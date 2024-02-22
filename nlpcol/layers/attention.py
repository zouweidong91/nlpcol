
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from nlpcol.layers.layer import LayerNorm
from nlpcol.models.base import BaseConfig as Config
from torch import Size, Tensor

# encoder 模型
class EncAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.dim_per_head = config.d_model // config.n_heads

        self.q = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.k = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.v = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout_rate)

    def shape(self, x:Tensor) -> Tensor:
        bs = x.shape[0]
        return x.view(bs, -1, self.n_heads, self.dim_per_head).transpose(1, 2)

    def unshape(self, x:Tensor) -> Tensor:
        # transpose、permute等维度变换操作后，tensor在内存中不再是连续存储的，
        # 而view操作要求tensor的内存连续存储，需要调用contiguous
        bs = x.shape[0]
        return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.dim_per_head)

    def get_padding_mask(self, key_mask:Tensor):
        # 目的是为了适配多头注意力机制，[batch_size, to_seq_length] -> [batch_size, 1, 1, to_seq_length]
        # 广播到[batch_size, num_heads, from_seq_length, to_seq_length]尺寸
        # bert源码中注明不考虑from_tensor的mask。因为在下游任务如ner中，也会对超出input_ids的部分忽略处理。
        # We don't assume that `from_tensor` is a mask (although it could be). We
        # don't actually care if we attend *from* padding tokens (only *to* padding)
        # tokens so we create a tensor of all ones.
        # enc_self_att, dec_cross_att
        key_mask = key_mask.unsqueeze(1).unsqueeze(2)           # (bs, None, None, klen)
        key_mask = (1.0 - key_mask) * - 10000.0 # 传入的mask的非padding部分为1, padding部分为0
        return key_mask

    def get_mask(self, scores:Tensor, key_mask:Tensor):
        return self.get_padding_mask(key_mask)

    def score_scale(self, scores:Tensor) -> Tensor:
        # atten_score是否缩放
        return scores / math.sqrt(self.dim_per_head)

    def applay_att_score_bias(self, scores:Tensor) -> Tensor:
        return scores

    def core_attention(self, q, k, v, key_mask):
        # 形状变换
        q = self.shape(q)                                       # (bs, n_heads, qlen, dim_per_head)
        k = self.shape(k)                                       # (bs, n_heads, klen, dim_per_head)
        v = self.shape(v)                                       # (bs, n_heads, klen, dim_per_head)
        
        # 如果只需要交换两个维度，使用 transpose 即可
        # 如果要进行更复杂的维度重排，使用 permute。
        scores = torch.matmul(q, k.transpose(-1, -2))           # (bs, n_heads, qlen, klen)
        scores = self.score_scale(scores)                       # (bs, n_heads, qlen, klen)
        scores = self.applay_att_score_bias(scores)             # (bs, n_heads, qlen, klen)

        key_mask = self.get_mask(scores, key_mask)
        scores = scores + key_mask                              # (bs, n_heads, qlen, klen)
        
        # scores归一化到0-1
        attention_probs = F.softmax(scores, dim=-1)             # (bs, n_heads, qlen, klen)
        attention_probs = self.dropout(attention_probs)         # (bs, n_heads, qlen, klen)
        context = torch.matmul(attention_probs, v)              # (bs, n_heads, qlen, dim_per_head)
        context = self.unshape(context)                         # (bs, qlen, d_model)
        return context

    def forward(self, query, key, value, key_mask):
        """
            key_mask (_type_): (bs, klen)
        """
        # 线性变换
        q = self.q(query)                                       # (bs, qlen, d_model)
        k = self.k(key)                                         # (bs, klen, d_model)
        v = self.v(value)                                       # (bs, klen, d_model)

        context = self.core_attention(q, k, v, key_mask)        # (bs, qlen, d_model)
        return context
        


# decoder 模型
class DecAttention(EncAttention):
    def __init__(self, config: Config):
        """_summary_

        Args:
            config (Config): _description_
        """
        super().__init__(config)

        # kv_cache
        self.cache_k = torch.zeros(
            config.max_batch_size, config.max_position, config.d_model, 
        )
        self.cache_v = torch.zeros(
            config.max_batch_size, config.max_position, config.d_model, 
        )

    def get_lm_mask(self, scores:Tensor):
        """定义下三角矩阵的atten_mask， 语言模型用
        scores: [batch_size, num_heads, from_seq_len, to_seq_len]
        key_mask: [batch_size, to_seq_length]
        """
        qlen, klen = scores.shape[2:]   # 推理阶段qlen==1
        key_mask = torch.tril(
            torch.ones(klen, klen, dtype=torch.long, device=scores.device), diagonal=0
        )
        # (batch_size, n_heads, klen, klen)
        key_mask = key_mask.unsqueeze(0).unsqueeze(1)               # (bs, n_heads, klen, klen)
        key_mask = key_mask[:, :, -qlen:, :]

        key_mask = (1.0 - key_mask) * - 10000.0     # 传入的mask的非padding部分为1, padding部分为0
        return key_mask

    def get_mask(self, scores:Tensor, key_mask:Tensor):
        return self.get_lm_mask(scores)

    def project(self, query:Tensor, key:Tensor, value:Tensor, start_pos:int):
        # 推理阶段，使用kv_cache
        # decoder-selfAtten:
        #     推理时seq_length=1
        bsz, q_len, _ = query.shape
        self.cache_k[:bsz, start_pos : start_pos + q_len] = self.k(key)
        self.cache_v[:bsz, start_pos : start_pos + q_len] = self.v(value)
        keys = self.cache_k[:bsz, : start_pos + q_len]
        values = self.cache_v[:bsz, : start_pos + q_len]

        return keys.to(key), values.to(value)

    def forward(self, query, key, value, key_mask:Tensor, start_pos:int):
        # query, key, value: [bsz, seqlen, head_size] 
        # start_pos: Starting position for caching.
        q = self.q(query)

        # train时无需使用kv_cache
        if self.training:
            k = self.k(key)
            v = self.v(value)
        else:
            k, v = self.project(query, key, value, start_pos)
    
        context = self.core_attention(q, k, v, key_mask)
        return context


# encoder-decoder 模型
class EncDecAttention(DecAttention):
    def __init__(self, config: Config, is_self_atten=True):
        """_summary_

        Args:
            config (Config): _description_
            is_self_atten (bool, optional): selfAtten or crossAtten. Defaults to True.
        """
        super().__init__(config)

        self.is_decoder = config.is_decoder
        self.is_self_atten = is_self_atten


    def get_mask(self, scores:Tensor, key_mask:Tensor):
        """
        scores: [batch_size, num_heads, from_seq_len, to_seq_len]
        key_mask: [batch_size, to_seq_length]
        """
        if key_mask is not None:
            # enc_self_atten  dec_cross_atten
            return self.get_padding_mask(key_mask)
        else:
            # dec_self_atten
            return self.get_lm_mask(scores)


    def project(self, query:Tensor, key:Tensor, value:Tensor, start_pos:int):
        # 推理阶段，使用kv_cache
        # decoder-selfAtten:
        #     推理时seq_length=1
        # decoder-crossAtten:
        #     推理时query_seq_length=1
        #     key, value的seq_length为src_len
        klen = vlen = key.shape[1]
        bsz, q_len, _ = query.shape
        
        # selfAtten
        if self.is_self_atten:
            self.cache_k[:bsz, start_pos : start_pos + q_len] = self.k(key)
            self.cache_v[:bsz, start_pos : start_pos + q_len] = self.v(value)
            keys = self.cache_k[:bsz, : start_pos + q_len]
            values = self.cache_v[:bsz, : start_pos + q_len]

        # crossAtten
        else:
            if start_pos == 0: # crossAtten时只需在0位置时计算一次
                self.cache_k[:bsz, :klen] = self.k(key)
                self.cache_v[:bsz, :vlen] = self.v(value)

            keys = self.cache_k[:bsz, :klen]
            values = self.cache_v[:bsz, :vlen]

        return keys.to(key), values.to(value)



class AttentionOutput(nn.Module):
    """每个attention层后都会有一个线性层  
    LayerNorm分pre_norm和post_norm
    顺序为： Drop --> Add --> LayerNorm(post)
    """
    def __init__(self, config:Config):
        super().__init__()
        self.layer_norm_type = config.layer_norm_type

        self.o = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout_rate)

        if self.layer_norm_type == "post":
            self.layer_norm = self.get_ln(config)

    def get_ln(self, config: Config):
        return LayerNorm(config.d_model, config.layer_norm_eps)

    def forward(self, context: Tensor, input_tensor: Tensor) -> Tensor:
        """
        Args:
            context (Tensor): attention层的输出结果
            input_tensor (Tensor): BertEmbeddings层的输出
        """
        context = self.o(context)
        context = self.dropout(context) + input_tensor

        if self.layer_norm_type == "post": 
            context = self.layer_norm(context)

        return context

