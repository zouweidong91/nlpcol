
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from nlpcol.activations import get_activation
from nlpcol.layers.layer import T5LayerNorm
from torch import Size, Tensor
from nlpcol.layers.pe import RelativePositionalT5
import copy

from .base import BaseModel


# T5-Small (60 million parameters): gs://t5-data/pretrained_models/small
# T5-Base (220 million parameters): gs://t5-data/pretrained_models/base
# T5-Large (770 million parameters): gs://t5-data/pretrained_models/large
# T5-3B (3 billion parameters): gs://t5-data/pretrained_models/3B
# T5-11B (11 billion parameters): gs://t5-data/pretrained_models/11B

# https://kexue.fm/archives/7867
# https://zhuanlan.zhihu.com/p/88438851 T5 模型：NLP Text-to-Text 预训练模型超大规模探索
# https://huggingface.co/docs/transformers/main/en/model_doc/mt5#transformers.MT5Model

# Transformer Encoder-Decoder 模型；
# BERT-style 式的破坏方法；
# Replace Span 的破坏策略；
# 15 %的破坏比；
# 3 的破坏时小段长度。

# T5.1.0和T5.1.1区别
# FFN 把relu激活的第一个变化层改为了gelu激活的门控线性单元，这样FFN层增加了50%参数，但是从论文效果看效果明显增加
# 此外，T5.1.1还对Embedding层做了改动，
# 原来在T5.1.0中，Encoder和Decoder的Embedding层、Decoder最后预测概率分布的Softmax层都是共享同一个Embedding矩阵的，
# 现在T5.1.1只让Encoder和Decoder的Embedding层共享，而Decoder最后预测概率分布的Softmax层则用了一个独立的Embedding矩阵，
# 当然这会让参数量大大增加，但Google的结论说这样做效果会更好


# 实现原始mt5  t5没有中文预训练权重。用Mt5实现 T5.1.1

# TODO 配置映射
class Config:
    # 以下参数来自mt5 base config.json文件
    # 显式声明，支持下文自动补全
    def __init__(self, **kwargs):
        self.architectures: str = kwargs.get('architectures')
        self.d_ff: int = kwargs.get('d_ff')
        self.d_kv: int = kwargs.get('d_kv')
        self.d_model: int = kwargs.get('d_model')
        self.decoder_start_token_id: int = kwargs.get('decoder_start_token_id')
        self.dropout_rate: float = kwargs.get('dropout_rate')
        self.eos_token_id: int = kwargs.get('eos_token_id')
        self.feed_forward_proj: str = kwargs.get('feed_forward_proj')
        self.initializer_factor: float = kwargs.get('initializer_factor')
        self.is_encoder_decoder: bool = kwargs.get('is_encoder_decoder')
        self.layer_norm_epsilon: float = kwargs.get('layer_norm_epsilon')
        self.model_type: str = kwargs.get('model_type')
        self.num_decoder_layers: int = kwargs.get('num_decoder_layers')
        self.num_heads: int = kwargs.get('num_heads')
        self.num_layers: int = kwargs.get('num_layers')
        self.output_past: bool = kwargs.get('output_past')
        self.pad_token_id: int = kwargs.get('pad_token_id')
        self.relative_attention_num_buckets: int = kwargs.get('relative_attention_num_buckets')
        self.tie_word_embeddings: bool = kwargs.get('tie_word_embeddings')
        self.tokenizer_class: str = kwargs.get('tokenizer_class')
        self.use_cache: bool = kwargs.get('use_cache')
        self.vocab_size: int = kwargs.get('vocab_size')
        self.is_decoder: bool = False  # 是否属于decoder
        self.max_seq_length:int = kwargs.get('max_seq_length', 512)  # 需要大于max(tar_len, src_len)， 相对位置编码用

"""
{
  "_name_or_path": "/home/patrick/hugging_face/t5/mt5-base",
  "architectures": [
    "MT5ForConditionalGeneration"
  ],
  "d_ff": 2048,
  "d_kv": 64,
  "d_model": 768,
  "decoder_start_token_id": 0,
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "mt5",
  "num_decoder_layers": 12,
  "num_heads": 12,
  "num_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "tokenizer_class": "T5Tokenizer",
  "transformers_version": "4.10.0.dev0",
  "use_cache": true,
  "vocab_size": 250112
}

"""

class T5Embeddings(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, input_ids: Optional[torch.LongTensor]) -> Tensor:
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.dropout(embeddings)
        return embeddings

class PositionWiseFeedForward(nn.Module):
    """
    位置感知前馈网络
        通常包括两个线性层和一个非线性激活函数，这两个线性变换是位置独立的，因此被称为“位置感知”
        通常的结构： 线性变换1 --> 非线性激活函数 --> 线性变换2
    """
    def __init__(self, config: Config):
        super().__init__()
        self.dense_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.dense_2 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.dense_output = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.drop = nn.Dropout(config.dropout_rate)

        dense_act_fn = "gelu_new"
        self.act = get_activation(dense_act_fn)
        
    def forward(self, hidden_states: Tensor) -> Tensor:
        # hidden_states shape: (btz, seq_len, hidden_size)
        hidden_gelu = self.act(self.dense_1(hidden_states))
        hidden_linear = self.dense_2(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.drop(hidden_states)

        hidden_states = self.dense_output(hidden_states)
        # hidden_states shape: (btz, seq_len, d_ff)
        return hidden_states

class FFN(nn.Module):
    """顺序为： LN --> FF --> Add
    """
    def __init__(self, config: Config):
        super().__init__()
        self.ff = PositionWiseFeedForward(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: Tensor) -> Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.ff(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)  # add为标准化之前的hidden_states
        return hidden_states


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, config: Config, has_relative_attention_bias=False, is_decoder=False):
        super().__init__()

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.is_decoder = is_decoder
        self.all_head_size = config.num_heads * config.d_kv
        self.num_heads = config.num_heads
        self.head_size = config.d_kv

        self.q = nn.Linear(config.d_model, self.all_head_size, bias=False)
        self.k = nn.Linear(config.d_model, self.all_head_size, bias=False)
        self.v = nn.Linear(config.d_model, self.all_head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.relative_attention_bias = None
        if has_relative_attention_bias: # 只有selfAtten时需要，且仅第0层初始化，其它层共享权重。
            self.relative_attention_bias = nn.Embedding(config.relative_attention_num_buckets, config.num_heads)
            # print(config.max_seq_length)
            self.relative_position = RelativePositionalT5(
                config.max_seq_length, config.max_seq_length, self.relative_attention_num_buckets, is_decoder=self.is_decoder
            )

    def mask(self, inputs:Tensor, key_mask:Tensor):
        """其他Module继承支持重写mask函数
        inputs: [batch_size, num_heads, seq_len, seq_len]
        key_mask: [batch_size, to_seq_length]
        """
        if key_mask is not None:
            # 目的是为了适配多头注意力机制，[batch_size, to_seq_length] -> [batch_size, 1, 1, to_seq_length]
            # 广播到[batch_size, num_heads, from_seq_length, to_seq_length]尺寸
            # bert源码中注明不考虑from_tensor的mask。因为在下游任务如ner中，也会对超出input_ids的部分忽略处理。
            # We don't assume that `from_tensor` is a mask (although it could be). We
            # don't actually care if we attend *from* padding tokens (only *to* padding)
            # tokens so we create a tensor of all ones.
            # enc_self_att, dec_cross_att
            key_mask = key_mask.unsqueeze(1).unsqueeze(2)
        else:
            # dec_self_att 下三角矩阵
            seq_len = inputs.shape[-1]
            key_mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.long, device=inputs.device), diagonal=0
            )
            key_mask = key_mask.unsqueeze(0).unsqueeze(1)

        key_mask = (1.0 - key_mask) * - 10000.0 # 传入的mask的非padding部分为1, padding部分为0
        return key_mask

    def applay_att_score_bias(self, attention_scores:Tensor) -> Tensor:
        """
        T5 位置编码通过在attention_scores上加一个可训练偏置项  TODO relative_positions重复计算，需优化
        """
        if self.relative_attention_bias is None:  # crossAtten不需要位置编码
            return attention_scores

        qlen, klen = attention_scores.shape[2:]
        relative_position = self.relative_position(qlen, klen)
        values:Tensor = self.relative_attention_bias(relative_position)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  #  (1, num_heads, qlen, klen)
        return attention_scores + values

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)   
        x = x.view(new_x_shape)  
        # torch.view 只能用于连续的张量  torch.reshape 可以用于不连续的张量  x_t.is_contiguous()  x:  B*L*num_head*head_size
        # view 的过程可以简单地理解为先将张量进行拉平（变成一维），然后再按照指定的新形状进行重塑
        return x.permute(0, 2, 1, 3)  # x:  B*num_head*L*head_size

    def forward(self, query, key, value, key_mask):
        # 线性变换
        query_layer = self.q(query)
        key_layer = self.k(key)
        value_layer = self.v(value)
        # 形状变换
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        # attention
        # transpose和permute的区别：如果你只需要交换两个维度，transpose 是一个简单的选择。如果你需要进行更复杂的维度重排，你应该使用 permute。
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # B*num_head*L*L
        attention_scores = self.applay_att_score_bias(attention_scores)

        # attention_scores = attention_scores / math.sqrt(self.attention_head_size) # T5不做scale TODO
        key_mask = self.mask(attention_scores, key_mask)
        attention_scores = attention_scores + key_mask
        
        # attention_scores归一化到0-1
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # 矩阵形状变换过程 B*num_head*L*L x B*num_head*L*head_size --> B*num_head*L*head_size
        context_layer = torch.matmul(attention_probs, value_layer)
        # transpose、permute等维度变换操作后，tensor在内存中不再是连续存储的，而view操作要求tensor的内存连续存储，
        # 需要contiguous来返回一个contiguous
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer

class AttentionOutput(nn.Module):
    """每个attention层后都会有一个线性层
    顺序为： Drop --> Add
    """
    def __init__(self, config:Config):
        super().__init__()

        self.all_head_size = config.num_heads * config.d_kv
        self.dense = nn.Linear(config.d_model, self.all_head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, context_layer: Tensor, input_tensor: Tensor) -> Tensor:
        """
        Args:
            context_layer (Tensor): attention层的输出结果
            input_tensor (Tensor): attention层的归一化之前的输入
        """
        output_layer = self.dense(context_layer)
        output_layer = input_tensor + self.dropout(output_layer)
        return output_layer


class T5Layer(nn.Module):
    """TODO 抽象为 Transformer类
    Encoder的顺序
        LN --> self_Att --> Add --> LN --> FFN --> Add
    Decoder的顺序
        LN --> self_Att --> Add --> LN --> cross_Att --> Add --> LN --> FFN --> Add
    """
    def __init__(self, config: Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.selfAttention = MultiHeadAttentionLayer(config, has_relative_attention_bias=has_relative_attention_bias)
        self.selfAttentionOutput = AttentionOutput(config)

        if self.is_decoder:
            self.layer_norm2 = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
            self.crossAttention = MultiHeadAttentionLayer(config, is_decoder=True)
            self.crossAttentionOutput = AttentionOutput(config)

        self.ffn = FFN(config)


    def forward(
        self, 
        hidden_states:Tensor, 
        attention_mask:Tensor=None, 
        encoder_hidden_states:Tensor=None, 
        encoder_attention_mask:Tensor=None
    ) -> Tensor:

        # self attention
        normed_hidden_states = self.layer_norm(hidden_states)
        context_layer = self.selfAttention(
            normed_hidden_states, normed_hidden_states, normed_hidden_states, attention_mask
        )
        hidden_states = self.selfAttentionOutput(context_layer, hidden_states) # add为标准化之前的hidden_states

        # cross attention
        # query: selfattntion的输出   key_value： Encoder端的输出
        if self.is_decoder:
            normed_hidden_states = self.layer_norm2(hidden_states)
            context_layer = self.crossAttention(
                normed_hidden_states, encoder_hidden_states, encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = self.crossAttentionOutput(context_layer, hidden_states)

        # feedforward
        ffn_output = self.ffn(hidden_states)
        return ffn_output

@dataclass
class T5StackOutput:
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[List[torch.FloatTensor]] = None


class T5Stack(nn.Module):
    """
    has_relative_attention_bias 只有第一层的selfAtten才需要相对位置编码
    """
    def __init__(self, config: Config, embed:nn.Embedding):
        super().__init__()
        self.embed = embed
        self.layers = nn.ModuleList(
            [T5Layer(config, has_relative_attention_bias=bool(i==0)) for i in range(config.num_layers)]
        )
        
        for layer in self.layers[1:]:
            layer.selfAttention.relative_attention_bias = self.layers[0].selfAttention.relative_attention_bias # relative_attention_bias权重共享
            layer.selfAttention.relative_position = self.layers[0].selfAttention.relative_position

        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self, 
        input_ids:Tensor, 
        attention_mask:Tensor=None,
        encoder_hidden_states:Tensor=None, 
        encoder_attention_mask:Tensor=None
    ) -> Tensor:
        hidden_states = self.embed(input_ids)

        all_hidden_states = []

        for i, layer_module in enumerate(self.layers):
            all_hidden_states.append(hidden_states)
            hidden_states = layer_module(
                hidden_states, 
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask
            )

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        all_hidden_states.append(hidden_states)

        return T5StackOutput(
            last_hidden_state = hidden_states,
            hidden_states = all_hidden_states
        )


@dataclass
class Seq2SeqLMOutput:
    lm_logits: torch.FloatTensor = None
    encoder_last_hidden_state: torch.FloatTensor = None
    decoder_last_hidden_state: torch.FloatTensor= None
    encoder_hidden_states: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[List[torch.FloatTensor]] = None


class T5Model(BaseModel):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config = Config(**config, **kwargs)
        embed = T5Embeddings(config) # 在enc和dec之间共享embedding, 不声明为self.embed,则模型参数中没有embed.weight这个权重

        enc_config = copy.deepcopy(config)
        enc_config.is_decoder = False
        self.encoder = T5Stack(enc_config, embed)

        dec_config = copy.deepcopy(config)
        dec_config.is_decoder = True
        dec_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(dec_config, embed)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)


    def forward(
        self,
        input_ids:torch.LongTensor,
        decoder_input_ids:torch.LongTensor,
        attention_mask:torch.FloatTensor=None,
        decoder_attention_mask:torch.FloatTensor=None,
    ):
        if not attention_mask: # [batch_size, to_seq_length]
            attention_mask = (input_ids != self.config.pad_token_id).long()
        
        enc_output:T5StackOutput = self.encoder(input_ids, attention_mask)

        dec_output:T5StackOutput = self.decoder(
            input_ids = decoder_input_ids, 
            attention_mask = decoder_attention_mask,
            encoder_hidden_states = enc_output.last_hidden_state,
            encoder_attention_mask = attention_mask
        )

        lm_logits = self.lm_head(dec_output.last_hidden_state)

        
        return Seq2SeqLMOutput(
            lm_logits = lm_logits,
            encoder_last_hidden_state = enc_output.last_hidden_state,
            decoder_last_hidden_state = dec_output.last_hidden_state,
            encoder_hidden_states = enc_output.hidden_states,
            decoder_hidden_states = dec_output.hidden_states,
        )


    def variable_mapping(self):
        """
        不同代码参数命名不同，需要做参数映射   new_key: old_key
        """
        mapping = {
            "encoder.embed.word_embeddings.weight": "encoder.embed_tokens.weight",
            "decoder.embed.word_embeddings.weight": "decoder.embed_tokens.weight",
            "encoder.layers.0.selfAttention.relative_attention_bias.weight": "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            "decoder.layers.0.selfAttention.relative_attention_bias.weight": "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
        }

        self_atten_fn = lambda stack, i, j: {
                    f"{stack}.layers.{i}.selfAttention.q.weight": f"{stack}.block.{i}.layer.{j}.SelfAttention.q.weight",
                    f"{stack}.layers.{i}.selfAttention.k.weight": f"{stack}.block.{i}.layer.{j}.SelfAttention.k.weight",
                    f"{stack}.layers.{i}.selfAttention.v.weight": f"{stack}.block.{i}.layer.{j}.SelfAttention.v.weight",
                    f"{stack}.layers.{i}.selfAttentionOutput.dense.weight": f"{stack}.block.{i}.layer.{j}.SelfAttention.o.weight",
                    f"{stack}.layers.{i}.layer_norm.weight": f"{stack}.block.{i}.layer.{j}.layer_norm.weight",
                }
        cross_atten_fn = lambda stack, i, j: {
                    f"{stack}.layers.{i}.crossAttention.q.weight": f"{stack}.block.{i}.layer.{j}.EncDecAttention.q.weight",
                    f"{stack}.layers.{i}.crossAttention.k.weight": f"{stack}.block.{i}.layer.{j}.EncDecAttention.k.weight",
                    f"{stack}.layers.{i}.crossAttention.v.weight": f"{stack}.block.{i}.layer.{j}.EncDecAttention.v.weight",
                    f"{stack}.layers.{i}.crossAttentionOutput.dense.weight": f"{stack}.block.{i}.layer.{j}.EncDecAttention.o.weight",
                    f"{stack}.layers.{i}.layer_norm2.weight": f"{stack}.block.{i}.layer.{j}.layer_norm.weight",
                }
        ffn_fn = lambda stack, i, j: {
                    f"{stack}.layers.{i}.ffn.ff.dense_1.weight": f"{stack}.block.{i}.layer.{j}.DenseReluDense.wi_0.weight",
                    f"{stack}.layers.{i}.ffn.ff.dense_2.weight": f"{stack}.block.{i}.layer.{j}.DenseReluDense.wi_1.weight",
                    f"{stack}.layers.{i}.ffn.ff.dense_output.weight": f"{stack}.block.{i}.layer.{j}.DenseReluDense.wo.weight",
                    f"{stack}.layers.{i}.ffn.layer_norm.weight": f"{stack}.block.{i}.layer.{j}.layer_norm.weight",
                }

        for i in range(self.config.num_layers):
            mapping.update(self_atten_fn('encoder', i, 0))
            mapping.update(ffn_fn('encoder', i, 1))

        for i in range(self.config.num_decoder_layers):
            mapping.update(self_atten_fn('decoder', i, 0))
            mapping.update(cross_atten_fn('decoder', i, 1))
            mapping.update(ffn_fn('decoder', i, 2))

        return mapping
        
