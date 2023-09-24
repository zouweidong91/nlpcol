
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from nlpcol.activations import get_activation
from nlpcol.layers.layer import LayerNorm
from torch import Size, Tensor

from .base import BaseModel


@dataclass
class Config:
    # 以下参数来自config.json文件
    # 显式声明，支持下文自动补全
    architectures: str
    attention_probs_dropout_prob: float
    directionality: str
    hidden_act: str
    hidden_dropout_prob: float
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    layer_norm_eps: float
    max_position_embeddings: int
    model_type: str
    num_attention_heads: int
    num_hidden_layers: int
    pad_token_id: int
    pooler_fc_size: int
    pooler_num_attention_heads: int
    pooler_num_fc_layers: int
    pooler_size_per_head: int
    pooler_type: str
    type_vocab_size: int
    vocab_size: int

"""
{
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "type_vocab_size": 2,
  "vocab_size": 21128
}
"""


class BertEmbeddings(nn.Module):
    """用 word, position and token_type embeddings 构造embedding
    """
    def __init__(self, config: Config):
        super().__init__()
        # padding_idx 将相应位置的embedding全部设置为0， 不参与梯度计算，训练过程参数也不更新
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,

    ) -> Tensor:

        device = input_ids.device
        btz, seq_len = input_ids.shape

        inputs_embeddings = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.shape, dtype=torch.long, device=device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeddings + token_type_embeddings

        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(btz, -1)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
        
        
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        
    def mask(self, inputs, key_masks):
        """其他Module继承支持重写mask函数
        """

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        """对输入张量做形状变换
        B:batch_size  L:seq_length  H: hidden_size

        Args:
            x (Tensor): B*L*H
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)   
        x = x.view(new_x_shape)  
        # torch.view 只能用于连续的张量  torch.reshape 可以用于不连续的张量  x_t.is_contiguous()  x:  B*L*num_head*head_size
        # view 的过程可以简单地理解为先将张量进行拉平（变成一维），然后再按照指定的新形状进行重塑
        return x.permute(0, 2, 1, 3)  # x:  B*num_head*L*head_size

    def forward(self, query, key, value, key_mask):
        """_summary_

        Args:
            query (_type_): B*L*H
            key (_type_): B*L*H
            value (_type_): B*L*H
            key_mask (_type_): [btz, 1, 1, seq_len]
        """
        # 线性变换
        query_layer = self.query(query)
        key_layer = self.key(key)
        value_layer = self.value(value)
        # 形状变换
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        # attention
        # transpose和permute的区别：如果你只需要交换两个维度，transpose 是一个简单的选择。如果你需要进行更复杂的维度重排，你应该使用 permute。
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # B*num_head*L*L
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        key_mask = (1.0 - key_mask) * - 10000.0 # 传入的mask的非padding部分为1, padding部分为0
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
    顺序为： Drop --> Add --> LayerNorm
    """
    def __init__(self, config:Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, context_layer: Tensor, input_tensor: Tensor) -> Tensor:
        """
        Args:
            context_layer (Tensor): attention层的输出结果
            input_tensor (Tensor): BertEmbeddings层的输出
        """
        output_layer = self.dense(context_layer)
        output_layer = self.dropout(output_layer)
        output_layer = self.LayerNorm(output_layer + input_tensor)
        return output_layer
        

# TODO  合并FeedForward和FFNOutput
class FeedForward(nn.Module):
    """前馈层
    """
    def __init__(self, config:Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = get_activation(config.hidden_act)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        return hidden_states

class FFNOutput(nn.Module):
    """顺序为： Drop --> Add --> LayerNorm
    """
    def __init__(self, config: Config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        """_summary_

        Args:
            hidden_states (Tensor): FeedForward的输出结果
            input_tensor (Tensor): AttentionOutput的输出结果

        Returns:
            Tensor: _description_
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """Transformer层 encoder block
        顺序为： Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm

    Args:
        nn (_type_): _description_
    """
    def __init__(self, config: Config):
        super().__init__()
        self.multiHeadAttention = MultiHeadAttentionLayer(config)
        self.attentionOutput = AttentionOutput(config)
        self.feedForward = FeedForward(config)
        self.ffnOutput = FFNOutput(config)
    
    def forward(self, hidden_states:Tensor, attention_mask:Tensor) -> Tensor:
        context_layer = self.multiHeadAttention(
            hidden_states, hidden_states, hidden_states, attention_mask 
        )
        attention_output = self.attentionOutput(context_layer, hidden_states)
        feed_forward = self.feedForward(attention_output)
        layer_output = self.ffnOutput(feed_forward, attention_output)
        return layer_output
        
@dataclass
class BertEncoderOutput:
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[List[torch.FloatTensor]] = None


class BertEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states:Tensor, attention_mask:Tensor) -> BertEncoderOutput:
        """这里控制整个enceder层的输出格式, 暂时只输出最后一个隐藏藏 TODO
        """
        all_hidden_states = [hidden_states]

        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, attention_mask)
            all_hidden_states.append(hidden_states)

        return BertEncoderOutput(
            last_hidden_state = hidden_states,
            hidden_states = all_hidden_states
        )
                

class BertPool(nn.Module):
    """pool层"""
    def __init__(self, config: Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        hidden_states (Tensor): BertEncoder.last_hidden_state
        """
        first_token_tensor = hidden_states[:, 0] # 获取第一个每行的token btz*hidden_size
        pool_output = self.dense(first_token_tensor)
        pool_output = self.activation(pool_output)
        return pool_output


class BertMLM(nn.Module):
    """bert MLM任务"""
    def __init__(self, config: Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = get_activation(config.hidden_act)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        hidden_states (Tensor): BertEncoder.last_hidden_state
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertNsp(nn.Module):
    """bert nsp任务"""
    def __init__(self, config:Config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
    
    def forward(self, pooled_output: Tensor) -> Tensor:
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

@dataclass
class BertOutput:
    pooled_output: torch.FloatTensor = None
    nsp_scores: torch.FloatTensor = None
    mlm_scores: torch.FloatTensor = None
    encoded_layers: List[torch.FloatTensor] = None # 所有encoder层的输出
    last_hidden_state: torch.FloatTensor = None # 最后一层encoer的输出


class BertModel(BaseModel):
    """bert 模型"""
    def __init__(self,
        config: dict,
        with_pool = False, # 是否包含pool部分
        with_nsp = False,  # 是否包含nsp部分
        with_mlm = False,  # 是否包含mlm部分
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config = Config(**config)
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        
        if self.with_pool:
            self.pooler = BertPool(config)
            # nsp的输入为pooled_output
            if self.with_nsp:
                self.nsp = BertNsp(config)

        if self.with_mlm:
            self.mlm = BertMLM(config)


    def forward(
        self,
        input_ids: Optional[Tensor],
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ):
        # ========================= attention_mask =========================
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long() # bert默认0为mask_value
            # 目的是为了适配多头注意力机制，[batch_size, to_seq_length] -> [batch_size, 1, 1, to_seq_length]
            # 广播到[batch_size, num_heads, from_seq_length, to_seq_length]尺寸
            # bert源码中注明不考虑from_tensor的mask。因为在下游任务如ner中，也会对超出input_ids的部分忽略处理。
            # We don't assume that `from_tensor` is a mask (although it could be). We
            # don't actually care if we attend *from* padding tokens (only *to* padding)
            # tokens so we create a tensor of all ones.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        input_embedding = self.embeddings(input_ids, token_type_ids, position_ids)
        encoder_output:BertEncoderOutput = self.encoder(input_embedding, attention_mask)
        hidden_states = encoder_output.last_hidden_state
        pooled_output, nsp_scores, mlm_scores = None, None, None

        if self.with_pool:
            pooled_output = self.pooler(hidden_states)
        if self.with_pool and self.nsp:
            nsp_scores = self.nsp(pooled_output)
        if self.with_mlm:
            mlm_scores = self.mlm(hidden_states)
            mlm_activation = get_activation('softmax')
            mlm_scores = mlm_activation(mlm_scores)

        return BertOutput(
            pooled_output = pooled_output,
            nsp_scores = nsp_scores,
            mlm_scores = mlm_scores,
            encoded_layers = encoder_output.hidden_states,
            last_hidden_state = encoder_output.last_hidden_state
        )

    def variable_mapping(self):
        """
        不同代码参数命名不同，需要做参数映射   new_key: old_key
        """
        prefix = 'bert'
        mapping = {
            'embeddings.word_embeddings.weight':  f'{prefix}.embeddings.word_embeddings.weight',
            'embeddings.position_embeddings.weight':  f'{prefix}.embeddings.position_embeddings.weight',
            'embeddings.token_type_embeddings.weight':  f'{prefix}.embeddings.token_type_embeddings.weight',
            'embeddings.LayerNorm.weight':  f'{prefix}.embeddings.LayerNorm.gamma',
            'embeddings.LayerNorm.bias':  f'{prefix}.embeddings.LayerNorm.beta',
            'pooler.dense.weight': f'{prefix}.pooler.dense.weight',
            'pooler.dense.bias': f'{prefix}.pooler.dense.bias',
            'nsp.seq_relationship.weight': 'cls.seq_relationship.weight',
            'nsp.seq_relationship.bias': 'cls.seq_relationship.bias', 
            'mlm.dense.weight': 'cls.predictions.transform.dense.weight',
            'mlm.dense.bias': 'cls.predictions.transform.dense.bias',
            'mlm.LayerNorm.weight': 'cls.predictions.transform.LayerNorm.gamma',
            'mlm.LayerNorm.bias': 'cls.predictions.transform.LayerNorm.beta',
            'mlm.bias': 'cls.predictions.bias',
            'mlm.decoder.weight': 'cls.predictions.decoder.weight',
            'mlm.decoder.bias': 'cls.predictions.bias'
        }

        for i in range(self.config.num_hidden_layers):
            prefix_i = f'{prefix}.encoder.layer.%d.' % i

            mapping.update(
                {
                    f"encoder.layers.{i}.multiHeadAttention.query.weight": prefix_i + 'attention.self.query.weight',
                    f"encoder.layers.{i}.multiHeadAttention.query.bias": prefix_i + 'attention.self.query.bias',
                    f"encoder.layers.{i}.multiHeadAttention.key.weight": prefix_i + 'attention.self.key.weight',
                    f"encoder.layers.{i}.multiHeadAttention.key.bias": prefix_i + 'attention.self.key.bias',
                    f"encoder.layers.{i}.multiHeadAttention.value.weight": prefix_i + 'attention.self.value.weight',
                    f"encoder.layers.{i}.multiHeadAttention.value.bias": prefix_i + 'attention.self.value.bias',
                    f"encoder.layers.{i}.attentionOutput.dense.weight": prefix_i + 'attention.output.dense.weight',
                    f"encoder.layers.{i}.attentionOutput.dense.bias": prefix_i + 'attention.output.dense.bias',
                    f"encoder.layers.{i}.attentionOutput.LayerNorm.weight": prefix_i + 'attention.output.LayerNorm.gamma',
                    f"encoder.layers.{i}.attentionOutput.LayerNorm.bias": prefix_i + 'attention.output.LayerNorm.beta',
                    f"encoder.layers.{i}.feedForward.dense.weight": prefix_i + 'intermediate.dense.weight',
                    f"encoder.layers.{i}.feedForward.dense.bias": prefix_i + 'intermediate.dense.bias',
                    f"encoder.layers.{i}.ffnOutput.dense.weight": prefix_i + 'output.dense.weight',
                    f"encoder.layers.{i}.ffnOutput.dense.bias": prefix_i + 'output.dense.bias',
                    f"encoder.layers.{i}.ffnOutput.LayerNorm.weight": prefix_i + 'output.LayerNorm.gamma',
                    f"encoder.layers.{i}.ffnOutput.LayerNorm.bias": prefix_i + 'output.LayerNorm.beta'
                }
            )

        return mapping
        
