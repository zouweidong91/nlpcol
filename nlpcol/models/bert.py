
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from nlpcol.activations import get_activation
from nlpcol.generation import UnilmGenerationMixin
from nlpcol.layers.embed import BertEmbeddings
from nlpcol.layers.layer import LayerNorm
from torch import Size, Tensor

from .base import BaseConfig, BaseModel
from .base.encoder import Stack, StackOutput


# config.json
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


class Config(BaseConfig):
    def __init__(self, **kwargs):
        # 通用配置
        self.d_model: int = kwargs.get('hidden_size')
        self.d_ff: int = kwargs.get('intermediate_size')
        self.n_heads: int = kwargs.get('num_attention_heads')
        self.vocab_size: int = kwargs.get('vocab_size')
        self.dropout_rate: float = kwargs.get('hidden_dropout_prob')
        self.initializer_range: float = kwargs.get('initializer_range')
        self.layer_norm_eps: float = kwargs.get('layer_norm_eps')
        self.num_layers: int = kwargs.get('num_hidden_layers')
        
        self.pad_token_id: int = kwargs.get('pad_token_id')

        self.unilm:bool = kwargs.get('unilm', False)  # 是否使用Unilm模式
        self.max_batch_size:int = kwargs.get('max_batch_size', 16)  # 推理过程中batch_size不能大于此值， kv_cache用
        self.max_position: int = kwargs.get('max_position_embeddings')
        
        # bert config文件配置
        self.architectures: list = kwargs.get('architectures')
        self.attention_probs_dropout_prob: float = kwargs.get('attention_probs_dropout_prob') # 直接用hidden_dropout_prob
        self.directionality: str = kwargs.get('directionality')
        self.hidden_act: str = kwargs.get('hidden_act')
        self.model_type: str = kwargs.get('model_type')
        self.pooler_fc_size: int = kwargs.get('pooler_fc_size')
        self.pooler_num_attention_heads: int = kwargs.get('pooler_num_attention_heads')
        self.pooler_num_fc_layers: int = kwargs.get('pooler_num_fc_layers')
        self.pooler_size_per_head: int = kwargs.get('pooler_size_per_head')
        self.pooler_type: str = kwargs.get('pooler_type')
        self.type_vocab_size: int = kwargs.get('type_vocab_size')
        

# *******************pool nsp mlm 下游任务***********************
class BertPool(nn.Module):
    """pool层"""
    def __init__(self, config: Config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.act = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        hidden_states (Tensor): BertEncoder.last_hidden_state
        """
        first_token_tensor = hidden_states[:, 0] # 获取第一个每行的token btz*hidden_size
        pool_output = self.dense(first_token_tensor)
        pool_output = self.act(pool_output)
        return pool_output


class BertMLM(nn.Module):
    """bert MLM任务"""
    def __init__(self, config: Config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.act = get_activation(config.hidden_act)
        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # mlm解码阶段需要做需要权重共享
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        hidden_states (Tensor): BertEncoder.last_hidden_state
        """
        hidden_states = self.layer_norm(self.act(self.dense(hidden_states)))
        hidden_states = self.lm_head(hidden_states)
        return hidden_states

class BertNsp(nn.Module):
    """bert nsp任务"""
    def __init__(self, config:Config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.d_model, 2)
    
    def forward(self, pooled_output: Tensor) -> Tensor:
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

@dataclass
class BertOutput:
    loss: torch.FloatTensor = None
    pooled_output: torch.FloatTensor = None
    nsp_scores: torch.FloatTensor = None
    lm_logits: torch.FloatTensor = None
    hidden_states: List[torch.FloatTensor] = None # 所有encoder层的输出
    last_hidden_state: torch.FloatTensor = None # 最后一层encoer的输出
    attention_mask: torch.LongTensor=None


class BertModel(BaseModel, UnilmGenerationMixin):
    """bert 模型"""
    def __init__(self,
        config: Config,
        with_pool = False, # 是否包含pool部分
        with_nsp = False,  # 是否包含nsp部分
        with_mlm = False,  # 是否包含mlm部分
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.config: Config
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm

        self.embed = BertEmbeddings(config)
        self.encoder = Stack(config)
        
        if self.with_pool:
            self.pooler = BertPool(config)
            # nsp的输入为pooled_output
            if self.with_nsp:
                self.nsp = BertNsp(config)

        if self.with_mlm:
            self.mlm = BertMLM(config)

        self.tie_weights()

    def get_output_embeddings(self):
        # tie_weights用
        if self.with_mlm:
            return self.mlm.lm_head

    def forward(
        self,
        input_ids: Optional[Tensor],
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor]=None, # unilm任务生成模式下使用
        start_pos:int=0
    ):
        # ========================= attention_mask =========================
        if attention_mask is None:
            # 非padding部分为1, padding部分为0
            attention_mask = (input_ids != self.config.pad_token_id).long() # bert默认0为mask_value

        # embed
        input_embedding = self.embed(input_ids, token_type_ids, position_ids, start_pos)

        # encoder
        enc_output:StackOutput = self.encoder(
            hidden_states = input_embedding, 
            attention_mask = attention_mask, 
            token_type_ids = token_type_ids, # unilm mask用
            start_pos = start_pos
        )

        hidden_states = enc_output.last_hidden_state
        pooled_output, nsp_scores, lm_logits, loss = None, None, None, None

        if self.with_pool:
            pooled_output = self.pooler(hidden_states)
        if self.with_pool and self.nsp:
            nsp_scores = self.nsp(pooled_output)

        if self.with_mlm:
            lm_logits:Tensor = self.mlm(hidden_states)
            if labels is not None: # unilm任务
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return BertOutput(
            loss = loss,
            pooled_output = pooled_output,
            nsp_scores = nsp_scores,
            lm_logits = lm_logits,
            hidden_states = enc_output.hidden_states,
            last_hidden_state = enc_output.last_hidden_state,
            attention_mask = attention_mask
        )

    @property
    def origin_embedding_keys(self) -> list:
        return [
            'bert.embeddings.word_embeddings.weight',
            'cls.predictions.decoder.weight',
            'cls.predictions.bias',
        ]

    def variable_mapping(self):
        """
        不同代码参数命名不同，需要做参数映射   new_key: old_key
        """
        prefix = 'bert'
        mapping = {
            'embed.token_embeddings.weight':  f'{prefix}.embeddings.word_embeddings.weight',
            'embed.position_embeddings.weight':  f'{prefix}.embeddings.position_embeddings.weight',
            'embed.token_type_embeddings.weight':  f'{prefix}.embeddings.token_type_embeddings.weight',
            'embed.layer_norm.weight':  f'{prefix}.embeddings.LayerNorm.gamma',
            'embed.layer_norm.bias':  f'{prefix}.embeddings.LayerNorm.beta',
            'pooler.dense.weight': f'{prefix}.pooler.dense.weight',
            'pooler.dense.bias': f'{prefix}.pooler.dense.bias',
            'nsp.seq_relationship.weight': 'cls.seq_relationship.weight',
            'nsp.seq_relationship.bias': 'cls.seq_relationship.bias', 
            'mlm.dense.weight': 'cls.predictions.transform.dense.weight',
            'mlm.dense.bias': 'cls.predictions.transform.dense.bias',
            'mlm.layer_norm.weight': 'cls.predictions.transform.LayerNorm.gamma',
            'mlm.layer_norm.bias': 'cls.predictions.transform.LayerNorm.beta',
            'mlm.bias': 'cls.predictions.bias',
            'mlm.lm_head.weight': 'cls.predictions.decoder.weight',
            'mlm.lm_head.bias': 'cls.predictions.bias'
        }

        for i in range(self.config.num_layers):
            prefix_i = f'{prefix}.encoder.layer.%d.' % i

            mapping.update(
                {
                    f"encoder.layers.{i}.self_attention.q.weight": prefix_i + 'attention.self.query.weight',
                    f"encoder.layers.{i}.self_attention.q.bias": prefix_i + 'attention.self.query.bias',
                    f"encoder.layers.{i}.self_attention.k.weight": prefix_i + 'attention.self.key.weight',
                    f"encoder.layers.{i}.self_attention.k.bias": prefix_i + 'attention.self.key.bias',
                    f"encoder.layers.{i}.self_attention.v.weight": prefix_i + 'attention.self.value.weight',
                    f"encoder.layers.{i}.self_attention.v.bias": prefix_i + 'attention.self.value.bias',
                    f"encoder.layers.{i}.self_attention_output.o.weight": prefix_i + 'attention.output.dense.weight',
                    f"encoder.layers.{i}.self_attention_output.o.bias": prefix_i + 'attention.output.dense.bias',
                    f"encoder.layers.{i}.self_attention_output.layer_norm.weight": prefix_i + 'attention.output.LayerNorm.gamma',
                    f"encoder.layers.{i}.self_attention_output.layer_norm.bias": prefix_i + 'attention.output.LayerNorm.beta',
                    f"encoder.layers.{i}.ffn.ff.dense_1.weight": prefix_i + 'intermediate.dense.weight',
                    f"encoder.layers.{i}.ffn.ff.dense_1.bias": prefix_i + 'intermediate.dense.bias',
                    f"encoder.layers.{i}.ffn.ff.dense_2.weight": prefix_i + 'output.dense.weight',
                    f"encoder.layers.{i}.ffn.ff.dense_2.bias": prefix_i + 'output.dense.bias',
                    f"encoder.layers.{i}.ffn.layer_norm.weight": prefix_i + 'output.LayerNorm.gamma',
                    f"encoder.layers.{i}.ffn.layer_norm.bias": prefix_i + 'output.LayerNorm.beta'
                }
            )

        return mapping
        
