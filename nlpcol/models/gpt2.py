

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from nlpcol.generation import DecGenerationMixin
from nlpcol.layers.attention import AttentionOutput, DecAttention
from nlpcol.layers.ffn import FFN
from nlpcol.layers.layer import LayerNorm
from torch import Size, Tensor

from .base import BaseConfig, BaseModel


"""
{
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 1024,
  "resid_pdrop": 0.1,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "vocab_size": 50257
}
"""


class Config(BaseConfig):
    def __init__(self, **kwargs):
        # 通用配置
        self.d_model: int = kwargs.get('n_embd')
        self.d_ff: int = kwargs.get('d_ff', 4 * self.d_model)
        self.n_heads: int = kwargs.get('n_head')
        self.vocab_size: int = kwargs.get('vocab_size')
        self.num_layers: int = kwargs.get('n_layer')
        self.dropout_rate: float = kwargs.get('dropout_rate', 0.1)
        self.initializer_range: float = kwargs.get('initializer_range')
        self.layer_norm_eps: float = kwargs.get('layer_norm_epsilon')
        
        # 自身配置没有，从extra_config获取
        self.pad_token_id: int = kwargs.get('pad_token_id')
        self.bos_token_id: int = kwargs.get('bos_token_id')
        self.eos_token_id: int = kwargs.get('eos_token_id')

        self.max_position:int = kwargs.get('n_positions', 512)  # 位置编码用
        self.max_batch_size:int = kwargs.get('max_batch_size', 16)  # 推理过程中batch_size不能大于此值， kv_cache用

        self.hidden_act: str = kwargs.get('hidden_act', 'gelu_new') # gpt使用gelu_new
        self.layer_norm_type: str = "pre" # gpt2使用pre_norm
        self.prefix: str = kwargs.get('prefix', '') # CDial-GPT相比gpt多了个transformer前缀



# decoder基础模型

# ？精简变量命名 TODO
class GptEmbeddings(nn.Module):
    """用 word, position 构造embedding
    """
    def __init__(self, config: Config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_position, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        token_type_ids: Optional[torch.LongTensor],
        position_ids: Optional[torch.LongTensor] = None,
        start_pos:int=0 # 解码时使用
    ) -> Tensor:

        device = input_ids.device
        btz, seq_len = input_ids.shape

        inputs_embeddings = self.token_embeddings(input_ids)

        if position_ids is None:
            position_ids = torch.arange(start_pos+seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(btz, -1)
            position_ids = position_ids[:, start_pos : start_pos+seq_len] # 解码时位置编码也要根据输入长度截取
        position_embeddings = self.position_embeddings(position_ids)

        # 统一embedding写法 后续模型继承
        # TODO token_type_ids is None   token_type_embedding = 0
        # ? 没有归一化
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(1))
            token_type_embeddings = self.token_embeddings(token_type_ids)
        else:
            token_type_embeddings = 0
            

        embeddings = inputs_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
        


class Gpt2Layer(nn.Module):
    """
    Attention --> Feed Forward
    """
    def __init__(self, config: Config):
        super().__init__()

        self.self_attention = DecAttention(config)
        self.self_attention_output = AttentionOutput(config)
        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = FFN(config)

        self.layer_norm_type = config.layer_norm_type

    def forward(
        self, 
        hidden_states:Tensor, 
        attention_mask:Tensor=None, 
        start_pos:int=0
    ) -> Tensor:
        
        # self attention
        normed_hidden_states = self.layer_norm(hidden_states)
        context_layer = self.self_attention(
            normed_hidden_states, normed_hidden_states, normed_hidden_states, attention_mask, start_pos
        )
        hidden_states = self.self_attention_output(context_layer, hidden_states) # add为标准化之前的hidden_states

        # feedforward
        ffn_output = self.ffn(hidden_states)
        return ffn_output

@dataclass
class Gpt2StackOutput:
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[List[torch.FloatTensor]] = None
    attention_mask: Optional[torch.LongTensor] = None # 推理时还需要用到


class Gpt2Stack(nn.Module):
    def __init__(self, config: Config, embed:nn.Module):
        super().__init__()
        self.embed = embed
        self.layers = nn.ModuleList([Gpt2Layer(config) for _ in range(config.num_layers)])
        
    def forward(
        self, 
        input_ids:Tensor, 
        token_type_ids:Tensor, 
        attention_mask:Tensor=None,
        start_pos:int=0
    ) -> Tensor:
        hidden_states = self.embed(input_ids, token_type_ids, start_pos=start_pos)
        all_hidden_states = []

        for i, layer_module in enumerate(self.layers):
            all_hidden_states.append(hidden_states)
            hidden_states = layer_module(
                hidden_states, 
                attention_mask,
                start_pos = start_pos
            )

        all_hidden_states.append(hidden_states)

        return Gpt2StackOutput(
            last_hidden_state = hidden_states,
            hidden_states = all_hidden_states,
            attention_mask = attention_mask
        )


@dataclass
class CausalLMOutput:
    loss: torch.FloatTensor = None
    lm_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor= None
    hidden_states: Optional[List[torch.FloatTensor]] = None


class Gpt2Model(BaseModel, DecGenerationMixin):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)
        self.config: Config
        
        self.embed = GptEmbeddings(config)
        self.decoder = Gpt2Stack(config, self.embed)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.ln_final = LayerNorm(config.d_model, config.layer_norm_eps)
        
        self.tie_weights()

    def forward(
        self,
        input_ids:torch.LongTensor=None,
        token_type_ids:torch.LongTensor=None,
        labels:torch.LongTensor=None,
        start_pos:int=0 # 训练状态下 start_pos=0
    ):  
        # decoder
        dec_output:Gpt2StackOutput = self.decoder(
            input_ids = input_ids, 
            token_type_ids = token_type_ids, 
            attention_mask = None,
            start_pos = start_pos
        )

        lm_logits: Tensor = self.lm_head(self.ln_final(dec_output.last_hidden_state))
        loss = None

        # label shift right  
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        return CausalLMOutput(
            loss = loss,
            lm_logits = lm_logits,
            last_hidden_state = dec_output.last_hidden_state,
            hidden_states = dec_output.hidden_states,
        )

    def parameter_spilt_or_transpose(self, state_dict: dict) -> dict:
        prefix = self.config.prefix

        for i in range(self.config.num_layers):
            # new_key: old_ley
            # atten层切分+转置
            mapping = {
                "decoder.layers.{}.self_attention.{}.weight": f"{prefix}h.{i}.attn.c_attn.weight",
                "decoder.layers.{}.self_attention.{}.bias": f"{prefix}h.{i}.attn.c_attn.bias",
            }
            
            for new_key, old_key in mapping.items():
                is_weight = old_key.endswith('weight')
                qkv = state_dict.pop(old_key)
                qkv = torch.chunk(qkv, 3, dim=1 if is_weight else 0)
                for i_k, i_v in zip(['q', 'k', 'v'], qkv):
                    state_dict[new_key.format(i, i_k)] = i_v.T if is_weight else i_v
            
            # 其他线性层转置
            mapping = {
                f'decoder.layers.{i}.self_attention_output.o.weight': f'{prefix}h.{i}.attn.c_proj.weight',  # atten_output 层
                f'decoder.layers.{i}.ffn.ff.dense_1.weight': f'{prefix}h.{i}.mlp.c_fc.weight',  # ffn 第一层
                f'decoder.layers.{i}.ffn.ff.dense_2.weight': f'{prefix}h.{i}.mlp.c_proj.weight'   # ffn 第二层
            }
            for new_key, old_key in mapping.items():
                state_dict[new_key] = state_dict.pop(old_key).T
                
        return state_dict


    def variable_mapping(self):
        """
        不同代码参数命名不同，需要做参数映射   new_key: old_key
        """
        prefix = self.config.prefix

        mapping = {
            "embed.token_embeddings.weight": f"{prefix}wte.weight",
            "embed.position_embeddings.weight": f"{prefix}wpe.weight",
            'ln_final.weight': f"{prefix}ln_f.weight",
            'ln_final.bias': f"{prefix}ln_f.bias",
        }

        for i in range(self.config.num_layers):
            mapping.update( 
            {
            f'decoder.layers.{i}.self_attention_output.o.bias': f'{prefix}h.{i}.attn.c_proj.bias',
            f'decoder.layers.{i}.layer_norm.weight': f'{prefix}h.{i}.ln_1.weight',
            f'decoder.layers.{i}.layer_norm.bias': f'{prefix}h.{i}.ln_1.bias',
            f'decoder.layers.{i}.ffn.ff.dense_1.bias': f'{prefix}h.{i}.mlp.c_fc.bias',
            f'decoder.layers.{i}.ffn.ff.dense_2.bias': f'{prefix}h.{i}.mlp.c_proj.bias',
            f'decoder.layers.{i}.ffn.layer_norm.weight': f'{prefix}h.{i}.ln_2.weight',
            f'decoder.layers.{i}.ffn.layer_norm.bias': f'{prefix}h.{i}.ln_2.bias'
            })
        return mapping
        

