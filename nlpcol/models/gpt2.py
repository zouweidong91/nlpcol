
import torch
from nlpcol.layers.embed import GptEmbeddings

from .base import BaseConfig, Decoder
from .gpt import GptModel

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
        self.has_final_layernorm: bool = True


class Gpt2Model(GptModel):
    """
    与gpt1区别，
        gpt2为pre_norm
        gpt2 lm_head之前有final_norm层
    """
    config: Config

    def variable_mapping(self):
        """
        不同代码参数命名不同，需要做参数映射   new_key: old_key
        """
        prefix = self.config.prefix

        mapping = {
            "embed.token_embeddings.weight": f"{prefix}wte.weight",
            "embed.position_embeddings.weight": f"{prefix}wpe.weight",
            'final_norm.weight': f"{prefix}ln_f.weight",
            'final_norm.bias': f"{prefix}ln_f.bias",
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
    