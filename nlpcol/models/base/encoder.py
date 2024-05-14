# encoder基础模型

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from nlpcol.layers.attention import (AttentionOutput, EncAttention,
                                     UnilmAttention)
from nlpcol.layers.ffn import FFN
from nlpcol.layers.layer import LayerNorm
from torch import Size, Tensor

from ._base import BaseConfig as Config


class Block(nn.Module):
    """
    包含模块： Attention --> Feed Forward
    """
    def __init__(self, config: Config):
        super().__init__()

        self.self_attention = self.get_attention(config)
        self.self_attention_output = AttentionOutput(config)
        self.ffn = self.get_ffn(config)

        self.layer_norm_type = config.layer_norm_type
        if self.layer_norm_type == "pre":
            self.layer_norm = self.get_ln(config)

    def get_attention(self, config: Config):
        return UnilmAttention(config) if config.unilm else EncAttention(config)

    def get_ffn(self, config: Config):
        return FFN(config)

    def get_ln(self, config: Config):
        return LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self, 
        hidden_states:Tensor, 
        attention_mask:Tensor=None, 
        start_pos:int=0,
        **kwargs
    ) -> Tensor:
        
        # atten pre_norm
        if self.layer_norm_type == "pre":
            _hidden_states = self.layer_norm(hidden_states)
        else:
            _hidden_states = hidden_states

        # self attention
        context_layer = self.self_attention(
            _hidden_states, _hidden_states, _hidden_states, attention_mask, start_pos=start_pos, **kwargs
        )
        hidden_states = self.self_attention_output(context_layer, hidden_states) # add为标准化之前的hidden_states

        # feedforward
        ffn_output = self.ffn(hidden_states)
        return ffn_output


@dataclass
class StackOutput:
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[List[torch.FloatTensor]] = None
    attention_mask: Optional[torch.LongTensor] = None # 推理时还需要用到


class Stack(nn.Module):
    """多个Block的叠加"""
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList([self.get_block(config) for _ in range(config.num_layers)])

    def get_block(self, config: Config):
        """下游模型用其他组件时继承用"""
        return Block(config)
        
    def forward(
        self, 
        hidden_states:Tensor, 
        attention_mask:Tensor=None,
        start_pos:int=0,
        **kwargs
    ) -> Tensor:
        all_hidden_states = []

        for i, layer_module in enumerate(self.layers):
            all_hidden_states.append(hidden_states)
            hidden_states = layer_module(
                hidden_states, 
                attention_mask,
                start_pos = start_pos,
                **kwargs
            )

        all_hidden_states.append(hidden_states)

        return StackOutput(
            last_hidden_state = hidden_states,
            hidden_states = all_hidden_states,
            attention_mask = attention_mask
        )


