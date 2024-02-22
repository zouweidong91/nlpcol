
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from nlpcol.activations import get_activation
from nlpcol.layers.layer import LayerNorm, RMSNorm
from nlpcol.models.base import BaseConfig as Config
from torch import Size, Tensor


# 前馈层 FeedForward
class DenseActDense(nn.Module):
    """线性变换1 --> 非线性激活函数 --> 线性变换2
    """
    def __init__(self, config:Config):
        super().__init__()
        self.dense_1 = nn.Linear(config.d_model, config.d_ff)
        self.dense_2 = nn.Linear(config.d_ff, config.d_model)
        self.act = get_activation(config.hidden_act)

    def forward(self, x):
        return self.dense_2(self.act(self.dense_1(x)))


class DenseGatedActDense(nn.Module):
    """gated-gelu
    """
    def __init__(self, config: Config):
        super().__init__()
        self.dense_1 = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.dense_2 = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.dense_output = nn.Linear(config.d_ff, config.d_model, bias=config.use_bias)
        self.drop = nn.Dropout(config.dropout_rate)
        self.act = get_activation(config.hidden_act)
        
    def forward(self, x: Tensor) -> Tensor:
        # x shape: (bs, seq_len, d_model)
        x_gelu = self.act(self.dense_1(x))     # (bs, seq_len, d_ff)
        x_linear = self.dense_2(x)
        x = x_gelu * x_linear
        x = self.dense_output(self.drop(x))
        return x


class FFN(nn.Module):
    """
    pre_norm顺序为： LayerNorm --> Drop --> Add
    post_norm顺序为： Drop --> Add --> LayerNorm
    """
    def __init__(self, config: Config):
        super().__init__()
        self.layer_norm_type = config.layer_norm_type

        self.ff = self.get_ff(config)
        self.layer_norm = self.get_ln(config)
        self.dropout = nn.Dropout(config.dropout_rate)

    def get_ff(self, config):
        return DenseActDense(config)

    def get_ln(self, config: Config):
        return LayerNorm(config.d_model, config.layer_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        if self.layer_norm_type == "pre":
            xx = self.ff(self.layer_norm(x))
            return self.dropout(xx) + x

        if self.layer_norm_type == "post":
            xx = self.ff(x)
            return self.layer_norm(self.dropout(xx) + x)
        
