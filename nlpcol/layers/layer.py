
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size, Tensor

from .pe import RotaryPositionalEmbedding


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """layernorm层  TODO 后期兼容其他模型
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, input:Tensor) -> Tensor:
        """
        >>> input = torch.randn(2, 3)
        >>> input
        tensor([[-1.1199,  2.0004,  0.7479],
                [ 0.5189, -1.2847,  0.2426]])
        >>> mean = input.mean(-1, keepdim=True)
        >>> mean
        tensor([[ 0.5428],
                [-0.1744]])
        >>> var = (input - mean).pow(2).mean(-1, keepdim=True)
        >>> var
        tensor([[1.6437],
                [0.6291]])
        >>> o = (input - mean) / torch.sqrt(var + 1e-12)
        >>> o
        tensor([[-1.2969,  1.1369,  0.1600],
                [ 0.8741, -1.3998,  0.5258]])
        """
        mean = input.mean(-1, keepdim=True)  # 最后一位计算均值
        var = (input - mean).pow(2).mean(-1, keepdim=True)  # 方差
        o = (input - mean) / torch.sqrt(var + self.eps)

        return self.weight * o + self.bias


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """t5使用的是RMSnorm (Root Mean Square)
        No bias and no subtraction of mean
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, input:Tensor) -> Tensor:
        # 与layerNorm相比，RMS Norm的主要区别在于去掉了减去均值的部分
        var = input.float().pow(2).mean(-1, keepdim=True)
        o = (input.float() / torch.sqrt(var + self.eps)).type_as(input)

        return self.weight * o


class GlobalPointer(nn.Module):
        """全局指针模块
        将序列的每个(start, end)作为整体来进行判断
        以实体为基本单位进行判别，每一个实体都是“n(n+1)/2选k”的多标签分类问题
        参考：https://kexue.fm/archives/8373
        """
        def __init__(self, hidden_size, heads, head_size, RoPE=True, use_bias=True, tril_mask=True):
            super().__init__()
            self.heads = heads
            self.head_size = head_size
            self.RoPE = RoPE
            self.tril_mask = tril_mask

            self.dense = nn.Linear(hidden_size, heads * head_size * 2, bias=use_bias) # 此处当然可以用两个[heads * head_size] self.dense代替
            if self.RoPE:
                self.position_embedding = RotaryPositionalEmbedding(head_size)
            
        def forward(self, inputs:Tensor, mask:Tensor=None) -> Tensor:
            """类似于attention的处理方式
            Args:
                inputs (Tensor): shape=[btz, seq_len, hdsz]
                mask (_type_, optional): shape=[btz, seq_len], padding部分为0
            return: shape=[btz, heads, seq_len, seq_len]
            """
            sequence_output = self.dense(inputs) # [btz, seq_len, heads * head_size * 2]
            # stack dim=-2， 表示stack后-2维度为self.heads
            sequence_output = torch.stack(torch.chunk(sequence_output, self.heads, dim=-1), dim=-2) # [btz, seq_len, heads, head_size*2] 
            qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:] # qk, kw: [btz, seq_len, heads, head_size]

            if self.RoPE:
                qw = self.position_embedding(qw.transpose(1, 2), seq_dim=-2).transpose(1, 2)
                kw = self.position_embedding(kw.transpose(1, 2), seq_dim=-2).transpose(1, 2)

            # 计算内积
            logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw) # [btz, heads, seq_len, seq_len]

            # 排除padding 要分别在最后2个seq_len维度上做mask
            if mask is not None:
                attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3) # [btz, 1, seq_len, 1]
                attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2) # [btz, 1, 1, seq_len]
                logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf')) # 填充为负无穷大，后续计算loss时忽略这部分数据
                logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))

            # 排除下三角
            # torch.tril(torch.ones(3, 3), -1)
            # ([[0., 0., 0.],
            #   [1., 0., 0.],
            #   [1., 1., 0.]])
            if self.tril_mask:
                logits -= torch.tril(torch.ones_like(logits), -1) * 1e12

            # scale返回
            return logits / self.head_size ** 0.5



            