
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size, Tensor

# 位置编码  PositionalEmbedding
# 各类位置编码文章参考： https://kexue.fm/archives/8130


class SinusoidalPositionalEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(self, max_position:int, embedding_size:int):
        """_summary_

        Args:
            max_position (int): 位置长度
            embedding_size (int): 位置编码 hidden_size
        """
        super().__init__()
        position_enc = self.get_sinusoid_encoding_table(max_position, embedding_size)
        self.embeddings_table = nn.Embedding.from_pretrained(position_enc, freeze=True)
    
    @staticmethod
    def get_sinusoid_encoding_table(max_position, embedding_size):
        # First part of the PE function: sin and cos argument
        # np.sin(pos/(np.power(10000, 2i/d_model)))  第2i个分量
        # np.cos(pos/(np.power(10000, 2i/d_model)))  第2i+1个分量
        position_enc = torch.tensor(
            [[pos / np.power(10000, 2 * (j // 2) / embedding_size) for j in range(embedding_size)]
            for pos in range(max_position)]
        ).float()
        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, ::2] = torch.sin(position_enc[:, ::2])
        position_enc[:, 1::2] = torch.cos(position_enc[:, 1::2])
        return position_enc
    
    def forward(self, position_ids:Tensor) -> Tensor:
        return self.embeddings_table(position_ids)


class RotaryPositionalEmbedding(nn.Module):
    """旋转式位置编码:https://kexue.fm/archives/8265   无最大长度限制。乘性位置编码，区别于绝对位置编码中的加性
    严格按照原论文的实现方式:  https://nn.labml.ai/transformers/rope/index.html
    TODO llama代码以复数形式实现  对比代码
    """
    def __init__(self, embedding_size:int):
        """
        Args:
            embedding_size (int): 位置编码 hidden_size
        """
        super().__init__()
        self.embedding_size = embedding_size

    def forward(self, qw:Tensor, seq_dim=-2) -> Tensor:
        """
        MultiHeadAttentionLayer中qw是[btz, n_heads, seq_len, head_size]
        GlobalPointer中*转置*后qw是[btz, n_heads, seq_len, head_size]
        """
        # 生成 sinusoidal_pos 编码
        seq_len = qw.shape[seq_dim]
        position_enc = SinusoidalPositionalEmbedding.get_sinusoid_encoding_table(seq_len, self.embedding_size).to(qw.device)

        # cos [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos = position_enc[:, 1::2].repeat_interleave(2, dim=-1) # [seq_len, head_size]
        # sin [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin = position_enc[:, ::2].repeat_interleave(2, dim=-1) # [seq_len, head_size]

        cos = cos[None, None, :, :] # [1, 1, seq_len, head_size]
        sin = sin[None, None, :, :] # [1, 1, seq_len, head_size]

        # to(device)
        # qw2 按照[-q1,q0,-q3,q2......,-qd-1,qd-2]排列
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1).reshape_as(qw)
        return qw * cos + qw2 * sin




