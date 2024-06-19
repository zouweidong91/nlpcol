
from typing import Optional

import numpy as np
import torch
import math
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
    llama中是基于复数乘法来实现，参见 https://github.com/meta-llama/llama3/blob/main/llama/model.py#L65
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


class RelativePositionalT5(nn.Module):
    """Google T5的相对位置编码
    来自论文：https://arxiv.org/abs/1910.10683
    本质是在attention score上加一个可训练的偏置项
    """
    def __init__(self, qlen, klen, num_buckets:int, max_distance:int=128, is_decoder=False):
        """
        Args:
            qlen (_type_): query length
            klen (_type_): key length
            num_buckets (int): 相对位置编码分桶数量
            max_distance(int): 相对位置编码的最大相对距离
            is_decoder (bool, optional): 是否是decoder. Defaults to False.
        """
        super().__init__()
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position # (qlen, klen)
        self.relative_position = self._relative_position_bucket(
            relative_position,
            bidirectional = not is_decoder,
            num_buckets = num_buckets,
            max_distance = max_distance,
        )

    def forward(self, qlen, klen) -> Tensor:
        return self.relative_position[:qlen, :klen]

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        比较邻近的位置（0～7），需要比较得精细一些，所以给它们都分配一个独立的位置编码，至于稍远的位置（比如8～11），我们不用区分得太清楚，
        所以它们可以共用一个位置编码，距离越远，共用的范围就可以越大，直到达到指定范围再clip
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets


