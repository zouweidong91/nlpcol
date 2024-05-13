import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nlpcol.layers.layer import GlobalPointer, RMSNorm
from nlpcol.layers.pe import (RelativePositionalT5, RotaryPositionalEmbedding,
                              SinusoidalPositionalEmbedding)
from nlpcol.utils import logger
from torch import Size, Tensor

random_seed = 42
torch.manual_seed(random_seed)

class LayerTest(unittest.TestCase):

    def test_SinusoidalPositionalEmbedding(self):
        """正余弦位置编码"""
        max_position = 2
        embedding_size = 4
        position_embedding = SinusoidalPositionalEmbedding(max_position, embedding_size)
        print(position_embedding.embeddings_table.weight)

        position_ids = torch.arange(0, max_position, dtype=torch.long)
        output = position_embedding(position_ids)
        print(output)

    def test_RotaryPositionalEmbedding(self):
        """旋转式位置编码"""
        torch.manual_seed(42)
        r = RotaryPositionalEmbedding(64)
        qw = torch.randn(12, 3, 100, 64)
        o = r(qw)
        print(o[0][0][0])
        print(o.shape)


    def test_Attention(self):
        """自注意力"""


    def test_attentionMask(self):
        """GlobalPointer attentionMask测试"""
        torch.manual_seed(42)

        # 创建一个示例的注意力分数矩阵（logits）
        btz = 2  # 批量大小
        seq_len = 5  # 序列长度
        logits = torch.rand(btz, seq_len, seq_len)  # 假设随机生成注意力分数

        # 创建一个示例的mask，其中某些位置是padding位置
        mask = torch.ones(btz, seq_len)
        mask[0][3:] = 0  # 在第一个样本中，后两个位置是padding
        mask[1][2:] = 0  # 在第二个样本中，后三个位置是padding

        # 根据您提供的代码进行注意力屏蔽操作
        attention_mask1 = 1 - mask.unsqueeze(1)  # [btz, 1, seq_len, 1]
        attention_mask2 = 1 - mask.unsqueeze(2)  # [btz, 1, 1, seq_len]

        # 使用屏蔽操作将注意力分数矩阵中的一些位置设为负无穷大
        logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
        logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))

        # 打印结果
        print("注意力分数矩阵（logits）:")
        print(logits)
    
    def test_attention_bias(self):
        # attention_bias 用以控制注意力机制中，不同段落能关注到哪些信息
        # 假设我们有一个长度为5的序列，其中包含两个不同的段落  第一个段落包含3个元素，第二个段落包含2个元素
        segment_ids = torch.tensor([[
            0, 0, 0,   # 第一个段落的ID，连续3个0
            1, 1      # 第二个段落的ID，连续2个1
        ]])

        # 计算segment_ids在dim=1上的累积和
        cumsum_segment_ids = torch.cumsum(segment_ids, dim=1)
        # tensor([[0, 0, 0, 1, 2]])

        # 将累积和张量扩展为三维张量，并进行比较操作
        attention_bias = (cumsum_segment_ids.unsqueeze(1) <= cumsum_segment_ids.unsqueeze(2))
        logger.info(cumsum_segment_ids.unsqueeze(1))
        # tensor([[[0, 0, 0, 1, 2]]])

        logger.info(cumsum_segment_ids.unsqueeze(2))
        # tensor([[[0],
        #         [0],
        #         [0],
        #         [1],
        #         [2]]])

        logger.info(attention_bias)
        # tensor([[[1, 1, 1, 0, 0],
        #         [1, 1, 1, 0, 0],
        #         [1, 1, 1, 0, 0],
        #         [1, 1, 1, 1, 0],
        #         [1, 1, 1, 1, 1]]])


    def test_GlobalPointer(self):
        torch.manual_seed(42)
        hidden_size = 7
        heads = 3
        head_size = 4
        gp = GlobalPointer(hidden_size, heads, head_size, RoPE=True)
        
        btz = 2
        seq_len = 5
        inputs = torch.randn(btz, seq_len, hidden_size)
        mask = torch.ones(btz, seq_len)
        mask[0][3:] = 0
        mask[1][2:] = 0
        o:Tensor = gp(inputs, mask)
        self.assertListEqual(list(o.shape), [btz, heads, seq_len, seq_len])

    def test_RMSNorm(self):
        input = torch.randn(2,4)
        rms_norm = RMSNorm(4)
        o = rms_norm(input)
        print(o)


    def test_RelativePositionalT5(self):
        pe = RelativePositionalT5(10, 10, 32)
        o = pe.relative_position
        print(o)



if __name__ == '__main__':
    unittest.main()
