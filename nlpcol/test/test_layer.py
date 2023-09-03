import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nlpcol.layers.layer import GlobalPointer
from nlpcol.layers.pe import (RotaryPositionalEmbedding,
                              SinusoidalPositionalEmbedding)
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


if __name__ == '__main__':
    unittest.main()
