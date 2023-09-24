import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size, Tensor

random_seed = 42
torch.manual_seed(random_seed)



class LayerTest(unittest.TestCase):

    def test_random(self):
        # self.assertEqual()
        # 每次运行结果一致
        print(torch.rand(1)) # 从0到1的均匀分布中随机抽取
        print(torch.rand(1))
        print(torch.randn(2, 3))  # 值从标准正态分布（均值为0，标准差为1）中随机抽取

    def test_tril(self):
        """下三角矩阵"""
        len = 10
        key_mask = torch.tril(
            torch.ones(len, len, dtype=torch.long), diagonal=0
        )
        key_mask = key_mask.unsqueeze(0).unsqueeze(1)
        print(key_mask)

