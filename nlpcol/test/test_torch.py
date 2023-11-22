import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size, Tensor
from scipy.stats import spearmanr

random_seed = 42
torch.manual_seed(random_seed)



class LayerTest(unittest.TestCase):

    def test_random(self):
        # self.assertEqual()
        # 每次运行结果一致
        print(torch.rand(1)) # 从0到1的均匀分布中随机抽取  uniform:均匀分布
        print(torch.rand(1))
        print(torch.randn(2, 3))  # 值从标准正态分布（均值为0，标准差为1）中随机抽取

    def test_tril(self):
        """下三角矩阵  tri_lower"""
        len = 10
        key_mask = torch.tril(
            torch.ones(len, len, dtype=torch.long), diagonal=0
        )
        key_mask = key_mask.unsqueeze(0).unsqueeze(1)
        print(key_mask)

    def test_spearmanr(self):
        """spearmanr相关系数--向量模型评估时用
        （Spearman's rank correlation coefficient）衡量了两个变量之间的相关性，但不要求这两个变量是线性相关的，而是通过对这两个变量的排名来计算它们的相关性
        """
        x = [3, 2, 3, 4, 5]
        y = [4, 4, 5, 6, 7]

        # 计算Spearman相关系数
        corr, p_value = spearmanr(x, y)
        print("Spearman's correlation coefficient:", corr)


        # numpys实现 结果有差异？
        # 对 x 和 y 进行排名
        rank_x = np.argsort(np.argsort(x))
        rank_y = np.argsort(np.argsort(y))
        print(rank_x, rank_y)

        # 计算排名之间的差异
        rank_diff = rank_x - rank_y
        print(rank_diff)

        # 计算 Spearman 等级相关系数
        n = len(x)
        r_s = 1 - 6 * np.sum(rank_diff**2) / (n * (n**2 - 1))
        print("Spearman's correlation coefficient:", r_s)
    


