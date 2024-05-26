import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nlpcol.utils import logger
from torch import Size, Tensor

random_seed = 42
torch.manual_seed(random_seed)



class LayerTest(unittest.TestCase):

    def test_random(self):
        # self.assertEqual()
        # 每次运行结果一致
        logger.info(torch.rand(1)) # 从0到1的均匀分布中随机抽取  uniform:均匀分布
        logger.info(torch.rand(1))
        logger.info(torch.randn(2, 3))  # 值从标准正态分布（均值为0，标准差为1）中随机抽取

    def test_tril(self):
        """下三角矩阵  tri_lower"""
        len = 10
        key_mask = torch.tril(
            torch.ones(len, len, dtype=torch.long), diagonal=0
        )
        key_mask = key_mask.unsqueeze(0).unsqueeze(1)
        logger.info('\n %s', key_mask)

    def test_spearmanr(self):
        """spearmanr相关系数--向量模型评估时用
        （Spearman's rank correlation coefficient）衡量了两个变量之间的相关性，但不要求这两个变量是线性相关的，而是通过对这两个变量的排名来计算它们的相关性
        """
        from scipy.stats import spearmanr
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
    

    def test_padding(self):
        from torch.nn.utils.rnn import pad_sequence
        a = torch.ones(5)
        b = torch.ones(6)
        c = torch.ones(7)
        pad = pad_sequence([a, b, c])
        logger.info(pad.size())
        
        from nlpcol.utils.snippets import sequence_padding
        input_ids = [
            [1,2,3],
            [1,2,3,4],
            [1,2,3,4,5]
        ]
        pad = sequence_padding(input_ids, padding_side='right')
        logger.info('\n %s', pad)
        # [[1 2 3 0 0]
        # [1 2 3 4 0]
        # [1 2 3 4 5]]

        # left padding: decoder-only 模型 batch generation时使用
        pad = sequence_padding(input_ids, padding_side='left')
        logger.info('\n %s', pad)
        # [[0 0 1 2 3]
        # [0 1 2 3 4]
        # [1 2 3 4 5]]

    def test_vector_multiplication(self):
        # 向量
        u = torch.tensor([1, 2, 3])
        v = torch.tensor([4, 5, 6])

        # wise乘法 *  对应位置元素相乘
        wise = u * v
        logger.info('\n %s', wise)  # tensor([ 4, 10, 18]) 

        # 内积(点积\标量积) dot  内积是两个向量相乘的结果,是一个标量值。
        dot = torch.dot(u, v)
        logger.info('\n %s', dot)  # tensor(32)

        # 外积  outer  w = u ⊗ v 是一个矩阵,其中 w[i, j] = u[i] * v[j]
        # 等价于矩阵乘法： u[:, None] @ v[None, :]
        outer = torch.outer(u, v)
        logger.info('\n %s', outer)  # tensor(32)
        # tensor([[ 4,  5,  6],
        #         [ 8, 10, 12],
        #         [12, 15, 18]])

    def test_matrix_multiplication(self):
        # 矩阵
        A = torch.tensor([[1, 2], [3, 4]])
        B = torch.tensor([[5, 6], [7, 8]])

        # wise乘法  * 运算符执行的是元素wise乘法,对应位置的两个元素相乘。
        C = A * B
        logger.info('\n %s', C)

        # 矩阵乘法 A行与B的列做内积
        D = torch.matmul(A, B)  # 等价于 A @ B 
        logger.info('\n %s', D)

    def test_polar(self):
        # 极坐标转为直角坐标表示
        # https://pytorch.org/docs/stable/generated/torch.polar.html
        import numpy as np
        abs = torch.tensor([1, 2], dtype=torch.float64)
        angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
        # out=abs⋅cos(angle)+abs⋅sin(angle)⋅j
        z = torch.polar(abs, angle) # 长度和角度
        logger.info('\n %s', z)
        # tensor([(0.0000+1.0000j), (-1.4142-1.4142j)], dtype=torch.complex128)


