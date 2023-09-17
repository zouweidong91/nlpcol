import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nlpcol.losses import (CrossEntropyLoss, FocalLoss,
                           MultilabelCategoricalCrossentropy)
from torch import Size, Tensor

random_seed = 42
torch.manual_seed(random_seed)

class LayerTest(unittest.TestCase):

    def test_loss(self):
        loss_1 = nn.CrossEntropyLoss(reduction='mean') # 取batch内的平均
        loss_2 = CrossEntropyLoss(reduction='mean')
        loss_3 = FocalLoss()

        input = torch.randn(3, 5, requires_grad=True)
        target = torch.empty(3, dtype=torch.long).random_(5)
        
        output_1 = loss_1(input, target)
        output_2 = loss_2(input, target)
        output_3 = loss_3(input, target)

        print(output_1, output_2, output_3)
        output_2.backward()

        self.assertEqual(output_1, output_2)
        self.assertEqual(output_1, output_2)


    def test_MultilabelCategoricalCrossentropy(self):
        """多标签分类的交叉熵"""
        loss = MultilabelCategoricalCrossentropy()
        batch_size = 3
        num_classes = 5

        # 生成随机的多个目标类索引，每个样例有多个目标类
        num_targets_per_sample = torch.randint(1, num_classes + 1, (batch_size,)) # 定义每个样例的target数量
        target_classes = torch.randint(0, num_classes, (batch_size, max(num_targets_per_sample)))
        # 生成对应的y_true
        y_true = torch.zeros((batch_size, num_classes))
        for i, num_targets in enumerate(num_targets_per_sample):
            y_true[i, target_classes[i, :num_targets]] = 1

        # 生成y_pred
        logits = torch.rand((batch_size, num_classes))
        y_pred = F.softmax(logits, dim=1)

        output = loss(y_pred, y_true)
        print(output)
        


if __name__ == '__main__':
    unittest.main()