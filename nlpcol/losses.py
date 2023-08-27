import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size, Tensor


class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss简单实现，只是为了和其他loss做对比
    loss的reduction默认为mean, 即取一个batch内的loss平均值
    """
    def __init__(self, weight:Tensor=None, ignore_index:int=-100, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        target标签值class，并不参与直接计算，而是作为一个索引,索引对象为实际类别
        假设输入target = torch.tensor([0]) input=torch.Tensor([[-0.7715, -0.6205,-0.2562]])， target为0
        loss = -x[0]+log(exp(x[0])+exp(x[1])+exp(x[2])) 

        Args:
            input (Tensor): shape=[N, C]
            target (Tensor): shape=[N,]
        """
        # LogSoftmax能够解决函数上溢和下溢的问题,加快运算速度,提高数据稳定性。
        logpt = F.log_softmax(input, dim=1)
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        return loss


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation
    ref: https://zhuanlan.zhihu.com/p/49981234
    目的是通过减少易分类样本的权重，从而使得模型在训练时更专注于难分类的样本。
    Focal Loss 公式：FL(p_t) = -alpha_t * (1 - p_t) ^ gamma * log(p_t)
    '''
    def __init__(self, gamma=2, weight:Tensor=None, ignore_index:int=-100):
        """
        Args:
            gamma (int, optional): _description_. Defaults to 2.
            weight (Tensor, optional):  a manual rescaling weight given to each
                class. If given, has to be a Tensor of size `C`
            ignore_index (int, optional): _description_. Defaults to -100.
        """
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): shape=[N, C]
            target (Tensor): shape=[N,]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt) # pt 等价于 F.softmax(input, dim=1)
        logpt = (1 - pt) ** self.gamma *logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss
        
class MultilabelCategoricalCrossentropy(nn.Module):
    """多标签分类的交叉熵；
    目标类的分数都大于0，非目标类的分数都小于0
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！
        预测阶段则输出y_pred大于0的类。
    ref: https://kexue.fm/archives/7359
    """
    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Args:
            y_true (Tensor): shape=[..., num_classes]
            y_pred (Tensor): shape=[..., num_classes]
        """
        # 将y_pred调整为目标类分数大于0，非目标类分数小于0的形式(注：在实际公式中，目标分值前有'-')
        y_pred = (1 - 2*y_true) * y_pred

        # 将非目标类位置的分数设置为负无穷大，以确保在计算logsumexp时不考虑这些位置
        y_pred_pos = y_pred - (1-y_true) * 1e12 # 非pos位置位负无穷大
        y_pred_neg = y_pred - y_true * 1e12 # 非neg位置位负无穷大

        y_pred_pos = torch.cat([y_pred_pos, torch.zeros_like(y_pred_pos[..., :1])], dim=-1) # 最后一位加0
        y_pred_neg = torch.cat([y_pred_neg, torch.zeros_like(y_pred_neg[..., :1])], dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1) # [..., num_classes] --> [...]
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        return (pos_loss + neg_loss).mean()
        


