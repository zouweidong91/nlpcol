
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size, Tensor


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

             self.dense = nn.Linear(hidden_size, heads * head_size * 2, bias=use_bias)
             if self.RoPE:
                self.position
                

             
        

