
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


