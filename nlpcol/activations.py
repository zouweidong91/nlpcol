
import math
import torch
from torch import nn
from packaging import version


def _gelu_python(x):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


if version.parse(torch.__version__) < version.parse("1.4"):
    gelu = _gelu_python
else:
    gelu = nn.functional.gelu


ACT2FN = {
    "relu": nn.functional.relu,  # max(0, x)
    "gelu": gelu,
    "softmax": nn.Softmax(dim=-1)
}

def get_activation(activation_str:str):
    if activation_str in ACT2FN:
        return ACT2FN[activation_str]
    else:
        raise KeyError(f"function {activation_str} not found in ACT2FN mapping {list(ACT2FN.keys())}")