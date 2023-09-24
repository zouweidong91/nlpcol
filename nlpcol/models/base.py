

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size, Tensor
from nlpcol.layers.layer import LayerNorm

class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.skip_init = kwargs.get('skip_init', False)
        
    def _init_weights(self, module:nn.Module):
        """初始化权重  大部分神经网络层都是由以下三种层组合成的
        不同的初始化策略，微调阶段影响不是很大
        """
        if self.skip_init: # 跳过初始化
            module.to_empty(device='cpu')

        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_() # 默认就是0，此处应该多余了 TODO
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def load_weight(self, checkpoint_path):
        """加载checkpoint_path参数"""
        state_dict:dict = torch.load(checkpoint_path, map_location='cpu')
        state_dict_new = {}
        mapping = self.variable_mapping()

        for new_key in self.state_dict():
            if new_key in state_dict: # 新旧参数名一样的变量
                state_dict_new[new_key] = state_dict.pop(new_key)

            elif new_key in mapping:
                old_key = mapping[new_key]
                state_dict_new[new_key] = state_dict.pop(old_key)
                
            else:
                print(new_key, '忽略')
                # TODO 增加warning日志
                continue


        self.load_state_dict(state_dict_new, strict=True)

    def variable_mapping(self) -> dict:
        """构建moedl变量与checkpoint权重变量间的映射
           new_key: old_key
        """
        return {}

    @torch.no_grad()
    def predict(self, X:list):
        # model.eval() 不启用 Batch Normalization 和 Dropout。
        self.eval()
        output = self.forward(*X)
        return output

