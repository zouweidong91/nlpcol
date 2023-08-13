

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size, Tensor
from nlpcol.layers import LayerNorm

class BaseModel(nn.Module):

    def _init_weights(self, module):
        """初始化权重  大部分神经网络层都是由以下三种层组合成的"""
        if isinstance(module, nn.Linear):
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
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        state_dict_new = {}
        mapping = self.variable_mapping()
        for new_key, old_key in mapping.items():
            if new_key not in self.state_dict():
                print(new_key, '忽略')
                # TODO 增加warning日志
                continue
            
            state_dict_new[new_key] = state_dict[old_key]

        self.load_state_dict(state_dict_new, strict=True)

    def variable_mapping(self):
        """构建moedl变量与checkpoint权重变量间的映射"""
        return {}

    @torch.no_grad()
    def predict(self, X:list):
        # model.eval() 不启用 Latch Normalization 和 Dropout。
        self.eval()
        output = self.forward(*X)
        return output

