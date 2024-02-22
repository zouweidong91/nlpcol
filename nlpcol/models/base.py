

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size, Tensor
from nlpcol.layers.layer import LayerNorm, RMSNorm
from nlpcol.generation import GenerationMixin


class BaseConfig:
    # 做代码补全用
    d_ff: int # ffn层的维度
    d_model: int # 模型维度 hidden_size
    # head_size: int # 每个atten_head的宽度
    n_heads: int # atten_heads 数量
    vocab_size: int
    num_layers: int

    dropout_rate: float
    initializer_range: float # 权重初始化标准差值
    layer_norm_eps: float # 很小的正数，确保不会出现除以零
    hidden_act: str # 激活函数

    eos_token_id: int
    bos_token_id: int # bos_token_id 默认为 pad_token_id
    pad_token_id: int
    max_position:int  # 最大位置编码
    max_batch_size:int  # 推理过程中batch_size不能大于此值， kv_cache用

    # decoder 模型配置
    is_decoder: bool # 是否是decoder端

    # 其他额外的默认配置
    use_bias: bool = True # nn.liner是否使用偏置  e.g. t5不使用
    layer_norm_type: str = 'post' # [pre, post] 是否在atten操作之前归一化
    tie_word_embeddings: bool = True # token_embeddings和lm_head权重共享
    



class BaseModel(nn.Module, GenerationMixin):
    def __init__(self, config: dict, **kwargs):
        super().__init__()
        self.skip_init = kwargs.get('skip_init', False) # 是否跳过初始化 TODO 待删除
        self.keep_tokens: list = kwargs.get('keep_tokens', None) # 要保留的词ID列表
        self.update_config(config)

        self.config: BaseConfig

    def update_config(self, config:dict):
        """更新更新config参数
        1、根据keep_tokens更新vocab_size
        """
        if self.keep_tokens is not None:
            config['vocab_size'] = len(self.keep_tokens)
        
        
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
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(self.config.initializer_range)

    def parameter_spilt_or_transpose(self, state_dict: dict) -> dict:
        """
        1、部分transformers模型如gpt, qkv在一个参数矩阵中，需要split
        2、gpt实现时Conv1d和标准nn.Linear输入输出位置相反，需要对weight转置
        """
        return state_dict

    def load_weight(self, checkpoint_path):
        """加载checkpoint_path参数"""
        state_dict:dict = torch.load(checkpoint_path, map_location='cpu')
        state_dict = self.parameter_spilt_or_transpose(state_dict)

        state_dict_new = {}
        mapping = self.variable_mapping()

        for new_key in self.state_dict():
            if new_key in state_dict: # 新旧参数名一样的变量
                state_dict_new[new_key] = self.load_variable(state_dict, new_key)

            elif new_key in mapping:
                old_key = mapping[new_key]
                state_dict_new[new_key] = self.load_variable(state_dict, old_key)
                
            else:
                print(new_key, '忽略')
                # TODO 增加warning日志
                continue

        del state_dict
        self.load_state_dict(state_dict_new, strict=False)
        del state_dict_new

    def variable_mapping(self) -> dict:
        """不同模型要单独实现
        构建moedl变量与checkpoint权重变量间的映射
           new_key: old_key
        """
        return {}

    def load_variable(self, state_dict:dict, name:str):
        """部分参数如embedding或者pos_embedding需要进行修改
        每个模型要单独实现
        name为原始模型的参数名
        """
        variable = state_dict.pop(name)
        if name in [  # TODO gpt 在下游模型中各自继承
            'encoder.embed_tokens.weight',
            'decoder.embed_tokens.weight',
            'lm_head.weight',
        ]:
            return self.load_embedding(variable)
        
        return variable

    def load_embedding(self, embeddings):
        """根据keep_tokens对embedding进行修改"""
        if self.keep_tokens is not None:
            embeddings = embeddings[self.keep_tokens]
        return embeddings


    def tie_weights(self):
        """token_embeddings和lm_head权重共享
        Tie the weights between the input embeddings and the output embeddings.
        # TODO enc dec weight tie
        """
        input_embeddings = self.get_input_embeddings()
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is None: return

        if self.config.tie_word_embeddings: 
            output_embeddings.weight = input_embeddings.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed.token_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    @torch.no_grad()
    def predict(self, X:list):
        # model.eval() 不启用 Batch Normalization 和 Dropout。
        self.eval()
        output = self.forward(*X)
        return output

