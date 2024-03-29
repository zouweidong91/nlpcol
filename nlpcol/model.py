
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size, Tensor
from nlpcol.models import BertModel, BaseModel, T5Model, GptModel


def load_config(config_path) -> dict:
    with open(config_path, 'r') as f:
        config = json.loads(f.read())
        return config


def build_transformer_model(checkpoint_path:str=None, config_path:str=None, model='bert', extra_config:dict={}, **kwargs) -> BaseModel:
    """_summary_

    Args:
        checkpoint_path (str): _description_
        config_path (str): _description_
        model (str, optional): _description_. Defaults to 'bert'.
        extra_config (dict): 如需修改config_path中参数如，dropout_rate，在此传入即可
    """
    config = load_config(config_path)
    config.update(extra_config)

    models = {
        "bert": BertModel,
        "t5": T5Model,
        "GPT": GptModel,
    }

    MODEL = models[model]
    transformer: BaseModel = MODEL(config, **kwargs)

    # 初始化权重 为transformer的每个submodule应用_init_weights函数
    transformer.apply(transformer._init_weights)

    # 权重加载
    if checkpoint_path:
        transformer.load_weight(checkpoint_path)

    return transformer


