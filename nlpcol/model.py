
import json

from nlpcol.models import (BaseConfig, BaseModel, BertConfig, BertModel,
                           Gpt2Config, Gpt2Model, GptConfig, GptModel,
                           T5Config, T5Model)


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
        extra_config (dict): 如需修改config_path中参数如：dropout_rate，pad_token_id等 在此传入即可 NOTE

        config相关参数透传说明：
            step1: 加载config.json文件 
            step2: extra_config，添加或覆盖config参数
            step3: Config实例化
            step4: BaseModel中对部分参数进行更新，update_config
    """
    config = load_config(config_path)
    config.update(extra_config)

    models = {
        "bert": (BertModel, BertConfig),
        "t5": (T5Model, T5Config),
        "GPT": (GptModel, GptConfig),
        "GPT2": (Gpt2Model, Gpt2Config),
    }

    MODEL, CONFIG = models[model]
    _config: BaseConfig = CONFIG(**config)
    transformer: BaseModel = MODEL(_config, **kwargs)

    # 初始化权重 为transformer的每个submodule应用_init_weights函数
    transformer.apply(transformer._init_weights)

    # 权重加载
    if checkpoint_path:
        transformer.load_weight(checkpoint_path)

    return transformer


