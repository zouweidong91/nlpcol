
import json
import os
import random
from typing import List, Union

import numpy as np
import sentencepiece as spm
import torch
from nlpcol.tokenizers.tokenizers import Tokenizer
from nlpcol.utils._file import FileTool
from torch import Size, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset


def seed_everything(seed:int=42):
    """固定seed

    Args:
        seed (int, optional): 随机种子. Defaults to None.
    """
    print(f"Global seed set to {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed

def get_device():
    if torch.cuda.is_available(): 
        gpu = os.getenv('device') if os.getenv('device') else "0"
    else: 
        gpu = ""

    device = torch.device(f"cuda:{gpu}") if gpu else torch.device("cpu")
    print(f'Using {device}')
    return device

def torch_gc() -> None:
    r"""
    Collects GPU memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

class ListDataset(Dataset):
    """数据是List格式Dataset
    """
    def __init__(self, file_path:str=None, data_list:list=None):
        """_summary_

        Args:
            file_path (str): 待读取文件路径
        """
        if file_path:
            self.data = self.load_data(file_path)
        elif data_list:
            self.data = data_list
        else:
            raise ValueError("the input args error")

    @staticmethod
    def load_data(file_path) -> list:
        """不同文件格式继承ListDataset类时需要重写该函数
        """
        D = []
        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                sample = json.loads(line.strip())
                D.append(sample)
        return D

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    

class IterDataset(IterableDataset):
    """流式读取文件，用于大数据量、多小文件使用时候需要注意steps_per_epoch
    
    Example:
        train_data = IterDataset(file_path)
        print(next(iter(train_data)))
    """
    def __init__(self, file_path:str):
        """_summary_

        Args:
            file_path (str): 待读取文件路径
        """
        self.file_path = file_path

    def __iter__(self):
        return self.load_data(self.file_path)

    @staticmethod
    def load_data(file_path):
        """不同文件格式继承IterDataset类时需要重写该函数
        """
        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                sample = json.loads(line.strip())
                yield sample


def print_paras(model:torch.nn.Module):
    """打印模型参数"""
    total_paras = 0
    for key, value in model.state_dict().items():
    # for para in list(model.parameters()):
        value:torch.Tensor
        print(key, value.shape, value.numel())
        total_paras += value.numel()
    print("total_paras: ", total_paras)


def save_model_parameter(state_dict: dict, save_path: str):
    """保存模型参数信息至txt文件"""
    columns = '\t'.join(["Key", "Shape", "Count"])
    paramrter_list = [columns]
    total_paras = 0 # 计算总参量

    for key, value in state_dict.items():
        total_paras += value.numel()
        paramrter_list.append( '\t'.join([key, str(list(value.shape)), str(value.numel())]))
    paramrter_list.append('总参数量' + str(total_paras))
    FileTool.write(paramrter_list, save_path)

def model_parameter_diff(state_dict_1:dict, state_dict_2:dict=None):
    """模型参数输出到文件，方便对比差异
    """
    os.makedirs('logs/', exist_ok=True)
    save_model_parameter(state_dict_1, "logs/model_para1.txt")
    save_model_parameter(state_dict_2, "logs/model_para2.txt")


def sequence_padding(inputs, length=None, value=0, seq_dims=1, padding_side='right'):
    """将序列padding到同一长度  部分模型如CDial-GPT，pad_id为1，请注意修改value
    padding_side 等价于transformers中 left padding 或者 right padding
    """
    if isinstance(inputs, (np.ndarray, list)):
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if padding_side == 'right':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif padding_side == 'left':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"padding_side" argument must be "right" or "left".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)
    
    elif isinstance(inputs[0], torch.Tensor):
        assert padding_side == 'right', '"padding_side" argument must be "right" when element is torch.Tensor'
        if length is not None:
            inputs = [i[:length] for i in inputs]
        return pad_sequence(inputs, padding_value=value, batch_first=True)
    else:
        raise ValueError('"input" argument must be tensor/list/ndarray.')
    

def text_segmentate(text, maxlen, seps='\n', strips=None):
    """将文本按照标点符号划分为若干个短句
    """
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
        return texts
    else:
        return [text]


def export_vocab_from_spm(spm_path:str, vocab_path:str):
    """从sp_model到处词典，方便查看
    """
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(spm_path)
    vocab_size = sp_model.get_piece_size()

    with open(vocab_path, 'w', encoding='utf8') as f:
        for i in range(vocab_size):
            token = sp_model.id_to_piece(i)
            f.write(
                f"{token}\t{str(i)}\n"
            )


def get_pool_emb(outputs, attention_mask:Tensor, pool_strategy='cls'):
    """获取句向量
    outputs: Model的输出 TODO 类型注释
    attention_mask: [btz, seq_len]
    return: [btz, hidden_size]
    """
    last_hidden = outputs.last_hidden_state
    pooled_output = outputs.pooled_output
    hidden_states: List[Tensor] = outputs.hidden_states

    if pool_strategy == 'pooler':
        return pooled_output

    if pool_strategy in "cls":
        return last_hidden[:, 0]

    elif pool_strategy == "avg":
        return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))

    elif pool_strategy == "avg_first_last":
        first_hidden = hidden_states[1]
        last_hidden = hidden_states[-1]
        pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        return pooled_result

    elif pool_strategy == "avg_top2":
        second_last_hidden = hidden_states[-2]
        last_hidden = hidden_states[-1]
        pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        return pooled_result

    else:
        raise NotImplementedError


def get_special_token(tokenizer: Tokenizer) -> dict:
    """从tokenizer获取三个token_id, 部分模型config.json文件没有配置

    Args:
        tokenizer (Tokenizer): 分词器

    Returns:
        dict: _description_
    """
    
    return {
        "pad_token_id": tokenizer.pad_token_id, 
        "bos_token_id": tokenizer.start_token_id, 
        "eos_token_id": tokenizer.end_token_id
    }



