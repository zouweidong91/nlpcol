
import os
import numpy as np
import random
import torch
import json

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
    print(f'Using {device} device')
    return device


class ListDataset(Dataset):
    """数据是List格式Dataset
    """
    def __init__(self, file_path:str):
        """_summary_

        Args:
            file_path (str): 待读取文件路径
        """
        self.data = self.load_data(file_path)

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






