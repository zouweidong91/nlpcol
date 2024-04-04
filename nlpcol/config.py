
from dataclasses import dataclass, field
from typing import Optional

from nlpcol.utils.snippets import get_device


class WordDir:
    dataset_dir = '/home/dataset'
    tmp_dir = '/home/tmp/'  # 临时模型文件目录

device = get_device()



@dataclass
class TrainConfig:
    """训练超参数配置
    """
    batch_size: Optional[int] = field(default=8, metadata={"help": "batch_size."})
    epochs: Optional[int] = field(default=3, metadata={"help": "epochs."})
    max_seq_length: Optional[int] = field(default=128, metadata={"help": "max_seq_length."})

