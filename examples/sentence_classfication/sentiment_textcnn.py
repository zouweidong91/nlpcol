

import os

import torch
import torch.nn as nn
from nlpcol.callback import Callback
from nlpcol.config import WordDir, device
from nlpcol.models.textcnn import Config, TextCNN
from nlpcol.trainer import TrainConfig, Trainer
from nlpcol.utils.snippets import ListDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from nlpcol.tokenizers import SampleTokenizer


# 基本训练参数
train_config = TrainConfig(batch_size = 128, epochs = 5, max_seq_length=32)


class THUDataset(ListDataset):
    @staticmethod
    def load_data(file_path) -> list:
        """不同文件格式继承ListDataset类时需要重写该函数
        """
        D = []
        with open(file_path, 'r', encoding='utf8') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line: continue
                D.append(line)
        return D

train_data = THUDataset('/home/dataset/nlpcol/THUCNews/train.txt')
valid_data = THUDataset('/home/dataset/nlpcol/THUCNews/dev.txt')
model_dir = "/home/tmp/textcnn/THUCNews/"
os.makedirs(model_dir, exist_ok=True)
vocab_path = model_dir + "vocab.pkl"

tokenizer = SampleTokenizer(train_data, vocab_path)


def collate_fn(batch:list):
    """不同的数据集要分别实现该函数
    """
    x, y = [], []

    for sample in batch:
        content, label = sample.split('\t')
        token_ids = tokenizer.encode(content, train_config.max_seq_length)

        x.append(token_ids)
        y.append(int(label))

    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return (x, ), y


# 评估函数
# TODO 不同任务做一个模板 不要写来写去
class Evaluate(Callback):
    def __init__(self, trainer:Trainer, save_path:str, valid_dataloader:DataLoader):
        self.trainer = trainer
        self.save_path = save_path
        self.valid_dataloader = valid_dataloader
        self.best_val_acc = 0.

    def on_epoch_end(self, *args):
        val_acc = self.evaluate()

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model.pt')
        print(f'val_acc: {val_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')

    def on_train_end(self, *args):
        self.trainer.save_weights(self.save_path)

    def evaluate(self):
        total, correct = 0, 0
        for batch in self.valid_dataloader:
            X, y = batch
            pred = self.trainer.predict(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += len(y)
    
        return correct / total


# 定义训练流程
config = Config()
model = TextCNN(config, tokenizer.vocab_size, 10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)
loss_fn = nn.CrossEntropyLoss()


model_name = '{}_{}_{}.bin'.format('out', train_config.batch_size, train_config.epochs)  # 定义模型名字
save_path = os.path.join(model_dir, model_name)
print("saved model path: ", save_path)
trainer = Trainer(model, train_config, loss_fn, optimizer, collate_fn)

valid_dataloader = trainer.get_test_dataloader(valid_data)
evaluate = Evaluate(trainer, save_path, valid_dataloader)

trainer.train(train_data, [evaluate])

