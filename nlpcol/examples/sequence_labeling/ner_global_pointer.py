
# bert+crf用来做实体识别
# 数据集：http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# [valid_f1]  token_level: 96.83； entity_level: 95.77


import os

import numpy as np
import torch
import torch.nn as nn
from nlpcol.callback import Callback
from nlpcol.config import TrainConfig, WordDir, device
from nlpcol.layers.layer import GlobalPointer
from nlpcol.losses import MultilabelCategoricalCrossentropy
from nlpcol.model import build_transformer_model
from nlpcol.models.bert import BertModel, BertOutput
from nlpcol.tokenizers import Tokenizer
from nlpcol.trainer import Trainer
from nlpcol.utils.data import PeopleDailyDataset
from nlpcol.utils.snippets import (save_model_parameter, seed_everything,
                                   sequence_padding, torch_gc)
from nlpcol.utils.snippets4examples import model_name_gene, trans_entity2tuple
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

# 固定seed
seed_everything(42)


# 基本训练参数
train_config = TrainConfig(batch_size = 8, epochs = 10, max_seq_length=256)

model_path = "/home/dataset/pretrain_ckpt/bert/chinese_L-12_H-768_A-12"
vocab_path = model_path + "/vocab.txt"
config_path = model_path + "/config.json"
checkpoint_path = model_path + '/pytorch_model.bin.bfr_convert'


# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)


maxlen = 256
batch_size = 16
categories_label2id = {"LOC": 0, "ORG": 1, "PER": 2}
categories_id2label = {value: key for key,value in categories_label2id.items()}
ner_vocab_size = len(categories_label2id)
ner_head_size = 64


def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for d in batch:
        tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
        mapping = tokenizer.rematch(d[0], tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        token_ids = tokenizer.tokens_to_ids(tokens)
        labels = np.zeros((len(categories_label2id), maxlen, maxlen))
        for start, end, label in d[1:]:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                label_id = categories_label2id[label]
                labels[label_id, start, end] = 1
                
        batch_token_ids.append(token_ids)
        batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(sequence_padding(batch_labels, seq_dims=3), dtype=torch.long, device=device)
    return batch_token_ids, batch_labels


# 加载数据集
train_data = PeopleDailyDataset('/home/dataset/corpus/ner/china-people-daily-ner-corpus/example.train')
valid_data = PeopleDailyDataset('/home/dataset/corpus/ner/china-people-daily-ner-corpus/example.dev')


# 定义bert上的模型结构
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert: BertModel = build_transformer_model(checkpoint_path, config_path)
        seed_everything(42)
        self.global_pointer = GlobalPointer(hidden_size=768, heads=ner_vocab_size, head_size=ner_head_size)

    def forward(self, token_ids:torch.Tensor):
        bert_output:BertOutput = self.bert(token_ids)
        sequence_output = bert_output.last_hidden_state # [btz, seq_len, hdsz]
        mask = token_ids.gt(0).long()
        logit = self.global_pointer(sequence_output, mask) # [btz, heads, seq_len, seq_len]
        return logit


class Loss(MultilabelCategoricalCrossentropy):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        shape = y_true.shape
        y_true = y_true.view(shape[0]*shape[1], -1) # [btz*ner_vocab_size, seq_len*seq_len]
        y_pred = y_pred.view(shape[0]*shape[1], -1) # [btz*ner_vocab_size, seq_len*seq_len]
        return super(Loss, self).forward(y_pred, y_true)


def evaluate(data, threshold=0):
    """目标类的分数都大于0，非目标类的分数都小于0"""
    X, Y, Z = 0, 1e-10, 1e-10
    for x_true, label in tqdm(data):
        scores = model(x_true)
        # torch_gc()
        for i, score in enumerate(scores):
            R = set()
            for l, start, end in zip(*np.where(score.cpu() > threshold)):
                R.add((start, end, categories_id2label[l]))  

            T = set()
            for l, start, end in zip(*np.where(label[i].cpu() > 0)):
                T.add((start, end, categories_id2label[l]))
            X += len(R & T)
            Y += len(R)
            Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self, trainer:Trainer, save_path:str, valid_dataloader:DataLoader):
        self.trainer = trainer
        self.save_path = save_path
        self.valid_dataloader = valid_dataloader
        self.best_val_f1 = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        f1, precision, recall = evaluate(valid_dataloader)
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            self.trainer.save_weights(self.save_path)
        print(f'[val] f1: {f1:.5f}, p: {precision:.5f} r: {recall:.5f} best_f1: {self.best_val_f1:.5f}')


# 定义训练流程
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = Loss()
# save_model_parameter(model.state_dict(), 'logs/bertcrf_para1.txt')

save_path = model_name_gene(train_config, 'bert', 'people_daily', prefix='test_gp')
trainer = Trainer(model, train_config, loss_fn, optimizer, collate_fn)


if __name__ == "__main__":
    valid_dataloader = trainer.get_dataloader(valid_data)
    evaluator = Evaluator(trainer, save_path, valid_dataloader)
    trainer.train(train_data, [evaluator])
    
