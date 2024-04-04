
# 情感分类任务, 加载bert权重
# valid_acc: 94.72, test_acc: 94.11

import os

import torch
import torch.nn as nn
from nlpcol.callback import Callback
from nlpcol.config import TrainConfig, WordDir, device
from nlpcol.model import build_transformer_model
from nlpcol.models.bert import BertModel, BertOutput
from nlpcol.tokenizers import Tokenizer
from nlpcol.trainer import Trainer
from nlpcol.utils.data import SentimentDataset
from nlpcol.utils.snippets import seed_everything, sequence_padding
from torch.utils.data import DataLoader

# 固定seed
seed_everything(42)


# 基本训练参数
train_config = TrainConfig(batch_size = 16, epochs = 10, max_seq_length=256)

model_path = "/home/dataset/pretrain_ckpt/bert/bert-base-chinese"
vocab_path = model_path + "/vocab.txt"
config_path = model_path + "/config.json"
checkpoint_path = model_path + '/pytorch_model.bin'


# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)


def collate_fn(batch):
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for text, label in batch:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=train_config.max_seq_length)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append([label])

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], batch_labels.flatten()


# 加载数据集
SentimentDataset.maxlen = train_config.max_seq_length
train_data = SentimentDataset('/home/dataset/corpus/sentence_classification/sentiment/sentiment.train.data')
valid_data = SentimentDataset('/home/dataset/corpus/sentence_classification/sentiment/sentiment.valid.data')
test_data = SentimentDataset('/home/dataset/corpus/sentence_classification/sentiment/sentiment.test.data')


# 定义bert上的模型结构
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert: BertModel = build_transformer_model(checkpoint_path, config_path, with_mlm=True, with_pool=True, with_nsp=True)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, token_ids, segment_ids):
        bert_output:BertOutput = self.bert(token_ids, segment_ids)
        output = self.dropout(bert_output.pooled_output)
        output = self.dense(output)
        return output


class Evaluator(Callback):
    def __init__(self, trainer:Trainer, save_path:str, valid_dataloader:DataLoader):
        self.trainer = trainer
        self.save_path = save_path
        self.valid_dataloader = valid_dataloader
        self.best_val_acc = 0.

    def on_epoch_end(self, *args):
        val_acc = self.evaluate()

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.trainer.save_weights(self.save_path)
        print(f'val_acc: {val_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')

    # def on_train_end(self, *args):
    #     self.trainer.save_weights(self.save_path)

    def evaluate(self):
        total, correct = 0, 0
        for batch in self.valid_dataloader:
            X, y = batch
            pred = self.trainer.predict(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += len(y)
    
        return correct / total


# 定义训练流程
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()


model_dir = "/home/tmp/bert/sentence_classification/"
os.makedirs(model_dir, exist_ok=True)
model_name = '{}_{}_{}.bin'.format('test', train_config.batch_size, train_config.epochs)  # 定义模型名字
save_path = os.path.join(model_dir, model_name)
print("saved model path: ", save_path)
trainer = Trainer(model, train_config, loss_fn, optimizer, collate_fn)


valid_dataloader = trainer.get_dataloader(valid_data)
evaluator = Evaluator(trainer, save_path, valid_dataloader)

trainer.train(train_data, [evaluator])


"""
export PYTHONPATH=/home/app/nlpcol CUDA_VISIBLE_DEVICES=1 && python  "/home/app/nlpcol/nlpcol/examples/task_sentiment_classification.py"
"""
