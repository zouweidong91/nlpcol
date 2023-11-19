# SimCSE 中文测试
# bert4keras链接：https://kexue.fm/archives/8348
# https://github.com/bojone/SimCSE
# https://github.com/princeton-nlp/SimCSE/tree/main
# 直接把Dropout当作数据扩增
# |     solution    |   ATEC  |  BQ  |  LCQMC  |  PAWSX  |  STS-B  |
# |      SimCSE     |  33.90  
# ATEC 0.33648 epoch3

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from nlpcol.callback import Callback
from nlpcol.config import TrainConfig, WordDir, device
from nlpcol.model import build_transformer_model
from nlpcol.models.bert import BertModel, BertOutput
from nlpcol.tokenizers import Tokenizer
from nlpcol.trainer import Trainer
from nlpcol.utils.snippets import (ListDataset, get_pool_emb,
                                   save_model_parameter, seed_everything,
                                   sequence_padding, torch_gc)
from nlpcol.utils.snippets4examples import model_name_gene, trans_entity2tuple
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

# 固定seed
seed_everything(42)

model_type, pooling, task_name, dropout_rate = 'BERT', 'cls', 'ATEC', 0.3  # debug使用

assert model_type in {'BERT', 'RoBERTa', 'NEZHA', 'RoFormer', 'SimBERT'}
assert pooling in {'first-last-avg', 'last-avg', 'cls', 'pooler'}
assert task_name in {'ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B'}
if model_type in {'BERT', 'RoBERTa', 'SimBERT'}:
    model_name = 'bert'
elif model_type in {'RoFormer'}:
    model_name = 'roformer'
elif model_type in {'NEZHA'}:
    model_name = 'nezha'


# 基本训练参数
train_config = TrainConfig(batch_size = 32, epochs = 10, max_seq_length=128)

model_path = "/home/dataset/pretrain_ckpt/bert/chinese_L-12_H-768_A-12"
vocab_path = model_path + "/vocab.txt"
config_path = model_path + "/config.json"
checkpoint_path = model_path + '/pytorch_model.bin.bfr_convert'


# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
data_path = '/home/dataset/corpus/sentence_embedding/'
all_names = [f'{data_path}{task_name}/{task_name}.{f}.data' for f in ['train', 'valid', 'test']]
print(all_names)


def load_data(filenames):
    """加载数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = l.strip().split('\t')
                if len(l) == 3:
                    D.append((l[0], l[1], float(l[2])))
    return D


all_texts = load_data(all_names)
train_texts = [j for i in all_texts for j in i[:2]]


if task_name != 'PAWSX':
    np.random.shuffle(train_texts)
    train_texts = train_texts[:10000]

# 加载训练数据集
def collate_fn(batch):
    """同一个句子分两次送入模型，每次drop后向量编码会不同，以此句子对作为正样本
    [a,a,b,b,c,c]
    """
    text_list = [[] for _ in range(2)]
    for text in batch:
        token_ids = tokenizer.encode(text, maxlen=128)[0]
        text_list[0].append(token_ids)
        text_list[1].append(token_ids)
    for i, texts in enumerate(text_list):
        text_list[i] = torch.tensor(sequence_padding(texts), dtype=torch.long, device=device)
    labels = torch.arange(text_list[0].size(0), device=device)
    return text_list, labels

# 加载测试数据集
def collate_fn_eval(batch):
    texts_list = [[] for _ in range(2)]
    labels = []
    for text1, text2, label in batch:
        texts_list[0].append(tokenizer.encode(text1, maxlen=128)[0])
        texts_list[1].append(tokenizer.encode(text2, maxlen=128)[0])
        labels.append(label)
    for i, texts in enumerate(texts_list):
        texts_list[i] = torch.tensor(sequence_padding(texts), dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.float, device=device) # batch内与自身是正样本，其他为负样本
    return texts_list, labels


# 加载数据集
train_dataset = ListDataset(data_list=train_texts)
valid_dataloader = DataLoader(ListDataset(data_list=all_texts), batch_size=32, collate_fn=collate_fn_eval)


# 定义bert上的模型结构
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        extra_config = {"hidden_dropout_prob": 0.3} # hidden_dropout_prob过低，会导致正样本对差异过小
        self.bert: BertModel = build_transformer_model(checkpoint_path, config_path, with_mlm=True, with_pool=True, with_nsp=True, 
            extra_config=extra_config)
        self.scale = 20.0
        self.pool_strategy = "cls"

    def forward(self, *token_ids_list):
        reps = []
        for token_ids in token_ids_list:
            bert_output:BertOutput = self.bert(token_ids)
            rep = get_pool_emb(bert_output, token_ids.gt(0).long(), self.pool_strategy)
            reps.append(rep)

        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:], dim=0) # 负样本拼接
        scores = self.cos_sim(embeddings_a, embeddings_b) * self.scale  # [btz, btz]
        return scores

    @torch.no_grad()
    def encoder(self, token_ids):
        self.eval()
        bert_output:BertOutput = self.bert(token_ids)
        output = get_pool_emb(bert_output, token_ids.gt(0).long(), self.pool_strategy)
        return output

    @staticmethod
    def cos_sim(a, b):
        """cosine相似度"""
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1) # l2归一化 即 除以向量模
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return a_norm @ b_norm.T
        

class Evaluator(Callback):
    """评估与保存
    """

    def __init__(self, trainer:Trainer, save_path:str, valid_dataloader:DataLoader):
        self.trainer = trainer
        self.save_path = save_path
        self.valid_dataloader = valid_dataloader
        self.best_val_consine = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_consine = self.evaluate(valid_dataloader)
        if val_consine > self.best_val_consine:
            self.best_val_consine = val_consine
            # self.trainer.save_weights(self.save_path)
        print(f'val_consine: {val_consine:.5f}, best_val_consine: {self.best_val_consine:.5f}\n')

    def evaluate(self, dataloader):
        # 模型预测
        # 标准化，相似度，相关系数
        sims_list, labels = [], []
        for (a_token_ids, b_token_ids), label in tqdm(dataloader):
            a_vecs = model.encoder(a_token_ids)
            b_vecs = model.encoder(b_token_ids)
            a_vecs = torch.nn.functional.normalize(a_vecs, p=2, dim=1).cpu().numpy()
            b_vecs = torch.nn.functional.normalize(b_vecs, p=2, dim=1).cpu().numpy()
            sims = (a_vecs * b_vecs).sum(axis=1)
            sims_list.append(sims)
            labels.append(label.cpu().numpy())

        corrcoef = scipy.stats.spearmanr(np.concatenate(labels), np.concatenate(sims_list)).correlation
        return corrcoef

# 定义训练流程
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

save_path = model_name_gene(train_config, 'bert', 'ATEC', prefix='test_unsup_simcse')
trainer = Trainer(model, train_config, loss_fn, optimizer, collate_fn)


if __name__ == "__main__":
    evaluator = Evaluator(trainer, save_path, valid_dataloader)
    trainer.train(train_dataset, [evaluator])
    

