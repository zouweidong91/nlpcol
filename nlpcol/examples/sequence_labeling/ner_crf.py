
# 情感分类任务, 加载bert权重
# valid_acc: 94.72, test_acc: 94.11

import os

import numpy as np
import torch
import torch.nn as nn
from nlpcol.callback import Callback
from nlpcol.config import WordDir, device
from nlpcol.crf import CRF
from nlpcol.model import build_transformer_model
from nlpcol.models.bert import BertModel, BertOutput
from nlpcol.tokenizers import Tokenizer
from nlpcol.trainer import TrainConfig, Trainer
from nlpcol.utils.data import PeopleDailyDataset
from nlpcol.utils.snippets import seed_everything, sequence_padding
from tqdm import tqdm

# 固定seed
seed_everything(42)


# 基本训练参数
train_config = TrainConfig(batch_size = 16, epochs = 10, max_seq_length=256)

model_path = "/home/dataset/pretrain_ckpt/bert/chinese_L-12_H-768_A-12"
vocab_path = model_path + "/vocab.txt"
config_path = model_path + "/config.json"
checkpoint_path = model_path + '/pytorch_model.bin.bfr_convert'


# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)


maxlen = 256
batch_size = 16
categories = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']
categories_id2label = {i: k for i, k in enumerate(categories)}
categories_label2id = {k: i for i, k in enumerate(categories)}


def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for d in batch:
        tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
        mapping = tokenizer.rematch(d[0], tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        token_ids = tokenizer.tokens_to_ids(tokens)
        labels = np.zeros(len(token_ids))
        for start, end, label in d[1:]:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                labels[start] = categories_label2id['B-'+label]
                labels[start + 1:end + 1] = categories_label2id['I-'+label]
        batch_token_ids.append(token_ids)
        batch_labels.append(labels)
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(sequence_padding(batch_labels), dtype=torch.long, device=device)
    return batch_token_ids, batch_labels


# 加载数据集
train_data = PeopleDailyDataset('/home/dataset/corpus/ner/china-people-daily-ner-corpus/example.train')
valid_data = PeopleDailyDataset('/home/dataset/corpus/ner/china-people-daily-ner-corpus/example.dev')


# 定义bert上的模型结构
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert: BertModel = build_transformer_model(checkpoint_path, config_path, with_mlm=True, with_pool=True, with_nsp=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, len(categories))
        self.crf = CRF(len(categories))

    def forward(self, token_ids:torch.Tensor):
        bert_output:BertOutput = self.bert(token_ids)
        sequence_output = bert_output.last_hidden_state # [btz, seq_len, hdsz]
        output = self.fc(sequence_output)  # [btz, seq_len, tag_size]
        attention_mask = token_ids.gt(0).long()
        return output, attention_mask

    @torch.no_grad()
    def predict(self, token_ids):
        self.eval()
        emission_score, attention_mask = self.forward(token_ids)
        best_path = self.crf.decode(emission_score, attention_mask)  # [btz, seq_len]
        best_path = torch.tensor(sequence_padding(best_path), device=device)
        return best_path


class Loss(nn.Module):
    def forward(self, outputs, labels):
        output, attention_mask = outputs
        # return model.crf(output, labels, attention_mask.byte())
        return model.crf(*outputs, labels)


def evaluate(data):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    X2, Y2, Z2 = 1e-10, 1e-10, 1e-10
    for token_ids, label in tqdm(data):
        scores = model.predict(token_ids)  # [btz, seq_len]
        attention_mask = label.gt(0)

        # token粒度
        X += (scores.eq(label) * attention_mask).sum().item()
        Y += scores.gt(0).sum().item()
        Z += label.gt(0).sum().item()

        # entity粒度
        entity_pred = trans_entity2tuple(scores)
        entity_true = trans_entity2tuple(label)
        X2 += len(entity_pred.intersection(entity_true))
        Y2 += len(entity_pred)
        Z2 += len(entity_true)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    f2, precision2, recall2 = 2 * X2 / (Y2 + Z2), X2/ Y2, X2 / Z2
    return f1, precision, recall, f2, precision2, recall2


def trans_entity2tuple(scores):
    '''把tensor转为(样本id, start, end, 实体类型)的tuple用于计算指标
    '''
    batch_entity_ids = set()
    for i, one_samp in enumerate(scores):
        entity_ids = []
        for j, item in enumerate(one_samp):
            flag_tag = categories_id2label[item.item()]
            if flag_tag.startswith('B-'):  # B
                entity_ids.append([i, j, j, flag_tag[2:]])
            elif len(entity_ids) == 0:
                continue
            elif (len(entity_ids[-1]) > 0) and flag_tag.startswith('I-') and (flag_tag[2:]==entity_ids[-1][-1]):  # I
                entity_ids[-1][-2] = j
            elif len(entity_ids[-1]) > 0:
                entity_ids.append([])

        for i in entity_ids:
            if i:
                batch_entity_ids.add(tuple(i))
    return batch_entity_ids

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        f1, precision, recall, f2, precision2, recall2 = evaluate(valid_dataloader)
        if f2 > self.best_val_f1:
            self.best_val_f1 = f2
            # model.save_weights('best_model.pt')
        print(f'[val-token  level] f1: {f1:.5f}, p: {precision:.5f} r: {recall:.5f}')
        print(f'[val-entity level] f1: {f2:.5f}, p: {precision2:.5f} r: {recall2:.5f} best_f1: {self.best_val_f1:.5f}\n')


# 定义训练流程
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = Loss()


model_dir = "/home/tmp/bert/people_daily/"
os.makedirs(model_dir, exist_ok=True)
model_name = '{}_{}_{}.bin'.format('test', train_config.batch_size, train_config.epochs)  # 定义模型名字
save_path = os.path.join(model_dir, model_name)
print("saved model path: ", save_path)
trainer = Trainer(model, train_config, loss_fn, optimizer, collate_fn)


valid_dataloader = trainer.get_dataloader(valid_data)
evaluator = Evaluator()

trainer.train(train_data, [evaluator])


"""
export PYTHONPATH=/home/app/nlpcol CUDA_VISIBLE_DEVICES=1 && python  "/home/app/nlpcol/nlpcol/examples/sequence_labeling/task_sequence_labeling_ner_crf.py"
"""
