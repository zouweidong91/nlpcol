# bert做Seq2Seq任务，采用UNILM方案
# 介绍链接：https://kexue.fm/archives/6933
# 数据集：https://github.com/CLUEbenchmark/CLGE 中的CSL数据集
# 补充了评测指标bleu、rouge-1、rouge-2、rouge-l

import json
import os

import torch
import torch.nn as nn
from nlpcol.callback import Callback
from nlpcol.config import TrainConfig
from nlpcol.model import build_transformer_model
from nlpcol.models.bert import BertModel, BertOutput
from nlpcol.tokenizers import Tokenizer, load_vocab
from nlpcol.trainer import Trainer
from nlpcol.utils.snippets import (ListDataset, seed_everything,
                                   sequence_padding, text_segmentate)
from nlpcol.utils.snippets4examples import model_name_gene
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge  # pip install rouge
from tqdm import tqdm

seed_everything(42)
# 基本参数
maxlen = 256
batch_size = 16
epochs = 50
steps_per_epoch = None

# 基本训练参数
train_config = TrainConfig(batch_size = batch_size, epochs = epochs, max_seq_length=maxlen)

model_path = "/home/dataset/pretrain_ckpt/bert/bert-base-chinese"
dict_path = model_path + "/vocab.txt"
config_path = model_path + "/bert4torch_config.json"
checkpoint_path = model_path + '/pytorch_model.bin'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据
        单条格式：(标题, 正文)
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                try:
                    title, content = l['title'], l['abst']
                    D.append((title, content))
                except:
                    # print(l)
                    pass
        return D


# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)

print(len(token_dict))
print(len(keep_tokens))

tokenizer = Tokenizer(dict_path, token_dict, do_lower_case=True)

def collate_fn(batch):
    """单条样本格式：[CLS]文章[SEP]标题[SEP]
        tokenizer.encode('content', 'title', maxlen=64)
        [[2, 9329, 3, 9853, 3], [0, 0, 0, 1, 1]]
    """
    batch_token_ids, batch_segment_ids = [], []
    for title, content in batch:
        token_ids, segment_ids = tokenizer.encode(content, title, maxlen=maxlen)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    labels = batch_token_ids * batch_segment_ids # segment_ids，刚好指示了要预测的部分
    labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100) # padding_id替换为-100，不参与loss计算
    return {
        "input_ids": batch_token_ids,
        "token_type_ids": batch_segment_ids,
        "labels": labels,
    }
    

train_dataset = MyDataset('/home/dataset/corpus/seq2seq/summary/csl_title_public/csl_title_train.json')
valid_dataset = MyDataset('/home/dataset/corpus/seq2seq/summary/csl_title_public/csl_title_dev.json')


model:BertModel = build_transformer_model(
    checkpoint_path,
    config_path,
    with_mlm=True,
    extra_config = {"unilm": True},  # unilm模式
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    skip_init=True
).to(device)


def generate(text):
    max_c_len = maxlen - 64
    input_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
    # segment_ids  [0, 0, 0, 0, 0, 0, 0]
    
    input_ids = torch.tensor([input_ids]).to(device)
    segment_ids = torch.tensor([segment_ids]).to(device)

    # token_ids = model.generate(input_ids, token_type_ids = segment_ids, mode='do_sample', top_k=20, top_p=0.9, temperature=0.9)
    token_ids = model.generate(
        input_ids = input_ids,
        token_type_ids = segment_ids,
        mode='greedy_search',  
        max_new_tokens=64
    )

    predict_label = [tokenizer.decode(i) for i in token_ids]
    return predict_label[0]


# 定义训练流程
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
save_path = model_name_gene(train_config, 'bert', 'csl_title')
trainer = Trainer(model, train_config, optimizer=optimizer, collate_fn=collate_fn)


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self, trainer:Trainer, save_path:str):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.
        self.trainer = trainer
        self.save_path = save_path

    def on_epoch_end(self, steps, epoch, logs=None):
        just_show()
        metrics = self.evaluate()  # 评测模型
        if metrics['bleu'] > self.best_bleu:
            self.best_bleu = metrics['bleu']
            self.trainer.save_weights(self.save_path)
        metrics['best_bleu'] = self.best_bleu
        print('valid_data:', metrics)
    
    def evaluate(self,):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for title, content in tqdm(valid_dataset):
            total += 1
            title = ' '.join(title).lower()
            pred_title = ' '.join(generate(content)).lower()
            if pred_title.strip():
                scores = self.rouge.get_scores(hyps=pred_title, refs=title)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu += sentence_bleu(references=[title.split(' ')], hypothesis=pred_title.split(' '),
                                      smoothing_function=self.smooth)
        rouge_1, rouge_2, rouge_l, bleu = rouge_1/total, rouge_2/total, rouge_l/total, bleu/total
        return {'rouge-1': rouge_1, 'rouge-2': rouge_2, 'rouge-l': rouge_l, 'bleu': bleu}

def just_show():
    s1 = u'抽象了一种基于中心的战术应用场景与业务,并将网络编码技术应用于此类场景的实时数据多播业务中。在分析基于中心网络与Many-to-all业务模式特性的基础上,提出了仅在中心节点进行编码操作的传输策略以及相应的贪心算法。分析了网络编码多播策略的理论增益上界,仿真试验表明该贪心算法能够获得与理论相近的性能增益。最后的分析与仿真试验表明,在这种有中心网络的实时数据多播应用中,所提出的多播策略的实时性能要明显优于传统传输策略。'
    s2 = u'普适计算环境中未知移动节点的位置信息是定位服务要解决的关键技术。在普适计算二维空间定位过程中,通过对三角形定位单元区域的误差分析,提出了定位单元布局(LUD)定理。在此基础上,对多个定位单元布局进行了研究,定义了一个新的描述定位单元中定位参考点覆盖效能的物理量——覆盖基,提出了在误差最小情况下定位单元布局的覆盖基定理。仿真实验表明定位单元布局定理能更好地满足对普适终端实时定位的需求,且具有较高的精度和最大覆盖效能。'
    s3 = """网络的不断发展对包括电信级以太网系统在内的网络节点设备的硬件资源容量提出了更高的要求。交换芯片通常是电信级以太网产品的主要功能单元,因不能编程导致可扩展性较差。在硬件结构不改变的情况下,电信级以太网设备提供的二层转发表、三层路由表、访问控制列表等重要硬件资源的容量就基本确定。但在实际应用中,经常会遇到这些资源不能满足应用需要的难题。当前的以太网产品,除了升级硬件,未采取措施来解决或弥补这个缺陷。另一方面,由帕累托原则可知,实际硬件转发表中的少数表项处于相对重要的地位,对网络流量的影响较大,而其它表项则处于重要程度相对较低的地位,对网络流量的贡献较小。基于这个事实,利用某种方法来管理和维护硬件资源,使其充分利用是可行的。提出了一种面向应用的解决硬件资源不足问题的方法,该方法借鉴操作系统的内存管理技术,采用特定的算法,在应用意义上等价扩充二层转发表、三层路由表、访问控制列表等硬件资源从而提高系统转发性能。该方法在内存中建立软二层转发表、软三层路由表以及软访问控制列表等,实际的硬件二层转发表、三层路由表以及访问控制列表可看作它们对应的高速缓存,通过生成访问频度等级信息,采用LFU算法来决定软表和硬表之间的数据交换,使得硬表中尽可能存放重要程度更高的条目,从而改善系统的转发性能。实验表明,该方法能达到预期的目标。"""
    for s in [s3]:
        print(u'生成标题:', generate(s))


if __name__ == '__main__':
    just_show()
    evaluator = Evaluator(trainer, save_path)
    # '<extra_id_0>北京 <extra_id_1>北京  <extra_id_2>北京 首都'

    trainer.train(train_dataset, callbacks=[evaluator])

