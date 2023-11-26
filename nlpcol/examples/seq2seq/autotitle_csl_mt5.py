
# 微调多国语言版T5做Seq2Seq任务
# 介绍链接：https://kexue.fm/archives/7867
# 数据集：https://github.com/CLUEbenchmark/CLGE 中的CSL数据集
# 补充了评测指标bleu、rouge-1、rouge-2、rouge-l
# mt5主要特点：gated-gelu, decoder的最后的dense层独立权重，rmsnorm
# valid_data: {'rouge-1': 0.43454686332522263, 'rouge-2': 0.3217250949304608, 'rouge-l': 0.42204007502153934, 'bleu': 0.16675070297852404, 'best_bleu': 0.16675070297852404}

import json

import torch
import torch.nn as nn
import torch.optim as optim
from nlpcol.callback import Callback
from nlpcol.config import TrainConfig, WordDir, device
from nlpcol.model import build_transformer_model
from nlpcol.tokenizers import SpTokenizer
from nlpcol.trainer import Trainer
from nlpcol.utils.snippets import (ListDataset, get_pool_emb,
                                   save_model_parameter, seed_everything,
                                   sequence_padding, torch_gc)
from nlpcol.utils.snippets4examples import model_name_gene, trans_entity2tuple
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge  # pip install rouge
from tqdm import tqdm

# 基本参数
max_c_len = 256
max_t_len = 32
batch_size = 16
epochs = 50
steps_per_epoch = None
pad_token_id = -100

train_config = TrainConfig(batch_size=batch_size, epochs=epochs, max_seq_length=128)


model_path = "/home/dataset/pretrain_ckpt/t5/mt5-base"
config_path = model_path + "/config.json"
checkpoint_path = model_path + '/pytorch_model.bin'
# spm_path = model_path + '/spiece.model'
# 下面两个config是从bert4keras中拿的，项目连接https://github.com/bojone/t5_in_bert4keras
spm_path = model_path + '/sentencepiece_cn.model'
keep_tokens_path = model_path + '/sentencepiece_cn_keep_tokens.json'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(42)


tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>')
keep_tokens = json.load(open(keep_tokens_path))


model = build_transformer_model(
    checkpoint_path,
    config_path,
    model='t5',
    extra_config={"max_seq_length": 512},
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    skip_init=True
).to(device)


def generate(text):
    input_ids, _ = tokenizer.encode(text)
    input_ids = torch.tensor([input_ids], device=device)

    # logits = model.generate(input_ids, mode='do_sample', top_k=20, top_p=0.9, temperature=0.9)
    logits = model.generate(input_ids, mode='greedy_search')
    # logits = model.generate(input_ids, mode='beam_search', num_beams=4)

    logits=logits[:,1:] # 去掉bos
    predict_label = [tokenizer.decode(i) for i in logits]
    return predict_label[0]


# 准备数据
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
                title, content = l['title'], l['abst']
                D.append((title, content))
        return D

def collate_fn(batch):
    """TODO padding 转入 tokenizer中
    """
    batch_content_ids, batch_titile_ids = [], []
    for title, content in batch:
        content_ids, _ = tokenizer.encode(content, maxlen=max_c_len)
        batch_content_ids.append(content_ids)

        titile_ids, _ = tokenizer.encode(title, maxlen=max_t_len)
        batch_titile_ids.append(titile_ids)

    batch_content_ids = torch.tensor(sequence_padding(batch_content_ids), dtype=torch.long, device=device)
    batch_titile_ids = torch.tensor(sequence_padding(batch_titile_ids, value=-100), dtype=torch.long, device=device)
    
    return {
        "input_ids": batch_content_ids,
        "labels": batch_titile_ids,
    }
    
train_dataset = MyDataset('/home/dataset/corpus/seq2seq/summary/csl_title_public/csl_title_train.json')
valid_dataset = MyDataset('/home/dataset/corpus/seq2seq/summary/csl_title_public/csl_title_dev.json')


# 定义训练流程
optimizer = optim.Adam(model.parameters(), 1e-4)
save_path = model_name_gene(train_config, 'mt5', 'csl_title_public', prefix='test')


# loss_fn = CrossEntropyLoss()
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
    
    def evaluate(self):
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
    for s in [s1, s2]:
        print(u'生成标题:', generate(s))


if __name__ == '__main__':
    evaluator = Evaluator(trainer, save_path)
    print(u'生成标题:', generate(u'中国的首都是extra0京'))  # 和huggingface的结果一致 
    # '<extra_id_0>北京 <extra_id_1>北京  <extra_id_2>北京 首都'

    trainer.train(train_dataset, callbacks=[evaluator])

# else:
#     trainer.load_weights(save_path)
