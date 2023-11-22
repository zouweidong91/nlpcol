
# 基础测试：lm预测

import torch
import torch.nn as nn
import json
from nlpcol.model import build_transformer_model, load_config
from nlpcol.models.bert import BertModel, BertOutput
from nlpcol.tokenizers import Tokenizer, SpTokenizer
from nlpcol.utils.snippets import model_parameter_diff, seed_everything, save_model_parameter
from nlpcol.config import device
from torch.nn import functional as F

seed_everything(42)



model_path = "/home/dataset/pretrain_ckpt/t5/mt5-base"
config_path = model_path + "/config.json"
checkpoint_path = model_path + '/pytorch_model.bin'
spm_path = model_path + '/spiece.model'

# 建立分词器
tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>')

# 需要传入参数with_mlm
model = build_transformer_model(checkpoint_path, config_path, 't5', skip_init=True, max_seq_length=512)  # 建立模型，加载权重 下游任务无额外参数 暂时不需初始化
model.eval()
# model.to(device)

# model_parameter_diff(
#     state_dict_1=model.state_dict(), 
#     state_dict_2=torch.load(checkpoint_path, map_location='cpu')
# ) 

# training
def get_loss():
    input_ids, _ = tokenizer.encode("The <extra_id_0> walks in <extra_id_1> park")
    labels, _ = tokenizer.encode("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>")
    input_ids = torch.tensor([input_ids])
    labels = torch.tensor([labels])

    print(input_ids)
    print(labels)

    # shift_right
    decoder_input_ids = torch.zeros_like(labels)
    decoder_input_ids[..., 1:] = labels[..., :-1].clone() # 向右偏移一位
    decoder_input_ids[..., 0] = 0 # 起始位置用padding代表
    print(decoder_input_ids)

    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

    lm_logits = outputs.lm_logits
    loss_fn = nn.CrossEntropyLoss()
    # move labels to correct device to enable PP 计算损失时将所有张量转移到同一设备上
    labels = labels.to(lm_logits.device)
    loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    print(loss)


def generate():
    # tokenize
    text = u"中国的首都是 <extra_id_0>京"
    input_ids, _ = tokenizer.encode(text)
    tokens = tokenizer.tokenize(text)
    print(tokens, input_ids, _)

    input_ids = torch.tensor([input_ids, input_ids])
    # decoder_input_ids = model.generate(input_ids, mode='do_sample', top_k=20, top_p=0.9, temperature=0.9)
    decoder_input_ids = model.generate(input_ids, mode='greedy_search')
    # decoder_input_ids = model.generate(input_ids, mode='beam_search', num_beams=4)
    decoder_input_ids=decoder_input_ids[:,1:] # 去掉bos
    predict_label = [tokenizer.decode(i) for i in decoder_input_ids]
    print(predict_label)
    # ['<extra_id_0>北京,简称 <extra_id_1>。']

    decoder_input_ids = model.generate(input_ids, mode='greedy_search')
    # decoder_input_ids = model.generate(input_ids, mode='beam_search', num_beams=4)
    decoder_input_ids=decoder_input_ids[:,1:] # 去掉bos
    predict_label = [tokenizer.decode(i) for i in decoder_input_ids]
    print(predict_label)


# get_loss()  # tensor(4.2471, grad_fn=<NllLossBackward0>)
generate()





