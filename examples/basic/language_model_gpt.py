
# 基础测试：mlm预测

import torch
import torch.nn as nn
from nlpcol.config import device
from nlpcol.generation import DecGenerationMixin
from nlpcol.model import build_transformer_model
from nlpcol.models.bert import BertModel, BertOutput
from nlpcol.models.gpt import GptModel
from nlpcol.tokenizers import Tokenizer
from nlpcol.utils.snippets import model_parameter_diff, seed_everything
from torch.nn.utils.rnn import pad_sequence

seed_everything(42)

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

model_path = "/home/dataset/pretrain_ckpt/gpt/openai-gpt/"
vocab_path = model_path + "/vocab.txt"
config_path = model_path + "/config.json"
checkpoint_path = model_path + '/pytorch_model.bin'

# 建立分词器
# tokenizer = Tokenizer(vocab_path, do_lower_case=True)
tokenizer = OpenAIGPTTokenizer.from_pretrained(model_path)

model:GptModel = build_transformer_model(checkpoint_path, config_path=config_path, model='GPT')  # 建立模型，加载权重
model.eval() # 关掉nn.dropout model.training=False
model.to(device)

# model_parameter_diff(
#     state_dict_1=model.state_dict(), 
#     state_dict_2=torch.load(checkpoint_path, map_location='cpu')
# )


# training
def get_loss():
    # tokenize
    tokens = tokenizer.tokenize("hello, my dog is cute.")
    print(tokens)
    # ['hello</w>', ',</w>', 'my</w>', 'dog</w>', 'is</w>', 'cute</w>', '.</w>']

    # 注意decode only的输入和输出。input 和 label要拼接在一起
    # input_ids 与 labels 形状必须一致
    
    # 准备数据时label做处理
    # 北 京 在 哪 里 在 中 国 </s>      input_ids
    # 哈 哈 哈 哈 哈 在 中 国 </s>      labels  哈为-100

    # 模型内部做shift right
    # 北 京 在 哪 里 在 中 国 </s>      原句
    # 北 京 在 哪 里 在 中 国           shift_logits
    # 哈 哈 哈 哈 在 中 国 </s>         labels  哈为-100

    cat = tokenizer("hello, my dog is cute. i'm sorry, i didn't mean to make", return_tensors="pt").input_ids.to(device)
    input_ids = cat.clone()
    labels = cat.clone()
    labels[:, :len(tokens)] = -100 # 真正ipt的位置不参与损失

    print(input_ids)
    print(labels)

    outputs = model(input_ids=input_ids, labels=labels)
    print(outputs.loss)
    # tensor([[ 3570,   240,   547,  2585,   544,  4957,   239,   249,   256,   258,
    #           1458,   240,   249, 21647,   256,   241,  1315,   485,   925]],
    #        device='cuda:0')
    # tensor([[ -100,  -100,  -100,  -100,  -100,  -100,  -100,   249,   256,   258,
    #           1458,   240,   249, 21647,   256,   241,  1315,   485,   925]],
    #        device='cuda:0')
    # tensor(2.9709, device='cuda:0', grad_fn=<NllLossBackward0>)


def generate():
    input_ids = tokenizer("My name is Julien and I like to", return_tensors="pt").input_ids.to(device)
    input_ids_2 = tokenizer("My name is Julien and I", return_tensors="pt").input_ids.to(device)
    # input_ids = input_ids.repeat(2, 1)
    input_ids = pad_sequence([input_ids.squeeze(), input_ids_2.squeeze()], batch_first=True, padding_value=DecGenerationMixin.PADDING_ID)   # (bsz, max_len)
    print(input_ids)

    # generate answer  greed search
    logits = model.generate(input_ids = input_ids, max_length=32, num_beams=1)
    predict_label = [tokenizer.decode(i,skip_special_tokens=True) for i in logits]
    print(predict_label)
    # ['my name is julien and i like to think that i am a very good person. i am a very good person. i am a very good person. i']

get_loss()
generate()


