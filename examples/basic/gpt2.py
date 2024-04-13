
# openai-gpt2测试

import torch
import torch.nn as nn
from nlpcol.config import device
from nlpcol.model import build_transformer_model
from nlpcol.models import Gpt2Model
from nlpcol.tokenizers import Tokenizer
from nlpcol.utils.snippets import model_parameter_diff, seed_everything, sequence_padding

seed_everything(42)


# This is the smallest version of GPT-2, with 124M parameters.
model_path = "/home/dataset/pretrain_ckpt/gpt2/openai-gpt2/"
config_path = model_path + "/config.json"
checkpoint_path = model_path + '/pytorch_model.bin'


# 建立分词器
tokenizer = Tokenizer(
    model_path, 
    tokenizer_type='bbpe', 
    token_start=None, 
    token_end=None, 
    token_pad='<|endoftext|>', 
    token_unk='<|endoftext|>',
    do_lower_case=False,
    do_basic_tokenize=False,
    )


model:Gpt2Model = build_transformer_model(
    checkpoint_path, 
    config_path=config_path, 
    extra_config={"pad_token_id": tokenizer.pad_token_id},
    model='GPT2'
)  # 建立模型，加载权重
model.eval() # 关掉nn.dropout model.training=False
model.to(device)
# print(model) # 打印模型网络结构

# 对比模型参数
# model_parameter_diff(
#     state_dict_1=model.state_dict(), 
#     state_dict_2=torch.load(checkpoint_path, map_location='cpu')
# )


# training
def get_loss():
    # tokenize
    token_ids = tokenizer.encode("hello, my dog is cute.")[0]
    tokens = tokenizer.ids_to_tokens(token_ids)
    print(tokens)
    # ['hello', ',', 'Ġmy', 'Ġdog', 'Ġis', 'Ġcute', '.']

    cat = tokenizer.encode("hello, my dog is cute. i'm sorry, i didn't mean to make")[0]
    cat = sequence_padding([cat], length=20, value=tokenizer.pad_token_id) # 模拟batch下padding效果
    cat = torch.tensor(cat).to(device)
    
    input_ids = cat.clone()
    labels = cat.clone()
    labels[:, :len(tokens)] = -100 # 真正ipt的位置不参与损失
    labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)

    print(input_ids)
    print(labels)

    outputs = model(input_ids=input_ids, labels=labels)
    print(outputs.loss)
    # tensor([[31373,    11,   616,  3290,   318, 13779,    13,  1312,  1101,  7926,
    #             11,  1312,  1422,   470,  1612,   284,   787, 50256, 50256, 50256]],
    #     device='cuda:0')
    # tensor([[-100, -100, -100, -100, -100, -100, -100, 1312, 1101, 7926,   11, 1312,
    #         1422,  470, 1612,  284,  787, -100, -100, -100]], device='cuda:0')
    # tensor(2.4846, device='cuda:0', grad_fn=<NllLossBackward0>)


def generate():
    input_ids= tokenizer.encode("My name is Julien and I like to")[0]
    input_ids = sequence_padding([input_ids], length=20, value=tokenizer.pad_token_id, padding_side='left')[0] # 模拟batch下padding效果
    input_ids = torch.tensor([input_ids]).to(device)

    print(input_ids)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    print(attention_mask)

    # generate answer  greed search
    token_ids = model.generate(input_ids = input_ids, max_new_tokens=64, num_beams=1, attention_mask=attention_mask)
    predict_label = [tokenizer.decode(i) for i in token_ids]
    print(predict_label)
    # ["play with my friends. I 'm a big fan of the game and I 'm looking forward to playing with my friends. I 'm looking forward to playing with my"]

# get_loss()
generate()



