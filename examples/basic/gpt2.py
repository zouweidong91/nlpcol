
# openai-gpt2测试

import torch
import torch.nn as nn
from nlpcol.config import device
from nlpcol.model import build_transformer_model
from nlpcol.models import Gpt2Model
from nlpcol.tokenizers import Tokenizer
from nlpcol.utils.snippets import model_parameter_diff, seed_everything

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
    token_unk='<|endoftext|>',
    do_lower_case=False,
    do_basic_tokenize=False,
    )

model:Gpt2Model = build_transformer_model(checkpoint_path, config_path=config_path, model='GPT2')  # 建立模型，加载权重
model.eval() # 关掉nn.dropout model.training=False
model.to(device)

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
    cat = torch.tensor([cat]).to(device)

    input_ids = cat.clone()
    labels = cat.clone()
    labels[:, :len(tokens)] = -100 # 真正ipt的位置不参与损失

    print(input_ids)
    print(labels)

    outputs = model(input_ids=input_ids, labels=labels)
    print(outputs.loss)
    # tensor([[31373,    11,   616,  3290,   318, 13779,    13,  1312,  1101,  7926,
    #             11,  1312,  1422,   470,  1612,   284,   787]], device='cuda:0')
    # tensor([[-100, -100, -100, -100, -100, -100, -100, 1312, 1101, 7926,   11, 1312,
    #         1422,  470, 1612,  284,  787]], device='cuda:0')
    # tensor(2.4846, device='cuda:0', grad_fn=<NllLossBackward0>)


def generate():
    # input_ids = tokenizer.encode("My name is Julien and I like to")[0]
    # input_ids_2 = tokenizer.encode("My name is Julien and I like to")[0]

    input_ids= tokenizer.encode("My name is Julien and I like to")[0]
    input_ids = torch.tensor([input_ids]).to(device)

    # input_ids = pad_sequence([input_ids.squeeze(), input_ids_2.squeeze()], batch_first=True, padding_value=DecGenerationMixin.PADDING_ID)   # (bsz, max_len)
    print(input_ids)

    # generate answer  greed search
    token_ids = model.generate(input_ids = input_ids, max_new_tokens=32, num_beams=1)
    predict_label = [tokenizer.decode(i) for i in token_ids]
    print(predict_label)
    # ["play with my friends. I 'm a big fan of the game and I 'm looking forward to playing with my friends. I 'm looking forward to playing with my"]

get_loss()
generate()


