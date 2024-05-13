
# openai-gpt测试

import torch
import torch.nn as nn
from nlpcol.config import device
from nlpcol.model import build_transformer_model
from nlpcol.models import GptModel
from nlpcol.tokenizers import Tokenizer
from nlpcol.utils.snippets import model_parameter_diff, seed_everything

seed_everything(42)

from nlpcol.tokenizers import Tokenizer

model_path = "/home/dataset/pretrain_ckpt/gpt/openai-gpt/"
config_path = model_path + "/config.json"
checkpoint_path = model_path + '/pytorch_model.bin'

# 建立分词器
tokenizer = Tokenizer(
    model_path, 
    tokenizer_type='bpe', 
    token_start=None, 
    token_end=None, 
    token_unk="<unk>", 
    token_pad='<unk>'
)

model:GptModel = build_transformer_model(
    checkpoint_path, 
    config_path=config_path, 
    model='GPT',
    extra_config={"pad_token_id": tokenizer.pad_token_id},
)  # 建立模型，加载权重
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
    token_ids, _ = tokenizer.encode("hello, my dog is cute.")
    tokens = tokenizer.ids_to_tokens(token_ids)
    print(tokens)
    # ['hello</w>', ',</w>', 'my</w>', 'dog</w>', 'is</w>', 'cute</w>', '.</w>']

    # 注意decode only的输入和输出。input 和 label要拼接在一起
    # input_ids 与 labels 形状必须一致
    
    # 准备数据时label做处理
    #   北   京   在   哪   里   在   中   国   </s>      input_ids
    # -100 -100 -100 -100 -100  在   中   国   </s>      labels  

    # 模型内部做shift right
    #   北   京   在   哪   里   在   中   国   </s>      原句
    # -100 -100 -100 -100 -100  在   中   国   </s>      labels  
    #   北   京   在   哪   里   在   中   国             shift_logits
    # -100 -100 -100 -100   在  中   国   </s>           shift_labels  

    cat = tokenizer.encode("hello, my dog is cute. i'm sorry, i didn't mean to make")[0]
    cat = torch.tensor([cat]).to(device)

    input_ids = cat.clone()
    labels = cat.clone()
    labels[:, :len(tokens)] = -100 # 真正ipt的位置不参与损失
    labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)

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
    # ['my name is julien and i like to think that i am a very good person. i am a very good person. i am a very good person. i']
    # ['think that i am a very good person . i am a very good person . i am a very good person . i']  rm_prompt_token

get_loss()
generate()


