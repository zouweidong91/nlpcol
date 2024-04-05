# 基本测试：中文GPT模型，base版本，CDial-GPT版
# 项目链接：https://github.com/thu-coai/CDial-GPT
# 参考项目：https://github.com/bojone/CDial-GPT-tf


from itertools import chain

import torch
from nlpcol.config import device
from nlpcol.model import build_transformer_model
from nlpcol.models import GptModel
from nlpcol.tokenizers import Tokenizer
from nlpcol.utils.snippets import get_special_token

model_path = "/home/dataset/pretrain_ckpt/gpt/CDial-GPT_LCCC-base"
vocab_path = model_path + "/vocab.txt"
config_path = model_path + "/config.json"
checkpoint_path = model_path + '/pytorch_model.bin'

# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
speaker1, speaker2 = [tokenizer.token_to_id('[speaker1]'), tokenizer.token_to_id('[speaker2]')]
speakers = [speaker1, speaker2]
bos, eos, pad = tokenizer.start_token_id, tokenizer.end_token_id, tokenizer.pad_token_id


model:GptModel = build_transformer_model(
    checkpoint_path, 
    config_path=config_path, 
    model='GPT', 
    extra_config={"prefix": "transformer", **get_special_token(tokenizer)}
)
model.to(device)


def generate():
    texts = ['别爱我没结果', '你这样会失去我的', '失去了又能怎样']
    token_ids = [tokenizer._token_start_id, speakers[0]]
    segment_ids = [tokenizer._token_start_id, speakers[0]]

    for i, text in enumerate(texts):
        ids = tokenizer.encode(text)[0][1:-1] + [speakers[(i + 1) % 2]] 
        token_ids.extend(ids)
        segment_ids.extend([speakers[i % 2]] * len(ids))
        segment_ids[-1] = speakers[(i + 1) % 2]

    print(token_ids, '\n', segment_ids)
    input_ids = torch.tensor([token_ids]).to(device)
    segment_ids = torch.tensor([segment_ids]).to(device)

    token_ids = model.generate(
        input_ids = input_ids, 
        token_type_ids = segment_ids, 
        mode = "do_sample", max_new_tokens=32, top_k=5
    )
    predict_label = [tokenizer.decode(i) for i in token_ids]
    print(predict_label)

generate()

"""
回复是随机的，例如：你还有我 | 那就不要爱我 | 你是不是傻 | 等等。
"""
