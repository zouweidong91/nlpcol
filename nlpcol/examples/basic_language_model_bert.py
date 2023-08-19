
# 基础测试：mlm预测

import torch
import torch.nn as nn
from nlpcol.model import build_transformer_model
from nlpcol.models.bert import BertModel, BertOutput
from nlpcol.tokenizers import Tokenizer
from nlpcol.utils.snippets import model_parameter_diff, seed_everything

seed_everything(42)


model_path = "/home/dataset/pretrain_ckpt/bert/chinese_L-12_H-768_A-12"
vocab_path = model_path + "/vocab.txt"
config_path = model_path + "/config.json"
checkpoint_path = model_path + '/pytorch_model.bin.bfr_convert'

# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)

# 需要传入参数with_mlm
model = build_transformer_model(checkpoint_path, config_path, with_mlm=True, with_pool=True, with_nsp=True)  # 建立模型，加载权重
# with_mlm 会导致权重多60M？ 但参量都是一样的啊

model_parameter_diff(
    state_dict_1=model.state_dict(), 
    state_dict_2=torch.load(checkpoint_path, map_location='cpu')
) 


token_ids, segments_ids = tokenizer.encode("广东省的省会在[MASK][MASK]市。")
print(''.join(tokenizer.ids_to_tokens(token_ids)))

tokens_ids_tensor = torch.tensor([token_ids, token_ids])
segment_ids_tensor = torch.tensor([segments_ids, segments_ids])

bert_output:BertOutput = model.predict([tokens_ids_tensor, segment_ids_tensor])
mlm_scores = bert_output.mlm_scores
# print(mlm_scores)
result = torch.argmax(mlm_scores[0, :], dim=-1).numpy()
print(tokenizer.decode(result))

