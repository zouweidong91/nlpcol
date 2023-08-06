
# 基础测试：mlm预测

from nlpcol.models.bert import BertModel, Config, load_config, variable_mapping
from nlpcol.tokenizers import Tokenizer
from nlpcol.utils.snippets import seed_everything
import torch
import torch.nn as nn

seed_everything(42)

from nlpcol.utils._file import FileTool
import json


root_model_path = "/home/dataset/pretrain_ckpt/bert/chinese_L-12_H-768_A-12"
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin.bfr_convert'

# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
config = load_config(config_path)
config = Config(**config)
model = BertModel(config)  # 建立模型，加载权重


def model_parameter_diff(state_dict_1:dict, state_dict_2:dict=None):
    """模型参数输出到文件，方便对比差异
    """
    columns = '\t'.join(["Key", "Shape", "Count"])

    paramrter_list_1 = [columns]
    paramrter_list_2 = [columns]

    for key, value in state_dict_1.items():
        paramrter_list_1.append( '\t'.join([key, str(list(value.shape)), str(value.numel())]))
    FileTool.write(paramrter_list_1, "logs/t1.txt")

    for key, value in state_dict_2.items():
        paramrter_list_2.append( '\t'.join([key, str(list(value.shape)), str(value.numel())]))
    FileTool.write(paramrter_list_2, "logs/t2.txt")



# model_parameter_diff(
#     state_dict_1=model.state_dict(), 
#     state_dict_2=torch.load(checkpoint_path, map_location='cpu')
# ) 


def load_weight(model:nn.Module, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    state_dict_new = {}
    mapping = variable_mapping()
    for new_key, old_key in mapping.items():
        if new_key not in model.state_dict():
            print(new_key, '忽略')
            continue
        
        state_dict_new[new_key] = state_dict[old_key]
    # print(state_dict_new['mlm.decoder.weight'])

    model.load_state_dict(state_dict_new, strict=True)


load_weight(model, checkpoint_path)

token_ids, segments_ids = tokenizer.encode("湖北省省会在[MASK][MASK]市。")
print(''.join(tokenizer.ids_to_tokens(token_ids)))

tokens_ids_tensor = torch.tensor([token_ids])
segment_ids_tensor = torch.tensor([segments_ids])

# 需要传入参数with_mlm
model.eval()
with torch.no_grad():
    probas = model(tokens_ids_tensor, segment_ids_tensor)
    result = torch.argmax(probas[0, :], dim=-1).numpy()
    print(tokenizer.decode(result))
