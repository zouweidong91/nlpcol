

import torch
import torch.nn as nn
from nlpcol.model import build_transformer_model
from nlpcol.models.bert import BertModel, BertOutput
from nlpcol.tokenizers import Tokenizer, SpTokenizer
from nlpcol.utils.snippets import model_parameter_diff, seed_everything
from nlpcol.config import device

seed_everything(42)


# transformers版本
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
pretraine_dir = '/home/dataset/pretrain_ckpt/t5/mt5-base/'
tokenizer = AutoTokenizer.from_pretrained(pretraine_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(pretraine_dir).to(device)
model.eval()

# tokenize
text = u"The <extra_id_0> walks in <extra_id_1> park" #  <extra_id_1> -->  _<extra_id_1>  空格 --> _

text = u"中国的首都是 <extra_id_0>京"
encode_dict = tokenizer(text, max_length=64, padding='max_length',truncation=True)
tokens = tokenizer.tokenize(text)
print(tokens)
inputs = {"input_ids": torch.tensor([encode_dict['input_ids']]).long()}


# training
input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids

print(input_ids)
print(labels)
outputs = model(input_ids=input_ids, labels=labels)

encoder_last_hidden_state = outputs.encoder_last_hidden_state
decoder_hidden_states = outputs.decoder_hidden_states

print(encoder_last_hidden_state.shape)
print(outputs.loss)


# tensor([[   486, 250099,  12747,    263,    281, 250098,  10676,      1]])
# tensor([[250099,  64712,  10990, 250098,    287, 250097,      1]])
# torch.Size([1, 8, 768])
# tensor(4.2471, grad_fn=<NllLossBackward0>)

# # generate answer
# logits = model.generate(input_ids = inputs['input_ids'],max_length=100, num_beams=1)
# logits=logits[:,1:]
# predict_label = [tokenizer.decode(i,skip_special_tokens=True) for i in logits]
# print(predict_label)



