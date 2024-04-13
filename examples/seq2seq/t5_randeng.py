# https://huggingface.co/IDEA-CCNL/Randeng-T5-784M
# 我们基于mT5-large，训练了它的中文版。为了加速训练，我们仅使用T5分词器(sentence piece)中的中英文对应的词表，
# 并且使用了语料库自适应预训练(Corpus-Adaptive Pre-Training, CAPT)技术在悟道语料库(180G版本)继续预训练。
# 预训练目标为破坏span。具体地，我们在预训练阶段中使用了封神框架大概花费了16张A100约96小时。



import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

# load tokenizer and model 
pretrained_model = "/home/dataset/pretrain_ckpt/t5/Randeng-T5-784M-MultiTask-Chinese"

special_tokens = ["<extra_id_{}>".format(i) for i in range(100)]
tokenizer = T5Tokenizer.from_pretrained(
    pretrained_model,
    do_lower_case=True,
    max_length=512,
    truncation=True,
    additional_special_tokens=special_tokens,
)
config = T5Config.from_pretrained(pretrained_model)
model = T5ForConditionalGeneration.from_pretrained(pretrained_model, config=config)
model.resize_token_embeddings(len(tokenizer))
model.eval()

# tokenize
text = "新闻分类任务：【微软披露拓扑量子计算机计划！】这篇文章的类别是什么？故事/文化/娱乐/体育/财经/房产/汽车/教育/科技"
encode_dict = tokenizer([text], max_length=512, padding='max_length',truncation=True)

inputs = {
  "input_ids": torch.tensor([encode_dict['input_ids']]).long(),
  "attention_mask": torch.tensor([encode_dict['attention_mask']]).long(),
  }

# generate answer
logits = model.generate(
  input_ids = inputs['input_ids'],
  max_length=100, 
  do_sample= True
  # early_stopping=True,
  )

logits=logits[:,1:]
predict_label = [tokenizer.decode(i,skip_special_tokens=True) for i in logits]
print(predict_label)

# model output: 科技
