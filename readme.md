## 🗂️ nlpcol
nlpcol(nlp collction)： 一个简单的nlp基础模型集合。

相较于transformers，本项目尽量以较低的代码实现一部分经典模型。旨在通过构建简单易懂的代码，让初学者能够快速接触及掌握各经典模型以及训练推理过程的各个细节。当前已集成模型: bert, gpt, mt5, gpt2。

## 💡 TODO
- 训练逻辑优化：混合精度、grad_checkoutpoint 
- 训练日志优化
- 常用优化器实现 
- xlnet unilm bart 
- llm: llama chatgpt 
- config的加载方式  extra_config


## 🚀 预训练权重
**所有权重均为torch版本**

| 模型分类| 模型名称 | 权重来源 | 官方项目地址 |
| ----- | ----- | ----- | ----- |
| bert | bert-base-chinese | [谷歌中文bert](https://huggingface.co/google-bert/bert-base-chinese) | [bert](https://github.com/google-research/bert) |
| mt5 | mt5-base | [谷歌多语言版T5](https://huggingface.co/google/mt5-base) | [t5](https://github.com/google-research/text-to-text-transfer-transformer) |
| gpt | openai-gpt | [openai-gpt1](https://huggingface.co/openai-community/openai-gpt) | [finetune-transformer-lm](https://github.com/openai/finetune-transformer-lm) |
| gpt | CDial-GPT_LCCC-base | [清华coai](https://huggingface.co/thu-coai/CDial-GPT_LCCC-base) | [CDial-GPT](https://github.com/thu-coai/CDial-GPT) |
| gpt2 | openai-gpt2 | [openai-gpt2](https://huggingface.co/openai-community/gpt2) | [gpt-2](https://github.com/openai/gpt-2) |


# ref: 
https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master
https://github.com/NVIDIA/Megatron-LM/tree/main
