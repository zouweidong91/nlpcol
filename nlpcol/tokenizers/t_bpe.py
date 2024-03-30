import json
import os

from .helpers import get_pairs

# 训练方法：从字符级的小词表出发，训练产生合并规则以及一个词表
# 编码方法：将文本切分成字符，再应用训练阶段获得的合并规则
# 经典模型：GPT, GPT-2, RoBERTa, BART, LLaMA, ChatGLM等


VOCAB_FILE = "vocab.json"
MERGES_FILE = "merges.txt"


class BPETokenizer:
    
    def __init__(self, model_path):
        vocab_file = os.path.join(model_path, VOCAB_FILE)
        merges_file = os.path.join(model_path, MERGES_FILE)
        
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}


    def bpe(self, token:str) -> str:
        word = tuple(token[:-1]) + (token[-1] + "</w>",)    # ('h', 'e', 'l', 'l', 'o</w>')
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        if word == "\n  </w>":
            word = "\n</w>"
        self.cache[token] = word
        return word

    
    def tokenize(self, token:str) -> list:
        token = self.bpe(token)
        return list(token.split(" "))
