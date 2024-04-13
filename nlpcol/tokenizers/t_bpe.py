import json
import os
from functools import lru_cache

# 训练方法：从字符级的小词表出发，训练产生合并规则以及一个词表
# 编码方法：将文本切分成字符，再应用训练阶段获得的合并规则
# 经典模型：GPT, GPT-2, RoBERTa, BART, LLaMA, ChatGLM等


VOCAB_FILE = "vocab.json"
MERGES_FILE = "merges.txt"


def get_pairs(word):
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class BPETokenizer:
    # gpt1
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


    def bpe(self, token:str, suffix:str="</w>") -> str:
        """_summary_

        Args:
            token (str): _description_
            suffix (str, optional): 分词后缀，gpt为"</w>". gpt2为"" 

        Returns:
            str: _description_
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + suffix,)    # ('h', 'e', 'l', 'l', 'o</w>')
        pairs = get_pairs(word)

        if not pairs:
            return token + suffix

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

        if word == f"\n  {suffix}":
            word = f"\n{suffix}"

        self.cache[token] = word
        return word

    
    def tokenize(self, token:str) -> list:
        token = self.bpe(token)
        return list(token.split(" "))



class BBPETokenizer(BPETokenizer):
    # gpt2
    def __init__(self, model_path):
        super().__init__(model_path)
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    def tokenize(self, token:str) -> list:
        # hello
        token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
        token = self.bpe(token, suffix="")
        return list(token.split(" "))

    def convert_tokens_to_string(self, tokens):
        """TODO Converts a sequence of tokens (string) in a single string.  Ċ  -> \n"""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace",)
        return text

