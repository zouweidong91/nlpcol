
import os
import pickle as pkl

from tqdm import tqdm

# 关于transformers库中不同模型的Tokenizer https://zhuanlan.zhihu.com/p/121787628

# 大模型基础组件 - Tokenizer https://zhuanlan.zhihu.com/p/651430181
# 根据不同的切分粒度可以把tokenizer分为: 基于词的切分，基于字的切分和基于subword的切分。 基于subword的切分是目前的主流切分方式。
# 基于subword的切分能很好平衡基于词切分和基于字切分的优缺点，也是目前主流最主流的切分方式。基于词和字的切分都会存在一定的问题，直接应用的效果比较差。

# 基于词的切分，会造成:
# 词表规模过大
# 一定会存在UNK，造成信息丢失
# 不能学习到词缀之间的关系，例如：dog与dogs，happy与unhappy

# 基于字的切分，会造成:
# 每个token的信息密度低
# 序列过长，解码效率很低

# 所以基于词和基于字的切分方式是两个极端，其优缺点也是互补的。而折中的subword就是一种相对平衡的方案。
# subword的基本切分原则是：
# 高频词依旧切分成完整的整词
# 低频词被切分成有意义的子词，例如 dogs => [dog, ##s]

# 基于subword的切分可以实现：
# 词表规模适中，解码效率较高
# 不存在UNK，信息不丢失
# 能学习到词缀之间的关系
# 基于subword的切分包括：BPE，WordPiece 和 Unigram 三种分词模型。

# https://blog.csdn.net/ljp1919/article/details/113616226
# Bert采用的是字符级别的BPE编码，直接生成词表文件。Roberta采用的是**byte level的BPE(BBPE)**编码，预训练结果中的merges.txt中存储了BBPE过程中merge得到的所有token，
# 可以简单理解成就是字典。vocab.json则是一个字典中基本单元到索引的映射。转换的过程是，根据merges.txt将输入的文本tokenize化，再根据vocab.json中的字典映射到对应的索id。




class SampleTokenizer:
    """
    需要先根据训练集构建模型所需的词表
    根据训练样本生成vocab
    """

    def __init__(self, train_data, vocab_path:str, ues_word=False):
        self.train_data = train_data
        self.ues_word = ues_word

        self.UNK = '<UNK>'  # 未知字，
        self.PAD = '<PAD>'  # padding符号
        self.MAX_VOCAB_SIZE = 10000 # 词表长度限制

        self.vocab = self.get_vocab(vocab_path)
        self.vocab_size = len(self.vocab)
    
    def build_vocab(self, min_freq=1) -> dict:
        """构建词表

        Args:
            min_freq (_type_): 过滤低频词
        """
        vocab_dic = {}
        for line in tqdm(iter(self.train_data)):
            content = line.split('\t')[0]
            for word in self.tokenize(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1

        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:self.MAX_VOCAB_SIZE]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({self.UNK: len(vocab_dic), self.PAD: len(vocab_dic) + 1})
    
        return vocab_dic


    def get_vocab(self, vocab_path) -> dict:
        if os.path.exists(vocab_path):
            vocab = pkl.load(open(vocab_path, 'rb'))
        else:
            vocab = self.build_vocab()
            pkl.dump(vocab, open(vocab_path, 'wb'))
        print(f"Vocab size: {len(vocab)}")
        return vocab


    def tokenize(self, text:str, max_seq_size:int=None):
        if self.ues_word:
            tokenizer = lambda x: x.split(' ') # 以要求数据集词以空格隔开，word-level
        else:
            tokenizer = lambda x: [y for y in x]  # char-level


        tokens = tokenizer(text)
        if not max_seq_size:
            return tokens


        if len(tokens) < max_seq_size:
            tokens.extend([self.PAD] * (max_seq_size - len(tokens)))
        else:
            tokens = tokens[:max_seq_size]
        
        return tokens

    def token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.UNK))

    
    def encode(self, text, max_seq_size):
        tokens = self.tokenize(text, max_seq_size)
        token_ids = [self.token_to_id(token) for token in tokens]
        return token_ids


    

"""Tokenization classes."""
import collections
import re
import unicodedata
from io import open

from nlpcol.utils import logger

from .base import TokenizerBase
from .helpers import lowercase_and_normalize
from .t_basic import BasicTokenizer
from .t_bpe import BPETokenizer
from .t_wordpiece import WordpieceTokenizer


def load_vocab(dict_path, encoding="utf-8", simplified=False, startswith=None):
    """加载词典文件到dict
    兼容bbpe vocab.json数据。 TODO 兼容bbpe
    """
    token_dict = collections.OrderedDict()
    index = 0
    with open(dict_path, "r", encoding=encoding) as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            token_dict[token] = index
            index += 1

    if simplified:  # 过滤冗余部分token，如[unused1]
        new_token_dict, keep_tokens = {}, []
        startswith = startswith or []
        for t in startswith:
            new_token_dict[t] = len(new_token_dict)
            keep_tokens.append(token_dict[t])

        for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
            if t not in new_token_dict and not Tokenizer._is_redundant(t):
                new_token_dict[t] = len(new_token_dict)
                keep_tokens.append(token_dict[t])

        return new_token_dict, keep_tokens
    else:
        return token_dict



class Tokenizer(TokenizerBase):
    """Bert原生分词器
    流程： 先执行BasicTokenizer，按照标点符号或者中文进行单字切分，再进行WordpieceTokenizer切分
    bpe是从最小char级别开始，一步步合并， bpe详见tokenization_openai, bpe有一个配套的merges.txt文件
    wordpiece word开始，如果不在词表中就一步步剪切word，执行最大前向匹配，直到出现在词表
    
    """
    def __init__(
        self, 
        vocab_or_model_path:str, 
        do_lower_case=True, 
        do_basic_tokenize=True, 
        do_tokenize_unk=False, 
        tokenizer_type='wordpiece', 
        **kwargs
    ):
        """
        参数:
            vocab_or_model_path:
                词典文件
            do_lower_case:
                是否转换成小写
            do_basic_tokenize:
                分词前，是否进行基础的分词
            do_tokenize_unk:
                分词后，是否生成[UNK]标记，还是在encode阶段生成
        """
        super(Tokenizer, self).__init__(**kwargs)

        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, never_split=self.never_split)

        if tokenizer_type == "wordpiece":
            token_dict = load_vocab(vocab_or_model_path)
            self._tokenizer = WordpieceTokenizer(vocab=token_dict, unk_token=self._token_unk, do_tokenize_unk=do_tokenize_unk)
        else:
            self._tokenizer = BPETokenizer(vocab_or_model_path)
            token_dict = self._tokenizer.encoder

        self._do_lower_case = do_lower_case
        self._vocab_size = len(token_dict)
        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}

        # 以下写在外面是方便有代码提示
        self._token_pad_id = self.pad_token_id = None
        self._token_unk_id = self.unk_token_id = None
        self._token_mask_id = self.mask_token_id = None
        self._token_start_id = self.start_token_id = None
        self._token_end_id = self.end_token_id = None
        # 设置5个特殊token的id
        for token in ['pad', 'unk', 'mask', 'start', 'end']:
            try:
                _token_id = token_dict[getattr(self, '_token_%s' % token)]
                setattr(self, '_token_%s_id' % token, _token_id)
                setattr(self, '%s_token_id' % token, _token_id)
            except:
                logger.warning(f"不存在token: {token}")
                delattr(self, '_token_%s_id' % token)
                delattr(self, '%s_token_id' % token)


    def _tokenize(self, text, pre_tokenize=True):
        """基本分词函数
        """
        # 以下pre_tokenizer逻辑参考bert4keras
        if self._do_lower_case:
            text = lowercase_and_normalize(text, never_split=self.never_split)

        if pre_tokenize and self._pre_tokenize is not None:
            tokens = []
            for token in self._pre_tokenize(text):
                if token in self._token_dict:
                    tokens.append(token)
                else:
                    tokens.extend(self._tokenize(token, False))
            return tokens

        # 以下逻辑参考pytorch版本bert分词器自己的
        text_pieces = self.tokens_trie.split(text)  # 新增逻辑，主要是special_tokens的分词
        split_tokens = []
        for text_piece in text_pieces:
            if not text_piece:
                continue
            elif text_piece in self._token_dict:
                split_tokens.append(text_piece)
            elif self.do_basic_tokenize:
                for token in self.basic_tokenizer.tokenize(text_piece):
                    for sub_token in self._tokenizer.tokenize(token):
                        split_tokens.append(sub_token)
            else:
                split_tokens.extend(self._tokenizer.tokenize(text_piece))
        return split_tokens

    def token_to_id(self, token):
        """token转为vocab中的id"""
        return self._token_dict.get(token, self._token_unk_id)

    def id_to_token(self, id):
        """id转为词表中的token"""
        return self._token_dict_inv[id]

    def decode(self, ids):
        """转为可读文本
        """
        tokens = self.ids_to_tokens(ids)
        tokens = [token for token in tokens if not self._is_special(token)] # 过滤特殊token

        text, flag = '', False
        for i, token in enumerate(tokens):
            if token[:2] == '##':
                text += token[2:]

            #原始gpt 为英文
            elif token[-4:] == '</w>':
                text += ' ' + token[:-4]

            elif len(token) == 1 and self._is_cjk_character(token):
                text += token
            elif len(token) == 1 and self._is_punctuation(token):
                text += token
                text += ' '
            elif i > 0 and self._is_cjk_character(text[-1]):
                text += token
            else:
                text += ' '
                text += token

        text = re.sub(' +', ' ', text)
        text = re.sub('\' (re|m|s|t|ve|d|ll) ', '\'\\1 ', text)
        punctuation = self._cjk_punctuation() + '+-/={(<['
        punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
        punctuation_regex = '(%s) ' % punctuation_regex
        text = re.sub(punctuation_regex, '\\1', text)
        text = re.sub('(\d\.) (\d)', '\\1\\2', text)

        return text.strip()

    @staticmethod
    def stem(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_space(ch):
        """空格类字符判断
        """
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
            unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _is_punctuation(ch):
        """标点符号类字符判断（全/半角均在此内）
        提醒：unicodedata.category这个函数在py2和py3下的
        表现可能不一样，比如u'§'字符，在py2下的结果为'So'，
        在py3下的结果是'Po'。
        """
        code = ord(ch)
        return 33 <= code <= 47 or \
            58 <= code <= 64 or \
            91 <= code <= 96 or \
            123 <= code <= 126 or \
            unicodedata.category(ch).startswith('P')

    @staticmethod
    def _cjk_punctuation():
        return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\u00b7\uff01\uff1f\uff61\u3002'

    @staticmethod
    def _is_cjk_character(ch):
        """CJK类字符判断（包括中文字符也在此列）
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    @staticmethod
    def _is_redundant(token):
        """判断该token是否冗余（默认情况下不可能分出来）
        """
        if len(token) > 1:
            for ch in Tokenizer.stem(token):
                if (
                    Tokenizer._is_cjk_character(ch) or
                    Tokenizer._is_punctuation(ch)
                ):
                    return True

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = lowercase_and_normalize(ch, self.never_split)
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping
