
import pickle as pkl
from tqdm import tqdm
import os



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


    
