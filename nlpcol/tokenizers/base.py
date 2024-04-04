
from .helpers import truncate_sequences
from .t_trie import Trie


class TokenizerBase(object):
    """分词器基类
    """
    def __init__(self, token_start='[CLS]', token_end='[SEP]', token_unk='[UNK]', token_pad='[PAD]', token_mask='[MASK]', 
                 add_special_tokens=None, pre_tokenize=None, token_translate=None):
        """参数说明：
        token_unk: 未知词标记
        token_end: 句子切分标记，当只有一句话作为输入时，此标记知识作为结束符；当有两句话作为输入时，此标记作为分隔符、最后一句话的结束符
        pad_token: padding填充标记
        token_start: 分类标记，位于整个序列的第一个
        mask_token: mask标记
        pre_tokenize: 外部传入的分词函数，用作对文本进行预分词。如果传入pre_tokenize，则先执行pre_tokenize(text)，然后在它的基础上执行原本的tokenize函数；
        token_translate: 映射字典，主要用在tokenize之后，将某些特殊的token替换为对应的token。

        不同模型，5个特殊token需要查看vocab文件分别传入。如果词表中不存在，则后续不设置相应token_id
        """
        self._token_pad = self.pad_token = token_pad
        self._token_unk = self.unk_token = token_unk
        self._token_mask = self.mask_token = token_mask
        self._token_start = self.start_token = token_start
        self._token_end = self.end_token = token_end
        self.never_split = [i for i in [self._token_unk, self._token_end, self._token_pad, self._token_start, self._token_mask] if isinstance(i, str)]
        if add_special_tokens is not None:
            if isinstance(add_special_tokens, (tuple, list)):
                self.never_split.extend(add_special_tokens)
            elif isinstance(add_special_tokens, str):
                self.never_split.append(add_special_tokens)
        self.tokens_trie = self._create_trie(self.never_split)  # trie树主要是为了special_tokens的分词
        self._pre_tokenize = pre_tokenize
        self._token_translate = token_translate or {}
        self._token_translate_inv = {v: k for k, v in self._token_translate.items()}

    def _create_trie(self, unique_no_split_tokens):
        trie = Trie()
        for token in unique_no_split_tokens:
            trie.add(token)
        return trie

    def tokenize(self, text, maxlen=None):
        """分词函数
        """
        tokens = [self._token_translate.get(token) or token for token in self._tokenize(text)]
        if self._token_start is not None:
            tokens.insert(0, self._token_start)
        if self._token_end is not None:
            tokens.append(self._token_end)

        if maxlen is not None:
            index = int(self._token_end is not None) + 1
            truncate_sequences(maxlen, -index, tokens)

        return tokens

    def token_to_id(self, token):
        """token转换为对应的id
        """
        raise NotImplementedError

    def tokens_to_ids(self, tokens):
        """token序列转换为对应的id序列
        """
        return [self.token_to_id(token) for token in tokens]

    def _encode(self, first_text, second_text=None, maxlen=None, pattern='S*E*E', truncate_from='right', return_offsets=False):
        """输出文本对应token id和segment id
        """
        first_tokens = self.tokenize(first_text) if isinstance(first_text, str) else first_text

        if second_text is None:
            second_tokens = None
        elif isinstance(second_text, str):
            second_tokens = self.tokenize(second_text)
        else:
            second_tokens = second_text

        if maxlen is not None:
            # 这里截断思路是优先截断最长的子句
            if truncate_from == 'right':
                index = -int(self._token_end is not None) - 1
            elif truncate_from == 'left':
                index = int(self._token_start is not None)
            else:
                index = truncate_from
            if second_text is not None and pattern == 'S*E*E':
                maxlen += 1
            truncate_sequences(maxlen, index, first_tokens, second_tokens)

        first_token_ids = self.tokens_to_ids(first_tokens)
        first_segment_ids = [0] * len(first_token_ids)

        if second_text is not None:
            if pattern == 'S*E*E':
                idx = int(bool(self._token_start))
                second_tokens = second_tokens[idx:]
            second_token_ids = self.tokens_to_ids(second_tokens)
            second_segment_ids = [1] * len(second_token_ids)
            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)
        
        encode_output = [first_token_ids, first_segment_ids]
        if return_offsets != False:
            offset = self.rematch(first_text, first_tokens) + self.rematch(second_text, second_tokens)
            if return_offsets == 'transformers':  # transformers包中tokenizer的形式
                encode_output.append([[0, 0] if not k else [k[0], k[-1]+1] for k in offset])
            else:
                encode_output.append(offset)
        return encode_output

    def encode(self, first_texts, second_texts=None, maxlen=None, pattern='S*E*E', truncate_from='right', return_offsets=False):
        '''可以处理多条或者单条
        '''
        return_list = False if isinstance(first_texts, str) else True
        first_texts = [first_texts] if isinstance(first_texts, str) else first_texts
        second_texts = [second_texts] if isinstance(second_texts, str) else second_texts

        first_token_ids, first_segment_ids, offsets = [], [], []
        if second_texts is None:
            second_texts = [None] * len(first_texts)
        assert len(first_texts) == len(second_texts), 'first_texts and second_texts should be same length'
        
        # 循环处理每条样本
        for first_text, second_text in zip(first_texts, second_texts):
            outputs = self._encode(first_text, second_text, maxlen, pattern, truncate_from, return_offsets)
            first_token_ids.append(outputs[0])
            first_segment_ids.append(outputs[1])
            if len(outputs) >= 3:
                offsets.append(outputs[2])

        encode_outputs = [first_token_ids, first_segment_ids]
        if return_offsets:
            encode_outputs.append(offsets)

        if not return_list:  # 如果输入是string
            encode_outputs = [item[0] for item in encode_outputs]
        return encode_outputs

    def id_to_token(self, i):
        """id序列为对应的token
        """
        raise NotImplementedError

    def ids_to_tokens(self, ids):
        """id序列转换为对应的token序列
        """
        return [self.id_to_token(int(i)) for i in ids]

    def decode(self, ids):
        """转为可读文本
        """
        raise NotImplementedError

    def _tokenize(self, text):
        """基本分词函数
        """
        raise NotImplementedError

