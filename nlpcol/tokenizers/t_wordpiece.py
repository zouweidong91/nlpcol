import collections
from .helpers import whitespace_tokenize


# 训练方法：从字符级的小词表出发，训练产生合并规则以及一个词表
# 编码方法：将文本切分成词，对每个词在词表中进行最大前向匹配
# 经典模型：BERT及其系列DistilBERT，MobileBERT等

class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100, do_tokenize_unk=False):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.do_tokenize_unk = do_tokenize_unk


    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token if self.do_tokenize_unk else token)  # 超长
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break

                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if self.do_tokenize_unk and is_bad:  # 是否在tokenize阶段转UNK
                output_tokens.append(self.unk_token)
            elif (not self.do_tokenize_unk) and is_bad:
                output_tokens.append(substr)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
