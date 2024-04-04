

import unittest
from nlpcol.utils import logger


class LayerTest(unittest.TestCase):

    def test_trie(self):
        # 基于前缀树分词
        from nlpcol.tokenizers import Trie

        trie = Trie()
        trie.add("dog")
        trie.add("北京")
        logger.info(trie.data)
        logger.info(trie.split("hello, my dog is cute."))
        # ['hello, my ', 'dog', ' is cute.']


    def test_bert_tokenizer(self):
        from nlpcol.tokenizers import Tokenizer
        
        # 建立分词器
        vocab_path = "/home/dataset/pretrain_ckpt/bert/chinese_L-12_H-768_A-12/vocab.txt"
        tokenizer = Tokenizer(vocab_path, do_lower_case=True)
        text = "湖北省省会在[MASK][MASK]市。"
        text = "hello, my dog is cute."
        token_ids, segments_ids = tokenizer.encode(text)
        logger.info(token_ids)
        logger.info(tokenizer.ids_to_tokens(token_ids))
        # [CLS]湖北省省会在[MASK][MASK]市。[SEP]


    def test_t5_tokenizer(self):
        from nlpcol.tokenizers import SpTokenizer
        
        # 建立分词器
        model_path = "/home/dataset/pretrain_ckpt/t5/mt5-base"
        spm_path = model_path + '/spiece.model'
        tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>')
        token_ids, _ = tokenizer.encode("The <extra_id_0> walks in <extra_id_1> park")
        logger.info(token_ids)
        logger.info(tokenizer.ids_to_tokens(token_ids))
        # ['▁The', '▁<extra_id_0>', '▁walk', 's', '▁in', '▁<extra_id_1>', '▁park', '</s>']


    def test_gpt_tokenizer(self):
        # 和OpenAIGPTTokenizer分词效果对比
        from nlpcol.tokenizers import Tokenizer
        from transformers import OpenAIGPTTokenizer

        # 原生openai分词
        model_path = "/home/dataset/pretrain_ckpt/gpt/openai-gpt/"
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_path)

        text = "hello, my dog is cute"
        tokens = tokenizer.tokenize(text)
        logger.info(tokens)

        # nlpcol分词
        tokenizer = Tokenizer(model_path, tokenizer_type='bpe', token_start=None, token_end=None, token_unk="<unk>")
        token_ids, segments_ids = tokenizer.encode(text)
        logger.info(token_ids)
        logger.info(tokenizer.ids_to_tokens(token_ids))
        # ['hello</w>', ',</w>', 'my</w>', 'dog</w>', 'is</w>', 'cute</w>'] 

        
    def test_gpt_CDial_tokenizer(self):
        from nlpcol.tokenizers import Tokenizer
        model_path = "/home/dataset/pretrain_ckpt/gpt/CDial-GPT_LCCC-base"
        vocab_path = model_path + "/vocab.txt"

        text = '别爱我没结果'
        tokenizer = Tokenizer(vocab_path, do_lower_case=True)
        token_ids, segments_ids = tokenizer.encode(text)
        logger.info(token_ids)
        logger.info(tokenizer.ids_to_tokens(token_ids))
        # ['[CLS]', '别', '爱', '我', '没', '结', '果', '[SEP]']




