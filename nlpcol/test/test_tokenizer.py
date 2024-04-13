

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

    def test_transformers_tokenizer(self):
        from transformers import AutoTokenizer
        pretraine_dir = '/home/dataset/pretrain_ckpt/t5/mt5-base/'
        tokenizer = AutoTokenizer.from_pretrained(pretraine_dir)
        text = ["The <extra_id_0> walks in <extra_id_1> park", "The <extra_id_0> walks"]
        inputs = tokenizer(
            text,
            padding='max_length',  # pading补全max_length， 默认longest
            max_length=10,
            return_tensors='pt'  # 返回张量
        )
        logger.info(inputs['input_ids'])
        logger.info(inputs['attention_mask'])


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

    def test_gpt2_tokenizer(self):
        # 和OpenAIGPTTokenizer分词效果对比
        from nlpcol.tokenizers import Tokenizer
        from transformers import GPT2Tokenizer
        # 原生openai分词
        model_path = "/home/dataset/pretrain_ckpt/gpt2/openai-gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)

        text = "My name is Lewis and I like to"
        tokens = tokenizer.tokenize(text)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.tolist()
        logger.info("\n %s \n %s \n", tokens, input_ids)

        # nlpcol分词
        tokenizer = Tokenizer(
            model_path, 
            tokenizer_type='bbpe', 
            token_start=None, 
            token_end=None, 
            token_unk='<|endoftext|>',
            do_lower_case=False,
            do_basic_tokenize=False,
            )

        token_ids, segments_ids = tokenizer.encode(text)
        logger.info(tokenizer.ids_to_tokens(token_ids))
        logger.info(tokenizer.decode(token_ids))
        logger.info(token_ids)
        # [3666, 1438, 318, 10174, 290, 314, 588, 284]
        # ['My', 'Ġname', 'Ġis', 'ĠLewis', 'Ġand', 'ĠI', 'Ġlike', 'Ġto']  
        # Ġ 为空格
        

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




