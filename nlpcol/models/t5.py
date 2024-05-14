
import copy
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from nlpcol.activations import get_activation
from nlpcol.generation import EncDecGenerationMixin
from nlpcol.layers.attention import AttentionOutput, EncDecAttention
from nlpcol.layers.ffn import FFN, DenseGatedActDense
from nlpcol.layers.layer import RMSNorm
from nlpcol.layers.pe import RelativePositionalT5
from torch import Size, Tensor

from .base import BaseConfig, BaseModel

# T5-Small (60 million parameters): gs://t5-data/pretrained_models/small
# T5-Base (220 million parameters): gs://t5-data/pretrained_models/base
# T5-Large (770 million parameters): gs://t5-data/pretrained_models/large
# T5-3B (3 billion parameters): gs://t5-data/pretrained_models/3B
# T5-11B (11 billion parameters): gs://t5-data/pretrained_models/11B

# https://kexue.fm/archives/7867
# https://zhuanlan.zhihu.com/p/88438851 T5 模型：NLP Text-to-Text 预训练模型超大规模探索
# https://huggingface.co/docs/transformers/main/en/model_doc/mt5#transformers.MT5Model

# Transformer Encoder-Decoder 模型；
# BERT-style 式的破坏方法；
# Replace Span 的破坏策略；
# 15 %的破坏比；
# 3 的破坏时小段长度。

# T5.1.0和T5.1.1区别
# FFN 把relu激活的第一个变化层改为了gelu激活的门控线性单元，这样FFN层增加了50%参数，但是从论文效果看效果明显增加
# 此外，T5.1.1还对Embedding层做了改动，
# 原来在T5.1.0中，Encoder和Decoder的Embedding层、Decoder最后预测概率分布的Softmax层都是共享同一个Embedding矩阵的，
# 现在T5.1.1只让Encoder和Decoder的Embedding层共享，而Decoder最后预测概率分布的Softmax层则用了一个独立的Embedding矩阵，
# 当然这会让参数量大大增加，但Google的结论说这样做效果会更好
# t5不使用bias，并且使用rmsnorm
# t5为pre_norm

# 实现原始mt5  t5没有中文预训练权重。用Mt5实现 T5.1.1

"""
{
    "_name_or_path": "/home/patrick/hugging_face/t5/mt5-base",
    "architectures": [
        "MT5ForConditionalGeneration"
    ],
    "d_ff": 2048,
    "d_kv": 64,
    "d_model": 768,
    "decoder_start_token_id": 0,
    "dropout_rate": 0.1,
    "eos_token_id": 1,
    "feed_forward_proj": "gated-gelu",
    "initializer_factor": 1.0,
    "is_encoder_decoder": true,
    "layer_norm_epsilon": 1e-06,
    "model_type": "mt5",
    "num_decoder_layers": 12,
    "num_heads": 12,
    "num_layers": 12,
    "output_past": true,
    "pad_token_id": 0,
    "relative_attention_num_buckets": 32,
    "tie_word_embeddings": false,
    "tokenizer_class": "T5Tokenizer",
    "transformers_version": "4.10.0.dev0",
    "use_cache": true,
    "vocab_size": 250112
}
"""


class Config(BaseConfig):
    def __init__(self, **kwargs):
        # 通用配置
        self.d_model: int = kwargs.get('d_model')
        self.d_ff: int = kwargs.get('d_ff')
        self.n_heads: int = kwargs.get('num_heads')
        self.vocab_size: int = kwargs.get('vocab_size')
        self.num_layers: int = kwargs.get('num_layers')
        self.dropout_rate: float = kwargs.get('dropout_rate')
        self.initializer_range: float = kwargs.get('initializer_factor')
        self.layer_norm_eps: float = kwargs.get('layer_norm_epsilon')
        self.hidden_act: str = kwargs.get('hidden_act', "gelu_new")

        self.pad_token_id: int = kwargs.get('pad_token_id')
        self.bos_token_id: int = kwargs.get('bos_token_id', self.pad_token_id) # T5 bos_token_id 默认为 pad_token_id
        self.eos_token_id: int = kwargs.get('eos_token_id')
        
        self.max_position:int = kwargs.get('max_seq_length', 512)  # 需要大于max(tar_len, src_len)， 相对位置编码用
        self.max_batch_size:int = kwargs.get('max_batch_size', 16)  # 推理过程中batch_size不能大于此值， kv_cache用

        self.use_bias: bool = False
        self.layer_norm_type = 'pre'
        self.tie_word_embeddings: bool = False


        # T5 config文件配置
        self.architectures: str = kwargs.get('architectures')
        self.model_type: str = kwargs.get('model_type')
        self.decoder_start_token_id: int = kwargs.get('decoder_start_token_id')
        self.feed_forward_proj: str = kwargs.get('feed_forward_proj')
        self.is_encoder_decoder: bool = kwargs.get('is_encoder_decoder')
        self.output_past: bool = kwargs.get('output_past')
        self.relative_attention_num_buckets: int = kwargs.get('relative_attention_num_buckets')
        self.tokenizer_class: str = kwargs.get('tokenizer_class')
        self.use_cache: bool = kwargs.get('use_cache')



class T5Embeddings(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, input_ids: Optional[torch.LongTensor]) -> Tensor:
        embeddings = self.token_embeddings(input_ids)
        embeddings = self.dropout(embeddings)
        return embeddings


class T5FFN(FFN):
    def get_ff(self, config):
        return DenseGatedActDense(config)

    def get_ln(self, config: Config):
        return RMSNorm(config.d_model, eps=config.layer_norm_eps)


class T5AttentionLayer(EncDecAttention):
    def __init__(self, config: Config, has_relative_attention_bias=False, is_self_atten=True):
        """_summary_

        Args:
            config (Config): _description_
            has_relative_attention_bias (bool, optional): 是否有相对位置编码. Defaults to False.
            is_self_atten (bool, optional): selfAtten or crossAtten. Defaults to True.
        """
        super().__init__(config, is_self_atten)

        self.relative_attention_num_buckets = config.relative_attention_num_buckets

        self.relative_attention_bias = None
        if has_relative_attention_bias: # 只有selfAtten时需要，且仅第0层初始化，其它层共享权重。
            self.relative_attention_bias = nn.Embedding(config.relative_attention_num_buckets, config.n_heads)
            # print(config.max_seq_length)
            self.relative_position = RelativePositionalT5(
                config.max_position, config.max_position, self.relative_attention_num_buckets, is_decoder=self.is_decoder
            )

    def compute_bias(self, qlen, klen):
        """计算偏置 selfAtten 处添加偏置"""
        device = self.relative_attention_bias.weight.device
        relative_position = self.relative_position(klen, klen).to(device)
        bias:Tensor = self.relative_attention_bias(relative_position)  # shape (klen, klen, num_heads)
        bias = bias.permute([2, 0, 1]).unsqueeze(0)  #  (1, num_heads, klen, klen)

        # if self.training or not self.is_decoder:
        #     pass
        # if self.k_cache is not None: # 推理阶段decoder,qlen=1, 只选取query最后一个position bias。 注意，crossAtten没有位置bias
        bias = bias[:, :, -qlen:, :]

        return bias

    def applay_att_score_bias(self, scores:Tensor) -> Tensor:
        """
        T5 位置编码通过在attention_scores上加一个可训练偏置项  TODO relative_positions重复计算，需优化
        """
        if self.relative_attention_bias is None:  # crossAtten不需要位置编码
            return scores

        qlen, klen = scores.shape[2:] # 推理时decoder qlen = 1
        bias = self.compute_bias(qlen, klen)
        return scores + bias

    def score_scale(self, scores:Tensor) -> Tensor:
        # T5 scores不需要缩放
        return scores


class T5Layer(nn.Module):
    """TODO 抽象为 Transformer类
    Encoder的顺序
        LN --> self_Att --> Add --> LN --> FFN --> Add
    Decoder的顺序
        LN --> self_Att --> Add --> LN --> cross_Att --> Add --> LN --> FFN --> Add
    """
    def __init__(self, config: Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.self_attention = T5AttentionLayer(config, has_relative_attention_bias=has_relative_attention_bias, is_self_atten=True)
        self.self_attention_output = AttentionOutput(config)

        if self.is_decoder:
            self.layer_norm2 = RMSNorm(config.d_model, eps=config.layer_norm_eps)
            self.cross_attention = T5AttentionLayer(config, is_self_atten=False)
            self.cross_attention_output = AttentionOutput(config)

        self.ffn = T5FFN(config)


    def forward(
        self, 
        hidden_states:Tensor, 
        attention_mask:Tensor=None,   
        encoder_hidden_states:Tensor=None, 
        encoder_attention_mask:Tensor=None,
        start_pos:int=0
    ) -> Tensor:
        """
        Args:
            attention_mask (Tensor, optional): 
                即输入的padding_mask。
                encoder端输入有padding信息，为非None
                decoder端输入没有padding信息，为None
        """

        # self attention
        normed_hidden_states = self.layer_norm(hidden_states)
        context_layer = self.self_attention(
            normed_hidden_states, normed_hidden_states, normed_hidden_states, attention_mask, start_pos
        )
        hidden_states = self.self_attention_output(context_layer, hidden_states) # add为标准化之前的hidden_states

        # cross attention
        # query: selfattntion的输出   key_value： Encoder端的输出
        if self.is_decoder:
            normed_hidden_states = self.layer_norm2(hidden_states)
            context_layer = self.cross_attention(
                normed_hidden_states, encoder_hidden_states, encoder_hidden_states, encoder_attention_mask, start_pos
            )
            hidden_states = self.cross_attention_output(context_layer, hidden_states)

        # feedforward
        ffn_output = self.ffn(hidden_states)
        return ffn_output

@dataclass
class T5StackOutput:
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[List[torch.FloatTensor]] = None
    attention_mask: Optional[torch.LongTensor] = None # 推理时还需要用到


class T5Stack(nn.Module):
    """
    has_relative_attention_bias 只有第一层的selfAtten才需要相对位置编码
    """
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [T5Layer(config, has_relative_attention_bias=bool(i==0)) for i in range(config.num_layers)]
        )
        
        for layer in self.layers[1:]:
            layer.self_attention.relative_attention_bias = self.layers[0].self_attention.relative_attention_bias # relative_attention_bias权重共享
            layer.self_attention.relative_position = self.layers[0].self_attention.relative_position

        self.final_layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self, 
        hidden_states:Tensor, 
        attention_mask:Tensor=None,
        encoder_hidden_states:Tensor=None, 
        encoder_attention_mask:Tensor=None,
        start_pos:int=0
    ) -> Tensor:

        all_hidden_states = []

        for i, layer_module in enumerate(self.layers):
            all_hidden_states.append(hidden_states)
            hidden_states = layer_module(
                hidden_states, 
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                start_pos = start_pos
            )

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        all_hidden_states.append(hidden_states)

        return T5StackOutput(
            last_hidden_state = hidden_states,
            hidden_states = all_hidden_states,
            attention_mask = attention_mask
        )


@dataclass
class Seq2SeqLMOutput:
    loss: torch.FloatTensor = None
    lm_logits: torch.FloatTensor = None
    encoder_last_hidden_state: torch.FloatTensor = None
    decoder_last_hidden_state: torch.FloatTensor= None
    encoder_attention_mask: Optional[torch.LongTensor] = None
    encoder_hidden_states: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[List[torch.FloatTensor]] = None


class T5Model(BaseModel, EncDecGenerationMixin):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)
        self.config: Config
        
        self.embed = T5Embeddings(config) # 在enc和dec之间共享embedding

        enc_config = copy.deepcopy(config)
        enc_config.is_decoder = False
        self.encoder = T5Stack(enc_config)

        dec_config = copy.deepcopy(config)
        dec_config.is_decoder = True
        self.decoder = T5Stack(dec_config)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.tie_weights()


    def forward(
        self,
        input_ids:torch.LongTensor=None,
        decoder_input_ids:torch.LongTensor=None,
        attention_mask:torch.FloatTensor=None, # encoder端的selfAtten以及decoder端的crossAtten
        encoder_outputs=None,
        labels:torch.LongTensor=None,
        start_pos:int=0 # 训练状态下 start_pos=0
    ):
        # encoder
        if attention_mask is None: # [batch_size, to_seq_length]
            attention_mask = (input_ids != self.config.pad_token_id).long()
        
        if encoder_outputs is None: # 训练或者第一次推理时才执行
            hidden_states_enc = self.embed(input_ids)
            enc_output:T5StackOutput = self.encoder(hidden_states_enc, attention_mask, start_pos = 0)
        else:
            enc_output = T5StackOutput(
                last_hidden_state=encoder_outputs, 
                attention_mask=attention_mask
            )
        
        # label shift right  
        # 北 京 在 中 国 </s>  labels
        # <s> 北 京 在 中 国   decoder_input_ids
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = torch.zeros_like(labels)
            decoder_input_ids[..., 1:] = labels[..., :-1].clone() # 向右偏移一位
            decoder_input_ids[..., 0] = self.config.bos_token_id # 起始位置用padding代表
            
            # batch模式下，外部数据组装时用-100进行padding，这里再进行替换
            decoder_input_ids.masked_fill_(decoder_input_ids == -100, self.config.pad_token_id)
            
        # decoder
        hidden_states_dec = self.embed(decoder_input_ids)
        dec_output:T5StackOutput = self.decoder(
            hidden_states = hidden_states_dec, 
            attention_mask = None, # 后续生成下三角矩阵
            encoder_hidden_states = enc_output.last_hidden_state,
            encoder_attention_mask = attention_mask,
            start_pos = start_pos
        )

        lm_logits: Tensor = self.lm_head(dec_output.last_hidden_state)

        # 计算损失
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device)
            loss = loss_fn(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        return Seq2SeqLMOutput(
            loss = loss,
            lm_logits = lm_logits,
            encoder_last_hidden_state = enc_output.last_hidden_state,
            decoder_last_hidden_state = dec_output.last_hidden_state,
            encoder_attention_mask = enc_output.attention_mask,
            encoder_hidden_states = enc_output.hidden_states,
            decoder_hidden_states = dec_output.hidden_states,
        )

    @property
    def origin_embedding_keys(self) -> list:
        """原始模型和token_embedding相关的参数名 keep_token精简embedding用"""
        return [
            'shared.weight',
            'lm_head.weight',
        ]

    def variable_mapping(self):
        """
        不同代码参数命名不同，需要做参数映射   new_key: old_key
        """
        mapping = {
            "embed.token_embeddings.weight": "shared.weight",
            "encoder.layers.0.self_attention.relative_attention_bias.weight": "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            "decoder.layers.0.self_attention.relative_attention_bias.weight": "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
        }

        self_atten_fn = lambda stack, i, j: {
                    f"{stack}.layers.{i}.self_attention.q.weight": f"{stack}.block.{i}.layer.{j}.SelfAttention.q.weight",
                    f"{stack}.layers.{i}.self_attention.k.weight": f"{stack}.block.{i}.layer.{j}.SelfAttention.k.weight",
                    f"{stack}.layers.{i}.self_attention.v.weight": f"{stack}.block.{i}.layer.{j}.SelfAttention.v.weight",
                    f"{stack}.layers.{i}.self_attention_output.o.weight": f"{stack}.block.{i}.layer.{j}.SelfAttention.o.weight",
                    f"{stack}.layers.{i}.layer_norm.weight": f"{stack}.block.{i}.layer.{j}.layer_norm.weight",
                }
        cross_atten_fn = lambda stack, i, j: {
                    f"{stack}.layers.{i}.cross_attention.q.weight": f"{stack}.block.{i}.layer.{j}.EncDecAttention.q.weight",
                    f"{stack}.layers.{i}.cross_attention.k.weight": f"{stack}.block.{i}.layer.{j}.EncDecAttention.k.weight",
                    f"{stack}.layers.{i}.cross_attention.v.weight": f"{stack}.block.{i}.layer.{j}.EncDecAttention.v.weight",
                    f"{stack}.layers.{i}.cross_attention_output.o.weight": f"{stack}.block.{i}.layer.{j}.EncDecAttention.o.weight",
                    f"{stack}.layers.{i}.layer_norm2.weight": f"{stack}.block.{i}.layer.{j}.layer_norm.weight",
                }
        ffn_fn = lambda stack, i, j: {
                    f"{stack}.layers.{i}.ffn.ff.dense_1.weight": f"{stack}.block.{i}.layer.{j}.DenseReluDense.wi_0.weight",
                    f"{stack}.layers.{i}.ffn.ff.dense_2.weight": f"{stack}.block.{i}.layer.{j}.DenseReluDense.wi_1.weight",
                    f"{stack}.layers.{i}.ffn.ff.dense_output.weight": f"{stack}.block.{i}.layer.{j}.DenseReluDense.wo.weight",
                    f"{stack}.layers.{i}.ffn.layer_norm.weight": f"{stack}.block.{i}.layer.{j}.layer_norm.weight",
                }

        for i in range(self.config.num_layers):
            mapping.update(self_atten_fn('encoder', i, 0))
            mapping.update(ffn_fn('encoder', i, 1))

        for i in range(self.config.num_layers):
            mapping.update(self_atten_fn('decoder', i, 0))
            mapping.update(cross_atten_fn('decoder', i, 1))
            mapping.update(ffn_fn('decoder', i, 2))

        return mapping
        
