# decoder基础模型

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from nlpcol.generation import DecGenerationMixin
from nlpcol.layers.attention import DecAttention
from nlpcol.layers.embed import GptEmbeddings
from nlpcol.layers.layer import LayerNorm
from torch import Size, Tensor

from ._base import BaseConfig as Config
from ._base import BaseModel
from .encoder import Block, Stack, StackOutput


class DecBlock(Block):
    def get_attention(self, config: Config):
        return DecAttention(config)
    

class DecStack(Stack):
    def get_block(self, config: Config):
        """下游模型用其他组件时继承用"""
        return DecBlock(config)

@dataclass
class CausalLMOutput:
    loss: torch.FloatTensor = None
    lm_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor= None
    hidden_states: Optional[List[torch.FloatTensor]] = None
    attention_mask: torch.LongTensor=None


class Decoder(BaseModel, DecGenerationMixin):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)
        
        self.embed = self.get_embed(config)
        self.decoder = DecStack(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.has_final_ln = config.has_final_layernorm

        if self.has_final_ln:
            self.final_norm = LayerNorm(config.d_model, config.layer_norm_eps)
        
        self.tie_weights()

    def get_embed(self, config: Config) -> GptEmbeddings:
        raise NotImplementedError

    def forward(
        self,
        input_ids:torch.LongTensor=None,
        token_type_ids:torch.LongTensor=None,
        labels:torch.LongTensor=None,
        attention_mask:torch.LongTensor=None,
        position_ids:torch.LongTensor=None,
        start_pos:int=0
    ): 
        """_summary_

        Args:
            attention_mask (torch.LongTensor, optional): 
                也就是input输入的padding_mask
                对于decoder-only模型
                train模式下：
                    right-padding，attention_mask为None。因为loss计算时，label的padding为-100，会自动忽略掉padding位置
                infer模式下：
                    batch模式下为left-padding，每个step在attention时时需要mask掉padding位，attention_mask为非None


            position_ids (torch.LongTensor, optional): 
                batch时generation才传入

            start_pos (int, optional): 
                训练状态下 start_pos=0
                推理时，为生成的step步

        """
        # embed
        hidden_states = self.embed(input_ids, token_type_ids, position_ids, start_pos=start_pos)

        # decoder
        dec_output:StackOutput = self.decoder(
            hidden_states = hidden_states, 
            attention_mask = attention_mask,
            start_pos = start_pos
        )

        # 是否执行final_norm
        hidden_state = dec_output.last_hidden_state
        if self.has_final_ln:
            hidden_state = self.final_norm(hidden_state)

        # lm_head
        lm_logits: Tensor = self.lm_head(hidden_state)   # (bsz, seq_len, vocab_size)
        loss = None

        # label shift right
        #   北   京   在   哪   里   在   中   国   </s>      原句
        # -100 -100 -100 -100 -100  在   中   国   </s>      labels  
        # -100 -100 -100 -100  在   中   国   </s>           shift_labels  
        #   北   京   在   哪   里   在   中   国             shift_logits
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        return CausalLMOutput(
            loss = loss,
            lm_logits = lm_logits,
            last_hidden_state = dec_output.last_hidden_state,
            hidden_states = dec_output.hidden_states,
            attention_mask = attention_mask
        )
