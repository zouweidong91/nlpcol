
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from nlpcol.layers.layer import LayerNorm
from nlpcol.models.base import BaseConfig as Config
from torch import Size, Tensor


class BaseEmbeddings(nn.Module):

    dropout: nn.Dropout
    token_embeddings: nn.Embedding

    def get_token_embeddings(self, input_ids:Tensor):
        return self.token_embeddings(input_ids)

    def get_position_embeddings(self, position_ids:Tensor, input_ids:Tensor, start_pos:int):
        return 0

    def get_token_type_embeddings(self, token_type_ids:Tensor, input_ids:Tensor, start_pos:int):
        return 0

    def embed_layer_norm(self, embeddings):
        return embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        start_pos:int=0
    ) -> Tensor:

        inputs_embeddings = self.get_token_embeddings(input_ids)
        position_embeddings = self.get_position_embeddings(position_ids, input_ids, start_pos)
        token_type_embeddings = self.get_token_type_embeddings(token_type_ids, input_ids, start_pos)
        embeddings = inputs_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.embed_layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertEmbeddings(BaseEmbeddings):
    """bert用 word, position and token_type embeddings 构造embedding
    """
    def __init__(self, config: Config):
        super().__init__()
        # padding_idx 将相应位置的embedding全部设置为0， 不参与梯度计算，训练过程参数也不更新
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position, config.d_model)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.d_model)

        self.layer_norm = LayerNorm(config.d_model, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_rate)

    def get_position_embeddings(self, position_ids:Tensor, input_ids:Tensor, start_pos:int):
        device = input_ids.device
        btz, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(start_pos+seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(btz, -1)

        position_ids = position_ids[:, -seq_len:]   # unilm推理阶段seq_len=1
        return self.position_embeddings(position_ids)

    def get_token_type_embeddings(self, token_type_ids:Tensor, input_ids:Tensor, start_pos:int):
        device = input_ids.device
        btz, seq_len = input_ids.shape

        if token_type_ids is None:
            token_type_ids = torch.zeros([btz, seq_len], dtype=torch.long, device=device)

        token_type_ids = token_type_ids[:, -seq_len:] # 推理阶段seq_len=1
        return self.token_type_embeddings(token_type_ids)

    def embed_layer_norm(self, embeddings):
        return self.layer_norm(embeddings)


class GptEmbeddings(BaseEmbeddings):
    """用 word, position 构造embedding
    """
    def __init__(self, config: Config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_position, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def get_position_embeddings(self, position_ids:Tensor, input_ids:Tensor, start_pos:int):
        device = input_ids.device
        btz, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(start_pos+seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(btz, -1)

        position_ids = position_ids[:, -seq_len:]   # 推理阶段seq_len=1
        return self.position_embeddings(position_ids)

    def get_token_type_embeddings(self, token_type_ids:Tensor, input_ids:Tensor, start_pos:int):
        # 部分gpt模型如CDial有token_type_ids
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(1))
            return self.token_embeddings(token_type_ids)
        else:
            return 0


class T5Embeddings(BaseEmbeddings):
    """使用相对位置编码
    """
    def __init__(self, config: Config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.dropout = nn.Dropout(config.dropout_rate)


