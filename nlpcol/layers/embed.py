
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from nlpcol.models.base import BaseConfig as Config
from torch import Size, Tensor


class GptEmbeddings(nn.Module):
    """用 word, position 构造embedding
    """
    def __init__(self, config: Config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_position, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        token_type_ids: Optional[torch.LongTensor],
        position_ids: Optional[torch.LongTensor] = None,
        start_pos:int=0 # 解码时使用
    ) -> Tensor:

        device = input_ids.device
        btz, seq_len = input_ids.shape

        inputs_embeddings = self.token_embeddings(input_ids)

        if position_ids is None:
            position_ids = torch.arange(start_pos+seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(btz, -1)
        
        position_ids = position_ids[:, start_pos: start_pos+seq_len] # 解码时位置编码也要根据输入长度截取
        position_embeddings = self.position_embeddings(position_ids)

        # 统一embedding写法 后续模型继承
        # TODO token_type_ids is None   token_type_embedding = 0
        # ? 没有归一化
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(1))
            token_type_embeddings = self.token_embeddings(token_type_ids)
        else:
            token_type_embeddings = 0
            

        embeddings = inputs_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
        