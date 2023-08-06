
import torch
import torch.nn as nn
from torch import Tensor, Size
import torch.nn.functional as F
import math

from typing import List, Optional, Tuple, Union

from nlpcol.activations import get_activation


class Config:
    def __init__(self, *args, **kwargs):
        # 以下参数来自config.json文件
        # 显式声明，支持下文自动补全
        self.attention_probs_dropout_prob = kwargs.get('attention_probs_dropout_prob')
        self.directionality = kwargs.get('directionality')
        self.hidden_act = kwargs.get('hidden_act')
        self.hidden_dropout_prob = kwargs.get('hidden_dropout_prob')
        self.hidden_size = kwargs.get('hidden_size')
        self.initializer_range = kwargs.get('initializer_range')
        self.intermediate_size = kwargs.get('intermediate_size')
        self.layer_norm_eps = kwargs.get('layer_norm_eps')
        self.max_position_embeddings = kwargs.get('max_position_embeddings')
        self.model_type = kwargs.get('model_type')
        self.num_attention_heads = kwargs.get('num_attention_heads')
        self.num_hidden_layers = kwargs.get('num_hidden_layers')
        self.pad_token_id = kwargs.get('pad_token_id')
        self.pooler_fc_size = kwargs.get('pooler_fc_size')
        self.pooler_num_attention_heads = kwargs.get('pooler_num_attention_heads')
        self.pooler_num_fc_layers = kwargs.get('pooler_num_fc_layers')
        self.pooler_size_per_head = kwargs.get('pooler_size_per_head')
        self.pooler_type = kwargs.get('pooler_type')
        self.type_vocab_size = kwargs.get('type_vocab_size')
        self.vocab_size = kwargs.get('vocab_size')



class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """layernorm层  TODO 后期兼容其他模型
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, input:Tensor) -> Tensor:
        """
        >>> input = torch.randn(2, 3)
        >>> input
        tensor([[-1.1199,  2.0004,  0.7479],
                [ 0.5189, -1.2847,  0.2426]])
        >>> mean = input.mean(-1, keepdim=True)
        >>> mean
        tensor([[ 0.5428],
                [-0.1744]])
        >>> var = (input - mean).pow(2).mean(-1, keepdim=True)
        >>> var
        tensor([[1.6437],
                [0.6291]])
        >>> o = (input - mean) / torch.sqrt(var + 1e-12)
        >>> o
        tensor([[-1.2969,  1.1369,  0.1600],
                [ 0.8741, -1.3998,  0.5258]])
        """
        mean = input.mean(-1, keepdim=True)  # 最后一位计算均值
        var = (input - mean).pow(2).mean(-1, keepdim=True)  # 方差
        o = (input - mean) / torch.sqrt(var + self.eps)

        return self.weight * o + self.bias


class BertEmbeddings(nn.Module):
    """用 word, position and token_type embeddings 构造embedding
    """
    def __init__(self, config: Config):
        super().__init__()
        # padding_idx 将相应位置的embedding全部设置为0， 不参与梯度计算，训练过程参数也不更新
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,

    ) -> torch.Tensor:

        device = input_ids.device
        btz, seq_len = input_ids.shape

        inputs_embeddings = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids, dtype=torch.long, device=device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeddings + token_type_embeddings

        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(btz, -1)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
        
        
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        
    def mask(self, inputs, key_masks):
        """其他Module继承支持重写mask函数
        """

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """对输入张量做形状变换
        B:batch_size  L:seq_length  H: hidden_size

        Args:
            x (torch.Tensor): B*L*H
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)   
        x = x.view(new_x_shape)  # torch.view 只能用于连续的张量  torch.reshape 可以用于不连续的张量  x_t.is_contiguous()  x:  B*L*num_head*head_size
        return x.permute(0, 2, 1, 3)  # x:  B*num_head*L*head_size

    def forward(self, query, key, value, key_mask):
        """_summary_

        Args:
            query (_type_): B*L*H
            key (_type_): B*L*H
            value (_type_): B*L*H
            key_mask (_type_): _description_
        """
        # 线性变换
        query_layer = self.query(query)
        key_layer = self.key(key)
        value_layer = self.value(value)
        # 形状变换
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        # attention
        # transpose和permute的区别：如果你只需要交换两个维度，transpose 是一个简单的选择。如果你需要进行更复杂的维度重排，你应该使用 permute。
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # B*num_head*L*L
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        key_mask = (1.0 - key_mask) * - 10000.0 # 传入的mask的非padding部分为1, padding部分为0
        attention_scores = attention_scores + key_mask
        
        # attention_scores归一化到0-1
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # 矩阵形状变换过程 B*num_head*L*L x B*num_head*L*head_size --> B*num_head*L*head_size
        context_layer = torch.matmul(attention_probs, value_layer)
        # transpose、permute等维度变换操作后，tensor在内存中不再是连续存储的，而view操作要求tensor的内存连续存储，
        # 需要contiguous来返回一个contiguous
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer


class AttentionOutput(nn.Module):
    """每个attention层后都会有一个线性层
    顺序为： Drop --> Add --> LayerNorm
    """
    def __init__(self, config:Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, context_layer: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_layer (torch.Tensor): attention层的输出结果
            input_tensor (torch.Tensor): BertEmbeddings层的输出
        """
        context_layer = self.dense(context_layer)
        context_layer = self.dropout(context_layer)
        context_layer = self.LayerNorm(context_layer + input_tensor)
        return context_layer
        
class FeedForward(nn.Module):
    """前馈层
    """
    def __init__(self, config:Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = get_activation(config.hidden_act)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.dense(hidden_state)
        hidden_state = self.intermediate_act_fn(hidden_state)
        return hidden_state

class FFNOutput(nn.Module):
    """顺序为： Drop --> Add --> LayerNorm
    """
    def __init__(self, config: Config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_state: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            hidden_state (torch.Tensor): FeedForward的输出结果
            input_tensor (torch.Tensor): AttentionOutput的输出结果

        Returns:
            torch.Tensor: _description_
        """
        hidden_state = self.dense(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.LayerNorm(hidden_state + input_tensor)
        return hidden_state


class BertLayer(nn.Module):
    """Transformer层 encoder block
        顺序为： Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm

    Args:
        nn (_type_): _description_
    """
    def __init__(self, config: Config):
        super().__init__()
        self.multiHeadAttention = MultiHeadAttentionLayer(config)
        self.attentionOutput = AttentionOutput(config)
        self.feedForward = FeedForward(config)
        self.ffnOutput = FFNOutput(config)
    
    def forward(self, hidden_states:torch.Tensor, attention_mask:torch.Tensor) -> torch.Tensor:
        context_layer = self.multiHeadAttention(
            hidden_states, hidden_states, hidden_states, attention_mask 
        )
        attention_output = self.attentionOutput(context_layer, hidden_states)
        feed_forward = self.feedForward(attention_output)
        layer_output = self.ffnOutput(feed_forward, attention_output)
        return layer_output
        

class BertEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states:torch.Tensor, attention_mask:torch.Tensor) -> torch.Tensor:
        """这里控制整个enceder层的输出格式, 暂时只输出最后一个隐藏藏 TODO
        """
        for i, layer_module in enumerate(self.layers):
            layer_output = layer_module(hidden_states, attention_mask)
            hidden_states = layer_output
        return hidden_states
                


class ModelBase(nn.Module):
    config: Config
    def _init_weights(self, module):
        """初始化权重  整个神经网络几乎都是由以下三种层组合成的"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_() # 默认就是0，此处应该多余了 TODO
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class BertPool(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pool_output = self.dense(first_token_tensor)
        pool_output = self.activation(pool_output)
        return pool_output


class BertMLM(nn.Module):
    """bert MLM任务"""
    def __init__(self, config: Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = get_activation(config.hidden_act)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias
        

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertModel(ModelBase):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.pooler = BertPool(config)
        self.mlm = BertMLM(config)

        # 初始化权重
        self._init_weights(self)
        

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # ========================= attention_mask =========================
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long() # bert默认0为mask_value


        # TODO 修改输入输出
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        print(encoder_outputs)

        mlm_scores = self.mlm(encoder_outputs)
        mlm_activation = get_activation('softmax')
        mlm_scores = mlm_activation(mlm_scores)

        return mlm_scores


def load_config(config_path):
    import json
    with open(config_path, 'r') as f:
        config = json.loads(f.read())
        print(config)
        return config

# config_path = '/home/dataset/pretrain_ckpt/bert/chinese_L-12_H-768_A-12/config.json'
# config = load_config(config_path)
# config = Config(**config)
# bert = BertLayer(config)

# print(1)


def variable_mapping(prefix = 'bert'):
    """
    不同代码参数命名不同，需要做参数映射   new_key: old_key
    """
    num_hidden_layers = 12

    mapping = {
        'embeddings.word_embeddings.weight':  f'{prefix}.embeddings.word_embeddings.weight',
        'embeddings.position_embeddings.weight':  f'{prefix}.embeddings.position_embeddings.weight',
        'embeddings.token_type_embeddings.weight':  f'{prefix}.embeddings.token_type_embeddings.weight',
        'embeddings.LayerNorm.weight':  f'{prefix}.embeddings.LayerNorm.gamma',
        'embeddings.LayerNorm.bias':  f'{prefix}.embeddings.LayerNorm.beta',
        'pooler.dense.weight': f'{prefix}.pooler.dense.weight',
        'pooler.dense.bias': f'{prefix}.pooler.dense.bias',
        'nsp.weight': 'cls.seq_relationship.weight',
        'nsp.bias': 'cls.seq_relationship.bias', 
        'mlm.dense.weight': 'cls.predictions.transform.dense.weight',
        'mlm.dense.bias': 'cls.predictions.transform.dense.bias',
        'mlm.LayerNorm.weight': 'cls.predictions.transform.LayerNorm.gamma',
        'mlm.LayerNorm.bias': 'cls.predictions.transform.LayerNorm.beta',
        'mlm.bias': 'cls.predictions.bias',
        'mlm.decoder.weight': 'cls.predictions.decoder.weight',
        'mlm.decoder.bias': 'cls.predictions.bias'
    }

    for i in range(num_hidden_layers):
        prefix_i = f'{prefix}.encoder.layer.%d.' % i

        mapping.update(
            {
                f"encoder.layers.{i}.multiHeadAttention.query.weight": prefix_i + 'attention.self.query.weight',
                f"encoder.layers.{i}.multiHeadAttention.query.bias": prefix_i + 'attention.self.query.bias',
                f"encoder.layers.{i}.multiHeadAttention.key.weight": prefix_i + 'attention.self.key.weight',
                f"encoder.layers.{i}.multiHeadAttention.key.bias": prefix_i + 'attention.self.key.bias',
                f"encoder.layers.{i}.multiHeadAttention.value.weight": prefix_i + 'attention.self.value.weight',
                f"encoder.layers.{i}.multiHeadAttention.value.bias": prefix_i + 'attention.self.value.bias',
                f"encoder.layers.{i}.attentionOutput.dense.weight": prefix_i + 'attention.output.dense.weight',
                f"encoder.layers.{i}.attentionOutput.dense.bias": prefix_i + 'attention.output.dense.bias',
                f"encoder.layers.{i}.attentionOutput.LayerNorm.weight": prefix_i + 'attention.output.LayerNorm.gamma',
                f"encoder.layers.{i}.attentionOutput.LayerNorm.bias": prefix_i + 'attention.output.LayerNorm.beta',
                f"encoder.layers.{i}.feedForward.dense.weight": prefix_i + 'intermediate.dense.weight',
                f"encoder.layers.{i}.feedForward.dense.bias": prefix_i + 'intermediate.dense.bias',
                f"encoder.layers.{i}.ffnOutput.dense.weight": prefix_i + 'output.dense.weight',
                f"encoder.layers.{i}.ffnOutput.dense.bias": prefix_i + 'output.dense.bias',
                f"encoder.layers.{i}.ffnOutput.LayerNorm.weight": prefix_i + 'output.LayerNorm.gamma',
                f"encoder.layers.{i}.ffnOutput.LayerNorm.bias": prefix_i + 'output.LayerNorm.beta'
            }
        )

    return mapping
        



