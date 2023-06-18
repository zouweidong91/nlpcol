
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self):
        self.model_name = 'TextCNN'
        self.embedding_pretrained = None                                       # 预训练词向量

        self.dropout = 0.5                                              # 随机失活
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)



class TextCNN(nn.Module):
    def __init__(self, config:Config, vocab_size:int, num_classes:int):
        """_summary_

        Args:
            config (Config): 模型配置信息
        """
        super(TextCNN, self).__init__()
        if config.embedding_pretrained:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, config.embed, padding_idx=vocab_size - 1)


        # 卷积核宽度和词向量维度相同， 这样卷积扫描时只会在长度方向移动
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=config.num_filters, kernel_size=(k, config.embed))
            for k in config.filter_sizes])

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        """
        conv: 1,256,(k,300)  卷积运算是扫描过程，与卷积核元素对应位置相乘再求和，得到的是个标量

        Args:
            x (_type_): x.shape:  btz, 1, seq_len, hdz
            conv (_type_): conv.weight  num_filters个kernel_size形状的矩阵
        """
        x = F.relu(conv(x)).squeeze(3)  # (btz, num_filters, seq_len-k+1, 1): 128 256 31 1 --> 128 256 31
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # 最大池化： 128 256 31 --> 128 256 1 --> 128 256
        return x

    def forward(self, token_ids):
        """
        Args:
            input (tuple): 输入数据，token_ids
        """
        out:torch.Tensor = self.embedding(token_ids)  # 128,32,300
        out = out.unsqueeze(1)    # 输入通道为1 128,1,32,300

        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)  # 128 256*3
        out = self.dropout(out)
        out = self.fc(out)
        return out
