
不同任务类型对应的损失函数：

1、回归问题 regression

**MSELoss 均方根误差**

$\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,$

2、单分类问题 single_label_classification

**CrossEntropyLoss 交叉熵**

> 文本分类: 对整个文本单分类
ner： 对每个token分类
生成模型： 对每个token在vocab_size尺寸分类

$\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})} \cdot \mathbb{1}$


3、多分类问题 multi_label_classification

**BCEWithLogitsLoss 二元交叉熵损失函数**
BCELoss需要在计算前做sigmoi处理

$\ell_c(x, y) = L_c = \{l_{1,c},\dots,l_{N,c}\}^\top, \quad
        l_{n,c} = - w_{n,c} \left[ p_c y_{n,c} \cdot \log \sigma(x_{n,c})
        + (1 - y_{n,c}) \cdot \log (1 - \sigma(x_{n,c})) \right]$


