"""
常见数据集的加载实现
"""
import json
from nlpcol.utils.snippets import ListDataset, IterDataset, text_segmentate



class SentimentDataset(ListDataset):
    maxlen: int # 最大长度
    @classmethod
    def load_data(cls, filename):
        """加载数据，并尽量划分为不超过maxlen的句子
        样本示例：
            贝贝好爱干净 每天出门都要洗澡 还喜欢喝蒙牛 不喜欢蹲地方 喜欢坐凳子上还喜欢和我坐在一起~ 1
            感觉好像是文科生看一本《高等数学》的教材一样，流水账一般，只是背景很好罢了，选择在这样一个竞争激烈的时代，写了那么一个催人奋进的故事，文笔不咋地。      0
        """
        D = []
        seps, strips = u'\n。！？!?；;，, ', u'；;，, '
        with open(filename, encoding='utf-8') as f:
            for l in f:
                text, label = l.strip().split('\t')
                for t in text_segmentate(text, cls.maxlen - 2, seps, strips):
                    D.append((t, int(label)))
        print(D[:2])
        return D


# 一般文件格式尽量用jsonl, 且限制字段名为 text label
class JsonlDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """
        {"label": "xx", "text": "xxxxx"}
        {"label": "xx", "text": "xxxxxx"}      
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                text, label = str(l['text']), str(l['label'])
                D.append((text, label))
        return D



class PeopleDailyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            for l in f.split('\n\n'):
                if not l:
                    continue
                d = ['']
                for i, c in enumerate(l.split('\n')):
                    char, flag = c.split(' ')
                    d[0] += char
                    if flag[0] == 'B':
                        d.append([i, i, flag[2:]])
                    elif flag[0] == 'I':
                        d[-1][1] = i
                D.append(d)
        return D[:1000]

