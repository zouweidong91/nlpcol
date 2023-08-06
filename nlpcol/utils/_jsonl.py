

import json, re
from ._file import FileTool

class JsonlCat:
    """jsonl数据查看工具
    """
    def __init__(self):
        self.offset = 0

    def _open(self, file_path):
        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                item = json.loads(line)
                yield item

    def _read(self, item_iter, count = 10):
        """读取文件
        """
        offset = self.offset
        self.offset += count
        
        for item in item_iter:
            if offset > 0:
                offset -= 1
                continue
            
            self._print(item)
            count -= 1
            if count <= 0:
                break
    
    def read_file(self, file_path, count = 10):
        """读取文件
        """
        item_iter = self._open(file_path)
        self._read(item_iter, count)

    def read_list(self, item_list, count=10):
        """读取列表
        """
        self._read(item_list, count)
                
    def _print(self, item, indent= ' '):
        """可复写此函数，适配不同的数据结构
            {'text': '2、首次入选“中国500强”企业名单。',
            'pred': [{'pos': [2, 17], 'name': '首次入选“中国500强”企业名单', 'type': '科目名_无值'}],
            'label': [{'pos': [2, 17], 'name': '首次入选“中国500强”企业名单', 'type': '科目名_无值'}],
            'project_name': '深圳市企业首次入选“中国500强”奖励'}
        """
        label_str = lambda label: "\t\t".join([f"{k}: {v}" for k, v in label.items()])
        label_list_str = lambda label_list: "\n".join([indent*2 + label_str(label) for label in label_list])

        sort_label_list = lambda label_list: sorted(label_list, key=lambda ele: ele['pos'][0])  # 排序
        labels_list = sort_label_list(item['label'])
        pred_list = sort_label_list(item['pred'])

        print("\n".join([
            "id: {}".format(item["id"]),
            "text: ",
            indent*2 + item['text'],
            "label: ",
            label_list_str(labels_list),
            "pred",
            label_list_str(pred_list),
            '\n'
        ]))


class JsonlTool(FileTool):
    """jsonl读写操作
    """
    def _open(self, file_path):
        with open(file_path, 'r', encoding='utf8') as in_f:
            for line in in_f:
                try:
                    item = json.loads(line)
                except:
                    item = line.strip()   # 纯文本
                yield item

    @classmethod
    def write(self, item_iter, file_path):
        """jsonl写入

        Args:
            item_iter ([list or iter]): [description]
            file_path ([str]): [description]
        """
        with open(file_path, 'w', encoding='utf8') as out_f:
            for item in item_iter:
                out_f.write(
                    json.dumps(item, ensure_ascii=False) + '\n'
                )
                
