import json

class FileTool:
    """File读写操作
    """
    def _open(self, file_path):
        with open(file_path, 'r', encoding='utf8') as in_f:
            for line in in_f:
                item = line.strip()   # 纯文本
                yield item

    def read_iter(self, file_path):
        """FileTool().read_iter
        """
        yield from self._open(file_path)

    def read_list(self, file_path):
        item_list = []
        for item in self._open(file_path):
            item_list.append(item)
        
        return item_list

    @classmethod
    def write(cls, item_iter, file_path):
        """写入 FileTool.write

        Args:
            item_iter ([list or iter]): [description]
            file_path ([str]): [description]
        """
        with open(file_path, 'w', encoding='utf8') as out_f:
            for item in item_iter:
                out_f.write(
                    item + '\n'
                )
                

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
    def write(cls, item_iter, file_path):
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
                
