
'''
excel文件读写工具
'''

import pandas as pd
from typing import List



class ExcelHandle:

    @classmethod
    def read(cls, file_path, sheet_name='Sheet1', todict=False):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df = df.fillna("")
        columns = list(df.columns)
        print(columns)

        for index, line in df.iterrows():
            if todict:
                line = cls().to_dict(line)
            yield line

    def to_dict(self, item):
        '''
        item.keys()
        item.values
        '''
        return {k:item[k] for k in item.keys()}


    @classmethod
    def write(cls, file_path, data:List[list], columns:list, sheet_name='Sheet1', width=10):
        df = pd.DataFrame(data, columns=columns)

        # 设置列宽
        # 需要安装pip install xlsxwriter 
        with pd.ExcelWriter(file_path) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            [
                worksheet.set_column(i, i, width)
                for i, c in enumerate(columns)
            ]


    @classmethod
    def write2(cls, file_path, item_list:List[dict], columns=None, key_map:dict=None, width=10):
        """_summary_

        Args:
            file_path (_type_): 输出文件路径
            item_list (List[dict]): 输入数据流
            columns (_type_, optional): 字段名. Defaults to None.
            key_map (dict, optional): 字段名映射. Defaults to None.
        """
        D = []
        for item in item_list:

            if not columns:
                columns = item.keys()

            D.append(
                [item[k] for k in columns]
            )
        
        if key_map:
            columns = [key_map.get(k, k) for k in columns]
        
        if not columns:
            return
            
        cls.write(file_path, D, columns, width=width)



    def foo(self):
        print(2)

    @staticmethod
    def foo1():
        ExcelHandle().foo()

    @classmethod
    def foo2(cls):
        cls().foo()


class CsvHandle(ExcelHandle):
    @classmethod
    def read(cls, file_path, todict=False):
        df = pd.read_csv(file_path)
        columns = list(df.columns)
        print(columns)

        for index, line in df.iterrows():
            if todict:
                line = cls().to_dict(line)
            yield line


if __name__=='__main__':

    ExcelHandle.foo1()
    ExcelHandle.foo2()
