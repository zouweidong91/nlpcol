"""
examples依赖的工具函数
"""
import os

from nlpcol.config import TrainConfig


def model_name_gene(train_config:TrainConfig, model_name, dataset_name) -> str:
    """训练过程模型名字生成

    Args:
        train_config (TrainConfig): 训练参数配置
        model_name: 模型名
        dataset_name: 数据集名

    Returns:
        _type_: 模型名字
    """
    model_dir = f"/home/tmp/{model_name}/{dataset_name}/"
    os.makedirs(model_dir, exist_ok=True)
    model_name = '{}_{}_{}.bin'.format('test', train_config.batch_size, train_config.epochs)  # 定义模型名字
    save_path = os.path.join(model_dir, model_name)
    print("saved model path: ", save_path)
    return save_path


# crf ner识别结果转换
def trans_entity2tuple(scores, id2label:dict, text_pos:bool=False, mapping:list=None):
    '''
    把tensor转为(样本id, start, end, 实体类型)的tuple用于计算指标
    mapping: [[], [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]]

    text_pos 为False, 则start, end 为token的pos
    text_pos 为True, 则start, end 为text的pos
    '''
    batch_entity_ids = set()
    for i, one_samp in enumerate(scores):
        entity_ids = []
        for j, item in enumerate(one_samp):
            flag_tag = id2label[item.item()]
            if flag_tag.startswith('B-'):  # B
                entity_ids.append([i, j, j, flag_tag[2:]])
            elif len(entity_ids) == 0:
                continue
            elif (len(entity_ids[-1]) > 0) and flag_tag.startswith('I-') and (flag_tag[2:]==entity_ids[-1][-1]):  
                entity_ids[-1][-2] = j
            elif len(entity_ids[-1]) > 0:
                entity_ids.append([])

        for i in entity_ids:
            if i:
                batch_entity_ids.add(tuple(i))

    if text_pos:
        return token2text(batch_entity_ids, mapping)

    return batch_entity_ids

def token2text(batch_entity_ids, mapping):
    """token_pos转换为text_pos，方便后续处理
    """
    _list = []
    for entity in batch_entity_ids:
        sample_id, start, end, label = entity  # start, end 为token的pos
        t_start, t_end = mapping[start][0], mapping[end][-1]

        _list.append([sample_id, t_start, t_end, label])
    return _list

