# coding=utf-8
# Filename:    build_vec_index.py
# Author:      ZENGGUANRONG
# Date:        2023-12-12
# description: 构造向量索引脚本

import json,torch,copy,random
from tqdm import tqdm
from loguru import logger
from sklearn.model_selection import train_test_split

from src.utils.data_processing import load_toutiao_data
from src.models.vec_model.vec_model import VectorizeModel
from src.searcher.vec_searcher.vec_searcher import VecSearcher 

if __name__ == "__main__":
    # 0. 必要配置
    MODE = "DEBUG"
    # MODE = "PRO"
    MODE = "FEW"

    VERSION = "20240702"
    VEC_MODEL_PATH = "C:/work/tool/huggingface/models/simcse-chinese-roberta-wwm-ext"
    SOURCE_INDEX_DATA_PATH = "./data/toutiao_cat_data/toutiao_cat_data.txt" # 数据来源：https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset
    VEC_INDEX_DATA = "vec_index_toutiao_{}_{}".format(VERSION,MODE)
    TESE_DATA_PATH = "./data/toutiao_cat_data/test_set_{}_{}.txt".format(VERSION,MODE)
    RANDOM_SEED = 100

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    TEST_SIZE = 0.1
    # 类目体系
    CLASS_INFO = [
        ["100", '民生-故事', 'news_story'],
        ["101", '文化-文化', 'news_culture'],
        ["102", '娱乐-娱乐', 'news_entertainment'],
        ["103", '体育-体育', 'news_sports'],
        ["104", '财经-财经', 'news_finance'],
        # ["105", '时政 新时代', 'nineteenth'],
        ["106", '房产-房产', 'news_house'],
        ["107", '汽车-汽车', 'news_car'],
        ["108", '教育-教育', 'news_edu' ],
        ["109", '科技-科技', 'news_tech'],
        ["110", '军事-军事', 'news_military'],
        # ["111" 宗教 无，凤凰佛教等来源],
        ["112", '旅游-旅游', 'news_travel'],
        ["113", '国际-国际', 'news_world'],
        ["114", '证券-股票', 'stock'],
        ["115", '农业-三农', 'news_agriculture'],
        ["116", '电竞-游戏', 'news_game']
    ]
    ID2CN_MAPPING = {}
    for idx in range(len(CLASS_INFO)):
        ID2CN_MAPPING[CLASS_INFO[idx][0]] = CLASS_INFO[idx][1]

    # 1. 加载数据、模型
    # 1.1 加载模型
    vec_model = VectorizeModel(VEC_MODEL_PATH, DEVICE)
    index_dim = len(vec_model.predict_vec("你好啊")[0])
    # 1.2 加载数据
    toutiao_index_data = load_toutiao_data(SOURCE_INDEX_DATA_PATH)
    source_index_data = copy.deepcopy(toutiao_index_data)
    logger.info("load data done: {}".format(len(source_index_data)))
    if MODE == "DEBUG":
        random.shuffle(source_index_data)
        source_index_data = source_index_data[:10000]
    elif MODE == "FEW":
        new_source_data = []
        class_dict_cal = {}
        test_list = []
        tmp_idx = 0

        for key in ID2CN_MAPPING:
            class_dict_cal[key] = 0
        for idx in range(len(source_index_data)):
            if class_dict_cal[source_index_data[idx][1][1]] < 10:
                class_dict_cal[source_index_data[idx][1][1]] += 1
                new_source_data.append(source_index_data[idx])
            if sum([class_dict_cal[i] for i in class_dict_cal]) >= len(class_dict_cal) * 10:
                break
        source_index_data = new_source_data

    for item in source_index_data:
        item[1].append(ID2CN_MAPPING[item[1][1]])
    # 1.3 训练集测试集划分
    if MODE != "FEW":
        train_list, test_list = train_test_split(source_index_data, test_size=TEST_SIZE, random_state=66)
    else:
        train_list = source_index_data
        test_list = toutiao_index_data[idx:idx + 1000]
        for item in test_list:
            item[1].append(ID2CN_MAPPING[item[1][1]])

    # 2. 创建索引并灌入数据
    # 2.1 构造索引
    vec_searcher = VecSearcher()
    vec_searcher.build(index_dim, VEC_INDEX_DATA)

    # 2.2 推理向量
    vectorize_result = []
    for q in tqdm(train_list, desc="VEC MODEL RUNNING"):
        vec = vec_model.predict_vec(q[0]).cpu().numpy()
        tmp_result = copy.deepcopy(q)
        tmp_result.append(vec)
        vectorize_result.append(copy.deepcopy(tmp_result))

    # 2.3 开始存入
    for idx in tqdm(range(len(vectorize_result)), desc="INSERT INTO INDEX"):
        vec_searcher.insert(vectorize_result[idx][2], vectorize_result[idx][:2])

    # 3. 保存
    # 3.1 索引保存
    vec_searcher.save()
    # 3.2 测试集保存
    with open(TESE_DATA_PATH, "w", encoding="utf8") as f:
        for item in test_list:
            f.write("_!_".join(item[1]) + "\n")
