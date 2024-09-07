# coding=utf-8
# Filename:    data_processing.py
# Author:      ZENGGUANRONG
# Date:        2024-06-25
# description: 数据处理函数

def load_toutiao_data(path):
    source_data = []
    with open(path, encoding="utf8") as f:
        for line in f:
            ll = line.strip().split("_!_") # 新闻ID，分类code，分类名称，新闻字符串（仅含标题），新闻关键词
            source_data.append([ll[3], ll])
    return source_data

def load_class_def(path):
    source_data = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            ll = line.strip().split("\t")
            source_data[ll[0]] = ll[1]
    return source_data