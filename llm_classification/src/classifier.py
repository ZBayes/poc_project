# coding=utf-8
# Filename:    classifier.py
# Author:      ZENGGUANRONG
# Date:        2024-06-25
# description: 分类器主函数

import copy
import torch
from loguru import logger

from config.toutiao_config import (VEC_INDEX_DATA, VEC_MODEL_PATH,
                                    LLM_CONFIG, LLM_PATH, PROMPT_TEMPLATE,CLASS_DEF_PATH)
from src.searcher.searcher import Searcher
from src.models.llm.llm_model import QWen2Model
from src.utils.data_processing import load_class_def

class VecLlmClassifier:
    def __init__(self) -> None:
        self.searcher = Searcher(VEC_MODEL_PATH, VEC_INDEX_DATA)
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.llm = QWen2Model(LLM_PATH, LLM_CONFIG, self.device)
        self.PROMPT_TEMPLATE = PROMPT_TEMPLATE
        self.class_def = load_class_def(CLASS_DEF_PATH)

    def predict(self, query):
        # 1. query预处理
        logger.info("request: {}".format(query))
        # 2. query向量召回
        recall_result = self.searcher.search(query, nums=5)
        # logger.debug(recall_result)

        # 3. 请求大模型
        # 3.1 PROMPT拼接
        request_prompt= copy.deepcopy(self.PROMPT_TEMPLATE)
        # 3.1.1 子模块拼接
        examples = []
        options = []
        options_detail = []
        for item in recall_result:
            tmp_examples = "——".join([item[1][0], item[1][1][5]])
            if tmp_examples not in examples:
                examples.append(tmp_examples)
            opt_detail_str = "：".join(["【" + item[1][1][5] + "】",self.class_def[item[1][1][5]]])
            opt = item[1][1][5]
            if opt not in options:
                options.append(opt)
                options_detail.append(opt_detail_str)
        # options.append("拒识：含义不明或用户query所属类目不在列举内时，分为此类")
        examples_str = "\n".join(examples)
        options_str = "，".join(options)
        options_detail_str = "\n".join(options_detail)

        # 3.1.2 整体组装
        request_prompt = request_prompt.replace("<examples>", examples_str)
        request_prompt = request_prompt.replace("<options>", options_str)
        request_prompt = request_prompt.replace("<options_detail>", options_detail_str)
        request_prompt = request_prompt.replace("<query>", query)
        # logger.info(request_prompt)

        # 3.2 请求大模型
        llm_response = self.llm.predict(request_prompt)
        # logger.info("llm response: {}".format(llm_response))

        # 3.3 大模型结果解析
        result = "拒识"
        for option in options:
            if option in llm_response:
                result = option
                break
        # logger.info("parse result: {}".format(result))

        # 4. 返回结果
        logger.info("response: {}".format(result))
        return result

if __name__ == "__main__":
    import sys
    vlc = VecLlmClassifier()
    if len(sys.argv) > 1:
        logger.info(vlc.predict("".join(sys.argv[1:])))

    # # 性能测试
    # from tqdm import tqdm
    # for i in tqdm(range(20), desc="warm up"):
    #     vlc.predict("感冒发烧怎么治疗")
    # for i in tqdm(range(20), desc="running speed"):
    #     vlc.predict("王阳明到底顿悟了什么？")