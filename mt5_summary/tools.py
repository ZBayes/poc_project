# coding=utf-8
# Filename:    tools.py
# Author:      ZENGGUANRONG
# Date:        2024-10-06
# description: 关键工具
# reference:   https://github.com/jsksxs360/How-to-use-Transformers
import random
import os
import numpy as np
import torch

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True