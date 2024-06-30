import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

class SimcseModel(nn.Module):
    # https://blog.csdn.net/qq_44193969/article/details/126981581
    def __init__(self, pretrained_bert_path, pooling="cls") -> None:
        super(SimcseModel, self).__init__()

        self.pretrained_bert_path = pretrained_bert_path
        self.config = BertConfig.from_pretrained(self.pretrained_bert_path)
        
        self.model = BertModel.from_pretrained(self.pretrained_bert_path, config=self.config)
        self.model.eval()
        
        # self.model = None
        self.pooling = pooling
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        return out.last_hidden_state[:, 0]