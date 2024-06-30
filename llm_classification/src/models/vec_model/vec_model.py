import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from transformers import BertTokenizer

from src.models.vec_model.simcse_model import SimcseModel

import onnxruntime as ort

class VectorizeModel:
    def __init__(self, ptm_model_path, device = "cpu") -> None:
        self.tokenizer = BertTokenizer.from_pretrained(ptm_model_path)
        self.model = SimcseModel(pretrained_bert_path=ptm_model_path, pooling="cls")
        # print(self.model)
        self.model.eval()
        
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        # self.DEVICE = device
        logger.info(self.DEVICE)
        self.model.to(self.DEVICE)
        
        self.pdist = nn.PairwiseDistance(2)
    
    def predict_vec(self,query):
        q_id = self.tokenizer(query, max_length = 200, truncation=True, padding="max_length", return_tensors='pt')
        with torch.no_grad():
            q_id_input_ids = q_id["input_ids"].squeeze(1).to(self.DEVICE)
            q_id_attention_mask = q_id["attention_mask"].squeeze(1).to(self.DEVICE)
            q_id_token_type_ids = q_id["token_type_ids"].squeeze(1).to(self.DEVICE)
            q_id_pred = self.model(q_id_input_ids, q_id_attention_mask, q_id_token_type_ids)

        return q_id_pred

    def predict_vec_request(self, query):
        q_id_pred = self.predict_vec(query)
        return q_id_pred.cpu().numpy().tolist()
    
    def predict_sim(self, q1, q2):
        q1_v = self.predict_vec(q1)
        q2_v = self.predict_vec(q2)
        sim = F.cosine_similarity(q1_v[0], q2_v[0], dim=-1)
        return sim.cpu().numpy().tolist()

class VectorizeModel_v2(VectorizeModel):
    def __init__(self, ptm_model_path, onnx_path, providers=['CUDAExecutionProvider']) -> None:
        # ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        self.tokenizer = BertTokenizer.from_pretrained(ptm_model_path)
        self.model = ort.InferenceSession(onnx_path, providers=providers)
        
        self.pdist = nn.PairwiseDistance(2)
    
    def _to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    def predict_vec(self,query):
        q_id = self.tokenizer(query, max_length = 200, truncation=True, padding="max_length", return_tensors='pt')
        input_feed = {
            self.model.get_inputs()[0].name: self._to_numpy(q_id["input_ids"]),
            self.model.get_inputs()[1].name: self._to_numpy(q_id["attention_mask"]),
            self.model.get_inputs()[2].name: self._to_numpy(q_id["token_type_ids"]),
        }
        return torch.tensor(self.model.run(None, input_feed=input_feed)[0])
    
    def predict_sim(self, q1, q2):
        q1_v = self.predict_vec(q1)
        q2_v = self.predict_vec(q2)
        sim = F.cosine_similarity(q1_v[0], q2_v[0], dim=-1)
        return sim.numpy().tolist()

if __name__ == "__main__":
    import time,random
    from tqdm import tqdm
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # device = ""
    # vec_model = VectorizeModel('C:/work/tool/huggingface/models/simcse-chinese-roberta-wwm-ext', device=device)
    vec_model = VectorizeModel_v2('C:/work/tool/huggingface/models/simcse-chinese-roberta-wwm-ext',
                                 "./data/model_simcse_roberta_output_20240211.onnx",providers=['CUDAExecutionProvider'])
    # vec_model = VectorizeModel_v2('C:/work/tool/huggingface/models/simcse-chinese-roberta-wwm-ext',
    #                              "./data/model_simcse_roberta_output_20240211.onnx",providers=['TensorrtExecutionProvider'])
    # 单测
    # q = ["你好啊"]
    # print(vec_model.predict_vec(q))
    # print(vec_model.predict_sim("你好呀","你好啊"))
    tmp_queries = ["你好啊", "今天天气怎么样", "我要暴富"]
    # 开始批跑
    batch_sizes = [1,2,4,8,16]
    for b in batch_sizes:
        for i in tqdm(range(100),desc="warmup"):
            tmp_q = []
            for i in range(b):
                tmp_q.append(random.choice(tmp_queries))
            vec_model.predict_vec(tmp_q)
        for i in tqdm(range(1000),desc="batch_size={}".format(b)):
            tmp_q = []
            for i in range(b):
                tmp_q.append(random.choice(tmp_queries))
            vec_model.predict_vec(tmp_q)
