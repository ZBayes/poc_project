
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, List
from tqdm import tqdm
import random

model_path = "C:\\work\\tool\\huggingface\\models\\Qwen-1_8b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().quantize(8).cuda()

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True
).eval().cuda()
response, history = model.chat(tokenizer, "你好", history=None)
print(response)
print(history)
random_query = ["你好","你是谁","你是谁创造出来的"]
for i in tqdm(range(100)):
    model.chat(tokenizer, random.choice(random_query), history=None)