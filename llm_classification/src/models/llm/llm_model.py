# coding=utf-8
# Filename:    llm_model.py
# Author:      ZENGGUANRONG
# Date:        2023-12-17
# description: 大模型调用模块，这里默认用的chatglm2

# from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, List
from loguru import logger

class QWen2Model:
    def __init__(self, model_path, config = {}, device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = self.model.eval()
        self.device = device

        self.generate_config = self._read_config_(config)
        logger.info("load LLM Model done")
    
    def _read_config_(self, config):
        tmp_config = {}
        # tmp_config["max_length"] = config.get("max_length", 2048)
        tmp_config["num_beams"] = config.get("num_beams", 1)
        tmp_config["do_sample"] = config.get("do_sample", False)
        tmp_config["top_k"] = config.get("top_k", 1)
        tmp_config["temperature"] = config.get("temperature", 0.8)
        return tmp_config

    def predict(self, query):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # Directly use generate() and tokenizer.decode() to get the output.
        # Use `max_new_tokens` to control the maximum output length.
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            **self.generate_config
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

if __name__ == "__main__":
    from config.toutiao_config import LLM_CONFIG,LLM_PATH
    print(LLM_CONFIG)
    llm_model = QWen2Model(LLM_PATH, config = LLM_CONFIG, device="cuda")
    print(llm_model.predict("如何做番茄炒蛋"))
