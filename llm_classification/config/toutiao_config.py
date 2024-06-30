VEC_MODEL_PATH = "C:/work/tool/huggingface/models/simcse-chinese-roberta-wwm-ext"
VEC_INDEX_DATA = "vec_index_toutiao_20240629"

LLM_PATH = "C:/work/tool/qwen2-1.5b-instruct"
LLM_CONFIG = {"max_length": 2048,
              "do_sample": False,
              "top_k": 1,
              "temperature": 0.8}

CLASS_DEF_PATH = "data/toutiao_cat_data/class_def.tsv"

PROMPT_TEMPLATE = """你是一个优秀的句子分类师，能把给定的用户query划分到正确的类目中。现在请你根据给定信息和要求，为给定用户query，从备选类目中选择最合适的类目。

下面是“参考案例”即被标注的正确结果，可供参考：
<examples>

备选类目：
<options>

类目概念：
<options_detail>

用户query：
<query>

请注意：
1. 用户query所选类目，仅能在【备选类目】中进行选择，用户query仅属于一个类目。
2. “参考案例”中的内容可供推理分析，可以仿照案例来分析用户query的所选类目。
3. 请仔细比对【备选类目】的概念和用户query的差异。
4. 如果用户quer也不属于【备选类目】中给定的类目，或者比较模糊，请选择“拒识”。
5. 请在“所选类目：”后回复结果，不需要说明理由。

所选类目："""