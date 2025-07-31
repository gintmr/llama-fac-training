"""
pip install transformers>=4.40.0 torch sentencepiece accelerate
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 指定模型路径或 HuggingFace Hub 名
model_name_or_path = "/mnt/lyc/wuxinrui/LLaMA-Factory/TCMv4_8ratio_FULL_1_5B/TCMv4_8ratio_FULL_1_5B_6epoch/models"     # 本地路径 or HF Hub

# 2. 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    padding_side="left"
)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 3. 输入文本
text = "\n<remaining>1/8</remaining>\n\n<remaining>2/8</remaining>\n\n<remaining>3/8</remaining>\n<think></think>hello"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
input_ids = inputs["input_ids"][0]           # (seq_len, )

# 4. 取最后一层的 hidden states
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
hidden_states = outputs.hidden_states[-1][0]  # (seq_len, hidden_size)

# 5. 计算 pairwise 余弦相似度
norm = hidden_states / hidden_states.norm(dim=-1, keepdim=True)  # 归一化
cos_sim = norm @ norm.T                                         # (seq_len, seq_len)

# 6. 把 token 和相似度一起打印出来
tokens = tokenizer.convert_ids_to_tokens(input_ids)
print("Token 列表:", tokens)
print("余弦相似度矩阵 (形状: {}×{})".format(*cos_sim.shape))
print(cos_sim.cpu().float())