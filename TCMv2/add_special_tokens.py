from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import json
# model = AutoModelForCausalLM.from_pretrained("/data/sunyi/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/6602cadec947dbb53e64f3d8d6425320b2197247")
# tokenizer = AutoTokenizer.from_pretrained("/data/sunyi/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/6602cadec947dbb53e64f3d8d6425320b2197247")

model = AutoModelForCausalLM.from_pretrained("/mnt/lyc/wuxinrui/DS_Huggingface/DS_QW_1_5B")
tokenizer = AutoTokenizer.from_pretrained("/mnt/lyc/wuxinrui/DS_Huggingface/DS_QW_1_5B")
print(model.get_input_embeddings())
print(model.lm_head)
print(len(tokenizer))


with open('TCMv2/special_tokens.json') as f:
    special_tokens = json.load(f)
    
bins_tokens = [
    special_tokens[f"{i}"] for i in range(200)
]

tokenizer.add_special_tokens({'additional_special_tokens': bins_tokens})
model.resize_token_embeddings(len(tokenizer))

print('Vocab size after adding special tokens:', len(tokenizer))

# # # 保存新的tokenizer和model
NEW_MODEL = 'TCMv2/1_5B_TCM2_models'
tokenizer.save_pretrained(NEW_MODEL)
model.save_pretrained(NEW_MODEL)

model = AutoModelForCausalLM.from_pretrained("TCMv2/1_5B_TCM2_models")
tokenizer = AutoTokenizer.from_pretrained("TCMv2/1_5B_TCM2_models")
print(model.get_input_embeddings())
print(model.lm_head)
print(len(tokenizer))


# model = AutoModelForCausalLM.from_pretrained(NEW_MODEL)
# tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL)

# new_token_ids = tokenizer.convert_tokens_to_ids(bins_tokens)
# embeddings = model.get_input_embeddings().weight
# print(embeddings.requires_grad)  # 应为 True（默认可训练）new_token_ids = 将"[TOKEN1]"和"[TOKEN2]"转换为 token 的 ID
