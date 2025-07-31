from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import json
# model = AutoModelForCausalLM.from_pretrained("/data/sunyi/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/6602cadec947dbb53e64f3d8d6425320b2197247")
# tokenizer = AutoTokenizer.from_pretrained("/data/sunyi/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/6602cadec947dbb53e64f3d8d6425320b2197247")




def gen_special_tokens_json():
    special_tokens_list = {}
    for i in range(7):
        special_tokens_list[f"{i}"] = f"\n<remaining>{i+1}/8</remaining>\n"
    print(special_tokens_list)
    
    with open('TCMv4_8ratio/special_tokens.json', 'w') as f:
        json.dump(special_tokens_list, f)
    print('special_tokens.json has been generated.')

if __name__ == "__main__":
    
    model = AutoModelForCausalLM.from_pretrained("/mnt/lyc/wuxinrui/DS_Huggingface/DS_QW_7B/")
    tokenizer = AutoTokenizer.from_pretrained("/mnt/lyc/wuxinrui/DS_Huggingface/DS_QW_7B/")
    print(model.get_input_embeddings())
    print(model.lm_head)
    print(len(tokenizer))

    gen_special_tokens_json()
    with open('TCMv4_8ratio/special_tokens.json') as f:
        special_tokens = json.load(f)
        
    bins_tokens = [
        special_tokens[f"{i}"] for i in range(7)
    ]

    tokenizer.add_special_tokens({'additional_special_tokens': bins_tokens})
    model.resize_token_embeddings(len(tokenizer))

    print('Vocab size after adding special tokens:', len(tokenizer))

    # # # 保存新的tokenizer和model
    NEW_MODEL = 'TCMv4_8ratio/7B_TCMv4_8ratio_models'
    tokenizer.save_pretrained(NEW_MODEL)
    model.save_pretrained(NEW_MODEL)

    model = AutoModelForCausalLM.from_pretrained("TCMv4_8ratio/7B_TCMv4_8ratio_models")
    tokenizer = AutoTokenizer.from_pretrained("TCMv4_8ratio/7B_TCMv4_8ratio_models")
    print(model.get_input_embeddings())
    print(model.lm_head)
    print(len(tokenizer))


# model = AutoModelForCausalLM.from_pretrained(NEW_MODEL)
# tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL)

# new_token_ids = tokenizer.convert_tokens_to_ids(bins_tokens)
# embeddings = model.get_input_embeddings().weight
# print(embeddings.requires_grad)  # 应为 True（默认可训练）new_token_ids = 将"[TOKEN1]"和"[TOKEN2]"转换为 token 的 ID
