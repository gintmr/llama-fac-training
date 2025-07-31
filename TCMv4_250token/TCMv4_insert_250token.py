## 在数据中穿插remaining token(输入未处理的数据,自动从答案的开头往后连续添加)
## 同时，insert操作向上以50为跨度取整

import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# 加载模型

tokenizer = AutoTokenizer.from_pretrained("/mnt/lyc/wuxinrui/DS_Huggingface/DS_QW_7B", trust_remote_code=True)

data_path = "TCMv4_250token/TCMv4_format.jsonl"
# data_path = "/mnt/lyc/wuxinrui/LLaMA-Factory/TCMv4/TCMv4_format_random2000_answer_prompt_generate.jsonl"


bins = [i*250 + 250 for i in range(40)]
# print(bins)
with open('TCMv4_250token/special_tokens.json') as f:
    special_tokens = json.load(f)
    
bins_tokens = [
    special_tokens[f"{i}"] for i in range(40)
]
# print(bins_tokens)

def split_array_by_bins(input_array, bins):
    # 定义区间值
    # intervals = [2000]
    intervals = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]
    
    # 计算新输入数组的长度
    array_length = len(input_array)
    ori_length = array_length
    
    # 找到合适的区间值
    for interval in intervals:
        if array_length <= interval:
            array_length = interval
            break
    else:
        # 如果输入数组长度大于所有区间值，选择最后一个区间值
        array_length = intervals[-1]
    
    # 确保 array_length 在 bins 中
    assert array_length in bins, f"array_length {array_length} not found in bins {bins}"
    index = bins.index(array_length)
    
    result = []
    
    # 从分档数组的最后一个元素开始向前遍历
    i = index 
    
    while i >= 0:
        start_index = (array_length - bins[i])
        if i == 0:
            end_index = ori_length
        else:
            end_index = (array_length - bins[i-1])
        
        result.append(input_array[start_index:end_index])
        i -= 1
    
    return result, index, array_length


def split_string(input_string):
    # 要匹配的字符串
    match_string = "\n</think>\n"
    
    # 找到匹配字符串的起始位置
    start_index = input_string.find(match_string)
    
    if start_index == -1:
        print("匹配的字符串未找到")
        return None, None
    
    # 获取匹配字符串之前的字符串
    before_string = input_string[:start_index]
    
    # 获取匹配字符串之后的所有字符串
    after_string = input_string[start_index:]
    
    return before_string, after_string

def count_down_RL(sub_cot, indice, TCMv4_length):
    inserted_cot = f""
    for i in (range(len(sub_cot))):
        if indice - i - 1 >= 0:
            inserted_cot = inserted_cot + tokenizer.decode(sub_cot[i]) + bins_tokens[indice - i - 1]
        else:
            inserted_cot = inserted_cot + tokenizer.decode(sub_cot[i])
    return inserted_cot



def insert_token_RL(data_path):
    inserted_data_path = data_path.replace(".jsonl", "_below10000_TCMv4_250token.jsonl")
    if os.path.exists(inserted_data_path):
        os.remove(inserted_data_path)
    with open(data_path, "r") as f:
        datas = [json.loads(line) for line in f]
        inserted_datas  ={}
        for data in tqdm(datas, desc="inserting token with RL format"):
            prompt = data["prompt"]
            response = data["response"]
            
            cot, answer = split_string(response)
            if cot is None:
                continue
            tokenized_cot = tokenizer(cot, return_tensors="pt").input_ids[0]
            chunk_size = 100
            length_of_tokenized_cot = len(tokenized_cot)
            if length_of_tokenized_cot > 10050:
                continue
            else:
                sub_cot, indice, TCMv4_length = split_array_by_bins(tokenized_cot, bins)
                inserted_cot = count_down_RL(sub_cot, indice, TCMv4_length)  
                response = inserted_cot + answer

                # add_prompt = f'\n(Respond in {TCMv4_length} tokens or fewer. Complete the process between <think> and </think> within the token budget. Display the countdown exponentially as <remaining>xxx</remaining>, where xxx = 50 * 2^n, n >= 0. Think more concisely as countdown decreases.)\n'
                # add_response = f"\n(I will complete the process within {TCMv4_length} tokens and show the countdown as <remaining>xxx</remaining>, following the exponential rule.I will think more concisely as countdown decreases.)\n"
                
                # add_prompt = f"\n(Complete thinking within {TCMv4_length} tokens or fewer.)\n"
                add_prompt = f"\n(Complete thinking within \n<remaining>{TCMv4_length}</remaining>\n tokens or fewer.)"
                add_response = ""
                
                inserted_data = {
                    # "prompt": prompt + f"\n<remaining>{TCMv4_length}</remaining>\n",
                    "prompt": prompt + add_prompt,
                    "response": add_response + response
                }
                # print(inserted_data)
                with open(inserted_data_path, "a") as f:
                    f.write(json.dumps(inserted_data) + "\n")



insert_token_RL(data_path=data_path)