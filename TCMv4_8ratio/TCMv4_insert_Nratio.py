## 在数据中穿插remaining token(输入未处理的数据,自动从答案的开头往后连续添加)
## 同时，insert操作向上以50为跨度取整

import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# 加载模型

tokenizer = AutoTokenizer.from_pretrained("/mnt/lyc/wuxinrui/DS_Huggingface/DS_QW_7B", trust_remote_code=True)

data_path = "TCMv4_8ratio/TCMv4_format.jsonl"
# data_path = "/mnt/lyc/wuxinrui/LLaMA-Factory/TCMv4/TCMv4_format_random2000_answer_prompt_generate.jsonl"



ratios_tokens = [
    f"\n<remaining>{i+1}/8</remaining>\n" for i in range(7) #g 这里使用range(7)是因为7个token将thinkig分成8份
]
# print(bins_tokens)

def split_array_by_ratios(input_array, array_length = None):
    
    array_length = len(input_array) if array_length is None else array_length
    
    
    result = []
    
    delta = (array_length // 8) + 1
    for i in range(8):
        #g 分成8份
        start_index = i * delta
        if i == 7:
            end_index = array_length
        else:
            end_index = (i + 1) * delta
        result.append(input_array[start_index:end_index])

    assert len(result) == 8
    
    
    return result, array_length


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

def count_down_RL(sub_cot, TCMv4_length):
    inserted_cot = f""
    for i in (range(len(sub_cot))):
        if 7 - i - 1 >= 0:
            inserted_cot = inserted_cot + tokenizer.decode(sub_cot[i]) + ratios_tokens[7 - i - 1]
        else:
            inserted_cot = inserted_cot + tokenizer.decode(sub_cot[i])
    return inserted_cot



def insert_token_RL(data_path):
    inserted_data_path = data_path.replace(".jsonl", "_below10000_TCMv4_8ratio_below2k.jsonl")
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
            if length_of_tokenized_cot > 2050:
                continue
            else:
                #g v1
                # N_50 = length_of_tokenized_cot // 50 + 1
                # array_length = N_50 * 50

                #g v2
                bins = [100, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]
                array_length = min(bins[i] for i in range(len(bins)) if bins[i] > length_of_tokenized_cot)

                sub_cot, array_length = split_array_by_ratios(tokenized_cot, array_length = array_length)
                inserted_cot = count_down_RL(sub_cot, array_length)  
                response = inserted_cot + answer



                # add_prompt = f'\n(Respond in {TCMv4_length} tokens or fewer. Complete the process between <think> and </think> within the token budget. Display the countdown exponentially as <remaining>xxx</remaining>, where xxx = 50 * 2^n, n >= 0. Think more concisely as countdown decreases.)\n'
                # add_response = f"\n(I will complete the process within {TCMv4_length} tokens and show the countdown as <remaining>xxx</remaining>, following the exponential rule.I will think more concisely as countdown decreases.)\n"
                
                # add_prompt = f"\n(Complete thinking within {TCMv4_length} tokens or fewer.)\n"
                add_prompt = f"\n(Complete thinking within {array_length} tokens or fewer, 7 special tokens ( \n<remaining>7/8</remaining>\n , \n<remaining>6/8</remaining>\n , \n<remaining>5/8</remaining>\n , \n<remaining>4/8</remaining>\n , \n<remaining>3/8</remaining>\n , \n<remaining>2/8</remaining>\n , \n<remaining>1/8</remaining>\n ) will split the thinking process into 8 parts.)"
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