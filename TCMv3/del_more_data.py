import json
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import random

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/lyc/wuxinrui/LLaMA-Factory/FULL7B_SFT/outputs_full")
len_list = []

dataset_path = "/mnt/lyc/wuxinrui/LLaMA-Factory/TCMv3/RL_QA_format.jsonl"
data_base_name = dataset_path.split("/")[-1].split(".")[0]

# 读取并处理原始数据
data_list = []
with open(dataset_path, "r") as f:
    for line in tqdm(f):
        data = json.loads(line)
        response = data["response"]
        answer = response.split("</think>")[0]
        tokenized_answer = tokenizer(answer, return_tensors="pt")
        length = tokenized_answer["input_ids"].shape[1]
        
        # 保存原始数据和长度
        data_list.append({
            "data": data,
            "length": length
        })
        len_list.append(length)

# 筛选数据
filtered_data = []
count_750_1000 = 0
count_1000_1250 = 0
count_1250_1500 = 0

# 打乱数据以确保随机选择
random.shuffle(data_list)

for item in data_list:
    length = item["length"]
    data = item["data"]
    
    # 跳过0-250区间的数据
    if 0 <= length < 250:
        continue
    
    # 处理750-1000区间
    elif 750 <= length < 1000:
        if count_750_1000 < 887:
            filtered_data.append(data)
            count_750_1000 += 1
    
    # 处理1000-1250区间
    elif 1000 <= length < 1250:
        if count_1000_1250 < 2075:
            filtered_data.append(data)
            count_1000_1250 += 1
    
    # 处理1250-1500区间
    elif 1250 <= length < 1500:
        if count_1250_1500 < 2880:
            filtered_data.append(data)
            count_1250_1500 += 1
    
    # 其他区间保持不变
    else:
        filtered_data.append(data)

# 保存筛选后的数据
filtered_path = f"./{data_base_name}_filtered.jsonl"
with open(filtered_path, "w") as f:
    for data in filtered_data:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
print(f"筛选后的数据已保存到 {filtered_path}")

# 重新计算长度分布
filtered_len_list = []
for data in filtered_data:
    response = data["response"]
    answer = response.split("</think>")[0]
    tokenized_answer = tokenizer(answer, return_tensors="pt")
    length = tokenized_answer["input_ids"].shape[1]
    filtered_len_list.append(length)

# 保存长度列表
len_list_path = f"./{data_base_name}_filtered_len_list.npy"
np.save(len_list_path, np.array(filtered_len_list))
print(f"筛选后的长度列表已保存到 {len_list_path}")

# 计算并保存长度分布
len_array = np.array(filtered_len_list)
max_length = np.max(len_array)
interval = 250
length_counts = []

for i in range(0, max_length + interval, interval):
    lower_bound = i
    upper_bound = i + interval
    count = np.sum((len_array >= lower_bound) & (len_array < upper_bound))
    length_counts.append([f"{lower_bound}-{upper_bound}", count])

csv_path = f"./{data_base_name}_filtered_len_distribution.csv"
df = pd.DataFrame(length_counts, columns=["Interval", "Count"])
df.to_csv(csv_path, index=False)
print(f"筛选后的长度分布已保存到 {csv_path}")

# 输出统计结果
print("\n筛选后的长度分布统计结果：")
print(df)
print("\n各区间的实际保留数量：")
print(f"750-1000区间: {count_750_1000}条")
print(f"1000-1250区间: {count_1000_1250}条")
print(f"1250-1500区间: {count_1250_1500}条")