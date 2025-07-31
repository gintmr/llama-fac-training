import json
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import pandas as pd
tokenizer = AutoTokenizer.from_pretrained("/mnt/lyc/wuxinrui/LLaMA-Factory/FULL7B_SFT/outputs_full")
len_list = []

dataset_path =  "/mnt/lyc/wuxinrui/LLaMA-Factory/RL_QA_format_filtered.jsonl"

data_base_name = dataset_path.split("/")[-1].split(".")[0]

with open(dataset_path, "r") as f:
    for lien in tqdm(f):
        data = json.loads(lien)
        response = data["response"]
        # print(response)
        answer = response.split("</think>")[0]
        
        tokenized_answer = tokenizer(answer, return_tensors="pt")
        
        length = tokenized_answer["input_ids"].shape[1]
        
        len_list.append(length)

# print(sum(len_list) / len(len_list))
# # print(len_list)
# min_len = min(len_list)
# max_len = max(len_list)
# print(min_len, max_len)

# bins = range(min_len, max_len + 1)

# plt.figure(figsize=(10, 6))
# sns.histplot(len_list, bins=bins, kde=True)
# plt.title(f"Distribution of lens of answer in {dataset_path}")
# plt.xlabel("Token Count")
# plt.ylabel("Frequency")
# plt.grid(True)

# plt.savefig("len_of_answer.png")
# plt.show()



# 将长度列表保存为文件
len_list_path = f"./{data_base_name}_len_list.npy"
# len_list_path = "./formatted_clean_OT_long_len_list.npy"
np.save(len_list_path, np.array(len_list))
print(f"长度列表已保存到 {len_list_path}")

# 将长度列表转换为 NumPy 数组方便操作
len_array = np.array(len_list)

# 定义间隔
interval = 250

# 计算最大长度
max_length = np.max(len_array)

# 初始化统计结果列表
length_counts = []

# 统计每个间隔内的数据条数
for i in range(0, max_length + interval, interval):
    lower_bound = i
    upper_bound = i + interval
    count = np.sum((len_array >= lower_bound) & (len_array < upper_bound))
    length_counts.append([f"{lower_bound}-{upper_bound}", count])

# 将统计结果保存为 CSV 文件
csv_path = f"./{data_base_name}_len_distribution.csv"
# csv_path = "/data/wuxinrui/LLaMA-Factory/data/formatted_clean_OT_long_length_distribution.csv"
df = pd.DataFrame(length_counts, columns=["Interval", "Count"])
df.to_csv(csv_path, index=False)
print(f"长度分布已保存到 {csv_path}")

# 输出统计结果
print("长度分布统计结果：")
print(df)