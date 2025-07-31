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

dataset_path =  "TCMv3/OT_long_short_formatted_cleaned.jsonl"

data_base_name = dataset_path.split("/")[-1].split(".")[0]



blow_path = f"TCMv3/{data_base_name}_below10000.jsonl"
blow_data = []
with open(dataset_path, "r") as f:
    for lien in tqdm(f):
        data = json.loads(lien)
        response = data["response"]
        # print(response)
        answer = response.split("</think>")[0]
        
        tokenized_answer = tokenizer(answer, return_tensors="pt")
        
        length = tokenized_answer["input_ids"].shape[1]
        
        if length < 10000:
            blow_data.append(data)

with open(blow_path, "w") as f:
    for data in blow_data:
        f.write(json.dumps(data) + "\n")
