import os 
import json
import re


def get_answer_prompt(file_path):
    answer_prompt_data = []
    clean_data_list = []
    with open("/mnt/lyc/wuxinrui/LLaMA-Factory/TCMv4/TCMv4_format.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            clean_data_list.append(data)
            
    
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            response = data["response"]
            prompt = data["prompt"]
            for clean_data in clean_data_list:
                if clean_data["prompt"] in prompt:
                    response = clean_data["response"]
            pattern = "</think>\n**Final Answer**"
            answer = response.split(pattern)[-1]
            print(answer)
            # answer = matches[0]
            
            answer_prompt = f"The answer to this question is {answer}. Based on the answer and the constraints of the thought chain length, you should deduce the most logical reasoning process. Note: During the thought process, you should pretend not to have seen the answer, but you must rationally infer the correct answer mentioned earlier based on the content of the thought chain."

            data['prompt'] = prompt+answer_prompt
            print(data)
            answer_prompt_data.append(data)

    with open(file_path.replace(".jsonl", "_answer_prompt.jsonl"), "w") as f:
        for data in answer_prompt_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    get_answer_prompt("/mnt/lyc/wuxinrui/LLaMA-Factory/TCMv4/TCMv4_format_below500_TCMv4.jsonl")
    