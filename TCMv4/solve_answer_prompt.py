import json
import re

def process_text_to_jsonl(input_file, output_file):
    # 读取整个文本文件
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式分割数据，每组数据以"<｜｜>"开头
    # 使用re.DOTALL标志使.匹配包括换行符在内的所有字符
    data_groups = re.split(r'(?=<｜User｜>)', content)
    
    # 过滤掉空字符串
    data_groups = [group for group in data_groups if group.strip()]
    
    # 确保我们只处理前2000组数据（如果有多余的）
    data_groups = data_groups[:2000]
    
    # 准备写入JSONL文件
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for i, group in enumerate(data_groups, 1):
            # 替换换行符为\n
            
            processed_content = group.replace('<｜User｜>', '')
            prompt, response = processed_content.split("<｜Assistant｜><｜begin▁of▁sentence｜>")[0], processed_content.split("<｜Assistant｜><｜begin▁of▁sentence｜>")[-1]
            # pattern = r"The answer to this question is (\\{1,2})boxed\{(.*?)\}\. Based on the answer and the constraints of the thought chain length, you should deduce the most logical reasoning process. Note: During the thought process, you should pretend not to have seen the answer, but you must rationally infer the correct answer mentioned earlier based on the content of the thought chain."
            template_pattern = r"The answer to this question is .*?the thought chain\."
            prompt = re.sub(template_pattern, "", prompt, flags=re.DOTALL)

            # prompt = re.sub(pattern, "", prompt)
            with open("/mnt/lyc/wuxinrui/LLaMA-Factory/TCMv2/special_tokens.json", "r") as f:
                special_tokens = json.load(f)
                for key, value in special_tokens.items():
                    if value in response:
                        # print(f"Found {value} in response {response}")
                        response = response.replace(value, "")
            json_obj = {
                "prompt": prompt,
                "response": response
            }
            
            # 写入JSONL文件
            out_f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    
    print(f"成功处理并保存了{len(data_groups)}条数据到{output_file}")

# 使用示例
input_txt = "/mnt/lyc/wuxinrui/LLaMA-Factory/TCMv4/TCMv4_format_random2000_answer_prompt_generate.txt"  # 替换为您的输入文件路径
output_jsonl = "/mnt/lyc/wuxinrui/LLaMA-Factory/TCMv4/TCMv4_format_random2000_answer_prompt_generate.jsonl"  # 输出文件路径

process_text_to_jsonl(input_txt, output_jsonl)