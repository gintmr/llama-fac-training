import json
import os
input_file = '/mnt/lyc/wuxinrui/LLaMA-Factory/TCMv5/TCMv5_format_TCMv5.jsonl'

output_file = input_file.replace('.jsonl', '_delete_begin.jsonl')
# output_file = input_file.replace('.jsonl', '_cleaned.jsonl')

pattern_to_remove = '<\uff5cbegin\u2581of\u2581sentence\uff5c>'

if os.path.exists(output_file):
    os.remove(output_file)
    
# if not os.path.exists(output_file):
#     os.mk(output_file)

# 打开输入文件和输出文件
with open(input_file, 'r') as input_file, \
     open(output_file, 'w') as output_file:
    # 逐行读取和处理文件
    # print(json.loads(input_file[0]))
    length = 0
    for line in input_file:
        # 将每行的JSON字符串转换为字典
        data = json.loads(line)
        # print(data)
        
    
        data['response'] = data['response'].replace(pattern_to_remove, '')
        # data['response'] = data['response'].replace("<|begin_of_thought|>", '<think>')
        # data['response'] = data['response'].replace("<|end_of_thought|>", "</think>")
        # data['response'] = data['response'].replace("<|begin_of_solution|>", "")
        # data['response'] = data['response'].replace("<|end_of_solution|>", "")
        
        # 将修改后的字典转换回JSON字符串，并写入输出文件
        output_file.write(json.dumps(data) + '\n')
        length += 1
    print("length of output_file:" + str(length))