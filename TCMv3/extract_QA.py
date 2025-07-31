import os
import json

def process_jsonl_files(root_dir):
    """
    遍历文件夹中的所有jsonl文件（包括子文件夹），读取并处理数据
    
    参数:
        root_dir: 要遍历的根目录路径
        
    返回:
        处理后的字典列表
    """
    result_list = []
    
    import tqdm
    for root, dirs, files in os.walk(root_dir):
        for file in tqdm.tqdm(files):
            if file.endswith('.jsonl'):
                file_path = os.path.join(root, file)
                
                # 读取jsonl文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            data.pop("score")
                            data.pop("gt")
                            # 检查是否是问答对（包含prompt和response键）
                            if 'prompt' in data and 'response' in data:
                                prompt = data['prompt']
                                current_response = data['response']
                                current_len = len(current_response)
                                
                                # 查找是否已有相同prompt的记录
                                existing_entries = [item for item in result_list if item['prompt'] == prompt]
                                
                                if not existing_entries:
                                    # 如果没有相同prompt的记录，直接添加
                                    result_list.append(data)
                                else:
                                    # 检查所有相同prompt的response长度差
                                    should_add = True
                                    for entry in existing_entries:
                                        existing_len = len(entry['response'])
                                        if abs(current_len - existing_len) < 40:
                                            should_add = False
                                            break
                                    
                                    if should_add:
                                        result_list.append(data)
                                        
                        except json.JSONDecodeError as e:
                            print(f"解析错误在文件 {file_path}, 行: {line}. 错误: {e}")
    
    return result_list

# 使用示例
if __name__ == "__main__":
    directory = "/mnt/lyc/wuxinrui/Qwen2.5-Math/evaluation/MODEL-FULL7B_SFT-TIP-TCMv2-STAGE-add-DATA-RL_QA"  # 替换为你的文件夹路径
    processed_data = process_jsonl_files(directory)
    
    # 打印结果或保存到文件
    print(f"处理后的数据数量: {len(processed_data)}")
    with open('TCMv3/RL_QA_format.jsonl', 'w', encoding='utf-8') as outfile:
        for entry in processed_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')