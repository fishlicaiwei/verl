import json
import os

def process_json_data(input_path, output_path):
    processed_data = []
    processed_count = 0
    error_count = 0

    print(f"开始处理文件: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        # 尝试兼容两种格式：1. 每一行是一个json (JSONL)  2. 整个文件是一个json list
        lines = f.readlines()
        
        # 判断是 JSONL 还是单个大 JSON
        first_line = lines[0].strip()
        if first_line.startswith('['):
            # 处理标准 JSON List
            full_content = "".join(lines)
            raw_items = json.loads(full_content)
        else:
            # 处理 JSONL 格式
            raw_items = []
            for i, line in enumerate(lines):
                if not line.strip(): continue
                try:
                    raw_items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"第 {i+1} 行解析失败，跳过。错误: {e}")
                    error_count += 1

    for item in raw_items:
        response = item.get("response", "")
        
        # 1. 逻辑分界点解析
        if "</think>" in response:
            parts = response.split("</think>")
            raw_think = parts[0]
            raw_answer = parts[1] if len(parts) > 1 else ""
        elif "<functioncall>" in response:
            parts = response.split("<functioncall>")
            raw_think = parts[0]
            raw_answer = parts[1] if len(parts) > 1 else ""
        else:
            raw_think = response
            raw_answer = ""

        # 2. 清理所有指定标签
        tags_to_remove = ["<think>", "</think>", "<functioncall>", "</functioncall>"]
        clean_think = raw_think
        clean_answer = raw_answer
        
        for tag in tags_to_remove:
            clean_think = clean_think.replace(tag, "")
            clean_answer = clean_answer.replace(tag, "")
        
        clean_think = clean_think.strip()
        clean_answer = clean_answer.strip()

        # 3. 重新赋值并计算长度
        item["think"] = clean_think
        item["answer"] = clean_answer
        item["think_len"] = len(clean_think)
        item["answer_len"] = len(clean_answer)
        
        processed_data.append(item)
        processed_count += 1

    # 4. 统一写回为一个标准的 JSON 数组文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    
    print("-" * 30)
    print(f"处理完成！")
    print(f"成功处理: {processed_count} 条")
    print(f"失败跳过: {error_count} 条")
    print(f"结果已保存至: {output_path}")

# 路径设置
input_file = "/data2/jyyan10/verl-0.3.0.post1-ws/default_project_skyworkqwen_generic.json"
output_file = "/data2/cwli16/data/rollout_for_r1/agent_default_project_skyworkqwen_generic_fixed.json"

if __name__ == "__main__":
    if os.path.exists(input_file):
        process_json_data(input_file, output_file)
    else:
        print(f"错误：找不到输入文件 {input_file}")

