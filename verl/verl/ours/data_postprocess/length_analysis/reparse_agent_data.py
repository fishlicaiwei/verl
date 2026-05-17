import json
import os
import re

# --- 配置路径 ---
source_file_path = "/data2/cwli16/data/rollout1/llm-as-judge_agent_grpo_qwen2-5_7b_skywork_generic(1).json"
output_path = "/data2/cwli16/data/rollout2/agent_reparsed_cleaned.json"

def clean_tags(text):
    """清除所有定义的标签"""
    tags = ["<think>", "</think>", "<functioncall>", "</functioncall>", "<answer>", "</answer>"]
    for tag in tags:
        text = text.replace(tag, "")
    return text.strip()

def reparse_agent_data():
    if not os.path.exists(source_file_path):
        print(f"❌ 错误: 找不到源文件 {source_file_path}")
        return

    print(f"📖 正在重新解析: {source_file_path} ...")
    
    try:
        with open(source_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        parsed_data = []
        fail_count = 0
        total_count = len(data)

        for item in data:
            response = item.get("response", "")
            think_part = ""
            answer_part = ""
            found = False

            # 1. 优先逻辑：尝试匹配 <functioncall>
            if "<functioncall>" in response:
                parts = response.split("<functioncall>")
                think_part = parts[0]
                answer_part = parts[1]
                found = True
            
            # 2. 次选逻辑：尝试匹配 <answer>
            elif "<answer>" in response:
                parts = response.split("<answer>")
                think_part = parts[0]
                answer_part = parts[1]
                found = True

            # 3. 备选逻辑：根据 </think> 截断
            elif "</think>" in response:
                parts = response.split("</think>")
                think_part = parts[0]
                answer_part = parts[1]
                found = True

            # 如果解析成功，清理标签并填入
            if found and (think_part.strip() or answer_part.strip()):
                # 清洗标签
                clean_think = clean_tags(think_part)
                clean_answer = clean_tags(answer_part)

                # 更新数据字段
                item["think"] = clean_think
                item["answer"] = clean_answer
                item["think_len"] = len(clean_think)
                item["answer_len"] = len(clean_answer)
                
                # 剔除清洗后长度依然完全相等且内容一致的异常情况（可选，根据你之前的需求）
                if item["think_len"] == item["answer_len"] and clean_think == clean_answer:
                    fail_count += 1
                else:
                    parsed_data.append(item)
            else:
                fail_count += 1

        # --- 结果报告 ---
        print("-" * 40)
        print(f"📊 解析报告:")
        print(f"   - 原始总数: {total_count}")
        print(f"   - 成功解析并保留: {len(parsed_data)}")
        print(f"   - 无法解析或格式错误被删除: {fail_count}")
        print("-" * 40)

        # 写入新文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(parsed_data, out_f, ensure_ascii=False, indent=4)
        
        print(f"🎉 处理完成! 文件保存至: {output_path}")

    except Exception as e:
        print(f"❌ 运行出错: {e}")

if __name__ == "__main__":
    reparse_agent_data()