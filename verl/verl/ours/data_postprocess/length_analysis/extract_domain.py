import json
import os

# --- 配置路径 ---
source_file_path = "/data2/cwli16/data/rollout1/llm-as-judge_writting_grpo_qwen2-5_7b_skywork_generic(1).json"
output_dir = "/data2/cwli16/data/rollout2/"
target_domain = "DeepWriting-20K"


# 构建输出文件名
output_file_name = f"{target_domain}_cleaned.json"
output_file_path = os.path.join(output_dir, output_file_name)

def extract_and_clean_data():
    # 1. 检查源文件
    if not os.path.exists(source_file_path):
        print(f"❌ 错误: 找不到源文件: {source_file_path}")
        return

    # 2. 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"📖 正在读取并处理: {source_file_path} ...")
    
    try:
        with open(source_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if not isinstance(data, list):
                print("❌ 错误: JSON 格式非列表")
                return

            # --- 核心逻辑：领域筛选 + 长度清洗 ---
            # 只有 domain 匹配 且 think_len 不等于 answer_len 的才保留
            final_data = []
            domain_match_count = 0  # 仅匹配 domain 的总数
            cleaned_count = 0       # 被剔除的异常数据总数

            for item in data:
                if item.get('domain') == target_domain:
                    domain_match_count += 1
                    # 检查长度是否相等（格式错误剔除）
                    if item.get('think_len') == item.get('answer_len'):
                        cleaned_count += 1
                    else:
                        final_data.append(item)

            # 3. 输出统计信息
            print("-" * 30)
            print(f"📊 统计报告 ({target_domain}):")
            print(f"   - 该领域总数据量: {domain_match_count} 条")
            print(f"   - 格式错误被剔除: {cleaned_count} 条 (think_len == answer_len)")
            print(f"   - 最终有效数据量: {len(final_data)} 条")
            print("-" * 30)

            if len(final_data) > 0:
                # 4. 写入新文件
                with open(output_file_path, 'w', encoding='utf-8') as out_f:
                    json.dump(final_data, out_f, ensure_ascii=False, indent=4)
                print(f"🎉 成功! 清洗后的数据已保存至: {output_file_path}")
            else:
                print("⚠️ 警告: 清洗后没有剩余有效数据。")

    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析错误: {e}")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

if __name__ == "__main__":
    extract_and_clean_data()