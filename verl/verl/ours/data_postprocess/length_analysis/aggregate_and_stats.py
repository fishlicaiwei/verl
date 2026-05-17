import json
from collections import defaultdict
import os

# --- 配置路径 ---
# 使用你上一步清洗后的文件
input_file_path = "/data2/cwli16/data/rollout2/xlam-function-calling-60k_cleaned.json"

def aggregate_and_stats():
    if not os.path.exists(input_file_path):
        print(f"❌ 错误: 找不到文件 {input_file_path}")
        return

    print(f"📖 正在加载数据: {input_file_path}")
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. 使用字典聚合，Key 是 prompt，Value 是该 prompt 对应的所有 rollout 列表
        prompt_groups = defaultdict(list)
        
        for item in data:
            p_text = item.get('prompt')
            prompt_groups[p_text].append(item)

        # 2. 统计信息
        total_samples = len(data)
        unique_prompts_count = len(prompt_groups)
        
        # 统计每个 Prompt 拥有的 Rollout 数量分布
        rollout_counts = defaultdict(int)
        for p, rollouts in prompt_groups.items():
            rollout_counts[len(rollouts)] += 1

        print("-" * 40)
        print(f"📊 聚合统计报告:")
        print(f"   - 总样本数 (Total Rows): {total_samples}")
        print(f"   - 唯一 Prompt 类数 (Unique Prompts): {unique_prompts_count}")
        print("-" * 40)
        print("📈 Rollout 分布情况:")
        for count in sorted(rollout_counts.keys()):
            num_prompts = rollout_counts[count]
            print(f"   - 包含 {count} 个 Rollout 的 Prompt 有: {num_prompts} 类")
        print("-" * 40)

        # 3. (可选) 如果你想看一眼具体的聚合结构，可以取消下面注释
        # for i, (p, rollouts) in enumerate(prompt_groups.items()):
        #     if i < 1: # 只看第一个
        #         print(f"示例 Prompt 的第一个 Rollout Score: {rollouts[0].get('score')}")

    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    aggregate_and_stats()