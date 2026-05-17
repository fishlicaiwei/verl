import argparse
import os
import datasets
import json

# ================= 配置区域 =================

# 1. 静态指令文本 (保持与 xLAM 处理脚本完全一致)
INSTRUCTION_BLOCK = """### Instructions:
1. **Analyze**: Deeply understand the user's intent and match it with the available tools.
2. **Think**: You MUST first outline your reasoning and planning process. Wrap this strictly inside <think> and </think> tags.
3. **Act**: Select the appropriate tool and generate the function call.
   - The output must be wrapped in <functioncall> and <|endoftext|>.
   - The content inside must be a valid JSON object with "name" and "arguments".

### Output Format:
<think>
[Your step-by-step reasoning and parameter extraction process goes here]
</think>
<functioncall> {"name": "selected_tool_name", "arguments": {"arg1": "value1", ...}} <|endoftext|>"""

def process_argilla_data(args):
    # 1. 加载数据
    data_source_repo = 'argilla/Synth-APIGen-v0.1'
    print(f"Loading dataset: {data_source_repo}...")
    
    try:
        # 加载训练集 (该数据集通常只有 train split)
        full_dataset = datasets.load_dataset(data_source_repo, split='train')
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    print(f"Original dataset size: {len(full_dataset)}")

    # 2. 关键步骤: 过滤 Tools 为空的无效数据
    print("Filtering out samples with empty tools...")
    
    def filter_empty_tools(example):
        tools = example.get('tools')
        # 如果是 None，丢弃
        if tools is None:
            return False
        # 如果是空列表，丢弃
        if isinstance(tools, list) and len(tools) == 0:
            return False
        # 如果是字符串类型的空列表 "[]"，丢弃
        if isinstance(tools, str) and tools.strip() == "[]":
            return False
        return True

    filtered_dataset = full_dataset.filter(filter_empty_tools)
    print(f"Dataset size after filtering: {len(filtered_dataset)}")

    # 3. 采样逻辑 (只需要 500 条测试集)
    TARGET_TEST_SIZE = 500
    
    if len(filtered_dataset) < TARGET_TEST_SIZE:
        print(f"❌ Error: Filtered dataset size ({len(filtered_dataset)}) is smaller than required {TARGET_TEST_SIZE}")
        return

    print(f"Shuffling and selecting {TARGET_TEST_SIZE} samples for Test set...")
    # 打乱并取前 500 条
    test_dataset = filtered_dataset.shuffle(seed=42).select(range(TARGET_TEST_SIZE))
    
    print(f"✅ Final Test Dataset size: {len(test_dataset)}")

    # 4. 定义格式化处理函数
    def format_argilla_entry(example, idx):
        # --- 提取原始数据 ---
        original_query = example.get('query', '')
        original_tools = example.get('tools', [])
        
        # --- 处理 Tools 格式 ---
        # 确保 tools 是 JSON 字符串格式，用于拼接到 Prompt
        if isinstance(original_tools, (list, dict)):
            tools_str = json.dumps(original_tools, ensure_ascii=False)
        else:
            tools_str = str(original_tools)

        # --- 构造 System Message ---
        system_msg = {
            "role": "system",
            "content": "You are a helpful assistant. Your goal is to solve the user's problem by using the provided tools effectively"
        }

        # --- 构造 User Message (复杂拼接) ---
        # 逻辑: Tools -> Instructions -> Output Format -> Query
        user_content_str = (
            f"### Available Tools:\n{tools_str}\n\n"
            f"{INSTRUCTION_BLOCK}\n\n"
            f"### User Question:\n{original_query}"
        )

        user_msg = {
            "role": "user",
            "content": user_content_str
        }

        # --- 构造最终字典 ---
        return {
            "data_source": "argilla-Synth-APIGen-v0.1",
            "prompt": [system_msg, user_msg],
            "ability": "function_calling",
            "reward_model": {
                "style": "rule",
                "ground_truth": None  # 🎯 你的要求：没有 Ground Truth
            },
            "extra_info": {
                'split': 'test',
                'index': idx,
                "query_raw": original_query
            }
        }

    # 5. 执行映射处理 (Map)
    final_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

    print("\nProcessing Test Set...")
    test_dataset = test_dataset.map(
        format_argilla_entry,
        with_indices=True,
        remove_columns=test_dataset.column_names # 移除原始列
    )
    
    # 确保只保留定义的列
    test_dataset = test_dataset.select_columns(final_columns)

    # 6. 保存结果
    os.makedirs(args.local_dir, exist_ok=True)
    
    # 只保存这一个测试文件
    output_path = os.path.join(args.local_dir, 'test_argilla_500.parquet')

    test_dataset.to_parquet(output_path)

    print(f"\n🎉 Success!")
    print(f"Saved Argilla Test set ({len(test_dataset)}) to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 🎯 默认输出路径 (可以根据需要修改)
    parser.add_argument('--local_dir', default='/data2/cwli16/data/train_covalid/argilla-processing') 
    args = parser.parse_args()
    
    process_argilla_data(args)