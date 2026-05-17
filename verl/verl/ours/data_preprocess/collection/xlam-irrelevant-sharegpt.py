import argparse
import os
import datasets
import json

# ================= 配置区域 =================

# 保持与正样本完全一致的 Prompt 模板 (包含 Relevance Check)
INSTRUCTION_BLOCK = """### Instructions:
1. **Analyze**: Deeply understand the user's intent and the specific requirements of their query.
2. **Check Relevance**: CRITICAL! You must first evaluate if the provided tools are relevant to the user's request.
   - If the user's request **cannot** be fulfilled by any of the available tools, or if the tools are completely unrelated, you MUST NOT hallucinate a function call.
   - If the request is "chitchat" (e.g., "hello", "who are you") and requires no tools, treat it as irrelevant to tool usage.
3. **Think**: You MUST first outline your reasoning and planning process. Wrap this strictly inside <think> and </think> tags.
   - Explicitly state whether a matching tool was found or if the request is irrelevant.
4. **Act**: 
   - If a relevant tool is found: Generate the function call.
   - If NO relevant tool is found: Output an empty list `[]`.
   - The output must be wrapped in <functioncall> and <|endoftext|>.

### Output Format:

**Scenario 1: Tool Found**
<think>
1. User wants to [intent].
2. Checked tool [tool_name], it matches because [reason].
3. Extracting parameters: [param] -> [value].
</think>
<functioncall> {"name": "selected_tool_name", "arguments": {"arg1": "value1", ...}} <|endoftext|>

**Scenario 2: No Relevant Tool (Irrelevance)**
<think>
1. User wants to [intent].
2. Checked tool [tool_name_A], it is for [function], not matching.
3. Checked tool [tool_name_B], it is for [function], not matching.
4. Conclusion: No tools provided are suitable for this request.
</think>
<functioncall> [] <|endoftext|>"""

def process_irrelevant_data(args):
    # 1. 加载数据
    data_source_repo = 'sanjay920/xlam-irrelevant-sharegpt'
    print(f"Loading dataset: {data_source_repo} (Split: train)...")
    
    try:
        # 加载数据集
        full_dataset = datasets.load_dataset(data_source_repo, split='train')
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # 2. 修改采样逻辑 (1000 Train + 500 Val)
    TARGET_TRAIN_SIZE = 1000
    TARGET_VAL_SIZE = 500
    TOTAL_REQUIRED = TARGET_TRAIN_SIZE + TARGET_VAL_SIZE
    
    print(f"Original dataset size: {len(full_dataset)}")
    
    if len(full_dataset) < TOTAL_REQUIRED:
        print(f"❌ Error: Dataset size ({len(full_dataset)}) is smaller than required {TOTAL_REQUIRED}")
        return

    # 打乱数据并只取前 1500 条
    print(f"Shuffling and selecting top {TOTAL_REQUIRED} samples...")
    sampled_dataset = full_dataset.shuffle(seed=42).select(range(TOTAL_REQUIRED))
    
    # 切分数据：Test size = 500, 剩下的 1000 自动归为 Train
    split_dataset = sampled_dataset.train_test_split(test_size=TARGET_VAL_SIZE, seed=42)
    
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    
    print(f"✅ Final sizes -> Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 3. 定义处理函数
    def format_irrelevant_entry(example, idx, split_name):
        # --- A. 提取关键内容 ---
        # 按照你的要求提取 query 和 tools
        original_query = example.get('query', '')
        original_tools = example.get('tools', [])
        
        # --- B. 构造 Answers ---
        # 核心修改点：既然是无关数据，Ground Truth 必须是空列表
        original_answers = "[]"

        # --- C. 处理 Tools 格式 ---
        if isinstance(original_tools, (list, dict)):
            tools_str = json.dumps(original_tools, ensure_ascii=False)
        else:
            tools_str = str(original_tools)

        # --- D. 构造 System Message ---
        system_msg = {
            "role": "system",
            "content": "You are a helpful assistant. Your goal is to solve the user's problem by using the provided tools effectively"
        }

        # --- E. 构造 User Message ---
        # 使用相同的 Prompt 模版，确保模型行为一致
        user_content_str = (
            f"### Available Tools:\n{tools_str}\n\n"
            f"{INSTRUCTION_BLOCK}\n\n"
            f"### User Question:\n{original_query}"
        )

        user_msg = {
            "role": "user",
            "content": user_content_str
        }

        # --- F. 返回标准化格式 ---
        return {
            "data_source": "xlam-irrelevant-sharegpt",
            "prompt": [system_msg, user_msg],
            "ability": "function_calling_irrelevant", # 标记能力类型
            "reward_model": {
                "style": "rule",
                "ground_truth": original_answers # 这里是 []
            },
            "extra_info": {
                'split': split_name,
                'index': idx,
                "query_raw": original_query
            }
        }

    # 4. 执行映射处理 (Map)
    final_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    
    # 你指定的要清除的列（实际上我们会移除所有原始列，但这里列出来作为确认）
    # columns_to_remove = ['query', 'tools', 'gpt4o_response', 'row_id', 'conversations']
    
    print("\nProcessing Training Set...")
    train_dataset = train_dataset.map(
        lambda x, i: format_irrelevant_entry(x, i, 'train'),
        with_indices=True,
        remove_columns=train_dataset.column_names # 直接移除所有原始列，包括 query, tools, gpt4o_response 等
    )
    train_dataset = train_dataset.select_columns(final_columns)

    print("\nProcessing Validation Set...")
    val_dataset = val_dataset.map(
        lambda x, i: format_irrelevant_entry(x, i, 'val'),
        with_indices=True,
        remove_columns=val_dataset.column_names
    )
    val_dataset = val_dataset.select_columns(final_columns)

    # 5. 保存结果
    os.makedirs(args.local_dir, exist_ok=True)
    
    # 修改文件名，避免覆盖之前的正样本文件
    train_output_path = os.path.join(args.local_dir, 'train_irrelevant_1k.parquet')
    val_output_path = os.path.join(args.local_dir, 'test_irrelevant_500.parquet')

    train_dataset.to_parquet(train_output_path)
    val_dataset.to_parquet(val_output_path)

    print(f"\n🎉 Success!")
    print(f"Saved Irrelevant Train set ({len(train_dataset)}) to: {train_output_path}")
    print(f"Saved Irrelevant Test set ({len(val_dataset)}) to: {val_output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 建议保存到同一个目录下
    parser.add_argument('--local_dir', default='/data2/cwli16/data/train_covalid/xlam-function-calling') 
    args = parser.parse_args()
    
    process_irrelevant_data(args)