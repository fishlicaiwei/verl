import argparse
import os
import datasets
import json

# ================= 配置区域 =================

# 定义静态指令文本 (包含 Instructions 和 Output Format)
# 更新：增加了 Step 2 Check Relevance 和 两种 Scenario 的输出格式
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

# 定义需要移除的原始列
COLUMNS_TO_REMOVE = ['id', 'query', 'answers', 'tools']

def process_xlam_data(args):
    # 1. 加载数据
    data_source_repo = 'Salesforce/xlam-function-calling-60k'
    print(f"Loading dataset: {data_source_repo} (Split: train)...")
    
    try:
        # xLAM 只有 'train' split
        full_dataset = datasets.load_dataset(data_source_repo, split='train')
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # 2. 精确采样逻辑 (10000 Train + 500 Val)
    TARGET_TRAIN_SIZE = 10000
    TARGET_VAL_SIZE = 500
    TOTAL_REQUIRED = TARGET_TRAIN_SIZE + TARGET_VAL_SIZE
    
    print(f"Original dataset size: {len(full_dataset)}")
    
    if len(full_dataset) < TOTAL_REQUIRED:
        print(f"❌ Error: Dataset size is smaller than required {TOTAL_REQUIRED}")
        return

    # 打乱数据并只取前 10500 条
    print(f"Shuffling and selecting top {TOTAL_REQUIRED} samples...")
    sampled_dataset = full_dataset.shuffle(seed=42).select(range(TOTAL_REQUIRED))
    
    # 切分数据：Test size = 500, 剩下的 10000 自动归为 Train
    split_dataset = sampled_dataset.train_test_split(test_size=TARGET_VAL_SIZE, seed=42)
    
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    
    print(f"✅ Final sizes -> Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 3. 定义处理函数
    def format_xlam_entry(example, idx, split_name):
        # --- 提取原始数据 ---
        original_query = example.get('query', '')
        original_tools = example.get('tools', [])
        original_answers = example.get('answers', '')

        # --- 处理 Tools 格式 (转为格式化的 JSON 字符串) ---
        # xLAM 的 tools 通常已经是 list[dict]，我们需要将其转为 string 拼接到 prompt 中
        if isinstance(original_tools, (list, dict)):
            tools_str = json.dumps(original_tools, ensure_ascii=False) # 紧凑格式
        else:
            tools_str = str(original_tools)

        # --- 构造 System Message ---
        system_msg = {
            "role": "system",
            "content": "You are a helpful assistant. Your goal is to solve the user's problem by using the provided tools effectively"
        }

        # --- 构造 User Message (复杂拼接) ---
        # 顺序: Tools -> Instructions -> Output Format -> User Question
        # 注意：INSTRUCTION_BLOCK 已经包含了 Instructions 和 Output Format
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
            "data_source": "xlam-function-calling-60k",
            "prompt": [system_msg, user_msg],
            "ability": "function_calling",
            "reward_model": {
                "style": "rule",
                "ground_truth": original_answers # 将原始 answers 放入 ground_truth
            },
            "extra_info": {
                'split': split_name,
                'index': idx,
                "query_raw": original_query
            }
        }

    # 4. 执行映射处理 (Map)
    # 定义最终需要的列
    final_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

    print("\nProcessing Training Set...")
    train_dataset = train_dataset.map(
        lambda x, i: format_xlam_entry(x, i, 'train'),
        with_indices=True,
        remove_columns=train_dataset.column_names # 移除原始列
    )
    # 确保只保留我们定义的列
    train_dataset = train_dataset.select_columns(final_columns)

    print("\nProcessing Validation Set...")
    val_dataset = val_dataset.map(
        lambda x, i: format_xlam_entry(x, i, 'val'),
        with_indices=True,
        remove_columns=val_dataset.column_names # 移除原始列
    )
    val_dataset = val_dataset.select_columns(final_columns)

    # 5. 保存结果
    os.makedirs(args.local_dir, exist_ok=True)
    
    train_output_path = os.path.join(args.local_dir, 'train_function_calling_10k.parquet')
    val_output_path = os.path.join(args.local_dir, 'test_function_calling_500.parquet')

    train_dataset.to_parquet(train_output_path)
    val_dataset.to_parquet(val_output_path)

    print(f"\n🎉 Success!")
    print(f"Saved Train set ({len(train_dataset)}) to: {train_output_path}")
    print(f"Saved Test set ({len(val_dataset)}) to: {val_output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 默认输出路径
    parser.add_argument('--local_dir', default='/data2/cwli16/data/train_covalid/xlam-function-calling') 
    args = parser.parse_args()
    
    process_xlam_data(args)