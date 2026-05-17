import argparse
import os
import datasets
import json

# 定义需要从原始数据中移除的所有列的列表
# openai_humaneval 原始列包括：task_id, prompt, canonical_solution, test, entry_point
COLUMNS_TO_REMOVE = [
    'task_id', 
    'prompt', 
    'canonical_solution', 
    'test',
    'entry_point',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 🎯 目标目录改为 openai_humaneval 路径
    parser.add_argument('--local_dir', default='/data2/cwli16/data/openai_humaneval_processed_val')
    args = parser.parse_args()

    # 🎯 切换数据源
    data_source_repo = 'openai/openai_humaneval' 
    # HumanEval 通常只有 'test' 分割，或者只有一个默认的 'train' 分割包含所有数据
    DATA_SPLIT = 'test' 
    data_source_value = 'openai_humaneval'
    
    VAL_SIZE_TARGET = 500 # 保持验证集大小目标
    
    # 📝 只需要加载一次数据集
    print(f"Loading dataset: {data_source_repo} (Split: {DATA_SPLIT})...")
    try:
        # 加载整个数据集。如果 'test' 不存在，尝试 'train'
        try:
            full_dataset = datasets.load_dataset(data_source_repo, split=DATA_SPLIT)
        except:
            print(f"Warning: '{DATA_SPLIT}' split not found. Trying 'train' split.")
            DATA_SPLIT = 'train'
            full_dataset = datasets.load_dataset(data_source_repo, split=DATA_SPLIT)

    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    # --- 关键步骤 1: 数据分割 (HumanEval 通常只有 164 个样本，我们直接使用全部数据) ---
    
    current_size = len(full_dataset)
    
    # HumanEval 只有 164 个样本，少于 500，直接使用全部
    print(f"✅ Using all available data ({current_size}) as validation set.")
    val_dataset = full_dataset
    
    print(f"Final Validation Dataset size: {len(val_dataset)}")


    # --- 关键步骤 2: 定义数据处理函数 ---

    def make_map_fn(split_name):
        def process_fn(example, idx):
            
            # 1. 提取原始的 'prompt' 字段作为 prompt 内容
            # prompt 包含函数签名和 docstring
            prompt_raw = example.get('prompt', "") 
            
            # 2. 提取 'canonical_solution' 字段作为 ground_truth (正确代码)
            canonical_solution = example.get('canonical_solution')
            
            # 3. 构造输出数据结构
            data = {
                "data_source": data_source_value, 
                "prompt": [{
                    "role": "user",
                    "content": str(prompt_raw).strip(), # <--- 最终使用的 prompt 内容
                }],
                "ability": "code", # 代码生成任务使用 'code'
                "reward_model": {
                    "style": "execution", # 这种代码任务通常需要执行测试来验证
                    # 'ground_truth' 对应标准解决方案
                    "ground_truth": str(canonical_solution) if canonical_solution is not None else None
                },
                "extra_info": {
                    'split': split_name, 
                    'index': idx,
                    "task_id": example.get('task_id'), # 保留任务ID
                    "test_code": example.get('test'), # 保留测试代码，用于评估
                    "entry_point": example.get('entry_point'), # 保留函数入口
                }
            }
            return data

        return process_fn
    
    # 最终保留的列
    final_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

    # --- 关键步骤 3: 处理和保存 Validation Set ---
    print("\nProcessing Validation Set...")
    
    val_dataset = val_dataset.map(
        function=make_map_fn('val'), 
        with_indices=True, 
        remove_columns=val_dataset.column_names # 移除所有原始列
    )
    
    val_dataset = val_dataset.remove_columns([col for col in val_dataset.column_names if col not in final_columns])

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True) 
    # 输出文件名为 test_code.parquet 
    output_val_path = os.path.join(local_dir, 'test_code.parquet') 
    
    val_dataset.to_parquet(output_val_path)
    print(f"✅ Validation data saved to: {output_val_path} (Size: {len(val_dataset)})")
    print(f"字段结构检查: {val_dataset.column_names}")

    print("\n数据集处理完成。")