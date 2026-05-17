"""
Preprocess the PKU-Alignment/PKU-SafeRLHF dataset (subset alpaca3-8b, train split, sampled)
强制移除所有不需要的列，只保留 'prompt' 内容。
"""

import argparse
import os
import datasets
import json

# 定义需要从原始数据中移除的所有列的列表
# 我们将移除所有原始列，除了 'prompt'，因为它需要被提取到新的结构中。
COLUMNS_TO_REMOVE_AFTER_PROMPT = [
    # 常见的 SFT/RLHF 数据集残留列
    'response_0', 
    'response_1',
    'id', 
    'messages',
    'data_source', 
    'solution', 
    'ability', 
    'reward_model', 
    'extra_info', 
    '__index_level_0__',
    # PKU-SafeRLHF 特有列，必须清除
    'is_response_0_safe', 
    'is_response_1_safe', 
    'better_response_id', 
    'safer_response_id',
    'prompt_source',
    'response_0_source',
    'response_1_source',
    'response_0_harm_category',
    'response_1_harm_category',
    'response_0_severity_level',
    'response_1_severity_level',
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/data2/cwli16/data/PKU-SafeRLHF-sampled/test_safety.parquet') 
    args = parser.parse_args()

    data_source_repo = 'PKU-Alignment/PKU-SafeRLHF'
    data_source_value = 'PKU-SafeRLHF' # 最终要硬编码的值
    data_source_subset = 'default'
    
    TRAIN_SIZE = 10000
    VAL_SIZE = 500
    RANDOM_SEED = 42
    
    # 最终保留的列
    final_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    
    os.makedirs(args.local_dir, exist_ok=True) 

    # --- 关键步骤 1: 处理 Validation Set (从原始 test 分割中抽取 500 条) ---
    print(f"Loading original TEST split from {data_source_repo}...")
    try:
        # 1. 加载原始的 test 分割
        val_dataset_orig = datasets.load_dataset(data_source_repo, data_source_subset, split='test')
    except Exception as e:
        print(f"Error loading original test split: {e}")
        exit()
    
    if len(val_dataset_orig) < VAL_SIZE:
        print(f"❌ ERROR: Original test size ({len(val_dataset_orig)}) is less than required validation size ({VAL_SIZE}). Aborting.")
        exit()

    print(f"Shuffling and sampling {VAL_SIZE} items for Validation...")
    # 抽取 500 条验证集
    val_dataset = val_dataset_orig.shuffle(seed=RANDOM_SEED).select(range(VAL_SIZE))
    
    print(f"Validation Dataset size: {len(val_dataset)}")


    # --- 关键步骤 2: 处理 Training Set (从原始 train 分割中抽取 10000 条) ---
    print(f"\nLoading original TRAIN split from {data_source_repo}...")
    try:
        # 1. 加载原始的 train 分割
        train_dataset_orig = datasets.load_dataset(data_source_repo, data_source_subset, split='train')
    except Exception as e:
        print(f"Error loading original train split: {e}")
        exit()

    if len(train_dataset_orig) < TRAIN_SIZE:
        print(f"❌ ERROR: Original train size ({len(train_dataset_orig)}) is less than required train size ({TRAIN_SIZE}). Aborting.")
        exit()

    print(f"Shuffling and sampling {TRAIN_SIZE} items for Training...")
    # 抽取 10000 条训练集
    train_dataset = train_dataset_orig.shuffle(seed=RANDOM_SEED).select(range(TRAIN_SIZE))
    
    print(f"Training Dataset size: {len(train_dataset)}")


    # --- 关键步骤 3: 定义数据处理函数 (保持不变) ---

    def make_map_fn(split_name):
        def process_fn(example, idx):
            
            # 1. 提取原始的 'prompt' 字段
            question_raw = example.pop('prompt') 
            question = str(question_raw) 

            # 2. 移除所有不需要的列 
            for col in COLUMNS_TO_REMOVE_AFTER_PROMPT:
                example.pop(col, None) 
            
            # 3. 构造输出数据结构
            data = {
                # 🎯 data_source 硬编码为指定的值
                "data_source": data_source_value, 
                "prompt": [{
                    "role": "user",
                    "content": question, 
                }],
                "ability": "safety", # 标记为安全能力
                "reward_model": {
                    "style": "rule",
                    "ground_truth": None 
                },
                "extra_info": {
                    'split': split_name, 
                    'index': idx,
                    "question_raw": question_raw,
                }
            }
            return data

        return process_fn

    # --- 关键步骤 4: 处理和保存 Validation Set (500 条) ---
    print("\nProcessing Validation Set...")
    # 使用 remove_columns 清理原始字段
    val_dataset = val_dataset.map(function=make_map_fn('val'), with_indices=True, remove_columns=val_dataset.column_names)
    
    # 显式移除 map 后剩余的未声明字段，并确保只保留 final_columns
    val_dataset = val_dataset.remove_columns([col for col in val_dataset.column_names if col not in final_columns])

    output_val_path = os.path.join(args.local_dir, 'test_safety.parquet') 
    
    # 修正：移除 index=False 参数
    val_dataset.to_parquet(output_val_path) 
    print(f"✅ Validation data saved to: {output_val_path} (Size: {len(val_dataset)})")


    # --- 关键步骤 5: 处理和保存 Training Set (10000 条) ---
    print("\nProcessing Training Set...")
    # 使用 remove_columns 清理原始字段
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=train_dataset.column_names)

    # 显式移除 map 后剩余的未声明字段，并确保只保留 final_columns
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in final_columns])
    
    output_train_path = os.path.join(args.local_dir, 'train_safety_partial.parquet')
    
    # 修正：移除 index=False 参数
    train_dataset.to_parquet(output_train_path)
    print(f"✅ Training data saved to: {output_train_path} (Size: {len(train_dataset)})")