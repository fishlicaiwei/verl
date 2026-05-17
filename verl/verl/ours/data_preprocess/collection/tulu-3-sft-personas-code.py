import argparse
import os
import datasets
import json

# 必须移除的原始列列表 (根据您的要求和常见的 SFT 数据结构)
COLUMNS_TO_REMOVE = [
    'id', 
    'messages', # 原始数据集的对话历史/完整内容
    # 额外移除我们在目标格式中不需要的字段 (如果原始数据集包含的话)
    'data_source', # 原始 data_source 也会被移除，因为我们要用新的值覆盖它
    'solution', 
    'ability', 
    'reward_model', 
    'extra_info', 
    '__index_level_0__',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 路径改为目标目录
    parser.add_argument('--local_dir', default='/data2/cwli16/data/tulu-3-sft-personas-code') 
    args = parser.parse_args()

    # ❗ 移除 data_source_repo 和 data_source_subset 变量的依赖
    
    TRAIN_SIZE = 10000
    VAL_SIZE = 500
    RANDOM_SEED = 42
    
    data_source_repo = 'allenai/tulu-3-sft-personas-code' # 仅用于加载数据集的标识

    print(f"Loading dataset: {data_source_repo} (Split: train)...")
    try:
        # 加载数据集
        full_dataset = datasets.load_dataset(data_source_repo, split='train')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    # --- 关键步骤 1: 数据隔离与分割 ---
    
    if len(full_dataset) < TRAIN_SIZE + VAL_SIZE:
        print(f"❌ ERROR: Dataset size ({len(full_dataset)}) is less than required size ({TRAIN_SIZE + VAL_SIZE}). Aborting.")
        exit()

    print(f"Splitting dataset: {TRAIN_SIZE} for Train, {VAL_SIZE} for Validation...")
    
    # 确保没有交叉污染，使用 train_test_split (默认先shuffle)
    split_dataset = full_dataset.train_test_split(
        test_size=VAL_SIZE,  # 确保 Validation 部分是 500 条
        train_size=TRAIN_SIZE, # 确保 Train 部分是 10000 条
        seed=RANDOM_SEED # 保证可重现性
    )
    
    # 获取隔离后的数据集
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    
    print(f"Train Dataset size: {len(train_dataset)}")
    print(f"Validation Dataset size: {len(val_dataset)}")


    # --- 关键步骤 2: 定义数据处理函数 ---

    def make_map_fn(split_name):
        def process_fn(example, idx):
            
            # 1. 提取原始的 'prompt' 字段
            question_raw = example.pop('prompt') 
            
            # 确保 question 是字符串类型
            question = str(question_raw) 

            # 2. 移除所有不需要的列 (id, messages, etc.)
            for col in COLUMNS_TO_REMOVE:
                example.pop(col, None) 
            
            # 3. 构造输出数据结构
            data = {
                # ❗ 关键修改 A: data_source 直接硬编码为目标字符串
                "data_source": "tulu-3-sft-personas-code", 
                "prompt": [{
                    "role": "user",
                    "content": question, # <--- 最终使用的 prompt 内容
                }],
                "ability": "code", # 标记为代码能力
                "reward_model": {
                    "style": "rule",
                    "ground_truth": None 
                },
                "extra_info": {
                    'split': split_name, # 标记是 train 还是 val
                    'index': idx,
                    "question_raw": question_raw,
                }
            }
            return data

        return process_fn
    
    # ❗ 关键修改 B: 最终保留的列不再包含 'source_name'
    final_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

    # --- 关键步骤 3: 处理和保存 Validation Set (500 条) ---
    print("\nProcessing Validation Set...")
    # 使用 remove_columns 清理 map 过程中可能残留的原始字段
    val_dataset = val_dataset.map(function=make_map_fn('val'), with_indices=True, remove_columns=val_dataset.column_names)
    
    # 显式移除 map 后剩余的未声明字段
    val_dataset = val_dataset.remove_columns([col for col in val_dataset.column_names if col not in final_columns])

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True) 
    output_val_path = os.path.join(local_dir, 'test_code.parquet') 
    
    # ❗ 确保 to_parquet 使用 index=False 避免残留索引列
    val_dataset.to_parquet(output_val_path)
    print(f"✅ Validation data saved to: {output_val_path} (Size: {len(val_dataset)})")


    # --- 关键步骤 4: 处理和保存 Training Set (10000 条) ---
    print("\nProcessing Training Set...")
    # 使用 remove_columns 清理 map 过程中可能残留的原始字段
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=train_dataset.column_names)

    # 显式移除 map 后剩余的未声明字段
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in final_columns])
    
    output_train_path = os.path.join(local_dir, 'train_code_partial.parquet')
    
    # ❗ 确保 to_parquet 使用 index=False 避免残留索引列
    train_dataset.to_parquet(output_train_path)
    print(f"✅ Training data saved to: {output_train_path} (Size: {len(train_dataset)})")