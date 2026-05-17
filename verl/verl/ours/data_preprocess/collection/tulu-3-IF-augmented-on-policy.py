import argparse
import os
import datasets
import json

# 定义需要从原始数据中移除的所有列的列表
# 目标是移除所有原始列，只保留最终需要的结构字段。
# 由于我们会在 map 函数中手动 pop 'chosen'，这里列出其他需要移除的字段。
COLUMNS_TO_REMOVE = [
    'id', 
    'rejected', # 偏好数据集中被拒绝的回答
    # 以下为可能存在的其他残留列（Tulu-3 系列数据中可能没有，但为安全起见保留）
    'messages', 
    'data_source', 
    'solution', 
    'ability', 
    'reward_model', 
    'extra_info', 
    '__index_level_0__',
    'response', 
    'source', 
    'conversation',
    'rating',
    'input',
    'prompt', # 如果原始数据集意外存在 'prompt'，也移除
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 🎯 目标目录
    parser.add_argument('--local_dir', default='/data2/cwli16/data/tulu-3-if-augmented-sampled') 
    args = parser.parse_args()

    # 🎯 切换数据源
    data_source_repo = 'allenai/tulu-3-IF-augmented-on-policy-8b' 
    # 🎯 切换 data_source 最终值
    data_source_value = 'tulu-3-IF-augmented-on-policy-8b'
    
    TRAIN_SIZE = 10000
    VAL_SIZE = 500
    RANDOM_SEED = 42
    
    print(f"Loading dataset: {data_source_repo} (Split: train)...")
    try:
        # Tulu-3 数据集通常只有一个 'train' 分割
        full_dataset = datasets.load_dataset(data_source_repo, split='train')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    # --- 关键步骤 1: 数据隔离与分割 ---
    
    required_size = TRAIN_SIZE + VAL_SIZE
    if len(full_dataset) < required_size:
        print(f"⚠️ WARNING: Dataset size ({len(full_dataset)}) is less than required size ({required_size}). Adjusting sizes.")
        VAL_SIZE = min(VAL_SIZE, len(full_dataset) // 10) 
        TRAIN_SIZE = len(full_dataset) - VAL_SIZE
        if TRAIN_SIZE <= 0:
            print(f"❌ ERROR: Dataset size is too small to split. Aborting.")
            exit()
    
    print(f"Splitting dataset: {TRAIN_SIZE} for Train, {VAL_SIZE} for Validation...")
    
    split_dataset = full_dataset.train_test_split(
        test_size=VAL_SIZE, 
        train_size=TRAIN_SIZE, 
        seed=RANDOM_SEED 
    )
    
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    
    print(f"Train Dataset size: {len(train_dataset)}")
    print(f"Validation Dataset size: {len(val_dataset)}")


    # --- 关键步骤 2: 定义数据处理函数 ---

    def make_map_fn(split_name):
        def process_fn(example, idx):
            
            # 1. 从 'chosen' 字段中提取用户 Prompt
            # chosen 是一个消息列表，第一个元素是用户消息，包含 role 和 content
            chosen_messages = example.pop('chosen') 
            
            # 提取第一个消息的内容作为 Prompt
            question_raw = chosen_messages[0]['content'] 
            question = str(question_raw) 

            # 2. 移除所有不需要的列
            # 'chosen' 已经在上面 pop 了
            for col in COLUMNS_TO_REMOVE:
                example.pop(col, None) 
            
            # 3. 构造输出数据结构
            data = {
                "data_source": data_source_value, 
                "prompt": [{
                    "role": "user",
                    "content": question, # <--- 最终使用的 prompt 内容
                }],
                "ability": "chat", 
                "reward_model": {
                    "style": "rule",
                    "ground_truth": None 
                },
                "extra_info": {
                    'split': split_name, 
                    'index': idx,
                    "chosen_messages_raw": chosen_messages, # 保留完整 chosen 消息列表作为原始信息
                }
            }
            return data

        return process_fn
    
    # 最终保留的列
    final_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

    # --- 关键步骤 3: 处理和保存 Validation Set ---
    print("\nProcessing Validation Set...")
    # 使用 remove_columns 清理 map 过程中可能残留的原始字段
    val_dataset = val_dataset.map(function=make_map_fn('val'), with_indices=True, remove_columns=val_dataset.column_names)
    
    # 显式移除 map 后剩余的未声明字段
    val_dataset = val_dataset.remove_columns([col for col in val_dataset.column_names if col not in final_columns])

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True) 
    output_val_path = os.path.join(local_dir, 'test_chat.parquet')
    
    val_dataset.to_parquet(output_val_path)
    print(f"✅ Validation data saved to: {output_val_path} (Size: {len(val_dataset)})")


    # --- 关键步骤 4: 处理和保存 Training Set ---
    print("\nProcessing Training Set...")
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=train_dataset.column_names)

    # 显式移除 map 后剩余的未声明字段
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in final_columns])
    
    output_train_path = os.path.join(local_dir, 'train_chat_partial.parquet')
    
    train_dataset.to_parquet(output_train_path)
    print(f"✅ Training data saved to: {output_train_path} (Size: {len(train_dataset)})")