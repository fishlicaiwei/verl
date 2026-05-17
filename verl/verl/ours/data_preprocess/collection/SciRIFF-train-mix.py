import argparse
import os
import datasets
import json

# 定义需要从原始数据中移除的所有列的列表
# 我们需要显式移除 'id' 和 'dataset' (如果存在)，并让 map 函数处理 'messages'
COLUMNS_TO_REMOVE_AFTER_EXTRACTION = [
    'id', 
    'dataset', # SciRIFF 数据集可能会有的字段
    'messages', # 在提取内容后移除
    'prompt', # 如果原始数据集意外存在
    # 以下为其他可能存在的原始列，全部移除
    'data_source', 
    'solution', 
    'ability', 
    'reward_model', 
    'extra_info', 
    '__index_level_0__',
    'instruction', 
    'output', 
    'response',
    'text', 
    'input',
    'chosen', # 确保移除无关列
    'rejected', # 确保移除无关列
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 🎯 目标目录改为 SciRIFF 路径
    parser.add_argument('--local_dir', default='/data2/cwli16/data/SciRIFF-sampled') 
    args = parser.parse_args()

    # 🎯 切换数据源
    data_source_repo = 'allenai/SciRIFF-train-mix' 
    # 🎯 切换 data_source 最终值
    data_source_value = 'SciRIFF-train-mix'
    
    TRAIN_SIZE = 10000
    VAL_SIZE = 500
    RANDOM_SEED = 42
    
    print(f"Loading dataset: {data_source_repo} (Split: train)...")
    try:
        # SciRIFF-train-mix 只有一个 'train' 分割
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
            
            # 1. 提取原始的 'messages' 字段
            messages = example.pop('messages')
            
            question_content = ""
            ground_truth_content = ""
            
            if isinstance(messages, list) and len(messages) > 0:
                # 提取 Prompt (第一个 role: user 的 content)
                for msg in messages:
                    if msg.get('role') == 'user' and 'content' in msg:
                        question_content = msg['content']
                        break # 找到第一个用户消息后停止
                
                # 提取 Gold Truth (最后一个 role: assistant 的 content)
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant' and 'content' in msg:
                        ground_truth_content = msg['content']
                        break # 找到最后一个助手消息后停止
            
            # 2. 移除所有不需要的原始列 (在提取内容后移除)
            for col in COLUMNS_TO_REMOVE_AFTER_EXTRACTION:
                example.pop(col, None) 
            
            # 3. 构造输出数据结构
            data = {
                "data_source": data_source_value, 
                # 仅包含用户 prompt，作为模型的输入
                "prompt": [{
                    "role": "user",
                    "content": str(question_content), 
                }],
                "ability": "general", # 标记为通用指令遵循
                "reward_model": {
                    "style": "sft", # 标记为 SFT 格式
                    # Gold Truth 存储在 ground_truth 中
                    "ground_truth": str(ground_truth_content), 
                },
                "extra_info": {
                    'split': split_name, 
                    'index': idx,
                    "messages_raw": messages, # 保留原始消息列表，方便调试
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
    # 输出文件名为 test_general.parquet，反映其通用性
    output_val_path = os.path.join(local_dir, 'test_general.parquet') 
    
    val_dataset.to_parquet(output_val_path)
    print(f"✅ Validation data saved to: {output_val_path} (Size: {len(val_dataset)})")


    # --- 关键步骤 4: 处理和保存 Training Set ---
    print("\nProcessing Training Set...")
    # 使用 remove_columns 清理 map 过程中可能残留的原始字段
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=train_dataset.column_names)

    # 显式移除 map 后剩余的未声明字段
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in final_columns])
    
    output_train_path = os.path.join(local_dir, 'train_general_partial.parquet')
    
    train_dataset.to_parquet(output_train_path)
    print(f"✅ Training data saved to: {output_train_path} (Size: {len(train_dataset)})")