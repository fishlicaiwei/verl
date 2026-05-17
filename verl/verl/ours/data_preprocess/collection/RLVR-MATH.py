import argparse
import os
import datasets
import json

# 定义需要从原始数据中移除的所有列的列表
# 我们将移除所有原始列，除了 'messages' 和 'ground_truth'，因为它们需要被提取内容。
COLUMNS_TO_REMOVE_AFTER_EXTRACTION = [
    # RLVR-MATH 常见的字段
    'id', 
    'messages', # 在提取内容后移除
    'ground_truth', # 在提取内容后移除
    'data_source', 
    'solution', 
    'ability', 
    'reward_model', 
    'extra_info', 
    '__index_level_0__',
    # RLVR-MATH 可能包含的其他列
    'split', 
    'model_output',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 🎯 目标目录改为 RLVR-MATH 路径
    parser.add_argument('--local_dir', default='/data2/cwli16/data/RLVR-MATH-sampled') 
    args = parser.parse_args()

    # 🎯 切换数据源
    data_source_repo = 'allenai/RLVR-MATH' 
    # 🎯 切换 data_source 最终值
    data_source_value = 'RLVR-MATH'
    
    TRAIN_SIZE = 7000  # 训练集目标大小
    VAL_SIZE = 500     # 验证集目标大小 (基于数据总量限制)
    RANDOM_SEED = 42
    
    print(f"Loading dataset: {data_source_repo} (Split: train)...")
    try:
        # RLVR-MATH 只有一个 'train' 分割
        full_dataset = datasets.load_dataset(data_source_repo, split='train')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    # --- 关键步骤 1: 数据隔离与分割 ---
    
    required_size = TRAIN_SIZE + VAL_SIZE
    if len(full_dataset) < required_size:
        # ⚠️ 警告用户实际的分割大小
        print(f"⚠️ WARNING: Dataset size ({len(full_dataset)}) is less than required size ({required_size}). Using all available data for train/val split.")
        TRAIN_SIZE = len(full_dataset) - VAL_SIZE # 调整训练集大小
        if TRAIN_SIZE < 0: TRAIN_SIZE = 0 # 避免负数
    
    print(f"Splitting dataset: {TRAIN_SIZE} for Train, {VAL_SIZE} for Validation...")
    
    # 确保没有交叉污染，使用 train_test_split (默认先shuffle)
    split_dataset = full_dataset.train_test_split(
        test_size=VAL_SIZE,  # 确保 Validation 部分是 500 条
        train_size=TRAIN_SIZE, # 使用调整后的训练集大小
        seed=RANDOM_SEED # 保证可重现性
    )
    
    # 获取隔离后的数据集
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    
    print(f"Final Train Dataset size: {len(train_dataset)}")
    print(f"Final Validation Dataset size: {len(val_dataset)}")


    # --- 关键步骤 2: 定义数据处理函数 ---

    def make_map_fn(split_name):
        def process_fn(example, idx):
            
            # 1. 提取原始的 'messages' 字段作为 prompt
            messages = example.pop('messages')
            
            # 2. 提取原始的 'ground_truth' 字段
            ground_truth = example.pop('ground_truth')
            
            # 3. 🎯 核心修改：从 messages 列表中提取最后一个用户的 'content'
            question_raw = ""
            if isinstance(messages, list) and messages:
                # 遍历 messages 列表，找到最后一个 role 为 'user' 的 content
                for message in reversed(messages):
                    if message.get('role') == 'user':
                        question_raw = message.get('content', '')
                        break
            
            # 确保 question 是字符串类型
            question = str(question_raw) 

            # 4. 移除所有不需要的列 (在提取内容后移除)
            for col in COLUMNS_TO_REMOVE_AFTER_EXTRACTION:
                example.pop(col, None) 
            
            # 5. 构造输出数据结构
            data = {
                # data_source 硬编码为目标字符串
                "data_source": data_source_value, 
                "prompt": [{
                    "role": "user",
                    "content": question, # <--- 最终使用的 prompt 内容
                }],
                "ability": "math", # 🎯 标记为数学能力
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth, # 🎯 提取 ground_truth
                },
                "extra_info": {
                    'split': split_name, # 标记是 train 还是 val
                    'index': idx,
                    "question_raw": question_raw,
                }
            }
            return data

        return process_fn
    
    # 最终保留的列
    final_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

    # --- 关键步骤 3: 处理和保存 Validation Set (500 条) ---
    print("\nProcessing Validation Set...")
    # 使用 remove_columns 清理 map 过程中可能残留的原始字段
    val_dataset = val_dataset.map(function=make_map_fn('val'), with_indices=True, remove_columns=val_dataset.column_names)
    
    # 显式移除 map 后剩余的未声明字段
    val_dataset = val_dataset.remove_columns([col for col in val_dataset.column_names if col not in final_columns])

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True) 
    output_val_path = os.path.join(local_dir, 'test_math.parquet') # 输出文件名为 test_math.parquet
    
    # 修正：移除 index=False 参数
    val_dataset.to_parquet(output_val_path)
    print(f"✅ Validation data saved to: {output_val_path} (Size: {len(val_dataset)})")


    # --- 关键步骤 4: 处理和保存 Training Set (7000 条) ---
    print("\nProcessing Training Set...")
    # 使用 remove_columns 清理 map 过程中可能残留的原始字段
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=train_dataset.column_names)

    # 显式移除 map 后剩余的未声明字段
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in final_columns])
    
    output_train_path = os.path.join(local_dir, 'train_math_partial.parquet')
    
    # 修正：移除 index=False 参数
    train_dataset.to_parquet(output_train_path)
    print(f"✅ Training data saved to: {output_train_path} (Size: {len(train_dataset)})")