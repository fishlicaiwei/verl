import argparse
import os
import datasets
import json

# 定义需要从原始数据中移除的所有列的列表
# 目标是移除所有原始列，只保留最终需要的结构字段。
COLUMNS_TO_REMOVE = [
    # 原始 SFT/RLHF 数据集常见的残留列
    'id', 
    'messages', 
    'data_source', 
    'solution', 
    'ability', 
    'reward_model', 
    'extra_info', 
    '__index_level_0__',
    # Tulu-3-Ultrafeedback 可能包含的列
    'response', 
    'source', 
    'conversation',
    'rating',
    'input',
    'chosen', 
    'rejected', 
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 🎯 目标目录改为 Ultrafeedback 路径
    parser.add_argument('--local_dir', default='/data2/cwli16/data/tulu-3-ultrafeedback-sampled-val') 
    args = parser.parse_args()

    # 🎯 切换数据源
    data_source_repo = 'allenai/tulu-3-ultrafeedback-cleaned-on-policy-8b' 
    # 🎯 切换 data_source 最终值
    data_source_value = 'tulu-3-ultrafeedback-cleaned-on-policy-8b'
    
    VAL_SIZE_TARGET = 500
    RANDOM_SEED = 42
    
    # 📝 只需要加载一次数据集
    print(f"Loading dataset: {data_source_repo} (Split: train)...")
    try:
        # Tulu-3 数据集通常只有一个 'train' 分割
        full_dataset = datasets.load_dataset(data_source_repo, split='train')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    # --- 关键步骤 1: 数据隔离与分割 (仅抽取验证集) ---
    
    current_size = len(full_dataset)
    
    if current_size < VAL_SIZE_TARGET:
        print(f"⚠️ WARNING: Dataset size ({current_size}) is less than required validation size ({VAL_SIZE_TARGET}). Using all available data.")
        VAL_SIZE = current_size
        val_dataset = full_dataset
    else:
        VAL_SIZE = VAL_SIZE_TARGET
        # 随机抽取 500 条作为验证集
        val_dataset = full_dataset.shuffle(seed=RANDOM_SEED).select(range(VAL_SIZE_TARGET))
    
    print(f"Final Validation Dataset size: {len(val_dataset)} (Target: {VAL_SIZE_TARGET})")


    # --- 关键步骤 2: 定义数据处理函数 ---

    def make_map_fn(split_name):
        def process_fn(example, idx):
            
            # 1. 提取原始的 'prompt' 字段
            # 注意：如果 'prompt' 字段不存在，需要根据实际列名调整 (如 'messages' 或 'input')
            question_raw = example.pop('prompt', None) 
            
            if question_raw is None:
                # 尝试从 'messages' 中提取用户消息（如果数据集结构是对话格式）
                messages = example.get('messages')
                if isinstance(messages, list) and messages and messages[0]['role'] == 'user':
                    question_raw = messages[0]['content']
                else:
                    # 最终降级：使用一个空字符串或警告
                    question_raw = ""

            # 确保 question 是字符串类型
            question = str(question_raw) 

            # 2. 移除所有不需要的列 
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
                    "ground_truth": None # 验证集不要求 Gold Truth
                },
                "extra_info": {
                    'split': split_name, 
                    'index': idx,
                    "question_raw": question_raw,
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
    output_val_path = os.path.join(local_dir, 'test_chat.parquet') # 输出文件名为 test_chat.parquet
    
    val_dataset.to_parquet(output_val_path)
    print(f"✅ Validation data saved to: {output_val_path} (Size: {len(val_dataset)})")

    # 📝 注意：这里省略了处理和保存 Training Set 的步骤
    print("\nTraining Set processing skipped as requested.")