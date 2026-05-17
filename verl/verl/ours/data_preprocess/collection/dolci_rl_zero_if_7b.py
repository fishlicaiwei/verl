import argparse
import os
import datasets
import json

# 定义需要从原始数据中移除的所有列的列表
# 目标是移除所有原始列，只保留最终需要的结构字段。
COLUMNS_TO_REMOVE = [
    'metadata', 
    'is_safety',
    'score',
    'prompt', # 在提取内容后移除
    # 保持列表形式，与你提供的 RLVR 脚本结构一致
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 🎯 目标目录改为 Dolci 路径 (使用 /data2 路径以匹配你的环境风格)
    parser.add_argument('--local_dir', default='/data2/cwli16/data/dolci-sampled-val') 
    args = parser.parse_args()

    # 🎯 数据集信息
    data_source_repo = 'allenai/Dolci-RL-Zero-IF-7B' 
    config_name = 'default' # 默认配置
    data_source_value = 'dolci_rl_zero_if_7b'
    
    # 📝 目标验证集大小。我们抽取 500 条作为示例
    VAL_SIZE_TARGET = 500
    RANDOM_SEED = 42
    
    # 定义要去除的前缀
    USER_PREFIX = "user: "
    
    # --- 加载数据集：指定分割 ---
    print(f"Loading dataset: {data_source_repo} (Config: {config_name}, Split: train)...")
    try:
        # Dolci-RL-Zero-IF-7B 的主分割是 'train'
        full_dataset = datasets.load_dataset(data_source_repo, config_name, split='train')
    except Exception as e:
        print(f"❌ Error loading {data_source_repo}: {e}")
        exit()


    # --- 关键步骤 1: 数据抽样 (仅抽取 500 条验证集) ---
    
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
            
            # 1. 提取原始的 'prompt' 字段内容
            prompt_raw = example.get('prompt', "")
            question = str(prompt_raw).strip() 
            
            # 2. 移除开头的 "user: " 字符串（大小写不敏感）
            if question.lower().startswith(USER_PREFIX):
                # 只移除匹配的前缀部分
                question = question[len(USER_PREFIX):].strip()

            # 3. 移除所有不需要的列 
            for col in COLUMNS_TO_REMOVE:
                example.pop(col, None) 
            
            # 4. 构造输出数据结构
            data = {
                "data_source": data_source_value, 
                "prompt": [{
                    "role": "user",
                    "content": question, 
                }],
       
                "ability": "instruction_following", 
                "reward_model": {
                    "style": "rule",
                    "ground_truth": None 
                },
                "extra_info": {
                    'split': split_name, 
                    'index': idx,
                }
            }

            return data

        return process_fn
    
    # 最终保留的列 (与你的 RLVR 脚本保持一致)
    final_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

    # --- 关键步骤 3: 处理和保存 Validation Set ---
    print("\nProcessing Validation Set...")
    # 使用 remove_columns=val_dataset.column_names 来移除所有原始列
    val_dataset = val_dataset.map(function=make_map_fn('val'), with_indices=True, remove_columns=val_dataset.column_names)
    
    # 显式移除 map 后剩余的未声明字段
    val_dataset = val_dataset.remove_columns([col for col in val_dataset.column_names if col not in final_columns])

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True) 
    # 输出文件名为 test_dolci.parquet
    output_val_path = os.path.join(local_dir, 'test_dolci.parquet') 
    
    val_dataset.to_parquet(output_val_path)
    print(f"✅ Validation data saved to: {output_val_path} (Size: {len(val_dataset)})")