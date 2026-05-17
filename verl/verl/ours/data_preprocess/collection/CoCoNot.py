import argparse
import os
import datasets
import json

# 定义需要从原始数据中移除的所有列的列表
# 目标是移除所有原始列，只保留最终需要的结构字段。
COLUMNS_TO_REMOVE = [
    # CoCoNot 原始字段
    'id', 
    'messages', 
    'prompt', # 在提取内容后移除
    'grounded_answer', 
    'context', 
    'safety_category', 
    'compliant', 
    'reason', 
    'split', 
    # 通用字段
    'data_source', 'solution', 'ability', 'reward_model', 'extra_info', '__index_level_0__',
    'response', 'source', 'conversation', 'rating', 'input',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 🎯 目标目录改为 CoCoNot 验证集路径
    parser.add_argument('--local_dir', default='/data2/cwli16/data/coconot-sampled-val') 
    args = parser.parse_args()

    # 🎯 切换数据源
    data_source_repo = 'allenai/coconot' 
    # 🎯 切换子集/配置名称
    config_name = 'original'
    # 🎯 切换 data_source 最终值
    data_source_value = f'coconot'
    
    # 📝 目标验证集大小。CoCoNot Original Eval set 约为 1000 条，但我们只取 500 条
    VAL_SIZE_TARGET = 500
    RANDOM_SEED = 42
    
    # --- 加载数据集：指定子集和分割 ---
    print(f"Loading dataset: {data_source_repo} (Config: {config_name}, Split: test)...")
    try:
        # 使用 load_dataset 时，第二个参数是配置名称 (Subset)
        full_dataset = datasets.load_dataset(data_source_repo, config_name, split='test')
    except Exception as e:
        print(f"❌ Error loading CoCoNot with specified config/split: {e}")
        exit()


    # --- 关键步骤 1: 数据隔离与抽样 (仅抽取 500 条验证集) ---
    
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
            
            # 1. 提取原始的 'prompt' 字段作为 prompt
            # CoCoNot Original Set 确实包含一个明确的 'prompt' 字段
            question_raw = example.pop('prompt', "") 
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
                # 🎯 标记为安全/不服从评估能力
                "ability": "safety_noncompliance", 
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
    val_dataset = val_dataset.map(function=make_map_fn('val'), with_indices=True, remove_columns=val_dataset.column_names)
    
    # 显式移除 map 后剩余的未声明字段
    val_dataset = val_dataset.remove_columns([col for col in val_dataset.column_names if col not in final_columns])

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True) 
    # 输出文件名为 test_safety.parquet，反映其用途
    output_val_path = os.path.join(local_dir, 'test_safety.parquet') 
    
    val_dataset.to_parquet(output_val_path)
    print(f"✅ Validation data saved to: {output_val_path} (Size: {len(val_dataset)})")

    # 📝 跳过处理和保存 Training Set 的步骤
    print("\nTraining Set processing skipped as requested.")