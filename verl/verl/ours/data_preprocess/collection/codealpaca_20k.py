import argparse
import os
import datasets
import json

# 定义需要从原始数据中移除的所有列的列表
# CodeAlpaca_20K 原始列包括：prompt, completion
COLUMNS_TO_REMOVE = [
    'prompt', 
    'completion', 
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 🎯 目标目录改为 CodeAlpaca_20K 路径
    parser.add_argument('--local_dir', default='/data2/cwli16/data/codealpaca_processed_val')
    args = parser.parse_args()

    # 🎯 切换数据源
    data_source_repo = 'HuggingFaceH4/CodeAlpaca_20K' 
    # 🎯 切换 data_source 最终值
    data_source_value = 'codealpaca_20k'
    
    # CodeAlpaca_20K 通常只包含 'train' 分割
    DATA_SPLIT = 'train'
    VAL_SIZE_TARGET = 500 # 沿用之前的验证集大小标准
    RANDOM_SEED = 42
    
    # 📝 只需要加载一次数据集
    print(f"Loading dataset: {data_source_repo} (Split: {DATA_SPLIT})...")
    try:
        # 加载整个数据集
        full_dataset = datasets.load_dataset(data_source_repo, split=DATA_SPLIT)
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

    # 通用指令作为新的 prompt
    GENERAL_CODE_PROMPT = "请阅读以下代码指令，并给出最佳的实现（仅代码）："

    def make_map_fn(split_name):
        def process_fn(example, idx):
            
            # 1. 提取原始的 'prompt' 字段作为 answer
            original_prompt = example.get('prompt', "") 
            
            # 2. 提取 'completion' 字段作为 ground_truth
            ground_truth_code = example.get('completion')
            
            # 3. 构造输出数据结构
            data = {
                "data_source": data_source_value, 
                "prompt": [{
                    "role": "user",
                    "content": GENERAL_CODE_PROMPT, # <--- 通用指令作为新的 prompt
                }],
                # 原始 prompt (指令) 成为模型的回答 (answer)
                "answer": original_prompt, 
                "ability": "code", # 代码生成任务使用 'code'
                "reward_model": {
                    "style": "rule",
                    # 原始 completion (代码) 成为 ground_truth
                    "ground_truth": str(ground_truth_code) if ground_truth_code is not None else None
                },
                "extra_info": {
                    'split': split_name, 
                    'index': idx,
                    "original_prompt_raw": original_prompt,
                }
            }
            return data

        return process_fn
    
    # 最终保留的列
    # 注意这里增加了 'answer' 字段
    final_columns = ["data_source", "prompt", "answer", "ability", "reward_model", "extra_info"]

    # --- 关键步骤 3: 处理和保存 Validation Set ---
    print("\nProcessing Validation Set...")
    
    # 使用 remove_columns 清理 map 过程中可能残留的原始字段
    val_dataset = val_dataset.map(
        function=make_map_fn('val'), 
        with_indices=True, 
        remove_columns=val_dataset.column_names # 移除所有原始列
    )
    
    # 显式移除 map 后剩余的未声明字段
    val_dataset = val_dataset.remove_columns([col for col in val_dataset.column_names if col not in final_columns])

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True) 
    # 输出文件名为 test_code.parquet (因为是代码任务)
    output_val_path = os.path.join(local_dir, 'test_code.parquet') 
    
    val_dataset.to_parquet(output_val_path)
    print(f"✅ Validation data saved to: {output_val_path} (Size: {len(val_dataset)})")
    print(f"字段结构检查: {val_dataset.column_names}")

    print("\n处理完成。")