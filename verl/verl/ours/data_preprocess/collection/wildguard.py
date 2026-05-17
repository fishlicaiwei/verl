import argparse
import os
import datasets
import json

# --- 配置信息 ---
# 待处理的本地 Parquet 文件路径
FILE_PATH = '/data2/jyyan10/verl-0.3.0.post1-ws/verl/verl/ours/data_preprocess.py/wildguard_test.parquet'
# 最终输出的本地目录
LOCAL_DIR = '/data2/cwli16/data/wildguard_processed_val'
# 最终输出的文件名
OUTPUT_FILENAME = 'test_general.parquet'
# 数据集的来源名称
DATA_SOURCE_VALUE = 'wildguard'
# 🎯 目标采样大小
VAL_SIZE_TARGET = 500 
RANDOM_SEED = 42

if __name__ == '__main__':
    
    print(f"尝试加载本地 Parquet 文件: {FILE_PATH}")
    
    if not os.path.exists(FILE_PATH):
        print(f"❌ 错误: 文件不存在于路径 '{FILE_PATH}'。请确保路径正确。")
        exit()

    try:
        # 使用 datasets 库加载本地 Parquet 文件
        raw_datasets = datasets.load_dataset(
            'parquet', 
            data_files={'val': FILE_PATH},
            split='val'
        )
        
    except Exception as e:
        print(f"❌ 读取 Parquet 文件时发生错误: {e}")
        exit()

    full_dataset = raw_datasets
    total_rows = len(full_dataset)
    print(f"✅ 文件加载成功。总行数: {total_rows}")

    # --- 关键步骤 1: 数据隔离与采样 (抽取 500 条) ---
    if total_rows > VAL_SIZE_TARGET:
        # 随机抽取 500 条作为验证集
        val_dataset = full_dataset.shuffle(seed=RANDOM_SEED).select(range(VAL_SIZE_TARGET))
        print(f"Sampling done. Final Validation Dataset size: {len(val_dataset)}")
    else:
        # 如果数据总量不足 500 条，则使用全部数据
        val_dataset = full_dataset
        print(f"⚠️ WARNING: Dataset size ({total_rows}) is less than required size ({VAL_SIZE_TARGET}). Using all available data.")


    # --- 关键步骤 2: 定义数据处理函数 ---

    def process_fn(example, idx):
        """
        处理数据样本，仅提取 'prompt' 字段并转换为目标格式。
        """
        # 仅提取 'prompt' 字段
        prompt_raw = example.get('prompt', "") 
        
        # 构造输出数据结构
        data = {
            "data_source": DATA_SOURCE_VALUE, 
            "prompt": [{
                "role": "user",
                "content": str(prompt_raw).strip(),
            }],
            "ability": "general", 
            "reward_model": {
                "style": "human_feedback", 
                "ground_truth": None # 按照要求，不设置 ground_truth
            },
            "extra_info": {
                'split': 'val', 
                'index': idx,
                "prompt_raw": prompt_raw,
            }
        }
        return data

    
    # 最终保留的列
    final_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

    # --- 关键步骤 3: 格式转换和保存 ---
    print("\nProcessing and Saving Dataset...")
    
    # 映射数据，并移除所有原始列
    val_dataset = val_dataset.map(
        function=process_fn, 
        with_indices=True, 
        remove_columns=val_dataset.column_names
    )
    
    # 显式移除 map 后剩余的未声明字段
    val_dataset = val_dataset.remove_columns([col for col in val_dataset.column_names if col not in final_columns])

    local_dir = LOCAL_DIR
    os.makedirs(local_dir, exist_ok=True) 
    output_val_path = os.path.join(local_dir, OUTPUT_FILENAME) 
    
    val_dataset.to_parquet(output_val_path)
    
    print("-" * 50)
    print(f"✅ 数据处理和转换完成。")
    print(f"新文件保存路径: {output_val_path}")
    print(f"最终记录数: {len(val_dataset)} (已采样到 {VAL_SIZE_TARGET} 或全部)")
    print("-" * 50)