# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# """
# Preprocess the wildchat-ultrafeedback dataset to parquet format
# """

import argparse
import os
import datasets
import json

# 定义需要从原始数据中移除的所有列的列表
# 我们将移除所有原始列，除了 'prompt'，因为它需要被提取到新的结构中。
COLUMNS_TO_REMOVE = [
    # 原始 SFT/RLHF 数据集常见的残留列
    'id', 
    'messages', # Wildchat/Tulu-3 经常包含完整的对话历史，需要移除
    'data_source', # 原始 data_source 会被移除，因为我们要用新的值覆盖它
    'solution', 
    'ability', 
    'reward_model', 
    'extra_info', 
    '__index_level_0__',
    # WildChat/Tulu-3 特有或可能存在的其他列 (确保全面清理)
    'response', 
    'source', 
    'conversation',
    'rating',
    'input', # 有些数据集 prompt 字段可能叫 input
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 🎯 目标目录改为 WildChat 路径
    parser.add_argument('--local_dir', default='/data2/cwli16/data/tulu-3-wildchat-sampled') 
    args = parser.parse_args()

    # 🎯 切换数据源
    data_source_repo = 'allenai/tulu-3-wildchat-if-on-policy-8b' 
    # 🎯 切换 data_source 最终值
    data_source_value = 'tulu-3-wildchat-if-on-policy-8b'
    
    TRAIN_SIZE = 10000
    VAL_SIZE = 500
    RANDOM_SEED = 42
    
    print(f"Loading dataset: {data_source_repo} (Split: train)...")
    try:
        # 加载数据集
        full_dataset = datasets.load_dataset(data_source_repo, split='train')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    # --- 关键步骤 1: 数据隔离与分割 ---
    
    required_size = TRAIN_SIZE + VAL_SIZE
    if len(full_dataset) < required_size:
        print(f"❌ ERROR: Dataset size ({len(full_dataset)}) is less than required size ({required_size}). Aborting.")
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

            # 2. 移除所有不需要的列 (在提取 prompt 之后进行)
            for col in COLUMNS_TO_REMOVE:
                example.pop(col, None) 
            
            # 3. 构造输出数据结构
            data = {
                # 🎯 关键修改 A: data_source 硬编码为目标字符串
                "data_source": data_source_value, 
                "prompt": [{
                    "role": "user",
                    "content": question, # <--- 最终使用的 prompt 内容
                }],
                "ability": "chat", # 🎯 标记为聊天能力
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
    output_val_path = os.path.join(local_dir, 'test_chat.parquet') # 输出文件名为 test_chat.parquet
    
    # 修正：移除 index=False 参数 (使用 datasets 库的正确方法)
    val_dataset.to_parquet(output_val_path)
    print(f"✅ Validation data saved to: {output_val_path} (Size: {len(val_dataset)})")


    # --- 关键步骤 4: 处理和保存 Training Set (10000 条) ---
    print("\nProcessing Training Set...")
    # 使用 remove_columns 清理 map 过程中可能残留的原始字段
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=train_dataset.column_names)

    # 显式移除 map 后剩余的未声明字段
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in final_columns])
    
    output_train_path = os.path.join(local_dir, 'train_chat_partial.parquet')
    
    # 修正：移除 index=False 参数
    train_dataset.to_parquet(output_train_path)
    print(f"✅ Training data saved to: {output_train_path} (Size: {len(train_dataset)})")