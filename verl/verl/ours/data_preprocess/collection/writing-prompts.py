import argparse
import os
import datasets
import json
import random

# 定义需要从原始数据中移除的所有列的列表
COLUMNS_TO_REMOVE = [
    'prompt', 
    'story', 
]

# 🎯 英文通用写作指令：更简洁，且符合用户的要求
WRITING_INSTRUCTION = "Write a compelling and coherent story based on the prompt below. Start the narration directly:"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 🎯 目标目录改为 llm-aes/writing-prompts 路径
    parser.add_argument('--local_dir', default='./llmaes_writingprompts_processed_val')
    args = parser.parse_args()

    # --- 配置信息 ---
    data_source_repo = 'llm-aes/writing-prompts' 
    data_source_value = 'llmaes_writingprompts_val'
    DATA_SPLIT = 'train'
    VAL_SIZE_TARGET = 500
    RANDOM_SEED = 42
    
    # 📝 只需要加载一次数据集
    print(f"Loading dataset: {data_source_repo} (Split: {DATA_SPLIT})...")
    try:
        full_dataset = datasets.load_dataset(data_source_repo, split=DATA_SPLIT)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    # --- 关键步骤 1: 数据隔离与分割 (仅抽取验证集) ---
    current_size = len(full_dataset)
    
    if current_size < VAL_SIZE_TARGET:
        val_dataset = full_dataset
    else:
        val_dataset = full_dataset.shuffle(seed=RANDOM_SEED).select(range(VAL_SIZE_TARGET))
    
    print(f"Final Validation Dataset size: {len(val_dataset)}")

    # --- 关键步骤 2: 定义数据处理函数 ---

    def make_map_fn(split_name):
        def process_fn(example, idx):
            
            prompt_raw = example.get('prompt', "") 
            story_raw = example.get('story')
            
            # 🎯 组合：将固定的通用指令和变化的原始提示拼接起来
            # 注意：这里将原始提示放在新的一行，以保持清晰的分隔
            final_prompt_content = f"{WRITING_INSTRUCTION}\n\n{str(prompt_raw).strip()}"
            
            # 构造输出数据结构
            data = {
                "data_source": data_source_value, 
                "prompt": [{
                    "role": "user",
                    "content": final_prompt_content, # <--- 组合后的英文 Prompt
                }],
                "ability": "creative", 
                "reward_model": {
                    "style": "human_feedback",
                    "ground_truth": str(story_raw) if story_raw is not None else None
                },
                "extra_info": {
                    'split': split_name, 
                    'index': idx,
                    "prompt_raw": prompt_raw,
                }
            }
            return data

        return process_fn
    
    # 最终保留的列
    final_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

    # --- 关键步骤 3: 处理和保存 Validation Set ---
    print("\nProcessing Validation Set...")
    
    val_dataset = val_dataset.map(
        function=make_map_fn('val'), 
        with_indices=True, 
        remove_columns=val_dataset.column_names
    )
    
    val_dataset = val_dataset.remove_columns([col for col in val_dataset.column_names if col not in final_columns])

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True) 
    output_val_path = os.path.join(local_dir, 'test_creative.parquet') 
    
    val_dataset.to_parquet(output_val_path)
    print(f"✅ Validation data saved to: {output_val_path} (Size: {len(val_dataset)})")
    print(f"字段结构检查: {val_dataset.column_names}")

    print("\n数据集处理完成。")