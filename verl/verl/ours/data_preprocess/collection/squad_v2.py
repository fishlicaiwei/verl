import argparse
import os
import datasets
import json

# 定义需要从原始数据中移除的所有列的列表
# SQuAD v2 原始列包括：id, title, context, question, answers, is_impossible
COLUMNS_TO_REMOVE = [
    'id',
    'title',
    'context', # 修正后，这个字段现在用于构建 prompt，但最终仍应被移除
    'question', # 修正后，这个字段现在用于构建 prompt，但最终仍应被移除
    'answers',
    'is_impossible',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 🎯 目标目录改为 SQuAD 路径
    parser.add_argument('--local_dir', default='/data2/cwli16/data/squad_v2_processed_val')
    args = parser.parse_args()

    # 🎯 切换数据源
    data_source_repo = 'rajpurkar/squad_v2' 
    data_source_value = 'squad_v2_val'
    
    DATA_SPLIT = 'validation'
    
    print(f"Loading dataset: {data_source_repo} (Split: {DATA_SPLIT})...")
    try:
        val_dataset = datasets.load_dataset(data_source_repo, split=DATA_SPLIT)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    print(f"Initial Validation Dataset size: {len(val_dataset)}")


    # --- 关键步骤 1: 定义数据处理函数 (修正 prompt 逻辑) ---

    def make_map_fn(split_name):
        def process_fn(example, idx):
            
            # 1. 提取 context 和 question
            context_raw = example.get('context', "")
            question_raw = example.get('question', "")
            
            # 2. 组合 context 和 question 作为最终 prompt 内容
            # 使用换行符分隔，形成一个完整的问答输入
            final_prompt_content = f"{context_raw.strip()}\n\n{question_raw.strip()}"
            
            # 3. 提取 'answers' 字段作为 ground_truth
            answers = example.get('answers', {})
            ground_truth = None
            if answers and isinstance(answers, dict) and answers.get('text'):
                if answers['text']:
                     # 提取 answers['text'] 列表中的第一个元素作为 ground_truth
                     ground_truth = answers['text'][0] 
                
            # 4. 构造输出数据结构
            data = {
                "data_source": data_source_value, 
                "prompt": [{
                    "role": "user",
                    "content": final_prompt_content, # <--- 修正后的 prompt 内容
                }],
                "ability": "qa", 
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth 
                },
                "extra_info": {
                    'split': split_name, 
                    'index': idx,
                    "context_raw": context_raw,
                    "question_raw": question_raw,
                    "is_impossible": example.get('is_impossible', False),
                    "original_id": example.get('id', f"idx_{idx}"),
                }
            }
            return data

        return process_fn
    
    # 最终保留的列
    final_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

    # --- 关键步骤 2: 处理和保存 Validation Set ---
    print("\nProcessing Validation Set...")
    # 使用 remove_columns=val_dataset.column_names 来确保所有原始字段被移除
    val_dataset = val_dataset.map(
        function=make_map_fn('val'), 
        with_indices=True, 
        remove_columns=val_dataset.column_names # 移除所有原始列
    )
    
    # 显式检查并移除 map 后可能剩余的任何意外字段
    val_dataset = val_dataset.remove_columns([col for col in val_dataset.column_names if col not in final_columns])

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True) 
    output_val_path = os.path.join(local_dir, 'test_qa.parquet') 
    
    val_dataset.to_parquet(output_val_path)
    print(f"✅ Validation data saved to: {output_val_path} (Size: {len(val_dataset)})")
    print(f"字段结构检查: {val_dataset.column_names}")

    print("\n处理完成。")