import argparse
import os
import datasets
import json

# 可移除字段
COLUMNS_TO_REMOVE = [
    'title',
    'context',
    'all_answers',
    'metadata',
    'annotations',
]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/data2/cwli16/data/nq-sampled-val')
    args = parser.parse_args()

    # 数据集信息
    data_source_repo = 'sentence-transformers/natural-questions'
    config_name = None
    data_source_value = 'natural_questions'

    VAL_SIZE_TARGET = 500
    RANDOM_SEED = 42

    # ------- 加载数据 -------
    print(f"Loading dataset: {data_source_repo} (Split: train)...")
    try:
        full_dataset = datasets.load_dataset(data_source_repo, split='train')
    except Exception as e:
        print(f"❌ Error loading {data_source_repo}: {e}")
        exit()

    current_size = len(full_dataset)

    if current_size < VAL_SIZE_TARGET:
        print(f"⚠️ Dataset smaller than target ({VAL_SIZE_TARGET}), using full set.")
        VAL_SIZE = current_size
        val_dataset = full_dataset
    else:
        VAL_SIZE = VAL_SIZE_TARGET
        val_dataset = full_dataset.shuffle(seed=RANDOM_SEED).select(range(VAL_SIZE_TARGET))

    print(f"Final Validation Dataset size: {len(val_dataset)} (Target: {VAL_SIZE_TARGET})")

    # ------- 数据处理函数 -------
    def make_map_fn(split_name):
        def process_fn(example, idx):

            # 1. 取 query 作为 prompt 内容
            question = str(example.get('query', "")).strip()

            # 2. 直接取 answer（字符串）
            answer_text = str(example.get('answer', "")).strip()

            # 3. 移除不必要字段
            for col in COLUMNS_TO_REMOVE:
                example.pop(col, None)

            # 4. 构造 RLVR 统一结构
            data = {
                "data_source": data_source_value,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "instruction_following",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer_text,  
                },
                "extra_info": {
                    "split": split_name,
                    "index": idx,
                },
            }

            return data

        return process_fn

    # 保留的最终列
    final_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

    # ------- 执行 map -------
    print("\nProcessing Validation Set...")
    val_dataset = val_dataset.map(
        function=make_map_fn('val'),
        with_indices=True,
        remove_columns=val_dataset.column_names,
    )

    val_dataset = val_dataset.remove_columns([c for c in val_dataset.column_names if c not in final_columns])

    # ------- 保存 parquet -------
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    output_val_path = os.path.join(local_dir, 'test_nq.parquet')
    val_dataset.to_parquet(output_val_path)

    print(f"✅ Validation data saved to: {output_val_path} (Size: {len(val_dataset)})")
