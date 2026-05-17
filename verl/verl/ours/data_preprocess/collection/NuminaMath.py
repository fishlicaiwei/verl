import argparse
import os
import datasets
import json

# 定义需要从原始数据中移除的所有列的列表
COLUMNS_TO_REMOVE_AFTER_EXTRACTION = [
    'problem',      
    'solution',     
    'messages',     
    'source',       
    'reasoning',    
    'id',
    'data_source',
    '__index_level_0__',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/data2/cwli16/data/NuminaMath-sampled') 
    args = parser.parse_args()

    data_source_repo = 'AI-MO/NuminaMath-CoT' 
    data_source_value = 'NuminaMath-CoT'
    
    TRAIN_SIZE = 10000  
    VAL_SIZE = 500     
    # 为了保证安全，我们提取稍微多一点的数据，比如 10000 条，然后再做切分
    PRE_LOAD_SIZE = 12000 
    RANDOM_SEED = 42
    
    print(f"Loading dataset: {data_source_repo} (Streaming mode)...")
    
    try:
        # 🎯 核心修改 1: 启用 streaming=True
        # 这样不会下载整个数据集，而是建立一个流式连接
        iterable_dataset = datasets.load_dataset(data_source_repo, split='train', streaming=True)
        
        print(f"Stream established. Extracting first {PRE_LOAD_SIZE} examples...")
        
        # 🎯 核心修改 2: 只取前 N 条数据
        # take() 会返回一个迭代器，我们需要把它转换成列表，再转回 Dataset 对象
        # 这样只下载大约几十 MB 的数据就会停止
        data_list = list(iterable_dataset.take(PRE_LOAD_SIZE))
        
        # 将列表转换回标准的 HuggingFace Dataset 对象
        # 这样后面的 train_test_split 和 map 也就是兼容的了
        full_dataset = datasets.Dataset.from_list(data_list)
        
        print(f"Successfully loaded {len(full_dataset)} examples from stream.")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        # 如果流式失败，可能是网络完全不通，建议检查 HF_ENDPOINT
        exit()

    # --- 后面的代码几乎不用动，逻辑保持一致 ---

    # --- 关键步骤 1: 数据隔离与分割 ---
    
    required_size = TRAIN_SIZE + VAL_SIZE
    
    if len(full_dataset) < required_size:
        print(f"⚠️ WARNING: Dataset size ({len(full_dataset)}) is less than required size ({required_size}).")
        TRAIN_SIZE = len(full_dataset) - VAL_SIZE
    
    print(f"Splitting dataset: {TRAIN_SIZE} for Train, {VAL_SIZE} for Validation...")
    
    # 此时 full_dataset 已经是内存中的普通 Dataset 了，可以直接 split
    split_dataset = full_dataset.train_test_split(
        test_size=VAL_SIZE, 
        train_size=TRAIN_SIZE, 
        seed=RANDOM_SEED
    )
    
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    
    print(f"Final Train Dataset size: {len(train_dataset)}")
    print(f"Final Validation Dataset size: {len(val_dataset)}")


    # --- 关键步骤 2: 定义数据处理函数 (保持不变) ---

    def make_map_fn(split_name):
        def process_fn(example, idx):
            
            question_raw = example.get('problem', '')
            if question_raw is None: question_raw = ""
            
            ground_truth = example.get('solution', '')
            if ground_truth is None: ground_truth = ""

            for col in COLUMNS_TO_REMOVE_AFTER_EXTRACTION:
                example.pop(col, None) 
            
            data = {
                "data_source": data_source_value, 
                "prompt": [{
                    "role": "user",
                    "content": str(question_raw),
                }],
                "ability": "math", 
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(ground_truth),
                    "solution": str(solution),
                },
                "extra_info": {
                    'split': split_name,
                    'index': idx,
                    "question_raw": str(question_raw),
                }
            }
            return data

        return process_fn
    
    final_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

    # --- 关键步骤 3: 处理和保存 Validation Set ---
    print("\nProcessing Validation Set...")
    val_dataset = val_dataset.map(
        function=make_map_fn('val'), 
        with_indices=True, 
        remove_columns=val_dataset.column_names
    )
    val_dataset = val_dataset.remove_columns([col for col in val_dataset.column_names if col not in final_columns])

    os.makedirs(args.local_dir, exist_ok=True) 
    output_val_path = os.path.join(args.local_dir, 'test_numina.parquet') 
    
    val_dataset.to_parquet(output_val_path)
    print(f"✅ Validation data saved to: {output_val_path} (Size: {len(val_dataset)})")


    # --- 关键步骤 4: 处理和保存 Training Set ---
    print("\nProcessing Training Set...")
    train_dataset = train_dataset.map(
        function=make_map_fn('train'), 
        with_indices=True, 
        remove_columns=train_dataset.column_names
    )

    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in final_columns])
    
    output_train_path = os.path.join(args.local_dir, 'train_numina_partial.parquet')
    
    train_dataset.to_parquet(output_train_path)
    print(f"✅ Training data saved to: {output_train_path} (Size: {len(train_dataset)})")