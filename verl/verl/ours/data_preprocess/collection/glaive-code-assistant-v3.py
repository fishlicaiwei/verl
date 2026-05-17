import argparse
import os
import datasets
import json

# 定义需要从原始数据中移除的所有列的列表
# 针对 Glaive Code Assistant 的常见列
COLUMNS_TO_REMOVE_AFTER_EXTRACTION = [
    'question',     # 提取后移除
    'answer',       # 提取后移除
    'id',           # 如果存在
    'source',       # 如果存在
    'type',         # 如果存在
    'data_source',
    '__index_level_0__',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 🎯 目标目录改为 Glaive Code 路径
    parser.add_argument('--local_dir', default='/data2/cwli16/data/glaive-code-sampled') 
    args = parser.parse_args()

    # 🎯 切换数据源
    data_source_repo = 'glaiveai/glaive-code-assistant-v3' 
    # 🎯 切换 data_source 最终值
    data_source_value = 'glaive-code-assistant-v3'
    
    TRAIN_SIZE = 10000  # 训练集目标大小
    VAL_SIZE = 500     # 验证集目标大小
    
    # 预加载数量：为了安全起见，从流中读取 10000 条，然后从中切分
    PRE_LOAD_SIZE = 10000 
    RANDOM_SEED = 42
    
    print(f"Loading dataset: {data_source_repo} (Streaming mode)...")
    
    try:
        # 1. 建立流式连接 (不下载全量数据)
        iterable_dataset = datasets.load_dataset(data_source_repo, split='train', streaming=False)
        
        print(f"Stream established. Extracting first {PRE_LOAD_SIZE} examples...")
        
        # 2. 获取前 N 条数据并转为列表
        # .take() 也是 lazy 的，只有转换为 list 时才会真正触发网络下载
        data_list = list(iterable_dataset.take(PRE_LOAD_SIZE))
        
        # 3. 转换为内存中的 Dataset 对象，方便后续 split 和 map
        full_dataset = datasets.Dataset.from_list(data_list)
        
        print(f"Successfully loaded {len(full_dataset)} examples from stream.")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    # --- 关键步骤 1: 数据隔离与分割 ---
    
    required_size = TRAIN_SIZE + VAL_SIZE
    
    if len(full_dataset) < required_size:
        print(f"⚠️ WARNING: Dataset size ({len(full_dataset)}) is less than required size ({required_size}).")
        TRAIN_SIZE = len(full_dataset) - VAL_SIZE
        if TRAIN_SIZE < 0: TRAIN_SIZE = 0
    
    print(f"Splitting dataset: {TRAIN_SIZE} for Train, {VAL_SIZE} for Validation...")
    
    split_dataset = full_dataset.train_test_split(
        test_size=VAL_SIZE, 
        train_size=TRAIN_SIZE, 
        seed=RANDOM_SEED
    )
    
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    
    print(f"Final Train Dataset size: {len(train_dataset)}")
    print(f"Final Validation Dataset size: {len(val_dataset)}")


    # --- 关键步骤 2: 定义数据处理函数 ---

    def make_map_fn(split_name):
        def process_fn(example, idx):
            
            # 1. 🎯 核心修改：提取 'question'
            question_raw = example.get('question', '')
            if question_raw is None: question_raw = ""
            
            # 2. 🎯 核心修改：提取 'answer'
            ground_truth = example.get('answer', '')
            if ground_truth is None: ground_truth = ""

            # 3. 移除旧列
            for col in COLUMNS_TO_REMOVE_AFTER_EXTRACTION:
                example.pop(col, None) 
            
            # 4. 构造输出数据结构
            data = {
                "data_source": data_source_value, 
                "prompt": [{
                    "role": "user",
                    "content": str(question_raw), # <--- question 放这里
                }],
                # 🎯 修改 Ability 为 code
                "ability": "code", 
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(ground_truth), # <--- answer 放这里
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
    # 输出文件名改为 test_code.parquet
    output_val_path = os.path.join(args.local_dir, 'test_code.parquet') 
    
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
    
    # 输出文件名改为 train_code_partial.parquet
    output_train_path = os.path.join(args.local_dir, 'train_code_partial.parquet')
    
    train_dataset.to_parquet(output_train_path)
    print(f"✅ Training data saved to: {output_train_path} (Size: {len(train_dataset)})")