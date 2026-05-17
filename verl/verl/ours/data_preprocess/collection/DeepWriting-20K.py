import argparse
import os
import datasets
import json

# 定义需要从原始数据中移除的所有列的列表
# 我们将移除所有原始列，因为只需要 'prompt' 的内容来构建新的标准结构。
COLUMNS_TO_REMOVE = [
    # 原始 DeepWriting-20K 常见的字段
    'id', 
    'messages',
    # 额外移除我们在目标格式中不需要的字段
    'data_source', 
    'solution', 
    'ability', 
    'reward_model', 
    'extra_info', 
    '__index_level_0__',
    # DeepWriting-20K 可能会有的字段
    'instruction', 
    'output', 
    'response',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 🎯 目标目录改为 DeepWriting 路径
    parser.add_argument('--local_dir', default='/data2/cwli16/data/m-a-p-deepwriting-sampled') 
    args = parser.parse_args()

    # 🎯 切换数据源
    data_source_repo = 'm-a-p/DeepWriting-20K' 
    # 🎯 切换 data_source 最终值
    data_source_value = 'DeepWriting-20K'
    
    TRAIN_SIZE = 10000
    VAL_SIZE = 500
    RANDOM_SEED = 42
    
    print(f"Loading dataset: {data_source_repo} (Split: train)...")
    try:
        # DeepWriting-20K 只有一个 'train' 分割
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
            prompt_raw = example.pop('prompt') # 使用 prompt_raw 来存储原始数据
            
            # 🎯 关键修复 A: 初始化 question_raw
            question_raw = ""
            prompt_list = None
            
            try:
                # 检查类型：如果已经是列表/字典，则无需解析
                if isinstance(prompt_raw, (list, dict)):
                    prompt_list = prompt_raw # 直接赋值
                elif isinstance(prompt_raw, str):
                    # 如果是字符串，则尝试解析 JSON
                    prompt_list = json.loads(prompt_raw) 
                else:
                    # 无法识别的类型，使用原始数据并跳过内容提取
                    question_raw = str(prompt_raw)
                    
                # 提取内容 (只有在成功获得 prompt_list 时才执行)
                if prompt_list and isinstance(prompt_list, list) and len(prompt_list) > 0:
                    # 提取第一个元素的 'content' 作为最终的指令
                    question_raw = prompt_list[0]['content'] 
                elif not question_raw:
                    # 如果 prompt_list 存在但不是预期的格式，或者为空，使用默认值
                    question_raw = str(prompt_raw)

            except Exception as e:
                # 降级为使用原始数据的字符串表示，并打印警告
                # print(f"Warning: Failed to process prompt for index {idx}. Error: {e}") 
                question_raw = str(prompt_raw)
                
            # 确保 question 是字符串类型
            question = str(question_raw) 

            # 2. 移除所有不需要的列 (在提取 prompt 之后进行)
            for col in COLUMNS_TO_REMOVE:
                example.pop(col, None) 
            
            # 3. 构造输出数据结构
            data = {
                # data_source 硬编码为目标字符串
                "data_source": data_source_value, 
                "prompt": [{
                    "role": "user",
                    "content": question, 
                }],
                "ability": "writing", # 标记为写作能力
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
    output_val_path = os.path.join(local_dir, 'test_writing.parquet') # 输出文件名为 test_writing.parquet
    
    # 🎯 最终修复 B: 移除 index=False 参数
    val_dataset.to_parquet(output_val_path)
    print(f"✅ Validation data saved to: {output_val_path} (Size: {len(val_dataset)})")


    # --- 关键步骤 4: 处理和保存 Training Set (10000 条) ---
    print("\nProcessing Training Set...")
    # 使用 remove_columns 清理 map 过程中可能残留的原始字段
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=train_dataset.column_names)

    # 显式移除 map 后剩余的未声明字段
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in final_columns])
    
    output_train_path = os.path.join(local_dir, 'train_writing_partial.parquet')
    
    # 🎯 最终修复 B: 移除 index=False 参数
    train_dataset.to_parquet(output_train_path)
    print(f"✅ Training data saved to: {output_train_path} (Size: {len(train_dataset)})")