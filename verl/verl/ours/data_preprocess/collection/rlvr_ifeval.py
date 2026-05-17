import argparse
import os
import datasets
import json

# 定义需要从原始数据中移除的所有列的列表
# 目标是移除所有原始列，只保留最终需要的结构字段。
COLUMNS_TO_REMOVE = [
    'task_id', 
    'messages', # 在提取内容后移除
    'solution', 
    'source', 
    'difficulty', 
    'domain', 
    'type', 
    'topic', 
    'instruction_group', 
    'reference_answer', 
    'reward_model',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 🎯 目标目录改为 RLVR-IFeval 路径
    parser.add_argument('--local_dir', default='/data2/cwli16/data/rlvr-ifeval-sampled-val') 
    args = parser.parse_args()

    # 🎯 数据集信息
    data_source_repo = 'allenai/RLVR-IFeval' 
    config_name = 'default' # RLVR-IFeval 默认配置
    data_source_value = 'rlvr_ifeval'
    
    # 📝 目标验证集大小。RLVR-IFeval 数据集较大，我们抽取 500 条作为示例
    VAL_SIZE_TARGET = 500
    RANDOM_SEED = 42
    
    # --- 加载数据集：指定分割 ---
    print(f"Loading dataset: {data_source_repo} (Config: {config_name}, Split: train)...")
    try:
        # RLVR-IFeval 的主分割是 'train'
        full_dataset = datasets.load_dataset(data_source_repo, config_name, split='train')
    except Exception as e:
        print(f"❌ Error loading RLVR-IFeval: {e}")
        exit()


    # --- 关键步骤 1: 数据抽样 (仅抽取 500 条验证集) ---
    
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
            
            # 1. 提取 'messages' 字段中的用户 'content' 作为 prompt
            messages_raw = example.get('messages', [])
            question = ""
            if messages_raw and isinstance(messages_raw, list):
                # 假设 'messages' 是一个列表，并且第一个元素是用户提问
                # 结构为: [{"content": "...", "role": "user"}]
                first_message = messages_raw[0]
                if isinstance(first_message, dict) and first_message.get('role') == 'user':
                    question_raw = first_message.get('content', "")
                    question = str(question_raw)
                else:
                    # 如果格式不符合预期，尝试直接获取第一个 content
                    question = str(first_message.get('content', '')) if isinstance(first_message, dict) else ""
            
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
                # 🎯 标记为通用能力，因为这个数据集不全是安全相关的
                "ability": "general", 
                "reward_model": {
                    "style": "rule",
                    "ground_truth": None # 验证集不要求 Gold Truth
                },
                "extra_info": {
                    'split': split_name, 
                    'index': idx,
                }
            }
            # 可以在 extra_info 中保留一些有用的原始信息，比如原始的 instruction_group
            if 'instruction_group' in example:
                 data['extra_info']['instruction_group'] = example.pop('instruction_group')

            return data

        return process_fn
    
    # 最终保留的列
    final_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

    # --- 关键步骤 3: 处理和保存 Validation Set ---
    print("\nProcessing Validation Set...")
    # 使用 remove_columns=val_dataset.column_names 来移除所有原始列
    val_dataset = val_dataset.map(function=make_map_fn('val'), with_indices=True, remove_columns=val_dataset.column_names)
    
    # 显式移除 map 后剩余的未声明字段
    val_dataset = val_dataset.remove_columns([col for col in val_dataset.column_names if col not in final_columns])

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True) 
    # 输出文件名为 test_rlvr.parquet
    output_val_path = os.path.join(local_dir, 'test_rlvr.parquet') 
    
    val_dataset.to_parquet(output_val_path)
    print(f"✅ Validation data saved to: {output_val_path} (Size: {len(val_dataset)})")