import argparse
import os
import datasets
import json
import re # 导入正则表达式库用于解析答案

# 定义需要从原始数据中移除的所有列的列表
# GSM8K 原始列包括：question, answer
COLUMNS_TO_REMOVE = [
    'question', 
    'answer', 
]

def extract_gsm8k_ground_truth(answer_text):
    """
    从 GSM8K 的答案文本中提取最终的数值答案。
    答案格式示例: "...#### 5"
    """
    if not isinstance(answer_text, str):
        return None
        
    # 使用正则表达式查找 '####' 后面直到字符串末尾的所有内容，并去除空白
    match = re.search(r'####\s*(.*)$', answer_text.strip())
    if match:
        return match.group(1).strip()
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 🎯 目标目录改为 GSM8K 路径
    parser.add_argument('--local_dir', default='/data2/cwli16/data/gsm8k_processed_val')
    args = parser.parse_args()

    # 🎯 切换数据源
    data_source_repo = 'openai/gsm8k' 
    data_source_config = 'main' # GSM8K 的子集/配置
    # 🎯 切换 data_source 最终值
    data_source_value = 'gsm8k'
    
    # GSM8K 使用 'train' 分割作为主要数据
    DATA_SPLIT = 'train'
    VAL_SIZE_TARGET = 500 # 沿用之前的验证集大小标准
    RANDOM_SEED = 42
    
    # 📝 只需要加载一次数据集
    print(f"Loading dataset: {data_source_repo} (Config: {data_source_config}, Split: {DATA_SPLIT})...")
    try:
        # 加载整个数据集
        full_dataset = datasets.load_dataset(data_source_repo, data_source_config, split=DATA_SPLIT)
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

    def make_map_fn(split_name):
        def process_fn(example, idx):
            
            # 1. 提取原始的 'question' 字段作为 prompt
            question_raw = example.get('question', "") 
            
            # 2. 提取并解析 'answer' 字段作为 ground_truth
            answer_full = example.get('answer', "")
            ground_truth = extract_gsm8k_ground_truth(answer_full)
            
            # 3. 构造输出数据结构
            data = {
                "data_source": data_source_value, 
                "prompt": [{
                    "role": "user",
                    "content": str(question_raw).strip(), # <--- 最终使用的 prompt 内容
                }],
                "ability": "math", # 基础算术应用题使用 'math'
                "reward_model": {
                    "style": "rule",
                    # 'ground_truth' 对应提取出的最终答案（'####' 后的部分）
                    "ground_truth": ground_truth 
                },
                "extra_info": {
                    'split': split_name, 
                    'index': idx,
                    "question_raw": question_raw,
                    "answer_full_raw": answer_full, # 保留完整的原始答案，用于调试
                }
            }
            return data

        return process_fn
    
    # 最终保留的列
    final_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

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
    # 输出文件名为 test_math.parquet (因为是数学任务)
    output_val_path = os.path.join(local_dir, 'test_math.parquet') 
    
    val_dataset.to_parquet(output_val_path)
    print(f"✅ Validation data saved to: {output_val_path} (Size: {len(val_dataset)})")
    print(f"字段结构检查: {val_dataset.column_names}")

    print("\n所有数据集处理完成。")