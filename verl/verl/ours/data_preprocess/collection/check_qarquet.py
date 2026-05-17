


# import pandas as pd
# import os
# import json

# # 配置你的文件路径
# DATA_DIR = '/data2/cwli16/data/train_covalid/xlam-function-calling/train_irrelevant_1k.parquet'


# def print_separator(title):
#     print("\n" + "="*60)
#     print(f" {title} ")
#     print("="*60 + "\n")

# def inspect_full_content(file_path):

#     print(f"路径: {file_path}")
    
#     if not os.path.exists(file_path):
#         print(f"❌ 文件不存在!")
#         return

#     # 读取数据
#     df = pd.read_parquet(file_path)
#     print(f"总行数: {len(df)}")
#     print(f"包含列: {df.columns.tolist()}")
    
#     # 提取第一行样本
#     sample = df.iloc[0]
    
#     print_separator(f"样本数据展示 (第 0 条) ")

#     # 遍历所有列进行打印
#     for col_name in df.columns:
#         content = sample[col_name]
        
#         print(f"⭐⭐⭐ [字段名]: {col_name} ⭐⭐⭐")
        
#         # 特殊处理 prompt 字段，为了看得更清楚，我们拆解 list 打印
#         if col_name == 'prompt' and isinstance(content, (list, np.ndarray if 'np' in locals() else list)):
#             print(f"数据类型: List (包含 {len(content)} 条消息)")
#             for i, msg in enumerate(content):
#                 role = msg.get('role', 'Unknown')
#                 text = msg.get('content', '')
#                 print(f"\n  --- Message {i+1} (Role: {role}) ---")
#                 print(f"  {text}")
#                 print("  " + "-"*40)
        
#         # 特殊处理 reward_model 和 extra_info (通常是 dict)
#         elif isinstance(content, dict):
#             print("数据类型: Dict")
#             # 使用 json dumps 格式化打印，方便看结构
#             print(json.dumps(content, indent=4, ensure_ascii=False))
            
#         else:
#             # 其他字段直接打印
#             print(content)
            
#         print("\n" + "_"*80 + "\n")

# if __name__ == '__main__':
#     # 为了避免输出太长刷屏，你可以只运行其中一个，或者把输出重定向到文件
#     inspect_full_content(DATA_DIR)
#     # inspect_full_content(TEST_FILE, "测试集 (Test)")



import pandas as pd
import os
import json
import numpy as np  # 记得导入 numpy

# 配置你的文件路径
DATA_DIR = '/data2/cwli16/data/train_covalid/glaive-code-sampled/train_code_partial.parquet'

# --- 核心修改：增加一个自定义类来处理 NumPy 数组 ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def print_separator(title):
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60 + "\n")

def inspect_full_content(file_path):
    print(f"路径: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在!")
        return

    # 读取数据
    df = pd.read_parquet(file_path)
    print(f"总行数: {len(df)}")
    print(f"包含列: {df.columns.tolist()}")
    
    if len(df) == 0:
        print("数据集为空")
        return

    # 提取第一行样本
    sample = df.iloc[19]
    
    print_separator(f"样本数据展示 (第 0 条) ")

    # 遍历所有列进行打印
    for col_name in df.columns:
        content = sample[col_name]
        
        print(f"⭐⭐⭐ [字段名]: {col_name} ⭐⭐⭐")
        
        # 特殊处理 prompt 字段 (通常是 numpy array 包含 dicts)
        # 注意：pandas 读取 parquet 里的 list 列时，经常会变成 numpy array
        if col_name == 'prompt':
            # 如果是 numpy 数组，先转成 list
            if isinstance(content, np.ndarray):
                content = content.tolist()
                
            print(f"数据类型: List (包含 {len(content)} 条消息)")
            for i, msg in enumerate(content):
                role = msg.get('role', 'Unknown')
                text = msg.get('content', '')
                print(f"\n --- Message {i+1} (Role: {role}) ---")
                print(f" {text}")
                print(" " + "-"*40)
        
        # 特殊处理 reward_model 和 extra_info (通常是 dict)
        elif isinstance(content, dict):
            print("数据类型: Dict")
            # --- 关键修改：在这里传入 cls=NumpyEncoder ---
            try:
                print(json.dumps(content, indent=4, ensure_ascii=False, cls=NumpyEncoder))
            except Exception as e:
                print(f"打印出错 (尝试直接打印): {e}")
                print(content)
            
        else:
            # 其他字段直接打印
            print(content)
            
        print("\n" + "_"*80 + "\n")

if __name__ == '__main__':
    inspect_full_content(DATA_DIR)