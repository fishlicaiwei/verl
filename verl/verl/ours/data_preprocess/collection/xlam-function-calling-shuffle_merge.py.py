import pandas as pd
import os
import numpy as np

# ================= 配置区域 =================

# 输入文件路径列表
INPUT_FILES = [
    '/data2/cwli16/data/train_covalid/(xlam-function-calling)-raw-data/train_function_calling_10k.parquet',
    '/data2/cwli16/data/train_covalid/(xlam-function-calling)-raw-data/train_irrelevant_1k.parquet'
]

# 输出文件路径 (合并后的新文件)
OUTPUT_DIR = '/data2/cwli16/data/train_covalid/xlam-function-calling'
OUTPUT_FILENAME = 'train_mixed_11k.parquet'

def shuffle_and_merge():
    print(f"🚀 开始处理...")
    
    data_frames = []
    
    # 1. 读取所有文件
    for file_path in INPUT_FILES:
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            continue
            
        print(f"📖 读取: {file_path}")
        df = pd.read_parquet(file_path)
        print(f"   -> 行数: {len(df)}")
        data_frames.append(df)
    
    if not data_frames:
        print("❌ 没有读取到任何数据，退出。")
        return

    # 2. 合并 (Concatenate)
    print("\n🔗 正在合并所有数据...")
    full_df = pd.concat(data_frames, ignore_index=True)
    print(f"✅ 合并完成，总行数: {len(full_df)}")

    # 3. 随机打乱 (Shuffle)
    # frac=1 表示抽取 100% 的数据（即全部），random_state=42 保证结果可复现
    print("🔀 正在随机打乱 (Seed=42)...")
    shuffled_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 4. 保存
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    print(f"💾 保存到: {output_path}")
    shuffled_df.to_parquet(output_path)
    
    # 5. 验证打印前几行
    print("\n🔍 数据预览 (前 3 条 'ability' 列):")
    # 打印 ability 列，确认正负样本是否混合了 (function_calling vs function_calling_irrelevant)
    print(shuffled_df['ability'].head(3).tolist())
    
    print(f"\n🎉 全部完成！最终文件包含 {len(shuffled_df)} 条数据。")

if __name__ == '__main__':
    shuffle_and_merge()