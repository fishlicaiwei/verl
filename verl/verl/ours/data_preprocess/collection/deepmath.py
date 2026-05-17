# import os
# from datasets import load_dataset, concatenate_datasets

# # 依然保留镜像加速，防止偶发的检查更新卡顿
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# print("1. 正在从本地缓存加载数据集...")
# dataset = load_dataset("zwhe99/DeepMath-103K", split="train")

# # === 核心修改区：按比例分层采样 ===
# TARGET_TOTAL_SIZE = 10000

# # 定义获取难度的安全转换函数，防止部分数据 difficulty 字段为空或非数字
# def get_difficulty(x):
#     try:
#         return float(x.get('difficulty', 0))
#     except (ValueError, TypeError):
#         return 0.0

# print("2. 开始按难度分层过滤...")
# # 第 1 层：锚点层 (难度 4.0 ~ 5.9)，占比 10%
# ds_level_1 = dataset.filter(lambda x: 5.0 <= get_difficulty(x) < 6.0)
# # 第 2 层：分水岭层 (难度 6.0 ~ 6.9)，占比 40%
# ds_level_2 = dataset.filter(lambda x: 6.0 <= get_difficulty(x) < 7.0)
# # 第 3 层：绝境层 (难度 7.0 及以上)，占比 50%
# ds_level_3 = dataset.filter(lambda x: get_difficulty(x) >= 7.0)

# # 计算各层实际应采样的数量（使用 min 防止库内存量不够）
# size_l1 = min(int(TARGET_TOTAL_SIZE * 0.10), len(ds_level_1))
# size_l2 = min(int(TARGET_TOTAL_SIZE * 0.40), len(ds_level_2))
# size_l3 = min(int(TARGET_TOTAL_SIZE * 0.50), len(ds_level_3))

# print(f"各层库存量 -> 4-5分: {len(ds_level_1)}, 6分: {len(ds_level_2)}, 7分+: {len(ds_level_3)}")
# print(f"实际采样量 -> 4-5分: {size_l1}, 6分: {size_l2}, 7分+: {size_l3}")

# # 独立打乱并截取对应的数量
# sampled_l1 = ds_level_1.shuffle(seed=42).select(range(size_l1))
# sampled_l2 = ds_level_2.shuffle(seed=42).select(range(size_l2))
# sampled_l3 = ds_level_3.shuffle(seed=42).select(range(size_l3))

# # 将三份数据拼接到一起，并做一次总打乱，防止训练时按难度扎堆
# print("3. 拼接并混合数据集...")
# combined_dataset = concatenate_datasets([sampled_l1, sampled_l2, sampled_l3]).shuffle(seed=42)
# # ==================================

# COLUMNS_TO_REMOVE_AFTER_EXTRACTION = dataset.column_names

# def make_map_fn(split_name, data_source_value="DeepMath-103K"):
#     def process_fn(example, idx):
#         question_raw = example.get('question', '')
#         ground_truth = example.get('final_answer', '')
#         solution = example.get('solution', '')
#         return {
#             "data_source": data_source_value, 
#             "prompt": [{"role": "user", "content": str(question_raw)}],
#             "ability": "math", 
#             "reward_model": {
#                 "style": "rule",
#                 "ground_truth": str(ground_truth),
#                 "solution": str(solution),
#             },
#             "extra_info": {
#                 "split": split_name,
#                 "index": idx,
#                 "question_raw": str(question_raw),
#                 "difficulty_label": get_difficulty(example) # 建议把难度存进额外信息，方便后续分析
#             }
#         }
#     return process_fn

# print("4. 格式化数据结构 (如果命中 map 缓存会瞬间完成)...")
# final_dataset = combined_dataset.map(
#     make_map_fn(split_name="train"), 
#     with_indices=True,
#     remove_columns=COLUMNS_TO_REMOVE_AFTER_EXTRACTION
# )

# # 5. 保存
# os.makedirs("/data2/jyyan10/", exist_ok=True)
# # 建议改个名字以区分之前的纯高难度数据
# save_path = "/data2/jyyan10/deepmath_mixed_10k_ratio_1_4_5.jsonl"

# print(f"5. 正在保存到: {save_path}")
# final_dataset.to_json(save_path, force_ascii=False)
# print("保存成功！你的临界点数据集已准备完毕。")

import os
from datasets import Dataset

# 1. 设置路径
jsonl_input_path = "/data2/jyyan10/deepmath_mixed_10k_ratio_1_4_5.jsonl"
parquet_output_path = "/data2/cwli16/deepmath_hard_10k2.parquet"

print(f"正在从本地读取 JSONL 文件: {jsonl_input_path}")

# 2. 从本地 JSONL 加载数据
# 注意：JSONL 文件需要使用 Dataset.from_json 加载
dataset = Dataset.from_json(jsonl_input_path)

# 3. 转换为 Parquet 格式
print(f"正在转换并保存为 Parquet: {parquet_output_path}")
dataset.to_parquet(parquet_output_path)

print(f"🎉 转换完成！文件已生成。")