
from datasets import load_dataset




# ==========================================
# 1. 继续严格防泄漏：跳过 SFT 数据
# ==========================================
SEED = 42
SFT_SAMPLES = 3000
RL_SAMPLES = 10000  # 这里填你这次想要抽取的 RL 数据量

print("📦 加载完整数据集...")
dataset = load_dataset("Elliott/Openr1-Math-46k-8192", split="train")

print(f"🔀 使用 seed={SEED} 全局打乱，跳过前 {SFT_SAMPLES} 条，抽取后续 {RL_SAMPLES} 条...")
rl_dataset = dataset.shuffle(seed=SEED).select(range(SFT_SAMPLES, SFT_SAMPLES + RL_SAMPLES))

# ==========================================
# 2. 极简过滤：只留下 user 的 prompt
# ==========================================
def filter_system_prompt(example):
    original_prompt = example.get("prompt", [])
    
    # 如果 prompt 是标准的列表字典格式：[{"role": "system", ...}, {"role": "user", ...}]
    if isinstance(original_prompt, list):
        # 列表推导式：只保留 role 为 user 的那一项
        example["prompt"] = [msg for msg in original_prompt if msg.get("role") == "user"]
        
    return example

print("🔄 正在剔除 system prompt，保留原有其他字段...")
# 直接 map 覆盖，不删除其他任何原有字段
rl_clean_dataset = rl_dataset.map(filter_system_prompt)

# ==========================================
# 3. 保存导出
# ==========================================
output_file = "/data2/cwli16/data/openr1_math_rl_10k.json"
rl_clean_dataset.to_json(output_file, force_ascii=False)
print(f"✅ 搞定！数据已原汁原味保存至 {output_file}")



from datasets import load_dataset

# 1. 明确路径
input_json = "/data2/cwli16/data/openr1_math_rl_10k.json"
output_parquet = "/data2/cwli16/data/openr1_math_rl_10k.parquet"

# 2. 读取并转换
print(f"🔄 正在读取 JSON 并清洗 Prompt...")
dataset = load_dataset("json", data_files=input_json, split="train")

# 只要 role 为 user 的内容
dataset = dataset.map(lambda x: {"prompt": [m for m in x["prompt"] if m["role"] == "user"]})

# 3. 核心保存动作：注意是用 dataset 对象去调用
print(f"💾 正在保存至: {output_parquet}")
dataset.to_parquet(output_parquet)

print("✅ 转换成功！")