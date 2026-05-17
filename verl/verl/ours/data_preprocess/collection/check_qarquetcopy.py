import pandas as pd
import json
import ast

# ==========================================
# 1. 配置文件路径
# ==========================================
file_path = "cwli16/data/train_covalid/glaive-code-sampled/train_code_partial.parquet"

print(f"🔄 正在读取文件: {file_path} ...")
try:
    df = pd.read_parquet(file_path)
    print(f"✅ 读取成功，共 {len(df)} 行。")
except Exception as e:
    print(f"❌ 读取 Parquet 失败: {e}")
    exit()

# 检查是否存在 'reward_model' 列
if 'reward_model' not in df.columns:
    print(f"❌ 错误：在列名中找不到 'reward_model'。现有列名: {df.columns.tolist()}")
    exit()

# ==========================================
# 2. 定义核心检查函数
# ==========================================
def check_nested_ground_truth(reward_model_val):
    # --- 第一步：解析 reward_model ---
    data = reward_model_val
    
    # 如果 reward_model 是字符串 (JSON string)，先解包
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except:
            try:
                data = ast.literal_eval(data)
            except:
                return False, "reward_model 无法解析为字典"
    
    # 确保它是字典
    if not isinstance(data, dict):
        return False, f"reward_model 不是字典，而是 {type(data).__name__}"
    
    # --- 第二步：提取 ground_truth ---
    if 'ground_truth' not in data:
        return False, "reward_model 中缺少 'ground_truth' 键"
    
    gt_val = data['ground_truth']
    
    # --- 第三步：验证 ground_truth 的具体格式 ---
    # 目标格式: [{"name": "...", "arguments": {...}}]
    
    # 如果 gt_val 是字符串 (二次嵌套 JSON)，尝试再解包一次
    if isinstance(gt_val, str):
        try:
            gt_val = json.loads(gt_val)
        except:
            try:
                gt_val = ast.literal_eval(gt_val)
            except:
                return False, "ground_truth 字段是字符串且无法解析"

    # 必须是列表
    if not isinstance(gt_val, list):
        return False, f"ground_truth 不是列表 (List)，而是 {type(gt_val).__name__}"

    # 遍历列表中的每一项
    for idx, item in enumerate(gt_val):
        if not isinstance(item, dict):
            return False, f"列表第 {idx} 项不是字典"
        
        # 检查必要的键
        if "name" not in item:
            return False, f"列表第 {idx} 项缺少 'name'"
        if "arguments" not in item:
            return False, f"列表第 {idx} 项缺少 'arguments'"
        
        # (可选) 严格检查 arguments 是否为字典
        # 如果你的数据里 arguments 允许是字符串，可以注释掉下面两行
        if not isinstance(item['arguments'], dict):
             return False, f"列表第 {idx} 项的 'arguments' 不是字典"

    return True, "Pass"

# ==========================================
# 3. 执行检查
# ==========================================
print("🕵️  正在深入 reward_model 内部检查 ground_truth 格式...")

# 应用检查函数
results = df['reward_model'].apply(check_nested_ground_truth)

# 拆分结果
df['is_valid'] = results.apply(lambda x: x[0])
df['error_msg'] = results.apply(lambda x: x[1])

# ==========================================
# 4. 输出报告
# ==========================================
invalid_rows = df[~df['is_valid']]

print("\n" + "="*50)
print("📊 最终检查报告")
print("="*50)

if invalid_rows.empty:
    print("✅ 完美通过！")
    print("所有数据的 reward_model['ground_truth'] 都符合 `[{\"name\":..., \"arguments\":{...}}]` 格式。")
    
    # 打印一个样本让用户放心
    print("\n🔍 抽样展示一个解析后的结果：")
    sample_rm = df['reward_model'].iloc[0]
    # 简单的展示逻辑，适应 sample_rm 是 dict 还是 str
    print(sample_rm)
else:
    print(f"❌ 发现 {len(invalid_rows)} 条数据格式不符！")
    print("\n前 3 个错误详情：")
    for i, (idx, row) in enumerate(invalid_rows.head(3).iterrows()):
        print(f"--- 错误样本 {i+1} (Row {idx}) ---")
        print(f"错误原因: {row['error_msg']}")
        print(f"原始内容: {row['reward_model']}")