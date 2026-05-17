import pandas as pd
import ast
from datasets import load_dataset

# ==========================================
# 1. 加载 Hugging Face 数据集
# ==========================================
print("正在加载数据集 gorilla-llm/Berkeley-Function-Calling-Leaderboard ...")
dataset = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard")

# 打印一下看看有哪些 split (例如 'train', 'test')
print("包含的 Splits:", dataset.keys())

# 通常 Leaderboard 数据集主要在 'test' split 中，或者 'train'。
# 这里我们需要选取一个 split 转换为 df。
# 假设我们检查 'test' 集（如果报错说没有 test，请改为 'train'）
target_split = 'test' if 'test' in dataset else 'train'
df = dataset[target_split].to_pandas()

print(f"已加载 split '{target_split}'，共 {len(df)} 行数据。")
print("当前数据的列名:", df.columns.tolist())
print("-" * 30)

# ==========================================
# 2. 列名映射 (关键步骤！)
# ==========================================
# 原始数据集的列名可能不是 'answer' 和 'tools'。
# 你需要根据上面打印的 "当前数据的列名" 来修改下面的变量。
# 常见的对应关系可能是：
# 你的逻辑字段 -> 数据集实际字段
col_answer = 'answer'  # 如果实际叫 'ground_truth'，请修改这里
col_tools = 'tools'    # 如果实际叫 'function' 或 'functions'，请修改这里

# 自动检查列名是否存在，不存在则提示
if col_answer not in df.columns or col_tools not in df.columns:
    print(f"❌ 错误：在数据中找不到列名 '{col_answer}' 或 '{col_tools}'")
    print(f"⚠️  请检查上方打印的列名，并将代码中的 col_answer 和 col_tools 修改为正确的列名。")
    # 比如：BFCL 数据集通常 Tools 存在 'function' 列，Answer 存在 'ground_truth' 列
    exit()

# ==========================================
# 3. 执行你的异常检查逻辑
# ==========================================
target_text = "The query cannot be answered, no tools were provided"

def is_tools_not_empty(tools_value):
    if tools_value is None: return False
    # 如果是字符串形式的列表，尝试解析
    if isinstance(tools_value, str):
        try:
            tools_list = ast.literal_eval(tools_value)
        except:
            return False # 解析失败视为空
    else:
        tools_list = tools_value
    
    # 判断是否为列表且不为空
    return isinstance(tools_list, list) and len(tools_list) > 0

# 开始筛选
anomalies = df[
    (df[col_answer].astype(str).str.contains(target_text, na=False, regex=False)) & 
    (df[col_tools].apply(is_tools_not_empty))
]

if not anomalies.empty:
    print(f"发现异常！共有 {len(anomalies)} 条数据。")
    print("前 2 条异常示例：")
    print(anomalies[[col_answer, col_tools]].head(2))
else:
    print("验证通过：所有包含该 Answer 的数据，其 Tools 确实都为空。")