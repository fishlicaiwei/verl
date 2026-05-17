import matplotlib.pyplot as plt
import seaborn as sns

def analyze_rubric_length(df):
    # 计算字符长度 (也可以改为计算 token 长度，如果安装了 transformers)
    df['rubric_length'] = df['parsed_rubric'].str.len()
    
    # 统计基本信息
    max_len = df['rubric_length'].max()
    avg_len = df['rubric_length'].mean()
    median_len = df['rubric_length'].median()
    
    print("-" * 30)
    print(f"统计结果:")
    print(f"最大长度: {max_len:.0f} 字符")
    print(f"平均长度: {avg_len:.2f} 字符")
    print(f"中位数长度: {median_len:.0f} 字符")
    print("-" * 30)

    # 绘图
    plt.figure(figsize=(10, 6))
    sns.histplot(df['rubric_length'], bins=30, kde=True, color='skyblue')
    
    # 标注平均值和最大值
    plt.axvline(avg_len, color='red', linestyle='--', label=f'Average: {avg_len:.2f}')
    plt.axvline(max_len, color='green', linestyle=':', label=f'Max: {max_len}')
    
    plt.title('Distribution of Parsed Rubric Lengths')
    plt.xlabel('Length (characters)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

# 调用示例
# df = load_data(INPUT_FILE)
# analyze_rubric_length(df)