PROMPT_TEMPLATE='''
You are an expert in mathematics and logical reasoning. Your tassk is to evaluate the cor
rectness of a solution to a given math problem, with a *strong emphasis on the reasoning
process**, not just the final answer.
Below is the **Problem** and the **Solution (Provided by aanother AI model)**:
**Problem**:
{question}
**Solution (Provided by another AI model)**:
{solution}
Please perform the following tasks:
1. **Analyze the solution step-by-step**, paying close attention to: - Computational accu
racy - Logical consistency - Conceptual understanding - Whether the reassoning is valid and
complete
2. **Identify any issues or errors in the reasoning**, even if the finaI answer is correct. Clas
sify them into the following categories (if applicable): - **Calculation ETror*: Mistakes in
arithmetic, algebraic manipulation, or numerical computation. - **Logical Error*: Invalid
reasoning, flawed logic, or incorrect inference. - **Conceptual Error**: Misunderstanding
or misuse of mathematical concepts or definitions. - *Omission / IIncompleteness**: Miss
ing steps, incomplete justification, or not addressing all parts of the quther***
Any other type of error that does not fit into the above categories
3. **Provide a final judgment** on whether the solution is logically soundd and free of errors
in reasoning.
Please format your response as follows:
**Issues Identified:**
- [Issue 1]: [Classification] - [Brief explanation] - [Issue 2]: [Classifion] - [Brief expla-
nation] - ...
Let's think step by step and output your final judgment within \\boxed{{}}
\\boxed{{yes}} or \\boxed{{no}}
'''
import json
import re
import asyncio
from tqdm.asyncio import tqdm
import os
from openai import AsyncOpenAI
from collections import defaultdict

# --- 配置区 ---
API_KEY = "unused"  # SGLang 通常不需要 API Key
API_BASE = "http://localhost:30021/v1"
MODEL_NAME = "gpt-oss-120b"
DATA_PATH1 = "/data2/cwli16/opencompass/outputs/aime2025_hot_vr_exp1/20260412_224827/predictions/hot_vr_exp1/aime2025_eval.json" # 请替换为你的实际文件名
DATA_PATH2 = "/data2/cwli16/opencompass/outputs/aime2025_hot_dsr1_think_exp1/20260412_151358/predictions/hot_dsr1_think_exp1/aime2025_eval.json" # 请替换为你的实际文件名

CONCURRENCY_LIMIT = 32 # 根据 A800 的负荷调整并发数

client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)

# --- 核心逻辑 ---

async def call_oss_eval(question, solution, semaphore, error_counter):
    async with semaphore:
        for attempt in range(3):  # 最多尝试 3 次
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "user", "content": PROMPT_TEMPLATE.format(question=question, solution=solution)}
                    ],
                    # 注意：如果 SGLang 报错不支持直接传参数，请放进 extra_body
                    extra_body={"reasoning_effort": "high"}, 
                    temperature=0.7 + (attempt * 0.2), # 格式错时稍微增加随机性
                    max_tokens=8192
                )
                content = response.choices[0].message.content
                
                # 尝试解析结果
                match = re.search(r'\\boxed\{(yes|no)\}', content.lower())
                if match:
                    return 1 if match.group(1) == "yes" else 0
                
                # 如果没解析到，打印一下报错，准备重试
                # print(f"[Attempt {attempt+1}] Format Error: No boxed yes/no found. Retrying...")
                
            except Exception as e:
                # print(f"[Attempt {attempt+1}] API Error: {e}")
                await asyncio.sleep(1)
                continue
        
        # 3次重试均失败
        error_counter['count'] += 1
        return None
    
async def run_experiment(data_list, lable):
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    error_counter = {'count': 0}
    
    tasks = [call_oss_eval(item['origin_prompt'], item['prediction'], semaphore, error_counter) for item in data_list]
    
    results = []
    # 带有实时失败计数的进度条
    pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Run {lable}", unit="req")
    
    for task in pbar:
        res = await task
        results.append(res)
        pbar.set_postfix({"Sys_Errors": error_counter['count']})
    
    return results, error_counter['count']

async def main(data_path,lable):
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    data_list = [raw_data[str(i)] for i in range(len(raw_data))]
    total_samples = len(data_list)
    
    all_experiments = []


    results, final_error_count = await run_experiment(data_list, lable)
    all_experiments.append((results, final_error_count))

    # --- 最终统计输出 ---
    print("\n" + "="*50)
    print(f"{'EXPERIMENT SUMMARY':^50}")
    print("="*50)

    for idx, (res, err_count) in enumerate(all_experiments):
        valid_res = [r for r in res if r is not None]
        num_valid = len(valid_res)
        num_yes = sum(valid_res) if valid_res else 0
        
        print(f"\n[Run {idx + 1}]")
        print(f"  - Total Requests: {total_samples}")
        print(f"  - System/Format Errors (Excluded): {err_count}")
        print(f"  - Valid Evaluations: {num_valid}")
        
        if num_valid > 0:
            pass_rate = (num_yes / num_valid) * 100
            print(f"  - Effective Pass Rate: {pass_rate:.2f}%")
        else:
            print("  - Effective Pass Rate: N/A")

    print("\n" + "="*50)

if __name__ == "__main__":
    for i in range(3):
        print("\nexp {i}\n")
        # asyncio.run(main(DATA_PATH1,"vr"))
        # print("\n##############################\n")
        asyncio.run(main(DATA_PATH2,"dsr1"))
# VR
# [Run 1]
#   - Total Requests: 960
#   - System/Format Errors (Excluded): 76
#   - Valid Evaluations: 884
#   - Effective Pass Rate: 31.67%
# [Run 2]
#   - Total Requests: 960
#   - System/Format Errors (Excluded): 74
#   - Valid Evaluations: 886
#   - Effective Pass Rate: 32.62%
#dsr1
# [Run 1]
#   - Total Requests: 960
#   - System/Format Errors (Excluded): 83
#   - Valid Evaluations: 877
#   - Effective Pass Rate: 32.50%
# [Run 2]
#   - Total Requests: 960
#   - System/Format Errors (Excluded): 74
#   - Valid Evaluations: 886
#   - Effective Pass Rate: 32.84%
# [Run 3]
#   - Total Requests: 960
#   - System/Format Errors (Excluded): 80
#   - Valid Evaluations: 880
#   - Effective Pass Rate: 32.05%