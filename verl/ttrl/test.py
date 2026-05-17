import json
import os
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- 配置部分 ---
API_KEY = "AIzaSyC3g4Kk709e_72pUH7fEJpVz-tfJOiYyCI"  # 请替换为你的 API Key
INPUT_FILE = "/workspace/data/cot_answer_post_analysis/data_deduplicated.json"
OUTPUT_FILE = "/workspace/data/cot_answer_post_analysis/data_with_gemini_judge.json"
CONCURRENT_WORKERS = 5  # 设置并发数，注意 API 的 Rate Limit

# 初始化 Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash') # 或者 'gemini-1.5-flash' 速度更快

def get_gemini_response(item):
    """
    为单个 item 调用 Gemini 接口
    """
    question = item.get("prompt", "")
    cot_a = item.get("think", "")
    cot_b = item.get("baseline_think", "")

    # 组装 Prompt
    prompt = f"""You are a professional logical reasoning evaluator.
Please compare two Chain-of-Thought (CoT) reasoning processes for the same question.

Question: {question}

Chain-of-Thought A: {cot_a}

Chain-of-Thought B: {cot_b}

Task: Determine which CoT is better in terms of logical rigor, clarity, and depth of reasoning. 
Constraint: Output ONLY the character "A" or "B". No other text.

Result:"""

    try:
        # 使用 generation_config 强制模型尽量简洁并设置 temperature 为 0
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=2,
                temperature=0.0
            )
        )
        result = response.text.strip().upper()
        # 简单清洗，只保留 A 或 B
        if 'A' in result: final_res = "A"
        elif 'B' in result: final_res = "B"
        else: final_res = result # 原样返回异常输出以便后续排查

        print(f"Prompt: {question[:30]}... | Result: {final_res}")
        return final_res
    except Exception as e:
        print(f"Error processing item: {e}")
        return "Error"

def main():
    # 1. 读取数据
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"总计加载 {len(data)} 条数据，准备调用 Gemini API (并发数: {CONCURRENT_WORKERS})...")

    # 2. 使用线程池并发执行
    results = []
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        # 提交所有任务
        future_to_item = {executor.submit(get_gemini_response, item): item for item in data}

        # 按照完成顺序收集结果
        for future in tqdm(as_completed(future_to_item), total=len(data)):
            item = future_to_item[future]
            try:
                # 获取 Gemini 的输出并存入新字段 result
                item["result"] = future.result()
            except Exception as e:
                item["result"] = f"Exception: {e}"

    # 3. 保存新文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\n实验完成！结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()