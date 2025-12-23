import requests
import json
import time

###############################skywork##########################
HOST_URL = "http://127.0.0.1:30004" 
INFERENCE_URL = HOST_URL + "/classify"

# 待评分的问题
PROMPT = "什么是神经网络中Sigmoid节点的数值输出范围？"

# 两个不同的响应，用于比较评分
RESPONSE_1 = "Sigmoid节点的输出范围在-1到1之间。"
RESPONSE_2 = "Sigmoid节点的输出范围在0到1之间。"
# =================================================================

``
# --- 关键修改：扁平化对话结构 ---
# 将对话对格式化为模型可接受的单一文本格式
def format_conversation(prompt, response):
    """将 Prompt 和 Response 格式化为 InternLM2 奖励模型通常接受的单行文本。"""
    # 奖励模型通常需要角色标签来区分输入
    return f"Human: {prompt}\nAssistant: {response}"

# 构造请求体数据，使用服务器要求的 "text" 字段
request_data = {
    # "text" 字段包含一个列表，列表中的每一项是要评分的完整文本
    "text": [
        format_conversation(PROMPT, RESPONSE_1), # 第一个要评分的文本
        format_conversation(PROMPT, RESPONSE_2), # 第二个要评分的文本
    ],
}
# -----------------------------------


def send_score_request():
    """发送评分请求并处理响应。"""
    print(f"[{time.strftime('%H:%M:%S')}] 正在向 {INFERENCE_URL} 发送评分请求...")
    print(f"请求内容：对两个 Assistant 响应进行评分...")

    try:
        response = requests.post(
            INFERENCE_URL,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=15 # 增加超时时间以应对模型加载
        )
        response.raise_for_status()
        
        response_data = response.json()
        
        print("\n--- API 响应成功 ---")
        
        # 奖励模型返回的格式可能仍然是列表，我们保持提取逻辑不变
        if isinstance(response_data, list):
            scores = [x.get("embedding", "N/A") for x in response_data]

            print(f"原始响应：{json.dumps(response_data, indent=4, ensure_ascii=False)}")
            print("\n==============================")
            print(f"问题: {PROMPT}")
            print("==============================")
            print(f"响应 1 (错误, -1到1): {RESPONSE_1}")
            print(f"得分 1: {scores[0]}")
            print("-" * 30)
            print(f"响应 2 (正确, 0到1): {RESPONSE_2}")
            print(f"得分 2: {scores[1]}")
            print("==============================")
            print("\n诊断：得分越高的响应，奖励模型认为质量/正确性越好。")
        else:
            print(f"警告：响应格式异常。原始数据:\n{response_data}")

    except requests.exceptions.HTTPError as e:
        print(f"\n--- 调用 API 失败 (HTTP 错误) ---")
        print(f"错误信息: {e}")
        if hasattr(e.response, 'text'):
            print(f"服务器返回内容: {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"\n--- 调用 API 失败 (连接/网络错误) ---")
        print(f"错误信息: {e}")
    except json.JSONDecodeError:
        print("\n--- JSON 解析失败 ---")
        print(f"服务器返回了非 JSON 格式数据:\n{response.text}")


# if __name__ == "__main__":
#     send_score_request()






################################ Llama-3.1-8B-Instruct-RM-RB2  ######################################




import requests
import json
import time

# #################################################
# ### 配置部分：根据您的部署信息修改
# #################################################
HOST = "127.0.0.1" # 如果在同一台机器上运行，使用 127.0.0.1
PORT = 30006       # 您部署时指定的端口号
HOST_URL = f"http://{HOST}:{PORT}"

# SGLang 推荐使用 /score 接口用于获取奖励模型的分数
INFERENCE_URL = HOST_URL + "/classify" 

# #################################################
# ### 测试数据：用于验证奖励模型效果
# #################################################
# 待评分的问题：这是一个具有明确正确答案的问题
PROMPT = "什么是神经网络中 Sigmoid 激活函数的数值输出范围？"

# 响应 1：错误答案
RESPONSE_1 = "Sigmoid 节点的输出范围在 -1 到 1 之间。"
# 响应 2：正确答案
RESPONSE_2 = "Sigmoid 节点的输出范围在 0 到 1 之间。"
# =================================================================


# --- 关键函数：格式化对话结构 ---
def format_conversation(prompt: str, response: str) -> str:
    """
    将 Prompt 和 Response 格式化为 Llama/Instruct 模型通常接受的输入序列。
    奖励模型通常需要完整且带有角色标签的对话历史作为输入。
    """
    # Llama 3.1 Instruct 格式通常遵循 Chat Template，但对于 RM 来说，
    # 简单的 Human/Assistant 标签通常是有效的输入序列。
    return f"Human: {prompt}\nAssistant: {response}"

# 构造请求体数据
request_data = {
    # 'text' 字段包含一个列表，列表中的每一项是要评分的完整文本序列
    "text": [
        format_conversation(PROMPT, RESPONSE_1), # 序列 1 (错误)
        format_conversation(PROMPT, RESPONSE_2), # 序列 2 (正确)
    ],
}
# -----------------------------------


def send_score_request():
    """发送评分请求并处理响应。"""
    print(f"[{time.strftime('%H:%M:%S')}] 正在向 {INFERENCE_URL} 发送评分请求...")
    print(f"请求内容：对两个 Assistant 响应进行评分 (总数: {len(request_data['text'])})")

    try:
        response = requests.post(
            INFERENCE_URL,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30 # 增加超时时间以应对模型首次加载和计算
        )
        response.raise_for_status()
        
        response_data = response.json()
        
        print("\n--- API 响应成功 ---")
        
        # SGLang 的 /score 接口通常返回一个包含 'scores' 列表的 JSON 对象
        if isinstance(response_data, dict) and "scores" in response_data and isinstance(response_data["scores"], list):
            scores = response_data["scores"]
            
            if len(scores) < 2:
                print(f"错误: 预期的分数数量不足 (预期 2, 实际 {len(scores)})")
                return

            print("\n============================== 评分结果 ==============================")
            print(f"问题: {PROMPT}")
            print("===================================================================")
            print(f"响应 1 (错误): {RESPONSE_1}")
            print(f"**得分 1:** {scores[0]:.4f}")
            print("-" * 65)
            print(f"响应 2 (正确): {RESPONSE_2}")
            print(f"**得分 2:** {scores[1]:.4f}")
            print("===================================================================")
            
            # 诊断/结论
            if scores[1] > scores[0]:
                print(f"\n✅ 诊断: 模型表现良好！(得分 2 > 得分 1) - 正确响应获得了更高的奖励。")
            else:
                print(f"\n❌ 诊断: 模型表现异常！(得分 1 >= 得分 2) - 正确响应未获得更高的奖励。")
            
        else:
            print(f"警告：响应格式异常。原始数据:\n{json.dumps(response_data, indent=4, ensure_ascii=False)}")

    except requests.exceptions.HTTPError as e:
        print(f"\n--- 调用 API 失败 (HTTP 错误) ---")
        print(f"错误信息: {e}")
        if hasattr(e.response, 'text'):
            print(f"服务器返回内容:\n{e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"\n--- 调用 API 失败 (连接/网络错误) ---")
        print(f"错误信息: 请检查 SGLang 服务器是否在 {HOST_URL} 端口 {PORT} 运行，并且 Docker 容器状态正常。")
    except json.JSONDecodeError:
        print("\n--- JSON 解析失败 ---")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"服务器返回了非 JSON 格式数据:\n{response.text}")

if __name__ == "__main__":
    send_score_request()







################################ RM-R1  ######################################
# import requests

# url = "http://127.0.0.1:30004/generate"

# prompt = f"""
# You are a reward model.
# Evaluate the quality of the following answer to the question.
# Output only a single scalar score between 0 and 10.
# The output must contain only the score wrapped in <score> and </score>, with no extra text.

# === Input ===
# [Question]
# What are the main causes of climate change?

# [Answer]
# Climate change happens because the weather changes a lot and sometimes the Earth just gets hotter for no clear reason.


# === Output ===
# <score>?</score>
# """

# payload = {
#     "text": prompt,          # 必须是 text，不能叫 prompt
#     "max_new_tokens": 1024,
#     "temperature": 0
# }

# resp = requests.post(url, json=payload)

# if resp.status_code == 200:
#     print(resp.json()["text"])
# else:
#     print("--- 调用 API 失败 ---")
#     print(resp.status_code, resp.text)

