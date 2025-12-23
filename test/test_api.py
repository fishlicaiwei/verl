import asyncio
import time
from openai import AsyncOpenAI, OpenAI

def run_synchronous_tests():
    """
    包含您原始的同步测试代码，以便可以独立运行。
    """
    print("--- Running Synchronous Test 1 (xdeepseekr1) ---")
    t1 = time.time()
    try:
        client = OpenAI(
            api_key="527bdfdb89f9191848ef21902f912e82:ODBlYjI2ZTVkMTdjYmVjMDc2N2M5ZGMx",
            base_url="https://maas-api.cn-huabei-1.xf-yun.com/v2"
        )
        completion = client.chat.completions.create(
            model="xdeepseekr1",
            messages=[{"role": "system", "content": "你是一个猫娘"}, {"role": "user", "content": "你好"}],
            temperature=0.0,
            max_tokens=10000,
        )

        reason_text = completion.choices[0].message.reasoning_content
        print("Reasoning:", reason_text)
        
        response_text = completion.choices[0].message.content
        print("Response:", response_text)
        print("time:", time.time() - t1)
    except Exception as e:
        print(f"An error occurred: {e}")


    print("\n--- Running Synchronous Test 2 (xdeepseekv3devbo) ---")
    t1 = time.time()
    try:
        client = OpenAI(
            api_key="sk-mC2aMvB8pYgn89HF850c9dCdA2A14464957c3eFe7f895831",
            base_url="https://test-maas-api.cn-huabei-1.xf-yun.com/v1"
        )
        completion = client.chat.completions.create(
            model="xdeepseekv3devbo",
            messages=[{"role": "system", "content": "你是一个猫娘"}, {"role": "user", "content": "法国皇帝是谁？"}],
            temperature=0.0,
            max_tokens=10000,
        )
        reason_text = completion.choices[0].message.reasoning_content
        print("Reasoning:", reason_text)
        
        response_text = completion.choices[0].message.content
        print("Response:", response_text)
        print("time:", time.time() - t1)
    except Exception as e:
        print(f"An error occurred: {e}")


# --- 以下是新增的异步并发测试代码 ---

async def api_worker(client: AsyncOpenAI, semaphore: asyncio.Semaphore, request_idx: int):
    """
    单个API请求的工作单元，其并发执行受信号量控制。
    """
    print(f"[Task {request_idx:02d}] 等待获取信号量...")
    async with semaphore:
        print(f"[Task {request_idx:02d}] ✅ 信号量已获取，开始API调用。")
        start_time = time.time()
        try:
            completion = await client.chat.completions.create(
                model="xdeepseekv3devbo",
                messages=[
                    {"role": "system", "content": f"你是一个猫娘"},
                    {"role": "user", "content": "你好"}
                ],
                temperature=0.7,
                max_tokens=2000,
            )

            # completion = await client.chat.completions.create(
            #     model="xdeepseekr1",
            #     messages=[{"role": "system", "content": "你是一个猫娘"}, {"role": "user", "content": "你好"}],
            #     temperature=0.0,
            #     max_tokens=10000,
            # )
            response_text = completion.choices[0].message.content
            reason_text = completion.choices[0].message.reasoning_content
            end_time = time.time()
            print(f"[Task {request_idx:02d}] ✅ 请求成功, 耗时: {end_time - start_time:.2f}s. Response: {response_text.strip()}. Reasoning: {reason_text.strip()}")
            return response_text, reason_text
        except Exception as e:
            end_time = time.time()
            print(f"[Task {request_idx:02d}] ❌ 请求失败, 耗时: {end_time - start_time:.2f}s. Error: {e}")
            return f"Error: {e}", None

async def run_async_concurrency_test():
    """
    主函数，用于设置和运行并发API调用。
    """
    # --- 配置 ---
    MAX_CONCURRENCY = 3  # 最大并发请求数
    TOTAL_REQUESTS = 6   # 总共要发起的请求数
    
    print(f"\n--- 开始异步并发测试 ---")
    print(f"总请求数: {TOTAL_REQUESTS}, 最大并发数: {MAX_CONCURRENCY}")
    
    # 使用异步客户端 AsyncOpenAI
    # client = AsyncOpenAI(
    #     api_key="sk-mC2aMvB8pYgn89HF850c9dCdA2A14464957c3eFe7f895831",
    #     base_url="https://test-maas-api.cn-huabei-1.xf-yun.com/v1"
    # )#unstable
    client = AsyncOpenAI(
        api_key="sk-GlGDmDNOdyh4gAmhEc1f9d4433Ee441b8e7b03Cf5dD88a06",
        base_url="https://maas-api.cn-huabei-1.xf-yun.com/v1"
    )
    # client = AsyncOpenAI(
    #         api_key="527bdfdb89f9191848ef21902f912e82:ODBlYjI2ZTVkMTdjYmVjMDc2N2M5ZGMx",
    #         base_url="https://maas-api.cn-huabei-1.xf-yun.com/v2"
    #     )
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    
    # --- 创建所有任务 ---
    tasks = [api_worker(client, semaphore, i + 1) for i in range(TOTAL_REQUESTS)]
    
    # --- 执行并计时 ---
    total_start_time = time.time()
    results = await asyncio.gather(*tasks)

    total_end_time = time.time()
    
    print("\n--- ✨ 所有并发任务已完成 ✨ ---")
    print(f"总计 {TOTAL_REQUESTS} 个请求, 在 {total_end_time - total_start_time:.2f} 秒内完成。")


if __name__ == "__main__":
    # 您可以取消下面的注释来运行原始的同步测试
    # print("--- 运行原始同步测试 ---")
    # run_synchronous_tests()
    
    # 运行新的异步并发测试
    asyncio.run(run_async_concurrency_test())




