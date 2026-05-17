import json
import pandas as pd
import numpy as np
import re
import torch
import torch.multiprocessing as mp
from sentence_transformers import SentenceTransformer
import xml.etree.ElementTree as ET
import math

# ==========================================
# 配置区域
# ==========================================
MODEL_PATH = '/data2/cwli16/share-models/Qwen3-Embedding-4B' 
INPUT_FILE = '/data2/jyyan10/verl-0.3.0.post1-ws/verl/dsr1_writting_rubric_dataset.json'
OUTPUT_FILE = 'clustered_writting_rubric.csv'

# 指定你的设备列表
GPU_IDS = [0, 5, 6, 7] 
# ==========================================
# 数据处理工具
# ==========================================
def parse_rubric_xml(xml_string):
    try:
        xml_string = f"<root>{xml_string}</root>"
        root = ET.fromstring(xml_string)
        parsed_items = []
        for criteria in root.findall('.//criteria'):
            name = criteria.find('name').text or ""
            weight = criteria.find('weight').text or ""
            desc = criteria.find('description').text or ""
            parsed_items.append(f"{name} (Weight: {weight}): {desc}")
        return " | ".join(parsed_items)
    except:
        return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', str(xml_string))).strip()

def load_data(filepath):
    print(f"正在读取数据: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['parsed_rubric'] = df['rubric'].apply(parse_rubric_xml)
    df['embedding_input'] = df.apply(
        lambda row: f"Query: {row['prompt']} \n Rule Criteria: {row['parsed_rubric']}", 
        axis=1
    )
    return df

# ==========================================
# 核心：独立进程工作函数
# ==========================================
def worker_process(gpu_id, sentences, output_queue):
    """
    每个进程独立加载模型，独立计算，结果放入队列
    """
    try:
        device = f"cuda:{gpu_id}"
        print(f"[GPU-{gpu_id}] 正在加载模型到 {device} ...")
        
        # ⚠️ 关键点：在子进程内部加载模型，避免主进程复制
        model = SentenceTransformer(MODEL_PATH, device=device, trust_remote_code=True)
        model.max_seq_length = 8192 
        
        print(f"[GPU-{gpu_id}] 模型加载完毕，开始计算 {len(sentences)} 条数据...")
        
        # 开始编码
        # batch_size=2 比较保险，8B模型+8k长度非常吃显存
        embeddings = model.encode(
            sentences, 
            batch_size=8, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        
        # 将结果放入队列传回主进程
        # 为了防止结果顺序混乱，我们返回 (gpu_id, embeddings)
        output_queue.put((gpu_id, embeddings))
        print(f"[GPU-{gpu_id}] 计算完成！")
        
    except Exception as e:
        print(f"[GPU-{gpu_id}] 发生错误: {e}")
        output_queue.put((gpu_id, None))

# ==========================================
# 主流程
# ==========================================
if __name__ == "__main__":
    # 必须设置启动方式为 spawn
    mp.set_start_method('spawn', force=True)

    # 1. 加载数据
    df = load_data(INPUT_FILE)
    all_sentences = df['embedding_input'].tolist()
    total_len = len(all_sentences)
    
    # 2. 数据切片 (Sharding)
    num_gpus = len(GPU_IDS)
    chunk_size = math.ceil(total_len / num_gpus)
    chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_len)
        chunks.append(all_sentences[start_idx:end_idx])
    
    print(f"数据已切分为 {num_gpus} 份，每份约 {chunk_size} 条")

    # 3. 启动多进程
    queue = mp.Queue()
    processes = []
    
    for i, gpu_id in enumerate(GPU_IDS):
        # 如果该分片没数据，就跳过
        if len(chunks[i]) == 0: continue
            
        p = mp.Process(target=worker_process, args=(gpu_id, chunks[i], queue))
        p.start()
        processes.append(p)
    
    # 4. 收集结果
    results_dict = {}
    print("等待子进程计算结果...")
    
    completed_count = 0
    while completed_count < len(processes):
        # 从队列获取结果
        gpu_id, emb = queue.get()
        if emb is not None:
            results_dict[gpu_id] = emb
        completed_count += 1
    
    # 等待所有进程彻底结束
    for p in processes:
        p.join()

    # 5. 按顺序拼接结果
    # 因为多进程返回顺序不一定，需要按切片顺序拼回去
    final_embeddings_list = []
    for i in range(len(GPU_IDS)):
        # 对应 GPU_IDS[i] 的结果
        target_gpu = GPU_IDS[i]
        if target_gpu in results_dict:
            final_embeddings_list.append(results_dict[target_gpu])
            
    if not final_embeddings_list:
        print("错误：没有计算出任何向量")
        exit()

    final_embeddings = np.vstack(final_embeddings_list)
    print(f"所有计算完成，最终向量形状: {final_embeddings.shape}")

    # 6. 后续聚类 (单机 CPU 进行即可)
    import umap
    import hdbscan
    
    print("正在进行 UMAP 降维...")
    # 注意：如果数据量太大，UMAP 可能会慢，这里用 CPU 跑没问题
    umap_embeddings = umap.UMAP(
        n_neighbors=15, n_components=10, metric='cosine', random_state=42
    ).fit_transform(final_embeddings)

    print("正在进行 HDBSCAN 聚类...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10, min_samples=1, metric='euclidean', cluster_selection_method='eom'
    )
    df['cluster'] = clusterer.fit_predict(umap_embeddings)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"搞定！结果已保存至 {OUTPUT_FILE}")