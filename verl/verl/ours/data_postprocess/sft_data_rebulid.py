import asyncio
import json
import os
import argparse
import time
import pandas as pd
import aiohttp
from typing import List, Dict, Any, Tuple
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from datetime import datetime
from zoneinfo import ZoneInfo

# --- Imports from Project Structure ---
try:
    from verl.ours.prompt import SYSTEM_PROMPT, PROMPT_TEMPLATES
except ImportError:
    raise ImportError("Failed to import prompts from verl.ours.prompt. Please ensure you are in the project root.")

# --- Configuration & Constants ---
# Keys from llmasjudge.py
API_KEY_LOW = "527bdfdb89f9191848ef21902f912e82:ODBlYjI2ZTVkMTdjYmVjMDc2N2M5ZGMx"
BASE_URL_LOW = "https://maas-api.cn-huabei-1.xf-yun.com/v2"

API_KEY_HIGH = "sk-GlGDmDNOdyh4gAmhEc1f9d4433Ee441b8e7b03Cf5dD88a06"
BASE_URL_HIGH = "https://maas-api.xf-yun.com/v1"

# SGLang Configuration
SGLANG_BASE_URL = "http://localhost:30000/v1"
SGLANG_MODEL = "default"

# --- Domain/Task Type Mapping (From llm_reward.py) ---
MATH_DATA_SOURCES = {'RLVR-MATH','math_500','gsm8k','NuminaMath-CoT'}
SAFETY_DATA_SOURCES = {'PKU-SafeRLHF','wildguard','coconot'}
IF_DATA_SOURCES = {'tulu-3-IF-augmented-on-policy-8b','rlvr_ifeval','dolci_rl_zero_if_7b'}
KNOWLEDGE_DATA_SOURCES = {'SciRIFF-train-mix','natural_questions','squad_v2_val'}
CHAT_DATA_SOURCES = {'tulu-3-wildchat-if-on-policy-8b','tulu-3-ultrafeedback-cleaned-on-policy-8b','OpenAssistant-oasst1'}
CODE_DATA_SOURCES = {'tulu-3-sft-personas-code','codealpaca_20k','openai_humaneval'}
WRITTING_DATA_SOURCES = {'DeepWriting-20K','llmaes_writingprompts_val'}
AGENT_DATA_SOURCES = {'xlam-function-calling-60k','function_calling_irrelevant'}

def get_task_type(data_source: str) -> str:
    """Map data source to task type to select correct prompt."""
    if data_source in MATH_DATA_SOURCES: return 'REASONING'
    if data_source in CODE_DATA_SOURCES: return 'REASONING'
    if data_source in SAFETY_DATA_SOURCES: return 'SAFETY'
    if data_source in IF_DATA_SOURCES: return 'IF'
    if data_source in KNOWLEDGE_DATA_SOURCES: return 'CHAT'
    if data_source in CHAT_DATA_SOURCES: return 'CHAT'
    if data_source in WRITTING_DATA_SOURCES: return 'CHAT'
    if data_source in AGENT_DATA_SOURCES: return 'AGENT'
    return 'CHAT' # Default

# --- Helpers ---

def extract_think_content(response: str) -> str:
    """Extract content inside <think> tags."""
    start_tag = "<think>"
    end_tag = "</think>"
    start_idx = response.find(start_tag)
    end_idx = response.rfind(end_tag)
    if start_idx != -1 and end_idx != -1:
        return response[start_idx + len(start_tag):end_idx].strip()
    return response.strip()

def parse_judge_response_strict(response_text: str, default_score: float = -1.0) -> Tuple[float, str, str]:
    """Parses Score, Rubric, and Justification from Judge Output."""
    if not response_text:
        return default_score, "", ""

    # Extract Rubric
    rubric_text = ""
    r_start = response_text.find("<rubric>")
    r_end = response_text.find("</rubric>")
    if r_start != -1 and r_end != -1:
        rubric_text = response_text[r_start+8:r_end].strip()

    # Extract Eval/Critique
    # Note: Depending on the prompt, this might be <eval> or <justify> or just text
    # Assuming <eval> based on prompt.py templates usually used
    justify_text = ""
    for tag in ["<eval>", "<justify>"]:
        j_start = response_text.find(tag)
        j_end = response_text.find(tag.replace("<", "</"))
        if j_start != -1 and j_end != -1:
            justify_text = response_text[j_start+len(tag):j_end].strip()
            break

    # Extract Score
    score = default_score
    # Try multiple tags if prompt varies, but standard is usually <think_score> or <score>
    for tag_name in ["think_score", "outcome_score", "score"]:
        start_tag = f"<{tag_name}>"
        end_tag = f"</{tag_name}>"
        end_idx = response_text.rfind(end_tag)
        if end_idx != -1:
            start_idx = response_text.rfind(start_tag, 0, end_idx)
            if start_idx != -1:
                val_str = response_text[start_idx + len(start_tag):end_idx].strip()
                try:
                    s = float(val_str)
                    if s > 10.0: s /= 10.0
                    score = s
                    break # Found valid score
                except: continue
    
    return score, rubric_text, justify_text

def get_current_time_stage() -> str:
    """Time-based stage determination (UTC+8)."""
    try:
        tz_shanghai = ZoneInfo("Asia/Shanghai")
        now = datetime.now(tz_shanghai)
    except:
        now = datetime.now()
    is_weekend = now.weekday() >= 5
    is_high_hours = 0 <= now.hour <= 7
    return 'extra_high' if is_weekend or is_high_hours else 'high'

class SFTDataRebuilder:
    def __init__(self, args):
        self.args = args
        self.sglang_sem = asyncio.Semaphore(args.rollout_concurrency)
        
        # Dual Client Setup
        self.client_low = AsyncOpenAI(api_key=API_KEY_LOW, base_url=BASE_URL_LOW)
        self.client_high = AsyncOpenAI(api_key=API_KEY_HIGH, base_url=BASE_URL_HIGH)
        
        # Concurrency Settings (from llmasjudge.py)
        self.concurrency_extra_high = 500
        self.concurrency_high = 500
        self.concurrency_low = 150
        
        # Semaphores
        self.sem_high = asyncio.Semaphore(self.concurrency_high) # Also used for extra_high cap
        self.sem_low = asyncio.Semaphore(self.concurrency_low)

    async def rollout_one(self, session, prompt: str) -> str:
        """SGLang Rollout."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        payload = {
            "model": SGLANG_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        async with self.sglang_sem:
            try:
                async with session.post(f"{SGLANG_BASE_URL}/chat/completions", json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data['choices'][0]['message']['content']
                    return ""
            except Exception as e:
                print(f"SGLang Error: {e}")
                return ""

    async def _call_judge_api(self, client: AsyncOpenAI, messages: List[Dict], semaphore: asyncio.Semaphore) -> str:
        """Raw API call with semaphore and retry. Includes content validation."""
        async with semaphore:
            model_name = "xdeepseekv3devbo" if client == self.client_high else "xdeepseekr1"
            
            for attempt in range(3):
                try:
                    completion = await client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=4096,
                        timeout=300
                    )
                    content = completion.choices[0].message.content
                    
                    # --- Content Validation ---
                    # If the response doesn't look like XML/Rubric format, force a retry
                    # We check for at least one critical tag.
                    if "<rubric>" not in content and "rubric" not in content.lower():
                        # Raise error to trigger the except block and retry
                        raise ValueError("Malformed response: missing <rubric> tag")
                        
                    return content
                    
                except Exception as e:
                    # Only print verbose errors if it's the last attempt
                    if attempt == 2:
                        print(f"Judge API Failed after 3 retries. Last error: {e}")
                    await asyncio.sleep(2)
            return ""

    async def judge_batch(self, tasks: List[Dict]) -> List[str]:
        """
        Distribute judge tasks to High/Low clients based on Time Stage.
        tasks: List of {'messages': ..., 'index': ...}
        """
        time_stage = get_current_time_stage()
        print(f"Judge Batch ({len(tasks)} items) - Stage: {time_stage}")
        
        async_tasks = []
        
        if time_stage == 'extra_high':
            # Use High (up to extra limit) + Low
            limit_high = self.concurrency_extra_high
            total_ratio = limit_high + self.concurrency_low
            
            for i, task_data in enumerate(tasks):
                if i % total_ratio < limit_high:
                    async_tasks.append(self._call_judge_api(self.client_high, task_data['messages'], self.sem_high))
                else:
                    async_tasks.append(self._call_judge_api(self.client_low, task_data['messages'], self.sem_low))
                    
        elif time_stage == 'high':
            # Use High (normal limit) + Low
            limit_high = self.concurrency_high
            total_ratio = limit_high + self.concurrency_low
            
            for i, task_data in enumerate(tasks):
                if i % total_ratio < limit_high:
                    async_tasks.append(self._call_judge_api(self.client_high, task_data['messages'], self.sem_high))
                else:
                    async_tasks.append(self._call_judge_api(self.client_low, task_data['messages'], self.sem_low))
                    
        else: # Low stage
            # Use Low only
            for task_data in tasks:
                async_tasks.append(self._call_judge_api(self.client_low, task_data['messages'], self.sem_low))
        
        return await asyncio.gather(*async_tasks)

    def build_judge_messages(self, item: Dict, response: str) -> List[Dict]:
        """Construct prompt using PROMPT_TEMPLATES based on Data Source."""
        data_source = item.get('data_source', 'unknown')
        task_type = get_task_type(data_source)
        
        # Select Template (Defaulting to THINK mode as we want Rubric/Process eval)
        template_key = f"{task_type}_THINK"
        template = PROMPT_TEMPLATES.get(template_key, PROMPT_TEMPLATES['CHAT_THINK'])
        
        prompt_content = template['prompt']
        
        # Prepare Input
        think_str = extract_think_content(response)
        ground_truth = item.get('ability', item.get('answer', item.get('ground_truth', "No ground truth provided.")))
        
        # Format user input using the template's 'input' field
        input_content = template['input'].format(
            client_question=item['prompt'],
            think_str=think_str,
            ground_truth=ground_truth
        )
        
        return [
            {"role": "system", "content": prompt_content},
            {"role": "user", "content": input_content}
        ]

    async def process_micro_batch(self, session, batch_data: List[Dict]):
        # 1. Rollout (All items)
        prompts = [item['prompt'] for item in batch_data]
        rollout_tasks = [self.rollout_one(session, p) for p in prompts]
        responses = await asyncio.gather(*rollout_tasks)
        
        # 2. Prepare Judge Tasks (Only for valid responses)
        judge_payloads = []
        valid_indices = []
        
        for i, r in enumerate(responses):
            if r:
                msgs = self.build_judge_messages(batch_data[i], r)
                judge_payloads.append({'messages': msgs, 'index': i})
                valid_indices.append(i)
        
        if not judge_payloads:
            return []
            
        # 3. Execute Judge (Dual Client Logic)
        judge_outputs = await self.judge_batch(judge_payloads)
        
        # 4. Assemble Results
        results = []
        for j_idx, raw_output in enumerate(judge_outputs):
            if raw_output:
                original_idx = valid_indices[j_idx]
                item = batch_data[original_idx].copy()
                
                score, rubric, eval_text = parse_judge_response_strict(raw_output)
                
                item['response'] = responses[original_idx]
                item['rubric'] = rubric
                # item['critique'] = eval_text # User requested to remove this
                item['score'] = score
                
                # Filter: Keep if we got a score OR a rubric (relaxed filter)
                if score != -1.0 or rubric:
                    results.append(item)
                    
        return results

    async def run_pipeline(self):
        # Load Data
        if self.args.input_file.endswith('.parquet'):
            df = pd.read_parquet(self.args.input_file)
        elif self.args.input_file.endswith('.jsonl'):
             df = pd.read_json(self.args.input_file, lines=True)
        else:
            raise ValueError("Use .parquet or .jsonl")

        if self.args.limit > 0:
            df = df.head(self.args.limit)
            
        records = df.to_dict('records')
        print(f"Processing {len(records)} records...")
        
        # Check required columns
        if 'data_source' not in records[0] and 'data_source' not in df.columns:
            print("Warning: 'data_source' column missing. Defaulting to 'CHAT' prompts.")

        # Resume Logic
        output_file = self.args.output_file
        processed_prompts = set()
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        processed_prompts.add(d['prompt'])
                    except: pass
        
        to_process = [r for r in records if r['prompt'] not in processed_prompts]
        print(f"Remaining: {len(to_process)}")

        # Micro-batching
        chunk_size = self.args.micro_batch_size
        chunks = [to_process[i:i+chunk_size] for i in range(0, len(to_process), chunk_size)]
        
        async with aiohttp.ClientSession() as session:
            active_tasks = set()
            max_active = self.args.max_active_batches
            pbar = tqdm(total=len(to_process))
            
            for chunk in chunks:
                task = asyncio.create_task(self.process_micro_batch(session, chunk))
                active_tasks.add(task)
                
                if len(active_tasks) >= max_active:
                    done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                    for t in done:
                        res = await t
                        if res:
                            with open(output_file, 'a', encoding='utf-8') as f:
                                for r in res:
                                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                            pbar.update(len(res)) # Approximate update
                            
            if active_tasks:
                done, _ = await asyncio.wait(active_tasks)
                for t in done:
                    res = await t
                    if res:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            for r in res:
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                                
            pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", default="sft_rebuilt.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--micro_batch_size", type=int, default=32)
    parser.add_argument("--max_active_batches", type=int, default=5)
    parser.add_argument("--rollout_concurrency", type=int, default=64)
    # Reward concurrency is now managed by time-stage logic, but arg kept for compatibility
    parser.add_argument("--reward_concurrency", type=int, default=0)
    
    args = parser.parse_args()
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(SFTDataRebuilder(args).run_pipeline())
