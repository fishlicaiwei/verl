# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import torch
import time
import os # Added for file operations
from datetime import datetime
from zoneinfo import ZoneInfo # Added for timezone handling

from typing import Tuple # Added for type hinting

from openai import AsyncOpenAI
import openai # For exception handling
import requests
import json
from concurrent.futures import ThreadPoolExecutor

from verl import DataProto
from verl.ours.llm_reward import parse_judge_response, prepare_data_for_all_models

from verl.ours.prompt import SYSTEM_PROMPT

class LLMasJudgeRewardManager:
    """
    A Reward Manager designed for API-based reward functions like LLM-as-a-Judge.
    It manages concurrent API calls efficiently using asyncio.
    """

    def _get_current_time_stage(self) -> str:
        """Determines the current time stage ('high' or 'low') based on UTC+8 time."""
        # Define the timezone for China Standard Time (UTC+8)
        tz_shanghai = ZoneInfo("Asia/Shanghai")
        
        # Get the current time in the specified timezone
        now = datetime.now(tz_shanghai)
        
        is_weekend = now.weekday() >= 5  # 5 for Saturday, 6 for Sunday
        is_high_hours = now.hour >= 0 and now.hour <= 7  # 0 AM to 7 AM
        
        if is_weekend or is_high_hours:
            # return 'extra_high'
            return 'low'
        else:
            # return 'high'
            return 'low'

    def __init__(self, tokenizer, num_examine, compute_score="dsr1_think", request_timeout=600.0) -> None:
        # --- Constants for Reward Logic ---
        # Placeholder for scores that fail due to API/parsing errors. Will be imputed by the batch mean.
        self.FAILURE_PLACEHOLDER = -99999.0
        # Explicit penalty for when the policy model generates a response with an incorrect format.
        self.FORMAT_PENALTY = -1.0
        # Scale factor for normalized simple model rewards (default 1.0). 
        # Adjust this to control the weight of the auxiliary reward relative to the main DSR1 score.
        self.SIMPLE_REWARD_SCALE = 10.0
        self.CONSISTENCY_WEIGHT = 0.1 # Weight for the consistency score
        
        # Batch processing settings
        self.micro_batch_size = 64

        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.request_timeout = request_timeout
        
        # Model-specific configurations
        self.model_configs = {
            "internlm2": {"port": 30002, "score_key": "embedding", "scaling_func": lambda s: s, "max_concurrency": 5},
            "skywork":   {"port": 30005, "score_key": "embedding", "scaling_func": lambda s: s, "max_concurrency": 5},
            "rmrb2":     {"port": 30006, "score_key": "embedding", "scaling_func": lambda s: s, "max_concurrency": 5},
            "skyworkqwen":{"port": 30008, "score_key": "embedding", "scaling_func": lambda s: s, "max_concurrency": 5},
        }

        # New parsing for multi-task, multi-model compute_score string (e.g., "dsr1-dsr1_think-consistency")
        parts = compute_score.split("_")
        self.reward_model_names = parts[0].split("-")  # e.g., ['dsr1', 'dsr1']
        type_parts = parts[-1].split("-")
        
        # Create a list of score modes, one for each model instance
        self.score_modes = []
        if len(type_parts) == 1:
             # Broadcast a single type to all models
             self.score_modes = [type_parts[0].upper()] * len(self.reward_model_names)
        elif len(type_parts) == len(self.reward_model_names):
             # Map types 1-to-1 if the count matches
             self.score_modes = [t.upper() for t in type_parts]
        else:
             # Fallback if there's a mismatch
             print(f"Warning: Mismatch between number of reward models ({len(self.reward_model_names)}) and reward types ({len(type_parts)}). Defaulting all models to 'think' scoring mode.")
             self.score_modes = ['THINK'] * len(self.reward_model_names)

        print(f"Initialized Reward Manager for models: {self.reward_model_names} with score modes: {self.score_modes}")

        # 1. Determine time stage by calling the helper method
        self.time_stage = self._get_current_time_stage()
        print(f"Initial time stage set to: '{self.time_stage}'")

        # 2. Initialize clients and concurrency based on time stage
        self.client_high = None
        self.client_low = None
        self.concurrency_extra_high = 500
        self.concurrency_high = 500
        self.concurrency_low = 150
        
        # Initialize all clients that might be needed.
        # The 'dsr1' type corresponds to the llmasjudge async flow.
        if "dsr1" in self.reward_model_names:
            # Low-speed client (cluster) is always available
            self.client_low = AsyncOpenAI(
                api_key="527bdfdb89f9191848ef21902f912e82:ODBlYjI2ZTVkMTdjYmVjMDc2N2M5ZGMx",
                base_url="https://maas-api.cn-huabei-1.xf-yun.com/v2",
            )
            # High-speed client is always initialized as well, to be used when stage becomes 'high'
            self.client_high = AsyncOpenAI(
                api_key="sk-GlGDmDNOdyh4gAmhEc1f9d4433Ee441b8e7b03Cf5dD88a06",
                base_url="https://maas-api.cn-huabei-1.xf-yun.com/v1",
            )
        
        # ... (rest of the __init__ method is unchanged)

    async def _call_api_with_retry(self, client: AsyncOpenAI, messages: list, sample_index: int) -> str:
        """
        Calls a specific OpenAI-compatible API with a retry mechanism.
        """
        max_retries = 3
        retry_delay = 2  # seconds
        last_exception = None

        # Determine model based on which client is being used
        model_name = "xdeepseekv3devbo" if client == self.client_high else "xdeepseekr1"
        
        for attempt in range(max_retries):
            try:
                completion = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=8196,
                    timeout=self.request_timeout,
                )
                
                response_content = completion.choices[0].message.content
                reasoning_content = completion.choices[0].message.reasoning_content
                return response_content if response_content and response_content.strip() else reasoning_content

            except openai.APIConnectionError as e:
                last_exception = e
                print(f"Connection error on attempt {attempt + 1}/{max_retries} for sample {sample_index}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            except Exception as e:
                error_message = f"Error calling LLM judge for sample {sample_index} (non-retriable): {e}"
                print(error_message)
                return error_message # Will be parsed as a failure
        
        final_error_message = f"LLM judge call failed for sample {sample_index} after {max_retries} retries. Last error: {last_exception}"
        print(final_error_message)
        return final_error_message

    async def _call_dsr1_api_async(self, api_requests: list, model_name: str) -> list:
        """
        Handles API calls for 'dsr1' type models, including high/low speed client logic and concurrency.
        """
        tasks = []
        async def task_wrapper(request, client, semaphore):
            async with semaphore:
                return await self._call_api_with_retry(client, request['messages'], request['index'])

        if self.time_stage == 'high' and self.client_high is not None:
            print(f"[{model_name}] Running in 'high' mode: using both clients.")
            sem_high = asyncio.Semaphore(self.concurrency_high)
            sem_low = asyncio.Semaphore(self.concurrency_low)
            total_ratio = self.concurrency_high + self.concurrency_low
            for i, req in enumerate(api_requests):
                tasks.append(task_wrapper(req, self.client_high if i % total_ratio < self.concurrency_high else self.client_low, sem_high if i % total_ratio < self.concurrency_high else sem_low))
        
        elif self.time_stage == 'extra_high' and self.client_high is not None:
            print(f"[{model_name}] Running in 'extra_high' mode: using both clients with higher speed.")
            sem_extra_high = asyncio.Semaphore(self.concurrency_extra_high)
            sem_low = asyncio.Semaphore(self.concurrency_low)
            total_ratio = self.concurrency_extra_high + self.concurrency_low
            for i, req in enumerate(api_requests):
                tasks.append(task_wrapper(req, self.client_high if i % total_ratio < self.concurrency_extra_high else self.client_low, sem_extra_high if i % total_ratio < self.concurrency_extra_high else sem_low))
        else:
            print(f"[{model_name}] Running in 'low' mode: using low-speed client only.")
            sem_low = asyncio.Semaphore(self.concurrency_low)
            tasks = [task_wrapper(req, self.client_low, sem_low) for req in api_requests]
        
        raw_responses = await asyncio.gather(*tasks, return_exceptions=True)
        print(f"[{model_name}] All API calls completed.")
        
        parsed_results = []
        for i, res in enumerate(raw_responses):
            if isinstance(res, Exception):
                parsed_results.append((self.FAILURE_PLACEHOLDER, "", ""))
            else:
                # The 'score_tag' is now correctly passed with each request
                parsed_results.append(parse_judge_response(res, api_requests[i]['score_tag'], default_value=self.FAILURE_PLACEHOLDER))
        
        # Return a list of tuples, as expected by the calling function
        return parsed_results
    
    def _send_batch_request_sync(self, batch_requests: list, model_name: str) -> list:
        """
        Sends a batch of synchronous requests to the local endpoint with retry mechanism.
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unsupported model name for sync request: {model_name}")

        config = self.model_configs[model_name]
        HOST_URL = f"http://127.0.0.1:{config['port']}"
        INFERENCE_URL = HOST_URL + "/classify"
        
        messages = [req['messages'] for req in batch_requests]
        request_data = {"text": messages}
        headers = {"Content-Type": "application/json"}

        max_retries = 3 # Can make this configurable later if needed
        retry_delay = 2 # seconds
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = requests.post(INFERENCE_URL, json=request_data, headers=headers, timeout=60.0)
                response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
                response_data = response.json()
                
                scores = []
                if isinstance(response_data, list):
                    for item in response_data:
                        val = item.get(config['score_key']) if isinstance(item, dict) else None
                        if isinstance(val, list) and val:
                            scores.append(config['scaling_func'](float(val[0])))
                        elif isinstance(val, (int, float)):
                            scores.append(config['scaling_func'](float(val)))
                        else:
                            scores.append(self.FAILURE_PLACEHOLDER)
                else:
                    print(f"Warning: Unexpected response format from {model_name}. Batch failed.")
                    scores = [self.FAILURE_PLACEHOLDER] * len(batch_requests)
                
                if len(scores) != len(batch_requests):
                    print(f"Warning: Batch size mismatch for {model_name}. Sent {len(batch_requests)}, got {len(scores)}.")
                    if len(scores) < len(batch_requests):
                        scores.extend([self.FAILURE_PLACEHOLDER] * (len(batch_requests) - len(scores)))
                    else:
                        scores = scores[:len(batch_requests)]
                
                return scores

            except requests.exceptions.RequestException as e: # Catch network errors, timeouts, HTTP errors
                last_exception = e
                print(f"Request error on attempt {attempt + 1}/{max_retries} for batch (model: {model_name}, size: {len(batch_requests)}). Retrying in {retry_delay}s... Error: {e}")
                time.sleep(retry_delay)
            except Exception as e: # Catch other unexpected errors during JSON parsing, etc.
                last_exception = e
                print(f"Unexpected error on attempt {attempt + 1}/{max_retries} for batch (model: {model_name}, size: {len(batch_requests)}). Retrying in {retry_delay}s... Error: {e}")
                time.sleep(retry_delay)
        
        # If all retries fail
        print(f"Batch request failed for model {model_name} after {max_retries} retries. Last error: {last_exception}")
        return [self.FAILURE_PLACEHOLDER] * len(batch_requests)

    async def _call_request_with_retry_async(self, api_requests: list, model_name: str) -> list:
        """
        Uses a ThreadPoolExecutor to send requests concurrently using batching.
        """
        print(f"Sending {len(api_requests)} requests to {model_name}/classify with batch size {self.micro_batch_size}...")
        
        loop = asyncio.get_running_loop()
        
        # Create batches
        batches = [api_requests[i:i + self.micro_batch_size] for i in range(0, len(api_requests), self.micro_batch_size)]
        
        # Use model-specific concurrency limit
        # The pool size determines how many batch requests are sent in parallel.
        # e.g., if max_concurrency is 16, we can process 16 * 64 items at once.
        max_workers = self.model_configs[model_name].get('max_concurrency', 4)
        max_workers = min(max_workers, len(batches)) if batches else 1
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            blocking_tasks = [
                loop.run_in_executor(executor, self._send_batch_request_sync, batch, model_name)
                for batch in batches
            ]
            batch_results = await asyncio.gather(*blocking_tasks)
            
        # Flatten the list of lists
        all_scores = [score for batch in batch_results for score in batch]
        return all_scores

    async def verify(self, data: DataProto) -> Tuple[list, list, list, list, list]:
        """
        Computes raw scores for a batch for ALL reward models concurrently.
        Manages concurrency by grouping requests for the same model type.
        """
        prompts_str = [text.split(SYSTEM_PROMPT)[-1].strip() for text in self.tokenizer.batch_decode(data.batch['prompts'], skip_special_tokens=True)]
        responses_str = self.tokenizer.batch_decode(data.batch['responses'], skip_special_tokens=True)
        ground_truth = [item.non_tensor_batch['reward_model']['ground_truth'] for item in data]
        data_sources_list = data.non_tensor_batch['data_source'] # Renamed to avoid conflict with `data` parameter
        
        model_requests_map, invalid_format_mask_list, think_lens, answer_lens = prepare_data_for_all_models(
            data_sources_list, prompts_str, responses_str, ground_truth, self.reward_model_names, self.score_modes
        )
        invalid_format_mask = torch.tensor(invalid_format_mask_list, dtype=torch.bool)
        
        # --- Group requests by model name to share concurrency pools ---
        grouped_requests = {model_name: [] for model_name in set(self.reward_model_names)}
        for i, model_name in enumerate(self.reward_model_names):
            unique_key = f"{model_name}_{i}"
            grouped_requests[model_name].extend(model_requests_map.get(unique_key, []))

        # --- Spawn API Tasks ---
        tasks = []
        for model_name, api_requests in grouped_requests.items():
            if not api_requests:
                print(f"[{model_name}] Warning: No requests prepared.")
                continue

            async def call_model_api(m_name, reqs):
                print(f"[{m_name}] invalid content format count in batch: {invalid_format_mask.sum().item()}")
                if m_name == "dsr1":
                    # For dsr1, we get back a list of (score, rubric, justify) tuples
                    results = await self._call_dsr1_api_async(reqs, m_name)
                elif m_name in ["internlm2", "skywork", "rmrb2", "skyworkqwen"]:
                    scores_list = await self._call_request_with_retry_async(reqs, m_name)
                    # Simple models don't return rubric/justify, so we create placeholder results
                    results = [(score, "", "") for score in scores_list]
                elif m_name == "rule":
                    # For rule-based models, we process them locally based on score_mode
                    results = [None] * len(reqs)
                    
                    # Group requests by their specific rule type (score_mode)
                    from collections import defaultdict
                    requests_by_mode = defaultdict(list)
                    for i, req in enumerate(reqs):
                        requests_by_mode[req['score_mode']].append((i, req))

                    # Process MATH rules
                    if 'MATH' in requests_by_mode:
                        from verl.ours.math_verify_reward import compute_math_reward
                        
                        math_req_indices = [item[0] for item in requests_by_mode['MATH']]
                        math_reqs = [item[1] for item in requests_by_mode['MATH']]
                        
                        # The 'answer' part of the response is in the last message's content
                        predictions = [r['messages'][-1]['content'] for r in math_reqs]
                        references = [r['ground_truth'] for r in math_reqs]
                        
                        math_scores = compute_math_reward(predictions, references)
                        
                        for i, score in zip(math_req_indices, math_scores):
                            results[i] = (score, "math_verify", "math_verify") # Use placeholder rubric/justify

                    # Future 'CODE' or other rules can be added as 'elif' blocks here
                    
                    # Fill any unprocessed rules with a failure score
                    for i in range(len(results)):
                        if results[i] is None:
                            results[i] = (self.FAILURE_PLACEHOLDER, "unknown_rule", "unknown_rule")
                else:
                    raise ValueError(f"Unknown reward model name in verify: {m_name}")
                
                # --- Unpack results back to original per-mode/per-instance lists ---
                # This is complex because requests were shuffled. We must use the 'index'
                # and 'score_mode' from the original request to put results back in order.
                # Initialize unpacked_results to hold results for each model instance, in correct order
                unpacked_results = {}
                for j, m_n in enumerate(self.reward_model_names):
                    if m_n == m_name: # Only create entries for instances of the current model_name
                        unpacked_results[f"{m_name}_{j}"] = [None] * len(data)

                for i, req in enumerate(reqs):
                    original_index = req['index']
                    score_mode = req['score_mode']
                    
                    # Find the original unique_key this request belonged to
                    original_instance_idx = -1
                    for j, m_n in enumerate(self.reward_model_names):
                        s_m = self.score_modes[j]
                        if m_n == m_name and s_m == score_mode:
                            original_instance_idx = j
                            break
                    
                    if original_instance_idx != -1:
                        original_unique_key = f"{m_name}_{original_instance_idx}"
                        if original_unique_key in unpacked_results and original_index < len(unpacked_results[original_unique_key]):
                            unpacked_results[original_unique_key][original_index] = results[i]
                        else:
                            print(f"Warning: Could not place result for request {i} (model: {m_name}, mode: {score_mode}, original_index: {original_index}). Key or index out of bounds.")
                    else:
                        print(f"Warning: Could not find original instance key for request {i} (model: {m_name}, mode: {score_mode}).")


                return unpacked_results

            tasks.append(call_model_api(model_name, api_requests))
        
        # Run all model tasks concurrently
        all_grouped_results = await asyncio.gather(*tasks)
        
        # --- Final Re-ordering ---
        # `all_grouped_results` is a list of dicts. We need to flatten it into a single list
        # that matches the original order of `self.reward_model_names`.
        final_results_list = [None] * len(self.reward_model_names)
        for group_result_dict in all_grouped_results:
            for key, results_list in group_result_dict.items():
                # key is e.g., "dsr1_0"
                model_name, index_str = key.rsplit('_', 1)
                original_index = int(index_str)
                
                # Re-zip the per-sample results back into lists of (scores, rubrics, justifies)
                scores_list, rubrics_list, justifies_list = [], [], []
                for res in results_list:
                    if res is None: # Handle cases where a sample might have failed
                        s, r, j = self.FAILURE_PLACEHOLDER, "", ""
                    else:
                        s, r, j = res
                    scores_list.append(s)
                    rubrics_list.append(r)
                    justifies_list.append(j)

                raw_scores_tensor = torch.tensor(scores_list, dtype=torch.float32)
                rm_error_mask = (raw_scores_tensor == self.FAILURE_PLACEHOLDER)
                log_key = f"{model_name}_{self.score_modes[original_index]}"
                print(f"[{log_key}] reward model system error count in batch: {rm_error_mask.sum().item()}")

                final_results_list[original_index] = (raw_scores_tensor, rm_error_mask, invalid_format_mask, rubrics_list, justifies_list)

        return final_results_list, invalid_format_mask, think_lens, answer_lens, data_sources_list

    def _save_dataset_records(self, prompts_str, responses_str, all_results_tuple):
        """
        Saves one consolidated record per sample, containing all scores and a single rubric/justify.
        """
        all_results_list, invalid_format_mask, think_lens, answer_lens, data_sources_list = all_results_tuple # Unpack the tuple
        
        records_to_save = []
        # Dynamic save file name based on data source
        save_file_base = "320_dsr1_math_hotstart_rubric_dataset"
        # save_file_base = "dsr1_think_rubric_dataset"
        # Determine if all data sources are the same
        if len(data_sources_list) > 0 and all(ds == data_sources_list[0] for ds in data_sources_list):
            save_file = f"{save_file_base}_{data_sources_list[0].replace(' ', '_').replace('-', '_').lower()}.json"
        else:
            save_file = f"{save_file_base}_mixed_batch.json"

        num_samples = len(prompts_str)
        if num_samples == 0:
            return

        for i in range(num_samples):
            # --- Find the single Rubric/Justify ---
            rubric_text, justify_text = "", ""
            for j, model_name in enumerate(self.reward_model_names):
                score_mode = self.score_modes[j]
                # Try to get rubric/justify from any DSR1 non-CONSISTENCY score
                if model_name == "dsr1" and score_mode != 'CONSISTENCY':
                    # Need to check if `all_results_list[j]` is not None or if its elements are valid
                    if j < len(all_results_list) and all_results_list[j] is not None:
                        _, _, _, rubrics, justifies = all_results_list[j]
                        if i < len(rubrics) and rubrics[i].strip():
                            rubric_text = rubrics[i]
                            justify_text = justifies[i]
                            break # Found it, exit loop
            
            # If still no rubric, try to get from CONSISTENCY score for DSR1 (as a fallback, user said CONSISTENCY doesn't produce it)
            if not rubric_text:
                for j, model_name in enumerate(self.reward_model_names):
                    score_mode = self.score_modes[j]
                    if model_name == "dsr1" and score_mode == 'CONSISTENCY':
                        if j < len(all_results_list) and all_results_list[j] is not None:
                            _, _, _, rubrics, justifies = all_results_list[j]
                            if i < len(rubrics) and rubrics[i].strip():
                                rubric_text = rubrics[i]
                                justify_text = justifies[i]
                                break


            if not rubric_text: # Only save if we found a valid rubric
                continue

            # --- Collect all scores for this sample ---
            scores_dict = {}
            for j, model_name in enumerate(self.reward_model_names):
                score_mode = self.score_modes[j]
                if all_results_list[j] is None: continue # Skip if no results for this model instance
                raw_scores, _, _, _, _ = all_results_list[j]
                
                score_val = raw_scores[i].item()
                if score_val != self.FAILURE_PLACEHOLDER:
                    # Use a unique key combining model name and score mode to prevent collisions
                    log_key = f"{model_name}_{score_mode}"
                    scores_dict[log_key.lower()] = score_val

            if scores_dict:
                record = {
                    "prompt": prompts_str[i],
                    "response": responses_str[i],
                    "rubric": rubric_text,
                    "justify": justify_text,
                    "think_len": think_lens[i] if i < len(think_lens) else 0, # Add lengths to record
                    "answer_len": answer_lens[i] if i < len(answer_lens) else 0,
                    "scores": scores_dict
                }
                records_to_save.append(record)

        if records_to_save:
            try:
                existing_data = []
                if os.path.exists(save_file):
                    with open(save_file, 'r', encoding='utf-8') as f:
                        try:
                            existing_data = json.load(f)
                            if not isinstance(existing_data, list): existing_data = []
                        except json.JSONDecodeError: existing_data = []
                
                existing_data.extend(records_to_save);
                
                with open(save_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"Error saving dataset: {e}")

    def __call__(self, data: DataProto):
        """
        Main entry point. Orchestrates parallel scoring and returns a dict compatible with the trainer.
        """
        call_start_time = time.time()
        print(f"start {self.reward_model_names} reward manager call...")
        if 'rm_scores' in data.batch and isinstance(data.batch['rm_scores'], dict):
            return data.batch['rm_scores']
        self.time_stage = self._get_current_time_stage()
        # Unpack here
        all_results_list, invalid_format_mask, think_lens, answer_lens, data_sources_list = asyncio.run(self.verify(data))

        all_reward_tensors = {}
        processed_scores_list = []

        # Process results for each model instance, applying penalties and imputation
        for i, model_name in enumerate(self.reward_model_names):
            # A result might be None if a model group had no requests
            if all_results_list[i] is None:
                processed_scores_list.append(None)
                continue
            
            raw_scores, rm_error_mask, invalid_format_mask_per_model, _, _ = all_results_list[i]
            
            penalty_mask = invalid_format_mask & ~rm_error_mask
            penalized_scores = raw_scores.clone()
            penalized_scores[penalty_mask] += self.FORMAT_PENALTY
            
            valid_mask = ~rm_error_mask
            valid_scores_mean = penalized_scores[valid_mask].mean().item() if valid_mask.any() else 5.0
            
            final_scores = torch.where(rm_error_mask, valid_scores_mean, penalized_scores)
            processed_scores_list.append(final_scores)

            unique_key = f"{model_name}_{self.score_modes[i]}"
            all_reward_tensors[f'invalid_format_cnt_{unique_key}'] = invalid_format_mask_per_model.sum().item()

        # --- Aggregation and preparation for Trainer ---
        # total_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        if self.reward_model_names:
            prompt_length = data.batch['prompts'].shape[-1]
            valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)

            for i, model_name in enumerate(self.reward_model_names):
                if processed_scores_list[i] is None: continue
                
                scores_tensor = processed_scores_list[i]
                score_mode = self.score_modes[i]
                
                # Create sparse reward tensor for this model instance
                reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
                for j in range(len(data)):
                    if valid_response_length[j].item() > 0:
                        reward_tensor[j, valid_response_length[j].item() - 1] = scores_tensor[j]

                # The log key should be unique based on model and score mode.
                log_key = f"{model_name}_{score_mode}"

                # Warning for potential overwrite if a user accidentally configures duplicates.
                if log_key in all_reward_tensors:
                    print(f"Warning: Duplicate reward key '{log_key}' detected. This may be due to using the same model with the same score_mode multiple times. The score will be overwritten.")

                all_reward_tensors[log_key] = reward_tensor

                # Apply weights and normalization for the final aggregated 'reward_score'
                if model_name == "dsr1":
                    weight = self.CONSISTENCY_WEIGHT if score_mode == 'CONSISTENCY' else 1.0
                    # total_reward_tensor += reward_tensor * weight
                else: # Simple models
                    valid_mask = reward_tensor != 0
                    if valid_mask.any():
                        valid_scores = reward_tensor[valid_mask]
                        min_val, max_val = valid_scores.min(), valid_scores.max()
                        print(f"[{model_name}_{score_mode}] Valid score range: min={min_val:.4f}, max={max_val:.4f}")
                        if max_val > min_val:
                            norm_scores = (valid_scores - min_val) / (max_val - min_val + 1e-6)
                            norm_tensor = torch.zeros_like(reward_tensor)
                            norm_tensor[valid_mask] = norm_scores
                            # total_reward_tensor += norm_tensor * self.SIMPLE_REWARD_SCALE
            
            # all_reward_tensors['reward_score'] = total_reward_tensor

        total_batch_time = time.time() - call_start_time
        print(f"\nBatch Processing Summary (Size: {len(data)}, Total time: {total_batch_time:.2f}s)")

        # Add average lengths to returned dict
        if think_lens:
            all_reward_tensors['avg_think_len'] = sum(think_lens) / len(think_lens)
        else:
            all_reward_tensors['avg_think_len'] = 0.0
        if answer_lens:
            all_reward_tensors['avg_answer_len'] = sum(answer_lens) / len(answer_lens)
        else:
            all_reward_tensors['avg_answer_len'] = 0.0
        
        prompts_str_for_debug = [text.split(SYSTEM_PROMPT)[-1].strip() for text in self.tokenizer.batch_decode(data.batch['prompts'], skip_special_tokens=True)]
        responses_str_for_debug = self.tokenizer.batch_decode(data.batch['responses'], skip_special_tokens=True)

        # --- Enhanced Debug Printing ---
        if self.num_examine > 0:
            for i in range(min(self.num_examine, len(data))):
                print("-" * 20)
                print(f"[Prompt {i}]: {prompts_str_for_debug[i]}")
                print(f"[Response {i}]: {responses_str_for_debug[i]}")
                for j, model_name in enumerate(self.reward_model_names):
                    if processed_scores_list[j] is None: continue
                    score_mode = self.score_modes[j]
                    final_score = processed_scores_list[j][i].item()
                    _, rm_error_mask_per_model, _, _, _ = all_results_list[j] # Use per-model error mask here

                    info = "(Imputed)" if rm_error_mask_per_model[i] else "(Penalized)" if invalid_format_mask[i] else ""
                    print(f"  - [Score - {model_name}_{score_mode}]: {final_score:.2f} {info}")
                
                prompt_len = data.batch['prompts'][i].shape[-1]
                resp_len = data.batch['attention_mask'][i, prompt_len:].sum().item()
                if resp_len > 0 and 'reward_score' in all_reward_tensors:
                    final_agg_reward = all_reward_tensors['reward_score'][i, resp_len - 1].item()
                    print(f"  - [Aggregated Reward]: {final_agg_reward:.2f}")
                print(f"  - [Think Char Len]: {think_lens[i] if i < len(think_lens) else 'N/A'}")
                print(f"  - [Answer Char Len]: {answer_lens[i] if i < len(answer_lens) else 'N/A'}")
                print("-" * 20)
        
        self._save_dataset_records(prompts_str_for_debug, responses_str_for_debug, (all_results_list, invalid_format_mask, think_lens, answer_lens, data_sources_list))

        return all_reward_tensors