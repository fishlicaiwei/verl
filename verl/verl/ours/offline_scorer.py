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

import argparse
import asyncio
import json
import os
import time
from openai import AsyncOpenAI
import openai
from verl.ours.llm_reward import parse_judge_response
from verl.ours.prompt import PROMPT_TEMPLATES

class OfflineScorer:
    """
    A lightweight scorer to process a dataset file offline, adding new scores
    based on specified content (e.g., 'answer' or 'think').
    """
    def __init__(self, reward_model, score_mode):
        if reward_model != "dsr1":
            raise ValueError("Currently, only 'dsr1' is supported for offline scoring.")
        
        self.score_mode = score_mode.upper()
        self.reward_model = reward_model
        self.client = AsyncOpenAI(
            api_key="527bdfdb89f9191848ef21902f912e82:ODBlYjI2ZTVkMTdjYmVjMDc2N2M5ZGMx",
            base_url="https://maas-api.cn-huabei-1.xf-yun.com/v2",
        )
        self.request_timeout = 600.0
        self.concurrency = 150
        self.failure_placeholder = -99999.0

    async def _call_api_with_retry(self, client: AsyncOpenAI, messages: list, sample_index: int) -> str:
        max_retries = 3
        retry_delay = 2
        last_exception = None
        for attempt in range(max_retries):
            try:
                completion = await client.chat.completions.create(
                    model="xdeepseekr1",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=8196,
                    timeout=self.request_timeout,
                )
                response_content = completion.choices[0].message.content
                return response_content if response_content and response_content.strip() else ""
            except openai.APIConnectionError as e:
                last_exception = e
                print(f"Connection error on attempt {attempt + 1}/{max_retries} for sample {sample_index}. Retrying...")
                await asyncio.sleep(retry_delay)
            except Exception as e:
                print(f"Error calling LLM judge for sample {sample_index} (non-retriable): {e}")
                return f"Error: {e}"
        final_error_message = f"LLM judge call failed for sample {sample_index} after {max_retries} retries. Last error: {last_exception}"
        print(final_error_message)
        return final_error_message

    def _prepare_requests(self, records: list) -> list:
        api_requests = []
        for i, record in enumerate(records):
            prompt = record.get("prompt", "")
            content_to_score = record.get(self.score_mode.lower(), "")

            if not prompt or not content_to_score:
                print(f"Warning: Skipping record {i} due to missing 'prompt' or '{self.score_mode.lower()}' field.")
                continue
            
            # Using a generic CHAT_THINK template for simplicity
            template_key = f"AGENT_{self.score_mode}"
            template = PROMPT_TEMPLATES.get(template_key, PROMPT_TEMPLATES['AGENT_THINK'])
            prompt_content = template['prompt']
            
            if self.score_mode == 'THINK':
                input_content = template['input'].format(client_question=prompt, think_str=content_to_score, ground_truth="")
                score_tag = "think_score"
            else: # 'ANSWER' or 'OUTCOME'
                input_content = template['input'].format(client_question=prompt, outcome_str=content_to_score, ground_truth="")
                score_tag = "outcome_score"

            messages = [{"role": "system", "content": prompt_content}, {"role": "user", "content": input_content}]
            api_requests.append({'messages': messages, 'score_tag': score_tag, 'index': i})
        return api_requests

    async def score_batch(self, api_requests: list) -> list:
        tasks = []
        semaphore = asyncio.Semaphore(self.concurrency)
        async def task_wrapper(request, client):
            async with semaphore:
                return await self._call_api_with_retry(client, request['messages'], request['index'])

        tasks = [task_wrapper(req, self.client) for req in api_requests]
        raw_responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        parsed_results = []
        for i, res in enumerate(raw_responses):
            if isinstance(res, Exception):
                parsed_results.append((self.failure_placeholder, "", ""))
            else:
                parsed_results.append(parse_judge_response(res, api_requests[i]['score_tag'], default_value=self.failure_placeholder))
        
        scores, _, _ = zip(*parsed_results)
        return list(scores)

async def main(args):
    print(f"Loading records from: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        records = json.load(f)

    scorer = OfflineScorer(reward_model=args.reward_model, score_mode=args.score_mode)
    
    print("Preparing API requests...")
    api_requests = scorer._prepare_requests(records)

    if not api_requests:
        print("No valid requests to send. Exiting.")
        return

    print(f"Scoring {len(api_requests)} records using model '{args.reward_model}' on '{args.score_mode}' content...")
    start_time = time.time()
    scores = await scorer.score_batch(api_requests)
    end_time = time.time()
    print(f"Scoring completed in {end_time - start_time:.2f} seconds.")

    # Augment records with new scores
    new_score_key = f"score_{args.reward_model}{args.model_name_suffix}"
    print(f"Augmenting records with new score key: '{new_score_key}'")
    
    # Since some records might have been skipped, we need to carefully align scores
    original_indices = [req['index'] for req in api_requests]
    for i, original_idx in enumerate(original_indices):
        records[original_idx][new_score_key] = scores[i]

    output_file = args.output_file if args.output_file else args.input_file
    print(f"Saving augmented records to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline scorer for augmenting datasets with new reward scores.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input JSON dataset file.")
    parser.add_argument("--output-file", type=str, help="Path to save the output. If not provided, overwrites the input file.")
    parser.add_argument("--reward-model", type=str, default="dsr1", help="The reward model to use for scoring.")
    parser.add_argument("--score-mode", type=str, default="think", choices=["think", "answer"], help="The content to score ('think' or 'answer').")
    parser.add_argument("--model-name-suffix", type=str, default="_answer", help="Suffix to append to the model name for the new score key.")
    
    args = parser.parse_args()
    asyncio.run(main(args))



#

    # python -m verl.ours.offline_scorer \
    # --input-file /workspace/data/rollout_for_r1/IF_project_skyworkqwen_generic.json \
    # --output-file /workspace/data/rollout_for_r1/IF_project_skyworkqwen_scored.json \
    # --score-mode think