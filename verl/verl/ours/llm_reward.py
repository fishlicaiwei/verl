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

from typing import Tuple, Dict

try:
    # This works when imported as part of a package
    from .prompt import PROMPT_TEMPLATES
except ImportError:
    # This works when run as a standalone script from the 'ours' directory
    from prompt import PROMPT_TEMPLATES

def prepare_data_for_all_models(
    data_sources: list,
    prompts_str: list,
    solutions_str: list,
    ground_truths: list,
    model_names: list,
    score_modes: list = None
) -> Tuple[Dict[str, list], list, list, list]:
    """
    Optimized batch data preparation for ALL models at once.
    This version supports multiple scoring tasks for the same model name.

    Features:
    - Regex extraction performed ONCE per sample.
    - Format checking performed ONCE per sample.
    - Message payload construction logic reused across compatible models.
    
    Returns:
    - model_requests_map: Dict[unique_model_key, List[request_dict]]
      where unique_model_key is e.g., "dsr1_0", "dsr1_1".
      request_dict is {'messages': ..., 'score_tag': ..., 'index': ...}
    - invalid_format_mask_list: List[bool]
    - think_lens: List[int] - Character length of extracted think strings
    - answer_lens: List[int] - Character length of extracted answer strings
    """
    batch_size = len(data_sources)
    if score_modes is None:
        score_modes = ['THINK'] * len(model_names)

    # --- 1. Fast Task Type Lookup ---
    MATH_DATA_SOURCES = {'RLVR-MATH','math_500','gsm8k','NuminaMath-CoT'}
    SAFETY_DATA_SOURCES = {'PKU-SafeRLHF','wildguard','coconot'}
    IF_DATA_SOURCES = {'tulu-3-IF-augmented-on-policy-8b','rlvr_ifeval','dolci_rl_zero_if_7b'}
    KNOWLEDGE_DATA_SOURCES = {'SciRIFF-train-mix','natural_questions','squad_v2_val'}
    CHAT_DATA_SOURCES = {'tulu-3-wildchat-if-on-policy-8b','tulu-3-ultrafeedback-cleaned-on-policy-8b','OpenAssistant-oasst1'}
    CODE_DATA_SOURCES = {'tulu-3-sft-personas-code','codealpaca_20k','openai_humaneval'}
    WRITTING_DATA_SOURCES = {'DeepWriting-20K','llmaes_writingprompts_val'}
    AGENT_DATA_SOURCES = {'xlam-function-calling-60k','function_calling_irrelevant'}
    source_to_type = {}
    for s in MATH_DATA_SOURCES: source_to_type[s] = 'REASONING'
    for s in CODE_DATA_SOURCES: source_to_type[s] = 'REASONING'
    for s in SAFETY_DATA_SOURCES: source_to_type[s] = 'SAFETY'
    for s in IF_DATA_SOURCES: source_to_type[s] = 'IF'
    for s in KNOWLEDGE_DATA_SOURCES: source_to_type[s] = 'KNOWLEDGE'
    for s in CHAT_DATA_SOURCES: source_to_type[s] = 'CHAT'
    for s in WRITTING_DATA_SOURCES: source_to_type[s] = 'WRITTING'
    for s in AGENT_DATA_SOURCES: source_to_type[s] = 'AGENT'
    task_types = [source_to_type.get(ds, 'CHAT') for ds in data_sources]

    # --- 2. Shared Processing (Fast String Extraction and Length Calculation) ---
    extracted_contents = [] # List of (think_str, outcome_str)
    invalid_format_mask_list = []
    think_lens = []
    answer_lens = []

    for i in range(batch_size):
        solution = solutions_str[i]
        think_start_tag = "<think>"
        think_end_tag = "</think>"
        
        start_idx = solution.find(think_start_tag)
        end_idx = solution.find(think_end_tag, start_idx)

        if start_idx != -1 and end_idx != -1:
            # Found <think>...</think> tags, apply relaxed parsing
            think_str = solution[start_idx + len(think_start_tag):end_idx].strip()
            # Everything after </think> is considered the answer
            outcome_str = solution[end_idx + len(think_end_tag):].strip()
            
            extracted_contents.append((think_str, outcome_str))
            invalid_format_mask_list.append(False)
            think_lens.append(len(think_str))
            answer_lens.append(len(outcome_str))
        else:
            # Fallback for complete invalid format (no <think> tag found)
            stripped = solution.strip()
            extracted_contents.append((stripped, stripped)) # Treat the whole response as both think and answer
            invalid_format_mask_list.append(True)
            think_lens.append(len(stripped))
            answer_lens.append(len(stripped))
        
    # --- 3. Payload Construction ---
    template_models, simple_models, rule_models = [], [], []
    for m in set(model_names): # Use set to get unique model types
        if m in ["dsr1"]: template_models.append(m)
        elif m in ["internlm2", "skywork", "rmrb2", "skyworkqwen"]: simple_models.append(m)
        elif m == "rule": rule_models.append(m)
        else: print(f"Warning: Unknown model type '{m}', skipping.")
    
    model_requests_map = {f"{m}_{i}": [] for i, m in enumerate(model_names)}
    
    for i in range(batch_size):
        think_str, outcome_str = extracted_contents[i]
        prompt = prompts_str[i]
        solution = solutions_str[i]
        ground_truth = ground_truths[i] if ground_truths[i] is not None else "No ground truth provided."
        
        for idx, model_name in enumerate(model_names):
            mode = score_modes[idx]
            unique_key = f"{model_name}_{idx}"

            if model_name in simple_models:
                content = think_str if mode == 'THINK' else outcome_str
                # The 'messages' key here is just a placeholder string for simple models, not a message list.
                msg_simple = f"Human: {prompt}\nAssistant: {content}"
                model_requests_map[unique_key].append({'messages': msg_simple, 'score_tag': mode, 'index': i, 'score_mode': mode})

            elif model_name in rule_models:
                # Rule-based models don't need a formatted prompt for an API.
                # We just need to package the data needed for the local rule function.
                # The 'messages' field will contain the model's answer to be evaluated.
                model_requests_map[unique_key].append({'messages': [ {"role": "assistant", "content": outcome_str} ], 'ground_truth': ground_truth, 'score_tag': mode, 'index': i, 'score_mode': mode})

            elif model_name in template_models:
                task_type = task_types[i]
                
                if mode == 'CONSISTENCY':
                    template = PROMPT_TEMPLATES['CONSISTENCY']
                    prompt_content = template['prompt']
                    input_content = template['input'].format(client_question=prompt, model_response=solution)
                    score_tag = "consistency_score"
                else: # THINK or OUTCOME
                    template_key = f"{task_type}_{mode}"
                    template = PROMPT_TEMPLATES.get(template_key, PROMPT_TEMPLATES['CHAT_THINK'])
                    prompt_content = template['prompt']
                    
                    if mode == 'THINK':
                        input_content = template['input'].format(client_question=prompt, think_str=think_str, ground_truth=ground_truth)
                        score_tag = "think_score"
                    else: # OUTCOME
                        input_content = template['input'].format(client_question=prompt, outcome_str=outcome_str, ground_truth=ground_truth)
                        score_tag = "outcome_score"
                
                msg_dsr1 = [{"role": "system", "content": prompt_content}, {"role": "user", "content": input_content}]
                model_requests_map[unique_key].append({'messages': msg_dsr1, 'score_tag': score_tag, 'index': i, 'score_mode': mode})

    return model_requests_map, invalid_format_mask_list, think_lens, answer_lens


def parse_judge_response(response_text: str, score_tag: str, default_value: float = -1.0) -> tuple[float, str]:
    """
    Parses the raw text response from the LLM judge to extract the score.
    It uses reverse string matching to find the last occurrence of the score tag.
    Returns:
        The extracted score as a float, or 'default_value' if parsing fails.
    """
    # #在新提示词系统下锁定score tag
    # score_tag = "score"

    rubric_start='<rubric>'
    rubric_end='</rubric>'
    start_idx = response_text.find(rubric_start)
    end_idx = response_text.find(rubric_end, start_idx)
    if start_idx != -1 and end_idx != -1:
        rubric_text = response_text[start_idx + len(rubric_start):end_idx].strip()
    else:
        rubric_text = ""

    justify_start='<justify>'
    justify_end='</justify>'
    start_idx = response_text.find(justify_start)
    end_idx = response_text.find(justify_end, start_idx)
    if start_idx != -1 and end_idx != -1:
        justify_text = response_text[start_idx + len(justify_start):end_idx].strip()
    else:
        justify_text = ""

    start_tag = f"<{score_tag}>"
    end_tag = f"</{score_tag}>"
    
    # 1. Find the last closing tag
    end_idx = response_text.rfind(end_tag)
    
    if end_idx != -1:
        # 2. Find the last opening tag BEFORE the closing tag
        start_idx = response_text.rfind(start_tag, 0, end_idx)
        
        if start_idx != -1:
            # Extract content between tags
            extracted_text = response_text[start_idx + len(start_tag):end_idx].strip()
            
            # Helper to check if string contains at least one digit
            if not any(char.isdigit() for char in extracted_text):
                 print(f"Warning: Content '{extracted_text}' inside <{score_tag}> does not contain digits. Defaulting to {default_value}.")
                 return default_value, rubric_text, justify_text

            try:
                # Check if the score is a fraction (e.g., "8/10")
                if '/' in extracted_text:
                    parts = extracted_text.split('/')
                    if len(parts) == 2:
                        numerator = float(parts[0])
                        denominator = float(parts[1])
                        if denominator != 0:
                            # Assuming a fraction like "8/10" should result in a score of 8.
                            score = (numerator / denominator) * 10.0
                        else:
                            score = default_value # Avoid division by zero
                    else:
                        # Handle malformed fractions like "8/9/10"
                        #print(f"Warning: Malformed fraction '{extracted_text}' found. Defaulting to {default_value}.")
                        score = default_value
                else:
                    # If not a fraction, convert directly to float
                    score = float(extracted_text)

                # Normalize score if it's out of the 0-10 range (e.g., LLM returned 100 instead of 10)
                if score > 10.0:
                    score = score / 10.0
                return score, rubric_text, justify_text

            except (ValueError, ZeroDivisionError) as e:
                #print(f"Warning: Could not process score from '{extracted_text}'. Error: {e}. Defaulting to {default_value}.")
                return default_value, rubric_text, justify_text
    
    # If we are here, tags were not found
    print(f"Warning: Could not parse <{score_tag}> tag with a valid score from response. Full response ({len(response_text)} tokens):\n{response_text[-100:]}")
    return default_value, rubric_text, justify_text