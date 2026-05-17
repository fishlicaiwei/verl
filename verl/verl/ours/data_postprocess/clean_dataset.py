import asyncio
import json
import os
import argparse
import sys
import random
from typing import List, Dict, Any, Optional
from collections import defaultdict
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# --- Configuration ---
API_KEY = "527bdfdb89f9191848ef21902f912e82:ODBlYjI2ZTVkMTdjYmVjMDc2N2M5ZGMx"
BASE_URL = "http://127.0.0.1:30010/v1"
MODEL_NAME = "xdeepseekr1"
MAX_RETRIES = 2 # Number of retries for malformed outputs

# System Prompt
CLEANER_SYSTEM_PROMPT = """You are an expert Data Engineer. Your task is to extract structured evaluation data from the provided "Rubric" text into a strict JSON format.

**CRITICAL INSTRUCTIONS:**
1. **Extraction Only:** Do not summarize, paraphrase, or generate new content. Extract text exactly as it appears in the source.
2. **Justification Handling:** - If the text contains a general/overall reasoning paragraph (usually at the start or end), extract it to `"overall_reasoning"`.
   - If the text contains specific reasons attached to individual criteria (e.g., "Weight 0.3 because..."), extract them to the `"specific_rationale"` field inside that criterion.
   - If no specific rationale is provided for a criterion, set `"specific_rationale"` to `null`.
3. **JSON Only:** Output raw JSON without markdown formatting (no ```json wrapper).

**OUTPUT SCHEMA:**
```json
{
  "criteria": [
    {
      "name": "Exact name from text",
      "weight": 0.5, 
      "description": "The description text...",
      "specific_rationale": "The specific reason for this weight/criterion (or null)"
    }
  ],
  "overall_reasoning": "The global reasoning text (or null)"
}
```
"""

async def call_llm(client: AsyncOpenAI, messages: List[Dict], temperature: float = 0.0) -> str:
    """Helper to make the API call."""
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=2048
    )
    content = response.choices[0].message.content.strip()
    # Basic Cleanup for Markdown
    if content.startswith("```json"): content = content[7:]
    if content.endswith("```"): content = content[:-3]
    return content.strip()

def validate_json_schema(data: Any) -> Optional[str]:
    """
    Validates if the parsed JSON matches the expected schema.
    Returns an error message string if invalid, or None if valid.
    """
    if not isinstance(data, dict):
        return "Root element must be a JSON object (dict)."
    
    required_keys = ["criteria", "overall_reasoning"]
    missing_keys = [k for k in required_keys if k not in data]
    
    if missing_keys:
        return f"Missing required top-level keys: {', '.join(missing_keys)}."
    
    if not isinstance(data.get("criteria"), list):
        return "'criteria' must be a list."
    
    # Check for hallucinated keys (optional strictness)
    # unexpected_keys = [k for k in data.keys() if k not in required_keys]
    # if unexpected_keys:
    #     return f"Unexpected keys found: {', '.join(unexpected_keys)}. Only 'criteria' and 'overall_reasoning' are allowed."

    return None

async def process_single_record(client: AsyncOpenAI, record: Dict, semaphore: asyncio.Semaphore, domain: str) -> Dict:
    """
    Process a single record with Retry-and-Feedback logic.
    """
    async with semaphore:
        raw_prompt = record.get("prompt", "")
        raw_rubric = record.get("rubric", "")

        # Initial output with hard constraints
        final_output = {
            "prompt": raw_prompt,
            "domain_tag": domain
        }

        if not raw_rubric or not isinstance(raw_rubric, str):
            final_output.update({"error": "empty_or_invalid_source_rubric"})
            return final_output

        # Initial Message History
        messages = [
            {"role": "system", "content": CLEANER_SYSTEM_PROMPT},
            {"role": "user", "content": f"RUBRIC:\n{raw_rubric}"}
        ]

        attempt = 0
        while attempt <= MAX_RETRIES:
            try:
                # Call LLM (increase temp slightly on retries to break loops)
                current_temp = 0.0 if attempt == 0 else 0.2
                response_content = await call_llm(client, messages, temperature=current_temp)
                
                # 1. Try Parse JSON
                try:
                    structured_data = json.loads(response_content)
                except json.JSONDecodeError:
                    error_msg = "Invalid JSON format. Please output strictly valid JSON only."
                    raise ValueError(error_msg)

                # 2. Validate Schema
                schema_error = validate_json_schema(structured_data)
                if schema_error:
                    raise ValueError(f"Schema Error: {schema_error}")

                # 3. Success! Merge and return (ignoring potential hallucinated keys by only picking what we validated if we wanted strictness, 
                # or just merging since we validated required keys)
                
                # Safe Merge: We trusted validate_json_schema, but to be ultra-safe against key overwrites:
                # Remove protected keys from LLM output if they exist
                structured_data.pop("prompt", None)
                structured_data.pop("domain_tag", None)
                
                final_output.update(structured_data)
                return final_output

            except ValueError as ve:
                # Logic Error (JSON or Schema)
                error_description = str(ve)
                # print(f"Attempt {attempt + 1} failed: {error_description}") # Optional logging
                
                if attempt < MAX_RETRIES:
                    # Append error feedback to history
                    messages.append({"role": "assistant", "content": response_content})
                    messages.append({"role": "user", "content": f"Error: {error_description}\nPlease correct your output format and try again."})
                    attempt += 1
                else:
                    # Final Failure
                    final_output.update({
                        "error": "max_retries_exceeded",
                        "last_error": error_description,
                        "raw_llm_output": response_content
                    })
                    return final_output

            except Exception as e:
                # System Error (Network, API) - usually not worth feedback-looping, just retry or fail
                final_output.update({
                    "error": "api_call_failed",
                    "details": str(e)
                })
                return final_output
        
        return final_output

def count_lines(filename):
    if not os.path.exists(filename): return 0
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

async def main():
    parser = argparse.ArgumentParser(description="Clean LLM Judge dataset (Domain-specific with Retry).")
    parser.add_argument("--domain", type=str, required=True, help="Domain name")
    parser.add_argument("--concurrency", type=int, default=400)
    parser.add_argument("--sample_k", type=int, default=1)
    parser.add_argument("--api_key", type=str, default=API_KEY)
    parser.add_argument("--base_url", type=str, default=BASE_URL)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    input_file = f"dsr1_{args.domain}_rubric_dataset.json"
    output_file = f"dsr1_{args.domain}_rubric_dataset_cleaned.jsonl"

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    print(f"Reading input: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        try:
            input_data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Failed to decode input JSON.")
            return

    # Grouping
    print("Grouping records by prompt...")
    grouped_data = defaultdict(list)
    for record in tqdm(input_data, desc="Grouping"):
        if "prompt" in record:
            grouped_data[record["prompt"]].append(record)
    
    # Sampling
    records_to_process = []
    for prompt, records in grouped_data.items():
        if len(records) > args.sample_k:
            sampled = random.sample(records, args.sample_k)
        else:
            sampled = records 
        records_to_process.extend(sampled)
    
    print(f"Total records to process: {len(records_to_process)}")

    # Resume Check
    processed_counts = defaultdict(int)
    if os.path.exists(output_file):
        print(f"Scanning output '{output_file}'...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "prompt" in rec and rec.get("domain_tag") == args.domain:
                        processed_counts[rec["prompt"]] += 1
                except: pass
        print(f"Found {sum(processed_counts.values())} processed records.")

    final_queue = []
    for r in records_to_process:
        p_text = r["prompt"]
        if processed_counts[p_text] > 0:
            processed_counts[p_text] -= 1
        else:
            final_queue.append(r)
    
    if not final_queue:
        print("All records processed!")
        return

    print(f"Remaining: {len(final_queue)}")

    # Processing
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    semaphore = asyncio.Semaphore(args.concurrency)

    with open(output_file, "a", encoding="utf-8") as outfile:
        batch_size = args.concurrency * 2
        
        with tqdm(total=len(final_queue), desc="Cleaning") as pbar:
            for i in range(0, len(final_queue), batch_size):
                batch = final_queue[i : i + batch_size]
                tasks = [process_single_record(client, record, semaphore, args.domain) for record in batch]
                batch_results = await asyncio.gather(*tasks)
                
                for res in batch_results:
                    outfile.write(json.dumps(res, ensure_ascii=False) + "\n")
                
                outfile.flush()
                pbar.update(len(batch))

    print(f"\nDone! Saved to {output_file}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
