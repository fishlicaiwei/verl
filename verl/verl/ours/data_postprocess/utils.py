import json
import random
import argparse
import os
import sys

def inspect_data(domain, n):
    # Construct filename based on the convention in clean_dataset.py
    filename = f"dsr1_{domain}_rubric_dataset_cleaned.jsonl"
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found in current directory: {os.getcwd()}")
        print("Available files:", [f for f in os.listdir('.') if f.endswith('.jsonl')])
        return

    # Load all records (efficient enough for viewing purposes unless file is massive)
    records = []
    print(f"Reading {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    total = len(records)
    print(f"Total records found: {total}")
    
    if total == 0:
        print("File is empty.")
        return

    # Sample
    k = min(n, total)
    sampled = random.sample(records, k)
    
    print(f"\nDisplaying {k} random records from '{domain}' domain:")
    print("=" * 80)

    for idx, record in enumerate(sampled, 1):
        print(f"Record {idx}/{k}")
        print("-" * 40)
        
        # 1. Prompt
        print(f"\n[PROMPT]:")
        print(record.get("prompt", "").strip())
        
        # 2. Overall Reasoning
        print(f"\n[OVERALL REASONING]:")
        reasoning = record.get("overall_reasoning")
        if reasoning:
            print(reasoning.strip())
        else:
            print("(None)")

        # 3. Criteria
        print(f"\n[CRITERIA]:")
        criteria = record.get("criteria", [])
        if not criteria:
            print("(No criteria found)")
        
        for c_idx, crit in enumerate(criteria, 1):
            name = crit.get("name", "Unnamed")
            weight = crit.get("weight", 0)
            desc = crit.get("description", "")
            rationale = crit.get("specific_rationale")
            
            print(f"  {c_idx}. {name} (Weight: {weight})")
            print(f"     Rule: {desc}")
            if rationale:
                print(f"     Why:  {rationale}")
            print("")
            
        print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly inspect cleaned rubric data.")
    parser.add_argument("--domain", type=str, required=True, help="Domain of the dataset (e.g., math, code).")
    parser.add_argument("-n", type=int, default=1, help="Number of records to display.")
    
    args = parser.parse_args()
    inspect_data(args.domain, args.n)
