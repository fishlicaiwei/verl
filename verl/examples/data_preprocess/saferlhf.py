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

"""
Preprocess the PKU-SafeRLHF dataset to parquet format
"""
import os
import datasets
import argparse
# Note: The original script contained HDFS-related imports (verl.utils.hdfs_io) 
# and a regex-based solution extractor, which are not needed for the new dataset/task.
# I've removed them and kept only the necessary imports.

def extract_solution(solution_str):
    """
    Placeholder function for solution extraction. 
    It is not used for the PKU-SafeRLHF dataset in this modified script, 
    but kept as a stub to minimally adapt the original structure if needed.
    """
    return ""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Updated default local directory name to reflect the new dataset
    parser.add_argument('--local_dir', default='~/data/pku_saferlhf') 
    parser.add_argument('--hdfs_dir', default=None) # HDFS logic is retained but you can ignore it if not using HDFS
    args = parser.parse_args()

    # --- Dataset Loading and Preparation ---
    
    # New dataset information
    data_source = 'PKU-Alignment/PKU-SafeRLHF'
    subset_name = 'alpaca3-8b'
    split_name = 'train'
    sample_size = 10000

    print(f"Loading dataset: {data_source}, subset: {subset_name}, split: {split_name}...")
    
    # Load the specified subset and split
    dataset = datasets.load_dataset(data_source, subset_name, split=split_name)
    
    # Sample the first 10,000 entries
    if len(dataset) > sample_size:
        print(f"Sampling the first {sample_size} examples...")
        sampled_dataset = dataset.select(range(sample_size))
    else:
        sampled_dataset = dataset
        print(f"Dataset size is {len(dataset)}, no sampling needed.")
        
    # --- Mapping Function ---
    
    def process_fn(example, idx):
        """
        Maps the 'prompt' column to the desired output format,
        retains 'index', and uses placeholders for other fields
        not relevant to the PKU-SafeRLHF 'alpaca3-8b' subset.
        """
        # The 'prompt' column in the alpaca3-8b subset contains the user prompt text.
        prompt_content = example['prompt']
        
        # Construct the desired output dictionary format
        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                # Map the original 'prompt' content to the new 'content' field
                "content": prompt_content,
            }],
            # Placeholders for fields that are not available/needed
            "ability": "safety_alignment", 
            "reward_model": {
                "style": "rule",
                "ground_truth": "", # No clear ground_truth/solution in this subset
            },
            "extra_info": {
                'split': split_name,
                'index': idx, # Index starting from 0
                'answer': "",
                "question": prompt_content, # Using prompt_content as question placeholder
            }
        }
        return data

    # Apply the mapping function
    print("Applying mapping function...")
    processed_dataset = sampled_dataset.map(function=process_fn, with_indices=True)

    # Remove the original columns to keep the dataset clean with only the new format
    cols_to_keep = [
        "data_source", "prompt", "ability", 
        "reward_model", "extra_info"
    ]
    # Keep only the generated columns, excluding the original ones like 'prompt', 'response', 'is_safe', etc.
    final_dataset = processed_dataset.remove_columns(
        [col for col in processed_dataset.column_names if col not in cols_to_keep]
    )


    # --- Save to Parquet ---

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    output_path = os.path.join(local_dir, f'{split_name}_{subset_name}.parquet')

    print(f"Saving to Parquet at: {output_path}...")
    final_dataset.to_parquet(output_path)
    print("Parquet file saved successfully.")

    # --- HDFS Copy (Optional) ---
    
    hdfs_dir = args.hdfs_dir
    if hdfs_dir is not None:
        print(f"Copying to HDFS at: {hdfs_dir}...")
        try:
            # You must ensure 'verl.utils.hdfs_io' is available and configured
            from verl.utils.hdfs_io import copy, makedirs
            makedirs(hdfs_dir)
            # Copy the entire local directory to HDFS
            copy(src=local_dir, dst=hdfs_dir)
            print("Copied to HDFS successfully.")
        except ImportError:
            print("Warning: Could not import 'verl.utils.hdfs_io'. Skipping HDFS copy.")
        except Exception as e:
            print(f"Error during HDFS copy: {e}")