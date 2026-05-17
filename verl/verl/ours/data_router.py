import os
import argparse

def get_paths(task: str) -> tuple[str, str]:
    """
    Determines the training and validation file paths based on the specified task.
    
    Args:
        task: The name of the task (e.g., 'chat', 'code', 'math').

    Returns:
        A tuple containing:
        - The path for the training data.
        - A Hydra-compatible string for the list of validation data paths.
    """
    
    # --- Base Paths ---
    FILE1 = "/workspace/data/train_covalid"
    FILE2 = "/workspace/data/valid"

    # --- Define a common set of validation paths, shared across all tasks ---
    common_val_paths = [
        # f"{FILE1}/tulu-3-wildchat-sampled/test_chat.parquet",
        # f"{FILE1}/PKU-SafeRLHF-sampled/test_safety.parquet",
        # f"{FILE1}/SciRIFF-sampled/test_general.parquet",
        f"{FILE1}/RLVR-MATH-sampled/test_math.parquet",
        # f"{FILE1}/tulu-3-if-augmented-sampled/test_if.parquet",
        # f"{FILE1}/m-a-p-deepwriting-sampled/test_writing.parquet",
        # f"{FILE1}/tulu-3-sft-personas-code/test_code.parquet",
        # f"{FILE1}/xlam-function-calling/test_function_calling_500.parquet"
    ]

    train_path = ""
    domain_specific_val_paths = []

    # --- Logic to select domain-specific data based on task ---
    if task == 'chat':
        train_path = f"{FILE1}/tulu-3-wildchat-sampled/train_chat_partial.parquet"
        domain_specific_val_paths = [
            f"{FILE2}/chat/oasst1-sampled-val/test_chat.parquet",
            f"{FILE2}/chat/tulu-3-ultrafeedback-sampled-val/test_chat.parquet",
        ]
    elif task == 'IF':
        train_path = f"{FILE1}/tulu-3-if-augmented-sampled/train_if_partial.parquet"
        domain_specific_val_paths = [
            f"{FILE2}/IF/dolci-sampled-val/test_dolci.parquet",
            f"{FILE2}/IF/rlvr-ifeval-sampled-val/test_rlvr.parquet",
        ]
    elif task == 'code':
        train_path = f"{FILE1}/tulu-3-sft-personas-code/train_code_partial.parquet"
        domain_specific_val_paths = [
            f"{FILE2}/code/codealpaca_processed_val/test_code.parquet",
            f"{FILE2}/code/openai_humaneval_processed_val/test_code.parquet",
        ]
    elif task == 'knowledge':
        # NOTE: Using 'SciRIFF' as the training set for the 'knowledge' task based on the file structure.
        train_path = f"{FILE1}/SciRIFF-sampled/train_general_partial.parquet"
        domain_specific_val_paths = [
            f"{FILE2}/knowledge/nq-sampled-val/test_nq.parquet",
            f"{FILE2}/knowledge/squad_v2_processed_val/test_qa.parquet",
        ]
    elif task == 'math':
        # train_path = f"{FILE1}/RLVR-MATH-sampled/train_math_partial.parquet"
        # train_path = f"{FILE1}/NuminaMath-sampled/train_numina_partial.parquet"
        # train_path = f"{FILE1}/NuminaMath-sampled/openr1_math_rl_10k.parquet"
        train_path = f"{FILE1}/NuminaMath-sampled/deepmath_hard_10k.parquet"
        domain_specific_val_paths = [
            f"{FILE1}/NuminaMath-sampled/test_numina.parquet",
            f"{FILE2}/math/math_500_processed_val/test_math.parquet",
        ]
    elif task == 'safe':
        train_path = f"{FILE1}/PKU-SafeRLHF-sampled/train_safety_partial.parquet"
        domain_specific_val_paths = [
            f"{FILE2}/safe/coconot-sampled-val/test_safety.parquet",
            f"{FILE2}/safe/wildguard_processed_val/test_general.parquet",
        ]
    elif task == 'writting': # Using user's spelling
        train_path = f"{FILE1}/m-a-p-deepwriting-sampled/train_writing_partial.parquet"
        domain_specific_val_paths = [
            f"{FILE2}/writting/llmaes_writingprompts_processed_val/test_creative.parquet",
        ]
    elif task == 'agent':
        train_path = f"{FILE1}/xlam-function-calling/train_mixed_11k.parquet"
        domain_specific_val_paths = [
            f"{FILE2}/agent/irrelevant-xlam/test_irrelevant_500.parquet",
        ]
    else:
        supported_tasks = ['chat', 'IF', 'code', 'knowledge', 'math', 'safe', 'writting','agent']
        raise ValueError(f"Unknown task: '{task}'. Please specify a valid task from: {supported_tasks}")

    # Combine common and domain-specific validation paths
    final_val_paths = common_val_paths + domain_specific_val_paths
    
    # Format the validation paths into a Hydra-compatible string: "[path1,path2,...]"
    val_paths_str = f"[{','.join(final_val_paths)}]"
    
    return train_path, val_paths_str


def main():
    """
    Main function to parse arguments and print the data paths to standard output.
    """
    parser = argparse.ArgumentParser(description="Data router for VERL training. Prints two lines: 1. train_path, 2. val_paths_string.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The name of the task to get data paths for (e.g., 'chat', 'code', 'math')."
    )
    args = parser.parse_args()
    
    try:
        train_path, val_paths_str = get_paths(args.task)
        
        # Print the results to standard output, one per line.
        # The shell script will capture these two lines.
        print(train_path)
        print(val_paths_str)
    except ValueError as e:
        print(f"Error: {e}", file=os.sys.stderr)
        exit(1)


if __name__ == "__main__":
    main()
