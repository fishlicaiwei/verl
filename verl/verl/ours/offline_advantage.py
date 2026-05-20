"""
Offline GRPO advantage computation from saved rollout_data.pt.
"""

import torch
import argparse


def compute_grpo_advantage(scores: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Compute GRPO-style outcome advantage by normalizing scores within each group.

    Args:
        scores: [num_prompts, n_responses] — raw scores per response
        epsilon: small constant for numerical stability

    Returns:
        advantages: [num_prompts, n_responses] — z-score normalized advantages
    """
    mean = scores.mean(dim=-1, keepdim=True)
    std = scores.std(dim=-1, keepdim=True)
    advantages = (scores - mean) / (std + epsilon)
    return advantages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to rollout_data.pt')
    parser.add_argument('--output', type=str, default=None, help='Output path (default: same dir, with _advantage suffix)')
    parser.add_argument('--epsilon', type=float, default=1e-6)
    args = parser.parse_args()

    data = torch.load(args.input, map_location='cpu', weights_only=False)
    print(f"Loaded step={data['step']}, prompts={len(data['prompts'])}")

    prm_scores = data.get('prm_scores')  # [num_prompts, n]
    orm_scores = data.get('orm_scores')  # [num_prompts, n]

    result = {'step': data['step']}

    if prm_scores is not None:
        result['prm_advantage'] = compute_grpo_advantage(prm_scores, args.epsilon)
        print(f"  PRM advantage: mean={result['prm_advantage'].mean():.4f}, std={result['prm_advantage'].std():.4f}")

    if orm_scores is not None:
        result['orm_advantage'] = compute_grpo_advantage(orm_scores, args.epsilon)
        print(f"  ORM advantage: mean={result['orm_advantage'].mean():.4f}, std={result['orm_advantage'].std():.4f}")

    # Combined advantage (if both available)
    if prm_scores is not None and orm_scores is not None:
        result['combined_advantage'] = result['prm_advantage'] + result['orm_advantage']
        print(f"  Combined: mean={result['combined_advantage'].mean():.4f}, std={result['combined_advantage'].std():.4f}")

    out_path = args.output or args.input.replace('.pt', '_advantage.pt')
    torch.save(result, out_path)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
