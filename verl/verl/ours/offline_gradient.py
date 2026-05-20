"""
Offline PPO gradient computation from saved rollout_data.pt and HF checkpoint.
Supports single-step mode and batch scan mode.

Single step:
  python offline_gradient.py --checkpoint .../global_step_4/actor/huggingface \
      --rollout_data .../global_step_4/rollout_data.pt

Batch scan (auto-discover global_step_*/):
  python offline_gradient.py \
      --ckpt_dir .../hot_dsr1_prm_online \
      --output_dir /workspace/gradient_output \
      --max_samples 64
"""

import torch
import torch.nn.functional as F
import argparse
import os
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Core functions ──────────────────────────────────────────────

def tokenize_sample(tokenizer, prompt, response, max_prompt_len, max_total_len):
    full = tokenizer(prompt + response, truncation=True, max_length=max_total_len,
                     return_tensors='pt')
    prompt_only = tokenizer(prompt, truncation=True, max_length=max_prompt_len,
                            return_tensors='pt')
    prompt_len = prompt_only['input_ids'].shape[1]
    response_len = full['input_ids'].shape[1] - prompt_len
    return full['input_ids'], full['attention_mask'], prompt_len, response_len


def compute_frozen_log_prob(model, input_ids, attention_mask, response_len):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    start = logits.shape[1] - response_len - 1
    lp = F.log_softmax(logits[:, start:-1, :], dim=-1)
    return lp.gather(dim=-1, index=input_ids[:, -response_len:].unsqueeze(-1)).squeeze(-1)


def compute_train_log_prob(model, input_ids, attention_mask, response_len):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    start = logits.shape[1] - response_len - 1
    lp = F.log_softmax(logits[:, start:-1, :], dim=-1)
    return lp.gather(dim=-1, index=input_ids[:, -response_len:].unsqueeze(-1)).squeeze(-1)


def ppo_loss(new_log_prob, old_log_prob, adv_val, response_len, clip_ratio, device):
    adv_tiled = torch.full((1, response_len), adv_val, device=device)
    resp_mask = torch.ones(1, response_len, device=device)
    ratio = torch.exp(new_log_prob - old_log_prob)
    pg_losses = -adv_tiled * ratio
    pg_losses2 = -adv_tiled * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    loss_per_token = torch.max(pg_losses, pg_losses2)
    return (loss_per_token * resp_mask).sum() / resp_mask.sum().clamp(min=1)


def grab_gradients(model):
    param_grads = {}
    total_gn2 = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad.detach().flatten()
            param_grads[name] = g
            total_gn2 += g.norm(2).item() ** 2
    return param_grads, total_gn2 ** 0.5


def accumulate_grads(grads_src, grads_dst):
    for k, v in grads_src.items():
        if k in grads_dst:
            grads_dst[k] += v
        else:
            grads_dst[k] = v.clone()


def grad_norm(grads):
    return sum(v.norm(2).item() ** 2 for v in grads.values()) ** 0.5


def per_sample_metrics(grads_prm, grads_orm):
    dot, norm_prm2, norm_orm2 = 0.0, 0.0, 0.0
    common = set(grads_prm.keys()) & set(grads_orm.keys())
    for k in common:
        gp, go = grads_prm[k], grads_orm[k]
        dot += (gp * go).sum().item()
        norm_prm2 += gp.norm(2).item() ** 2
        norm_orm2 += go.norm(2).item() ** 2
    eps = 1e-8
    gn_prm, gn_orm = norm_prm2 ** 0.5, norm_orm2 ** 0.5
    cos = dot / (gn_prm * gn_orm + eps)
    mag_sim = dot / (norm_orm2 + eps)
    norm_ratio = gn_prm / (gn_orm + eps)
    return cos, mag_sim, norm_ratio


# ── Single-step processor ───────────────────────────────────────

def process_step(checkpoint_path, rollout_path, output_dir, args):
    """Process one step: load model, compute gradients, save."""
    step_name = os.path.basename(os.path.dirname(rollout_path))  # e.g. "global_step_4"
    print(f"\n{'='*60}")
    print(f"Processing {step_name}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  rollout:    {rollout_path}")
    print(f"{'='*60}")

    # Load data
    data = torch.load(rollout_path, map_location='cpu', weights_only=False)
    prompts = data['prompts']
    responses_list = data['responses']
    prm_scores_raw = data.get('prm_scores')
    orm_scores_raw = data.get('orm_scores')
    step = data['step']
    print(f"  step={step}, prompts={len(prompts)}, responses_per={len(responses_list[0])}")

    # Advantages: compute or load
    advantage_path = rollout_path.replace('.pt', '_advantage.pt')
    if os.path.exists(advantage_path):
        print(f"  Loading cached advantages from {advantage_path}")
        adv_data = torch.load(advantage_path, map_location='cpu', weights_only=False)
    else:
        print("  Computing GRPO advantages...")
        def grpo_adv(scores, eps=1e-6):
            if scores is None: return None
            mean = scores.float().mean(dim=-1, keepdim=True)
            std = scores.float().std(dim=-1, keepdim=True)
            return (scores.float() - mean) / (std + eps)
        adv_data = {'step': step}
        if prm_scores_raw is not None:
            adv_data['prm_advantage'] = grpo_adv(prm_scores_raw)
        if orm_scores_raw is not None:
            adv_data['orm_advantage'] = grpo_adv(orm_scores_raw)
        if prm_scores_raw is not None and orm_scores_raw is not None:
            adv_data['combined_advantage'] = adv_data['prm_advantage'] + adv_data['orm_advantage']
        torch.save(adv_data, advantage_path)
    prm_adv = adv_data['prm_advantage']
    orm_adv = adv_data['orm_advantage']
    comb_adv = adv_data['combined_advantage']

    # Load model
    print(f"  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Shuffle prompt indices for fair sampling (fixed seed for reproducibility)
    import numpy as np
    num_prompts = len(prompts)
    shuf_idx = np.random.RandomState(42).permutation(num_prompts)
    max_prompts = min(num_prompts, args.max_samples // args.n)
    selected_idx = shuf_idx[:max_prompts]
    print(f"  Shuffled {num_prompts} prompts, selected {len(selected_idx)} for gradient analysis")

    # Per-sample accumulators
    sample_ids, cos_sim_list, norm_ratio_list, magnitude_sim_list = [], [], [], []
    prm_advantages, orm_advantages, response_lengths = [], [], []
    prm_scores_lst, orm_scores_lst = [], []
    acc_prm, acc_orm, acc_comb = {}, {}, {}
    processed = 0
    total = min(len(prompts) * args.n, args.max_samples)

    print(f"  Processing {total} samples...")
    for p_idx in selected_idx:
        if processed >= total: break
        for r_idx in range(args.n):
            if processed >= total: break

            try:
                input_ids, attn_mask, p_len, r_len = tokenize_sample(
                    tokenizer, prompts[p_idx], responses_list[p_idx][r_idx],
                    args.max_prompt_len, args.max_total_len)
                if r_len <= 0: continue
                input_ids, attn_mask = input_ids.cuda(), attn_mask.cuda()

                old_log_prob = compute_frozen_log_prob(model, input_ids, attn_mask, r_len)
                model.train()
                for p in model.parameters(): p.requires_grad_(True)

                # PRM
                nl = compute_train_log_prob(model, input_ids, attn_mask, r_len)
                ppo_loss(nl, old_log_prob.detach(), prm_adv[p_idx, r_idx].item(),
                         r_len, args.clip_ratio, input_ids.device).backward()
                g_prm, gn_prm = grab_gradients(model)
                accumulate_grads(g_prm, acc_prm)
                model.zero_grad()

                # ORM
                nl = compute_train_log_prob(model, input_ids, attn_mask, r_len)
                ppo_loss(nl, old_log_prob.detach(), orm_adv[p_idx, r_idx].item(),
                         r_len, args.clip_ratio, input_ids.device).backward()
                g_orm, gn_orm = grab_gradients(model)
                accumulate_grads(g_orm, acc_orm)
                model.zero_grad()

                # Combined
                nl = compute_train_log_prob(model, input_ids, attn_mask, r_len)
                ppo_loss(nl, old_log_prob.detach(), comb_adv[p_idx, r_idx].item(),
                         r_len, args.clip_ratio, input_ids.device).backward()
                g_comb, gn_comb = grab_gradients(model)
                accumulate_grads(g_comb, acc_comb)
                model.zero_grad()

                cos_s, mag_s, norm_r = per_sample_metrics(g_prm, g_orm)
                cos_sim_list.append(cos_s)
                magnitude_sim_list.append(mag_s)
                norm_ratio_list.append(norm_r)
                sample_ids.append(f"p{p_idx}_r{r_idx}")
                prm_advantages.append(prm_adv[p_idx, r_idx].item())
                orm_advantages.append(orm_adv[p_idx, r_idx].item())
                response_lengths.append(r_len)
                if prm_scores_raw is not None:
                    prm_scores_lst.append(prm_scores_raw[p_idx, r_idx].item())
                if orm_scores_raw is not None:
                    orm_scores_lst.append(orm_scores_raw[p_idx, r_idx].item())

                for p in model.parameters(): p.requires_grad_(False)
                model.eval()
                processed += 1
                if processed % 8 == 0:
                    print(f"    [{processed}/{total}] cos={cos_s:.4f} mag_sim={mag_s:.4f} norm_r={norm_r:.4f}")

            except torch.cuda.OutOfMemoryError:
                print(f"    OOM p={p_idx} r={r_idx}, skip")
                torch.cuda.empty_cache()
                continue

    print(f"  Done: {processed} samples.")

    # Aggregated
    if processed == 0: return None
    avg_prm = {k: v / processed for k, v in acc_prm.items()}
    avg_orm = {k: v / processed for k, v in acc_orm.items()}
    avg_comb = {k: v / processed for k, v in acc_comb.items()}

    gn_prm_avg = grad_norm(avg_prm)
    gn_orm_avg = grad_norm(avg_orm)
    gn_comb_avg = grad_norm(avg_comb)

    dot_agg, nprm2, norm2 = 0.0, 0.0, 0.0
    for k in set(avg_prm.keys()) & set(avg_orm.keys()):
        gp, go = avg_prm[k], avg_orm[k]
        dot_agg += (gp * go).sum().item()
        nprm2 += gp.norm(2).item() ** 2
        norm2 += go.norm(2).item() ** 2
    eps = 1e-8
    cos_agg = dot_agg / ((nprm2 ** 0.5) * (norm2 ** 0.5) + eps)
    mag_sim_agg = dot_agg / (norm2 + eps)
    norm_ratio_agg = gn_prm_avg / (gn_orm_avg + eps)

    head_prm = avg_prm.get('lm_head.weight', None)
    head_orm = avg_orm.get('lm_head.weight', None)
    head_comb = avg_comb.get('lm_head.weight', None)

    output = {
        'step_idx': step,
        'n_samples': processed,
        'sample_ids': sample_ids,
        'prm_advantages': prm_advantages,
        'orm_advantages': orm_advantages,
        'cos_sim_list': cos_sim_list,
        'norm_ratio_list': norm_ratio_list,
        'magnitude_sim_list': magnitude_sim_list,
        'response_lengths': response_lengths,
        'prm_scores_raw': prm_scores_lst,
        'orm_scores_raw': orm_scores_lst,
        'avg_cos_sim': cos_agg,
        'avg_magnitude_sim': mag_sim_agg,
        'avg_norm_ratio': norm_ratio_agg,
        'grad_norm_prm': gn_prm_avg,
        'grad_norm_orm': gn_orm_avg,
        'grad_norm_combined': gn_comb_avg,
        'aggregated_grad_head': {
            'prm': head_prm,
            'orm': head_orm,
            'combined': head_comb,
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'gradient_step{step}.pt')
    torch.save(output, out_path)
    print(f"  Saved → {out_path}")
    print(f"  grad_norm: prm={gn_prm_avg:.4f} orm={gn_orm_avg:.4f} comb={gn_comb_avg:.4f}")
    print(f"  avg_cos={cos_agg:.4f}  avg_mag_sim={mag_sim_agg:.4f}  avg_norm_ratio={norm_ratio_agg:.4f}")

    del model
    torch.cuda.empty_cache()
    return output


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Root checkpoint dir (auto-discover global_step_*/). '
                             'Alternative to --checkpoint + --rollout_data.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to HF model (e.g. global_step_4/actor/huggingface)')
    parser.add_argument('--rollout_data', type=str, default=None,
                        help='Path to rollout_data.pt')
    parser.add_argument('--output_dir', type=str, default='./gradient_output')
    parser.add_argument('--max_prompt_len', type=int, default=1024)
    parser.add_argument('--max_total_len', type=int, default=10240)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--max_samples', type=int, default=64)
    parser.add_argument('--n', type=int, default=8)
    args = parser.parse_args()

    if args.ckpt_dir:
        # ── Batch scan mode ──
        ckpt_dir = args.ckpt_dir
        print(f"Scanning: {ckpt_dir}")
        step_dirs = sorted(glob.glob(os.path.join(ckpt_dir, 'global_step_*')))
        if not step_dirs:
            print(f"  No global_step_* found in {ckpt_dir}")
            return
        print(f"  Found {len(step_dirs)} steps: {[os.path.basename(d) for d in step_dirs]}")

        for step_dir in step_dirs:
            rollout_path = os.path.join(step_dir, 'rollout_data.pt')
            if not os.path.exists(rollout_path):
                print(f"  Skip {os.path.basename(step_dir)}: no rollout_data.pt")
                continue
            ckpt_path = os.path.join(step_dir, 'actor', 'huggingface')
            if not os.path.exists(ckpt_path):
                print(f"  Skip {os.path.basename(step_dir)}: no actor/huggingface")
                continue
            process_step(ckpt_path, rollout_path, args.output_dir, args)

        print(f"\nAll done. Results in {args.output_dir}/")

    elif args.checkpoint and args.rollout_data:
        # ── Single-step mode ──
        process_step(args.checkpoint, args.rollout_data, args.output_dir, args)

    else:
        print("Error: provide either --ckpt_dir (batch mode) or both --checkpoint and --rollout_data (single mode)")
        return


if __name__ == '__main__':
    main()
