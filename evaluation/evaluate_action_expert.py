"""Isolated Action Expert evaluation.

Measures the action expert's standalone prediction accuracy on a held-out set,
decoupled from the world model by using the *real* goal-state subgoal (and,
optionally, the *generated* subgoal to quantify the WM's contribution).

Reports:
    - per-step action match (all H=8 actions)
    - first-action match (the action actually executed first)
    - first-K match (the execute_steps=2 prefix that gets executed before replanning)

Usage:
    python evaluation/evaluate_action_expert.py \
        --checkpoint checkpoints/action_expert_goal_wide_best.pt \
        [--world-model checkpoints/world_model_goal_wide_best.pt]
"""

import argparse

import torch
from torch.utils.data import DataLoader

from data.dataset import EpisodeDataset, collate
from training.utils import build_action_expert, build_world_model, tokenize_batch

ACTION_NAMES = ("W", "A", "S", "D")


@torch.no_grad()
def evaluate(ae, wm, data_dir, device, goal=True, execute_steps=2):
    ds = EpisodeDataset(data_dir, K=4, H=8, goal=goal, preload=True)
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate)

    def match(pred, chunk, k):
        return (pred[:, :k] == chunk[:, :k]).float().mean().item()

    res = {}
    for mode in (["real", "generated"] if wm is not None else ["real"]):
        tot_all = tot_first = tot_k = 0.0
        n = 0
        for b in loader:
            hist = b["history"].cuda()
            real_sg = b["subgoal"].unsqueeze(1).cuda()
            chunk = b["action_chunk"].cuda()
            lang = tokenize_batch(b["instruction"], device)
            sub = tokenize_batch(b["subtask"], device)
            meta = b["metadata"]
            sg = real_sg if mode == "real" else wm.sample(hist, lang, sub, meta)
            pred = ae.sample(hist, sg, lang, sub, meta).argmax(-1)  # (B,H)
            tot_all += match(pred, chunk, 8) * hist.shape[0]
            tot_first += match(pred, chunk, 1) * hist.shape[0]
            tot_k += match(pred, chunk, execute_steps) * hist.shape[0]
            n += hist.shape[0]
        res[mode] = {
            "per_step": tot_all / n,
            "first_action": tot_first / n,
            f"first_{execute_steps}": tot_k / n,
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--world-model", default=None)
    ap.add_argument("--data", default="data/val")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ae, _ = build_action_expert(args.checkpoint, args.device)
    wm = None
    if args.world_model:
        wm, _ = build_world_model(args.world_model, args.device)

    res = evaluate(ae, wm, args.data, args.device)
    for mode, r in res.items():
        print(f"[{mode} subgoal]")
        print(f"  per-step (all 8): {r['per_step']*100:.1f}%")
        print(f"  first action:      {r['first_action']*100:.1f}%")
        print(f"  first 2 (executed):{r['first_2']*100:.1f}%")


if __name__ == "__main__":
    main()
