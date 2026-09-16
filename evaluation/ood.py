"""OOD benchmark: evaluate an agent across distribution-shift splits.

Splits (from ``data.generator.SPLIT_CONFIGS``):
    test           -- same distribution as training (fresh seeds)
    ood_combo      -- held-out shape x fill combinations (compositional OOD)
    ood_distractors-- more distractors than training
    ood_layout     -- more obstacles / unseen layouts
    ood_long       -- long-horizon navigation
"""

from __future__ import annotations

import json

from data.generator import SPLIT_CONFIGS, split_to_task_kwargs
from evaluation.evaluate_policy import evaluate_agent

OOD_SPLITS = ["test", "ood_combo", "ood_distractors", "ood_layout", "ood_long"]


def run_ood_benchmark(agent, *, n_episodes: int = 200, execute_steps: int = 2,
                      seed: int = 0, splits: list[str] | None = None) -> dict:
    """Evaluate ``agent`` on each OOD split and return a per-split summary."""
    splits = splits or OOD_SPLITS
    out = {}
    for split in splits:
        cfg = SPLIT_CONFIGS[split]
        kw = split_to_task_kwargs(cfg)
        print(f"[{split}]", flush=True)
        summary = evaluate_agent(agent, n_episodes=n_episodes,
                                 execute_steps=execute_steps, seed=seed,
                                 task_kwargs=kw, verbose=True)
        summary["split_config"] = {k: (list(v) if isinstance(v, list) else v)
                                   for k, v in vars(cfg).items() if v is not None}
        out[split] = summary
    return out


def print_ood_table(results: dict):
    header = f"{'split':18s} {'success':>8s} {'avg_len':>8s} {'avg_dist':>9s} {'coll':>6s}"
    print(header)
    print("-" * len(header))
    for split, r in results.items():
        print(f"{split:18s} {r['success_rate']*100:7.1f}% {r['avg_episode_length']:8.1f} "
              f"{r['avg_final_distance']:9.2f} {r['avg_collisions']:6.1f}")


if __name__ == "__main__":
    import argparse
    from training.utils import build_simple_policy
    from evaluation.evaluate_policy import SimplePolicyAgent
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--execute-steps", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    model, _ = build_simple_policy(args.checkpoint, args.device)
    agent = SimplePolicyAgent(model, args.device)
    results = run_ood_benchmark(agent, n_episodes=args.episodes,
                                execute_steps=args.execute_steps)
    print_ood_table(results)
    print(json.dumps(results, indent=2, default=str))
