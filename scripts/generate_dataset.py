"""Generate an expert dataset split.

Usage:
    python scripts/generate_dataset.py --episodes 10000 --split train --output data/train
"""

import argparse
import os

from data.generator import generate_split, SPLIT_CONFIGS, SplitConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--split", default="train", choices=list(SPLIT_CONFIGS) + ["custom"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", default="data/train")
    ap.add_argument("--min-distractors", type=int, default=None)
    ap.add_argument("--max-distractors", type=int, default=None)
    ap.add_argument("--n-obstacles", type=int, default=None)
    ap.add_argument("--shape-only-p", type=float, default=None)
    args = ap.parse_args()

    cfg = SPLIT_CONFIGS.get(args.split, SplitConfig())
    if args.min_distractors is not None:
        cfg.min_distractors = args.min_distractors
    if args.max_distractors is not None:
        cfg.max_distractors = args.max_distractors
    if args.n_obstacles is not None:
        cfg.n_obstacles = args.n_obstacles
    if args.shape_only_p is not None:
        cfg.shape_only_p = args.shape_only_p

    print(f"generating split={args.split} episodes={args.episodes} "
          f"workers={args.workers} -> {args.output}")
    results = generate_split(args.split, args.episodes, args.output,
                             seed=args.seed, workers=args.workers, cfg=cfg)
    succ = sum(1 for _, s, _ in results if s)
    print(f"done: {len(results)} episodes, {succ} success "
          f"({100.0 * succ / max(1, len(results)):.1f}%)")


if __name__ == "__main__":
    main()
