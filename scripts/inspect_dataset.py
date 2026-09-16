"""Dataset inspector: statistics + visualizations.

Usage:
    python scripts/inspect_dataset.py --data data/train --out _inspect
"""

import argparse
import json
import os
from collections import Counter

import numpy as np

from data.dataset import EpisodeDataset


def stats(data_dir: str) -> dict:
    ds = EpisodeDataset(data_dir, max_episodes=None)
    n_ep = len(ds.paths)
    n_samples = len(ds.index)

    lens, succ, coll, n_obj, n_ob, deltas = [], 0, 0, [], [], []
    task_counter, obj_counter, action_counter = Counter(), Counter(), Counter()

    for i in range(n_ep):
        d = ds._load(i)
        m = d["meta"]
        lens.append(m["episode_length"])
        succ += m["success"]
        coll += m["collision_count"]
        n_obj.append(m["n_objects"])
        n_ob.append(m["n_obstacles"])
        task_counter[m["target"]["shape"] + " " + m["target"]["fill"]] += 1
        obj_counter[m["target"]["shape"] + " " + m["target"]["fill"]] += 1
        for a in d["actions"]:
            action_counter[int(a)] += 1

    for _, _, delta in ds.index:
        deltas.append(delta)

    return {
        "n_episodes": n_ep,
        "n_samples": n_samples,
        "success_rate": succ / max(1, n_ep),
        "collision_rate": coll / max(1, n_ep),
        "episode_length_mean": float(np.mean(lens)),
        "episode_length_std": float(np.std(lens)),
        "episode_length_min": int(np.min(lens)) if lens else 0,
        "episode_length_max": int(np.max(lens)) if lens else 0,
        "n_objects_mean": float(np.mean(n_obj)),
        "n_obstacles_mean": float(np.mean(n_ob)),
        "task_distribution": dict(task_counter),
        "object_distribution": dict(obj_counter),
        "action_distribution": {int(k): v for k, v in action_counter.items()},
        "subgoal_delta_distribution": dict(Counter(deltas)),
    }


def visualize(ds: EpisodeDataset, out_dir: str, n: int = 6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    for j, idx in enumerate(idxs):
        s = ds[idx]
        hist = s["history"]              # (K,48,48)
        sub = s["subgoal"]               # (48,48)
        actions = s["action_chunk"]
        fig, axes = plt.subplots(1, hist.shape[0] + 2, figsize=(14, 3))
        for k in range(hist.shape[0]):
            axes[k].imshow(hist[k], cmap="gray", vmin=-1, vmax=1)
            axes[k].set_title(f"t-{hist.shape[0]-1-k}")
            axes[k].axis("off")
        axes[-2].imshow(sub, cmap="gray", vmin=-1, vmax=1)
        axes[-2].set_title(f"subgoal +{s['subgoal_delta']}")
        axes[-2].axis("off")
        axes[-1].axis("off")
        axes[-1].text(0.05, 0.5,
                      f"'{s['instruction']}'\n{s['subtask']}\nactions: {list(actions)}",
                      fontsize=8, va="center")
        fig.suptitle(f"target={s['target_shape']} {s['target_fill']}", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sample_{j:02d}.png"), dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--out", default="_inspect")
    ap.add_argument("--viz", type=int, default=8)
    args = ap.parse_args()

    st = stats(args.data)
    print(json.dumps(st, indent=2, ensure_ascii=False))

    if args.viz > 0:
        ds = EpisodeDataset(args.data)
        visualize(ds, args.out, n=args.viz)
        print(f"visualizations written to {args.out}")


if __name__ == "__main__":
    main()
