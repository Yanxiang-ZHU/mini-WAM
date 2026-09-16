"""Sample visual subgoals from a trained world model and save side-by-side
comparisons with the real future frame (for visual inspection).

Usage:
    python scripts/sample_subgoal.py --checkpoint checkpoints/world_model_goal_wide_best.pt --goal --n 8 --out _subgoals
"""

import argparse
import os

import numpy as np
import torch

from data.dataset import EpisodeDataset
from training.utils import build_world_model, tokenize_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/val")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="_subgoals")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--goal", action="store_true", help="use goal-state subgoals (terminal frame)")
    args = ap.parse_args()

    wm, _ = build_world_model(args.checkpoint, args.device)
    ds = EpisodeDataset(args.data, preload=True, goal=args.goal)
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(args.out, exist_ok=True)

    for j, idx in enumerate(idxs):
        s = ds[idx]
        history = torch.from_numpy(s["history"]).unsqueeze(0).to(args.device)
        lang = tokenize_batch([s["instruction"]], args.device)
        sub = tokenize_batch([s["subtask"]], args.device)
        meta = [s["metadata"]]
        gen = wm.sample(history, lang, sub, meta)[0, 0].cpu().numpy()

        fig, axes = plt.subplots(1, history.shape[1] + 2, figsize=(16, 3))
        for k in range(history.shape[1]):
            axes[k].imshow(s["history"][k], cmap="gray", vmin=-1, vmax=1)
            axes[k].set_title(f"t-{history.shape[1]-1-k}")
            axes[k].axis("off")
        axes[-2].imshow(gen, cmap="gray", vmin=-1, vmax=1)
        axes[-2].set_title("generated subgoal")
        axes[-2].axis("off")
        axes[-1].imshow(s["subgoal"], cmap="gray", vmin=-1, vmax=1)
        axes[-1].set_title(f"real +{s['subgoal_delta']}")
        axes[-1].axis("off")
        fig.suptitle(f"'{s['instruction']}'  target={s['target_shape']} {s['target_fill']}",
                     fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"subgoal_{j:02d}.png"), dpi=110)
        plt.close(fig)

    print(f"saved {len(idxs)} subgoal comparisons to {args.out}")


if __name__ == "__main__":
    main()
