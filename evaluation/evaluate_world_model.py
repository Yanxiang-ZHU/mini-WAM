"""World Model evaluation: pixel fidelity + semantic correctness.

Loads the trained world model, generates subgoals for held-out samples, and
reports pixel MSE plus a template-matching score for whether the generated
subgoal preserves the requested target object (shape x fill identity).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from data.dataset import EpisodeDataset, normalize_frames
from training.utils import build_world_model, tokenize_batch
from evaluation.metrics import object_presence_score, pixel_mse, classify_object


@torch.no_grad()
def evaluate_world_model(wm, data_dir: str, n: int = 200, device: str = "cuda",
                         seed: int = 0) -> dict:
    ds = EpisodeDataset(data_dir, preload=True)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=min(n, len(ds)), replace=False)

    mses, target_presence_real, target_presence_gen = [], [], []
    identity_correct = 0
    for idx in idxs:
        s = ds[idx]
        history = torch.from_numpy(s["history"]).unsqueeze(0).to(device)
        real = s["subgoal"]  # (48,48) in [-1,1]
        lang = tokenize_batch([s["instruction"]], device)
        sub = tokenize_batch([s["subtask"]], device)
        meta = [s["metadata"]]

        gen = wm.sample(history, lang, sub, meta)  # (1,1,48,48)
        gen_np = gen[0, 0].cpu().numpy()

        mses.append(pixel_mse(real, gen_np))

        shape, fill = s["target_shape"], s["target_fill"]
        target_presence_real.append(object_presence_score(real, shape, fill))
        target_presence_gen.append(object_presence_score(gen_np, shape, fill))

        pred = classify_object(gen_np)
        if pred == (shape, fill):
            identity_correct += 1

    return {
        "n": len(idxs),
        "pixel_mse_mean": float(np.mean(mses)),
        "target_presence_real": float(np.mean(target_presence_real)),
        "target_presence_generated": float(np.mean(target_presence_gen)),
        "target_identity_accuracy": identity_correct / max(1, len(idxs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/val")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    wm, _ = build_world_model(args.checkpoint, args.device)
    res = evaluate_world_model(wm, args.data, args.n, args.device)
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
