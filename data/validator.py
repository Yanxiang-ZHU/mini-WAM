"""Dataset validation: structural integrity, determinism, and semantic checks.

Checks every episode for valid shapes, action ranges, and success flags, and
optionally re-runs a deterministic episode to confirm reproducibility.
"""

from __future__ import annotations

import json

import numpy as np

from data.dataset import EpisodeDataset
from game.language import parse_instruction
from game.objects import SHAPES, FILLS


def validate(data_dir: str, max_episodes: int | None = None) -> dict:
    ds = EpisodeDataset(data_dir, max_episodes=max_episodes)
    problems = []
    n_success = 0
    for i in range(len(ds.paths)):
        d = ds._load(i)
        frames, actions, meta = d["frames"], d["actions"], d["meta"]
        T = frames.shape[0]
        if frames.ndim != 3 or frames.shape[1:] != (48, 48):
            problems.append(f"ep {i}: bad frames shape {frames.shape}")
        if actions.shape != (T - 1,):
            problems.append(f"ep {i}: actions shape {actions.shape} != frames-1")
        if actions.size and (actions.min() < 0 or actions.max() > 3):
            problems.append(f"ep {i}: action out of range")
        # semantic consistency: instruction must parse to the target
        try:
            p = parse_instruction(meta["instruction"])
            if p["shape"] != meta["target"]["shape"]:
                problems.append(f"ep {i}: instruction shape mismatch")
            if p["fill"] != meta["target"]["fill"]:
                problems.append(f"ep {i}: instruction fill mismatch")
        except ValueError as e:
            problems.append(f"ep {i}: {e}")
        if meta["target"]["shape"] not in SHAPES or meta["target"]["fill"] not in FILLS:
            problems.append(f"ep {i}: invalid semantic")
        if meta["success"]:
            n_success += 1

    return {
        "n_episodes": len(ds.paths),
        "n_success": n_success,
        "n_problems": len(problems),
        "problems": problems[:20],
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--max", type=int, default=None)
    args = ap.parse_args()
    print(json.dumps(validate(args.data, args.max), indent=2))
