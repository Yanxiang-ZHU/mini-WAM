"""Evaluation metrics and a deterministic CV parser for generated subgoals.

The parser uses template matching against rendered canonical shapes (legitimate:
it uses the same geometric object definitions, not the model) to score whether
a generated subgoal contains a requested (shape, fill) object.
"""

from __future__ import annotations

import numpy as np

from game.objects import Object
from game.renderer import render_frame

_TPL = {}
_TPL_HALF = 10  # crop half-width around the object centre for the template


def _templates():
    if _TPL:
        return _TPL
    c = 24
    for shape in ("circle", "triangle", "square"):
        for fill in ("solid", "hollow"):
            fr = render_frame((10, 10), [Object(shape, fill, c, c, 5.0)], [])
            # crop a small window around the centred object
            crop = fr[c - _TPL_HALF:c + _TPL_HALF, c - _TPL_HALF:c + _TPL_HALF]
            _TPL[(shape, fill)] = crop.astype(np.float32)
    return _TPL


def object_presence_score(frame: np.ndarray, shape: str, fill: str) -> float:
    """Max normalised cross-correlation of ``frame`` with the (shape,fill)
    template (a small crop), slid across the frame.  Values near 1 indicate the
    object is present."""
    tpl = _templates()[(shape, fill)]
    f = frame.astype(np.float32)
    t = tpl - tpl.mean()
    t_norm = np.sqrt((t * t).sum()) + 1e-8
    H, W = f.shape
    th, tw = t.shape
    best = -1.0
    for y in range(0, H - th + 1, 2):
        for x in range(0, W - tw + 1, 2):
            patch = f[y:y + th, x:x + tw]
            patch = patch - patch.mean()
            num = (patch * t).sum()
            d = np.sqrt((patch * patch).sum()) * t_norm + 1e-8
            best = max(best, num / d)
    return float(best)


def classify_object(frame: np.ndarray) -> tuple[str, str] | None:
    """Classify the dominant object in ``frame`` into (shape, fill)."""
    best, best_key = -1.0, None
    for shape in ("circle", "triangle", "square"):
        for fill in ("solid", "hollow"):
            s = object_presence_score(frame, shape, fill)
            if s > best:
                best, best_key = s, (shape, fill)
    return best_key


def pixel_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a - b) ** 2).mean())


def summarize(results: list[dict]) -> dict:
    """Aggregate a list of per-episode result dicts into summary metrics."""
    n = len(results)
    succ = sum(1 for r in results if r["success"])
    lens = [r["episode_length"] for r in results]
    dists = [r["final_distance"] for r in results]
    colls = [r.get("collision_count", 0) for r in results]
    out = {
        "n_episodes": n,
        "success_rate": succ / max(1, n),
        "avg_episode_length": float(np.mean(lens)) if lens else 0.0,
        "avg_final_distance": float(np.mean(dists)) if dists else 0.0,
        "avg_collisions": float(np.mean(colls)) if colls else 0.0,
    }
    if all("expert_length" in r for r in results):
        eff = [r["expert_length"] / max(1, r["episode_length"]) for r in results]
        out["action_efficiency"] = float(np.mean(eff)) if eff else 0.0
    return out
