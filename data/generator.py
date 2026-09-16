"""Procedural dataset generation using the deterministic expert.

Each episode is stored as a single compressed ``.npz`` containing the full frame
sequence (uint8), the expert action sequence, and a JSON metadata blob.  Frames
are stored in full so the visual-subgoal horizon can be varied later without
regenerating the data.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from game.env import GameEnv
from game.expert import Expert
from game.renderer import to_uint8

ALL_COMBOS = [(s, f) for s in ("circle", "triangle", "square")
              for f in ("solid", "hollow")]

# Held-out combos for the compositional OOD split (unseen shape x fill pairs).
HELD_OUT_COMBOS = [("circle", "hollow"), ("triangle", "solid"), ("square", "hollow")]
SEEN_COMBOS = [c for c in ALL_COMBOS if c not in HELD_OUT_COMBOS]


@dataclass
class SplitConfig:
    min_distractors: int = 1
    max_distractors: int = 3
    n_obstacles: int = 1
    shape_only_p: float = 0.15
    force_obstacle: bool = False
    force_long: bool = False
    allowed_targets: list | None = None
    allowed_distractors: list | None = None
    accel_range: tuple = (0.32, 0.38)
    friction_range: tuple = (0.78, 0.86)
    max_steps: int = 150


SPLIT_CONFIGS: dict[str, SplitConfig] = {
    # standard training distribution: distractors + occasional obstacles
    "train": SplitConfig(min_distractors=1, max_distractors=3, n_obstacles=1,
                         shape_only_p=0.15),
    "val": SplitConfig(min_distractors=1, max_distractors=3, n_obstacles=1,
                       shape_only_p=0.15),
    # same distribution, fresh seeds
    "test": SplitConfig(min_distractors=1, max_distractors=3, n_obstacles=1,
                        shape_only_p=0.15),
    # compositional OOD: targets restricted to held-out shape x fill combos
    "ood_combo": SplitConfig(min_distractors=2, max_distractors=4, n_obstacles=1,
                             shape_only_p=0.0, allowed_targets=HELD_OUT_COMBOS),
    # distractor-heavy: more distractors than training
    "ood_distractors": SplitConfig(min_distractors=5, max_distractors=8, n_obstacles=1,
                                   shape_only_p=0.0),
    # unseen layouts: more obstacles, forced
    "ood_layout": SplitConfig(min_distractors=1, max_distractors=3, n_obstacles=3,
                              force_obstacle=True, shape_only_p=0.15),
    # long-horizon: forced long distance
    "ood_long": SplitConfig(min_distractors=1, max_distractors=3, n_obstacles=1,
                            force_long=True, shape_only_p=0.15),
}


def generate_episode(rng: np.random.Generator, cfg: SplitConfig) -> dict:
    """Run one expert episode and collect frames + actions + metadata."""
    task_kwargs = dict(
        min_distractors=cfg.min_distractors,
        max_distractors=cfg.max_distractors,
        n_obstacles=cfg.n_obstacles,
        shape_only_p=cfg.shape_only_p,
        force_obstacle=cfg.force_obstacle,
        force_long=cfg.force_long,
        allowed_targets=cfg.allowed_targets,
        allowed_distractors=cfg.allowed_distractors,
    )
    accel = float(rng.uniform(*cfg.accel_range))
    friction = float(rng.uniform(*cfg.friction_range))

    env = GameEnv(task_kwargs=task_kwargs)
    env.cfg.accel = accel
    env.cfg.friction = friction
    env.cfg.max_steps = cfg.max_steps
    env.reset(seed=int(rng.integers(0, 2 ** 31)))

    expert = Expert(env.task)
    expert.reset((env.player.x, env.player.y))

    frames = [env.render()]
    actions = []
    while not env.is_done() and len(actions) < cfg.max_steps:
        a = expert.action((env.player.x, env.player.y), (env.player.vx, env.player.vy))
        env.step(int(a))
        actions.append(int(a))
        frames.append(env.render())

    info = env.get_info()
    frames_arr = np.stack([to_uint8(f) for f in frames]).astype(np.uint8)
    actions_arr = np.asarray(actions, dtype=np.int8)

    meta = {
        "instruction": env.task.instruction,
        "subtask": env.task.subtask,
        "target": {"shape": env.task.target.shape, "fill": env.task.target.fill},
        "metadata": env.task.metadata,
        "shape_only": env.task.shape_only,
        "success": bool(info["success"]),
        "episode_length": int(len(actions)),
        "collision_count": int(info["collision_count"]),
        "n_objects": int(len(env.task.objects())),
        "n_obstacles": int(len(env.task.obstacles)),
        "accel": accel,
        "friction": friction,
    }
    return {"frames": frames_arr, "actions": actions_arr, "meta": meta}


def save_episode(episode: dict, outdir: str, idx: int) -> str:
    path = os.path.join(outdir, f"episode_{idx:06d}.npz")
    np.savez_compressed(
        path,
        frames=episode["frames"],
        actions=episode["actions"],
        meta_json=np.array(json.dumps(episode["meta"])),
    )
    return path


def _worker(args):
    idx, base_seed, cfg, outdir = args
    rng = np.random.default_rng(base_seed + idx)
    ep = generate_episode(rng, cfg)
    path = save_episode(ep, outdir, idx)
    return idx, ep["meta"]["success"], ep["meta"]["episode_length"]


def split_to_task_kwargs(cfg: SplitConfig) -> dict:
    """Convert a SplitConfig into generate_task kwargs for live OOD evaluation."""
    return dict(
        min_distractors=cfg.min_distractors,
        max_distractors=cfg.max_distractors,
        n_obstacles=cfg.n_obstacles,
        shape_only_p=cfg.shape_only_p,
        force_obstacle=cfg.force_obstacle,
        force_long=cfg.force_long,
        allowed_targets=cfg.allowed_targets,
        allowed_distractors=cfg.allowed_distractors,
    )


def generate_split(split: str, n_episodes: int, outdir: str, seed: int = 0,
                   workers: int = 8, cfg: SplitConfig | None = None) -> list[tuple]:
    """Generate ``n_episodes`` episodes for a split, writing ``.npz`` files."""
    os.makedirs(outdir, exist_ok=True)
    cfg = cfg if cfg is not None else SPLIT_CONFIGS[split]
    args = [(i, seed, cfg, outdir) for i in range(n_episodes)]
    results = []
    if workers > 1 and n_episodes > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for r in pool.map(_worker, args, chunksize=16):
                results.append(r)
    else:
        for a in args:
            results.append(_worker(a))
    # manifest
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump({
            "split": split, "n_episodes": n_episodes, "seed": seed,
            "config": {k: v for k, v in vars(cfg).items()},
        }, f, indent=2)
    return results
