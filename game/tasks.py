"""Task / scene generation.

Randomly samples a target object (shape + optional fill), places the player,
target, distractors and obstacles, and emits a natural instruction + subtask.

The task types (A-F) are realised by controlling the number of distractors,
whether shape-only instructions are used, obstacle placement, and distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .objects import SHAPES, FILLS, Object, Rect
from .language import make_instruction, make_subtask

W = 48.0
H = 48.0
PLAYER_RADIUS = 1.6
OBJECT_SIZE = 4.0
TARGET_REACH_RADIUS = 4.5


@dataclass
class Task:
    target: Object
    distractors: list[Object]
    obstacles: list[Rect]
    player_xy: tuple[float, float]
    instruction: str
    subtask: str
    shape_only: bool = False
    metadata: dict = field(default_factory=dict)

    def objects(self) -> list[Object]:
        return [self.target] + self.distractors


def _rand_point(rng, margin: float = 8.0):
    return float(rng.uniform(margin, W - margin)), float(rng.uniform(margin, H - margin))


def _sample_object(rng, shape: str, fill: str, x: float, y: float) -> Object:
    return Object(shape=shape, fill=fill, x=x, y=y, size=OBJECT_SIZE)


def _too_close(ax, ay, bx, by, dmin):
    return math.hypot(ax - bx, ay - by) < dmin


def _obstacle_blocks(ob: Rect, ax, ay, bx, by) -> bool:
    """Rough segment-vs-rect test for whether an obstacle intersects the direct
    line from a to b (used to decide if navigation is required)."""
    # sample the segment
    steps = 24
    for i in range(steps + 1):
        t = i / steps
        px = ax + (bx - ax) * t
        py = ay + (by - ay) * t
        if ob.contains(px, py):
            return True
    return False


def generate_task(rng: np.random.Generator, *, min_distractors: int = 0,
                  max_distractors: int = 3, shape_only_p: float = 0.0,
                  force_obstacle: bool = False, force_long: bool = False,
                  force_shape_only: bool = False, n_obstacles: int = 1,
                  allowed_targets: list | None = None,
                  allowed_distractors: list | None = None) -> Task:
    """Generate one random task.

    ``allowed_targets`` / ``allowed_distractors`` restrict the semantic object
    vocabulary (used for compositional OOD splits).  Each entry is (shape, fill).
    """
    if allowed_targets is not None:
        shape, fill = allowed_targets[rng.integers(len(allowed_targets))]
    else:
        shape = rng.choice(list(SHAPES))
        fill = rng.choice(list(FILLS))
    shape_only = force_shape_only or (rng.random() < shape_only_p)

    # place player and target reasonably far apart
    px, py = _rand_point(rng, margin=6.0)
    for _ in range(100):
        tx, ty = _rand_point(rng, margin=6.0)
        d = math.hypot(tx - px, ty - py)
        min_d = 30.0 if force_long else 12.0
        if d >= min_d:
            break
    else:
        tx, ty = _rand_point(rng, margin=6.0)

    target = _sample_object(rng, shape, fill, tx, ty)

    # distractors: distinct semantic objects, placed away from player and target
    n_dist = int(rng.integers(min_distractors, max_distractors + 1))
    distractors: list[Object] = []
    if allowed_distractors is not None:
        dist_pool = list(allowed_distractors)
    else:
        dist_pool = [(s, f) for s in SHAPES for f in FILLS]
    attempts = 0
    while len(distractors) < n_dist and attempts < 200:
        attempts += 1
        s, f = dist_pool[rng.integers(len(dist_pool))]
        if (s, f) == (shape, fill):
            continue
        dx, dy = _rand_point(rng, margin=6.0)
        if _too_close(dx, dy, px, py, 8.0):
            continue
        if _too_close(dx, dy, tx, ty, 8.0):
            continue
        if any(_too_close(dx, dy, o.x, o.y, 8.0) for o in distractors):
            continue
        distractors.append(_sample_object(rng, s, f, dx, dy))

    # obstacles (axis-aligned rectangles), possibly blocking the direct path
    obstacles: list[Rect] = []
    if force_obstacle or n_obstacles > 0:
        placed = 0
        for _ in range(100):
            if placed >= n_obstacles:
                break
            hw = float(rng.uniform(3.0, 6.0))
            hh = float(rng.uniform(5.0, 12.0))
            ox = float(rng.uniform(10.0, W - 10.0))
            oy = float(rng.uniform(10.0, H - 10.0))
            ob = Rect(ox, oy, hw, hh)
            # keep clear of player, target, objects
            if math.hypot(ox - px, oy - py) < 8.0 or math.hypot(ox - tx, oy - ty) < 8.0:
                continue
            if any(_too_close(ox, oy, o.x, o.y, 9.0) for o in distractors):
                continue
            obstacles.append(ob)
            placed += 1

    instruction = make_instruction(shape, None if shape_only else fill, rng, shape_only)
    subtask = make_subtask(shape, None if shape_only else fill, rng)
    metadata = {"speed": "normal", "quality": 1.0, "mistake": False,
                "control_mode": "keyboard"}

    return Task(target=target, distractors=distractors, obstacles=obstacles,
                player_xy=(px, py), instruction=instruction, subtask=subtask,
                shape_only=shape_only, metadata=metadata)
