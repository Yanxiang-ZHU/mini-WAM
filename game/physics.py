"""Simple continuous-time-ish player dynamics.

    v_{t+1} = lambda * v_t + a_t
    x_{t+1} = x_t + v_{t+1}

WASD selects the acceleration direction (in image coordinates, y grows
downward, so "up" is -y).  ``lambda`` is the friction coefficient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Action index -> (ax, ay) unit direction, scaled by accel magnitude at call site.
ACTION_DIRS = {
    0: (0.0, -1.0),  # W = up
    1: (-1.0, 0.0),  # A = left
    2: (0.0, 1.0),   # S = down
    3: (1.0, 0.0),   # D = right
}

ACTION_NAMES = ("W", "A", "S", "D")
NUM_ACTIONS = 4


@dataclass
class Player:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


def step_player(p: Player, accel: tuple[float, float], friction: float,
                max_speed: float) -> None:
    """Advance one step.  ``accel`` is the already-scaled acceleration vector."""
    p.vx = friction * p.vx + accel[0]
    p.vy = friction * p.vy + accel[1]

    speed = math.hypot(p.vx, p.vy)
    if speed > max_speed:
        scale = max_speed / speed
        p.vx *= scale
        p.vy *= scale

    p.x += p.vx
    p.y += p.vy


def action_to_accel(action_idx: int, accel_mag: float) -> tuple[float, float]:
    ax, ay = ACTION_DIRS[action_idx]
    return (ax * accel_mag, ay * accel_mag)
