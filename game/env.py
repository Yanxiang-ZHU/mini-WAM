"""Core game environment: a deterministic 2D top-down navigation task.

Exposes a clean Gym-style API and maintains a K-frame observation history.  The
true state is available via ``get_state()`` for the expert / evaluation /
debugging, but the neural policy only ever sees rendered grayscale frames plus
language / subtask / metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .objects import Object, Rect
from .physics import Player, step_player, action_to_accel, NUM_ACTIONS
from .renderer import render_frame
from .collision import resolve_all
from .tasks import generate_task, Task, TARGET_REACH_RADIUS, PLAYER_RADIUS


@dataclass
class EnvConfig:
    width: int = 48
    height: int = 48
    accel: float = 0.35
    friction: float = 0.82
    max_speed: float = 3.0
    history: int = 4
    max_steps: int = 200
    reach_radius: float = TARGET_REACH_RADIUS
    player_radius: float = PLAYER_RADIUS


class GameEnv:
    def __init__(self, config: EnvConfig | None = None, *, task_kwargs: dict | None = None):
        self.cfg = config or EnvConfig()
        self.task_kwargs = task_kwargs or {}
        self.rng = np.random.default_rng(0)
        self._reset_state()

    # -- state ------------------------------------------------------------ #
    def _reset_state(self):
        self.player = Player(24.0, 24.0)
        self.task: Task | None = None
        self.steps = 0
        self.done = False
        self.success = False
        self.collision_count = 0
        self.history = np.zeros((self.cfg.history, self.cfg.height, self.cfg.width),
                                dtype=np.float32)
        self._last_frame = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)

    # -- API -------------------------------------------------------------- #
    def reset(self, seed: int | None = None, task: Task | None = None) -> np.ndarray:
        self.rng = np.random.default_rng(seed if seed is not None else 0)
        self._reset_state()
        self.task = task if task is not None else generate_task(self.rng, **self.task_kwargs)
        self.player = Player(self.task.player_xy[0], self.task.player_xy[1])
        self._last_frame = render_frame(
            (self.player.x, self.player.y), self.task.objects(), self.task.obstacles,
            self.cfg.width, self.cfg.height)
        for k in range(self.cfg.history):
            self.history[k] = self._last_frame
        return self.get_observation()

    def step(self, action: int):
        ax, ay = action_to_accel(action, self.cfg.accel)
        step_player(self.player, (ax, ay), self.cfg.friction, self.cfg.max_speed)
        self.player.x, self.player.y, hit = resolve_all(
            self.player.x, self.player.y, self.cfg.player_radius,
            self.task.obstacles, self.cfg.width, self.cfg.height)
        if hit:
            self.collision_count += 1
        self.steps += 1

        d = math.hypot(self.player.x - self.task.target.x,
                       self.player.y - self.task.target.y)
        if d < self.cfg.reach_radius:
            self.success = True
            self.done = True
        elif self.steps >= self.cfg.max_steps:
            self.done = True

        self._last_frame = render_frame(
            (self.player.x, self.player.y), self.task.objects(), self.task.obstacles,
            self.cfg.width, self.cfg.height)
        self.history[:-1] = self.history[1:]
        self.history[-1] = self._last_frame

        reward = 1.0 if self.success else 0.0
        return self.get_observation(), reward, self.done, self.get_info()

    def get_observation(self) -> np.ndarray:
        return self.history.copy()

    def render(self) -> np.ndarray:
        return self._last_frame.copy()

    def get_state(self) -> dict:
        t = self.task.target
        return {
            "player": (self.player.x, self.player.y),
            "player_vel": (self.player.vx, self.player.vy),
            "target": (t.x, t.y),
            "target_semantic": (t.shape, t.fill),
            "objects": [(o.x, o.y, o.shape, o.fill) for o in self.task.objects()],
            "obstacles": [(o.cx, o.cy, o.hw, o.hh) for o in self.task.obstacles],
        }

    def get_info(self) -> dict:
        t = self.task.target
        d = math.hypot(self.player.x - t.x, self.player.y - t.y)
        return {
            "success": self.success,
            "episode_length": self.steps,
            "final_distance": d,
            "collision_count": self.collision_count,
            "distance_to_target": d,
            "target_reached": self.success,
            "instruction": self.task.instruction,
            "subtask": self.task.subtask,
            "target_semantic": (t.shape, t.fill),
        }

    def is_done(self) -> bool:
        return self.done
