import numpy as np

from game.env import GameEnv
from game.expert import Expert


def _solve(seed, **kwargs):
    env = GameEnv(task_kwargs=kwargs)
    env.reset(seed=seed)
    exp = Expert(env.task)
    exp.reset((env.player.x, env.player.y))
    steps = 0
    while not env.is_done() and steps < 400:
        a = exp.action((env.player.x, env.player.y), (env.player.vx, env.player.vy))
        env.step(a)
        steps += 1
    return env.get_info(), steps


def test_expert_direct_navigation():
    info, steps = _solve(0, n_obstacles=0)
    assert info["success"], info
    assert steps < 400


def test_expert_obstacle_navigation():
    info, steps = _solve(5, force_obstacle=True, n_obstacles=2)
    assert info["success"], info


def test_expert_many_seeds():
    for seed in range(10):
        info, _ = _solve(seed, n_obstacles=1)
        assert info["success"], f"seed {seed}: {info}"
