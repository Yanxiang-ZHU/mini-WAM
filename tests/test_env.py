import numpy as np

from game.env import GameEnv


def test_reset_observation_shape():
    env = GameEnv()
    obs = env.reset(seed=0)
    assert obs.shape == (4, 48, 48)


def test_deterministic_reset():
    e1, e2 = GameEnv(), GameEnv()
    o1 = e1.reset(seed=42)
    o2 = e2.reset(seed=42)
    assert np.array_equal(o1, o2)
    assert e1.task.instruction == e2.task.instruction


def test_step_returns_tuple():
    env = GameEnv()
    env.reset(seed=0)
    obs, reward, done, info = env.step(3)  # D
    assert obs.shape == (4, 48, 48)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "success" in info


def test_history_updates():
    env = GameEnv()
    env.reset(seed=0)
    o0 = env.get_observation()
    for _ in range(3):
        env.step(3)
    o1 = env.get_observation()
    # last frame should differ; buffer slides
    assert not np.array_equal(o0[-1], o1[-1])


def test_get_state_has_positions():
    env = GameEnv()
    env.reset(seed=0)
    s = env.get_state()
    assert "player" in s and "target" in s and "objects" in s


def test_success_terminates():
    # place player right next to target
    env = GameEnv()
    env.reset(seed=0)
    env.player.x = env.task.target.x
    env.player.y = env.task.target.y - 1.0
    obs, r, done, info = env.step(0)
    assert done and info["success"]
