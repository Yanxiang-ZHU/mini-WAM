import numpy as np

from game.tasks import generate_task
from game.objects import SHAPES, FILLS


def test_task_generation():
    rng = np.random.default_rng(0)
    t = generate_task(rng)
    assert t.target.shape in SHAPES
    assert t.target.fill in FILLS
    assert t.instruction
    assert t.subtask
    assert t.metadata["control_mode"] == "keyboard"


def test_distractors_are_distinct_semantics():
    rng = np.random.default_rng(1)
    for _ in range(100):
        t = generate_task(rng, min_distractors=2, max_distractors=2)
        target_sem = t.target.semantic()
        for d in t.distractors:
            assert d.semantic() != target_sem


def test_obstacle_task_has_obstacles():
    rng = np.random.default_rng(2)
    t = generate_task(rng, force_obstacle=True, n_obstacles=2)
    assert len(t.obstacles) >= 1


def test_objects_list_includes_target():
    rng = np.random.default_rng(3)
    t = generate_task(rng)
    assert t.objects()[0] is t.target


def test_shape_only_flag():
    rng = np.random.default_rng(4)
    t = generate_task(rng, force_shape_only=True)
    assert t.shape_only
