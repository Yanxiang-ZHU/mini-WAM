import numpy as np

from game.renderer import render_frame, to_uint8, BG
from game.objects import Object, Rect


def test_output_shape_and_range():
    fr = render_frame((24, 24), [], [])
    assert fr.shape == (48, 48)
    assert fr.dtype == np.float32
    assert fr.min() >= 0.0 and fr.max() <= 1.0


def test_background_when_empty():
    fr = render_frame((24, 24), [], [])
    # the player cross is drawn at (24,24); corners far from it stay background
    assert np.allclose(fr[:4, :4], BG, atol=1e-6)
    assert np.allclose(fr[-4:, -4:], BG, atol=1e-6)


def test_deterministic():
    a = render_frame((10, 20), [Object("circle", "solid", 30, 30, 5.0)], [Rect(20, 20, 3, 5)])
    b = render_frame((10, 20), [Object("circle", "solid", 30, 30, 5.0)], [Rect(20, 20, 3, 5)])
    assert np.array_equal(a, b)


def test_obstacle_rendered():
    fr = render_frame((24, 24), [], [Rect(24, 24, 4, 6)])
    assert (fr > 0.2).any()


def test_player_rendered():
    empty = render_frame((24, 24), [], [])
    with_player = render_frame((30, 30), [], [])
    assert not np.array_equal(empty, with_player)


def test_to_uint8():
    fr = render_frame((24, 24), [Object("circle", "solid", 30, 30, 5.0)], [])
    u8 = to_uint8(fr)
    assert u8.dtype == np.uint8
    assert u8.max() <= 255
