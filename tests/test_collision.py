from game.collision import resolve_rect, resolve_boundary, resolve_all
from game.objects import Rect


def test_rect_push_out():
    # disc overlapping from the left of a rect (rect spans x=[6,14])
    px, py, hit = resolve_rect(7.0, 10.0, 2.0, Rect(10, 10, 4, 4))
    assert hit
    assert px < 7.0  # pushed left (away from rect)
    assert abs(py - 10.0) < 1e-6


def test_rect_no_overlap():
    px, py, hit = resolve_rect(0.0, 10.0, 2.0, Rect(10, 10, 4, 4))
    # already resolved; test a far point
    px2, py2, hit2 = resolve_rect(0.0, 0.0, 1.0, Rect(30, 30, 2, 2))
    assert not hit2
    assert (px2, py2) == (0.0, 0.0)


def test_boundary_clamp():
    x, y, hit = resolve_boundary(-5.0, 50.0, 2.0, 48.0, 48.0)
    assert hit
    assert x == 2.0
    assert y == 46.0


def test_resolve_all_no_obstacles():
    x, y, hit = resolve_all(5.0, 5.0, 2.0, [], 48.0, 48.0)
    assert not hit
    assert (x, y) == (5.0, 5.0)
