import math

import numpy as np

from game.objects import (SHAPES, FILLS, Object, Rect, shape_contains, shape_hit,
                          AREA_SCALE, INSET)


def test_six_categories():
    cats = {(s, f) for s in SHAPES for f in FILLS}
    assert len(cats) == 6


def test_solid_center_inside():
    for s in SHAPES:
        assert shape_contains(s, 0, 0, 0, 0, 5.0), s


def test_hollow_center_is_empty():
    for s in SHAPES:
        assert not shape_hit(s, "hollow", 0, 0, 0, 0, 5.0, 1.5), s


def test_hollow_boundary_hit():
    for s in SHAPES:
        # a point on the boundary should be hit for hollow
        r = 5.0 * AREA_SCALE[s]
        # sample along a ray to find an outline pixel
        hit = False
        for t in np.linspace(0.0, r, 200):
            if shape_hit(s, "hollow", t, 0.0, 0.0, 0.0, 5.0, 1.5):
                hit = True
                break
        assert hit, f"hollow {s} outline not detected"


def test_solid_vs_hollow_different():
    for s in SHAPES:
        # a point near the centre (inside the hollow empty core): solid hit, hollow miss
        inner = 0.1 * AREA_SCALE[s] * 5.0
        assert shape_hit(s, "solid", inner, 0, 0, 0, 5.0, 1.5)
        assert not shape_hit(s, "hollow", inner, 0, 0, 0, 5.0, 1.5)


def test_area_normalization_roughly_equal():
    # solid areas should be comparable (within a factor) across shapes
    from game.renderer import render_frame
    counts = {}
    for s in SHAPES:
        fr = render_frame((10, 10), [Object(s, "solid", 24, 24, 5.0)], [])
        counts[s] = int((fr > 0.5).sum())
    mx, mn = max(counts.values()), min(counts.values())
    assert mn / mx > 0.5, counts  # no wildly smaller shape


def test_rect_contains():
    r = Rect(10, 10, 2, 3)
    assert r.contains(10, 10)
    assert r.contains(12, 13)
    assert not r.contains(13, 10)


def test_semantic_tuple():
    o = Object("circle", "solid", 1, 2)
    assert o.semantic() == ("circle", "solid")
