"""Object vocabulary and geometric primitives for the 2D grayscale game.

Six semantic object categories are defined by shape x fill:

    shape in {circle, triangle, square}
    fill  in {solid, hollow}

No colour information is used anywhere.  Identity is recovered purely from
geometry (shape outline) and fill state (interior vs. outline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SHAPES = ("circle", "triangle", "square")
FILLS = ("solid", "hollow")

# Per-shape scale factor so that, for a given nominal ``size``, all three shapes
# have EQUAL area (circle radius = size as reference).  This removes an easy
# "the smallest object is the triangle" shortcut and forces real shape grounding.
#   circle   area = pi * r^2          (r = size)
#   square   area = (2*half)^2        (half = size * scale)
#   triangle area = (3*sqrt3/4) * R^2 (R = circumradius = size * scale)
_AREA = math.pi  # target area for size == 1
AREA_SCALE = {
    "circle": 1.0,
    "square": math.sqrt(_AREA) / 2.0,
    "triangle": math.sqrt(4.0 * _AREA / (3.0 * math.sqrt(3.0))),
}

# To shrink a shape by a uniform edge thickness ``t``, its ``size`` parameter must
# be reduced by ``t * INSET`` (for a triangle the inradius is size/2, so reducing
# size by t only moves each edge inward by t/2; we compensate).
INSET = {"circle": 1.0, "square": 1.0, "triangle": 2.0}


@dataclass
class Object:
    """A semantic game object (target or distractor).

    ``x``/``y`` is the centre in pixel coordinates; ``size`` is the half-extent
    (circle radius, square half-side, triangle circumradius) in pixels.
    """

    shape: str
    fill: str
    x: float
    y: float
    size: float = 4.0

    def semantic(self) -> tuple[str, str]:
        return (self.shape, self.fill)


@dataclass
class Rect:
    """An axis-aligned rectangular obstacle, given by centre + half-extents."""

    cx: float
    cy: float
    hw: float  # half width
    hh: float  # half height

    def contains(self, px: float, py: float) -> bool:
        return abs(px - self.cx) <= self.hw and abs(py - self.cy) <= self.hh


# --------------------------------------------------------------------------- #
# Point-in-shape tests (solid interior)
# --------------------------------------------------------------------------- #

def _in_circle(px, py, cx, cy, size):
    dx, dy = px - cx, py - cy
    return dx * dx + dy * dy <= size * size


def _in_square(px, py, cx, cy, size):
    return abs(px - cx) <= size and abs(py - cy) <= size


_SQRT3_2 = math.sqrt(3.0) / 2.0


def _tri_verts(cx, cy, size):
    """Vertices of an equilateral triangle pointing up, circumradius ``size``."""
    return [
        (cx, cy - size),
        (cx - _SQRT3_2 * size, cy + 0.5 * size),
        (cx + _SQRT3_2 * size, cy + 0.5 * size),
    ]


def _in_triangle(px, py, cx, cy, size):
    a, b, c = _tri_verts(cx, cy, size)

    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    s1 = cross(a, b, (px, py))
    s2 = cross(b, c, (px, py))
    s3 = cross(c, a, (px, py))
    has_neg = (s1 < 0) or (s2 < 0) or (s3 < 0)
    has_pos = (s1 > 0) or (s2 > 0) or (s3 > 0)
    return not (has_neg and has_pos)


_SHAPE_FN = {
    "circle": _in_circle,
    "square": _in_square,
    "triangle": _in_triangle,
}


def shape_contains(shape: str, px: float, py: float, cx: float, cy: float, size: float) -> bool:
    """True if point (px,py) is inside the solid shape centred at (cx,cy)."""
    return _SHAPE_FN[shape](px, py, cx, cy, size)


def shape_hit(shape: str, fill: str, px: float, py: float, cx: float, cy: float,
              size: float, thickness: float) -> bool:
    """True if point (px,py) is on the rendered pixels of the object.

    ``solid`` -> interior of the shape.  ``hollow`` -> a band of width
    ``thickness`` around the boundary (interior minus a shrunken core).
    """
    if fill == "solid":
        return shape_contains(shape, px, py, cx, cy, size)
    outer = shape_contains(shape, px, py, cx, cy, size)
    if not outer:
        return False
    inner = shape_contains(shape, px, py, cx, cy, max(size - thickness * INSET[shape], 1e-6))
    return not inner
