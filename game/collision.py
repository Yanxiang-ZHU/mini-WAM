"""Collision resolution: player (a disc) vs. axis-aligned rectangles and the
world boundary.  Obstacles are impassable and resolve by pushing the player out
along the minimum-penetration axis."""

from __future__ import annotations

from .objects import Rect


def resolve_rect(px: float, py: float, r: float, rect: Rect) -> tuple[float, float, bool]:
    """Push the disc at (px,py) with radius r out of ``rect``.

    Returns (new_x, new_y, collided).
    """
    # closest point on the rectangle to the disc centre
    cx = max(rect.cx - rect.hw, min(px, rect.cx + rect.hw))
    cy = max(rect.cy - rect.hh, min(py, rect.cy + rect.hh))
    dx = px - cx
    dy = py - cy
    d2 = dx * dx + dy * dy
    if d2 >= r * r:
        return px, py, False
    if d2 < 1e-12:
        # centre inside the rectangle: push out along the shallowest axis
        left = px - (rect.cx - rect.hw)
        right = (rect.cx + rect.hw) - px
        top = py - (rect.cy - rect.hh)
        bottom = (rect.cy + rect.hh) - py
        m = min(left, right, top, bottom)
        if m == left:
            px = rect.cx - rect.hw - r
        elif m == right:
            px = rect.cx + rect.hw + r
        elif m == top:
            py = rect.cy - rect.hh - r
        else:
            py = rect.cy + rect.hh + r
        return px, py, True
    d = (d2 ** 0.5)
    nx, ny = dx / d, dy / d
    overlap = r - d
    return px + nx * overlap, py + ny * overlap, True


def resolve_boundary(px: float, py: float, r: float, w: float, h: float) -> tuple[float, float, bool]:
    """Clamp the disc inside [r, w-r] x [r, h-r]."""
    nx = min(max(px, r), w - r)
    ny = min(max(py, r), h - r)
    hit = (nx != px) or (ny != py)
    return nx, ny, hit


def resolve_all(px: float, py: float, r: float, obstacles: list[Rect],
                w: float, h: float) -> tuple[float, float, bool]:
    """Resolve boundary + all obstacles.  Returns (x, y, any_collision)."""
    px, py, bhit = resolve_boundary(px, py, r, w, h)
    any_hit = bhit
    # a few iterations for stacked corners
    for _ in range(4):
        hit_any = False
        for ob in obstacles:
            px, py, hit = resolve_rect(px, py, r, ob)
            hit_any = hit_any or hit
        px, py, bhit = resolve_boundary(px, py, r, w, h)
        hit_any = hit_any or bhit
        any_hit = any_hit or hit_any
        if not hit_any:
            break
    return px, py, any_hit
