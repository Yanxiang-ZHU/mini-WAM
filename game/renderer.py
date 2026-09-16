"""Grayscale renderer producing 48x48x1 observations.

Rendering is supersampled (default 4x per axis) for smooth shape edges, then
area-averaged down to the target resolution.  Brightness levels are the only
visual channel; no colour is used.

    background : 0.08
    obstacles  : 0.43
    objects    : 0.75   (target + distractors, solid or hollow)
    player     : 1.00   (a small cross marker, distinct from all 6 categories)
"""

from __future__ import annotations

import numpy as np

from .objects import Object, Rect, shape_hit, AREA_SCALE

BG = 0.08
OBSTACLE = 0.35
OBJECT = 0.75
PLAYER = 1.00

HOLLOW_THICKNESS = 1.5  # world px; thick enough to reach full brightness after AA

PLAYER_ARM = 2.0   # half-length of the cross arms (px)
PLAYER_THICK = 0.7  # half-thickness of the cross arms (px)


def _cross_hit(px: float, py: float, cx: float, cy: float) -> bool:
    dx, dy = abs(px - cx), abs(py - cy)
    horiz = dx <= PLAYER_ARM and dy <= PLAYER_THICK
    vert = dy <= PLAYER_ARM and dx <= PLAYER_THICK
    return horiz or vert


def render_frame(player_xy, objects: list[Object], obstacles: list[Rect],
                 width: int = 48, height: int = 48, ss: int = 4) -> np.ndarray:
    """Render the scene to a float32 array of shape (height, width) in [0, 1]."""
    frame = np.full((height * ss, width * ss), BG, dtype=np.float32)
    for ob in obstacles:
        x0 = int(round((ob.cx - ob.hw) * ss))
        x1 = int(round((ob.cx + ob.hw) * ss))
        y0 = int(round((ob.cy - ob.hh) * ss))
        y1 = int(round((ob.cy + ob.hh) * ss))
        x0, x1 = max(0, x0), min(width * ss, x1)
        y0, y1 = max(0, y0), min(height * ss, y1)
        if x1 > x0 and y1 > y0:
            frame[y0:y1, x0:x1] = OBSTACLE

    for obj in objects:
        _draw_object(frame, obj, ss)

    _draw_player(frame, player_xy, ss)

    # area-average down to target resolution
    out = frame.reshape(height, ss, width, ss).mean(axis=(1, 3))
    return out.astype(np.float32)


def _draw_object(frame: np.ndarray, obj: Object, ss: int) -> None:
    cx, cy = obj.x * ss, obj.y * ss
    world_size = obj.size * AREA_SCALE[obj.shape]
    size = world_size * ss
    # bounding box of the shape
    half = int(size) + 2
    x0 = max(0, int(cx) - half)
    x1 = min(frame.shape[1], int(cx) + half + 1)
    y0 = max(0, int(cy) - half)
    y1 = min(frame.shape[0], int(cy) + half + 1)
    for y in range(y0, y1):
        py = (y + 0.5) / ss
        for x in range(x0, x1):
            px = (x + 0.5) / ss
            if shape_hit(obj.shape, obj.fill, px, py, obj.x, obj.y, world_size, HOLLOW_THICKNESS):
                frame[y, x] = max(frame[y, x], OBJECT)


def _draw_player(frame: np.ndarray, player_xy, ss: int) -> None:
    cx, cy = player_xy[0] * ss, player_xy[1] * ss
    half = int(PLAYER_ARM * ss) + 2
    x0 = max(0, int(cx) - half)
    x1 = min(frame.shape[1], int(cx) + half + 1)
    y0 = max(0, int(cy) - half)
    y1 = min(frame.shape[0], int(cy) + half + 1)
    for y in range(y0, y1):
        py = (y + 0.5) / ss
        for x in range(x0, x1):
            px = (x + 0.5) / ss
            if _cross_hit(px, py, player_xy[0], player_xy[1]):
                frame[y, x] = PLAYER


def to_uint8(frame: np.ndarray) -> np.ndarray:
    return (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
