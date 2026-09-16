"""Deterministic expert controller for dataset generation.

Uses A* on a coarse grid when obstacles lie between the player and the target,
and direct steering otherwise.  Outputs a WASD action index per step.  The
expert has access to full state; the neural policy never does.

The controller is deliberately robust: start/goal cells are always traversable,
obstacles are inflated only by the player radius plus a small margin, paths are
4-connected (no corner cutting), and a stuck detector re-plans when the agent
presses against an obstacle.
"""

from __future__ import annotations

import heapq
import math
from collections import deque

from .objects import Rect
from .physics import ACTION_DIRS, NUM_ACTIONS

_CELL = 2.0
_PLAYER_R = 1.6
_MARGIN = 0.6
_INFLATE = _PLAYER_R + _MARGIN


def _segment_blocked(ax, ay, bx, by, obstacles, steps=48):
    for ob in obstacles:
        for i in range(steps + 1):
            t = i / steps
            px = ax + (bx - ax) * t
            py = ay + (by - ay) * t
            if ob.contains(px, py):
                return True
    return False


class Expert:
    def __init__(self, task, world_w: float = 48.0, world_h: float = 48.0):
        self.target = task.target
        self.obstacles = task.obstacles
        self.world_w = world_w
        self.world_h = world_h
        self.path: list[tuple[float, float]] = []
        self._wp_idx = 0
        self._pos_hist: deque = deque(maxlen=8)
        self._stuck = 0

    # -- planning --------------------------------------------------------- #
    def reset(self, player_xy):
        self._pos_hist.clear()
        self._stuck = 0
        self._plan(player_xy)

    def _plan(self, start):
        tx, ty = self.target.x, self.target.y
        if not self.obstacles or not _segment_blocked(start[0], start[1], tx, ty, self.obstacles):
            self.path = [(tx, ty)]
        else:
            self.path = self._astar(start, (tx, ty))
        self._wp_idx = 0

    def _blocked(self, cx: float, cy: float) -> bool:
        for ob in self.obstacles:
            if (cx > ob.cx - ob.hw - _INFLATE and cx < ob.cx + ob.hw + _INFLATE and
                    cy > ob.cy - ob.hh - _INFLATE and cy < ob.cy + ob.hh + _INFLATE):
                return True
        return False

    def _astar(self, start, goal):
        nw = max(1, int(self.world_w / _CELL))
        nh = max(1, int(self.world_h / _CELL))

        def cell_center(gx, gy):
            return (gx + 0.5) * _CELL, (gy + 0.5) * _CELL

        sx, sy = int(start[0] / _CELL), int(start[1] / _CELL)
        gx, gy = int(goal[0] / _CELL), int(goal[1] / _CELL)
        sx, sy = min(max(sx, 0), nw - 1), min(max(sy, 0), nh - 1)
        gx, gy = min(max(gx, 0), nw - 1), min(max(gy, 0), nh - 1)
        start_i = sy * nw + sx
        goal_i = gy * nw + gx

        def idx(x, y):
            return y * nw + x

        def blocked(gcx, gcy):
            if (gcx, gcy) == (sx, sy) or (gcx, gcy) == (gx, gy):
                return False
            ccx, ccy = cell_center(gcx, gcy)
            return self._blocked(ccx, ccy)

        open_set = [(0, start_i)]
        came = {}
        gscore = {start_i: 0.0}
        closed = set()

        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur in closed:
                continue
            if cur == goal_i:
                path = []
                while cur in came:
                    path.append(cell_center(cur % nw, cur // nw))
                    cur = came[cur]
                path.append(cell_center(sx, sy))
                path.reverse()
                return path
            closed.add(cur)
            cx, cy = cur % nw, cur // nw
            for dxx, dyy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dxx, cy + dyy
                if nx < 0 or ny < 0 or nx >= nw or ny >= nh:
                    continue
                if blocked(nx, ny):
                    continue
                ni = idx(nx, ny)
                ng = gscore[cur] + 1.0
                if ni not in gscore or ng < gscore[ni]:
                    gscore[ni] = ng
                    came[ni] = cur
                    heapq.heappush(open_set, (ng + math.hypot(nx - gx, ny - gy), ni))
        return []

    # -- control ---------------------------------------------------------- #
    def action(self, player_xy, player_vel) -> int:
        self._pos_hist.append(tuple(player_xy))

        # stuck detection: little net movement over the window while far from goal
        if len(self._pos_hist) == self._pos_hist.maxlen:
            p0 = self._pos_hist[0]
            p1 = self._pos_hist[-1]
            moved = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            d_target = math.hypot(player_xy[0] - self.target.x, player_xy[1] - self.target.y)
            if moved < 1.2 and d_target > 6.0:
                self._stuck += 1
            else:
                self._stuck = 0
            if self._stuck >= 2:
                self._plan(player_xy)
                self._stuck = 0

        if not self.path:
            # no path found: steer directly toward target as a fallback
            return self._steer(player_xy, (self.target.x, self.target.y))

        wx, wy = self.path[self._wp_idx]
        d = math.hypot(wx - player_xy[0], wy - player_xy[1])
        if d < 2.5 and self._wp_idx < len(self.path) - 1:
            self._wp_idx += 1
            wx, wy = self.path[self._wp_idx]
        return self._steer(player_xy, (wx, wy))

    def _steer(self, player_xy, waypoint) -> int:
        dx = waypoint[0] - player_xy[0]
        dy = waypoint[1] - player_xy[1]
        best, best_dot = 0, -1e9
        for a in range(NUM_ACTIONS):
            ax, ay = ACTION_DIRS[a]
            dot = ax * dx + ay * dy
            if dot > best_dot:
                best_dot, best = dot, a
        return best
