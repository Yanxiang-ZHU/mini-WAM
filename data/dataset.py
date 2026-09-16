"""Episode dataset and training-sample extraction.

Loads ``.npz`` episodes produced by the generator, builds a flat index of
(episode, timestep, subgoal-delta) triples, and yields samples of the form:

    history (K,48,48)  ·  instruction  ·  subtask  ·  metadata
    ·  subgoal (48,48)  ·  subgoal_delta  ·  action chunk (H,4 one-hot)

Frames are stored as uint8 [0,255] and normalised to [-1,1] (or [0,1]) on load.
The visual subgoal is always the *real* future frame here; generated subgoals
are mixed in by the training loop when required (Phase 6).
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_frames(u8: np.ndarray, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    x = u8.astype(np.float32) / 255.0
    if lo == 0.0 and hi == 1.0:
        return x
    return x * (hi - lo) + lo


def extract_sample(frames_u8, actions_u8, t, delta, K, H, lo=-1.0, hi=1.0, goal=False):
    """Extract a single sample at timestep ``t`` from a full episode.

    ``goal=True`` uses the terminal frame (player at/near the target) as the
    visual subgoal — a goal-state conditioning à la π0 image goals — instead of a
    short-horizon future frame (``frame[t+delta]``).
    """
    T = frames_u8.shape[0]
    history = normalize_frames(frames_u8[t - K + 1:t + 1], lo, hi)          # (K,48,48)
    if goal:
        sg_idx = T - 1                       # terminal frame = goal state
        subgoal_delta = T - 1 - t            # steps remaining to the goal
    else:
        sg_idx = t + delta
        subgoal_delta = delta
    subgoal = normalize_frames(frames_u8[sg_idx:sg_idx + 1], lo, hi)        # (1,48,48)
    chunk = actions_u8[t:t + H]                                             # (H,)
    # one-hot action chunk (H, 4)
    onehot = np.zeros((H, 4), dtype=np.float32)
    onehot[np.arange(H), chunk] = 1.0
    return {
        "history": history,           # (K,48,48) float
        "subgoal": subgoal[0],        # (48,48) float
        "subgoal_delta": subgoal_delta,
        "action_chunk": chunk,        # (H,) int
        "action_onehot": onehot,      # (H,4) float
    }


class EpisodeDataset(Dataset):
    def __init__(self, data_dir: str, *, K: int = 4, H: int = 8,
                 deltas: tuple = (8, 16, 24, 32), lo: float = -1.0, hi: float = 1.0,
                 max_episodes: int | None = None, preload: bool = True,
                 goal: bool = False):
        self.data_dir = data_dir
        self.K, self.H = K, H
        self.deltas = deltas
        self.lo, self.hi = lo, hi
        self.goal = goal

        self.paths = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
        if max_episodes is not None:
            self.paths = self.paths[:max_episodes]

        self._cache: dict[int, dict] = {}
        self._preloaded: list[dict] | None = None
        if preload:
            self._preload_all()
        self._build_index()

    def _preload_all(self):
        self._preloaded = []
        for p in self.paths:
            z = np.load(p)
            self._preloaded.append({
                "frames": z["frames"],
                "actions": z["actions"],
                "meta": json.loads(str(z["meta_json"])),
            })

    def _load(self, ep_idx: int) -> dict:
        if self._preloaded is not None:
            return self._preloaded[ep_idx]
        if ep_idx in self._cache:
            return self._cache[ep_idx]
        z = np.load(self.paths[ep_idx])
        frames = z["frames"]            # (T,48,48) uint8
        actions = z["actions"]          # (T-1,) int8
        meta = json.loads(str(z["meta_json"]))
        if len(self._cache) > 64:       # simple LRU-ish eviction
            self._cache.pop(next(iter(self._cache)))
        self._cache[ep_idx] = {"frames": frames, "actions": actions, "meta": meta}
        return self._cache[ep_idx]

    def _build_index(self):
        self.index = []  # list of (ep_idx, t, delta)
        for i in range(len(self.paths)):
            d = self._load(i)
            T = d["frames"].shape[0]
            H = self.H
            if self.goal:
                # goal mode: one sample per timestep (subgoal = terminal frame)
                for t in range(self.K - 1, T - H):
                    self.index.append((i, t, 0))
            else:
                for t in range(self.K - 1, T - H):      # need K history + H actions
                    for delta in self.deltas:
                        if t + delta < T:
                            self.index.append((i, t, delta))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_idx, t, delta = self.index[idx]
        d = self._load(ep_idx)
        sample = extract_sample(d["frames"], d["actions"], t, delta,
                                self.K, self.H, self.lo, self.hi, goal=self.goal)
        sample["episode"] = ep_idx
        sample["t"] = t
        sample["instruction"] = d["meta"]["instruction"]
        sample["subtask"] = d["meta"]["subtask"]
        sample["metadata"] = d["meta"]["metadata"]
        sample["target_shape"] = d["meta"]["target"]["shape"]
        sample["target_fill"] = d["meta"]["target"]["fill"]
        return sample

    def episode(self, idx: int) -> dict:
        return self._load(idx)


def collate(batch: list[dict]) -> dict:
    """Collate a list of samples into a tensor batch."""
    out = {}
    out["history"] = torch.stack([torch.as_tensor(s["history"]) for s in batch])
    out["subgoal"] = torch.stack([torch.as_tensor(s["subgoal"]) for s in batch])
    out["subgoal_delta"] = torch.as_tensor([s["subgoal_delta"] for s in batch])
    out["action_onehot"] = torch.stack([torch.as_tensor(s["action_onehot"]) for s in batch])
    out["action_chunk"] = torch.stack([torch.as_tensor(s["action_chunk"]) for s in batch]).long()
    out["instruction"] = [s["instruction"] for s in batch]
    out["subtask"] = [s["subtask"] for s in batch]
    out["metadata"] = [s["metadata"] for s in batch]
    out["target_shape"] = [s["target_shape"] for s in batch]
    out["target_fill"] = [s["target_fill"] for s in batch]
    return out
