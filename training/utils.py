# Shared training utilities: config loading, tokenization, loaders, checkpointing.

from __future__ import annotations

import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import EpisodeDataset, collate
from models.language_encoder import tokenize


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_loader(data_dir: str, batch_size: int, *, K: int = 4, H: int = 8,
                deltas=(8, 16, 24, 32), workers: int = 4, shuffle: bool = True,
                max_episodes: int | None = None, preload: bool = True) -> DataLoader:
    ds = EpisodeDataset(data_dir, K=K, H=H, deltas=tuple(deltas),
                        max_episodes=max_episodes, preload=preload)
    # preloaded datasets live in the parent process; avoid pickling them to
    # workers by using a single process (in-memory access is fast enough).
    n_workers = 0 if preload else workers
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=n_workers, collate_fn=collate, drop_last=True,
                      persistent_workers=False)


def tokenize_batch(strings: list[str], device: str) -> torch.Tensor:
    return torch.tensor([tokenize(s) for s in strings], device=device)


def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int,
                    cfg: dict, seed: int, extra: dict | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "config": cfg,
        "seed": seed,
        "extra": extra or {},
    }, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt["optimizer"]:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt["scheduler"]:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.n = 0.0, 0

    def update(self, v, n=1):
        self.val += v * n
        self.n += n

    @property
    def avg(self):
        return self.val / max(1, self.n)


# -- model reconstruction from checkpoints -------------------------------- #

def build_world_model(ckpt_path: str, device: str = "cuda"):
    from models.world_model import WorldModel
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    mc = dict(cfg["model"])
    mc["history"] = cfg["history"]["frames"]
    model = WorldModel(mc).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def build_action_expert(ckpt_path: str, device: str = "cuda"):
    from models.action_expert import ActionExpert
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    mc = dict(cfg["model"])
    mc["history"] = cfg["history"]["frames"]
    mc["action_horizon"] = cfg["action"]["horizon"]
    mc["num_actions"] = cfg["action"]["num_actions"]
    model = ActionExpert(mc).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def build_simple_policy(ckpt_path: str, device: str = "cuda"):
    from models.simple_policy import SimplePolicy
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    mc = dict(cfg["model"])
    mc["history"] = cfg["history"]["frames"]
    model = SimplePolicy(mc).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg
