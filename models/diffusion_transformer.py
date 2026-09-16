"""Diffusion Transformer (DiT) building blocks and conditioning helpers.

Provides sinusoidal timestep embeddings, a metadata encoder, an adaLN-zero DiT
block, and a conditioning combiner shared by the world model and action expert.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.mlp[0].in_features))


class MetadataEncoder(nn.Module):
    """Encodes the small fixed metadata dict into a vector.

    ``speed`` and ``control_mode`` are categorical; ``quality`` is a scalar;
    ``mistake`` is binary.  For the first experiments these are constant.
    """

    SPEEDS = ["normal", "slow", "fast"]
    MODES = ["keyboard", "ai"]

    def __init__(self, dim: int = 256):
        super().__init__()
        self.speed_emb = nn.Embedding(len(self.SPEEDS), dim // 4)
        self.mode_emb = nn.Embedding(len(self.MODES), dim // 4)
        self.quality_proj = nn.Linear(1, dim // 4)
        self.mistake_proj = nn.Linear(1, dim // 4)

    def forward(self, meta_list: list) -> torch.Tensor:
        # meta_list: list of per-sample metadata dicts
        dev = self.speed_emb.weight.device
        speed = torch.tensor([self.SPEEDS.index(m["speed"]) for m in meta_list],
                             device=dev)
        mode = torch.tensor([self.MODES.index(m["control_mode"]) for m in meta_list],
                            device=dev)
        quality = torch.tensor([[m["quality"]] for m in meta_list],
                               dtype=torch.float32, device=dev)
        mistake = torch.tensor([[1.0 if m["mistake"] else 0.0] for m in meta_list],
                               dtype=torch.float32, device=dev)
        parts = [self.speed_emb(speed), self.mode_emb(mode),
                 self.quality_proj(quality), self.mistake_proj(mistake)]
        return torch.cat(parts, dim=-1)


class DiTBlock(nn.Module):
    """adaLN-zero diffusion transformer block."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        # zero-init the final modulation so the block starts as identity
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s1, g1, s2, g2, s3, g3 = self.adaLN(c).chunk(6, dim=-1)
        # attention
        h = self.norm1(x) * (1 + s1.unsqueeze(1)) + g1.unsqueeze(1)
        x = x + self.attn(h, h, h)[0] * g2.unsqueeze(1)
        # mlp
        h = self.norm2(x) * (1 + s2.unsqueeze(1)) + g3.unsqueeze(1)
        x = x + self.mlp(h)
        return x


class ConditioningEncoder(nn.Module):
    """Combines language, subtask, metadata, and time into one condition vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.time_mlp = TimestepMLP(dim)
        self.meta_enc = MetadataEncoder(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim * 4, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, lang: torch.Tensor, subtask: torch.Tensor,
                meta: dict, t: torch.Tensor) -> torch.Tensor:
        m = self.meta_enc(meta)
        tm = self.time_mlp(t)
        c = torch.cat([lang, subtask, m, tm], dim=-1)
        return self.proj(c)
