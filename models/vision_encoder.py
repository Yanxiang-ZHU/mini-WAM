"""Vision encoder: patch-embeds 48x48x1 frames into tokens.

A single 48x48 grayscale frame becomes (48/P)^2 tokens of dimension ``dim``.
History is encoded frame-by-frame with a shared patch projector and given
learnable spatial + temporal positional embeddings.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, patch: int = 8, in_ch: int = 1, dim: int = 256,
                 height: int = 48, width: int = 48):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.nh, self.nw = height // patch, width // patch
        self.n_patches = self.nh * self.nw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, H, W) -> (B, n_patches, dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionEncoder(nn.Module):
    """Encodes a single frame or a K-frame history into token sequences."""

    def __init__(self, patch: int = 8, dim: int = 256, history: int = 4,
                 height: int = 48, width: int = 48):
        super().__init__()
        self.history = history
        self.patch_embed = PatchEmbed(patch, 1, dim, height, width)
        n = self.patch_embed.n_patches
        self.spatial_pos = nn.Parameter(torch.zeros(1, n, dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, history, 1, dim))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def encode_frame(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W) -> (B, n, dim)
        return self.patch_embed(x) + self.spatial_pos

    def encode_history(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, K, H, W) -> (B, K*n, dim)
        B, K = h.shape[0], h.shape[1]
        h = h.reshape(B * K, 1, h.shape[2], h.shape[3])
        toks = self.encode_frame(h)                       # (B*K, n, dim)
        toks = toks.reshape(B, K, toks.shape[1], toks.shape[2])
        toks = toks + self.temporal_pos                    # (B, K, n, dim)
        return toks.reshape(B, K * toks.shape[2], toks.shape[3])
