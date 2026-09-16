"""Baseline 1: a simple observation->action classifier (no world model, no
flow matching).  Encodes observation history + language, then predicts a single
WASD action with a cross-entropy loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .vision_encoder import VisionEncoder
from .language_encoder import LanguageEncoder
from .diffusion_transformer import DiTBlock


class SimplePolicy(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        dim = cfg.get("hidden_dim", 256)
        layers = cfg.get("layers", 4)
        heads = cfg.get("heads", 8)
        patch = cfg.get("patch", 8)
        history = cfg.get("history", 4)
        self.num_actions = cfg.get("num_actions", 4)

        self.vision = VisionEncoder(patch, dim, history)
        self.lang = LanguageEncoder(dim, cfg.get("lang_layers", 2), cfg.get("lang_heads", 4))
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(layers)])
        # a zero conditioning vector keeps DiTBlock in identity mode initially
        self.head = nn.Linear(dim, self.num_actions)

    def forward(self, history: torch.Tensor, lang_ids: torch.Tensor) -> torch.Tensor:
        B = history.shape[0]
        hist = self.vision.encode_history(history)          # (B, K*n, dim)
        lang = self.lang(lang_ids).unsqueeze(1)             # (B, 1, dim)
        x = torch.cat([lang, hist], dim=1)                  # (B, 1+K*n, dim)
        c = torch.zeros(B, hist.shape[-1], device=history.device)  # identity adaLN
        for blk in self.blocks:
            x = blk(x, c)
        return self.head(x[:, 0])                            # (B, 4) logits

    def loss(self, history: torch.Tensor, action: torch.Tensor,
             lang_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(history, lang_ids)
        return nn.functional.cross_entropy(logits, action)
