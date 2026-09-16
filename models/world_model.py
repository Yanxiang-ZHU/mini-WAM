"""World Model: a conditional flow-matching generator of visual subgoals.

    g_t ~ p_psi(g | H_t, L_t, S_t, M_t)

The model patches a noisy subgoal image into tokens, attends over them jointly
with the observation-history tokens, and is conditioned on language, subtask,
metadata and flow time via adaLN.  It predicts the flow velocity field, which is
unpatchified back to a 48x48 grayscale frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .vision_encoder import VisionEncoder
from .language_encoder import LanguageEncoder
from .diffusion_transformer import DiTBlock, ConditioningEncoder
from .flow_matching import interpolate, flow_matching_loss, euler_sample


def unpatchify(tokens: torch.Tensor, patch: int, height: int, width: int) -> torch.Tensor:
    """(B, n_patches, patch*patch) -> (B, 1, H, W)."""
    B = tokens.shape[0]
    nh, nw = height // patch, width // patch
    x = tokens.reshape(B, nh, nw, patch, patch)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    x = x.reshape(B, 1, height, width)
    return x


class WorldModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        dim = cfg.get("hidden_dim", 256)
        layers = cfg.get("layers", 6)
        heads = cfg.get("heads", 8)
        patch = cfg.get("patch", 8)
        history = cfg.get("history", 4)
        height = cfg.get("height", 48)
        width = cfg.get("width", 48)
        self.denoise_steps = cfg.get("denoise_steps", 4)
        # object-weighted loss: up-weight object/player pixels (in the target
        # frame) so the flow-matching loss is not dominated by the background.
        self.obj_weight = cfg.get("obj_weight", 1.0)

        self.vision = VisionEncoder(patch, dim, history, height, width)
        self.lang = LanguageEncoder(dim, cfg.get("lang_layers", 2), cfg.get("lang_heads", 4))
        self.subtask_enc = LanguageEncoder(dim, cfg.get("lang_layers", 2), cfg.get("lang_heads", 4))
        self.cond = ConditioningEncoder(dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch * patch)

        self.patch = patch
        self.height = height
        self.width = width
        self.n_patches = self.vision.patch_embed.n_patches

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, history: torch.Tensor,
                lang_ids: torch.Tensor, subtask_ids: torch.Tensor, meta_list: list) -> torch.Tensor:
        """Return the predicted velocity field for the noisy subgoal x_t."""
        B = x_t.shape[0]
        hist_tokens = self.vision.encode_history(history)       # (B, K*n, dim)
        img_tokens = self.vision.encode_frame(x_t)              # (B, n, dim)
        lang = self.lang(lang_ids)
        subtask = self.subtask_enc(subtask_ids)
        c = self.cond(lang, subtask, meta_list, t)              # (B, dim)

        x = torch.cat([hist_tokens, img_tokens], dim=1)
        for blk in self.blocks:
            x = blk(x, c)
        v = self.norm(x[:, -self.n_patches:])                   # (B, n, dim)
        v = self.head(v)                                        # (B, n, patch*patch)
        return unpatchify(v, self.patch, self.height, self.width)

    def loss(self, subgoal: torch.Tensor, history: torch.Tensor,
             lang_ids: torch.Tensor, subtask_ids: torch.Tensor, meta_list: list) -> torch.Tensor:
        B = subgoal.shape[0]
        t = torch.rand(B, device=subgoal.device)
        x_t, v_star = interpolate(subgoal, t)
        v = self.forward(x_t, t, history, lang_ids, subtask_ids, meta_list)
        if self.obj_weight > 1.0:
            # object/player pixels (value > 0 in [-1,1] target) get up-weighted
            obj_mask = (subgoal > 0.0).float()
            weight = 1.0 + (self.obj_weight - 1.0) * obj_mask
            return (weight * (v - v_star) ** 2).mean()
        return flow_matching_loss(v, v_star)

    @torch.no_grad()
    def sample(self, history: torch.Tensor, lang_ids: torch.Tensor,
               subtask_ids: torch.Tensor, meta_list: list,
               n_steps: int | None = None) -> torch.Tensor:
        n_steps = n_steps or self.denoise_steps
        B = history.shape[0]
        # fixed encodings (independent of the denoising step) — computed once
        hist_tokens = self.vision.encode_history(history)
        lang = self.lang(lang_ids)
        subtask = self.subtask_enc(subtask_ids)
        x = torch.randn(B, 1, self.height, self.width, device=history.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i / n_steps, device=history.device)
            img_tokens = self.vision.encode_frame(x)          # depends on x
            c = self.cond(lang, subtask, meta_list, t)        # depends on t
            xt = torch.cat([hist_tokens, img_tokens], dim=1)
            for blk in self.blocks:
                xt = blk(xt, c)
            v = self.norm(xt[:, -self.n_patches:])
            v = self.head(v)
            x = x + unpatchify(v, self.patch, self.height, self.width) * dt
        return x
