"""Action Expert: a conditional flow-matching generator of action chunks.

    A_{t:t+H} ~ p_theta(A | H_t, L_t, S_t, G_t, M_t)

Noisy action chunks (H x 4 continuous WASD vectors) are embedded into tokens and
denoised by a transformer that attends over history tokens, subgoal tokens, and
the action tokens, conditioned on language/subtask/metadata/time via adaLN.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .vision_encoder import VisionEncoder
from .language_encoder import LanguageEncoder
from .diffusion_transformer import DiTBlock, ConditioningEncoder
from .flow_matching import interpolate, flow_matching_loss


class ActionExpert(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        dim = cfg.get("hidden_dim", 256)
        layers = cfg.get("layers", 6)
        heads = cfg.get("heads", 8)
        patch = cfg.get("patch", 8)
        history = cfg.get("history", 4)
        height = cfg.get("height", 48)
        width = cfg.get("width", 48)
        self.action_horizon = cfg.get("action_horizon", 8)
        self.num_actions = cfg.get("num_actions", 4)
        self.denoise_steps = cfg.get("denoise_steps", 4)
        self.use_subgoal = cfg.get("use_subgoal", True)

        self.vision = VisionEncoder(patch, dim, history, height, width)
        self.lang = LanguageEncoder(dim, cfg.get("lang_layers", 2), cfg.get("lang_heads", 4))
        self.subtask_enc = LanguageEncoder(dim, cfg.get("lang_layers", 2), cfg.get("lang_heads", 4))
        self.cond = ConditioningEncoder(dim)

        self.action_embed = nn.Linear(self.num_actions, dim)
        self.action_pos = nn.Parameter(torch.zeros(1, self.action_horizon, dim))
        nn.init.trunc_normal_(self.action_pos, std=0.02)

        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.num_actions)

        self.n_patches = self.vision.patch_embed.n_patches

    def forward(self, a_t: torch.Tensor, t: torch.Tensor, history: torch.Tensor,
                subgoal: torch.Tensor | None, lang_ids: torch.Tensor, subtask_ids: torch.Tensor,
                meta_list: list) -> torch.Tensor:
        """Return the predicted velocity for the noisy action chunk a_t (B,H,4)."""
        B, H = a_t.shape[0], a_t.shape[1]
        hist_tokens = self.vision.encode_history(history)       # (B, K*n, dim)
        a_tokens = self.action_embed(a_t) + self.action_pos[:, :H]   # (B, H, dim)

        lang = self.lang(lang_ids)
        subtask = self.subtask_enc(subtask_ids)
        c = self.cond(lang, subtask, meta_list, t)

        if self.use_subgoal and subgoal is not None:
            subgoal_tokens = self.vision.encode_frame(subgoal)   # (B, n, dim)
            x = torch.cat([hist_tokens, subgoal_tokens, a_tokens], dim=1)
        else:
            x = torch.cat([hist_tokens, a_tokens], dim=1)
        for blk in self.blocks:
            x = blk(x, c)
        v = self.norm(x[:, -H:])
        return self.head(v)                                     # (B, H, 4)

    def loss(self, action_chunk: torch.Tensor, history: torch.Tensor,
             subgoal: torch.Tensor, lang_ids: torch.Tensor, subtask_ids: torch.Tensor,
             meta_list: list) -> torch.Tensor:
        B = action_chunk.shape[0]
        t = torch.rand(B, device=action_chunk.device)
        a_t, v_star = interpolate(action_chunk, t)
        v = self.forward(a_t, t, history, subgoal, lang_ids, subtask_ids, meta_list)
        return flow_matching_loss(v, v_star)

    @torch.no_grad()
    def sample(self, history: torch.Tensor, subgoal: torch.Tensor | None,
               lang_ids: torch.Tensor, subtask_ids: torch.Tensor, meta_list: list,
               n_steps: int | None = None) -> torch.Tensor:
        n_steps = n_steps or self.denoise_steps
        B = history.shape[0]
        H = self.action_horizon
        # fixed encodings (independent of the denoising step) — computed once
        hist_tokens = self.vision.encode_history(history)
        subgoal_tokens = (self.vision.encode_frame(subgoal)
                          if (self.use_subgoal and subgoal is not None) else None)
        lang = self.lang(lang_ids)
        subtask = self.subtask_enc(subtask_ids)
        a = torch.randn(B, H, self.num_actions, device=history.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i / n_steps, device=history.device)
            a_tokens = self.action_embed(a) + self.action_pos[:, :H]   # depends on a
            c = self.cond(lang, subtask, meta_list, t)                 # depends on t
            if subgoal_tokens is not None:
                xt = torch.cat([hist_tokens, subgoal_tokens, a_tokens], dim=1)
            else:
                xt = torch.cat([hist_tokens, a_tokens], dim=1)
            for blk in self.blocks:
                xt = blk(xt, c)
            v = self.head(self.norm(xt[:, -H:]))
            a = a + v * dt
        return a  # (B, H, 4) continuous; argmax -> discrete
