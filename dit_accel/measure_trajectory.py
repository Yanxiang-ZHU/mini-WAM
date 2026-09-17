"""Confirm the trajectory-decay hypothesis behind time-conditioned precision.

Few-step flow matching sets x_t = (1-t)*eps + t*x1, so the noise coefficient
(1-t) shrinks monotonically from 1 (step 0) to ~0.25 (step 3). The input to the
DiT therefore has a DECREASING dynamic range across denoising steps. If true,
precision can be allocated per step (high-bit early, low-bit late), which is a
per-TIMESTEP lever -- orthogonal to the per-DIFFERENTIAL quantization used by
Cambricon-D / DSTAR / DITTO / EXION.

We measure, at each of the 4 steps, the std and max|.| of:
  - the noisy subgoal input x_t (the changing img tokens at block-1 input),
  - the full block-1 activation,
and the quantization SNR of x_t at 8/6/4/2 bits.
"""

import torch

from training.utils import build_world_model
from models.world_model import unpatchify

CKPT = "checkpoints/world_model_goal_wide_best.pt"


def snr(x, bits):
    if bits >= 32:
        return float("inf")
    mn, mx = x.min(), x.max()
    s = (mx - mn) / (2 ** bits - 1)
    if s == 0:
        return float("inf")
    xq = ((x - mn) / s).round().clamp(0, 2 ** bits - 1) * s + mn
    return 10 * torch.log10(x.pow(2).mean() / ((x - xq).pow(2).mean() + 1e-12)).item()


def main():
    device = "cuda"
    wm, _ = build_world_model(CKPT, device)
    wm.eval()
    n_img = wm.n_patches  # 36

    torch.manual_seed(0)
    B = 1
    history = torch.randn(B, 4, 48, 48, device=device)
    lang = torch.randint(0, 21, (B, 8), device=device)
    subtask = torch.randint(0, 21, (B, 8), device=device)
    meta = [{"speed": "normal", "quality": 1.0, "mistake": False, "control_mode": "keyboard"}] * B

    hist_tokens = wm.vision.encode_history(history)
    langv, subv = wm.lang(lang), wm.subtask_enc(subtask)
    n_steps = 4
    x = torch.randn(B, 1, 48, 48, device=device)
    dt = 1.0 / n_steps

    print(f"{'step':>5} {'t':>5} | {'x_t std':>8} {'x_t max':>8} {'x_t SNR@8/6/4/2':>18} | "
          f"{'blk1 full std':>13} {'|v|':>8}")
    with torch.no_grad():
        for i in range(n_steps):
            t = torch.full((B,), i / n_steps, device=device)
            img = wm.vision.encode_frame(x)
            c = wm.cond(langv, subv, meta, t)
            xt = torch.cat([hist_tokens, img], dim=1)
            blk1_in = wm.blocks[0].norm1(xt)  # representative full activation
            for blk in wm.blocks:
                xt = blk(xt, c)
            v = wm.head(wm.norm(xt[:, -n_img:]))
            vf = unpatchify(v, wm.patch, wm.height, wm.width)
            s = "/".join(f"{snr(img, b):.0f}" for b in (8, 6, 4, 2))
            print(f"{i:>5} {i/n_steps:>5.2f} | {img.std().item():>8.3f} {img.abs().max().item():>8.3f} "
                  f"{s:>18} | {blk1_in.std().item():>13.3f} {vf.flatten().norm().item():>8.3f}")
            x = x + vf * dt


if __name__ == "__main__":
    main()
