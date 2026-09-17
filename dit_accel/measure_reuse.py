"""Quantify the velocity-convergence step-reuse opportunity (training-free).

Because the flow-matching velocity converges in direction (cos 0.855->0.982) and
decays in magnitude (43->17) across the 4 steps, the late steps are near-redundant.
We measure the final-image error when reusing earlier velocities for later steps
(which skips those DiT forward passes entirely -- no retraining), and the effect
of quantizing each step's velocity to fewer bits (velocity-magnitude-proportional
precision: late steps have small |v| so a fixed absolute error needs fewer bits).
"""

import torch

from training.utils import build_world_model
from models.world_model import unpatchify

CKPT = "checkpoints/world_model_goal_wide_best.pt"


def run_sample(wm, history, lang, sub, meta, x0, reuse_from=None, v_bits=None):
    """Run 4-step sample from fixed initial noise x0; optionally reuse velocity
    from `reuse_from[i]` (or None) for step i, and quantize each velocity."""
    hist_tokens = wm.vision.encode_history(history)
    langv, subv = wm.lang(lang), wm.subtask_enc(sub)
    n = 4
    x = x0.clone()
    dt = 1.0 / n
    prev_v = None
    vels = []
    with torch.no_grad():
        for i in range(n):
            if reuse_from is not None and i in reuse_from and reuse_from[i] is not None:
                vf = vels[reuse_from[i]].clone()
            else:
                t = torch.full((1,), i / n, device=history.device)
                img = wm.vision.encode_frame(x)
                c = wm.cond(langv, subv, meta, t)
                xt = torch.cat([hist_tokens, img], dim=1)
                for blk in wm.blocks:
                    xt = blk(xt, c)
                v = wm.head(wm.norm(xt[:, -wm.n_patches:]))
                vf = unpatchify(v, wm.patch, wm.height, wm.width)
            if v_bits is not None and v_bits < 32:
                mn, mx = vf.min(), vf.max()
                s = (mx - mn) / (2 ** v_bits - 1)
                vf = ((vf - mn) / s).round().clamp(0, 2 ** v_bits - 1) * s + mn
            vels.append(vf)
            x = x + vf * dt
    return x, vels


def main():
    device = "cuda"
    wm, _ = build_world_model(CKPT, device)
    wm.eval()
    torch.manual_seed(0)
    history = torch.randn(1, 4, 48, 48, device=device)
    lang = torch.randint(0, 21, (1, 8), device=device)
    sub = torch.randint(0, 21, (1, 8), device=device)
    meta = [{"speed": "normal", "quality": 1.0, "mistake": False, "control_mode": "keyboard"}]

    x0 = torch.randn(1, 1, 48, 48, device=device)
    x_true, _ = run_sample(wm, history, lang, sub, meta, x0)
    print("final-image MSE (normalized to [-1,1], lower=better):\n")

    # step reuse: skip later DiT passes by reusing earlier velocities
    for label, rf in [
        ("reuse v2 for step3 (3 DiT passes)", {3: 2}),
        ("reuse v1 for steps2,3 (2 DiT passes)", {2: 1, 3: 1}),
        ("reuse v0 for all (1 DiT pass)", {1: 0, 2: 0, 3: 0}),
    ]:
        x, _ = run_sample(wm, history, lang, sub, meta, x0, reuse_from=rf)
        mse = ((x_true - x) ** 2).mean().item()
        print(f"  {label:<40} MSE = {mse:.5f}")

    # velocity-magnitude-proportional precision: quantize the (small) late-step
    # velocities more aggressively than the (large) early-step velocities
    print("\nvelocity quantization (bits per step), final MSE:")
    for bits in [8, 6, 4, 3]:
        x, vels = run_sample(wm, history, lang, sub, meta, x0, v_bits=bits)
        mse = ((x_true - x) ** 2).mean().item()
        print(f"  all steps @ {bits}-bit velocity: MSE = {mse:.5f}")


if __name__ == "__main__":
    main()
