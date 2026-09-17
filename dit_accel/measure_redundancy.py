"""Quantify cross-step redundancy in the World Model's DiT sampling loop.

For each DiT block, capture the history-token portion of the block input
(first 144 tokens) across the 4 denoising steps, and report how much it drifts
between steps. This is the empirical ceiling for any cross-step caching scheme
(Delta-DiT / L2C / FasterCache): if the fixed-token features barely move, caching
them has room; if they move a lot, caching them is lossy.
"""

import torch
import torch.nn as nn

from training.utils import build_world_model

CKPT = "checkpoints/world_model_goal_wide_best.pt"


def main():
    device = "cuda"
    wm, _ = build_world_model(CKPT, device)
    wm.eval()
    n_hist = wm.vision.encode_history(torch.zeros(1, 4, 48, 48, device=device)).shape[1]
    n_img = wm.n_patches
    print(f"hist tokens = {n_hist}, img tokens = {n_img}, total = {n_hist + n_img}")

    # capture each block's input, hist portion
    caps = [[] for _ in wm.blocks]
    handles = []
    for bi, blk in enumerate(wm.blocks):
        def hook(mod, inp, out, bi=bi):
            caps[bi].append(inp[0][:, :n_hist].detach().clone())
        handles.append(blk.register_forward_hook(hook))

    torch.manual_seed(0)
    B = 1
    history = torch.randn(B, 4, 48, 48, device=device)
    lang = torch.randint(0, 21, (B, 8), device=device)
    subtask = torch.randint(0, 21, (B, 8), device=device)
    meta = [{"speed": "normal", "quality": 1.0, "mistake": False, "control_mode": "keyboard"}] * B

    with torch.no_grad():
        wm.sample(history, lang, subtask, meta)  # 4 denoising steps
    for h in handles:
        h.remove()

    print("\nHist-token activation drift across the 4 denoising steps (normalized):")
    print(f"{'block':>6} {'mean|delta|/|x|':>18} {'max|delta|/|x|':>18}")
    for bi, cap in enumerate(caps):
        assert len(cap) == 4, f"block {bi}: {len(cap)} caps"
        rels = []
        for i in range(3):
            d = (cap[i + 1] - cap[i]).abs()
            rels.append((d / cap[i].abs().clamp_min(1e-6)).mean().item())
        print(f"{bi+1:>6} {sum(rels)/3:>18.4f} {max(rels):>18.4f}")

    # also: how much does the *conditioning* vector c change across steps?
    print("\nConditioning vector c drift across steps (this gates the adaLN modulation):")
    cs = []
    with torch.no_grad():
        for i in range(4):
            t = torch.full((B,), i / 4, device=device)
            cs.append(wm.cond(wm.lang(lang), wm.subtask_enc(subtask), meta, t))
    for i in range(3):
        d = (cs[i + 1] - cs[i]).abs() / cs[i].abs().clamp_min(1e-6)
        print(f"  step {i}->{i+1}: mean rel diff = {d.mean().item():.4f}")


if __name__ == "__main__":
    main()
