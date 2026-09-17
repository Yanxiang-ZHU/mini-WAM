"""Verify the core empirical claim behind a token-class-aware precision proposal.

Hypothesis: in a WAM/VLA DiT the token sequence is HETEROGENEOUS --
a FIXED observation-context stream (144 tokens, deterministic, non-noisy)
and a CHANGING generation-target stream (36 tokens, noisy subgoal/action).
If the two streams have different dynamic ranges / quantization sensitivity,
precision can be assigned per token-class (low-bit context, high-bit target),
which is a STRUCTURAL (not temporal) redundancy absent from image/video-gen DiT.

We measure, WITHIN one forward pass (one denoising step), at each block:
  - std and max|.| of the context-token vs target-token activations,
  - the quantization SNR of each stream at 8 / 6 / 4 / 3 bits.
"""

import torch

from training.utils import build_world_model

CKPT = "checkpoints/world_model_goal_wide_best.pt"


def quant_snr(x: torch.Tensor, bits: int) -> float:
    """Signal-to-quantization-noise ratio (dB) for uniform int-bits quantization."""
    if bits >= 32:
        return float("inf")
    mn, mx = x.min(), x.max()
    scale = (mx - mn) / (2 ** bits - 1)
    if scale == 0:
        return float("inf")
    xq = ((x - mn) / scale).round().clamp(0, 2 ** bits - 1) * scale + mn
    err = (x - xq).pow(2).mean()
    sig = x.pow(2).mean()
    return 10 * torch.log10(sig / (err + 1e-12)).item()


def main():
    device = "cuda"
    wm, _ = build_world_model(CKPT, device)
    wm.eval()
    n_hist = 144
    n_img = wm.n_patches  # 36

    # capture each block's input, split into context (hist) and target (img) parts
    caps = [[] for _ in wm.blocks]
    handles = []
    for bi, blk in enumerate(wm.blocks):
        def hook(mod, inp, out, bi=bi):
            x = inp[0]
            caps[bi].append((x[:, :n_hist].detach(), x[:, n_hist:].detach()))
        handles.append(blk.register_forward_hook(hook))

    torch.manual_seed(0)
    B = 1
    history = torch.randn(B, 4, 48, 48, device=device)
    lang = torch.randint(0, 21, (B, 8), device=device)
    subtask = torch.randint(0, 21, (B, 8), device=device)
    meta = [{"speed": "normal", "quality": 1.0, "mistake": False, "control_mode": "keyboard"}] * B

    with torch.no_grad():
        # a single forward at t=0.25 (one denoising step)
        x_t = torch.randn(B, 1, 48, 48, device=device)
        t = torch.full((B,), 0.25, device=device)
        wm.forward(x_t, t, history, lang, subtask, meta)
    for h in handles:
        h.remove()

    print(f"{'blk':>4} | {'ctx std':>8} {'tgt std':>8} | {'ctx max':>8} {'tgt max':>8} | "
          f"{'ctx SNR@8/6/4':>13} | {'tgt SNR@8/6/4':>13}")
    for bi, cap in enumerate(caps):
        ctx, tgt = cap[0]
        cs, ts = ctx.std().item(), tgt.std().item()
        cm, tm = ctx.abs().max().item(), tgt.abs().max().item()
        c_snr = "/".join(f"{quant_snr(ctx, b):.0f}" for b in (8, 6, 4))
        t_snr = "/".join(f"{quant_snr(tgt, b):.0f}" for b in (8, 6, 4))
        print(f"{bi+1:>4} | {cs:>8.3f} {ts:>8.3f} | {cm:>8.3f} {tm:>8.3f} | "
              f"{c_snr:>13} | {t_snr:>13}")


if __name__ == "__main__":
    main()
