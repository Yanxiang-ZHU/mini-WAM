"""Measure the KEY flow-matching property: velocity-field constancy.

Rectified/linear flow matching defines x_t = (1-t)*eps + t*x1, so the ground-truth
velocity v* = x1 - eps is CONSTANT along the trajectory. If the *learned* velocity
field is also near-constant across the 4 denoising steps, then the model's OUTPUT
is highly temporally correlated even though its INPUT (x_t) is not. That is a
redundancy no existing DiT accelerator (Cambricon-D / DSTAR / DITTO / EXION, which
all exploit *input* temporal deltas) targets.

We capture the velocity v at each of the 4 steps and report:
  - cosine similarity between consecutive velocities,
  - relative L2 change between consecutive velocities.
If these are ~1.0 / ~0.0, the velocity is nearly reusable across steps.
"""

import torch

from training.utils import build_world_model

CKPT = "checkpoints/world_model_goal_wide_best.pt"


def main():
    device = "cuda"
    wm, _ = build_world_model(CKPT, device)
    wm.eval()

    # capture the velocity (head output) at each step
    vels = []
    def hook(mod, inp, out):
        vels.append(out.detach().clone())  # (B, n_patches, patch*patch)
    h = wm.head.register_forward_hook(hook)

    torch.manual_seed(0)
    B = 1
    history = torch.randn(B, 4, 48, 48, device=device)
    lang = torch.randint(0, 21, (B, 8), device=device)
    subtask = torch.randint(0, 21, (B, 8), device=device)
    meta = [{"speed": "normal", "quality": 1.0, "mistake": False, "control_mode": "keyboard"}] * B

    with torch.no_grad():
        wm.sample(history, lang, subtask, meta)  # 4 steps
    h.remove()

    print(f"captured {len(vels)} velocity vectors, shape {tuple(vels[0].shape)}")
    print("\nVelocity correlation across consecutive denoising steps:")
    print(f"{'step i->i+1':>12} {'cosine sim':>12} {'rel L2 diff':>12} {'|v_i|':>10} {'|v_i+1|':>10}")
    for i in range(len(vels) - 1):
        a, b = vels[i].flatten(), vels[i + 1].flatten()
        cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)
        rel = (b - a).norm() / (a.norm() + 1e-12)
        print(f"{i:>4} -> {i+1:>4} {cos.item():>12.4f} {rel.item():>12.4f} "
              f"{a.norm().item():>10.3f} {b.norm().item():>10.3f}")

    # also: how much does reusing step-0's velocity hurt the final image?
    print("\nAblation: reuse v_0 for all steps vs true per-step v (final-image MSE):")
    # replicate the loop manually with full x capture
    hist_tokens = wm.vision.encode_history(history)
    langv = wm.lang(lang); subv = wm.subtask_enc(subtask)
    n_steps = 4
    x_true = torch.randn(B, 1, 48, 48, device=device)
    x_reuse = x_true.clone()
    dt = 1.0 / n_steps
    with torch.no_grad():
        for i in range(n_steps):
            t = torch.full((B,), i / n_steps, device=device)
            img = wm.vision.encode_frame(x_true)
            c = wm.cond(langv, subv, meta, t)
            xt = torch.cat([hist_tokens, img], dim=1)
            for blk in wm.blocks:
                xt = blk(xt, c)
            v = wm.head(wm.norm(xt[:, -wm.n_patches:]))
            v_img = v.reshape(B, 1, 48, 48) if False else v.reshape(B, 1, wm.height // wm.patch * wm.patch, wm.width)
            # use the same unpatchify logic
            from models.world_model import unpatchify
            vf = unpatchify(v, wm.patch, wm.height, wm.width)
            if i == 0:
                v0 = vf.clone()
            x_true = x_true + vf * dt
        # reuse: apply v0 every step
        for i in range(n_steps):
            x_reuse = x_reuse + v0 * dt
    mse = ((x_true - x_reuse) ** 2).mean().item()
    print(f"  final-image MSE (reuse v0 vs true): {mse:.6f}")


if __name__ == "__main__":
    main()
