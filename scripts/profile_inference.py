"""Inference-time profiling: decompose one cascade step into sub-components.

Measures:
  - top-level: language encoder, vision encoder, World Model sample, Action
    Expert sample, cascade orchestration overhead.
  - within WM and AE: vision (history/frame), language, conditioning, DiT blocks,
    output head — in absolute ms and as a fraction of the forward pass.
"""

import time

import torch

from training.utils import build_world_model, build_action_expert, tokenize_batch

DEVICE = "cuda"
B = 1


def timeit(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def main():
    wm, _ = build_world_model("checkpoints/world_model_goal_wide_best.pt", DEVICE)
    ae, _ = build_action_expert("checkpoints/action_expert_goal_wide_best.pt", DEVICE)
    wm.eval(); ae.eval()

    history = torch.randn(B, 4, 48, 48, device=DEVICE)
    subgoal = torch.randn(B, 1, 48, 48, device=DEVICE)
    lang_ids = tokenize_batch(["go to the solid circle"], DEVICE)
    subtask_ids = tokenize_batch(["move toward the solid circle"], DEVICE)
    meta = [{"speed": "normal", "quality": 1.0, "mistake": False, "control_mode": "keyboard"}]
    x_t = torch.randn(B, 1, 48, 48, device=DEVICE)
    t = torch.rand(B, device=DEVICE)

    # precompute fixed intermediates (reused across denoising steps)
    with torch.no_grad():
        hist_tokens = wm.vision.encode_history(history)
        img_tokens = wm.vision.encode_frame(x_t)
        lang = wm.lang(lang_ids)
        subtask = wm.subtask_enc(subtask_ids)
        c = wm.cond(lang, subtask, meta, t)

    print("=" * 72)
    print("TOP-LEVEL (per call)")
    print("=" * 72)

    t_lang = timeit(lambda: (wm.lang(lang_ids), wm.subtask_enc(subtask_ids)))
    t_vision_hist = timeit(lambda: wm.vision.encode_history(history))
    t_wm_sample = timeit(lambda: wm.sample(history, lang_ids, subtask_ids, meta), iters=50)
    t_ae_sample = timeit(lambda: ae.sample(history, subgoal, lang_ids, subtask_ids, meta), iters=50)

    # cascade orchestration: a full step WITHOUT the AE inference (simulate)
    from models.cascade_wam import CascadeWAM
    wam = CascadeWAM(wm, ae, device=DEVICE, subgoal_update_every=5, async_mode=False)
    wam.set_instruction("go to the solid circle")
    wam.observe(history[0, -1].cpu().numpy())
    # measure observe + step orchestration by patching AE.sample to a no-op
    orig = ae.sample
    ae.sample = lambda *a, **k: torch.zeros(B, 8, 4, device=DEVICE)
    t_cascade_overhead = timeit(lambda: wam.step(), iters=200)
    ae.sample = orig
    wam.shutdown()

    print(f"  language encoder (instr + subtask): {t_lang:.3f} ms")
    print(f"  vision encoder (4-frame history)  : {t_vision_hist:.3f} ms")
    print(f"  World Model sample (4 denoise)    : {t_wm_sample:.3f} ms")
    print(f"  Action Expert sample (4 denoise)  : {t_ae_sample:.3f} ms")
    print(f"  cascade orchestration (no infer)  : {t_cascade_overhead:.4f} ms")
    print()

    # --- WM internals (per forward pass) ---
    print("=" * 72)
    print("WORLD MODEL internals (per single forward pass)")
    print("=" * 72)

    def wm_dit():
        x = torch.cat([hist_tokens, img_tokens], dim=1)
        for blk in wm.blocks:
            x = blk(x, c)
        return x

    def wm_head():
        x = torch.cat([hist_tokens, img_tokens], dim=1)
        for blk in wm.blocks:
            x = blk(x, c)
        v = wm.norm(x[:, -wm.n_patches:])
        return wm.head(v)

    t_vis_hist = timeit(lambda: wm.vision.encode_history(history))
    t_vis_frame = timeit(lambda: wm.vision.encode_frame(x_t))
    t_lang_wm = timeit(lambda: (wm.lang(lang_ids), wm.subtask_enc(subtask_ids)))
    t_cond = timeit(lambda: wm.cond(lang, subtask, meta, t))
    t_dit = timeit(wm_dit)
    t_head = timeit(wm_head)

    dit_only = t_dit
    head_only = t_head - t_dit  # head = (dit + head) - dit
    total_fwd = t_vis_hist + t_vis_frame + t_lang_wm + t_cond + t_dit + max(head_only, 0)

    print(f"  vision (history 4 frames) : {t_vis_hist:.3f} ms ({t_vis_hist/total_fwd*100:.1f}%)")
    print(f"  vision (noisy subgoal)    : {t_vis_frame:.3f} ms ({t_vis_frame/total_fwd*100:.1f}%)")
    print(f"  language (instr+subtask)  : {t_lang_wm:.3f} ms ({t_lang_wm/total_fwd*100:.1f}%)")
    print(f"  conditioning (adaLN)      : {t_cond:.3f} ms ({t_cond/total_fwd*100:.1f}%)")
    print(f"  DiT blocks (6 layers)     : {t_dit:.3f} ms ({t_dit/total_fwd*100:.1f}%)")
    print(f"  head + unpatchify         : {max(head_only,0):.3f} ms ({max(head_only,0)/total_fwd*100:.1f}%)")
    print(f"  TOTAL forward             : {total_fwd:.3f} ms")
    print(f"  -> x4 denoise steps = WM sample ~{total_fwd*4:.3f} ms (measured {t_wm_sample:.3f} ms)")
    print()

    # --- AE internals (per forward pass) ---
    print("=" * 72)
    print("ACTION EXPERT internals (per single forward pass)")
    print("=" * 72)

    with torch.no_grad():
        ae_hist = ae.vision.encode_history(history)
        ae_sub = ae.vision.encode_frame(subgoal)
        ae_lang = ae.lang(lang_ids)
        ae_subtask = ae.subtask_enc(subtask_ids)
        ae_c = ae.cond(ae_lang, ae_subtask, meta, t)
        a_t = torch.randn(B, 8, 4, device=DEVICE)
        a_tokens = ae.action_embed(a_t) + ae.action_pos

    def ae_dit():
        x = torch.cat([ae_hist, ae_sub, a_tokens], dim=1)
        for blk in ae.blocks:
            x = blk(x, c)
        return x

    def ae_head():
        x = torch.cat([ae_hist, ae_sub, a_tokens], dim=1)
        for blk in ae.blocks:
            x = blk(x, c)
        return ae.head(ae.norm(x[:, -8:]))

    t_vis_hist_ae = timeit(lambda: ae.vision.encode_history(history))
    t_vis_sub_ae = timeit(lambda: ae.vision.encode_frame(subgoal))
    t_lang_ae = timeit(lambda: (ae.lang(lang_ids), ae.subtask_enc(subtask_ids)))
    t_cond_ae = timeit(lambda: ae.cond(ae_lang, ae_subtask, meta, t))
    t_act_embed = timeit(lambda: ae.action_embed(a_t) + ae.action_pos)
    t_dit_ae = timeit(ae_dit)
    t_head_ae = timeit(ae_head)

    head_only_ae = t_head_ae - t_dit_ae
    total_fwd_ae = t_vis_hist_ae + t_vis_sub_ae + t_lang_ae + t_cond_ae + t_act_embed + t_dit_ae + max(head_only_ae, 0)

    print(f"  vision (history 4 frames) : {t_vis_hist_ae:.3f} ms ({t_vis_hist_ae/total_fwd_ae*100:.1f}%)")
    print(f"  vision (subgoal)          : {t_vis_sub_ae:.3f} ms ({t_vis_sub_ae/total_fwd_ae*100:.1f}%)")
    print(f"  language (instr+subtask)  : {t_lang_ae:.3f} ms ({t_lang_ae/total_fwd_ae*100:.1f}%)")
    print(f"  conditioning (adaLN)      : {t_cond_ae:.3f} ms ({t_cond_ae/total_fwd_ae*100:.1f}%)")
    print(f"  action embedding          : {t_act_embed:.3f} ms ({t_act_embed/total_fwd_ae*100:.1f}%)")
    print(f"  DiT blocks (6 layers)     : {t_dit_ae:.3f} ms ({t_dit_ae/total_fwd_ae*100:.1f}%)")
    print(f"  head                      : {max(head_only_ae,0):.3f} ms ({max(head_only_ae,0)/total_fwd_ae*100:.1f}%)")
    print(f"  TOTAL forward             : {total_fwd_ae:.3f} ms")
    print(f"  -> x4 denoise steps = AE sample ~{total_fwd_ae*4:.3f} ms (measured {t_ae_sample:.3f} ms)")


if __name__ == "__main__":
    main()
