"""CLI demo: run the cascade WAM closed-loop with a debug overlay.

Usage:
    python scripts/run_agent.py --world-model checkpoints/world_model_best.pt \
        --action-expert checkpoints/action_expert_best.pt \
        --instruction "Go to the hollow triangle." --out demo

* The visual subgoal is generated asynchronously at low frequency (π0.7-style).
* Omit ``--instruction`` to type it interactively.
* Omit ``--seed`` for a random scene layout each run.
* Saves overlay frames + an animated GIF (up to ``--frames`` steps).
"""

import argparse
import os

import numpy as np

from game.env import GameEnv
from game.language import parse_instruction
from training.utils import build_world_model, build_action_expert
from models.cascade_wam import CascadeWAM

ACTION_NAMES = ("W", "A", "S", "D")


def render_overlay(obs, subgoal, instruction, subtask, chunk, exec_chunk, step, success):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].imshow(obs, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Current Observation", fontsize=11)
    axes[0].axis("off")
    if subgoal is not None:
        axes[1].imshow(subgoal, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Predicted Subgoal (async, low-freq)", fontsize=11)
    else:
        axes[1].set_title("Predicted Subgoal (generating…)", fontsize=11)
    axes[1].axis("off")
    axes[2].axis("off")
    text = (f"Instruction: {instruction}\n"
            f"Subtask: {subtask}\n"
            f"Step: {step}\n"
            f"Action chunk: {' '.join(ACTION_NAMES[a] for a in chunk)}\n"
            f"Executed: {' '.join(ACTION_NAMES[a] for a in exec_chunk)}\n"
            f"Success: {success}")
    axes[2].text(0.02, 0.98, text, fontsize=11, va="top", family="monospace")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-model", default="checkpoints/world_model_goal_wide_best.pt")
    ap.add_argument("--action-expert", default="checkpoints/action_expert_goal_wide_best.pt")
    ap.add_argument("--instruction", default=None, help="omit to type interactively")
    ap.add_argument("--execute-steps", type=int, default=2)
    ap.add_argument("--subgoal-update-every", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--frames", type=int, default=100, help="max overlay frames to save")
    ap.add_argument("--seed", type=int, default=None, help="omit for a random scene")
    ap.add_argument("--out", default="demo")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    instruction = args.instruction
    if instruction is None:
        instruction = input("Instruction (e.g. 'Go to the hollow triangle.'): ").strip()
        if not instruction:
            instruction = "Go to the hollow triangle."

    device = args.device
    wm, _ = build_world_model(args.world_model, device)
    ae, _ = build_action_expert(args.action_expert, device)
    wam = CascadeWAM(wm, ae, device=device,
                     subgoal_update_every=args.subgoal_update_every, async_mode=True)

    # parse instruction -> force the target semantics, random layout
    sem = parse_instruction(instruction)
    shape, fill = sem["shape"], sem["fill"]
    allowed = [(shape, fill)] if fill else [(shape, "solid"), (shape, "hollow")]

    from game.tasks import generate_task
    seed = args.seed if args.seed is not None else int(np.random.randint(0, 2 ** 31))
    rng = np.random.default_rng(seed)
    task = generate_task(rng, allowed_targets=allowed, min_distractors=2,
                         max_distractors=3, n_obstacles=1)
    task.instruction = instruction

    env = GameEnv()
    env.reset(seed=0, task=task)
    wam.set_instruction(instruction)

    os.makedirs(args.out, exist_ok=True)
    import matplotlib.pyplot as plt

    obs = env.get_observation()
    done = False
    step = 0
    frames = []
    while not done and step < args.max_steps:
        wam.observe(obs[-1])
        chunk = wam.step()                                  # (H,4)
        chunk_idx = chunk.argmax(-1).tolist()
        exec_chunk = chunk_idx[:args.execute_steps]
        for a in exec_chunk:
            obs, r, done, info = env.step(int(a))
            if done:
                break
        step += 1

        sg = wam.subgoal
        sg_np = (sg[0, 0].cpu().numpy() + 1) / 2 if sg is not None else None
        fig = render_overlay(env.render(), sg_np, instruction, wam.subtask,
                             chunk_idx, exec_chunk, step, done)
        path = os.path.join(args.out, f"step_{step:03d}.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        frames.append(path)

        print(f"step {step}: chunk={' '.join(ACTION_NAMES[a] for a in chunk_idx)} "
              f"exec={' '.join(ACTION_NAMES[a] for a in exec_chunk)} success={done}",
              flush=True)
        if len(frames) >= args.frames:
            break

    info = env.get_info()
    print(f"\nFINAL: success={info['success']} steps={step} "
          f"final_dist={info['final_distance']:.2f} (seed={seed})")

    # assemble GIF with PIL
    try:
        from PIL import Image
        imgs = [Image.open(p) for p in frames]
        imgs[0].save(os.path.join(args.out, "demo.gif"), save_all=True,
                     append_images=imgs[1:], duration=300, loop=0)
        print(f"saved {len(frames)} frames + demo.gif -> {args.out}")
    except Exception as e:
        print(f"saved {len(frames)} frames -> {args.out} (GIF skipped: {e})")

    wam.shutdown()


if __name__ == "__main__":
    main()
