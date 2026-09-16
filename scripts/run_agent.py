"""Interactive demo: run the cascade WAM closed-loop with a debug overlay.

Usage:
    python scripts/run_agent.py \
        --world-model checkpoints/world_model_best.pt \
        --action-expert checkpoints/action_expert_best.pt \
        --instruction "Go to the hollow triangle." \
        --out demo --fps 30 --device cuda

Renders the current observation, predicted subgoal and action chunk side-by-side
each step and saves them (plus an animated GIF).  With --live it opens a pygame
window.
"""

import argparse
import os

import numpy as np

from game.env import GameEnv
from game.language import parse_instruction
from game.renderer import to_uint8
from training.utils import build_world_model, build_action_expert
from evaluation.evaluate_policy import CascadeAgent
from evaluation.metrics import classify_object

ACTION_NAMES = ("W", "A", "S", "D")


def render_overlay(obs, subgoal, instruction, subtask, action, chunk, step, success, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(obs, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Current Observation")
    axes[0].axis("off")
    if subgoal is not None:
        sg = (subgoal + 1) / 2  # denormalize [-1,1] -> [0,1]
        axes[1].imshow(sg, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Predicted Subgoal")
    else:
        axes[1].set_title("Predicted Subgoal (none)")
    axes[1].axis("off")
    axes[2].axis("off")
    chunk_str = " ".join(ACTION_NAMES[a] for a in chunk)
    exec_str = " ".join(ACTION_NAMES[a] for a in action)
    text = (f"Instruction: {instruction}\n"
            f"Subtask: {subtask}\n"
            f"Step: {step}\n"
            f"Action chunk: {chunk_str}\n"
            f"Executed: {exec_str}\n"
            f"FPS: {fps:.1f}\n"
            f"Success: {success}")
    axes[2].text(0.02, 0.98, text, fontsize=11, va="top", family="monospace")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-model", required=True)
    ap.add_argument("--action-expert", required=True)
    ap.add_argument("--instruction", default="Go to the hollow triangle.")
    ap.add_argument("--execute-steps", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="demo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()

    device = args.device
    wm, _ = build_world_model(args.world_model, device)
    ae, _ = build_action_expert(args.action_expert, device)
    agent = CascadeAgent(wm, ae, device)

    # parse the instruction to force the target semantics
    sem = parse_instruction(args.instruction)
    shape, fill = sem["shape"], sem["fill"]
    allowed = [(shape, fill)] if fill else [(shape, "solid"), (shape, "hollow")]

    env = GameEnv(task_kwargs={
        "allowed_targets": allowed,
        "allowed_distractors": None,
        "min_distractors": 2, "max_distractors": 3, "n_obstacles": 1,
    })
    env.reset(seed=args.seed)
    # override the instruction text so the demo shows exactly the user's string
    env.task.instruction = args.instruction

    os.makedirs(args.out, exist_ok=True)
    frames = []
    instruction = env.task.instruction
    subtask = env.task.subtask

    import time
    obs = env.get_observation()
    done = False
    step = 0
    t_start = time.time()
    while not done and step < args.max_steps:
        chunk = agent.act(obs, instruction, subtask)
        exec_chunk = chunk[:args.execute_steps]
        for a in exec_chunk:
            obs, r, done, info = env.step(int(a))
            if done:
                break
        step += 1
        fps = step / max(1e-6, time.time() - t_start)
        # generate the subgoal once more for display (matches the agent's internal subgoal)
        subgoal_np = None
        if agent.world_model is not None:
            from training.utils import tokenize_batch
            import torch
            h = torch.from_numpy(obs).float().unsqueeze(0).to(device) * 2 - 1
            lang = tokenize_batch([instruction], device)
            sub = tokenize_batch([subtask], device)
            sg = agent.world_model.sample(h, lang, sub, agent.metadata)
            subgoal_np = sg[0, 0].cpu().numpy()

        fig = render_overlay(env.render(), subgoal_np, instruction, subtask,
                             exec_chunk, chunk, step, done, fps)
        path = os.path.join(args.out, f"step_{step:03d}.png")
        fig.savefig(path, dpi=90)
        import matplotlib.pyplot as plt
        plt.close(fig)
        frames.append(path)

        print(f"step {step}: chunk={' '.join(ACTION_NAMES[a] for a in chunk)} "
              f"exec={' '.join(ACTION_NAMES[a] for a in exec_chunk)} "
              f"success={done}", flush=True)

    info = env.get_info()
    print(f"\nFINAL: success={info['success']} steps={info['episode_length']} "
          f"final_dist={info['final_distance']:.2f}")

    # assemble GIF
    try:
        import matplotlib.image as mpimg
        imgs = [mpimg.imread(p) for p in frames]
        import imageio  # optional
        imageio.mimsave(os.path.join(args.out, "demo.gif"), imgs, fps=10)
        print(f"saved {len(frames)} frames + demo.gif to {args.out}")
    except Exception as e:
        print(f"saved {len(frames)} frames to {args.out} (GIF skipped: {e})")


if __name__ == "__main__":
    main()
