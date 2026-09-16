"""Full closed-loop evaluation of the cascade WAM using the ASYNC CascadeWAM
(π0.7-style low-frequency subgoal), matching what the demo actually runs.

Runs the main (test) split plus the five OOD splits and prints a summary table.

Usage:
    python scripts/eval_cascade_full.py \
        --world-model checkpoints/world_model_goal_wide_best.pt \
        --action-expert checkpoints/action_expert_goal_wide_best.pt \
        --episodes 200
"""

import argparse
import json

import numpy as np

from training.utils import build_world_model, build_action_expert
from models.cascade_wam import CascadeWAM
from game.env import GameEnv
from data.generator import SPLIT_CONFIGS, split_to_task_kwargs


def run_episode(wm, ae, task_kwargs, seed, execute_steps=2, max_steps=200,
                subgoal_every=5):
    wam = CascadeWAM(wm, ae, device="cuda", subgoal_update_every=subgoal_every,
                     async_mode=True)
    env = GameEnv(task_kwargs=task_kwargs)
    env.reset(seed=seed)
    wam.set_instruction(env.task.instruction)
    obs = env.get_observation()
    done = False
    steps = 0
    while not done and steps < max_steps:
        wam.observe(obs[-1])
        chunk = wam.step()
        for a in chunk.argmax(-1).tolist()[:execute_steps]:
            obs, r, done, info = env.step(int(a))
            if done:
                break
        steps += 1
    wam.shutdown()
    return info["success"], steps, info["final_distance"]


def evaluate_split(wm, ae, task_kwargs, n, base_seed):
    succ = 0
    lens = []
    for i in range(n):
        s, l, _ = run_episode(wm, ae, task_kwargs, base_seed + i)
        succ += s
        lens.append(l)
    return succ / n, np.mean(lens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-model", required=True)
    ap.add_argument("--action-expert", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    wm, _ = build_world_model(args.world_model, "cuda")
    ae, _ = build_action_expert(args.action_expert, "cuda")

    splits = ["test", "ood_combo", "ood_distractors", "ood_layout", "ood_long"]
    results = {}
    for split in splits:
        kw = split_to_task_kwargs(SPLIT_CONFIGS[split])
        succ, avg_len = evaluate_split(wm, ae, kw, args.episodes, args.seed)
        results[split] = {"success_rate": round(succ, 4), "avg_len": round(avg_len, 1)}
        print(f"{split:16s}  success={succ*100:5.1f}%  avg_len={avg_len:6.1f}",
              flush=True)

    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
