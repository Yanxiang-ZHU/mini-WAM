"""Top-level closed-loop evaluation for any trained policy.

Usage:
    python scripts/evaluate.py --model cascade --checkpoint checkpoints/action_expert_best.pt \
        --world-model checkpoints/world_model_best.pt --episodes 200
    python scripts/evaluate.py --model simple_policy --checkpoint checkpoints/simple_policy_best.pt --ood
"""

import argparse
import json

from training.utils import (build_simple_policy, build_action_expert, build_world_model)
from evaluation.evaluate_policy import (SimplePolicyAgent, ActionExpertAgent,
                                        CascadeAgent, evaluate_agent)
from evaluation.ood import run_ood_benchmark, print_ood_table
from data.generator import SPLIT_CONFIGS, split_to_task_kwargs


def build_agent(args):
    device = args.device
    if args.model == "simple_policy":
        model, _ = build_simple_policy(args.checkpoint, device)
        return SimplePolicyAgent(model, device)
    if args.model == "action_expert":
        model, _ = build_action_expert(args.checkpoint, device)
        return ActionExpertAgent(model, device, subgoal_mode="none")
    if args.model == "cascade":
        wm, _ = build_world_model(args.world_model, device)
        ae, _ = build_action_expert(args.checkpoint, device)
        return CascadeAgent(wm, ae, device)
    raise ValueError(f"unknown model {args.model}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["simple_policy", "action_expert", "cascade"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--world-model", default=None)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--execute-steps", type=int, default=2)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ood", action="store_true", help="run the full OOD benchmark")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    agent = build_agent(args)

    if args.ood:
        results = run_ood_benchmark(agent, n_episodes=args.episodes,
                                    execute_steps=args.execute_steps, seed=args.seed)
        print_ood_table(results)
        print(json.dumps(results, indent=2, default=str))
    else:
        kw = split_to_task_kwargs(SPLIT_CONFIGS[args.split])
        summary = evaluate_agent(agent, n_episodes=args.episodes,
                                 execute_steps=args.execute_steps, seed=args.seed,
                                 task_kwargs=kw)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
