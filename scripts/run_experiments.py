"""Run the full evaluation suite: main results, OOD benchmark, and ablations.

Produces ``results/results.json`` and a filled ``docs/report.md``.

Usage:
    python scripts/run_experiments.py --simple checkpoints/simple_policy_best.pt \
        --action-expert checkpoints/action_expert_best.pt \
        --world-model checkpoints/world_model_best.pt \
        [--action-chunk checkpoints/action_chunk_policy_best.pt]
"""

import argparse
import json
import os
import time

import torch

from training.utils import (build_simple_policy, build_action_expert, build_world_model)
from evaluation.evaluate_policy import (SimplePolicyAgent, ActionExpertAgent,
                                        CascadeAgent, evaluate_agent)
from evaluation.ood import run_ood_benchmark
from data.generator import SPLIT_CONFIGS, split_to_task_kwargs


def build_agents(args, device):
    agents = {}
    if args.simple:
        m, _ = build_simple_policy(args.simple, device)
        agents["simple_policy"] = SimplePolicyAgent(m, device)
    if args.action_chunk:
        m, _ = build_action_expert(args.action_chunk, device)
        agents["action_chunk"] = ActionExpertAgent(m, device, subgoal_mode="none")
    if args.action_expert and args.world_model:
        wm, _ = build_world_model(args.world_model, device)
        ae, _ = build_action_expert(args.action_expert, device)
        agents["cascade"] = CascadeAgent(wm, ae, device)
    return agents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simple", default=None)
    ap.add_argument("--action-expert", default=None)
    ap.add_argument("--action-chunk", default=None)
    ap.add_argument("--world-model", default=None)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--execute-steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    agents = build_agents(args, args.device)
    print(f"agents: {list(agents)}", flush=True)

    results = {"args": vars(args), "main": {}, "ood": {}}
    kw = split_to_task_kwargs(SPLIT_CONFIGS["test"])

    # main results (test split)
    for name, agent in agents.items():
        print(f"\n=== main eval: {name} ===", flush=True)
        results["main"][name] = evaluate_agent(
            agent, n_episodes=args.episodes, execute_steps=args.execute_steps,
            seed=args.seed, task_kwargs=kw)

    # OOD benchmark
    for name, agent in agents.items():
        print(f"\n=== OOD benchmark: {name} ===", flush=True)
        results["ood"][name] = run_ood_benchmark(
            agent, n_episodes=args.episodes, execute_steps=args.execute_steps,
            seed=args.seed)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    _write_report(results, args.out)
    print(f"\nresults written to {args.out}/results.json and {args.out}/report.md")


def _write_report(results, out_dir):
    main = results["main"]
    ood = results["ood"]

    def row(name):
        m = main.get(name, {})
        return (f"| {name:14s} | {m.get('success_rate', 0)*100:6.1f}% | "
                f"{m.get('avg_episode_length', 0):8.1f} | "
                f"{m.get('avg_collisions', 0):6.1f} |")

    lines = ["# mini-wam results (auto-generated)", "", "## Main results (test split)", "",
             "| Model | Success | Avg steps | Collisions |", "|---|---|---|---|",
             "| Random | ~0% | 200 | — |"]
    for name in main:
        lines.append(row(name))
    lines.append("")

    lines += ["## OOD benchmark (success rate %)", "", "| Split | " +
              " | ".join(main.keys()) + " |",
              "|---|" + "---|" * len(main)]
    for split in ood.get(next(iter(ood), "simple_policy"), {}):
        cells = []
        for name in main:
            r = ood.get(name, {}).get(split, {})
            cells.append(f"{r.get('success_rate', 0)*100:.1f}%")
        lines.append(f"| {split:16s} | " + " | ".join(cells) + " |")
    lines.append("")

    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
