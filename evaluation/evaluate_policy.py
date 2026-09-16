"""Closed-loop evaluation of trained policies in the actual game environment.

Supports three agent types under a unified ``act(history, instruction, subtask)
-> list[int]`` interface (a chunk of discrete actions, executed with receding
horizon):

    SimplePolicyAgent   -- single-action classifier (Baseline 1)
    ActionExpertAgent   -- flow-matching chunk, optional subgoal (Baseline 2 / main)
    CascadeAgent        -- world model -> generated subgoal -> action expert
"""

from __future__ import annotations

import numpy as np
import torch

from game.env import GameEnv
from game.tasks import generate_task
from game.expert import Expert
from training.utils import tokenize_batch
from evaluation.metrics import summarize


def _norm(history: np.ndarray) -> torch.Tensor:
    """History [0,1] -> [-1,1] tensor (B=1, K, 48, 48)."""
    return torch.from_numpy(history).float().unsqueeze(0).to("cuda") * 2 - 1


class SimplePolicyAgent:
    def __init__(self, model, device="cuda"):
        self.model = model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def act(self, history, instruction, subtask):
        h = _norm(history).to(self.device)
        lang = tokenize_batch([instruction], self.device)
        logits = self.model(h, lang)
        return [int(logits.argmax(-1).item())]


class ActionExpertAgent:
    def __init__(self, model, device="cuda", world_model=None, subgoal_mode="generated",
                 metadata=None):
        self.model = model.eval().to(device)
        self.device = device
        self.world_model = world_model
        self.subgoal_mode = subgoal_mode  # 'none' | 'generated'
        self.metadata = metadata or [{"speed": "normal", "quality": 1.0,
                                      "mistake": False, "control_mode": "keyboard"}]

    @torch.no_grad()
    def act(self, history, instruction, subtask):
        h = _norm(history).to(self.device)
        lang = tokenize_batch([instruction], self.device)
        sub = tokenize_batch([subtask], self.device)

        subgoal = None
        if self.subgoal_mode == "generated":
            assert self.world_model is not None, "generated subgoal requires a world model"
            subgoal = self.world_model.sample(h, lang, sub, self.metadata)
        elif self.model.use_subgoal:
            # no subgoal available: feed a blank frame as a neutral placeholder
            subgoal = torch.zeros(1, 1, 48, 48, device=self.device)

        chunk = self.model.sample(h, subgoal, lang, sub, self.metadata)
        return chunk.argmax(-1).squeeze(0).tolist()


class CascadeAgent(ActionExpertAgent):
    """Action expert + world model (generated subgoal)."""

    def __init__(self, world_model, action_expert, device="cuda", metadata=None):
        super().__init__(action_expert, device=device, world_model=world_model,
                         subgoal_mode="generated", metadata=metadata)


def run_closed_loop(agent, env: GameEnv, execute_steps: int, max_steps: int = 200,
                    task=None, seed=None) -> dict:
    """Run one episode and return the info dict plus efficiency vs. expert."""
    env.reset(seed=seed, task=task)
    instruction = env.task.instruction
    subtask = env.task.subtask

    # expert reference length (for action efficiency)
    expert_len = _expert_length(env)

    obs = env.get_observation()
    done = False
    while not done and env.steps < max_steps:
        chunk = agent.act(obs, instruction, subtask)
        for a in chunk[:execute_steps]:
            obs, r, done, info = env.step(int(a))
            if done:
                break
    info = env.get_info()
    info["expert_length"] = expert_len
    return info


def _expert_length(env: GameEnv) -> int:
    from game.env import GameEnv as G
    # replay expert on a copy of the task to get reference length
    e = GameEnv()
    e.reset(seed=0, task=env.task)
    exp = Expert(e.task)
    exp.reset((e.player.x, e.player.y))
    n = 0
    while not e.is_done() and n < 300:
        a = exp.action((e.player.x, e.player.y), (e.player.vx, e.player.vy))
        e.step(a)
        n += 1
    return n if e.get_info()["success"] else env.cfg.max_steps


def evaluate_agent(agent, *, n_episodes: int = 200, execute_steps: int = 2,
                   seed: int = 0, task_kwargs: dict | None = None,
                   max_steps: int = 200, verbose: bool = True) -> dict:
    """Evaluate ``agent`` over ``n_episodes`` and return summary metrics."""
    env = GameEnv(task_kwargs=task_kwargs or {})
    results = []
    for i in range(n_episodes):
        info = run_closed_loop(agent, env, execute_steps, max_steps=max_steps,
                               seed=seed + i)
        results.append(info)
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_episodes} done", flush=True)
    return summarize(results)
