# mini-wam

A **compute-scaled π0.7-style World-Action Model / Vision-Language-Action system**
in a controlled 2D grayscale game.

The goal is *not* to build the strongest game-playing agent. It is to reproduce
the core computational structure of a modern WAM/VLA such as π0.7 — **World Model
→ visual subgoal → flow-matching action expert** — while shrinking the visual
complexity, model size, action dimension, dataset size and compute enough to
train and run the whole system on a single consumer GPU (NVIDIA RTX 5060 Ti,
16 GB). **And furthermore make quick testing environment for new architectures.**

```
Language Instruction
        ↓
Subtask (oracle / learned)
        ↓
Observation History (4 grayscale frames)
        ↓
World Model (flow-matching)  ──→  Visual Subgoal (future frame)
        ↓
Action Expert (flow-matching) ──→  Action Chunk (8 WASD)
        ↓
Game Environment ──→ New Observation ──↺  (receding-horizon, closed loop)
```

---

## Research questions

| Q | Question | Experiment |
|---|---|---|
| Q1 | Does visual-subgoal conditioning improve control? | `π(H,L)` vs `π(H,L,G)` |
| Q2 | Does language grounding improve compositional generalization? | OOD splits (held-out shape×fill combos, layouts, distractors) |
| Q3 | Does flow matching beat a plain action classifier? | Simple policy (classifier) vs action chunk (flow matching) |
| Q4 | Does the full WAM architecture improve OOD performance? | Simple policy vs cascade WAM |

---

## Environment

* 48×48 grayscale observations (no colour — forces genuine shape×fill grounding).
* Six semantic objects: `{circle, triangle, square} × {solid, hollow}`.
* Player is a small cross; obstacles are grey rectangles; simple damped physics
  `v ← λv + a`, `x ← x + v`.
* WASD controls; deterministic under a fixed seed; human- and AI-controllable.
* Reaching the target within a radius ends the episode successfully.

The task generator varies player/target/distractor/obstacle placement, object
vocabulary, and instruction wording (multiple templates per semantic).

---

## Architecture (all parameters configurable)

| Component | Description | Params |
|---|---|---|
| Vision encoder | patch-embed 48×48 → tokens (shared) | — |
| Language encoder | word-level vocab + small Transformer | — |
| **World Model** | conditional flow-matching subgoal generator (DiT) | ~11.5 M |
| **Action Expert** | conditional flow-matching action-chunk generator (DiT) | ~11.5 M |
| **Simple policy** | observation → single WASD classifier (baseline) | ~6.4 M |

The World Model learns `p(g | H, L, S, M)` (a *future visual subgoal*), the Action
Expert learns `p(A | H, L, S, G, M)` (an 8-step action chunk), both via flow
matching with 3–5 Euler denoising steps at inference. Action chunks are executed
with receding horizon (`H=8`, execute first `K=2`, replan).

---

## Layout

```
game/        environment, renderer, physics, collision, objects, tasks, expert, keyboard
data/        dataset, generator, sampler, validator
models/      vision_encoder, language_encoder, flow_matching, diffusion_transformer,
             world_model, action_expert, simple_policy, cascade_wam
training/    train_simple_policy, train_world_model, train_action_expert (+ utils)
evaluation/  evaluate_policy, evaluate_world_model, ood, metrics
scripts/     generate_dataset, inspect_dataset, train, evaluate, run_agent, sample_subgoal
configs/     YAML for game / each model / cascade
tests/       unit tests (physics, collision, rendering, semantics, flow matching, models)
```

---

## Quick start

```bash
# 1. install torch for Blackwell (RTX 50xx) + the package
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install -e .

# 2. tests
pytest

# 3. generate the wide training data (20k episodes, 1-6 distractors, 1-3 obstacles)
python scripts/generate_dataset.py --episodes 20000 --split train_wide --output data/train_wide --workers 8
python scripts/generate_dataset.py --episodes 1000  --split val   --output data/val

# 4. train the baseline
python training/train_simple_policy.py --config configs/simple_policy.yaml

# 5. train the world model (goal-state + object-weighted)
python training/train_world_model.py --config configs/world_model_goal_wide.yaml

# 6. train the action expert (Phase 6 + cosine LR)
python training/train_action_expert.py --config configs/action_expert_goal_wide.yaml

# 7. closed-loop evaluation (async cascade)
python scripts/eval_cascade_full.py \
    --world-model checkpoints/world_model_goal_wide_best.pt \
    --action-expert checkpoints/action_expert_goal_wide_best.pt --episodes 200

# 8. interactive demo (web UI)
python scripts/web_demo.py --port 8000
```

---

## Baselines & ablations

* **Baseline 1** — simple policy (obs → single action classifier).
* **Baseline 2** — action-chunk policy (flow matching, *no* subgoal; `use_subgoal: false`).
* **Main** — cascade WAM (world model → **goal-state visual subgoal** → action expert).
* Ablations: no-subgoal vs subgoal, short-horizon vs goal-state subgoal, real vs
  generated subgoal, object weighting on the world model, flow matching vs
  classifier, history length, subgoal horizon, model size.

**Key result** (200-episode closed-loop, test split): the cascade with a
**goal-state subgoal + object-weighted world model + diverse (wide) training data**
reaches **92.5%**, vs 52.5% for the simple policy and 62.5% for the no-subgoal
action-chunk policy — confirming that the world-model → subgoal → action-expert
structure helps, provided the subgoal is a goal state, the world model renders it
faithfully, and the training data is diverse. OOD: unseen combos 90.5%, dense
distractors 86%, multiple obstacles 81%.

See `docs/report.md` for the full experimental results.
