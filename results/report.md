# mini-wam results

Final results (async cascade, 200 episodes per split, NVIDIA RTX 5060 Ti).

## Final models

| Component | Checkpoint |
|---|---|
| World Model | `checkpoints/world_model_goal_wide_best.pt` |
| Action Expert | `checkpoints/action_expert_goal_wide_best.pt` |

## Main results (test split)

| Model | Success |
|---|---:|
| Random | ~0% |
| Simple policy (classifier) | 52.5% |
| Action-chunk policy (no subgoal) | 62.5% |
| Cascade — short-horizon subgoal | 61.0% |
| Cascade — goal-state subgoal (unweighted) | 45.0% |
| Cascade — goal-state + object-weighted | 71.0% |
| Cascade — + improved action expert (async) | 86.5% |
| **Cascade — + wider training data (async, final)** | **92.5%** |

## OOD benchmark (async cascade)

| Split | Narrow data | Wide data (final) |
|---|---:|---:|
| test (in-distribution) | 86.5% | **92.5%** |
| ood_combo (unseen shape×fill) | 84.5% | **90.5%** |
| ood_distractors (5–8 distractors) | 62.0% | **86.0%** |
| ood_layout (3 obstacles) | 63.0% | **81.0%** |
| ood_long (far target) | 79.5% | **93.0%** |

## Research questions

| Question | Answer |
|---|---|
| Q1 — visual subgoal helps control? | **YES** (goal-state subgoal, +30 pts over no-subgoal) |
| Q2 — language grounding → compositional generalization? | **YES** (with diverse data) |
| Q3 — flow matching beats a classifier? | **YES** (+10 pts) |
| Q4 — full WAM improves performance? | **YES** (+40 pts over simple policy) |

## Model sizes

| Model | Params |
|---|---|
| Simple policy | 6.35M |
| World Model | 11.50M |
| Action Expert | 11.49M |
| Cascade total | 23.0M |

## Training data

| Item | Value |
|---|---|
| Wide episodes | 20,000 |
| Distribution | 1–6 distractors, 1–3 obstacles |
| Expert success rate | 93.9% |
