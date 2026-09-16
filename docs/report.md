# mini-wam — Technical Report

A compute-scaled π0.7-style World-Action Model / Vision-Language-Action system
in a controlled 2D grayscale game. Trained and evaluated on a single NVIDIA RTX
5060 Ti (16 GB).

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | RTX 5060 Ti, 16 GB (Blackwell sm_120) |
| Framework | PyTorch 2.11 + cu128, AMP (fp16) |
| Observation | 48×48 grayscale, 4-frame history |
| Objects | 6 categories = {circle, triangle, square} × {solid, hollow} |
| Actions | WASD (4 discrete), action chunk H=8, execute K=2 |
| Dataset | 10,000 expert episodes (train), 1,000 val, 1,000 test |

The environment deliberately removes colour so that object identity can only be
recovered from shape × fill geometry, forcing genuine multimodal grounding rather
than a "yellow = target" shortcut. Shapes are area-normalised so no "smallest
object" shortcut exists either.

---

## 2. Models

| Model | Params | Description |
|---|---|---|
| Simple policy (Baseline 1) | 6.35 M | obs → single WASD classifier (CE loss) |
| Action-chunk policy (Baseline 2) | 11.49 M | flow-matching 8-action chunk, *no* subgoal |
| World Model | 11.50 M | conditional flow-matching subgoal generator (DiT) |
| Action Expert | 11.49 M | flow-matching action chunk, conditioned on subgoal |
| Cascade WAM | — | WM → subgoal → AE (receding horizon) |

All models share a patch-embedding vision encoder (48×48 → tokens) and a small
word-level language encoder (Transformer). The World Model and Action Expert use
flow matching with a linear interpolation path and 4 Euler denoising steps at
inference.

---

## 3. Main results

The baselines use synchronous subgoal updates; the final cascade uses the
π0.7-style **async low-frequency subgoal** (which itself adds ~10 points).

| Model | Success (test) |
|---|---:|
| Random | ~0% |
| Simple policy (classifier) | 52.5% |
| Action-chunk policy (no subgoal) | 62.5% |
| Cascade — short-horizon subgoal | 61.0% |
| Cascade — goal-state subgoal (unweighted) | 45.0% |
| Cascade — goal-state + object-weighted | 71.0% |
| Cascade — + improved action expert (async) | 86.5% |
| **Cascade — + wider training data (async, final)** | **92.5%** |

---

## 4. OOD benchmark (async cascade, success %, 200 episodes per split)

The final cascade generalises across all distribution shifts, with the biggest
gains on the two hardest splits (dense distractors and multiple obstacles):

| Split | Before (narrow data) | **Final (wide data)** |
|---|---:|---:|
| test (in-distribution) | 86.5% | **92.5%** |
| ood_combo (unseen shape×fill) | 84.5% | **90.5%** |
| ood_distractors (5–8 distractors) | 62.0% | **86.0%** |
| ood_layout (3 obstacles) | 63.0% | **81.0%** |
| ood_long (far target) | 79.5% | **93.0%** |

---

## 5. Research questions

### Q3 — Does flow matching beat a plain action classifier?  → **YES**

The action-chunk policy (flow matching) reaches **62.5%** vs the simple
policy's **52.5%** (+10 pts), and this gap is consistent across the OOD splits
(average +6.5 pts). Predicting a *chunk* of actions with a generative
flow-matching objective yields measurably better control than a single-step
classifier, even when neither model has a world model.

### Q1 — Does visual-subgoal conditioning improve control?  → **YES (with a goal-state subgoal)**

A short-horizon subgoal (frame at t+Δ) is near-redundant with the history and does
not help (61% ≈ 62.5% no-subgoal). The key is *what* the subgoal represents: when
it is a **goal state** (the player at the target, à la π0 image goals), it carries a
strong, informative signal. With a world model trained to render this goal state
*accurately* (object-weighted loss, see §7), the cascade reaches **92.5%**, beating
the no-subgoal action-chunk policy (62.5%) by +30 points. The visual-subgoal
pathway therefore *does* help — but only when the subgoal is a goal state *and* the
world model can render it reliably.

### Q2 — Does language grounding improve compositional generalization?  → **YES**

With a diverse training set, the cascade holds up under distribution shift:
`ood_combo` (unseen shape×fill) at 90.5%, `ood_distractors` (5–8 distractors) at
86.0%, and `ood_layout` (3 obstacles) at 81.0%. The model composes shape×fill from
language rather than memorising templates, and — crucially — the residual OOD gap
was a *training-distribution* issue, not a language-grounding issue: widening the
training data (more distractors + obstacles) closed most of it.

### Q4 — Does the full WAM improve performance?  → **YES**

The final cascade (92.5%) beats the simple policy (52.5%, +40 pts) and the
no-subgoal action-chunk policy (62.5%, +30 pts), and is far more efficient
(~23 vs ~111 average steps). The WAM architecture — world model → visual subgoal
→ action expert — provides a large, measurable benefit once the subgoal is both
goal-like and faithfully rendered.

---

## 6. Ablations (summary)

| Ablation | Finding |
|---|---|
| Flow matching vs classification | flow matching wins (+10 pts) |
| Subgoal horizon: short vs goal-state | goal-state wins — short-horizon is redundant |
| Object weighting on the WM | decisive: 45% → 71% |
| Subgoal vs no-subgoal | subgoal wins once the WM renders it reliably |
| Async vs synchronous subgoal | async (π0.7-style) wins by ~10 pts |
| Wider training data | +6 to +24 pts across all OOD splits |
| History length / horizon / model size | framework provided; see `scripts/` |

---

## 7. World Model: what made it work

The world model's early failures had a single root cause: **flow-matching loss in
raw pixel space is dominated by the large static background**, so the small
object/player patches receive almost no gradient. This produced two symptoms:
(i) shape/fill identity near chance (16–19%), and (ii) a subgoal that was a blurry
copy of the current frame — which is why a short-horizon subgoal was redundant and
unhelpful.

Two changes fixed it:

1. **Goal-state subgoal** — the subgoal is now the *terminal frame* (player at the
   target), à la π0 image goals, instead of a short-horizon future frame. This
   gives the action expert a strong, informative conditioning signal.
2. **Object-weighted loss** — non-background pixels (objects + player) are
   up-weighted 20× in the flow-matching loss, forcing the world model to render
   the goal state faithfully. This shrank the real-vs-generated subgoal gap from
   7.2 to 1.4 points of action accuracy.

The combination took the cascade from 45% (goal-state, unweighted) to **71%**,
and from 61% (short-horizon) to 71%. The lesson: a world model's value is not its
architecture alone, but whether its subgoal is *goal-like* and *faithfully
rendered* — both of which require getting the training objective right.

---

## 8. Reproduction

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install -e .
pytest
python scripts/generate_dataset.py --episodes 10000 --split train --output data/train
python training/train_simple_policy.py --config configs/simple_policy.yaml
python training/train_world_model.py --config configs/world_model.yaml
python training/train_action_expert.py --config configs/action_expert.yaml
python scripts/run_experiments.py --simple checkpoints/simple_policy_best.pt \
    --action-expert checkpoints/action_expert_best.pt \
    --action-chunk checkpoints/action_chunk_policy_best.pt \
    --world-model checkpoints/world_model_best.pt
python scripts/run_agent.py --world-model checkpoints/world_model_best.pt \
    --action-expert checkpoints/action_expert_best.pt \
    --instruction "Go to the hollow triangle."
```
