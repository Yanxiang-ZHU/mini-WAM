# Tuning Log — how we pushed the mini-WAM accuracy up

A chronological record of every change we made, *why*, and its effect on the
closed-loop success rate. The headline metric is **closed-loop success rate**
(test split, 200 episodes); where relevant we also report the action expert's
isolated action-match accuracy.

---

## Accuracy trajectory (summary)

| Step | Change | Closed-loop success (test) |
|---|---|---|
| 0 | Short-horizon subgoal, unweighted loss | 61% |
| 1 | Goal-state subgoal (still unweighted) | 45% ⚠️ (worse) |
| 2 | + object-weighted loss (20×) | 71% |
| 3 | + async low-frequency subgoal (π0.7-style) | ~81% |
| 4 | + improved action expert (Phase 6 + cosine LR, 20 epochs) | 86.5% |
| 5 | + wider training data (more distractors + obstacles) | **92.5%** |

The action expert's isolated accuracy improved along the way from **64.2%** to
**73.5%** (per-step action match, real subgoal, narrow val set).

---

## Step 0 — baseline: short-horizon subgoal

- **What**: the visual subgoal was `I_{t+Δ}` (the frame 8–32 steps ahead), and the
  world model used a plain flow-matching loss.
- **Result**: cascade 61%, barely different from the no-subgoal action-chunk policy
  (62.5%). The short-horizon subgoal is a blurry near-copy of the current frame,
  so it carries almost no new information.

## Step 1 — goal-state subgoal (the user's insight)

- **What**: the subgoal became the *terminal frame* (player standing at the target),
  à la π0 image goals — a strong, informative signal instead of a procedural one.
- **Result**: 45%, *worse*. The concept was right, but the world model could not
  yet render the goal state — its shape/fill identity was near chance (19%), so it
  often drew the player at the *wrong* object, actively misleading the action
  expert.

## Step 2 — object-weighted loss

- **What**: up-weighted non-background pixels (objects + player) by 20× in the
  world model's flow-matching loss. This forces the model to care about the
  objects instead of being dominated by the large static background.
- **Why it works**: pixel-space flow matching is otherwise dominated by background
  reconstruction, so the small object patches get almost no gradient.
- **Result**: 71% (up from 45%). The real-vs-generated subgoal gap in the action
  expert's accuracy shrank from 7.2 to 1.4 points — the world model now renders
  the goal faithfully.

## Step 3 — async low-frequency subgoal (π0.7-style)

- **What**: the world model runs in a background thread and refreshes the subgoal
  every 5 steps (instead of synchronously every 2); the action expert uses the
  latest (possibly stale) subgoal and never blocks.
- **Why it works**: a goal-state subgoal is *static* (player at target), so
  regenerating it every 2 steps just adds stochastic noise; regenerating it less
  often keeps it stable.
- **Result**: ~81% (vs 71% synchronous). This is the evaluation mode the demo uses.

## Step 4 — improved action expert

- **What** (three changes at once):
  1. **More epochs**: 8 → 20.
  2. **Phase 6**: during training, 25% of samples use a world-model-*generated*
     subgoal instead of the real one, so the action expert is robust to the
     imperfect subgoals it sees at inference.
  3. **Cosine LR schedule** (decays to zero over the 20 epochs).
- **Result**: action-match accuracy 64.2% → **73.5%**; closed-loop 81% → **86.5%**.
  The real-vs-generated subgoal gap narrowed further (1.7 → 1.1 points).

## Step 5 — wider training data

- **Problem**: the OOD benchmark exposed two weak spots — `ood_distractors`
  (5–8 distractors: 62%) and `ood_layout` (3 obstacles: 63%). The training set
  only had 1–3 distractors and 1 obstacle, so these distributions were unseen.
- **What**: regenerated 20k training episodes with a *wide* distribution
  (1–6 distractors, 1–3 obstacles), retraining the world model and action expert
  on it. This is the π0 lesson: robustness comes from a *diverse* training set.
- **Result**: every split improved, with the biggest gains exactly on the two weak
  spots (see the OOD table below).

### Final OOD results (async cascade, 200 episodes per split)

| Split | Before Step 5 | After Step 5 |
|---|---:|---:|
| test | 86.5% | **92.5%** |
| ood_combo (unseen shape×fill) | 84.5% | **90.5%** |
| ood_distractors (5–8 distractors) | 62.0% | **86.0%** |
| ood_layout (3 obstacles) | 63.0% | **81.0%** |
| ood_long (far target) | 79.5% | **93.0%** |

---

## Key lessons

1. **A world model's value is not its architecture, but whether its subgoal is
   *goal-like* and *faithfully rendered*** — both require getting the training
   *objective* right (goal state + object weighting), not adding parameters.
2. **Flow-matching loss has a non-zero floor** (the irreducible conditional-to-
   marginal gap `C > 0`), so loss plateauing is expected; judge the model by its
   downstream accuracy instead.
3. **The π0 recipe transfers to tiny scale**: goal-state subgoal + flow-matching
   action chunk + async low-frequency subgoal + a diverse training set — each
   piece contributed measurably.
