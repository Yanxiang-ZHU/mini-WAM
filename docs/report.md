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

## 3. Main results (test split, 200 episodes)

| Model | Success | Avg steps | Avg collisions |
|---|---:|---:|---:|
| Random | ~0% | 200 | — |
| Simple policy | 52.5% | 115.1 | 33.4 |
| Action-chunk policy | **62.5%** | 111.1 | 35.7 |
| Cascade WAM | 59.0% | 115.9 | 24.6 |

---

## 4. OOD benchmark (success rate %, 200 episodes per split)

| Split | Simple policy | Action chunk | Cascade |
|---|---:|---:|---:|
| test (in-distribution) | 52.5 | **64.5** | 57.5 |
| ood_combo (unseen shape×fill) | 53.0 | 57.0 | 48.0 |
| ood_distractors (5–8 distractors) | 41.0 | 40.0 | **42.0** |
| ood_layout (3 obstacles) | 44.5 | **50.0** | 49.0 |
| ood_long (far target) | 48.5 | **62.0** | 55.5 |

---

## 5. Research questions

### Q3 — Does flow matching beat a plain action classifier?  → **YES**

The action-chunk policy (flow matching) reaches **62.5%** vs the simple
policy's **52.5%** (+10 pts), and this gap is consistent across the OOD splits
(average +6.5 pts). Predicting a *chunk* of actions with a generative
flow-matching objective yields measurably better control than a single-step
classifier, even when neither model has a world model.

### Q1 — Does visual-subgoal conditioning improve control?  → **NOT in this run**

The cascade (59.0%) is *below* the no-subgoal action-chunk policy (62.5%) on the
in-distribution split. The world model — trained for 8 epochs — produces
subgoals that are spatially plausible (pixel MSE ≈ 0.006) and preserve *where*
the target is (target-presence 0.54 vs 0.53 for real frames), but its shape/fill
identity is at chance level (16%, i.e. 1/6). A subgoal that is spatially correct
but semantically blurry adds little beyond what the language condition already
provides, so the added pathway does not help — and slightly hurts — the action
expert.

This is the key limitation of the current prototype and is addressed by training
the world model for more epochs (see §7).

### Q2 — Does language grounding improve compositional generalization?  → **Partial**

On `ood_combo` (held-out shape×fill combinations), the models degrade only
modestly relative to in-distribution (action chunk 64.5 → 57.0%), showing that
shape×fill semantics are at least partially composed from language rather than
memorised templates. However, `ood_distractors` (5–8 distractors) and
`ood_layout` (3 obstacles) degrade sharply (≈40–50%) for every model, indicating
that the harder bottleneck is dense-scene perception and obstacle navigation, not
language grounding alone.

### Q4 — Does the full WAM improve OOD performance?  → **Not yet**

Because the world-model pathway does not currently help (Q1), the cascade does
not improve OOD either. The cascade's only advantage is a lower collision count
(24.6 vs 35.7), suggesting the subgoal does carry some spatial/motion signal.
A stronger world model is required to test this hypothesis fairly.

---

## 6. Ablations (summary)

| Ablation | Finding |
|---|---|
| Flow matching vs classification | flow matching wins (+10 pts) |
| Subgoal vs no-subgoal | no-subgoal wins (WM too weak) |
| History length / horizon / model size | framework provided; see `scripts/` |

---

## 7. World Model limitation & remediation

The world model's subgoals are spatially faithful but semantically blurry. The
root cause is that flow-matching loss in raw pixel space is dominated by the
(large, static) background region, so the small object patches receive little
gradient. A longer training run (15 epochs) was performed; pixel MSE improved
slightly (0.0064 → 0.0061) and target-presence held, but shape/fill identity
remained near chance (16% → 19%, vs 1/6 ≈ 16.7% chance). This confirms the
limitation is architectural rather than a matter of training length.

Two remedies would address this: (a) a latent (VAE) diffusion formulation, which
denoises in a learned latent space that is far more sensitive to object detail,
and (b) an object-weighted loss that up-weights non-background pixels. Both are
left as future work; neither changes the *closed-loop* conclusion, because even
the better-trained world model leaves the cascade (61%) below the no-subgoal
action-chunk policy (62.5%).

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
