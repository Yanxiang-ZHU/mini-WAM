# mini-wam — Engineering Documentation

This document describes every module, the data formats, the training/evaluation
flow, and the key design decisions. It is the definitive reference for extending
or modifying the project.

---

## 1. Overview

`mini-wam` reproduces the computational structure of a modern World-Action Model
/ Vision-Language-Action system (π0.7-style) in a 2D grayscale game:

```
Language → Subtask → Observation History → World Model → Visual Subgoal
                                              ↘
                       Action Expert → Action Chunk → Environment → (loop)
```

Three models are trained and compared:

| Model | Type | File |
|---|---|---|
| Simple policy (Baseline 1) | observation → single-action classifier | `models/simple_policy.py` |
| Action-chunk policy (Baseline 2) | flow-matching action chunk, no subgoal | `models/action_expert.py` (`use_subgoal: false`) |
| Cascade WAM | World Model → subgoal → Action Expert | `models/cascade_wam.py` |

---

## 2. Directory structure

```
game/          environment, renderer, physics, collision, objects, tasks, language, expert, keyboard
data/          generator (procedural data), dataset (loading/sampling), validator
models/        vision/language encoders, flow matching, DiT, WM, AE, simple policy, cascade
training/      utils (loaders, checkpointing), per-model train scripts
evaluation/    metrics (incl. CV parser), evaluate_policy, evaluate_world_model, ood
scripts/       generate_dataset, inspect_dataset, train, evaluate, run_agent, sample_subgoal, run_experiments
configs/       YAML for game / each model / cascade
tests/         unit tests (physics, collision, rendering, semantics, flow matching, models)
docs/          this doc + report.md
```

---

## 3. `game/` — environment

### `objects.py`
- `SHAPES = ("circle","triangle","square")`, `FILLS = ("solid","hollow")`.
- `Object(shape, fill, x, y, size)` — a semantic object (target or distractor).
  `size` is the half-extent (circle radius / square half-side / triangle circumradius).
- `Rect(cx, cy, hw, hh)` — an axis-aligned obstacle.
- Point-in-shape tests: `shape_contains` (solid interior) and `shape_hit`
  (solid **or** hollow, where hollow is the interior minus a shrunken core).
- **`AREA_SCALE`** — per-shape factor making all three shapes equal-area for a
  given `size` (removes the "smallest object = triangle" shortcut).
- **`INSET`** — per-shape factor so a hollow outline has uniform edge thickness
  (for a triangle, the inradius is size/2, so shrinking size by `t` moves each
  edge only `t/2`; `INSET["triangle"]=2` compensates).

### `physics.py`
- `Player(x, y, vx, vy)`; `step_player` applies `v ← λv + a`, `x ← x + v`, with
  speed clamping.
- `ACTION_DIRS = {W:(0,-1), A:(-1,0), S:(0,1), D:(1,0)}` (image coords, y down).

### `collision.py`
- `resolve_rect` pushes a disc out of a rectangle along the minimum-penetration
  axis; `resolve_boundary` clamps to the world; `resolve_all` iterates both.

### `renderer.py`
- `render_frame(player_xy, objects, obstacles)` → `float32 (48,48)` in `[0,1]`.
- Supersampled 4× per axis then area-averaged for smooth edges.
- Brightness levels: background `0.08`, obstacle `0.35`, object `0.75`, player
  `1.00` (player is a small cross, distinct from all six categories).
- `to_uint8` for PNG/`.npz` storage.

### `tasks.py`
- `generate_task(rng, **kwargs)` samples a random scene: target (shape/fill),
  player/target/distractor placement, obstacles, and a natural-language
  instruction + subtask. Supports `allowed_targets`/`allowed_distractors` (for
  held-out OOD splits), `shape_only_p`, `force_obstacle`, `force_long`.

### `language.py`
- `make_instruction` / `make_subtask` use varied templates so the policy cannot
  memorise exact strings.
- `parse_instruction(text)` → `{"shape", "fill"}` (fill may be `None` for
  shape-only instructions). This is the shared ground-truth parser.

### `expert.py`
- Deterministic A* controller for dataset generation. Uses 4-connected A* on a
  2px grid when obstacles block the direct path, direct steering otherwise, plus
  a stuck detector that re-plans. ~98% success across random seeds.

### `env.py`
- `GameEnv(config)` with `reset(seed, task)`, `step(action)`, `render()`,
  `get_observation()` (K-frame history), `get_state()`, `get_info()`,
  `is_done()`. `EnvConfig` holds physics + history + reach parameters.

### `keyboard.py`
- Optional pygame human WASD control (for manual data collection / debugging).

---

## 4. `data/` — dataset

### `generator.py`
- `SplitConfig` dataclass + `SPLIT_CONFIGS` for `train/val/test`, the OOD splits
  (`ood_combo`, `ood_distractors`, `ood_layout`, `ood_long`), and `train_wide`
  (the diverse training distribution: 1–6 distractors, 1–3 obstacles via
  `n_obstacles_max`).
- `HELD_OUT_COMBOS` — the three held-out shape×fill pairs used for compositional
  OOD (training never sees them).
- `generate_episode(rng, cfg)` runs the expert, collects frames + actions +
  metadata. `generate_split` parallelises with `ProcessPoolExecutor`.

**Episode `.npz` schema** (one per episode, `episode_XXXXXX.npz`):

| key | dtype / shape | meaning |
|---|---|---|
| `frames` | uint8 `(T,48,48)` | full frame sequence `[0,255]` |
| `actions` | int8 `(T-1,)` | expert action at each step `0..3` |
| `meta_json` | str (JSON) | instruction, subtask, target semantic, success, physics, counts |

### `dataset.py`
- `EpisodeDataset` preloads all episodes into memory, builds a flat index of
  `(episode, t, subgoal_delta)` triples, and yields samples.
- `extract_sample` returns `history (K,48,48)`, `subgoal (48,48)`,
  `subgoal_delta`, `action_chunk (H,)`, `action_onehot (H,4)`.
- **`goal` mode**: when `goal=True`, the subgoal is the *terminal frame* (player
  at/near the target) — a goal-state conditioning à la π0 image goals — instead
  of a short-horizon future frame. This is the mode that makes the subgoal
  informative (see `docs/report.md` §7).
- `collate` stacks a batch. Frames are normalised `[0,1] → [-1,1]`.
- **Why store full frames?** the subgoal horizon is a sampling choice at train
  time, so storing full trajectories lets you switch between short-horizon and
  goal-state subgoals (or ablate the horizon) without regenerating data.

### `validator.py`
- Structural + semantic integrity checks (shape, action range, instruction→target
  consistency).

---

## 5. `models/`

### `vision_encoder.py`
- `PatchEmbed`: Conv2d stride=patch → `(48/patch)²` tokens. `VisionEncoder` adds
  spatial + temporal positional embeddings and encodes a K-frame history.

### `language_encoder.py`
- A fixed word-level vocabulary (shapes, fills, template words + specials).
  `tokenize` lowercases, strips punctuation, pads to `MAX_LEN=12`.
- `LanguageEncoder`: embedding + small TransformerEncoder → CLS-token vector.
  Instruction and subtask share this encoder (two instances, identical arch).

### `flow_matching.py`
- `interpolate(x0, t)` → `(x_t, v*)` with `x_t = (1-t)ε + t·x0`, `v* = x0 − ε`.
- `flow_matching_loss` = MSE. `euler_sample` integrates the velocity with N steps.

### `diffusion_transformer.py`
- `timestep_embedding` (sinusoidal), `TimestepMLP`, `MetadataEncoder` (encodes
  speed / control_mode / quality / mistake), `DiTBlock` (adaLN-zero block), and
  `ConditioningEncoder` which fuses `[language; subtask; metadata; time] → c`.

### `world_model.py`
- `WorldModel(cfg)`: patches the noisy subgoal into tokens, attends over them
  jointly with history tokens, conditioned by `c` via adaLN, and predicts the
  velocity field (unpatchified back to 48×48). `loss()` does flow matching;
  `sample()` runs Euler integration.
- **Object-weighted loss** (`obj_weight`, default 20): non-background pixels
  (objects + player, value > 0 in the target frame) are up-weighted in the
  flow-matching loss, so the model learns to render objects sharply instead of
  being dominated by the background. This is the change that made the world model
  generate reliable goal states (real-vs-generated subgoal gap 7.2 → 1.4 points).

### `action_expert.py`
- `ActionExpert(cfg)`: embeds a noisy `(H,4)` action chunk into tokens, attends
  over `[history; subgoal?; action]` tokens, predicts the action velocity.
  `use_subgoal=False` drops the subgoal tokens (Baseline 2).

### `simple_policy.py`
- `SimplePolicy(cfg)`: history tokens + a language token → transformer → 4 logits,
  cross-entropy loss.

### `cascade_wam.py`
- `CascadeWAM` wires WM + AE with a history buffer, `observe(frame)`,
  `set_instruction(text)`, `step()` → chunk. Oracle subtask derived lexically.
- **Async low-frequency subgoal (π0.7-style)**: the world model runs in a
  background thread and refreshes the subgoal every `subgoal_update_every` steps
  (default 5); the action expert consumes the latest (possibly stale) subgoal and
  never blocks. `async_mode=False` falls back to synchronous low-frequency updates.

---

## 6. `training/`

- `utils.py`: `load_config`, `make_loader` (preloads, single-process),
  `tokenize_batch`, `save/load_checkpoint`, `build_{world_model,action_expert,
  simple_policy}` (reconstruct from checkpoint config), `AverageMeter`.
- `train_simple_policy.py`, `train_world_model.py`, `train_action_expert.py`:
  each is a self-contained AMP training loop (autocast + GradScaler), saving
  `{name}_best.pt` and `{name}_last.pt` every epoch.

**Checkpoint contents**: `model`, `optimizer`, `scheduler`, `epoch`, `config`,
`seed`, `extra`.

---

## 7. `evaluation/`

- `metrics.py`: `summarize` (success rate, avg length, efficiency),
  `object_presence_score` / `classify_object` (template-matching CV parser used to
  score whether a generated subgoal contains the requested shape×fill).
- `evaluate_policy.py`: `SimplePolicyAgent`, `ActionExpertAgent`, `CascadeAgent`
  under a unified `act(history, instruction, subtask) → chunk` interface;
  `run_closed_loop` (receding horizon) and `evaluate_agent`.
- `evaluate_world_model.py`: pixel MSE + target-presence + identity accuracy.
- `ood.py`: `run_ood_benchmark` across the five OOD splits.

---

## 8. `scripts/`

| Script | Purpose |
|---|---|
| `generate_dataset.py` | generate a split (`--episodes --split --output --workers`) |
| `inspect_dataset.py` | statistics + sample visualizations |
| `train.py` | dispatch to a training script (`--model --config`) |
| `evaluate.py` | closed-loop eval (`--model --checkpoint [--world-model] [--ood]`) |
| `run_agent.py` | interactive demo with debug overlay → frames + GIF |
| `sample_subgoal.py` | visualise generated vs real subgoals |
| `run_experiments.py` | full main + OOD suite → `results/results.json` + `report.md` |

---

## 9. Configuration

All YAML files live in `configs/`. Every architectural parameter is configurable:

```yaml
history.frames          # K (observation history length)
action.horizon          # H (chunk length)
action.execute_steps    # K' (receding-horizon prefix)
model.hidden_dim/layers/heads/patch   # DiT size
model.denoise_steps     # flow-matching integration steps
training.batch_size/epochs/learning_rate/deltas
```

The same dict is passed to every model constructor, so changing a value in YAML
changes the architecture without touching code.

---

## 10. Design decisions & pitfalls

1. **Colour removed** (grayscale) → identity is shape×fill only; forces grounding.
2. **Area-normalised shapes** → no "object size" shortcut.
3. **Supersampled rendering** → clean shape edges at 48×48.
4. **Full frames stored** → subgoal horizon is a train-time choice.
5. **Flow matching, not DDPM** → 3–5 Euler steps at inference (fast, matches π0.7).
6. **One-hot action targets** → continuous flow-matching over `(H,4)` vectors,
   `argmax` at inference.
7. **Preloaded dataset, single-process DataLoader** → the `.npz` decompression
   would otherwise dominate; in-memory access is ~10× faster than lazy loading.
8. **Pixel-space WM is the known weak point** — the flow loss is background-dominated,
   so shape/fill identity stays near chance. A latent (VAE) formulation or
   object-weighted loss is the documented fix (see `docs/report.md` §7).

---

## 11. Extending

- **New object type**: add to `SHAPES`/`FILLS`, add a shape function in
  `objects.py`, add its template to `language.py` and `language_encoder.py`.
- **New split**: add a `SplitConfig` to `SPLIT_CONFIGS` in `generator.py` and a
  name to `OOD_SPLITS` in `evaluation/ood.py`.
- **Joint WAM (Phase 2)**: a shared transformer denoising `(G, A)` jointly —
  see `docs/report.md`; not implemented (optional).
- **Async inference**: `CascadeWAM(synchronous=False)` already exposes a
  stale-subgoal buffer; a threaded world-model loop would complete it.
