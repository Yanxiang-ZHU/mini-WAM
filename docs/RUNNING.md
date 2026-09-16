# mini-wam — How to Run It Yourself

Everything runs on a single NVIDIA RTX 5060 Ti (or any CUDA GPU ≥ 8 GB, ideally
16 GB). Below is the exact sequence from a clean checkout.

---

## 0. Prerequisites

- Python 3.10–3.12
- NVIDIA GPU with CUDA 12.8+ (the RTX 50-series needs cu128/cu130)
- `uv` (recommended) or `pip`

---

## 1. Install

```bash
cd "D:/ICpro/Wan Lab/4mock"

# create a virtualenv
uv venv .venv --python 3.12        # or: python -m venv .venv

# install PyTorch with CUDA for Blackwell (RTX 50xx)
uv pip install --python .venv/Scripts/python.exe \
    --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# install the project + deps
uv pip install --python .venv/Scripts/python.exe -e .

# verify CUDA
.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#   -> True NVIDIA GeForce RTX 5060 Ti
```

> On a non-Blackwell GPU (RTX 30/40 series) the same cu128 wheel still works.
> On CPU only, training is possible but slow.

---

## 2. Run the tests

```bash
.venv/Scripts/python.exe -m pytest -q
#   -> 53 passed
```

This validates physics, collision, rendering, semantics, flow matching, and model
forward passes before you spend time on training.

---

## 3. Generate the dataset

```bash
# training set (10,000 episodes, ~3 min on 8 cores)
.venv/Scripts/python.exe scripts/generate_dataset.py --episodes 10000 --split train --output data/train --workers 8

# validation + test (1,000 each)
.venv/Scripts/python.exe scripts/generate_dataset.py --episodes 1000 --split val --output data/val --workers 8
.venv/Scripts/python.exe scripts/generate_dataset.py --episodes 1000 --split test --output data/test --workers 8
```

To sanity-check the data:

```bash
.venv/Scripts/python.exe scripts/inspect_dataset.py --data data/train --out _inspect --viz 6
```

---

## 4. Train the models (in order)

Each writes `checkpoints/{name}_best.pt` and `{name}_last.pt` every epoch. Approx.
times on the RTX 5060 Ti:

| Model | Command | ≈ time |
|---|---|---|
| Simple policy | `python training/train_simple_policy.py --config configs/simple_policy.yaml` | ~20 min |
| World Model (wide) | `python training/train_world_model.py --config configs/world_model_goal_wide.yaml` | ~70 min |
| Action Expert (wide) | `python training/train_action_expert.py --config configs/action_expert_goal_wide.yaml` | ~2 h |
| Action-chunk (no subgoal) | `python training/train_action_expert.py --config configs/action_chunk_policy.yaml` | ~25 min |

> **Recommended (best results)**: the wide-data configs (`world_model_goal_wide.yaml`
> and `action_expert_goal_wide.yaml`) train on 20k episodes with a **wide
> distribution** (1–6 distractors, 1–3 obstacles), using a **goal-state subgoal** +
> **object-weighted loss** + **Phase 6** (generated-subgoal mix). This gives
> **92.5%** closed-loop success (and 86–93% across OOD splits) — the strongest
> result. First generate the wide data:
>
> ```bash
> python scripts/generate_dataset.py --episodes 20000 --split train_wide --output data/train_wide --workers 8
> ```

```bash
.venv/Scripts/python.exe training/train_simple_policy.py --config configs/simple_policy.yaml
.venv/Scripts/python.exe training/train_world_model.py --config configs/world_model.yaml
.venv/Scripts/python.exe training/train_action_expert.py --config configs/action_expert.yaml
.venv/Scripts/python.exe training/train_action_expert.py --config configs/action_chunk_policy.yaml
```

> Tip: add `--epochs N` to override the epoch count, or edit `configs/*.yaml`
> (batch size, hidden dim, layers, patch size, deltas are all configurable).

---

## 5. Evaluate closed-loop

```bash
# simple policy
.venv/Scripts/python.exe scripts/evaluate.py --model simple_policy \
    --checkpoint checkpoints/simple_policy_best.pt --episodes 200

# action-chunk policy (no subgoal)
.venv/Scripts/python.exe scripts/evaluate.py --model action_expert \
    --checkpoint checkpoints/action_chunk_policy_best.pt --episodes 200

# cascade WAM (world model + action expert)
.venv/Scripts/python.exe scripts/evaluate.py --model cascade \
    --checkpoint checkpoints/action_expert_best.pt \
    --world-model checkpoints/world_model_best.pt --episodes 200
```

Run the full suite (main + OOD benchmark) in one shot:

```bash
.venv/Scripts/python.exe scripts/run_experiments.py \
    --simple checkpoints/simple_policy_best.pt \
    --action-expert checkpoints/action_expert_best.pt \
    --action-chunk checkpoints/action_chunk_policy_best.pt \
    --world-model checkpoints/world_model_best.pt \
    --episodes 200
#   -> writes results/results.json and results/report.md
```

---

## 6. Inspect the world model's subgoals

```bash
.venv/Scripts/python.exe scripts/sample_subgoal.py \
    --checkpoint checkpoints/world_model_best.pt --n 8 --out _subgoals
#   -> _subgoals/subgoal_XX.png (history → generated → real)
```

---

## 7. Run the interactive demo

The demo now generates the visual subgoal **asynchronously at low frequency**
(π0.7-style): the world model refreshes the subgoal every `--subgoal-update-every`
steps (default 5) in a background thread, while the action expert keeps using the
latest available subgoal and never blocks.

```bash
# CLI demo — random scene (omit --seed), manual instruction, N frames + GIF
.venv/Scripts/python.exe scripts/run_agent.py \
    --world-model checkpoints/world_model_best.pt \
    --action-expert checkpoints/action_expert_best.pt \
    --instruction "Go to the hollow triangle." \
    --out demo
#   -> demo/step_XXX.png + demo/demo.gif

# omit --instruction to type it interactively; omit --seed for a random layout
.venv/Scripts/python.exe scripts/run_agent.py --out demo2
```

## 8. Web demo (browser UI)

A live browser demo: image on the left, instruction text box on the right, press
Run, and watch the agent navigate (observation + async subgoal + action chunk
streamed live over WebSocket). Each run is a fresh random scene.

```bash
.venv/Scripts/python.exe scripts/web_demo.py \
    --world-model checkpoints/world_model_best.pt \
    --action-expert checkpoints/action_expert_best.pt \
    --port 8000
#   -> open http://localhost:8000
```

The web UI has preset example buttons and a 🎲 random-scene button. The subgoal
panel stays fixed for several steps (low-frequency async update), then jumps.

## 9. Human control (optional)

```bash
uv pip install --python .venv/Scripts/python.exe pygame
.venv/Scripts/python.exe -m game.keyboard   # WASD to move, ESC to quit
```

## 10. View the report

Open `project_report.html` in a browser (it references the generated images and
the demo GIF relative to the project folder), or read `docs/report.md` and
`docs/ENGINEERING.md`.

---

## Troubleshooting

- **`No module named torch` / CUDA not available** — re-run the cu128 install in
  step 1 and confirm `torch.cuda.is_available()`.
- **OOM** — reduce `training.batch_size` in the YAML, or lower `hidden_dim` /
  `layers`. The default models use ~6 GB VRAM at batch 128.
- **Slow training** — the dataset is preloaded into RAM; if you have less RAM,
  set `preload=False` in `training/utils.py::make_loader`.
- **`imageio` missing when making the demo GIF** — the demo still writes PNG
  frames; install `imageio` or build the GIF with PIL (see `scripts/run_agent.py`).
