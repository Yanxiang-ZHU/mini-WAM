"""Top-level training dispatcher.

Usage:
    python scripts/train.py --model simple_policy --config configs/simple_policy.yaml
    python scripts/train.py --model world_model --config configs/world_model.yaml
    python scripts/train.py --model action_expert --config configs/action_expert.yaml
"""

import argparse
import subprocess
import sys
import os

SCRIPTS = {
    "simple_policy": "train_simple_policy.py",
    "world_model": "train_world_model.py",
    "action_expert": "train_action_expert.py",
    "action_chunk": "train_action_expert.py",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SCRIPTS))
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    args, rest = ap.parse_known_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(here, "training", SCRIPTS[args.model])
    cmd = [sys.executable, script, "--config", args.config]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    if args.device:
        cmd += ["--device", args.device]
    cmd += rest
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
