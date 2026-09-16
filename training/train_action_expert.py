"""Train the Action Expert (flow-matching action chunk generator).

Phase 5 uses real future frames as subgoals.  Phase 6 mixes in generated
subgoals (``generated_subgoal_ratio``) to close the train/test mismatch.
"""

import argparse
import os
import time

import torch

from training.utils import (load_config, make_loader, tokenize_batch,
                            save_checkpoint, AverageMeter)
from models.action_expert import ActionExpert


def train_one_epoch(model, loader, opt, scaler, device, amp):
    model.train()
    loss_m = AverageMeter()
    for batch in loader:
        history = batch["history"].to(device)
        subgoal = batch["subgoal"].unsqueeze(1).to(device)
        chunk = batch["action_onehot"].to(device)
        lang_ids = tokenize_batch(batch["instruction"], device)
        subtask_ids = tokenize_batch(batch["subtask"], device)
        meta = batch["metadata"]

        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = model.loss(chunk, history, subgoal, lang_ids, subtask_ids, meta)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        loss_m.update(loss.item(), history.shape[0])
    return loss_m.avg


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_m = AverageMeter()
    for batch in loader:
        history = batch["history"].to(device)
        subgoal = batch["subgoal"].unsqueeze(1).to(device)
        chunk = batch["action_onehot"].to(device)
        lang_ids = tokenize_batch(batch["instruction"], device)
        subtask_ids = tokenize_batch(batch["subtask"], device)
        meta = batch["metadata"]
        loss = model.loss(chunk, history, subgoal, lang_ids, subtask_ids, meta)
        loss_m.update(loss.item(), history.shape[0])
    return loss_m.avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/action_expert.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.device:
        cfg["device"] = args.device
    device = cfg["device"]
    seed = cfg["seed"]
    torch.manual_seed(seed)

    t = cfg["training"]
    hist = cfg["history"]["frames"]
    H = cfg["action"]["horizon"]
    model_cfg = {**cfg["model"], "history": hist, "action_horizon": H,
                 "num_actions": cfg["action"]["num_actions"]}
    model = ActionExpert(model_cfg).to(device)
    print(f"action expert params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    train_loader = make_loader(t["data"], t["batch_size"], K=hist, H=H,
                               deltas=tuple(t["deltas"]), workers=t["workers"],
                               max_episodes=t.get("max_episodes"))
    val_loader = make_loader(t["val_data"], t["batch_size"], K=hist, H=H,
                             deltas=tuple(t["deltas"]), workers=t["workers"],
                             shuffle=False, max_episodes=t.get("max_episodes"))

    opt = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"],
                            weight_decay=t["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=t.get("mixed_precision", True))
    amp = t.get("mixed_precision", True)

    best_loss = float("inf")
    ckpt_dir = cfg["checkpoint"]["dir"]
    name = cfg["checkpoint"]["name"]
    os.makedirs(ckpt_dir, exist_ok=True)

    for ep in range(t["epochs"]):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, opt, scaler, device, amp)
        val_loss = evaluate(model, val_loader, device)
        print(f"epoch {ep+1}/{t['epochs']} loss={loss:.4f} val_loss={val_loss:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(os.path.join(ckpt_dir, f"{name}_best.pt"),
                            model, opt, None, ep + 1, cfg, seed)
        if (ep + 1) % t["save_every"] == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"{name}_last.pt"),
                            model, opt, None, ep + 1, cfg, seed)

    print(f"best val flow-matching loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
