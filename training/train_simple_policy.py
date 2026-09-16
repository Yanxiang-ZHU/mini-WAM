"""Train the simple policy baseline (observation -> single WASD classifier)."""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.utils import (load_config, make_loader, tokenize_batch,
                            save_checkpoint, AverageMeter)
from models.simple_policy import SimplePolicy


def train_one_epoch(model, loader, opt, scaler, device, amp):
    model.train()
    loss_m = AverageMeter()
    for batch in loader:
        history = batch["history"].to(device)
        lang_ids = tokenize_batch(batch["instruction"], device)
        action = batch["action_chunk"][:, 0].to(device)  # single action at t

        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = model.loss(history, action, lang_ids)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        loss_m.update(loss.item(), history.shape[0])
    return loss_m.avg


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        history = batch["history"].to(device)
        lang_ids = tokenize_batch(batch["instruction"], device)
        action = batch["action_chunk"][:, 0].to(device)
        logits = model(history, lang_ids)
        correct += (logits.argmax(-1) == action).sum().item()
        total += action.shape[0]
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/simple_policy.yaml")
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
    H = cfg["action"].get("horizon", 8) if "action" in cfg else 8
    model_cfg = {**cfg["model"], "history": hist}
    model = SimplePolicy(model_cfg).to(device)
    print(f"simple policy params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

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

    best_acc = 0.0
    ckpt_dir = cfg["checkpoint"]["dir"]
    name = cfg["checkpoint"]["name"]
    os.makedirs(ckpt_dir, exist_ok=True)

    for ep in range(t["epochs"]):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, opt, scaler, device, amp)
        acc = evaluate(model, val_loader, device)
        print(f"epoch {ep+1}/{t['epochs']} loss={loss:.4f} val_acc={acc:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(os.path.join(ckpt_dir, f"{name}_best.pt"),
                            model, opt, None, ep + 1, cfg, seed)
        if (ep + 1) % t["save_every"] == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"{name}_last.pt"),
                            model, opt, None, ep + 1, cfg, seed)

    print(f"best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
