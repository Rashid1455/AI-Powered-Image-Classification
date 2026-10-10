"""Train a classifier.

Transfer models use a two-stage schedule:
  Stage 1  backbone frozen, train the head at `head_lr`
  Stage 2  unfreeze the top `unfreeze_fraction` of backbone blocks, fine-tune at `finetune_lr`
The custom CNN trains end-to-end in a single stage.

Early stopping monitors validation loss; the checkpoint written to disk is
always the best validation-loss epoch, never simply the last one.

Usage:
  python -m src.train                       # arch from config.yaml
  python -m src.train --arch efficientnet_b0
  python -m src.train --arch custom_cnn --run-name baseline_cnn
"""
from __future__ import annotations

import argparse
import copy
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.data.dataset import make_loaders
from src.models import build_model, count_params, freeze_backbone, set_frozen_bn_eval, unfreeze_top
from src.utils import get_device, load_config, save_json, set_seed


def class_weights(loader, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised so the mean weight is 1."""
    labels = loader.dataset.df["label"].map(loader.dataset.class_to_idx).to_numpy()
    counts = np.bincount(labels, minlength=num_classes)
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    if train:
        set_frozen_bn_eval(model)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


class EarlyStopper:
    def __init__(self, patience: int, on_improve=None):
        self.patience, self.best, self.bad_epochs = patience, float("inf"), 0
        self.best_state, self.best_epoch = None, -1
        self.on_improve = on_improve  # called with (epoch, val_loss) after each new best

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """Record the epoch; return True when training should stop."""
        if val_loss < self.best - 1e-4:
            self.best, self.bad_epochs, self.best_epoch = val_loss, 0, epoch
            self.best_state = copy.deepcopy(model.state_dict())
            if self.on_improve:
                self.on_improve(epoch, val_loss)
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def train_stage(name, model, loaders, criterion, optimizer, epochs, device, stopper, history, epoch0):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=2)
    total, trainable = count_params(model)
    print(f"\n=== {name}: {trainable:,} / {total:,} params trainable ===")
    stopper.bad_epochs = 0  # each stage gets its own patience budget
    epoch = epoch0
    for epoch in range(epoch0, epoch0 + epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step(va_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append(dict(epoch=epoch + 1, stage=name, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, lr=lr, seconds=time.time() - t0))
        stopper.val_acc = va_acc
        stop = stopper.step(va_loss, model, epoch + 1)
        flag = "*" if stopper.best_epoch == epoch + 1 else " "
        print(f"[{name}] ep {epoch + 1:3d} | train {tr_loss:.4f} / {tr_acc:.3f} | "
              f"val {va_loss:.4f} / {va_acc:.3f} {flag}| lr {lr:.1e} | {time.time() - t0:.0f}s")
        if stop:
            print(f"[{name}] early stop: no val-loss improvement for {stopper.patience} epochs")
            break
    return epoch + 1


def plot_history(history: list[dict], path, title: str) -> None:
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, label in [(axes[0], "acc", "Accuracy"), (axes[1], "loss", "Loss")]:
        ax.plot(ep, [h[f"train_{key}"] for h in history], "o-", ms=3, label="train")
        ax.plot(ep, [h[f"val_{key}"] for h in history], "o-", ms=3, label="validation")
        stage_change = [h["epoch"] for i, h in enumerate(history[1:], 1) if h["stage"] != history[i - 1]["stage"]]
        for s in stage_change:
            ax.axvline(s - 0.5, color="grey", ls="--", lw=1)
            ax.text(s - 0.4, ax.get_ylim()[1], " fine-tune", va="top", fontsize=8, color="grey")
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    best = min(history, key=lambda h: h["val_loss"])
    axes[1].scatter([best["epoch"]], [best["val_loss"]], s=80, facecolors="none", edgecolors="red",
                    zorder=5, label="best (checkpoint)")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch")
    ap.add_argument("--run-name")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--image-size", type=int, help="override input size (e.g. 128 for the scratch CNN)")
    ap.add_argument("--no-rotation", action="store_true", help="augmentation ablation: drop RandomRotation")
    args = ap.parse_args()

    cfg = load_config()
    if args.image_size:
        cfg["image"]["size"] = args.image_size
        cfg["image"]["resize_short"] = round(args.image_size * 256 / 224)
    cfg["image"]["rotation"] = 0 if args.no_rotation else cfg["image"].get("rotation", 15)
    tc = cfg["train"]
    arch = args.arch or tc["arch"]
    run_name = args.run_name or arch
    set_seed(cfg["seed"])
    device = get_device()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    print(f"[train] arch={arch} run={run_name} device={device}")

    loaders, classes = make_loaders(cfg)
    model = build_model(arch, len(classes), tc["dropout"]).to(device)

    use_weights = tc["class_weighting"] and not args.no_class_weights
    weights = class_weights(loaders["train"], len(classes)).to(device) if use_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=tc["label_smoothing"])
    print(f"[train] class weights: {None if weights is None else weights.cpu().numpy().round(3)}")

    models_dir = cfg["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / f"{run_name}.pt"

    def save_checkpoint(epoch: int, val_loss: float) -> None:
        # Written on every new best epoch, so an interrupted CPU run still leaves a usable model.
        torch.save({
            "arch": arch,
            "classes": classes,
            "state_dict": stopper.best_state,
            "image": cfg["image"],
            "dropout": tc["dropout"],
            "best_epoch": epoch,
            "val_loss": val_loss,
            "val_acc": stopper.val_acc,
        }, ckpt_path)

    stopper = EarlyStopper(tc["early_stopping_patience"], on_improve=save_checkpoint)
    history: list[dict] = []
    t_start = time.time()

    if arch == "custom_cnn":
        opt = torch.optim.AdamW(model.parameters(), lr=tc["scratch_lr"], weight_decay=tc["weight_decay"])
        stopper.patience = tc["early_stopping_patience"] + 2  # scratch training is noisier
        train_stage("scratch", model, loaders, criterion, opt, tc["scratch_epochs"], device, stopper, history, 0)
    else:
        freeze_backbone(model)
        opt = torch.optim.AdamW(model.head.parameters(), lr=tc["head_lr"], weight_decay=tc["weight_decay"])
        ep = train_stage("head", model, loaders, criterion, opt, tc["head_epochs"], device, stopper, history, 0)

        # Fine-tuning starts from the best head checkpoint, not the last epoch.
        model.load_state_dict(stopper.best_state)
        n_blocks = unfreeze_top(model, tc["unfreeze_fraction"])
        print(f"[train] unfroze top {n_blocks} backbone blocks")
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=tc["finetune_lr"], weight_decay=tc["weight_decay"])
        train_stage("finetune", model, loaders, criterion, opt, tc["finetune_epochs"], device, stopper, history, ep)

    model.load_state_dict(stopper.best_state)
    best = next(h for h in history if h["epoch"] == stopper.best_epoch)
    minutes = (time.time() - t_start) / 60
    print(f"\n[train] best epoch {best['epoch']} ({best['stage']}): "
          f"val_loss {best['val_loss']:.4f} val_acc {best['val_acc']:.3f} | {minutes:.1f} min total")

    print(f"[train] saved {ckpt_path} (best epoch {best['epoch']})")

    run_dir = cfg["paths"]["reports_dir"] / "runs" / run_name
    save_json({
        "arch": arch,
        "run_name": run_name,
        "class_weights": None if weights is None else weights.cpu().tolist(),
        "train_config": tc,
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "best_val_acc": best["val_acc"],
        "train_minutes": round(minutes, 1),
        "params_total": count_params(model)[0],
        "history": history,
    }, run_dir / "history.json")
    plot_history(history, cfg["paths"]["figures_dir"] / f"curves_{run_name}.png", f"Training curves - {run_name}")


if __name__ == "__main__":
    main()
