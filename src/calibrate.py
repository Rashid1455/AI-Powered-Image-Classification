"""Temperature scaling (Guo et al., 2017) fitted on the validation set.

A single scalar T divides the logits before softmax. It does not change any
prediction (argmax is unchanged), only how confident the probabilities are,
so the app's confidence score means what it says. T is stored in the
checkpoint and applied automatically by the Predictor and evaluator.

Usage:  python -m src.calibrate --model models/mobilenet_v2.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import nn

from src.data.dataset import ManifestDataset, eval_transform
from src.models import build_model
from src.utils import load_config


@torch.no_grad()
def collect_logits(model, loader) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x))
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)  # optimise log T so T stays positive
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    cfg = load_config()
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg["image"] = ckpt.get("image", cfg["image"])  # preprocessing travels with the model
    model =build_model(ckpt["arch"], len(ckpt["classes"]), ckpt.get("dropout", 0.3), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    ds = ManifestDataset(cfg["paths"]["splits_dir"] / "val.csv", ckpt["classes"], eval_transform(cfg))
    loader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=cfg["train"]["num_workers"])

    logits, labels = collect_logits(model, loader)
    t = fit_temperature(logits, labels)
    y = labels.numpy()
    before = expected_calibration_error(torch.softmax(logits, 1).numpy(), y)
    after = expected_calibration_error(torch.softmax(logits / t, 1).numpy(), y)
    print(f"[calibrate] T = {t:.3f} | val ECE {before:.4f} -> {after:.4f}")

    ckpt["temperature"] = t
    ckpt["val_ece"] = {"before": before, "after": after}
    torch.save(ckpt, args.model)
    print(f"[calibrate] temperature written to {args.model}")


if __name__ == "__main__":
    main()
