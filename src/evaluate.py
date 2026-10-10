"""Evaluate a checkpoint on a split and write every metric and figure.

Outputs (under reports/eval/<run>_<split>/ and reports/figures/):
  metrics.json              accuracy, macro/weighted P/R/F1, ECE, per-class report
  predictions.csv           one row per image: truth, prediction, confidence,
                            per-class probabilities and image-quality features
  confusion_<run>_<split>.png       counts + row-normalised
  reliability_<run>_<split>.png     calibration diagram
  misclassified_<run>_<split>.png   gallery of errors, most confident first

The test split is only evaluated once per final model; all development
decisions are made on --split val.

Usage:
  python -m src.evaluate --model models/mobilenet_v2.pt --split val
  python -m src.evaluate --model models/mobilenet_v2.pt --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from src.calibrate import expected_calibration_error
from src.data.dataset import ManifestDataset, eval_transform
from src.models import build_model
from src.utils import PROJECT_ROOT, load_config, save_json


def image_quality(path: Path) -> dict:
    """Cheap quality signals used to test the 'poor image quality' hypothesis."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return {
        "blur_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),  # low = blurry
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "min_side": int(min(gray.shape)),
    }


def plot_confusion(cm: np.ndarray, classes: list[str], path: Path, title: str) -> None:
    cm_norm = cm / cm.sum(1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes,
                ax=axes[0], cbar=False)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes,
                ax=axes[1], vmin=0, vmax=1, cbar=False)
    axes[0].set_title("Counts")
    axes[1].set_title("Row-normalised (recall per true class)")
    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_reliability(conf: np.ndarray, correct: np.ndarray, path: Path, title: str, n_bins: int = 10) -> None:
    edges = np.linspace(0, 1, n_bins + 1)
    mids, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            mids.append(conf[m].mean())
            accs.append(correct[m].mean())
            counts.append(m.sum())
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.plot(mids, accs, "o-", label="model")
    for x, y, n in zip(mids, accs, counts):
        ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Accuracy in bin")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_misclassified(df: pd.DataFrame, path: Path, title: str, n: int = 20) -> None:
    wrong = df[~df["correct"]].sort_values("confidence", ascending=False).head(n)
    if wrong.empty:
        return
    cols = 5
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.3))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (_, r) in zip(np.atleast_1d(axes).ravel(), wrong.iterrows()):
        with Image.open(PROJECT_ROOT / r["path"]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"true: {r['true']}\npred: {r['pred']} ({r['confidence']:.0%})", fontsize=9, color="crimson")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


@torch.no_grad()
def predict_split(model, loader, temperature: float) -> np.ndarray:
    model.eval()
    return torch.cat([torch.softmax(model(x) / temperature, 1) for x, _ in loader]).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    cfg = load_config()
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg["image"] = ckpt.get("image", cfg["image"])  # preprocessing travels with the model
    classes = ckpt["classes"]
    run = Path(args.model).stem
    model = build_model(ckpt["arch"], len(classes), ckpt.get("dropout", 0.3), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    temperature = float(ckpt.get("temperature", 1.0))

    ds = ManifestDataset(cfg["paths"]["splits_dir"] / f"{args.split}.csv", classes, eval_transform(cfg))
    loader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=cfg["train"]["num_workers"])
    probs = predict_split(model, loader, temperature)

    y_true = ds.df["label"].map(ds.class_to_idx).to_numpy()
    y_pred = probs.argmax(1)
    conf = probs.max(1)

    df = ds.df[["path", "label"]].rename(columns={"label": "true"}).copy()
    df["pred"] = [classes[i] for i in y_pred]
    df["confidence"] = conf
    df["correct"] = y_true == y_pred
    for i, c in enumerate(classes):
        df[f"p_{c}"] = probs[:, i]
    quality = pd.DataFrame([image_quality(PROJECT_ROOT / p) for p in df["path"]])
    df = pd.concat([df.reset_index(drop=True), quality], axis=1)

    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    thr = cfg["inference"]["uncertain_threshold"]
    confident = conf >= thr
    metrics = {
        "run": run,
        "arch": ckpt["arch"],
        "split": args.split,
        "n": int(len(df)),
        "temperature": temperature,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro": {"precision": p_m, "recall": r_m, "f1": f_m},
        "weighted": {"precision": p_w, "recall": r_w, "f1": f_w},
        "ece": expected_calibration_error(probs, y_true),
        "uncertain_threshold": thr,
        "coverage_at_threshold": float(confident.mean()),
        "accuracy_when_confident": float((y_true == y_pred)[confident].mean()) if confident.any() else None,
        "per_class": classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }

    out_dir = cfg["paths"]["reports_dir"] / "eval" / f"{run}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, out_dir / "metrics.json")
    df.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0))

    fig_dir = cfg["paths"]["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{run} - {args.split} set"
    plot_confusion(cm, classes, fig_dir / f"confusion_{run}_{args.split}.png", f"Confusion matrix - {tag}")
    plot_reliability(conf, df["correct"].to_numpy(), fig_dir / f"reliability_{run}_{args.split}.png",
                     f"Reliability - {tag}\nECE = {metrics['ece']:.3f}")
    plot_misclassified(df, fig_dir / f"misclassified_{run}_{args.split}.png",
                       f"Most confident errors - {tag}")

    print(classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0))
    print(f"accuracy {metrics['accuracy']:.4f} | macro F1 {f_m:.4f} | ECE {metrics['ece']:.4f} | "
          f"coverage@{thr:.2f} {metrics['coverage_at_threshold']:.3f} -> acc {metrics['accuracy_when_confident']:.4f}")
    print(f"[evaluate] wrote {out_dir}")


if __name__ == "__main__":
    main()
