"""Inference: load a checkpoint and classify images.

The Predictor uses exactly the same eval transform as validation/test, so
there is no train/serve preprocessing skew.

Usage:
  python -m src.predict path/to/image.jpg
  python -m src.predict path/to/folder --model models/mobilenet_v2.pt --csv out.csv
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

from src.data.dataset import eval_transform
from src.models import build_model
from src.utils import PROJECT_ROOT, load_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Prediction:
    label: str                 # top-1 class, or "uncertain" when below threshold
    top_class: str             # top-1 class regardless of threshold
    confidence: float          # top-1 softmax probability
    uncertain: bool
    top_k: list[tuple[str, float]] = field(default_factory=list)
    probabilities: dict[str, float] = field(default_factory=dict)


class Predictor:
    def __init__(self, checkpoint: str | Path | None = None, threshold: float | None = None, device: str = "cpu"):
        cfg = load_config()
        checkpoint = Path(checkpoint) if checkpoint else PROJECT_ROOT / "models" / f"{cfg['train']['arch']}.pt"
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self.arch = ckpt["arch"]
        self.classes: list[str] = ckpt["classes"]
        self.model = build_model(self.arch, len(self.classes), ckpt.get("dropout", 0.3), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(device)
        self.device = device
        self.temperature = float(ckpt.get("temperature", 1.0))  # from src.calibrate
        cfg["image"] = ckpt.get("image", cfg["image"])  # preprocessing travels with the model
        self.cfg = cfg
        self.transform = eval_transform(cfg)
        self.threshold = cfg["inference"]["uncertain_threshold"] if threshold is None else threshold
        self.top_k = cfg["inference"]["top_k"]

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, img: Image.Image) -> Prediction:
        logits = self.model(self.preprocess(img)) / self.temperature
        probs = torch.softmax(logits, dim=1)[0].cpu()
        conf, idx = probs.topk(min(self.top_k, len(self.classes)))
        top_k = [(self.classes[i], float(c)) for c, i in zip(conf, idx)]
        top_class, top_conf = top_k[0]
        uncertain = top_conf < self.threshold
        return Prediction(
            label="uncertain" if uncertain else top_class,
            top_class=top_class,
            confidence=top_conf,
            uncertain=uncertain,
            top_k=top_k,
            probabilities={c: float(p) for c, p in zip(self.classes, probs)},
        )

    def predict_path(self, path: str | Path) -> Prediction:
        with Image.open(path) as im:
            return self.predict(im)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="image file or folder of images")
    ap.add_argument("--model", help="checkpoint path (default: models/<config arch>.pt)")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--csv", help="write folder results to this CSV")
    args = ap.parse_args()

    predictor = Predictor(args.model, args.threshold)
    src = Path(args.input)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS) if src.is_dir() else [src]
    rows = []
    for f in files:
        p = predictor.predict_path(f)
        topk = ", ".join(f"{c} {s:.1%}" for c, s in p.top_k)
        print(f"{f.name:40s} -> {p.label:12s} ({p.confidence:.1%})  [{topk}]")
        rows.append({"file": str(f), "prediction": p.label, "top_class": p.top_class, "confidence": p.confidence})
    if args.csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
