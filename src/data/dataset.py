"""PyTorch Dataset over the split manifests, plus the train/eval transforms.

Augmentation lives only in `train_transform`; validation, test and inference
all share `eval_transform`, so the model is never evaluated on augmented data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.utils import PROJECT_ROOT


def train_transform(cfg: dict) -> T.Compose:
    im = cfg["image"]
    rotation = im.get("rotation", 15)
    return T.Compose([
        # Random crop covering 60-100% of the photo: flowers are photographed at
        # many distances, so scale jitter teaches the model zoom invariance.
        T.RandomResizedCrop(im["size"], scale=(0.6, 1.0), ratio=(3 / 4, 4 / 3)),
        T.RandomHorizontalFlip(),
        *([T.RandomRotation(rotation)] if rotation else []),
        # Mild colour jitter only: hue is a key cue (yellow sunflower vs pink rose),
        # so hue shift is kept very small to avoid creating label noise.
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
        T.ToTensor(),
        T.Normalize(im["mean"], im["std"]),
    ])


def eval_transform(cfg: dict) -> T.Compose:
    im = cfg["image"]
    return T.Compose([
        T.Resize(im["resize_short"]),
        T.CenterCrop(im["size"]),
        T.ToTensor(),
        T.Normalize(im["mean"], im["std"]),
    ])


def denormalize(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Undo normalisation for display; works on (C,H,W) or (N,C,H,W)."""
    mean = torch.tensor(cfg["image"]["mean"]).view(-1, 1, 1)
    std = torch.tensor(cfg["image"]["std"]).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)


class ManifestDataset(Dataset):
    def __init__(self, csv_path: str | Path, classes: list[str], transform=None):
        self.df = pd.read_csv(csv_path)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        with Image.open(PROJECT_ROOT / row["path"]) as im:
            img = im.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[row["label"]]


def get_classes(cfg: dict) -> list[str]:
    df = pd.read_csv(cfg["paths"]["splits_dir"] / "train.csv")
    return sorted(df["label"].unique())


def make_loaders(cfg: dict) -> tuple[dict[str, DataLoader], list[str]]:
    classes = get_classes(cfg)
    sd, tc = cfg["paths"]["splits_dir"], cfg["train"]
    datasets = {
        "train": ManifestDataset(sd / "train.csv", classes, train_transform(cfg)),
        "val": ManifestDataset(sd / "val.csv", classes, eval_transform(cfg)),
        "test": ManifestDataset(sd / "test.csv", classes, eval_transform(cfg)),
    }
    g = torch.Generator().manual_seed(cfg["seed"])
    loaders = {
        name: DataLoader(
            ds,
            batch_size=tc["batch_size"],
            shuffle=(name == "train"),
            num_workers=tc["num_workers"],
            persistent_workers=tc["num_workers"] > 0,
            generator=g if name == "train" else None,
        )
        for name, ds in datasets.items()
    }
    return loaders, classes
