"""Stratified, group-aware train/val/test split with a fixed seed.

Near-duplicate groups from the cleaning step are never divided across
splits, so the test set cannot contain a near-copy of a training image.

Usage:  python -m src.data.split
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils import load_config, save_json


def group_split(df: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off approximately `frac` of rows, stratified by label, respecting groups."""
    n_splits = max(2, round(1 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, part_idx = next(sgkf.split(df, df["label"], groups=df["group"]))
    return df.iloc[rest_idx], df.iloc[part_idx]


def main() -> None:
    cfg = load_config()
    seed, ratios = cfg["seed"], cfg["split"]
    df = pd.read_csv(cfg["paths"]["clean_dir"] / "manifest.csv")

    rest, test = group_split(df, ratios["test"], seed)
    val_frac_of_rest = ratios["val"] / (ratios["train"] + ratios["val"])
    train, val = group_split(rest, val_frac_of_rest, seed)

    # Leakage guards: no file and no near-duplicate group may appear in two splits.
    for a, b in [(train, val), (train, test), (val, test)]:
        assert not set(a["path"]) & set(b["path"]), "file appears in two splits"
        assert not set(a["group"]) & set(b["group"]), "near-duplicate group crosses splits"

    out_dir = cfg["paths"]["splits_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part = part.sort_values("path")
        part.to_csv(out_dir / f"{name}.csv", index=False)
        summary[name] = {
            "n": len(part),
            "fraction": round(len(part) / len(df), 3),
            "class_counts": part["label"].value_counts().sort_index().to_dict(),
        }
    summary["seed"] = seed
    save_json(summary, cfg["paths"]["reports_dir"] / "split_summary.json")
    for name in ("train", "val", "test"):
        print(f"[split] {name:5s} n={summary[name]['n']:5d} ({summary[name]['fraction']:.0%})")


if __name__ == "__main__":
    main()
