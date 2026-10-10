"""Integrity scan and cleaning of the raw image folder.

Checks, in order:
  1. unreadable / truncated files      -> removed
  2. images below `dataset.min_side`    -> removed
  3. exact duplicates (MD5)             -> keep one copy; if the copies carry
                                           *different* labels, drop all of them
  4. manual exclusions (data/exclusions.csv, filled in during error analysis)
  5. near-duplicates (perceptual hash)  -> kept, but assigned a shared group id
                                           so the splitter keeps them together
                                           (prevents train/test leakage)

Outputs data/processed/manifest.csv and reports/data_cleaning.json.

Usage:  python -m src.data.clean
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

from src.utils import PROJECT_ROOT, load_config, save_json

ImageFile.LOAD_TRUNCATED_IMAGES = False  # truncated files must raise, not load silently
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NEAR_DUP_MAX_HAMMING = 4  # pHash distance (out of 64 bits) treated as "same picture"


def scan_file(path: Path) -> dict:
    rec = {"path": path.relative_to(PROJECT_ROOT).as_posix(), "label": path.parent.name}
    try:
        with Image.open(path) as im:
            im.verify()  # header / structure check
        with Image.open(path) as im:
            im.load()  # full decode catches truncation
            rec.update(width=im.width, height=im.height, mode=im.mode)
            rec["phash"] = str(imagehash.phash(im.convert("RGB")))
        rec["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001 - any decode failure means unusable
        rec["status"] = f"unreadable: {type(e).__name__}"
    return rec


def near_duplicate_groups(phashes: list[str]) -> np.ndarray:
    """Union-find over pairs whose pHash Hamming distance <= threshold."""
    bits = np.array(
        [np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)) for h in phashes],
        dtype=bool,
    )
    n = len(bits)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        dist = (bits[i + 1:] != bits[i]).sum(axis=1)
        for j in np.nonzero(dist <= NEAR_DUP_MAX_HAMMING)[0] + i + 1:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    return np.array([find(i) for i in range(n)])


def main() -> None:
    cfg = load_config()
    image_dir: Path = cfg["paths"]["image_dir"]
    files = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    print(f"[clean] scanning {len(files)} files")
    df = pd.DataFrame([scan_file(p) for p in tqdm(files)])
    removed: dict[str, list[str]] = defaultdict(list)

    # 1. unreadable
    bad = df["status"] != "ok"
    removed["unreadable_or_truncated"] = df.loc[bad, "path"].tolist()
    df = df[~bad].copy()

    # 2. too small
    min_side = cfg["dataset"]["min_side"]
    small = (df["width"] < min_side) | (df["height"] < min_side)
    removed["too_small"] = df.loc[small, "path"].tolist()
    df = df[~small].copy()

    # 3. exact duplicates
    conflict_md5 = df.groupby("md5")["label"].nunique()
    conflict_md5 = set(conflict_md5[conflict_md5 > 1].index)
    is_conflict = df["md5"].isin(conflict_md5)
    removed["duplicate_label_conflict"] = df.loc[is_conflict, "path"].tolist()
    df = df[~is_conflict].copy()
    is_dup = df.duplicated("md5", keep="first")
    removed["exact_duplicate"] = df.loc[is_dup, "path"].tolist()
    df = df[~is_dup].copy()

    # 4. manual exclusions (mislabelled / not a flower), added after inspection
    excl_file = PROJECT_ROOT / "data" / "exclusions.csv"
    if excl_file.exists():
        excl = pd.read_csv(excl_file)
        is_excl = df["path"].isin(set(excl["path"]))
        removed["manual_exclusion"] = df.loc[is_excl, "path"].tolist()
        df = df[~is_excl].copy()

    # 5. near-duplicate grouping (kept, but split together)
    df = df.reset_index(drop=True)
    df["group"] = near_duplicate_groups(df["phash"].tolist())
    group_sizes = df["group"].map(df["group"].value_counts())
    near_dup_groups = df[group_sizes > 1].groupby("group")
    cross_label = [g for g, d in near_dup_groups if d["label"].nunique() > 1]

    out = cfg["paths"]["clean_dir"] / "manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["status"]).to_csv(out, index=False)

    report = {
        "files_scanned": len(files),
        "files_kept": len(df),
        "removed_counts": {k: len(v) for k, v in removed.items()},
        "removed_total": sum(len(v) for v in removed.values()),
        "removed_files": dict(removed),
        "near_duplicate_groups": int(near_dup_groups.ngroups),
        "images_in_near_duplicate_groups": int((group_sizes > 1).sum()),
        "near_duplicate_groups_with_mixed_labels": [
            df.loc[df["group"] == g, ["path", "label"]].to_dict("records") for g in cross_label
        ],
        "colour_modes": dict(Counter(df["mode"])),
        "class_counts": df["label"].value_counts().sort_index().to_dict(),
    }
    save_json(report, cfg["paths"]["reports_dir"] / "data_cleaning.json")
    print(f"[clean] kept {len(df)} / {len(files)}; removed {report['removed_counts']}")
    print(f"[clean] near-duplicate groups: {report['near_duplicate_groups']} "
          f"({report['images_in_near_duplicate_groups']} images), "
          f"mixed-label groups: {len(cross_label)}")


if __name__ == "__main__":
    main()
