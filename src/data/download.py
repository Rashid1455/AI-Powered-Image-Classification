"""Download and extract the dataset archive.

Usage:  python -m src.data.download
"""
from __future__ import annotations

import tarfile
import urllib.request
from datetime import date
from pathlib import Path

from src.utils import load_config, save_json


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[download] archive already present: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] fetching {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"[download] saved {dest.stat().st_size / 1e6:.1f} MB")


def extract(archive: Path, out_dir: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        # Only accept regular files/dirs that stay inside out_dir (no path traversal).
        members = []
        for m in tar.getmembers():
            target = (out_dir / m.name).resolve()
            if not str(target).startswith(str(out_dir.resolve())):
                raise RuntimeError(f"Unsafe path in archive: {m.name}")
            if m.isfile() or m.isdir():
                members.append(m)
        tar.extractall(out_dir, members=members)


def main() -> None:
    cfg = load_config()
    raw_dir = cfg["paths"]["raw_dir"]
    archive = raw_dir / cfg["dataset"]["archive"]
    download(cfg["dataset"]["url"], archive)

    image_dir = cfg["paths"]["image_dir"]
    if image_dir.exists() and any(image_dir.iterdir()):
        print(f"[download] already extracted: {image_dir}")
    else:
        print("[download] extracting ...")
        extract(archive, raw_dir)

    classes = sorted(p.name for p in image_dir.iterdir() if p.is_dir())
    counts = {c: len(list((image_dir / c).glob("*"))) for c in classes}
    meta = {
        "name": cfg["dataset"]["name"],
        "url": cfg["dataset"]["url"],
        "licence": cfg["dataset"]["licence"],
        "accessed": date.today().isoformat(),
        "classes": classes,
        "counts": counts,
        "total": sum(counts.values()),
    }
    save_json(meta, cfg["paths"]["reports_dir"] / "dataset_meta.json")
    print(f"[download] {meta['total']} images across {len(classes)} classes: {counts}")


if __name__ == "__main__":
    main()
