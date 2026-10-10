#!/usr/bin/env bash
# Reproduce every experiment end to end (CPU: several hours; GPU: minutes).
# Usage:  bash scripts/run_all.sh        (from the project root, venv active)
set -euo pipefail
PY=${PYTHON:-python}

$PY -m src.data.download
$PY -m src.data.clean
$PY -m src.data.split

# Architecture comparison (identical data, augmentation and schedule)
$PY -m src.train --arch mobilenet_v2
$PY -m src.train --arch efficientnet_b0
$PY -m src.train --arch custom_cnn --image-size 128     # from-scratch baseline

# Calibrate + evaluate on validation (development decisions are made here)
for m in mobilenet_v2 efficientnet_b0 custom_cnn; do
  $PY -m src.calibrate --model models/$m.pt
  $PY -m src.evaluate --model models/$m.pt --split val
done
