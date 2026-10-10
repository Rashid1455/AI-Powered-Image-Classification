# Model checkpoints

Each `*.pt` file is a PyTorch dictionary saved by `src/train.py` and updated by `src/calibrate.py`:

| key | meaning |
|---|---|
| `arch` | `mobilenet_v2`, `efficientnet_b0` or `custom_cnn` |
| `classes` | class names in output-index order |
| `state_dict` | model weights from the **best validation-loss epoch** (not the last epoch) |
| `image` | preprocessing used in training (input size, resize, ImageNet mean/std); the Predictor reads it so inference always matches training |
| `temperature` | temperature-scaling constant fitted on the validation set |
| `best_epoch`, `val_loss`, `val_acc`, `val_ece` | training and calibration bookkeeping |

Load it for inference:

```python
from src.predict import Predictor
p = Predictor("models/mobilenet_v2.pt")
print(p.predict_path("some_flower.jpg"))
```

Only the final checkpoints are committed (see `.gitignore`). Every other run can be reproduced with `bash scripts/run_all.sh`.
