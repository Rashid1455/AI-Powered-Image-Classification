"""Grad-CAM: highlight the image regions that drove a prediction.

Hooks the last convolutional block of the backbone, weights its activation
maps by the spatially-averaged gradient of the target class score, and
upsamples the result to the input size.
"""
from __future__ import annotations

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.target_layer = list(model.backbone.children())[-1]
        self._acts = self._grads = None
        self.target_layer.register_forward_hook(self._save_acts)
        self.target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _m, _i, out):
        self._acts = out.detach()

    def _save_grads(self, _m, _gi, go):
        self._grads = go[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Return a (H, W) heatmap in [0, 1] for a single preprocessed image."""
        self.model.eval()
        x = x.clone().requires_grad_(True)  # allows grads to flow even when the backbone is frozen
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._acts).sum(1))[0]
        cam = cam / (cam.max() + 1e-8)
        cam = torch.nn.functional.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear",
                                              align_corners=False)[0, 0]
        return cam.cpu().numpy()


def overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a heatmap over the (centre-cropped) image."""
    h, w = cam.shape
    base = np.asarray(img.convert("RGB").resize((w, h)), dtype=np.float32) / 255
    heat = colormaps["jet"](cam)[..., :3]
    out = (1 - alpha) * base + alpha * heat
    return Image.fromarray((out * 255).astype(np.uint8))


def center_crop_view(img: Image.Image, resize_short: int, size: int) -> Image.Image:
    """The exact crop the model sees, so the overlay lines up with the heatmap."""
    img = img.convert("RGB")
    s = resize_short / min(img.size)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)
    left, top = (img.width - size) // 2, (img.height - size) // 2
    return img.crop((left, top, left + size, top + size))
