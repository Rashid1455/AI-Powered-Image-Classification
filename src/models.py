"""Model factory: two transfer-learning backbones and a from-scratch baseline CNN.

Every model exposes `.backbone` (feature extractor) and `.head` (classifier)
so the two-stage freeze -> fine-tune schedule works the same for all of them.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class TransferNet(nn.Module):
    def __init__(self, backbone: nn.Sequential, feat_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.backbone(x)))


class CustomCNN(nn.Module):
    """Small VGG-style CNN trained from scratch - the fundamentals baseline."""

    def __init__(self, num_classes: int, dropout: float):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.backbone = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.backbone(x)))


def build_model(arch: str, num_classes: int, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    if arch == "mobilenet_v2":
        w = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.mobilenet_v2(weights=w)
        return TransferNet(net.features, 1280, num_classes, dropout)
    if arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=w)
        return TransferNet(net.features, 1280, num_classes, dropout)
    if arch == "custom_cnn":
        return CustomCNN(num_classes, dropout)
    raise ValueError(f"unknown arch: {arch}")


def freeze_backbone(model: nn.Module) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_top(model: nn.Module, fraction: float) -> int:
    """Unfreeze the last `fraction` of backbone blocks. Returns number unfrozen.

    BatchNorm layers in still-frozen blocks stay in eval mode (see `set_bn_eval`)
    so their ImageNet running statistics are not overwritten by small batches.
    """
    blocks = list(model.backbone.children())
    n = max(1, round(len(blocks) * fraction))
    for blk in blocks[-n:]:
        for p in blk.parameters():
            p.requires_grad = True
    return n


def set_frozen_bn_eval(model: nn.Module) -> None:
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
