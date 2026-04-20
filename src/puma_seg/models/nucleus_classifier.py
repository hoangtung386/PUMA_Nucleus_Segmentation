"""
Per-nucleus classification model for the PUMA challenge.

``NucleusClassifier`` wraps a torchvision backbone (ResNet-18 by default)
pre-trained on ImageNet and replaces the final FC layer with a PUMA-specific
classification head.

Design choices:
  * ResNet-18 is lightweight enough to train on nucleus crops (64×64 px).
  * The penultimate features can optionally be frozen for fast fine-tuning
    when nucleus-crop data is limited.
  * ``build_classifier()`` is the recommended factory function.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models as tv_models
from torchvision.models import ResNet18_Weights

logger = logging.getLogger(__name__)


class NucleusClassifier(nn.Module):
    """ResNet-18-based nucleus type classifier.

    Args:
        n_classes:        Number of foreground nucleus classes
                          (3 for Track 1, 10 for Track 2).
        pretrained:       Load ImageNet weights for the backbone.
        freeze_backbone:  If ``True``, freeze all layers except the
                          classification head (useful for small datasets).
        dropout:          Dropout probability before the classification layer.
    """

    def __init__(
        self,
        n_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        # ── Strip the original FC layer ────────────────────────────────────
        in_features = backbone.fc.in_features  # 512 for ResNet-18
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # ── Custom classification head ─────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

        self.n_classes = n_classes

        if freeze_backbone:
            self.freeze_backbone()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) float tensor (ImageNet-normalised).

        Returns:
            logits: (B, n_classes) — raw logits (no softmax).
        """
        features = self.backbone(x)  # (B, 512)
        return self.head(features)   # (B, n_classes)

    # ── Freeze / unfreeze helpers ──────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters; only classification head is trained."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("NucleusClassifier: backbone frozen.")

    def unfreeze_backbone(self, layer_groups: Optional[int] = None) -> None:
        """Unfreeze backbone layers for fine-tuning.

        Args:
            layer_groups: How many ResNet layer groups to unfreeze from the end
                          (None → unfreeze everything).
        """
        if layer_groups is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("NucleusClassifier: entire backbone unfrozen.")
        else:
            # ResNet-18 groups: layer4, layer3, layer2, layer1, conv1/bn1
            resnet_layers = [
                self.backbone.layer4,
                self.backbone.layer3,
                self.backbone.layer2,
                self.backbone.layer1,
            ]
            for layer in resnet_layers[:layer_groups]:
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info(
                "NucleusClassifier: unfrozen last %d layer group(s).", layer_groups
            )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(), "n_classes": self.n_classes},
            str(path),
        )
        logger.info("NucleusClassifier saved to: %s", path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "NucleusClassifier":
        ckpt = torch.load(str(path), map_location="cpu")
        n_classes = ckpt.get("n_classes", 3)
        model = cls(n_classes=n_classes, pretrained=False, **kwargs)
        model.load_state_dict(ckpt["state_dict"])
        logger.info("NucleusClassifier loaded from: %s", path)
        return model

    # ── Embedding access (for t-SNE / analysis) ───────────────────────────────

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate features (before dropout + final linear)."""
        features = self.backbone(x)
        # Pass through everything except the last Linear layer
        h = self.head[:-1](features)
        return h


# ─── Factory ──────────────────────────────────────────────────────────────────


def build_classifier(
    track: int = 1,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.3,
) -> NucleusClassifier:
    """Build a ``NucleusClassifier`` configured for the given PUMA track.

    Args:
        track:           1 (3 foreground classes) or 2 (10 foreground classes).
        pretrained:      Use ImageNet pre-trained weights.
        freeze_backbone: Start with frozen backbone (recommended when data < 5k crops).
        dropout:         Dropout in the classification head.

    Returns:
        Configured ``NucleusClassifier`` instance.
    """
    from puma_seg.data.geojson_parser import NUM_CLASSES

    n_classes = NUM_CLASSES[track] - 1  # exclude background class 0
    model = NucleusClassifier(
        n_classes=n_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )
    logger.info(
        "Built NucleusClassifier: track=%d, n_classes=%d, frozen=%s",
        track,
        n_classes,
        freeze_backbone,
    )
    return model


