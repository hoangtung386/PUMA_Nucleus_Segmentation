from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IGNORE_INDEX, NUCLEI_CLASSES, TISSUE_CLASSES


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int | None = None) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    valid = torch.ones_like(target, dtype=torch.bool)
    safe_target = target.clone()
    if ignore_index is not None:
        valid = target != ignore_index
        safe_target = target.clone()
        safe_target[~valid] = 0

    one_hot = F.one_hot(safe_target.clamp(0, num_classes - 1), num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid_f = valid.unsqueeze(1).float()
    probs = probs * valid_f
    one_hot = one_hot * valid_f

    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    denominator = probs.sum(dims) + one_hot.sum(dims)
    present = one_hot.sum(dims) > 0
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    if present.any():
        return 1.0 - dice[present].mean()
    return logits.sum() * 0.0


class PumaMultiTaskLoss(nn.Module):
    def __init__(self, tissue_bg_weight: float = 0.25):
        super().__init__()
        tissue_weights = torch.ones(len(TISSUE_CLASSES), dtype=torch.float32)
        tissue_weights[0] = tissue_bg_weight
        self.register_buffer('tissue_weights', tissue_weights)
        self.tissue_classes = len(TISSUE_CLASSES)
        self.nuclei_classes = len(NUCLEI_CLASSES)

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tissue_target = batch['tissue']
        nuclei_fg_target = batch['nuclei_fg']
        nuclei_class_target = batch['nuclei_class']

        tissue_ce = F.cross_entropy(outputs['tissue'], tissue_target, weight=self.tissue_weights)
        tissue_dice = dice_loss_from_logits(outputs['tissue'], tissue_target, self.tissue_classes)

        fg_ce = F.cross_entropy(outputs['nuclei_fg'], nuclei_fg_target)
        fg_dice = dice_loss_from_logits(outputs['nuclei_fg'], nuclei_fg_target, 2)

        nclass_ce = F.cross_entropy(outputs['nuclei_class'], nuclei_class_target, ignore_index=IGNORE_INDEX)
        nclass_dice = dice_loss_from_logits(outputs['nuclei_class'], nuclei_class_target, self.nuclei_classes, ignore_index=IGNORE_INDEX)

        total = (
            1.0 * tissue_ce + 1.0 * tissue_dice
            + 1.0 * fg_ce + 1.0 * fg_dice
            + 1.0 * nclass_ce + 0.5 * nclass_dice
        )
        return {
            'loss': total,
            'tissue_ce': tissue_ce.detach(),
            'tissue_dice_loss': tissue_dice.detach(),
            'nuclei_fg_ce': fg_ce.detach(),
            'nuclei_fg_dice_loss': fg_dice.detach(),
            'nuclei_class_ce': nclass_ce.detach(),
            'nuclei_class_dice_loss': nclass_dice.detach(),
        }
