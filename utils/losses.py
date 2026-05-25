"""Rare-focused loss for v8 CellPath: 5 tissue + 10 nuclei + boundary-aware HV."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.constants import LOSS_MULTIPLIERS, NUCLEI_CLASS_WEIGHTS, TISSUE_CLASS_WEIGHTS
from training.logging_utils import logger


class SafeCrossEntropyLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, ignore_index: int = 255) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        if weight is None:
            self.weight = None
        else:
            self.register_buffer("weight", weight.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            ignore_index=self.ignore_index,
        )


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.25,
        smooth: float = 1e-5,
        ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        self.ignore_index = int(ignore_index)
        if class_weights is None:
            self.class_weights = None
        else:
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0
        c = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_safe = targets.clone()
        targets_safe[~valid] = 0
        one_hot = F.one_hot(targets_safe.clamp(0, c - 1), num_classes=c).permute(0, 3, 1, 2).float()
        valid_mask = valid.unsqueeze(1).float()
        probs = probs * valid_mask
        one_hot = one_hot * valid_mask
        tp = (probs * one_hot).sum(dim=(2, 3))
        fp = (probs * (1.0 - one_hot)).sum(dim=(2, 3))
        fn = ((1.0 - probs) * one_hot).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = torch.pow(1.0 - tversky, self.gamma)
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(loss.device).view(1, -1)
        return loss.mean()


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.squeeze(1))
        targets = targets.float()
        if targets.dim() > probs.dim():
            targets = targets.squeeze(1)
        inter = (probs * targets).sum(dim=(1, 2))
        den = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        return (1.0 - (2.0 * inter + self.smooth) / (den + self.smooth)).mean()


class FocalBCELoss(nn.Module):
    def __init__(self, alpha: float = 0.45, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.squeeze(1)
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - pt).pow(self.gamma) * bce).mean()


class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(
        self,
        tissue_weights: Optional[torch.Tensor] = None,
        nuclei_weights: Optional[torch.Tensor] = None,
        num_tasks: int = 5,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.use_focal_tversky = False
        self.focal_tversky_weight = 0.0

        if tissue_weights is None:
            tissue_weights = torch.tensor(TISSUE_CLASS_WEIGHTS, dtype=torch.float32)
        if nuclei_weights is None:
            nuclei_weights = torch.tensor(NUCLEI_CLASS_WEIGHTS, dtype=torch.float32)

        self.register_buffer("tissue_weights", tissue_weights.float())
        self.register_buffer("nuclei_weights", nuclei_weights.float())

        self.ce_tissue = SafeCrossEntropyLoss(weight=self.tissue_weights, ignore_index=ignore_index)
        self.ce_nc = SafeCrossEntropyLoss(weight=self.nuclei_weights, ignore_index=ignore_index)

        self.ft_tissue = FocalTverskyLoss(
            alpha=0.30, beta=0.70, gamma=1.25, ignore_index=ignore_index, class_weights=self.tissue_weights
        )
        self.ft_nc = FocalTverskyLoss(
            alpha=0.25, beta=0.75, gamma=1.50, ignore_index=ignore_index, class_weights=self.nuclei_weights
        )
        self.np_bce = FocalBCELoss(alpha=0.45, gamma=2.0)
        self.np_dice = SoftDiceLoss()

    def set_focal_tversky_weight(self, weight: float) -> None:
        weight = float(weight)
        weight = max(0.0, min(weight, 1.0))
        self.focal_tversky_weight = weight
        self.use_focal_tversky = weight > 0.0

    def switch_to_focal_tversky(self) -> None:
        if not self.use_focal_tversky or self.focal_tversky_weight < 1.0:
            self.set_focal_tversky_weight(1.0)
            logger.info("Switched tissue/nuclei losses to CE + full FocalTversky")

    def forward(
        self, preds: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, List[float]]:
        tissue_target = targets["tissue_sem"]
        nuclei_target = targets["nuclei_nc"]

        focal_weight = float(getattr(self, "focal_tversky_weight", 0.0))
        l_tissue = self.ce_tissue(preds["tissue"], tissue_target)
        l_nc = self.ce_nc(preds["nc"], nuclei_target)
        if focal_weight > 0.0:
            l_tissue = l_tissue + focal_weight * self.ft_tissue(preds["tissue"], tissue_target)
            l_nc = l_nc + focal_weight * self.ft_nc(preds["nc"], nuclei_target)

        l_np = self.np_bce(preds["np"], targets["nuclei_np"]) + self.np_dice(preds["np"], targets["nuclei_np"])

        hv_mask = targets["nuclei_np"].unsqueeze(1).bool().expand_as(preds["hv"])
        if torch.any(hv_mask):
            l_hv = F.smooth_l1_loss(preds["hv"][hv_mask], targets["nuclei_hv"][hv_mask], beta=0.5, reduction="mean")
        else:
            l_hv = preds["hv"].sum() * 0.0

        losses = [l_tissue, l_np, l_nc, l_hv, torch.tensor(0.0, device=l_tissue.device)]

        multipliers = LOSS_MULTIPLIERS + [0.0]
        total = 0.0
        for i, loss in enumerate(losses):
            total = total + multipliers[i] * (torch.exp(-self.log_vars[i]) * loss + self.log_vars[i])
        return total, [float(x.detach().item()) for x in losses[:4]]
