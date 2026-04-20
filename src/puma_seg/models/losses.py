"""
Loss functions for the PUMA nucleus segmentation pipeline.

CombinedSegLoss      — wraps Cellpose's internal flow + cellprob losses;
                       re-exported here for use outside of Cellpose's own
                       training loop (e.g. custom training loops).
ClassificationLoss   — weighted cross-entropy for imbalanced nucleus classes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Segmentation losses ──────────────────────────────────────────────────────


class FlowLoss(nn.Module):
    """MSE loss between predicted and target optical flow fields.

    Cellpose represents nuclei as a 2D flow field (dx, dy) pointing toward the
    nucleus centre.  This loss is applied only at foreground pixels.

    Args:
        reduction: ``"mean"`` or ``"sum"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_flow: torch.Tensor,
        target_flow: torch.Tensor,
        cellprob_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_flow:       (B, 2, H, W) — predicted (dx, dy) flows.
            target_flow:     (B, 2, H, W) — ground-truth flows.
            cellprob_target: (B, 1, H, W) — binary cell probability mask.

        Returns:
            Scalar flow loss value.
        """
        # Mask to foreground pixels only
        fg_mask = (cellprob_target > 0).float()  # (B, 1, H, W)
        n_fg = fg_mask.sum().clamp(min=1)

        sq_err = (pred_flow - target_flow) ** 2  # (B, 2, H, W)
        masked = sq_err * fg_mask  # broadcast over channel dim

        if self.reduction == "mean":
            return masked.sum() / (2 * n_fg)
        return masked.sum()


class CellProbLoss(nn.Module):
    """Binary cross-entropy loss for the cell probability map.

    Args:
        pos_weight: Weight for positive (cell) pixels to handle imbalance.
    """

    def __init__(self, pos_weight: float = 10.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        pred_prob: torch.Tensor,
        target_prob: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_prob:   (B, 1, H, W) raw logits.
            target_prob: (B, 1, H, W) binary targets {0, 1}.
        """
        pw = torch.tensor([self.pos_weight], device=pred_prob.device)
        return F.binary_cross_entropy_with_logits(
            pred_prob, target_prob, pos_weight=pw
        )


class CombinedSegLoss(nn.Module):
    """Combined segmentation loss = flow loss + cell-probability loss.

    This mirrors Cellpose's internal loss without coupling to their private API.

    Args:
        flow_weight: Weight for the optical flow MSE term.
        prob_weight: Weight for the cell probability BCE term.
        pos_weight:  Positive-pixel weight inside ``CellProbLoss``.
    """

    def __init__(
        self,
        flow_weight: float = 1.0,
        prob_weight: float = 1.0,
        pos_weight: float = 10.0,
    ) -> None:
        super().__init__()
        self.flow_loss = FlowLoss()
        self.prob_loss = CellProbLoss(pos_weight=pos_weight)
        self.flow_w = flow_weight
        self.prob_w = prob_weight

    def forward(
        self,
        pred: torch.Tensor,
        target_flow: torch.Tensor,
        target_prob: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:        (B, 3, H, W) — [dy, dx, cellprob] as output by CPnet.
            target_flow: (B, 2, H, W) — ground-truth [dy, dx].
            target_prob: (B, 1, H, W) — ground-truth cell probability.

        Returns:
            Weighted scalar loss.
        """
        pred_flow = pred[:, :2]   # (B, 2, H, W)
        pred_prob = pred[:, 2:3]  # (B, 1, H, W)

        l_flow = self.flow_loss(pred_flow, target_flow, target_prob)
        l_prob = self.prob_loss(pred_prob, target_prob)

        return self.flow_w * l_flow + self.prob_w * l_prob


# ─── Classification loss ──────────────────────────────────────────────────────


class ClassificationLoss(nn.Module):
    """Weighted cross-entropy loss for nucleus type classification.

    Args:
        class_weights: Optional 1-D tensor of per-class weights.
                       Typically computed from ``PUMAClassificationDataset.class_weights()``.
        label_smoothing: Smoothing factor in [0, 1).  ``0.1`` is a good default.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:  (B, n_classes) raw logits.
            targets: (B,) long tensor with class indices.

        Returns:
            Scalar loss.
        """
        weights = self.class_weights
        if weights is not None:
            weights = weights.to(logits.device)
        return F.cross_entropy(
            logits,
            targets,
            weight=weights,
            label_smoothing=self.label_smoothing,
        )


# ─── Focal loss (optional) ────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """Focal loss for severe class imbalance (alternative to weighted CE).

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017).

    Args:
        alpha:  Weighting factor for rare classes.
        gamma:  Focusing parameter.  ``gamma=0`` → standard cross-entropy.
        reduction: ``"mean"`` or ``"sum"``.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()
