
import torch
import torch.nn as nn
import torch.nn.functional as F

from symbiopan.common.logging import get_logger

logger = get_logger(__name__)


class SafeCrossEntropyLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, ignore_index: int | None = 255) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        if weight is None:
            self.weight = None
        else:
            self.register_buffer("weight", weight.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.ignore_index is not None:
            valid = targets != self.ignore_index
            if not torch.any(valid):
                return logits.sum() * 0.0
            ce_ignore = self.ignore_index
        else:
            ce_ignore = -100
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            ignore_index=ce_ignore,
        )


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.25,
        smooth: float = 1e-5,
        ignore_index: int | None = 255,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        self.ignore_index = ignore_index
        if class_weights is None:
            self.class_weights = None
        else:
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        c = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        if self.ignore_index is not None:
            valid = targets != self.ignore_index
            if not torch.any(valid):
                return logits.sum() * 0.0
            targets_safe = targets.clone()
            targets_safe[~valid] = 0
            valid_mask = valid.unsqueeze(1).float()
            probs = probs * valid_mask
        else:
            targets_safe = targets
            valid_mask = 1.0
        one_hot = F.one_hot(targets_safe.clamp(0, c - 1), num_classes=c).permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid_mask if torch.is_tensor(valid_mask) else one_hot
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
