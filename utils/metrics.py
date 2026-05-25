import math
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from data.constants import RARE_NUCLEI_IDS, RARE_TISSUE_IDS

EPS = 1e-8


class SemanticMetricAccumulator:
    """Validation-set level Dice/IoU accumulator."""

    def __init__(
        self, num_classes: int, prefix: str, ignore_index: Optional[int] = 255, device: Union[str, torch.device] = "cpu"
    ) -> None:
        self.num_classes = int(num_classes)
        self.prefix = prefix
        self.ignore_index = ignore_index
        self.tp = torch.zeros(self.num_classes, dtype=torch.float64, device=device)
        self.pred_sum = torch.zeros(self.num_classes, dtype=torch.float64, device=device)
        self.target_sum = torch.zeros(self.num_classes, dtype=torch.float64, device=device)
        self.union = torch.zeros(self.num_classes, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if preds.ndim == targets.ndim + 1:
            pred_labels = torch.argmax(preds, dim=1)
        else:
            pred_labels = preds
        pred_labels = pred_labels.detach()
        targets = targets.detach()
        valid = targets != self.ignore_index if self.ignore_index is not None else torch.ones_like(targets, dtype=torch.bool)
        for k in range(self.num_classes):
            p = (pred_labels == k) & valid
            t = (targets == k) & valid
            self.tp[k] += (p & t).sum(dtype=torch.float64)
            self.pred_sum[k] += p.sum(dtype=torch.float64)
            self.target_sum[k] += t.sum(dtype=torch.float64)
            self.union[k] += (p | t).sum(dtype=torch.float64)

    def compute(self) -> Dict[str, float]:
        out = {}
        dice_values = []
        iou_values = []
        for k in range(self.num_classes):
            tp = float(self.tp[k].item())
            pred = float(self.pred_sum[k].item())
            target = float(self.target_sum[k].item())
            union = float(self.union[k].item())
            if target == 0.0 and pred == 0.0:
                dice = math.nan
                iou = math.nan
            elif target == 0.0 and pred > 0.0:
                dice = 0.0
                iou = 0.0
            else:
                dice = (2.0 * tp) / max(pred + target, EPS)
                iou = tp / max(union, EPS)
            out[f"{self.prefix}_dice_{k}"] = dice
            out[f"{self.prefix}_iou_{k}"] = iou
            out[f"{self.prefix}_target_pixels_{k}"] = target
            out[f"{self.prefix}_pred_pixels_{k}"] = pred
            if not math.isnan(dice):
                dice_values.append(dice)
            if not math.isnan(iou):
                iou_values.append(iou)
        out[f"{self.prefix}_macro_dice_valid"] = float(np.mean(dice_values)) if dice_values else math.nan
        out[f"{self.prefix}_macro_iou_valid"] = float(np.mean(iou_values)) if iou_values else math.nan
        return out


class PUMAMetrics:
    """Convenience wrapper around semantic metric computation for PUMA tasks."""

    def new_semantic_accumulator(
        self, num_classes: int, prefix: str, ignore_index: Optional[int] = 255, device: Union[str, torch.device] = "cpu"
    ) -> SemanticMetricAccumulator:
        return SemanticMetricAccumulator(num_classes, prefix, ignore_index=ignore_index, device=device)

    @staticmethod
    def _nanmean(values: List[float]) -> float:
        clean = []
        for value in values:
            if value is None:
                continue
            value = float(value)
            if not math.isnan(value):
                clean.append(value)
        return float(np.mean(clean)) if clean else math.nan

    @staticmethod
    def _nan_to_zero(value: Optional[float]) -> float:
        if value is None:
            return 0.0
        value = float(value)
        return 0.0 if math.isnan(value) else value

    def calculate_semantic_metrics(
        self, logits: torch.Tensor, targets: torch.Tensor, num_classes: int, prefix: str, ignore_index: Optional[int] = 255
    ) -> Dict[str, float]:
        device = logits.device if torch.is_tensor(logits) else "cpu"
        acc = self.new_semantic_accumulator(num_classes, prefix, ignore_index=ignore_index, device=device)
        acc.update(logits, targets)
        return acc.compute()

    def calculate_all_metrics(
        self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        out = {}
        out.update(self.calculate_semantic_metrics(preds["tissue"], targets["tissue_sem"], 6, "tissue", ignore_index=None))
        out.update(self.calculate_semantic_metrics(preds["nc"], targets["nuclei_nc"], 10, "nuclei"))
        tissue_dice = [out.get(f"tissue_dice_{i}", math.nan) for i in range(6)]
        nuclei_dice = [out.get(f"nuclei_dice_{i}", math.nan) for i in range(10)]
        rare_tissue_dice = [out.get(f"tissue_dice_{i}", math.nan) for i in sorted(RARE_TISSUE_IDS)]
        rare_nuclei_dice = [out.get(f"nuclei_dice_{i}", math.nan) for i in sorted(RARE_NUCLEI_IDS)]
        rare_dice = rare_tissue_dice + rare_nuclei_dice
        out["avg_tissue_dice"] = self._nan_to_zero(self._nanmean(tissue_dice))
        out["avg_nuclei_dice"] = self._nan_to_zero(self._nanmean(nuclei_dice))
        out["rare_tissue_macro_dice"] = self._nan_to_zero(self._nanmean(rare_tissue_dice))
        out["rare_nuclei_macro_dice"] = self._nan_to_zero(self._nanmean(rare_nuclei_dice))
        out["rare_macro_dice"] = self._nan_to_zero(self._nanmean(rare_dice))
        # Rare-focused checkpoint selection. This intentionally gives rare classes
        # the largest weight, because common tumor/stroma/lymphocyte already learn well.
        out["selection_score"] = (
            0.20 * out["avg_tissue_dice"] + 0.25 * out["avg_nuclei_dice"] + 0.55 * out["rare_macro_dice"]
        )
        return out
