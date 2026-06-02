from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch

from .config import IGNORE_INDEX, NUCLEI_CLASSES, TISSUE_CLASSES, id_to_name


@dataclass
class SegmentationAccumulator:
    num_classes: int
    ignore_index: int | None = None
    intersection: np.ndarray = field(init=False)
    pred_sum: np.ndarray = field(init=False)
    gt_sum: np.ndarray = field(init=False)

    def __post_init__(self):
        self.intersection = np.zeros(self.num_classes, dtype=np.float64)
        self.pred_sum = np.zeros(self.num_classes, dtype=np.float64)
        self.gt_sum = np.zeros(self.num_classes, dtype=np.float64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_np = pred.detach().cpu().numpy().astype(np.int64)
        target_np = target.detach().cpu().numpy().astype(np.int64)

        valid = np.ones_like(target_np, dtype=bool)
        if self.ignore_index is not None:
            valid = target_np != self.ignore_index

        for cid in range(self.num_classes):
            p = (pred_np == cid) & valid
            g = (target_np == cid) & valid
            self.intersection[cid] += np.logical_and(p, g).sum()
            self.pred_sum[cid] += p.sum()
            self.gt_sum[cid] += g.sum()

    def dice(self) -> np.ndarray:
        denom = self.pred_sum + self.gt_sum
        out = np.full(self.num_classes, np.nan, dtype=np.float64)
        valid = denom > 0
        out[valid] = (2.0 * self.intersection[valid]) / denom[valid]
        return out

    def mean_dice(self, include_ids: list[int] | None = None) -> float:
        d = self.dice()
        if include_ids is not None:
            d = d[include_ids]
        return float(np.nanmean(d)) if np.isfinite(d).any() else 0.0

    def micro_dice(self, include_ids: list[int] | None = None) -> float:
        if include_ids is None:
            ids = np.arange(self.num_classes)
        else:
            ids = np.asarray(include_ids, dtype=np.int64)
        tp = float(self.intersection[ids].sum())
        pred = float(self.pred_sum[ids].sum())
        gt = float(self.gt_sum[ids].sum())
        denom = pred + gt
        if denom <= 0.0:
            return 0.0
        return (2.0 * tp) / denom


class PumaMetrics:
    def __init__(self):
        # Tissue labels are normally 0..5 in this encoder-probe dataset.
        # Keeping IGNORE_INDEX support makes the metric safe if 255 appears.
        self.tissue = SegmentationAccumulator(len(TISSUE_CLASSES), ignore_index=IGNORE_INDEX)
        self.nuclei_fg = SegmentationAccumulator(2, ignore_index=None)
        self.nuclei_class = SegmentationAccumulator(len(NUCLEI_CLASSES), ignore_index=IGNORE_INDEX)

    @torch.no_grad()
    def update(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> None:
        self.tissue.update(outputs['tissue'].argmax(dim=1), batch['tissue'])
        self.nuclei_fg.update(outputs['nuclei_fg'].argmax(dim=1), batch['nuclei_fg'])
        self.nuclei_class.update(outputs['nuclei_class'].argmax(dim=1), batch['nuclei_class'])

    def compute(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        tissue_names = id_to_name(TISSUE_CLASSES)
        nuclei_names = id_to_name(NUCLEI_CLASSES)

        scored_tissue_ids = [cid for cid in range(len(TISSUE_CLASSES)) if cid != 0]

        tissue_dice_by_class = self.tissue.dice()
        for cid, value in enumerate(tissue_dice_by_class):
            result[f'dice_{tissue_names[cid]}'] = float(value) if np.isfinite(value) else float('nan')

        # Existing column kept for backward compatibility.
        result['mean_tissue_dice_scored_1_to_5'] = self.tissue.mean_dice(include_ids=scored_tissue_ids)

        fg_dice = self.nuclei_fg.dice()
        result['dice_nuclei_foreground'] = float(fg_dice[1]) if np.isfinite(fg_dice[1]) else 0.0

        nuclei_dice_by_class = self.nuclei_class.dice()
        for cid, value in enumerate(nuclei_dice_by_class):
            result[f'dice_{nuclei_names[cid]}'] = float(value) if np.isfinite(value) else float('nan')

        # Existing column kept for backward compatibility.
        result['mean_nuclei_class_dice_on_nuclei_pixels'] = self.nuclei_class.mean_dice()

        # New official-style validation metrics.
        # Pixel-level binary nuclei foreground F1 is equivalent to Dice.
        result['nuclei_f1'] = result['dice_nuclei_foreground']
        # Use the existing macro tissue Dice definition over scored tissue classes.
        result['tissue_dice'] = result['mean_tissue_dice_scored_1_to_5']
        # Per-class nuclei F1 averaged equally across nuclei classes.
        result['nuclei_macro_f1'] = result['mean_nuclei_class_dice_on_nuclei_pixels']
        # Micro tissue Dice pools TP/pred/GT over scored tissue classes.
        result['tissue_micro_dice'] = self.tissue.micro_dice(include_ids=scored_tissue_ids)

        result['official_selection_score'] = (
            result['nuclei_f1']
            + result['tissue_dice']
            + result['nuclei_macro_f1']
            + result['tissue_micro_dice']
        ) / 4.0

        # Old selection_score remains in metrics.csv for compatibility, but
        # checkpointing and comparison now use official_selection_score.
        result['selection_score'] = (
            result['mean_tissue_dice_scored_1_to_5']
            + result['dice_nuclei_foreground']
            + result['mean_nuclei_class_dice_on_nuclei_pixels']
        ) / 3.0
        return result
