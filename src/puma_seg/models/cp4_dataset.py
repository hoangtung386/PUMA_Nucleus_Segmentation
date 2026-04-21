"""
CP4 dataset and loss helpers.

CP4Dataset converts instance masks to flow fields for CP4 fine-tuning.
CP4Loss combines L1 flow loss and BCE cell-probability loss.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CP4Loss(nn.Module):
    """Combined loss for CP4 training: L1 flow + BCE cell probability."""

    def __init__(self, flow_weight: float = 1.0, cellprob_weight: float = 1.0) -> None:
        super().__init__()
        self.flow_weight = flow_weight
        self.cellprob_weight = cellprob_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred: torch.Tensor,
        y_flow: torch.Tensor,
        x_flow: torch.Tensor,
        cellprob: torch.Tensor,
    ) -> torch.Tensor:
        pred_flow = pred[:, :2]
        pred_cellprob = pred[:, 2:3]
        _, _, pred_h, pred_w = pred_flow.shape
        target_h, target_w = y_flow.shape[1], y_flow.shape[2]
        if pred_h != target_h or pred_w != target_w:
            pred_flow = F.interpolate(pred_flow, size=(target_h, target_w), mode="bilinear", align_corners=False)
            pred_cellprob = F.interpolate(pred_cellprob, size=(target_h, target_w), mode="bilinear", align_corners=False)
        flow_target = torch.cat([y_flow, x_flow], dim=1)
        flow_loss = F.l1_loss(pred_flow, flow_target, reduction="mean")
        cellprob_loss = self.bce(pred_cellprob.squeeze(1), cellprob)
        return self.flow_weight * flow_loss + self.cellprob_weight * cellprob_loss


class CP4Dataset(torch.utils.data.Dataset):
    """Dataset for CP4 fine-tuning from image/mask arrays."""

    TARGET_SIZE = 256

    def __init__(
        self,
        images: List[np.ndarray],
        labels: List[np.ndarray],
        diameter: float = 30.0,
        augment: bool = True,
    ) -> None:
        self.images = images
        self.labels = labels
        self.diameter = diameter
        self.augment = augment

        import cv2
        self._resize = lambda img: cv2.resize(img, (self.TARGET_SIZE, self.TARGET_SIZE))
        self._resize_mask = lambda mask: cv2.resize(
            mask.astype(np.float32),
            (self.TARGET_SIZE, self.TARGET_SIZE),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        img = self.images[idx].astype(np.float32) / 255.0
        mask = self.labels[idx]

        img = self._resize(img)
        mask = self._resize_mask(mask)

        if self.augment:
            img, mask = self._augment(img, mask)

        img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        y_flow, x_flow, cellprob = self._masks_to_flows(mask, self.diameter)

        return (
            img_t,
            torch.from_numpy(y_flow).float(),
            torch.from_numpy(x_flow).float(),
            torch.from_numpy(cellprob).float(),
        )

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        if np.random.rand() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
        if np.random.rand() > 0.5:
            img = np.rot90(img, k=1, axes=(0, 1)).copy()
            mask = np.rot90(mask, k=1, axes=(0, 1)).copy()
        return img, mask

    def _masks_to_flows(
        self, mask: np.ndarray, diameter: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert instance masks to flow vectors (simplified variant)."""
        _ = diameter
        h, w = mask.shape
        y_flow = np.zeros((h, w), dtype=np.float32)
        x_flow = np.zeros((h, w), dtype=np.float32)
        cellprob = np.zeros((h, w), dtype=np.float32)

        if mask.max() == 0:
            return y_flow, x_flow, cellprob

        for inst_id in range(1, int(mask.max()) + 1):
            inst_mask = mask == inst_id
            if not inst_mask.any():
                continue

            ys, xs = np.where(inst_mask)
            if len(ys) < 3:
                continue

            y_center = ys.mean()
            x_center = xs.mean()
            cellprob[inst_mask] = 1.0

            dist = np.sqrt((ys - y_center) ** 2 + (xs - x_center) ** 2)
            dist = np.maximum(dist, 1e-6)
            y_flow[inst_mask] = (ys - y_center) / dist
            x_flow[inst_mask] = (xs - x_center) / dist

        return y_flow, x_flow, cellprob
