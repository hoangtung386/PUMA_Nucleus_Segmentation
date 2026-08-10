from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter


def _draw_gaussian_max(heatmap: np.ndarray, x: float, y: float, sigma: float) -> None:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    anchor_x, anchor_y = int(np.floor(x)), int(np.floor(y))
    left, right = max(0, anchor_x - radius), min(heatmap.shape[1], anchor_x + radius + 1)
    top, bottom = max(0, anchor_y - radius), min(heatmap.shape[0], anchor_y + radius + 1)
    if left >= right or top >= bottom:
        return
    dx = (np.arange(left, right, dtype=np.float32) + 0.5) - np.float32(x)
    dy = (np.arange(top, bottom, dtype=np.float32) + 0.5) - np.float32(y)
    kernel = np.exp(-(dy[:, None] ** 2 + dx[None, :] ** 2) / np.float32(2.0 * sigma * sigma))
    view = heatmap[top:bottom, left:right]
    np.maximum(view, kernel.astype(np.float32), out=view)
    heatmap[anchor_y, anchor_x] = 1.0


def build_dense_targets(
    centroids_xy: np.ndarray,
    height: int,
    width: int,
    *,
    fixed_sigma: float = 2.5,
    offset_radius: float = 5.0,
) -> dict[str, np.ndarray]:
    """Build the heatmap and offset targets used by A1_IFCRN_PP."""
    heatmap = np.zeros((height, width), dtype=np.float32)
    offset = np.zeros((2, height, width), dtype=np.float32)
    offset_valid = np.zeros((1, height, width), dtype=np.float32)

    for row in np.asarray(centroids_xy, dtype=np.float32):
        x, y = float(row[0]), float(row[1])
        xi, yi = int(np.floor(x)), int(np.floor(y))
        if not (0 <= xi < width and 0 <= yi < height):
            continue
        _draw_gaussian_max(heatmap, x, y, float(fixed_sigma))

        radius = max(float(offset_radius), 0.0)
        x0, x1 = max(0, int(np.floor(x - radius))), min(width, int(np.ceil(x + radius)) + 1)
        y0, y1 = max(0, int(np.floor(y - radius))), min(height, int(np.ceil(y + radius)) + 1)
        x_centres = np.arange(x0, x1, dtype=np.float32) + 0.5
        y_centres = np.arange(y0, y1, dtype=np.float32) + 0.5
        dx = x_centres - np.float32(x)
        dy = y_centres - np.float32(y)
        disk = dy[:, None] ** 2 + dx[None, :] ** 2 <= np.float32(max(radius, 0.5) ** 2)
        offset[0, y0:y1, x0:x1][disk] = np.broadcast_to(-dx[None, :], disk.shape)[disk]
        offset[1, y0:y1, x0:x1][disk] = np.broadcast_to(-dy[:, None], disk.shape)[disk]
        offset_valid[0, y0:y1, x0:x1][disk] = 1.0

    return {"heatmap": heatmap[None], "offset": offset, "offset_valid": offset_valid}


def modified_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    pred = logits.sigmoid().clamp(1e-5, 1 - 1e-5)
    pos = targets.eq(1).float()
    neg = targets.lt(1).float()
    pos_loss = -(1 - pred).pow(alpha) * pred.log() * pos
    neg_loss = -pred.pow(alpha) * (1 - pred).log() * (1 - targets).pow(beta) * neg
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp_min(1.0)


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.expand_as(pred).to(dtype=pred.dtype)
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _local_maxima(
    heatmap: np.ndarray,
    threshold: float,
    radius: int,
    max_candidates: int,
) -> np.ndarray:
    if heatmap.ndim == 3:
        heatmap = heatmap.squeeze(0)
    maxed = maximum_filter(heatmap, size=2 * radius + 1, mode="constant")
    yy, xx = np.where((heatmap >= maxed - 1e-8) & (heatmap >= threshold))
    if len(xx) == 0:
        return np.empty((0, 3), np.float32)
    scores = heatmap[yy, xx]
    order = np.argsort(-scores, kind="stable")[:max_candidates]
    return np.column_stack([xx[order], yy[order], scores[order]]).astype(np.float32)


@dataclass(slots=True)
class DecodedCandidates:
    coordinates: np.ndarray
    scores: np.ndarray


def decode_dense_predictions_multi_radius(
    outputs: dict[str, torch.Tensor],
    threshold: float,
    radii: tuple[int, ...] | list[int],
    max_candidates: int = 2000,
    stride: int = 1,
    heatmap_probabilities: np.ndarray | None = None,
) -> dict[int, list[DecodedCandidates]]:
    radii = tuple(dict.fromkeys(int(radius) for radius in radii)) or (3,)
    heat = (
        outputs["heatmap_logits"].detach().sigmoid().float().cpu().numpy()
        if heatmap_probabilities is None
        else np.asarray(heatmap_probabilities, dtype=np.float32)
    )
    offset = outputs.get("offset")
    decoded = {radius: [] for radius in radii}

    for batch_index in range(heat.shape[0]):
        peaks_by_radius = {
            radius: _local_maxima(heat[batch_index, 0], threshold, radius, max_candidates)
            for radius in radii
        }
        for radius in radii:
            peaks = peaks_by_radius[radius]
            if len(peaks) == 0:
                decoded[radius].append(
                    DecodedCandidates(
                        np.empty((0, 2), np.float32),
                        np.empty(0, np.float32),
                    )
                )
                continue

            xy = peaks[:, :2].copy() + np.float32(0.5)
            if offset is not None:
                x = torch.as_tensor(peaks[:, 0].astype(np.int64), device=offset.device)
                y = torch.as_tensor(peaks[:, 1].astype(np.int64), device=offset.device)
                residual = offset[batch_index, :, y, x].detach().T.float().cpu().numpy()
                xy += residual
            xy *= np.float32(stride)
            decoded[radius].append(
                DecodedCandidates(
                    coordinates=xy.astype(np.float32),
                    scores=peaks[:, 2].astype(np.float32),
                )
            )
    return decoded


def adaptive_suppress(
    candidates: DecodedCandidates,
    min_radius: float = 2.0,
    max_radius: float = 8.0,
) -> DecodedCandidates:
    if len(candidates.scores) <= 1:
        return candidates
    radius = float(np.clip(4.0, min_radius, max_radius))
    order = np.argsort(-candidates.scores, kind="stable")
    cell_size = max(radius, 1e-6)
    bins: dict[tuple[int, int], list[int]] = {}
    kept: list[int] = []
    coordinates = np.asarray(candidates.coordinates, np.float32)

    for raw_index in order:
        index = int(raw_index)
        xy = coordinates[index]
        cell = (
            int(math.floor(float(xy[0]) / cell_size)),
            int(math.floor(float(xy[1]) / cell_size)),
        )
        neighbours: list[int] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neighbours.extend(bins.get((cell[0] + dx, cell[1] + dy), ()))
        if neighbours:
            nearby = coordinates[np.asarray(neighbours, np.int64)]
            if np.any(np.square(nearby - xy).sum(axis=1) < radius * radius):
                continue
        kept.append(index)
        bins.setdefault(cell, []).append(index)

    selected = np.asarray(kept, np.int64)
    return DecodedCandidates(candidates.coordinates[selected], candidates.scores[selected])
