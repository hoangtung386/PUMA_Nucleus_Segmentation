"""
PUMA challenge evaluation metrics.

The official PUMA metric is the **summed macro F1-score** for nucleus
detection and classification:
  1. Match predicted nuclei to ground-truth nuclei using centroid distance ≤ 15 px.
  2. Compute per-class F1 for matched instances.
  3. PUMA score = sum of per-class F1 scores (not averaged).

Reference: https://puma.grand-challenge.org
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

PUMA_CENTROID_THRESHOLD: float = 15.0  # pixels


# ─── Instance matching ────────────────────────────────────────────────────────


def match_instances(
    pred_centroids: Dict[int, Tuple[float, float]],
    gt_centroids: Dict[int, Tuple[float, float]],
    threshold: float = PUMA_CENTROID_THRESHOLD,
) -> Tuple[List[Tuple[int, int]], Set[int], Set[int]]:
    """Match predicted nuclei to GT nuclei by centroid distance (Hungarian matching).

    Args:
        pred_centroids: {pred_id: (row, col)} — predicted nucleus centroids.
        gt_centroids:   {gt_id: (row, col)} — ground-truth nucleus centroids.
        threshold:      Maximum centroid distance (px) to consider a match.

    Returns:
        matched_pairs:     List of (pred_id, gt_id) matched pairs.
        unmatched_pred:    Set of unmatched prediction IDs (false positives).
        unmatched_gt:      Set of unmatched GT IDs (false negatives).
    """
    pred_ids = list(pred_centroids.keys())
    gt_ids = list(gt_centroids.keys())

    if not pred_ids or not gt_ids:
        return [], set(pred_ids), set(gt_ids)

    # Build pairwise distance matrix
    pred_coords = np.array([pred_centroids[p] for p in pred_ids])  # (P, 2)
    gt_coords = np.array([gt_centroids[g] for g in gt_ids])        # (G, 2)

    diff = pred_coords[:, None, :] - gt_coords[None, :, :]         # (P, G, 2)
    dist_matrix = np.linalg.norm(diff, axis=-1)                    # (P, G)

    # Hungarian assignment
    row_idx, col_idx = linear_sum_assignment(dist_matrix)

    matched_pairs: List[Tuple[int, int]] = []
    unmatched_pred: Set[int] = set(pred_ids)
    unmatched_gt: Set[int] = set(gt_ids)

    for r, c in zip(row_idx, col_idx):
        if dist_matrix[r, c] <= threshold:
            p_id = pred_ids[r]
            g_id = gt_ids[c]
            matched_pairs.append((p_id, g_id))
            unmatched_pred.discard(p_id)
            unmatched_gt.discard(g_id)

    return matched_pairs, unmatched_pred, unmatched_gt


# ─── Detection F1 ─────────────────────────────────────────────────────────────


def detection_f1(
    pred_centroids: Dict[int, Tuple[float, float]],
    gt_centroids: Dict[int, Tuple[float, float]],
    threshold: float = PUMA_CENTROID_THRESHOLD,
) -> Dict[str, float]:
    """Compute nucleus detection precision, recall, and F1 (class-agnostic).

    Args:
        pred_centroids: Predicted centroids dict.
        gt_centroids:   Ground-truth centroids dict.
        threshold:      Centroid distance threshold in pixels.

    Returns:
        dict with keys ``"precision"``, ``"recall"``, ``"f1"``, ``"tp"``,
        ``"fp"``, ``"fn"``.
    """
    matched, fp_set, fn_set = match_instances(pred_centroids, gt_centroids, threshold)
    tp = len(matched)
    fp = len(fp_set)
    fn = len(fn_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ─── Classification F1 ────────────────────────────────────────────────────────


def classification_f1(
    matched_pairs: List[Tuple[int, int]],
    pred_classes: Dict[int, int],
    gt_classes: Dict[int, int],
    n_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute per-class and macro F1 for matched nucleus pairs.

    Args:
        matched_pairs: Output of ``match_instances``.
        pred_classes:  {pred_id: class_id} — predicted class per nucleus.
        gt_classes:    {gt_id: class_id} — GT class per nucleus.
        n_classes:     Total number of foreground classes (excluding background).
        class_names:   Optional list of class name strings for the result keys.

    Returns:
        dict with per-class F1 under their names and ``"macro_f1"``.
    """
    if not matched_pairs:
        keys = class_names if class_names else [str(i) for i in range(1, n_classes + 1)]
        result = {k: 0.0 for k in keys}
        result["macro_f1"] = 0.0
        return result

    y_pred = [pred_classes.get(p, 0) for p, _ in matched_pairs]
    y_true = [gt_classes.get(g, 0) for _, g in matched_pairs]

    labels = list(range(1, n_classes + 1))  # exclude background (0)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    if class_names is None:
        class_names = [str(i) for i in labels]

    result: Dict[str, float] = {}
    for name, f1_val in zip(class_names, per_class_f1):
        result[name] = float(f1_val)

    result["macro_f1"] = float(per_class_f1.mean())
    return result


# ─── Official PUMA metric ─────────────────────────────────────────────────────


def compute_puma_score(
    pred_centroids: Dict[int, Tuple[float, float]],
    gt_centroids: Dict[int, Tuple[float, float]],
    pred_classes: Dict[int, int],
    gt_classes: Dict[int, int],
    n_classes: int,
    class_names: Optional[List[str]] = None,
    threshold: float = PUMA_CENTROID_THRESHOLD,
) -> Dict[str, float]:
    """Compute the official PUMA challenge score for a single image.

    PUMA score = **sum** of per-class F1 scores (not average).

    Args:
        pred_centroids: {pred_id: (row, col)} predicted centroids.
        gt_centroids:   {gt_id: (row, col)} ground-truth centroids.
        pred_classes:   {pred_id: class_id} predicted nucleus classes.
        gt_classes:     {gt_id: class_id} GT nucleus classes.
        n_classes:      Number of foreground classes.
        class_names:    Optional class name list.
        threshold:      Centroid distance threshold (default 15 px).

    Returns:
        dict with detection metrics, per-class F1, and the ``"puma_score"``.
    """
    det_metrics = detection_f1(pred_centroids, gt_centroids, threshold)
    matched, _, _ = match_instances(pred_centroids, gt_centroids, threshold)
    cls_metrics = classification_f1(
        matched, pred_classes, gt_classes, n_classes, class_names
    )

    puma_score = sum(
        v for k, v in cls_metrics.items() if k != "macro_f1"
    )

    return {
        **det_metrics,
        **cls_metrics,
        "puma_score": puma_score,
    }


# ─── Dataset-level evaluation ─────────────────────────────────────────────────


def evaluate_predictions(
    pred_list: List[Dict],
    gt_list: List[Dict],
    n_classes: int,
    class_names: Optional[List[str]] = None,
    threshold: float = PUMA_CENTROID_THRESHOLD,
) -> Dict[str, float]:
    """Aggregate PUMA metrics over a dataset split.

    Args:
        pred_list: List of prediction dicts, each with keys:
                   ``"centroids"`` ({id: (r,c)}) and ``"classes"`` ({id: cls_id}).
        gt_list:   Corresponding list of GT dicts with same keys.
        n_classes: Number of foreground classes.
        class_names: Optional list of string class names.
        threshold: Centroid distance threshold.

    Returns:
        Aggregated metrics dict with per-class F1 and ``"puma_score"``.
    """
    assert len(pred_list) == len(gt_list), "pred and gt lists must have equal length."

    if class_names is None:
        class_names = [str(i) for i in range(1, n_classes + 1)]

    per_image_scores = [
        compute_puma_score(
            pred["centroids"],
            gt["centroids"],
            pred["classes"],
            gt["classes"],
            n_classes,
            class_names,
            threshold,
        )
        for pred, gt in zip(pred_list, gt_list)
    ]

    # Average across images
    keys = list(per_image_scores[0].keys())
    aggregated: Dict[str, float] = {}
    for k in keys:
        vals = [s[k] for s in per_image_scores]
        aggregated[k] = float(np.mean(vals))

    return aggregated
