from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from puma.config import PUMA_CLASS_NAMES, TAIL_CLASS_IDS


@dataclass(slots=True)
class MatchResult:
    pred_indices: np.ndarray
    gt_indices: np.ndarray
    distances: np.ndarray
    unmatched_pred: np.ndarray
    unmatched_gt: np.ndarray


def match_centroids(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    radius: float = 15.0,
    pred_scores: np.ndarray | None = None,
) -> MatchResult:
    """Reproduce the official PUMA greedy matching order for one category.

    Distances are computed once for the ROI instead of being recomputed for every ground
    truth. The official ground-truth order and confidence/distance tie-breaking are unchanged.
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float32).reshape(-1, 2)
    gt_xy = np.asarray(gt_xy, dtype=np.float32).reshape(-1, 2)
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return MatchResult(
            np.empty(0, int), np.empty(0, int), np.empty(0, np.float32),
            np.arange(len(pred_xy), dtype=int), np.arange(len(gt_xy), dtype=int),
        )
    if pred_scores is None:
        scores = np.ones(len(pred_xy), dtype=np.float32)
    else:
        scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
        if len(scores) != len(pred_xy):
            raise ValueError(f"pred_scores length {len(scores)} != predictions {len(pred_xy)}")
        scores = np.where(np.isfinite(scores), scores, -np.inf)
    # Spatial queries avoid a dense prediction-by-ground-truth distance matrix.
    prediction_tree = cKDTree(pred_xy)
    neighbour_lists = prediction_tree.query_ball_point(
        gt_xy, r=float(radius), workers=1
    )
    available_pred = np.ones(len(pred_xy), dtype=bool)
    matched_gt_mask = np.zeros(len(gt_xy), dtype=bool)
    matched_p: list[int] = []
    matched_g: list[int] = []
    distances: list[float] = []
    for gi, neighbours in enumerate(neighbour_lists):
        if not neighbours:
            continue
        eligible = np.asarray(sorted(neighbours), dtype=np.int64)
        eligible = eligible[available_pred[eligible]]
        if len(eligible) == 0:
            continue
        eligible_distances = np.linalg.norm(
            pred_xy[eligible] - gt_xy[gi], axis=1
        ).astype(np.float32, copy=False)
        # query_ball_point includes the exact boundary; the official evaluator uses < r.
        inside = eligible_distances < float(radius)
        eligible = eligible[inside]
        eligible_distances = eligible_distances[inside]
        if len(eligible) == 0:
            continue
        ranking = np.lexsort((eligible_distances, -scores[eligible]))
        local = int(ranking[0])
        pi = int(eligible[local])
        available_pred[pi] = False
        matched_gt_mask[gi] = True
        matched_p.append(pi)
        matched_g.append(gi)
        distances.append(float(eligible_distances[local]))
    return MatchResult(
        np.asarray(matched_p, dtype=int),
        np.asarray(matched_g, dtype=int),
        np.asarray(distances, dtype=np.float32),
        np.flatnonzero(available_pred).astype(int),
        np.flatnonzero(~matched_gt_mask).astype(int),
    )


def _record_match(record: dict, radius: float) -> MatchResult:
    cache = record.setdefault("_puma_match_cache", {})
    key = float(radius)
    match = cache.get(key)
    if match is None:
        match = match_centroids(
            record["pred_xy"], record["gt_xy"], radius, record.get("pred_scores")
        )
        cache[key] = match
    return match


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def evaluate_binary_detection(records: Iterable[dict], radius: float = 15.0) -> dict[str, float]:
    records = list(records)
    tp = fp = fn = 0
    errors: list[float] = []
    duplicates = 0
    total_pred = 0
    for r in records:
        pred, gt = np.asarray(r["pred_xy"]), np.asarray(r["gt_xy"])
        m = _record_match(r, radius)
        tp += len(m.pred_indices); fp += len(m.unmatched_pred); fn += len(m.unmatched_gt)
        errors.extend(m.distances.tolist()); total_pred += len(pred)
        if len(pred) and len(gt):
            nearest_distance, nearest = cKDTree(gt).query(pred, k=1, workers=1)
            eligible_nearest = nearest[np.asarray(nearest_distance) < radius]
            if len(eligible_nearest):
                _, duplicate_counts = np.unique(eligible_nearest, return_counts=True)
                duplicates += int(np.maximum(duplicate_counts - 1, 0).sum())
    p, rec, f1 = _f1(tp, fp, fn)
    e = np.asarray(errors, np.float32)
    return {
        "binary_precision": p, "binary_recall": rec, "binary_f1": f1,
        "binary_tp": tp, "binary_fp": fp, "binary_fn": fn,
        "mean_localization_error": float(e.mean()) if len(e) else np.nan,
        "median_localization_error": float(np.median(e)) if len(e) else np.nan,
        "p90_localization_error": float(np.percentile(e, 90)) if len(e) else np.nan,
        "p95_localization_error": float(np.percentile(e, 95)) if len(e) else np.nan,
        "duplicate_rate": duplicates / max(total_pred, 1),
        "candidate_load_per_roi": total_pred / max(len(records), 1),
    }


def prepare_oracle_context(
    records: list[dict], number_of_classes: int = 10
) -> dict[str, object]:
    """Precompute Stage-1 oracle quantities that do not depend on predictions."""
    diameter_parts: list[np.ndarray] = []
    nearest_parts: list[np.ndarray] = []
    for record in records:
        gt_class = np.asarray(record["gt_class"], dtype=np.int64)
        extent = np.asarray(
            record.get("gt_extent", np.ones((len(gt_class), 2)) * 16),
            dtype=np.float32,
        )
        diameter_parts.append(
            np.sqrt(np.maximum(extent[:, 0] * extent[:, 1], 1.0)).astype(
                np.float32, copy=False
            )
        )
        nearest_parts.append(
            np.asarray(
                record.get("gt_nearest", np.ones(len(gt_class)) * 99),
                dtype=np.float32,
            )
        )
    diameters = (
        np.concatenate(diameter_parts) if diameter_parts else np.empty(0, np.float32)
    )
    nearest_all = (
        np.concatenate(nearest_parts) if nearest_parts else np.empty(0, np.float32)
    )
    size_q = (
        np.percentile(diameters, [5, 25, 75, 95])
        if len(diameters)
        else np.asarray([8, 14, 24, 40], np.float32)
    )
    density_q = (
        np.percentile(nearest_all, [20, 40, 60, 80])
        if len(nearest_all)
        else np.asarray([10, 20, 30, 40], np.float32)
    )
    per_record: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for record, diameter, nearest in zip(
        records, diameter_parts, nearest_parts, strict=True
    ):
        gt_class = np.asarray(record["gt_class"], dtype=np.int64)
        per_record.append((
            gt_class,
            np.searchsorted(size_q, diameter, side="right").astype(np.int8),
            np.searchsorted(density_q, nearest, side="right").astype(np.int8),
        ))
    return {
        "number_of_classes": int(number_of_classes),
        "size_q": np.asarray(size_q),
        "density_q": np.asarray(density_q),
        "per_record": per_record,
    }


def oracle_official_metrics(
    records: list[dict],
    radius: float = 15.0,
    number_of_classes: int = 10,
    context: dict[str, object] | None = None,
) -> dict[str, float]:
    """Perfect class assignment and perfect rejection of unmatched candidates: Stage-1 ceiling."""
    if context is None:
        context = prepare_oracle_context(records, number_of_classes)
    if int(context["number_of_classes"]) != int(number_of_classes):
        raise ValueError("Oracle context class count does not match number_of_classes.")
    size_q = np.asarray(context["size_q"])
    density_q = np.asarray(context["density_q"])
    per_record = context["per_record"]
    if len(per_record) != len(records):
        raise ValueError("Oracle context record count does not match records.")

    tp = np.zeros(number_of_classes, dtype=np.int64)
    fn = np.zeros(number_of_classes, dtype=np.int64)
    counts = np.zeros(number_of_classes, dtype=np.int64)
    size_hit_counts = np.zeros((5, 2), dtype=np.int64)
    density_hit_counts = np.zeros((5, 2), dtype=np.int64)

    for record, static in zip(records, per_record, strict=True):
        gt_class, size_bins, density_bins = static
        gt_class = np.asarray(gt_class, dtype=np.int64)
        match = _record_match(record, radius)
        hit = np.zeros(len(gt_class), dtype=bool)
        hit[match.gt_indices] = True
        counts += np.bincount(gt_class, minlength=number_of_classes)[:number_of_classes]
        if np.any(hit):
            tp += np.bincount(
                gt_class[hit], minlength=number_of_classes
            )[:number_of_classes]
        if np.any(~hit):
            fn += np.bincount(
                gt_class[~hit], minlength=number_of_classes
            )[:number_of_classes]
        size_hit_counts[:, 1] += np.bincount(size_bins, minlength=5)[:5]
        density_hit_counts[:, 1] += np.bincount(density_bins, minlength=5)[:5]
        if np.any(hit):
            size_hit_counts[:, 0] += np.bincount(
                size_bins[hit], minlength=5
            )[:5]
            density_hit_counts[:, 0] += np.bincount(
                density_bins[hit], minlength=5
            )[:5]

    f1s: list[float] = []
    result: dict[str, float] = {}
    for class_id, name in enumerate(PUMA_CLASS_NAMES[:number_of_classes]):
        _, recall, f1 = _f1(int(tp[class_id]), 0, int(fn[class_id]))
        result[f"oracle_f1_{name}"] = f1
        result[f"detection_recall_{name}"] = recall
        f1s.append(f1)
    result["oracle_macro_f1"] = float(np.mean(f1s))
    result["oracle_sum_f1"] = float(np.sum(f1s))
    result["tail_detection_recall"] = float(
        sum(tp[list(TAIL_CLASS_IDS)]) / max(sum(counts[list(TAIL_CLASS_IDS)]), 1)
    )
    size_names = ("lt_p5", "p5_p25", "p25_p75", "p75_p95", "gt_p95")
    density_names = ("q1_dense", "q2", "q3", "q4", "q5_sparse")
    for index, name in enumerate(size_names):
        result[f"recall_size_{name}"] = (
            int(size_hit_counts[index, 0]) / max(int(size_hit_counts[index, 1]), 1)
        )
    for index, name in enumerate(density_names):
        result[f"recall_density_{name}"] = (
            int(density_hit_counts[index, 0])
            / max(int(density_hit_counts[index, 1]), 1)
        )
    result["recall_size_tiny"] = result["recall_size_lt_p5"]
    result["recall_size_small"] = result["recall_size_p5_p25"]
    result["recall_size_medium"] = result["recall_size_p25_p75"]
    result["recall_size_large"] = result["recall_size_p75_p95"]
    result["recall_size_extreme_large"] = result["recall_size_gt_p95"]
    result["recall_density_dense"] = result["recall_density_q1_dense"]
    result["recall_density_sparse"] = result["recall_density_q5_sparse"]
    result["size_p5"], result["size_p25"], result["size_p75"], result["size_p95"] = map(float, size_q)
    result["density_p20"], result["density_p40"], result["density_p60"], result["density_p80"] = map(float, density_q)
    return result


def evaluate_typed_detection(records: list[dict], radius: float = 15.0, number_of_classes: int = 10) -> dict[str, float]:
    """Official class-specific PUMA matching and macro/summed F1.

    Each class is matched independently, exactly as the challenge evaluator groups predictions
    by category before matching. A wrong-class prediction therefore cannot steal a match from a
    correct-class prediction.
    """
    tp = np.zeros(number_of_classes, dtype=np.int64)
    fp = np.zeros(number_of_classes, dtype=np.int64)
    fn = np.zeros(number_of_classes, dtype=np.int64)
    for record in records:
        pred_xy = np.asarray(record["pred_xy"], dtype=np.float32).reshape(-1, 2)
        gt_xy = np.asarray(record["gt_xy"], dtype=np.float32).reshape(-1, 2)
        pred_cls = np.asarray(record["pred_class"], dtype=int).reshape(-1)
        gt_cls = np.asarray(record["gt_class"], dtype=int).reshape(-1)
        pred_scores = np.asarray(record.get("pred_scores", np.ones(len(pred_xy))), dtype=np.float32).reshape(-1)
        if len(pred_cls) != len(pred_xy) or len(gt_cls) != len(gt_xy) or len(pred_scores) != len(pred_xy):
            raise ValueError("Typed detection record has inconsistent coordinate/class/score lengths")
        for class_id in range(number_of_classes):
            pred_indices = np.flatnonzero(pred_cls == class_id)
            gt_indices = np.flatnonzero(gt_cls == class_id)
            match = match_centroids(
                pred_xy[pred_indices],
                gt_xy[gt_indices],
                radius,
                pred_scores[pred_indices],
            )
            tp[class_id] += len(match.pred_indices)
            fp[class_id] += len(pred_indices) - len(match.pred_indices)
            fn[class_id] += len(gt_indices) - len(match.gt_indices)
    output: dict[str, float] = {}
    f1_values: list[float] = []
    for class_id, name in enumerate(PUMA_CLASS_NAMES[:number_of_classes]):
        precision, recall, f1 = _f1(int(tp[class_id]), int(fp[class_id]), int(fn[class_id]))
        output[f"precision_{name}"] = precision
        output[f"recall_{name}"] = recall
        output[f"f1_{name}"] = f1
        output[f"tp_{name}"] = int(tp[class_id])
        output[f"fp_{name}"] = int(fp[class_id])
        output[f"fn_{name}"] = int(fn[class_id])
        f1_values.append(f1)
    output["macro_f1"] = float(np.mean(f1_values))
    output["sum_f1"] = float(np.sum(f1_values))
    return output


