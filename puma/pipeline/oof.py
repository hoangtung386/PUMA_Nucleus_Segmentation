from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from puma.config import (
    REJECT_CLASS_ID,
    RuntimeConfig,
    stage1_experiment_registry,
    stage1_model_config_from_dict,
    validate_folds,
)
from puma.data.datasets import PumaNpyStore
from puma.evaluation.metrics import match_centroids
from puma.models.stage1 import build_stage1_model
from puma.training.stage1 import STAGE1_EXPERIMENT, predict_roi
from puma.training.stage1 import _stage1_run_hash
from puma.utils import (
    atomic_save_numpy,
    atomic_write_json,
    load_checkpoint,
    release_cuda_memory,
    resolve_amp_dtype,
    resolve_device,
    utc_now_iso,
)

OOF_IMPLEMENTATION_REVISION = 2


CANDIDATE_DTYPE = np.dtype(
    [
        ("oof_row_id", "i8"),
        ("roi_index", "i4"),
        ("candidate_index", "i4"),
        ("x", "f4"),
        ("y", "f4"),
        ("confidence", "f4"),
        ("nearest_distance", "f4"),
        ("class_id", "i2"),
        ("matched_gt_index", "i4"),
        ("match_distance", "f4"),
        ("is_reject", "u1"),
        ("fold", "i1"),
    ]
)


def validate_full_oof(runtime: RuntimeConfig) -> Path:
    path = runtime.paths.stage1_existing_file("stage1_oof_candidates.npy")
    if not path.exists():
        raise FileNotFoundError(
            "stage1_oof_candidates.npy is missing. Generate all five A1 OOF folds first."
        )
    candidates = np.load(path, mmap_mode="r", allow_pickle=False)
    if candidates.dtype != CANDIDATE_DTYPE:
        raise RuntimeError(
            f"Stage-1 OOF schema mismatch. Expected {CANDIDATE_DTYPE}, got {candidates.dtype}."
        )
    expected = set(range(runtime.data.number_of_folds))
    observed = set(np.unique(candidates["fold"]).astype(int).tolist())
    if observed != expected:
        raise RuntimeError(
            f"Stage 2 requires OOF folds {sorted(expected)}, got {sorted(observed)}."
        )
    return path


def _candidate_rows_for_roi(
    roi: int,
    fold: int,
    prediction,
    ground_truth: np.ndarray,
    match_radius: float,
    image_shape: tuple[int, ...],
) -> list[tuple[Any, ...]]:
    gt_xy = np.column_stack([ground_truth["x"], ground_truth["y"]]).astype(np.float32)
    match = match_centroids(prediction.coordinates, gt_xy, match_radius, prediction.scores)
    matched = {
        int(pred_index): (int(gt_index), float(distance))
        for pred_index, gt_index, distance in zip(
            match.pred_indices, match.gt_indices, match.distances, strict=True
        )
    }
    if len(prediction.coordinates) > 1:
        nearest = cKDTree(np.asarray(prediction.coordinates, np.float32)).query(
            prediction.coordinates, k=2
        )[0][:, 1].astype(np.float32)
    else:
        # Match the Stage-2 geometry fallback used by GT-positive training and
        # challenge inference when only one candidate exists in a 1024 ROI.
        nearest = np.full(
            len(prediction.coordinates), float(max(image_shape[:2])), dtype=np.float32
        )

    rows: list[tuple[Any, ...]] = []
    for index in range(len(prediction.scores)):
        if index in matched:
            gt_index, distance = matched[index]
            class_id = int(ground_truth[gt_index]["class_id"])
            is_reject = 0
        else:
            gt_index, distance = -1, np.nan
            class_id = REJECT_CLASS_ID
            is_reject = 1
        rows.append(
            (
                -1,
                roi,
                index,
                float(prediction.coordinates[index, 0]),
                float(prediction.coordinates[index, 1]),
                float(prediction.scores[index]),
                float(nearest[index]),
                class_id,
                gt_index,
                float(distance),
                is_reject,
                fold,
            )
        )
    return rows


def _oof_cache_signature(
    runtime: RuntimeConfig,
    seed: int,
    folds: tuple[int, ...],
) -> dict[str, Any]:
    checkpoints: list[dict[str, Any]] = []
    for fold in folds:
        path = runtime.paths.stage1_existing_file(
            f"stage1_best_{STAGE1_EXPERIMENT}_fold{fold}_seed{seed}.pt"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage-1 checkpoint for fold {fold}: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected_hash = _stage1_run_hash(runtime, stage1_experiment_registry()[STAGE1_EXPERIMENT])
        observed_hash = str(payload.get("extra", {}).get("config_hash", ""))
        if observed_hash != expected_hash:
            raise RuntimeError(f"Stage-1 checkpoint identity mismatch for fold {fold}: {path.name}")
        stat = path.stat()
        checkpoints.append({
            "fold": fold, "name": path.name, "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns), "config_hash": observed_hash,
        })

    metadata_path = runtime.paths.preprocessing_file("puma_preprocessing_metadata.json")
    preprocessing = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        preprocessing = {
            "configuration_hash": metadata.get("configuration_hash"),
            "source_inventory_hash": metadata.get("source_inventory_hash"),
            "preprocessing_schema_version": metadata.get("preprocessing_schema_version"),
        }
    return {
        "oof_implementation_revision": OOF_IMPLEMENTATION_REVISION,
        "experiment": STAGE1_EXPERIMENT,
        "seed": int(seed),
        "run_folds": list(folds),
        "candidate_dtype": repr(CANDIDATE_DTYPE.descr),
        "checkpoints": checkpoints,
        "preprocessing": preprocessing,
    }


def _oof_cache_is_valid(
    output: Path,
    metadata_path: Path,
    signature: dict[str, Any],
) -> bool:
    if not output.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_signature") != signature:
            return False
        array = np.load(output, mmap_mode="r", allow_pickle=False)
        return array.dtype == CANDIDATE_DTYPE and int(metadata.get("number_of_candidates", -1)) == len(array)
    except Exception:
        return False


def generate_oof_candidates(
    runtime: RuntimeConfig,
    *,
    seed: int = 0,
    force: bool = False,
) -> Path:
    folds = validate_folds(tuple(range(runtime.data.number_of_folds)), runtime.data.number_of_folds)
    output = runtime.paths.stage1_file("stage1_oof_candidates.npy")
    metadata_path = runtime.paths.stage1_file("stage1_oof_candidates_metadata.json")
    cached_output = runtime.paths.stage1_existing_file(output.name)
    cached_metadata = runtime.paths.stage1_existing_file(metadata_path.name)
    signature = _oof_cache_signature(runtime, int(seed), folds)

    if not force and _oof_cache_is_valid(cached_output, cached_metadata, signature):
        print(f"Reusing Stage-1 OOF candidates: {cached_output}")
        return cached_output

    device = resolve_device()
    store = PumaNpyStore.open(runtime.paths.artifact_dir)
    fallback_config = stage1_experiment_registry()[STAGE1_EXPERIMENT]
    rows: list[tuple[Any, ...]] = []
    roi_candidate_counts: dict[str, int] = {}

    for fold in folds:
        checkpoint = runtime.paths.stage1_existing_file(
            f"stage1_best_{STAGE1_EXPERIMENT}_fold{fold}_seed{int(seed)}.pt"
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected_hash = _stage1_run_hash(runtime, fallback_config)
        if str(payload.get("extra", {}).get("config_hash", "")) != expected_hash:
            raise RuntimeError(f"Stage-1 checkpoint hash mismatch for OOF fold {fold}: {checkpoint}")
        config = stage1_model_config_from_dict(payload["config"]) if payload.get("config") else fallback_config
        model = build_stage1_model(config).to(device)
        payload = load_checkpoint(checkpoint, model, device)
        threshold = float(payload.get("extra", {}).get("threshold", 0.25))
        radius = int(payload.get("extra", {}).get("radius", 3))
        suppression_radius = float(payload.get("extra", {}).get("suppression_radius", 5.0))

        for roi_value in store.indices_for_fold(fold, train=False):
            roi = int(roi_value)
            prediction = predict_roi(
                model,
                np.asarray(store.images[roi]),
                runtime.data.tile_size,
                runtime.data.tile_overlap,
                device,
                threshold,
                radius,
                runtime.training.amp,
                resolve_amp_dtype(runtime.training.prefer_bfloat16, device),
                tile_batch_size=runtime.data.validation_tile_batch_size,
                suppression_radius=suppression_radius,
            )
            roi_rows = _candidate_rows_for_roi(
                roi,
                fold,
                prediction,
                store.roi_centroids(roi),
                runtime.data.official_match_radius_px,
                np.asarray(store.images[roi]).shape,
            )
            rows.extend(roi_rows)
            roi_candidate_counts[str(roi)] = len(roi_rows)

        del model, payload
        release_cuda_memory()

    array = np.asarray(rows, dtype=CANDIDATE_DTYPE)
    if len(array):
        array["oof_row_id"] = np.arange(len(array), dtype=np.int64)
    atomic_save_numpy(output, array)
    atomic_write_json(
        metadata_path,
        {
            "cache_signature": signature,
            "experiment": STAGE1_EXPERIMENT,
            "seed": int(seed),
            "run_folds": list(folds),
            "number_of_evaluated_rois": len(roi_candidate_counts),
            "number_of_zero_candidate_rois": sum(count == 0 for count in roi_candidate_counts.values()),
            "roi_candidate_counts": roi_candidate_counts,
            "number_of_candidates": len(array),
            "number_of_rejects": int(array["is_reject"].sum()) if len(array) else 0,
            "created_at": utc_now_iso(),
        },
    )
    print(f"Saved {len(array)} Stage-1 OOF candidates: {output}")
    return output
