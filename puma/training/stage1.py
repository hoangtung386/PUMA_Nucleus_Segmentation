from __future__ import annotations

import copy
import json
import math
import random
import time
import traceback
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from puma.config import RuntimeConfig, Stage1ModelConfig, select_inner_fold, stage1_experiment_registry, validate_folds
from puma.data.datasets import (
    PumaNpyStore,
    Stage1TileDataset,
    image_to_uint8_tensor,
    normalize_image_batch,
    stage1_collate,
    tile_starts,
)
from puma.data.preprocess import validate_fold_assignments
from puma.data.targets import (
    DecodedCandidates,
    adaptive_suppress,
    decode_dense_predictions_multi_radius,
    masked_smooth_l1,
    modified_focal_loss,
)
from puma.evaluation.metrics import evaluate_binary_detection, oracle_official_metrics, prepare_oracle_context
from puma.models.stage1 import build_stage1_model
from puma.utils import (
    append_csv_row_atomic,
    build_adamw,
    clip_grad_norm_fast,
    config_hash,
    count_trainable_parameters,
    dataloader_performance_kwargs,
    latest_completed_csv_row,
    load_checkpoint,
    peak_vram_mb,
    release_cuda_memory,
    rescale_partial_accumulation_gradients,
    reset_peak_vram,
    resolve_amp_dtype,
    resolve_artifact_reference,
    resolve_device,
    restore_checkpoint_payload,
    save_best_checkpoint,
    seed_everything,
    utc_now_iso,
    worker_seed_init,
)

STAGE1_EXPERIMENT = "A1_IFCRN_PP"
STAGE1_IMPLEMENTATION_REVISION = 2


def _stage1_run_hash(runtime: RuntimeConfig, model_config: Stage1ModelConfig) -> str:
    metadata_path = runtime.paths.preprocessing_file("puma_preprocessing_metadata.json")
    preprocessing: dict[str, Any] = {}
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        preprocessing = {key: payload.get(key) for key in (
            "preprocessing_schema_version", "configuration_hash", "source_inventory_hash",
            "number_of_rois", "number_of_nuclei",
        )}
    tr = runtime.training
    return config_hash({
        "version": "13.2",
        "implementation_revision": STAGE1_IMPLEMENTATION_REVISION,
        "model": model_config,
        "data": runtime.data,
        "training": {
            "epochs": tr.stage1_epochs,
            "effective_batch_size": tr.stage1_effective_batch_size,
            # Physical micro-batch/workers are intentionally excluded: CUDA-OOM
            # fallback changes them while preserving the exact effective batch, and
            # A1 uses GroupNorm rather than batch-dependent normalization.
            "amp": tr.amp,
            "prefer_bfloat16": tr.prefer_bfloat16,
            "deterministic": tr.deterministic,
            "validation_interval": tr.validation_interval,
            "early_stopping_enabled": tr.stage1_early_stopping_enabled,
            "early_stopping_patience": tr.stage1_early_stopping_patience,
            "early_stopping_min_delta": tr.early_stopping_min_delta,
            "gradient_clip_norm": tr.gradient_clip_norm,
            "threshold_grid": tr.threshold_grid,
            "local_max_radius_grid": tr.local_max_radius_grid,
            "suppression_radius_grid": tr.suppression_radius_grid,
            "stage1_recall_tolerance": tr.stage1_recall_tolerance,
        },
        "preprocessing": preprocessing,
    })


def _capture_rng_state() -> dict[str, Any]:
    out: dict[str, Any] = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        out["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return out


def _restore_rng_state(extra: dict[str, Any]) -> None:
    if "python_rng_state" in extra: random.setstate(extra["python_rng_state"])
    if "numpy_rng_state" in extra: np.random.set_state(extra["numpy_rng_state"])
    if "torch_rng_state" in extra: torch.set_rng_state(extra["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in extra:
        torch.cuda.set_rng_state_all(extra["cuda_rng_state_all"])


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device, non_blocking=True)


def _checkpoint_hash_matches(path: Path, expected_hash: str) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return str(payload.get("extra", {}).get("config_hash", "")) == expected_hash
    except Exception:
        return False


def stage1_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    collect_terms: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    targets = batch["targets"]
    heatmap = modified_focal_loss(outputs["heatmap_logits"], targets["heatmap"])
    offset = masked_smooth_l1(outputs["offset"], targets["offset"], targets["offset_valid"])
    total = 0.7 * heatmap + 0.3 * offset
    if not collect_terms:
        return total, {}
    return total, {"heatmap": float(heatmap.detach()), "offset": float(offset.detach())}


def _empty_candidates() -> DecodedCandidates:
    return DecodedCandidates(
        coordinates=np.empty((0, 2), np.float32),
        scores=np.empty(0, np.float32),
    )


def _subset_candidates(candidates: DecodedCandidates, mask: np.ndarray) -> DecodedCandidates:
    return DecodedCandidates(candidates.coordinates[mask], candidates.scores[mask])


def _predict_roi_bases_impl(
    model: torch.nn.Module,
    image: np.ndarray,
    tile_size: int,
    overlap: int,
    device: torch.device,
    minimum_threshold: float,
    radii: tuple[int, ...],
    amp: bool,
    amp_dtype: torch.dtype,
    tile_batch_size: int,
) -> dict[int, tuple[DecodedCandidates, np.ndarray]]:
    height, width = image.shape[:2]
    radii = tuple(dict.fromkeys(int(radius) for radius in radii)) or (3,)
    buckets = {radius: {"xy": [], "score": [], "raw": []} for radius in radii}
    tile_specs = [
        (y0, x0, image[y0 : y0 + tile_size, x0 : x0 + tile_size])
        for y0 in tile_starts(height, tile_size, overlap)
        for x0 in tile_starts(width, tile_size, overlap)
    ]

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tile_specs), max(1, int(tile_batch_size))):
            specs = tile_specs[start : start + max(1, int(tile_batch_size))]
            images = torch.stack([image_to_uint8_tensor(spec[2]) for spec in specs]).to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last if device.type == "cuda" else torch.preserve_format,
            )
            images = normalize_image_batch(images)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp and device.type == "cuda",
            ):
                outputs = model(images)
            heat = outputs["heatmap_logits"].detach().sigmoid().float().cpu().numpy()
            decoded_by_radius = decode_dense_predictions_multi_radius(
                outputs,
                minimum_threshold,
                radii,
                stride=getattr(model, "output_stride", 1),
                heatmap_probabilities=heat,
            )

            for sample_index, (y0, x0, _) in enumerate(specs):
                for radius in radii:
                    decoded = decoded_by_radius[radius][sample_index]
                    if len(decoded.scores) == 0:
                        continue
                    raw_scores = decoded.scores.astype(np.float32, copy=True)
                    local = decoded.coordinates
                    margin = np.minimum.reduce(
                        [
                            local[:, 0],
                            local[:, 1],
                            tile_size - 1 - local[:, 0],
                            tile_size - 1 - local[:, 1],
                        ]
                    )
                    taper = np.clip(margin / max(overlap / 2, 1), 0.5, 1.0).astype(np.float32)
                    bucket = buckets[radius]
                    bucket["xy"].append(
                        (local + np.asarray([x0, y0], np.float32)).astype(np.float32, copy=False)
                    )
                    bucket["score"].append(raw_scores * taper)
                    bucket["raw"].append(raw_scores)
            del outputs, images, decoded_by_radius, heat

    result: dict[int, tuple[DecodedCandidates, np.ndarray]] = {}
    for radius in radii:
        bucket = buckets[radius]
        if not bucket["xy"]:
            result[radius] = (_empty_candidates(), np.empty(0, np.float32))
            continue
        merged = DecodedCandidates(
            coordinates=np.concatenate(bucket["xy"]),
            scores=np.concatenate(bucket["score"]),
        )
        raw_scores = np.concatenate(bucket["raw"])
        inside = (
            (merged.coordinates[:, 0] >= 0)
            & (merged.coordinates[:, 0] < width)
            & (merged.coordinates[:, 1] >= 0)
            & (merged.coordinates[:, 1] < height)
        )
        result[radius] = (_subset_candidates(merged, inside), raw_scores[inside])
    return result


def _predict_roi_bases(
    model: torch.nn.Module,
    image: np.ndarray,
    tile_size: int,
    overlap: int,
    device: torch.device,
    minimum_threshold: float,
    radii: tuple[int, ...],
    amp: bool,
    amp_dtype: torch.dtype,
    tile_batch_size: int = 1,
) -> dict[int, tuple[DecodedCandidates, np.ndarray]]:
    batch_size = max(1, int(tile_batch_size))
    try:
        return _predict_roi_bases_impl(
            model,
            image,
            tile_size,
            overlap,
            device,
            minimum_threshold,
            radii,
            amp,
            amp_dtype,
            batch_size,
        )
    except RuntimeError as exc:
        if device.type != "cuda" or "out of memory" not in str(exc).lower() or batch_size == 1:
            raise
        fallback = max(1, batch_size // 2)
        print(f"Stage-1 validation tile batch {batch_size} OOM; retrying with {fallback}.")
        release_cuda_memory(synchronize=False)
        return _predict_roi_bases(
            model,
            image,
            tile_size,
            overlap,
            device,
            minimum_threshold,
            radii,
            amp,
            amp_dtype,
            tile_batch_size=fallback,
        )


def _finalize_roi_candidates(
    model: torch.nn.Module,
    base: DecodedCandidates,
    raw_scores: np.ndarray,
    threshold: float,
    suppression_radius: float | None = None,
) -> DecodedCandidates:
    if len(raw_scores) == 0:
        return base
    selected = _subset_candidates(base, raw_scores >= float(threshold))
    radius = float(
        getattr(model, "fixed_suppression_radius", 5.0)
        if suppression_radius is None
        else suppression_radius
    )
    return adaptive_suppress(selected, min_radius=radius, max_radius=radius)


def predict_roi(
    model: torch.nn.Module,
    image: np.ndarray,
    tile_size: int,
    overlap: int,
    device: torch.device,
    threshold: float,
    radius: int,
    amp: bool,
    amp_dtype: torch.dtype,
    tile_batch_size: int = 1,
    suppression_radius: float | None = None,
) -> DecodedCandidates:
    base, raw_scores = _predict_roi_bases(
        model,
        image,
        tile_size,
        overlap,
        device,
        float(threshold),
        (int(radius),),
        amp,
        amp_dtype,
        tile_batch_size=tile_batch_size,
    )[int(radius)]
    return _finalize_roi_candidates(
        model, base, raw_scores, threshold, suppression_radius=suppression_radius
    )


def _ground_truth_record(store: PumaNpyStore, roi: int) -> dict[str, Any]:
    gt = store.roi_centroids(roi)
    return {
        "roi_id": str(store.manifest[roi]["roi_id"]),
        "patient_id": str(store.manifest[roi]["case_id"]),
        "gt_xy": np.column_stack([gt["x"], gt["y"]]).astype(np.float32),
        "gt_class": gt["class_id"].astype(int),
        "gt_extent": np.column_stack([gt["width"], gt["height"]]).astype(np.float32),
        "gt_nearest": gt["nearest_neighbor_distance"].astype(np.float32),
    }


def _prediction_record(ground_truth: dict[str, Any], prediction: DecodedCandidates) -> dict[str, Any]:
    return {**ground_truth, "pred_xy": prediction.coordinates, "pred_scores": prediction.scores}


def _records_for_rois(
    model: torch.nn.Module,
    store: PumaNpyStore,
    roi_indices: np.ndarray,
    runtime: RuntimeConfig,
    threshold: float,
    radius: int,
    suppression_radius: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    amp_dtype = resolve_amp_dtype(runtime.training.prefer_bfloat16, device)
    records: list[dict[str, Any]] = []
    for roi_value in roi_indices:
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
            amp_dtype,
            tile_batch_size=runtime.data.validation_tile_batch_size,
            suppression_radius=suppression_radius,
        )
        records.append(_prediction_record(_ground_truth_record(store, roi), prediction))
    return records


def evaluate_stage1(
    model: torch.nn.Module,
    store: PumaNpyStore,
    roi_indices: np.ndarray,
    runtime: RuntimeConfig,
    device: torch.device,
) -> tuple[float, float, int, float, dict[str, float]]:
    """Select detector post-processing with a small recall preference near the best ceiling.

    We first maximize the Stage-1 oracle macro-F1 ceiling.  Any configuration within
    ``stage1_recall_tolerance`` of that maximum is considered statistically equivalent for
    model selection, and among those we prefer higher tail recall, then global recall.
    This biases the cascade mildly toward recall without accepting a materially worse F1.
    """
    thresholds = tuple(float(value) for value in runtime.training.threshold_grid)
    radii = tuple(int(value) for value in runtime.training.local_max_radius_grid)
    suppression_radii = tuple(float(value) for value in runtime.training.suppression_radius_grid)
    if not thresholds or not radii or not suppression_radii:
        raise ValueError("Stage-1 threshold/local-max/suppression grids must not be empty.")

    amp_dtype = resolve_amp_dtype(runtime.training.prefer_bfloat16, device)
    cached: list[tuple[dict[int, tuple[DecodedCandidates, np.ndarray]], dict[str, Any]]] = []
    for roi_value in roi_indices:
        roi = int(roi_value)
        bases = _predict_roi_bases(
            model,
            np.asarray(store.images[roi]),
            runtime.data.tile_size,
            runtime.data.tile_overlap,
            device,
            min(thresholds),
            radii,
            runtime.training.amp,
            amp_dtype,
            tile_batch_size=runtime.data.validation_tile_batch_size,
        )
        cached.append((bases, _ground_truth_record(store, roi)))

    oracle_context = prepare_oracle_context([ground_truth for _, ground_truth in cached])
    candidates: list[tuple[float, float, int, float, dict[str, float]]] = []
    for threshold in thresholds:
        for radius in radii:
            base_rows = [(bases[radius], ground_truth) for bases, ground_truth in cached]
            for suppression_radius in suppression_radii:
                records = [
                    _prediction_record(
                        ground_truth,
                        _finalize_roi_candidates(
                            model,
                            *base,
                            threshold,
                            suppression_radius=suppression_radius,
                        ),
                    )
                    for base, ground_truth in base_rows
                ]
                metrics = {
                    **evaluate_binary_detection(records, runtime.data.official_match_radius_px),
                    **oracle_official_metrics(
                        records,
                        runtime.data.official_match_radius_px,
                        context=oracle_context,
                    ),
                }
                candidates.append(
                    (
                        float(metrics["oracle_macro_f1"]),
                        float(threshold),
                        int(radius),
                        float(suppression_radius),
                        metrics,
                    )
                )
    if not candidates:
        raise RuntimeError("Stage-1 validation produced no metric combination.")
    best_ceiling = max(item[0] for item in candidates)
    tolerance = max(0.0, float(runtime.training.stage1_recall_tolerance))
    shortlist = [item for item in candidates if item[0] >= best_ceiling - tolerance]
    shortlist.sort(
        key=lambda item: (
            float(item[4].get("tail_detection_recall", 0.0)),
            float(item[4].get("binary_recall", 0.0)),
            item[0],
            float(item[4].get("binary_precision", 0.0)),
        ),
        reverse=True,
    )
    return shortlist[0]


def _stage1_lr(base_lr: float, epoch: int, total_epochs: int, *, warmup_epochs: int = 2) -> float:
    minimum_lr = min(1e-5, base_lr)
    if epoch <= warmup_epochs:
        return base_lr * float(epoch) / float(max(warmup_epochs, 1))
    progress = (epoch - warmup_epochs - 1) / max(total_epochs - warmup_epochs - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return minimum_lr + (base_lr - minimum_lr) * cosine


def _make_stage1_dataset(store: PumaNpyStore, roi_indices: np.ndarray, runtime: RuntimeConfig, seed: int, fixed_sigma: float) -> Stage1TileDataset:
    return Stage1TileDataset(
        store,
        roi_indices,
        tile_size=runtime.data.tile_size,
        tiles_per_roi=runtime.data.tiles_per_roi_per_epoch,
        seed=seed,
        augment=True,
        fixed_sigma=fixed_sigma,
        offset_radius=5.0,
        background_fraction=runtime.data.background_fraction,
        density_fraction=runtime.data.density_fraction,
        small_nucleus_fraction=runtime.data.small_nucleus_fraction,
        uniform_fraction=runtime.data.uniform_fraction,
        rare_nucleus_fraction=runtime.data.rare_nucleus_fraction,
    )


def _make_stage1_loader(dataset: Stage1TileDataset, runtime: RuntimeConfig, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=runtime.training.stage1_micro_batch_size,
        shuffle=True,
        collate_fn=stage1_collate,
        worker_init_fn=worker_seed_init,
        **dataloader_performance_kwargs(
            runtime.training.number_of_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=True,
        ),
    )


def _train_stage1_epoch(
    model: torch.nn.Module,
    dataset: Stage1TileDataset,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    runtime: RuntimeConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
    epoch: int,
) -> float:
    model.train()
    dataset.set_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)
    running_loss = torch.zeros((), device=device)
    accumulation = runtime.training.stage1_accumulation_steps
    step = 0
    for step, batch in enumerate(loader, 1):
        images = batch["image"].to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last if device.type == "cuda" else torch.preserve_format,
        )
        batch["image"] = normalize_image_batch(images, batch.get("stain_parameters"))
        batch["targets"] = {
            key: value.to(device, non_blocking=True) for key, value in batch["targets"].items()
        }
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=runtime.training.amp and device.type == "cuda",
        ):
            outputs = model(batch["image"])
            loss, _ = stage1_loss(outputs, batch, collect_terms=False)
            scaled_loss = loss / accumulation
        scaler.scale(scaled_loss).backward()
        running_loss.add_(loss.detach())
        if step % accumulation == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_fast(model.parameters(), runtime.training.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        del outputs, loss, scaled_loss, batch, images
    if step and step % accumulation:
        scaler.unscale_(optimizer)
        rescale_partial_accumulation_gradients(
            model.parameters(),
            accumulation_steps=accumulation,
            microbatches_in_group=step % accumulation,
        )
        clip_grad_norm_fast(model.parameters(), runtime.training.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return float(running_loss.item()) / max(step, 1)


def _train_stage1_fold_once(runtime: RuntimeConfig, fold: int, seed: int = 0) -> dict[str, Any]:
    model_config = stage1_experiment_registry()[STAGE1_EXPERIMENT]
    runtime.paths.ensure()
    results_csv = runtime.paths.stage1_file("stage1_results.csv")
    key = {"stage": "stage1", "experiment": STAGE1_EXPERIMENT, "fold": fold, "seed": seed}
    run_hash = _stage1_run_hash(runtime, model_config)
    checkpoint = runtime.paths.stage1_file(
        f"stage1_best_{STAGE1_EXPERIMENT}_fold{fold}_seed{seed}.pt"
    )
    inner_checkpoint = runtime.paths.stage1_file(
        f"stage1_inner_{STAGE1_EXPERIMENT}_fold{fold}_seed{seed}.pt"
    )
    selection_resume = runtime.paths.stage1_file(
        f"stage1_resume_selection_{STAGE1_EXPERIMENT}_fold{fold}_seed{seed}.pt"
    )
    refit_resume = runtime.paths.stage1_file(
        f"stage1_resume_refit_{STAGE1_EXPERIMENT}_fold{fold}_seed{seed}.pt"
    )

    if runtime.training.resume_from_results_csv:
        completed = latest_completed_csv_row(results_csv, key)
        if completed is not None and str(completed.get("config_hash", "")) == run_hash:
            resolved = resolve_artifact_reference(
                completed.get("best_checkpoint", ""), runtime.paths.stage1_output_search_dirs()
            )
            if resolved is not None and resolved.exists() and _checkpoint_hash_matches(resolved, run_hash):
                print(f"Stage 1 fold {fold} seed {seed}: already complete; skipping.")
                return {**key, "status": "skipped", "best_checkpoint": str(resolved)}
            print(f"Stage 1 fold {fold} seed {seed}: completed record is incomplete/stale; rebuilding.")

    append_csv_row_atomic(
        results_csv,
        {**key, "status": "running", "started_at": utc_now_iso(), "config_hash": run_hash},
    )
    started = time.time()
    try:
        seed_everything(seed, runtime.training.deterministic)
        device = resolve_device()
        reset_peak_vram()
        store = PumaNpyStore.open(runtime.paths.artifact_dir)
        inner_fold = select_inner_fold(fold, range(runtime.data.number_of_folds))
        train_idx = np.flatnonzero((store.folds != fold) & (store.folds != inner_fold))
        inner_idx = np.flatnonzero(store.folds == inner_fold)
        outer_idx = np.flatnonzero(store.folds == fold)
        refit_idx = np.flatnonzero(store.folds != fold)
        if min(len(train_idx), len(inner_idx), len(outer_idx), len(refit_idx)) == 0:
            raise RuntimeError(
                f"Empty Stage-1 split: train={len(train_idx)}, inner={len(inner_idx)}, "
                f"refit={len(refit_idx)}, outer={len(outer_idx)}."
            )

        # ---- 3/5 train + 1/5 inner selection ----
        model = build_stage1_model(model_config).to(device)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        dataset = _make_stage1_dataset(store, train_idx, runtime, seed, model_config.fixed_sigma)
        loader = _make_stage1_loader(dataset, runtime, device)
        optimizer = build_adamw(
            model.parameters(), lr=model_config.learning_rate,
            weight_decay=model_config.weight_decay, device=device
        )
        amp_dtype = resolve_amp_dtype(runtime.training.prefer_bfloat16, device)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=runtime.training.amp and device.type == "cuda" and amp_dtype == torch.float16,
        )
        best_score = float("-inf")
        best_epoch = -1
        epochs_trained = 0
        stopped_early = False
        selection_start_epoch = 1
        if selection_resume.exists():
            payload = torch.load(selection_resume, map_location="cpu", weights_only=False)
            extra_resume = dict(payload.get("extra", {}))
            resumable = (
                str(extra_resume.get("config_hash", "")) == run_hash
                and str(extra_resume.get("resume_phase", "")) == "selection"
                and (int(extra_resume.get("best_epoch", -1)) < 0 or inner_checkpoint.exists())
            )
            if resumable:
                restore_checkpoint_payload(payload, model)
                if payload.get("optimizer_state") is not None:
                    optimizer.load_state_dict(payload["optimizer_state"]); _move_optimizer_state_to_device(optimizer, device)
                if payload.get("scaler_state") is not None: scaler.load_state_dict(payload["scaler_state"])
                selection_complete = bool(extra_resume.get("selection_complete", False))
                selection_start_epoch = (
                    runtime.training.stage1_epochs + 1
                    if selection_complete else int(payload.get("epoch", 0)) + 1
                )
                epochs_trained = int(payload.get("epoch", 0))
                best_score = float(extra_resume.get("best_score", float("-inf")))
                best_epoch = int(extra_resume.get("best_epoch", -1))
                # Reconcile a newer inner-best file if interruption happened between
                # saving the best model and saving the resume state.
                if inner_checkpoint.exists() and _checkpoint_hash_matches(inner_checkpoint, run_hash):
                    inner_now = torch.load(inner_checkpoint, map_location="cpu", weights_only=False)
                    if float(inner_now.get("score", float("-inf"))) > best_score:
                        best_score = float(inner_now["score"]); best_epoch = int(inner_now["epoch"])
                    del inner_now
                _restore_rng_state(extra_resume)
                print(f"RESUME Stage-1 selection fold {fold}: epoch {selection_start_epoch}")
            else:
                selection_resume.unlink(missing_ok=True); inner_checkpoint.unlink(missing_ok=True)
            del payload
        elif inner_checkpoint.exists():
            inner_checkpoint.unlink(missing_ok=True)

        for epoch in range(selection_start_epoch, runtime.training.stage1_epochs + 1):
            epochs_trained = epoch
            lr = _stage1_lr(model_config.learning_rate, epoch, runtime.training.stage1_epochs)
            for group in optimizer.param_groups:
                group["lr"] = lr
            train_loss = _train_stage1_epoch(
                model, dataset, loader, optimizer, scaler, runtime, device, amp_dtype, epoch
            )
            do_validate = not (epoch % runtime.training.validation_interval and epoch != runtime.training.stage1_epochs)
            if do_validate:
                score, threshold, radius, suppression_radius, metrics = evaluate_stage1(
                    model, store, inner_idx, runtime, device
                )
                print(
                    f"[A1 V13.2 outer={fold} inner={inner_fold} seed={seed}] "
                    f"epoch {epoch:02d} loss={train_loss:.4f} inner_oracle={score:.4f} "
                    f"P={metrics['binary_precision']:.4f} R={metrics['binary_recall']:.4f} "
                    f"tailR={metrics.get('tail_detection_recall', float('nan')):.4f}"
                )
                improved = np.isfinite(score) and (
                    best_epoch < 0 or score > best_score + runtime.training.early_stopping_min_delta
                )
                if improved:
                    best_score, best_epoch = score, epoch
                    save_best_checkpoint(
                        inner_checkpoint, model=model, optimizer=None, scheduler=None, scaler=None,
                        epoch=epoch, score=score, config=model_config,
                        extra={
                            "threshold": threshold, "radius": radius,
                            "suppression_radius": suppression_radius, "metrics": metrics,
                            "inner_fold": inner_fold,
                            "selection_train_folds": sorted(set(range(runtime.data.number_of_folds)) - {fold, inner_fold}),
                            "config_hash": run_hash,
                        },
                    )
                elif (
                    runtime.training.stage1_early_stopping_enabled and best_epoch >= 0
                    and epoch - best_epoch >= runtime.training.stage1_early_stopping_patience
                ):
                    stopped_early = True
            if epoch % int(runtime.training.resume_checkpoint_interval) == 0 or stopped_early or epoch == runtime.training.stage1_epochs:
                resume_extra = {
                    "config_hash": run_hash, "resume_phase": "selection",
                    "best_score": best_score, "best_epoch": best_epoch, "inner_fold": inner_fold,
                    "selection_complete": bool(stopped_early or epoch == runtime.training.stage1_epochs),
                }
                resume_extra.update(_capture_rng_state())
                save_best_checkpoint(
                    selection_resume, model=model, optimizer=optimizer, scheduler=None, scaler=scaler,
                    epoch=epoch, score=best_score, config=model_config, extra=resume_extra,
                    include_training_state=True,
                )
            if stopped_early:
                break
        if not inner_checkpoint.exists():
            raise RuntimeError(f"Stage 1 fold {fold} produced no inner-selection checkpoint.")
        inner_payload = torch.load(inner_checkpoint, map_location="cpu", weights_only=False)
        if str(inner_payload.get("extra", {}).get("config_hash", "")) != run_hash:
            raise RuntimeError(f"Stage-1 inner checkpoint hash mismatch for fold {fold}.")
        selected_epoch = int(inner_payload["epoch"])
        inner_score = float(inner_payload["score"])
        extra = dict(inner_payload.get("extra", {}))
        threshold = float(extra.get("threshold", 0.25))
        radius = int(extra.get("radius", 3))
        suppression_radius = float(extra.get("suppression_radius", 5.0))
        inner_metrics = dict(extra.get("metrics", {}))
        # Keep the selection resume marker until the fold is fully refit. This closes
        # the interruption gap between inner selection and the first refit epoch.
        del inner_payload, optimizer, scaler, loader, dataset, model
        release_cuda_memory(synchronize=False)

        # ---- V13.2 refit: all 4 non-outer folds, exact selected epoch count ----
        seed_everything(seed + 100_003 + fold, runtime.training.deterministic)
        refit_model = build_stage1_model(model_config).to(device)
        if device.type == "cuda":
            refit_model = refit_model.to(memory_format=torch.channels_last)
        refit_dataset = _make_stage1_dataset(
            store, refit_idx, runtime, seed + 100_003 + fold, model_config.fixed_sigma
        )
        refit_loader = _make_stage1_loader(refit_dataset, runtime, device)
        refit_optimizer = build_adamw(
            refit_model.parameters(), lr=model_config.learning_rate,
            weight_decay=model_config.weight_decay, device=device
        )
        refit_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=runtime.training.amp and device.type == "cuda" and amp_dtype == torch.float16,
        )
        refit_start_epoch = 1
        if refit_resume.exists():
            payload = torch.load(refit_resume, map_location="cpu", weights_only=False)
            extra_resume = dict(payload.get("extra", {}))
            if (str(extra_resume.get("config_hash", "")) == run_hash
                and str(extra_resume.get("resume_phase", "")) == "refit"
                and int(extra_resume.get("selected_epoch", -1)) == selected_epoch):
                restore_checkpoint_payload(payload, refit_model)
                if payload.get("optimizer_state") is not None:
                    refit_optimizer.load_state_dict(payload["optimizer_state"]); _move_optimizer_state_to_device(refit_optimizer, device)
                if payload.get("scaler_state") is not None: refit_scaler.load_state_dict(payload["scaler_state"])
                refit_start_epoch = int(payload.get("epoch", 0)) + 1
                _restore_rng_state(extra_resume)
                print(f"RESUME Stage-1 refit fold {fold}: epoch {refit_start_epoch}/{selected_epoch}")
            else:
                refit_resume.unlink(missing_ok=True)
            del payload
        for epoch in range(refit_start_epoch, selected_epoch + 1):
            lr = _stage1_lr(model_config.learning_rate, epoch, selected_epoch)
            for group in refit_optimizer.param_groups:
                group["lr"] = lr
            refit_loss = _train_stage1_epoch(
                refit_model, refit_dataset, refit_loader, refit_optimizer,
                refit_scaler, runtime, device, amp_dtype, epoch
            )
            if epoch == 1 or epoch == selected_epoch or epoch % 5 == 0:
                print(
                    f"[A1 V13.2 REFIT outer={fold} seed={seed}] epoch {epoch:02d}/{selected_epoch:02d} "
                    f"loss={refit_loss:.4f}"
                )
            if epoch % int(runtime.training.resume_checkpoint_interval) == 0 or epoch == selected_epoch:
                resume_extra = {
                    "config_hash": run_hash, "resume_phase": "refit",
                    "selected_epoch": selected_epoch, "inner_fold": inner_fold,
                }
                resume_extra.update(_capture_rng_state())
                save_best_checkpoint(
                    refit_resume, model=refit_model, optimizer=refit_optimizer, scheduler=None, scaler=refit_scaler,
                    epoch=epoch, score=inner_score, config=model_config, extra=resume_extra,
                    include_training_state=True,
                )

        save_best_checkpoint(
            checkpoint,
            model=refit_model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            epoch=selected_epoch,
            score=inner_score,
            config=model_config,
            extra={
                "threshold": threshold,
                "radius": radius,
                "suppression_radius": suppression_radius,
                "metrics": inner_metrics,
                "inner_fold": inner_fold,
                "refit": True,
                "refit_folds": sorted(set(range(runtime.data.number_of_folds)) - {fold}),
                "outer_fold_untouched": fold,
                "selection_epoch": selected_epoch,
                "refit_epochs": selected_epoch,
                "config_hash": run_hash,
            },
        )

        outer_records = _records_for_rois(
            refit_model, store, outer_idx, runtime, threshold, radius, suppression_radius, device
        )
        outer_metrics = {
            **evaluate_binary_detection(outer_records, runtime.data.official_match_radius_px),
            **oracle_official_metrics(outer_records, runtime.data.official_match_radius_px),
        }
        total, trainable = count_trainable_parameters(refit_model)
        completed = {
            **key,
            "config_hash": run_hash,
            "status": "completed",
            "completed_at": utc_now_iso(),
            "duration_minutes": (time.time() - started) / 60.0,
            "best_epoch": selected_epoch,
            "best_checkpoint": str(checkpoint),
            "threshold": threshold,
            "local_max_radius": radius,
            "suppression_radius": suppression_radius,
            "inner_fold": inner_fold,
            "inner_best_oracle_macro_f1": inner_score,
            "epochs_trained_selection": epochs_trained,
            "refit_epochs": selected_epoch,
            "refit_roi_count": len(refit_idx),
            "stage1_effective_batch_size": runtime.training.stage1_effective_batch_size,
            "stage1_micro_batch_size": runtime.training.stage1_micro_batch_size,
            "stopped_early": stopped_early,
            "parameters_total": total,
            "parameters_trainable": trainable,
            "peak_vram_mb": peak_vram_mb(),
            **outer_metrics,
        }
        append_csv_row_atomic(results_csv, completed)
        inner_checkpoint.unlink(missing_ok=True)
        selection_resume.unlink(missing_ok=True)
        refit_resume.unlink(missing_ok=True)
        return completed
    except Exception as exc:
        append_csv_row_atomic(
            results_csv,
            {
                **key,
                "config_hash": run_hash,
                "status": "failed",
                "completed_at": utc_now_iso(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc()[-12000:],
            },
        )
        raise
    finally:
        release_cuda_memory()


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "cuda out of memory" in text or "outofmemoryerror" in text or "cuda error: out of memory" in text


def train_stage1_fold(runtime: RuntimeConfig, fold: int, seed: int = 0) -> dict[str, Any]:
    """Train/refit one OOF fold with automatic physical-batch fallback.

    Effective Stage-1 batch remains exactly 16.  The supplied configuration starts with
    physical batch 16 for maximum throughput and falls back to 8/4/2/1 only on CUDA OOM.
    """
    initial = int(runtime.training.stage1_micro_batch_size)
    effective = int(runtime.training.stage1_effective_batch_size)
    attempts = [initial] + [v for v in (8, 4, 2, 1) if v < initial and effective % v == 0]
    last_error: Exception | None = None
    for attempt, micro in enumerate(dict.fromkeys(attempts)):
        run_runtime = copy.deepcopy(runtime)
        run_runtime.training.stage1_micro_batch_size = int(micro)
        if attempt:
            print(
                f"Stage-1 CUDA-OOM fallback fold {fold}: micro batch {micro}; "
                f"effective batch remains {effective}."
            )
            # Remove any incomplete final/inner checkpoint from the failed physical batch.
            run_runtime.paths.stage1_file(
                f"stage1_best_{STAGE1_EXPERIMENT}_fold{fold}_seed{seed}.pt"
            ).unlink(missing_ok=True)
            run_runtime.paths.stage1_file(
                f"stage1_inner_{STAGE1_EXPERIMENT}_fold{fold}_seed{seed}.pt"
            ).unlink(missing_ok=True)
            run_runtime.paths.stage1_file(
                f"stage1_resume_selection_{STAGE1_EXPERIMENT}_fold{fold}_seed{seed}.pt"
            ).unlink(missing_ok=True)
            run_runtime.paths.stage1_file(
                f"stage1_resume_refit_{STAGE1_EXPERIMENT}_fold{fold}_seed{seed}.pt"
            ).unlink(missing_ok=True)
        try:
            return _train_stage1_fold_once(run_runtime, fold, seed)
        except Exception as exc:
            last_error = exc
            if not (_is_cuda_oom(exc) and attempt + 1 < len(attempts)):
                raise
            release_cuda_memory()
    assert last_error is not None
    raise last_error


def run_stage1_a1(runtime: RuntimeConfig) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    folds = validate_folds(runtime.training.run_folds, runtime.data.number_of_folds)
    # Every fold serves twice: as the untouched outer fold that receives the OOF
    # prediction, and as some other fold's inner split for checkpoint, threshold, and
    # radius selection. A degenerate split still trains and reports without error, so it
    # has to be rejected here rather than discovered in the results.
    store = PumaNpyStore.open(runtime.paths.artifact_dir)
    validate_fold_assignments(np.asarray(store.folds), runtime.data.number_of_folds)
    for fold in folds:
        for seed in runtime.training.seeds:
            results.append(train_stage1_fold(runtime, fold, int(seed)))
    return results
