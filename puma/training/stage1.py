from __future__ import annotations

import json
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
    save_best_checkpoint,
    seed_everything,
    utc_now_iso,
    worker_seed_init,
)

STAGE1_EXPERIMENT = "A1_IFCRN_PP"


def _stage1_run_hash(runtime: RuntimeConfig, model_config: Stage1ModelConfig) -> str:
    metadata_path = runtime.paths.preprocessing_file("puma_preprocessing_metadata.json")
    preprocessing: dict[str, Any] = {}
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        preprocessing = {
            key: payload.get(key)
            for key in (
                "preprocessing_schema_version",
                "configuration_hash",
                "source_inventory_hash",
                "number_of_rois",
                "number_of_nuclei",
            )
        }
    training = runtime.training
    return config_hash(
        {
            # Preserve the A1 run hash so completed checkpoints remain reusable.
            "model": {
                "name": model_config.name,
                "family": "ifcrn_pp",
                "learning_rate": model_config.learning_rate,
                "weight_decay": model_config.weight_decay,
                "query_count": 600,
                "hidden_dim": 256,
                "decoder_layers": 4,
                "query_init": "hybrid",
                "use_proximity": False,
                "adaptive_proximity": False,
                "faithful_downsample": 4,
                "fixed_sigma": model_config.fixed_sigma,
                "pretrained_backbone": True,
                "backbone_name": "convnextv2_tiny.fcmae_ft_in22k_in1k",
                "extra": {},
            },
            "data": runtime.data,
            "training": {
                "epochs": training.epochs,
                "effective_batch_size": training.effective_batch_size,
                "micro_batch_size": training.stage1_micro_batch_size,
                "number_of_workers": training.number_of_workers,
                "amp": training.amp,
                "prefer_bfloat16": training.prefer_bfloat16,
                "deterministic": training.deterministic,
                "validation_interval": training.validation_interval,
                "early_stopping_enabled": training.early_stopping_enabled,
                "early_stopping_patience": training.early_stopping_patience,
                "early_stopping_min_delta": training.early_stopping_min_delta,
                "gradient_clip_norm": training.gradient_clip_norm,
                "threshold_grid": training.threshold_grid,
                "local_max_radius_grid": training.local_max_radius_grid,
            },
            "execution": {"max_train_batches": None, "max_val_rois": None},
            "preprocessing": preprocessing,
        }
    )


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
) -> DecodedCandidates:
    if len(raw_scores) == 0:
        return base
    selected = _subset_candidates(base, raw_scores >= float(threshold))
    radius = float(getattr(model, "fixed_suppression_radius", 5.0))
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
    return _finalize_roi_candidates(model, base, raw_scores, threshold)


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
        )
        records.append(_prediction_record(_ground_truth_record(store, roi), prediction))
    return records


def evaluate_stage1(
    model: torch.nn.Module,
    store: PumaNpyStore,
    roi_indices: np.ndarray,
    runtime: RuntimeConfig,
    device: torch.device,
) -> tuple[float, float, int, dict[str, float]]:
    thresholds = tuple(float(value) for value in runtime.training.threshold_grid)
    radii = tuple(int(value) for value in runtime.training.local_max_radius_grid)
    if not thresholds or not radii:
        raise ValueError("Stage-1 threshold and radius grids must not be empty.")

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
    best: tuple[float, float, int, dict[str, float]] | None = None
    for threshold in thresholds:
        for radius in radii:
            records = [
                _prediction_record(
                    ground_truth,
                    _finalize_roi_candidates(model, *bases[radius], threshold),
                )
                for bases, ground_truth in cached
            ]
            metrics = {
                **evaluate_binary_detection(records, runtime.data.official_match_radius_px),
                **oracle_official_metrics(
                    records,
                    runtime.data.official_match_radius_px,
                    context=oracle_context,
                ),
            }
            score = float(metrics["oracle_macro_f1"])
            if best is None or score > best[0]:
                best = (score, threshold, radius, metrics)
    if best is None:
        raise RuntimeError("Stage-1 validation produced no metric combination.")
    return best


def train_stage1_fold(runtime: RuntimeConfig, fold: int, seed: int = 0) -> dict[str, Any]:
    model_config = stage1_experiment_registry()[STAGE1_EXPERIMENT]
    fold = int(fold)
    seed = int(seed)
    validate_folds((fold,), runtime.data.number_of_folds)
    runtime.paths.ensure()

    results_csv = runtime.paths.stage1_file("stage1_results.csv")
    key = {"stage": "stage1", "experiment": STAGE1_EXPERIMENT, "fold": fold, "seed": seed}
    run_hash = _stage1_run_hash(runtime, model_config)
    checkpoint = runtime.paths.stage1_file(
        f"stage1_best_{STAGE1_EXPERIMENT}_fold{fold}_seed{seed}.pt"
    )

    if runtime.training.resume_from_results_csv:
        completed = latest_completed_csv_row(results_csv, key)
        if completed is not None and str(completed.get("config_hash", "")) == run_hash:
            resolved = resolve_artifact_reference(
                completed.get("best_checkpoint", ""), runtime.paths.stage1_output_search_dirs()
            )
            if resolved is not None and resolved.exists():
                print(f"Stage 1 fold {fold} seed {seed}: already complete; skipping.")
                return {**key, "status": "skipped", "best_checkpoint": str(resolved)}

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

        # Outer fold is untouched until final OOF evaluation. A separate inner fold
        # controls early stopping and threshold/radius selection.
        inner_fold = select_inner_fold(fold, range(runtime.data.number_of_folds))
        train_idx = np.flatnonzero((store.folds != fold) & (store.folds != inner_fold))
        inner_idx = np.flatnonzero(store.folds == inner_fold)
        outer_idx = np.flatnonzero(store.folds == fold)
        if min(len(train_idx), len(inner_idx), len(outer_idx)) == 0:
            raise RuntimeError(
                f"Empty Stage-1 split: train={len(train_idx)}, inner={len(inner_idx)}, outer={len(outer_idx)}."
            )

        model = build_stage1_model(model_config).to(device)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)

        dataset = Stage1TileDataset(
            store,
            train_idx,
            tile_size=runtime.data.tile_size,
            tiles_per_roi=runtime.data.tiles_per_roi_per_epoch,
            seed=seed,
            augment=True,
            fixed_sigma=model_config.fixed_sigma,
            offset_radius=5.0,
        )
        loader = DataLoader(
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
        optimizer = build_adamw(
            model.parameters(),
            lr=model_config.learning_rate,
            weight_decay=model_config.weight_decay,
            device=device,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(runtime.training.epochs, 1)
        )
        amp_dtype = resolve_amp_dtype(runtime.training.prefer_bfloat16, device)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=runtime.training.amp and device.type == "cuda" and amp_dtype == torch.float16,
        )
        accumulation = runtime.training.stage1_accumulation_steps
        best_score = float("-inf")
        best_epoch = -1
        epochs_trained = 0
        stopped_early = False

        for epoch in range(1, runtime.training.epochs + 1):
            epochs_trained = epoch
            model.train()
            dataset.set_epoch(epoch)
            optimizer.zero_grad(set_to_none=True)
            running_loss = torch.zeros((), device=device)
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

            scheduler.step()
            if epoch % runtime.training.validation_interval and epoch != runtime.training.epochs:
                continue

            score, threshold, radius, metrics = evaluate_stage1(
                model, store, inner_idx, runtime, device
            )
            print(
                f"[A1 outer={fold} inner={inner_fold} seed={seed}] "
                f"epoch {epoch:02d} loss={float(running_loss.item()) / max(step, 1):.4f} "
                f"inner_oracle={score:.4f} P={metrics['binary_precision']:.4f}"
            )
            improved = np.isfinite(score) and (
                best_epoch < 0 or score > best_score + runtime.training.early_stopping_min_delta
            )
            if improved:
                best_score = score
                best_epoch = epoch
                save_best_checkpoint(
                    checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    score=score,
                    config=model_config,
                    extra={"threshold": threshold, "radius": radius, "metrics": metrics},
                )
            elif (
                runtime.training.early_stopping_enabled
                and best_epoch >= 0
                and epoch - best_epoch >= runtime.training.early_stopping_patience
            ):
                stopped_early = True
                print(f"Stage 1 fold {fold}: early stop at epoch {epoch}; best epoch {best_epoch}.")
                break

        if not checkpoint.exists():
            raise RuntimeError(f"Stage 1 fold {fold} produced no valid checkpoint.")

        optimizer.zero_grad(set_to_none=True)
        del optimizer, scheduler, scaler, loader, dataset
        release_cuda_memory(synchronize=False)

        payload = load_checkpoint(checkpoint, model, device)
        threshold = float(payload.get("extra", {}).get("threshold", 0.25))
        radius = int(payload.get("extra", {}).get("radius", 3))
        selected_epoch = int(payload.get("epoch", -1))
        inner_score = float(payload.get("score", np.nan))
        outer_records = _records_for_rois(
            model, store, outer_idx, runtime, threshold, radius, device
        )
        outer_metrics = {
            **evaluate_binary_detection(outer_records, runtime.data.official_match_radius_px),
            **oracle_official_metrics(outer_records, runtime.data.official_match_radius_px),
        }
        total, trainable = count_trainable_parameters(model)
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
            "inner_fold": inner_fold,
            "inner_best_oracle_macro_f1": inner_score,
            "epochs_trained": epochs_trained,
            "stopped_early": stopped_early,
            "parameters_total": total,
            "parameters_trainable": trainable,
            "peak_vram_mb": peak_vram_mb(),
            **outer_metrics,
        }
        append_csv_row_atomic(results_csv, completed)
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


def run_stage1_a1(runtime: RuntimeConfig) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    folds = validate_folds(runtime.training.run_folds, runtime.data.number_of_folds)
    for fold in folds:
        for seed in runtime.training.seeds:
            results.append(train_stage1_fold(runtime, fold, int(seed)))
    return results
