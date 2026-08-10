from __future__ import annotations

"""Final V13 training and deployment-lock helpers."""

import copy
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from puma.config import (
    PUMA_CLASS_NAMES,
    REJECT_CLASS_ID,
    RuntimeConfig,
    stage1_model_config_from_dict,
    Stage2ModelConfig,
    stage1_experiment_registry,
)
from puma.data.datasets import PumaNpyStore, Stage2CandidateDataset
from puma.models.stage2 import build_stage2_model, ensure_stage2_pretrained_checkpoints, split_optimizer_parameters
from puma.pipeline.oof import validate_full_oof
from puma.stage2.catalog import stage2_experiment_registry
from puma.training.stage2 import (
    _build_perfect_candidates,
    _forward_batch_streamed,
    _hard_reject_scores,
    _phase_epoch_number,
    _phase_for_epoch,
    _warmup_cosine_scheduler,
)
from puma.training.stage2_v13 import (
    _make_v13_epoch_encoder_cache,
    _make_v13_loader,
    _reset_v13_loader,
    v13_stage2_loss,
)
from puma.utils import (
    atomic_write_json,
    build_adamw_parameter_groups,
    clip_grad_norm_fast,
    config_hash,
    count_trainable_parameters,
    peak_vram_mb,
    release_cuda_memory,
    rescale_partial_accumulation_gradients,
    reset_peak_vram,
    resolve_amp_dtype,
    resolve_device,
    save_best_checkpoint,
    seed_everything,
    utc_now_iso,
)

FINAL_STAGE1_EXPERIMENT = "A1_IFCRN_PP"
FINAL_STAGE1_SEED = 0
V13_FINAL_LOCK_NAME = "stage2_v13_final_lock.json"
V13_FINAL_CHECKPOINT_PREFIX = "stage2_v13_final"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_fixed_stage1_lock_v13(
    runtime: RuntimeConfig,
    *,
    experiment: str = FINAL_STAGE1_EXPERIMENT,
    seed: int = FINAL_STAGE1_SEED,
    folds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Write the fixed five-fold A1 deployment lock."""
    registry = stage1_experiment_registry()
    if experiment not in registry:
        raise KeyError(f"Fixed Stage-1 experiment {experiment!r} is not in the registry.")
    expected_cfg = registry[experiment]
    folds = tuple(range(runtime.data.number_of_folds)) if folds is None else tuple(int(v) for v in folds)
    if set(folds) != set(range(runtime.data.number_of_folds)):
        raise ValueError(
            "V13 deployment expects all Stage-1 OOF folds for the detector ensemble; "
            f"got {folds}."
        )

    checkpoint_rows: list[dict[str, Any]] = []
    for fold in folds:
        path = runtime.paths.stage1_existing_file(
            f"stage1_best_{experiment}_fold{fold}_seed{int(seed)}.pt"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Missing fixed Stage-1 checkpoint for fold {fold}: {path}"
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        serialized = payload.get("config")
        cfg = stage1_model_config_from_dict(serialized) if serialized else expected_cfg
        if config_hash(cfg) != config_hash(expected_cfg):
            raise RuntimeError(
                f"Stage-1 checkpoint config mismatch at {path}. Expected fixed {experiment}."
            )
        checkpoint_rows.append({
            "fold": int(fold),
            "checkpoint": str(path),
            "threshold": float(payload.get("extra", {}).get("threshold", 0.25)),
            "radius": int(payload.get("extra", {}).get("radius", 3)),
        })

    lock = {
        "version": 13,
        "selected_experiment": experiment,
        "selected_at": utc_now_iso(),
        "selection_mode": "fixed_a1_oof",
        "selected_model_config": asdict(expected_cfg),
        "run_folds": [int(v) for v in folds],
        "seeds": [int(seed)],
        "inference_mode": "five_fold_ensemble",
        "checkpoints": checkpoint_rows,
    }
    atomic_write_json(runtime.paths.stage1_file("stage1_lock.json"), lock)
    return lock


def _final_recipe(
    runtime: RuntimeConfig,
    *,
    selected_experiment: str | None,
    final_epochs: int | None,
    validity_threshold: float | None,
) -> tuple[dict[str, Any], Stage2ModelConfig, int, int, float]:
    dev_lock_path = runtime.paths.stage2_existing_file("stage2_v13_lock.json")
    dev_lock = _load_json(dev_lock_path)
    selected = str(selected_experiment or dev_lock.get("selected_experiment", ""))
    registry = stage2_experiment_registry()
    if selected not in registry:
        raise KeyError(
            f"V13 selected experiment {selected!r} is not available. "
            "Lock a completed V13 winner first."
        )
    locked_cfg = dev_lock.get("selected_model_config")
    cfg = Stage2ModelConfig(**locked_cfg) if isinstance(locked_cfg, dict) else registry[selected]
    if cfg.name != selected:
        raise RuntimeError(
            f"V13 development lock/config mismatch: selected={selected!r}, config={cfg.name!r}."
        )
    plan = dict(dev_lock.get("final_training_plan") or {})
    epochs = int(final_epochs if final_epochs is not None else plan.get("recommended_epochs", 0))
    if epochs < 1:
        raise ValueError(
            "Final epoch count is unavailable. Recreate stage2_v13_lock.json with the updated V13 code "
            "or pass final_epochs explicitly."
        )
    schedule_reference_epochs = int(plan.get("schedule_reference_epochs", runtime.training.epochs))
    if schedule_reference_epochs < 1:
        raise ValueError("schedule_reference_epochs must be positive.")
    threshold = float(
        validity_threshold if validity_threshold is not None else plan.get("validity_threshold", 0.5)
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("validity_threshold must be in [0,1].")
    return dev_lock, cfg, epochs, schedule_reference_epochs, threshold


def _train_final_stage2_once(
    runtime: RuntimeConfig,
    cfg: Stage2ModelConfig,
    *,
    hf_token: str | None,
    final_epochs: int,
    schedule_reference_epochs: int,
    validity_threshold: float,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    paths = runtime.paths
    paths.ensure()
    validate_full_oof(runtime)
    ensure_stage2_pretrained_checkpoints(paths.root, hf_token=hf_token, pfm_keys=(cfg.pfm_key,))
    write_fixed_stage1_lock_v13(runtime)

    checkpoint = paths.stage2_file(f"{V13_FINAL_CHECKPOINT_PREFIX}_{cfg.name}_seed{seed}.pt")
    final_lock_path = paths.stage2_file(V13_FINAL_LOCK_NAME)
    if checkpoint.exists() and final_lock_path.exists() and not force:
        existing = _load_json(final_lock_path)
        if (
            str(existing.get("selected_experiment")) == cfg.name
            and int(existing.get("final_epochs", -1)) == int(final_epochs)
            and int(existing.get("schedule_reference_epochs", -1)) == int(schedule_reference_epochs)
        ):
            print(f"REUSE completed V13 final checkpoint: {checkpoint}")
            return existing

    seed_everything(seed, runtime.training.deterministic)
    device = resolve_device()
    reset_peak_vram()
    store = PumaNpyStore.open(paths.artifact_dir)
    all_candidates = np.load(paths.stage1_existing_file("stage1_oof_candidates.npy"), mmap_mode="r")
    observed_folds = set(np.unique(all_candidates["fold"]).astype(int).tolist())
    expected_folds = set(range(runtime.data.number_of_folds))
    if observed_folds != expected_folds:
        raise RuntimeError(
            f"Final V13 training requires complete OOF candidates for folds {sorted(expected_folds)}, "
            f"got {sorted(observed_folds)}."
        )
    train_base = np.asarray(all_candidates)
    train_oof_positive = train_base[train_base["class_id"] != REJECT_CLASS_ID]
    manifest = np.load(paths.preprocessing_file("puma_roi_manifest.npy"), mmap_mode="r", allow_pickle=False)
    all_roi_indices = np.arange(len(manifest), dtype=np.int64)
    perfect_train = _build_perfect_candidates(
        store,
        train_base,
        seed=seed,
        background_fraction=0.0,
        roi_indices=all_roi_indices,
    )

    model = build_stage2_model(cfg, hf_token=hf_token).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    head_parameters, lora_parameters = split_optimizer_parameters(model)
    trainable_parameters = [*head_parameters, *lora_parameters]
    groups: list[dict[str, Any]] = [{
        "params": head_parameters,
        "lr": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
    }]
    if lora_parameters:
        groups.append({
            "params": lora_parameters,
            "lr": cfg.lora_learning_rate,
            "weight_decay": 0.0,
        })
    optimizer = build_adamw_parameter_groups(groups, device=device)
    # Preserve the development schedule horizon. The final run stops at the selected
    # development epoch instead of compressing phase boundaries into a shorter run.
    scheduler = _warmup_cosine_scheduler(optimizer, schedule_reference_epochs, cfg.warmup_epochs)
    amp_dtype = resolve_amp_dtype(runtime.training.prefer_bfloat16, device)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(runtime.training.amp and device.type == "cuda" and amp_dtype == torch.float16),
    )
    accumulation = runtime.training.stage2_accumulation_steps

    type_counts = np.bincount(
        perfect_train["class_id"].astype(int), minlength=REJECT_CLASS_ID
    )[:REJECT_CLASS_ID]
    missing = [PUMA_CLASS_NAMES[i] for i in np.flatnonzero(type_counts == 0).astype(int)]
    if missing:
        raise RuntimeError("Final V13 training data is missing class(es): " + ", ".join(missing))
    reject_count = int(np.sum(train_base["class_id"] == REJECT_CLASS_ID))
    class_counts = torch.as_tensor(
        np.r_[type_counts, max(reject_count, 1)], dtype=torch.float32, device=device
    )

    persistent_phase: str | None = None
    persistent_dataset: Stage2CandidateDataset | None = None
    persistent_loader = None
    started = time.time()
    epochs_trained = 0
    try:
        for epoch in range(1, final_epochs + 1):
            epochs_trained = epoch
            phase = _phase_for_epoch(cfg.schedule_key, epoch, schedule_reference_epochs)
            phase_epoch = _phase_epoch_number(cfg.schedule_key, epoch, schedule_reference_epochs)
            if phase == "GT_POS":
                phase_data = perfect_train
            elif phase == "OOF_POS":
                phase_data = train_oof_positive
            elif phase == "OOF_ALL":
                phase_data = train_base
            else:
                raise RuntimeError(f"Unsupported final V13 phase {phase!r}")

            hard_scores = None
            if (
                phase == "OOF_ALL"
                and phase_epoch >= int(cfg.hard_negative_start_phase_epoch)
                and np.any(phase_data["class_id"] == REJECT_CLASS_ID)
            ):
                hard_scores = _hard_reject_scores(
                    model,
                    store,
                    phase_data,
                    cfg,
                    device,
                    batch_size=runtime.training.stage2_micro_batch_size,
                    workers=runtime.training.number_of_workers,
                    amp=runtime.training.amp,
                    prefer_bfloat16=runtime.training.prefer_bfloat16,
                )

            sampling_seed = seed + epoch * 10007
            if (
                persistent_phase == phase
                and persistent_dataset is not None
                and persistent_loader is not None
            ):
                dataset, loader = persistent_dataset, persistent_loader
                dataset.set_epoch(epoch)
                _reset_v13_loader(loader, sampling_seed, hard_scores)
            else:
                persistent_loader = None
                persistent_dataset = None
                dataset = Stage2CandidateDataset(
                    store,
                    phase_data,
                    views=cfg.views,
                    augment=cfg.use_stain_augmentation,
                    seed=seed,
                    interface_key=cfg.interface_key,
                )
                dataset.set_epoch(epoch)
                loader = _make_v13_loader(
                    dataset,
                    phase_data,
                    batch_size=runtime.training.stage2_micro_batch_size,
                    workers=runtime.training.number_of_workers,
                    model_cfg=cfg,
                    train=True,
                    seed=sampling_seed,
                    hard_reject_scores=hard_scores,
                    persistent_workers=True,
                )
                persistent_phase, persistent_dataset, persistent_loader = phase, dataset, loader

            encoder_cache = _make_v13_epoch_encoder_cache(model, loader)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = torch.zeros((), device=device)
            step = 0
            for step, batch in enumerate(loader, 1):
                labels = batch["label"].to(device, non_blocking=True)
                outputs, geometry = _forward_batch_streamed(
                    model,
                    batch,
                    device,
                    runtime.training.amp,
                    amp_dtype,
                    encoder_cache,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=runtime.training.amp and device.type == "cuda",
                ):
                    loss = v13_stage2_loss(outputs, labels, class_counts, cfg)
                    scaled = loss / accumulation
                scaler.scale(scaled).backward()
                running_loss.add_(loss.detach())
                if step % accumulation == 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_fast(trainable_parameters, runtime.training.gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                del outputs, geometry, labels, batch, loss, scaled
            if step and step % accumulation != 0:
                partial_group = int(step % accumulation)
                scaler.unscale_(optimizer)
                rescale_partial_accumulation_gradients(
                    trainable_parameters,
                    accumulation_steps=accumulation,
                    microbatches_in_group=partial_group,
                )
                clip_grad_norm_fast(trainable_parameters, runtime.training.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            encoder_cache = None
            print(
                f"[FINAL V13 {cfg.name} s{seed}] epoch {epoch:02d}/{final_epochs:02d} "
                f"phase={phase} phase_epoch={phase_epoch:02d} "
                f"loss={float(running_loss.item()) / max(step, 1):.4f}"
            )

        save_best_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=final_epochs,
            score=float("nan"),
            config=cfg,
            extra={
                "version": 13,
                "final_training": True,
                "all_labeled_rois": True,
                "schedule_reference_epochs": schedule_reference_epochs,
                "validity_threshold": validity_threshold,
                "stage1_oof_folds": sorted(expected_folds),
            },
            trainable_only=True,
            include_training_state=False,
        )
        total, trainable = count_trainable_parameters(model)
        final_lock = {
            "version": 13,
            "selected_experiment": cfg.name,
            "selected_at": utc_now_iso(),
            "selection_mode": "development_lock_then_100_percent_retrain",
            "selected_model_config": asdict(cfg),
            "final_checkpoint": str(checkpoint),
            "final_epochs": int(final_epochs),
            "schedule_reference_epochs": int(schedule_reference_epochs),
            "validity_threshold": float(validity_threshold),
            "seed": int(seed),
            "all_labeled_rois": True,
            "train_roi_count": int(len(all_roi_indices)),
            "train_oof_candidates": int(len(train_base)),
            "train_oof_positives": int(len(train_oof_positive)),
            "train_oof_rejects": int(reject_count),
            "effective_batch_size": int(runtime.training.effective_batch_size),
            "stage2_micro_batch_size": int(runtime.training.stage2_micro_batch_size),
            "encoder_micro_batch_size": int(cfg.encoder_micro_batch_size),
            "parameters_total": int(total),
            "parameters_trainable": int(trainable),
            "duration_minutes": (time.time() - started) / 60.0,
            "peak_vram_mb": peak_vram_mb(),
            "stage1_inference": {
                "experiment": FINAL_STAGE1_EXPERIMENT,
                "seed": FINAL_STAGE1_SEED,
                "folds": sorted(expected_folds),
                "mode": "five_fold_ensemble",
            },
        }
        atomic_write_json(final_lock_path, final_lock)
        return final_lock
    finally:
        release_cuda_memory()


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "cuda out of memory" in text or "outofmemoryerror" in text or "cuda error: out of memory" in text


def train_final_stage2_v13(
    runtime: RuntimeConfig,
    *,
    hf_token: str | None = None,
    selected_experiment: str | None = None,
    final_epochs: int | None = None,
    validity_threshold: float | None = None,
    seed: int = 0,
    force: bool = False,
    auto_oom_fallback: bool = True,
) -> dict[str, Any]:
    """Retrain the locked V13 winner on all labeled ROIs.

    Requested Stage-2 and UNI2-h physical micro-batches start at the runtime/config
    values (256/256 in the supplied notebook).  On CUDA OOM, optional fallback tries
    128/64/32 while preserving the effective optimizer batch through accumulation.
    """
    _, cfg, epochs, schedule_reference_epochs, threshold = _final_recipe(
        runtime,
        selected_experiment=selected_experiment,
        final_epochs=final_epochs,
        validity_threshold=validity_threshold,
    )
    attempts = [int(runtime.training.stage2_micro_batch_size)]
    if auto_oom_fallback:
        effective = int(runtime.training.effective_batch_size)
        attempts.extend(
            value for value in (128, 64, 32)
            if value < attempts[0] and effective % value == 0
        )
    last_error: Exception | None = None
    for attempt_index, micro in enumerate(dict.fromkeys(attempts)):
        run_runtime = copy.deepcopy(runtime)
        run_runtime.training.stage2_micro_batch_size = int(micro)
        run_cfg = replace(cfg, encoder_micro_batch_size=min(int(cfg.encoder_micro_batch_size), int(micro)))
        if attempt_index:
            print(
                f"FINAL V13 CUDA-OOM fallback: Stage-2/UNI2 micro-batch "
                f"{micro}/{run_cfg.encoder_micro_batch_size}; effective batch remains "
                f"{run_runtime.training.effective_batch_size}."
            )
        try:
            return _train_final_stage2_once(
                run_runtime,
                run_cfg,
                hf_token=hf_token,
                final_epochs=epochs,
                schedule_reference_epochs=schedule_reference_epochs,
                validity_threshold=threshold,
                seed=int(seed),
                force=force or attempt_index > 0,
            )
        except Exception as exc:
            last_error = exc
            release_cuda_memory()
            if not (auto_oom_fallback and _is_cuda_oom(exc) and attempt_index + 1 < len(attempts)):
                raise
    assert last_error is not None
    raise last_error


def final_v13_ready(runtime: RuntimeConfig) -> dict[str, Any]:
    """Validate the artifacts required for challenge inference without running it."""
    stage1_lock = write_fixed_stage1_lock_v13(runtime)
    final_lock_path = runtime.paths.stage2_existing_file(V13_FINAL_LOCK_NAME)
    final_lock = _load_json(final_lock_path)
    checkpoint = Path(final_lock["final_checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = runtime.paths.stage2_output_dir / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"V13 final Stage-2 checkpoint is missing: {checkpoint}")
    return {
        "ready": True,
        "stage1_experiment": stage1_lock["selected_experiment"],
        "stage1_folds": stage1_lock["run_folds"],
        "stage2_experiment": final_lock["selected_experiment"],
        "stage2_checkpoint": str(checkpoint),
        "validity_threshold": float(final_lock["validity_threshold"]),
    }
