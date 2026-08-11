from __future__ import annotations

"""Final all-data V13.2 training and deployment locks."""

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
    STAGE2_GEOMETRY_NAMES,
    RuntimeConfig,
    Stage2ModelConfig,
    stage1_experiment_registry,
    stage1_model_config_from_dict,
)
from puma.data.datasets import PumaNpyStore, Stage2CandidateDataset
from puma.models.stage2 import build_stage2_model, ensure_stage2_pretrained_checkpoints
from puma.pipeline.oof import validate_full_oof
from puma.stage2.catalog import stage2_experiment_registry
from puma.training.stage1 import _stage1_run_hash
from puma.training.stage2 import (
    _build_perfect_candidates, _forward_batch_streamed, _capture_rng_state,
    _restore_rng_state, _move_optimizer_state_to_device,
)
from puma.training.stage2_v132 import (
    _hardness_scores,
    _make_epoch_encoder_cache,
    _make_loader,
    _reset_epoch_loader,
    _set_optimizer_lrs,
    _split_trainable_parameters,
    phase_for_epoch,
    phase_learning_rates,
    phase_bounds,
    stage2_loss,
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
    resolve_artifact_reference,
    resolve_device,
    restore_checkpoint_payload,
    save_best_checkpoint,
    seed_everything,
    utc_now_iso,
)

FINAL_STAGE1_EXPERIMENT = "A1_IFCRN_PP"
FINAL_STAGE1_SEED = 0
V132_FINAL_LOCK_NAME = "stage2_v132_final_lock.json"
V132_FINAL_CHECKPOINT_PREFIX = "stage2_v132_final"
V132_FINAL_IMPLEMENTATION_REVISION = 3


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_stage1_deployment_lock_v132(
    runtime: RuntimeConfig,
    *,
    seed: int = FINAL_STAGE1_SEED,
) -> dict[str, Any]:
    registry = stage1_experiment_registry()
    cfg = registry[FINAL_STAGE1_EXPERIMENT]
    folds = tuple(range(runtime.data.number_of_folds))
    checkpoints: list[dict[str, Any]] = []
    for fold in folds:
        path = runtime.paths.stage1_existing_file(
            f"stage1_best_{FINAL_STAGE1_EXPERIMENT}_fold{fold}_seed{seed}.pt"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage-1 fold checkpoint: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        serialized = payload.get("config")
        observed_cfg = stage1_model_config_from_dict(serialized) if serialized else cfg
        if config_hash(observed_cfg) != config_hash(cfg):
            raise RuntimeError(f"Stage-1 config mismatch in {path.name}.")
        extra = dict(payload.get("extra", {}))
        expected_run_hash = _stage1_run_hash(runtime, cfg)
        if str(extra.get("config_hash", "")) != expected_run_hash:
            raise RuntimeError(f"Stage-1 checkpoint training identity mismatch in {path.name}")
        checkpoints.append({
            "fold": int(fold),
            "checkpoint": str(path),
            "threshold": float(extra.get("threshold", 0.25)),
            "radius": int(extra.get("radius", 3)),
            "suppression_radius": float(extra.get("suppression_radius", 5.0)),
            "refit_epochs": int(extra.get("refit_epochs", payload.get("epoch", 0))),
            "config_hash": str(extra.get("config_hash", "")),
        })
    lock = {
        "version": "13.2",
        "selected_experiment": FINAL_STAGE1_EXPERIMENT,
        "selected_at": utc_now_iso(),
        "inference_mode": "five_fold_refit_ensemble",
        "selected_model_config": asdict(cfg),
        "run_folds": list(folds),
        "seed": int(seed),
        "checkpoints": checkpoints,
    }
    atomic_write_json(runtime.paths.stage1_file("stage1_lock.json"), lock)
    return lock


def _resolve_final_recipe(
    runtime: RuntimeConfig,
    *,
    selected_experiment: str | None,
    final_epochs: int,
    validity_threshold: float | None,
) -> tuple[Stage2ModelConfig, float]:
    phase_bounds(final_epochs)
    dev_lock = _load_json(runtime.paths.stage2_existing_file("stage2_v132_lock.json"))
    selected = str(selected_experiment or dev_lock.get("selected_experiment", ""))
    registry = stage2_experiment_registry()
    if selected not in registry:
        raise KeyError(f"Unknown locked V13.2 experiment {selected!r}.")
    payload = dev_lock.get("selected_model_config")
    cfg = Stage2ModelConfig(**payload) if isinstance(payload, dict) else registry[selected]
    if cfg.name != selected:
        raise RuntimeError("V13.2 development lock/config mismatch.")
    if cfg.use_lora:
        raise ValueError("LoRA is excluded from V13.2 final training.")
    default_threshold = float((dev_lock.get("deployment") or {}).get("validity_threshold", 0.5))
    threshold = default_threshold if validity_threshold is None else float(validity_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("validity_threshold must be in [0,1].")
    return cfg, threshold


def _semantic_final_model_config(cfg: Stage2ModelConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload.pop("encoder_micro_batch_size", None)
    return payload


def _final_training_hash(runtime: RuntimeConfig, cfg: Stage2ModelConfig, *, final_epochs: int, seed: int, stage1_lock: dict[str, Any]) -> str:
    oof_path = runtime.paths.stage1_existing_file("stage1_oof_candidates_metadata.json")
    oof = {}
    if oof_path.exists():
        meta = _load_json(oof_path)
        oof = {k: meta.get(k) for k in ("cache_signature", "number_of_candidates", "number_of_evaluated_rois")}
    dev_lock_path = runtime.paths.stage2_existing_file("stage2_v132_lock.json")
    development = {}
    if dev_lock_path.exists():
        dev = _load_json(dev_lock_path)
        development = {
            "selected_experiment": dev.get("selected_experiment"),
            "selected_config_hash": dev.get("selected_config_hash"),
            "split_name": dev.get("split_name"),
            "split_hash": dev.get("split_hash"),
            "epoch_profile": dev.get("epoch_profile"),
        }
    stable_stage1 = {
        "selected_experiment": stage1_lock.get("selected_experiment"),
        "run_folds": stage1_lock.get("run_folds"), "seed": stage1_lock.get("seed"),
        "checkpoints": [{
            "fold": r.get("fold"), "checkpoint": Path(str(r.get("checkpoint", ""))).name,
            "threshold": r.get("threshold"), "radius": r.get("radius"),
            "suppression_radius": r.get("suppression_radius"), "refit_epochs": r.get("refit_epochs"),
            "config_hash": r.get("config_hash"),
        } for r in stage1_lock.get("checkpoints", [])],
    }
    return config_hash({
        "version": "13.2", "implementation_revision": V132_FINAL_IMPLEMENTATION_REVISION,
        "experiment": cfg.name, "model": _semantic_final_model_config(cfg),
        "geometry_names": list(STAGE2_GEOMETRY_NAMES),
        "epochs": int(final_epochs), "seed": int(seed),
        "effective_batch_size": int(runtime.training.stage2_effective_batch_size),
        # Physical micro-batch/encoder chunks are excluded so CUDA-OOM fallback
        # remains the same semantic final-training identity.
        "amp": runtime.training.amp, "prefer_bfloat16": runtime.training.prefer_bfloat16,
        "deterministic": runtime.training.deterministic,
        "gradient_clip_norm": runtime.training.gradient_clip_norm,
        "stage1": stable_stage1, "oof": oof, "development": development,
    })


def _deployment_hash(final_training_hash: str, validity_threshold: float) -> str:
    """Identity of inference-time decoding layered on an already trained model."""
    return config_hash({
        "final_training_hash": str(final_training_hash),
        "validity_threshold": float(validity_threshold),
        "revision": 1,
    })


def _train_final_once(
    runtime: RuntimeConfig,
    cfg: Stage2ModelConfig,
    *,
    final_epochs: int,
    validity_threshold: float,
    seed: int,
    hf_token: str | None,
    force: bool,
) -> dict[str, Any]:
    paths = runtime.paths
    paths.ensure()
    validate_full_oof(runtime)
    ensure_stage2_pretrained_checkpoints(paths.root, hf_token=hf_token, pfm_keys=("uni2_h",))
    stage1_lock = write_stage1_deployment_lock_v132(runtime, seed=seed)

    final_hash = _final_training_hash(runtime, cfg, final_epochs=final_epochs, seed=seed, stage1_lock=stage1_lock)
    run_tag = final_hash[:10]
    checkpoint = paths.stage2_file(
        f"{V132_FINAL_CHECKPOINT_PREFIX}_{cfg.name}_{final_epochs}ep_seed{seed}_{run_tag}.pt"
    )
    resume_checkpoint = paths.stage2_file(
        f"stage2_v132_final_resume_{cfg.name}_{final_epochs}ep_seed{seed}_{run_tag}.pt"
    )
    lock_path = paths.stage2_file(V132_FINAL_LOCK_NAME)
    if checkpoint.exists() and lock_path.exists() and not force:
        existing = _load_json(lock_path)
        try:
            check_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            checkpoint_ok = str(check_payload.get("extra", {}).get("config_hash", "")) == final_hash
            del check_payload
        except Exception:
            checkpoint_ok = False
        if (checkpoint_ok and str(existing.get("final_training_hash", "")) == final_hash
            and str(existing.get("selected_experiment")) == cfg.name
            and int(existing.get("final_epochs", -1)) == int(final_epochs)):
            requested_deployment_hash = _deployment_hash(final_hash, validity_threshold)
            if (
                abs(float(existing.get("validity_threshold", 0.5)) - float(validity_threshold)) > 1e-12
                or str(existing.get("deployment_hash", "")) != requested_deployment_hash
            ):
                # The trained weights are identical; only the inference decoding threshold
                # changed. Update the deployment lock atomically instead of wasting compute.
                existing["validity_threshold"] = float(validity_threshold)
                existing["deployment_hash"] = requested_deployment_hash
                existing["deployment_updated_at"] = utc_now_iso()
                atomic_write_json(lock_path, existing)
                print(
                    "REUSE V13.2 final checkpoint and update validity threshold: "
                    f"{checkpoint} -> {validity_threshold:.6f}"
                )
            else:
                print(f"REUSE V13.2 final checkpoint: {checkpoint}")
            return existing

    seed_everything(seed, runtime.training.deterministic)
    device = resolve_device()
    reset_peak_vram()
    started = time.time()
    store = PumaNpyStore.open(paths.artifact_dir)
    manifest = np.asarray(store.manifest)
    candidates = np.asarray(np.load(
        paths.stage1_existing_file("stage1_oof_candidates.npy"), mmap_mode="r", allow_pickle=False
    ))
    expected_folds = set(range(runtime.data.number_of_folds))
    observed = set(np.unique(candidates["fold"]).astype(int).tolist())
    if observed != expected_folds:
        raise RuntimeError(f"Final V13.2 needs all OOF folds {sorted(expected_folds)}, got {sorted(observed)}.")
    positives = candidates[candidates["class_id"] != REJECT_CLASS_ID]
    all_rois = np.arange(len(manifest), dtype=np.int64)
    perfect = _build_perfect_candidates(
        store, candidates, seed=seed, background_fraction=0.0, roi_indices=all_rois
    )

    model = build_stage2_model(cfg, hf_token=hf_token).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    type_parameters, validity_parameters = _split_trainable_parameters(model)
    trainable_parameters = [*type_parameters, *validity_parameters]
    optimizer = build_adamw_parameter_groups(
        [
            {"params": type_parameters, "lr": cfg.phase1_start_lr, "weight_decay": cfg.weight_decay, "puma_role": "type"},
            {"params": validity_parameters, "lr": 0.0, "weight_decay": cfg.weight_decay, "puma_role": "validity"},
        ],
        device=device,
    )
    amp_dtype = resolve_amp_dtype(runtime.training.prefer_bfloat16, device)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(runtime.training.amp and device.type == "cuda" and amp_dtype == torch.float16),
    )
    accumulation = runtime.training.stage2_accumulation_steps
    start_epoch = 1
    if resume_checkpoint.exists() and not force:
        payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        extra_resume = dict(payload.get("extra", {}))
        if str(extra_resume.get("config_hash", "")) == final_hash:
            restore_checkpoint_payload(payload, model)
            if payload.get("optimizer_state") is not None:
                optimizer.load_state_dict(payload["optimizer_state"]); _move_optimizer_state_to_device(optimizer, device)
            if payload.get("scaler_state") is not None: scaler.load_state_dict(payload["scaler_state"])
            start_epoch = int(payload.get("epoch", 0)) + 1
            _restore_rng_state(extra_resume)
            print(f"RESUME FINAL V13.2 {cfg.name}: epoch {start_epoch}/{final_epochs}")
        else:
            resume_checkpoint.unlink(missing_ok=True)
        del payload
    type_counts = np.bincount(perfect["class_id"].astype(int), minlength=REJECT_CLASS_ID)[:REJECT_CLASS_ID]
    missing = [PUMA_CLASS_NAMES[i] for i in np.flatnonzero(type_counts == 0).astype(int)]
    if missing:
        raise RuntimeError("Final V13.2 data is missing class(es): " + ", ".join(missing))
    raw_counts = torch.as_tensor(
        np.r_[type_counts, max(int(np.sum(candidates["class_id"] == REJECT_CLASS_ID)), 1)],
        dtype=torch.float32,
        device=device,
    )
    hard_reject_scores: np.ndarray | None = None
    hard_positive_scores: np.ndarray | None = None
    active_phase: str | None = None
    train_dataset: Stage2CandidateDataset | None = None
    train_loader = None

    try:
        for epoch in range(start_epoch, final_epochs + 1):
            phase, phase_epoch, phase_length = phase_for_epoch(epoch, final_epochs)
            phase_data = perfect if phase == "GT_POS" else positives if phase == "OOF_POS" else candidates
            type_lr, validity_lr = phase_learning_rates(cfg, epoch, final_epochs)
            _set_optimizer_lrs(optimizer, type_lr, validity_lr)
            validity_active = phase == "OOF_ALL"
            mining_active = (
                phase == "OOF_ALL"
                and phase_epoch >= min(cfg.hard_negative_start_phase_epoch, cfg.hard_positive_start_phase_epoch)
            )
            refresh = (
                mining_active
                and (
                    hard_reject_scores is None
                    or (phase_epoch - cfg.hard_negative_start_phase_epoch)
                    % max(1, cfg.hard_pool_refresh_interval) == 0
                )
            )
            if refresh:
                hard_reject_scores, hard_positive_scores = _hardness_scores(
                    model, store, phase_data, cfg, device,
                    batch_size=runtime.training.stage2_micro_batch_size,
                    workers=runtime.training.number_of_workers,
                    amp=runtime.training.amp,
                    prefer_bfloat16=runtime.training.prefer_bfloat16,
                )
            epoch_seed = seed + epoch * 10007
            if active_phase != phase or train_loader is None or train_dataset is None:
                if train_loader is not None:
                    del train_loader
                if train_dataset is not None:
                    del train_dataset
                train_dataset = Stage2CandidateDataset(
                    store, phase_data, views=cfg.views, augment=cfg.use_stain_augmentation,
                    seed=seed, interface_key=cfg.interface_key,
                )
                train_dataset.set_epoch(epoch)
                train_loader = _make_loader(
                    train_dataset, phase_data, cfg, manifest,
                    batch_size=runtime.training.stage2_micro_batch_size,
                    workers=runtime.training.number_of_workers,
                    phase=phase,
                    seed=epoch_seed,
                    hard_reject_scores=hard_reject_scores if mining_active else None,
                    hard_positive_scores=hard_positive_scores if mining_active else None,
                    persistent_workers=runtime.training.number_of_workers > 0,
                )
                active_phase = phase
            else:
                _reset_epoch_loader(
                    train_loader, train_dataset, epoch=epoch, seed=epoch_seed,
                    hard_reject_scores=hard_reject_scores if mining_active else None,
                    hard_positive_scores=hard_positive_scores if mining_active else None,
                )
            epoch_encoder_cache = _make_epoch_encoder_cache(model, train_loader.sampler)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running = torch.zeros((), device=device)
            step = 0
            for step, batch in enumerate(train_loader, 1):
                labels = batch["label"].to(device, non_blocking=True)
                outputs, geometry = _forward_batch_streamed(
                    model, batch, device, runtime.training.amp, amp_dtype, epoch_encoder_cache
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=runtime.training.amp and device.type == "cuda",
                ):
                    loss = stage2_loss(
                        outputs, labels, raw_counts, cfg, validity_active=validity_active
                    )
                    scaled = loss / accumulation
                scaler.scale(scaled).backward()
                running.add_(loss.detach())
                if step % accumulation == 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_fast(trainable_parameters, runtime.training.gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                del outputs, geometry, labels, batch, loss, scaled
            if step and step % accumulation != 0:
                partial = step % accumulation
                scaler.unscale_(optimizer)
                rescale_partial_accumulation_gradients(
                    trainable_parameters,
                    accumulation_steps=accumulation,
                    microbatches_in_group=partial,
                )
                clip_grad_norm_fast(trainable_parameters, runtime.training.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            print(
                f"[FINAL V13.2 {cfg.name}] epoch={epoch:03d}/{final_epochs} "
                f"phase={phase}:{phase_epoch:02d}/{phase_length:02d} "
                f"lr={type_lr:.2e}/{validity_lr:.2e} "
                f"loss={float(running.item()) / max(step,1):.4f}"
            )
            del epoch_encoder_cache
            if epoch % int(runtime.training.resume_checkpoint_interval) == 0 or epoch == final_epochs:
                resume_extra = {"config_hash": final_hash, "final_training": True, "phase": phase, "phase_epoch": phase_epoch}
                resume_extra.update(_capture_rng_state())
                save_best_checkpoint(
                    resume_checkpoint, model=model, optimizer=optimizer, scheduler=None, scaler=scaler,
                    epoch=epoch, score=float("nan"), config=cfg, extra=resume_extra,
                    trainable_only=True, include_training_state=True,
                )

        if train_loader is not None:
            del train_loader
        if train_dataset is not None:
            del train_dataset

        save_best_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=scaler,
            epoch=final_epochs,
            score=float("nan"),
            config=cfg,
            extra={
                "version": "13.2",
                "final_training": True,
                "all_labeled_rois": True,
                "epoch_profile": final_epochs,
                "validity_threshold": validity_threshold,
                "config_hash": final_hash,
            },
            trainable_only=True,
            include_training_state=False,
        )
        total, trainable = count_trainable_parameters(model)
        lock = {
            "version": "13.2",
            "final_training_hash": final_hash,
            "deployment_hash": _deployment_hash(final_hash, validity_threshold),
            "selected_experiment": cfg.name,
            "selected_at": utc_now_iso(),
            "selection_mode": "development_lock_then_all_data_retrain",
            "selected_model_config": asdict(cfg),
            "final_checkpoint": str(checkpoint),
            "final_epochs": int(final_epochs),
            "validity_threshold": float(validity_threshold),
            "seed": int(seed),
            "all_labeled_rois": True,
            "train_roi_count": int(len(all_rois)),
            "train_oof_candidates": int(len(candidates)),
            "train_oof_positives": int(len(positives)),
            "train_oof_rejects": int(np.sum(candidates["class_id"] == REJECT_CLASS_ID)),
            "stage2_effective_batch_size": int(runtime.training.stage2_effective_batch_size),
            "stage2_micro_batch_size": int(runtime.training.stage2_micro_batch_size),
            "encoder_micro_batch_size": int(cfg.encoder_micro_batch_size),
            "parameters_total": int(total),
            "parameters_trainable": int(trainable),
            "duration_minutes": (time.time() - started) / 60.0,
            "peak_vram_mb": peak_vram_mb(),
            "stage1_inference": stage1_lock,
        }
        atomic_write_json(lock_path, lock)
        resume_checkpoint.unlink(missing_ok=True)
        return lock
    finally:
        release_cuda_memory()


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(token in text for token in (
        "cuda out of memory", "outofmemoryerror", "cuda error: out of memory"
    ))


def train_final_stage2_v132(
    runtime: RuntimeConfig,
    *,
    hf_token: str | None = None,
    selected_experiment: str | None = None,
    final_epochs: int = 100,
    validity_threshold: float | None = None,
    seed: int = 0,
    force: bool = False,
    auto_oom_fallback: bool = True,
) -> dict[str, Any]:
    """Retrain the locked V13.2 winner on 100% labeled ROIs.

    ``final_epochs`` accepts 50 or 100. The intended submission profile is 100.
    Physical Stage-2/UNI2 batches fall back on OOM while effective batch stays 256.
    """
    cfg, threshold = _resolve_final_recipe(
        runtime,
        selected_experiment=selected_experiment,
        final_epochs=int(final_epochs),
        validity_threshold=validity_threshold,
    )
    initial = int(runtime.training.stage2_micro_batch_size)
    effective = int(runtime.training.stage2_effective_batch_size)
    attempts = [initial]
    if auto_oom_fallback:
        attempts.extend(
            value for value in (128, 64, 32, 16)
            if value < initial and effective % value == 0
        )
    last_error: Exception | None = None
    for attempt_index, micro in enumerate(dict.fromkeys(attempts)):
        run_runtime = copy.deepcopy(runtime)
        run_runtime.training.stage2_epochs = int(final_epochs)
        run_runtime.training.stage2_micro_batch_size = int(micro)
        run_cfg = replace(cfg, encoder_micro_batch_size=min(cfg.encoder_micro_batch_size, micro))
        if attempt_index:
            print(
                f"FINAL V13.2 OOM fallback: Stage2/UNI2 micro={micro}/{run_cfg.encoder_micro_batch_size}; "
                f"effective batch remains {effective}."
            )
        try:
            return _train_final_once(
                run_runtime,
                run_cfg,
                final_epochs=int(final_epochs),
                validity_threshold=threshold,
                seed=int(seed),
                hf_token=hf_token,
                force=force,
            )
        except Exception as exc:
            last_error = exc
            release_cuda_memory()
            if not (auto_oom_fallback and _is_cuda_oom(exc) and attempt_index + 1 < len(attempts)):
                raise
    assert last_error is not None
    raise last_error


def final_v132_ready(runtime: RuntimeConfig) -> dict[str, Any]:
    stage1 = write_stage1_deployment_lock_v132(runtime)
    lock = _load_json(runtime.paths.stage2_existing_file(V132_FINAL_LOCK_NAME))
    checkpoint = resolve_artifact_reference(
        lock.get("final_checkpoint", ""), runtime.paths.stage2_output_search_dirs()
    )
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(
            f"V13.2 final Stage-2 checkpoint missing from canonical output directory: "
            f"{lock.get('final_checkpoint')}"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_hash = str(lock.get("final_training_hash", ""))
    if expected_hash and str(payload.get("extra", {}).get("config_hash", "")) != expected_hash:
        raise RuntimeError(f"V13.2 final checkpoint identity mismatch: {checkpoint}")
    threshold = float(lock.get("validity_threshold", 0.5))
    deployment_hash = str(lock.get("deployment_hash", ""))
    if deployment_hash and deployment_hash != _deployment_hash(expected_hash, threshold):
        raise RuntimeError("V13.2 final deployment lock identity mismatch.")
    return {
        "ready": True,
        "stage1_experiment": stage1["selected_experiment"],
        "stage1_folds": stage1["run_folds"],
        "stage2_experiment": lock["selected_experiment"],
        "stage2_checkpoint": str(checkpoint),
        "final_epochs": int(lock["final_epochs"]),
        "validity_threshold": float(lock["validity_threshold"]),
    }
