from __future__ import annotations

"""Stage-2 V13 training on one fixed 80/20 development split."""

import json
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from puma.config import PUMA_CLASS_NAMES, REJECT_CLASS_ID, TAIL_CLASS_IDS, RuntimeConfig, Stage2ModelConfig
from puma.data.datasets import PumaNpyStore, Stage2CandidateDataset, stage2_collate
from puma.models.stage2 import (
    build_stage2_model,
    effective_number_weights,
    focal_binary_loss,
    split_optimizer_parameters,
)
from puma.stage2.train_val_split import SPLIT_TRAIN, SPLIT_VAL, create_split, validate_final_split
from puma.training.stage2 import (
    _build_perfect_candidates,
    _capture_rng_state,
    _final_schedule_phase,
    _forward_batch_streamed,
    _hard_reject_scores,
    _move_optimizer_state_to_device,
    _phase_epoch_number,
    _phase_for_epoch,
    _restore_rng_state,
    _resume_interval_epochs,
    _warmup_cosine_scheduler,
    evaluate_stage2,
)
from puma.utils import (
    append_csv_row_atomic,
    atomic_save_numpy,
    build_adamw_parameter_groups,
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
    resolve_device,
    restore_checkpoint_payload,
    save_best_checkpoint,
    seed_everything,
    sha256_file,
    utc_now_iso,
    worker_seed_init,
)

V13_SPLIT_NAME = "optimized_train80_val20"
V13_IMPLEMENTATION_REVISION = 2


def _split_paths(runtime: RuntimeConfig) -> tuple[Path, Path, Path]:
    directory = runtime.paths.artifact_dir / "train_val_split"
    return (
        directory / "puma_train_val_assignments.npy",
        directory / "puma_train_val_indices.npz",
        directory / "puma_train_val_split_metadata.json",
    )


def ensure_v13_split(
    runtime: RuntimeConfig,
    *,
    force: bool = False,
    val_fraction: float = 0.20,
    seed: int = 2026,
    check_sources: bool = True,
) -> dict[str, Any]:
    """Create (if needed), load, and validate the Version-13 development split."""
    assignments_path, indices_path, metadata_path = _split_paths(runtime)
    if force or not (assignments_path.exists() and indices_path.exists() and metadata_path.exists()):
        create_split(
            runtime.paths.root,
            val_fraction=val_fraction,
            seed=seed,
            output_dir=assignments_path.parent,
            check_sources=check_sources,
        )
    manifest = np.load(runtime.paths.preprocessing_file("puma_roi_manifest.npy"), mmap_mode="r", allow_pickle=False)
    assignments = np.load(assignments_path, allow_pickle=False)
    if assignments.shape != (len(manifest),):
        raise RuntimeError(
            f"V13 split length {assignments.shape} does not match manifest length {len(manifest)}. "
            "Regenerate the split."
        )
    if set(np.unique(assignments).astype(int).tolist()) != {int(SPLIT_TRAIN), int(SPLIT_VAL)}:
        raise RuntimeError("V13 split must contain exactly train=0 and validation=1 assignments.")
    diagnostics = validate_final_split(manifest, assignments)
    if diagnostics["case_leakage_count"] != 0:
        raise RuntimeError("Case leakage detected in V13 split.")
    if diagnostics["missing_train_classes"]:
        raise RuntimeError(
            "V13 training split is missing class(es): " + ", ".join(diagnostics["missing_train_classes"])
        )
    if diagnostics["missing_validation_classes"]:
        raise RuntimeError(
            "V13 validation split is missing class(es): " + ", ".join(diagnostics["missing_validation_classes"])
        )
    with np.load(indices_path, allow_pickle=False) as split_indices:
        train_roi_indices = np.asarray(split_indices["train_roi_indices"], dtype=np.int64)
        val_roi_indices = np.asarray(split_indices["val_roi_indices"], dtype=np.int64)
    return {
        "assignments_path": assignments_path,
        "indices_path": indices_path,
        "metadata_path": metadata_path,
        "assignments": assignments,
        "train_roi_indices": train_roi_indices,
        "val_roi_indices": val_roi_indices,
        "diagnostics": diagnostics,
        "split_hash": sha256_file(assignments_path)[:16],
    }


def _v13_run_hash(
    runtime: RuntimeConfig,
    model_cfg: Stage2ModelConfig,
    split_hash: str,
) -> str:
    metadata_path = runtime.paths.stage1_existing_file("stage1_oof_candidates_metadata.json")
    oof_signature: dict[str, Any] = {}
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        oof_signature = {
            "cache_signature": payload.get("cache_signature"),
            "number_of_candidates": payload.get("number_of_candidates"),
            "number_of_evaluated_rois": payload.get("number_of_evaluated_rois"),
        }
    tr = runtime.training
    return config_hash({
        "version": 13,
        "implementation_revision": V13_IMPLEMENTATION_REVISION,
        "model": model_cfg,
        "split_name": V13_SPLIT_NAME,
        "split_hash": split_hash,
        "training": {
            "epochs": tr.epochs,
            "effective_batch_size": tr.effective_batch_size,
            "micro_batch_size": tr.stage2_micro_batch_size,
            "workers": tr.number_of_workers,
            "amp": tr.amp,
            "prefer_bfloat16": tr.prefer_bfloat16,
            "deterministic": tr.deterministic,
            "validation_interval": tr.validation_interval,
            "early_stopping_enabled": tr.early_stopping_enabled,
            "early_stopping_patience": tr.early_stopping_patience,
            "early_stopping_min_delta": tr.early_stopping_min_delta,
        },
        "oof": oof_signature,
    })


class V13MixtureCandidateSampler(Sampler[int]):
    """Configurable natural + class-balanced sampler with extra tail-class repeat budget."""

    def __init__(
        self,
        candidates: np.ndarray,
        model_cfg: Stage2ModelConfig,
        seed: int,
        hard_reject_scores: np.ndarray | None = None,
    ) -> None:
        self.candidates = candidates
        self.cfg = model_cfg
        self.seed = int(seed)
        self.hard_reject_scores = hard_reject_scores
        self.indices: list[int] = []
        self.reset(seed, hard_reject_scores)

    @staticmethod
    def _draw_with_global_cap(
        pool: np.ndarray,
        number: int,
        rng: np.random.Generator,
        global_counts: np.ndarray,
        global_caps: np.ndarray,
    ) -> list[int]:
        """Draw without exceeding the per-candidate epoch cap."""
        if number <= 0 or len(pool) == 0:
            return []
        pool = np.asarray(pool, dtype=np.int64)
        result: list[int] = []
        remaining = int(number)
        while remaining > 0:
            eligible = pool[global_counts[pool] < global_caps[pool]]
            if len(eligible) == 0:
                break
            order = rng.permutation(eligible)
            take = min(remaining, len(order))
            chosen = np.asarray(order[:take], dtype=np.int64)
            result.extend(chosen.astype(int, copy=False).tolist())
            np.add.at(global_counts, chosen, 1)
            remaining -= take
        return result

    def reset(self, seed: int, hard_reject_scores: np.ndarray | None = None) -> None:
        self.seed = int(seed)
        self.hard_reject_scores = hard_reject_scores
        rng = np.random.default_rng(self.seed)
        labels = self.candidates["class_id"].astype(int)
        all_indices = np.arange(len(self.candidates), dtype=np.int64)
        positive = all_indices[labels != REJECT_CLASS_ID]
        reject = all_indices[labels == REJECT_CLASS_ID]

        # One repeat cap applies across all sampling branches.
        global_caps = np.full(len(self.candidates), int(self.cfg.sampler_max_repeats), dtype=np.int32)
        if len(positive):
            tail_mask = np.isin(labels, np.asarray(TAIL_CLASS_IDS, dtype=int))
            global_caps[tail_mask] = int(self.cfg.sampler_tail_max_repeats)
        global_counts = np.zeros(len(self.candidates), dtype=np.int32)

        if len(positive) == 0:
            self.indices = self._draw_with_global_cap(
                reject, len(reject), rng, global_counts, global_caps
            )
            self.repeat_counts = global_counts
            return

        n = len(self.candidates)
        positive_target = n if len(reject) == 0 else int(round(self.cfg.sampler_positive_fraction * n))
        positive_target = min(max(1, positive_target), n)
        reject_target = n - positive_target
        balanced_target = int(round(self.cfg.sampler_balanced_positive_fraction * positive_target))
        balanced_target = min(max(0, balanced_target), positive_target)
        natural_target = positive_target - balanced_target

        natural = self._draw_with_global_cap(
            positive, natural_target, rng, global_counts, global_caps
        )

        # Equal-class draws share the same global repeat counter.
        class_pools = {
            class_id: positive[labels[positive] == class_id]
            for class_id in range(REJECT_CLASS_ID)
            if np.any(labels[positive] == class_id)
        }
        balanced: list[int] = []
        active = list(class_pools)
        while len(balanced) < balanced_target and active:
            next_active: list[int] = []
            for class_id_value in rng.permutation(active):
                class_id = int(class_id_value)
                pool = class_pools[class_id]
                eligible = pool[global_counts[pool] < global_caps[pool]]
                if len(eligible) == 0:
                    continue
                chosen = int(eligible[int(rng.integers(len(eligible)))])
                balanced.append(chosen)
                global_counts[chosen] += 1
                if np.any(global_counts[pool] < global_caps[pool]):
                    next_active.append(class_id)
                if len(balanced) >= balanced_target:
                    break
            active = next_active

        # Use remaining positive capacity without exceeding repeat caps.
        positive_draws = natural + balanced
        if len(positive_draws) < positive_target:
            positive_draws.extend(
                self._draw_with_global_cap(
                    positive,
                    positive_target - len(positive_draws),
                    rng,
                    global_counts,
                    global_caps,
                )
            )

        reject_draws: list[int] = []
        if reject_target > 0 and len(reject):
            if hard_reject_scores is not None:
                scores = np.asarray(hard_reject_scores)
                if len(scores) != len(self.candidates):
                    raise ValueError(
                        "hard_reject_scores must have one value per candidate: "
                        f"{len(scores)} != {len(self.candidates)}"
                    )
                order = reject[np.argsort(-scores[reject])]
                hard_pool = order[: max(1, len(order) // 2)]
                reject_draws.extend(
                    self._draw_with_global_cap(
                        hard_pool, reject_target // 2, rng, global_counts, global_caps
                    )
                )
            reject_draws.extend(
                self._draw_with_global_cap(
                    reject,
                    reject_target - len(reject_draws),
                    rng,
                    global_counts,
                    global_caps,
                )
            )

        # Backfill legally so every epoch keeps its full length.
        output = positive_draws + reject_draws
        remaining = n - len(output)
        if remaining > 0:
            opposite_pool = reject if len(positive_draws) < positive_target else positive
            output.extend(
                self._draw_with_global_cap(
                    opposite_pool, remaining, rng, global_counts, global_caps
                )
            )
            remaining = n - len(output)
        if remaining > 0:
            output.extend(
                self._draw_with_global_cap(
                    all_indices, remaining, rng, global_counts, global_caps
                )
            )
        if len(output) != n:
            raise RuntimeError(
                f"V13 sampler could not construct a full epoch: {len(output)} != {n}."
            )
        rng.shuffle(output)
        self.indices = output
        self.repeat_counts = global_counts

        # Enforce the total per-candidate repeat cap.
        used = np.asarray(output, dtype=np.int64)
        if len(used):
            observed = np.bincount(used, minlength=len(self.candidates))
            bad = np.flatnonzero(observed > global_caps)
            if len(bad):
                raise RuntimeError(
                    "V13 sampler exceeded its global repeat cap for candidate(s): "
                    + ", ".join(map(str, bad[:20].tolist()))
                )

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _make_v13_loader(
    dataset: Stage2CandidateDataset,
    candidates: np.ndarray,
    *,
    batch_size: int,
    workers: int,
    model_cfg: Stage2ModelConfig,
    train: bool,
    seed: int = 0,
    hard_reject_scores: np.ndarray | None = None,
    persistent_workers: bool = False,
) -> DataLoader:
    common = {
        "collate_fn": stage2_collate,
        **dataloader_performance_kwargs(
            workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=persistent_workers,
        ),
    }
    if train:
        sampler = V13MixtureCandidateSampler(candidates, model_cfg, seed, hard_reject_scores)
        common["worker_init_fn"] = worker_seed_init
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, **common)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **common)


def _reset_v13_loader(
    loader: DataLoader,
    seed: int,
    hard_reject_scores: np.ndarray | None,
) -> None:
    if not isinstance(loader.sampler, V13MixtureCandidateSampler):
        raise TypeError("Unsupported reusable Version-13 Stage-2 loader.")
    loader.sampler.reset(seed, hard_reject_scores)


def _v13_type_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_counts: torch.Tensor,
    cfg: Stage2ModelConfig,
) -> torch.Tensor:
    positive = labels != REJECT_CLASS_ID
    if not bool(positive.any()):
        return logits.sum() * 0.0
    y = labels[positive]
    z = logits[positive]
    counts = class_counts[:REJECT_CLASS_ID].clamp_min(1.0)
    if cfg.type_loss_key == "BALANCED_SOFTMAX":
        return F.cross_entropy(z + torch.log(counts)[None], y)
    if cfg.type_loss_key == "CE":
        return F.cross_entropy(z, y)
    weights = effective_number_weights(counts, beta=cfg.class_balance_beta).to(z.device)
    if cfg.type_loss_key == "CB_CE":
        return F.cross_entropy(z, y, weight=weights)
    if cfg.type_loss_key == "CB_FOCAL":
        ce = F.cross_entropy(z, y, reduction="none")
        pt = torch.exp(-ce).clamp(1e-6, 1.0)
        sample_weights = weights.gather(0, y)
        weighted = sample_weights * (1.0 - pt).pow(cfg.type_focal_gamma) * ce
        return weighted.sum() / sample_weights.sum().clamp_min(1e-8)
    raise KeyError(f"Unknown V13 type_loss_key={cfg.type_loss_key!r}")


def v13_stage2_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    class_counts: torch.Tensor,
    cfg: Stage2ModelConfig,
) -> torch.Tensor:
    positive = labels != REJECT_CLASS_ID
    type_loss = _v13_type_loss(outputs["type_logits"], labels, class_counts, cfg)
    total = cfg.type_loss_weight * type_loss
    if cfg.loss_key != "TYPE_BALANCED":
        valid_targets = positive.float()
        if cfg.validity_loss_key == "BCE":
            validity = F.binary_cross_entropy_with_logits(outputs["validity_logits"], valid_targets)
        elif cfg.validity_loss_key == "FOCAL":
            validity = focal_binary_loss(
                outputs["validity_logits"],
                valid_targets,
                positive_alpha=cfg.validity_positive_alpha,
                gamma=2.0,
            )
        else:
            raise KeyError(f"Unknown validity_loss_key={cfg.validity_loss_key!r}")
        total = total + cfg.validity_loss_weight * validity
    return total


def _repeated_v13_indices(loader: DataLoader) -> set[int]:
    sampler = getattr(loader, "sampler", None)
    if not isinstance(sampler, V13MixtureCandidateSampler):
        return set()
    values, counts = np.unique(np.asarray(sampler.indices, dtype=np.int64), return_counts=True)
    return set(values[counts > 1].astype(int).tolist())


def _make_v13_epoch_encoder_cache(model, loader: DataLoader):
    # Cache frozen encoder features only for candidates repeated within the epoch.
    from puma.training.stage2 import _FrozenEncoderEpochCache, _epoch_feature_cache_budget_bytes
    if model.encoder_trainable:
        return None
    repeated = _repeated_v13_indices(loader)
    return _FrozenEncoderEpochCache(repeated, _epoch_feature_cache_budget_bytes()) if repeated else None


def train_stage2_experiment_v13(
    runtime: RuntimeConfig,
    model_cfg: Stage2ModelConfig,
    seed: int,
    *,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Train one V13 experiment on the optimized train split and select on validation."""
    paths = runtime.paths
    paths.ensure()
    split_info = ensure_v13_split(runtime, force=False, check_sources=False)
    split_hash = str(split_info["split_hash"])
    csv_path = paths.stage2_file("stage2_v13_results.csv")
    key = {
        "stage": "stage2_v13",
        "experiment": model_cfg.name,
        "split": V13_SPLIT_NAME,
        "seed": int(seed),
    }
    expected_hash = _v13_run_hash(runtime, model_cfg, split_hash)
    if runtime.training.resume_from_results_csv:
        row = latest_completed_csv_row(csv_path, key)
        if row is not None and str(row.get("config_hash", "")) == expected_hash:
            print(f"SKIP recorded V13 performance: {model_cfg.name}/seed{seed}")
            return {**key, "status": "skipped", "skip_reason": "completed_csv_record"}

    append_csv_row_atomic(
        csv_path,
        {**key, "status": "running", "started_at": utc_now_iso(), "config_hash": expected_hash},
    )
    started = time.time()
    try:
        seed_everything(seed, runtime.training.deterministic)
        device = resolve_device()
        reset_peak_vram()
        store = PumaNpyStore.open(paths.artifact_dir)
        all_candidates = np.load(paths.stage1_existing_file("stage1_oof_candidates.npy"), mmap_mode="r")
        observed_folds = set(np.unique(all_candidates["fold"]).astype(int).tolist())
        expected_folds = set(range(runtime.data.number_of_folds))
        if observed_folds != expected_folds:
            raise RuntimeError(
                f"V13 requires complete leakage-safe Stage-1 OOF candidates for folds {sorted(expected_folds)}, "
                f"got {sorted(observed_folds)}."
            )

        train_roi_indices = np.asarray(split_info["train_roi_indices"], dtype=np.int64)
        val_roi_indices = np.asarray(split_info["val_roi_indices"], dtype=np.int64)
        train_mask = np.isin(all_candidates["roi_index"], train_roi_indices)
        val_mask = np.isin(all_candidates["roi_index"], val_roi_indices)
        if np.any(train_mask & val_mask):
            raise RuntimeError("Candidate leakage between V13 train and validation split.")
        train_base = np.asarray(all_candidates[train_mask])
        val_oof_all = np.asarray(all_candidates[val_mask])
        train_oof_positive = train_base[train_base["class_id"] != REJECT_CLASS_ID]
        val_oof_positive = val_oof_all[val_oof_all["class_id"] != REJECT_CLASS_ID]
        if len(train_base) == 0 or len(val_oof_all) == 0 or len(train_oof_positive) == 0:
            raise RuntimeError("V13 train/validation candidate split is empty.")

        perfect_train = _build_perfect_candidates(
            store, train_base, seed=seed, background_fraction=0.0, roi_indices=train_roi_indices
        )
        perfect_val = _build_perfect_candidates(
            store, val_oof_all, seed=seed + 1, background_fraction=0.0, roi_indices=val_roi_indices
        )
        final_phase = _final_schedule_phase(model_cfg.schedule_key)
        if final_phase == "GT_POS":
            val_candidates = perfect_val
        elif final_phase == "OOF_POS":
            val_candidates = val_oof_positive
        else:
            val_candidates = val_oof_all

        model = build_stage2_model(model_cfg, hf_token=hf_token).to(device)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        head_parameters, lora_parameters = split_optimizer_parameters(model)
        trainable_parameters = [*head_parameters, *lora_parameters]
        groups: list[dict[str, Any]] = [{
            "params": head_parameters,
            "lr": model_cfg.learning_rate,
            "weight_decay": model_cfg.weight_decay,
        }]
        if lora_parameters:
            groups.append({
                "params": lora_parameters,
                "lr": model_cfg.lora_learning_rate,
                "weight_decay": 0.0,
            })
        optimizer = build_adamw_parameter_groups(groups, device=device)
        scheduler = _warmup_cosine_scheduler(
            optimizer, runtime.training.epochs, model_cfg.warmup_epochs
        )
        amp_dtype = resolve_amp_dtype(runtime.training.prefer_bfloat16, device)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(runtime.training.amp and device.type == "cuda" and amp_dtype == torch.float16),
        )

        type_counts = np.bincount(
            perfect_train["class_id"].astype(int), minlength=REJECT_CLASS_ID
        )[:REJECT_CLASS_ID]
        reject_count = int(np.sum(train_base["class_id"] == REJECT_CLASS_ID))
        missing = [PUMA_CLASS_NAMES[i] for i in np.flatnonzero(type_counts == 0).astype(int)]
        if missing:
            raise RuntimeError("V13 training data is missing class(es): " + ", ".join(missing))
        class_counts = torch.as_tensor(
            np.r_[type_counts, max(reject_count, 1)], dtype=torch.float32, device=device
        )
        accumulation = runtime.training.stage2_accumulation_steps
        if accumulation != 1:
            print(
                f"V13 uses gradient accumulation={accumulation}; effective batch={runtime.training.effective_batch_size}, "
                f"micro batch={runtime.training.stage2_micro_batch_size}."
            )

        checkpoint = paths.stage2_file(f"stage2_v13_best_{model_cfg.name}_seed{seed}.pt")
        resume_checkpoint = paths.stage2_file(f"stage2_v13_resume_{model_cfg.name}_seed{seed}.pt")
        best = float("-inf")
        best_epoch = -1
        selection_phase = _final_schedule_phase(model_cfg.schedule_key)
        entered_selection_phase = False
        stopped_early = False
        epochs_trained = 0
        start_epoch = 1
        persistent_phase: str | None = None
        persistent_dataset: Stage2CandidateDataset | None = None
        persistent_loader: DataLoader | None = None
        dataset: Stage2CandidateDataset | None = None
        loader: DataLoader | None = None

        print(
            f"V13 split train={len(train_roi_indices)} ROIs / val={len(val_roi_indices)} ROIs: "
            f"train OOF={len(train_base)} ({len(train_oof_positive)} positive, {reject_count} reject), "
            f"val OOF={len(val_oof_all)}; split_hash={split_hash}."
        )

        if resume_checkpoint.exists():
            payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            extra = dict(payload.get("extra", {}))
            if str(extra.get("config_hash", "")) == expected_hash:
                restore_checkpoint_payload(payload, model)
                optimizer.load_state_dict(payload["optimizer_state"])
                _move_optimizer_state_to_device(optimizer, device)
                scheduler.load_state_dict(payload["scheduler_state"])
                if payload.get("scaler_state") is not None:
                    scaler.load_state_dict(payload["scaler_state"])
                start_epoch = int(payload.get("epoch", 0)) + 1
                epochs_trained = start_epoch - 1
                best = float(extra.get("best", float("-inf")))
                best_epoch = int(extra.get("best_epoch", -1))
                entered_selection_phase = bool(extra.get("entered_selection_phase", False))
                stopped_early = bool(extra.get("stopped_early", False))
                _restore_rng_state(extra)
                if stopped_early:
                    start_epoch = runtime.training.epochs + 1
                print(f"RESUME V13 {model_cfg.name}: epoch {start_epoch}")
            else:
                print(f"IGNORE stale V13 resume checkpoint: {resume_checkpoint.name}")
                resume_checkpoint.unlink(missing_ok=True)
            del payload

        for epoch in range(start_epoch, runtime.training.epochs + 1):
            epochs_trained = epoch
            phase = _phase_for_epoch(model_cfg.schedule_key, epoch, runtime.training.epochs)
            phase_epoch = _phase_epoch_number(model_cfg.schedule_key, epoch, runtime.training.epochs)
            if phase == selection_phase and not entered_selection_phase:
                entered_selection_phase = True
                best, best_epoch = float("-inf"), -1
                checkpoint.unlink(missing_ok=True)

            if phase == "GT_POS":
                phase_data = perfect_train
            elif phase == "OOF_POS":
                phase_data = train_oof_positive
            elif phase == "OOF_ALL":
                phase_data = train_base
            else:
                raise RuntimeError(f"Unsupported V13 phase {phase!r}")

            hard_scores = None
            if (
                phase == "OOF_ALL"
                and phase_epoch >= int(model_cfg.hard_negative_start_phase_epoch)
                and np.any(phase_data["class_id"] == REJECT_CLASS_ID)
            ):
                hard_scores = _hard_reject_scores(
                    model,
                    store,
                    phase_data,
                    model_cfg,
                    device,
                    batch_size=runtime.training.stage2_micro_batch_size,
                    workers=runtime.training.number_of_workers,
                    amp=runtime.training.amp,
                    prefer_bfloat16=runtime.training.prefer_bfloat16,
                )

            reusable = phase in {"GT_POS", "OOF_POS", "OOF_ALL"}
            sampling_seed = seed + epoch * 10007
            if (
                reusable
                and persistent_phase == phase
                and persistent_dataset is not None
                and persistent_loader is not None
            ):
                dataset, loader = persistent_dataset, persistent_loader
                dataset.set_epoch(epoch)
                _reset_v13_loader(loader, sampling_seed, hard_scores)
            else:
                persistent_loader = None
                persistent_dataset = None
                persistent_phase = None
                dataset = Stage2CandidateDataset(
                    store,
                    phase_data,
                    views=model_cfg.views,
                    augment=model_cfg.use_stain_augmentation,
                    seed=seed,
                    interface_key=model_cfg.interface_key,
                )
                dataset.set_epoch(epoch)
                loader = _make_v13_loader(
                    dataset,
                    phase_data,
                    batch_size=runtime.training.stage2_micro_batch_size,
                    workers=runtime.training.number_of_workers,
                    model_cfg=model_cfg,
                    train=True,
                    seed=sampling_seed,
                    hard_reject_scores=hard_scores,
                    persistent_workers=reusable,
                )
                if reusable:
                    persistent_phase, persistent_dataset, persistent_loader = phase, dataset, loader

            epoch_encoder_cache = _make_v13_epoch_encoder_cache(model, loader)
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
                    epoch_encoder_cache,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=runtime.training.amp and device.type == "cuda",
                ):
                    loss = v13_stage2_loss(outputs, labels, class_counts, model_cfg)
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
                # Correct the last partial accumulation group before stepping.
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
            epoch_encoder_cache = None

            stop_after_epoch = False
            if epoch % runtime.training.validation_interval == 0 or epoch == runtime.training.epochs:
                metrics, _, _ = evaluate_stage2(
                    model,
                    store,
                    val_candidates,
                    model_cfg,
                    device,
                    batch_size=runtime.training.stage2_micro_batch_size,
                    workers=runtime.training.number_of_workers,
                    roi_indices=val_roi_indices,
                    amp=runtime.training.amp,
                    prefer_bfloat16=runtime.training.prefer_bfloat16,
                )
                score = float(metrics.get(model_cfg.selection_metric, np.nan))
                print(
                    f"[{model_cfg.name} split={V13_SPLIT_NAME} s{seed}] epoch {epoch:02d} "
                    f"phase={phase} phase_epoch={phase_epoch:02d} "
                    f"loss={float(running_loss.item()) / max(step,1):.4f} "
                    f"val_{model_cfg.selection_metric}={score:.4f} "
                    f"type_present={float(metrics.get('conditional_type_macro_f1_present', np.nan)):.4f} "
                    f"rejectP={float(metrics.get('reject_precision', np.nan)):.3f} "
                    f"rejectR={float(metrics.get('reject_recall', np.nan)):.3f} "
                    f"threshold={float(metrics.get('validity_threshold', 0.0)):.2f}"
                )
                eligible = (
                    phase == selection_phase
                    and phase_epoch >= int(model_cfg.checkpoint_selection_start_phase_epoch)
                )
                improved = eligible and np.isfinite(score) and (
                    best_epoch < 0 or score > best + runtime.training.early_stopping_min_delta
                )
                if improved:
                    best, best_epoch = score, epoch
                    save_best_checkpoint(
                        checkpoint,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        score=score,
                        config=model_cfg,
                        extra={
                            "metrics": metrics,
                            "phase": phase,
                            "phase_epoch": phase_epoch,
                            "validity_threshold": float(metrics.get("validity_threshold", 0.5)),
                            "split_hash": split_hash,
                        },
                        trainable_only=True,
                        include_training_state=False,
                    )
                elif (
                    eligible
                    and runtime.training.early_stopping_enabled
                    and best_epoch >= 0
                    and epoch - best_epoch >= runtime.training.early_stopping_patience
                ):
                    stopped_early = True
                    stop_after_epoch = True
                    print(
                        f"Early stopping V13 {model_cfg.name}: no val macro-F1 improvement for "
                        f"{runtime.training.early_stopping_patience} epochs (best={best:.4f} @ {best_epoch})."
                    )

            next_phase = (
                _phase_for_epoch(model_cfg.schedule_key, epoch + 1, runtime.training.epochs)
                if epoch < runtime.training.epochs else None
            )
            if (
                epoch % _resume_interval_epochs() == 0
                or next_phase != phase
                or epoch == runtime.training.epochs
                or stop_after_epoch
            ):
                extra = {
                    "config_hash": expected_hash,
                    "best": best,
                    "best_epoch": best_epoch,
                    "entered_selection_phase": entered_selection_phase,
                    "stopped_early": stopped_early,
                    "phase": phase,
                    "phase_epoch": phase_epoch,
                    "split_hash": split_hash,
                }
                extra.update(_capture_rng_state())
                save_best_checkpoint(
                    resume_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    score=best,
                    config=model_cfg,
                    extra=extra,
                    trainable_only=True,
                    include_training_state=True,
                )
            if stop_after_epoch:
                break

        if not checkpoint.exists():
            raise RuntimeError(f"No valid V13 checkpoint produced for {model_cfg.name}.")
        optimizer.zero_grad(set_to_none=True)
        persistent_loader = persistent_dataset = None
        del optimizer, scheduler, scaler, loader, dataset
        release_cuda_memory(synchronize=False)

        payload = load_checkpoint(checkpoint, model, device)
        selected_epoch = int(payload.get("epoch", -1))
        selected_score = float(payload.get("score", np.nan))
        threshold = float(payload.get("extra", {}).get("validity_threshold", 0.5))
        del payload
        getattr(model, "_puma_frozen_feature_caches", {}).clear()
        metrics, _, predictions = evaluate_stage2(
            model,
            store,
            val_candidates,
            model_cfg,
            device,
            batch_size=runtime.training.stage2_micro_batch_size,
            workers=runtime.training.number_of_workers,
            roi_indices=val_roi_indices,
            cache_frozen_encoder=False,
            amp=runtime.training.amp,
            prefer_bfloat16=runtime.training.prefer_bfloat16,
            validity_threshold=threshold,
        )
        source_ids, classes, confidence, probabilities = predictions
        detector_lookup = {int(row["oof_row_id"]): float(row["confidence"]) for row in val_candidates}
        detector_confidence = np.asarray(
            [detector_lookup.get(int(source), np.nan) for source in source_ids], dtype=np.float32
        )
        matrix = np.column_stack(
            [source_ids, classes, confidence, detector_confidence, probabilities]
        ).astype(np.float32)
        prediction_path = paths.stage2_file(
            f"stage2_v13_predictions_{model_cfg.name}_seed{seed}.npy"
        )
        atomic_save_numpy(prediction_path, matrix, allow_pickle=False)
        total, trainable = count_trainable_parameters(model)
        completed = {
            **key,
            "config_hash": expected_hash,
            "split_hash": split_hash,
            "status": "completed",
            "completed_at": utc_now_iso(),
            "duration_minutes": (time.time() - started) / 60.0,
            "best_epoch": selected_epoch,
            "val_best_metric": selected_score,
            "selection_metric": model_cfg.selection_metric,
            "validity_threshold": threshold,
            "epochs_trained": epochs_trained,
            "stopped_early": bool(stopped_early),
            "early_stopping_patience": runtime.training.early_stopping_patience,
            "best_checkpoint": str(checkpoint),
            "prediction_npy": str(prediction_path),
            "parameters_total": total,
            "parameters_trainable": trainable,
            "train_oof_candidates": len(train_base),
            "train_oof_positives": len(train_oof_positive),
            "train_oof_rejects": reject_count,
            "val_candidates": len(val_candidates),
            "train_roi_count": len(train_roi_indices),
            "val_roi_count": len(val_roi_indices),
            "train_type_counts_json": json.dumps(type_counts.astype(int).tolist()),
            "effective_batch_size": runtime.training.effective_batch_size,
            "stage2_micro_batch_size": runtime.training.stage2_micro_batch_size,
            "encoder_micro_batch_size": model_cfg.encoder_micro_batch_size,
            "peak_vram_mb": peak_vram_mb(),
            **metrics,
        }
        append_csv_row_atomic(csv_path, completed)
        resume_checkpoint.unlink(missing_ok=True)
        return completed
    except Exception as exc:
        append_csv_row_atomic(
            csv_path,
            {
                **key,
                "config_hash": expected_hash,
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
