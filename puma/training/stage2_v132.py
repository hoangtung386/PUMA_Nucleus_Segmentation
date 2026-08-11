from __future__ import annotations

"""Stage-2 V13.2 training on the fixed 80/20 development split.

Core invariants:
- Stage-1 candidates are complete five-fold OOF predictions.
- Stage-2 split is case-grouped and fixed before experiments.
- Curriculum is exactly 30% GT_POS, 30% OOF_POS, 40% OOF_ALL.
- 50 epochs => 15/15/20; 100 epochs => 30/30/40.
- Strong rare exposure is case-aware and repeat-capped.
- Validity/reject learning is enabled only in OOF_ALL.
- Phase-aware learning rates avoid starving the final OOF_ALL phase.
"""

import json
import math
from dataclasses import asdict
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from puma.config import (
    PUMA_CLASS_NAMES,
    REJECT_CLASS_ID,
    STAGE2_GEOMETRY_NAMES,
    TAIL_CLASS_IDS,
    RuntimeConfig,
    Stage2ModelConfig,
)
from puma.data.datasets import PumaNpyStore, Stage2CandidateDataset, stage2_collate
from puma.models.stage2 import build_stage2_model, effective_number_weights
from puma.stage2.train_val_split import SPLIT_TRAIN, SPLIT_VAL, create_split, validate_final_split
from puma.training.stage2 import (
    _FrozenEncoderEpochCache,
    _build_perfect_candidates,
    _capture_rng_state,
    _epoch_feature_cache_budget_bytes,
    _forward_batch_streamed,
    _move_optimizer_state_to_device,
    _predict_candidates,
    _restore_rng_state,
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

V132_SPLIT_NAME = "optimized_train80_val20"
V132_IMPLEMENTATION_REVISION = 3
_VALID_STAGE2_EPOCHS = {50, 100}


def _split_paths(runtime: RuntimeConfig) -> tuple[Path, Path, Path]:
    directory = runtime.paths.artifact_dir / "train_val_split"
    return (
        directory / "puma_train_val_assignments.npy",
        directory / "puma_train_val_indices.npz",
        directory / "puma_train_val_split_metadata.json",
    )


def ensure_v132_split(
    runtime: RuntimeConfig,
    *,
    force: bool = False,
    val_fraction: float = 0.20,
    seed: int = 2026,
    check_sources: bool = True,
) -> dict[str, Any]:
    assignments_path, indices_path, metadata_path = _split_paths(runtime)
    if force or not (assignments_path.exists() and indices_path.exists() and metadata_path.exists()):
        create_split(
            runtime.paths.root,
            val_fraction=val_fraction,
            seed=seed,
            output_dir=assignments_path.parent,
            check_sources=check_sources,
        )
    manifest = np.load(
        runtime.paths.preprocessing_file("puma_roi_manifest.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    assignments = np.load(assignments_path, allow_pickle=False)
    if assignments.shape != (len(manifest),):
        raise RuntimeError(
            f"V13.2 split length {assignments.shape} does not match manifest length {len(manifest)}."
        )
    if set(np.unique(assignments).astype(int).tolist()) != {int(SPLIT_TRAIN), int(SPLIT_VAL)}:
        raise RuntimeError("V13.2 split must contain exactly train=0 and validation=1 assignments.")
    diagnostics = validate_final_split(manifest, assignments)
    if diagnostics["case_leakage_count"] != 0:
        raise RuntimeError("Case leakage detected in V13.2 split.")
    if diagnostics["missing_train_classes"] or diagnostics["missing_validation_classes"]:
        raise RuntimeError(
            "Every PUMA class must occur in both V13.2 train and validation. "
            f"train missing={diagnostics['missing_train_classes']}, "
            f"val missing={diagnostics['missing_validation_classes']}"
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


def phase_bounds(epochs: int) -> tuple[int, int, int]:
    """Return exact inclusive phase lengths (GT_POS, OOF_POS, OOF_ALL)."""
    epochs = int(epochs)
    if epochs not in _VALID_STAGE2_EPOCHS:
        raise ValueError(f"V13.2 supports Stage-2 epochs 50 or 100 only, got {epochs}.")
    p1 = int(round(epochs * 0.30))
    p2 = int(round(epochs * 0.30))
    p3 = epochs - p1 - p2
    return p1, p2, p3


def phase_for_epoch(epoch: int, epochs: int) -> tuple[str, int, int]:
    p1, p2, p3 = phase_bounds(epochs)
    if epoch < 1 or epoch > epochs:
        raise ValueError(f"epoch must be in [1,{epochs}], got {epoch}")
    if epoch <= p1:
        return "GT_POS", epoch, p1
    if epoch <= p1 + p2:
        return "OOF_POS", epoch - p1, p2
    return "OOF_ALL", epoch - p1 - p2, p3


def _cosine_between(start: float, end: float, step: int, length: int) -> float:
    if length <= 1:
        return float(end)
    progress = (int(step) - 1) / float(length - 1)
    progress = min(max(progress, 0.0), 1.0)
    return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress)))


def phase_learning_rates(cfg: Stage2ModelConfig, epoch: int, epochs: int) -> tuple[float, float]:
    phase, phase_epoch, phase_length = phase_for_epoch(epoch, epochs)
    if phase == "GT_POS":
        warmup = min(max(int(cfg.warmup_epochs), 0), phase_length)
        if warmup > 0 and phase_epoch <= warmup:
            if warmup == 1:
                type_lr = float(cfg.phase1_start_lr)
            else:
                fraction = (phase_epoch - 1) / float(warmup - 1)
                type_lr = float(cfg.phase1_start_lr) * (0.20 + 0.80 * fraction)
        else:
            decay_step = phase_epoch - warmup
            decay_length = max(phase_length - warmup, 1)
            type_lr = _cosine_between(
                cfg.phase1_start_lr, cfg.phase1_end_lr, decay_step, decay_length
            )
        validity_lr = 0.0
    elif phase == "OOF_POS":
        type_lr = _cosine_between(
            cfg.phase2_start_lr, cfg.phase2_end_lr, phase_epoch, phase_length
        )
        validity_lr = 0.0
    else:
        type_lr = _cosine_between(
            cfg.phase3_start_lr, cfg.phase3_end_lr, phase_epoch, phase_length
        )
        validity_lr = _cosine_between(
            cfg.phase3_validity_start_lr,
            cfg.phase3_validity_end_lr,
            phase_epoch,
            phase_length,
        )
    return float(type_lr), float(validity_lr)


def _set_optimizer_lrs(optimizer: torch.optim.Optimizer, type_lr: float, validity_lr: float) -> None:
    for group in optimizer.param_groups:
        role = str(group.get("puma_role", "type"))
        group["lr"] = float(validity_lr if role == "validity" else type_lr)


def _should_validate(epoch: int, epochs: int, interval: int) -> bool:
    p1, p2, _ = phase_bounds(epochs)
    if epoch in {p1, p1 + p2, epochs}:
        return True
    phase, phase_epoch, _ = phase_for_epoch(epoch, epochs)
    return phase == "OOF_ALL" and phase_epoch % max(1, int(interval)) == 0


def _semantic_model_config(cfg: Stage2ModelConfig) -> dict[str, Any]:
    """Return result-affecting Stage-2 config, excluding execution-only chunk sizes."""
    payload = asdict(cfg)
    payload.pop("encoder_micro_batch_size", None)
    return payload


def _run_hash(runtime: RuntimeConfig, cfg: Stage2ModelConfig, split_hash: str) -> str:
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
        "version": "13.2",
        "implementation_revision": V132_IMPLEMENTATION_REVISION,
        "model": _semantic_model_config(cfg),
        "geometry_revision": "puma_roi_1024_7d_v3",
        "geometry_names": list(STAGE2_GEOMETRY_NAMES),
        "image_shape": [int(runtime.data.image_height), int(runtime.data.image_width)],
        "split": V132_SPLIT_NAME,
        "split_hash": split_hash,
        "training": {
            "stage2_epochs": tr.stage2_epochs,
            "stage2_effective_batch_size": tr.stage2_effective_batch_size,
            # Physical micro-batch, UNI2 encoder chunk size, and DataLoader workers are
            # execution-only knobs. OOM fallback may change them while preserving the
            # exact optimizer effective batch and model semantics.

            "amp": tr.amp,
            "prefer_bfloat16": tr.prefer_bfloat16,
            "deterministic": tr.deterministic,
            "validation_interval": tr.validation_interval,
            "early_stopping_enabled": tr.early_stopping_enabled,
            "early_stopping_patience": tr.early_stopping_patience,
            "early_stopping_min_delta": tr.early_stopping_min_delta,
            "gradient_clip_norm": tr.gradient_clip_norm,
        },
        "oof": oof_signature,
    })


class V132RareExposureSampler(Sampler[int]):
    """Batch-structured sampler with strong rare exposure and global repeat caps.

    Exact tail quotas are attempted per full batch. If a class is so small that the
    requested quota would exceed the configured max-repeat budget, its effective quota
    is reduced to the largest feasible value and reported in ``stats``.
    """

    def __init__(
        self,
        candidates: np.ndarray,
        cfg: Stage2ModelConfig,
        *,
        batch_size: int,
        phase: str,
        candidate_case_ids: np.ndarray,
        seed: int,
        hard_reject_scores: np.ndarray | None = None,
        hard_positive_scores: np.ndarray | None = None,
    ) -> None:
        self.candidates = candidates
        self.cfg = cfg
        self.batch_size = int(batch_size)
        self.phase = str(phase)
        self.candidate_case_ids = np.asarray(candidate_case_ids).astype(str)
        if len(self.candidate_case_ids) != len(candidates):
            raise ValueError("candidate_case_ids must align one-to-one with candidates.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.indices: list[int] = []
        self.repeat_counts = np.zeros(len(candidates), dtype=np.int32)
        self.stats: dict[str, Any] = {}
        self.reset(seed, hard_reject_scores, hard_positive_scores)

    @staticmethod
    def _quota_for_phase(cfg: Stage2ModelConfig, phase: str) -> int:
        if not cfg.use_strong_rare_sampling:
            return 0
        return {
            "GT_POS": int(cfg.rare_quota_gt_per_class),
            "OOF_POS": int(cfg.rare_quota_oof_pos_per_class),
            "OOF_ALL": int(cfg.rare_quota_oof_all_per_class),
        }[phase]

    @staticmethod
    def _case_interleaved_order(
        pool: np.ndarray,
        case_ids: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if len(pool) <= 1:
            return np.asarray(pool, dtype=np.int64)
        buckets: dict[str, list[int]] = {}
        for index in np.asarray(pool, dtype=np.int64):
            buckets.setdefault(str(case_ids[index]), []).append(int(index))
        for values in buckets.values():
            rng.shuffle(values)
        case_order = list(buckets)
        rng.shuffle(case_order)
        output: list[int] = []
        while case_order:
            next_cases: list[str] = []
            for case in case_order:
                values = buckets[case]
                if values:
                    output.append(values.pop())
                if values:
                    next_cases.append(case)
            rng.shuffle(next_cases)
            case_order = next_cases
        return np.asarray(output, dtype=np.int64)

    @classmethod
    def _draw_capped(
        cls,
        pool: np.ndarray,
        number: int,
        rng: np.random.Generator,
        counts: np.ndarray,
        caps: np.ndarray,
        case_ids: np.ndarray,
        *,
        case_aware: bool = False,
    ) -> list[int]:
        result: list[int] = []
        pool = np.asarray(pool, dtype=np.int64)
        remaining = int(number)
        while remaining > 0 and len(pool):
            eligible = pool[counts[pool] < caps[pool]]
            if not len(eligible):
                break
            order = (
                cls._case_interleaved_order(eligible, case_ids, rng)
                if case_aware
                else rng.permutation(eligible)
            )
            take = min(remaining, len(order))
            chosen = np.asarray(order[:take], dtype=np.int64)
            result.extend(chosen.astype(int, copy=False).tolist())
            np.add.at(counts, chosen, 1)
            remaining -= take
        return result

    def reset(
        self,
        seed: int,
        hard_reject_scores: np.ndarray | None = None,
        hard_positive_scores: np.ndarray | None = None,
    ) -> None:
        rng = np.random.default_rng(int(seed))
        labels = self.candidates["class_id"].astype(int)
        all_indices = np.arange(len(self.candidates), dtype=np.int64)
        positives = all_indices[labels != REJECT_CLASS_ID]
        rejects = all_indices[labels == REJECT_CLASS_ID]
        if len(positives) == 0:
            raise RuntimeError("V13.2 Stage-2 training phase contains no positive candidates.")

        n_batches = max(1, math.ceil(len(self.candidates) / self.batch_size))
        epoch_draws = n_batches * self.batch_size
        positive_per_batch = (
            self.batch_size
            if len(rejects) == 0
            else max(1, min(self.batch_size, int(round(self.cfg.sampler_positive_fraction * self.batch_size))))
        )
        reject_per_batch = self.batch_size - positive_per_batch

        counts = np.zeros(len(self.candidates), dtype=np.int32)
        caps = np.full(len(self.candidates), int(self.cfg.sampler_max_repeats), dtype=np.int32)
        tail_mask = np.isin(labels, np.asarray(TAIL_CLASS_IDS, dtype=int))
        caps[tail_mask] = int(self.cfg.sampler_tail_max_repeats)

        requested_quota = self._quota_for_phase(self.cfg, self.phase)
        effective_quota: dict[int, int] = {}
        for class_id in TAIL_CLASS_IDS:
            pool_size = int(np.sum(labels == class_id))
            if pool_size == 0:
                raise RuntimeError(
                    f"Rare class {PUMA_CLASS_NAMES[class_id]} is absent from phase {self.phase}."
                )
            if requested_quota <= 0:
                effective_quota[class_id] = 0
                continue
            max_total = pool_size * int(self.cfg.sampler_tail_max_repeats)
            feasible = max_total // n_batches
            effective_quota[class_id] = min(requested_quota, int(feasible))

        total_tail_quota = sum(effective_quota.values())
        if total_tail_quota > positive_per_batch:
            scale = positive_per_batch / float(total_tail_quota)
            effective_quota = {
                key: int(math.floor(value * scale)) for key, value in effective_quota.items()
            }

        class_pools = {
            class_id: positives[labels[positives] == class_id]
            for class_id in range(REJECT_CLASS_ID)
            if np.any(labels[positives] == class_id)
        }
        common_positive_pool = positives[~np.isin(labels[positives], np.asarray(TAIL_CLASS_IDS, dtype=int))]

        hard_reject_scores = (
            np.asarray(hard_reject_scores, dtype=np.float32)
            if hard_reject_scores is not None else None
        )
        hard_positive_scores = (
            np.asarray(hard_positive_scores, dtype=np.float32)
            if hard_positive_scores is not None else None
        )
        for scores, name in ((hard_reject_scores, "hard_reject_scores"), (hard_positive_scores, "hard_positive_scores")):
            if scores is not None and len(scores) != len(self.candidates):
                raise ValueError(f"{name} must have one score per candidate.")

        output: list[int] = []
        for _batch in range(n_batches):
            batch: list[int] = []
            # 1) Guaranteed tail exposure, case-aware and unique-first.
            for class_id in TAIL_CLASS_IDS:
                quota = int(effective_quota.get(class_id, 0))
                if quota <= 0:
                    continue
                pool = class_pools.get(class_id, np.empty(0, dtype=np.int64))
                hard_target = 0
                if hard_positive_scores is not None and self.cfg.hard_rare_fraction > 0:
                    hard_target = int(round(quota * float(self.cfg.hard_rare_fraction)))
                if hard_target > 0 and len(pool):
                    ranked = pool[np.argsort(-hard_positive_scores[pool])]
                    hard_pool = ranked[: max(1, len(ranked) // 2)]
                    batch.extend(self._draw_capped(
                        hard_pool, hard_target, rng, counts, caps,
                        self.candidate_case_ids, case_aware=True,
                    ))
                remaining = quota - sum(labels[index] == class_id for index in batch)
                if remaining > 0:
                    batch.extend(self._draw_capped(
                        pool, remaining, rng, counts, caps,
                        self.candidate_case_ids, case_aware=True,
                    ))

            # 2) Balanced positive component.
            remaining_positive = max(0, positive_per_batch - len(batch))
            balanced_target = min(
                remaining_positive,
                int(round(float(self.cfg.sampler_balanced_positive_fraction) * positive_per_batch)),
            )
            active_classes = (
                [class_id for class_id in class_pools if class_id not in TAIL_CLASS_IDS]
                if self.cfg.use_strong_rare_sampling else list(class_pools)
            )
            balanced_added = 0
            while balanced_added < balanced_target and active_classes:
                next_active: list[int] = []
                for class_id_value in rng.permutation(active_classes):
                    class_id = int(class_id_value)
                    chosen = self._draw_capped(
                        class_pools[class_id], 1, rng, counts, caps,
                        self.candidate_case_ids,
                        case_aware=(class_id in TAIL_CLASS_IDS),
                    )
                    if chosen:
                        batch.extend(chosen)
                        balanced_added += 1
                    pool = class_pools[class_id]
                    if np.any(counts[pool] < caps[pool]):
                        next_active.append(class_id)
                    if balanced_added >= balanced_target:
                        break
                active_classes = next_active

            # 3) Natural positive backfill.
            if len(batch) < positive_per_batch:
                natural_pool = (
                    common_positive_pool
                    if self.cfg.use_strong_rare_sampling and len(common_positive_pool)
                    else positives
                )
                batch.extend(self._draw_capped(
                    natural_pool,
                    positive_per_batch - len(batch),
                    rng,
                    counts,
                    caps,
                    self.candidate_case_ids,
                ))
                if len(batch) < positive_per_batch:
                    batch.extend(self._draw_capped(
                        positives,
                        positive_per_batch - len(batch),
                        rng,
                        counts,
                        caps,
                        self.candidate_case_ids,
                    ))

            # 4) Reject quota, half hard when scores are available.
            reject_draws: list[int] = []
            if reject_per_batch > 0 and len(rejects):
                hard_target = 0
                if hard_reject_scores is not None:
                    hard_target = int(round(reject_per_batch * float(self.cfg.hard_reject_fraction)))
                if hard_target > 0:
                    ranked = rejects[np.argsort(-hard_reject_scores[rejects])]
                    hard_pool = ranked[: max(1, len(ranked) // 2)]
                    reject_draws.extend(self._draw_capped(
                        hard_pool, hard_target, rng, counts, caps,
                        self.candidate_case_ids,
                    ))
                reject_draws.extend(self._draw_capped(
                    rejects,
                    reject_per_batch - len(reject_draws),
                    rng,
                    counts,
                    caps,
                    self.candidate_case_ids,
                ))
                batch.extend(reject_draws)

            # 5) Strict capped backfill. Configured repeat limits are real maxima.
            if len(batch) < self.batch_size:
                pool = positives if len(rejects) == 0 else all_indices
                batch.extend(self._draw_capped(
                    pool, self.batch_size - len(batch), rng, counts, caps, self.candidate_case_ids,
                ))
            if len(batch) < self.batch_size:
                raise RuntimeError(
                    "V13.2 sampler exhausted strict repeat caps before filling a batch. "
                    f"phase={self.phase}, missing={self.batch_size-len(batch)}. "
                    "Increase repeat caps explicitly; they are never relaxed implicitly."
                )
            rng.shuffle(batch)
            output.extend(batch[: self.batch_size])

        self.indices = output[:epoch_draws]
        self.repeat_counts = counts
        sampled_labels = labels[np.asarray(self.indices, dtype=np.int64)]
        self.stats = {
            "phase": self.phase,
            "number_of_batches": n_batches,
            "epoch_draws": len(self.indices),
            "positive_per_batch": positive_per_batch,
            "reject_per_batch": reject_per_batch,
            "requested_tail_quota_per_class": requested_quota,
            "effective_tail_quota_per_class": {
                PUMA_CLASS_NAMES[k]: int(v) for k, v in effective_quota.items()
            },
            "sampled_class_counts": np.bincount(
                sampled_labels, minlength=REJECT_CLASS_ID + 1
            ).astype(int).tolist(),
            "max_observed_repeat": int(counts.max()) if len(counts) else 0,
        }

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _make_epoch_encoder_cache(model, sampler: Sampler[int] | None) -> _FrozenEncoderEpochCache | None:
    """Cache frozen UNI2-h features only for candidates repeated inside this epoch."""
    if sampler is None or getattr(model, "encoder_trainable", False):
        return None
    indices = getattr(sampler, "indices", None)
    if not indices:
        return None
    values, counts = np.unique(np.asarray(indices, dtype=np.int64), return_counts=True)
    repeated = set(values[counts > 1].astype(int, copy=False).tolist())
    if not repeated:
        return None
    return _FrozenEncoderEpochCache(repeated, _epoch_feature_cache_budget_bytes())


def _reset_epoch_loader(
    loader: DataLoader,
    dataset: Stage2CandidateDataset,
    *,
    epoch: int,
    seed: int,
    hard_reject_scores: np.ndarray | None,
    hard_positive_scores: np.ndarray | None,
) -> None:
    dataset.set_epoch(epoch)
    sampler = loader.sampler
    if not isinstance(sampler, V132RareExposureSampler):
        raise TypeError("Unexpected Stage-2 sampler type in V13.2 loader.")
    sampler.reset(seed, hard_reject_scores, hard_positive_scores)


def _candidate_case_ids(candidates: np.ndarray, manifest: np.ndarray) -> np.ndarray:
    roi_indices = candidates["roi_index"].astype(np.int64)
    if "case_id" in (manifest.dtype.names or ()):
        return np.asarray(manifest["case_id"][roi_indices]).astype(str)
    if "roi_id" in (manifest.dtype.names or ()):
        return np.asarray(manifest["roi_id"][roi_indices]).astype(str)
    return roi_indices.astype(str)


def _make_loader(
    dataset: Stage2CandidateDataset,
    candidates: np.ndarray,
    cfg: Stage2ModelConfig,
    manifest: np.ndarray,
    *,
    batch_size: int,
    workers: int,
    phase: str,
    seed: int,
    hard_reject_scores: np.ndarray | None,
    hard_positive_scores: np.ndarray | None,
    persistent_workers: bool,
) -> DataLoader:
    sampler = V132RareExposureSampler(
        candidates,
        cfg,
        batch_size=batch_size,
        phase=phase,
        candidate_case_ids=_candidate_case_ids(candidates, manifest),
        seed=seed,
        hard_reject_scores=hard_reject_scores,
        hard_positive_scores=hard_positive_scores,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=stage2_collate,
        worker_init_fn=worker_seed_init,
        **dataloader_performance_kwargs(
            workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=persistent_workers,
        ),
    )


def _type_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    raw_class_counts: torch.Tensor,
    cfg: Stage2ModelConfig,
) -> torch.Tensor:
    positive = labels != REJECT_CLASS_ID
    if not bool(positive.any()):
        return logits.sum() * 0.0
    y = labels[positive]
    z = logits[positive]
    counts = raw_class_counts[:REJECT_CLASS_ID].clamp_min(1.0)
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
        return (
            sample_weights * (1.0 - pt).pow(cfg.type_focal_gamma) * ce
        ).sum() / sample_weights.sum().clamp_min(1e-8)
    raise KeyError(f"Unknown V13.2 type_loss_key={cfg.type_loss_key!r}")


def stage2_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    raw_class_counts: torch.Tensor,
    cfg: Stage2ModelConfig,
    *,
    validity_active: bool,
) -> torch.Tensor:
    positive = labels != REJECT_CLASS_ID
    total = cfg.type_loss_weight * _type_loss(
        outputs["type_logits"], labels, raw_class_counts, cfg
    )
    if validity_active:
        validity_logits = outputs.get("validity_logits")
        if validity_logits is None:
            raise RuntimeError("V13.2 OOF_ALL requires a validity head.")
        valid_targets = positive.float()
        validity = F.binary_cross_entropy_with_logits(validity_logits, valid_targets)
        total = total + cfg.validity_loss_weight * validity
    return total


def _hardness_scores(
    model,
    store: PumaNpyStore,
    candidates: np.ndarray,
    cfg: Stage2ModelConfig,
    device: torch.device,
    *,
    batch_size: int,
    workers: int,
    amp: bool,
    prefer_bfloat16: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """One inference pass computes both hard reject and hard positive scores."""
    dataset = Stage2CandidateDataset(
        store, candidates, views=cfg.views, augment=False, interface_key=cfg.interface_key
    )
    source_ids, _, _, probabilities = _predict_candidates(
        model,
        dataset,
        candidates,
        batch_size,
        workers,
        device,
        amp=amp,
        amp_dtype=resolve_amp_dtype(prefer_bfloat16, device),
        cache_frozen_encoder=True,
    )
    row_by_source = {
        int(candidate["oof_row_id"]): index for index, candidate in enumerate(candidates)
    }
    reject_scores = np.zeros(len(candidates), dtype=np.float32)
    positive_scores = np.zeros(len(candidates), dtype=np.float32)
    for source, probability in zip(source_ids, probabilities, strict=True):
        index = row_by_source.get(int(source))
        if index is None:
            continue
        label = int(candidates["class_id"][index])
        if label == REJECT_CLASS_ID:
            reject_scores[index] = float(1.0 - probability[REJECT_CLASS_ID])
        elif 0 <= label < REJECT_CLASS_ID:
            positive_scores[index] = float(1.0 - probability[label])
    return reject_scores, positive_scores


def _split_trainable_parameters(model) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    type_fusion: list[torch.nn.Parameter] = []
    validity: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("validity_classifier."):
            validity.append(parameter)
        else:
            type_fusion.append(parameter)
    if not type_fusion:
        raise RuntimeError("No trainable Stage-2 type/fusion parameters found.")
    if not validity:
        raise RuntimeError("V13.2 requires a trainable validity head.")
    return type_fusion, validity


def train_stage2_experiment_v132(
    runtime: RuntimeConfig,
    model_cfg: Stage2ModelConfig,
    seed: int,
    *,
    hf_token: str | None = None,
) -> dict[str, Any]:
    epochs = int(runtime.training.stage2_epochs)
    phase_bounds(epochs)  # validates 50/100 and exact 30/30/40.
    if model_cfg.use_lora:
        raise ValueError("LoRA is intentionally excluded from V13.2 screening/final profiles.")

    paths = runtime.paths
    paths.ensure()
    split_info = ensure_v132_split(runtime, force=False, check_sources=False)
    split_hash = str(split_info["split_hash"])
    csv_path = paths.stage2_file("stage2_v132_results.csv")
    key = {
        "stage": "stage2_v132",
        "experiment": model_cfg.name,
        "split": V132_SPLIT_NAME,
        "seed": int(seed),
        "epoch_profile": epochs,
    }
    expected_hash = _run_hash(runtime, model_cfg, split_hash)
    if runtime.training.resume_from_results_csv:
        row = latest_completed_csv_row(csv_path, key)
        if row is not None and str(row.get("config_hash", "")) == expected_hash:
            ck = paths.stage2_output_dir / Path(str(row.get("best_checkpoint", ""))).name
            pred = paths.stage2_output_dir / Path(str(row.get("prediction_npy", ""))).name
            ck_ok = False
            if ck.exists():
                try:
                    payload = torch.load(ck, map_location="cpu", weights_only=False)
                    ck_ok = str(payload.get("extra", {}).get("config_hash", "")) == expected_hash
                    del payload
                except Exception:
                    ck_ok = False
            if ck_ok and pred.exists():
                print(f"SKIP recorded V13.2 performance: {model_cfg.name}/seed{seed}/{epochs}ep")
                return {**key, "status": "skipped", "skip_reason": "completed_artifacts_verified",
                        "best_checkpoint": str(ck), "prediction_npy": str(pred)}
            print(f"REBUILD incomplete V13.2 record: checkpoint_ok={ck_ok}, predictions_ok={pred.exists()}")

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
        manifest = np.asarray(store.manifest)
        all_candidates = np.load(
            paths.stage1_existing_file("stage1_oof_candidates.npy"), mmap_mode="r", allow_pickle=False
        )
        observed_folds = set(np.unique(all_candidates["fold"]).astype(int).tolist())
        expected_folds = set(range(runtime.data.number_of_folds))
        if observed_folds != expected_folds:
            raise RuntimeError(
                "V13.2 requires complete leakage-safe Stage-1 OOF candidates for all five folds; "
                f"got {sorted(observed_folds)}."
            )

        train_roi_indices = np.asarray(split_info["train_roi_indices"], dtype=np.int64)
        val_roi_indices = np.asarray(split_info["val_roi_indices"], dtype=np.int64)
        train_mask = np.isin(all_candidates["roi_index"], train_roi_indices)
        val_mask = np.isin(all_candidates["roi_index"], val_roi_indices)
        if np.any(train_mask & val_mask):
            raise RuntimeError("Candidate leakage between V13.2 train and validation split.")
        train_base = np.asarray(all_candidates[train_mask])
        val_oof_all = np.asarray(all_candidates[val_mask])
        train_oof_positive = train_base[train_base["class_id"] != REJECT_CLASS_ID]
        if len(train_base) == 0 or len(val_oof_all) == 0 or len(train_oof_positive) == 0:
            raise RuntimeError("V13.2 train/validation candidate split is empty.")
        perfect_train = _build_perfect_candidates(
            store, train_base, seed=seed, background_fraction=0.0, roi_indices=train_roi_indices
        )

        model = build_stage2_model(model_cfg, hf_token=hf_token).to(device)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        type_parameters, validity_parameters = _split_trainable_parameters(model)
        trainable_parameters = [*type_parameters, *validity_parameters]
        optimizer = build_adamw_parameter_groups(
            [
                {
                    "params": type_parameters,
                    "lr": model_cfg.phase1_start_lr,
                    "weight_decay": model_cfg.weight_decay,
                    "puma_role": "type",
                },
                {
                    "params": validity_parameters,
                    "lr": 0.0,
                    "weight_decay": model_cfg.weight_decay,
                    "puma_role": "validity",
                },
            ],
            device=device,
        )
        amp_dtype = resolve_amp_dtype(runtime.training.prefer_bfloat16, device)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(runtime.training.amp and device.type == "cuda" and amp_dtype == torch.float16),
        )

        type_counts = np.bincount(
            perfect_train["class_id"].astype(int), minlength=REJECT_CLASS_ID
        )[:REJECT_CLASS_ID]
        missing = [PUMA_CLASS_NAMES[i] for i in np.flatnonzero(type_counts == 0).astype(int)]
        if missing:
            raise RuntimeError("V13.2 training data is missing class(es): " + ", ".join(missing))
        raw_class_counts = torch.as_tensor(
            np.r_[type_counts, max(int(np.sum(train_base["class_id"] == REJECT_CLASS_ID)), 1)],
            dtype=torch.float32,
            device=device,
        )

        accumulation = runtime.training.stage2_accumulation_steps
        if accumulation < 1:
            raise RuntimeError("Invalid Stage-2 gradient accumulation.")
        run_tag = expected_hash[:10]
        checkpoint = paths.stage2_file(
            f"stage2_v132_best_{model_cfg.name}_{epochs}ep_seed{seed}_{run_tag}.pt"
        )
        resume_checkpoint = paths.stage2_file(
            f"stage2_v132_resume_{model_cfg.name}_{epochs}ep_seed{seed}_{run_tag}.pt"
        )
        best, best_epoch = float("-inf"), -1
        epochs_trained = 0
        start_epoch = 1
        hard_reject_scores: np.ndarray | None = None
        hard_positive_scores: np.ndarray | None = None
        active_phase: str | None = None
        train_dataset: Stage2CandidateDataset | None = None
        train_loader: DataLoader | None = None

        print(
            f"V13.2 {epochs}ep split: train={len(train_roi_indices)} ROI / val={len(val_roi_indices)} ROI; "
            f"OOF train={len(train_base)}, val={len(val_oof_all)}, split_hash={split_hash}."
        )

        if resume_checkpoint.exists():
            payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            extra = dict(payload.get("extra", {}))
            recorded_best_epoch = int(extra.get("best_epoch", -1))
            resumable = (
                str(extra.get("config_hash", "")) == expected_hash
                and (recorded_best_epoch < 0 or checkpoint.exists())
            )
            if resumable:
                restore_checkpoint_payload(payload, model)
                if payload.get("optimizer_state") is not None:
                    optimizer.load_state_dict(payload["optimizer_state"])
                    _move_optimizer_state_to_device(optimizer, device)
                if payload.get("scaler_state") is not None:
                    scaler.load_state_dict(payload["scaler_state"])
                start_epoch = int(payload.get("epoch", 0)) + 1
                epochs_trained = start_epoch - 1
                best = float(extra.get("best", float("-inf")))
                best_epoch = int(extra.get("best_epoch", -1))
                _restore_rng_state(extra)
                print(f"RESUME V13.2 {model_cfg.name}: epoch {start_epoch}")
            else:
                print(f"IGNORE stale/incomplete V13.2 resume checkpoint: {resume_checkpoint.name}")
                resume_checkpoint.unlink(missing_ok=True)
                checkpoint.unlink(missing_ok=True)
            del payload

        for epoch in range(start_epoch, epochs + 1):
            epochs_trained = epoch
            phase, phase_epoch, phase_length = phase_for_epoch(epoch, epochs)
            if phase == "GT_POS":
                phase_data = perfect_train
            elif phase == "OOF_POS":
                phase_data = train_oof_positive
            else:
                phase_data = train_base

            type_lr, validity_lr = phase_learning_rates(model_cfg, epoch, epochs)
            _set_optimizer_lrs(optimizer, type_lr, validity_lr)
            validity_active = phase == "OOF_ALL"

            mining_active = (
                phase == "OOF_ALL"
                and phase_epoch >= min(
                    int(model_cfg.hard_negative_start_phase_epoch),
                    int(model_cfg.hard_positive_start_phase_epoch),
                )
            )
            refresh = (
                mining_active
                and (
                    hard_reject_scores is None
                    or (phase_epoch - int(model_cfg.hard_negative_start_phase_epoch))
                    % max(1, int(model_cfg.hard_pool_refresh_interval)) == 0
                )
            )
            if refresh:
                hard_reject_scores, hard_positive_scores = _hardness_scores(
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

            epoch_seed = seed + epoch * 10007
            if active_phase != phase or train_loader is None or train_dataset is None:
                # Reuse the loader for the complete phase. Persistent workers retain the
                # native-crop worker cache, avoiding repeated process startup and disk reads.
                if train_loader is not None:
                    del train_loader
                if train_dataset is not None:
                    del train_dataset
                train_dataset = Stage2CandidateDataset(
                    store,
                    phase_data,
                    views=model_cfg.views,
                    augment=model_cfg.use_stain_augmentation,
                    seed=seed,
                    interface_key=model_cfg.interface_key,
                )
                train_dataset.set_epoch(epoch)
                train_loader = _make_loader(
                    train_dataset,
                    phase_data,
                    model_cfg,
                    manifest,
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
                    train_loader,
                    train_dataset,
                    epoch=epoch,
                    seed=epoch_seed,
                    hard_reject_scores=hard_reject_scores if mining_active else None,
                    hard_positive_scores=hard_positive_scores if mining_active else None,
                )

            sampler_stats = dict(getattr(train_loader.sampler, "stats", {}))
            epoch_encoder_cache = _make_epoch_encoder_cache(model, train_loader.sampler)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = torch.zeros((), device=device)
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
                        outputs,
                        labels,
                        raw_class_counts,
                        model_cfg,
                        validity_active=validity_active,
                    )
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
                partial = int(step % accumulation)
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

            stop_after_epoch = False
            if _should_validate(epoch, epochs, runtime.training.validation_interval):
                metrics, _, _ = evaluate_stage2(
                    model,
                    store,
                    val_oof_all,
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
                    f"[{model_cfg.name} {epochs}ep s{seed}] epoch={epoch:03d} "
                    f"phase={phase}:{phase_epoch:02d}/{phase_length:02d} "
                    f"lr={type_lr:.2e}/{validity_lr:.2e} "
                    f"loss={float(running_loss.item()) / max(step,1):.4f} "
                    f"val_macro_f1={score:.4f} threshold={float(metrics.get('validity_threshold', .5)):.2f}"
                )
                eligible = (
                    phase == "OOF_ALL"
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
                        scheduler=None,
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
                            "epoch_profile": epochs,
                            "sampler_stats": sampler_stats,
                            "config_hash": expected_hash,
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
                    stop_after_epoch = True

            phase_next = None if epoch == epochs else phase_for_epoch(epoch + 1, epochs)[0]
            if (epoch % int(runtime.training.resume_checkpoint_interval) == 0
                or phase_next != phase or epoch == epochs or stop_after_epoch):
                extra = {
                    "config_hash": expected_hash,
                    "best": best,
                    "best_epoch": best_epoch,
                    "phase": phase,
                    "phase_epoch": phase_epoch,
                    "split_hash": split_hash,
                    "epoch_profile": epochs,
                }
                extra.update(_capture_rng_state())
                save_best_checkpoint(
                    resume_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    scaler=scaler,
                    epoch=epoch,
                    score=best,
                    config=model_cfg,
                    extra=extra,
                    trainable_only=True,
                    include_training_state=True,
                )
            del epoch_encoder_cache
            if stop_after_epoch:
                print(f"Early stopping V13.2 {model_cfg.name} at epoch {epoch}; best={best:.4f}@{best_epoch}.")
                break

        if train_loader is not None:
            del train_loader
        if train_dataset is not None:
            del train_dataset

        if not checkpoint.exists():
            # With the sparse validation schedule this should only happen when final-phase
            # selection never produced a finite score, which is a real error.
            raise RuntimeError(f"No eligible V13.2 checkpoint produced for {model_cfg.name}.")

        optimizer.zero_grad(set_to_none=True)
        del optimizer, scaler
        release_cuda_memory(synchronize=False)
        raw_best = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if str(raw_best.get("extra", {}).get("config_hash", "")) != expected_hash:
            raise RuntimeError(f"V13.2 best checkpoint hash mismatch: {checkpoint}")
        del raw_best
        payload = load_checkpoint(checkpoint, model, device)
        selected_epoch = int(payload.get("epoch", -1))
        selected_score = float(payload.get("score", np.nan))
        threshold = float(payload.get("extra", {}).get("validity_threshold", 0.5))
        del payload
        metrics, _, predictions = evaluate_stage2(
            model,
            store,
            val_oof_all,
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
        detector_lookup = {int(row["oof_row_id"]): float(row["confidence"]) for row in val_oof_all}
        detector_confidence = np.asarray(
            [detector_lookup.get(int(source), np.nan) for source in source_ids], dtype=np.float32
        )
        matrix = np.column_stack(
            [source_ids, classes, confidence, detector_confidence, probabilities]
        ).astype(np.float32)
        prediction_path = paths.stage2_file(
            f"stage2_v132_predictions_{model_cfg.name}_{epochs}ep_seed{seed}_{run_tag}.npy"
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
            "best_checkpoint": str(checkpoint),
            "prediction_npy": str(prediction_path),
            "parameters_total": total,
            "parameters_trainable": trainable,
            "train_oof_candidates": len(train_base),
            "train_oof_positives": len(train_oof_positive),
            "train_oof_rejects": int(np.sum(train_base["class_id"] == REJECT_CLASS_ID)),
            "val_candidates": len(val_oof_all),
            "train_roi_count": len(train_roi_indices),
            "val_roi_count": len(val_roi_indices),
            "train_type_counts_json": json.dumps(type_counts.astype(int).tolist()),
            "stage2_effective_batch_size": runtime.training.stage2_effective_batch_size,
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
