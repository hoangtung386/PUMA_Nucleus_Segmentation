from __future__ import annotations

from collections import OrderedDict
import hashlib
import math
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from puma.config import REJECT_CLASS_ID, Stage2ModelConfig
from puma.data.datasets import PumaNpyStore, Stage2CandidateDataset, prepare_stage2_view_batch, stage2_collate
from puma.evaluation.metrics import evaluate_typed_detection
from puma.models.stage2 import decode_stage2_probabilities, hierarchical_probabilities
from puma.utils import dataloader_performance_kwargs, resolve_amp_dtype

def _schedule_phases(schedule: str) -> tuple[str, ...]:
    mapping = {
        "GT_POS": ("GT_POS",),
        "OOF_POS": ("OOF_POS",),
        "OOF_ALL": ("OOF_ALL",),
        "GT_POS+OOF_POS": ("GT_POS", "OOF_POS"),
        "GT_POS+OOF_POS+OOF_ALL": ("GT_POS", "OOF_POS", "OOF_ALL"),
    }
    try:
        return mapping[schedule]
    except KeyError as exc:
        raise KeyError(f"Unknown Stage-2 schedule_key={schedule!r}.") from exc

def _phase_for_epoch(schedule: str, epoch: int, epochs: int) -> str:
    """Map an epoch to a curriculum phase, reserving most epochs for deployment data."""
    phases = _schedule_phases(schedule)
    if len(phases) == 1:
        return phases[0]
    if len(phases) == 2:
        boundary = max(1, int(round(0.40 * epochs)))
        return phases[0] if epoch <= boundary else phases[1]
    end_gt = max(1, int(round(0.30 * epochs)))
    end_oof_positive = max(end_gt + 1, int(round(0.60 * epochs)))
    if epoch <= end_gt:
        return "GT_POS"
    if epoch <= end_oof_positive:
        return "OOF_POS"
    return "OOF_ALL"

def _final_schedule_phase(schedule: str) -> str:
    return _schedule_phases(schedule)[-1]

def _phase_epoch_number(schedule: str, epoch: int, epochs: int) -> int:
    """Return the one-based epoch number inside the current curriculum phase."""
    phase = _phase_for_epoch(schedule, epoch, epochs)
    start = int(epoch)
    while start > 1 and _phase_for_epoch(schedule, start - 1, epochs) == phase:
        start -= 1
    return int(epoch) - start + 1

def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device, non_blocking=True)

def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state

def _restore_rng_state(extra: dict[str, Any]) -> None:
    if "python_rng_state" in extra:
        random.setstate(extra["python_rng_state"])
    if "numpy_rng_state" in extra:
        np.random.set_state(extra["numpy_rng_state"])
    if "torch_rng_state" in extra:
        torch.set_rng_state(extra["torch_rng_state"])
    if torch.cuda.is_available() and extra.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(extra["cuda_rng_state_all"])

def _write_field(array: np.ndarray, index: int, name: str, value: Any) -> None:
    if name in array.dtype.names:
        array[name][index] = value

def _build_perfect_candidates(
    store: PumaNpyStore,
    train_base: np.ndarray,
    seed: int,
    background_fraction: float = 0.0,
    roi_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Build GT-centred candidates for the Stage-2 curriculum."""
    if roi_indices is None:
        roi_indices = np.unique(train_base["roi_index"]).astype(int)
    else:
        roi_indices = np.asarray(roi_indices, dtype=int)
    if len(roi_indices) == 0:
        raise ValueError("Cannot build perfect candidates for an empty ROI split.")
    positive_count = sum(len(store.roi_centroids(int(roi))) for roi in roi_indices)
    background_count = int(math.ceil(background_fraction * positive_count))
    data = np.zeros(positive_count + background_count, dtype=train_base.dtype)
    rng = np.random.default_rng(seed)
    cursor = 0
    negative_source_id = -1

    for roi in roi_indices:
        gt = store.roi_centroids(int(roi))
        for gt_index, nucleus in enumerate(gt):
            _write_field(data, cursor, "oof_row_id", negative_source_id)
            negative_source_id -= 1
            _write_field(data, cursor, "roi_index", roi)
            _write_field(data, cursor, "candidate_index", gt_index)
            _write_field(data, cursor, "x", float(nucleus["x"]))
            _write_field(data, cursor, "y", float(nucleus["y"]))
            _write_field(data, cursor, "confidence", 1.0)
            _write_field(data, cursor, "nearest_distance", float(nucleus["nearest_neighbor_distance"]))
            _write_field(data, cursor, "class_id", int(nucleus["class_id"]))
            _write_field(data, cursor, "matched_gt_index", gt_index)
            _write_field(data, cursor, "match_distance", 0.0)
            _write_field(data, cursor, "is_reject", 0)
            _write_field(data, cursor, "fold", -1)
            cursor += 1

    if background_count:
        per_roi = rng.multinomial(background_count, np.ones(len(roi_indices)) / len(roi_indices))
        for roi, count in zip(roi_indices, per_roi, strict=True):
            gt = store.roi_centroids(int(roi))
            gt_xy = np.column_stack([gt["x"], gt["y"]]).astype(np.float32)
            height, width = store.images[int(roi)].shape[:2]
            for local_index in range(int(count)):
                x = y = 0.0
                nearest = float(max(height, width))
                for _ in range(200):
                    x = float(rng.uniform(0, width - 1))
                    y = float(rng.uniform(0, height - 1))
                    nearest = (
                        float(np.linalg.norm(gt_xy - np.asarray([x, y]), axis=1).min())
                        if len(gt_xy)
                        else float(max(height, width))
                    )
                    if nearest > 15.0:
                        break
                _write_field(data, cursor, "oof_row_id", negative_source_id)
                negative_source_id -= 1
                _write_field(data, cursor, "roi_index", roi)
                _write_field(data, cursor, "candidate_index", len(gt) + local_index)
                _write_field(data, cursor, "x", x)
                _write_field(data, cursor, "y", y)
                _write_field(data, cursor, "confidence", 0.5)
                _write_field(data, cursor, "nearest_distance", nearest)
                _write_field(data, cursor, "class_id", REJECT_CLASS_ID)
                _write_field(data, cursor, "matched_gt_index", -1)
                _write_field(data, cursor, "match_distance", np.nan)
                _write_field(data, cursor, "is_reject", 1)
                _write_field(data, cursor, "fold", -1)
                cursor += 1
    return data[:cursor]

def _make_loader(
    dataset: Stage2CandidateDataset,
    batch_size: int,
    workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=stage2_collate,
        **dataloader_performance_kwargs(
            workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        ),
    )

class _FrozenEncoderEpochCache:
    """Cache frozen encoder features for repeated candidates within an epoch."""

    def __init__(self, cacheable_indices: set[int], max_bytes: int) -> None:
        self.cacheable_indices = cacheable_indices
        self.max_bytes = int(max_bytes)
        self.current_bytes = 0
        self.entries: dict[tuple[str, int], torch.Tensor] = {}

    def get(self, view: str, index: int) -> torch.Tensor | None:
        return self.entries.get((view, int(index)))

    def put_many(
        self,
        view: str,
        indices: list[int],
        features: torch.Tensor,
    ) -> None:
        if not indices:
            return
        for row, index in enumerate(indices):
            index = int(index)
            key = (view, index)
            if index not in self.cacheable_indices or key in self.entries:
                continue
            source = features[row]
            nbytes = source.numel() * source.element_size()
            if self.current_bytes + nbytes > self.max_bytes:
                break
            # Clone each row so a single cached feature never retains the complete
            # temporary batch storage after the other rows are discarded.
            self.entries[key] = source.clone()
            self.current_bytes += nbytes

def _epoch_feature_cache_budget_bytes() -> int:
    try:
        import psutil
        return min(750_000_000, int(psutil.virtual_memory().available * 0.10))
    except Exception:
        return 384_000_000

def _subset_packed_view(
    packed: tuple[dict[str, torch.Tensor], ...] | list[dict[str, torch.Tensor]],
    positions: list[int],
) -> tuple[dict[str, torch.Tensor], ...]:
    """Select batch positions from grouped native crops and renumber their indices."""
    if not positions:
        return ()
    position_tensor = torch.tensor(positions, dtype=torch.long)
    batch_size = 1 + max(
        int(group["indices"].max().item()) for group in packed if len(group["indices"])
    )
    remap = torch.full((batch_size,), -1, dtype=torch.long)
    remap[position_tensor] = torch.arange(len(positions), dtype=torch.long)
    selected: list[dict[str, torch.Tensor]] = []
    for group in packed:
        mapped = remap[group["indices"]]
        keep = mapped >= 0
        if bool(keep.any()):
            selected.append({
                "indices": mapped[keep],
                "images": group["images"][keep],
            })
    return tuple(selected)

def _encode_view_with_epoch_cache(
    model,
    view: str,
    packed: tuple[dict[str, torch.Tensor], ...] | list[dict[str, torch.Tensor]],
    candidate_indices: torch.Tensor,
    device: torch.device,
    stain_parameters: torch.Tensor | None,
    cache: _FrozenEncoderEpochCache,
) -> torch.Tensor:
    ids = [int(value) for value in candidate_indices.tolist()]
    hit_positions: list[int] = []
    hit_features: list[torch.Tensor] = []
    miss_positions: list[int] = []
    duplicate_positions: list[int] = []
    duplicate_miss_rows: list[int] = []
    pending_misses: dict[int, int] = {}
    for position, index in enumerate(ids):
        cached = cache.get(view, index)
        if cached is not None:
            hit_positions.append(position)
            hit_features.append(cached)
            continue
        existing_row = pending_misses.get(index)
        if existing_row is not None:
            duplicate_positions.append(position)
            duplicate_miss_rows.append(existing_row)
            continue
        pending_misses[index] = len(miss_positions)
        miss_positions.append(position)

    raw_miss: torch.Tensor | None = None
    if miss_positions:
        miss_packed = _subset_packed_view(packed, miss_positions)
        miss_position_tensor = torch.tensor(miss_positions, dtype=torch.long)
        miss_stain = (
            None
            if stain_parameters is None or not stain_parameters.numel()
            else stain_parameters.index_select(0, miss_position_tensor)
        )
        image = prepare_stage2_view_batch(
            miss_packed, device, stain_parameters=miss_stain
        )
        raw_miss = model.extract_view_features(image)
        del image

    batch_size = len(ids)
    if raw_miss is not None:
        raw_batch = raw_miss.new_empty((batch_size, raw_miss.shape[-1]))
        miss_device_positions = torch.tensor(
            miss_positions, device=device, dtype=torch.long
        )
        raw_batch.index_copy_(0, miss_device_positions, raw_miss)
        if duplicate_positions:
            raw_batch.index_copy_(
                0,
                torch.tensor(duplicate_positions, device=device, dtype=torch.long),
                raw_miss.index_select(
                    0, torch.tensor(duplicate_miss_rows, device=device, dtype=torch.long)
                ),
            )
        cache_rows: list[int] = []
        cache_ids: list[int] = []
        for row, position in enumerate(miss_positions):
            index = ids[position]
            if index in cache.cacheable_indices:
                cache_rows.append(row)
                cache_ids.append(index)
        if cache_rows:
            cache_cpu = raw_miss.detach().index_select(
                0, torch.tensor(cache_rows, device=device, dtype=torch.long)
            ).cpu().contiguous()
            cache.put_many(view, cache_ids, cache_cpu)
    else:
        # Every row is a cache hit. Stack compact CPU features before one transfer.
        raw_batch = torch.stack(hit_features).to(device, non_blocking=True)

    if hit_positions:
        hit_batch = torch.stack(hit_features).to(
            device=device, dtype=raw_batch.dtype, non_blocking=True
        )
        raw_batch.index_copy_(
            0,
            torch.tensor(hit_positions, device=device, dtype=torch.long),
            hit_batch,
        )
        del hit_batch
    if raw_miss is not None:
        del raw_miss
    return model.project_view_features(raw_batch, view)

def _encode_views_streamed(
    model,
    cpu_views: dict[str, Any],
    device: torch.device,
    stain_parameters: dict[str, torch.Tensor] | None = None,
    candidate_indices: torch.Tensor | None = None,
    epoch_encoder_cache: _FrozenEncoderEpochCache | None = None,
) -> dict[str, torch.Tensor]:
    """Encode views sequentially, overlapping preparation of the next CUDA view."""
    keys = [key for key in model.cfg.views if key in cpu_views]
    projected: dict[str, torch.Tensor] = {}
    if not keys:
        return projected

    # Cached frozen features have variable hit/miss subsets and are already much cheaper
    # than a PFM pass. Use the direct path to avoid unnecessary double buffering.
    if epoch_encoder_cache is not None and candidate_indices is not None:
        for key in keys:
            view_stain = None if not stain_parameters else stain_parameters.get(key)
            projected[key] = _encode_view_with_epoch_cache(
                model, key, cpu_views[key], candidate_indices, device,
                view_stain, epoch_encoder_cache,
            )
        return projected

    def prepare(key: str) -> torch.Tensor:
        return prepare_stage2_view_batch(
            cpu_views[key], device,
            stain_parameters=None if not stain_parameters else stain_parameters.get(key),
        )

    if device.type != "cuda" or len(keys) == 1:
        for key in keys:
            image = prepare(key)
            projected[key] = model.encode_view(image, key)
            del image
        return projected

    # Prefetch one view while UNI2-h processes the current view.
    current_stream = torch.cuda.current_stream(device)
    prefetch_stream = getattr(model, "_puma_view_prefetch_stream", None)
    if prefetch_stream is None:
        prefetch_stream = torch.cuda.Stream(device=device)
        model._puma_view_prefetch_stream = prefetch_stream
    with torch.cuda.stream(prefetch_stream):
        next_image = prepare(keys[0])
    for position, key in enumerate(keys):
        current_stream.wait_stream(prefetch_stream)
        image = next_image
        image.record_stream(current_stream)
        if position + 1 < len(keys):
            next_key = keys[position + 1]
            with torch.cuda.stream(prefetch_stream):
                next_image = prepare(next_key)
        projected[key] = model.encode_view(image, key)
        del image
    return projected

def _fuse_batch_streamed(
    model,
    batch: dict[str, Any],
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
    epoch_encoder_cache: _FrozenEncoderEpochCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    geometry = batch["geometry"].to(device, non_blocking=True)
    with torch.autocast(
        device_type=device.type, dtype=amp_dtype,
        enabled=amp and device.type == "cuda",
    ):
        projected = _encode_views_streamed(
            model, batch["views"], device, batch.get("stain_parameters"),
            batch.get("candidate_index"), epoch_encoder_cache,
        )
        fused = model.fuse_projected_views(projected, geometry)
    del projected
    return fused, geometry

def _forward_batch_streamed(
    model,
    batch: dict[str, Any],
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
    epoch_encoder_cache: _FrozenEncoderEpochCache | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    fused, geometry = _fuse_batch_streamed(
        model, batch, device, amp, amp_dtype, epoch_encoder_cache
    )
    with torch.autocast(
        device_type=device.type, dtype=amp_dtype,
        enabled=amp and device.type == "cuda",
    ):
        outputs = model.classify_fused(fused)
    del fused
    return outputs, geometry

def _feature_cache_key(
    candidates: np.ndarray,
    model,
) -> tuple[Any, ...]:
    if len(candidates) and "oof_row_id" in (candidates.dtype.names or ()):
        source_ids = np.ascontiguousarray(candidates["oof_row_id"])
        identity = hashlib.blake2b(
            source_ids.view(np.uint8), digest_size=12
        ).hexdigest()
    else:
        identity = str(
            int(candidates.__array_interface__["data"][0]) if len(candidates) else 0
        )
    return (
        identity,
        len(candidates),
        model.cfg.pfm_key,
        tuple(model.cfg.views),
        model.cfg.pooling_key,
        model.cfg.interface_key,
        bool(model.cfg.use_geometry),
        bool(model.cfg.use_lora),
    )

def _feature_cache_budget_bytes() -> int:
    """Return the host-memory budget for cached frozen encoder features."""
    try:
        import psutil
        return min(1_500_000_000, int(psutil.virtual_memory().available * 0.20))
    except Exception:
        return 750_000_000

def _feature_cache_entry_bytes(entry: dict[str, Any]) -> int:
    total = 0
    for key in ("geometry",):
        tensor = entry.get(key)
        if isinstance(tensor, torch.Tensor):
            total += tensor.numel() * tensor.element_size()
    for tensor in entry.get("raw_features", {}).values():
        total += tensor.numel() * tensor.element_size()
    return int(total)

def _feature_cache_allowed(
    model, candidates: np.ndarray, amp: bool, amp_dtype: torch.dtype,
) -> bool:
    if model.encoder_trainable:
        return False
    count = len(candidates)
    encoder_dim = int(next(iter(model.view_projections.values()))[0].normalized_shape[0])
    element_size = 2 if amp and amp_dtype in {torch.float16, torch.bfloat16} else 4
    estimate = count * len(model.cfg.views) * encoder_dim * element_size
    budget = _feature_cache_budget_bytes()
    return estimate <= max(budget, 128_000_000)

def _predict_from_frozen_feature_cache(
    model,
    cache: dict[str, Any],
    batch_size: int,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
    validity_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_ids = cache["source_ids"]
    predicted_classes: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(source_ids), batch_size):
            stop = min(start + batch_size, len(source_ids))
            geometry = cache["geometry"][start:stop].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype,
                enabled=amp and device.type == "cuda",
            ):
                projected = {
                    key: model.project_view_features(
                        cache["raw_features"][key][start:stop].to(device, non_blocking=True),
                        key,
                    )
                    for key in model.cfg.views
                }
                fused = model.fuse_projected_views(projected, geometry)
                outputs = model.classify_fused(fused)
                probability = hierarchical_probabilities(outputs, model.cfg.loss_key)
            predicted_class, confidence = decode_stage2_probabilities(
                probability, model.cfg.loss_key, validity_threshold
            )
            predicted_classes.append(predicted_class.cpu().numpy())
            confidences.append(confidence.float().cpu().numpy())
            probabilities.append(probability.float().cpu().numpy())
            del geometry, projected, fused, outputs, probability
    return (
        source_ids.copy(), np.concatenate(predicted_classes),
        np.concatenate(confidences), np.concatenate(probabilities),
    )

def _predict_candidates(
    model,
    dataset: Stage2CandidateDataset,
    candidates: np.ndarray,
    batch_size: int,
    workers: int,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
    cache_frozen_encoder: bool = False,
    validity_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_key = _feature_cache_key(candidates, model)
    feature_caches = getattr(model, "_puma_frozen_feature_caches", None)
    if not isinstance(feature_caches, OrderedDict):
        feature_caches = OrderedDict(feature_caches or {})
        model._puma_frozen_feature_caches = feature_caches
    if cache_frozen_encoder and cache_key in feature_caches:
        cache_entry = feature_caches.pop(cache_key)
        feature_caches[cache_key] = cache_entry
        return _predict_from_frozen_feature_cache(
            model, cache_entry, batch_size, device, amp, amp_dtype,
            validity_threshold,
        )
    collect_feature_cache = bool(
        cache_frozen_encoder
        and _feature_cache_allowed(model, candidates, amp, amp_dtype)
    )

    loader = _make_loader(dataset, batch_size, workers)
    source_ids: list[np.ndarray] = []
    predicted_classes: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    cached_geometry: list[torch.Tensor] = []
    cached_raw: dict[str, list[torch.Tensor]] = {key: [] for key in model.cfg.views}
    model.eval()
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            if collect_feature_cache:
                geometry = batch["geometry"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype,
                    enabled=amp and device.type == "cuda",
                ):
                    projected: dict[str, torch.Tensor] = {}
                    for key in model.cfg.views:
                        image = prepare_stage2_view_batch(
                            batch["views"][key], device,
                            stain_parameters=batch.get("stain_parameters", {}).get(key),
                        )
                        raw = model.extract_view_features(image)
                        cached_raw[key].append(raw.detach().cpu())
                        projected[key] = model.project_view_features(raw, key)
                        del image, raw
                    fused = model.fuse_projected_views(projected, geometry)
                    outputs = model.classify_fused(fused)
                    probability = hierarchical_probabilities(outputs, model.cfg.loss_key)
                cached_geometry.append(batch["geometry"])
                del projected, fused
            else:
                outputs, geometry = _forward_batch_streamed(
                    model, batch, device, amp, amp_dtype
                )
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype,
                    enabled=amp and device.type == "cuda",
                ):
                    probability = hierarchical_probabilities(outputs, model.cfg.loss_key)
            predicted_class, confidence = decode_stage2_probabilities(
                probability, model.cfg.loss_key, validity_threshold
            )
            source_ids.append(batch["source_index"].numpy())
            predicted_classes.append(predicted_class.cpu().numpy())
            confidences.append(confidence.float().cpu().numpy())
            probabilities.append(probability.float().cpu().numpy())
            del outputs, probability, confidence, predicted_class
            del geometry, batch
    if not source_ids:
        return (
            np.empty(0, np.int64),
            np.empty(0, np.int64),
            np.empty(0, np.float32),
            np.empty((0, REJECT_CLASS_ID + 1), np.float32),
        )
    concatenated_source_ids = np.concatenate(source_ids)
    if collect_feature_cache and cached_geometry:
        cache_entry = {
            "source_ids": concatenated_source_ids.copy(),
            "geometry": torch.cat(cached_geometry, dim=0).contiguous(),
            "raw_features": {
                key: torch.cat(parts, dim=0).contiguous()
                for key, parts in cached_raw.items()
            },
        }
        cache_entry["_nbytes"] = _feature_cache_entry_bytes(cache_entry)
        budget = _feature_cache_budget_bytes()
        current_bytes = sum(
            int(entry.get("_nbytes", _feature_cache_entry_bytes(entry)))
            for entry in feature_caches.values()
        )
        while feature_caches and current_bytes + cache_entry["_nbytes"] > budget:
            _, evicted = feature_caches.popitem(last=False)
            current_bytes -= int(
                evicted.get("_nbytes", _feature_cache_entry_bytes(evicted))
            )
        if cache_entry["_nbytes"] <= budget:
            feature_caches[cache_key] = cache_entry
    return (
        concatenated_source_ids,
        np.concatenate(predicted_classes),
        np.concatenate(confidences),
        np.concatenate(probabilities),
    )

def _candidate_classification_diagnostics(
    candidates: np.ndarray,
    source_ids: np.ndarray,
    predicted_classes: np.ndarray,
    confidences: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    lookup = {
        int(source_id): index
        for index, source_id in enumerate(candidates["oof_row_id"])
    }
    true_labels: list[int] = []
    predicted: list[int] = []
    kept_confidence: list[float] = []
    type_predictions: list[int] = []
    for source_id, predicted_class, confidence, probability in zip(
        source_ids,
        predicted_classes,
        confidences,
        probabilities,
        strict=True,
    ):
        index = lookup.get(int(source_id))
        if index is None:
            continue
        true_labels.append(int(candidates["class_id"][index]))
        predicted.append(int(predicted_class))
        kept_confidence.append(float(confidence))
        type_predictions.append(int(np.argmax(probability[:REJECT_CLASS_ID])))
    if not true_labels:
        return {
            "candidate_accuracy": np.nan,
            "conditional_type_accuracy": np.nan,
            "conditional_type_macro_f1": np.nan,
            "conditional_type_macro_f1_present": np.nan,
            "number_of_present_type_classes": 0.0,
            "reject_precision": np.nan,
            "reject_recall": np.nan,
            "ece": np.nan,
        }

    true = np.asarray(true_labels, dtype=int)
    pred = np.asarray(predicted, dtype=int)
    type_pred = np.asarray(type_predictions, dtype=int)
    conf = np.asarray(kept_confidence, dtype=np.float32)
    true_reject = true == REJECT_CLASS_ID
    pred_reject = pred == REJECT_CLASS_ID
    reject_tp = int(np.sum(true_reject & pred_reject))
    reject_fp = int(np.sum(~true_reject & pred_reject))
    reject_fn = int(np.sum(true_reject & ~pred_reject))
    positive = ~true_reject

    conditional_f1: list[float] = []
    present_conditional_f1: list[float] = []
    result: dict[str, float] = {}
    for class_id in range(REJECT_CLASS_ID):
        class_support = int(np.sum(positive & (true == class_id)))
        tp = int(np.sum(positive & (true == class_id) & (type_pred == class_id)))
        fp = int(np.sum(positive & (true != class_id) & (type_pred == class_id)))
        fn = int(np.sum(positive & (true == class_id) & (type_pred != class_id)))
        denominator = 2 * tp + fp + fn
        score = (2.0 * tp / denominator) if denominator else 0.0
        conditional_f1.append(score)
        if class_support > 0:
            present_conditional_f1.append(score)
        result[f"conditional_type_f1_class_{class_id}"] = float(score)
        result[f"conditional_type_support_class_{class_id}"] = float(class_support)

    correct = pred == true
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (conf >= lower) & (
            conf < upper if upper < 1.0 else conf <= upper
        )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(conf[mask].mean())
            )
    result.update(
        {
            "candidate_accuracy": float(correct.mean()),
            "conditional_type_accuracy": (
                float((type_pred[positive] == true[positive]).mean())
                if bool(positive.any())
                else np.nan
            ),
            # Also report macro F1 over classes present in this validation subset.
            "conditional_type_macro_f1": float(np.mean(conditional_f1)),
            "conditional_type_macro_f1_present": (
                float(np.mean(present_conditional_f1))
                if present_conditional_f1
                else np.nan
            ),
            "number_of_present_type_classes": float(len(present_conditional_f1)),
            "reject_precision": float(
                reject_tp / max(reject_tp + reject_fp, 1)
            ),
            "reject_recall": float(
                reject_tp / max(reject_tp + reject_fn, 1)
            ),
            "ece": float(ece),
        }
    )
    return result

def _records_from_candidate_predictions(
    store: PumaNpyStore,
    candidates: np.ndarray,
    source_ids: np.ndarray,
    classes: np.ndarray,
    confidence: np.ndarray,
    roi_indices: np.ndarray | None,
) -> list[dict[str, Any]]:
    prediction_map = {
        int(source_id): (int(class_id), float(score))
        for source_id, class_id, score in zip(
            source_ids, classes, confidence, strict=True
        )
    }
    evaluated_rois = (
        np.unique(candidates["roi_index"]).astype(int)
        if roi_indices is None
        else np.asarray(roi_indices, dtype=int)
    )
    candidate_groups: dict[int, np.ndarray] = {}
    if len(candidates):
        order = np.argsort(candidates["roi_index"], kind="stable")
        ordered = candidates[order]
        values, starts = np.unique(ordered["roi_index"], return_index=True)
        stops = np.r_[starts[1:], len(ordered)]
        candidate_groups = {
            int(value): ordered[start:stop]
            for value, start, stop in zip(values, starts, stops, strict=True)
        }

    records: list[dict[str, Any]] = []
    for roi in evaluated_rois:
        roi_candidates = candidate_groups.get(int(roi), candidates[:0])
        selected_xy: list[list[float]] = []
        predicted_class: list[int] = []
        scores: list[float] = []
        for candidate in roi_candidates:
            result = prediction_map.get(int(candidate["oof_row_id"]))
            if result is None:
                continue
            class_id, class_confidence = result
            if class_id == REJECT_CLASS_ID:
                continue
            selected_xy.append([float(candidate["x"]), float(candidate["y"])])
            predicted_class.append(class_id)
            scores.append(
                class_confidence * float(candidate["confidence"])
            )
        gt = store.roi_centroids(int(roi))
        manifest = store.manifest[int(roi)]
        records.append(
            {
                "roi_id": str(manifest["roi_id"]),
                "patient_id": str(manifest["case_id"]),
                "pred_xy": np.asarray(selected_xy, np.float32).reshape(-1, 2),
                "pred_scores": np.asarray(scores, np.float32),
                "pred_class": np.asarray(predicted_class, int),
                "gt_xy": np.column_stack([gt["x"], gt["y"]]).astype(
                    np.float32
                ),
                "gt_class": gt["class_id"].astype(int),
            }
        )
    return records

def evaluate_stage2(
    model,
    store: PumaNpyStore,
    candidates: np.ndarray,
    cfg: Stage2ModelConfig,
    device: torch.device,
    batch_size: int,
    workers: int,
    roi_indices: np.ndarray | None = None,
    cache_frozen_encoder: bool = True,
    amp: bool = True,
    prefer_bfloat16: bool = True,
    validity_threshold: float | None = None,
) -> tuple[
    dict[str, float],
    list[dict[str, Any]],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    dataset = Stage2CandidateDataset(
        store,
        candidates,
        views=cfg.views,
        augment=False,
        interface_key=cfg.interface_key,
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
        cache_frozen_encoder=cache_frozen_encoder,
        validity_threshold=0.5,
    )

    probability_tensor = torch.from_numpy(probabilities)
    thresholds = (
        (0.0,)
        if cfg.loss_key == "TYPE_BALANCED"
        else (
            tuple(float(value) for value in cfg.validity_threshold_grid)
            if validity_threshold is None
            else (float(validity_threshold),)
        )
    )
    best_payload: tuple[
        float,
        float,
        dict[str, float],
        list[dict[str, Any]],
        np.ndarray,
        np.ndarray,
    ] | None = None
    for threshold in thresholds:
        classes_tensor, confidence_tensor = decode_stage2_probabilities(
            probability_tensor, cfg.loss_key, threshold
        )
        classes = classes_tensor.numpy().astype(np.int64, copy=False)
        confidence = confidence_tensor.numpy().astype(np.float32, copy=False)
        records = _records_from_candidate_predictions(
            store,
            candidates,
            source_ids,
            classes,
            confidence,
            roi_indices,
        )
        metrics = evaluate_typed_detection(
            records, radius=15.0, number_of_classes=REJECT_CLASS_ID
        )
        metrics.update(
            _candidate_classification_diagnostics(
                candidates,
                source_ids,
                classes,
                confidence,
                probabilities,
            )
        )
        metrics["validity_threshold"] = float(threshold)
        score = float(metrics.get(cfg.selection_metric, np.nan))
        tie_break = float(metrics.get("macro_f1", np.nan))
        candidate_key = (
            score if np.isfinite(score) else float("-inf"),
            tie_break if np.isfinite(tie_break) else float("-inf"),
        )
        if best_payload is None or candidate_key > best_payload[:2]:
            best_payload = (
                candidate_key[0],
                candidate_key[1],
                metrics,
                records,
                classes,
                confidence,
            )
    if best_payload is None:
        raise RuntimeError("Stage-2 evaluation produced no threshold candidate.")
    _, _, metrics, records, classes, confidence = best_payload
    return metrics, records, (
        source_ids,
        classes,
        confidence,
        probabilities,
    )

def _hard_reject_scores(
    model,
    store: PumaNpyStore,
    candidates: np.ndarray,
    cfg: Stage2ModelConfig,
    device: torch.device,
    batch_size: int,
    workers: int,
    *,
    amp: bool = True,
    prefer_bfloat16: bool = True,
) -> np.ndarray:
    scores = np.zeros(len(candidates), dtype=np.float32)
    reject_indices = np.flatnonzero(candidates["class_id"] == REJECT_CLASS_ID)
    if not len(reject_indices):
        return scores
    reject_candidates = candidates[reject_indices]
    dataset = Stage2CandidateDataset(store, reject_candidates, views=cfg.views, augment=False, interface_key=cfg.interface_key)
    source_ids, _, _, probability = _predict_candidates(
        model,
        dataset,
        reject_candidates,
        batch_size,
        workers,
        device,
        amp=amp,
        amp_dtype=resolve_amp_dtype(prefer_bfloat16, device),
        cache_frozen_encoder=True,
    )
    by_source = {int(source): float(1.0 - prob[REJECT_CLASS_ID]) for source, prob in zip(source_ids, probability, strict=True)}
    for local, global_index in enumerate(reject_indices):
        scores[global_index] = by_source.get(int(candidates["oof_row_id"][global_index]), 0.0)
    return scores

def _warmup_cosine_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int):
    """Linear warm-up followed by cosine decay, evaluated once per epoch."""
    total = max(int(epochs), 1)
    warmup = max(0, min(int(warmup_epochs), total - 1))

    def multiplier(step: int) -> float:
        epoch = step + 1
        if warmup and epoch <= warmup:
            return float(epoch) / float(warmup)
        progress = (epoch - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
