"""Leakage-safe train/val split using group-based splitting by source image."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from symbiopan.common.logging import get_logger

logger = get_logger(__name__)


def _safe_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return int(default)
    if arr.size == 1:
        return int(arr.reshape(-1)[0])
    raise RuntimeError(f"Expected scalar integer value, got shape={arr.shape}")


def _safe_str_list(value: Any) -> list[str]:
    arr = np.asarray(value)
    if arr.size == 0:
        return []
    return [str(x) for x in arr.reshape(-1).tolist()]


def make_or_load_group_split(
    source_names: Sequence[str],
    is_original: Sequence[bool],
    split_path: str | Path,
    seed: int = 42,
    train_fraction: float = 0.8,
    force_new: bool = False,
    val_original_only: bool = True,
) -> tuple[list[int], list[int]]:
    """Build (or reload) a deterministic train/val split grouped by source image.

    Persists the split as a ``.npz`` so subsequent runs reuse the same partition
    even when rare-augmented crops change.
    """
    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    source_names = [str(x) for x in source_names]
    is_original = [bool(x) for x in is_original]
    dataset_size = len(source_names)

    if dataset_size <= 1:
        raise RuntimeError(f"Dataset too small for split: dataset_size={dataset_size}")
    if len(is_original) != dataset_size:
        raise RuntimeError("source_names and is_original must have the same length")

    unique_sources = sorted(set(source_names))
    group_count = len(unique_sources)
    if group_count <= 1:
        raise RuntimeError(f"Need at least two original source groups for train/val split, got {group_count}")

    if split_path.exists() and not force_new:
        data = np.load(split_path, allow_pickle=True)
        split_type = str(np.asarray(data["split_type"]).reshape(-1)[0]) if "split_type" in data else "index"
        if split_type != "group":
            raise RuntimeError(
                f"Existing split file is not group-based: {split_path}\n"
                f"Delete it before using rare-centered crops to avoid leakage."
            )

        saved_size = _safe_int(data.get("dataset_size"), dataset_size)
        saved_groups = _safe_int(data.get("group_count"), group_count)
        train_sources = set(_safe_str_list(data["train_sources"]))
        val_sources = set(_safe_str_list(data["val_sources"]))

        if saved_size != dataset_size or saved_groups != group_count:
            raise RuntimeError(
                f"Group split mismatch. Split file={split_path}\n"
                f"Saved dataset_size/group_count={saved_size}/{saved_groups}\n"
                f"Current dataset_size/group_count={dataset_size}/{group_count}\n"
                f"Delete the split file and rerun Stage 1/Stage 2 after preprocessing changes."
            )
        if train_sources & val_sources:
            raise RuntimeError("Invalid split: same source appears in train and validation")

        train_idx = []
        val_idx = []
        for i, src in enumerate(source_names):
            if src in train_sources:
                train_idx.append(i)
            elif src in val_sources:
                if (not val_original_only) or is_original[i]:
                    val_idx.append(i)
            else:
                raise RuntimeError(f"Sample source {src!r} is not present in loaded split file")

        logger.info("Loaded existing GROUP split: %s", split_path)
        logger.info("Train source groups: %d", len(train_sources))
        logger.info("Val source groups: %d", len(val_sources))
        logger.info("Train samples: %d", len(train_idx))
        logger.info("Val samples: %d | val_original_only=%s", len(val_idx), val_original_only)
        return train_idx, val_idx

    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(group_count, generator=generator).numpy()
    ordered_sources = np.asarray(unique_sources, dtype=object)[perm]
    n_train_groups = int(round(float(train_fraction) * group_count))
    n_train_groups = max(1, min(n_train_groups, group_count - 1))

    train_sources = {str(x) for x in ordered_sources[:n_train_groups].tolist()}
    val_sources = {str(x) for x in ordered_sources[n_train_groups:].tolist()}

    train_idx = []
    val_idx = []
    for i, src in enumerate(source_names):
        if src in train_sources:
            train_idx.append(i)
        elif src in val_sources and ((not val_original_only) or is_original[i]):
            val_idx.append(i)

    if not train_idx or not val_idx:
        raise RuntimeError(
            f"Invalid group split: train_samples={len(train_idx)}, val_samples={len(val_idx)}. "
            f"Check preprocessing metadata and group count."
        )

    np.savez(
        split_path,
        split_type=np.asarray(["group"], dtype=object),
        train_indices=np.asarray(train_idx, dtype=np.int64),
        val_indices=np.asarray(val_idx, dtype=np.int64),
        train_sources=np.asarray(sorted(train_sources), dtype=object),
        val_sources=np.asarray(sorted(val_sources), dtype=object),
        dataset_size=np.asarray([dataset_size], dtype=np.int64),
        group_count=np.asarray([group_count], dtype=np.int64),
        seed=np.asarray([seed], dtype=np.int64),
        train_fraction=np.asarray([train_fraction], dtype=np.float32),
        val_original_only=np.asarray([bool(val_original_only)], dtype=bool),
    )

    logger.info("Created new GROUP split: %s", split_path)
    logger.info("Train source groups: %d", len(train_sources))
    logger.info("Val source groups: %d", len(val_sources))
    logger.info("Train samples: %d", len(train_idx))
    logger.info("Val samples: %d | val_original_only=%s", len(val_idx), val_original_only)
    return train_idx, val_idx


__all__ = ["make_or_load_group_split"]
