from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import NUCLEI_CLASSES, PROCESSED_DIR, SPLIT_DIR, TISSUE_CLASSES, TrainConfig

TISSUE_PIXEL_PREFIX = 'pixels_'
NUCLEI_PIXEL_PREFIX = 'pixels_'


def _label_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for name in TISSUE_CLASSES:
        col = f'pixels_{name}'
        if col in df.columns:
            cols.append(col)
    for name in NUCLEI_CLASSES:
        col = f'pixels_{name}'
        if col in df.columns:
            cols.append(col)
    if not cols:
        raise ValueError('index.csv does not contain per-class pixel columns. Re-run preprocessing with the new code.')
    return cols


def build_multilabel_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = _label_columns(df)
    y = (df[cols].to_numpy(dtype=np.float64) > 0).astype(np.int64)
    return y, cols


def multilabel_stratified_kfold_indices(
    y: np.ndarray,
    n_splits: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Greedy multi-label stratified fold assignment.

    This keeps rare labels distributed as evenly as possible without requiring
    extra packages such as iterative-stratification. It works on image-level
    class presence, not pixel counts, which is the right choice for preventing
    rare classes from disappearing from a validation fold.
    """
    if y.ndim != 2:
        raise ValueError(f'y must be 2D, got shape {y.shape}')
    n_samples, n_labels = y.shape
    if n_samples < n_splits:
        raise ValueError(f'Need at least {n_splits} samples, got {n_samples}')

    rng = np.random.default_rng(seed)
    label_totals = y.sum(axis=0).astype(np.float64)
    desired = label_totals / float(n_splits)
    desired_size = n_samples / float(n_splits)

    # Samples with rare labels go first. This is important for PUMA because
    # necrosis, epidermis, neutrophil, plasma cell, etc. can be very sparse.
    label_freq = np.maximum(label_totals, 1.0)
    rarity_score = (y / label_freq).sum(axis=1)
    random_tiebreak = rng.random(n_samples) * 1e-6
    order = np.argsort(-(rarity_score + random_tiebreak))

    fold_counts = np.zeros((n_splits, n_labels), dtype=np.float64)
    fold_sizes = np.zeros(n_splits, dtype=np.float64)
    assignment = np.full(n_samples, -1, dtype=np.int64)

    for idx in order:
        labels = y[idx].astype(np.float64)
        best_fold = 0
        best_score = None
        for fold in range(n_splits):
            new_counts = fold_counts[fold] + labels
            new_size = fold_sizes[fold] + 1.0

            # Penalize folds that would exceed expected rare-label demand.
            label_deficit_before = np.maximum(desired - fold_counts[fold], 0.0)
            label_deficit_after = np.maximum(desired - new_counts, 0.0)
            label_gain = (label_deficit_before - label_deficit_after).sum()

            size_penalty = abs(new_size - desired_size) / max(1.0, desired_size)
            overload_penalty = np.maximum(new_counts - desired, 0.0).sum() / max(1.0, labels.sum())

            # Lower is better. Negative label_gain rewards assigning rare labels
            # to folds that still need them.
            score = -label_gain + 0.20 * size_penalty + 0.05 * overload_penalty
            if best_score is None or score < best_score:
                best_score = score
                best_fold = fold

        assignment[idx] = best_fold
        fold_counts[best_fold] += labels
        fold_sizes[best_fold] += 1.0

    if (assignment < 0).any():
        raise RuntimeError('Internal split error: some samples were not assigned')
    return assignment


def _fold_summary(df: pd.DataFrame, assignment: np.ndarray, label_cols: list[str], n_folds: int) -> pd.DataFrame:
    rows = []
    for fold in range(n_folds):
        val = df.iloc[np.where(assignment == fold)[0]]
        train = df.iloc[np.where(assignment != fold)[0]]
        row = {
            'fold': fold,
            'train_images': int(len(train)),
            'val_images': int(len(val)),
        }
        for col in label_cols:
            row[f'train_has_{col.removeprefix("pixels_")}'] = int((train[col] > 0).sum()) if col in train else 0
            row[f'val_has_{col.removeprefix("pixels_")}'] = int((val[col] > 0).sum()) if col in val else 0
        rows.append(row)
    return pd.DataFrame(rows)


def generate_multilabel_folds(cfg: TrainConfig, force: bool = False) -> Path:
    index_path = PROCESSED_DIR / 'index.csv'
    if not index_path.exists():
        raise FileNotFoundError(f'Missing {index_path}. Run preprocess_data.py first.')

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    assignment_path = SPLIT_DIR / 'fold_assignments.csv'
    summary_path = SPLIT_DIR / 'folds_summary.csv'

    if assignment_path.exists() and not force:
        return SPLIT_DIR

    df = pd.read_csv(index_path)
    y, label_cols = build_multilabel_matrix(df)
    assignment = multilabel_stratified_kfold_indices(y, n_splits=cfg.n_folds, seed=cfg.split_seed)

    out = df[['id']].copy() if 'id' in df.columns else pd.DataFrame({'row_index': np.arange(len(df))})
    out['fold'] = assignment
    out.to_csv(assignment_path, index=False)

    for fold in range(cfg.n_folds):
        train_df = df.iloc[np.where(assignment != fold)[0]].reset_index(drop=True)
        val_df = df.iloc[np.where(assignment == fold)[0]].reset_index(drop=True)
        train_df.to_csv(SPLIT_DIR / f'fold_{fold}_train.csv', index=False)
        val_df.to_csv(SPLIT_DIR / f'fold_{fold}_val.csv', index=False)

    summary = _fold_summary(df, assignment, label_cols, cfg.n_folds)
    summary.to_csv(summary_path, index=False)

    metadata = {
        'n_images': int(len(df)),
        'n_folds': int(cfg.n_folds),
        'split_seed': int(cfg.split_seed),
        'label_columns': label_cols,
        'note': 'Greedy multi-label stratified fold split using image-level class presence.',
    }
    (SPLIT_DIR / 'folds_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return SPLIT_DIR


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame


def load_fold_split(cfg: TrainConfig) -> SplitFrames:
    generate_multilabel_folds(cfg, force=False)
    train_path = SPLIT_DIR / f'fold_{cfg.fold_id}_train.csv'
    val_path = SPLIT_DIR / f'fold_{cfg.fold_id}_val.csv'
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f'Missing split files for fold {cfg.fold_id}. Run generate_folds.py first.')
    return SplitFrames(
        train=pd.read_csv(train_path),
        val=pd.read_csv(val_path),
    )
