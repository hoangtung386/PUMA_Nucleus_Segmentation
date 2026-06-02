from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from .config import ENCODER_RUNS, IGNORE_INDEX, NUCLEI_CLASSES, PROCESSED_DIR, SPLIT_DIR, TISSUE_CLASSES, TrainConfig
from .data import PumaPatchDataset, make_split
from .models import PumaEncoderProbe
from .splits import generate_multilabel_folds


def check_processed_dataset() -> None:
    index_path = PROCESSED_DIR / 'index.csv'
    if not index_path.exists():
        raise FileNotFoundError(f'Missing {index_path}. Run preprocess_data.py first.')
    df = pd.read_csv(index_path)
    errors = []
    tissue_allowed = set(TISSUE_CLASSES.values())
    nuclei_allowed = set(NUCLEI_CLASSES.values()) | {IGNORE_INDEX}

    for _, row in df.iterrows():
        image = np.load(row.image_path)
        with np.load(row.mask_path) as masks:
            tissue = masks['tissue']
            nuclei_class = masks['nuclei_class']
            nuclei_fg = masks['nuclei_fg']
        if image.ndim != 3 or image.shape[-1] != 3:
            errors.append(f'{row.id}: bad image shape {image.shape}')
        if image.shape[:2] != tissue.shape or tissue.shape != nuclei_class.shape or tissue.shape != nuclei_fg.shape:
            errors.append(f'{row.id}: shape mismatch image={image.shape} tissue={tissue.shape} nuclei={nuclei_class.shape} fg={nuclei_fg.shape}')
        bad_tissue = set(np.unique(tissue).tolist()) - tissue_allowed
        bad_nuclei = set(np.unique(nuclei_class).tolist()) - nuclei_allowed
        bad_fg = set(np.unique(nuclei_fg).tolist()) - {0, 1}
        if bad_tissue:
            errors.append(f'{row.id}: invalid tissue ids {bad_tissue}')
        if bad_nuclei:
            errors.append(f'{row.id}: invalid nuclei ids {bad_nuclei}')
        if bad_fg:
            errors.append(f'{row.id}: invalid nuclei_fg ids {bad_fg}')
        if not np.array_equal(nuclei_fg.astype(bool), nuclei_class != IGNORE_INDEX):
            errors.append(f'{row.id}: nuclei_fg is not equal to nuclei_class != 255')

    if errors:
        raise RuntimeError('\n'.join(errors[:80]))
    print(f'Processed dataset OK: {len(df)} images')


def check_folds(cfg: TrainConfig | None = None, force: bool = False) -> None:
    cfg = cfg or TrainConfig()
    generate_multilabel_folds(cfg, force=force)
    summary_path = SPLIT_DIR / 'folds_summary.csv'
    if not summary_path.exists():
        raise FileNotFoundError(f'Missing {summary_path}')
    summary = pd.read_csv(summary_path)
    print(f'Multi-label folds OK: {cfg.n_folds} folds written to {SPLIT_DIR}')
    print(summary[['fold', 'train_images', 'val_images']].to_string(index=False))


def check_experiment_batch(run_key: str, cfg: TrainConfig) -> None:
    test_cfg = replace(cfg, batch_size=1, num_workers=0, pretrained=False, samples_per_train_image=1, val_samples_per_image=1)
    split = make_split(test_cfg)
    if split.train.empty or split.val.empty:
        raise RuntimeError(f'{run_key}: empty train or val split')
    ds = PumaPatchDataset(split.train, test_cfg, train=True)
    batch = torch.utils.data.default_collate([ds[0]])

    assert batch['image'].shape == (1, 3, test_cfg.image_size, test_cfg.image_size), batch['image'].shape
    assert batch['tissue'].shape == (1, test_cfg.image_size, test_cfg.image_size), batch['tissue'].shape
    assert batch['nuclei_fg'].shape == batch['tissue'].shape
    assert batch['nuclei_class'].shape == batch['tissue'].shape
    assert int(batch['tissue'].min()) >= 0 and int(batch['tissue'].max()) < len(TISSUE_CLASSES)
    assert set(batch['nuclei_fg'].unique().tolist()).issubset({0, 1})
    print(f'{run_key} batch OK: image={tuple(batch["image"].shape)}, tissue={tuple(batch["tissue"].shape)}')


def check_experiment_model_forward(run_key: str, cfg: TrainConfig, full_size: bool = False) -> None:
    # Full 1024 forward can be memory-heavy. Default synthetic forward uses one
    # 256 tile while still checking channel heads and tiled foundation logic.
    size = cfg.image_size if full_size else cfg.foundation_tile_size
    test_cfg = replace(
        cfg,
        image_size=size,
        batch_size=1,
        num_workers=0,
        pretrained=False,
        foundation_tile_batch=1,
    )
    model = PumaEncoderProbe(test_cfg)
    model.eval()
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        out = model(x)
    expected = {
        'tissue': (1, len(TISSUE_CLASSES), size, size),
        'nuclei_fg': (1, 2, size, size),
        'nuclei_class': (1, len(NUCLEI_CLASSES), size, size),
    }
    for key, shape in expected.items():
        if tuple(out[key].shape) != shape:
            raise RuntimeError(f'{run_key}: bad {key} shape {tuple(out[key].shape)}, expected {shape}')
    print(f'{run_key} model forward OK at {size}x{size}')


def check_all_experiments(model_forward: bool = False, full_size_forward: bool = False) -> None:
    for run_key, cfg in ENCODER_RUNS.items():
        for fold_id in range(cfg.n_folds):
            fold_cfg = replace(cfg, fold_id=fold_id)
            check_experiment_batch(f'{run_key}/fold_{fold_id}', fold_cfg)
        if model_forward:
            check_experiment_model_forward(run_key, cfg, full_size=full_size_forward)


def run_sanity(model_forward: bool = False, full_size_forward: bool = False, force_folds: bool = False) -> None:
    check_processed_dataset()
    check_folds(force=force_folds)
    check_all_experiments(model_forward=model_forward, full_size_forward=full_size_forward)
    print('All sanity checks passed.')


if __name__ == '__main__':
    run_sanity(model_forward=False)
