from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import IGNORE_INDEX, NUCLEI_CLASSES, TISSUE_BACKGROUND_ID, TISSUE_CLASSES, TrainConfig
from .splits import SplitFrames, load_fold_split

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

RARE_TISSUE_IDS = [
    TISSUE_CLASSES['tissue_blood_vessel'],
    TISSUE_CLASSES['tissue_epidermis'],
    TISSUE_CLASSES['tissue_necrosis'],
]
RARE_NUCLEI_IDS = [
    NUCLEI_CLASSES['nuclei_plasma_cell'],
    NUCLEI_CLASSES['nuclei_histiocyte'],
    NUCLEI_CLASSES['nuclei_melanophage'],
    NUCLEI_CLASSES['nuclei_neutrophil'],
    NUCLEI_CLASSES['nuclei_epithelium'],
    NUCLEI_CLASSES['nuclei_endothelium'],
    NUCLEI_CLASSES['nuclei_apoptosis'],
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_split(cfg: TrainConfig) -> SplitFrames:
    return load_fold_split(cfg)


def _pad_to_min_size(arr: np.ndarray, min_h: int, min_w: int, fill_value: int = 0) -> np.ndarray:
    h, w = arr.shape[:2]
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return arr
    if arr.ndim == 3:
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))
    else:
        pad_width = ((0, pad_h), (0, pad_w))
    return np.pad(arr, pad_width, mode='constant', constant_values=fill_value)


def _crop_at(arr: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    return arr[top:top + size, left:left + size]


def _choose_center_from_coords(coords: np.ndarray) -> Tuple[int, int]:
    y, x = coords[random.randrange(len(coords))]
    return int(y), int(x)


def _crop_origin_around(y: int, x: int, h: int, w: int, size: int) -> Tuple[int, int]:
    top = min(max(0, y - size // 2), max(0, h - size))
    left = min(max(0, x - size // 2), max(0, w - size))
    return top, left


class PumaPatchDataset(Dataset):
    """Loads processed PUMA ROI data.

    With image_size=1024 this returns the full ROI. If you later use smaller
    image_size, it will perform class-aware crops while keeping labels aligned.
    """

    def __init__(self, df: pd.DataFrame, cfg: TrainConfig, train: bool):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.train = train
        self.size = int(cfg.image_size)
        self.samples_per_image = int(cfg.samples_per_train_image if train else cfg.val_samples_per_image)
        if self.size <= 0:
            raise ValueError(f'image_size must be positive, got {self.size}')
        if cfg.foundation_tile_size > self.size:
            raise ValueError('foundation_tile_size cannot be larger than image_size')

    def __len__(self) -> int:
        return len(self.df) * self.samples_per_image

    def _load_item(self, row_idx: int):
        row = self.df.iloc[row_idx]
        image = np.load(row.image_path)
        with np.load(row.mask_path) as masks:
            tissue = masks['tissue']
            nuclei_class = masks['nuclei_class']
            nuclei_fg = masks['nuclei_fg']
        if image.shape[:2] != tissue.shape or tissue.shape != nuclei_class.shape or tissue.shape != nuclei_fg.shape:
            raise RuntimeError(
                f'Shape mismatch in {row.id}: image={image.shape}, tissue={tissue.shape}, '
                f'nuclei_class={nuclei_class.shape}, nuclei_fg={nuclei_fg.shape}'
            )
        return image, tissue, nuclei_class, nuclei_fg

    def _choose_crop(self, tissue: np.ndarray, nuclei_class: np.ndarray, nuclei_fg: np.ndarray) -> Tuple[int, int]:
        h, w = tissue.shape
        size = self.size
        if h == size and w == size:
            return 0, 0
        if not self.train:
            return max(0, (h - size) // 2), max(0, (w - size) // 2)

        mode = random.choices(
            population=['random', 'tissue_fg', 'rare_tissue', 'nuclei_fg', 'rare_nuclei'],
            weights=[0.20, 0.20, 0.25, 0.20, 0.15],
            k=1,
        )[0]
        coords = None
        if mode == 'tissue_fg':
            coords = np.argwhere(tissue != TISSUE_BACKGROUND_ID)
        elif mode == 'rare_tissue':
            coords = np.argwhere(np.isin(tissue, RARE_TISSUE_IDS))
        elif mode == 'nuclei_fg':
            coords = np.argwhere(nuclei_fg > 0)
        elif mode == 'rare_nuclei':
            coords = np.argwhere(np.isin(nuclei_class, RARE_NUCLEI_IDS))

        if coords is not None and len(coords) > 0:
            y, x = _choose_center_from_coords(coords)
            return _crop_origin_around(y, x, h, w, size)
        return random.randint(0, h - size), random.randint(0, w - size)

    def _augment(self, image, tissue, nuclei_class, nuclei_fg):
        if not self.train:
            return image, tissue, nuclei_class, nuclei_fg
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            tissue = np.flip(tissue, axis=1).copy()
            nuclei_class = np.flip(nuclei_class, axis=1).copy()
            nuclei_fg = np.flip(nuclei_fg, axis=1).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            tissue = np.flip(tissue, axis=0).copy()
            nuclei_class = np.flip(nuclei_class, axis=0).copy()
            nuclei_fg = np.flip(nuclei_fg, axis=0).copy()
        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k, axes=(0, 1)).copy()
            tissue = np.rot90(tissue, k).copy()
            nuclei_class = np.rot90(nuclei_class, k).copy()
            nuclei_fg = np.rot90(nuclei_fg, k).copy()
        return image, tissue, nuclei_class, nuclei_fg

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row_idx = idx % len(self.df)
        image, tissue, nuclei_class, nuclei_fg = self._load_item(row_idx)

        image = _pad_to_min_size(image, self.size, self.size, fill_value=255)
        tissue = _pad_to_min_size(tissue, self.size, self.size, fill_value=TISSUE_BACKGROUND_ID)
        nuclei_class = _pad_to_min_size(nuclei_class, self.size, self.size, fill_value=IGNORE_INDEX)
        nuclei_fg = _pad_to_min_size(nuclei_fg, self.size, self.size, fill_value=0)

        top, left = self._choose_crop(tissue, nuclei_class, nuclei_fg)
        image = _crop_at(image, top, left, self.size)
        tissue = _crop_at(tissue, top, left, self.size)
        nuclei_class = _crop_at(nuclei_class, top, left, self.size)
        nuclei_fg = _crop_at(nuclei_fg, top, left, self.size)

        image, tissue, nuclei_class, nuclei_fg = self._augment(image, tissue, nuclei_class, nuclei_fg)

        x = torch.from_numpy(np.ascontiguousarray(image)).float().permute(2, 0, 1) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD

        return {
            'image': x,
            'tissue': torch.from_numpy(np.ascontiguousarray(tissue)).long(),
            'nuclei_fg': torch.from_numpy(np.ascontiguousarray(nuclei_fg)).long(),
            'nuclei_class': torch.from_numpy(np.ascontiguousarray(nuclei_class)).long(),
        }
