"""Tiling and normalization helpers for WSI inference."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile
import torch

from data.constants import NORMALIZATION_MEAN, NORMALIZATION_STD
from training.logging_utils import logger


def find_single_tif(input_dir: str) -> Path:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff"}])
    if not files:
        files = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in {".tif", ".tiff"}])
    if not files:
        raise FileNotFoundError(f"No .tif/.tiff file found in {input_dir}")
    if len(files) > 1:
        logger.warning("Found %d TIFF files. Using first: %s", len(files), files[0])
    return files[0]


def read_rgb_uint8(path: Path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[0] in {3, 4} and img.shape[-1] not in {3, 4}:
        img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        x = img.astype(np.float32)
        x = x - x.min()
        denom = x.max() + 1e-8
        img = np.clip((x / denom) * 255.0, 0, 255).astype(np.uint8)
    return img


def normalize_tile(tile_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    """Normalize a uint8 tile to a float32 tensor on device using PUMA constants."""
    x = tile_uint8.astype(np.float32) / 255.0
    mean = np.asarray(NORMALIZATION_MEAN, dtype=np.float32)
    std = np.asarray(NORMALIZATION_STD, dtype=np.float32)
    x = (x - mean) / std
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return x


def make_tile_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] + tile_size < length:
        starts.append(length - tile_size)
    return starts


def pad_reflect(tile: np.ndarray, tile_size: int) -> Tuple[np.ndarray, int, int]:
    real_h, real_w = tile.shape[:2]
    pad_h = tile_size - real_h
    pad_w = tile_size - real_w
    if pad_h <= 0 and pad_w <= 0:
        return tile, real_h, real_w
    mode = "reflect" if real_h > 1 and real_w > 1 else "constant"
    return np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode), real_h, real_w


def autocast_enabled(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda")
