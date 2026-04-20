"""I/O utilities for loading images, masks, and saving results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_image(path: Union[str, Path], as_rgb: bool = True) -> np.ndarray:
    """Load an image from disk.

    Args:
        path:   Path to the image file.
        as_rgb: If ``True`` (default), convert BGR → RGB.

    Returns:
        np.ndarray of shape (H, W, 3) uint8.
    """
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if as_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_mask(path: Union[str, Path]) -> np.ndarray:
    """Load a pre-computed instance mask from a ``.npy`` file.

    Args:
        path: Path to ``.npy`` mask file.

    Returns:
        np.ndarray of shape (H, W), dtype int32.
    """
    return np.load(str(path)).astype(np.int32)


def save_mask(mask: np.ndarray, path: Union[str, Path]) -> None:
    """Save an instance mask as ``.npy``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), mask.astype(np.int32))


def save_results(
    results: Dict[str, Any],
    path: Union[str, Path],
) -> None:
    """Serialise a results dictionary to JSON.

    Converts numpy scalar types to native Python for JSON compatibility.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable.")

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=_convert)

    logger.info("Results saved to: %s", path)


def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON results file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def list_image_paths(
    directory: Union[str, Path],
    extensions: Tuple[str, ...] = (".png", ".tif", ".tiff", ".jpg"),
) -> List[Path]:
    """Return sorted list of image paths in a directory."""
    directory = Path(directory)
    paths = []
    for ext in extensions:
        paths.extend(directory.glob(f"*{ext}"))
    return sorted(paths)


def load_data_split(split_path: Union[str, Path]) -> Dict[str, List[str]]:
    """Load a train/val/test split JSON file.

    Expected format::

        {
            "train": ["stem1", "stem2", ...],
            "val":   ["stem3", ...],
            "test":  ["stem4", ...]
        }
    """
    with open(split_path, encoding="utf-8") as fh:
        return json.load(fh)
