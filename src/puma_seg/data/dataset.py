"""
PyTorch Dataset classes for the PUMA nucleus segmentation challenge.

Two datasets are provided:

PUMASegmentationDataset
    Returns ``(image, instance_mask)`` pairs to fine-tune Cellpose.
    ``image``         — ``np.ndarray`` (H, W, 3) uint8  (no tensor conversion
                        because Cellpose's ``train_seg`` expects numpy arrays)
    ``instance_mask`` — ``np.ndarray`` (H, W) int32

PUMAClassificationDataset
    Returns ``(crop_tensor, class_id)`` pairs to train the nucleus classifier.
    Crops are extracted from instance masks given at construction time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .geojson_parser import (
    CLASS_NAMES_T1,
    CLASS_NAMES_T2,
    extract_nucleus_crops,
)
from .transforms import get_crop_transforms, get_crop_val_transforms

logger = logging.getLogger(__name__)


# ─── Segmentation Dataset ─────────────────────────────────────────────────────


class PUMASegmentationDataset(Dataset):
    """Dataset for fine-tuning Cellpose segmentation on PUMA.

    Expects the processed data directory layout:
        processed_dir/
        ├── {split}/
        │   ├── images/   *.png
        │   └── masks/    *.npy  (instance_mask, int32)

    Args:
        processed_dir: Root of the processed data directory.
        split:         One of ``"train"`` | ``"val"`` | ``"test"``.
        track:         Challenge track (1 or 2).
        transform:     Optional callable applied to both image and mask.
                       Should accept ``dict(image=..., mask=...)`` and return
                       the same dict. If ``None``, no augmentation is applied.
        image_size:    Resize target (None → keep original 1024×1024).
    """

    def __init__(
        self,
        processed_dir: Path | str,
        split: str = "train",
        track: int = 1,
        transform=None,
        image_size: Optional[int] = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.track = track
        self.transform = transform
        self.image_size = image_size

        self.image_dir = self.processed_dir / split / "images"
        self.mask_dir = self.processed_dir / split / "masks"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.image_paths = sorted(self.image_dir.glob("*.png"))
        if not self.image_paths:
            self.image_paths = sorted(self.image_dir.glob("*.tif"))

        logger.info(
            "PUMASegmentationDataset [%s] — %d images found.", split, len(self.image_paths)
        )

    # ── dunder methods ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / (img_path.stem + ".npy")

        # ── Load image ──────────────────────────────────────────────────────
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # → (H, W, 3) uint8

        # ── Load mask ───────────────────────────────────────────────────────
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        instance_mask = np.load(str(mask_path)).astype(np.int32)

        # ── Optional resize ─────────────────────────────────────────────────
        if self.image_size is not None:
            image = cv2.resize(image, (self.image_size, self.image_size))
            instance_mask = cv2.resize(
                instance_mask,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )

        # ── Optional transform ──────────────────────────────────────────────
        if self.transform is not None:
            result = self.transform(image=image, mask=instance_mask)
            image = result["image"]
            instance_mask = result["mask"]

        return image, instance_mask

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def class_names(self) -> List[str]:
        return CLASS_NAMES_T1 if self.track == 1 else CLASS_NAMES_T2

    def get_all_numpy(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return (images, masks) as Python lists of numpy arrays.

        Used by ``cellpose.train.train_seg`` which expects lists, not tensors.
        """
        logger.info(f"Loading {len(self)} images and masks to memory...")
        images, masks = [], []
        for idx in range(len(self)):
            if idx % 20 == 0:
                logger.info(f"Loading data: {idx}/{len(self)}")
            img, msk = self[idx]
            images.append(img)
            masks.append(msk)
        logger.info(f"Loaded {len(images)} images ({sum(img.nbytes for img in images) / 1e9:.2f} GB)")
        return images, masks


# ─── Classification Dataset ──────────────────────────────────────────────────


class PUMAClassificationDataset(Dataset):
    """Dataset for training the per-nucleus classifier.

    Builds a flat list of nucleus crops extracted from pre-computed instance
    masks together with per-instance class labels.

    Args:
        processed_dir: Root of the processed data directory.
        split:         One of ``"train"`` | ``"val"`` | ``"test"``.
        track:         Challenge track (1 or 2).
        crop_size:     Square size to resize each nucleus crop to.
        augment:       If ``True``, apply training augmentations; otherwise
                       only resize + normalise.
    """

    def __init__(
        self,
        processed_dir: Path | str,
        split: str = "train",
        track: int = 1,
        crop_size: int = 64,
        augment: bool = True,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.track = track
        self.crop_size = crop_size

        self.transform = (
            get_crop_transforms(crop_size)
            if augment
            else get_crop_val_transforms(crop_size)
        )

        self.image_dir = self.processed_dir / split / "images"
        self.mask_dir = self.processed_dir / split / "masks"
        self.label_dir = self.processed_dir / split / "labels"  # per-instance class

        self.crops: List[np.ndarray] = []  # each entry: (crop_size, crop_size, 3)
        self.labels: List[int] = []

        self._load_all_crops()

        logger.info(
            "PUMAClassificationDataset [%s] — %d nucleus crops (track %d).",
            split,
            len(self.crops),
            track,
        )

    def _load_all_crops(self) -> None:
        """Pre-load all nucleus crops into memory."""
        image_paths = sorted(self.image_dir.glob("*.png"))
        if not image_paths:
            image_paths = sorted(self.image_dir.glob("*.tif"))

        for img_path in image_paths:
            mask_path = self.mask_dir / (img_path.stem + ".npy")
            label_path = self.label_dir / (img_path.stem + ".npy")

            if not mask_path.exists() or not label_path.exists():
                logger.warning("Skipping %s — mask or label file missing.", img_path.stem)
                continue

            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            instance_mask = np.load(str(mask_path)).astype(np.int32)
            # instance_classes[i] = class_id for instance (i+1)
            instance_classes: np.ndarray = np.load(str(label_path))

            crops, inst_ids = extract_nucleus_crops(
                image, instance_mask, crop_size=self.crop_size
            )
            for crop, inst_id in zip(crops, inst_ids):
                class_idx = int(inst_id) - 1  # convert 1-based to 0-based list index
                if 0 <= class_idx < len(instance_classes):
                    self.crops.append(crop)
                    self.labels.append(int(instance_classes[class_idx]))

    # ── dunder methods ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        crop = self.crops[idx].copy()
        label = self.labels[idx]

        result = self.transform(image=crop)
        tensor = result["image"]  # already a FloatTensor from ToTensorV2

        return tensor, label

    # ── Class stats ───────────────────────────────────────────────────────────

    @property
    def class_names(self) -> List[str]:
        return CLASS_NAMES_T1 if self.track == 1 else CLASS_NAMES_T2

    def class_counts(self) -> Dict[int, int]:
        """Return {class_id: count} for all loaded crops."""
        counts: Dict[int, int] = {}
        for lbl in self.labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    def class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced training."""
        from .geojson_parser import NUM_CLASSES

        n_classes = NUM_CLASSES[self.track]
        counts = self.class_counts()
        total = sum(counts.values()) or 1

        weights = []
        for cls_id in range(1, n_classes):  # skip background (0)
            cnt = counts.get(cls_id, 1)
            weights.append(total / (n_classes * cnt))

        return torch.tensor(weights, dtype=torch.float32)
