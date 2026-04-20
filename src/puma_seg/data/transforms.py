"""
Albumentations-based augmentation pipelines for H&E histopathology images.

Two pipelines are provided:
  * ``get_train_transforms`` — heavy augmentation including stain variation
  * ``get_val_transforms``   — normalisation only (no spatial distortion)

All transforms accept a dict with keys ``image`` and (optionally) ``mask``.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 512) -> A.Compose:
    """Return a training augmentation pipeline for H&E images.

    Augmentations included:
      - Random flip & 90° rotations
      - Random crop / resize
      - Elastic deformation (simulates tissue deformation)
      - Brightness / contrast jitter (simulates stain variation)
      - HueSaturation shift for stain normalisation invariance
      - Gaussian noise & blur
      - ImageNet normalisation + ToTensor

    Args:
        image_size: Spatial size after cropping (both H and W).

    Returns:
        Albumentations ``Compose`` pipeline compatible with ``image`` and
        ``mask`` keys.
    """
    return A.Compose(
        [
            # Spatial augmentations — applied to image AND mask identically
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.3),

            # Tissue deformation
            A.ElasticTransform(
                alpha=60,
                sigma=6,
                p=0.3,
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),

            # Colour augmentations — applied to IMAGE ONLY (no mask key here)
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1.0
                    ),
                    A.CLAHE(clip_limit=2.0, p=1.0),
                ],
                p=0.5,
            ),
            # H&E stain colour shift
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.4,
            ),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),

            # Noise / blur
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ],
                p=0.2,
            ),

            # Normalise with ImageNet stats and convert to tensor
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


def get_val_transforms(image_size: int = 512) -> A.Compose:
    """Return a validation / inference transform pipeline (no augmentation).

    Args:
        image_size: Spatial size for centre-crop.

    Returns:
        Albumentations ``Compose`` pipeline.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


def get_crop_transforms(crop_size: int = 64) -> A.Compose:
    """Return transforms for individual nucleus-crop classification.

    Used by the ``NucleusClassifier`` training pipeline.
    """
    return A.Compose(
        [
            A.Resize(crop_size, crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=8, p=0.3
            ),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_crop_val_transforms(crop_size: int = 64) -> A.Compose:
    """Return validation-only transforms for nucleus crops."""
    return A.Compose(
        [
            A.Resize(crop_size, crop_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
