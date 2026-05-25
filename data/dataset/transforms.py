from copy import deepcopy
from typing import Any

import albumentations as A
import cv2
import numpy as np


def _fix_vector_field(vec: np.ndarray | None, replay: dict[str, Any]) -> np.ndarray | None:
    if vec is None:
        return vec
    for tr in replay.get("transforms", []):
        if not tr.get("applied", False):
            continue
        name = tr.get("__class_fullname__", "")
        if name.endswith("HorizontalFlip"):
            vec[..., 0] *= -1
        elif name.endswith("VerticalFlip"):
            vec[..., 1] *= -1
        elif name.endswith("RandomRotate90"):
            k = int(tr.get("params", {}).get("factor", 0)) % 4
            x = vec[..., 0].copy()
            y = vec[..., 1].copy()
            if k == 1:
                vec[..., 0] = -y
                vec[..., 1] = x
            elif k == 2:
                vec[..., 0] = -x
                vec[..., 1] = -y
            elif k == 3:
                vec[..., 0] = y
                vec[..., 1] = -x
    return vec


class TrainTransform:
    def __init__(self, image_size: int, use_stain_aug: bool = False) -> None:
        spatial = [
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]
        pixel = [
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        ]
        if use_stain_aug:
            pixel.append(A.HEStain(p=0.5))
        self.spatial = A.ReplayCompose(
            spatial,
            additional_targets={
                "tissue_mask": "mask",
                "nuclei_mask": "mask",
                "hv_map": "image",
            },
        )
        self.pixel = A.Compose(pixel)

    def __call__(self, **kwargs: object) -> dict[str, Any]:
        out = self.spatial(**kwargs)
        replay = out.pop("replay", {})
        out["hv_map"] = _fix_vector_field(out.get("hv_map"), replay)
        img = self.pixel(image=out["image"])["image"]
        out["image"] = img
        return out


def get_train_transforms(image_size: int = 1024, use_stain_aug: bool = False) -> TrainTransform:
    return TrainTransform(image_size, use_stain_aug=use_stain_aug)


def get_val_transforms(image_size: int = 1024) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
        ],
        additional_targets={
            "tissue_mask": "mask",
            "nuclei_mask": "mask",
            "hv_map": "image",
        },
    )
