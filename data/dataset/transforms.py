from typing import Any

import albumentations as A
import cv2
import numpy as np


def _fix_vector_field(vec: np.ndarray | None, replay: dict[str, Any]) -> np.ndarray | None:
    """Correct x/y vector directions after flips and 90-degree rotations."""
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


class VectorSafeCompose:
    """Wrapper around ReplayCompose that corrects vector fields after
    flips and rotations.
    """

    def __init__(self, transforms: list[Any], additional_targets: dict[str, str]) -> None:
        """Initialize VectorSafeCompose.

        Args:
            transforms: List of albumentations transforms.
            additional_targets: dict of additional target names to their types.
        """
        self.aug = A.ReplayCompose(transforms, additional_targets=additional_targets)

    def __call__(self, **kwargs: object) -> dict[str, Any]:
        """Apply transforms and fix vector field orientations.

        Args:
            **kwargs: Keyword arguments passed to the compose pipeline
                (image, tissue_mask, nuclei_mask, cp_flow, hv_map).

        Returns:
            dict of augmented data with corrected vector fields.
        """
        out = self.aug(**kwargs)
        replay = out.get("replay", {})
        out["cp_flow"] = _fix_vector_field(out.get("cp_flow"), replay)
        out["hv_map"] = _fix_vector_field(out.get("hv_map"), replay)
        out.pop("replay", None)
        return out


def get_train_transforms(image_size: int = 1024) -> VectorSafeCompose:
    """Return training data augmentation pipeline.

    Includes resize, horizontal/vertical flips, and random 90-degree rotations
    with proper vector field correction.

    Args:
        image_size: Target spatial size (height and width).

    Returns:
        VectorSafeCompose instance configured for training.
    """
    return VectorSafeCompose(
        [
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ],
        additional_targets={
            "tissue_mask": "mask",
            "nuclei_mask": "mask",
            "cp_flow": "image",
            "hv_map": "image",
        },
    )


def get_val_transforms(image_size: int = 1024) -> A.Compose:
    """Return validation data transformation pipeline.

    Only resizes to the target size without random augmentations.

    Args:
        image_size: Target spatial size (height and width).

    Returns:
        A.Compose instance configured for validation.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
        ],
        additional_targets={
            "tissue_mask": "mask",
            "nuclei_mask": "mask",
            "cp_flow": "image",
            "hv_map": "image",
        },
    )
