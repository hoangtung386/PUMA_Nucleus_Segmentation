"""Tests for data transforms including stain augmentation."""

import numpy as np

from data.dataset.transforms import get_train_transforms, get_val_transforms


def test_val_transforms_output_keys():
    transforms = get_val_transforms(256)
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    tissue = np.random.randint(0, 5, (512, 512), dtype=np.uint8)
    nuclei = np.random.randint(0, 10, (512, 512), dtype=np.uint8)
    hv = np.random.randn(512, 512, 2).astype(np.float32)
    out = transforms(image=image, tissue_mask=tissue, nuclei_mask=nuclei, hv_map=hv)
    assert "image" in out
    assert "tissue_mask" in out
    assert "nuclei_mask" in out
    assert "hv_map" in out
    assert out["image"].shape == (256, 256, 3)


def test_train_transforms_with_stain_aug():
    transforms = get_train_transforms(256, use_stain_aug=True)
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    tissue = np.random.randint(0, 5, (512, 512), dtype=np.uint8)
    nuclei = np.random.randint(0, 10, (512, 512), dtype=np.uint8)
    hv = np.random.randn(512, 512, 2).astype(np.float32)
    out = transforms(image=image, tissue_mask=tissue, nuclei_mask=nuclei, hv_map=hv)
    assert out["image"].shape == (256, 256, 3)
