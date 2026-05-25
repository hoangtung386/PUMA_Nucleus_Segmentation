from data.dataset.puma_dataset import PUMADataset
from data.dataset.transforms import get_train_transforms, get_val_transforms


def get_train_transforms_stain_aug(image_size: int = 1024):
    return get_train_transforms(image_size, use_stain_aug=True)


__all__ = [
    "PUMADataset",
    "get_train_transforms",
    "get_train_transforms_stain_aug",
    "get_val_transforms",
]
