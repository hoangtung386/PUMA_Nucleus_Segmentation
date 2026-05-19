from .puma_dataset import (
    PUMADataset,
    PUMA_TISSUE_ID_TO_NAME,
    INTERNAL_TISSUE_ID_TO_NAME,
    PUMA_NUCLEI_ID_TO_NAME,
    internal_tissue_to_puma,
    puma_tissue_to_internal,
)

try:
    from .transforms import get_train_transforms, get_val_transforms
except Exception as exc:  # Allows inference imports without albumentations installed.
    def get_train_transforms(*args, **kwargs):
        raise ImportError("albumentations is required for training transforms") from exc

    def get_val_transforms(*args, **kwargs):
        raise ImportError("albumentations is required for validation transforms") from exc
