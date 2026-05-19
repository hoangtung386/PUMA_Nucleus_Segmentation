import numpy as np
import torch

from data.constants import NORMALIZATION_MEAN, NORMALIZATION_STD


def normalize_image(
    img: np.ndarray | torch.Tensor,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> np.ndarray | torch.Tensor:
    mean = mean or NORMALIZATION_MEAN
    std = std or NORMALIZATION_STD
    if isinstance(img, np.ndarray):
        x = img.astype(np.float32) / 255.0
        x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
        return x
    mean_t = torch.tensor(mean, dtype=img.dtype, device=img.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=img.dtype, device=img.device).view(3, 1, 1)
    return (img - mean_t) / std_t
