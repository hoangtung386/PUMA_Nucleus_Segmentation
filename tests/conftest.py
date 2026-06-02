"""Shared pytest fixtures for SymbioPan."""

import numpy as np
import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_batch(device: torch.device) -> dict:
    batch, height, width = 2, 256, 256
    return {
        "image": torch.randn(batch, 3, height, width, device=device),
        "tissue_sem": torch.randint(0, 6, (batch, height, width), device=device),
        "nuclei_np": (torch.rand(batch, 1, height, width, device=device) > 0.5).float(),
        "nuclei_nc": torch.randint(0, 10, (batch, height, width), device=device),
        "nuclei_hv": torch.randn(batch, 2, height, width, device=device),
        "site_id": torch.zeros(batch, dtype=torch.long, device=device),
    }


@pytest.fixture
def temp_dataset_dir(tmp_path):
    """Create a temporary directory mimicking ``PUMADataset`` on-disk layout."""
    data_dir = tmp_path / "dataset"
    (data_dir / "images").mkdir(parents=True)
    (data_dir / "tissue_sem").mkdir()
    (data_dir / "nuclei_nc").mkdir()
    (data_dir / "nuclei_hv").mkdir()
    for i in range(5):
        np.save(data_dir / "images" / f"sample_{i}.npy", np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        np.save(data_dir / "tissue_sem" / f"sample_{i}.npy", np.random.randint(0, 6, (256, 256), dtype=np.uint8))
        np.save(data_dir / "nuclei_nc" / f"sample_{i}.npy", np.random.randint(0, 10, (256, 256), dtype=np.uint8))
        np.save(data_dir / "nuclei_hv" / f"sample_{i}.npy", np.random.randn(256, 256, 2).astype(np.float16))
    return data_dir
