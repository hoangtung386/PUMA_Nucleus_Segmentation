
import numpy as np
import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_batch(device):
    B, H, W = 2, 256, 256
    return {
        "image": torch.randn(B, 3, H, W, device=device),
        "tissue_sem": torch.randint(0, 6, (B, H, W), device=device),
        "nuclei_np": (torch.rand(B, 1, H, W, device=device) > 0.5).float(),
        "nuclei_nc": torch.randint(0, 10, (B, H, W), device=device),
        "nuclei_hv": torch.randn(B, 2, H, W, device=device),
        "site_id": torch.zeros(B, dtype=torch.long, device=device),
    }


@pytest.fixture
def temp_dataset_dir(tmp_path):
    data_dir = tmp_path / "dataset"
    (data_dir / "images").mkdir(parents=True)
    (data_dir / "tissue_sem").mkdir()
    (data_dir / "nuclei_np").mkdir()
    (data_dir / "nuclei_nc").mkdir()
    (data_dir / "nuclei_hv").mkdir()
    for i in range(5):
        np.save(data_dir / "images" / f"sample_{i}.npy", np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        np.save(data_dir / "tissue_sem" / f"sample_{i}.npy", np.random.randint(0, 6, (256, 256), dtype=np.uint8))
        np.save(data_dir / "nuclei_np" / f"sample_{i}.npy", np.random.randint(0, 2, (256, 256), dtype=np.uint8))
        np.save(data_dir / "nuclei_nc" / f"sample_{i}.npy", np.random.randint(0, 10, (256, 256), dtype=np.uint8))
        np.save(data_dir / "nuclei_hv" / f"sample_{i}.npy", np.random.randn(256, 256, 2).astype(np.float16))
    return data_dir
