import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from data.constants import (
    IGNORE_INDEX,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
)
from data.dataset.sampling import compute_all_sample_weights
from training.logging_utils import logger


def puma_tissue_to_internal(tissue_puma: np.ndarray) -> np.ndarray:
    """Convert stored PUMA tissue IDs 0..5 into targets 0..4 plus ignore=255."""
    out = tissue_puma.astype(np.int64).copy()
    out[out == 0] = IGNORE_INDEX
    valid = out != IGNORE_INDEX
    out[valid] = out[valid] - 1
    return out


def internal_tissue_to_puma(tissue_internal: np.ndarray) -> np.ndarray:
    """Convert model predictions 0..4 back to PUMA tissue IDs 1..5."""
    return tissue_internal.astype(np.uint8) + 1


def source_name_from_base(base_name: str) -> str:
    """Original ROI/source name. Rare augmented samples have suffix __rareXX_..."""
    return str(base_name).split("__rare", 1)[0]


class PUMADataset(Dataset):
    """
    Loader for rare-focused Version-4 processed 1024 data.

    Expected folders:
        images/*.npy          RGB uint8, HWC
        tissue_sem/*.npy      PUMA tissue IDs: 0 background, 1..5 tissue classes
        nuclei_nc/*.npy       nuclei IDs: 0..9, 255 non-nucleus
        nuclei_hv/*.npy       [2,H,W] or [H,W,2] float HV map
        cellpose_flows/*.npy  [2,H,W] or [H,W,2] float flow map, optional
        sample_metadata.json  optional metadata from preprocess.py

    Important leakage control:
        self.source_names maps every rare crop back to its original image.
    """

    def __init__(
        self,
        data_dir: str | Path,
        transforms: Callable[..., dict] | None = None,
        zero_cellpose_prob: float = 0.0,
    ) -> None:
        """Initialize the PUMADataset.

        Args:
            data_dir: Path to the processed data directory containing images/,
                tissue_sem/, nuclei_nc/, nuclei_hv/, and cellpose_flows/ subdirs.
            transforms: Optional albumentations transform pipeline.
            zero_cellpose_prob: Probability of zeroing out the cellpose flow
                for regularization.
        """
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.zero_cellpose_prob = float(zero_cellpose_prob)
        self.image_files = sorted((self.data_dir / "images").glob("*.npy"))
        if not self.image_files:
            raise FileNotFoundError(f"No .npy files found in {self.data_dir / 'images'}")
        self.metadata = self._load_metadata()
        self.base_names = [p.stem for p in self.image_files]
        self.source_names = [self.get_source_name(i) for i in range(len(self.image_files))]
        self.is_original = [not self.is_rare_augmented(i) for i in range(len(self.image_files))]
        self._validate_files()

    def _load_metadata(self) -> dict[str, Any]:
        """Load sample metadata from sample_metadata.json if it exists.

        Returns:
            dict mapping base_name to its metadata row.
        """
        path = self.data_dir / "sample_metadata.json"
        out = {}
        if not path.exists():
            return out
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            for row in rows:
                name = row.get("base_name")
                if name:
                    out[str(name)] = row
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
        return out

    def _validate_files(self) -> None:
        """Verify that all required annotation files exist for each sample.

        Raises:
            FileNotFoundError: If any required files are missing (up to 10
                reported).
        """
        required_dirs = ["tissue_sem", "nuclei_nc", "nuclei_hv"]
        missing = []
        for base in self.base_names:
            for folder in required_dirs:
                path = self.data_dir / folder / f"{base}.npy"
                if not path.exists():
                    missing.append(str(path))
                    if len(missing) >= 10:
                        break
            if len(missing) >= 10:
                break
        if missing:
            preview = "\n".join(missing)
            raise FileNotFoundError(f"Processed dataset is incomplete. Missing files include:\n{preview}")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.image_files)

    def get_base_name(self, idx: int) -> str:
        """Return the base filename (without extension) for the given index.

        Args:
            idx: Sample index.

        Returns:
            Stem of the image file name.
        """
        return self.image_files[int(idx)].stem

    def get_source_name(self, idx: int) -> str:
        """Return the original ROI/source name for the given index.

        Args:
            idx: Sample index.

        Returns:
            Source name from metadata, or inferred from the base name.
        """
        base_name = self.get_base_name(idx)
        meta = self.metadata.get(base_name, {})
        return str(meta.get("source_name") or source_name_from_base(base_name))

    def is_rare_augmented(self, idx: int) -> bool:
        """Check whether the sample at the given index is a rare-class crop.

        Args:
            idx: Sample index.

        Returns:
            True if the sample is a rare-augmented crop.
        """
        base_name = self.get_base_name(idx)
        meta = self.metadata.get(base_name, {})
        if "is_rare_augmented" in meta:
            return bool(meta["is_rare_augmented"])
        return "__rare" in base_name

    def get_split_metadata(self) -> dict[str, Any]:
        """Return metadata dictionaries for all samples.

        Returns:
            dict with keys 'source_names', 'is_original', and 'base_names',
            each a list over all dataset indices.
        """
        return {
            "source_names": [self.get_source_name(i) for i in range(len(self))],
            "is_original": [not self.is_rare_augmented(i) for i in range(len(self))],
            "base_names": [self.get_base_name(i) for i in range(len(self))],
        }

    def _infer_site_type(self, base_name: str) -> str:
        """Infer tissue site type (primary/metastatic) from the source name.

        Args:
            base_name: Source name string.

        Returns:
            'primary' or 'metastatic'.
        """
        name_lower = base_name.lower()
        if "primary" in name_lower:
            return "primary"
        if "metastatic" in name_lower or "metastasis" in name_lower or "meta" in name_lower:
            return "metastatic"
        return "metastatic"

    def compute_sample_weights(self, indices: list[int] | None = None) -> list[float]:
        """Compute weighted sampling probabilities focusing on rare classes.

        Args:
            indices: Optional list of sample indices to compute weights for.
                If None, all samples are used.

        Returns:
            list of float weights, one per index.
        """
        if indices is None:
            indices = list(range(len(self)))
        base_names = [self.get_base_name(i) for i in indices]
        is_rare = [self.is_rare_augmented(i) for i in indices]
        return compute_all_sample_weights(self.data_dir, base_names, is_rare, self.metadata)

    @staticmethod
    def _ensure_hwc_2ch(x: np.ndarray, h: int, w: int) -> np.ndarray:
        """Validate and convert a vector map to [H, W, 2] format.

        Args:
            x: Input array, either [2, H, W] or [H, W, 2].
            h: Expected height.
            w: Expected width.

        Returns:
            Array in [H, W, 2] float32 format.

        Raises:
            RuntimeError: If the array has unexpected shape or channel count.
        """
        if x.ndim != 3:
            raise RuntimeError(f"Expected 3D vector map, got shape={x.shape}")
        if x.shape[0] == 2:
            x = x.transpose(1, 2, 0)
        if x.shape[-1] != 2:
            raise RuntimeError(f"Expected vector map with 2 channels, got shape={x.shape}")
        if x.shape[0] != h or x.shape[1] != w:
            raise RuntimeError(f"Vector map shape {x.shape[:2]} does not match image shape {(h, w)}")
        return x.astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load and return a sample from the dataset.

        Loads the image, tissue semantic mask, nuclei classification mask,
        nuclei HV map, and cellpose flow from disk. Applies transforms if
        configured, normalizes the image, and converts arrays to tensors.

        Args:
            idx: Sample index.

        Returns:
            dict containing:
                - image: [3, H, W] normalized float32 tensor.
                - tissue_sem: [H, W] long tensor of internal tissue IDs.
                - nuclei_np: [H, W] long binary mask (nucleus present).
                - nuclei_nc: [H, W] long tensor of nuclei class IDs.
                - nuclei_hv: [2, H, W] float32 HV map.
                - cellpose_flow: [2, H, W] float32 flow map.
                - site_type: str, 'primary' or 'metastatic'.
                - base_name: str, stem of the sample file.
                - source_name: str, original ROI name.
                - is_rare_augmented: bool.
        """
        idx = int(idx)
        base_name = self.get_base_name(idx)
        site_type = self._infer_site_type(self.get_source_name(idx))

        image = np.load(self.data_dir / "images" / f"{base_name}.npy")
        if image.ndim != 3 or image.shape[-1] != 3:
            raise RuntimeError(f"Expected RGB HWC image for {base_name}, got shape={image.shape}")
        h, w = image.shape[:2]

        tissue_puma = np.load(self.data_dir / "tissue_sem" / f"{base_name}.npy").astype(np.uint8)
        tissue_internal = puma_tissue_to_internal(tissue_puma).astype(np.uint8)
        nuclei_nc = np.load(self.data_dir / "nuclei_nc" / f"{base_name}.npy").astype(np.uint8)

        if tissue_internal.shape != (h, w):
            raise RuntimeError(
                f"Tissue mask shape {tissue_internal.shape} does not match image {(h, w)} for {base_name}"
            )
        if nuclei_nc.shape != (h, w):
            raise RuntimeError(f"Nuclei mask shape {nuclei_nc.shape} does not match image {(h, w)} for {base_name}")

        nuclei_hv = np.load(self.data_dir / "nuclei_hv" / f"{base_name}.npy")
        nuclei_hv = self._ensure_hwc_2ch(nuclei_hv, h, w)

        cp_path = self.data_dir / "cellpose_flows" / f"{base_name}.npy"
        if cp_path.exists():
            cellpose_flow = np.load(cp_path)
            cellpose_flow = self._ensure_hwc_2ch(cellpose_flow, h, w)
        else:
            cellpose_flow = np.zeros((h, w, 2), dtype=np.float32)

        if self.zero_cellpose_prob > 0.0 and np.random.rand() < self.zero_cellpose_prob:
            cellpose_flow = np.zeros_like(cellpose_flow, dtype=np.float32)

        if self.transforms is not None:
            augmented = self.transforms(
                image=image,
                tissue_mask=tissue_internal,
                nuclei_mask=nuclei_nc,
                cp_flow=cellpose_flow,
                hv_map=nuclei_hv,
            )
            image = augmented["image"]
            tissue_internal = augmented["tissue_mask"]
            nuclei_nc = augmented["nuclei_mask"]
            cellpose_flow = augmented["cp_flow"]
            nuclei_hv = augmented["hv_map"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mean = torch.tensor(NORMALIZATION_MEAN, dtype=image.dtype).view(3, 1, 1)
            std = torch.tensor(NORMALIZATION_STD, dtype=image.dtype).view(3, 1, 1)
            image = (image - mean) / std

        tissue_internal = torch.as_tensor(tissue_internal, dtype=torch.long)
        nuclei_nc = torch.as_tensor(nuclei_nc, dtype=torch.long)
        nuclei_np = (nuclei_nc != IGNORE_INDEX).long()

        if isinstance(cellpose_flow, np.ndarray):
            cellpose_flow = torch.from_numpy(cellpose_flow.transpose(2, 0, 1)).float()
        if isinstance(nuclei_hv, np.ndarray):
            nuclei_hv = torch.from_numpy(nuclei_hv.transpose(2, 0, 1)).float()

        return {
            "image": image,
            "tissue_sem": tissue_internal,
            "nuclei_np": nuclei_np,
            "nuclei_nc": nuclei_nc,
            "nuclei_hv": nuclei_hv,
            "cellpose_flow": cellpose_flow,
            "site_type": site_type,
            "base_name": base_name,
            "source_name": self.get_source_name(idx),
            "is_rare_augmented": self.is_rare_augmented(idx),
        }
