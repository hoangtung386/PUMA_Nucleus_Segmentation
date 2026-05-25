import functools
import json
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.constants import IGNORE_INDEX, NORMALIZATION_MEAN, NORMALIZATION_STD, SITE_MAP
from data.dataset.sampling import compute_all_sample_weights
from training.logging_utils import logger


_CONTEXT_CACHE: dict[str, np.ndarray] = {}
_CONTEXT_ROI_SIZE = 320


def _load_context_roi(context_dir: Path, source_name: str) -> Optional[np.ndarray]:
    if source_name in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[source_name]
    for ext in (".tif", ".tiff", ".png", ".jpg"):
        path = context_dir / f"{source_name}_context{ext}"
        if path.exists():
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (_CONTEXT_ROI_SIZE, _CONTEXT_ROI_SIZE), interpolation=cv2.INTER_LINEAR)
                _CONTEXT_CACHE[source_name] = img.astype(np.float32)
                return _CONTEXT_CACHE[source_name]
            break
    for ext in (".tif", ".tiff"):
        path = context_dir / f"{source_name}_context{ext}"
        if path.exists():
            try:
                import tifffile
                img = tifffile.imread(str(path))
                if img.ndim == 3 and img.shape[0] in (3, 4):
                    img = img.transpose(1, 2, 0)
                if img.shape[-1] == 4:
                    img = img[..., :3]
                if img.dtype != np.uint8:
                    img = img.astype(np.float32)
                    img = (img - img.min()) / (img.max() + 1e-6) * 255.0
                    img = img.clip(0, 255).astype(np.uint8)
                img = cv2.resize(img, (_CONTEXT_ROI_SIZE, _CONTEXT_ROI_SIZE), interpolation=cv2.INTER_LINEAR)
                _CONTEXT_CACHE[source_name] = img.astype(np.float32)
                return _CONTEXT_CACHE[source_name]
            except Exception:
                pass
    return None


def puma_tissue_to_internal(tissue_puma: np.ndarray) -> np.ndarray:
    return tissue_puma.astype(np.int64).copy()


def internal_tissue_to_puma(tissue_internal: np.ndarray) -> np.ndarray:
    return tissue_internal.astype(np.uint8)


def source_name_from_base(base_name: str) -> str:
    return str(base_name).split("__rare", 1)[0]


def infer_site_id(source_name: str) -> int:
    name_lower = source_name.lower()
    if "primary" in name_lower:
        return 0
    if "lymph" in name_lower:
        return 1
    if "brain" in name_lower:
        return 2
    if "bone" in name_lower:
        return 3
    if "soft_tissue" in name_lower or "softtissue" in name_lower:
        return 4
    if "liver" in name_lower:
        return 5
    if "lung" in name_lower:
        return 6
    if "gastro" in name_lower or "intestinal" in name_lower:
        return 7
    if "skin" in name_lower and "primary" not in name_lower:
        return 8
    return 1


class PUMADataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        transforms: Callable[..., dict] | None = None,
        context_dir: str | Path | None = None,
        use_context: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.image_files = sorted((self.data_dir / "images").glob("*.npy"))
        if not self.image_files:
            raise FileNotFoundError(f"No .npy files found in {self.data_dir / 'images'}")
        self.metadata = self._load_metadata()
        self.base_names = [p.stem for p in self.image_files]
        self.source_names = [self.get_source_name(i) for i in range(len(self.image_files))]
        self.is_original = [not self.is_rare_augmented(i) for i in range(len(self.image_files))]
        self._validate_files()
        self.use_context = use_context
        if use_context:
            if context_dir is None:
                context_dir = Path(data_dir).parent / "Dataset" / "PUMA" / "01_training_dataset_tif_context_ROIs"
            self.context_dir = Path(context_dir)
            if not self.context_dir.is_dir():
                logger.warning("Context dir %s does not exist; disabling context ROI", self.context_dir)
                self.use_context = False
            else:
                logger.info("Context ROIs enabled from %s", self.context_dir)

    def _load_metadata(self) -> dict[str, Any]:
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
        return len(self.image_files)

    def get_base_name(self, idx: int) -> str:
        return self.image_files[int(idx)].stem

    def get_source_name(self, idx: int) -> str:
        base_name = self.get_base_name(idx)
        meta = self.metadata.get(base_name, {})
        return str(meta.get("source_name") or source_name_from_base(base_name))

    def is_rare_augmented(self, idx: int) -> bool:
        base_name = self.get_base_name(idx)
        meta = self.metadata.get(base_name, {})
        if "is_rare_augmented" in meta:
            return bool(meta["is_rare_augmented"])
        return "__rare" in base_name

    def get_split_metadata(self) -> dict[str, Any]:
        return {
            "source_names": [self.get_source_name(i) for i in range(len(self))],
            "is_original": [not self.is_rare_augmented(i) for i in range(len(self))],
            "base_names": [self.get_base_name(i) for i in range(len(self))],
        }

    def compute_sample_weights(self, indices: list[int] | None = None) -> list[float]:
        if indices is None:
            indices = list(range(len(self)))
        base_names = [self.get_base_name(i) for i in indices]
        is_rare = [self.is_rare_augmented(i) for i in indices]
        return compute_all_sample_weights(self.data_dir, base_names, is_rare, self.metadata)

    @staticmethod
    def _ensure_hwc_2ch(x: np.ndarray, h: int, w: int) -> np.ndarray:
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
        idx = int(idx)
        base_name = self.get_base_name(idx)
        site_id = infer_site_id(self.get_source_name(idx))

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

        if self.transforms is not None:
            augmented = self.transforms(
                image=image,
                tissue_mask=tissue_internal,
                nuclei_mask=nuclei_nc,
                hv_map=nuclei_hv,
            )
            image = augmented["image"]
            tissue_internal = augmented["tissue_mask"]
            nuclei_nc = augmented["nuclei_mask"]
            nuclei_hv = augmented["hv_map"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mean = torch.tensor(NORMALIZATION_MEAN, dtype=image.dtype).view(3, 1, 1)
            std = torch.tensor(NORMALIZATION_STD, dtype=image.dtype).view(3, 1, 1)
            image = (image - mean) / std

        tissue_internal = torch.as_tensor(tissue_internal, dtype=torch.long)
        nuclei_nc = torch.as_tensor(nuclei_nc, dtype=torch.long)
        nuclei_np = (nuclei_nc != IGNORE_INDEX).long()

        if isinstance(nuclei_hv, np.ndarray):
            nuclei_hv = torch.from_numpy(nuclei_hv.transpose(2, 0, 1)).float()

        sample = {
            "image": image,
            "tissue_sem": tissue_internal,
            "nuclei_np": nuclei_np,
            "nuclei_nc": nuclei_nc,
            "nuclei_hv": nuclei_hv,
            "site_id": site_id,
            "base_name": base_name,
            "source_name": self.get_source_name(idx),
            "is_rare_augmented": self.is_rare_augmented(idx),
        }

        if self.use_context:
            source_name = self.get_source_name(idx)
            ctx = _load_context_roi(self.context_dir, source_name)
            if ctx is not None:
                mean = np.array(NORMALIZATION_MEAN, dtype=np.float32).reshape(1, 1, 3)
                std = np.array(NORMALIZATION_STD, dtype=np.float32).reshape(1, 1, 3)
                ctx = (ctx / 255.0 - mean) / std
                sample["context_roi"] = torch.from_numpy(ctx.transpose(2, 0, 1)).float()

        return sample
