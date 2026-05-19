from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset


PUMA_TISSUE_ID_TO_NAME = {
    0: "background",
    1: "tissue_stroma",
    2: "tissue_blood_vessel",
    3: "tissue_tumor",
    4: "tissue_epidermis",
    5: "tissue_necrosis",
}

INTERNAL_TISSUE_ID_TO_NAME = {
    0: "tissue_stroma",
    1: "tissue_blood_vessel",
    2: "tissue_tumor",
    3: "tissue_epidermis",
    4: "tissue_necrosis",
}

PUMA_NUCLEI_ID_TO_NAME = {
    0: "nuclei_tumor",
    1: "nuclei_lymphocyte",
    2: "nuclei_plasma_cell",
    3: "nuclei_histiocyte",
    4: "nuclei_melanophage",
    5: "nuclei_neutrophil",
    6: "nuclei_stroma",
    7: "nuclei_epithelium",
    8: "nuclei_endothelium",
    9: "nuclei_apoptosis",
}

RARE_TISSUE_IDS_PUMA = [2, 4, 5]
RARE_NUCLEI_IDS = [2, 4, 5, 8, 9]


def puma_tissue_to_internal(tissue_puma):
    """Convert stored PUMA tissue IDs 0..5 into targets 0..4 plus ignore=255."""
    out = tissue_puma.astype(np.int64).copy()
    out[out == 0] = 255
    valid = out != 255
    out[valid] = out[valid] - 1
    return out


def internal_tissue_to_puma(tissue_internal):
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
        train_stage1.py/train_stage2.py use group splitting on source_names.
        Validation uses original samples only by default.
    """

    def __init__(self, data_dir, transforms=None, zero_cellpose_prob=0.0):
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

    def _load_metadata(self):
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
            print(f"[Dataset][WARN] Could not read {path}: {exc}")
        return out

    def _validate_files(self):
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

    def __len__(self):
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

    def get_split_metadata(self):
        return {
            "source_names": [self.get_source_name(i) for i in range(len(self))],
            "is_original": [not self.is_rare_augmented(i) for i in range(len(self))],
            "base_names": [self.get_base_name(i) for i in range(len(self))],
        }

    def _infer_site_type(self, base_name: str):
        name_lower = base_name.lower()
        if "primary" in name_lower:
            return "primary"
        if "metastatic" in name_lower or "metastasis" in name_lower or "meta" in name_lower:
            return "metastatic"
        return "metastatic"

    def compute_sample_weights(self, indices=None):
        if indices is None:
            indices = list(range(len(self)))
        weights = []
        for idx in indices:
            idx = int(idx)
            base_name = self.get_base_name(idx)
            meta = self.metadata.get(base_name)
            if meta is not None and "sample_weight" in meta:
                weights.append(float(meta["sample_weight"]))
                continue

            weight = 1.0
            tissue_path = self.data_dir / "tissue_sem" / f"{base_name}.npy"
            nuclei_path = self.data_dir / "nuclei_nc" / f"{base_name}.npy"

            if tissue_path.exists():
                tissue = np.load(tissue_path, mmap_mode="r")
                if np.any(tissue == 2):
                    weight += 3.0
                if np.any(tissue == 4):
                    weight += 2.0
                if np.any(tissue == 5):
                    weight += 6.0

            if nuclei_path.exists():
                nuclei = np.load(nuclei_path, mmap_mode="r")
                if np.any(nuclei == 2):
                    weight += 6.0
                if np.any(nuclei == 4):
                    weight += 4.0
                if np.any(nuclei == 5):
                    weight += 8.0
                if np.any(nuclei == 8):
                    weight += 5.0
                if np.any(nuclei == 9):
                    weight += 8.0

            if self.is_rare_augmented(idx):
                weight *= 1.5
            weights.append(float(weight))
        return weights

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

    def __getitem__(self, idx):
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
            raise RuntimeError(f"Tissue mask shape {tissue_internal.shape} does not match image {(h, w)} for {base_name}")
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
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(3, 1, 1)
            image = (image - mean) / std

        tissue_internal = torch.as_tensor(tissue_internal, dtype=torch.long)
        nuclei_nc = torch.as_tensor(nuclei_nc, dtype=torch.long)
        nuclei_np = (nuclei_nc != 255).long()

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
