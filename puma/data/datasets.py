from __future__ import annotations

import hashlib
import math
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from puma.config import PUMA_MICRONS_PER_PIXEL, STAGE2_GEOMETRY_DIM
from puma.data.targets import build_dense_targets


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]



def _colab_local_artifact_paths(
    artifact_dir: Path,
    names: dict[str, str],
) -> dict[str, Path]:
    """Mirror read-hot immutable preprocessing files from Drive to local Colab disk.

    Random memory-mapped access through Google Drive is often the dominant DataLoader
    bottleneck. The cache is byte-for-byte, signature-keyed, and never modifies the source.
    Large segmentation maps that training does not read are intentionally left on Drive.
    """
    source_paths = {name: artifact_dir / filename for name, filename in names.items()}
    disabled = os.environ.get("PUMA_DISABLE_LOCAL_DATA_CACHE", "").strip().lower()
    is_colab_drive = str(artifact_dir).startswith("/content/drive/")
    if disabled in {"1", "true", "yes"} or not is_colab_drive:
        return source_paths
    hot_names = ("images", "manifest", "centroids", "offsets", "folds")
    try:
        signature_parts = []
        total_bytes = 0
        for name in hot_names:
            path = source_paths[name]
            stat = path.stat()
            signature_parts.append(
                f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
            )
            total_bytes += int(stat.st_size)
        free_bytes = shutil.disk_usage("/content").free
        # Keep a conservative reserve for Colab packages, checkpoints, and temporary files.
        if total_bytes > max(0, free_bytes - 4 * 1024**3):
            print(
                "Skipping local preprocessing cache: insufficient /content disk "
                f"({total_bytes / 1024**3:.1f} GiB required)."
            )
            return source_paths
        signature = hashlib.sha256(
            "|".join(signature_parts).encode("utf-8")
        ).hexdigest()[:16]
        cache_dir = Path("/content/puma_preprocessing_cache") / signature
        cache_dir.mkdir(parents=True, exist_ok=True)
        resolved = dict(source_paths)
        copied_any = False
        for name in hot_names:
            source = source_paths[name]
            destination = cache_dir / source.name
            if not destination.exists() or destination.stat().st_size != source.stat().st_size:
                temporary = destination.with_name(
                    destination.name + f".{os.getpid()}.tmp"
                )
                shutil.copyfile(source, temporary)
                os.replace(temporary, destination)
                copied_any = True
            resolved[name] = destination
        if copied_any:
            print(
                "Cached read-hot preprocessing arrays on local Colab disk: "
                f"{cache_dir}"
            )
        else:
            print(f"Reusing local preprocessing cache: {cache_dir}")
        return resolved
    except Exception as exc:
        print(f"Local preprocessing cache unavailable; reading from Drive: {exc}")
        return source_paths


@dataclass(slots=True)
class PumaNpyStore:
    images: np.ndarray
    instances: np.ndarray
    classes: np.ndarray
    canonical_heatmaps: np.ndarray
    manifest: np.ndarray
    centroids: np.ndarray
    offsets: np.ndarray
    folds: np.ndarray

    @classmethod
    def open(cls, artifact_dir: Path) -> "PumaNpyStore":
        names = {
            "images": "puma_rgb_images.npy",
            "instances": "puma_instance_maps.npy",
            "classes": "puma_class_maps.npy",
            "canonical_heatmaps": "puma_centroid_heatmaps.npy",
            "manifest": "puma_roi_manifest.npy",
            "centroids": "puma_nuclei_centroids.npy",
            "offsets": "puma_roi_centroid_offsets.npy",
            "folds": "puma_fold_assignments.npy",
        }
        artifact_dir = Path(artifact_dir)
        missing = [name for name, fn in names.items() if not (artifact_dir / fn).exists()]
        if missing:
            raise FileNotFoundError(f"Missing preprocessed artifacts: {missing}. Run 00_Preprocess.ipynb first.")
        resolved_paths = _colab_local_artifact_paths(artifact_dir, names)
        return cls(**{
            name: np.load(resolved_paths[name], mmap_mode="r")
            for name in names
        })

    def roi_centroids(self, roi_index: int) -> np.ndarray:
        start, end = int(self.offsets[roi_index]), int(self.offsets[roi_index + 1])
        return self.centroids[start:end]

    def indices_for_fold(self, fold: int, train: bool) -> np.ndarray:
        return np.flatnonzero(self.folds != fold if train else self.folds == fold)


def image_to_uint8_tensor(image: np.ndarray) -> torch.Tensor:
    """Create a compact CHW uint8 tensor with at most one host copy.

    NumPy memmaps and D4 views are commonly read-only or non-contiguous, so they need one
    owned C-order copy before ``torch.from_numpy``. Already-owned C-order crops can be
    wrapped directly. The CHW view intentionally stays non-contiguous: DataLoader collation
    creates the final contiguous batch once, avoiding a second copy for every sample.
    """
    array = np.asarray(image, dtype=np.uint8)
    if not array.flags.c_contiguous or not array.flags.writeable:
        array = np.array(array, dtype=np.uint8, copy=True, order="C")
    return torch.from_numpy(array).permute(2, 0, 1)


def sample_stain_parameters(rng: np.random.Generator) -> np.ndarray:
    """Draw the original stain jitter parameters without materializing a float CPU image."""
    gain = rng.uniform(0.90, 1.10, size=3).astype(np.float32)
    bias = rng.uniform(-0.04, 0.04, size=3).astype(np.float32)
    gamma = np.float32(rng.uniform(0.90, 1.10))
    return np.concatenate([gain, bias, np.asarray([gamma], dtype=np.float32)])


def apply_stain_parameters_(
    images: torch.Tensor,
    stain_parameters: torch.Tensor | None,
) -> torch.Tensor:
    """Apply stain jitter on-device while preserving uint8 quantization."""
    if stain_parameters is None or stain_parameters.numel() == 0:
        return images
    parameters = stain_parameters.to(
        device=images.device, dtype=torch.float32, non_blocking=True
    )
    if parameters.ndim == 1:
        parameters = parameters.unsqueeze(0)
    gain = parameters[:, 0:3, None, None]
    bias = parameters[:, 3:6, None, None]
    gamma = parameters[:, 6:7, None, None]
    images.mul_(1.0 / 255.0).mul_(gain).add_(bias).clamp_(0.0, 1.0)
    images.pow_(gamma).mul_(255.0).add_(0.5).floor_().clamp_(0.0, 255.0)
    return images


def normalize_image_batch(
    images: torch.Tensor,
    stain_parameters: torch.Tensor | None = None,
) -> torch.Tensor:
    squeeze = images.ndim == 3
    if squeeze:
        images = images.unsqueeze(0)
    images = images.to(dtype=torch.float32)
    apply_stain_parameters_(images, stain_parameters)
    images.mul_(1.0 / 255.0)
    images[:, 0].sub_(0.485).div_(0.229)
    images[:, 1].sub_(0.456).div_(0.224)
    images[:, 2].sub_(0.406).div_(0.225)
    return images[0] if squeeze else images




def _apply_dihedral_xy(
    image: np.ndarray,
    coordinates: np.ndarray,
    width: int,
    height: int,
    code: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a D4 transform to the image and pixel-centre coordinates."""
    rotation = int(code) % 4
    flip = int(code) // 4
    output = image
    xy = np.asarray(coordinates, dtype=np.float32).copy()
    current_width, current_height = int(width), int(height)
    for _ in range(rotation):
        output = np.rot90(output, k=1)
        if len(xy):
            x, y = xy[:, 0].copy(), xy[:, 1].copy()
            xy[:, 0], xy[:, 1] = y, current_width - x
        current_width, current_height = current_height, current_width
    if flip:
        output = np.fliplr(output)
        if len(xy):
            xy[:, 0] = current_width - xy[:, 0]
    return np.ascontiguousarray(output), xy



def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile >= length:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, max(length - tile, 0) + 1, stride))
    final = length - tile
    if starts[-1] != final:
        starts.append(final)
    return starts


class Stage1TileDataset(Dataset):
    def __init__(
        self,
        store: PumaNpyStore,
        roi_indices: np.ndarray,
        tile_size: int = 512,
        tiles_per_roi: int = 8,
        seed: int = 2026,
        augment: bool = True,
        fixed_sigma: float = 2.5,
        offset_radius: float = 5.0,
    ) -> None:
        self.store = store
        self.roi_indices = np.asarray(roi_indices, dtype=np.int64)
        self.tile_size = int(tile_size)
        self.tiles_per_roi = int(tiles_per_roi)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.fixed_sigma = float(fixed_sigma)
        self.offset_radius = float(offset_radius)
        self._epoch_shared = torch.zeros((), dtype=torch.int64).share_memory_()
        self._sampling_weights: dict[tuple[int, str], np.ndarray] = {}

        for roi_value in self.roi_indices:
            roi = int(roi_value)
            nuclei = self.store.roi_centroids(roi)
            if not len(nuclei):
                continue
            dense = 1.0 / np.maximum(nuclei["nearest_neighbor_distance"].astype(np.float64), 1.0)
            small = 1.0 / np.maximum(nuclei["equivalent_diameter"].astype(np.float64), 1.0)
            self._sampling_weights[(roi, "dense")] = dense / dense.sum()
            self._sampling_weights[(roi, "small")] = small / small.sum()
            self._sampling_weights[(roi, "uniform")] = np.full(
                len(nuclei), 1.0 / len(nuclei), dtype=np.float64
            )

    def set_epoch(self, epoch: int) -> None:
        self._epoch_shared.fill_(int(epoch))

    def __len__(self) -> int:
        return len(self.roi_indices) * self.tiles_per_roi

    def _choose_origin(
        self,
        roi: int,
        image_shape: tuple[int, ...],
        nuclei: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        height, width = image_shape[:2]
        mode = rng.random()
        if len(nuclei) and mode < 0.70:
            if mode < 0.30:
                weights = self._sampling_weights[(roi, "dense")]
            elif mode < 0.50:
                weights = self._sampling_weights[(roi, "small")]
            else:
                weights = self._sampling_weights[(roi, "uniform")]
            nucleus = nuclei[int(rng.choice(len(nuclei), p=weights))]
            x0 = int(round(float(nucleus["x"]) - self.tile_size / 2 + rng.uniform(-0.2, 0.2) * self.tile_size))
            y0 = int(round(float(nucleus["y"]) - self.tile_size / 2 + rng.uniform(-0.2, 0.2) * self.tile_size))
        else:
            x0 = int(rng.integers(0, max(width - self.tile_size + 1, 1)))
            y0 = int(rng.integers(0, max(height - self.tile_size + 1, 1)))
        return (
            int(np.clip(x0, 0, max(width - self.tile_size, 0))),
            int(np.clip(y0, 0, max(height - self.tile_size, 0))),
        )

    def __getitem__(self, item: int) -> dict[str, Any]:
        roi = int(self.roi_indices[item // self.tiles_per_roi])
        epoch = int(self._epoch_shared.item())
        rng = np.random.default_rng(self.seed + epoch * 1_000_003 + item)
        roi_image = self.store.images[roi]
        nuclei = self.store.roi_centroids(roi)
        x0, y0 = self._choose_origin(roi, roi_image.shape, nuclei, rng)
        image = np.asarray(roi_image[y0 : y0 + self.tile_size, x0 : x0 + self.tile_size])
        inside = (
            (nuclei["x"] >= x0)
            & (nuclei["x"] < x0 + image.shape[1])
            & (nuclei["y"] >= y0)
            & (nuclei["y"] < y0 + image.shape[0])
        )
        selected = nuclei[inside]
        coordinates = np.empty((len(selected), 2), dtype=np.float32)
        if len(selected):
            coordinates[:, 0] = selected["x"] - x0
            coordinates[:, 1] = selected["y"] - y0

        stain_parameters = torch.empty(0, dtype=torch.float32)
        if self.augment:
            stain_parameters = torch.from_numpy(sample_stain_parameters(rng))
            image, coordinates = _apply_dihedral_xy(
                image,
                coordinates,
                image.shape[1],
                image.shape[0],
                int(rng.integers(0, 8)),
            )

        targets = build_dense_targets(
            coordinates,
            image.shape[0],
            image.shape[1],
            fixed_sigma=self.fixed_sigma,
            offset_radius=self.offset_radius,
        )
        return {
            "image": image_to_uint8_tensor(image),
            "targets": {key: torch.from_numpy(value) for key, value in targets.items()},
            "stain_parameters": stain_parameters,
        }


def stage1_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = tuple(batch[0]["targets"])
    return {
        "image": torch.stack([sample["image"] for sample in batch]),
        "targets": {
            key: torch.stack([sample["targets"][key] for sample in batch])
            for key in keys
        },
        "stain_parameters": (
            torch.stack([sample["stain_parameters"] for sample in batch])
            if batch[0]["stain_parameters"].numel()
            else torch.empty((len(batch), 0), dtype=torch.float32)
        ),
    }

def _reflect_indices(indices: np.ndarray, length: int) -> np.ndarray:
    """Map arbitrary integer indices with NumPy ``pad(mode='reflect')`` semantics."""
    if length <= 1:
        return np.zeros_like(indices, dtype=np.int64)
    period = 2 * int(length) - 2
    folded = np.mod(indices, period)
    return np.where(folded < length, folded, period - folded).astype(np.int64, copy=False)


def _extract_reflected_window(
    image: np.ndarray,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Extract only the requested reflected window, never pad/copy the complete ROI."""
    x1, y1 = int(x0) + int(width), int(y0) + int(height)
    if x0 >= 0 and y0 >= 0 and x1 <= image.shape[1] and y1 <= image.shape[0]:
        return np.array(
            image[y0:y1, x0:x1], dtype=np.uint8, copy=True, order="C"
        )
    x_indices = _reflect_indices(np.arange(x0, x1, dtype=np.int64), image.shape[1])
    y_indices = _reflect_indices(np.arange(y0, y1, dtype=np.int64), image.shape[0])
    return np.array(
        np.asarray(image)[y_indices[:, None], x_indices[None, :], :],
        dtype=np.uint8, copy=True, order="C",
    )


def extract_crop(image: np.ndarray, x: float, y: float, size: int, pad_mode: str = "reflect") -> np.ndarray:
    size = int(size); half = size // 2
    x0, y0 = int(math.floor(x)) - half, int(math.floor(y)) - half
    if pad_mode != "reflect":
        raise ValueError(f"Only reflect padding is supported, got {pad_mode!r}.")
    return _extract_reflected_window(image, x0, y0, size, size)





def build_stage2_geometry(
    *,
    image_shape: tuple[int, ...],
    x: float,
    y: float,
    confidence: float,
    nearest_distance: float,
    interface_key: str = "Fixed-MV",
) -> np.ndarray:
    """Build only metadata that is valid for A1_IFCRN_PP at deployment."""
    if interface_key != "Fixed-MV":
        raise ValueError(
            f"Stage 2 supports only the leakage-safe Fixed-MV interface, got {interface_key!r}."
        )
    nearest = (
        max(float(nearest_distance), 1e-3)
        if np.isfinite(nearest_distance)
        else float(max(image_shape[:2]))
    )
    local_density = 1.0 / (math.pi * nearest * nearest)
    border_distance = min(x, y, image_shape[1] - x, image_shape[0] - y)
    geometry = np.asarray([
        math.log1p(nearest),
        local_density,
        float(np.clip(confidence, 0.0, 1.0)),
        float(np.clip(border_distance / max(min(image_shape[:2]), 1), 0.0, 1.0)),
        PUMA_MICRONS_PER_PIXEL,
        float(np.clip(x / image_shape[1], 0.0, 1.0)),
        float(np.clip(y / image_shape[0], 0.0, 1.0)),
    ], dtype=np.float32)
    if geometry.shape != (STAGE2_GEOMETRY_DIM,) or not np.isfinite(geometry).all():
        raise ValueError(
            f"Invalid Stage-2 metadata: shape={geometry.shape}, values={geometry}"
        )
    return geometry




class Stage2CandidateDataset(Dataset):
    """
    Reads a structured candidate NPY generated by puma.pipeline.oof.

    Required fields are read from the OOF candidate contract. The revised A1 interface
    uses only roi_index, x, y, confidence, nearest_distance, class_id, and is_reject;
    no extent, orientation, uncertainty, or detector-embedding fields are required.
    """
    def __init__(
        self,
        store: PumaNpyStore,
        candidates: np.ndarray,
        views=("V2", "V3"),
        augment: bool = True,
        seed: int = 2026,
        interface_key: str = "Fixed-MV",
    ):
        self.store = store; self.candidates = candidates; self.views = tuple(views)
        unknown_views = sorted(set(self.views) - {"V2", "V3", "V4"})
        if unknown_views:
            raise ValueError(
                "A1_IFCRN_PP Stage 2 supports fixed views V2/V3/V4 only; "
                f"unsupported views: {unknown_views}."
            )
        if not self.views:
            raise ValueError("At least one fixed Stage-2 view is required.")
        if interface_key != "Fixed-MV":
            raise ValueError(
                f"Only interface_key='Fixed-MV' is supported, got {interface_key!r}."
            )
        self.augment = augment; self.seed = seed
        self._epoch_shared = torch.zeros((), dtype=torch.int64).share_memory_()
        self.interface_key = str(interface_key)
        self.view_sizes = {"V2": 64, "V3": 128, "V4": 256}
        # Cache repeated native crops in worker memory; stain jitter stays epoch-specific.
        try:
            cache_mb = max(0, int(os.environ.get("PUMA_STAGE2_CROP_CACHE_MB", "128")))
        except ValueError:
            cache_mb = 128
        self._crop_cache_limit = cache_mb * 1024 * 1024
        self._crop_cache_bytes = 0
        self._crop_cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()

    def set_epoch(self, epoch: int): self._epoch_shared.fill_(int(epoch))
    def __len__(self): return len(self.candidates)

    def _cached_crops(self, idx: int) -> dict[str, torch.Tensor] | None:
        if self._crop_cache_limit <= 0:
            return None
        cached = self._crop_cache.pop(int(idx), None)
        if cached is not None:
            self._crop_cache[int(idx)] = cached
        return cached

    def _store_crops(
        self, idx: int, crops: dict[str, torch.Tensor]
    ) -> None:
        if self._crop_cache_limit <= 0:
            return
        size_bytes = sum(tensor.numel() * tensor.element_size() for tensor in crops.values())
        if size_bytes > self._crop_cache_limit:
            return
        while self._crop_cache and self._crop_cache_bytes + size_bytes > self._crop_cache_limit:
            _, evicted = self._crop_cache.popitem(last=False)
            self._crop_cache_bytes -= sum(
                tensor.numel() * tensor.element_size() for tensor in evicted.values()
            )
        # Cached crops are immutable uint8 tensors; augmentation runs after collation.
        self._crop_cache[int(idx)] = crops
        self._crop_cache_bytes += size_bytes

    def __getitem__(self, idx: int) -> dict[str, Any]:
        c = self.candidates[idx]
        epoch = int(self._epoch_shared.item())
        rng = np.random.default_rng(self.seed + epoch * 1_000_003 + idx)
        x, y = float(c["x"]), float(c["y"])
        cached = self._cached_crops(idx)
        if cached is None:
            # Read the ROI by memmap and copy only the requested crop.
            image = np.asarray(self.store.images[int(c["roi_index"])])
            crops: dict[str, torch.Tensor] = {}
            # Extract the largest view once and center-slice smaller views.
            axis_base_size = max(self.view_sizes[view] for view in self.views)
            axis_tensor = image_to_uint8_tensor(
                extract_crop(image, x, y, axis_base_size)
            )
            for view in self.views:
                size = self.view_sizes[view]
                start = (axis_base_size - size) // 2
                crops[view] = axis_tensor[
                    :, start:start + size, start:start + size
                ]
                # Native uint8 crops are resized, stained and normalized after transfer.
            self._store_crops(idx, crops)
            image_shape = image.shape
        else:
            crops = cached
            image_shape = self.store.images.shape[1:]

        stain_parameters: dict[str, torch.Tensor] = {}
        if self.augment:
            # Apply one stain perturbation consistently across all views.
            shared_stain = torch.from_numpy(sample_stain_parameters(rng))
            stain_parameters = {view: shared_stain for view in self.views}
        geometry = build_stage2_geometry(
            image_shape=image_shape,
            x=x,
            y=y,
            confidence=float(c["confidence"]),
            nearest_distance=float(c["nearest_distance"]),
            interface_key=self.interface_key,
        )
        label = int(c["class_id"])
        source_index = int(c["oof_row_id"]) if "oof_row_id" in c.dtype.names else idx
        return {
            "views": crops, "geometry": torch.from_numpy(geometry),
            "label": label,
            "candidate_index": idx,
            "source_index": source_index,
            "stain_parameters": stain_parameters,
        }


def pack_stage2_view_tensors(
    tensors: list[torch.Tensor],
) -> tuple[dict[str, torch.Tensor], ...]:
    """Pack native crops into shape-homogeneous tensors while preserving order."""
    groups: dict[tuple[int, int], list[tuple[int, torch.Tensor]]] = {}
    for index, tensor in enumerate(tensors):
        shape = (int(tensor.shape[-2]), int(tensor.shape[-1]))
        groups.setdefault(shape, []).append((index, tensor))
    packed: list[dict[str, torch.Tensor]] = []
    for items in groups.values():
        packed.append({
            "indices": torch.tensor([index for index, _ in items], dtype=torch.long),
            "images": torch.stack([tensor for _, tensor in items]),
        })
    return tuple(packed)


def prepare_stage2_view_batch(
    packed: tuple[dict[str, torch.Tensor], ...] | list[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    output_size: int = 224,
    stain_parameters: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transfer compact native crops, resize, and normalize on the accelerator."""
    if not packed:
        return torch.empty((0, 3, output_size, output_size), device=device)
    device_stain = None
    if stain_parameters is not None and stain_parameters.numel():
        device_stain = stain_parameters.to(
            device=device, dtype=torch.float32, non_blocking=True
        )

    final_dtype = torch.float32
    if device.type == "cuda":
        try:
            autocast_enabled = bool(torch.is_autocast_enabled("cuda"))
        except TypeError:  # Compatibility with older Colab PyTorch builds.
            autocast_enabled = bool(torch.is_autocast_enabled())
        if autocast_enabled:
            if hasattr(torch, "get_autocast_dtype"):
                final_dtype = torch.get_autocast_dtype("cuda")
            else:  # pragma: no cover - retained for older supported PyTorch.
                final_dtype = torch.get_autocast_gpu_dtype()

    def finalize(images: torch.Tensor) -> torch.Tensor:
        # Autocast would create this lower-precision input at the first PFM operation
        # anyway. Casting here avoids keeping both float32 and AMP copies in VRAM.
        return images if images.dtype == final_dtype else images.to(dtype=final_dtype)

    def prepare(group: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        images = group["images"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
            memory_format=torch.channels_last if device.type == "cuda" else torch.preserve_format,
        )
        indices = group["indices"].to(device, non_blocking=True)
        group_stain = (
            None if device_stain is None else device_stain.index_select(0, indices)
        )
        apply_stain_parameters_(images, group_stain)
        if images.shape[-2:] != (output_size, output_size):
            images = torch.nn.functional.interpolate(
                images, size=(output_size, output_size),
                mode="bicubic", align_corners=False,
            )
        images.mul_(1.0 / 255.0)
        images[:, 0].sub_(0.485).div_(0.229)
        images[:, 1].sub_(0.456).div_(0.224)
        images[:, 2].sub_(0.406).div_(0.225)
        return images, indices

    if len(packed) == 1:
        images, _ = prepare(packed[0])
        return finalize(images)

    batch_size = sum(int(group["indices"].numel()) for group in packed)
    output = torch.empty(
        (batch_size, 3, output_size, output_size),
        device=device,
        dtype=final_dtype,
        memory_format=torch.channels_last if device.type == "cuda" else torch.contiguous_format,
    )
    for group in packed:
        images, indices = prepare(group)
        if images.dtype != final_dtype:
            images = images.to(dtype=final_dtype)
        output.index_copy_(0, indices, images)
        del images, indices
    return output


def stage2_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    view_keys = batch[0]["views"].keys()
    return {
        "views": {
            key: pack_stage2_view_tensors([sample["views"][key] for sample in batch])
            for key in view_keys
        },
        "geometry": torch.stack([b["geometry"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "candidate_index": torch.tensor([b["candidate_index"] for b in batch], dtype=torch.long),
        "source_index": torch.tensor([b["source_index"] for b in batch], dtype=torch.long),
        "stain_parameters": ({
            key: torch.stack([sample["stain_parameters"][key] for sample in batch])
            for key in view_keys
        } if batch[0]["stain_parameters"] else {}),
    }
