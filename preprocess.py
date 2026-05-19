"""
Rare-class-focused preprocessing for PUMA Track 2.

Click-to-run. No argparse.
All paths are relative to root = Path.cwd().

Expected raw folders:
    Dataset/01_training_dataset_tif_ROIs/*.tif
    Dataset/01_training_dataset_geojson_tissue/*_tissue.geojson
    Dataset/01_training_dataset_geojson_nuclei/*_nuclei.geojson

Output folders:
    dataset_processed/images/*.npy
    dataset_processed/tissue_sem/*.npy       PUMA tissue IDs: 0 background, 1..5 tissue
    dataset_processed/nuclei_nc/*.npy        nuclei class IDs: 0..9, 255 non-nucleus
    dataset_processed/nuclei_hv/*.npy        HoVer map [2,H,W]
    dataset_processed/cellpose_flows/*.npy   Cellpose flow [2,H,W]
    dataset_processed/sample_metadata.json   rare-class flags and sampling weights

Important design:
    The model still uses only 5 tissue classes. Background tissue ID 0 is stored here,
    but the dataset converts it to 255 ignore during training.

Rare-class strategy:
    1. Always save the original 1024 sample.
    2. If a raw ROI contains rare tissue/nuclei, create extra rare-centered translated
       crops with suffix __rareXX. This increases the number of useful rare samples.
    3. Also store per-sample rare metadata so train_stage1.py/train_stage2.py can use
       a WeightedRandomSampler.
"""

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm

# ============================================================
# Click-to-run configuration
# ============================================================

root = Path.cwd()
raw_dir = root / "Dataset"
out_dir = root / "dataset_processed"

image_dir = raw_dir / "01_training_dataset_tif_ROIs"
tissue_geojson_dir = raw_dir / "01_training_dataset_geojson_tissue"
nuclei_geojson_dir = raw_dir / "01_training_dataset_geojson_nuclei"

image_size = 1024
crop_size = 1024

# Generate real Cellpose flow files during preprocessing.
# Set to False only if you want zero Cellpose flows during training.
generate_cellpose_flows = True
cellpose_model_type = "cyto3"
cellpose_batch_size = 1

# Rare crop generation.
make_rare_centered_crops = True
max_rare_crops_per_image = 3
rare_crop_jitter_px = 96
random_seed = 42

# If True, existing .npy files are kept. Set True for fast resume.
skip_existing = True

# If True, wipe sample_metadata.json and rebuild it from this run.
# The .npy files themselves are not deleted.
rebuild_metadata = True

# PUMA tissue output IDs.
PUMA_TISSUE_NAME_TO_ID = {
    "tissue_stroma": 1,
    "tissue_blood_vessel": 2,
    "tissue_tumor": 3,
    "tissue_epidermis": 4,
    "tissue_necrosis": 5,
}

PUMA_NUCLEI_NAME_TO_ID = {
    "nuclei_tumor": 0,
    "nuclei_lymphocyte": 1,
    "nuclei_plasma_cell": 2,
    "nuclei_histiocyte": 3,
    "nuclei_melanophage": 4,
    "nuclei_neutrophil": 5,
    "nuclei_stroma": 6,
    "nuclei_epithelium": 7,
    "nuclei_endothelium": 8,
    "nuclei_apoptosis": 9,
}

# Rare classes to focus on. These are based on your Stage 1 logs.
RARE_TISSUE_IDS = {
    2: "tissue_blood_vessel",
    4: "tissue_epidermis",
    5: "tissue_necrosis",
}
RARE_NUCLEI_IDS = {
    2: "nuclei_plasma_cell",
    4: "nuclei_melanophage",
    5: "nuclei_neutrophil",
    8: "nuclei_endothelium",
    9: "nuclei_apoptosis",
}

# Higher = stronger oversampling in training.
RARE_TISSUE_SAMPLE_BONUS = {
    2: 3.0,   # blood vessel
    4: 2.0,   # epidermis
    5: 6.0,   # necrosis
}
RARE_NUCLEI_SAMPLE_BONUS = {
    2: 6.0,   # plasma cell
    4: 4.0,   # melanophage
    5: 8.0,   # neutrophil
    8: 5.0,   # endothelium
    9: 8.0,   # apoptosis
}


# ============================================================
# GeoJSON parsing
# ============================================================

def _feature_class_name(feature: dict) -> Optional[str]:
    props = feature.get("properties", {}) or {}
    cls = props.get("classification", {}) or {}
    name = cls.get("name")
    if name is None:
        name = props.get("name") or props.get("class") or props.get("label")
    return name


def _polygon_arrays_from_geometry(geometry: Optional[dict]) -> List[np.ndarray]:
    if not geometry:
        return []
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    polys: List[np.ndarray] = []

    if gtype == "Polygon":
        if coords:
            arr = np.asarray(coords[0], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] >= 3:
                polys.append(arr)
    elif gtype == "MultiPolygon":
        for poly in coords:
            if not poly:
                continue
            arr = np.asarray(poly[0], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] >= 3:
                polys.append(arr)
    return polys


def _polygon_arrays_from_multiple_polygons(data: dict) -> Iterable[Tuple[str, np.ndarray]]:
    """Support Grand-Challenge-style multiple-polygon JSON if present."""
    for poly in data.get("polygons", []):
        name = poly.get("name") or poly.get("classification")
        pts = poly.get("path_points") or poly.get("coordinates") or poly.get("points")
        if name is None or pts is None:
            continue
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 3:
            yield name, arr


def parse_geojson_masks(
    geojson_path: Path,
    class_dict: Dict[str, int],
    shape_hw: Tuple[int, int],
    is_instance: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Rasterize polygons at the raw image shape."""
    h, w = shape_hw
    background_value = 255 if is_instance else 0
    sem_mask = np.full((h, w), background_value, dtype=np.uint8)
    inst_mask = np.zeros((h, w), dtype=np.int32) if is_instance else None

    if not geojson_path.exists():
        return sem_mask, inst_mask

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    inst_id = 1

    if "features" in data:
        for feature in data.get("features", []):
            class_name = _feature_class_name(feature)
            if class_name not in class_dict:
                continue
            class_id = int(class_dict[class_name])
            polygons = _polygon_arrays_from_geometry(feature.get("geometry"))
            for poly in polygons:
                poly_i = np.round(poly).astype(np.int32)
                cv2.fillPoly(sem_mask, [poly_i], color=class_id)
                if is_instance and inst_mask is not None:
                    cv2.fillPoly(inst_mask, [poly_i], color=inst_id)
                    inst_id += 1
    else:
        for class_name, poly in _polygon_arrays_from_multiple_polygons(data):
            if class_name not in class_dict:
                continue
            class_id = int(class_dict[class_name])
            poly_i = np.round(poly).astype(np.int32)
            cv2.fillPoly(sem_mask, [poly_i], color=class_id)
            if is_instance and inst_mask is not None:
                cv2.fillPoly(inst_mask, [poly_i], color=inst_id)
                inst_id += 1

    return sem_mask, inst_mask


# ============================================================
# Mask/flow helpers
# ============================================================

def compute_hv_map(inst_mask: np.ndarray) -> np.ndarray:
    """HoVer-Net style horizontal/vertical maps, shape [2,H,W]."""
    h_map = np.zeros_like(inst_mask, dtype=np.float32)
    v_map = np.zeros_like(inst_mask, dtype=np.float32)

    for inst_id in np.unique(inst_mask):
        if inst_id == 0:
            continue
        ys, xs = np.where(inst_mask == inst_id)
        if len(xs) == 0:
            continue
        x_center = float(xs.mean())
        y_center = float(ys.mean())
        x_radius = max((float(xs.max()) - float(xs.min())) / 2.0, 1.0)
        y_radius = max((float(ys.max()) - float(ys.min())) / 2.0, 1.0)
        h_map[ys, xs] = np.clip((xs - x_center) / (x_radius + 1e-8), -1.0, 1.0)
        v_map[ys, xs] = np.clip((ys - y_center) / (y_radius + 1e-8), -1.0, 1.0)

    return np.stack([h_map, v_map], axis=0).astype(np.float16)


def read_rgb_tif(path: Path) -> np.ndarray:
    image = tiff.imread(str(path))
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[0] in [3, 4] and image.shape[-1] not in [3, 4]:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        image = image - image.min()
        image = image / (image.max() + 1e-6)
        image = (image * 255.0).clip(0, 255).astype(np.uint8)
    return image


def resize_all(
    image: np.ndarray,
    tissue: np.ndarray,
    nuclei: np.ndarray,
    inst: np.ndarray,
    size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if image.shape[0] == size and image.shape[1] == size:
        return image, tissue, nuclei, inst
    image_r = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    tissue_r = cv2.resize(tissue, (size, size), interpolation=cv2.INTER_NEAREST)
    nuclei_r = cv2.resize(nuclei, (size, size), interpolation=cv2.INTER_NEAREST)
    inst_r = cv2.resize(inst.astype(np.int32), (size, size), interpolation=cv2.INTER_NEAREST)
    return image_r, tissue_r.astype(np.uint8), nuclei_r.astype(np.uint8), inst_r.astype(np.int32)


def translate_to_center(
    image: np.ndarray,
    tissue: np.ndarray,
    nuclei: np.ndarray,
    inst: np.ndarray,
    center_xy: Tuple[float, float],
    out_size: int,
    jitter_px: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Translate a rare object toward the crop center while keeping 1024x1024 output."""
    h, w = image.shape[:2]
    cx, cy = center_xy
    jx = random.randint(-jitter_px, jitter_px) if jitter_px > 0 else 0
    jy = random.randint(-jitter_px, jitter_px) if jitter_px > 0 else 0
    target_x = out_size / 2.0 + jx
    target_y = out_size / 2.0 + jy
    dx = target_x - cx
    dy = target_y - cy
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)

    image_t = cv2.warpAffine(
        image,
        matrix,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    tissue_t = cv2.warpAffine(
        tissue,
        matrix,
        (out_size, out_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    nuclei_t = cv2.warpAffine(
        nuclei,
        matrix,
        (out_size, out_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    # OpenCV warpAffine is not reliable for int32 on all builds.
    # Use float32 with nearest interpolation, then cast back to int32.
    inst_t = cv2.warpAffine(
        inst.astype(np.float32),
        matrix,
        (out_size, out_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return image_t, tissue_t.astype(np.uint8), nuclei_t.astype(np.uint8), np.rint(inst_t).astype(np.int32)


def component_centers(mask: np.ndarray, class_ids: Iterable[int], max_per_class: int = 2) -> List[Tuple[int, Tuple[float, float], int]]:
    """Return class_id, center_xy, area for connected components of selected classes."""
    out: List[Tuple[int, Tuple[float, float], int]] = []
    for cls in class_ids:
        binary = (mask == cls).astype(np.uint8)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        comps = []
        for comp_id in range(1, n):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            cx, cy = centroids[comp_id]
            comps.append((area, (float(cx), float(cy))))
        comps.sort(reverse=True, key=lambda x: x[0])
        for area, center in comps[:max_per_class]:
            out.append((int(cls), center, int(area)))
    return out


def sample_weight_from_masks(tissue: np.ndarray, nuclei: np.ndarray, is_rare_augmented: bool) -> Tuple[float, List[int], List[int]]:
    tissue_present = sorted(int(x) for x in np.unique(tissue) if int(x) in RARE_TISSUE_IDS)
    nuclei_present = sorted(int(x) for x in np.unique(nuclei) if int(x) in RARE_NUCLEI_IDS)

    weight = 1.0
    for cls in tissue_present:
        weight += RARE_TISSUE_SAMPLE_BONUS.get(cls, 0.0)
    for cls in nuclei_present:
        weight += RARE_NUCLEI_SAMPLE_BONUS.get(cls, 0.0)
    if is_rare_augmented:
        weight *= 1.5
    return float(weight), tissue_present, nuclei_present


class CellposeFlowGenerator:
    def __init__(self, enabled: bool, model_type: str):
        self.enabled = bool(enabled)
        self.model = None
        if not self.enabled:
            return
        try:
            import torch
            from cellpose import models
            use_gpu = torch.cuda.is_available()
            self.model = models.CellposeModel(gpu=use_gpu, model_type=model_type)
            print(f"[Cellpose] Loaded model_type={model_type} gpu={use_gpu}")
        except Exception as exc:
            print(f"[Cellpose][WARN] Could not load Cellpose. Zero flows will be stored. Error: {exc}")
            self.model = None

    def make_flow(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        if self.model is None:
            return np.zeros((2, h, w), dtype=np.float16)
        try:
            result = self.model.eval(
                image_rgb,
                diameter=None,
                channels=[0, 0],
                flow_threshold=None,
                cellprob_threshold=0.0,
            )
            if len(result) == 4:
                _, flows, _, _ = result
            else:
                _, flows, _ = result
            flow = flows[1] if isinstance(flows, list) and len(flows) > 1 else flows
            flow = np.asarray(flow)
            if flow.ndim == 3 and flow.shape[0] >= 2:
                flow = flow[:2]
            elif flow.ndim == 3 and flow.shape[-1] >= 2:
                flow = flow[..., :2].transpose(2, 0, 1)
            else:
                raise RuntimeError(f"Unexpected Cellpose flow shape: {flow.shape}")
            if flow.shape[1] != h or flow.shape[2] != w:
                flow = np.stack([
                    cv2.resize(flow[0], (w, h), interpolation=cv2.INTER_LINEAR),
                    cv2.resize(flow[1], (w, h), interpolation=cv2.INTER_LINEAR),
                ], axis=0)
            return flow.astype(np.float16)
        except Exception as exc:
            print(f"[Cellpose][WARN] Flow generation failed; storing zero flow. Error: {exc}")
            return np.zeros((2, h, w), dtype=np.float16)


def save_processed_sample(
    base_name: str,
    image: np.ndarray,
    tissue: np.ndarray,
    nuclei: np.ndarray,
    inst: np.ndarray,
    flow_generator: CellposeFlowGenerator,
    metadata: List[dict],
    is_rare_augmented: bool,
    source_name: str,
) -> None:
    paths = {
        "image": out_dir / "images" / f"{base_name}.npy",
        "tissue": out_dir / "tissue_sem" / f"{base_name}.npy",
        "nuclei": out_dir / "nuclei_nc" / f"{base_name}.npy",
        "hv": out_dir / "nuclei_hv" / f"{base_name}.npy",
        "cp": out_dir / "cellpose_flows" / f"{base_name}.npy",
    }

    weight, rare_tissue, rare_nuclei = sample_weight_from_masks(tissue, nuclei, is_rare_augmented)

    if not (skip_existing and all(p.exists() for p in paths.values())):
        hv = compute_hv_map(inst)
        cp_flow = flow_generator.make_flow(image)
        np.save(paths["image"], image.astype(np.uint8))
        np.save(paths["tissue"], tissue.astype(np.uint8))
        np.save(paths["nuclei"], nuclei.astype(np.uint8))
        np.save(paths["hv"], hv)
        np.save(paths["cp"], cp_flow)

    metadata.append({
        "base_name": base_name,
        "source_name": source_name,
        "is_rare_augmented": bool(is_rare_augmented),
        "rare_tissue_ids": rare_tissue,
        "rare_nuclei_ids": rare_nuclei,
        "sample_weight": weight,
    })


def find_annotation_file(folder: Path, base: str, suffix: str) -> Path:
    """Find annotation robustly across common PUMA/QuPath naming variants."""
    candidates = [
        folder / f"{base}_{suffix}.geojson",
        folder / f"{base}.geojson",
        folder / f"{base}-{suffix}.geojson",
        folder / f"{base} {suffix}.geojson",
    ]
    for path in candidates:
        if path.exists():
            return path
    hits = sorted(folder.glob(f"*{base}*{suffix}*.geojson"))
    if hits:
        return hits[0]
    hits = sorted(folder.glob(f"*{base}*.geojson"))
    if hits:
        return hits[0]
    return candidates[0]


def process_one_roi(img_path: Path, flow_generator: CellposeFlowGenerator, metadata: List[dict]) -> None:
    base = img_path.stem
    image = read_rgb_tif(img_path)
    h, w = image.shape[:2]

    tissue_geojson = find_annotation_file(tissue_geojson_dir, base, "tissue")
    nuclei_geojson = find_annotation_file(nuclei_geojson_dir, base, "nuclei")

    tissue, _ = parse_geojson_masks(tissue_geojson, PUMA_TISSUE_NAME_TO_ID, (h, w), is_instance=False)
    nuclei, inst = parse_geojson_masks(nuclei_geojson, PUMA_NUCLEI_NAME_TO_ID, (h, w), is_instance=True)
    assert inst is not None

    image_1024, tissue_1024, nuclei_1024, inst_1024 = resize_all(image, tissue, nuclei, inst, image_size)

    save_processed_sample(
        base_name=base,
        image=image_1024,
        tissue=tissue_1024,
        nuclei=nuclei_1024,
        inst=inst_1024,
        flow_generator=flow_generator,
        metadata=metadata,
        is_rare_augmented=False,
        source_name=base,
    )

    if not make_rare_centered_crops:
        return

    centers: List[Tuple[str, int, Tuple[float, float], int]] = []
    for cls, center, area in component_centers(tissue_1024, RARE_TISSUE_IDS.keys(), max_per_class=2):
        centers.append(("tissue", cls, center, area))
    for cls, center, area in component_centers(nuclei_1024, RARE_NUCLEI_IDS.keys(), max_per_class=3):
        centers.append(("nuclei", cls, center, area))

    if not centers:
        return

    # Prioritize the rarest classes and larger components.
    def priority(item):
        kind, cls, _, area = item
        bonus = RARE_NUCLEI_SAMPLE_BONUS.get(cls, 0.0) if kind == "nuclei" else RARE_TISSUE_SAMPLE_BONUS.get(cls, 0.0)
        return bonus * 100000.0 + area

    centers.sort(key=priority, reverse=True)
    centers = centers[:max_rare_crops_per_image]

    for j, (kind, cls, center, _) in enumerate(centers):
        aug_name = f"{base}__rare{j:02d}_{kind}{cls}"
        im_t, tissue_t, nuclei_t, inst_t = translate_to_center(
            image_1024,
            tissue_1024,
            nuclei_1024,
            inst_1024,
            center_xy=center,
            out_size=image_size,
            jitter_px=rare_crop_jitter_px,
        )
        save_processed_sample(
            base_name=aug_name,
            image=im_t,
            tissue=tissue_t,
            nuclei=nuclei_t,
            inst=inst_t,
            flow_generator=flow_generator,
            metadata=metadata,
            is_rare_augmented=True,
            source_name=base,
        )


def main() -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)

    for subdir in ["images", "tissue_sem", "nuclei_nc", "nuclei_hv", "cellpose_flows"]:
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    img_files = sorted(image_dir.glob("*.tif")) + sorted(image_dir.glob("*.tiff"))
    if not img_files:
        raise FileNotFoundError(f"No TIFF files found in {image_dir}")

    print(f"[Root] {root}")
    print(f"[Raw] {raw_dir}")
    print(f"[Output] {out_dir}")
    print(f"[Images] {len(img_files)}")
    print(f"[Rare crops] enabled={make_rare_centered_crops}, max_per_image={max_rare_crops_per_image}")
    print(f"[Cellpose] generate_cellpose_flows={generate_cellpose_flows}")

    flow_generator = CellposeFlowGenerator(generate_cellpose_flows, cellpose_model_type)
    metadata: List[dict] = []

    for img_path in tqdm(img_files, desc="Preprocess rare-focused"):
        process_one_roi(img_path, flow_generator, metadata)

    metadata_path = out_dir / "sample_metadata.json"
    if rebuild_metadata or not metadata_path.exists():
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    n_rare_aug = sum(1 for m in metadata if m["is_rare_augmented"])
    print("\n[Done]")
    print(f"Processed samples in metadata: {len(metadata)}")
    print(f"Rare augmented samples: {n_rare_aug}")
    print(f"Metadata: {metadata_path}")
    print("Stored PUMA tissue IDs in tissue_sem; dataset converts background 0 to ignore 255.")


if __name__ == "__main__":
    main()
