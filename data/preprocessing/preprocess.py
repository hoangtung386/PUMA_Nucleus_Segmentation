"""Rare-class-focused preprocessing for PUMA Track 2."""

import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import tifffile as tiff
import torch
from tqdm import tqdm

from configs import PATHS, PREPROCESS_DEFAULT_CONFIG
from data.constants import (
    PUMA_NUCLEI_NAME_TO_ID,
    PUMA_TISSUE_NAME_TO_ID,
    RARE_NUCLEI_IDS,
    RARE_NUCLEI_SAMPLE_BONUS,
    RARE_TISSUE_IDS_PUMA,
    RARE_TISSUE_SAMPLE_BONUS,
)
from data.preprocessing.flow_generator import CellposeFlowGenerator, compute_hv_map
from data.preprocessing.geojson_parser import find_annotation_file, parse_geojson_masks
from training.logging_utils import logger

cfg = PREPROCESS_DEFAULT_CONFIG


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


def resize_all(image, tissue, nuclei, inst, size: int):
    if image.shape[0] == size and image.shape[1] == size:
        return image, tissue, nuclei, inst
    image_r = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    tissue_r = cv2.resize(tissue, (size, size), interpolation=cv2.INTER_NEAREST)
    nuclei_r = cv2.resize(nuclei, (size, size), interpolation=cv2.INTER_NEAREST)
    inst_r = cv2.resize(inst.astype(np.int32), (size, size), interpolation=cv2.INTER_NEAREST)
    return image_r, tissue_r.astype(np.uint8), nuclei_r.astype(np.uint8), inst_r.astype(np.int32)


def translate_to_center(image, tissue, nuclei, inst, center_xy, out_size, jitter_px):
    h, w = image.shape[:2]
    cx, cy = center_xy
    import random
    jx = random.randint(-jitter_px, jitter_px) if jitter_px > 0 else 0
    jy = random.randint(-jitter_px, jitter_px) if jitter_px > 0 else 0
    target_x = out_size / 2.0 + jx
    target_y = out_size / 2.0 + jy
    dx = target_x - cx
    dy = target_y - cy
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)

    image_t = cv2.warpAffine(
        image, matrix, (out_size, out_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )
    tissue_t = cv2.warpAffine(
        tissue, matrix, (out_size, out_size),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    nuclei_t = cv2.warpAffine(
        nuclei, matrix, (out_size, out_size),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    inst_t = cv2.warpAffine(
        inst.astype(np.float32), matrix, (out_size, out_size),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return image_t, tissue_t.astype(np.uint8), nuclei_t.astype(np.uint8), np.rint(inst_t).astype(np.int32)


def component_centers(mask: np.ndarray, class_ids: Iterable[int], max_per_class: int = 2) -> List[Tuple[int, Tuple[float, float], int]]:
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
    tissue_present = sorted(int(x) for x in np.unique(tissue) if int(x) in RARE_TISSUE_IDS_PUMA)
    nuclei_present = sorted(int(x) for x in np.unique(nuclei) if int(x) in RARE_NUCLEI_IDS)

    weight = 1.0
    for cls in tissue_present:
        weight += RARE_TISSUE_SAMPLE_BONUS.get(cls, 0.0)
    for cls in nuclei_present:
        weight += RARE_NUCLEI_SAMPLE_BONUS.get(cls, 0.0)
    if is_rare_augmented:
        weight *= 1.5
    return float(weight), tissue_present, nuclei_present


def save_processed_sample(base_name, image, tissue, nuclei, inst, flow_generator, metadata, is_rare_augmented, source_name):
    out_dir = PATHS.data_dir
    paths = {
        "image": out_dir / "images" / f"{base_name}.npy",
        "tissue": out_dir / "tissue_sem" / f"{base_name}.npy",
        "nuclei": out_dir / "nuclei_nc" / f"{base_name}.npy",
        "hv": out_dir / "nuclei_hv" / f"{base_name}.npy",
        "cp": out_dir / "cellpose_flows" / f"{base_name}.npy",
    }

    weight, rare_tissue, rare_nuclei = sample_weight_from_masks(tissue, nuclei, is_rare_augmented)

    if not (cfg.skip_existing and all(p.exists() for p in paths.values())):
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


def process_one_roi(img_path: Path, flow_generator: CellposeFlowGenerator, metadata: List[dict]):

    raw_dir = PATHS.raw_dir
    tissue_geojson_dir = raw_dir / "01_training_dataset_geojson_tissue"
    nuclei_geojson_dir = raw_dir / "01_training_dataset_geojson_nuclei"

    base = img_path.stem
    image = read_rgb_tif(img_path)
    h, w = image.shape[:2]

    tissue_geojson = find_annotation_file(tissue_geojson_dir, base, "tissue")
    nuclei_geojson = find_annotation_file(nuclei_geojson_dir, base, "nuclei")

    tissue, _ = parse_geojson_masks(tissue_geojson, PUMA_TISSUE_NAME_TO_ID, (h, w), is_instance=False)
    nuclei, inst = parse_geojson_masks(nuclei_geojson, PUMA_NUCLEI_NAME_TO_ID, (h, w), is_instance=True)
    assert inst is not None

    image_1024, tissue_1024, nuclei_1024, inst_1024 = resize_all(image, tissue, nuclei, inst, cfg.image_size)

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

    if not cfg.make_rare_centered_crops:
        return

    centers: List[Tuple[str, int, Tuple[float, float], int]] = []
    for cls, center, area in component_centers(tissue_1024, RARE_TISSUE_IDS_PUMA, max_per_class=2):
        centers.append(("tissue", cls, center, area))
    for cls, center, area in component_centers(nuclei_1024, RARE_NUCLEI_IDS, max_per_class=3):
        centers.append(("nuclei", cls, center, area))

    if not centers:
        return

    def priority(item):
        kind, cls, _, area = item
        bonus = RARE_NUCLEI_SAMPLE_BONUS.get(cls, 0.0) if kind == "nuclei" else RARE_TISSUE_SAMPLE_BONUS.get(cls, 0.0)
        return bonus * 100000.0 + area

    centers.sort(key=priority, reverse=True)
    centers = centers[:cfg.max_rare_crops_per_image]

    for j, (kind, cls, center, _) in enumerate(centers):
        aug_name = f"{base}__rare{j:02d}_{kind}{cls}"
        im_t, tissue_t, nuclei_t, inst_t = translate_to_center(
            image_1024, tissue_1024, nuclei_1024, inst_1024,
            center_xy=center,
            out_size=cfg.image_size,
            jitter_px=cfg.rare_crop_jitter_px,
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
    import random
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    raw_dir = PATHS.raw_dir
    image_dir = raw_dir / "01_training_dataset_tif_ROIs"

    out_dir = PATHS.data_dir
    for subdir in ["images", "tissue_sem", "nuclei_nc", "nuclei_hv", "cellpose_flows"]:
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    img_files = sorted(image_dir.glob("*.tif")) + sorted(image_dir.glob("*.tiff"))
    if not img_files:
        raise FileNotFoundError(f"No TIFF files found in {image_dir}")

    logger.info("Root: %s", PATHS.root)
    logger.info("Raw: %s", raw_dir)
    logger.info("Output: %s", out_dir)
    logger.info("Images: %d", len(img_files))
    logger.info("Rare crops: enabled=%s max_per_image=%d", cfg.make_rare_centered_crops, cfg.max_rare_crops_per_image)
    logger.info("Cellpose: generate_cellpose_flows=%s", cfg.generate_cellpose_flows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_generator = CellposeFlowGenerator(cfg.generate_cellpose_flows, cfg.cellpose_model_type, device=device)
    metadata: List[dict] = []
    meta_lock = Lock()
    num_workers = min(os.cpu_count() or 4, 8)

    def _process_one(img_path):
        local_meta: List[dict] = []
        process_one_roi(img_path, flow_generator, local_meta)
        with meta_lock:
            metadata.extend(local_meta)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(_process_one, p) for p in img_files]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Preprocess rare-focused"):
            f.result()

    metadata_path = out_dir / "sample_metadata.json"
    if cfg.rebuild_metadata or not metadata_path.exists():
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    n_rare_aug = sum(1 for m in metadata if m["is_rare_augmented"])
    logger.info("Done")
    logger.info("Processed samples in metadata: %d", len(metadata))
    logger.info("Rare augmented samples: %d", n_rare_aug)
    logger.info("Metadata: %s", metadata_path)
    logger.info("Stored PUMA tissue IDs in tissue_sem; dataset converts background 0 to ignore 255.")
