"""
Parse PUMA GeoJSON annotation files into instance masks and class maps.

PUMA annotations are stored as GeoJSON FeatureCollections where each Feature
represents a nucleus polygon with a ``classification.name`` property.

Track 1  — 3 classes : tumor | lymphocyte | other
Track 2  — 10 classes: tumor | lymphocyte | plasma cell | histiocyte |
                        melanophage | neutrophil | stroma | endothelium |
                        epithelium | apoptosis
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from skimage.draw import polygon as sk_polygon

logger = logging.getLogger(__name__)

# ─── Class maps ──────────────────────────────────────────────────────────────

CLASS_NAMES_T1: List[str] = ["background", "tumor", "lymphocyte", "other"]
CLASS_NAMES_T2: List[str] = [
    "background",
    "tumor",
    "lymphocyte",
    "plasma cell",
    "histiocyte",
    "melanophage",
    "neutrophil",
    "stroma",
    "endothelium",
    "epithelium",
    "apoptosis",
]

_CLASS_MAP_T1: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES_T1)}
_CLASS_MAP_T2: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES_T2)}

NUM_CLASSES: Dict[int, int] = {1: len(CLASS_NAMES_T1), 2: len(CLASS_NAMES_T2)}

_ALIAS_MAP: Dict[str, str] = {
    "til": "lymphocyte",
    "tils": "lymphocyte",
    "lymphocytes": "lymphocyte",
    "plasma": "plasma cell",
    "stromal": "stroma",
    "endothelial": "endothelium",
    "apoptotic": "apoptosis",
    "necrosis": "necrotic",
}


def get_class_map(track: int = 1) -> Dict[str, int]:
    """Return the {class_name: class_id} dict for the requested track."""
    if track == 1:
        return _CLASS_MAP_T1
    elif track == 2:
        return _CLASS_MAP_T2
    else:
        raise ValueError(f"track must be 1 or 2, got {track}")


def get_class_names(track: int = 1) -> List[str]:
    """Return list of class names for the requested track (index 0 = background)."""
    return CLASS_NAMES_T1 if track == 1 else CLASS_NAMES_T2


def normalize_class_name(raw_name: str) -> str:
    """Normalize challenge labels to canonical class names.

    Examples:
        nuclei_tumor -> tumor
        tissue_blood_vessel -> blood vessel
        tils -> lymphocyte
    """
    name = raw_name.lower().strip().replace("-", " ").replace("_", " ")
    for prefix in ("nuclei ", "tissue "):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    name = " ".join(name.split())
    return _ALIAS_MAP.get(name, name)


# ─── Core parser ─────────────────────────────────────────────────────────────


def parse_geojson(
    geojson_path: Path,
    image_shape: Tuple[int, int],
    track: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Convert a PUMA GeoJSON annotation file to rasterised masks.

    Args:
        geojson_path: Path to the ``.geojson`` annotation file.
        image_shape:  ``(H, W)`` of the corresponding image.
        track:        Challenge track — 1 (3 classes) or 2 (10 classes).

    Returns:
        instance_mask:    ``(H, W)`` int32 array; 0 = background, 1..N = instance IDs.
        class_mask:       ``(H, W)`` uint8 array; class ID at each pixel.
        instance_classes: List of length N with the class ID for each instance
                          (``instance_classes[i]`` corresponds to instance ID ``i+1``).
    """
    class_map = get_class_map(track)
    fallback_id = class_map.get("other", len(class_map) - 1)
    H, W = image_shape

    with open(geojson_path, encoding="utf-8") as fh:
        data = json.load(fh)

    features = data.get("features", [])

    instance_mask = np.zeros((H, W), dtype=np.int32)
    class_mask = np.zeros((H, W), dtype=np.uint8)
    instance_classes: List[int] = []
    instance_id = 0

    for feat in features:
        geom = feat.get("geometry") or {}
        props = feat.get("properties") or {}

        # ── Resolve class name ────────────────────────────────────────────────
        cls_info = props.get("classification") or {}
        raw_name = cls_info.get("name", "other")
        normalized_name = normalize_class_name(raw_name)
        class_id = class_map.get(normalized_name, fallback_id)

        # ── Rasterise geometry ─────────────────────────────────────────────────
        geom_type = geom.get("type", "")
        coords_list: List[List] = []

        if geom_type == "Polygon":
            coords_list = [geom["coordinates"][0]]  # outer ring only
        elif geom_type == "MultiPolygon":
            coords_list = [poly[0] for poly in geom["coordinates"]]
        else:
            logger.debug("Unsupported geometry type '%s' — skipping.", geom_type)
            continue

        for ring in coords_list:
            rr, cc = _rasterize_ring(ring, H, W)
            if rr.size == 0:
                continue
            instance_id += 1
            instance_mask[rr, cc] = instance_id
            class_mask[rr, cc] = class_id
            instance_classes.append(class_id)

    logger.debug("Parsed %d nuclei from '%s'.", instance_id, geojson_path.name)
    return instance_mask, class_mask, instance_classes


def _rasterize_ring(
    ring: List[List[float]], H: int, W: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterise a GeoJSON polygon ring (list of [x, y] pairs).

    GeoJSON convention: x = column, y = row.
    Returns (row_indices, col_indices) that lie inside the polygon.
    """
    if len(ring) < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    xs = np.clip([c[0] for c in ring], 0, W - 1)
    ys = np.clip([c[1] for c in ring], 0, H - 1)

    # skimage.draw.polygon expects (row_coords, col_coords)
    rr, cc = sk_polygon(ys, xs, shape=(H, W))
    return rr, cc


# ─── Spatial helpers ─────────────────────────────────────────────────────────


def get_nucleus_centroids(instance_mask: np.ndarray) -> Dict[int, Tuple[float, float]]:
    """Compute (row, col) centroids for every instance.

    Returns:
        dict mapping instance_id → (row_centroid, col_centroid).
    """
    centroids: Dict[int, Tuple[float, float]] = {}
    instance_ids = np.unique(instance_mask)
    for inst_id in instance_ids[instance_ids > 0]:
        rows, cols = np.where(instance_mask == inst_id)
        centroids[int(inst_id)] = (float(rows.mean()), float(cols.mean()))
    return centroids


def get_nucleus_bboxes(
    instance_mask: np.ndarray,
    padding: int = 4,
) -> Dict[int, Tuple[int, int, int, int]]:
    """Compute bounding boxes for every instance.

    Returns:
        dict mapping instance_id → (y_min, x_min, y_max, x_max) (clipped to image).
    """
    H, W = instance_mask.shape
    bboxes: Dict[int, Tuple[int, int, int, int]] = {}
    instance_ids = np.unique(instance_mask)
    for inst_id in instance_ids[instance_ids > 0]:
        rows, cols = np.where(instance_mask == inst_id)
        y_min = max(0, int(rows.min()) - padding)
        y_max = min(H, int(rows.max()) + padding + 1)
        x_min = max(0, int(cols.min()) - padding)
        x_max = min(W, int(cols.max()) + padding + 1)
        bboxes[int(inst_id)] = (y_min, x_min, y_max, x_max)
    return bboxes


def extract_nucleus_crops(
    image: np.ndarray,
    instance_mask: np.ndarray,
    crop_size: int = 64,
    padding: int = 4,
) -> Tuple[List[np.ndarray], List[int]]:
    """Extract and resize square image crops for each nucleus.

    Args:
        image:         RGB image, shape (H, W, 3), uint8.
        instance_mask: Instance mask, shape (H, W).
        crop_size:     Output square size in pixels.
        padding:       Extra pixels around the bounding box.

    Returns:
        crops:       List of (crop_size, crop_size, 3) uint8 arrays.
        instance_ids: Corresponding instance IDs.
    """
    import cv2

    bboxes = get_nucleus_bboxes(instance_mask, padding=padding)
    crops: List[np.ndarray] = []
    ids: List[int] = []

    for inst_id, (y_min, x_min, y_max, x_max) in bboxes.items():
        crop = image[y_min:y_max, x_min:x_max]
        crop_resized = cv2.resize(crop, (crop_size, crop_size))
        crops.append(crop_resized)
        ids.append(inst_id)

    return crops, ids
