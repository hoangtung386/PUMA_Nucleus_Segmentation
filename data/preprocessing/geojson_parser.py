"""GeoJSON parsing and rasterization utilities."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


def _feature_class_name(feature: dict) -> Optional[str]:
    """Extract the class name from a GeoJSON feature.

    Checks 'classification.name', 'name', 'class', and 'label' properties.

    Args:
        feature: A GeoJSON feature dictionary.

    Returns:
        The class name string, or None if not found.
    """
    props = feature.get("properties", {}) or {}
    cls = props.get("classification", {}) or {}
    name = cls.get("name")
    if name is None:
        name = props.get("name") or props.get("class") or props.get("label")
    return name


def _polygon_arrays_from_geometry(geometry: Optional[dict]) -> List[np.ndarray]:
    """Extract polygon coordinate arrays from a GeoJSON geometry dict.

    Supports Polygon and MultiPolygon geometry types.

    Args:
        geometry: A GeoJSON geometry dictionary, or None.

    Returns:
        list of numpy arrays, each of shape [N, 2] containing polygon
        vertex coordinates.
    """
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
