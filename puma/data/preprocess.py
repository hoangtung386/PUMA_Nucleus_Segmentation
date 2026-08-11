from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import tifffile
from numpy.lib.format import open_memmap
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import center_of_mass
from scipy.spatial import cKDTree
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid

from puma.config import DataConfig, PUMA_CLASS_NAMES, PUMA_CLASS_TO_ID, PathConfig, RuntimeConfig
from puma.utils import atomic_save_numpy, atomic_write_json, config_hash, utc_now_iso


PREPROCESSING_SCHEMA_VERSION = 3


ROI_DTYPE = np.dtype(
    [
        ("roi_index", "i4"),
        ("roi_id", "U128"),
        ("image_file", "U256"),
        ("geojson_file", "U256"),
        ("melanoma_type", "U16"),
        ("case_id", "U128"),
        ("fold", "i1"),
        ("height", "i4"),
        ("width", "i4"),
        ("n_nuclei", "i4"),
        ("overlap_pixels", "i8"),
        ("repaired_geometries", "i4"),
        ("failed_geometries", "i4"),
        ("rasterization_fallbacks", "i4"),
    ]
    + [(f"count_class_{i}", "i4") for i in range(len(PUMA_CLASS_NAMES))]
)

CENTROID_DTYPE = np.dtype(
    [
        ("roi_index", "i4"),
        ("nucleus_index", "i4"),
        ("annotation_id", "U96"),
        ("class_id", "i1"),
        ("class_name", "U32"),
        ("x", "f4"),
        ("y", "f4"),
        ("column", "i4"),
        ("row", "i4"),
        ("polygon_centroid_x", "f4"),
        ("polygon_centroid_y", "f4"),
        ("mask_centroid_x", "f4"),
        ("mask_centroid_y", "f4"),
        ("centroid_difference", "f4"),
        ("mask_centroid_difference", "f4"),
        ("area_pixels", "f4"),
        ("equivalent_diameter", "f4"),
        ("width", "f4"),
        ("height", "f4"),
        ("theta_radians", "f4"),
        ("nearest_neighbor_distance", "f4"),
        ("border_distance", "f4"),
        ("bbox_min_x", "f4"),
        ("bbox_min_y", "f4"),
        ("bbox_max_x", "f4"),
        ("bbox_max_y", "f4"),
        ("mask_pixel_count", "i4"),
        ("geometry_repaired", "u1"),
        ("rasterization_all_touched_fallback", "u1"),
        ("centroid_inside_geometry", "u1"),
    ]
)


class PreprocessingError(RuntimeError):
    pass


def infer_melanoma_type(name: str) -> str:
    lower = name.lower()
    if "metastatic" in lower:
        return "metastatic"
    if "primary" in lower:
        return "primary"
    return "unknown"


def infer_case_id(roi_id: str) -> str:
    """Use ROI ID when patient metadata is unavailable; replace from a sidecar if available."""
    return roi_id


def load_case_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    """Load optional ROI-to-case metadata using case-insensitive ROI identifiers."""
    if path is None or not Path(path).exists():
        return {}
    mapping: dict[str, dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"roi_id", "case_id"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise PreprocessingError(
                f"Optional case metadata must contain columns {sorted(required)}: {path}"
            )
        for row in reader:
            roi_id = str(row.get("roi_id", "")).strip()
            case_id = str(row.get("case_id", "")).strip()
            if not roi_id or not case_id:
                continue
            key = roi_id.casefold()
            if key in mapping:
                raise PreprocessingError(f"Duplicate roi_id in case metadata: {roi_id}")
            mapping[key] = {
                "case_id": case_id,
                "melanoma_type": str(row.get("melanoma_type", "")).strip().lower(),
            }
    return mapping


def _discover_files_case_insensitive(directory: Path, suffixes: set[str]) -> list[Path]:
    directory = Path(directory)
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes),
        key=lambda path: path.name.lower(),
    )


def _build_unique_stem_map(
    paths: list[Path],
    *,
    remove_nuclei_suffix: bool,
) -> dict[str, tuple[str, Path]]:
    result: dict[str, tuple[str, Path]] = {}
    for path in paths:
        stem = path.stem
        if remove_nuclei_suffix and stem.lower().endswith("_nuclei"):
            stem = stem[: -len("_nuclei")]
        key = stem.casefold()
        if key in result:
            raise PreprocessingError(
                f"Duplicate case-insensitive dataset stem {stem!r}: "
                f"{result[key][1].name!r} and {path.name!r}"
            )
        result[key] = (stem, path)
    return result


def discover_pairs(
    image_dir: Path,
    geojson_dir: Path,
    case_metadata: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Match .tif/.tiff images with *_nuclei.geojson annotations case-insensitively."""
    case_metadata = case_metadata or {}
    image_files = _discover_files_case_insensitive(Path(image_dir), {".tif", ".tiff"})
    geojson_files = _discover_files_case_insensitive(Path(geojson_dir), {".geojson"})
    images = _build_unique_stem_map(image_files, remove_nuclei_suffix=False)
    geojsons = _build_unique_stem_map(geojson_files, remove_nuclei_suffix=True)
    common = sorted(set(images).intersection(geojsons))
    missing_geo = sorted(images[key][0] for key in set(images).difference(geojsons))
    missing_img = sorted(geojsons[key][0] for key in set(geojsons).difference(images))
    if not common:
        raise FileNotFoundError(
            f"No matched TIFF/GeoJSON pairs. image_dir={image_dir}, geojson_dir={geojson_dir}"
        )
    if missing_geo or missing_img:
        raise FileNotFoundError(
            "Every training ROI must have both files. "
            f"Images without GeoJSON: {missing_geo[:20]}; "
            f"GeoJSON without image: {missing_img[:20]}."
        )
    pairs: list[dict[str, Any]] = []
    for key in common:
        roi_id, image_path = images[key]
        _, geojson_path = geojsons[key]
        metadata = case_metadata.get(key, {})
        inferred_type = infer_melanoma_type(roi_id)
        metadata_type = str(metadata.get("melanoma_type", "")).lower()
        melanoma_type = metadata_type if metadata_type in {"primary", "metastatic"} else inferred_type
        pairs.append(
            {
                "roi_id": roi_id,
                "image_path": image_path,
                "geojson_path": geojson_path,
                "melanoma_type": melanoma_type,
                "case_id": metadata.get("case_id", infer_case_id(roi_id)),
            }
        )
    return pairs


CLASS_ALIASES = {
    "tumor": "nuclei_tumor", "tumour": "nuclei_tumor", "tumor_cell": "nuclei_tumor",
    "tumour_cell": "nuclei_tumor", "nuclei_tumor_cell": "nuclei_tumor", "nuclei_tumour": "nuclei_tumor",
    "nuclei_tumour_cell": "nuclei_tumor",
    "lymphocyte": "nuclei_lymphocyte", "plasma_cell": "nuclei_plasma_cell",
    "histiocyte": "nuclei_histiocyte", "histocyte": "nuclei_histiocyte",
    "melanophage": "nuclei_melanophage", "neutrophil": "nuclei_neutrophil",
    "stroma": "nuclei_stroma", "stromal_cell": "nuclei_stroma", "stroma_cell": "nuclei_stroma",
    "epithelium": "nuclei_epithelium", "epithelial_cell": "nuclei_epithelium",
    "endothelium": "nuclei_endothelium", "endothelial_cell": "nuclei_endothelium",
    "apoptosis": "nuclei_apoptosis", "apoptotic_cell": "nuclei_apoptosis",
}

def normalize_class_name(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if normalized in PUMA_CLASS_TO_ID:
        return normalized
    without_prefix = normalized.removeprefix("nuclei_")
    return CLASS_ALIASES.get(normalized, CLASS_ALIASES.get(without_prefix, normalized))

def extract_class_name(properties: dict[str, Any]) -> str | None:
    classification = properties.get("classification")
    if isinstance(classification, dict) and classification.get("name"):
        return normalize_class_name(classification["name"])
    for key in ("class_name", "class", "label", "name"):
        value = properties.get(key)
        if value is not None and str(value).strip():
            return normalize_class_name(value)
    return None


def repair_polygonal_geometry(geometry: BaseGeometry) -> tuple[BaseGeometry, bool]:
    if geometry.is_valid:
        return geometry, False
    repaired = make_valid(geometry)
    if isinstance(repaired, GeometryCollection):
        parts: list[Polygon] = []
        for component in repaired.geoms:
            if isinstance(component, Polygon):
                parts.append(component)
            elif isinstance(component, MultiPolygon):
                parts.extend(list(component.geoms))
        if not parts:
            return repaired, True
        repaired = parts[0] if len(parts) == 1 else MultiPolygon(parts)
    return repaired, True


def local_rasterize_geometry(
    geometry: BaseGeometry,
    *,
    all_touched: bool,
    padding: int = 1,
) -> tuple[np.ndarray, int, int, bool]:
    """Rasterize one annotation, preserving the configured rule when possible.

    Very small/sliver polygons can contain no pixel centre, so Rasterio legitimately
    returns an empty mask when ``all_touched=False``.  In that narrow case only, retry
    the individual geometry with ``all_touched=True`` rather than changing the
    rasterization semantics for the whole dataset.
    """
    min_x_float, min_y_float, max_x_float, max_y_float = geometry.bounds
    min_x = int(np.floor(min_x_float)) - padding
    min_y = int(np.floor(min_y_float)) - padding
    max_x = int(np.ceil(max_x_float)) + padding
    max_y = int(np.ceil(max_y_float)) + padding
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid raster window {width}x{height}")

    def _rasterize(touch_all: bool) -> np.ndarray:
        return rasterize(
            [(mapping(geometry), 1)],
            out_shape=(height, width),
            transform=Affine.translation(min_x, min_y),
            fill=0,
            default_value=1,
            dtype=np.uint8,
            all_touched=touch_all,
        )

    mask = _rasterize(all_touched)
    used_fallback = False
    if not mask.any() and not all_touched:
        mask = _rasterize(True)
        used_fallback = bool(mask.any())
    if not mask.any():
        raise ValueError(
            "Rasterized polygon has no foreground pixels even with all_touched=True."
        )
    return mask, min_x, min_y, used_fallback


def pixel_mask_centroid(mask: np.ndarray, min_x: int, min_y: int) -> tuple[float, float]:
    pixel_count = int(mask.sum())
    if pixel_count <= 0:
        raise ValueError("Rasterized polygon has no foreground pixels.")
    local_y, local_x = center_of_mass(mask)
    return float(min_x + local_x + 0.5), float(min_y + local_y + 0.5)


def official_vertex_mean_centroid(geometry_json: dict[str, Any]) -> tuple[float, float]:
    """Reproduce the official PUMA evaluator centroid: arithmetic mean of path vertices.

    The evaluator uses every exterior path point exactly as supplied, including a repeated
    closing vertex when present. Polygon holes are not part of ``path_points`` and are ignored.
    For the rare MultiPolygon case, all exterior paths are concatenated deterministically.
    """
    geometry_type = str(geometry_json.get("type", ""))
    coordinates = geometry_json.get("coordinates")
    paths: list[Any]
    if geometry_type == "Polygon" and isinstance(coordinates, list) and coordinates:
        paths = [coordinates[0]]
    elif geometry_type == "MultiPolygon" and isinstance(coordinates, list):
        paths = [polygon[0] for polygon in coordinates if isinstance(polygon, list) and polygon]
    else:
        raise ValueError(f"Unsupported geometry for official centroid: {geometry_type!r}")
    points: list[list[float]] = []
    for exterior in paths:
        for coordinate in exterior:
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) < 2:
                raise ValueError("Invalid coordinate in polygon exterior path")
            points.append([float(coordinate[0]), float(coordinate[1])])
    if len(points) < 3:
        raise ValueError("A valid PUMA polygon requires at least three path points")
    centroid = np.asarray(points, dtype=np.float64).mean(axis=0)
    return float(centroid[0]), float(centroid[1])


def minimum_rotated_extent(geometry: BaseGeometry) -> tuple[float, float, float]:
    rectangle = geometry.minimum_rotated_rectangle
    coords = np.asarray(rectangle.exterior.coords[:-1], dtype=np.float64)
    if coords.shape[0] < 4:
        min_x, min_y, max_x, max_y = geometry.bounds
        return float(max_x - min_x), float(max_y - min_y), 0.0
    edges = np.roll(coords, -1, axis=0) - coords
    lengths = np.linalg.norm(edges, axis=1)
    order = np.argsort(lengths)[::-1]
    long_edge = edges[order[0]]
    width = float(lengths[order[0]])
    height = float(lengths[order[-1]])
    theta = float(math.atan2(long_edge[1], long_edge[0]))
    if height > width:
        width, height = height, width
        theta += math.pi / 2
    theta = (theta + math.pi) % math.pi
    return width, height, theta


def draw_gaussian_max(
    heatmap: np.ndarray,
    x: float,
    y: float,
    sigma: float,
) -> None:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x0 = max(0, int(math.floor(x)) - radius)
    x1 = min(heatmap.shape[1], int(math.floor(x)) + radius + 1)
    y0 = max(0, int(math.floor(y)) - radius)
    y1 = min(heatmap.shape[0], int(math.floor(y)) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    kernel = np.exp(-((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2) / (2.0 * sigma**2))
    view = heatmap[y0:y1, x0:x1]
    np.maximum(view, kernel.astype(heatmap.dtype), out=view)
    anchor_x, anchor_y = int(math.floor(x)), int(math.floor(y))
    if 0 <= anchor_x < heatmap.shape[1] and 0 <= anchor_y < heatmap.shape[0]:
        heatmap[anchor_y, anchor_x] = max(float(heatmap[anchor_y, anchor_x]), 1.0)


def _copy_local_mask(
    destination: np.ndarray,
    local_mask: np.ndarray,
    min_x: int,
    min_y: int,
    value: int,
) -> int:
    height, width = destination.shape
    dst_x0 = max(0, min_x)
    dst_y0 = max(0, min_y)
    dst_x1 = min(width, min_x + local_mask.shape[1])
    dst_y1 = min(height, min_y + local_mask.shape[0])
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return 0
    src_x0 = dst_x0 - min_x
    src_y0 = dst_y0 - min_y
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    mask_view = local_mask[src_y0:src_y1, src_x0:src_x1].astype(bool)
    destination_view = destination[dst_y0:dst_y1, dst_x0:dst_x1]
    overlap = int(np.count_nonzero(mask_view & (destination_view != 0)))
    destination_view[mask_view] = value
    return overlap



def _convert_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert common TIFF numeric types to uint8 without modulo truncation."""
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if array.dtype == np.bool_:
        return np.ascontiguousarray(array.astype(np.uint8) * 255)
    if np.issubdtype(array.dtype, np.integer):
        minimum = float(array.min(initial=0))
        maximum = float(array.max(initial=0))
        if minimum >= 0.0 and maximum <= 255.0:
            return np.ascontiguousarray(array.astype(np.uint8))
        if np.issubdtype(array.dtype, np.unsignedinteger):
            scale_max = float(np.iinfo(array.dtype).max)
            scaled = np.clip(array.astype(np.float32) / max(scale_max, 1.0), 0.0, 1.0)
        else:
            if maximum <= minimum:
                return np.zeros(array.shape, dtype=np.uint8)
            scaled = (array.astype(np.float32) - minimum) / (maximum - minimum)
        return np.ascontiguousarray(np.rint(scaled * 255.0).astype(np.uint8))
    if np.issubdtype(array.dtype, np.floating):
        working = np.asarray(array, dtype=np.float32)
        finite = np.isfinite(working)
        if not finite.any():
            return np.zeros(working.shape, dtype=np.uint8)
        working = np.where(finite, working, 0.0)
        minimum = float(working[finite].min())
        maximum = float(working[finite].max())
        if minimum >= 0.0 and maximum <= 1.0 + 1e-6:
            scaled = np.clip(working, 0.0, 1.0)
        elif minimum >= 0.0 and maximum <= 255.0 + 1e-6:
            scaled = np.clip(working / 255.0, 0.0, 1.0)
        else:
            low, high = np.percentile(working[finite], [0.1, 99.9])
            if high <= low:
                return np.zeros(working.shape, dtype=np.uint8)
            scaled = np.clip((working - low) / (high - low), 0.0, 1.0)
        return np.ascontiguousarray(np.rint(scaled * 255.0).astype(np.uint8))
    raise PreprocessingError(f"Unsupported TIFF dtype: {array.dtype}")


def ensure_rgb_uint8(image: np.ndarray, *, source_name: str = "TIFF") -> np.ndarray:
    """Normalize grayscale/RGB/RGBA, channel-first or channel-last arrays to HxWx3 uint8."""
    array = np.asarray(image)
    if array.ndim > 3:
        array = np.squeeze(array)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    elif array.ndim == 3:
        first_is_channel = array.shape[0] in (1, 2, 3, 4)
        last_is_channel = array.shape[-1] in (1, 2, 3, 4)
        if first_is_channel and not last_is_channel:
            array = np.moveaxis(array, 0, -1)
        elif last_is_channel and not first_is_channel:
            pass
        elif first_is_channel and last_is_channel:
            # Resolve tiny/ambiguous arrays using the usual RGB/RGBA preference.
            if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
                array = np.moveaxis(array, 0, -1)
            else:
                pass
        else:
            raise PreprocessingError(
                f"Could not identify a channel axis for {source_name}: shape={array.shape}"
            )
        channels = array.shape[-1]
        if channels == 1:
            array = np.repeat(array, 3, axis=-1)
        elif channels == 2:
            array = np.repeat(array[..., :1], 3, axis=-1)
        else:
            array = array[..., :3]
    else:
        raise PreprocessingError(f"Unsupported image shape {array.shape} for {source_name}")
    return _convert_to_uint8(array)


def read_tiff_rgb(path: Path) -> np.ndarray:
    """Read a TIFF/TIFF image and return a contiguous RGB uint8 array."""
    path = Path(path)
    try:
        image = tifffile.imread(path)
    except Exception as exc:
        raise PreprocessingError(f"Could not read TIFF {path}: {exc}") from exc
    return ensure_rgb_uint8(image, source_name=path.name)

def process_one_roi(task: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(task["image_path"])
    geojson_path = Path(task["geojson_path"])
    all_touched = bool(task["all_touched"])
    sigma_scale = float(task["sigma_scale"])
    sigma_min = float(task["sigma_min"])
    sigma_max = float(task["sigma_max"])

    image = read_tiff_rgb(image_path)
    height, width = image.shape[:2]
    expected_height = int(task.get("expected_height", height))
    expected_width = int(task.get("expected_width", width))
    if (height, width) != (expected_height, expected_width):
        raise PreprocessingError(
            f"Unexpected ROI shape for {image_path.name}: {(height, width)}; "
            f"expected {(expected_height, expected_width)}"
        )

    data = json.loads(geojson_path.read_text(encoding="utf-8-sig"))
    features = data.get("features")
    if not isinstance(features, list):
        raise PreprocessingError(f"No valid features list in {geojson_path.name}")

    instance_map = np.zeros((height, width), dtype=np.uint16)
    class_map_background_id = int(task.get("class_map_background_id", 255))
    foreground_values = set(range(1, len(PUMA_CLASS_NAMES) + 1))
    if not 0 <= class_map_background_id <= 255 or class_map_background_id in foreground_values:
        raise PreprocessingError(
            "class_map_background_id must be a uint8 value outside the encoded foreground "
            f"values 1-{len(PUMA_CLASS_NAMES)}, got {class_map_background_id}"
        )
    class_map = np.full((height, width), class_map_background_id, dtype=np.uint8)
    heatmap = np.zeros((height, width), dtype=np.float32)
    match_disk_map = np.zeros((height, width), dtype=np.uint8)
    records: list[dict[str, Any]] = []
    valid_items: list[dict[str, Any]] = []
    repaired_count = 0
    failed_count = 0
    rasterization_fallback_count = 0
    failure_messages: list[str] = []

    if len(features) > np.iinfo(np.uint16).max:
        raise PreprocessingError(
            f"Too many annotations for uint16 instance IDs in {geojson_path.name}: {len(features)}"
        )

    for feature_index, feature in enumerate(features):
        try:
            class_name = extract_class_name(feature.get("properties") or {})
            if class_name not in PUMA_CLASS_TO_ID:
                raise ValueError(f"Unknown PUMA nucleus class: {class_name!r}")
            geometry_json = feature.get("geometry")
            if not geometry_json:
                raise ValueError("Missing geometry")
            geometry = shape(geometry_json)
            if geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
                raise ValueError(f"Unsupported geometry {geometry.geom_type}")
            geometry, repaired = repair_polygonal_geometry(geometry)
            repaired_count += int(repaired)
            if geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
                raise ValueError("Geometry repair did not yield polygons")
            mask, min_x, min_y, used_rasterization_fallback = local_rasterize_geometry(
                geometry, all_touched=all_touched, padding=1
            )
            rasterization_fallback_count += int(used_rasterization_fallback)
            mask_centroid_x, mask_centroid_y = pixel_mask_centroid(mask, min_x, min_y)
            centroid_x, centroid_y = official_vertex_mean_centroid(geometry_json)
            if not (0.0 <= centroid_x < width and 0.0 <= centroid_y < height):
                raise ValueError(
                    f"Centroid {(centroid_x, centroid_y)} lies outside image bounds {(width, height)}"
                )
            polygon_centroid = geometry.centroid
            width_extent, height_extent, theta = minimum_rotated_extent(geometry)
            area = float(geometry.area)
            equivalent_diameter = float(2.0 * math.sqrt(max(area, 0.0) / math.pi))
            min_bx, min_by, max_bx, max_by = geometry.bounds
            border_distance = float(min(centroid_x, centroid_y, width - centroid_x, height - centroid_y))
            record = {
                "feature_index": feature_index,
                "annotation_id": str(feature.get("id") or f"feature_{feature_index}"),
                "class_id": PUMA_CLASS_TO_ID[class_name],
                "class_name": class_name,
                "x": centroid_x,
                "y": centroid_y,
                "column": int(np.floor(centroid_x)),
                "row": int(np.floor(centroid_y)),
                "polygon_centroid_x": float(polygon_centroid.x),
                "polygon_centroid_y": float(polygon_centroid.y),
                "mask_centroid_x": float(mask_centroid_x),
                "mask_centroid_y": float(mask_centroid_y),
                "centroid_difference": float(math.hypot(centroid_x - polygon_centroid.x, centroid_y - polygon_centroid.y)),
                "mask_centroid_difference": float(math.hypot(centroid_x - mask_centroid_x, centroid_y - mask_centroid_y)),
                "area_pixels": area,
                "equivalent_diameter": equivalent_diameter,
                "width": width_extent,
                "height": height_extent,
                "theta_radians": theta,
                "nearest_neighbor_distance": float("inf"),
                "border_distance": border_distance,
                "bbox_min_x": float(min_bx),
                "bbox_min_y": float(min_by),
                "bbox_max_x": float(max_bx),
                "bbox_max_y": float(max_by),
                "mask_pixel_count": int(mask.sum()),
                "geometry_repaired": int(repaired),
                "rasterization_all_touched_fallback": int(used_rasterization_fallback),
                "centroid_inside_geometry": int(geometry.covers(Point(centroid_x, centroid_y))),
            }
            records.append(record)
            valid_items.append(
                {
                    "record": record,
                    "mask": mask,
                    "min_x": min_x,
                    "min_y": min_y,
                    "area": area,
                }
            )
        except Exception as exc:
            failed_count += 1
            message = f"feature {feature_index}: {type(exc).__name__}: {exc}"
            failure_messages.append(message)
            print(f"[{geojson_path.name}] {message}")

    if failed_count and bool(task.get("fail_on_annotation_error", True)):
        preview = "; ".join(failure_messages[:10])
        raise PreprocessingError(
            f"{geojson_path.name} contains {failed_count} annotation errors. "
            f"First errors: {preview}"
        )

    if records:
        coordinates = np.asarray([[r["x"], r["y"]] for r in records], dtype=np.float32)
        if len(records) == 1:
            nearest = np.asarray([float(max(height, width))], dtype=np.float32)
        else:
            distances, _ = cKDTree(coordinates).query(coordinates, k=2)
            nearest = distances[:, 1].astype(np.float32)
        for record, distance in zip(records, nearest, strict=True):
            record["nearest_neighbor_distance"] = float(distance)

    # Larger instances first, smaller instances overwrite in rare overlap pixels.
    overlap_pixels = 0
    for item in sorted(valid_items, key=lambda item: item["area"], reverse=True):
        record = item["record"]
        instance_id = int(record["feature_index"] + 1)
        overlap_pixels += _copy_local_mask(
            instance_map, item["mask"], item["min_x"], item["min_y"], instance_id
        )
        _copy_local_mask(
            class_map,
            item["mask"],
            item["min_x"],
            item["min_y"],
            int(record["class_id"] + 1),  # foreground encoding is 1..10
        )

    for record in records:
        sigma = float(np.clip(sigma_scale * record["equivalent_diameter"], sigma_min, sigma_max))
        draw_gaussian_max(heatmap, record["x"], record["y"], sigma)
        # Explicit 15-pixel eligibility disk for evaluator QA. It is not used as the training Gaussian.
        radius = float(task.get("official_match_radius_px", 15.0))
        x, y = float(record["x"]), float(record["y"])
        x0, x1 = max(0, int(np.floor(x-radius))), min(width, int(np.ceil(x+radius))+1)
        y0, y1 = max(0, int(np.floor(y-radius))), min(height, int(np.ceil(y+radius))+1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        match_disk_map[y0:y1, x0:x1][(xx-x)**2 + (yy-y)**2 <= radius**2] = 1

    class_counts = np.zeros(len(PUMA_CLASS_NAMES), dtype=np.int32)
    for record in records:
        class_counts[int(record["class_id"])] += 1

    return {
        "roi_id": task["roi_id"],
        "image_file": image_path.name,
        "geojson_file": geojson_path.name,
        "melanoma_type": task["melanoma_type"],
        "case_id": task["case_id"],
        "image": image,
        "instance_map": instance_map,
        "class_map": class_map,
        "centroid_heatmap": heatmap.astype(np.float16),
        "centroid_match_disks": match_disk_map,
        "records": records,
        "class_counts": class_counts,
        "overlap_pixels": overlap_pixels,
        "repaired_geometries": repaired_count,
        "failed_geometries": failed_count,
        "rasterization_fallbacks": rasterization_fallback_count,
    }


def _feature_vector_for_folds(roi_result: dict[str, Any]) -> np.ndarray:
    counts = np.asarray(roi_result["class_counts"], dtype=np.float64)
    presence = (counts > 0).astype(np.float64)
    log_counts = np.log1p(counts)
    if log_counts.max() > 0:
        log_counts /= log_counts.max()
    melanoma = np.asarray(
        [
            float(roi_result["melanoma_type"] == "primary"),
            float(roi_result["melanoma_type"] == "metastatic"),
        ],
        dtype=np.float64,
    )
    return np.concatenate([melanoma, presence * 2.0, log_counts], axis=0)


def _fold_capacities(total_weight: int, number_of_folds: int) -> np.ndarray:
    """Split ``total_weight`` ROIs into per-fold capacities that differ by at most one."""
    base, remainder = divmod(int(total_weight), int(number_of_folds))
    capacities = np.full(number_of_folds, base, dtype=np.int64)
    capacities[:remainder] += 1
    return capacities


def _greedy_multilabel_assignments(
    results: list[dict[str, Any]],
    number_of_folds: int,
    seed: int,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy assignment for already-grouped records.

    Fold size is a hard capacity, not a cost term: every fold is given a quota of
    ``total_rois / number_of_folds`` (differing by at most one) and a group may only go
    to a fold that still has room. Class balance is then optimised *within* that
    constraint, by placing groups rarest-first into the eligible fold whose totals stay
    closest to the pro-rata share of everything placed so far.

    Both details matter, and an earlier version got both wrong:

    * It compared each fold against the *final* per-fold target. While every fold still
      sits far below that target, the fold closest to it is whichever is already
      fullest, so each group lands in the biggest fold and feeds a rich-get-richer
      loop. On PUMA (205 ROIs, 5 folds) that collapsed to sizes [1, 45, 87, 1, 71],
      decided by a cost margin of 0.003 on the very first placement.
    * It expressed size balance as a squared distance to the final size target, added
      with weight 0.35. That penalises an *empty* fold for exactly the same reason, so
      no weight can repair the collapse -- at weight 1000 the sizes were still
      [1, 79, 44, 1, 80]. Switching to the pro-rata share alone still left one fold
      starved at 11 ROIs, because a fold seeded with an outlier group stays above its
      share long enough to be skipped for most of the loop.

    A capacity is immune to both failure modes: no fold can be starved, whatever the
    feature cost says.
    """
    if len(results) < number_of_folds:
        raise ValueError(f"Need at least {number_of_folds} ROIs, found {len(results)}")
    if weights is None:
        weights = np.ones(len(results), dtype=np.int64)
    weights = np.asarray(weights, dtype=np.int64)
    if weights.shape != (len(results),) or np.any(weights < 1):
        raise ValueError("weights must hold one positive ROI count per group.")

    rng = np.random.default_rng(seed)
    features = np.stack([_feature_vector_for_folds(r) for r in results])
    totals = features.sum(axis=0)
    # Per-dimension scale, so no single feature dimension dominates the cost.
    scale = np.maximum(totals / number_of_folds, 1.0)
    rarity = 1.0 / np.maximum(totals, 1.0)
    priorities = (features * rarity).sum(axis=1) + rng.uniform(0, 1e-6, len(results))
    order = np.argsort(priorities)[::-1]

    capacities = _fold_capacities(int(weights.sum()), number_of_folds)
    fold_sums = np.zeros((number_of_folds, features.shape[1]), dtype=np.float64)
    fold_weights = np.zeros(number_of_folds, dtype=np.int64)
    assignments = np.full(len(results), -1, dtype=np.int8)

    # Seed the rarest group into every fold, so the rarest classes are spread across all
    # folds instead of concentrating wherever the feature cost happens to point first.
    seed_folds = rng.permutation(number_of_folds)
    seeded: set[int] = set()
    for position, idx in enumerate(order[:number_of_folds]):
        chosen = int(seed_folds[position])
        assignments[idx] = chosen
        fold_sums[chosen] += features[idx]
        fold_weights[chosen] += int(weights[idx])
        seeded.add(int(idx))

    placed_sum = fold_sums.sum(axis=0)
    for index in order:
        idx = int(index)
        if idx in seeded:
            continue
        placed_sum = placed_sum + features[idx]
        # Share every fold should hold once this group has landed somewhere.
        share = placed_sum / number_of_folds
        room = capacities - fold_weights
        eligible = room >= int(weights[idx])
        if not eligible.any():
            # A group larger than any remaining quota: put it where it overflows least.
            eligible = room >= room.max()
        candidate_sums = fold_sums + features[idx]
        costs = np.mean(((candidate_sums - share) / scale) ** 2, axis=1)
        costs = np.where(eligible, costs, np.inf)
        chosen = int(np.argmin(costs))
        assignments[idx] = chosen
        fold_sums[chosen] += features[idx]
        fold_weights[chosen] += int(weights[idx])
    if np.any(assignments < 0):
        raise AssertionError("Incomplete fold assignment")
    return _refine_by_equal_weight_swaps(
        assignments, features, weights, number_of_folds, scale
    )


def _refine_by_equal_weight_swaps(
    assignments: np.ndarray,
    features: np.ndarray,
    weights: np.ndarray,
    number_of_folds: int,
    scale: np.ndarray,
    *,
    max_sweeps: int = 12,
) -> np.ndarray:
    """Swap equal-weight groups between folds while it reduces class imbalance.

    The capacity-constrained greedy pass fixes fold *sizes*, but it still leaves an
    end-game skew: groups are placed rarest-first, so the common tumour-heavy ROIs are
    placed last and land in whichever fold still has quota rather than where they
    balance best. On PUMA that left one fold with 15467 tumour and 933 lymphocyte nuclei
    against roughly 10000/5000 elsewhere.

    Swapping two groups that cover the same number of ROIs leaves every fold size
    untouched, so this pass can only improve class balance, never break the capacity
    guarantee. Sweeps run to a fixed point or ``max_sweeps``, whichever comes first.
    """
    target = features.sum(axis=0) / number_of_folds
    fold_sums = np.stack([
        features[assignments == fold].sum(axis=0) for fold in range(number_of_folds)
    ])

    def deviation(vector: np.ndarray) -> float:
        return float((((vector - target) / scale) ** 2).sum())

    fold_cost = np.array([deviation(fold_sums[fold]) for fold in range(number_of_folds)])
    # Only equal-weight groups may swap, so pair candidates up by ROI count.
    by_weight: dict[int, list[int]] = {}
    for index, weight in enumerate(weights):
        by_weight.setdefault(int(weight), []).append(int(index))

    for _ in range(max_sweeps):
        improved = False
        for candidates in by_weight.values():
            for position, left in enumerate(candidates):
                for right in candidates[position + 1:]:
                    fold_left, fold_right = int(assignments[left]), int(assignments[right])
                    if fold_left == fold_right:
                        continue
                    new_left = fold_sums[fold_left] - features[left] + features[right]
                    new_right = fold_sums[fold_right] - features[right] + features[left]
                    delta = (
                        deviation(new_left)
                        + deviation(new_right)
                        - fold_cost[fold_left]
                        - fold_cost[fold_right]
                    )
                    if delta < -1e-12:
                        assignments[left], assignments[right] = fold_right, fold_left
                        fold_sums[fold_left], fold_sums[fold_right] = new_left, new_right
                        fold_cost[fold_left] = deviation(new_left)
                        fold_cost[fold_right] = deviation(new_right)
                        improved = True
        if not improved:
            break
    return assignments


def validate_fold_assignments(
    assignments: np.ndarray,
    number_of_folds: int,
    *,
    minimum_fold_fraction: float = 0.5,
) -> dict[str, Any]:
    """Fail loudly when a fold is too small to serve as a validation split.

    Stage 1 uses every fold twice: once as the untouched outer fold that receives the
    OOF prediction, and once as the inner fold that selects the checkpoint, heatmap
    threshold, and local-max radius. A fold holding a handful of ROIs makes both of
    those meaningless while still training and reporting without error, so a degenerate
    split has to be a hard failure rather than a warning.
    """
    counts = np.bincount(np.asarray(assignments, dtype=np.int64), minlength=number_of_folds)
    if counts.shape[0] != number_of_folds:
        raise ValueError(
            f"Fold labels outside [0,{number_of_folds - 1}] were assigned: {counts.shape[0]} labels."
        )
    expected = len(assignments) / number_of_folds
    floor = max(2.0, minimum_fold_fraction * expected)
    if int(counts.min()) < floor:
        raise ValueError(
            f"Degenerate fold assignment: sizes {counts.tolist()} for {len(assignments)} ROIs "
            f"across {number_of_folds} folds (expected about {expected:.0f} each, minimum "
            f"allowed {floor:.0f}). Nested validation and OOF coverage would both be invalid."
        )
    return {
        "fold_sizes": counts.tolist(),
        "expected_fold_size": round(float(expected), 2),
        "minimum_fold_size": int(counts.min()),
        "maximum_fold_size": int(counts.max()),
        "size_imbalance_ratio": round(float(counts.max() / max(int(counts.min()), 1)), 3),
    }


def multilabel_greedy_folds(
    results: list[dict[str, Any]],
    number_of_folds: int,
    seed: int,
) -> np.ndarray:
    """Assign complete case groups to one fold while balancing type and all ten classes."""
    grouped: dict[str, list[int]] = {}
    for index, result in enumerate(results):
        grouped.setdefault(str(result["case_id"]), []).append(index)
    if len(grouped) < number_of_folds:
        raise ValueError(
            f"Need at least {number_of_folds} distinct case groups, found {len(grouped)}. "
            "Provide Dataset/puma_case_metadata.csv when multiple ROIs belong to one patient."
        )
    group_ids = sorted(grouped)
    group_records: list[dict[str, Any]] = []
    for group_id in group_ids:
        members = [results[i] for i in grouped[group_id]]
        counts = np.sum([np.asarray(m["class_counts"], dtype=np.int64) for m in members], axis=0)
        types = {str(m["melanoma_type"]) for m in members}
        melanoma_type = next(iter(types)) if len(types) == 1 else "mixed"
        group_records.append({
            "case_id": group_id,
            "class_counts": counts,
            "melanoma_type": melanoma_type,
        })
    # Capacity is counted in ROIs, not case groups, so a multi-ROI patient consumes its
    # full share of the fold quota and the resulting ROI counts stay balanced.
    group_weights = np.array([len(grouped[group_id]) for group_id in group_ids], dtype=np.int64)
    group_folds = _greedy_multilabel_assignments(
        group_records, number_of_folds, seed, weights=group_weights
    )
    assignments = np.full(len(results), -1, dtype=np.int8)
    for group_id, fold in zip(group_ids, group_folds, strict=True):
        assignments[np.asarray(grouped[group_id], dtype=np.int64)] = int(fold)
    if np.any(assignments < 0):
        raise AssertionError("Incomplete case-group fold assignment")
    validate_fold_assignments(assignments, number_of_folds)
    return assignments


def fold_assignment_report(
    results: list[dict[str, Any]],
    assignments: np.ndarray,
    number_of_folds: int,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Summarise ROI counts, melanoma type, and per-class spread across the folds."""
    assignments = np.asarray(assignments, dtype=np.int64)
    counts = np.stack([np.asarray(r["class_counts"], dtype=np.int64) for r in results])
    types = np.array([str(r["melanoma_type"]) for r in results])
    per_fold: dict[str, Any] = {}
    for fold in range(number_of_folds):
        mask = assignments == fold
        class_totals = counts[mask].sum(axis=0)
        per_fold[str(fold)] = {
            "rois": int(mask.sum()),
            "nuclei": int(class_totals.sum()),
            "primary": int(np.sum(types[mask] == "primary")),
            "metastatic": int(np.sum(types[mask] == "metastatic")),
            "class_counts": {
                str(name): int(value) for name, value in zip(class_names, class_totals, strict=True)
            },
            "classes_absent": [
                str(name)
                for name, value in zip(class_names, class_totals, strict=True)
                if int(value) == 0
            ],
        }
    return {
        **validate_fold_assignments(assignments, number_of_folds),
        "per_fold": per_fold,
    }


def _submit_bounded(
    executor: ProcessPoolExecutor,
    tasks: list[dict[str, Any]],
    max_in_flight: int,
) -> Iterable[dict[str, Any]]:
    iterator = iter(tasks)
    futures = set()
    for _ in range(min(max_in_flight, len(tasks))):
        try:
            futures.add(executor.submit(process_one_roi, next(iterator)))
        except StopIteration:
            break
    while futures:
        done, futures = wait(futures, return_when=FIRST_COMPLETED)
        for future in done:
            yield future.result()
            try:
                futures.add(executor.submit(process_one_roi, next(iterator)))
            except StopIteration:
                pass



def _source_inventory_hash(pairs: list[dict[str, Any]]) -> str:
    inventory: list[dict[str, Any]] = []
    for pair in pairs:
        for role in ("image_path", "geojson_path"):
            path = Path(pair[role])
            stat = path.stat()
            inventory.append(
                {
                    "role": role,
                    "name": path.name,
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
    return config_hash(inventory, length=32)


def _preprocessed_cache_is_valid(
    output_paths: dict[str, Path],
    data_config: DataConfig,
    source_inventory_hash: str,
) -> tuple[bool, str]:
    missing = [key for key, path in output_paths.items() if not path.exists()]
    if missing:
        return False, f"missing artifacts: {missing}"
    try:
        metadata = json.loads(output_paths["metadata"].read_text(encoding="utf-8"))
        expected_hash = config_hash(asdict(data_config))
        if metadata.get("preprocessing_schema_version") != PREPROCESSING_SCHEMA_VERSION:
            return False, "preprocessing artifact schema changed"
        if metadata.get("configuration_hash") != expected_hash:
            return False, "preprocessing configuration changed"
        if metadata.get("source_inventory_hash") != source_inventory_hash:
            return False, "source TIFF/GeoJSON inventory changed"

        images = np.load(output_paths["images"], mmap_mode="r", allow_pickle=False)
        instances = np.load(output_paths["instances"], mmap_mode="r", allow_pickle=False)
        classes = np.load(output_paths["classes"], mmap_mode="r", allow_pickle=False)
        heatmaps = np.load(output_paths["heatmaps"], mmap_mode="r", allow_pickle=False)
        match_disks = np.load(output_paths["match_disks"], mmap_mode="r", allow_pickle=False)
        manifest = np.load(output_paths["roi_manifest"], mmap_mode="r", allow_pickle=False)
        centroids = np.load(output_paths["centroids"], mmap_mode="r", allow_pickle=False)
        offsets = np.load(output_paths["centroid_offsets"], mmap_mode="r", allow_pickle=False)
        folds = np.load(output_paths["folds"], mmap_mode="r", allow_pickle=False)

        if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
            return False, f"invalid image artifact shape/dtype: {images.shape}/{images.dtype}"
        n, height, width, _ = images.shape
        expected_spatial = (data_config.image_height, data_config.image_width)
        if (height, width) != expected_spatial:
            return False, f"cached spatial shape {(height, width)} != {expected_spatial}"
        for name, array in (
            ("instances", instances),
            ("classes", classes),
            ("heatmaps", heatmaps),
            ("match_disks", match_disks),
        ):
            if array.shape != (n, height, width):
                return False, f"{name} shape mismatch: {array.shape}"
        if manifest.dtype != ROI_DTYPE:
            return False, f"ROI manifest dtype mismatch: {manifest.dtype}"
        if centroids.dtype != CENTROID_DTYPE:
            return False, f"centroid dtype mismatch: {centroids.dtype}"
        if len(manifest) != n or len(folds) != n or len(offsets) != n + 1:
            return False, "manifest/fold/offset length mismatch"
        if int(offsets[0]) != 0 or int(offsets[-1]) != len(centroids):
            return False, "centroid offsets do not match centroid array"
        if np.any(np.diff(np.asarray(offsets)) < 0):
            return False, "centroid offsets are not monotonic"
        fold_array = np.asarray(folds)
        if np.any((fold_array < 0) | (fold_array >= data_config.number_of_folds)):
            return False, "fold assignments are outside the configured range"
        if set(np.unique(fold_array).tolist()) != set(range(data_config.number_of_folds)):
            return False, "one or more configured folds are empty"
    except Exception as exc:
        return False, f"cache validation failed: {type(exc).__name__}: {exc}"
    return True, "valid"

def preprocess_dataset(config: RuntimeConfig, *, force: bool = False) -> dict[str, Path]:
    paths = config.paths
    data_config = config.data
    paths.ensure()
    output_paths = {
        "images": paths.preprocessing_file("puma_rgb_images.npy"),
        "instances": paths.preprocessing_file("puma_instance_maps.npy"),
        "classes": paths.preprocessing_file("puma_class_maps.npy"),
        "heatmaps": paths.preprocessing_file("puma_centroid_heatmaps.npy"),
        "match_disks": paths.preprocessing_file("puma_centroid_match_disks_15px.npy"),
        "roi_manifest": paths.preprocessing_file("puma_roi_manifest.npy"),
        "centroids": paths.preprocessing_file("puma_nuclei_centroids.npy"),
        "centroid_offsets": paths.preprocessing_file("puma_roi_centroid_offsets.npy"),
        "folds": paths.preprocessing_file("puma_fold_assignments.npy"),
        "metadata": paths.preprocessing_file("puma_preprocessing_metadata.json"),
    }
    case_metadata = load_case_metadata(paths.case_metadata_csv)
    pairs = discover_pairs(paths.image_dir, paths.nuclei_geojson_dir, case_metadata)
    source_inventory_hash = _source_inventory_hash(pairs)
    if not force:
        cache_valid, cache_reason = _preprocessed_cache_is_valid(
            output_paths, data_config, source_inventory_hash
        )
        if cache_valid:
            print("Preprocessed NPY artifacts are complete and current; skipping. Use force=True to rebuild.")
            return output_paths
        if any(path.exists() for path in output_paths.values()):
            print(f"Existing preprocessing cache is stale or incomplete ({cache_reason}); rebuilding.")
    tasks = [
        {
            **pair,
            "image_path": str(pair["image_path"]),
            "geojson_path": str(pair["geojson_path"]),
            "all_touched": data_config.rasterize_all_touched,
            "sigma_scale": data_config.canonical_heatmap_sigma_scale,
            "sigma_min": data_config.canonical_heatmap_sigma_min,
            "sigma_max": data_config.canonical_heatmap_sigma_max,
            "official_match_radius_px": data_config.official_match_radius_px,
            "expected_height": data_config.image_height,
            "expected_width": data_config.image_width,
            "class_map_background_id": data_config.class_map_background_id,
            "fail_on_annotation_error": data_config.fail_on_annotation_error,
        }
        for pair in pairs
    ]
    workers = data_config.preprocessing_workers or (os.cpu_count() or 1)
    print(f"Processing {len(tasks)} ROIs with {workers} CPU workers...")
    results_by_id: dict[str, dict[str, Any]] = {}
    if workers == 1:
        for task in tasks:
            result = process_one_roi(task)
            results_by_id[result["roi_id"]] = result
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in _submit_bounded(executor, tasks, max_in_flight=max(1, workers)):
                results_by_id[result["roi_id"]] = result
                print(f"  completed {len(results_by_id)}/{len(tasks)}: {result['roi_id']}")
    results = [results_by_id[pair["roi_id"]] for pair in pairs]

    shapes = {tuple(result["image"].shape[:2]) for result in results}
    if len(shapes) != 1:
        raise PreprocessingError(f"All PUMA ROIs must have one fixed size; found {sorted(shapes)}")
    height, width = next(iter(shapes))
    number_of_rois = len(results)
    images_mm = open_memmap(output_paths["images"], mode="w+", dtype=np.uint8, shape=(number_of_rois, height, width, 3))
    instances_mm = open_memmap(output_paths["instances"], mode="w+", dtype=np.uint16, shape=(number_of_rois, height, width))
    classes_mm = open_memmap(output_paths["classes"], mode="w+", dtype=np.uint8, shape=(number_of_rois, height, width))
    heatmaps_mm = open_memmap(output_paths["heatmaps"], mode="w+", dtype=np.float16, shape=(number_of_rois, height, width))
    match_disks_mm = open_memmap(output_paths["match_disks"], mode="w+", dtype=np.uint8, shape=(number_of_rois, height, width))

    fold_assignments = multilabel_greedy_folds(results, data_config.number_of_folds, data_config.random_seed)
    roi_manifest = np.zeros(number_of_rois, dtype=ROI_DTYPE)
    centroid_offsets = np.zeros(number_of_rois + 1, dtype=np.int64)
    all_centroid_rows: list[tuple[Any, ...]] = []

    for roi_index, (result, fold) in enumerate(zip(results, fold_assignments, strict=True)):
        images_mm[roi_index] = result["image"]
        instances_mm[roi_index] = result["instance_map"]
        classes_mm[roi_index] = result["class_map"]
        heatmaps_mm[roi_index] = result["centroid_heatmap"]
        match_disks_mm[roi_index] = result["centroid_match_disks"]
        roi_row: list[Any] = [
            roi_index,
            result["roi_id"],
            result["image_file"],
            result["geojson_file"],
            result["melanoma_type"],
            result["case_id"],
            int(fold),
            height,
            width,
            len(result["records"]),
            result["overlap_pixels"],
            result["repaired_geometries"],
            result["failed_geometries"],
            result["rasterization_fallbacks"],
        ] + result["class_counts"].astype(int).tolist()
        roi_manifest[roi_index] = tuple(roi_row)
        centroid_offsets[roi_index] = len(all_centroid_rows)
        for nucleus_index, record in enumerate(result["records"]):
            all_centroid_rows.append(
                (
                    roi_index,
                    nucleus_index,
                    record["annotation_id"],
                    record["class_id"],
                    record["class_name"],
                    record["x"],
                    record["y"],
                    record["column"],
                    record["row"],
                    record["polygon_centroid_x"],
                    record["polygon_centroid_y"],
                    record["mask_centroid_x"],
                    record["mask_centroid_y"],
                    record["centroid_difference"],
                    record["mask_centroid_difference"],
                    record["area_pixels"],
                    record["equivalent_diameter"],
                    record["width"],
                    record["height"],
                    record["theta_radians"],
                    record["nearest_neighbor_distance"],
                    record["border_distance"],
                    record["bbox_min_x"],
                    record["bbox_min_y"],
                    record["bbox_max_x"],
                    record["bbox_max_y"],
                    record["mask_pixel_count"],
                    record["geometry_repaired"],
                    record["rasterization_all_touched_fallback"],
                    record["centroid_inside_geometry"],
                )
            )
    centroid_offsets[number_of_rois] = len(all_centroid_rows)
    centroids = np.asarray(all_centroid_rows, dtype=CENTROID_DTYPE)

    del images_mm, instances_mm, classes_mm, heatmaps_mm, match_disks_mm
    atomic_save_numpy(output_paths["roi_manifest"], roi_manifest)
    atomic_save_numpy(output_paths["centroids"], centroids)
    atomic_save_numpy(output_paths["centroid_offsets"], centroid_offsets)
    atomic_save_numpy(output_paths["folds"], fold_assignments.astype(np.int8))
    metadata = {
        "created_at": utc_now_iso(),
        "preprocessing_schema_version": PREPROCESSING_SCHEMA_VERSION,
        "configuration": asdict(data_config),
        "configuration_hash": config_hash(asdict(data_config)),
        "source_inventory_hash": source_inventory_hash,
        "number_of_rois": number_of_rois,
        "number_of_nuclei": int(len(centroids)),
        "rasterization_fallback_count": int(sum(result["rasterization_fallbacks"] for result in results)),
        "shape": [height, width],
        "class_names": list(PUMA_CLASS_NAMES),
        "centroid_definition": (
            "Arithmetic mean of all exterior path vertices, including a repeated closing point when present, "
            "matching the official PUMA Track-2 evaluator. Raster-mask and geometric polygon centroids are "
            "retained only for QA."
        ),
        "official_match_radius_px": data_config.official_match_radius_px,
        "class_map_background_id": data_config.class_map_background_id,
        "class_map_encoding": "background=class_map_background_id; nuclei pixels=class_id+1 (1..10)",
        "important_note": (
            "The 15-pixel PUMA tolerance is used by evaluation and candidate matching, not as the Gaussian target sigma."
        ),
        "fold_counts": {str(f): int(np.sum(fold_assignments == f)) for f in range(data_config.number_of_folds)},
        "fold_report": fold_assignment_report(
            results, fold_assignments, data_config.number_of_folds, PUMA_CLASS_NAMES
        ),
        "case_group_count": int(len(set(str(row["case_id"]) for row in roi_manifest))),
        "case_metadata_csv": str(paths.case_metadata_csv) if paths.case_metadata_csv.exists() else None,
        "artifact_files": {key: str(path) for key, path in output_paths.items()},
    }
    atomic_write_json(output_paths["metadata"], metadata)
    print(f"Finished preprocessing: {number_of_rois} ROIs, {len(centroids)} nuclei")
    print(f"Preprocessing artifacts: {paths.preprocessing_dir}")
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess PUMA TIFF/GeoJSON data into NPY memmaps.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    cfg = RuntimeConfig(paths=PathConfig(root=args.root))
    cfg.data.preprocessing_workers = args.workers
    preprocess_dataset(cfg, force=args.force)


if __name__ == "__main__":
    main()
