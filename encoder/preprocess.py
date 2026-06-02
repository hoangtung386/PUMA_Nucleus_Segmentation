from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw
from tqdm import tqdm

from .config import (
    CLASS_ALIASES,
    IGNORE_INDEX,
    IMAGE_DIR,
    NUCLEI_CLASSES,
    NUCLEI_GEOJSON_DIR,
    PROCESSED_DIR,
    TISSUE_BACKGROUND_ID,
    TISSUE_CLASSES,
    TISSUE_GEOJSON_DIR,
)

IMAGE_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
PROPERTY_KEYS = ('classification', 'class', 'name', 'label', 'type', 'objectType')


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {'.tif', '.tiff'}:
        image = tifffile.imread(str(path))
    else:
        image = np.asarray(Image.open(path))

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3:
        raise ValueError(f'Unsupported image shape {image.shape} for {path}')
    if image.shape[-1] > 3:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        lo = float(np.nanmin(image))
        hi = float(np.nanmax(image))
        if hi > lo:
            image = (255.0 * (image - lo) / (hi - lo)).clip(0, 255)
        image = image.astype(np.uint8)
    return np.ascontiguousarray(image)


def find_geojson(folder: Path, image_stem: str, suffix: str) -> Optional[Path]:
    candidates = [
        folder / f'{image_stem}_{suffix}.geojson',
        folder / f'{image_stem}_{suffix}.json',
        folder / f'{image_stem}.geojson',
        folder / f'{image_stem}.json',
    ]
    for p in candidates:
        if p.exists():
            return p
    globbed = sorted(folder.glob(f'{image_stem}*{suffix}*.geojson')) + sorted(folder.glob(f'{image_stem}*{suffix}*.json'))
    return globbed[0] if globbed else None


def extract_class_name(properties: Dict[str, Any]) -> Optional[str]:
    for key in PROPERTY_KEYS:
        if key not in properties:
            continue
        value = properties[key]
        if isinstance(value, dict):
            for inner_key in ('name', 'class', 'label', 'type', 'value'):
                if inner_key in value and value[inner_key] is not None:
                    return str(value[inner_key])
        elif value is not None:
            return str(value)
    return None



def normalize_class_for_domain(raw_name: str | None, class_map: Dict[str, int], domain_prefix: str) -> Optional[str]:
    """Resolve a GeoJSON class label inside the correct output head.

    This avoids a common bug where bare labels such as "tumor" or "stroma" are
    globally normalized to tissue_* and then skipped when rasterizing nuclei.
    """
    if raw_name is None:
        return None

    s = str(raw_name).strip().lower()
    s = s.replace('-', '_').replace('/', '_')
    s = ' '.join(s.split())
    s = s.replace(' ', '_')

    candidates = [
        s,
        f'{domain_prefix}{s}',
        CLASS_ALIASES.get(s, s),
    ]

    # If the global alias points to the wrong domain, still try the same suffix
    # inside this domain. Example: tumor -> tissue_tumor globally, but while
    # rasterizing nuclei we also need nuclei_tumor.
    aliased = CLASS_ALIASES.get(s)
    if aliased and '_' in aliased:
        suffix = aliased.split('_', 1)[1]
        candidates.append(f'{domain_prefix}{suffix}')

    for candidate in candidates:
        if candidate in class_map:
            return candidate
    return None

def polygon_rings(geometry: Dict[str, Any]) -> Iterator[List[Tuple[float, float]]]:
    if not geometry:
        return
    gtype = geometry.get('type')
    coords = geometry.get('coordinates', [])
    if gtype == 'Polygon':
        if coords:
            yield [(float(x), float(y)) for x, y, *rest in coords[0]]
    elif gtype == 'MultiPolygon':
        for polygon in coords:
            if polygon and polygon[0]:
                yield [(float(x), float(y)) for x, y, *rest in polygon[0]]


def rasterize_geojson(
    geojson_path: Optional[Path],
    height: int,
    width: int,
    class_map: Dict[str, int],
    default_value: int,
    domain_prefix: str,
) -> tuple[np.ndarray, Counter, Counter]:
    mask = np.full((height, width), int(default_value), dtype=np.uint8)
    seen = Counter()
    skipped = Counter()
    if geojson_path is None:
        return mask, seen, skipped

    with geojson_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    features = data.get('features', []) if isinstance(data, dict) else []

    canvas = Image.fromarray(mask)
    draw = ImageDraw.Draw(canvas)

    for feature in features:
        properties = feature.get('properties', {}) or {}
        raw_name = extract_class_name(properties)
        canonical = normalize_class_for_domain(raw_name, class_map, domain_prefix)
        if canonical is None:
            if raw_name is None:
                skipped['<missing_class_property>'] += 1
            else:
                skipped[str(raw_name)] += 1
            continue

        class_id = int(class_map[canonical])
        drew_any = False
        for ring in polygon_rings(feature.get('geometry', {}) or {}):
            if len(ring) >= 3:
                draw.polygon(ring, fill=class_id)
                drew_any = True
        if drew_any:
            seen[canonical] += 1

    return np.asarray(canvas, dtype=np.uint8), seen, skipped


def preprocess_dataset(overwrite: bool = False) -> Path:
    image_out = PROCESSED_DIR / 'images_npy'
    mask_out = PROCESSED_DIR / 'masks_npz'
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f'Missing image dir: {IMAGE_DIR}')
    if not TISSUE_GEOJSON_DIR.exists():
        raise FileNotFoundError(f'Missing tissue GeoJSON dir: {TISSUE_GEOJSON_DIR}')
    if not NUCLEI_GEOJSON_DIR.exists():
        raise FileNotFoundError(f'Missing nuclei GeoJSON dir: {NUCLEI_GEOJSON_DIR}')

    image_paths = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise FileNotFoundError(f'No images found in {IMAGE_DIR}')

    rows = []
    all_seen = Counter()
    all_skipped = Counter()

    for image_path in tqdm(image_paths, desc='Preprocess PUMA ROIs'):
        stem = image_path.stem
        image_npy = image_out / f'{stem}.npy'
        mask_npz = mask_out / f'{stem}.npz'

        if image_npy.exists() and mask_npz.exists() and not overwrite:
            with np.load(mask_npz) as masks:
                tissue = masks['tissue']
                nuclei_class = masks['nuclei_class']
                nuclei_fg = masks['nuclei_fg']
            rows.append(make_index_row(stem, image_npy, mask_npz, tissue, nuclei_class, nuclei_fg, None, None))
            continue

        image = read_image(image_path)
        h, w = image.shape[:2]
        tissue_geojson = find_geojson(TISSUE_GEOJSON_DIR, stem, 'tissue')
        nuclei_geojson = find_geojson(NUCLEI_GEOJSON_DIR, stem, 'nuclei')

        tissue, tissue_seen, tissue_skipped = rasterize_geojson(
            tissue_geojson, h, w, TISSUE_CLASSES, TISSUE_BACKGROUND_ID, 'tissue_'
        )
        nuclei_class, nuclei_seen, nuclei_skipped = rasterize_geojson(
            nuclei_geojson, h, w, NUCLEI_CLASSES, IGNORE_INDEX, 'nuclei_'
        )
        nuclei_fg = (nuclei_class != IGNORE_INDEX).astype(np.uint8)

        if tissue.shape != (h, w) or nuclei_class.shape != (h, w) or nuclei_fg.shape != (h, w):
            raise RuntimeError(f'Mask shape mismatch for {stem}')

        np.save(image_npy, image)
        np.savez_compressed(mask_npz, tissue=tissue, nuclei_class=nuclei_class, nuclei_fg=nuclei_fg)

        all_seen.update(tissue_seen)
        all_seen.update(nuclei_seen)
        all_skipped.update(tissue_skipped)
        all_skipped.update(nuclei_skipped)
        rows.append(make_index_row(stem, image_npy, mask_npz, tissue, nuclei_class, nuclei_fg, tissue_geojson, nuclei_geojson))

    index = pd.DataFrame(rows)
    index_path = PROCESSED_DIR / 'index.csv'
    index.to_csv(index_path, index=False)

    report = {
        'num_images': len(index),
        'seen_feature_counts': dict(all_seen),
        'skipped_feature_counts': dict(all_skipped),
        'tissue_classes': TISSUE_CLASSES,
        'nuclei_classes': NUCLEI_CLASSES,
        'notes': [
            'Tissue background is class 0.',
            'Nuclei has no background class; non-nucleus pixels use ignore_index=255 for nuclei_class.',
            'Each ROI can contain only a subset of labels; missing classes are allowed.',
        ],
    }
    with (PROCESSED_DIR / 'preprocess_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f'Wrote {index_path}')
    print(f'Wrote {PROCESSED_DIR / "preprocess_report.json"}')
    if all_skipped:
        print('[WARNING] Some GeoJSON features were skipped. Check preprocess_report.json')
    return index_path


def make_index_row(
    stem: str,
    image_npy: Path,
    mask_npz: Path,
    tissue: np.ndarray,
    nuclei_class: np.ndarray,
    nuclei_fg: np.ndarray,
    tissue_geojson: Optional[Path],
    nuclei_geojson: Optional[Path],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        'id': stem,
        'image_path': str(image_npy),
        'mask_path': str(mask_npz),
        'tissue_geojson': '' if tissue_geojson is None else str(tissue_geojson),
        'nuclei_geojson': '' if nuclei_geojson is None else str(nuclei_geojson),
        'height': int(tissue.shape[0]),
        'width': int(tissue.shape[1]),
        'nuclei_fg_pixels': int(nuclei_fg.sum()),
    }
    for name, cid in TISSUE_CLASSES.items():
        row[f'pixels_{name}'] = int((tissue == cid).sum())
        row[f'present_{name}'] = int(np.any(tissue == cid))
    for name, cid in NUCLEI_CLASSES.items():
        row[f'pixels_{name}'] = int((nuclei_class == cid).sum())
        row[f'present_{name}'] = int(np.any(nuclei_class == cid))
    return row
