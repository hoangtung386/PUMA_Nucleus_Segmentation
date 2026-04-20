"""
prepare_data.py — Convert PUMA raw GeoJSON annotations to instance masks.

Usage:
    python scripts/prepare_data.py \\
        --raw-dir   data/raw \\
        --out-dir   data/processed \\
        --track     1 \\
        --val-split 0.15 \\
        --test-split 0.10

Expected raw layout:
    raw-dir/
    ├── images/        *.png  (1024×1024)
    └── annotations/   *.geojson

Output layout:
    out-dir/
    ├── train/
    │   ├── images/    *.png
    │   ├── masks/     *.npy   (int32 instance mask)
    │   └── labels/    *.npy   (int list: class per instance)
    ├── val/
    └── test/
    data/splits/split.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from puma_seg.data.geojson_parser import parse_geojson
from puma_seg.utils.io_utils import list_image_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PUMA GeoJSON to instance masks.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--track", type=int, choices=[1, 2], default=1)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-ext", type=str, default=".png",
        help="Extension of image files (default: .png)."
    )
    return parser.parse_args()


def make_split(
    stems: list[str],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict:
    """Split file stems into train / val / test."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(stems)).tolist()

    n_test = max(1, int(len(stems) * test_frac))
    n_val = max(1, int(len(stems) * val_frac))

    test_idx = indices[:n_test]
    val_idx = indices[n_test: n_test + n_val]
    train_idx = indices[n_test + n_val:]

    return {
        "train": [stems[i] for i in train_idx],
        "val":   [stems[i] for i in val_idx],
        "test":  [stems[i] for i in test_idx],
    }


def process_one(
    image_path: Path,
    ann_dir: Path,
    out_dir: Path,
    track: int,
    image_ext: str,
) -> bool:
    """Convert one sample: copy image, save instance mask + label array."""
    import cv2

    ann_path = ann_dir / (image_path.stem + ".geojson")
    if not ann_path.exists():
        logger.warning("Annotation not found for %s — skipping.", image_path.stem)
        return False

    # Load image to get shape
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Cannot read image %s — skipping.", image_path)
        return False
    h, w = img.shape[:2]

    # Parse annotation
    instance_mask, _, instance_classes = parse_geojson(ann_path, (h, w), track=track)

    # Copy image
    img_out = out_dir / "images" / (image_path.stem + image_ext)
    img_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, img_out)

    # Save masks
    mask_out = out_dir / "masks" / (image_path.stem + ".npy")
    mask_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(mask_out), instance_mask)

    # Save per-instance class labels
    label_out = out_dir / "labels" / (image_path.stem + ".npy")
    label_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(label_out), np.array(instance_classes, dtype=np.int32))

    return True


def main() -> None:
    args = parse_args()

    img_dir = args.raw_dir / "images"
    ann_dir = args.raw_dir / "annotations"

    if not img_dir.exists():
        logger.error("Image directory not found: %s", img_dir)
        sys.exit(1)
    if not ann_dir.exists():
        logger.error("Annotation directory not found: %s", ann_dir)
        sys.exit(1)

    image_paths = list_image_paths(img_dir, extensions=(args.image_ext,))
    if not image_paths:
        logger.error("No images found in %s with extension '%s'.", img_dir, args.image_ext)
        sys.exit(1)

    logger.info("Found %d images.", len(image_paths))

    # Build split
    stems = [p.stem for p in image_paths]
    split = make_split(stems, args.val_split, args.test_split, args.seed)

    # Save split file
    args.split_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.split_dir / "split.json"
    with open(split_path, "w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=2)
    logger.info(
        "Split: %d train / %d val / %d test → %s",
        len(split["train"]), len(split["val"]), len(split["test"]),
        split_path,
    )

    # Process each split
    stem_to_path = {p.stem: p for p in image_paths}
    stats = {"ok": 0, "skipped": 0}

    for subset in ("train", "val", "test"):
        out_subset = args.out_dir / subset
        logger.info("Processing subset: %s (%d images)", subset, len(split[subset]))
        for stem in tqdm(split[subset], desc=subset):
            img_path = stem_to_path.get(stem)
            if img_path is None:
                stats["skipped"] += 1
                continue
            ok = process_one(img_path, ann_dir, out_subset, args.track, args.image_ext)
            if ok:
                stats["ok"] += 1
            else:
                stats["skipped"] += 1

    logger.info(
        "Done. %d processed, %d skipped. Output: %s",
        stats["ok"], stats["skipped"], args.out_dir,
    )


if __name__ == "__main__":
    main()
