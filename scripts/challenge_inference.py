"""Grand Challenge entrypoint for PUMA algorithm submissions.

Reads input TIFF images and writes:
  - nuclei detection GeoJSON (Task 2)
  - tissue segmentation TIFF mask (Task 1)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from skimage import measure

from puma_seg.data.geojson_parser import get_class_names
from puma_seg.data.transforms import get_crop_val_transforms
from puma_seg.models.cellpose_wrapper import CellposeSegmentor
from puma_seg.models.nucleus_classifier import NucleusClassifier
from puma_seg.utils.io_utils import list_image_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("puma.challenge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PUMA Grand Challenge inference")
    parser.add_argument("--input-dir", type=Path, default=Path("/input/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("/output/images"))
    parser.add_argument("--seg-model", type=str, default="cyto3")
    parser.add_argument("--cls-model", type=Path, default=None)
    parser.add_argument("--track", type=int, choices=[1, 2], default=1)
    parser.add_argument("--diameter", type=float, default=17.0)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--crop-size", type=int, default=64)
    return parser.parse_args()


def mask_to_geojson(instance_mask: np.ndarray, class_by_instance: Dict[int, str]) -> Dict[str, Any]:
    features: List[Dict[str, Any]] = []
    for instance_id in np.unique(instance_mask):
        if int(instance_id) <= 0:
            continue
        binary = instance_mask == int(instance_id)
        contours = measure.find_contours(binary.astype(np.uint8), 0.5)
        if not contours:
            continue

        largest = max(contours, key=len)
        if len(largest) < 3:
            continue
        ring = [[float(col), float(row)] for row, col in largest]
        if ring[0] != ring[-1]:
            ring.append(ring[0])

        class_name = class_by_instance.get(int(instance_id), "other")
        if not class_name.startswith("nuclei_"):
            class_name = f"nuclei_{class_name.replace(' ', '_')}"
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {"classification": {"name": class_name}},
            }
        )

    return {"type": "FeatureCollection", "features": features}


def write_tissue_placeholder(output_path: Path, shape_hw: tuple[int, int]) -> None:
    """Write a valid TIFF mask for task 1 when no tissue model is configured."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tissue_mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.imwrite(str(output_path), tissue_mask)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = get_class_names(args.track)

    segmentor = CellposeSegmentor(
        pretrained_model=args.seg_model,
        gpu=(device == "cuda"),
        diameter=args.diameter,
    )

    classifier = None
    cls_transform = None
    if args.cls_model and args.cls_model.exists():
        classifier = NucleusClassifier.load(args.cls_model)
        classifier.eval().to(device)
        cls_transform = get_crop_val_transforms(args.crop_size)

    input_images = list_image_paths(args.input_dir, extensions=(".tif", ".tiff", ".png", ".jpg"))
    logger.info("Found %d input images in %s", len(input_images), args.input_dir)

    nuclei_dir = args.output_dir / "melanoma-cell-detection"
    tissue_dir = args.output_dir / "melanoma-tissue-mask-segmentation"
    nuclei_dir.mkdir(parents=True, exist_ok=True)
    tissue_dir.mkdir(parents=True, exist_ok=True)

    for image_path in input_images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Cannot read %s, skipping.", image_path)
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred_mask, _ = segmentor.predict(
            image_rgb,
            diameter=args.diameter,
            flow_threshold=args.flow_threshold,
        )

        pred_classes: Dict[int, str] = {}
        if classifier is not None and pred_mask.max() > 0 and cls_transform is not None:
            from puma_seg.data.geojson_parser import extract_nucleus_crops

            crops, instance_ids = extract_nucleus_crops(image_rgb, pred_mask, crop_size=args.crop_size)
            if crops:
                tensors = torch.stack([cls_transform(image=crop)["image"] for crop in crops]).to(device)
                with torch.no_grad():
                    logits = classifier(tensors)
                    preds = logits.argmax(dim=1).cpu().tolist()
                pred_classes = {
                    instance_id: class_names[pred_idx + 1]
                    for instance_id, pred_idx in zip(instance_ids, preds)
                }

        nuclei_geojson = mask_to_geojson(pred_mask, pred_classes)
        nuclei_path = nuclei_dir / f"{image_path.stem}.json"
        with nuclei_path.open("w", encoding="utf-8") as file:
            json.dump(nuclei_geojson, file)

        tissue_path = tissue_dir / f"{image_path.stem}.tif"
        write_tissue_placeholder(tissue_path, pred_mask.shape)

        logger.info("Wrote %s and %s", nuclei_path.name, tissue_path.name)


if __name__ == "__main__":
    main()
