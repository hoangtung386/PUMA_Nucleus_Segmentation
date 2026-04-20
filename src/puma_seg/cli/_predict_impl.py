"""Shared predict implementation used by script and CLI entry point."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch

from puma_seg.data.geojson_parser import extract_nucleus_crops, get_class_names
from puma_seg.data.transforms import get_crop_val_transforms
from puma_seg.models.cellpose_wrapper import CellposeSegmentor
from puma_seg.models.nucleus_classifier import NucleusClassifier
from puma_seg.utils.io_utils import list_image_paths
from puma_seg.utils.visualization import overlay_instances

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("puma.predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PUMA nucleus prediction.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Path to a single input image.")
    group.add_argument("--image-dir", type=Path, help="Directory of images to process.")
    parser.add_argument("--seg-model", type=str, default="cyto3", help="Cellpose model name or path.")
    parser.add_argument(
        "--cls-model", type=Path, default=None, help="NucleusClassifier checkpoint (.pth)."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--diameter", type=float, default=17.0)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--track", type=int, choices=[1, 2], default=1)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--save-overlay", action="store_true", help="Save color-coded overlays.")
    return parser.parse_args()


def predict_single(
    image_path: Path,
    segmentor: CellposeSegmentor,
    classifier: Optional[NucleusClassifier],
    cls_transform: Optional[Any],
    args: argparse.Namespace,
    device: str,
) -> Dict[str, Any]:
    """Run full segmentation + classification pipeline on one image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        logger.error("Cannot read: %s", image_path)
        return {}
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pred_mask, _ = segmentor.predict(
        image, diameter=args.diameter, flow_threshold=args.flow_threshold
    )
    logger.info("%s: %d nuclei detected.", image_path.name, pred_mask.max())

    pred_classes: Dict[int, str] = {}
    class_names = get_class_names(args.track)
    if classifier is not None and pred_mask.max() > 0 and cls_transform is not None:
        crops, inst_ids = extract_nucleus_crops(image, pred_mask, crop_size=args.crop_size)
        if crops:
            tensors = torch.stack([cls_transform(image=crop)["image"] for crop in crops]).to(device)
            with torch.no_grad():
                logits = classifier(tensors)
                preds = logits.argmax(dim=1).cpu().tolist()
            pred_classes = {
                instance_id: class_names[pred_idx + 1]
                for instance_id, pred_idx in zip(inst_ids, preds)
            }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    np.save(str(output_dir / f"{stem}_mask.npy"), pred_mask)

    result = {
        "n_nuclei": int(pred_mask.max()),
        "classes": pred_classes,
    }
    with (output_dir / f"{stem}_result.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    if args.save_overlay:
        class_mask = np.zeros_like(pred_mask, dtype=np.uint8)
        for instance_id, class_name in pred_classes.items():
            class_id = class_names.index(class_name) if class_name in class_names else 0
            class_mask[pred_mask == instance_id] = class_id
        overlay = overlay_instances(image, pred_mask, class_mask, track=args.track)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"{stem}_overlay.png"), overlay_bgr)

    return result


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    image_paths = [args.image] if args.image else list_image_paths(args.image_dir)
    logger.info("Running inference on %d image(s)...", len(image_paths))
    for image_path in image_paths:
        predict_single(image_path, segmentor, classifier, cls_transform, args, device)
    logger.info("Done. Results saved to: %s", args.output_dir)
