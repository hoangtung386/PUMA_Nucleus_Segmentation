"""Shared evaluate implementation used by script and CLI entry point."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from puma_seg.data.geojson_parser import (
    NUM_CLASSES,
    extract_nucleus_crops,
    get_class_names,
    get_nucleus_centroids,
)
from puma_seg.data.transforms import get_crop_val_transforms
from puma_seg.evaluation.metrics import evaluate_predictions
from puma_seg.models.cellpose_wrapper import CellposeSegmentor
from puma_seg.models.nucleus_classifier import NucleusClassifier
from puma_seg.utils.io_utils import list_image_paths, load_mask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("puma.evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PUMA pipeline on a data split.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--seg-model", type=str, default="cyto3", help="Cellpose model name or checkpoint path."
    )
    parser.add_argument(
        "--cls-model", type=Path, default=None, help="NucleusClassifier checkpoint path (.pth)."
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument(
        "--threshold", type=float, default=15.0, help="Centroid distance threshold for matching."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    data_cfg = cfg["data"]
    seg_cfg = cfg["segmentation"]
    track = data_cfg["track"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    segmentor = CellposeSegmentor(
        pretrained_model=args.seg_model,
        gpu=(device == "cuda"),
        diameter=seg_cfg.get("diameter"),
    )

    classifier = None
    cls_transform = None
    if args.cls_model and args.cls_model.exists():
        classifier = NucleusClassifier.load(args.cls_model)
        classifier.eval().to(device)
        cls_transform = get_crop_val_transforms(data_cfg.get("crop_size", 64))

    processed_dir = Path(data_cfg["processed_dir"])
    image_dir = processed_dir / args.split / "images"
    mask_dir = processed_dir / args.split / "masks"
    label_dir = processed_dir / args.split / "labels"
    image_paths = list_image_paths(image_dir)
    logger.info("Evaluating %d images (split=%s)...", len(image_paths), args.split)

    foreground_names = get_class_names(track)[1:]
    n_classes = NUM_CLASSES[track] - 1
    pred_list = []
    gt_list = []

    for img_path in tqdm(image_paths, desc="Evaluating"):
        gt_mask_path = mask_dir / f"{img_path.stem}.npy"
        gt_label_path = label_dir / f"{img_path.stem}.npy"
        if not gt_mask_path.exists():
            continue

        gt_mask = load_mask(gt_mask_path)
        gt_instance_labels = {}
        if gt_label_path.exists():
            inst_cls = np.load(str(gt_label_path))
            gt_instance_labels = {i + 1: int(inst_cls[i]) for i in range(len(inst_cls))}
        gt_centroids = get_nucleus_centroids(gt_mask)

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred_mask, info = segmentor.predict(image, channels=seg_cfg.get("channels"), flow_threshold=0.4)
        pred_centroids = info["centroids"]

        pred_classes = {}
        if classifier is not None and pred_mask.max() > 0 and cls_transform is not None:
            crops, inst_ids = extract_nucleus_crops(image, pred_mask, crop_size=data_cfg.get("crop_size", 64))
            if crops:
                tensors = torch.stack([cls_transform(image=c)["image"] for c in crops]).to(device)
                with torch.no_grad():
                    logits = classifier(tensors)
                    class_preds = logits.argmax(dim=1).cpu().tolist()
                pred_classes = {
                    inst_id: cls_pred + 1 for inst_id, cls_pred in zip(inst_ids, class_preds)
                }
        else:
            pred_classes = {instance_id: 1 for instance_id in pred_centroids}

        pred_list.append({"centroids": pred_centroids, "classes": pred_classes})
        gt_list.append({"centroids": gt_centroids, "classes": gt_instance_labels})

    metrics = evaluate_predictions(
        pred_list,
        gt_list,
        n_classes=n_classes,
        class_names=foreground_names,
        threshold=args.threshold,
    )
    logger.info("\n%s", "=" * 50)
    logger.info("PUMA Evaluation — split: %s", args.split)
    logger.info("%s", "=" * 50)
    for key, value in metrics.items():
        logger.info("  %-25s %.4f", key, value)
    logger.info("%s", "=" * 50)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"metrics_{args.split}.json"
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    logger.info("Metrics saved to: %s", out_path)
