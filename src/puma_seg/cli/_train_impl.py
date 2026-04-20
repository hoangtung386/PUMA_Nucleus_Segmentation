"""Shared train implementation used by script and CLI entry point."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("puma.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PUMA nucleus segmentation trainer.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--mode",
        choices=["segmentation", "classification", "both"],
        default="both",
        help="Training mode.",
    )
    parser.add_argument(
        "--resume-seg",
        type=Path,
        default=None,
        help="Path to a pre-trained Cellpose checkpoint to resume from.",
    )
    parser.add_argument(
        "--resume-cls",
        type=Path,
        default=None,
        help="Path to a NucleusClassifier checkpoint to resume training from.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 2 iterations with synthetic data to verify the pipeline.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _dry_run_data(n: int = 4) -> Tuple[list[Any], list[Any]]:
    """Generate tiny synthetic arrays for dry-run validation."""
    import numpy as np

    images = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(n)]
    masks = []
    for _ in range(n):
        mask = np.zeros((256, 256), dtype=np.int32)
        mask[50:80, 50:80] = 1
        mask[150:180, 150:180] = 2
        masks.append(mask)
    return images, masks


def run_segmentation(cfg: Dict[str, Any], args: argparse.Namespace) -> str:
    """Fine-tune Cellpose and return saved model path."""
    from puma_seg.data.dataset import PUMASegmentationDataset
    from puma_seg.models.cellpose_wrapper import CellposeSegmentor
    from puma_seg.training.trainer import SegmentationTrainer

    seg_cfg = cfg["segmentation"]
    data_cfg = cfg["data"]
    paths_cfg = cfg["paths"]

    pretrained = str(args.resume_seg) if args.resume_seg else seg_cfg["pretrained_model"]
    segmentor = CellposeSegmentor(
        pretrained_model=pretrained,
        gpu=seg_cfg.get("gpu", True),
        diameter=seg_cfg.get("diameter"),
        nchan=seg_cfg.get("nchan", 2),
    )

    if args.dry_run:
        logger.info("[DRY RUN] Generating synthetic segmentation data...")
        train_images, train_labels = _dry_run_data(4)
        val_images, val_labels = _dry_run_data(2)
        n_epochs = 2
    else:
        logger.info("Loading segmentation datasets...")
        train_ds = PUMASegmentationDataset(
            data_cfg["processed_dir"],
            split="train",
            track=data_cfg["track"],
            image_size=data_cfg.get("image_size"),
        )
        val_ds = PUMASegmentationDataset(
            data_cfg["processed_dir"],
            split="val",
            track=data_cfg["track"],
            image_size=data_cfg.get("image_size"),
        )
        train_images, train_labels = train_ds.get_all_numpy()
        val_images, val_labels = val_ds.get_all_numpy()
        n_epochs = seg_cfg["n_epochs"]

    trainer = SegmentationTrainer(
        segmentor=segmentor,
        save_dir=paths_cfg["save_dir"],
        log_dir=paths_cfg["log_dir"],
        model_name=seg_cfg["model_name"],
    )
    result = trainer.train(
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        learning_rate=seg_cfg["learning_rate"],
        weight_decay=seg_cfg["weight_decay"],
        n_epochs=n_epochs,
        batch_size=seg_cfg["batch_size"],
        channels=seg_cfg.get("channels"),
    )
    logger.info("Segmentation training done. Model: %s", result["model_path"])
    return result["model_path"]


def run_classification(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """Train NucleusClassifier on extracted nucleus crops."""
    from torch.utils.data import DataLoader, TensorDataset

    from puma_seg.data.dataset import PUMAClassificationDataset
    from puma_seg.models.nucleus_classifier import NucleusClassifier, build_classifier
    from puma_seg.training.trainer import ClassificationTrainer

    cls_cfg = cfg["classification"]
    data_cfg = cfg["data"]
    paths_cfg = cfg["paths"]
    track = data_cfg["track"]
    device = "cuda" if (cls_cfg.get("use_amp", True) and torch.cuda.is_available()) else "cpu"

    if args.resume_cls:
        logger.info("Resuming classifier from: %s", args.resume_cls)
        model = NucleusClassifier.load(args.resume_cls)
    else:
        model = build_classifier(
            track=track,
            pretrained=cls_cfg.get("pretrained", True),
            freeze_backbone=cls_cfg.get("freeze_backbone", True),
            dropout=cls_cfg.get("dropout", 0.3),
        )

    if args.dry_run:
        logger.info("[DRY RUN] Using synthetic classification data...")
        n_classes = 3 if track == 1 else 10
        dummy_x = torch.randn(8, 3, 64, 64)
        dummy_y = torch.randint(0, n_classes, (8,))
        ds = TensorDataset(dummy_x, dummy_y)
        train_loader = DataLoader(ds, batch_size=4)
        val_loader = DataLoader(ds, batch_size=4)
        class_weights = None
        phase1_epochs, phase2_epochs = 1, 1
    else:
        logger.info("Loading classification datasets...")
        train_ds = PUMAClassificationDataset(
            data_cfg["processed_dir"],
            split="train",
            track=track,
            crop_size=data_cfg.get("crop_size", 64),
            augment=True,
        )
        val_ds = PUMAClassificationDataset(
            data_cfg["processed_dir"],
            split="val",
            track=track,
            crop_size=data_cfg.get("crop_size", 64),
            augment=False,
        )
        class_weights = train_ds.class_weights()
        train_loader = DataLoader(
            train_ds,
            batch_size=cls_cfg.get("batch_size", 64),
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cls_cfg.get("batch_size", 64),
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        phase1_epochs = cls_cfg["phase1_epochs"]
        phase2_epochs = cls_cfg["phase2_epochs"]

    trainer = ClassificationTrainer(
        model=model,
        device=device,
        save_dir=paths_cfg["save_dir"],
        log_dir=paths_cfg["log_dir"],
        track=track,
    )
    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        phase1_epochs=phase1_epochs,
        phase1_lr=cls_cfg.get("phase1_lr", 1e-3),
        phase2_epochs=phase2_epochs,
        phase2_lr_head=cls_cfg.get("phase2_lr_head", 1e-4),
        phase2_lr_backbone=cls_cfg.get("phase2_lr_backbone", 1e-6),
        unfreeze_groups=cls_cfg.get("unfreeze_groups", 2),
        weight_decay=cls_cfg.get("weight_decay", 1e-4),
        label_smoothing=cls_cfg.get("label_smoothing", 0.1),
        patience=cls_cfg.get("patience", 15),
        use_amp=cls_cfg.get("use_amp", True),
    )
    logger.info(
        "Classification training done. Best val F1: %.4f → %s",
        result["best_val_f1"],
        result["checkpoint_path"],
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger.info("Config: %s | Mode: %s | Dry-run: %s", args.config, args.mode, args.dry_run)

    if args.mode in ("segmentation", "both"):
        run_segmentation(cfg, args)
    if args.mode in ("classification", "both"):
        run_classification(cfg, args)

    logger.info("All training complete.")
