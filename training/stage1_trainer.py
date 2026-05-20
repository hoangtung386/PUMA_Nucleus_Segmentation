"""Rare-focused Stage 1 training with config-based parameters."""

from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

try:
    import bitsandbytes as bnb
except Exception:
    bnb = None

from configs import PATHS, STAGE1_DEFAULT_CONFIG
from configs.defaults import get_device, linear_ramp
from configs.serialization import make_inference_config_from_stage1
from data.constants import (
    INTERNAL_TISSUE_ID_TO_NAME,
    NUCLEI_CLASS_WEIGHTS,
    NUM_NUCLEI_CLASSES,
    NUM_TISSUE_CLASSES,
    PUMA_NUCLEI_ID_TO_NAME,
    TISSUE_CLASS_WEIGHTS,
)
from data.dataset import PUMADataset, get_train_transforms, get_val_transforms
from models import UnifiedPanopticNet, build_cnn_backbone
from training import safe_torch_save, safe_torch_save_entity
from training.logging_utils import logger
from training.train_loop import train_one_epoch, validate
from utils import MultiTaskUncertaintyLoss, PUMAMetrics
from utils.split_utils import make_or_load_group_split

cfg = STAGE1_DEFAULT_CONFIG


def apply_smooth_schedule(model, criterion, epoch):
    """Apply smooth Stage 1 schedule for rare semantic loss, SC-DFA, and prior."""
    core = model.module if hasattr(model, "module") else model

    focal_weight = linear_ramp(epoch, cfg.focal_start_epoch, cfg.focal_full_epoch, cfg.focal_max_weight)
    sc_dfa_weight = linear_ramp(epoch, cfg.sc_dfa_start_epoch, cfg.sc_dfa_full_epoch, cfg.sc_dfa_max_weight)
    prior_weight = linear_ramp(epoch, cfg.prior_start_epoch, cfg.prior_full_epoch, cfg.prior_max_weight)

    if hasattr(criterion, "set_focal_tversky_weight"):
        criterion.set_focal_tversky_weight(focal_weight)
    else:
        criterion.focal_tversky_weight = focal_weight

    if hasattr(core, "set_sc_dfa_lambda"):
        core.set_sc_dfa_lambda(sc_dfa_weight)
    elif hasattr(core, "enable_sc_dfa"):
        core.enable_sc_dfa(sc_dfa_weight > 0.0)

    if hasattr(core, "set_spatial_prior_lambda"):
        core.set_spatial_prior_lambda(prior_weight)

    logger.info(
        "SmoothSchedule epoch=%03d focal=%.3f sc_dfa=%.3f prior=%.3f",
        epoch, focal_weight, sc_dfa_weight, prior_weight,
    )
    return focal_weight, sc_dfa_weight, prior_weight


def make_inference_config(core_model):
    return make_inference_config_from_stage1(cfg, core_model)


def save_checkpoint(path, model, criterion, optimizer, scheduler, scaler, epoch, best_score, val_report):
    core = model.module if hasattr(model, "module") else model
    payload = {
        "epoch": int(epoch),
        "model_state": core.state_dict(),
        "criterion_state": criterion.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_score": float(best_score),
        "best_val_report": val_report,
        "inference_config": make_inference_config(core),
    }
    safe_torch_save(payload, path)
    entity_path = path.with_name(path.stem + "_full" + path.suffix)
    safe_torch_save_entity(core, entity_path)


def make_rare_weighted_sampler(dataset, indices):
    weights = np.asarray(dataset.compute_sample_weights(indices), dtype=np.float64)
    weights = np.clip(weights, 1.0, cfg.max_sample_weight)
    num_samples = int(round(len(indices) * cfg.samples_per_epoch_multiplier))
    num_samples = max(num_samples, len(indices))
    logger.info(
        "Sampler rare weighted sampler: n_indices=%d num_samples=%d min_w=%.2f mean_w=%.2f max_w=%.2f",
        len(indices), num_samples, weights.min(), weights.mean(), weights.max(),
    )
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )


def print_report(epoch, train_loss, val):
    tissue_names = [INTERNAL_TISSUE_ID_TO_NAME[i] for i in range(NUM_TISSUE_CLASSES)]
    nuclei_names = [PUMA_NUCLEI_ID_TO_NAME[i] for i in range(NUM_NUCLEI_CLASSES)]
    logger.info("=" * 88)
    logger.info(
        "Epoch %03d | train_loss=%.4f | val_loss=%.4f | selection=%.4f | rare=%.4f",
        epoch, train_loss, val.get("val_loss", 0), val.get("selection_score", 0), val.get("rare_macro_dice", 0),
    )
    logger.info("-" * 88)
    logger.info("Tissue loss=%.4f", val.get("loss_tissue", 0))
    for i, name in enumerate(tissue_names):
        logger.info("  %d: %-22s dice=%.4f iou=%.4f", i, name, val.get(f"tissue_dice_{i}", 0), val.get(f"tissue_iou_{i}", 0))
    logger.info("Nuclei loss=%.4f", val.get("loss_nc", 0))
    for i, name in enumerate(nuclei_names):
        logger.info("  %d: %-22s dice=%.4f iou=%.4f", i, name, val.get(f"nuclei_dice_{i}", 0), val.get(f"nuclei_iou_{i}", 0))
    logger.info("NP loss=%.4f | HV loss=%.4f", val.get("loss_np", 0), val.get("loss_hv", 0))
    logger.info(
        "avg_tissue=%.4f | avg_nuclei=%.4f | rare_tissue=%.4f | rare_nuclei=%.4f",
        val.get("avg_tissue_dice", 0), val.get("avg_nuclei_dice", 0),
        val.get("rare_tissue_macro_dice", 0), val.get("rare_nuclei_macro_dice", 0),
    )
    logger.info("=" * 88)


def main():
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = get_device()
    torch.backends.cudnn.benchmark = True
    PATHS.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Root: %s", PATHS.root)
    logger.info("Data: %s", PATHS.data_dir)
    logger.info("Checkpoints: %s", PATHS.checkpoint_dir)
    logger.info("Config: batch_size=%d epochs=%d zero_cellpose_prob=%.2f", cfg.batch_size, cfg.epochs, cfg.zero_cellpose_prob)

    train_ds = PUMADataset(
        PATHS.data_dir,
        transforms=get_train_transforms(cfg.image_size),
        zero_cellpose_prob=cfg.zero_cellpose_prob,
    )
    val_ds = PUMADataset(
        PATHS.data_dir,
        transforms=get_val_transforms(cfg.image_size),
        zero_cellpose_prob=0.0,
    )

    split_meta = train_ds.get_split_metadata()
    train_idx, val_idx = make_or_load_group_split(
        source_names=split_meta["source_names"],
        is_original=split_meta["is_original"],
        split_path=PATHS.split_file,
        seed=cfg.seed,
        train_fraction=1.0 - cfg.val_ratio,
        force_new=cfg.force_new_split,
        val_original_only=cfg.val_original_only,
    )
    logger.info("Split: train=%d val=%d file=%s", len(train_idx), len(val_idx), PATHS.split_file)
    logger.info("Split: Leakage-safe; validation uses originals only")

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=cfg.batch_size,
        sampler=make_rare_weighted_sampler(train_ds, train_idx),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    cnn = build_cnn_backbone(pretrained=True)
    model = UnifiedPanopticNet(
        vit_model=PATHS.uni_weight_dir,
        cnn_model=cnn,
        num_tissue=NUM_TISSUE_CLASSES,
        num_nuclei=NUM_NUCLEI_CLASSES,
        load_uni_weights=True,
    ).to(device)

    if cfg.multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    criterion = MultiTaskUncertaintyLoss(
        tissue_weights=torch.tensor(TISSUE_CLASS_WEIGHTS, dtype=torch.float32),
        nuclei_weights=torch.tensor(NUCLEI_CLASS_WEIGHTS, dtype=torch.float32),
    ).to(device)

    params = list(model.parameters()) + list(criterion.parameters())

    if bnb is not None and device.type == "cuda":
        try:
            optimizer = bnb.optim.AdamW8bit(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        except Exception:
            optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs * max(len(train_loader), 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.use_fp16)
    metrics = PUMAMetrics()

    best_score = -1.0
    best_epoch = 0
    best_val_report = None
    start_epoch = 1

    if cfg.resume is not None:
        resume_path = Path(cfg.resume)
        if resume_path.is_file():
            logger.info("Resuming from checkpoint: %s", resume_path)
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

            core = model.module if hasattr(model, "module") else model
            core.load_state_dict(checkpoint["model_state"])

            if "criterion_state" in checkpoint and checkpoint["criterion_state"] is not None:
                criterion.load_state_dict(checkpoint["criterion_state"], strict=False)

            if "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state"])

            if "scheduler_state" in checkpoint and checkpoint["scheduler_state"] is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state"])

            if "scaler_state" in checkpoint and checkpoint["scaler_state"] is not None:
                scaler.load_state_dict(checkpoint["scaler_state"])

            start_epoch = checkpoint["epoch"] + 1
            best_score = checkpoint.get("best_score", -1.0)
            best_val_report = checkpoint.get("best_val_report", None)

            logger.info("Resumed successfully. Starting from epoch %d, current best score: %.4f", start_epoch, best_score)
        else:
            logger.warning("Resume path %s does not exist. Starting from scratch.", resume_path)

    for epoch in range(start_epoch, cfg.epochs + 1):
        core = model.module if hasattr(model, "module") else model

        apply_smooth_schedule(model, criterion, epoch)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            device,
            scaler,
            epoch,
        )
        val = validate(model, val_loader, criterion, metrics, device, epoch)
        print_report(epoch, train_loss, val)

        score = float(val.get("selection_score", -val.get("val_loss", 1e9)))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val_report = dict(val)
            save_path = PATHS.checkpoint_dir / "puma_epoch_best_s1.pth"
            save_checkpoint(save_path, model, criterion, optimizer, scheduler, scaler, epoch, best_score, best_val_report)
            logger.info("Saved best checkpoint: %s | epoch=%d score=%.4f", save_path, best_epoch, best_score)

        last_path = PATHS.checkpoint_dir / "puma_epoch_last_s1.pth"
        save_checkpoint(last_path, model, criterion, optimizer, scheduler, scaler, epoch, best_score, best_val_report)

    logger.info("=" * 88)
    logger.info("Stage 1 complete. Best epoch: %d", best_epoch)
    logger.info("Best selection score: %.4f", best_score)
    if best_val_report is not None:
        logger.info("Best rare macro dice: %.4f", best_val_report.get("rare_macro_dice", 0))
        logger.info("Best rare tissue dice: %.4f", best_val_report.get("rare_tissue_macro_dice", 0))
        logger.info("Best rare nuclei dice: %.4f", best_val_report.get("rare_nuclei_macro_dice", 0))
    logger.info("Best checkpoint: %s", PATHS.checkpoint_dir / "puma_epoch_best_s1.pth")
