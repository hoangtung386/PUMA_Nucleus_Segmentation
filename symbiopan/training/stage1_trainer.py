"""Rare-focused Stage 1 training with Virchow2 encoder + ConvNeXt-Tiny backbone."""

import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, Subset, WeightedRandomSampler

try:
    import bitsandbytes as bnb
except Exception:
    bnb = None

from configs import PATHS, STAGE1_DEFAULT_CONFIG
from symbiopan.common.device import get_device
from symbiopan.common.logging import get_logger
from symbiopan.data.constants import (
    NUM_NUCLEI_CLASSES,
    NUM_TISSUE_CLASSES,
    PUMA_NUCLEI_ID_TO_NAME,
    PUMA_TISSUE_ID_TO_NAME,
)
from symbiopan.data.dataset import PUMADataset, get_train_transforms, get_val_transforms
from symbiopan.losses import MultiTaskUncertaintyLoss
from symbiopan.metrics import PUMAMetrics
from symbiopan.models import UnifiedPanopticNet, build_cnn_backbone
from symbiopan.modules.scheduler import build_warmup_cosine_scheduler, linear_ramp
from symbiopan.modules.split import make_or_load_group_split
from symbiopan.training import extract_state_dict, safe_torch_save_entity
from symbiopan.training.train_loop import train_one_epoch, validate

logger = get_logger(__name__)


def apply_smooth_schedule(cfg, model: torch.nn.Module, criterion: torch.nn.Module, epoch: int) -> tuple[float, float]:
    core = model.module if hasattr(model, "module") else model

    focal_weight = linear_ramp(epoch, cfg.focal_start_epoch, cfg.focal_full_epoch, cfg.focal_max_weight)
    sc_dfa_weight = linear_ramp(epoch, cfg.sc_dfa_start_epoch, cfg.sc_dfa_full_epoch, cfg.sc_dfa_max_weight)

    if hasattr(criterion, "set_focal_tversky_weight"):
        criterion.set_focal_tversky_weight(focal_weight)
    else:
        criterion.focal_tversky_weight = focal_weight

    if hasattr(core, "set_sc_dfa_lambda"):
        core.set_sc_dfa_lambda(sc_dfa_weight)
    elif hasattr(core, "enable_sc_dfa"):
        core.enable_sc_dfa(sc_dfa_weight > 0.0)

    logger.info("SmoothSchedule epoch=%03d focal=%.3f sc_dfa=%.3f", epoch, focal_weight, sc_dfa_weight)
    return focal_weight, sc_dfa_weight


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_score: float,
    val_report: dict,
) -> None:
    core = model.module if hasattr(model, "module") else model
    core._metadata = {
        "epoch": int(epoch),
        "best_score": float(best_score),
        "best_val_report": val_report,
        "architecture": "symbiopan_v8_cellpath",
    }
    safe_torch_save_entity(core, path)
    logger.info("Entity model saved: %s | epoch=%d score=%.4f", path, epoch, best_score)


def _build_loader(
    cfg,
    dataset: PUMADataset,
    indices: list[int],
    sampler: Sampler | None = None,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=sampler is not None,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else 2,
    )


def make_rare_weighted_sampler(cfg, dataset: PUMADataset, indices: list[int]) -> WeightedRandomSampler:
    weights = np.asarray(dataset.compute_sample_weights(indices), dtype=np.float64)
    weights = np.clip(weights, 1.0, cfg.max_sample_weight)
    num_samples = int(round(len(indices) * cfg.samples_per_epoch_multiplier))
    num_samples = max(num_samples, len(indices))
    logger.info(
        "Sampler: n_indices=%d num_samples=%d min_w=%.2f mean_w=%.2f max_w=%.2f",
        len(indices),
        num_samples,
        weights.min(),
        weights.mean(),
        weights.max(),
    )
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )


def print_report(cfg, epoch: int, train_loss: float, val: dict) -> None:
    tissue_names = [PUMA_TISSUE_ID_TO_NAME[i] for i in range(NUM_TISSUE_CLASSES)]
    nuclei_names = [PUMA_NUCLEI_ID_TO_NAME[i] for i in range(NUM_NUCLEI_CLASSES)]
    logger.info("=" * 88)
    logger.info(
        "Epoch %03d | train_loss=%.4f | val_loss=%.4f | selection=%.4f | rare=%.4f",
        epoch,
        train_loss,
        val.get("val_loss", 0),
        val.get("selection_score", 0),
        val.get("rare_macro_dice", 0),
    )
    logger.info("-" * 88)
    logger.info("Tissue loss=%.4f", val.get("loss_tissue", 0))
    for i, name in enumerate(tissue_names):
        logger.info(
            "  %d: %-22s dice=%.4f iou=%.4f", i, name, val.get(f"tissue_dice_{i}", 0), val.get(f"tissue_iou_{i}", 0)
        )
    logger.info("Nuclei loss=%.4f", val.get("loss_nc", 0))
    for i, name in enumerate(nuclei_names):
        logger.info(
            "  %d: %-22s dice=%.4f iou=%.4f", i, name, val.get(f"nuclei_dice_{i}", 0), val.get(f"nuclei_iou_{i}", 0)
        )
    logger.info("NP loss=%.4f | HV loss=%.4f", val.get("loss_np", 0), val.get("loss_hv", 0))
    logger.info(
        "avg_tissue=%.4f | avg_nuclei=%.4f | rare_tissue=%.4f | rare_nuclei=%.4f",
        val.get("avg_tissue_dice", 0),
        val.get("avg_nuclei_dice", 0),
        val.get("rare_tissue_macro_dice", 0),
        val.get("rare_nuclei_macro_dice", 0),
    )
    logger.info("=" * 88)


def _optimize_gpu() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    logger.info("GPU optimizations: cudnn.benchmark=True tf32=True matmul_precision=high")


def main(override_cfg=None, test_loader=None) -> dict:
    """Run full Stage 1 training.

    Args:
        override_cfg: Optional Stage1Config to override defaults.
        test_loader: Optional DataLoader for held-out test evaluation.

    Returns:
        dict with keys: history, best_epoch, best_score, best_val_report, test_metrics.
    """
    cfg = override_cfg or STAGE1_DEFAULT_CONFIG
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = get_device()
    _optimize_gpu()
    PATHS.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Root: %s", PATHS.root)
    logger.info("Data: %s", PATHS.data_dir)
    logger.info("Checkpoints: %s", PATHS.checkpoint_dir)
    logger.info("Config: batch_size=%d epochs=%d num_workers=%d", cfg.batch_size, cfg.epochs, cfg.num_workers)

    context_dir = PATHS.raw_dir / "01_training_dataset_tif_context_ROIs" if cfg.use_context_encoder else None
    train_transform = get_train_transforms(cfg.image_size, use_stain_aug=cfg.use_stain_aug)
    train_ds = PUMADataset(
        PATHS.data_dir,
        transforms=train_transform,
        context_dir=context_dir,
        use_context=cfg.use_context_encoder,
    )
    val_ds = PUMADataset(
        PATHS.data_dir,
        transforms=get_val_transforms(cfg.image_size),
        context_dir=context_dir,
        use_context=cfg.use_context_encoder,
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

    train_loader = _build_loader(cfg, train_ds, train_idx, sampler=make_rare_weighted_sampler(cfg, train_ds, train_idx))
    val_loader = _build_loader(cfg, val_ds, val_idx)

    cnn = build_cnn_backbone(pretrained=True)
    model = UnifiedPanopticNet(
        virchow2_model_name=cfg.virchow2_model_name,
        cnn_model=cnn,
        num_tissue=cfg.num_tissue,
        num_nuclei=cfg.num_nuclei,
        num_sites=cfg.num_sites,
        site_embed_dim=cfg.site_embed_dim,
        fine_tune_last_n_blocks=cfg.fine_tune_last_n_blocks,
        load_encoder_weights=True,
        use_context_encoder=cfg.use_context_encoder,
    ).to(device)

    if cfg.multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if cfg.compile_model:
        try:
            model = torch.compile(model, mode="default", fullgraph=False)
            logger.info("Model compiled with torch.compile (default)")
        except Exception as e:
            logger.warning("torch.compile failed (%s); running uncompiled", e)

    criterion = MultiTaskUncertaintyLoss(
        tissue_weights=torch.tensor(cfg.tissue_class_weights, dtype=torch.float32),
        nuclei_weights=torch.tensor(cfg.nuclei_class_weights, dtype=torch.float32),
        loss_multipliers=cfg.loss_multipliers,
        focal_tversky_tissue=cfg.focal_tversky_tissue,
        focal_tversky_nuclei=cfg.focal_tversky_nuclei,
        focal_bce=cfg.focal_bce,
        smooth_l1_beta=cfg.smooth_l1_beta,
    ).to(device)

    params = list(model.parameters()) + list(criterion.parameters())

    if bnb is not None and device.type == "cuda":
        try:
            optimizer = bnb.optim.AdamW8bit(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        except Exception:
            optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=cfg.warmup_epochs,
        total_epochs=cfg.epochs,
        steps_per_epoch=max(len(train_loader), 1),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.use_fp16)
    metrics = PUMAMetrics(selection_score_weights=cfg.selection_score_weights)

    best_score = -1.0
    best_epoch = 0
    best_val_report = None
    start_epoch = 1
    metrics_history: list[dict] = []

    if cfg.resume is not None:
        resume_path = Path(cfg.resume)
        if resume_path.is_file():
            logger.info("Resuming from checkpoint: %s", resume_path)
            obj = torch.load(resume_path, map_location=device, weights_only=False)
            core = model.module if hasattr(model, "module") else model
            sd = obj if isinstance(obj, dict) else obj.state_dict()
            core.load_state_dict(extract_state_dict(sd))
            metadata = (
                obj._metadata
                if isinstance(obj, torch.nn.Module) and hasattr(obj, "_metadata")
                else obj
                if isinstance(obj, dict)
                else {}
            )
            start_epoch = metadata.get("epoch", 0) + 1 if isinstance(metadata, dict) else 1
            best_score = metadata.get("best_score", -1.0) if isinstance(metadata, dict) else -1.0
            best_val_report = metadata.get("best_val_report", None) if isinstance(metadata, dict) else None
            logger.info("Resumed from epoch %d, best score: %.4f", start_epoch, best_score)
        else:
            logger.warning("Resume path %s does not exist. Starting from scratch.", resume_path)

    # Warmup torch.compile graphs (train + eval modes) to avoid lazy-compile lag in epoch 1
    if cfg.compile_model:
        logger.info("Warming up torch.compile graphs...")
        dummy_img = torch.randn(1, 3, 1024, 1024, device=device)
        dummy_site = torch.zeros(1, dtype=torch.long, device=device)
        model.train()
        _ = model(dummy_img, dummy_site)
        model.eval()
        _ = model(dummy_img, dummy_site)
        torch.cuda.synchronize()
        logger.info("torch.compile warmup done")

    for epoch in range(start_epoch, cfg.epochs + 1):
        apply_smooth_schedule(cfg, model, criterion, epoch)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            device,
            scaler,
            epoch,
            grad_accum_steps=cfg.grad_accum_steps,
        )
        val = validate(model, val_loader, criterion, metrics, device, epoch)
        print_report(cfg, epoch, train_loss, val)

        metrics_history.append({"epoch": epoch, "train_loss": train_loss, **val})

        score = float(val.get("selection_score", -val.get("val_loss", 1e9)))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val_report = dict(val)
            save_path = PATHS.checkpoint_dir / "puma_epoch_best_s1.pth"
            save_checkpoint(
                save_path, model, criterion, optimizer, scheduler, scaler, epoch, best_score, best_val_report
            )
            logger.info("Saved best checkpoint: %s | epoch=%d score=%.4f", save_path, best_epoch, best_score)

        last_path = PATHS.checkpoint_dir / "puma_epoch_last_s1.pth"
        save_checkpoint(last_path, model, criterion, optimizer, scheduler, scaler, epoch, best_score, best_val_report)

    # ── Test set evaluation ──
    test_metrics = None
    if test_loader is not None:
        logger.info("=" * 88)
        logger.info("Evaluating best model on held-out test set...")
        best_ckpt_path = PATHS.checkpoint_dir / "puma_epoch_best_s1.pth"
        if best_ckpt_path.exists():
            obj = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            core = model.module if hasattr(model, "module") else model
            sd = obj if isinstance(obj, dict) else obj.state_dict()
            core.load_state_dict(extract_state_dict(sd))
            test_metrics = validate(model, test_loader, criterion, metrics, device, 0)
            logger.info("Test metrics:")
            for k, v in test_metrics.items():
                if isinstance(v, float):
                    logger.info("  %s: %.4f", k, v)

    logger.info("=" * 88)
    logger.info("Stage 1 complete. Best epoch: %d | score: %.4f", best_epoch, best_score)
    if best_val_report is not None:
        logger.info("Best rare macro dice: %.4f", best_val_report.get("rare_macro_dice", 0))

    return {
        "history": metrics_history,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_val_report": best_val_report,
        "test_metrics": test_metrics,
    }
