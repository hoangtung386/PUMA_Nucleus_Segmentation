"""Rare-focused Stage 2 residual nuclei refiner training."""

import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

try:
    import bitsandbytes as bnb
except Exception:
    bnb = None

from configs import PATHS, STAGE2_DEFAULT_CONFIG
from configs.defaults import get_device, linear_ramp
from data.constants import PUMA_NUCLEI_ID_TO_NAME
from data.dataset import PUMADataset, get_train_transforms, get_val_transforms
from models import (
    ResidualNucleiRefinerUNet,
    UnifiedPanopticNet,
    build_cnn_backbone,
    build_stage2_input,
)
from training.checkpoint import extract_state_dict, load_large_checkpoint, safe_torch_save, safe_torch_save_entity
from training.logging_utils import logger
from utils import PUMAMetrics
from utils.losses import FocalTverskyLoss, SafeCrossEntropyLoss
from utils.split_utils import make_or_load_group_split

cfg = STAGE2_DEFAULT_CONFIG


def alpha_schedule(epoch):
    return linear_ramp(epoch, 0, cfg.alpha_warmup_epochs, cfg.alpha_end) + cfg.alpha_start


def keep_lambda_schedule(epoch):
    return linear_ramp(epoch, 0, cfg.keep_lambda_decay_epochs, cfg.keep_lambda_end - cfg.keep_lambda_start) + cfg.keep_lambda_start


def masked_kl_preservation_loss(refined_logits, s1_logits, targets, temperature=2.0, ignore_index=255):
    valid = targets != ignore_index
    if not torch.any(valid):
        return refined_logits.sum() * 0.0
    log_p_refined = F.log_softmax(refined_logits / temperature, dim=1)
    p_s1 = F.softmax(s1_logits.detach() / temperature, dim=1)
    kl = F.kl_div(log_p_refined, p_s1, reduction="none").sum(dim=1)
    return kl[valid].mean() * (temperature ** 2)


def make_rare_weighted_sampler(dataset, indices):
    weights = np.asarray(dataset.compute_sample_weights(indices), dtype=np.float64)
    weights = np.clip(weights, 1.0, cfg.max_sample_weight)
    num_samples = int(round(len(indices) * cfg.samples_per_epoch_multiplier))
    num_samples = max(num_samples, len(indices))
    logger.info(
        "Sampler Stage 2 rare weighted sampler: n_indices=%d num_samples=%d min_w=%.2f mean_w=%.2f max_w=%.2f",
        len(indices), num_samples, weights.min(), weights.mean(), weights.max(),
    )
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )


def compute_stage2_scores(metrics_calc, s1_metrics, s2_metrics):
    out = {}
    s1_dice = []
    s2_dice = []
    s1_rare = []
    s2_rare = []

    for k in range(cfg.num_nuclei_classes):
        s1_val = s1_metrics.get(f"s1_nuclei_dice_{k}", math.nan)
        s2_val = s2_metrics.get(f"s2_nuclei_dice_{k}", math.nan)
        out[f"s1_nuclei_dice_{k}"] = s1_val
        out[f"s1_nuclei_iou_{k}"] = s1_metrics.get(f"s1_nuclei_iou_{k}", math.nan)
        out[f"s2_nuclei_dice_{k}"] = s2_val
        out[f"s2_nuclei_iou_{k}"] = s2_metrics.get(f"s2_nuclei_iou_{k}", math.nan)
        s1_dice.append(s1_val)
        s2_dice.append(s2_val)

    for k in cfg.rare_nuclei_ids:
        s1_rare.append(s1_metrics.get(f"s1_nuclei_dice_{k}", math.nan))
        s2_rare.append(s2_metrics.get(f"s2_nuclei_dice_{k}", math.nan))

    s1_macro = metrics_calc._nanmean(s1_dice)
    s2_macro = metrics_calc._nanmean(s2_dice)
    s1_rare_macro = metrics_calc._nanmean(s1_rare)
    s2_rare_macro = metrics_calc._nanmean(s2_rare)

    out["s1_macro_dice"] = s1_macro
    out["s2_macro_dice"] = s2_macro
    out["s1_rare_macro_dice"] = s1_rare_macro
    out["s2_rare_macro_dice"] = s2_rare_macro
    out["selection_score"] = 0.25 * metrics_calc._nan_to_zero(s2_macro) + 0.75 * metrics_calc._nan_to_zero(s2_rare_macro)
    out["improvement_score"] = 0.25 * (metrics_calc._nan_to_zero(s2_macro) - metrics_calc._nan_to_zero(s1_macro)) + 0.75 * (metrics_calc._nan_to_zero(s2_rare_macro) - metrics_calc._nan_to_zero(s1_rare_macro))
    out["beats_stage1"] = out["improvement_score"] > 0.0
    return out


def fmt(v):
    try:
        v = float(v)
    except Exception:
        return "N/A"
    return "N/A" if math.isnan(v) else f"{v:.4f}"


def print_report(epoch, train_loss, val_loss, results):
    logger.info("=" * 92)
    logger.info("Stage 2 epoch %03d | train=%.4f val=%.4f alpha=%.3f keep=%.3f", epoch, train_loss, val_loss, results["alpha"], results["keep_lambda"])
    logger.info("-" * 92)
    for k in range(cfg.num_nuclei_classes):
        name = PUMA_NUCLEI_ID_TO_NAME[k]
        s1 = results.get(f"s1_nuclei_dice_{k}")
        s2 = results.get(f"s2_nuclei_dice_{k}")
        delta = math.nan if s1 is None or s2 is None or math.isnan(float(s1)) or math.isnan(float(s2)) else float(s2) - float(s1)
        logger.info("%02d %-22s S1=%-8s S2=%-8s Delta=%-8s", k, name, fmt(s1), fmt(s2), fmt(delta))
    logger.info("S1 macro=%s | S2 macro=%s", fmt(results.get("s1_macro_dice")), fmt(results.get("s2_macro_dice")))
    logger.info("S1 rare =%s | S2 rare =%s", fmt(results.get("s1_rare_macro_dice")), fmt(results.get("s2_rare_macro_dice")))
    logger.info("selection=%s improvement=%s beats_stage1=%s", fmt(results.get("selection_score")), fmt(results.get("improvement_score")), results.get("beats_stage1"))
    logger.info("=" * 92)


def main():
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = get_device()
    torch.backends.cudnn.benchmark = True
    PATHS.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    stage1_ckpt_path = PATHS.checkpoint_dir / "puma_epoch_best_s1.pth"

    logger.info("Root: %s", PATHS.root)
    logger.info("Data: %s", PATHS.data_dir)
    logger.info("Stage 1 checkpoint: %s", stage1_ckpt_path)
    logger.info("Checkpoints: %s", PATHS.checkpoint_dir)

    train_ds = PUMADataset(PATHS.data_dir, transforms=get_train_transforms(cfg.image_size), zero_cellpose_prob=0.0)
    val_ds = PUMADataset(PATHS.data_dir, transforms=get_val_transforms(cfg.image_size), zero_cellpose_prob=0.0)

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

    model_s1 = UnifiedPanopticNet(
        vit_model=PATHS.uni_weight_dir,
        cnn_model=build_cnn_backbone(pretrained=False),
        num_tissue=5,
        num_nuclei=10,
        load_uni_weights=False,
    ).to(device)

    ckpt_s1 = load_large_checkpoint(stage1_ckpt_path, device)
    model_s1.load_state_dict(extract_state_dict(ckpt_s1), strict=True)

    cfg_s1 = ckpt_s1.get("inference_config", {}) if isinstance(ckpt_s1, dict) else {}
    model_s1.enable_sc_dfa(bool(cfg_s1.get("use_sc_dfa", True)))
    model_s1.set_spatial_prior_lambda(float(cfg_s1.get("lambda_prior", 1.0)))
    model_s1.eval()
    for p in model_s1.parameters():
        p.requires_grad = False

    model_s2 = ResidualNucleiRefinerUNet(
        in_channels=cfg.stage2_in_channels,
        out_classes=cfg.num_nuclei_classes,
    ).to(device)

    class_weights = torch.tensor(cfg.nuclei_weights, dtype=torch.float32, device=device)
    ce_loss = SafeCrossEntropyLoss(weight=class_weights, ignore_index=cfg.ignore_index).to(device)
    ft_loss = FocalTverskyLoss(
        alpha=0.20,
        beta=0.80,
        gamma=1.60,
        class_weights=class_weights,
        ignore_index=cfg.ignore_index,
    ).to(device)

    if bnb is not None and device.type == "cuda":
        try:
            optimizer = bnb.optim.AdamW8bit(model_s2.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        except Exception:
            optimizer = optim.AdamW(model_s2.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = optim.AdamW(model_s2.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.use_fp16)
    metrics_calc = PUMAMetrics()

    best_score = -1.0
    best_epoch = 0
    best_improvement = -999.0

    if cfg.resume is not None:
        logger.warning("Resume path is set to %s, but resume is not fully implemented in click-to-run mode.", cfg.resume)

    for epoch in range(1, cfg.epochs + 1):
        alpha = alpha_schedule(epoch)
        keep_lambda = keep_lambda_schedule(epoch)
        model_s2.train()
        train_loss_sum = 0.0

        for batch in tqdm(train_loader, desc=f"Train Stage2 {epoch:03d}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            targets_nc = batch["nuclei_nc"].to(device, non_blocking=True)
            cellpose_flows = batch["cellpose_flow"].to(device, non_blocking=True)
            site_types = batch.get("site_type") or [cfg.default_site_type] * images.shape[0]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda" and cfg.use_fp16):
                with torch.no_grad():
                    preds_s1 = model_s1(images, cellpose_flows, site_types)
                    s1_nc_logits = preds_s1["nc"].detach()
                    s2_input = build_stage2_input(images, preds_s1)
                delta_nc = model_s2(s2_input)
                refined_nc = s1_nc_logits + alpha * delta_nc
                loss_ce = ce_loss(refined_nc, targets_nc)
                loss_ft = ft_loss(refined_nc, targets_nc)
                loss_keep = masked_kl_preservation_loss(
                    refined_nc,
                    s1_nc_logits,
                    targets_nc,
                    temperature=cfg.kd_temperature,
                    ignore_index=cfg.ignore_index,
                )
                loss = loss_ce + loss_ft + keep_lambda * loss_keep

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_s2.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.detach().item())

        scheduler.step()
        avg_train_loss = train_loss_sum / max(len(train_loader), 1)

        model_s2.eval()
        val_loss_sum = 0.0
        s1_acc = metrics_calc.new_semantic_accumulator(cfg.num_nuclei_classes, "s1_nuclei", ignore_index=cfg.ignore_index, device=device)
        s2_acc = metrics_calc.new_semantic_accumulator(cfg.num_nuclei_classes, "s2_nuclei", ignore_index=cfg.ignore_index, device=device)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Valid Stage2 {epoch:03d}", leave=False):
                images = batch["image"].to(device, non_blocking=True)
                targets_nc = batch["nuclei_nc"].to(device, non_blocking=True)
                cellpose_flows = batch["cellpose_flow"].to(device, non_blocking=True)
                site_types = batch.get("site_type") or [cfg.default_site_type] * images.shape[0]

                with torch.amp.autocast("cuda", enabled=device.type == "cuda" and cfg.use_fp16):
                    preds_s1 = model_s1(images, cellpose_flows, site_types)
                    s1_nc_logits = preds_s1["nc"]
                    s2_input = build_stage2_input(images, preds_s1)
                    delta_nc = model_s2(s2_input)
                    refined_nc = s1_nc_logits + alpha * delta_nc
                    loss_ce = ce_loss(refined_nc, targets_nc)
                    loss_ft = ft_loss(refined_nc, targets_nc)
                    loss_keep = masked_kl_preservation_loss(
                        refined_nc,
                        s1_nc_logits,
                        targets_nc,
                        temperature=cfg.kd_temperature,
                        ignore_index=cfg.ignore_index,
                    )
                    val_loss = loss_ce + loss_ft + keep_lambda * loss_keep

                val_loss_sum += float(val_loss.detach().item())
                s1_acc.update(s1_nc_logits, targets_nc)
                s2_acc.update(refined_nc, targets_nc)

        avg_val_loss = val_loss_sum / max(len(val_loader), 1)
        results = compute_stage2_scores(metrics_calc, s1_acc.compute(), s2_acc.compute())
        results["alpha"] = alpha
        results["keep_lambda"] = keep_lambda
        print_report(epoch, avg_train_loss, avg_val_loss, results)

        ckpt_payload = {
            "model_state": model_s2.state_dict(),
            "epoch": epoch,
            "alpha": alpha,
            "keep_lambda": keep_lambda,
            "selection_score": results["selection_score"],
            "improvement_score": results["improvement_score"],
            "beats_stage1": results["beats_stage1"],
            "config": {
                "in_channels": cfg.stage2_in_channels,
                "out_classes": cfg.num_nuclei_classes,
                "residual": True,
                "alpha_start": cfg.alpha_start,
                "alpha_end": cfg.alpha_end,
                "alpha_warmup_epochs": cfg.alpha_warmup_epochs,
                "kd_temperature": cfg.kd_temperature,
                "keep_lambda_start": cfg.keep_lambda_start,
                "keep_lambda_end": cfg.keep_lambda_end,
                "nuclei_weights": cfg.nuclei_weights,
                "rare_nuclei_ids": cfg.rare_nuclei_ids,
                "uses_5_tissue_probs_no_background": True,
                "stage2_input_channels": cfg.stage2_in_channels,
                "split_is_group_based": True,
                "validation_original_only": cfg.val_original_only,
            },
        }

        if epoch % 5 == 0 or epoch == cfg.epochs:
            last_path = PATHS.checkpoint_dir / "nuclei_refiner_residual_last.pth"
            safe_torch_save(ckpt_payload, last_path)
            safe_torch_save_entity(model_s2, last_path.with_name(last_path.stem + "_full" + last_path.suffix))

        score = float(results["selection_score"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_improvement = float(results["improvement_score"])
            best_path = PATHS.checkpoint_dir / "nuclei_refiner_residual_best.pth"
            safe_torch_save(ckpt_payload, best_path)
            safe_torch_save_entity(model_s2, best_path.with_name(best_path.stem + "_full" + best_path.suffix))
            logger.info("Saved Stage 2 best: epoch=%d score=%.4f improvement=%+.4f", best_epoch, best_score, best_improvement)

        if not results["beats_stage1"]:
            logger.warning("Stage 2 has not beaten Stage 1 yet. For Docker inference, prefer Stage 1-only or validate hybrid before enabling Stage 2.")

    logger.info("=" * 92)
    logger.info("Stage 2 complete. Best epoch: %d", best_epoch)
    logger.info("Best score: %.4f", best_score)
    logger.info("Best improvement over Stage 1: %+.4f", best_improvement)
    logger.info("Best checkpoint: %s", PATHS.checkpoint_dir / "nuclei_refiner_residual_best.pth")
