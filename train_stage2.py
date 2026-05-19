"""
Rare-focused Stage 2 residual nuclei refiner.

Click-to-run. No argparse.
All paths are relative to root = Path.cwd().

Leakage control:
    Uses the exact same group-based split file as Stage 1.
    Train includes rare-centered crops from train source images only.
    Validation uses original validation images only.
"""

import math
import shutil
from pathlib import Path

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

from dataloaders import PUMA_NUCLEI_ID_TO_NAME, PUMADataset, get_train_transforms, get_val_transforms
from models import ResidualNucleiRefinerUNet, UnifiedPanopticNet, build_stage2_input, get_cnn_spatial_prior
from utils import PUMAMetrics
from utils.losses import FocalTverskyLoss, SafeCrossEntropyLoss
from utils.split_utils import make_or_load_group_split


# ============================================================
# Click-to-run config
# ============================================================

root = Path.cwd()

data_dir = root / "dataset_processed"
uni_weight_dir = root
checkpoint_dir = root / "checkpoints"
split_file = checkpoint_dir / "split_seed42.npz"
stage1_ckpt = checkpoint_dir / "puma_epoch_best_s1.pth"

image_size = 1024
batch_size = 16
epochs = 30
num_workers = 2
seed = 42
train_fraction = 0.8
force_new_split = False
val_original_only = True

lr = 1e-4
weight_decay = 1e-4

default_site_type = "metastatic"
use_fp16 = True
resume = None

num_nuclei_classes = 10
stage2_in_channels = 21
ignore_index = 255

rare_nuclei_ids = [2, 4, 5, 8, 9]
nuclei_weights = [0.6, 0.9, 9.0, 2.5, 5.0, 10.0, 2.0, 2.5, 6.0, 10.0]

samples_per_epoch_multiplier = 2.5
kd_temperature = 2.0
keep_lambda_start = 0.80
keep_lambda_end = 0.15
keep_lambda_decay_epochs = 30
alpha_start = 0.05
alpha_end = 0.45
alpha_warmup_epochs = 30


def safe_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if Path("/content").exists():
        local_tmp = Path("/content") / (path.name + ".tmp")
        local_final = Path("/content") / path.name
    else:
        local_tmp = path.with_suffix(path.suffix + ".tmp")
        local_final = path

    torch.save(obj, local_tmp)
    _ = torch.load(local_tmp, map_location="cpu", weights_only=False)

    if local_final != local_tmp:
        local_tmp.replace(local_final)

    if local_final != path:
        shutil.copy2(local_final, path)
        _ = torch.load(path, map_location="cpu", weights_only=False)

    print(f"[Checkpoint] Saved and verified: {path}")


def load_large_checkpoint(path, device):
    """Copy Drive checkpoint to local /content before loading, avoiding Drive FUSE seek issues."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if str(path).startswith("/content/drive") and Path("/content").exists():
        local_dir = Path("/content/checkpoints")
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / path.name
        if (not local_path.exists()) or local_path.stat().st_size != path.stat().st_size:
            print(f"[Checkpoint] Copying Stage 1 checkpoint to local runtime: {local_path}")
            shutil.copy2(path, local_path)
        path = local_path

    return torch.load(path, map_location=device, weights_only=False)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model_state", "model_state_dict", "state_dict"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format")
    return {k.replace("module.", "", 1): v for k, v in checkpoint.items()}


def alpha_schedule(epoch):
    ratio = min(max(epoch / float(max(alpha_warmup_epochs, 1)), 0.0), 1.0)
    return alpha_start + ratio * (alpha_end - alpha_start)


def keep_lambda_schedule(epoch):
    ratio = min(max(epoch / float(max(keep_lambda_decay_epochs, 1)), 0.0), 1.0)
    return keep_lambda_start + ratio * (keep_lambda_end - keep_lambda_start)


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
    weights = np.clip(weights, 1.0, 50.0)
    num_samples = int(round(len(indices) * samples_per_epoch_multiplier))
    num_samples = max(num_samples, len(indices))
    print(
        f"[Sampler] Stage 2 rare weighted sampler: n_indices={len(indices)} "
        f"num_samples={num_samples} min_w={weights.min():.2f} "
        f"mean_w={weights.mean():.2f} max_w={weights.max():.2f}"
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

    for k in range(num_nuclei_classes):
        s1_val = s1_metrics.get(f"s1_nuclei_dice_{k}", math.nan)
        s2_val = s2_metrics.get(f"s2_nuclei_dice_{k}", math.nan)
        out[f"s1_nuclei_dice_{k}"] = s1_val
        out[f"s1_nuclei_iou_{k}"] = s1_metrics.get(f"s1_nuclei_iou_{k}", math.nan)
        out[f"s2_nuclei_dice_{k}"] = s2_val
        out[f"s2_nuclei_iou_{k}"] = s2_metrics.get(f"s2_nuclei_iou_{k}", math.nan)
        s1_dice.append(s1_val)
        s2_dice.append(s2_val)

    for k in rare_nuclei_ids:
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
    print("\n" + "=" * 92)
    print(f"Stage 2 epoch {epoch:03d} | train={train_loss:.4f} val={val_loss:.4f} alpha={results['alpha']:.3f} keep={results['keep_lambda']:.3f}")
    print("-" * 92)
    for k in range(num_nuclei_classes):
        name = PUMA_NUCLEI_ID_TO_NAME[k]
        s1 = results.get(f"s1_nuclei_dice_{k}")
        s2 = results.get(f"s2_nuclei_dice_{k}")
        delta = math.nan if s1 is None or s2 is None or math.isnan(float(s1)) or math.isnan(float(s2)) else float(s2) - float(s1)
        print(f"{k:02d} {name:<22} S1={fmt(s1):<8} S2={fmt(s2):<8} Δ={fmt(delta):<8}")
    print(f"S1 macro={fmt(results.get('s1_macro_dice'))} | S2 macro={fmt(results.get('s2_macro_dice'))}")
    print(f"S1 rare ={fmt(results.get('s1_rare_macro_dice'))} | S2 rare ={fmt(results.get('s2_rare_macro_dice'))}")
    print(f"selection={fmt(results.get('selection_score'))} improvement={fmt(results.get('improvement_score'))} beats_stage1={results.get('beats_stage1')}")
    print("=" * 92 + "\n")


def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Root] {root}")
    print(f"[Data] {data_dir}")
    print(f"[Stage 1 checkpoint] {stage1_ckpt}")
    print(f"[Checkpoints] {checkpoint_dir}")

    train_ds = PUMADataset(data_dir, transforms=get_train_transforms(image_size), zero_cellpose_prob=0.0)
    val_ds = PUMADataset(data_dir, transforms=get_val_transforms(image_size), zero_cellpose_prob=0.0)

    split_meta = train_ds.get_split_metadata()
    train_idx, val_idx = make_or_load_group_split(
        source_names=split_meta["source_names"],
        is_original=split_meta["is_original"],
        split_path=split_file,
        seed=seed,
        train_fraction=train_fraction,
        force_new=force_new_split,
        val_original_only=val_original_only,
    )
    print(f"[Split] train={len(train_idx)} val={len(val_idx)} file={split_file}")
    print("[Split] Leakage-safe: all rare crops stay with their source image; validation uses originals only.")

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        sampler=make_rare_weighted_sampler(train_ds, train_idx),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model_s1 = UnifiedPanopticNet(
        vit_model=uni_weight_dir,
        cnn_model=get_cnn_spatial_prior(pretrained=False),
        num_tissue=5,
        num_nuclei=10,
        load_uni_weights=False,
    ).to(device)

    ckpt_s1 = load_large_checkpoint(stage1_ckpt, device)
    model_s1.load_state_dict(extract_state_dict(ckpt_s1), strict=True)

    cfg_s1 = ckpt_s1.get("inference_config", {}) if isinstance(ckpt_s1, dict) else {}
    model_s1.enable_sc_dfa(bool(cfg_s1.get("use_sc_dfa", True)))
    model_s1.set_spatial_prior_lambda(float(cfg_s1.get("lambda_prior", 1.0)))
    model_s1.eval()
    for p in model_s1.parameters():
        p.requires_grad = False

    model_s2 = ResidualNucleiRefinerUNet(
        in_channels=stage2_in_channels,
        out_classes=num_nuclei_classes,
    ).to(device)

    class_weights = torch.tensor(nuclei_weights, dtype=torch.float32, device=device)
    ce_loss = SafeCrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    ft_loss = FocalTverskyLoss(
        alpha=0.20,
        beta=0.80,
        gamma=1.60,
        class_weights=class_weights,
        ignore_index=ignore_index,
    ).to(device)
    ce_loss = ce_loss.to(device)

    if bnb is not None and device.type == "cuda":
        try:
            optimizer = bnb.optim.AdamW8bit(model_s2.parameters(), lr=lr, weight_decay=weight_decay)
        except Exception:
            optimizer = optim.AdamW(model_s2.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model_s2.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and use_fp16)
    metrics_calc = PUMAMetrics()

    best_score = -1.0
    best_epoch = 0
    best_improvement = -999.0

    if resume is not None:
        print(f"[WARN] resume path is set to {resume}, but exact resume loading is not implemented in this click-to-run file.")

    for epoch in range(1, epochs + 1):
        alpha = alpha_schedule(epoch)
        keep_lambda = keep_lambda_schedule(epoch)
        model_s2.train()
        train_loss_sum = 0.0

        for batch in tqdm(train_loader, desc=f"Train Stage2 {epoch:03d}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            targets_nc = batch["nuclei_nc"].to(device, non_blocking=True)
            cellpose_flows = batch["cellpose_flow"].to(device, non_blocking=True)
            site_types = batch.get("site_type") or [default_site_type] * images.shape[0]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda" and use_fp16):
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
                    temperature=kd_temperature,
                    ignore_index=ignore_index,
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
        s1_acc = metrics_calc.new_semantic_accumulator(num_nuclei_classes, "s1_nuclei", ignore_index=ignore_index, device=device)
        s2_acc = metrics_calc.new_semantic_accumulator(num_nuclei_classes, "s2_nuclei", ignore_index=ignore_index, device=device)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Valid Stage2 {epoch:03d}", leave=False):
                images = batch["image"].to(device, non_blocking=True)
                targets_nc = batch["nuclei_nc"].to(device, non_blocking=True)
                cellpose_flows = batch["cellpose_flow"].to(device, non_blocking=True)
                site_types = batch.get("site_type") or [default_site_type] * images.shape[0]

                with torch.amp.autocast("cuda", enabled=device.type == "cuda" and use_fp16):
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
                        temperature=kd_temperature,
                        ignore_index=ignore_index,
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
                "in_channels": stage2_in_channels,
                "out_classes": num_nuclei_classes,
                "residual": True,
                "alpha_start": alpha_start,
                "alpha_end": alpha_end,
                "alpha_warmup_epochs": alpha_warmup_epochs,
                "kd_temperature": kd_temperature,
                "keep_lambda_start": keep_lambda_start,
                "keep_lambda_end": keep_lambda_end,
                "nuclei_weights": nuclei_weights,
                "rare_nuclei_ids": rare_nuclei_ids,
                "uses_5_tissue_probs_no_background": True,
                "stage2_input_channels": stage2_in_channels,
                "split_is_group_based": True,
                "validation_original_only": val_original_only,
            },
        }

        if epoch % 5 == 0 or epoch == epochs:
            safe_torch_save(ckpt_payload, checkpoint_dir / "nuclei_refiner_residual_last.pth")

        score = float(results["selection_score"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_improvement = float(results["improvement_score"])
            safe_torch_save(ckpt_payload, checkpoint_dir / "nuclei_refiner_residual_best.pth")
            print(f"Saved Stage 2 best: epoch={best_epoch} score={best_score:.4f} improvement={best_improvement:+.4f}")

        if not results["beats_stage1"]:
            print("[WARN] Stage 2 has not beaten Stage 1 yet. For Docker inference, prefer Stage 1-only or validate hybrid before enabling Stage 2.\n")

    print("\n" + "=" * 92)
    print(f"Stage 2 complete. Best epoch used as checkpoint: {best_epoch}")
    print(f"Best score: {best_score:.4f}")
    print(f"Best improvement over Stage 1: {best_improvement:+.4f}")
    print(f"Best checkpoint: {checkpoint_dir / 'nuclei_refiner_residual_best.pth'}")
    print("=" * 92)


if __name__ == "__main__":
    main()
