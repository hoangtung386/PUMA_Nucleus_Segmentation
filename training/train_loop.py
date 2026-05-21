"""Shared training loop primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from utils import PUMAMetrics


def _autocast_context(device: torch.device) -> torch.amp.autocast:
    """Return an autocast context manager for the given device.

    Args:
        device: A ``torch.device`` instance.

    Returns:
        torch.autocast: Autocast context for FP16 on CUDA (no-op on CPU).
    """
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda")


def _batch_to_device(
    batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, list[str]]:
    """Move a training batch to the target device.

    Extracts and transfers images, all target tensors, cellpose flows, and
    site-type metadata.

    Args:
        batch: Dict returned by the PUMADataset.
        device: Target ``torch.device``.

    Returns:
        tuple: ``(images, targets_dict, cellpose_flows, site_types)``.
    """
    images = batch["image"].to(device, non_blocking=True)
    targets = {
        "tissue_sem": batch["tissue_sem"].to(device, non_blocking=True),
        "nuclei_np": batch["nuclei_np"].to(device, non_blocking=True),
        "nuclei_nc": batch["nuclei_nc"].to(device, non_blocking=True),
        "nuclei_hv": batch["nuclei_hv"].to(device, non_blocking=True),
    }
    cellpose_flows = batch["cellpose_flow"].to(device, non_blocking=True)
    site_types = batch.get("site_type")
    if site_types is None:
        site_types = ["metastatic"] * images.shape[0]
    return images, targets, cellpose_flows, site_types


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    epoch: int,
) -> float:
    """Train the model for a single epoch.

    Args:
        model: The Stage 1 model to train.
        dataloader: Training DataLoader.
        optimizer: Optimizer (AdamW or 8-bit AdamW).
        criterion: Multi-task loss criterion.
        scheduler: LR scheduler (stepped per batch).
        device: Target torch device.
        scaler: ``GradScaler`` for mixed-precision training.
        epoch: Current epoch number (used only for logging).

    Returns:
        float: Average training loss over the epoch.
    """
    model.train()
    core = model.module if hasattr(model, "module") else model
    if hasattr(core, "encoder") and hasattr(core.encoder, "vit_model"):
        core.encoder.vit_model.eval()

    running = 0.0
    pbar = tqdm(dataloader, desc=f"Train {epoch}", leave=False)
    for batch in pbar:
        images, targets, cellpose_flows, site_types = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device):
            preds = model(images, cellpose_flows, site_types)
            loss, branch_losses = criterion(preds, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        running += float(loss.detach().item())
        pbar.set_postfix(loss=f"{float(loss.detach().item()):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    return running / max(len(dataloader), 1)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    metrics_calculator: PUMAMetrics,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Run validation for a single epoch.

    Args:
        model: The Stage 1 model to evaluate.
        dataloader: Validation DataLoader.
        criterion: Multi-task loss criterion.
        metrics_calculator: ``PUMAMetrics`` instance.
        device: Target torch device.
        epoch: Current epoch number (used only for logging).

    Returns:
        dict: Validation metrics including ``val_loss``, per-branch losses,
            and all per-class dice/iou scores.
    """
    model.eval()
    running = 0.0
    metric_sum = {}
    branch_sum = [0.0, 0.0, 0.0, 0.0]

    pbar = tqdm(dataloader, desc=f"Valid {epoch}", leave=False)
    for batch in pbar:
        images, targets, cellpose_flows, site_types = _batch_to_device(batch, device)
        with _autocast_context(device):
            preds = model(images, cellpose_flows, site_types)
            loss, branch_losses = criterion(preds, targets)
        running += float(loss.detach().item())
        for i, v in enumerate(branch_losses):
            branch_sum[i] += float(v)

        metrics = metrics_calculator.calculate_all_metrics(preds, targets)
        for k, v in metrics.items():
            if isinstance(v, float) and v != v:
                continue
            metric_sum[k] = metric_sum.get(k, 0.0) + float(v)

    n = max(len(dataloader), 1)
    out = {k: v / n for k, v in metric_sum.items()}
    out["val_loss"] = running / n
    out["loss_tissue"] = branch_sum[0] / n
    out["loss_np"] = branch_sum[1] / n
    out["loss_nc"] = branch_sum[2] / n
    out["loss_hv"] = branch_sum[3] / n
    return out
