"""Shared training loop primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from utils import PUMAMetrics


def _autocast_context(device: torch.device) -> torch.amp.autocast:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda")


def _batch_to_device(
    batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
    images = batch["image"].to(device, non_blocking=True)
    targets = {
        "tissue_sem": batch["tissue_sem"].to(device, non_blocking=True),
        "nuclei_np": batch["nuclei_np"].to(device, non_blocking=True),
        "nuclei_nc": batch["nuclei_nc"].to(device, non_blocking=True),
        "nuclei_hv": batch["nuclei_hv"].to(device, non_blocking=True),
    }
    site_ids = (
        batch["site_id"].to(device, non_blocking=True) if isinstance(batch.get("site_id"), torch.Tensor) else None
    )
    context_roi = (
        batch["context_roi"].to(device, non_blocking=True)
        if isinstance(batch.get("context_roi"), torch.Tensor)
        else None
    )
    return images, targets, site_ids, context_roi


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    epoch: int,
    grad_accum_steps: int = 1,
) -> float:
    model.train()
    core = model.module if hasattr(model, "module") else model
    if hasattr(core, "encoder") and hasattr(core.encoder, "vit_model"):
        core.encoder.vit_model.eval() if not core.encoder.fine_tune else None

    running = 0.0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc=f"Train {epoch}", leave=False)

    for i, batch in enumerate(pbar):
        images, targets, site_ids, context_roi = _batch_to_device(batch, device)
        with _autocast_context(device):
            preds = model(images, site_ids, context_roi)
            loss, branch_losses = criterion(preds, targets)
            loss = loss / grad_accum_steps
        scaler.scale(loss).backward()

        if (i + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        running += float(loss.detach().item()) * grad_accum_steps
        pbar.set_postfix(
            loss=f"{float(loss.detach().item() * grad_accum_steps):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

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
    model.eval()
    running = 0.0
    metric_sum = {}
    branch_sum = [0.0, 0.0, 0.0, 0.0]

    pbar = tqdm(dataloader, desc=f"Valid {epoch}", leave=False)
    for batch in pbar:
        images, targets, site_ids, context_roi = _batch_to_device(batch, device)
        with _autocast_context(device):
            preds = model(images, site_ids, context_roi)
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
