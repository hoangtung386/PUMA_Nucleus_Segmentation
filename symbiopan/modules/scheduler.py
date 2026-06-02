"""Warm-up + cosine annealing LR scheduler."""


import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    min_lr_ratio: float = 0.01,
) -> SequentialLR:
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr_ratio)

    milestones: list[int] = [warmup_steps]
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=milestones)


def linear_ramp(epoch: int, start: int, end: int, max_value: float) -> float:
    if epoch < start:
        return 0.0
    if epoch >= end:
        return float(max_value)
    progress = (epoch - start + 1) / max(end - start + 1, 1)
    return float(max_value) * float(progress)


__all__ = ["build_warmup_cosine_scheduler", "linear_ramp"]
