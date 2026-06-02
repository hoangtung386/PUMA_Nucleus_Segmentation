"""Tests for warm-up cosine scheduler + linear_ramp."""

import torch
from torch import optim

from symbiopan.modules.scheduler import build_warmup_cosine_scheduler, linear_ramp


def test_linear_ramp_clamps():
    assert linear_ramp(epoch=0, start=10, end=20, max_value=1.0) == 0.0
    assert linear_ramp(epoch=25, start=10, end=20, max_value=1.0) == 1.0
    middle = linear_ramp(epoch=15, start=10, end=20, max_value=1.0)
    assert 0.0 < middle < 1.0


def test_warmup_cosine_scheduler_runs():
    params = [torch.nn.Parameter(torch.randn(1))]
    optimizer = optim.SGD(params, lr=1.0)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_epochs=2, total_epochs=10, steps_per_epoch=4)
    lrs = []
    for _ in range(40):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    assert lrs[0] < lrs[7]  # warm-up phase increases LR
    assert lrs[-1] < lrs[10]  # post-warmup cosine decreases
