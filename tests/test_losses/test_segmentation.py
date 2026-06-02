"""Tests for segmentation losses."""

import torch

from symbiopan.losses.segmentation import (
    FocalBCELoss,
    FocalTverskyLoss,
    SafeCrossEntropyLoss,
    SoftDiceLoss,
)


def test_safe_cross_entropy_ignores_index():
    loss_fn = SafeCrossEntropyLoss(ignore_index=255)
    logits = torch.randn(2, 5, 10, 10)
    targets = torch.full((2, 10, 10), 255, dtype=torch.long)
    loss = loss_fn(logits, targets)
    assert loss.item() == 0.0


def test_safe_cross_entropy_valid():
    loss_fn = SafeCrossEntropyLoss(ignore_index=255)
    logits = torch.randn(2, 5, 10, 10)
    targets = torch.randint(0, 5, (2, 10, 10), dtype=torch.long)
    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0


def test_focal_tversky_loss_range():
    loss_fn = FocalTverskyLoss(ignore_index=255)
    logits = torch.randn(2, 5, 10, 10)
    targets = torch.randint(0, 5, (2, 10, 10), dtype=torch.long)
    loss = loss_fn(logits, targets)
    assert 0.0 <= loss.item() <= 10.0


def test_focal_tversky_ignores_index():
    loss_fn = FocalTverskyLoss(ignore_index=255)
    logits = torch.randn(2, 5, 10, 10)
    targets = torch.full((2, 10, 10), 255, dtype=torch.long)
    loss = loss_fn(logits, targets)
    assert loss.item() == 0.0


def test_soft_dice_loss():
    loss_fn = SoftDiceLoss()
    logits = torch.randn(2, 1, 10, 10)
    targets = torch.randint(0, 2, (2, 10, 10), dtype=torch.float32)
    loss = loss_fn(logits, targets)
    assert 0.0 <= loss.item() <= 1.0


def test_focal_bce_loss():
    loss_fn = FocalBCELoss()
    logits = torch.randn(2, 1, 10, 10)
    targets = torch.randint(0, 2, (2, 10, 10), dtype=torch.float32)
    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
