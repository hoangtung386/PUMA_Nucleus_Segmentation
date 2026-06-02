"""Tests for MultiTaskUncertaintyLoss."""

import torch

from symbiopan.losses.multitask import MultiTaskUncertaintyLoss


def _make_batch(batch: int = 2, hw: int = 10) -> tuple[dict, dict]:
    preds = {
        "tissue": torch.randn(batch, 6, hw, hw),
        "np": torch.randn(batch, 1, hw, hw),
        "nc": torch.randn(batch, 10, hw, hw),
        "hv": torch.randn(batch, 2, hw, hw),
    }
    targets = {
        "tissue_sem": torch.randint(0, 6, (batch, hw, hw), dtype=torch.long),
        "nuclei_nc": torch.randint(0, 10, (batch, hw, hw), dtype=torch.long),
        "nuclei_np": torch.randint(0, 2, (batch, hw, hw), dtype=torch.long),
        "nuclei_hv": torch.randn(batch, 2, hw, hw),
    }
    return preds, targets


def test_multi_task_uncertainty_loss_output():
    loss_fn = MultiTaskUncertaintyLoss()
    preds, targets = _make_batch()
    total, branch_losses = loss_fn(preds, targets)
    assert total.item() > 0.0
    assert len(branch_losses) == 4
    assert all(isinstance(x, float) for x in branch_losses)


def test_set_focal_tversky_weight_clamps():
    loss_fn = MultiTaskUncertaintyLoss()
    loss_fn.set_focal_tversky_weight(-1.0)
    assert loss_fn.focal_tversky_weight == 0.0
    loss_fn.set_focal_tversky_weight(2.5)
    assert loss_fn.focal_tversky_weight == 1.0


def test_focal_tversky_increases_loss():
    loss_fn = MultiTaskUncertaintyLoss()
    preds, targets = _make_batch()

    loss_fn.set_focal_tversky_weight(0.0)
    base_total, _ = loss_fn(preds, targets)

    loss_fn.set_focal_tversky_weight(1.0)
    ramped_total, _ = loss_fn(preds, targets)

    assert ramped_total.item() != base_total.item()
