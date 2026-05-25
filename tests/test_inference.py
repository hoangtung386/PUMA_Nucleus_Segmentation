"""Tests for inference utilities including TTA."""

import torch

from inference.infer_wsi import apply_tta


def test_tta_no_aug_output_shapes():
    model = _DummyModel()
    tensor = torch.randn(1, 3, 256, 256)
    site_ids = torch.zeros(1, dtype=torch.long)
    out = apply_tta(model, tensor, site_ids, use_tta=False)
    assert "tissue" in out
    assert "nc" in out
    assert out["tissue"].shape == (1, 5, 256, 256)
    assert out["nc"].shape == (1, 10, 256, 256)


class _DummyModel(torch.nn.Module):
    def forward(self, x, site_ids=None, context_roi=None):
        return {
            "tissue": torch.randn(1, 5, x.shape[2], x.shape[3]),
            "np": torch.randn(1, 1, x.shape[2], x.shape[3]),
            "nc": torch.randn(1, 10, x.shape[2], x.shape[3]),
            "hv": torch.randn(1, 2, x.shape[2], x.shape[3]),
            "boundary": torch.randn(1, 1, x.shape[2], x.shape[3]),
        }
