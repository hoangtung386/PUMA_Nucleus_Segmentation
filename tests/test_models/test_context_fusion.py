"""Tests for context conditioning modules."""

import torch

from symbiopan.models.components.context_fusion import ContextFusionModule


def test_context_fusion_initializes_as_identity():
    fusion = ContextFusionModule(context_dim=32, fpn_dim=16)
    feats = {"p2": torch.randn(2, 16, 8, 8), "p3": torch.randn(2, 16, 4, 4)}
    context = torch.randn(2, 32)

    out = fusion(feats, context)

    assert torch.allclose(out["p2"], feats["p2"])
    assert torch.allclose(out["p3"], feats["p3"])
