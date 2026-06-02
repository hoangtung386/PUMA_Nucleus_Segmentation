"""Tests for HierarchicalFPN."""

import torch

from symbiopan.models.fpn_aggregator import HierarchicalFPN


def test_hierarchical_fpn_output_keys():
    fpn = HierarchicalFPN(vit_dim=1280, cnn_dims=[96, 192, 384, 768], fpn_dim=256)
    vit_tokens = torch.randn(1, 256, 1280)
    cnn_features = [
        torch.randn(1, 96, 256, 256),
        torch.randn(1, 192, 128, 128),
        torch.randn(1, 384, 64, 64),
        torch.randn(1, 768, 32, 32),
    ]
    out, low_feat = fpn(vit_tokens, cnn_features)
    assert set(out.keys()) == {"p1", "p2", "p3", "p4", "p5"}
    assert out["p1"].shape[1] == 256
    assert low_feat.shape[1] == 96


def test_hierarchical_fpn_uses_image_grid_with_special_tokens():
    fpn = HierarchicalFPN(vit_dim=1280, cnn_dims=[96, 192, 384, 768], fpn_dim=256, patch_size=14)
    vit_tokens = torch.randn(1, 5330, 1280)
    cnn_features = [
        torch.randn(1, 96, 256, 256),
        torch.randn(1, 192, 128, 128),
        torch.randn(1, 384, 64, 64),
        torch.randn(1, 768, 32, 32),
    ]
    out, _ = fpn(vit_tokens, cnn_features, img_size=(1024, 1024))
    assert out["p4"].shape[-2:] == (64, 64)


def test_hierarchical_fpn_validates_cnn_count():
    fpn = HierarchicalFPN(vit_dim=1280, cnn_dims=[96, 192, 384, 768], fpn_dim=256)
    vit_tokens = torch.randn(1, 256, 1280)
    bad_cnn = [torch.randn(1, 96, 256, 256)]
    try:
        fpn(vit_tokens, bad_cnn)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for mismatched CNN feature count")
