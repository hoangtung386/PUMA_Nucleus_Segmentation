"""Tests for decoder heads."""

import torch

from symbiopan.models.decoders import (
    CellViTPlusPlusNucleiDecoder,
    DeepLabV3PlusTissueHead,
    MutualFeatureExchange,
    ParallelDecoders,
)


def test_parallel_decoders_output_shapes():
    decoders = ParallelDecoders(fpn_dim=256, num_tissue=6, num_nuclei=10)
    fpn_feats = {
        "p1": torch.randn(1, 256, 256, 256),
        "p2": torch.randn(1, 256, 128, 128),
        "p3": torch.randn(1, 256, 64, 64),
        "p4": torch.randn(1, 256, 32, 32),
        "p5": torch.randn(1, 256, 16, 16),
    }
    low_level_feat = torch.randn(1, 96, 256, 256)
    vit_intermediate = torch.randn(4, 1, 1280, 64, 64)
    cnn_features = [
        torch.randn(1, 96, 128, 128),
        torch.randn(1, 192, 64, 64),
        torch.randn(1, 384, 32, 32),
        torch.randn(1, 768, 16, 16),
    ]

    tissue, np_logits, nc, hv = decoders(fpn_feats, low_level_feat, vit_intermediate, cnn_features)
    assert tissue.shape[1] == 6
    assert np_logits.shape[1] == 1
    assert nc.shape[1] == 10
    assert hv.shape[1] == 2


def test_mutual_feature_exchange_preserves_shapes():
    mfe = MutualFeatureExchange(dim=256)
    f_t = torch.randn(1, 256, 32, 32)
    f_n = torch.randn(1, 256, 32, 32)
    ft_out, fn_out = mfe(f_t, f_n)
    assert ft_out.shape == f_t.shape
    assert fn_out.shape == f_n.shape


def test_deep_lab_v3_plus_tissue_head():
    head = DeepLabV3PlusTissueHead(fpn_dim=256, num_tissue=6, low_level_channels=96)
    aspp_feat = torch.randn(1, 256, 64, 64)
    low_feat = torch.randn(1, 96, 256, 256)
    out = head(aspp_feat, low_feat)
    assert out.shape[1] == 6


def test_cell_vit_plus_plus_decoder():
    decoder = CellViTPlusPlusNucleiDecoder(fpn_dim=256, vit_dims=(1280, 1280, 1280, 1280), num_nuclei=10)
    vit_intermediate = torch.stack(
        [
            torch.randn(1, 1280, 64, 64),
            torch.randn(1, 1280, 64, 64),
            torch.randn(1, 1280, 64, 64),
            torch.randn(1, 1280, 64, 64),
        ],
        dim=0,
    )
    out = decoder(vit_intermediate)
    assert out.shape[1] == 10
