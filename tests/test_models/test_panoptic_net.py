"""Tests for UnifiedPanopticNet end-to-end instantiation."""

import pytest

from symbiopan.models import UnifiedPanopticNet, build_cnn_backbone


@pytest.fixture
def tiny_panoptic_net():
    cnn = build_cnn_backbone(pretrained=False)
    return UnifiedPanopticNet(
        virchow2_model_name="paige-ai/Virchow2",
        cnn_model=cnn,
        num_tissue=6,
        num_nuclei=10,
        num_sites=9,
        fine_tune_last_n_blocks=0,
        load_encoder_weights=False,
    )


def test_unified_panoptic_net_instantiates(tiny_panoptic_net):
    assert tiny_panoptic_net is not None
    assert hasattr(tiny_panoptic_net, "encoder")
    assert hasattr(tiny_panoptic_net, "fpn")
    assert hasattr(tiny_panoptic_net, "decoders")
    assert tiny_panoptic_net.site_embed.num_embeddings == 9
    assert tiny_panoptic_net.site_embed.embedding_dim == 256


def test_unified_panoptic_net_sc_dfa_lambda(tiny_panoptic_net):
    tiny_panoptic_net.enable_sc_dfa(True)
    assert tiny_panoptic_net.use_sc_dfa is True
    assert tiny_panoptic_net.lambda_sc_dfa == 1.0

    tiny_panoptic_net.set_sc_dfa_lambda(0.5)
    assert tiny_panoptic_net.lambda_sc_dfa == 0.5

    tiny_panoptic_net.set_sc_dfa_lambda(0.0)
    assert tiny_panoptic_net.use_sc_dfa is False
