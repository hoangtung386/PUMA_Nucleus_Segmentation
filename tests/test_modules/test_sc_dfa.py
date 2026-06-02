"""Tests for SCDFA module."""

import torch

from symbiopan.modules.sc_dfa import SCDFA


def test_scdfa_output_shape():
    sc = SCDFA(num_tissue_classes=6, num_nuclei_classes=10)
    tissue_logits = torch.randn(2, 6, 32, 32)
    out = sc(tissue_logits)
    assert out.shape == (2, 10, 32, 32)


def test_scdfa_validates_input_channels():
    sc = SCDFA(num_tissue_classes=6, num_nuclei_classes=10)
    bad_input = torch.randn(2, 5, 32, 32)
    try:
        sc(bad_input)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for mismatched tissue channels")
