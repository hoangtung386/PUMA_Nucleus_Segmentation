"""Tests for checkpoint save / load / extract utilities."""

import pytest
import torch
import torch.nn as nn

from symbiopan.training.checkpoint import extract_state_dict, load_large_checkpoint, safe_torch_save


def test_safe_torch_save_round_trip(tmp_path):
    path = tmp_path / "ckpt.pt"
    obj = {"state_dict": {"a": torch.zeros(1)}, "epoch": 5}
    safe_torch_save(obj, path)
    assert path.exists()
    loaded = load_large_checkpoint(path)
    assert loaded["epoch"] == 5


def test_extract_state_dict_from_module():
    model = nn.Linear(4, 4)
    state = extract_state_dict(model)
    assert "weight" in state and "bias" in state


def test_extract_state_dict_strips_module_prefix():
    raw = {"module.layer.weight": torch.zeros(1), "module.layer.bias": torch.zeros(1)}
    extracted = extract_state_dict(raw)
    assert "layer.weight" in extracted
    assert "module.layer.weight" not in extracted


def test_extract_state_dict_unwraps_known_keys():
    raw = {"state_dict": {"x": torch.zeros(1)}}
    extracted = extract_state_dict(raw)
    assert "x" in extracted


def test_load_large_checkpoint_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_large_checkpoint(tmp_path / "nope.pt")
