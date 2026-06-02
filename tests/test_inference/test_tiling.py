"""Tests for WSI tiling utilities."""

import numpy as np
import pytest
import torch

from symbiopan.inference.tiling import (
    find_single_tif,
    make_tile_starts,
    normalize_tile,
    pad_reflect,
)


def test_make_tile_starts_covers_full_image():
    starts = make_tile_starts(length=2048, tile_size=1024, stride=768)
    assert starts[0] == 0
    assert starts[-1] + 1024 >= 2048


def test_make_tile_starts_short_image():
    assert make_tile_starts(length=512, tile_size=1024, stride=768) == [0]


def test_pad_reflect_pads_to_target_size():
    tile = np.zeros((512, 512, 3), dtype=np.uint8)
    padded, real_h, real_w = pad_reflect(tile, tile_size=1024)
    assert padded.shape == (1024, 1024, 3)
    assert real_h == 512 and real_w == 512


def test_pad_reflect_no_op_when_already_target():
    tile = np.zeros((1024, 1024, 3), dtype=np.uint8)
    padded, real_h, real_w = pad_reflect(tile, tile_size=1024)
    assert padded.shape == (1024, 1024, 3)
    assert real_h == 1024 and real_w == 1024


def test_normalize_tile_returns_tensor():
    tile = np.zeros((32, 32, 3), dtype=np.uint8)
    out = normalize_tile(tile, device=torch.device("cpu"))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 3, 32, 32)


def test_find_single_tif_raises_on_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_single_tif(str(tmp_path))
