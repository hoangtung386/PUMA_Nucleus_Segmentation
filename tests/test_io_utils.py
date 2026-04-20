"""Tests for I/O utility helpers."""

from __future__ import annotations

import numpy as np

from puma_seg.utils.io_utils import (
    list_image_paths,
    load_mask,
    load_results,
    save_mask,
    save_results,
)


def test_save_and_load_mask(tmp_path):
    mask = np.zeros((16, 16), dtype=np.int32)
    mask[2:5, 2:5] = 1
    out_path = tmp_path / "mask.npy"
    save_mask(mask, out_path)
    loaded = load_mask(out_path)
    assert loaded.dtype == np.int32
    assert np.array_equal(mask, loaded)


def test_save_and_load_results_json(tmp_path):
    result = {"score": 0.9, "count": 2}
    out_path = tmp_path / "result.json"
    save_results(result, out_path)
    loaded = load_results(out_path)
    assert loaded["score"] == 0.9
    assert loaded["count"] == 2


def test_list_image_paths_filters_extensions(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("x", encoding="utf-8")
    paths = list_image_paths(tmp_path)
    names = [path.name for path in paths]
    assert "a.png" in names
    assert "b.jpg" in names
    assert "c.txt" not in names
