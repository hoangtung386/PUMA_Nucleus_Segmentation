"""Tests for sample-weight computation."""

import numpy as np

from symbiopan.data.sampling import compute_all_sample_weights, compute_sample_weight


def test_compute_sample_weight_uses_metadata():
    tissue = np.zeros((4, 4), dtype=np.uint8)
    nuclei = np.zeros((4, 4), dtype=np.uint8)
    weight = compute_sample_weight(tissue, nuclei, is_rare_augmented=False, metadata_weight=7.5)
    assert weight == 7.5


def test_compute_sample_weight_rare_class_bonus():
    tissue = np.full((4, 4), fill_value=4, dtype=np.uint8)
    nuclei = np.zeros((4, 4), dtype=np.uint8)
    weight = compute_sample_weight(tissue, nuclei, is_rare_augmented=False)
    assert weight > 1.0


def test_compute_sample_weight_rare_augmented_multiplier():
    tissue = np.zeros((4, 4), dtype=np.uint8)
    nuclei = np.zeros((4, 4), dtype=np.uint8)
    base = compute_sample_weight(tissue, nuclei, is_rare_augmented=False)
    boosted = compute_sample_weight(tissue, nuclei, is_rare_augmented=True)
    assert boosted == 1.5 * base


def test_compute_all_sample_weights_falls_back_to_disk(tmp_path):
    data_dir = tmp_path / "ds"
    (data_dir / "tissue_sem").mkdir(parents=True)
    (data_dir / "nuclei_nc").mkdir()
    np.save(data_dir / "tissue_sem" / "s0.npy", np.full((4, 4), 2, dtype=np.uint8))
    np.save(data_dir / "nuclei_nc" / "s0.npy", np.zeros((4, 4), dtype=np.uint8))

    weights = compute_all_sample_weights(data_dir, ["s0"], [False])
    assert weights[0] > 1.0


def test_compute_all_sample_weights_prefers_metadata(tmp_path):
    data_dir = tmp_path / "ds"
    data_dir.mkdir()
    weights = compute_all_sample_weights(data_dir, ["x"], [False], metadata={"x": {"sample_weight": 4.2}})
    assert weights == [4.2]
