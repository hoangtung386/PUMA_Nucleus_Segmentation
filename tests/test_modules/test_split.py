"""Tests for group-based train/val split."""

import pytest

from symbiopan.modules.split import make_or_load_group_split


def test_make_split_disjoint_sources(tmp_path):
    sources = ["a", "a", "b", "b", "c", "c", "d"]
    original = [True, False, True, False, True, False, True]
    split_path = tmp_path / "split.npz"

    train_idx, val_idx = make_or_load_group_split(
        sources, original, split_path, seed=42, train_fraction=0.5, force_new=True, val_original_only=True
    )

    train_sources = {sources[i] for i in train_idx}
    val_sources = {sources[i] for i in val_idx}
    assert not (train_sources & val_sources), "Source groups must be disjoint"
    assert len(train_idx) > 0 and len(val_idx) > 0


def test_make_split_persisted_and_reloaded(tmp_path):
    sources = ["a", "a", "b", "b", "c"]
    original = [True, False, True, False, True]
    split_path = tmp_path / "split.npz"

    train_a, val_a = make_or_load_group_split(sources, original, split_path, seed=7, force_new=True)
    train_b, val_b = make_or_load_group_split(sources, original, split_path, seed=7, force_new=False)

    assert train_a == train_b
    assert val_a == val_b


def test_make_split_requires_multiple_groups(tmp_path):
    with pytest.raises(RuntimeError):
        make_or_load_group_split(["only_one"] * 5, [True] * 5, tmp_path / "x.npz", force_new=True)
