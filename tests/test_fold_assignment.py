"""Regression tests for stratified fold assignment.

Run with the project virtualenv, from the project root:

    ./.venv/bin/python tests/test_fold_assignment.py

Kept dependency-free (no pytest) so it runs in the same environment as the notebooks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puma.config import PUMA_CLASS_NAMES  # noqa: E402
from puma.data.preprocess import (  # noqa: E402
    _fold_capacities,
    fold_assignment_report,
    multilabel_greedy_folds,
    validate_fold_assignments,
)

NUMBER_OF_CLASSES = len(PUMA_CLASS_NAMES)


def synthetic_rois(count: int, seed: int, *, rois_per_case: int = 1) -> list[dict]:
    """ROIs shaped like PUMA: tumour-dominated, with a long tail of very rare classes."""
    rng = np.random.default_rng(seed)
    results = []
    for index in range(count):
        counts = np.zeros(NUMBER_OF_CLASSES, dtype=np.int64)
        counts[0] = rng.integers(20, 900)          # tumour
        counts[1] = rng.integers(0, 400)           # lymphocyte
        counts[3] = rng.integers(0, 90)            # histiocyte
        counts[6] = rng.integers(0, 60)            # stroma
        for rare in (2, 4, 5, 7, 8, 9):            # plasma, melanophage, neutrophil, ...
            if rng.random() < 0.25:
                counts[rare] = rng.integers(1, 12)
        results.append({
            "class_counts": counts,
            "melanoma_type": "primary" if index % 2 == 0 else "metastatic",
            "case_id": f"case_{index // rois_per_case:04d}",
        })
    return results


def test_capacities_differ_by_at_most_one() -> None:
    for total, folds in ((205, 5), (100, 5), (7, 5), (5, 5), (1000, 3)):
        capacities = _fold_capacities(total, folds)
        assert capacities.sum() == total, (total, folds, capacities)
        assert capacities.max() - capacities.min() <= 1, (total, folds, capacities)


def test_fold_sizes_are_balanced() -> None:
    """The rich-get-richer collapse produced [1, 45, 87, 1, 71] on 205 ROIs / 5 folds."""
    for seed in (0, 7, 2026):
        results = synthetic_rois(205, seed)
        assignments = multilabel_greedy_folds(results, 5, 2026)
        sizes = np.bincount(assignments, minlength=5)
        assert sizes.tolist() == [41, 41, 41, 41, 41], (seed, sizes.tolist())


def test_no_case_group_spans_two_folds() -> None:
    results = synthetic_rois(200, 3, rois_per_case=4)
    assignments = multilabel_greedy_folds(results, 5, 2026)
    per_case: dict[str, set[int]] = {}
    for record, fold in zip(results, assignments, strict=True):
        per_case.setdefault(record["case_id"], set()).add(int(fold))
    split = {case: folds for case, folds in per_case.items() if len(folds) > 1}
    assert not split, split
    # Grouping must not cost size balance: capacity is counted in ROIs, not in cases.
    sizes = np.bincount(assignments, minlength=5)
    assert sizes.max() - sizes.min() <= 1, sizes.tolist()


def test_common_classes_are_evenly_spread() -> None:
    results = synthetic_rois(205, 11)
    assignments = multilabel_greedy_folds(results, 5, 2026)
    counts = np.stack([np.asarray(r["class_counts"]) for r in results])
    per_fold = np.stack([counts[assignments == fold].sum(axis=0) for fold in range(5)])
    for class_id in (0, 1, 3, 6):  # classes with enough mass to be balanced
        target = counts[:, class_id].sum() / 5
        worst = np.abs(per_fold[:, class_id] - target).max() / max(target, 1.0)
        assert worst < 0.25, (PUMA_CLASS_NAMES[class_id], worst, per_fold[:, class_id].tolist())


def test_melanoma_type_is_stratified() -> None:
    results = synthetic_rois(205, 5)
    assignments = multilabel_greedy_folds(results, 5, 2026)
    types = np.array([r["melanoma_type"] for r in results])
    primary = np.array([int((types[assignments == fold] == "primary").sum()) for fold in range(5)])
    assert primary.max() - primary.min() <= 3, primary.tolist()


def test_assignment_is_deterministic() -> None:
    results = synthetic_rois(205, 1)
    first = multilabel_greedy_folds(results, 5, 2026)
    second = multilabel_greedy_folds(results, 5, 2026)
    assert np.array_equal(first, second)


def test_validator_rejects_the_historical_collapse() -> None:
    collapsed = np.array([0] + [1] * 45 + [2] * 87 + [3] + [4] * 71, dtype=np.int8)
    assert collapsed.shape[0] == 205
    try:
        validate_fold_assignments(collapsed, 5)
    except ValueError as exc:
        assert "Degenerate fold assignment" in str(exc), exc
    else:
        raise AssertionError("validate_fold_assignments accepted a 1-ROI fold")


def test_validator_accepts_a_balanced_split() -> None:
    balanced = np.repeat(np.arange(5, dtype=np.int8), 41)
    summary = validate_fold_assignments(balanced, 5)
    assert summary["fold_sizes"] == [41] * 5, summary
    assert summary["size_imbalance_ratio"] == 1.0, summary


def test_report_covers_every_fold_and_class() -> None:
    results = synthetic_rois(205, 2)
    assignments = multilabel_greedy_folds(results, 5, 2026)
    report = fold_assignment_report(results, assignments, 5, PUMA_CLASS_NAMES)
    assert set(report["per_fold"]) == {"0", "1", "2", "3", "4"}, report["per_fold"].keys()
    total = sum(entry["nuclei"] for entry in report["per_fold"].values())
    assert total == int(np.stack([r["class_counts"] for r in results]).sum()), total
    for entry in report["per_fold"].values():
        assert set(entry["class_counts"]) == set(PUMA_CLASS_NAMES)
        assert entry["primary"] + entry["metastatic"] == entry["rois"]


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}\n      {exc}")
        else:
            print(f"ok    {test.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
