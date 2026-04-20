"""Tests for PUMA evaluation metrics."""

from __future__ import annotations

import pytest

from puma_seg.evaluation.metrics import (
    classification_f1,
    compute_puma_score,
    detection_f1,
    evaluate_predictions,
    match_instances,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def perfect_pred():
    """Predictions that exactly match GT (3 nuclei)."""
    gt_centroids = {1: (10.0, 10.0), 2: (50.0, 50.0), 3: (90.0, 90.0)}
    pred_centroids = {1: (10.0, 10.0), 2: (50.0, 50.0), 3: (90.0, 90.0)}
    gt_classes = {1: 1, 2: 2, 3: 3}
    pred_classes = {1: 1, 2: 2, 3: 3}
    return gt_centroids, pred_centroids, gt_classes, pred_classes


@pytest.fixture()
def half_pred():
    """Predictions with 1 TP, 1 FP, 1 FN."""
    gt_centroids = {1: (10.0, 10.0), 2: (50.0, 50.0)}
    pred_centroids = {1: (10.0, 10.0), 2: (200.0, 200.0)}  # 2nd is far off
    gt_classes = {1: 1, 2: 2}
    pred_classes = {1: 1, 2: 2}
    return gt_centroids, pred_centroids, gt_classes, pred_classes


# ─── match_instances ──────────────────────────────────────────────────────────


class TestMatchInstances:
    def test_perfect_match(self, perfect_pred):
        gt_c, pred_c, _, _ = perfect_pred
        matched, fp, fn = match_instances(pred_c, gt_c, threshold=15.0)
        assert len(matched) == 3
        assert len(fp) == 0
        assert len(fn) == 0

    def test_no_instances(self):
        matched, fp, fn = match_instances({}, {}, threshold=15.0)
        assert matched == []
        assert len(fp) == 0
        assert len(fn) == 0

    def test_threshold_respected(self):
        pred = {1: (0.0, 0.0)}
        gt = {1: (20.0, 0.0)}  # distance = 20 > 15
        matched, fp, fn = match_instances(pred, gt, threshold=15.0)
        assert len(matched) == 0
        assert 1 in fp
        assert 1 in fn

    def test_threshold_just_inside(self):
        pred = {1: (0.0, 0.0)}
        gt = {1: (14.9, 0.0)}  # just within threshold
        matched, _, _ = match_instances(pred, gt, threshold=15.0)
        assert len(matched) == 1


# ─── detection_f1 ─────────────────────────────────────────────────────────────


class TestDetectionF1:
    def test_perfect_f1_is_one(self, perfect_pred):
        gt_c, pred_c, _, _ = perfect_pred
        metrics = detection_f1(pred_c, gt_c)
        assert abs(metrics["f1"] - 1.0) < 1e-6
        assert abs(metrics["precision"] - 1.0) < 1e-6
        assert abs(metrics["recall"] - 1.0) < 1e-6

    def test_f1_with_fp(self, half_pred):
        gt_c, pred_c, _, _ = half_pred
        metrics = detection_f1(pred_c, gt_c)
        # 1 TP, 1 FP, 1 FN → precision=0.5, recall=0.5, F1=0.5
        assert abs(metrics["precision"] - 0.5) < 1e-6
        assert abs(metrics["recall"] - 0.5) < 1e-6
        assert abs(metrics["f1"] - 0.5) < 1e-6


# ─── classification_f1 ────────────────────────────────────────────────────────


class TestClassificationF1:
    def test_perfect_classification(self, perfect_pred):
        gt_c, pred_c, gt_cls, pred_cls = perfect_pred
        matched, _, _ = match_instances(pred_c, gt_c)
        result = classification_f1(matched, pred_cls, gt_cls, n_classes=3)
        assert abs(result["macro_f1"] - 1.0) < 1e-6

    def test_empty_matched_gives_zeros(self):
        result = classification_f1([], {}, {}, n_classes=3)
        assert result["macro_f1"] == 0.0


# ─── compute_puma_score ───────────────────────────────────────────────────────


class TestPUMAScore:
    def test_perfect_score_equals_n_classes(self, perfect_pred):
        gt_c, pred_c, gt_cls, pred_cls = perfect_pred
        result = compute_puma_score(pred_c, gt_c, pred_cls, gt_cls, n_classes=3)
        # All 3 per-class F1 = 1.0 → puma_score = 3.0
        assert abs(result["puma_score"] - 3.0) < 1e-4

    def test_puma_score_is_nonnegative(self, half_pred):
        gt_c, pred_c, gt_cls, pred_cls = half_pred
        result = compute_puma_score(pred_c, gt_c, pred_cls, gt_cls, n_classes=3)
        assert result["puma_score"] >= 0.0


# ─── evaluate_predictions ─────────────────────────────────────────────────────


class TestEvaluatePredictions:
    def test_aggregation_shape(self, perfect_pred):
        gt_c, pred_c, gt_cls, pred_cls = perfect_pred
        pred_list = [{"centroids": pred_c, "classes": pred_cls}] * 2
        gt_list = [{"centroids": gt_c, "classes": gt_cls}] * 2
        metrics = evaluate_predictions(pred_list, gt_list, n_classes=3)
        assert "puma_score" in metrics
        assert "f1" in metrics

    def test_mismatched_lengths_raises(self, perfect_pred):
        gt_c, pred_c, gt_cls, pred_cls = perfect_pred
        pred_list = [{"centroids": pred_c, "classes": pred_cls}]
        gt_list = [{"centroids": gt_c, "classes": gt_cls}] * 2
        with pytest.raises(AssertionError):
            evaluate_predictions(pred_list, gt_list, n_classes=3)
