import torch

from symbiopan.metrics.panoptic import PUMAMetrics, SemanticMetricAccumulator


def test_semantic_accumulator_basic():
    acc = SemanticMetricAccumulator(num_classes=3, prefix="test", ignore_index=255)
    preds = torch.tensor([[[0, 1, 2]]])
    targets = torch.tensor([[[0, 1, 2]]])
    acc.update(preds, targets)
    result = acc.compute()
    assert result["test_dice_0"] == 1.0
    assert result["test_dice_1"] == 1.0
    assert result["test_dice_2"] == 1.0


def test_semantic_accumulator_ignores_index():
    acc = SemanticMetricAccumulator(num_classes=3, prefix="test", ignore_index=255)
    preds = torch.tensor([[[0, 1, 255]]])
    targets = torch.tensor([[[0, 1, 255]]])
    acc.update(preds, targets)
    result = acc.compute()
    assert result["test_dice_0"] == 1.0
    assert result["test_dice_1"] == 1.0


def test_puma_metrics_all_metrics():
    metrics = PUMAMetrics()
    batch = 2
    preds = {
        "tissue": torch.randn(batch, 6, 10, 10),
        "nc": torch.randn(batch, 10, 10, 10),
    }
    targets = {
        "tissue_sem": torch.randint(0, 6, (batch, 10, 10), dtype=torch.long),
        "nuclei_nc": torch.randint(0, 10, (batch, 10, 10), dtype=torch.long),
    }
    result = metrics.calculate_all_metrics(preds, targets)
    assert "selection_score" in result
    assert "avg_tissue_dice" in result
    assert "avg_nuclei_dice" in result
    assert "rare_macro_dice" in result


def test_nanmean():
    assert PUMAMetrics._nanmean([1.0, 2.0, 3.0]) == 2.0
    assert PUMAMetrics._nanmean([]) != PUMAMetrics._nanmean([])
    assert PUMAMetrics._nan_to_zero(float("nan")) == 0.0
    assert PUMAMetrics._nan_to_zero(1.0) == 1.0
