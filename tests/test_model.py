"""Tests for model forward passes and builder functions."""

from __future__ import annotations

import pytest
import torch

from puma_seg.models.nucleus_classifier import NucleusClassifier, build_classifier


class TestNucleusClassifier:
    """Test NucleusClassifier construction and forward pass."""

    @pytest.fixture()
    def model_t1(self):
        return NucleusClassifier(n_classes=3, pretrained=False, freeze_backbone=False)

    @pytest.fixture()
    def model_frozen(self):
        return NucleusClassifier(n_classes=3, pretrained=False, freeze_backbone=True)

    def test_forward_shape(self, model_t1):
        x = torch.randn(4, 3, 64, 64)
        logits = model_t1(x)
        assert logits.shape == (4, 3)

    def test_forward_dtype(self, model_t1):
        x = torch.randn(2, 3, 64, 64)
        logits = model_t1(x)
        assert logits.dtype == torch.float32

    def test_frozen_backbone_no_grad(self, model_frozen):
        for name, param in model_frozen.backbone.named_parameters():
            assert not param.requires_grad, f"Param '{name}' should be frozen."

    def test_head_always_trainable(self, model_frozen):
        for name, param in model_frozen.head.named_parameters():
            assert param.requires_grad, f"Head param '{name}' should be trainable."

    def test_unfreeze_last_1_group(self, model_frozen):
        model_frozen.unfreeze_backbone(layer_groups=1)
        # layer4 should be unfrozen
        for param in model_frozen.backbone.layer4.parameters():
            assert param.requires_grad

    def test_embeddings_shape(self, model_t1):
        x = torch.randn(8, 3, 64, 64)
        emb = model_t1.get_embeddings(x)
        # Before last linear: should be 256-dim (defined in head)
        # The head is: Linear(512→256) → BN → ReLU → Dropout → Linear(256→3)
        # get_embeddings returns output of head[:-1] = (B, 256) after BN+ReLU+Dropout
        # But Dropout outputs same dim as input from the linear
        assert emb.shape == (8, 256)

    def test_save_and_load(self, tmp_path, model_t1):
        path = tmp_path / "cls.pth"
        model_t1.save(path)
        loaded = NucleusClassifier.load(path)
        assert loaded.n_classes == model_t1.n_classes

        # Weights should be identical
        for (n1, p1), (n2, p2) in zip(
            model_t1.named_parameters(), loaded.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Mismatch in '{n1}'"


class TestBuildClassifier:
    def test_track1_n_classes(self):
        model = build_classifier(track=1, pretrained=False)
        x = torch.randn(2, 3, 64, 64)
        assert model(x).shape == (2, 3)  # 3 foreground classes

    def test_track2_n_classes(self):
        model = build_classifier(track=2, pretrained=False)
        x = torch.randn(2, 3, 64, 64)
        assert model(x).shape == (2, 10)  # 10 foreground classes

    def test_invalid_track_raises(self):
        with pytest.raises(KeyError):
            build_classifier(track=99, pretrained=False)


class TestLossFunctions:
    def test_classification_loss_forward(self):
        from puma_seg.models.losses import ClassificationLoss

        loss_fn = ClassificationLoss(label_smoothing=0.0)
        logits = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0.0

    def test_focal_loss_forward(self):
        from puma_seg.models.losses import FocalLoss

        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0

    def test_combined_seg_loss_forward(self):
        from puma_seg.models.losses import CombinedSegLoss

        loss_fn = CombinedSegLoss()
        B, H, W = 2, 64, 64
        pred = torch.randn(B, 3, H, W)          # [dy, dx, cellprob]
        target_flow = torch.randn(B, 2, H, W)
        target_prob = torch.randint(0, 2, (B, 1, H, W)).float()
        loss = loss_fn(pred, target_flow, target_prob)
        assert loss.item() >= 0.0
