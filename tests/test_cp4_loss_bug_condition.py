"""
Bug Condition Exploration Test for CP4Loss Tensor Size Mismatch

This test is designed to FAIL on unfixed code to confirm the bug exists.
It tests the bug condition where predicted output dimensions differ from
target dimensions and the interpolation logic fails to properly align
tensor shapes before loss computation.

EXPECTED OUTCOME ON UNFIXED CODE: Test FAILS with RuntimeError
EXPECTED OUTCOME ON FIXED CODE: Test PASSES
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, strategies as st, settings, HealthCheck

from puma_seg.models.cp4_dataset import CP4Loss


class TestCP4LossBugCondition:
    """
    Property 1: Bug Condition - Tensor Shape Alignment After Interpolation
    
    For any input where the predicted output dimensions differ from target
    dimensions (isBugCondition returns true), the CP4Loss.forward method
    should handle tensor shape alignment correctly and compute loss without
    RuntimeError.
    
    This test will FAIL on unfixed code, confirming the bug exists.
    """

    def test_dimension_mismatch_pred_smaller(self):
        """
        Test Case 1: pred shape [8, 3, 256, 256], targets shape [8, 512, 512]
        
        EXPECTED ON UNFIXED CODE: RuntimeError about tensor size mismatch
        EXPECTED ON FIXED CODE: Loss computation succeeds
        """
        loss_fn = CP4Loss(flow_weight=1.0, cellprob_weight=1.0)
        
        # Create inputs with dimension mismatch
        batch_size = 8
        pred = torch.randn(batch_size, 3, 256, 256)
        y_flow = torch.randn(batch_size, 512, 512)
        x_flow = torch.randn(batch_size, 512, 512)
        cellprob = torch.randint(0, 2, (batch_size, 512, 512)).float()
        
        # On unfixed code, this will raise RuntimeError
        # On fixed code, this should succeed
        loss = loss_fn(pred, y_flow, x_flow, cellprob)
        
        # Verify loss is valid
        assert loss.ndim == 0, "Loss should be a scalar tensor"
        assert loss.item() >= 0.0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_dimension_mismatch_pred_larger(self):
        """
        Test Case 2: pred shape [4, 3, 512, 512], targets shape [4, 256, 256]
        
        EXPECTED ON UNFIXED CODE: RuntimeError about tensor size mismatch
        EXPECTED ON FIXED CODE: Loss computation succeeds
        """
        loss_fn = CP4Loss(flow_weight=1.0, cellprob_weight=1.0)
        
        batch_size = 4
        pred = torch.randn(batch_size, 3, 512, 512)
        y_flow = torch.randn(batch_size, 256, 256)
        x_flow = torch.randn(batch_size, 256, 256)
        cellprob = torch.randint(0, 2, (batch_size, 256, 256)).float()
        
        loss = loss_fn(pred, y_flow, x_flow, cellprob)
        
        assert loss.ndim == 0
        assert loss.item() >= 0.0
        assert torch.isfinite(loss)

    def test_extra_channel_dimension_in_flow(self):
        """
        Test Case 3: y_flow/x_flow with extra channel dimension [4, 1, 256, 256]
        
        This tests the case where y_flow and x_flow already have a channel
        dimension, which causes the unsqueeze operation to add an extra
        dimension, resulting in shape mismatch.
        
        EXPECTED ON UNFIXED CODE: RuntimeError due to extra dimension
        EXPECTED ON FIXED CODE: Loss computation succeeds
        """
        loss_fn = CP4Loss(flow_weight=1.0, cellprob_weight=1.0)
        
        batch_size = 4
        pred = torch.randn(batch_size, 3, 128, 128)
        # y_flow and x_flow already have channel dimension
        y_flow = torch.randn(batch_size, 1, 256, 256)
        x_flow = torch.randn(batch_size, 1, 256, 256)
        cellprob = torch.randint(0, 2, (batch_size, 256, 256)).float()
        
        loss = loss_fn(pred, y_flow, x_flow, cellprob)
        
        assert loss.ndim == 0
        assert loss.item() >= 0.0
        assert torch.isfinite(loss)

    def test_batch_size_greater_than_one_with_mismatch(self):
        """
        Test Case 4: Various batch sizes with dimension mismatch
        
        EXPECTED ON UNFIXED CODE: RuntimeError
        EXPECTED ON FIXED CODE: Loss computation succeeds for all batch sizes
        """
        loss_fn = CP4Loss(flow_weight=1.0, cellprob_weight=1.0)
        
        for batch_size in [1, 2, 8, 16]:
            pred = torch.randn(batch_size, 3, 128, 128)
            y_flow = torch.randn(batch_size, 256, 256)
            x_flow = torch.randn(batch_size, 256, 256)
            cellprob = torch.randint(0, 2, (batch_size, 256, 256)).float()
            
            loss = loss_fn(pred, y_flow, x_flow, cellprob)
            
            assert loss.ndim == 0
            assert loss.item() >= 0.0
            assert torch.isfinite(loss)

    @given(
        batch_size=st.integers(min_value=1, max_value=16),
        pred_h=st.integers(min_value=64, max_value=256),
        pred_w=st.integers(min_value=64, max_value=256),
        target_h=st.integers(min_value=64, max_value=512),
        target_w=st.integers(min_value=64, max_value=512),
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    def test_property_dimension_mismatch_always_succeeds(
        self, batch_size: int, pred_h: int, pred_w: int, target_h: int, target_w: int
    ):
        """
        Property-Based Test: For all inputs where pred dimensions differ from
        target dimensions, CP4Loss.forward should compute loss successfully.
        
        This property test generates many random dimension combinations to
        ensure the fix works across the entire input domain.
        
        EXPECTED ON UNFIXED CODE: Fails with RuntimeError for many inputs
        EXPECTED ON FIXED CODE: Passes for all generated inputs
        """
        # Skip cases where dimensions already match (not bug condition)
        if pred_h == target_h and pred_w == target_w:
            return
        
        loss_fn = CP4Loss(flow_weight=1.0, cellprob_weight=1.0)
        
        pred = torch.randn(batch_size, 3, pred_h, pred_w)
        y_flow = torch.randn(batch_size, target_h, target_w)
        x_flow = torch.randn(batch_size, target_h, target_w)
        cellprob = torch.randint(0, 2, (batch_size, target_h, target_w)).float()
        
        # This should succeed after fix
        loss = loss_fn(pred, y_flow, x_flow, cellprob)
        
        assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert loss.item() >= 0.0, f"Loss should be non-negative, got {loss.item()}"
        assert torch.isfinite(loss), "Loss should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
