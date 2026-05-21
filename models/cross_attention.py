from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialInjector(nn.Module):
    """Inject spatial information from CNN into ViT using cross-attention with memory efficiency.

    Multi-scale CNN features are adaptively pooled to a fixed grid, projected to
    ViT dimension, and used as key/value in flash attention with ViT tokens.
    """

    def __init__(self, vit_dim: int = 1024, cnn_dims: Optional[List[int]] = None, target_grid: int = 64) -> None:
        """Initialize the spatial injector.

        Args:
            vit_dim: ViT token dimension (default 1024).
            cnn_dims: List of CNN feature channel dimensions per scale.
            target_grid: Spatial size to pool CNN features to (default 64).
        """
        if cnn_dims is None:
            cnn_dims = [40, 80, 160, 320]
        super().__init__()
        # AdaptivePooling compresses large feature maps (1/4, 1/8) to a safe
        # grid size (64x64) to prevent VRAM overflow during attention computation.
        self.cnn_projections = nn.ModuleList(
            [
                nn.Sequential(nn.AdaptiveAvgPool2d((target_grid, target_grid)), nn.Conv2d(dim, vit_dim, kernel_size=1))
                for dim in cnn_dims
            ]
        )

        self.norm = nn.LayerNorm(vit_dim)
        self.scale = vit_dim**-0.5

    def forward(self, vit_tokens: torch.Tensor, cnn_features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass: attend ViT tokens to multi-scale CNN features.

        Args:
            vit_tokens: ViT patch tokens of shape (B, N, vit_dim).
            cnn_features: List of CNN feature maps at different scales.

        Returns:
            torch.Tensor: ViT tokens updated with spatial context,
                same shape as input vit_tokens.
        """
        flattened_cnn = []

        for i, feat in enumerate(cnn_features):
            feat_proj = self.cnn_projections[i](feat)
            b, c, h, w = feat_proj.shape
            feat_flat = feat_proj.view(b, c, -1).transpose(1, 2)
            flattened_cnn.append(feat_flat)

        s_flat = torch.cat(flattened_cnn, dim=1)

        # Flash attention via PyTorch's optimized scaled_dot_product_attention.
        attn_output = F.scaled_dot_product_attention(vit_tokens, s_flat, s_flat, dropout_p=0.0, is_causal=False)

        return self.norm(vit_tokens + attn_output)
