"""SpatialInjector cross-attention bridge for ViT + CNN fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialInjector(nn.Module):
    def __init__(self, vit_dim: int = 1280, cnn_dims: list[int] | None = None, target_grid: int = 64) -> None:
        if cnn_dims is None:
            cnn_dims = [96, 192, 384, 768]
        super().__init__()
        self.cnn_projections = nn.ModuleList(
            [
                nn.Sequential(nn.AdaptiveAvgPool2d((target_grid, target_grid)), nn.Conv2d(dim, vit_dim, kernel_size=1))
                for dim in cnn_dims
            ]
        )
        self.norm = nn.LayerNorm(vit_dim)
        self.scale = vit_dim**-0.5

    def forward(self, vit_tokens: torch.Tensor, cnn_features: list[torch.Tensor]) -> torch.Tensor:
        flattened_cnn = []
        for i, feat in enumerate(cnn_features):
            feat_proj = self.cnn_projections[i](feat)
            b, c, h, w = feat_proj.shape
            feat_flat = feat_proj.view(b, c, -1).transpose(1, 2)
            flattened_cnn.append(feat_flat)

        s_flat = torch.cat(flattened_cnn, dim=1)
        attn_output = F.scaled_dot_product_attention(vit_tokens, s_flat, s_flat, dropout_p=0.0, is_causal=False)
        return self.norm(vit_tokens + attn_output)
