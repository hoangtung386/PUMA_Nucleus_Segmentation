"""SpatialInjector cross-attention bridge for ViT + CNN fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialInjector(nn.Module):
    def __init__(
        self,
        vit_dim: int = 1280,
        cnn_dims: list[int] | None = None,
        target_grid: int = 32,
        num_heads: int = 8,
    ) -> None:
        if cnn_dims is None:
            cnn_dims = [96, 192, 384, 768]
        super().__init__()
        if vit_dim % num_heads != 0:
            raise ValueError(f"vit_dim={vit_dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = vit_dim // self.num_heads
        self.cnn_projections = nn.ModuleList(
            [
                nn.Sequential(nn.AdaptiveAvgPool2d((target_grid, target_grid)), nn.Conv2d(dim, vit_dim, kernel_size=1))
                for dim in cnn_dims
            ]
        )
        self.q_proj = nn.Linear(vit_dim, vit_dim)
        self.k_proj = nn.Linear(vit_dim, vit_dim)
        self.v_proj = nn.Linear(vit_dim, vit_dim)
        self.out_proj = nn.Linear(vit_dim, vit_dim)
        self.norm_q = nn.LayerNorm(vit_dim)
        self.norm_kv = nn.LayerNorm(vit_dim)
        self.norm_out = nn.LayerNorm(vit_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, vit_tokens: torch.Tensor, cnn_features: list[torch.Tensor]) -> torch.Tensor:
        flattened_cnn = []
        for i, feat in enumerate(cnn_features):
            feat_proj = self.cnn_projections[i](feat)
            b, c, h, w = feat_proj.shape
            feat_flat = feat_proj.view(b, c, -1).transpose(1, 2)
            flattened_cnn.append(feat_flat)

        s_flat = torch.cat(flattened_cnn, dim=1)
        q = self._split_heads(self.q_proj(self.norm_q(vit_tokens)))
        k = self._split_heads(self.k_proj(self.norm_kv(s_flat)))
        v = self._split_heads(self.v_proj(s_flat))
        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn_output = attn_output.transpose(1, 2).contiguous().view_as(vit_tokens)
        return self.norm_out(vit_tokens + self.out_proj(attn_output))
