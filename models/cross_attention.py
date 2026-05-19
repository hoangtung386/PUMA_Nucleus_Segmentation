import torch
import torch.nn as nn
import torch.nn.functional as f


class SpatialInjector(nn.Module):
    def __init__(self, vit_dim=1024, cnn_dims=[40, 80, 160, 320], target_grid=64):
        """Inject spatial information from CNN into ViT with memory efficiency."""
        super().__init__()
        # AdaptivePooling compresses large feature maps (1/4, 1/8) to a safe
        # grid size (64x64) to prevent VRAM overflow during attention computation.
        self.cnn_projections = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((target_grid, target_grid)),
                nn.Conv2d(dim, vit_dim, kernel_size=1)
            ) for dim in cnn_dims
        ])

        self.norm = nn.LayerNorm(vit_dim)
        self.scale = vit_dim ** -0.5

    def forward(self, vit_tokens, cnn_features):
        flattened_cnn = []

        for i, feat in enumerate(cnn_features):
            feat_proj = self.cnn_projections[i](feat)
            b, c, h, w = feat_proj.shape
            feat_flat = feat_proj.view(b, c, -1).transpose(1, 2)
            flattened_cnn.append(feat_flat)

        s_flat = torch.cat(flattened_cnn, dim=1)

        # Flash attention via PyTorch's optimized scaled_dot_product_attention.
        attn_output = f.scaled_dot_product_attention(vit_tokens, s_flat, s_flat, dropout_p=0.0, is_causal=False)

        return self.norm(vit_tokens + attn_output)
