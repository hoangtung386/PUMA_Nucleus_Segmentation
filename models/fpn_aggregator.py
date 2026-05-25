"""Hierarchical FPN with CNN + multi-scale ViT feature fusion."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalFPN(nn.Module):
    def __init__(self, vit_dim: int = 1280, cnn_dims: Optional[List[int]] = None, fpn_dim: int = 256) -> None:
        super().__init__()
        if cnn_dims is None:
            cnn_dims = [96, 192, 384, 768]
        self.fpn_dim = fpn_dim
        self.latents = nn.ModuleList([nn.Conv2d(int(dim), fpn_dim, 1) for dim in cnn_dims])
        self.vit_proj = nn.Conv2d(vit_dim, fpn_dim, 1)
        self.vit_intermediate_projs = nn.ModuleList([nn.Conv2d(vit_dim, fpn_dim, 1) for _ in range(4)])
        self.smooth4 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.smooth1 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)

        self.low_level_fuse = nn.Sequential(
            nn.Conv2d(fpn_dim, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        vit_tokens: torch.Tensor,
        cnn_features: List[torch.Tensor],
        vit_intermediate: torch.Tensor,
        img_size: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if len(cnn_features) != len(self.latents):
            raise ValueError(f"Expected {len(self.latents)} CNN feature maps, got {len(cnn_features)}")

        patch_tokens = vit_tokens[:, 1:, :] if vit_tokens.shape[1] > 1 else vit_tokens
        b, n, c = patch_tokens.shape
        grid_size = int(round(n**0.5))
        if grid_size * grid_size != n:
            raise ValueError(f"ViT token count {n} is not square after removing CLS token")

        vit_2d = patch_tokens.transpose(1, 2).reshape(b, c, grid_size, grid_size)
        vit_2d = self.vit_proj(vit_2d)

        s1, s2, s3, s4 = [proj(feat) for proj, feat in zip(self.latents, cnn_features)]

        low_level_feat = self.low_level_fuse(s1)

        p5 = s4
        p4 = self.smooth4(
            s3
            + F.interpolate(p5, size=s3.shape[-2:], mode="bilinear", align_corners=False)
            + F.interpolate(vit_2d, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        )
        p3 = self.smooth3(s2 + F.interpolate(p4, size=s2.shape[-2:], mode="bilinear", align_corners=False))
        p2 = self.smooth2(s1 + F.interpolate(p3, size=s1.shape[-2:], mode="bilinear", align_corners=False))
        p1 = self.smooth1(F.interpolate(p2, scale_factor=2, mode="bilinear", align_corners=False))

        return {"p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5}, low_level_feat
