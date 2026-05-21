from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNAggregator(nn.Module):
    """Robust FPN for Version-2.2 architecture, safe for 512 or 1024 inputs."""

    def __init__(self, vit_dim: int = 1024, cnn_dims: Optional[List[int]] = None, fpn_dim: int = 256) -> None:
        """Initialize the FPN aggregator.

        Args:
            vit_dim: ViT token channel dimension (default 1024).
            cnn_dims: List of CNN feature channel dimensions per scale.
            fpn_dim: Target FPN feature dimension (default 256).
        """
        super().__init__()
        if cnn_dims is None:
            cnn_dims = [40, 80, 160, 320]
        self.fpn_dim = fpn_dim
        self.latents = nn.ModuleList([nn.Conv2d(int(dim), fpn_dim, 1) for dim in cnn_dims])
        self.vit_proj = nn.Conv2d(vit_dim, fpn_dim, 1)
        self.smooth4 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.smooth1 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)

    def forward(
        self, vit_tokens: torch.Tensor, cnn_features: List[torch.Tensor], img_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the FPN aggregator.

        Args:
            vit_tokens: ViT patch tokens of shape (B, N, vit_dim).
            cnn_features: List of CNN feature maps from each stage.
            img_size: Optional image size (unused, kept for API compatibility).

        Returns:
            dict[str, torch.Tensor]: Dictionary of FPN feature maps at
                scales p1 (highest res) through p5 (lowest res), each of
                shape (B, fpn_dim, H_i, W_i).
        """
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

        p5 = s4
        p4 = self.smooth4(
            s3
            + F.interpolate(p5, size=s3.shape[-2:], mode="bilinear", align_corners=False)
            + F.interpolate(vit_2d, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        )
        p3 = self.smooth3(s2 + F.interpolate(p4, size=s2.shape[-2:], mode="bilinear", align_corners=False))
        p2 = self.smooth2(s1 + F.interpolate(p3, size=s1.shape[-2:], mode="bilinear", align_corners=False))
        p1 = self.smooth1(F.interpolate(p2, scale_factor=2, mode="bilinear", align_corners=False))
        return {"p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5}
