import timm
import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    def __init__(self, output_dim: int = 256, output_mode: str = "both") -> None:
        super().__init__()
        if output_mode not in ("spatial", "global", "both"):
            raise ValueError(f"output_mode must be 'spatial', 'global', or 'both', got {output_mode!r}")
        self.output_mode = output_mode
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, features_only=True, out_indices=(4,))
        self.proj = nn.Sequential(
            nn.Conv2d(1280, output_dim, kernel_size=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )
        if self.output_mode in ("global", "both"):
            self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, context_roi: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(context_roi)
        spatial = self.proj(features[0])
        if self.output_mode == "spatial":
            return spatial
        global_feat = self.gap(spatial).flatten(1)
        if self.output_mode == "global":
            return global_feat
        return spatial, global_feat
