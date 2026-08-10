from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from puma.config import Stage1ModelConfig


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        groups = min(8, out_ch)
        while out_ch % groups and groups > 1:
            groups -= 1
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw1 = nn.Conv2d(channels, 4 * channels, 1)
        self.pw2 = nn.Conv2d(4 * channels, channels, 1)
        groups = min(8, channels)
        while channels % groups and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(self.dw(x))
        y = F.gelu(self.pw1(y))
        return x + self.pw2(y)


class PyramidEncoder(nn.Module):
    def __init__(self, channels: tuple[int, ...] = (48, 96, 192, 384)):
        super().__init__()
        self.channels = channels
        self.stem = nn.Sequential(ConvNormAct(3, channels[0]), ResidualBlock(channels[0]))
        self.downs = nn.ModuleList(
            nn.Sequential(ConvNormAct(a, b, stride=2), ResidualBlock(b), ResidualBlock(b))
            for a, b in zip(channels[:-1], channels[1:], strict=True)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = [self.stem(x)]
        for down in self.downs:
            features.append(down(features[-1]))
        return features


class FPNDecoder(nn.Module):
    def __init__(self, channels: tuple[int, ...], out_ch: int = 96):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(ch, out_ch, 1) for ch in channels)
        self.smooth = nn.ModuleList(ConvNormAct(out_ch, out_ch) for _ in channels)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        pyramid: list[torch.Tensor | None] = [None] * len(features)
        current = self.lateral[-1](features[-1])
        pyramid[-1] = self.smooth[-1](current)
        for index in range(len(features) - 2, -1, -1):
            current = F.interpolate(
                current, size=features[index].shape[-2:], mode="bilinear", align_corners=False
            ) + self.lateral[index](features[index])
            pyramid[index] = self.smooth[index](current)
        return [feature for feature in pyramid if feature is not None]


class DenseHeads(nn.Module):
    """A1 prediction heads. Unused serialized heads are kept for checkpoint compatibility."""

    def __init__(self, in_ch: int = 96, embedding_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(ConvNormAct(in_ch, in_ch), ResidualBlock(in_ch))
        self.heatmap = nn.Conv2d(in_ch, 1, 1)
        self.offset = nn.Conv2d(in_ch, 2, 1)
        self.extent = nn.Conv2d(in_ch, 2, 1)
        self.orientation = nn.Conv2d(in_ch, 2, 1)
        self.uncertainty = nn.Conv2d(in_ch, 1, 1)
        self.quality = nn.Conv2d(in_ch, 1, 1)
        self.embedding = nn.Conv2d(in_ch, embedding_dim, 1)
        self.proximity = None
        nn.init.constant_(self.heatmap.bias, -2.19)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.shared(x)
        return {"heatmap_logits": self.heatmap(x), "offset": self.offset(x)}


class IFCRNPP(nn.Module):
    output_stride = 1
    fixed_suppression_radius = 5.0

    def __init__(self):
        super().__init__()
        self.enable_extent = False  # checkpoint-compatible A1 module layout
        self.encoder = PyramidEncoder()
        self.fpn = FPNDecoder(self.encoder.channels, 96)
        self.heads = DenseHeads(96, embedding_dim=64)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.fpn(self.encoder(x))[0]
        return self.heads(feature)


def build_stage1_model(config: Stage1ModelConfig) -> nn.Module:
    if config.name != "A1_IFCRN_PP":
        raise ValueError(f"Version 13 supports only A1_IFCRN_PP, got {config.name!r}.")
    return IFCRNPP()
