from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

"""Decoders for UnifiedPanopticNet (Ablation 1: HoVerNeXt heads, Ablation 2: ASPP tissue)."""


class MutualFeatureExchange(nn.Module):
    """Bidirectional feature exchange between tissue and nuclei branches.

    Each branch generates a prompt from the other branch via grouped convolutions,
    then fuses its own features with the cross-branch prompt through a 1x1 conv.
    """

    def __init__(self, dim: int = 256) -> None:
        """Initialize the mutual feature exchange module.

        Args:
            dim: Channel dimension for all features (default 256).
        """
        super().__init__()
        self.w_t = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.w_n = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm_t = nn.GroupNorm(32, dim)
        self.norm_n = nn.GroupNorm(32, dim)
        self.conv_t = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.conv_n = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

    def forward(self, f_tissue: torch.Tensor, f_nuclei: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exchange prompts between tissue and nuclei branches.

        Args:
            f_tissue: Tissue features of shape (B, dim, H, W).
            f_nuclei: Nuclei features of shape (B, dim, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Refined tissue and nuclei
                features, each of shape (B, dim, H, W).
        """
        prompt_n = F.relu(self.norm_t(self.w_t(f_nuclei)))
        f_tissue_out = self.conv_t(torch.cat([f_tissue, prompt_n], dim=1))
        prompt_t = F.relu(self.norm_n(self.w_n(f_tissue)))
        f_nuclei_out = self.conv_n(torch.cat([f_nuclei, prompt_t], dim=1))
        return f_tissue_out, f_nuclei_out


class HoVerNeXtConv2dReLU(nn.Sequential):
    """Single convolutional block: Conv2d + optional BatchNorm + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        use_batchnorm: bool = False,
    ) -> None:
        """Initialize the conv-bn-relu block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            padding: Convolution padding.
            stride: Convolution stride.
            use_batchnorm: Whether to include BatchNorm after conv.
        """
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        relu = nn.ReLU()
        super().__init__(conv, bn, relu)


class HoVerNeXtNucleiHead(nn.Sequential):
    """HoVer-NeXt-style two-conv decoder head plus 1x1 prediction layer."""

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        """Initialize the HoVer-NeXt-style nuclei head.

        Args:
            in_channels: Number of input channels.
            mid_channels: Number of intermediate channels.
            out_channels: Number of output channels.
        """
        super().__init__(
            HoVerNeXtConv2dReLU(in_channels, mid_channels, kernel_size=3, padding=1, use_batchnorm=False),
            HoVerNeXtConv2dReLU(mid_channels, mid_channels, kernel_size=3, padding=1, use_batchnorm=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
        )


class ASPPBranch(nn.Sequential):
    """Single ASPP branch: atrous conv + BatchNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        """Initialize an ASPP branch.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            dilation: Atrous dilation rate (1 results in a 1x1 conv).
        """
        if dilation == 1:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
        super().__init__(conv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale context.

    Applies parallel atrous convolutions at multiple dilation rates and
    concatenates the results through a 1x1 projection.
    """

    def __init__(self, in_channels: int, out_channels: int, rates: Tuple[int, ...] = (1, 3, 6, 9)) -> None:
        """Initialize ASPP module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels per branch.
            rates: Tuple of dilation rates for parallel atrous branches.
        """
        super().__init__()
        self.branches = nn.ModuleList([ASPPBranch(in_channels, out_channels, r) for r in rates])
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(rates), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ASPP.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Multi-scale aggregated features of shape
                (B, out_channels, H, W).
        """
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class ParallelDecoders(nn.Module):
    """Version 5.2 + Ablation 1/2.

    Changes from baseline:
      1. NP/HV heads use HoVer-NeXt-style two-conv decoder heads.
      2. Tissue head uses ASPP for multi-scale context.

    Unchanged:
      - 5 tissue classes, no background channel.
      - 10 nuclei classes.
      - Existing FPN inputs and P3 mutual feature exchange.
      - NC head, because Ablation 1 targets instance quality only.
    """

    def __init__(self, fpn_dim: int = 256, num_tissue: int = 5, num_nuclei: int = 10) -> None:
        """Initialize parallel decoders for tissue, nuclei class, nuclei presence, and HoVer maps.

        Args:
            fpn_dim: FPN feature dimension (default 256).
            num_tissue: Number of tissue classes (must be 5).
            num_nuclei: Number of nuclei classes (default 10).
        """
        super().__init__()
        if num_tissue != 5:
            raise ValueError("This merged model intentionally uses 5 tissue classes, no background channel.")
        self.tissue_proj = nn.Conv2d(fpn_dim, fpn_dim, 1)
        self.nuclei_proj = nn.Conv2d(fpn_dim, fpn_dim, 1)
        self.exchange = MutualFeatureExchange(dim=fpn_dim)

        head_in = fpn_dim * 3

        # Ablation 2: ASPP for tissue macro-context only.
        self.tissue_head = nn.Sequential(
            ASPP(head_in, fpn_dim, rates=(1, 3, 6, 9)),
            nn.Conv2d(fpn_dim, num_tissue, 1),
        )

        # Keep NC baseline in Ablation 1/2, so NP/HV and tissue changes are separable.
        self.nc_head = nn.Sequential(
            nn.Conv2d(head_in, fpn_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, num_nuclei, 1),
        )

        # Ablation 1: HoVer-NeXt-style instance heads.
        micro_in = fpn_dim * 2 + 2
        self.np_head = HoVerNeXtNucleiHead(micro_in, 64, 1)
        self.hv_head = HoVerNeXtNucleiHead(micro_in, 64, 2)

    def forward(
        self, fpn_feats: Dict[str, torch.Tensor], cellpose_prior: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through parallel decoders.

        Args:
            fpn_feats: Dictionary of FPN feature maps at scales p1-p5.
            cellpose_prior: Cellpose flow prior of shape (B, 2, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                tissue_logits, np_logits, nc_logits, hv_logits.
        """
        p1, p2, p3, p4, p5 = fpn_feats["p1"], fpn_feats["p2"], fpn_feats["p3"], fpn_feats["p4"], fpn_feats["p5"]
        f_t, f_n = self.exchange(self.tissue_proj(p3), self.nuclei_proj(p3))

        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        p5_up = F.interpolate(p5, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        tissue_logits = self.tissue_head(torch.cat([f_t, p4_up, p5_up], dim=1))
        nc_logits = self.nc_head(torch.cat([f_n, p4_up, p5_up], dim=1))

        p2_up = F.interpolate(p2, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        cp_resized = F.interpolate(cellpose_prior, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        high_res = torch.cat([p1, p2_up, cp_resized], dim=1)
        np_logits = self.np_head(high_res)
        hv_logits = self.hv_head(high_res)
        return tissue_logits, np_logits, nc_logits, hv_logits
