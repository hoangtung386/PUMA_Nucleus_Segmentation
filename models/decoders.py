import torch
import torch.nn as nn
import torch.nn.functional as F


class MutualFeatureExchange(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.w_t = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.w_n = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm_t = nn.GroupNorm(32, dim)
        self.norm_n = nn.GroupNorm(32, dim)
        self.conv_t = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.conv_n = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

    def forward(self, f_tissue, f_nuclei):
        prompt_n = F.relu(self.norm_t(self.w_t(f_nuclei)))
        f_tissue_out = self.conv_t(torch.cat([f_tissue, prompt_n], dim=1))
        prompt_t = F.relu(self.norm_n(self.w_n(f_tissue)))
        f_nuclei_out = self.conv_n(torch.cat([f_nuclei, prompt_t], dim=1))
        return f_tissue_out, f_nuclei_out


mutual_feature_exchange = MutualFeatureExchange


# =============================================================================
# Ablation 1: HoVer-NeXt-style nuclei heads
# -----------------------------------------------------------------------------
# HoVer-NeXt's public training model uses a ConvNeXtV2 encoder with U-Net decoder
# blocks. The relevant decoder block is two Conv2dReLU layers, and in the
# published training config path the decoder is built with use_batchnorm=False.
#
# We do NOT use the fake "ConvNeXtBlock" from the feedback. That block is not the
# HoVer-NeXt decoder block. Here the NP/HV heads follow the original HoVer-NeXt
# decoder-head style as closely as possible inside your existing FPN architecture:
#   Conv2d -> Identity/no BN -> ReLU -> Conv2d -> Identity/no BN -> ReLU -> 1x1.
# This keeps your encoder/FPN/MFE unchanged and only changes the nuclei instance
# heads for the ablation.
# =============================================================================
class HoVerNeXtConv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=False):
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

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__(
            HoVerNeXtConv2dReLU(in_channels, mid_channels, kernel_size=3, padding=1, use_batchnorm=False),
            HoVerNeXtConv2dReLU(mid_channels, mid_channels, kernel_size=3, padding=1, use_batchnorm=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
        )


# =============================================================================
# Ablation 2: ASPP tissue head
# -----------------------------------------------------------------------------
# Tissue segmentation needs larger context than NP/HV. ASPP is isolated to the
# tissue branch only; NC/MFE/FPN remain unchanged so the ablation remains clean.
# Rates are moderate for PUMA because blood vessel can be small. If tissue recall
# improves but vessels degrade, reduce rates to (1, 2, 4, 6).
# =============================================================================
class ASPPBranch(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
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
    def __init__(self, in_channels, out_channels, rates=(1, 3, 6, 9)):
        super().__init__()
        self.branches = nn.ModuleList([ASPPBranch(in_channels, out_channels, r) for r in rates])
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(rates), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

    def forward(self, x):
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

    def __init__(self, fpn_dim=256, num_tissue=5, num_nuclei=10):
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

    def forward(self, fpn_feats, cellpose_prior):
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
