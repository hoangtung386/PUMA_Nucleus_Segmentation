import torch
import torch.nn as nn
import torch.nn.functional as F


class MutualFeatureExchange(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.w_t = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.w_n = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm_t = nn.GroupNorm(32, dim)
        self.norm_n = nn.GroupNorm(32, dim)
        self.conv_t = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.conv_n = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

    def forward(self, f_tissue: torch.Tensor, f_nuclei: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_n = F.relu(self.norm_t(self.w_t(f_nuclei)))
        f_tissue_out = self.conv_t(torch.cat([f_tissue, prompt_n], dim=1))
        prompt_t = F.relu(self.norm_n(self.w_n(f_tissue)))
        f_nuclei_out = self.conv_n(torch.cat([f_nuclei, prompt_t], dim=1))
        return f_tissue_out, f_nuclei_out


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rates: tuple[int, ...] = (1, 3, 6, 9)) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            if r == 1:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=r, dilation=r, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    )
                )
        self.branches.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(out_channels // 16 if out_channels >= 16 else out_channels, out_channels),
                nn.ReLU(inplace=True),
            )
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for branch in self.branches:
            out = branch(x)
            if out.shape[-2:] != x.shape[-2:]:
                out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
            outputs.append(out)
        return self.bottleneck(torch.cat(outputs, dim=1))


class HoVerNeXtNucleiHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 64, out_channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DeepLabV3PlusTissueHead(nn.Module):
    def __init__(self, fpn_dim: int = 256, num_tissue: int = 6, low_level_channels: int = 96) -> None:
        super().__init__()
        self.aspp = ASPP(fpn_dim, fpn_dim, rates=(1, 3, 6, 9))
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_dim + 48, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(fpn_dim, num_tissue, kernel_size=1)

    def forward(self, aspp_feat: torch.Tensor, low_level_feat: torch.Tensor) -> torch.Tensor:
        aspp_out = self.aspp(aspp_feat)
        low = self.low_level_conv(low_level_feat)
        aspp_up = F.interpolate(aspp_out, size=low.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([aspp_up, low], dim=1))
        return self.classifier(fused)


class CellViTPlusPlusNucleiDecoder(nn.Module):
    def __init__(
        self, fpn_dim: int = 256, vit_dims: tuple[int, ...] = (1280, 1280, 1280, 1280), num_nuclei: int = 10
    ) -> None:
        super().__init__()
        self.vit_projs = nn.ModuleList([nn.Conv2d(d, fpn_dim, kernel_size=1) for d in vit_dims])
        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_dim * len(vit_dims), fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
        )
        self.nc_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, num_nuclei, kernel_size=1),
        )

    def forward(self, vit_intermediate: torch.Tensor) -> torch.Tensor:
        upsampled = []
        target_size = vit_intermediate[-1].shape[-2:]
        for proj, feat in zip(self.vit_projs, vit_intermediate, strict=False):
            x = proj(feat)
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            upsampled.append(x)
        fused = self.fuse(torch.cat(upsampled, dim=1))
        return self.nc_head(fused)


class ParallelDecoders(nn.Module):
    def __init__(
        self, fpn_dim: int = 256, num_tissue: int = 6, num_nuclei: int = 10, low_level_channels: int = 96
    ) -> None:
        super().__init__()
        self.tissue_proj = nn.Conv2d(fpn_dim, fpn_dim, 1)
        self.nuclei_proj = nn.Conv2d(fpn_dim, fpn_dim, 1)
        self.exchange = MutualFeatureExchange(dim=fpn_dim)

        self.tissue_decoder = DeepLabV3PlusTissueHead(
            fpn_dim=fpn_dim, num_tissue=num_tissue, low_level_channels=low_level_channels
        )

        self.nc_head = CellViTPlusPlusNucleiDecoder(
            fpn_dim=fpn_dim, vit_dims=(1280, 1280, 1280, 1280), num_nuclei=num_nuclei
        )

        self.tissue_fuse = nn.Sequential(
            nn.Conv2d(fpn_dim * 3, fpn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
        )
        micro_in = fpn_dim * 2
        self.np_head = HoVerNeXtNucleiHead(micro_in, 64, 1)
        self.hv_head = HoVerNeXtNucleiHead(micro_in, 64, 2)

    def forward(
        self,
        fpn_feats: dict[str, torch.Tensor],
        low_level_feat: torch.Tensor,
        vit_intermediate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p1, p2, p3, p4, p5 = fpn_feats["p1"], fpn_feats["p2"], fpn_feats["p3"], fpn_feats["p4"], fpn_feats["p5"]
        f_t, f_n = self.exchange(self.tissue_proj(p3), self.nuclei_proj(p3))

        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        p5_up = F.interpolate(p5, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        tissue_input = self.tissue_fuse(torch.cat([f_t, p4_up, p5_up], dim=1))
        tissue_logits = self.tissue_decoder(tissue_input, low_level_feat)

        nc_logits = self.nc_head(vit_intermediate)

        p2_up = F.interpolate(p2, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        high_res = torch.cat([p1, p2_up], dim=1)
        np_logits = self.np_head(high_res)
        hv_logits = self.hv_head(high_res)

        return tissue_logits, np_logits, nc_logits, hv_logits
