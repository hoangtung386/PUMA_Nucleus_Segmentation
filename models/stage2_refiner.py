from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualNucleiRefinerUNet(nn.Module):
    """
    Stage 2 nuclei refiner.

    It predicts only a residual correction delta for Stage-1 nuclei-class logits:
        refined_nc_logits = stage1_nc_logits + alpha * delta_nc_logits

    The final 1x1 convolution is zero-initialized, so at step 0 the refiner is
    exactly identity. This prevents the common failure where Stage 2 overwrites
    Stage 1 and kills rare classes.

    For the merged no-background tissue model, the input has 21 channels:
         3 RGB-normalized image channels
       + 5 Stage-1 tissue probabilities
       + 10 Stage-1 nuclei-class probabilities
       + 1 Stage-1 NP foreground probability
       + 2 Stage-1 HV channels
       = 21 channels
    """

    def __init__(self, in_channels: int = 21, out_classes: int = 10) -> None:
        """Initialize the residual nuclei refiner U-Net.

        Args:
            in_channels: Number of input channels (default 21 for merged model).
            out_classes: Number of output nuclei classes (default 10).
        """
        super().__init__()

        def conv_block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.inc = conv_block(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), conv_block(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), conv_block(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), conv_block(256, 512))

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = conv_block(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = conv_block(128, 64)
        self.outc = nn.Conv2d(64, out_classes, kernel_size=1)

        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the refiner U-Net.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Residual delta logits of shape (B, out_classes, H, W).
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = _pad_or_crop_to_match(x, x3)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = _pad_or_crop_to_match(x, x2)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = _pad_or_crop_to_match(x, x1)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        return self.outc(x)


def _pad_or_crop_to_match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Pad or crop x to match the spatial dimensions of ref.

    Central padding is applied if x is smaller; center-cropping if x is larger.

    Args:
        x: Tensor to pad or crop of shape (B, C, H, W).
        ref: Reference tensor of shape (B, C, H_ref, W_ref).

    Returns:
        torch.Tensor: Tensor with spatial dimensions matching ref.
    """
    _, _, h, w = x.shape
    _, _, rh, rw = ref.shape
    dh = rh - h
    dw = rw - w
    if dh > 0 or dw > 0:
        x = F.pad(
            x,
            [
                max(dw // 2, 0),
                max(dw - dw // 2, 0),
                max(dh // 2, 0),
                max(dh - dh // 2, 0),
            ],
        )
    if x.shape[-2] > rh or x.shape[-1] > rw:
        x = x[..., :rh, :rw]
    return x


def build_stage2_input(images: torch.Tensor, preds_s1: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Build 21-channel Stage-2 input.

    Stage-1 NP logits have 2 channels. Stage 2 should receive only the
    foreground probability channel, not both background and foreground.
    """
    tissue_prob = F.softmax(preds_s1["tissue"], dim=1)
    nuclei_prob = F.softmax(preds_s1["nc"], dim=1)
    # Stage-1 NP head in this merged code has one foreground-logit channel.
    # Use sigmoid, not softmax. Softmax over one channel would always be 1.
    np_fg_prob = torch.sigmoid(preds_s1["np"][:, :1])

    return torch.cat(
        [
            images,
            tissue_prob,
            nuclei_prob,
            np_fg_prob,
            preds_s1["hv"],
        ],
        dim=1,
    ).detach()
