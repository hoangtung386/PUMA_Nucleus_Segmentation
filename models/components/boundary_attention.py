"""BoundaryAttentionModule for nuclei boundary detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryAttentionModule(nn.Module):
    def __init__(self, fpn_dim: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(fpn_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, high_res_features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(high_res_features)))
        boundary = torch.sigmoid(self.conv2(x))
        return boundary
