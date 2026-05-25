"""ContextFusionModule — FiLM-style conditioning of FPN features using global context descriptors."""

from typing import Dict

import torch
import torch.nn as nn


class ContextFusionModule(nn.Module):
    def __init__(self, context_dim: int = 256, fpn_dim: int = 256) -> None:
        super().__init__()
        self.gamma = nn.Linear(context_dim, fpn_dim)
        self.beta = nn.Linear(context_dim, fpn_dim)

    def forward(self, fpn_feats: Dict[str, torch.Tensor], context_descriptor: torch.Tensor) -> Dict[str, torch.Tensor]:
        gamma = self.gamma(context_descriptor).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(context_descriptor).unsqueeze(-1).unsqueeze(-1)
        return {k: gamma * v + beta for k, v in fpn_feats.items()}
