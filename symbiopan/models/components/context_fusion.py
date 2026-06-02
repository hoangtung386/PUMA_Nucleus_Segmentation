"""ContextFusionModule — FiLM-style conditioning of FPN features using global context descriptors."""

import torch
import torch.nn as nn


class ContextFusionModule(nn.Module):
    def __init__(self, context_dim: int = 256, fpn_dim: int = 256) -> None:
        super().__init__()
        self.gamma = nn.Linear(context_dim, fpn_dim)
        self.beta = nn.Linear(context_dim, fpn_dim)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, fpn_feats: dict[str, torch.Tensor], context_descriptor: torch.Tensor) -> dict[str, torch.Tensor]:
        gamma = self.gamma(context_descriptor).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(context_descriptor).unsqueeze(-1).unsqueeze(-1)
        return {k: (1.0 + gamma) * v + beta for k, v in fpn_feats.items()}
