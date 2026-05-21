import torch
import torch.nn as nn
import torch.nn.functional as F


class SCDFA(nn.Module):
    """SC-DFA with Version-2.2 shape: 5 tissue classes x 10 nuclei classes."""

    def __init__(self, num_tissue_classes: int = 5, num_nuclei_classes: int = 10) -> None:
        super().__init__()
        if num_tissue_classes != 5:
            raise ValueError("Merged no-background setup requires num_tissue_classes=5")
        self.W_k = nn.Parameter(torch.empty(num_tissue_classes, num_nuclei_classes))
        nn.init.xavier_uniform_(self.W_k)

    def forward(self, tissue_logits: torch.Tensor) -> torch.Tensor:
        if tissue_logits.shape[1] != self.W_k.shape[0]:
            raise ValueError(f"SCDFA expected {self.W_k.shape[0]} tissue channels, got {tissue_logits.shape[1]}")
        tissue_probs = F.softmax(tissue_logits, dim=1).detach()
        bias = torch.matmul(tissue_probs.permute(0, 2, 3, 1), self.W_k)
        return bias.permute(0, 3, 1, 2).contiguous()
