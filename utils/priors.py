import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialLogitAdjuster(nn.Module):
    """Spatial prior with Version-2.2 shape: 5 tissue classes x 10 nuclei classes."""

    def __init__(self, num_tissue_classes=5, num_nuclei_classes=10):
        super().__init__()
        if num_tissue_classes != 5:
            raise ValueError("Merged no-background setup requires num_tissue_classes=5")
        self.register_buffer("primary_prior", torch.ones(num_tissue_classes, num_nuclei_classes))
        self.register_buffer("metastatic_prior", torch.ones(num_tissue_classes, num_nuclei_classes))

    @staticmethod
    def _normalize_to_log_prior(prior):
        prior = prior.clamp_min(1e-8)
        return torch.log(prior / prior.sum(dim=1, keepdim=True))

    def forward(self, nuclei_logits, tissue_logits, site_type, lambda_scale):
        if float(lambda_scale) == 0.0:
            return nuclei_logits
        if tissue_logits.shape[1] != self.primary_prior.shape[0]:
            raise ValueError(f"Spatial prior expected {self.primary_prior.shape[0]} tissue channels, got {tissue_logits.shape[1]}")

        tissue_probs = F.softmax(tissue_logits, dim=1)
        adjusted = nuclei_logits.clone()
        primary_log = self._normalize_to_log_prior(self.primary_prior).to(nuclei_logits.device)
        metastatic_log = self._normalize_to_log_prior(self.metastatic_prior).to(nuclei_logits.device)

        for b in range(nuclei_logits.shape[0]):
            st = site_type[b] if isinstance(site_type, (list, tuple)) else site_type
            log_prior = primary_log if st == "primary" else metastatic_log
            penalty = torch.matmul(tissue_probs[b].permute(1, 2, 0), log_prior)
            adjusted[b] = nuclei_logits[b] + penalty.permute(2, 0, 1) * float(lambda_scale)
        return adjusted
