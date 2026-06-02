
import torch
import torch.nn as nn
import torch.nn.functional as F

from symbiopan.common.logging import get_logger
from symbiopan.losses.segmentation import FocalBCELoss, FocalTverskyLoss, SafeCrossEntropyLoss, SoftDiceLoss

logger = get_logger(__name__)


class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(
        self,
        tissue_weights: torch.Tensor | None = None,
        nuclei_weights: torch.Tensor | None = None,
        num_tasks: int = 5,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.use_focal_tversky = False
        self.focal_tversky_weight = 0.0

        if tissue_weights is None:
            tissue_weights = torch.tensor([1.0, 1.0, 2.0, 3.0, 3.0, 4.0], dtype=torch.float32)
        if nuclei_weights is None:
            nuclei_weights = torch.tensor([1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0], dtype=torch.float32)

        self.register_buffer("tissue_weights", tissue_weights.float())
        self.register_buffer("nuclei_weights", nuclei_weights.float())

        self.ce_tissue = SafeCrossEntropyLoss(weight=self.tissue_weights, ignore_index=None)
        self.ce_nc = SafeCrossEntropyLoss(weight=self.nuclei_weights, ignore_index=ignore_index)

        self.ft_tissue = FocalTverskyLoss(
            alpha=0.30, beta=0.70, gamma=1.25, ignore_index=None, class_weights=self.tissue_weights
        )
        self.ft_nc = FocalTverskyLoss(
            alpha=0.25, beta=0.75, gamma=1.50, ignore_index=ignore_index, class_weights=self.nuclei_weights
        )
        self.np_bce = FocalBCELoss(alpha=0.45, gamma=2.0)
        self.np_dice = SoftDiceLoss()

    def set_focal_tversky_weight(self, weight: float) -> None:
        weight = float(weight)
        weight = max(0.0, min(weight, 1.0))
        self.focal_tversky_weight = weight
        self.use_focal_tversky = weight > 0.0

    def switch_to_focal_tversky(self) -> None:
        if not self.use_focal_tversky or self.focal_tversky_weight < 1.0:
            self.set_focal_tversky_weight(1.0)
            logger.info("Switched tissue/nuclei losses to CE + full FocalTversky")

    def forward(
        self, preds: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, list[float]]:
        tissue_target = targets["tissue_sem"]
        nuclei_target = targets["nuclei_nc"]

        focal_weight = float(getattr(self, "focal_tversky_weight", 0.0))
        l_tissue = self.ce_tissue(preds["tissue"], tissue_target)
        l_nc = self.ce_nc(preds["nc"], nuclei_target)
        if focal_weight > 0.0:
            l_tissue = l_tissue + focal_weight * self.ft_tissue(preds["tissue"], tissue_target)
            l_nc = l_nc + focal_weight * self.ft_nc(preds["nc"], nuclei_target)

        l_np = self.np_bce(preds["np"], targets["nuclei_np"]) + self.np_dice(preds["np"], targets["nuclei_np"])

        hv_mask = targets["nuclei_np"].unsqueeze(1).bool().expand_as(preds["hv"])
        if torch.any(hv_mask):
            l_hv = F.smooth_l1_loss(preds["hv"][hv_mask], targets["nuclei_hv"][hv_mask], beta=0.5, reduction="mean")
        else:
            l_hv = preds["hv"].sum() * 0.0

        losses = [l_tissue, l_np, l_nc, l_hv, torch.tensor(0.0, device=l_tissue.device)]

        multipliers = [2.5, 1.0, 2.8, 1.0, 0.0]
        total = 0.0
        for i, loss in enumerate(losses):
            total = total + multipliers[i] * (torch.exp(-self.log_vars[i]) * loss + self.log_vars[i])
        return total, [float(x.detach().item()) for x in losses[:4]]
