import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super().__init__()
        self.ignore_index = int(ignore_index)
        if weight is None:
            self.weight = None
        else:
            self.register_buffer("weight", weight.float())

    def forward(self, logits, targets):
        valid = targets != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            ignore_index=self.ignore_index,
        )


class FocalTverskyLoss(nn.Module):
    """
    Multi-class focal Tversky loss.

    alpha weights false positives, beta weights false negatives.
    For rare classes, beta should be larger than alpha because the main failure is
    missing rare objects/classes.
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=1.25, smooth=1e-5, ignore_index=255, class_weights=None):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        self.ignore_index = int(ignore_index)
        if class_weights is None:
            self.class_weights = None
        else:
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits, targets):
        valid = targets != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0

        c = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_safe = targets.clone()
        targets_safe[~valid] = 0
        one_hot = F.one_hot(targets_safe.clamp(0, c - 1), num_classes=c).permute(0, 3, 1, 2).float()
        valid_mask = valid.unsqueeze(1).float()
        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        tp = (probs * one_hot).sum(dim=(2, 3))
        fp = (probs * (1.0 - one_hot)).sum(dim=(2, 3))
        fn = ((1.0 - probs) * one_hot).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = torch.pow(1.0 - tversky, self.gamma)

        # Do not let absent classes dominate a batch, but keep false-positive penalty
        # if class is predicted in a batch with zero target pixels.
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(loss.device).view(1, -1)
        return loss.mean()


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits.squeeze(1))
        targets = targets.float()
        inter = (probs * targets).sum(dim=(1, 2))
        den = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        return (1.0 - (2.0 * inter + self.smooth) / (den + self.smooth)).mean()


class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.45, gamma=2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        logits = logits.squeeze(1)
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - pt).pow(self.gamma) * bce).mean()


class MultiTaskUncertaintyLoss(nn.Module):
    """
    Rare-focused loss for merged 5-tissue/no-background model.

    targets['tissue_sem'] must be 0..4 for real tissue and 255 for background.
    targets['nuclei_nc'] must be 0..9 for nuclei and 255 for non-nucleus.
    """

    def __init__(self, tissue_weights=None, nuclei_weights=None, num_tasks=4, ignore_index=255):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.use_focal_tversky = False
        self.focal_tversky_weight = 0.0

        if tissue_weights is None:
            # Internal order: stroma, blood_vessel, tumor, epidermis, necrosis.
            # Strong weights for blood vessel/epidermis/necrosis.
            tissue_weights = torch.tensor([1.0, 4.0, 0.8, 3.0, 7.0], dtype=torch.float32)
        if nuclei_weights is None:
            # Order: tumor, lymphocyte, plasma, histiocyte, melanophage,
            # neutrophil, stroma, epithelium, endothelium, apoptosis.
            nuclei_weights = torch.tensor([0.8, 1.0, 7.0, 2.5, 4.5, 8.0, 2.0, 2.5, 5.5, 8.0], dtype=torch.float32)

        self.register_buffer("tissue_weights", tissue_weights.float())
        self.register_buffer("nuclei_weights", nuclei_weights.float())

        self.ce_tissue = SafeCrossEntropyLoss(weight=self.tissue_weights, ignore_index=ignore_index)
        self.ce_nc = SafeCrossEntropyLoss(weight=self.nuclei_weights, ignore_index=ignore_index)

        # beta > alpha penalizes false negatives more than false positives.
        self.ft_tissue = FocalTverskyLoss(
            alpha=0.30,
            beta=0.70,
            gamma=1.25,
            ignore_index=ignore_index,
            class_weights=self.tissue_weights,
        )
        self.ft_nc = FocalTverskyLoss(
            alpha=0.25,
            beta=0.75,
            gamma=1.50,
            ignore_index=ignore_index,
            class_weights=self.nuclei_weights,
        )
        self.np_bce = FocalBCELoss(alpha=0.45, gamma=2.0)
        self.np_dice = SoftDiceLoss()

    def set_focal_tversky_weight(self, weight):
        """Smoothly blend FN-focused FocalTversky into the semantic losses.

        weight=0.0 means CE only.
        weight=0.5 means CE + 0.5 * FocalTversky.
        This avoids the abrupt objective jump caused by a binary loss switch.
        """
        weight = float(weight)
        weight = max(0.0, min(weight, 1.0))
        self.focal_tversky_weight = weight
        self.use_focal_tversky = weight > 0.0

    def switch_to_focal_tversky(self):
        # Backward-compatible API for older scripts.
        if not self.use_focal_tversky or self.focal_tversky_weight < 1.0:
            self.set_focal_tversky_weight(1.0)
            print("[Loss] Switched tissue/nuclei semantic losses to CE + full FN-focused FocalTversky")

    def forward(self, preds, targets):
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
            l_hv = F.smooth_l1_loss(
                preds["hv"][hv_mask],
                targets["nuclei_hv"][hv_mask],
                beta=0.5,
                reduction="mean",
            )
        else:
            l_hv = preds["hv"].sum() * 0.0

        losses = [l_tissue, l_np, l_nc, l_hv]

        # More semantic emphasis than before because rare class recognition is weak.
        multipliers = [2.5, 1.0, 2.8, 1.0]
        total = 0.0
        for i, loss in enumerate(losses):
            total = total + multipliers[i] * (torch.exp(-self.log_vars[i]) * loss + self.log_vars[i])
        return total, [float(x.detach().item()) for x in losses]


# Compatibility alias.
DecoupledPUMALoss = MultiTaskUncertaintyLoss
