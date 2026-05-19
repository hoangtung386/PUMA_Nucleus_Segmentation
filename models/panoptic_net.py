import torch.nn as nn
import torch.nn.functional as F

from models.decoders import ParallelDecoders
from models.encoder import UnifiedPanopticEncoder
from models.fpn_aggregator import FPNAggregator
from utils.priors import SpatialLogitAdjuster
from utils.sc_dfa import SCDFA


class UnifiedPanopticNet(nn.Module):
    """
    Merged architecture:
      - Version-2.2 dual encoder + FPN + parallel decoder structure.
      - Version-4 class names/preprocessed labels.
      - No tissue background output channel: tissue logits are [B,5,H,W].
      - SC-DFA and spatial prior remain [5 tissue x 10 nuclei].
    """

    def __init__(self, vit_model, cnn_model, num_tissue=5, num_nuclei=10, load_uni_weights=True):
        super().__init__()
        if num_tissue != 5:
            raise ValueError("This merged model uses exactly 5 tissue classes. Background is ignored, not predicted.")
        self.encoder = UnifiedPanopticEncoder(
            cnn_model=cnn_model,
            local_weight_dir=vit_model,
            load_uni_weights=load_uni_weights,
        )
        cnn_dims = cnn_model.feature_info.channels() if hasattr(cnn_model, "feature_info") else [40, 80, 160, 320]
        self.fpn = FPNAggregator(cnn_dims=cnn_dims)
        self.decoders = ParallelDecoders(num_tissue=num_tissue, num_nuclei=num_nuclei)

        self.cellpose_adapter = nn.Sequential(
            nn.Conv2d(2, 2, 3, padding=1),
            nn.InstanceNorm2d(2),
            nn.GELU(),
            nn.Conv2d(2, 2, 3, padding=1),
        )

        self.sc_dfa = SCDFA(num_tissue_classes=num_tissue, num_nuclei_classes=num_nuclei)
        self.spatial_prior = SpatialLogitAdjuster(num_tissue_classes=num_tissue, num_nuclei_classes=num_nuclei)
        self.use_sc_dfa = False
        self.lambda_sc_dfa = 0.0
        self.lambda_prior = 0.0

    def enable_sc_dfa(self, state=True):
        self.use_sc_dfa = bool(state)
        # Backward compatibility: older training code expected enable_sc_dfa(True)
        # to apply the full SC-DFA correction. Smooth-schedule training should call
        # set_sc_dfa_lambda(...) each epoch.
        if self.use_sc_dfa and self.lambda_sc_dfa <= 0.0:
            self.lambda_sc_dfa = 1.0
        if not self.use_sc_dfa:
            self.lambda_sc_dfa = 0.0

    def set_sc_dfa_lambda(self, value):
        value = float(value)
        value = max(0.0, min(value, 1.0))
        self.lambda_sc_dfa = value
        self.use_sc_dfa = value > 0.0

    def set_spatial_prior_lambda(self, value):
        self.lambda_prior = float(value)

    def forward(self, images, cellpose_flows, site_types=None):
        vit_tokens, cnn_features = self.encoder(images)
        fpn_feats = self.fpn(vit_tokens, cnn_features, img_size=images.shape[-1])
        cp_prior = self.cellpose_adapter(cellpose_flows)

        tissue_logits, np_logits, nc_logits, hv_logits = self.decoders(fpn_feats, cp_prior)
        out_size = images.shape[-2:]
        tissue_logits = F.interpolate(tissue_logits, size=out_size, mode="bilinear", align_corners=False)
        np_logits = F.interpolate(np_logits, size=out_size, mode="bilinear", align_corners=False)
        nc_logits = F.interpolate(nc_logits, size=out_size, mode="bilinear", align_corners=False)
        hv_logits = F.interpolate(hv_logits, size=out_size, mode="bilinear", align_corners=False)

        if self.use_sc_dfa and self.lambda_sc_dfa > 0.0:
            nc_logits = nc_logits + self.lambda_sc_dfa * self.sc_dfa(tissue_logits)

        if self.lambda_prior > 0.0 and site_types is not None:
            nc_logits = self.spatial_prior(nc_logits, tissue_logits, site_types, self.lambda_prior)

        return {"tissue": tissue_logits, "np": np_logits, "nc": nc_logits, "hv": hv_logits}
