from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components import ContextEncoder
from models.components.context_fusion import ContextFusionModule
from models.decoders import ParallelDecoders
from models.encoder import UnifiedPanopticEncoder
from models.fpn_aggregator import HierarchicalFPN
from utils.sc_dfa import SCDFA


class UnifiedPanopticNet(nn.Module):
    def __init__(
        self,
        virchow2_model_name: str = "paige-ai/Virchow2",
        cnn_model: Any = None,
        num_tissue: int = 5,
        num_nuclei: int = 10,
        fine_tune_last_n_blocks: int = 6,
        load_encoder_weights: bool = True,
        use_context_encoder: bool = False,
    ) -> None:
        super().__init__()
        if num_tissue != 5:
            raise ValueError("This model uses exactly 5 tissue classes. Background is ignored, not predicted.")

        self.encoder = UnifiedPanopticEncoder(
            virchow2_model_name=virchow2_model_name,
            cnn_model=cnn_model,
            fine_tune_last_n_blocks=fine_tune_last_n_blocks,
            load_weights=load_encoder_weights,
        )
        cnn_dims = cnn_model.feature_info.channels() if hasattr(cnn_model, "feature_info") else [96, 192, 384, 768]
        self.fpn = HierarchicalFPN(cnn_dims=cnn_dims)
        self.decoders = ParallelDecoders(num_tissue=num_tissue, num_nuclei=num_nuclei, low_level_channels=cnn_dims[0])

        self.use_context_encoder = use_context_encoder
        if use_context_encoder:
            self.context_encoder = ContextEncoder(output_dim=256, output_mode="global")
            self.context_fusion = ContextFusionModule(context_dim=256, fpn_dim=256)

        self.site_embed = nn.Embedding(9, 256)
        self.sc_dfa = SCDFA(num_tissue_classes=num_tissue, num_nuclei_classes=num_nuclei)
        self.use_sc_dfa = False
        self.lambda_sc_dfa = 0.0

    def enable_sc_dfa(self, state: bool = True) -> None:
        self.use_sc_dfa = bool(state)
        if self.use_sc_dfa and self.lambda_sc_dfa <= 0.0:
            self.lambda_sc_dfa = 1.0
        if not self.use_sc_dfa:
            self.lambda_sc_dfa = 0.0

    def set_sc_dfa_lambda(self, value: float) -> None:
        value = float(value)
        value = max(0.0, min(value, 1.0))
        self.lambda_sc_dfa = value
        self.use_sc_dfa = value > 0.0

    def forward(
        self,
        images: torch.Tensor,
        site_ids: Optional[torch.Tensor] = None,
        context_roi: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        vit_tokens, cnn_features, vit_intermediate = self.encoder(images)
        fpn_feats, low_level_feat = self.fpn(vit_tokens, cnn_features, vit_intermediate, img_size=images.shape[-1])

        if self.use_context_encoder and context_roi is not None:
            ctx_desc = self.context_encoder(context_roi)
            fpn_feats = self.context_fusion(fpn_feats, ctx_desc)

        if site_ids is not None:
            site_bias = self.site_embed(site_ids).unsqueeze(-1).unsqueeze(-1)
            for k in fpn_feats:
                fpn_feats[k] = fpn_feats[k] + site_bias

        tissue_logits, np_logits, nc_logits, hv_logits, boundary_map = self.decoders(
            fpn_feats, low_level_feat, vit_intermediate
        )
        out_size = images.shape[-2:]
        tissue_logits = F.interpolate(tissue_logits, size=out_size, mode="bilinear", align_corners=False)
        np_logits = F.interpolate(np_logits, size=out_size, mode="bilinear", align_corners=False)
        nc_logits = F.interpolate(nc_logits, size=out_size, mode="bilinear", align_corners=False)
        hv_logits = F.interpolate(hv_logits, size=out_size, mode="bilinear", align_corners=False)
        boundary_map = F.interpolate(boundary_map, size=out_size, mode="bilinear", align_corners=False)

        if self.use_sc_dfa and self.lambda_sc_dfa > 0.0:
            nc_logits = nc_logits + self.lambda_sc_dfa * self.sc_dfa(tissue_logits)

        return {
            "tissue": tissue_logits,
            "np": np_logits,
            "nc": nc_logits,
            "hv": hv_logits,
            "boundary": boundary_map,
        }
