"""Virchow2 ViT-H/14 encoder with fine-tuning and multi-block feature extraction."""

import json
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModel, ViTConfig

from symbiopan.common.logging import get_logger
from symbiopan.models.cross_attention import SpatialInjector

logger = get_logger(__name__)

_DEFAULT_VIRCHOW2_CFG: dict[str, int | float] = {
    "hidden_size": 1280,
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "intermediate_size": 5120,
    "patch_size": 14,
    "image_size": 1024,
    "num_channels": 3,
    "layer_norm_eps": 1e-6,
}


def _load_virchow2_config(model_name: str) -> ViTConfig:
    cfg = dict(_DEFAULT_VIRCHOW2_CFG)
    try:
        config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        with open(config_path) as f:
            cfg.update({k: v for k, v in json.load(f).items() if k in _DEFAULT_VIRCHOW2_CFG})
        logger.info("Loaded Virchow2 config from cache: %s", config_path)
    except Exception:
        logger.warning("Virchow2 config not available; using hardcoded ViT-H/14 config")
    return ViTConfig(**cfg)


def build_virchow2_vit(
    model_name: str = "paige-ai/Virchow2",
    load_weights: bool = True,
    fine_tune_last_n_blocks: int = 6,
) -> Any:
    """Build a Virchow2 ViT-H/14 (32 blocks, dim=1280) with selective fine-tuning."""
    config = _load_virchow2_config(model_name)

    if hasattr(config, "num_labels") and config.num_labels is None:
        config.num_labels = 0

    if load_weights:
        try:
            model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
            logger.info("Loaded Virchow2: %s", model_name)
        except Exception:
            logger.warning("Failed to load Virchow2 weights; using random init")
            model = AutoModel.from_config(config, trust_remote_code=True)
    else:
        model = AutoModel.from_config(config, trust_remote_code=True)
        logger.info("Built Virchow2 from config (no weights): %s", model_name)

    for param in model.parameters():
        param.requires_grad = False

    if fine_tune_last_n_blocks > 0 and hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        total_blocks = len(model.encoder.layer)
        start_finetune = total_blocks - fine_tune_last_n_blocks
        for i in range(start_finetune, total_blocks):
            for param in model.encoder.layer[i].parameters():
                param.requires_grad = True
        logger.info("Fine-tuning last %d/%d ViT blocks", fine_tune_last_n_blocks, total_blocks)

    return model


class UnifiedPanopticEncoder(nn.Module):
    """ViT + CNN encoder with 4 cross-attention bridges and 4 intermediate ViT taps."""

    DEFAULT_BRIDGE_INTERVAL = 8
    DEFAULT_INTERMEDIATE_INDICES = (8, 16, 24, 31)

    def __init__(
        self,
        virchow2_model_name: str = "paige-ai/Virchow2",
        cnn_model: Any = None,
        fine_tune_last_n_blocks: int = 6,
        load_weights: bool = True,
    ) -> None:
        super().__init__()
        self.vit_model = build_virchow2_vit(
            model_name=virchow2_model_name,
            load_weights=load_weights,
            fine_tune_last_n_blocks=fine_tune_last_n_blocks,
        )
        self.cnn_model = cnn_model
        self.fine_tune = fine_tune_last_n_blocks > 0

        vit_dim = self._get_vit_dim()
        if hasattr(self.cnn_model, "feature_info"):
            cnn_dims = self.cnn_model.feature_info.channels()
        else:
            cnn_dims = [96, 192, 384, 768]

        self.bridges = nn.ModuleList([SpatialInjector(vit_dim=vit_dim, cnn_dims=cnn_dims) for _ in range(4)])
        self.vit_intermediate_indices = self.DEFAULT_INTERMEDIATE_INDICES

    def _get_vit_dim(self) -> int:
        if hasattr(self.vit_model, "config"):
            return getattr(self.vit_model.config, "hidden_size", 1280)
        return 1280

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        cnn_features = self.cnn_model(img)

        x = self.vit_model.get_input_embeddings()(img) if hasattr(self.vit_model, "get_input_embeddings") else None
        if x is None and hasattr(self.vit_model, "embeddings"):
            x = self.vit_model.embeddings(img)
        if x is None and hasattr(self.vit_model, "patch_embed"):
            x = self.vit_model.patch_embed(img)
        if x is None:
            raise RuntimeError("Virchow2 model has no recognizable patch-embedding module")

        if hasattr(self.vit_model, "encoder") and hasattr(self.vit_model.encoder, "layer"):
            blocks = self.vit_model.encoder.layer
        else:
            blocks = self.vit_model.blocks if hasattr(self.vit_model, "blocks") else []

        bridge_idx = 0
        intermediate_features = []
        for i, block in enumerate(blocks):
            out = block(x)
            x = out[0] if isinstance(out, tuple) else out
            if i in self.vit_intermediate_indices and bridge_idx < 4:
                intermediate_features.append(x.clone())
            if (i + 1) % self.DEFAULT_BRIDGE_INTERVAL == 0 and bridge_idx < len(self.bridges):
                x = self.bridges[bridge_idx](vit_tokens=x, cnn_features=cnn_features)
                bridge_idx += 1

        if hasattr(self.vit_model, "norm"):
            x = self.vit_model.norm(x)
        elif hasattr(self.vit_model, "ln_f"):
            x = self.vit_model.ln_f(x)

        spatial_list: list[torch.Tensor] = []
        gh = img.shape[-2] // _DEFAULT_VIRCHOW2_CFG["patch_size"]
        gw = img.shape[-1] // _DEFAULT_VIRCHOW2_CFG["patch_size"]
        n_spatial = gh * gw
        for feat in intermediate_features:
            feat_spatial = feat[:, -n_spatial:, :]
            spatial_list.append(feat_spatial.transpose(1, 2).reshape(-1, feat_spatial.shape[-1], gh, gw))
        vit_intermediate_tensor = torch.stack(spatial_list, dim=0) if spatial_list else x.unsqueeze(0)

        return x, cnn_features, vit_intermediate_tensor
