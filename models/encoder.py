"""Virchow2 ViT-H/14 encoder with fine-tuning and multi-block feature extraction."""

import json
import math
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModel, ViTConfig

from models.cross_attention import SpatialInjector
from training.logging_utils import logger

_VIRCHOW2_CFG = dict(
    hidden_size=1280,
    num_hidden_layers=32,
    num_attention_heads=16,
    intermediate_size=5120,
    patch_size=14,
    image_size=1024,
    num_channels=3,
    layer_norm_eps=1e-6,
)


def _load_virchow2_config(model_name: str) -> ViTConfig:
    try:
        config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        logger.info("Loaded Virchow2 config from cache: %s", config_path)
        return ViTConfig(
            **{k: cfg[k] for k in _VIRCHOW2_CFG if k in cfg},
        )
    except Exception:
        pass

    logger.warning("Virchow2 config not available; using hardcoded ViT-H/14 config")
    return ViTConfig(**_VIRCHOW2_CFG)


def build_virchow2_vit(
    model_name: str = "paige-ai/Virchow2",
    load_weights: bool = True,
    fine_tune_last_n_blocks: int = 6,
) -> Any:
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


def extract_intermediate_features(
    model: Any, hidden_states: torch.Tensor, bridge_indices: Tuple[int, ...]
) -> List[torch.Tensor]:
    intermediate = []
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        for i, layer in enumerate(model.encoder.layer):
            hidden_states = layer(hidden_states)[0]
            if i in bridge_indices:
                intermediate.append(hidden_states)
    return intermediate


class UnifiedPanopticEncoder(nn.Module):
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

        self.vit_intermediate_indices = (8, 16, 24, 31)
        self._patch_proj: nn.Linear | None = None

    def _get_vit_dim(self) -> int:
        if hasattr(self.vit_model, "config"):
            return getattr(self.vit_model.config, "hidden_size", 1280)
        return 1280

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        cnn_features = self.cnn_model(img)

        x = self.vit_model.get_input_embeddings()(img) if hasattr(self.vit_model, "get_input_embeddings") else None
        if x is None:
            x = self.vit_model.embeddings(img) if hasattr(self.vit_model, "embeddings") else None
        if x is None:
            x = (
                self.vit_model.patch_embed(img)
                if hasattr(self.vit_model, "patch_embed")
                else self._simple_patch_embed(img)
            )

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
            if (i + 1) % 8 == 0 and bridge_idx < len(self.bridges):
                x = self.bridges[bridge_idx](vit_tokens=x, cnn_features=cnn_features)
                bridge_idx += 1

        if hasattr(self.vit_model, "norm"):
            x = self.vit_model.norm(x)
        elif hasattr(self.vit_model, "ln_f"):
            x = self.vit_model.ln_f(x)

        spatial_list = []
        for feat in intermediate_features:
            feat_no_cls = feat[:, 1:, :]
            n_tokens = feat_no_cls.shape[1]
            grid = math.isqrt(n_tokens)
            spatial_list.append(feat_no_cls.transpose(1, 2).reshape(-1, feat_no_cls.shape[-1], grid, grid))
        vit_intermediate_tensor = torch.stack(spatial_list, dim=0) if spatial_list else x.unsqueeze(0)

        return x, cnn_features, vit_intermediate_tensor

    def _simple_patch_embed(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        patch_size = 14
        h, w = H // patch_size, W // patch_size
        x = img.reshape(B, C, h, patch_size, w, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, C * patch_size * patch_size)
        if hasattr(self.vit_model, "config") and hasattr(self.vit_model.config, "hidden_size"):
            hidden = self.vit_model.config.hidden_size
            if self._patch_proj is None or self._patch_proj.out_features != hidden:
                self._patch_proj = nn.Linear(C * patch_size * patch_size, hidden).to(x.device, x.dtype)
            x = self._patch_proj(x)
        return x
