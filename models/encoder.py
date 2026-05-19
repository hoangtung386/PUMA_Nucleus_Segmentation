import os
from pathlib import Path

import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from models.cross_attention import SpatialInjector
from training.logging_utils import logger


def build_uni_vit():
    timm_kwargs = {
        "model_name": "vit_large_patch16_224",
        "img_size": 224,
        "patch_size": 16,
        "depth": 24,
        "num_heads": 16,
        "init_values": 1e-5,
        "embed_dim": 1024,
        "num_classes": 0,
        "dynamic_img_size": True,
    }
    return timm.create_model(pretrained=False, **timm_kwargs)


def get_frozen_uni_model(local_dir=Path.cwd(), load_weights=True, allow_download=True):
    """
    Build UNI ViT-L/16.

    Training: pass local_dir containing pytorch_model.bin.
    Docker inference with a full checkpoint: pass load_weights=False, then load_state_dict(strict=True).
    """
    model = build_uni_vit()

    if load_weights:
        if local_dir is None:
            raise ValueError("local_dir cannot be None when load_weights=True")
        local_dir = Path(local_dir)
        os.makedirs(local_dir, exist_ok=True)
        weight_path = local_dir / "pytorch_model.bin"
        if not weight_path.exists():
            if not allow_download:
                raise FileNotFoundError(f"UNI weight file not found: {weight_path}")
            logger.info("Downloading UNI weights from HuggingFace...")
            hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=str(local_dir))
        model.load_state_dict(torch.load(weight_path, map_location="cpu"), strict=True)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


class UnifiedPanopticEncoder(nn.Module):
    def __init__(self, cnn_model, local_weight_dir=Path.cwd(), load_uni_weights=True):
        super().__init__()
        self.vit_model = get_frozen_uni_model(local_dir=local_weight_dir, load_weights=load_uni_weights)
        self.cnn_model = cnn_model

        if hasattr(self.cnn_model, "feature_info"):
            cnn_dims = self.cnn_model.feature_info.channels()
        else:
            cnn_dims = [40, 80, 160, 320]

        self.bridges = nn.ModuleList([
            SpatialInjector(vit_dim=1024, cnn_dims=cnn_dims) for _ in range(4)
        ])

    def forward(self, img):
        cnn_features = self.cnn_model(img)

        x = self.vit_model.patch_embed(img)
        x = self.vit_model._pos_embed(x)

        bridge_idx = 0
        for i, block in enumerate(self.vit_model.blocks):
            x = block(x)
            if (i + 1) % 6 == 0:
                x = self.bridges[bridge_idx](vit_tokens=x, cnn_features=cnn_features)
                bridge_idx += 1

        x = self.vit_model.norm(x)
        return x, cnn_features
