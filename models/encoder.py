import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from models.cross_attention import SpatialInjector
from training.logging_utils import logger


def build_uni_vit() -> Any:
    """Build a UNI ViT-L/16 backbone without pretrained weights.

    Returns:
        timm.models.VisionTransformer: A ViT-L/16 model configured with
            UNI architecture parameters (embed_dim=1024, 24 blocks, 16 heads).
    """
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


def get_frozen_uni_model(
    local_dir: Optional[Path] = Path.cwd(),
    load_weights: bool = True,
    allow_download: bool = True,
) -> Any:
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
    """Dual-encoder that fuses CNN features with frozen UNI ViT features via spatial attention bridges.

    Processes the input image through both a CNN backbone and a frozen UNI ViT-L/16,
    injecting multi-scale CNN features into the ViT stream at every 6th ViT block
    using SpatialInjector modules.
    """

    def __init__(self, cnn_model: Any, local_weight_dir: Path = Path.cwd(), load_uni_weights: bool = True) -> None:
        """Initialize the unified panoptic encoder.

        Args:
            cnn_model: A timm CNN backbone model with feature_info attribute.
            local_weight_dir: Directory containing UNI pretrained weights
                (pytorch_model.bin).
            load_uni_weights: Whether to load pretrained UNI weights.
        """
        super().__init__()
        self.vit_model = get_frozen_uni_model(local_dir=local_weight_dir, load_weights=load_uni_weights)
        self.cnn_model = cnn_model

        if hasattr(self.cnn_model, "feature_info"):
            cnn_dims = self.cnn_model.feature_info.channels()
        else:
            cnn_dims = [40, 80, 160, 320]

        self.bridges = nn.ModuleList([SpatialInjector(vit_dim=1024, cnn_dims=cnn_dims) for _ in range(4)])

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through dual encoder.

        Args:
            img: Input image tensor of shape (B, 3, H, W).

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]: The final ViT
                class token and patch embeddings after norm, and the list
                of CNN feature maps from each stage.
        """
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
