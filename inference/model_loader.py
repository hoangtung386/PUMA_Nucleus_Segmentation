"""Model loading utilities for inference."""

from pathlib import Path
from typing import Optional

import torch

from models import (
    ResidualNucleiRefinerUNet,
    UnifiedPanopticNet,
    build_cnn_backbone,
)
from training.checkpoint import extract_state_dict
from training.logging_utils import logger


def load_stage1(checkpoint_path: str, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("inference_config", {}) if isinstance(ckpt, dict) else {}

    model = UnifiedPanopticNet(
        vit_model=None,
        cnn_model=build_cnn_backbone(pretrained=False),
        num_tissue=5,
        num_nuclei=10,
        load_uni_weights=False,
    )
    model.load_state_dict(extract_state_dict(ckpt), strict=True)
    model.enable_sc_dfa(bool(cfg.get("use_sc_dfa", False)))
    model.set_spatial_prior_lambda(float(cfg.get("lambda_prior", 0.0)))
    model.to(device).eval()

    logger.info("Loaded Stage 1: %s", checkpoint_path)
    logger.info("Stage 1 settings: use_sc_dfa=%s, lambda_prior=%.4f", model.use_sc_dfa, model.lambda_prior)
    return model, cfg


def load_stage2(checkpoint_path: Optional[str], device: torch.device):
    if checkpoint_path is None:
        return None, 0.0
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.info("Stage 2 checkpoint not found: %s. Running Stage 1 only.", checkpoint_path)
        return None, 0.0

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    in_channels = int(cfg.get("in_channels", 21))
    out_classes = int(cfg.get("out_classes", 10))
    if in_channels != 21:
        raise RuntimeError(f"Stage 2 checkpoint expects {in_channels} input channels, but merged model requires 21.")

    model = ResidualNucleiRefinerUNet(in_channels=in_channels, out_classes=out_classes)
    model.load_state_dict(extract_state_dict(ckpt), strict=True)
    alpha = float(ckpt.get("alpha", cfg.get("alpha_end", 0.35))) if isinstance(ckpt, dict) else 0.35
    model.to(device).eval()
    logger.info("Loaded Stage 2: %s | alpha=%.3f", checkpoint_path, alpha)
    return model, alpha
