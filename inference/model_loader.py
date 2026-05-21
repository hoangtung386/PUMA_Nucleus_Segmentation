"""Model loading utilities for inference.

Supports both:
  - Entity models (``torch.save(model, path)``) — full architecture + weights
  - Legacy state-dict checkpoints (``dict`` with ``model_state`` key)
"""

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


def _load_obj(path, device):
    """Loads a checkpoint from disk, handling both entity and state-dict formats.

    Args:
        path: Path to the checkpoint file.
        device: Torch device to map the checkpoint to.

    Returns:
        Tuple of (model_or_state_dict, metadata_dict).
    """
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, torch.nn.Module):
        metadata = getattr(obj, "_metadata", {})
        return obj, metadata
    return obj, obj if isinstance(obj, dict) else {}


def load_stage1(checkpoint_path: str, device: torch.device):
    """Loads the Stage 1 panoptic model from a checkpoint.

    Supports both entity-saved models and legacy state-dict checkpoints.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Torch device.

    Returns:
        Tuple of (loaded model in eval mode, config dict).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint_path}")

    obj, metadata = _load_obj(checkpoint_path, device)

    if isinstance(obj, torch.nn.Module):
        model = obj
        cfg = metadata.get("inference_config", {})
    else:
        cfg = metadata
        model = UnifiedPanopticNet(
            vit_model=None,
            cnn_model=build_cnn_backbone(pretrained=False),
            num_tissue=5,
            num_nuclei=10,
            load_uni_weights=False,
        )
        model.load_state_dict(extract_state_dict(obj), strict=True)

    model.enable_sc_dfa(bool(cfg.get("use_sc_dfa", False)))
    model.set_spatial_prior_lambda(float(cfg.get("lambda_prior", 0.0)))
    model.to(device).eval()

    logger.info("Loaded Stage 1: %s", checkpoint_path)
    logger.info("Stage 1 settings: use_sc_dfa=%s, lambda_prior=%.4f", model.use_sc_dfa, model.lambda_prior)
    return model, cfg


def load_stage2(checkpoint_path: Optional[str], device: torch.device):
    """Loads the Stage 2 residual nuclei refiner.

    Args:
        checkpoint_path: Optional path to the checkpoint. If None, returns (None, 0.0).
        device: Torch device.

    Returns:
        Tuple of (model in eval mode or None, alpha blending weight).
    """
    if checkpoint_path is None:
        return None, 0.0
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.info("Stage 2 checkpoint not found: %s. Running Stage 1 only.", checkpoint_path)
        return None, 0.0

    obj, metadata = _load_obj(checkpoint_path, device)

    if isinstance(obj, torch.nn.Module):
        model = obj
        cfg = metadata.get("config", {})
    else:
        cfg = metadata
        in_channels = int(cfg.get("in_channels", 21))
        out_classes = int(cfg.get("out_classes", 10))
        if in_channels != 21:
            raise RuntimeError(
                f"Stage 2 checkpoint expects {in_channels} input channels, but merged model requires 21."
            )
        model = ResidualNucleiRefinerUNet(in_channels=in_channels, out_classes=out_classes)
        model.load_state_dict(extract_state_dict(obj), strict=True)

    alpha = float(cfg.get("alpha", cfg.get("alpha_end", 0.35)))
    model.to(device).eval()
    logger.info("Loaded Stage 2: %s | alpha=%.3f", checkpoint_path, alpha)
    return model, alpha
