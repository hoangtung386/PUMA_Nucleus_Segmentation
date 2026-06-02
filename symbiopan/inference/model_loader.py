"""Model loading utilities for inference — v8 CellPath."""

from pathlib import Path

import torch

from symbiopan.common.logging import get_logger
from symbiopan.models import UnifiedPanopticNet, build_cnn_backbone
from symbiopan.training.checkpoint import extract_state_dict

logger = get_logger(__name__)


def load_stage1(checkpoint_path: str, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint_path}")

    obj = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(obj, torch.nn.Module):
        model = obj
        logger.info("Loaded Stage 1 entity model: %s", checkpoint_path)
    else:
        cnn = build_cnn_backbone(pretrained=False)
        model = UnifiedPanopticNet(
            virchow2_model_name="paige-ai/Virchow2",
            cnn_model=cnn,
            num_tissue=6,
            num_nuclei=10,
            fine_tune_last_n_blocks=6,
            load_encoder_weights=False,
        )
        model.load_state_dict(extract_state_dict(obj), strict=True)
        logger.info("Loaded Stage 1 from state dict: %s", checkpoint_path)

    model.enable_sc_dfa(True)
    model.to(device).eval()
    return model
