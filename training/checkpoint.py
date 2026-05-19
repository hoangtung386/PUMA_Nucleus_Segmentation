"""Checkpoint save/load utilities shared across training stages."""

from pathlib import Path

import torch

from training.logging_utils import logger


def safe_torch_save(obj: dict, path: str | Path) -> None:
    """Save checkpoint atomically to avoid partial-write corruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, str(tmp))
    _ = torch.load(str(tmp), map_location="cpu", weights_only=False)
    tmp.replace(path)
    logger.info("Checkpoint saved and verified: %s", path)


def load_large_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict:
    """Load checkpoint with DataParallel key stripping."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(str(path), map_location=device, weights_only=False)


def extract_state_dict(checkpoint: dict) -> dict:
    """Extract model state dict from various checkpoint formats."""
    if isinstance(checkpoint, dict):
        for key in ["model_state", "model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format")
    return {k.removeprefix("module."): v for k, v in checkpoint.items()}
