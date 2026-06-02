"""Checkpoint save/load utilities shared across training stages."""

from pathlib import Path

import torch

from symbiopan.common.logging import get_logger

logger = get_logger(__name__)


def safe_torch_save(obj: dict, path: str | Path) -> None:
    """Save checkpoint atomically to avoid partial-write corruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, str(tmp))
    _ = torch.load(str(tmp), map_location="cpu", weights_only=False)
    tmp.replace(path)
    logger.info("Checkpoint saved and verified: %s", path)


def safe_torch_save_entity(model: torch.nn.Module, path: str | Path) -> None:
    """Save full model (architecture + weights) atomically.

    The saved file contains the complete model object, so loading only requires
    ``torch.load(path, map_location='cpu', weights_only=False)`` — no need to
    manually reconstruct the architecture.

    IMPORTANT: Saves a CPU copy without moving the original model from its device.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    orig_device = next(model.parameters()).device
    model.to("cpu")
    torch.save(model, str(tmp))
    model.to(orig_device)
    _ = torch.load(str(tmp), map_location="cpu", weights_only=False)
    tmp.replace(path)
    logger.info("Entity model saved and verified: %s", path)


def load_large_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict:
    """Load checkpoint with DataParallel key stripping."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(str(path), map_location=device, weights_only=False)


def extract_state_dict(checkpoint: dict | torch.nn.Module) -> dict[str, torch.Tensor]:
    """Extract model state dict from various checkpoint formats.

    Supports:
    - ``dict`` with keys ``model_state``, ``model_state_dict``, ``state_dict``, or ``model``
    - ``nn.Module`` (entity model) — returns ``model.state_dict()``
    """
    if isinstance(checkpoint, torch.nn.Module):
        return {k.removeprefix("module."): v for k, v in checkpoint.state_dict().items()}
    if isinstance(checkpoint, dict):
        for key in ["model_state", "model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format")
    return {k.removeprefix("module."): v for k, v in checkpoint.items()}
