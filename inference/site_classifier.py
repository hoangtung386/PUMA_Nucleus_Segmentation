"""Site-type classifier (primary vs. metastatic) for inference."""

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F

from inference.tiling import autocast_enabled, normalize_tile
from training.checkpoint import extract_state_dict
from training.logging_utils import logger


def load_site_classifier(checkpoint_path: Optional[str], device: torch.device, arch: str = "convnext_atto"):
    """Loads a site classifier (primary vs. metastatic) from a checkpoint.

    Tries multiple timm architectures if the initial one fails to load.

    Args:
        checkpoint_path: Optional path to the classifier checkpoint.
        device: Torch device.
        arch: Preferred timm architecture name.

    Returns:
        Loaded model in eval mode, or None if checkpoint_path is None/missing.

    Raises:
        RuntimeError: If the checkpoint cannot be loaded with any architecture.
    """
    if checkpoint_path is None:
        return None
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.info("Site classifier not found: %s", checkpoint_path)
        return None

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = extract_state_dict(ckpt)

    candidate_arches = []
    for a in [arch, "convnext_atto", "convnextv2_atto", "convnextv2_femto", "convnext_femto"]:
        if a not in candidate_arches:
            candidate_arches.append(a)

    last_error = None
    for candidate in candidate_arches:
        try:
            model = timm.create_model(candidate, pretrained=False, num_classes=2)
            model.load_state_dict(state, strict=True)
            model.to(device).eval()
            logger.info("Loaded site classifier: %s | arch=%s", checkpoint_path, candidate)
            return model
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not load site classifier {checkpoint_path}. Tried {candidate_arches}. Last error: {last_error}"
    )


@torch.no_grad()
def predict_site_type(site_model, image_rgb: np.ndarray, device: torch.device, image_size: int = 256) -> str:
    """Predicts whether a whole-slide image is primary or metastatic.

    Args:
        site_model: Loaded site classifier model.
        image_rgb: uint8 RGB image array.
        device: Torch device.
        image_size: Resize dimension for the classifier input.

    Returns:
        'primary' or 'metastatic'.
    """
    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = normalize_tile(resized, device)
    with autocast_enabled(device):
        logits = site_model(x)
        prob = F.softmax(logits.float(), dim=1)[0]
    pred = int(prob.argmax().item())
    site = "primary" if pred == 0 else "metastatic"
    logger.info("Site classifier predicted: %s | primary=%.4f, metastatic=%.4f", site, prob[0].item(), prob[1].item())
    return site


def resolve_site_type(args, cfg: Dict, image_rgb: np.ndarray, device: torch.device) -> str:
    """Resolves the site type (primary vs. metastatic) for inference.

    Priority: manual override -> site classifier -> config default.

    Args:
        args: Parsed command-line arguments.
        cfg: Stage 1 config dict.
        image_rgb: uint8 RGB image array.
        device: Torch device.

    Returns:
        'primary' or 'metastatic'.
    """
    if args.site_type is not None:
        logger.info("Using manual site type: %s", args.site_type)
        return args.site_type

    site_model = load_site_classifier(args.site_classifier_cp, device, arch=args.site_classifier_arch)
    if site_model is not None:
        return predict_site_type(site_model, image_rgb, device, image_size=args.site_classifier_size)

    default_site = cfg.get("default_site_type", "metastatic")
    if isinstance(default_site, int):
        default_site = "primary" if default_site == 0 else "metastatic"
    if default_site not in {"primary", "metastatic"}:
        default_site = "metastatic"
    logger.warning("No site classifier found. Falling back to default_site_type=%s", default_site)
    return default_site
