"""Site-type classifier (9-class: primary + 8 metastatic sites) for inference."""

from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F

from inference.tiling import autocast_enabled, normalize_tile
from training.checkpoint import extract_state_dict
from training.logging_utils import logger

SITE_NAMES = [
    "primary",
    "lymph_node",
    "brain",
    "bone",
    "soft_tissue",
    "liver",
    "lung",
    "gastrointestinal",
    "skin",
]


def load_site_classifier(checkpoint_path: Optional[str], device: torch.device, arch: str = "convnext_tiny"):
    if checkpoint_path is None:
        return None
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.info("Site classifier not found: %s", checkpoint_path)
        return None

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = extract_state_dict(ckpt)

    candidate_arches = []
    for a in [arch, "convnext_tiny", "convnext_atto", "convnextv2_tiny"]:
        if a not in candidate_arches:
            candidate_arches.append(a)

    last_error = None
    for candidate in candidate_arches:
        try:
            num_classes = state.get("head.fc.weight", state.get("fc.weight", torch.zeros(9, 64))).shape[0]
            is_binary = num_classes == 2
            n_classes = 2 if is_binary else 9
            model = timm.create_model(candidate, pretrained=False, num_classes=n_classes)
            model.load_state_dict(state, strict=True)
            model.to(device).eval()
            logger.info("Loaded site classifier: %s | arch=%s | classes=%d", checkpoint_path, candidate, n_classes)
            return model, is_binary
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not load site classifier {checkpoint_path}. Last error: {last_error}")


@torch.no_grad()
def predict_site_type(
    site_model: Union[torch.nn.Module, tuple], image_rgb: np.ndarray, device: torch.device, image_size: int = 256
) -> int:
    if isinstance(site_model, tuple):
        site_model, is_binary = site_model
    else:
        is_binary = False
    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = normalize_tile(resized, device)
    with autocast_enabled(device):
        logits = site_model(x)
        prob = F.softmax(logits.float(), dim=1)[0]
    pred = int(prob.argmax().item())
    if is_binary:
        logger.info(
            "Binary site classifier: %s (primary=%.4f, metastatic=%.4f)",
            "primary" if pred == 0 else "metastatic",
            prob[0].item(),
            prob[1].item(),
        )
        return 0 if pred == 0 else 1
    logger.info("9-class site classifier: %s (id=%d, conf=%.4f)", SITE_NAMES[pred], pred, prob[pred].item())
    return pred


def resolve_site_type(args, cfg: Dict, image_rgb: np.ndarray, device: torch.device) -> int:
    if hasattr(args, "site_type") and args.site_type is not None:
        logger.info("Using manual site type: %s", args.site_type)
        id_map = {"primary": 0, "metastatic": 1}
        return id_map.get(args.site_type, 1)

    result = load_site_classifier(
        getattr(args, "site_classifier_cp", None),
        device,
        arch=getattr(args, "site_classifier_arch", "convnext_tiny"),
    )
    if result is not None:
        model, is_binary = result if isinstance(result, tuple) else (result, False)
        return predict_site_type(
            (model, is_binary), image_rgb, device, image_size=getattr(args, "site_classifier_size", 256)
        )

    logger.warning("No site classifier found. Defaulting to primary (id=0).")
    return 0
