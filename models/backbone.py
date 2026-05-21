from typing import Any

import timm


def build_cnn_backbone(pretrained: bool = True) -> Any:
    """Build a ConvNeXt Atto CNN backbone for multi-scale feature extraction.

    Args:
        pretrained: Whether to load pretrained weights.

    Returns:
        timm.models.FeatureListNet: A ConvNeXt Atto model with
            features_only=True and out_indices=(0, 1, 2, 3).
    """
    model = timm.create_model(
        "convnext_atto",
        pretrained=pretrained,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )
    return model


# Backward compatibility alias
get_cnn_spatial_prior = build_cnn_backbone
