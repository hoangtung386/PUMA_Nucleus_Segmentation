from typing import Any

import timm


def build_cnn_backbone(pretrained: bool = True) -> Any:
    model = timm.create_model(
        "convnext_tiny",
        pretrained=pretrained,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )
    return model
