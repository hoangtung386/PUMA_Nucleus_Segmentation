import timm


def build_cnn_backbone(pretrained=True):
    model = timm.create_model(
        'convnext_atto',
        pretrained=pretrained,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )
    return model


# Backward compatibility alias
get_cnn_spatial_prior = build_cnn_backbone
