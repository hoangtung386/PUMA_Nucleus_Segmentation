import timm

def get_cnn_spatial_prior(pretrained=True):
    model = timm.create_model(
        #'convnext_tiny',
        'convnext_atto',
        pretrained=pretrained,
        features_only=True,
        out_indices=(0, 1, 2, 3)  # S1, S2, S3, S4
    )
    return model
