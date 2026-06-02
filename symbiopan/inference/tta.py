"""Test-time augmentation for WSI inference."""


import torch

TTA_TRANSFORMS = [
    ("identity", lambda x: x),
    ("hflip", lambda x: torch.flip(x, dims=[-1])),
    ("vflip", lambda x: torch.flip(x, dims=[-2])),
    ("h+v_flip", lambda x: torch.flip(x, dims=[-1, -2])),
    ("rot90", lambda x: torch.rot90(x, k=1, dims=[-2, -1])),
    ("rot180", lambda x: torch.rot90(x, k=2, dims=[-2, -1])),
    ("rot270", lambda x: torch.rot90(x, k=3, dims=[-2, -1])),
    ("rot90_hflip", lambda x: torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-1])),
]

TTA_INVERSE = {
    "identity": lambda x: x,
    "hflip": lambda x: torch.flip(x, dims=[-1]),
    "vflip": lambda x: torch.flip(x, dims=[-2]),
    "h+v_flip": lambda x: torch.flip(x, dims=[-1, -2]),
    "rot90": lambda x: torch.rot90(x, k=-1, dims=[-2, -1]),
    "rot180": lambda x: torch.rot90(x, k=-2, dims=[-2, -1]),
    "rot270": lambda x: torch.rot90(x, k=-3, dims=[-2, -1]),
    "rot90_hflip": lambda x: torch.rot90(torch.flip(x, dims=[-1]), k=-1, dims=[-2, -1]),
}


@torch.no_grad()
def apply_tta(model: torch.nn.Module, tensor: torch.Tensor, site_ids: torch.Tensor | None, use_tta: bool = True) -> dict[str, torch.Tensor]:
    from symbiopan.inference.tiling import autocast_enabled

    if not use_tta:
        with autocast_enabled(tensor.device):
            preds = model(tensor, site_ids)
        return {k: v.float() for k, v in preds.items()}

    accumulated: dict[str, torch.Tensor] = {}
    count = 0
    for name, aug_fn in TTA_TRANSFORMS:
        x_aug = aug_fn(tensor)
        with autocast_enabled(tensor.device):
            out = model(x_aug, site_ids)
        inv_fn = TTA_INVERSE[name]
        for key, val in out.items():
            val_inv = inv_fn(val.float())
            if key not in accumulated:
                accumulated[key] = val_inv
            else:
                accumulated[key] = accumulated[key] + val_inv
        count += 1
    return {k: v / count for k, v in accumulated.items()}
