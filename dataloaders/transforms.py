import albumentations as A
import cv2


def _fix_vector_field(vec, replay):
    """Correct x/y vector directions after flips and 90-degree rotations."""
    if vec is None:
        return vec
    for tr in replay.get("transforms", []):
        if not tr.get("applied", False):
            continue
        name = tr.get("__class_fullname__", "")
        if name.endswith("HorizontalFlip"):
            vec[..., 0] *= -1
        elif name.endswith("VerticalFlip"):
            vec[..., 1] *= -1
        elif name.endswith("RandomRotate90"):
            k = int(tr.get("params", {}).get("factor", 0)) % 4
            x = vec[..., 0].copy()
            y = vec[..., 1].copy()
            if k == 1:
                vec[..., 0] = -y
                vec[..., 1] = x
            elif k == 2:
                vec[..., 0] = -x
                vec[..., 1] = -y
            elif k == 3:
                vec[..., 0] = y
                vec[..., 1] = -x
    return vec


class VectorSafeCompose:
    def __init__(self, transforms, additional_targets):
        self.aug = A.ReplayCompose(transforms, additional_targets=additional_targets)

    def __call__(self, **kwargs):
        out = self.aug(**kwargs)
        replay = out.get("replay", {})
        out["cp_flow"] = _fix_vector_field(out.get("cp_flow"), replay)
        out["hv_map"] = _fix_vector_field(out.get("hv_map"), replay)
        out.pop("replay", None)
        return out


def get_train_transforms(image_size=1024):
    return VectorSafeCompose([
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ], additional_targets={
        "tissue_mask": "mask",
        "nuclei_mask": "mask",
        "cp_flow": "image",
        "hv_map": "image",
    })


def get_val_transforms(image_size=1024):
    return A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
    ], additional_targets={
        "tissue_mask": "mask",
        "nuclei_mask": "mask",
        "cp_flow": "image",
        "hv_map": "image",
    })
