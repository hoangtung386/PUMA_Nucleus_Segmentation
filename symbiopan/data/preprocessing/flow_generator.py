"""HV map computation for preprocessing and inference (Cellpose removed)."""

import numpy as np


def compute_hv_map(inst_mask: np.ndarray) -> np.ndarray:
    """HoVer-Net style horizontal/vertical maps, shape [2, H, W]."""
    h_map = np.zeros_like(inst_mask, dtype=np.float32)
    v_map = np.zeros_like(inst_mask, dtype=np.float32)

    for inst_id in np.unique(inst_mask):
        if inst_id == 0:
            continue
        ys, xs = np.where(inst_mask == inst_id)
        if len(xs) == 0:
            continue
        x_center = float(xs.mean())
        y_center = float(ys.mean())
        x_radius = max((float(xs.max()) - float(xs.min())) / 2.0, 1.0)
        y_radius = max((float(ys.max()) - float(ys.min())) / 2.0, 1.0)
        h_map[ys, xs] = np.clip((xs - x_center) / (x_radius + 1e-8), -1.0, 1.0)
        v_map[ys, xs] = np.clip((ys - y_center) / (y_radius + 1e-8), -1.0, 1.0)

    return np.stack([h_map, v_map], axis=0).astype(np.float16)
