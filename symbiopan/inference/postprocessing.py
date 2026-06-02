"""Post-processing: instance segmentation and polygon generation."""

import cv2
import numpy as np
import torch
from scipy.ndimage import find_objects

from symbiopan.data.constants import HV_GRAD_THRESHOLD, PUMA_NUCLEI_ID_TO_NAME


def hv_instance_segmentation(np_logits: np.ndarray, hv_map: np.ndarray, threshold: float, min_size: int) -> np.ndarray:
    prob = 1.0 / (1.0 + np.exp(-np_logits))
    fg = (prob >= threshold).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if int(fg.sum()) == 0:
        return np.zeros_like(fg, dtype=np.int32)

    hv = hv_map.astype(np.float32)
    gx = cv2.Sobel(hv[0], cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(hv[1], cv2.CV_32F, 0, 1, ksize=3)
    grad = np.maximum(np.abs(gx), np.abs(gy))
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)

    seed = ((grad < HV_GRAD_THRESHOLD) & (fg > 0)).astype(np.uint8)
    n_markers, markers = cv2.connectedComponents(seed, connectivity=8)
    if n_markers <= 1:
        _n, markers = cv2.connectedComponents(fg, connectivity=8)

    surface = (grad * 255).astype(np.uint8)
    inst = cv2.watershed(cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR), markers.astype(np.int32))
    inst = np.clip(inst, 0, None).astype(np.int32)
    inst[fg == 0] = 0

    cleaned = np.zeros_like(inst, dtype=np.int32)
    new_id = 1
    for i, sl in enumerate(find_objects(inst)):
        if sl is None:
            continue
        old_id = i + 1
        region = inst[sl] == old_id
        if int(region.sum()) < min_size:
            continue
        cleaned[sl][region] = new_id
        new_id += 1
    return cleaned


def classify_instances(inst_map: np.ndarray, nc_logits: np.ndarray) -> dict[int, tuple[int, float]]:
    probs = torch.softmax(torch.from_numpy(nc_logits), dim=0).numpy()
    cls_map = probs.argmax(axis=0).astype(np.uint8)
    conf_map = probs.max(axis=0)
    out = {}
    for i, sl in enumerate(find_objects(inst_map)):
        if sl is None:
            continue
        inst_id = i + 1
        mask = inst_map[sl] == inst_id
        if not np.any(mask):
            continue
        cls_vals = cls_map[sl][mask]
        counts = np.bincount(cls_vals, minlength=10)
        cls = int(counts.argmax())
        conf = float(conf_map[sl][mask].mean())
        out[inst_id] = (cls, conf)
    return out


def instances_to_polygons(
    inst_map: np.ndarray,
    id_to_class_conf: dict[int, tuple[int, float]],
    tile_offset,
    valid_r,
    valid_c,
) -> list[dict]:
    polygons = []
    h, w = inst_map.shape
    r0, r1 = valid_r[0], valid_r[1] if valid_r[1] is not None else h
    c0, c1 = valid_c[0], valid_c[1] if valid_c[1] is not None else w

    for inst_id, (class_idx, conf) in id_to_class_conf.items():
        ys, xs = np.where(inst_map == inst_id)
        if len(xs) == 0:
            continue
        cy = float(ys.mean())
        cx = float(xs.mean())
        if not (r0 <= cy < r1 and c0 <= cx < c1):
            continue

        binary = (inst_map == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            continue

        points = []
        for pt in contour:
            x = float(pt[0][0]) + tile_offset[1]
            y = float(pt[0][1]) + tile_offset[0]
            points.append([x, y, 0.5])
        if len(points) < 3:
            continue

        polygons.append(
            {
                "name": PUMA_NUCLEI_ID_TO_NAME[int(class_idx)],
                "seed_point": points[0],
                "path_points": points,
                "sub_type": "",
                "groups": [],
                "probability": float(max(0.0, min(1.0, conf))),
            }
        )
    return polygons
