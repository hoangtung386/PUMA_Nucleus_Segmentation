"""Main WSI inference pipeline for PUMA Track 2 — v8 CellPath."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tifffile
import torch

from configs import INFERENCE_DEFAULT_CONFIG
from symbiopan.common.logging import get_logger
from symbiopan.inference.model_loader import load_stage1
from symbiopan.inference.postprocessing import (
    classify_instances,
    hv_instance_segmentation,
    instances_to_polygons,
)
from symbiopan.inference.site_classifier import resolve_site_type
from symbiopan.inference.tiling import (
    find_single_tif,
    make_tile_starts,
    normalize_tile,
    pad_reflect,
    read_rgb_uint8,
)
from symbiopan.inference.tta import apply_tta

logger = get_logger(__name__)

__all__ = ["main"]


def parse_args():
    parser = argparse.ArgumentParser(description="PUMA Track 2 inference — v8 CellPath")
    parser.add_argument("--input", default=os.environ.get("SYMBIOPAN_INPUT", "/input/images/melanoma-whole-slide-image"))
    parser.add_argument("--output", default=os.environ.get("SYMBIOPAN_OUTPUT", "/output"))
    parser.add_argument("--cp", default=os.environ.get("SYMBIOPAN_CKPT", "checkpoints/best_model.pth"), help="Stage 1 panoptic checkpoint")
    parser.add_argument("--tile-size", type=int, default=INFERENCE_DEFAULT_CONFIG.tile_size)
    parser.add_argument("--overlap", type=int, default=INFERENCE_DEFAULT_CONFIG.overlap)
    parser.add_argument(
        "--site-type", choices=["primary", "metastatic"], default=None, help="Manual site-type override"
    )
    parser.add_argument("--site-classifier-cp", default=INFERENCE_DEFAULT_CONFIG.site_classifier_cp)
    parser.add_argument("--site-classifier-arch", default=INFERENCE_DEFAULT_CONFIG.site_classifier_arch)
    parser.add_argument("--site-classifier-size", type=int, default=INFERENCE_DEFAULT_CONFIG.site_classifier_size)
    parser.add_argument("--np-threshold", type=float, default=INFERENCE_DEFAULT_CONFIG.np_threshold)
    parser.add_argument("--min-nucleus-area", type=int, default=INFERENCE_DEFAULT_CONFIG.min_nucleus_area)
    parser.add_argument("--tta", action="store_true", default=False, help="Enable test-time augmentation (8 augs)")
    return parser.parse_args()


@torch.no_grad()
def run_tile(model_s1, tile_uint8, tensor, device, site_id, use_tta):
    site_ids = torch.tensor([site_id], device=device) if site_id is not None else None
    preds = apply_tta(model_s1, tensor, site_ids, use_tta=use_tta)

    tissue_logits = preds["tissue"][0].cpu().numpy()
    np_logits = preds["np"][0, 0].cpu().numpy()
    nc_logits = preds["nc"][0].cpu().numpy()
    hv = preds["hv"][0].cpu().numpy()
    return tissue_logits, np_logits, nc_logits, hv


def process_image(
    input_path, output_dir, model_s1, device, tile_size, overlap, site_id, np_threshold, min_nucleus_area, use_tta
):
    img = read_rgb_uint8(input_path)
    h, w = img.shape[:2]
    logger.info("Processing %s: %dx%d | site_id=%s", input_path.name, h, w, site_id)

    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError(f"Invalid tiling: tile_size={tile_size}, overlap={overlap}")

    rows = make_tile_starts(h, tile_size, stride)
    cols = make_tile_starts(w, tile_size, stride)
    tissue_acc = np.zeros((6, h, w), dtype=np.float32)
    tissue_count = np.zeros((h, w), dtype=np.float32)
    polygons = []
    half = overlap // 2

    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            logger.info("Tile row=%d/%d col=%d/%d", ri + 1, len(rows), ci + 1, len(cols))
            tile = img[r : min(r + tile_size, h), c : min(c + tile_size, w)]
            tile, real_h, real_w = pad_reflect(tile, tile_size)
            tensor = normalize_tile(tile, device)
            t_logits, np_logit, nc_logits, hv = run_tile(
                model_s1,
                tile,
                tensor,
                device,
                site_id,
                use_tta,
            )

            t_logits = t_logits[:, :real_h, :real_w]
            np_logit = np_logit[:real_h, :real_w]
            nc_logits = nc_logits[:, :real_h, :real_w]
            hv = hv[:, :real_h, :real_w]

            tissue_acc[:, r : r + real_h, c : c + real_w] += t_logits
            tissue_count[r : r + real_h, c : c + real_w] += 1.0

            inst = hv_instance_segmentation(np_logit, hv, threshold=np_threshold, min_size=min_nucleus_area)
            ids = classify_instances(inst, nc_logits)
            valid_r = (0 if ri == 0 else half, real_h if ri == len(rows) - 1 else max(real_h - half, 0))
            valid_c = (0 if ci == 0 else half, real_w if ci == len(cols) - 1 else max(real_w - half, 0))
            polygons.extend(instances_to_polygons(inst, ids, tile_offset=(r, c), valid_r=valid_r, valid_c=valid_c))

    tissue_avg = tissue_acc / np.maximum(tissue_count[None, ...], 1e-6)
    tissue_puma = tissue_avg.argmax(axis=0).astype(np.uint8)

    output_dir = Path(output_dir)
    tissue_dir = output_dir / "images" / "melanoma-tissue-mask-segmentation"
    tissue_dir.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(str(tissue_dir / input_path.name), tissue_puma, photometric="minisblack")

    json_path = output_dir / "melanoma-10-class-nuclei-segmentation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"type": "Multiple polygons", "polygons": polygons, "version": {"major": 1, "minor": 0}}, f)

    logger.info("Saved tissue mask: %s", tissue_dir / input_path.name)
    logger.info("Saved nuclei JSON: %s (%d polygons)", json_path, len(polygons))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    input_path = find_single_tif(args.input)
    image_rgb = read_rgb_uint8(input_path)

    model_s1 = load_stage1(args.cp, device)
    site_id = resolve_site_type(args, {}, image_rgb, device)

    tile_size = int(getattr(args, "tile_size", 1024))
    overlap = int(getattr(args, "overlap", 256))

    process_image(
        input_path=input_path,
        output_dir=args.output,
        model_s1=model_s1,
        device=device,
        tile_size=tile_size,
        overlap=overlap,
        site_id=site_id,
        np_threshold=args.np_threshold,
        min_nucleus_area=args.min_nucleus_area,
        use_tta=args.tta,
    )
