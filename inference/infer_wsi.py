"""Main WSI inference pipeline for PUMA Track 2."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from configs import INFERENCE_DEFAULT_CONFIG
from data.dataset.puma_dataset import internal_tissue_to_puma
from inference.cellpose_flow import CellposeFlowGenerator
from inference.model_loader import load_stage1, load_stage2
from inference.postprocessing import (
    classify_instances,
    hv_instance_segmentation,
    instances_to_polygons,
)
from inference.site_classifier import resolve_site_type
from inference.tiling import (
    autocast_enabled,
    find_single_tif,
    make_tile_starts,
    normalize_tile,
    pad_reflect,
    read_rgb_uint8,
)
from models import build_stage2_input
from training.logging_utils import logger

DEFAULT_STAGE2_CP = "/opt/app/checkpoint/nuclei_refiner_residual_best.pth"
DEFAULT_INPUT_DIR = "/input/images/melanoma-whole-slide-image"
DEFAULT_OUTPUT_DIR = "/output"
DEFAULT_STAGE1_CP = "/opt/app/checkpoint/best_model.pth"


def parse_args():
    """Parses command-line arguments for WSI inference.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PUMA Track 2 inference")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cp", default=DEFAULT_STAGE1_CP, help="Stage 1 panoptic checkpoint")
    parser.add_argument("--stage2-cp", default=None, help="Optional Stage 2 residual refiner checkpoint")
    parser.add_argument("--tile-size", type=int, default=INFERENCE_DEFAULT_CONFIG.tile_size)
    parser.add_argument("--overlap", type=int, default=INFERENCE_DEFAULT_CONFIG.overlap)

    parser.add_argument(
        "--site-type",
        choices=["primary", "metastatic"],
        default=None,
        help="Manual site-type override. If omitted, uses site_classifier_atto.pth if available.",
    )
    parser.add_argument(
        "--site-classifier-cp",
        default=INFERENCE_DEFAULT_CONFIG.site_classifier_cp,
        help="Optional site classifier checkpoint. Expected mapping: 0=primary, 1=metastatic.",
    )
    parser.add_argument(
        "--site-classifier-arch",
        default=INFERENCE_DEFAULT_CONFIG.site_classifier_arch,
        help="timm architecture for the site classifier.",
    )
    parser.add_argument("--site-classifier-size", type=int, default=INFERENCE_DEFAULT_CONFIG.site_classifier_size)

    parser.add_argument(
        "--cellpose-mode", choices=["auto", "generate", "zero"], default=INFERENCE_DEFAULT_CONFIG.cellpose_mode
    )
    parser.add_argument("--cellpose-model-type", default=INFERENCE_DEFAULT_CONFIG.cellpose_model_type)
    parser.add_argument("--np-threshold", type=float, default=INFERENCE_DEFAULT_CONFIG.np_threshold)
    parser.add_argument("--min-nucleus-area", type=int, default=INFERENCE_DEFAULT_CONFIG.min_nucleus_area)
    return parser.parse_args()


@torch.no_grad()
def run_tile(model_s1, model_s2, stage2_alpha, tile_uint8, tensor, flow_generator, device, site_type):
    """Runs inference on a single tile.

    Args:
        model_s1: Stage 1 panoptic model.
        model_s2: Optional Stage 2 residual refiner.
        stage2_alpha: Blending weight for Stage 2 residuals.
        tile_uint8: Raw uint8 tile array.
        tensor: Normalized tile tensor.
        flow_generator: Cellpose flow generator.
        device: Torch device.
        site_type: 'primary' or 'metastatic'.

    Returns:
        Tuple of (tissue_logits, np_logits, nc_logits, hv) as float numpy arrays.
    """
    flow = flow_generator.make_flow(tile_uint8, device)
    if flow.shape[-2:] != tensor.shape[-2:]:
        flow = F.interpolate(flow, size=tensor.shape[-2:], mode="bilinear", align_corners=False)

    site_types = [site_type] if getattr(model_s1, "lambda_prior", 0.0) > 0.0 else None
    with autocast_enabled(device):
        preds_s1 = model_s1(tensor, flow, site_types)
        if model_s2 is not None:
            s2_input = build_stage2_input(tensor, preds_s1)
            delta_nc = model_s2(s2_input)
            preds_s1["nc"] = preds_s1["nc"] + float(stage2_alpha) * delta_nc

    tissue_logits = preds_s1["tissue"][0].float().cpu().numpy()
    np_logits = preds_s1["np"][0, 0].float().cpu().numpy()
    nc_logits = preds_s1["nc"][0].float().cpu().numpy()
    hv = preds_s1["hv"][0].float().cpu().numpy()
    return tissue_logits, np_logits, nc_logits, hv


def process_image(
    input_path,
    output_dir,
    model_s1,
    model_s2,
    stage2_alpha,
    flow_generator,
    device,
    tile_size,
    overlap,
    site_type,
    np_threshold,
    min_nucleus_area,
):
    """Processes a single WSI image end-to-end.

    Tiles the image, runs inference on each tile with overlap handling,
    aggregates tissue predictions, performs instance segmentation, and
    saves outputs (tissue mask as TIFF, nuclei polygons as JSON).

    Args:
        input_path: Path to the input TIFF image.
        output_dir: Output directory path.
        model_s1: Stage 1 model.
        model_s2: Optional Stage 2 model.
        stage2_alpha: Stage 2 blending weight.
        flow_generator: Cellpose flow generator.
        device: Torch device.
        tile_size: Tile size in pixels.
        overlap: Tile overlap in pixels.
        site_type: 'primary' or 'metastatic'.
        np_threshold: Nucleus presence confidence threshold.
        min_nucleus_area: Minimum nucleus area in pixels.
    """
    img = read_rgb_uint8(input_path)
    h, w = img.shape[:2]
    logger.info("Processing %s: %dx%d | site_type=%s", input_path.name, h, w, site_type)

    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError(f"Invalid tiling: tile_size={tile_size}, overlap={overlap}")

    rows = make_tile_starts(h, tile_size, stride)
    cols = make_tile_starts(w, tile_size, stride)
    tissue_acc = np.zeros((5, h, w), dtype=np.float32)
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
                model_s1, model_s2, stage2_alpha, tile, tensor, flow_generator, device, site_type
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
    tissue_internal = tissue_avg.argmax(axis=0).astype(np.uint8)
    tissue_puma = internal_tissue_to_puma(tissue_internal).astype(np.uint8)
    if not set(np.unique(tissue_puma).tolist()).issubset({1, 2, 3, 4, 5}):
        raise RuntimeError(f"Unexpected tissue values: {np.unique(tissue_puma).tolist()}")

    output_dir = Path(output_dir)
    tissue_dir = output_dir / "images" / "melanoma-tissue-mask-segmentation"
    tissue_dir.mkdir(parents=True, exist_ok=True)

    import tifffile

    tifffile.imwrite(str(tissue_dir / input_path.name), tissue_puma, photometric="minisblack")

    json_path = output_dir / "melanoma-10-class-nuclei-segmentation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"type": "Multiple polygons", "polygons": polygons, "version": {"major": 1, "minor": 0}}, f)

    logger.info("Saved tissue mask: %s", tissue_dir / input_path.name)
    logger.info("Saved nuclei JSON: %s (%d polygons)", json_path, len(polygons))


def main():
    """Main entry point for WSI inference.

    Parses arguments, loads models, resolves site type, generates cellpose
    flow, and runs process_image on the input WSI.
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    input_path = find_single_tif(args.input)
    image_rgb = read_rgb_uint8(input_path)

    model_s1, cfg = load_stage1(args.cp, device)
    stage2_cp = args.stage2_cp
    if stage2_cp is None and Path(DEFAULT_STAGE2_CP).exists():
        stage2_cp = DEFAULT_STAGE2_CP
    model_s2, stage2_alpha = load_stage2(stage2_cp, device)

    site_type = resolve_site_type(args, cfg, image_rgb, device)
    flow_generator = CellposeFlowGenerator(mode=args.cellpose_mode, model_type=args.cellpose_model_type, device=device)

    tile_size = int(cfg.get("tile_size", args.tile_size))
    overlap = int(cfg.get("overlap", args.overlap))
    if "stride" in cfg and "overlap" not in cfg:
        overlap = max(tile_size - int(cfg["stride"]), 0)

    process_image(
        input_path=input_path,
        output_dir=args.output,
        model_s1=model_s1,
        model_s2=model_s2,
        stage2_alpha=stage2_alpha,
        flow_generator=flow_generator,
        device=device,
        tile_size=tile_size,
        overlap=overlap,
        site_type=site_type,
        np_threshold=args.np_threshold,
        min_nucleus_area=args.min_nucleus_area,
    )
