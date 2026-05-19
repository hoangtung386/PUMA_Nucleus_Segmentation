import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from scipy.ndimage import find_objects

from dataloaders.puma_dataset import PUMA_NUCLEI_ID_TO_NAME, internal_tissue_to_puma
from models import ResidualNucleiRefinerUNet, UnifiedPanopticNet, build_stage2_input, get_cnn_spatial_prior


DEFAULT_INPUT_DIR = "/input/images/melanoma-whole-slide-image"
DEFAULT_OUTPUT_DIR = "/output"
DEFAULT_STAGE1_CP = "/opt/app/checkpoint/best_model.pth"
DEFAULT_STAGE2_CP = "/opt/app/checkpoint/nuclei_refiner_residual_best.pth"
DEFAULT_SITE_CLASSIFIER_CP = "/opt/app/checkpoint/site_classifier_atto.pth"
DEFAULT_TILE_SIZE = 1024
DEFAULT_OVERLAP = 256


# -----------------------------------------------------------------------------
# CLI: defaults are Docker-ready, so the file can run without manual arguments.
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="PUMA Track 2 inference")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cp", default=DEFAULT_STAGE1_CP, help="Stage 1 panoptic checkpoint")
    parser.add_argument("--stage2-cp", default=None, help="Optional Stage 2 residual refiner checkpoint")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)

    parser.add_argument(
        "--site-type",
        choices=["primary", "metastatic"],
        default=None,
        help="Manual site-type override. If omitted, uses site_classifier_atto.pth if available.",
    )
    parser.add_argument(
        "--site-classifier-cp",
        default=DEFAULT_SITE_CLASSIFIER_CP,
        help="Optional site classifier checkpoint. Expected mapping: 0=primary, 1=metastatic.",
    )
    parser.add_argument(
        "--site-classifier-arch",
        default="convnext_atto",
        help="timm architecture for the site classifier. Auto-fallbacks are tried if strict load fails.",
    )
    parser.add_argument("--site-classifier-size", type=int, default=256)

    parser.add_argument("--cellpose-mode", choices=["auto", "generate", "zero"], default="auto")
    parser.add_argument("--cellpose-model-type", default="nuclei")
    parser.add_argument("--np-threshold", type=float, default=0.50)
    parser.add_argument("--min-nucleus-area", type=int, default=20)
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------
def find_single_tif(input_dir: str) -> Path:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff"}])
    if not files:
        files = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in {".tif", ".tiff"}])
    if not files:
        raise FileNotFoundError(f"No .tif/.tiff file found in {input_dir}")
    if len(files) > 1:
        print(f"[WARN] Found {len(files)} TIFF files. Using first: {files[0]}")
    return files[0]


def read_rgb_uint8(path: Path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[0] in {3, 4} and img.shape[-1] not in {3, 4}:
        img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        x = img.astype(np.float32)
        x = x - x.min()
        denom = x.max() + 1e-8
        img = np.clip((x / denom) * 255.0, 0, 255).astype(np.uint8)
    return img


def normalize_tile(tile_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    x = tile_uint8.astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return x


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model_state", "model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format; expected state_dict or dict containing model_state/state_dict.")
    return {k.replace("module.", "", 1): v for k, v in checkpoint.items()}


def autocast_enabled(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda")


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_stage1(checkpoint_path: str, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("inference_config", {}) if isinstance(ckpt, dict) else {}

    model = UnifiedPanopticNet(
        vit_model=None,
        cnn_model=get_cnn_spatial_prior(pretrained=False),
        num_tissue=5,
        num_nuclei=10,
        load_uni_weights=False,
    )
    model.load_state_dict(extract_state_dict(ckpt), strict=True)
    model.enable_sc_dfa(bool(cfg.get("use_sc_dfa", False)))
    model.set_spatial_prior_lambda(float(cfg.get("lambda_prior", 0.0)))
    model.to(device).eval()

    print(f"[OK] Loaded Stage 1: {checkpoint_path}")
    print(f"[OK] Stage 1 settings: use_sc_dfa={model.use_sc_dfa}, lambda_prior={model.lambda_prior}")
    return model, cfg


def load_stage2(checkpoint_path: Optional[str], device: torch.device):
    if checkpoint_path is None:
        return None, 0.0
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"[INFO] Stage 2 checkpoint not found: {checkpoint_path}. Running Stage 1 only.")
        return None, 0.0

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    in_channels = int(cfg.get("in_channels", 21))
    out_classes = int(cfg.get("out_classes", 10))
    if in_channels != 21:
        raise RuntimeError(f"Stage 2 checkpoint expects {in_channels} input channels, but merged model requires 21.")

    model = ResidualNucleiRefinerUNet(in_channels=in_channels, out_classes=out_classes)
    model.load_state_dict(extract_state_dict(ckpt), strict=True)
    alpha = float(ckpt.get("alpha", cfg.get("alpha_end", 0.35))) if isinstance(ckpt, dict) else 0.35
    model.to(device).eval()
    print(f"[OK] Loaded Stage 2: {checkpoint_path} | alpha={alpha:.3f}")
    return model, alpha


# -----------------------------------------------------------------------------
# Site classifier: inference-only primary/metastatic prediction.
# -----------------------------------------------------------------------------
def load_site_classifier(checkpoint_path: Optional[str], device: torch.device, arch: str = "convnext_atto"):
    if checkpoint_path is None:
        return None
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"[INFO] Site classifier not found: {checkpoint_path}")
        return None

    import timm

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = extract_state_dict(ckpt)

    candidate_arches = []
    for a in [arch, "convnext_atto", "convnextv2_atto", "convnextv2_femto", "convnext_femto"]:
        if a not in candidate_arches:
            candidate_arches.append(a)

    last_error = None
    for candidate in candidate_arches:
        try:
            model = timm.create_model(candidate, pretrained=False, num_classes=2)
            model.load_state_dict(state, strict=True)
            model.to(device).eval()
            print(f"[OK] Loaded site classifier: {checkpoint_path} | arch={candidate}")
            return model
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not load site classifier {checkpoint_path}. Tried {candidate_arches}. "
        f"Last error: {last_error}"
    )


@torch.no_grad()
def predict_site_type(site_model, image_rgb: np.ndarray, device: torch.device, image_size: int = 256) -> str:
    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = normalize_tile(resized, device)
    with autocast_enabled(device):
        logits = site_model(x)
        prob = F.softmax(logits.float(), dim=1)[0]
    pred = int(prob.argmax().item())
    site = "primary" if pred == 0 else "metastatic"
    print(f"[OK] Site classifier predicted: {site} | primary={prob[0].item():.4f}, metastatic={prob[1].item():.4f}")
    return site


def resolve_site_type(args, cfg: Dict, image_rgb: np.ndarray, device: torch.device) -> str:
    if args.site_type is not None:
        print(f"[OK] Using manual site type: {args.site_type}")
        return args.site_type

    site_model = load_site_classifier(args.site_classifier_cp, device, arch=args.site_classifier_arch)
    if site_model is not None:
        return predict_site_type(site_model, image_rgb, device, image_size=args.site_classifier_size)

    default_site = cfg.get("default_site_type", "metastatic")
    if isinstance(default_site, int):
        default_site = "primary" if default_site == 0 else "metastatic"
    if default_site not in {"primary", "metastatic"}:
        default_site = "metastatic"
    print(f"[WARN] No site classifier found. Falling back to default_site_type={default_site}")
    return default_site


# -----------------------------------------------------------------------------
# Cellpose flow generation.
# -----------------------------------------------------------------------------
class CellposeFlowGenerator:
    def __init__(self, mode: str = "auto", model_type: str = "nuclei", device: Optional[torch.device] = None):
        self.mode = mode
        self.model_type = model_type
        self.device = device or torch.device("cpu")
        self.model = None

        if mode == "zero":
            print("[Cellpose] Using zero flow.")
            return

        try:
            from cellpose import models as cellpose_models
            self.model = cellpose_models.CellposeModel(gpu=(self.device.type == "cuda"), model_type=model_type)
            print(f"[Cellpose] Loaded Cellpose model_type={model_type}")
        except Exception as exc:
            if mode == "generate":
                raise RuntimeError("Cellpose was required but could not be loaded.") from exc
            print(f"[WARN] Cellpose could not be loaded ({exc}). Falling back to zero flow.")
            self.mode = "zero"

    def make_flow(self, tile_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
        h, w = tile_uint8.shape[:2]
        if self.mode == "zero" or self.model is None:
            return torch.zeros((1, 2, h, w), dtype=torch.float32, device=device)

        try:
            result = self.model.eval(
                tile_uint8,
                diameter=None,
                channels=[0, 0],
                flow_threshold=None,
                cellprob_threshold=0.0,
            )
            if len(result) == 4:
                _masks, flows, _styles, _diams = result
            else:
                _masks, flows, _styles = result

            flow = flows[1] if isinstance(flows, list) and len(flows) > 1 else flows
            flow = np.asarray(flow)
            if flow.ndim == 3 and flow.shape[0] >= 2:
                flow = flow[:2]
            elif flow.ndim == 3 and flow.shape[-1] >= 2:
                flow = flow[..., :2].transpose(2, 0, 1)
            else:
                raise RuntimeError(f"Unexpected Cellpose flow shape: {flow.shape}")

            flow = flow.astype(np.float32)
            if flow.shape[1:] != (h, w):
                flow = np.stack([
                    cv2.resize(flow[0], (w, h), interpolation=cv2.INTER_LINEAR),
                    cv2.resize(flow[1], (w, h), interpolation=cv2.INTER_LINEAR),
                ], axis=0)
            return torch.from_numpy(flow).unsqueeze(0).float().to(device)
        except Exception as exc:
            if self.mode == "generate":
                raise RuntimeError("Cellpose flow generation failed.") from exc
            print(f"[WARN] Cellpose flow failed ({exc}). Using zero flow for this tile.")
            return torch.zeros((1, 2, h, w), dtype=torch.float32, device=device)


# -----------------------------------------------------------------------------
# Post-processing.
# -----------------------------------------------------------------------------
def make_tile_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] + tile_size < length:
        starts.append(length - tile_size)
    return starts


def pad_reflect(tile: np.ndarray, tile_size: int) -> Tuple[np.ndarray, int, int]:
    real_h, real_w = tile.shape[:2]
    pad_h = tile_size - real_h
    pad_w = tile_size - real_w
    if pad_h <= 0 and pad_w <= 0:
        return tile, real_h, real_w
    # Reflect padding fails if a dimension is 1. Constant fallback is safer.
    mode = "reflect" if real_h > 1 and real_w > 1 else "constant"
    return np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode), real_h, real_w


def hv_instance_segmentation(np_logits: np.ndarray, hv_map: np.ndarray, threshold: float, min_size: int) -> np.ndarray:
    # np_logits shape [H,W], raw foreground logit. hv_map shape [2,H,W].
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

    seed = ((grad < 0.35) & (fg > 0)).astype(np.uint8)
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


def classify_instances(inst_map: np.ndarray, nc_logits: np.ndarray) -> Dict[int, Tuple[int, float]]:
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


def instances_to_polygons(inst_map: np.ndarray, id_to_class_conf: Dict[int, Tuple[int, float]], tile_offset, valid_r, valid_c) -> List[dict]:
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

        polygons.append({
            "name": PUMA_NUCLEI_ID_TO_NAME[int(class_idx)],
            "seed_point": points[0],
            "path_points": points,
            "sub_type": "",
            "groups": [],
            "probability": float(max(0.0, min(1.0, conf))),
        })
    return polygons


@torch.no_grad()
def run_tile(model_s1, model_s2, stage2_alpha: float, tile_uint8: np.ndarray, tensor: torch.Tensor, flow_generator, device: torch.device, site_type: str):
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


def process_image(input_path: Path, output_dir: str, model_s1, model_s2, stage2_alpha: float, flow_generator, device: torch.device, tile_size: int, overlap: int, site_type: str, np_threshold: float, min_nucleus_area: int):
    img = read_rgb_uint8(input_path)
    h, w = img.shape[:2]
    print(f"[INFO] Processing {input_path.name}: {h}x{w} | site_type={site_type}")

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
            print(f"[Tile] row={ri + 1}/{len(rows)} col={ci + 1}/{len(cols)}")
            tile = img[r:min(r + tile_size, h), c:min(c + tile_size, w)]
            tile, real_h, real_w = pad_reflect(tile, tile_size)
            tensor = normalize_tile(tile, device)
            t_logits, np_logit, nc_logits, hv = run_tile(model_s1, model_s2, stage2_alpha, tile, tensor, flow_generator, device, site_type)

            t_logits = t_logits[:, :real_h, :real_w]
            np_logit = np_logit[:real_h, :real_w]
            nc_logits = nc_logits[:, :real_h, :real_w]
            hv = hv[:, :real_h, :real_w]

            tissue_acc[:, r:r + real_h, c:c + real_w] += t_logits
            tissue_count[r:r + real_h, c:c + real_w] += 1.0

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
    tifffile.imwrite(str(tissue_dir / input_path.name), tissue_puma, photometric="minisblack")

    json_path = output_dir / "melanoma-10-class-nuclei-segmentation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"type": "Multiple polygons", "polygons": polygons, "version": {"major": 1, "minor": 0}}, f)

    print(f"[OK] Saved tissue mask: {tissue_dir / input_path.name}")
    print(f"[OK] Saved nuclei JSON: {json_path} ({len(polygons)} polygons)")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Device: {device}")

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


if __name__ == "__main__":
    main()
