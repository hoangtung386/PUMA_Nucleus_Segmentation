from __future__ import annotations

"""V13.2 local and PUMA Grand-Challenge inference.

The nuclei submission contract follows the official Track-2 evaluator: one
``melanoma-10-class-nuclei-segmentation.json`` with ``type=Multiple polygons``.
Each centroid prediction is encoded as a small symmetric polygon whose arithmetic
vertex mean is exactly the model centroid. A structurally valid tissue-mask output
is also supported because the official Track-2 algorithm interface expects it.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from puma.config import (
    PUMA_CLASS_NAMES,
    REJECT_CLASS_ID,
    RuntimeConfig,
    Stage2ModelConfig,
    stage1_experiment_registry,
    stage1_model_config_from_dict,
)
from puma.data.datasets import (
    build_stage2_geometry,
    extract_crop,
    image_to_uint8_tensor,
    pack_stage2_view_tensors,
    prepare_stage2_view_batch,
)
from puma.data.preprocess import read_tiff_rgb
from puma.data.targets import DecodedCandidates, adaptive_suppress
from puma.models.stage1 import build_stage1_model
from puma.models.stage2 import (
    build_stage2_model,
    decode_stage2_probabilities,
    hierarchical_probabilities,
    uni2_checkpoint_path,
)
from puma.training.stage1 import predict_roi
from puma.utils import (
    atomic_save_numpy,
    config_hash,
    resolve_amp_dtype,
    resolve_artifact_reference,
    resolve_device,
    restore_checkpoint_payload,
)

PUMA_GC_INPUT_DIR = Path("/input/images/melanoma-wsi")
PUMA_GC_INPUT_DIR_ALIASES: tuple[Path, ...] = (
    PUMA_GC_INPUT_DIR,
    # Compatibility alias used by some Grand-Challenge interfaces/documentation.
    Path("/input/images/melanoma-whole-slide-image"),
)
PUMA_GC_NUCLEI_JSON = Path("/output/melanoma-10-class-nuclei-segmentation.json")
PUMA_GC_TISSUE_DIR = Path("/output/images/melanoma-tissue-mask-segmentation")
PUMA_TISSUE_LABELS = frozenset(range(6))

FINAL_DTYPE = np.dtype([
    ("image_id", "U128"),
    ("x", "f4"),
    ("y", "f4"),
    ("class_id", "i2"),
    ("class_name", "U32"),
    ("confidence", "f4"),
    ("detector_confidence", "f4"),
    ("classifier_confidence", "f4"),
])


def read_rgb(path: Path) -> np.ndarray:
    return read_tiff_rgb(Path(path))


def _empty_candidates() -> DecodedCandidates:
    return DecodedCandidates(
        coordinates=np.empty((0, 2), np.float32),
        scores=np.empty(0, np.float32),
    )


def ensemble_stage1(
    models,
    image: np.ndarray,
    runtime: RuntimeConfig,
    postprocess: list[tuple[float, int, float]],
) -> DecodedCandidates:
    device = resolve_device()
    amp_dtype = resolve_amp_dtype(runtime.training.prefer_bfloat16, device)
    predictions: list[DecodedCandidates] = []
    for model, (threshold, radius, suppression_radius) in zip(models, postprocess, strict=True):
        prediction = predict_roi(
            model,
            image,
            runtime.data.tile_size,
            runtime.data.tile_overlap,
            device,
            threshold,
            radius,
            runtime.training.amp,
            amp_dtype,
            tile_batch_size=runtime.data.validation_tile_batch_size,
            suppression_radius=suppression_radius,
        )
        if len(prediction.scores):
            predictions.append(prediction)
    if not predictions:
        return _empty_candidates()
    merged = DecodedCandidates(
        coordinates=np.concatenate([item.coordinates for item in predictions]),
        scores=np.concatenate([item.scores for item in predictions]),
    )
    ensemble_radius = float(np.median([item[2] for item in postprocess]))
    return adaptive_suppress(merged, min_radius=ensemble_radius, max_radius=ensemble_radius)


def _nearest_distances(coordinates: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    if len(coordinates) <= 1:
        return np.full(len(coordinates), float(max(image_shape[:2])), np.float32)
    distances, _ = cKDTree(np.asarray(coordinates, np.float32)).query(coordinates, k=2)
    return distances[:, 1].astype(np.float32)


def predict_candidate_probabilities(
    model,
    image: np.ndarray,
    prediction: DecodedCandidates,
    cfg: Stage2ModelConfig,
    batch_size: int,
    *,
    amp: bool,
    prefer_bfloat16: bool,
    nearest_distances: np.ndarray | None = None,
) -> np.ndarray:
    if len(prediction.scores) == 0:
        return np.empty((0, REJECT_CLASS_ID + 1), dtype=np.float32)
    if cfg.interface_key != "Fixed-MV" or any(view not in {"V2", "V3", "V4"} for view in cfg.views):
        raise ValueError("V13.2 inference supports only fixed V2/V3/V4 views.")
    device = resolve_device()
    amp_dtype = resolve_amp_dtype(prefer_bfloat16, device)
    fixed_sizes = {"V2": 64, "V3": 128, "V4": 256}
    if nearest_distances is None:
        nearest = _nearest_distances(prediction.coordinates, image.shape)
    else:
        nearest = np.asarray(nearest_distances, dtype=np.float32)
        if nearest.shape != (len(prediction.scores),):
            raise ValueError(
                f"nearest_distances shape {nearest.shape} does not match candidates {len(prediction.scores)}."
            )
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(prediction.scores), int(batch_size)):
            stop = min(start + int(batch_size), len(prediction.scores))
            packed_views: dict[str, list[torch.Tensor]] = {view: [] for view in cfg.views}
            geometry_rows: list[np.ndarray] = []
            largest = max(fixed_sizes[view] for view in cfg.views)
            for index in range(start, stop):
                x, y = map(float, prediction.coordinates[index])
                base = image_to_uint8_tensor(extract_crop(image, x, y, largest))
                for view in cfg.views:
                    size = fixed_sizes[view]
                    offset = (largest - size) // 2
                    packed_views[view].append(base[:, offset:offset + size, offset:offset + size])
                geometry_rows.append(build_stage2_geometry(
                    image_shape=image.shape,
                    x=x,
                    y=y,
                    confidence=float(prediction.scores[index]),
                    nearest_distance=float(nearest[index]),
                    interface_key="Fixed-MV",
                ))
            geometry = torch.from_numpy(np.stack(geometry_rows)).to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp and device.type == "cuda",
            ):
                projected = {}
                for view in cfg.views:
                    image_batch = prepare_stage2_view_batch(
                        pack_stage2_view_tensors(packed_views[view]), device
                    )
                    projected[view] = model.encode_view(image_batch, view)
                fused = model.fuse_projected_views(projected, geometry)
                outputs = model.classify_fused(fused)
                probability = hierarchical_probabilities(outputs, cfg.loss_key)
            chunks.append(probability.float().cpu().numpy())
            del projected, fused, outputs, probability, geometry
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def load_final_models(runtime: RuntimeConfig, hf_token: str | None = None):
    device = resolve_device()
    stage1_lock_path = runtime.paths.stage1_existing_file("stage1_lock.json")
    if not stage1_lock_path.exists():
        from puma.pipeline.final_v132 import write_stage1_deployment_lock_v132
        write_stage1_deployment_lock_v132(runtime)
    stage1_lock = json.loads(stage1_lock_path.read_text(encoding="utf-8"))
    folds = tuple(int(value) for value in stage1_lock["run_folds"])
    expected_folds = tuple(range(runtime.data.number_of_folds))
    if set(folds) != set(expected_folds):
        raise RuntimeError(f"Stage-1 deployment requires folds {expected_folds}, got {folds}.")
    stage1_name = str(stage1_lock["selected_experiment"])
    stage1_seed = int(stage1_lock.get("seed", 0))
    fallback = stage1_experiment_registry().get(stage1_name)
    lock_rows = {int(row["fold"]): row for row in stage1_lock.get("checkpoints", [])}
    stage1_models = []
    postprocess: list[tuple[float, int, float]] = []
    for fold in folds:
        lock_row = lock_rows.get(fold, {})
        saved = resolve_artifact_reference(
            lock_row.get("checkpoint", ""), runtime.paths.stage1_output_search_dirs()
        )
        path = saved if saved is not None and saved.exists() else runtime.paths.stage1_existing_file(
            f"stage1_best_{stage1_name}_fold{fold}_seed{stage1_seed}.pt"
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected_stage1_hash = str(lock_row.get("config_hash", ""))
        if expected_stage1_hash and str(payload.get("extra", {}).get("config_hash", "")) != expected_stage1_hash:
            raise RuntimeError(f"Stage-1 deployment checkpoint identity mismatch: {path}")
        cfg = stage1_model_config_from_dict(payload["config"]) if payload.get("config") else fallback
        if cfg is None:
            raise KeyError(f"No Stage-1 config available for {path}.")
        model = build_stage1_model(cfg).to(device)
        restore_checkpoint_payload(payload, model)
        model.eval()
        extra = dict(payload.get("extra", {}))
        stage1_models.append(model)
        postprocess.append((
            float(lock_row.get("threshold", extra.get("threshold", 0.25))),
            int(lock_row.get("radius", extra.get("radius", 3))),
            float(lock_row.get("suppression_radius", extra.get("suppression_radius", 5.0))),
        ))

    final_lock_path = runtime.paths.stage2_existing_file("stage2_v132_final_lock.json")
    if not final_lock_path.exists():
        raise FileNotFoundError(
            "Missing stage2_v132_final_lock.json. Run final V13.2 all-data training first."
        )
    final_lock = json.loads(final_lock_path.read_text(encoding="utf-8"))
    serialized = final_lock.get("selected_model_config")
    if not isinstance(serialized, dict):
        raise KeyError("stage2_v132_final_lock.json is missing selected_model_config.")
    stage2_cfg = Stage2ModelConfig(**serialized)
    checkpoint = resolve_artifact_reference(
        final_lock.get("final_checkpoint", ""), runtime.paths.stage2_output_search_dirs()
    )
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"Missing final Stage-2 checkpoint: {final_lock.get('final_checkpoint')}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_final_hash = str(final_lock.get("final_training_hash", ""))
    if expected_final_hash and str(payload.get("extra", {}).get("config_hash", "")) != expected_final_hash:
        raise RuntimeError(f"Final Stage-2 training identity mismatch at {checkpoint}.")
    threshold = float(final_lock.get("validity_threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError(f"Invalid locked validity threshold: {threshold}.")
    locked_deployment_hash = str(final_lock.get("deployment_hash", ""))
    if locked_deployment_hash:
        expected_deployment_hash = config_hash({
            "final_training_hash": expected_final_hash,
            "validity_threshold": threshold,
            "revision": 1,
        })
        if locked_deployment_hash != expected_deployment_hash:
            raise RuntimeError("Final Stage-2 deployment lock identity mismatch.")
    checkpoint_cfg = Stage2ModelConfig(**payload["config"]) if payload.get("config") else stage2_cfg
    if config_hash(checkpoint_cfg) != config_hash(stage2_cfg):
        raise RuntimeError(f"Final Stage-2 checkpoint config mismatch at {checkpoint}.")
    # Grand Challenge containers run offline. Inference must therefore be fully
    # self-contained and must never depend on a Hugging Face download at runtime.
    uni2_binary = uni2_checkpoint_path(runtime.paths.root)
    if not uni2_binary.is_file() or uni2_binary.stat().st_size < 1_000_000:
        raise FileNotFoundError(
            "Missing local UNI2-h binary required for offline inference: "
            f"{uni2_binary}. Include PUMA_pretrained_checkpoints/UNI2-h/"
            "uni2_h_model.bin in the submission/container before deployment."
        )
    stage2_model = build_stage2_model(checkpoint_cfg, hf_token=None).to(device)
    restore_checkpoint_payload(payload, stage2_model)
    stage2_model.eval()
    return (
        stage1_models,
        postprocess,
        stage2_model,
        checkpoint_cfg,
        threshold,
    )


def _classified_rows(
    image_id: str,
    prediction: DecodedCandidates,
    probabilities: np.ndarray,
    cfg: Stage2ModelConfig,
    validity_threshold: float,
    *,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> np.ndarray:
    classes_tensor, confidence_tensor = decode_stage2_probabilities(
        torch.from_numpy(probabilities), cfg.loss_key, validity_threshold
    )
    classes = classes_tensor.numpy().astype(int, copy=False)
    classifier_confidence = confidence_tensor.numpy().astype(np.float32, copy=False)
    final_scores = (prediction.scores * classifier_confidence).astype(np.float32)
    rows = []
    for index, class_id in enumerate(classes):
        if class_id == REJECT_CLASS_ID:
            continue
        rows.append((
            str(image_id),
            float(prediction.coordinates[index, 0] + x_offset),
            float(prediction.coordinates[index, 1] + y_offset),
            int(class_id),
            PUMA_CLASS_NAMES[int(class_id)],
            float(final_scores[index]),
            float(prediction.scores[index]),
            float(classifier_confidence[index]),
        ))
    return np.asarray(rows, dtype=FINAL_DTYPE)


def _predict_loaded_roi(
    runtime: RuntimeConfig,
    image_id: str,
    image: np.ndarray,
    *,
    models,
) -> np.ndarray:
    """Run the complete two-stage pipeline on one in-memory PUMA ROI.

    Geometry is computed only after Stage-1 ensemble/suppression using the complete
    candidate set of the ROI, so internal Stage-1 tile boundaries never affect
    nearest-neighbour or density features.
    """
    stage1_models, postprocess, stage2_model, stage2_cfg, validity_threshold = models
    prediction = ensemble_stage1(stage1_models, image, runtime, postprocess)
    probabilities = predict_candidate_probabilities(
        stage2_model,
        image,
        prediction,
        stage2_cfg,
        runtime.training.stage2_micro_batch_size,
        amp=runtime.training.amp,
        prefer_bfloat16=runtime.training.prefer_bfloat16,
    )
    rows = _classified_rows(
        image_id, prediction, probabilities, stage2_cfg, validity_threshold
    )
    return rows


def predict_challenge_roi(
    runtime: RuntimeConfig,
    image_path: Path,
    *,
    models=None,
    hf_token: str | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Run PUMA challenge inference on the official 1024x1024 ROI input.

    The official PUMA dataset/challenge uses 1024x1024 ROIs.  Failing loudly on a
    different spatial shape prevents an accidental train/inference geometry shift.
    """
    if models is None:
        models = load_final_models(runtime, hf_token)
    image_path = Path(image_path)
    image = read_rgb(image_path)
    height, width = map(int, image.shape[:2])
    expected = (int(runtime.data.image_height), int(runtime.data.image_width))
    if (height, width) != expected:
        raise ValueError(
            f"PUMA challenge input must be {expected[0]}x{expected[1]} pixels, "
            f"got {height}x{width} for {image_path}. Refusing to run because Stage-2 "
            "ROI-relative geometry was trained on 1024x1024 inputs."
        )
    rows = _predict_loaded_roi(runtime, image_path.stem, image, models=models)
    print(
        f"{image_path.name}: PUMA ROI {width}x{height} -> {len(rows)} retained nuclei"
    )
    return rows, (height, width)

def predict_image(
    runtime: RuntimeConfig,
    image_path: Path,
    *,
    models=None,
    hf_token: str | None = None,
) -> np.ndarray:
    if models is None:
        models = load_final_models(runtime, hf_token)
    image_path = Path(image_path)
    image = read_rgb(image_path)
    rows = _predict_loaded_roi(runtime, image_path.stem, image, models=models)
    print(f"{image_path.name}: {len(rows)} retained nuclei")
    return rows


def run_inference(
    runtime: RuntimeConfig,
    input_dir: Path | None = None,
    hf_token: str | None = None,
) -> dict[str, Any]:
    input_dir = Path(input_dir or runtime.paths.image_dir)
    files = sorted(
        (p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}),
        key=lambda p: p.name.lower(),
    )
    if not files:
        raise FileNotFoundError(f"No TIFF images found in {input_dir}.")
    models = load_final_models(runtime, hf_token)
    arrays = [predict_image(runtime, path, models=models) for path in files]
    array = np.concatenate(arrays) if arrays and any(len(a) for a in arrays) else np.empty(0, dtype=FINAL_DTYPE)
    npy_path = runtime.paths.stage2_file("puma_final_predictions.npy")
    atomic_save_numpy(npy_path, array, allow_pickle=False)
    csv_path = runtime.paths.stage2_file("puma_final_predictions.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FINAL_DTYPE.names)
        for row in array:
            writer.writerow([row[name] for name in FINAL_DTYPE.names])
    return {"npy": npy_path, "csv": csv_path, "count": len(array)}


def predictions_to_puma_polygons(predictions: np.ndarray, half_size: float = 1.0) -> dict[str, Any]:
    """Encode centroid predictions as evaluator-safe symmetric polygons."""
    half_size = float(half_size)
    if half_size <= 0:
        raise ValueError("half_size must be positive.")
    polygons: list[dict[str, Any]] = []
    for row in predictions:
        x, y = float(row["x"]), float(row["y"])
        score = float(np.clip(row["confidence"], 0.0, 1.0))
        points = [
            [x - half_size, y - half_size, 0.0],
            [x + half_size, y - half_size, 0.0],
            [x + half_size, y + half_size, 0.0],
            [x - half_size, y + half_size, 0.0],
        ]
        polygons.append({
            "name": str(row["class_name"]),
            "seed_point": [x, y, 0.0],
            "path_points": points,
            "sub_type": None,
            "groups": [],
            # Official evaluator reads score when present; baseline-compatible probability
            # is retained as well for forward/backward compatibility.
            "score": score,
            "probability": score,
        })
    payload = {
        "type": "Multiple polygons",
        "polygons": polygons,
        "version": {"major": 1, "minor": 0},
    }
    validate_puma_nuclei_json(payload)
    return payload


def validate_puma_nuclei_json(payload: dict[str, Any]) -> None:
    if payload.get("type") != "Multiple polygons":
        raise ValueError("PUMA nuclei JSON must have type='Multiple polygons'.")
    polygons = payload.get("polygons")
    if not isinstance(polygons, list):
        raise TypeError("PUMA nuclei JSON 'polygons' must be a list.")
    allowed = set(PUMA_CLASS_NAMES)
    for index, polygon in enumerate(polygons):
        if polygon.get("name") not in allowed:
            raise ValueError(f"Polygon {index} has unknown PUMA class {polygon.get('name')!r}.")
        points = np.asarray(polygon.get("path_points", []), dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
            raise ValueError(f"Polygon {index} must contain at least 3 path_points with x/y.")
        if not np.all(np.isfinite(points[:, :2])):
            raise ValueError(f"Polygon {index} contains non-finite coordinates.")
        score = float(polygon.get("score", polygon.get("probability", 1.0)))
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Polygon {index} has invalid confidence {score}.")
        seed = np.asarray(polygon.get("seed_point", []), dtype=np.float64)
        if seed.size >= 2:
            centroid = points[:, :2].mean(axis=0)
            if not np.allclose(centroid, seed[:2], atol=1e-5):
                raise ValueError(
                    f"Polygon {index} vertex mean {centroid.tolist()} does not match seed {seed[:2].tolist()}."
                )


def _find_single_tiff(directory: Path) -> Path:
    files = sorted(
        p for p in Path(directory).rglob("*")
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one TIFF under {directory}, found {len(files)}.")
    return files[0]


def _resolve_grand_challenge_input(input_image: Path | None) -> Path:
    if input_image is not None:
        path = Path(input_image)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Grand-Challenge input TIFF does not exist: {path}")
        if path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"Grand-Challenge input must be TIFF, got {path}.")
        return path

    existing = [directory for directory in PUMA_GC_INPUT_DIR_ALIASES if directory.exists()]
    if not existing:
        expected = ", ".join(str(path) for path in PUMA_GC_INPUT_DIR_ALIASES)
        raise FileNotFoundError(f"No PUMA input directory found. Expected one of: {expected}")
    errors: list[str] = []
    for directory in existing:
        try:
            return _find_single_tiff(directory)
        except RuntimeError as exc:
            errors.append(str(exc))
    raise RuntimeError("; ".join(errors))


def _write_tissue_mask(
    image_shape: tuple[int, int],
    destination: Path,
    source: Path | None,
    *,
    allow_background_fallback: bool,
) -> str:
    """Write the Track-2 tissue TIFF for the same 1024x1024 challenge ROI."""
    try:
        import rasterio
        from rasterio.windows import Window
    except Exception as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError("PUMA tissue-mask I/O requires rasterio.") from exc

    height, width = map(int, image_shape[:2])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source is not None:
        with rasterio.open(source) as src:
            if (int(src.height), int(src.width)) != (height, width):
                raise ValueError(
                    f"Tissue mask shape {(src.height, src.width)} != image shape {(height, width)}."
                )
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                count=1,
                dtype="uint8",
                height=height,
                width=width,
                compress="DEFLATE",
                tiled=True,
                blockxsize=512,
                blockysize=512,
                BIGTIFF="IF_SAFER",
            )
            with rasterio.open(destination, "w", **profile) as dst:
                for y0 in range(0, height, 1024):
                    h = min(1024, height - y0)
                    for x0 in range(0, width, 1024):
                        w = min(1024, width - x0)
                        window = Window(x0, y0, w, h)
                        block = src.read(1, window=window)
                        labels = np.unique(block).astype(int)
                        if not set(labels.tolist()).issubset(PUMA_TISSUE_LABELS):
                            raise ValueError(
                                f"Tissue mask contains labels outside 0..5: {labels.tolist()}"
                            )
                        dst.write(block.astype(np.uint8, copy=False), 1, window=window)
        return "provided"
    if not allow_background_fallback:
        raise FileNotFoundError(
            "Official PUMA Track-2 output also expects a tissue-mask TIFF. Provide tissue_mask_source "
            "or explicitly allow the background-only structural fallback."
        )

    # Structural fallback only. This satisfies the file contract but is not a competitive
    # tissue prediction; provide a real 0..5 tissue mask for Track-2 tissue scoring.
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "compress": "DEFLATE",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }
    zero = np.zeros((1024, 1024), dtype=np.uint8)
    with rasterio.open(destination, "w", **profile) as dst:
        for y0 in range(0, height, 1024):
            h = min(1024, height - y0)
            for x0 in range(0, width, 1024):
                w = min(1024, width - x0)
                dst.write(zero[:h, :w], 1, window=Window(x0, y0, w, h))
    return "background_fallback"

def run_grand_challenge_inference(
    runtime: RuntimeConfig,
    *,
    input_image: Path | None = None,
    output_root: Path = Path("/output"),
    tissue_mask_source: Path | None = None,
    allow_background_tissue_fallback: bool = True,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Write the official PUMA Track-2 outputs for one 1024x1024 ROI."""
    image_path = _resolve_grand_challenge_input(input_image)
    output_root = Path(output_root)
    nuclei_path = output_root / PUMA_GC_NUCLEI_JSON.name
    tissue_path = output_root / "images" / "melanoma-tissue-mask-segmentation" / f"{image_path.stem}.tif"
    models = load_final_models(runtime, hf_token)
    predictions, image_shape = predict_challenge_roi(
        runtime, image_path, models=models
    )
    payload = predictions_to_puma_polygons(predictions)
    nuclei_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = nuclei_path.with_suffix(nuclei_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    os.replace(temporary, nuclei_path)
    tissue_mode = _write_tissue_mask(
        image_shape,
        tissue_path,
        Path(tissue_mask_source) if tissue_mask_source is not None else None,
        allow_background_fallback=allow_background_tissue_fallback,
    )
    return {
        "input_image": image_path,
        "image_shape": tuple(map(int, image_shape)),
        "nuclei_json": nuclei_path,
        "nuclei_count": len(predictions),
        "tissue_mask": tissue_path,
        "tissue_mode": tissue_mode,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--grand-challenge", action="store_true")
    parser.add_argument("--input-image", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("/output"))
    parser.add_argument("--tissue-mask-source", type=Path)
    parser.add_argument("--no-background-tissue-fallback", action="store_true")
    args = parser.parse_args()
    from puma.runtime import create_runtime
    runtime = create_runtime(args.root)
    if args.grand_challenge:
        print(run_grand_challenge_inference(
            runtime,
            input_image=args.input_image,
            output_root=args.output_root,
            tissue_mask_source=args.tissue_mask_source,
            allow_background_tissue_fallback=not args.no_background_tissue_fallback,
            hf_token=os.environ.get("HF_TOKEN"),
        ))
    else:
        print(run_inference(runtime, args.input_dir, os.environ.get("HF_TOKEN")))


if __name__ == "__main__":
    main()
