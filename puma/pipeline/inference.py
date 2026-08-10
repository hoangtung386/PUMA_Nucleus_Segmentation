from __future__ import annotations

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
    PathConfig,
    RuntimeConfig,
    stage1_model_config_from_dict,
    Stage2ModelConfig,
    stage1_experiment_registry,
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


FINAL_DTYPE = np.dtype(
    [
        ("image_id", "U128"),
        ("x", "f4"),
        ("y", "f4"),
        ("class_id", "i2"),
        ("class_name", "U32"),
        ("confidence", "f4"),
        ("detector_confidence", "f4"),
        ("classifier_confidence", "f4"),
        ("width", "f4"),
        ("height", "f4"),
        ("theta_radians", "f4"),
    ]
)


def read_rgb(path: Path) -> np.ndarray:
    return read_tiff_rgb(Path(path))


def _empty_candidates() -> DecodedCandidates:
    return DecodedCandidates(
        coordinates=np.empty((0, 2), np.float32),
        scores=np.empty(0, np.float32),
    )


def ensemble_stage1(models, image, runtime, threshold_radius) -> DecodedCandidates:
    """Average deployment coverage through the five A1 fold models."""
    device = resolve_device()
    amp_dtype = resolve_amp_dtype(runtime.training.prefer_bfloat16, device)
    predictions: list[DecodedCandidates] = []
    for model, (threshold, radius) in zip(models, threshold_radius, strict=True):
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
        )
        if len(prediction.scores):
            predictions.append(prediction)
    if not predictions:
        return _empty_candidates()

    merged = DecodedCandidates(
        coordinates=np.concatenate([item.coordinates for item in predictions]),
        scores=np.concatenate([item.scores for item in predictions]),
    )
    return adaptive_suppress(merged, min_radius=2, max_radius=6)


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
) -> np.ndarray:
    """Predict V13 Stage-2 probabilities for A1 candidates."""
    if len(prediction.scores) == 0:
        return np.empty((0, REJECT_CLASS_ID + 1), dtype=np.float32)
    if cfg.interface_key != "Fixed-MV" or any(view not in {"V2", "V3", "V4"} for view in cfg.views):
        raise ValueError("V13 inference supports only fixed V2/V3/V4 views.")

    device = resolve_device()
    amp_dtype = resolve_amp_dtype(prefer_bfloat16, device)
    fixed_sizes = {"V2": 64, "V3": 128, "V4": 256}
    nearest = _nearest_distances(prediction.coordinates, image.shape)
    chunks: list[np.ndarray] = []
    model.eval()

    with torch.inference_mode():
        for start in range(0, len(prediction.scores), batch_size):
            stop = min(start + batch_size, len(prediction.scores))
            packed_views: dict[str, list[torch.Tensor]] = {view: [] for view in cfg.views}
            geometry_rows: list[np.ndarray] = []
            largest = max(fixed_sizes[view] for view in cfg.views)

            for index in range(start, stop):
                x, y = map(float, prediction.coordinates[index])
                base = image_to_uint8_tensor(extract_crop(image, x, y, largest))
                for view in cfg.views:
                    size = fixed_sizes[view]
                    offset = (largest - size) // 2
                    packed_views[view].append(base[:, offset : offset + size, offset : offset + size])
                geometry_rows.append(
                    build_stage2_geometry(
                        image_shape=image.shape,
                        x=x,
                        y=y,
                        confidence=float(prediction.scores[index]),
                        nearest_distance=float(nearest[index]),
                        interface_key="Fixed-MV",
                    )
                )

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
    """Load the fixed five-fold A1 ensemble and the final V13 Stage-2 model."""
    device = resolve_device()
    stage1_lock_path = runtime.paths.stage1_existing_file("stage1_lock.json")
    if not stage1_lock_path.exists():
        from puma.pipeline.final_v13 import write_fixed_stage1_lock_v13

        write_fixed_stage1_lock_v13(runtime)
    stage1_lock = json.loads(stage1_lock_path.read_text(encoding="utf-8"))
    folds = tuple(int(value) for value in stage1_lock["run_folds"])
    expected_folds = tuple(range(runtime.data.number_of_folds))
    if set(folds) != set(expected_folds):
        raise RuntimeError(f"Stage-1 deployment requires folds {expected_folds}, got {folds}.")

    stage1_name = str(stage1_lock["selected_experiment"])
    stage1_seed = int(stage1_lock["seeds"][0])
    fallback = stage1_experiment_registry().get(stage1_name)
    stage1_models = []
    threshold_radius: list[tuple[float, int]] = []
    for fold in folds:
        path = runtime.paths.stage1_existing_file(
            f"stage1_best_{stage1_name}_fold{fold}_seed{stage1_seed}.pt"
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = stage1_model_config_from_dict(payload["config"]) if payload.get("config") else fallback
        if cfg is None:
            raise KeyError(f"No Stage-1 config is available for {path}.")
        model = build_stage1_model(cfg).to(device)
        restore_checkpoint_payload(payload, model)
        model.eval()
        stage1_models.append(model)
        threshold_radius.append(
            (
                float(payload.get("extra", {}).get("threshold", 0.25)),
                int(payload.get("extra", {}).get("radius", 3)),
            )
        )

    final_lock_path = runtime.paths.stage2_existing_file("stage2_v13_final_lock.json")
    if not final_lock_path.exists():
        raise FileNotFoundError(
            "Missing stage2_v13_final_lock.json. Train the locked V13 winner before inference."
        )
    final_lock = json.loads(final_lock_path.read_text(encoding="utf-8"))
    serialized = final_lock.get("selected_model_config")
    if not isinstance(serialized, dict):
        raise KeyError("stage2_v13_final_lock.json is missing selected_model_config.")
    stage2_cfg = Stage2ModelConfig(**serialized)
    checkpoint = resolve_artifact_reference(
        final_lock.get("final_checkpoint", ""), runtime.paths.stage2_output_search_dirs()
    )
    if checkpoint is None or not checkpoint.exists():
        expected = runtime.paths.stage2_file(
            f"stage2_v13_final_{stage2_cfg.name}_seed{int(final_lock.get('seed', 0))}.pt"
        )
        checkpoint = expected if expected.exists() else checkpoint
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing final Stage-2 checkpoint: {final_lock.get('final_checkpoint')}"
        )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_cfg = Stage2ModelConfig(**payload["config"]) if payload.get("config") else stage2_cfg
    if config_hash(checkpoint_cfg) != config_hash(stage2_cfg):
        raise RuntimeError(f"Final Stage-2 checkpoint config mismatch at {checkpoint}.")
    stage2_model = build_stage2_model(checkpoint_cfg, hf_token=hf_token).to(device)
    restore_checkpoint_payload(payload, stage2_model)
    stage2_model.eval()

    return (
        stage1_models,
        threshold_radius,
        stage2_model,
        checkpoint_cfg,
        float(final_lock.get("validity_threshold", 0.5)),
    )


def run_inference(
    runtime: RuntimeConfig,
    input_dir: Path | None = None,
    hf_token: str | None = None,
) -> dict[str, Any]:
    input_dir = Path(input_dir or runtime.paths.image_dir)
    files = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ),
        key=lambda path: path.name.lower(),
    )
    if not files:
        raise FileNotFoundError(f"No TIFF images found in {input_dir}.")

    stage1_models, thresholds, stage2_model, stage2_cfg, validity_threshold = load_final_models(
        runtime, hf_token
    )
    rows = []
    for path in files:
        image = read_rgb(path)
        prediction = ensemble_stage1(stage1_models, image, runtime, thresholds)
        probabilities = predict_candidate_probabilities(
            stage2_model,
            image,
            prediction,
            stage2_cfg,
            runtime.training.stage2_micro_batch_size,
            amp=runtime.training.amp,
            prefer_bfloat16=runtime.training.prefer_bfloat16,
        )
        classes_tensor, confidence_tensor = decode_stage2_probabilities(
            torch.from_numpy(probabilities), stage2_cfg.loss_key, validity_threshold
        )
        classes = classes_tensor.numpy().astype(int, copy=False)
        classifier_confidence = confidence_tensor.numpy().astype(np.float32, copy=False)
        final_scores = (prediction.scores * classifier_confidence).astype(np.float32)

        for index, class_id in enumerate(classes):
            if class_id == REJECT_CLASS_ID:
                continue
            rows.append(
                (
                    path.stem,
                    prediction.coordinates[index, 0],
                    prediction.coordinates[index, 1],
                    class_id,
                    PUMA_CLASS_NAMES[class_id],
                    final_scores[index],
                    prediction.scores[index],
                    classifier_confidence[index],
                    16.0,
                    16.0,
                    0.0,
                )
            )
        print(
            f"{path.name}: {len(prediction.scores)} Stage-1 candidates -> "
            f"{int(np.sum(classes != REJECT_CLASS_ID))} retained"
        )

    array = np.asarray(rows, dtype=FINAL_DTYPE)
    npy_path = runtime.paths.stage2_file("puma_final_predictions.npy")
    atomic_save_numpy(npy_path, array, allow_pickle=False)
    csv_path = runtime.paths.stage2_file("puma_final_predictions.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FINAL_DTYPE.names)
        for row in array:
            writer.writerow([row[name] for name in FINAL_DTYPE.names])
    return {"npy": npy_path, "csv": csv_path, "count": len(array)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args()
    runtime = RuntimeConfig(paths=PathConfig(root=args.root))
    print(run_inference(runtime, args.input_dir, os.environ.get("HF_TOKEN")))


if __name__ == "__main__":
    main()
