from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

from puma.config import PathConfig, RuntimeConfig, validate_folds
from puma.utils import cuda_supports_native_bfloat16, resolve_amp_dtype


def configure_project_checkpoint_cache(project_root: Path | str) -> dict[str, str]:
    """Route Hugging Face/TIMM/Torch downloads into a persistent project folder.

    The project stores one persistent UNI2-h ``.bin`` under ``PROJECT_DIR`` and uses
    these Hugging Face directories only during the first gated download/conversion.
    """
    root = Path(project_root).expanduser().resolve()
    cache_root = root / "PUMA_pretrained_checkpoints"
    hf_home = cache_root / "huggingface"
    hf_hub_cache = hf_home / "hub"
    hf_xet_cache = hf_home / "xet"
    hf_assets_cache = hf_home / "assets"
    torch_home = cache_root / "torch"
    for directory in (hf_home, hf_hub_cache, hf_xet_cache, hf_assets_cache, torch_home):
        directory.mkdir(parents=True, exist_ok=True)

    uni2_checkpoint = cache_root / "UNI2-h" / "uni2_h_model.bin"
    uni2_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    values = {
        "PUMA_PROJECT_ROOT": str(root),
        "PUMA_UNI2_CHECKPOINT": str(uni2_checkpoint),
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_hub_cache),
        "HF_XET_CACHE": str(hf_xet_cache),
        "HF_ASSETS_CACHE": str(hf_assets_cache),
        # Google Drive is a shared/FUSE filesystem; avoid fragile cache symlinks.
        "HF_HUB_DISABLE_SYMLINKS": "1",
        # Compatibility with older huggingface_hub releases used by some Colab images.
        "HUGGINGFACE_HUB_CACHE": str(hf_hub_cache),
        "TORCH_HOME": str(torch_home),
    }
    os.environ.update(values)

    # If huggingface_hub was imported earlier in the notebook, update its cached
    # constants as well. Normally create_runtime() runs before that import.
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        for name, value in (
            ("HF_HOME", values["HF_HOME"]),
            ("HF_HUB_CACHE", values["HF_HUB_CACHE"]),
            ("HF_XET_CACHE", values["HF_XET_CACHE"]),
            ("HF_ASSETS_CACHE", values["HF_ASSETS_CACHE"]),
            ("HUGGINGFACE_HUB_CACHE", values["HUGGINGFACE_HUB_CACHE"]),
        ):
            if hasattr(constants, name):
                setattr(constants, name, value)
        if hasattr(constants, "HF_HUB_DISABLE_SYMLINKS"):
            constants.HF_HUB_DISABLE_SYMLINKS = True
    return values


_REQUIRED_IMPORTS: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("tifffile", "tifffile"),
    ("shapely", "shapely"),
    ("rasterio", "rasterio"),
    ("torch", "torch"),
    ("timm", "timm"),
    ("huggingface_hub", "huggingface_hub"),
    ("safetensors", "safetensors"),
    ("tqdm", "tqdm"),
    ("psutil", "psutil"),
)


def create_runtime(
    root: Path | str | None = None,
    *,
    run_folds: Iterable[int] = (0, 1, 2, 3, 4),
    seeds: Iterable[int] = (0,),
    epochs: int = 30,
    effective_batch_size: int = 32,
    stage1_micro_batch_size: int = 2,
    stage2_micro_batch_size: int = 32,
    preprocessing_workers: int = 0,
    early_stopping_enabled: bool = True,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
) -> RuntimeConfig:
    """Build and validate the shared project configuration used by every notebook."""
    project_root = Path(root or Path.cwd()).expanduser().resolve()
    configure_project_checkpoint_cache(project_root)
    runtime = RuntimeConfig(
        paths=PathConfig(
            root=project_root,
            nuclei_geojson_dir=project_root / "Dataset" / "01_training_dataset_geojson_nuclei",
            image_dir=project_root / "Dataset" / "01_training_dataset_tif_ROIs",
            artifact_dir=project_root / "PUMA_outputs",
            stage1_output_dir=project_root / "PUMA_stage1_training_outputs",
            stage2_output_dir=project_root / "PUMA_stage2_training_outputs",
            case_metadata_csv=project_root / "Dataset" / "puma_case_metadata.csv",
        )
    )
    runtime.training.run_folds = validate_folds(run_folds, runtime.data.number_of_folds)
    runtime.training.seeds = tuple(int(seed) for seed in seeds)
    runtime.training.epochs = int(epochs)
    runtime.training.effective_batch_size = int(effective_batch_size)
    runtime.training.stage1_micro_batch_size = int(stage1_micro_batch_size)
    runtime.training.stage2_micro_batch_size = int(stage2_micro_batch_size)
    runtime.training.early_stopping_enabled = bool(early_stopping_enabled)
    runtime.training.early_stopping_patience = int(early_stopping_patience)
    runtime.training.early_stopping_min_delta = float(early_stopping_min_delta)
    runtime.data.preprocessing_workers = int(preprocessing_workers)

    if not runtime.training.seeds:
        raise ValueError("At least one random seed is required.")
    if runtime.training.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {runtime.training.epochs}")
    if runtime.training.effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive.")
    for name, value in (
        ("stage1_micro_batch_size", runtime.training.stage1_micro_batch_size),
        ("stage2_micro_batch_size", runtime.training.stage2_micro_batch_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        if value > runtime.training.effective_batch_size:
            raise ValueError(
                f"{name}={value} cannot exceed effective_batch_size="
                f"{runtime.training.effective_batch_size}."
            )
        if runtime.training.effective_batch_size % value != 0:
            raise ValueError(
                f"effective_batch_size={runtime.training.effective_batch_size} must be divisible "
                f"by {name}={value} so gradient accumulation is exact."
            )
    if runtime.training.validation_interval <= 0:
        raise ValueError("validation_interval must be positive.")
    if runtime.training.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive.")
    if runtime.training.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta cannot be negative.")
    if runtime.data.preprocessing_workers < 0:
        raise ValueError("preprocessing_workers must be 0 (all logical cores) or a positive integer.")
    return runtime


def _files_with_suffixes(directory: Path, suffixes: set[str]) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes),
        key=lambda path: path.name.lower(),
    )


def _keyed_files(paths: list[Path], *, remove_nuclei_suffix: bool) -> dict[str, Path]:
    keyed: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    for path in paths:
        stem = path.stem
        if remove_nuclei_suffix and stem.lower().endswith("_nuclei"):
            stem = stem[: -len("_nuclei")]
        key = stem.casefold()
        if key in keyed:
            duplicates.setdefault(key, [keyed[key].name]).append(path.name)
        else:
            keyed[key] = path
    if duplicates:
        details = "; ".join(f"{key}: {names}" for key, names in sorted(duplicates.items()))
        raise ValueError(f"Duplicate case-insensitive dataset stems detected: {details}")
    return keyed


def _spatial_shape_from_tiff_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    squeezed = tuple(int(value) for value in shape if int(value) != 1)
    if len(squeezed) == 2:
        return squeezed
    if len(squeezed) != 3:
        raise ValueError(f"Unsupported TIFF series shape: {shape}")
    if squeezed[-1] in (1, 2, 3, 4):
        return squeezed[0], squeezed[1]
    if squeezed[0] in (1, 2, 3, 4):
        return squeezed[1], squeezed[2]
    raise ValueError(f"Could not identify a channel axis in TIFF series shape: {shape}")


def validate_dataset(runtime: RuntimeConfig, *, inspect_contents: bool = True) -> dict[str, Any]:
    """Validate TIFF/GeoJSON presence, pairing, readability, and basic PUMA dimensions."""
    geojson_dir = runtime.paths.nuclei_geojson_dir
    image_dir = runtime.paths.image_dir
    if not geojson_dir.exists() or not geojson_dir.is_dir():
        raise FileNotFoundError(f"GeoJSON directory does not exist: {geojson_dir}")
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    image_files = _files_with_suffixes(image_dir, {".tif", ".tiff"})
    geojson_files = _files_with_suffixes(geojson_dir, {".geojson"})
    if not image_files or not geojson_files:
        raise FileNotFoundError(
            f"Dataset is incomplete: {len(geojson_files)} GeoJSON files and "
            f"{len(image_files)} TIFF files."
        )

    images = _keyed_files(image_files, remove_nuclei_suffix=False)
    geojsons = _keyed_files(geojson_files, remove_nuclei_suffix=True)
    missing_geojson = sorted(set(images) - set(geojsons))
    missing_images = sorted(set(geojsons) - set(images))
    if missing_geojson or missing_images:
        raise FileNotFoundError(
            "TIFF/GeoJSON pairing is incomplete. "
            f"Images without GeoJSON: {missing_geojson[:20]}; "
            f"GeoJSON without image: {missing_images[:20]}."
        )

    feature_count = 0
    observed_shapes: set[tuple[int, int]] = set()
    if inspect_contents:
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError("tifffile is required to validate TIFF inputs.") from exc
        for key in sorted(images):
            image_path = images[key]
            geojson_path = geojsons[key]
            try:
                with tifffile.TiffFile(image_path) as tif:
                    if not tif.series:
                        raise ValueError("TIFF has no readable image series")
                    spatial_shape = _spatial_shape_from_tiff_shape(tuple(tif.series[0].shape))
            except Exception as exc:
                raise ValueError(f"Cannot read TIFF metadata from {image_path}: {exc}") from exc
            observed_shapes.add(spatial_shape)

            try:
                payload = json.loads(geojson_path.read_text(encoding="utf-8-sig"))
            except Exception as exc:
                raise ValueError(f"Cannot parse GeoJSON {geojson_path}: {exc}") from exc
            features = payload.get("features")
            if not isinstance(features, list):
                raise ValueError(f"GeoJSON has no valid 'features' list: {geojson_path}")
            feature_count += len(features)

    expected_shape = (runtime.data.image_height, runtime.data.image_width)
    nonstandard_shapes = sorted(shape for shape in observed_shapes if shape != expected_shape)
    report: dict[str, Any] = {
        "geojson_files": len(geojson_files),
        "image_files": len(image_files),
        "matched_pairs": len(images),
        "annotated_features": feature_count,
        "observed_spatial_shapes": [list(shape) for shape in sorted(observed_shapes)],
        "expected_spatial_shape": list(expected_shape),
        "nonstandard_spatial_shapes": [list(shape) for shape in nonstandard_shapes],
        "case_metadata_present": bool(runtime.paths.case_metadata_csv.exists()),
    }
    if nonstandard_shapes:
        raise ValueError(
            f"Expected PUMA ROIs of shape {expected_shape}, but found {nonstandard_shapes}."
        )
    return report


def _dependency_report(require: bool) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for display_name, import_name in _REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", None)
            report[display_name] = {"available": True, "version": str(version or "unknown")}
        except Exception as exc:
            report[display_name] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            missing.append(display_name)
    if require and missing:
        raise ImportError(
            "Missing or broken training dependencies: " + ", ".join(missing) + 
            ". Run `%pip install -r requirements_colab.txt` and restart the runtime if needed."
        )
    return report


def preflight_environment(
    runtime: RuntimeConfig,
    *,
    require_dataset: bool = True,
    require_training_dependencies: bool = True,
) -> dict[str, Any]:
    """Run fail-fast checks before preprocessing/training and return a printable report.

    This function intentionally verifies that the imported ``puma`` package comes from
    ``runtime.paths.root``. That prevents a stale ``/content/puma`` directory or an older
    package cached in ``sys.modules`` from silently shadowing the uploaded project.
    """
    package_dir = Path(__file__).resolve().parent
    expected_package_dir = (runtime.paths.root / "puma").resolve()
    if package_dir != expected_package_dir:
        raise ImportError(
            "The wrong PUMA package is loaded. "
            f"Loaded: {package_dir}; expected: {expected_package_dir}. "
            "Restart the Colab runtime and rerun the notebook bootstrap cell, which places "
            "PROJECT_DIR at the front of sys.path and clears stale puma modules."
        )

    dependencies = _dependency_report(require_training_dependencies)
    dataset = validate_dataset(runtime, inspect_contents=True) if require_dataset else None

    torch_module = importlib.import_module("torch") if dependencies.get("torch", {}).get("available") else None
    cuda: dict[str, Any] = {"available": False}
    if torch_module is not None:
        cuda["available"] = bool(torch_module.cuda.is_available())
        cuda["torch_version"] = str(torch_module.__version__)
        if torch_module.cuda.is_available():
            device_index = int(torch_module.cuda.current_device())
            properties = torch_module.cuda.get_device_properties(device_index)
            cuda.update(
                {
                    "device_index": device_index,
                    "device_name": str(properties.name),
                    "total_vram_gb": round(float(properties.total_memory) / (1024**3), 2),
                    "bfloat16_supported": bool(torch_module.cuda.is_bf16_supported()),
                    "native_bfloat16_supported": cuda_supports_native_bfloat16(device_index),
                    "selected_amp_dtype": str(resolve_amp_dtype(runtime.training.prefer_bfloat16, device_index)).replace("torch.", ""),
                }
            )

    report = {
        "project_root": str(runtime.paths.root),
        "puma_package": str(package_dir),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset": dataset,
        "dependencies": dependencies,
        "cuda": cuda,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if require_training_dependencies and not cuda["available"]:
        print("WARNING: CUDA GPU is not available. Preprocessing can run, but model training will be very slow.")
    return report


def resolve_hf_token() -> str | None:
    """Read the Hugging Face token from Colab secrets or the process environment."""
    token: str | None = None
    try:
        from google.colab import userdata  # type: ignore

        token = userdata.get("HF_TOKEN")
    except Exception:
        token = os.environ.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
    return token
