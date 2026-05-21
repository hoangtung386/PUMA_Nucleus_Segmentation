"""Centralized configuration with sensible defaults."""

from dataclasses import dataclass, field
from pathlib import Path

import torch

from data.constants import (
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    NUCLEI_CLASS_WEIGHTS,
    STAGE2_NUCLEI_WEIGHTS,
    TISSUE_CLASS_WEIGHTS,
)


@dataclass(frozen=True)
class PathsConfig:
    root: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "dataset_processed")
    checkpoint_dir: Path = field(default_factory=lambda: Path.cwd() / "checkpoints")
    uni_weight_dir: Path = field(default_factory=lambda: Path.cwd())
    split_file: Path = field(default_factory=lambda: Path.cwd() / "checkpoints" / "split_seed42.npz")
    raw_dir: Path = field(default_factory=lambda: Path.cwd() / "Dataset")


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 1024
    generate_cellpose_flows: bool = True
    cellpose_model_type: str = "cyto3"  # Cyto3 for training: better boundary detection for rare classes
    # NOTE: Inference uses "nuclei" model - see InferenceConfig. Do NOT change without regenerating training data.
    cellpose_batch_size: int = 1
    make_rare_centered_crops: bool = True
    max_rare_crops_per_image: int = 3
    rare_crop_jitter_px: int = 96
    random_seed: int = 42
    skip_existing: bool = True
    rebuild_metadata: bool = True


@dataclass(frozen=True)
class Stage1Config:
    image_size: int = 1024
    stride: int = 768
    batch_size: int = 12
    epochs: int = 50
    num_workers: int = 2
    seed: int = 42
    val_ratio: float = 0.2
    force_new_split: bool = False
    val_original_only: bool = True
    max_sample_weight: float = 15.0

    lr: float = 1e-4
    weight_decay: float = 1e-4

    focal_start_epoch: int = 10
    focal_full_epoch: int = 16
    focal_max_weight: float = 0.5

    sc_dfa_start_epoch: int = 15
    sc_dfa_full_epoch: int = 22
    sc_dfa_max_weight: float = 0.3

    prior_start_epoch: int = 20
    prior_full_epoch: int = 28
    prior_max_weight: float = 0.2

    default_site_type: str = "metastatic"
    zero_cellpose_prob: float = 0.0
    samples_per_epoch_multiplier: float = 1.0
    multi_gpu: bool = False
    use_fp16: bool = True
    resume: str | None = None

    tissue_class_weights: tuple = field(default_factory=lambda: tuple(TISSUE_CLASS_WEIGHTS))
    nuclei_class_weights: tuple = field(default_factory=lambda: tuple(NUCLEI_CLASS_WEIGHTS))
    normalization_mean: tuple = field(default_factory=lambda: tuple(NORMALIZATION_MEAN))
    normalization_std: tuple = field(default_factory=lambda: tuple(NORMALIZATION_STD))


@dataclass(frozen=True)
class Stage2Config:
    image_size: int = 1024
    batch_size: int = 16
    epochs: int = 30
    num_workers: int = 2
    seed: int = 42
    val_ratio: float = 0.2
    max_sample_weight: float = 50.0
    force_new_split: bool = False
    val_original_only: bool = True

    lr: float = 1e-4
    weight_decay: float = 1e-4

    default_site_type: str = "metastatic"
    use_fp16: bool = True
    resume: str | None = None
    samples_per_epoch_multiplier: float = 2.5

    num_nuclei_classes: int = 10
    stage2_in_channels: int = 21
    ignore_index: int = 255

    nuclei_weights: tuple = field(default_factory=lambda: tuple(STAGE2_NUCLEI_WEIGHTS))
    rare_nuclei_ids: tuple = field(default_factory=lambda: (2, 4, 5, 8, 9))

    kd_temperature: float = 2.0
    keep_lambda_start: float = 0.80
    keep_lambda_end: float = 0.15
    keep_lambda_decay_epochs: int = 30
    alpha_start: float = 0.05
    alpha_end: float = 0.45
    alpha_warmup_epochs: int = 30


@dataclass(frozen=True)
class InferenceConfig:
    input_dir: str = "/input/images/melanoma-whole-slide-image"
    output_dir: str = "/output"
    cp: str = "/opt/app/checkpoints/best_model.pth"
    stage2_cp: str | None = None
    tile_size: int = 1024
    overlap: int = 256
    site_type: str | None = None
    site_classifier_cp: str = "/opt/app/checkpoints/site_classifier_atto.pth"
    site_classifier_arch: str = "convnext_atto"
    site_classifier_size: int = 256
    cellpose_mode: str = "auto"
    cellpose_model_type: str = "nuclei"  # Nuclei for inference: faster, focuses on nuclear boundaries
    # NOTE: Training data was generated with "cyto3" - do NOT change without regenerating training data.
    np_threshold: float = 0.50
    min_nucleus_area: int = 20


PATHS = PathsConfig()

STAGE1_DEFAULT_CONFIG = Stage1Config()
STAGE2_DEFAULT_CONFIG = Stage2Config()
PREPROCESS_DEFAULT_CONFIG = PreprocessConfig()
INFERENCE_DEFAULT_CONFIG = InferenceConfig()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_ramp(epoch: int, start: int, end: int, max_value: float) -> float:
    if epoch < start:
        return 0.0
    if epoch >= end:
        return float(max_value)
    progress = (epoch - start + 1) / max(end - start + 1, 1)
    return float(max_value) * float(progress)
