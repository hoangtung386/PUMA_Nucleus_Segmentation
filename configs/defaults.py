"""Centralized configuration with sensible defaults — v8 CellPath."""

from dataclasses import dataclass, field
from pathlib import Path

from symbiopan.data.constants import (
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    NUCLEI_CLASS_WEIGHTS,
    TISSUE_CLASS_WEIGHTS,
)


@dataclass(frozen=True)
class PathsConfig:
    root: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "dataset_processed")
    checkpoint_dir: Path = field(default_factory=lambda: Path.cwd() / "checkpoints")
    split_file: Path = field(default_factory=lambda: Path.cwd() / "checkpoints" / "split_seed42.npz")
    raw_dir: Path = field(default_factory=lambda: Path.cwd() / "Dataset")


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 1024
    make_rare_centered_crops: bool = True
    max_rare_crops_per_image: int = 3
    rare_crop_jitter_px: int = 96
    random_seed: int = 42
    skip_existing: bool = True
    rebuild_metadata: bool = True


@dataclass(frozen=True)
class Stage1Config:
    image_size: int = 1024
    batch_size: int = 8
    grad_accum_steps: int = 2
    epochs: int = 50
    num_workers: int = 2
    seed: int = 42
    val_ratio: float = 0.2
    force_new_split: bool = False
    val_original_only: bool = True
    max_sample_weight: float = 15.0

    lr: float = 1e-4
    weight_decay: float = 1e-4

    warmup_epochs: int = 5

    focal_start_epoch: int = 10
    focal_full_epoch: int = 16
    focal_max_weight: float = 0.5

    sc_dfa_start_epoch: int = 15
    sc_dfa_full_epoch: int = 22
    sc_dfa_max_weight: float = 0.3

    fine_tune_last_n_blocks: int = 6

    use_context_encoder: bool = False
    context_roi_size: int = 320
    use_stain_aug: bool = False
    mixup_prob: float = 0.0
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0

    samples_per_epoch_multiplier: float = 1.0
    multi_gpu: bool = False
    use_fp16: bool = True
    compile_model: bool = True
    resume: str | None = None

    tissue_class_weights: tuple = field(default_factory=lambda: tuple(TISSUE_CLASS_WEIGHTS))
    nuclei_class_weights: tuple = field(default_factory=lambda: tuple(NUCLEI_CLASS_WEIGHTS))
    normalization_mean: tuple = field(default_factory=lambda: tuple(NORMALIZATION_MEAN))
    normalization_std: tuple = field(default_factory=lambda: tuple(NORMALIZATION_STD))


@dataclass(frozen=True)
class InferenceConfig:
    input_dir: str = "/input/images/melanoma-whole-slide-image"
    output_dir: str = "/output"
    cp: str = "/opt/app/checkpoints/best_model.pth"
    tile_size: int = 1024
    overlap: int = 256
    site_type: str | None = None
    site_classifier_cp: str = "/opt/app/checkpoints/site_classifier_atto.pth"
    site_classifier_arch: str = "convnext_tiny"
    site_classifier_size: int = 256
    use_tta: bool = False
    np_threshold: float = 0.50
    min_nucleus_area: int = 20


PATHS = PathsConfig()

STAGE1_DEFAULT_CONFIG = Stage1Config()
PREPROCESS_DEFAULT_CONFIG = PreprocessConfig()
INFERENCE_DEFAULT_CONFIG = InferenceConfig()
