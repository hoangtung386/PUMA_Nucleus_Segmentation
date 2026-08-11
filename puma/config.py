from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable


PUMA_CLASS_NAMES: tuple[str, ...] = (
    "nuclei_tumor",
    "nuclei_lymphocyte",
    "nuclei_plasma_cell",
    "nuclei_histiocyte",
    "nuclei_melanophage",
    "nuclei_neutrophil",
    "nuclei_stroma",
    "nuclei_endothelium",
    "nuclei_epithelium",
    "nuclei_apoptosis",
)
PUMA_CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(PUMA_CLASS_NAMES)}
REJECT_CLASS_ID = len(PUMA_CLASS_NAMES)
PUMA_MICRONS_PER_PIXEL = 0.23
STAGE2_GEOMETRY_NAMES: tuple[str, ...] = (
    # PUMA training and challenge inference both operate on 1024x1024 ROIs.
    # Keep the original 7-D ROI geometry so Stage-2 sees the same feature semantics
    # during training, validation, local inference, and Grand-Challenge inference.
    "log_nearest_distance",
    "local_density",
    "detector_confidence",
    "border_distance_normalized",
    "microns_per_pixel",
    "x_normalized",
    "y_normalized",
)
STAGE2_GEOMETRY_DIM = len(STAGE2_GEOMETRY_NAMES)
TAIL_CLASS_IDS: tuple[int, ...] = tuple(
    PUMA_CLASS_TO_ID[name]
    for name in (
        "nuclei_plasma_cell",
        "nuclei_neutrophil",
        "nuclei_apoptosis",
        "nuclei_melanophage",
        "nuclei_endothelium",
    )
)


@dataclass(slots=True)
class PathConfig:
    """Project, artifact, and training-output paths."""

    root: Path = field(default_factory=Path.cwd)
    nuclei_geojson_dir: Path | None = None
    image_dir: Path | None = None
    artifact_dir: Path | None = None
    stage1_output_dir: Path | None = None
    stage2_output_dir: Path | None = None
    case_metadata_csv: Path | None = None

    _STAGE1_SMALL_OUTPUTS: ClassVar[tuple[str, ...]] = (
        "stage1_results.csv",
        "stage1_lock.json",
        "stage1_oof_candidates_metadata.json",
    )
    _STAGE2_SMALL_OUTPUTS: ClassVar[tuple[str, ...]] = (
        "stage2_v132_lock.json",
        "stage2_v132_final_lock.json",
    )

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.nuclei_geojson_dir = Path(self.nuclei_geojson_dir or (
            self.root / "Dataset" / "01_training_dataset_geojson_nuclei"
        ))
        self.image_dir = Path(self.image_dir or (
            self.root / "Dataset" / "01_training_dataset_tif_ROIs"
        ))
        self.artifact_dir = Path(self.artifact_dir or (self.root / "PUMA_outputs"))
        self.stage1_output_dir = Path(
            self.stage1_output_dir or (self.root / "PUMA_stage1_training_outputs")
        )
        self.stage2_output_dir = Path(
            self.stage2_output_dir or (self.root / "PUMA_stage2_training_outputs")
        )
        self.case_metadata_csv = Path(
            self.case_metadata_csv or (self.root / "Dataset" / "puma_case_metadata.csv")
        )

    @property
    def pretrained_checkpoint_dir(self) -> Path:
        """Persistent project-local cache for downloaded pretrained model weights."""
        return self.root / "PUMA_pretrained_checkpoints"

    @property
    def uni2_checkpoint_file(self) -> Path:
        """Single persistent UNI2-h binary reused by all Stage-2 sessions."""
        return self.pretrained_checkpoint_dir / "UNI2-h" / "uni2_h_model.bin"

    @property
    def huggingface_home_dir(self) -> Path:
        return self.pretrained_checkpoint_dir / "huggingface"

    @property
    def huggingface_hub_cache_dir(self) -> Path:
        return self.huggingface_home_dir / "hub"

    @property
    def huggingface_xet_cache_dir(self) -> Path:
        return self.huggingface_home_dir / "xet"

    @property
    def huggingface_assets_cache_dir(self) -> Path:
        return self.huggingface_home_dir / "assets"

    @property
    def torch_home_dir(self) -> Path:
        return self.pretrained_checkpoint_dir / "torch"

    @property
    def preprocessing_dir(self) -> Path:
        return self.artifact_dir

    def ensure(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.stage1_output_dir.mkdir(parents=True, exist_ok=True)
        self.stage2_output_dir.mkdir(parents=True, exist_ok=True)
        self.uni2_checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self.huggingface_hub_cache_dir.mkdir(parents=True, exist_ok=True)
        self.huggingface_xet_cache_dir.mkdir(parents=True, exist_ok=True)
        self.huggingface_assets_cache_dir.mkdir(parents=True, exist_ok=True)
        self.torch_home_dir.mkdir(parents=True, exist_ok=True)

    def preprocessing_file(self, name: str) -> Path:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        return self.artifact_dir / name

    def stage1_file(self, name: str) -> Path:
        self.ensure()
        return self.stage1_output_dir / name

    def stage2_file(self, name: str) -> Path:
        self.ensure()
        return self.stage2_output_dir / name

    def stage1_output_search_dirs(self) -> tuple[Path, ...]:
        # One canonical training-output location. Deleting this directory must
        # intentionally force a fresh Stage-1 run rather than reuse legacy files.
        self.ensure()
        return (self.stage1_output_dir,)

    def stage2_output_search_dirs(self) -> tuple[Path, ...]:
        # One canonical training-output location. Deleting this directory must
        # intentionally force a fresh Stage-2 run rather than reuse legacy files.
        self.ensure()
        return (self.stage2_output_dir,)

    def stage1_existing_file(self, name: str) -> Path:
        if Path(name).name in self._STAGE1_SMALL_OUTPUTS:
            return self.stage1_output_dir / name
        for directory in self.stage1_output_search_dirs():
            candidate = directory / name
            if candidate.exists():
                return candidate
        return self.stage1_output_dir / name

    def stage2_existing_file(self, name: str) -> Path:
        if Path(name).name in self._STAGE2_SMALL_OUTPUTS:
            return self.stage2_output_dir / name
        for directory in self.stage2_output_search_dirs():
            candidate = directory / name
            if candidate.exists():
                return candidate
        return self.stage2_output_dir / name



@dataclass(slots=True)
class DataConfig:
    image_height: int = 1024
    image_width: int = 1024
    channels: int = 3
    number_of_folds: int = 5
    random_seed: int = 2026
    preprocessing_workers: int = 0  # 0 = all available logical CPU cores
    rasterize_all_touched: bool = False
    centroid_method: str = "official_vertex_mean"
    canonical_heatmap_sigma_scale: float = 0.15
    canonical_heatmap_sigma_min: float = 1.5
    canonical_heatmap_sigma_max: float = 5.0
    official_match_radius_px: float = 15.0
    tile_size: int = 512
    tile_overlap: int = 128
    tiles_per_roi_per_epoch: int = 8
    validation_tile_batch_size: int = 4
    background_fraction: float = 0.05
    density_fraction: float = 0.30
    small_nucleus_fraction: float = 0.20
    uniform_fraction: float = 0.30
    rare_nucleus_fraction: float = 0.15
    use_reflection_padding: bool = True
    class_map_background_id: int = 255
    fail_on_annotation_error: bool = True


@dataclass(slots=True)
class TrainingConfig:
    """User-requested defaults. Batch size means effective batch size."""

    run_folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    seeds: tuple[int, ...] = (0,)
    stage1_epochs: int = 40
    stage2_epochs: int = 50
    stage1_effective_batch_size: int = 16
    stage2_effective_batch_size: int = 256
    stage1_micro_batch_size: int = 16
    stage2_micro_batch_size: int = 256
    number_of_workers: int = 2
    amp: bool = True
    prefer_bfloat16: bool = True
    deterministic: bool = True
    validation_interval: int = 2
    stage1_early_stopping_enabled: bool = True
    stage1_early_stopping_patience: int = 10
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    gradient_clip_norm: float = 1.0
    save_best_only: bool = True
    resume_from_results_csv: bool = True
    resume_checkpoint_interval: int = 1
    threshold_grid: tuple[float, ...] = (0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40)
    local_max_radius_grid: tuple[int, ...] = (2, 3, 4, 5)
    suppression_radius_grid: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0)
    stage1_recall_tolerance: float = 0.005

    @property
    def stage1_accumulation_steps(self) -> int:
        return max(1, self.stage1_effective_batch_size // self.stage1_micro_batch_size)

    @property
    def stage2_accumulation_steps(self) -> int:
        return max(1, self.stage2_effective_batch_size // self.stage2_micro_batch_size)


@dataclass(slots=True)
class Stage1ModelConfig:
    name: str
    learning_rate: float
    weight_decay: float
    fixed_sigma: float = 2.5


def stage1_model_config_from_dict(payload: dict[str, Any]) -> Stage1ModelConfig:
    """Load the retained A1 fields from a saved config."""
    return Stage1ModelConfig(
        name=str(payload["name"]),
        learning_rate=float(payload["learning_rate"]),
        weight_decay=float(payload["weight_decay"]),
        fixed_sigma=float(payload.get("fixed_sigma", 2.5)),
    )


@dataclass(slots=True)
class Stage2ModelConfig:
    """Configuration for one V13.2 UNI2-h experiment."""

    name: str
    pfm_key: str = "uni2_h"
    views: tuple[str, ...] = ("V2", "V3")
    pooling_key: str = "cls_center_ring"
    use_geometry: bool = False
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_last_blocks: int = 8
    loss_key: str = "HIERARCHICAL"
    schedule_key: str = "GT_POS+OOF_POS+OOF_ALL"
    interface_key: str = "Fixed-MV"
    hidden_dim: int = 512
    fusion_layers: int = 2
    learning_rate: float = 1e-4
    validity_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    phase1_start_lr: float = 1e-4
    phase1_end_lr: float = 5e-5
    phase2_start_lr: float = 7.5e-5
    phase2_end_lr: float = 3e-5
    phase3_start_lr: float = 5e-5
    phase3_end_lr: float = 5e-6
    phase3_validity_start_lr: float = 1e-4
    phase3_validity_end_lr: float = 1e-5
    warmup_epochs: int = 3
    encoder_micro_batch_size: int = 256
    type_loss_key: str = "BALANCED_SOFTMAX"
    validity_loss_key: str = "BCE"
    type_loss_weight: float = 1.0
    validity_loss_weight: float = 1.0
    type_focal_gamma: float = 2.0
    class_balance_beta: float = 0.9999
    sampler_positive_fraction: float = 2.0 / 3.0
    use_strong_rare_sampling: bool = True
    sampler_balanced_positive_fraction: float = 0.25
    sampler_max_repeats: int = 4
    sampler_tail_max_repeats: int = 12
    rare_quota_gt_per_class: int = 16
    rare_quota_oof_pos_per_class: int = 12
    rare_quota_oof_all_per_class: int = 8
    hard_negative_start_phase_epoch: int = 4
    hard_positive_start_phase_epoch: int = 4
    hard_pool_refresh_interval: int = 3
    hard_reject_fraction: float = 0.50
    hard_rare_fraction: float = 0.25
    checkpoint_selection_start_phase_epoch: int = 6
    use_stain_augmentation: bool = True
    validity_threshold_grid: tuple[float, ...] = (
        0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90
    )
    selection_metric: str = "macro_f1"
    extra: dict[str, Any] = field(default_factory=dict)


PFM_SPECS: dict[str, dict[str, Any]] = {
    "uni2_h": {
        "hf_model": "hf-hub:MahmoodLab/UNI2-h",
        "embedding_dim": 1536,
        "gated": True,
        "license": "CC-BY-NC-ND-4.0",
    },
}


def stage1_experiment_registry() -> dict[str, Stage1ModelConfig]:
    """Return the fixed A1_IFCRN_PP Stage-1 configuration."""
    return {
        "A1_IFCRN_PP": Stage1ModelConfig(
            name="A1_IFCRN_PP",
            learning_rate=3e-4,
            weight_decay=1e-4,
            fixed_sigma=2.5,
        ),
    }



def stage2_experiment_registry() -> dict[str, Stage2ModelConfig]:
    """Return the V13.2 Stage-2 experiment registry."""
    from puma.stage2.catalog import stage2_experiment_registry as build_registry

    return build_registry()


def stage2_experiment_groups() -> dict[str, tuple[str, ...]]:
    """Compatibility facade for active Version-13.2 experiment groups."""
    from puma.stage2.catalog import stage2_experiment_groups as build_groups

    return build_groups()


@dataclass(slots=True)
class RuntimeConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def as_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(v) for v in value]
            return value

        return convert(asdict(self))


def validate_folds(folds: Iterable[int], number_of_folds: int = 5) -> tuple[int, ...]:
    resolved = tuple(sorted(set(int(f) for f in folds)))
    if not resolved:
        raise ValueError("At least one fold must be selected.")
    bad = [f for f in resolved if f < 0 or f >= number_of_folds]
    if bad:
        raise ValueError(f"Fold indices out of range [0,{number_of_folds - 1}]: {bad}")
    return resolved


def select_inner_fold(outer_fold: int, candidate_folds: Iterable[int]) -> int:
    """Choose a deterministic grouped inner-validation fold distinct from the outer fold.

    The next available fold in cyclic order is used. This keeps threshold selection, early
    stopping, and model promotion separate from the outer evaluation fold.
    """
    available = tuple(sorted(set(int(value) for value in candidate_folds)))
    if outer_fold not in available:
        raise ValueError(f"Outer fold {outer_fold} is not present in candidate_folds={available}.")
    alternatives = [value for value in available if value != outer_fold]
    if not alternatives:
        raise ValueError("Nested validation requires at least two distinct folds.")
    larger = [value for value in alternatives if value > outer_fold]
    return int(larger[0] if larger else alternatives[0])
