"""Configuration serialization utilities."""

from typing import Any

from configs.defaults import Stage1Config
from data.constants import (
    INTERNAL_TISSUE_ID_TO_NAME,
    NUM_NUCLEI_CLASSES,
    NUM_TISSUE_CLASSES,
    PUMA_NUCLEI_ID_TO_NAME,
)


def make_inference_config_from_stage1(stage1_config: Stage1Config, core_model: Any) -> dict[str, Any]:
    """Convert Stage1Config + model state to inference-compatible dictionary."""
    return {
        "architecture": "merged_v22_architecture_v4_labels_no_tissue_background_rare_focused",
        "image_size": stage1_config.image_size,
        "tile_size": stage1_config.image_size,
        "stride": stage1_config.stride,
        "num_tissue_classes": NUM_TISSUE_CLASSES,
        "num_nuclei_classes": NUM_NUCLEI_CLASSES,
        "use_sc_dfa": bool(core_model.use_sc_dfa),
        "lambda_sc_dfa": float(getattr(core_model, "lambda_sc_dfa", 0.0)),
        "lambda_prior": float(core_model.lambda_prior),
        "default_site_type": stage1_config.default_site_type,
        "internal_tissue_id_to_name": INTERNAL_TISSUE_ID_TO_NAME,
        "puma_nuclei_id_to_name": PUMA_NUCLEI_ID_TO_NAME,
        "tissue_internal_to_puma_rule": "puma_id = internal_id + 1; no model background channel",
        "normalization_mean": list(stage1_config.normalization_mean),
        "normalization_std": list(stage1_config.normalization_std),
        "cellpose_mode_at_training": "real_cellpose_flows_from_dataset_processed_cellpose_flows",
        "zero_cellpose_prob": stage1_config.zero_cellpose_prob,
        "rare_focused_training": True,
        "batch_size": stage1_config.batch_size,
        "epochs": stage1_config.epochs,
        "tissue_class_weights": list(stage1_config.tissue_class_weights),
        "nuclei_class_weights": list(stage1_config.nuclei_class_weights),
        "samples_per_epoch_multiplier": stage1_config.samples_per_epoch_multiplier,
        "max_sample_weight": stage1_config.max_sample_weight,
        "smooth_stage1_schedule": {
            "focal_start_epoch": stage1_config.focal_start_epoch,
            "focal_full_epoch": stage1_config.focal_full_epoch,
            "focal_max_weight": stage1_config.focal_max_weight,
            "sc_dfa_start_epoch": stage1_config.sc_dfa_start_epoch,
            "sc_dfa_full_epoch": stage1_config.sc_dfa_full_epoch,
            "sc_dfa_max_weight": stage1_config.sc_dfa_max_weight,
            "prior_start_epoch": stage1_config.prior_start_epoch,
            "prior_full_epoch": stage1_config.prior_full_epoch,
            "prior_max_weight": stage1_config.prior_max_weight,
        },
        "stage1_checkpoint_name": "puma_epoch_best_s1.pth",
        "stage2_checkpoint_name": "nuclei_refiner_residual_best.pth",
        "split_is_group_based": True,
        "validation_original_only": stage1_config.val_original_only,
    }
