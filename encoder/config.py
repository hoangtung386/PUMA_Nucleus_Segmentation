from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Literal

ROOT = Path('/content/drive/MyDrive/Research/PUMA')

IMAGE_DIR = ROOT / 'Dataset/01_training_dataset_tif_ROIs'
TISSUE_GEOJSON_DIR = ROOT / 'Dataset/01_training_dataset_geojson_tissue'
NUCLEI_GEOJSON_DIR = ROOT / 'Dataset/01_training_dataset_geojson_nuclei'

PROCESSED_DIR = ROOT / 'dataset_processed_encoder'
OUTPUT_DIR = ROOT / 'checkpoints_encoder'
SPLIT_DIR = PROCESSED_DIR / 'splits_3fold_multilabel'

# Official PUMA Track 2 tissue output ids.
# Background stays class 0 because the submission tissue mask uses background=0.
TISSUE_CLASSES: Dict[str, int] = {
    'tissue_background': 0,
    'tissue_stroma': 1,
    'tissue_blood_vessel': 2,
    'tissue_tumor': 3,
    'tissue_epidermis': 4,
    'tissue_necrosis': 5,
}

# PUMA Track 2 nuclei foreground classes only.
# There is intentionally NO nuclei background class.
NUCLEI_CLASSES: Dict[str, int] = {
    'nuclei_tumor': 0,
    'nuclei_lymphocyte': 1,
    'nuclei_plasma_cell': 2,
    'nuclei_histiocyte': 3,
    'nuclei_melanophage': 4,
    'nuclei_neutrophil': 5,
    'nuclei_stroma': 6,
    'nuclei_epithelium': 7,
    'nuclei_endothelium': 8,
    'nuclei_apoptosis': 9,
}

CLASS_ALIASES: Dict[str, str] = {
    # tissue aliases
    'background': 'tissue_background',
    'white_background': 'tissue_background',
    'tissue_background': 'tissue_background',
    'tissue_white_background': 'tissue_background',
    'stroma': 'tissue_stroma',
    'tissue_stroma': 'tissue_stroma',
    'blood_vessel': 'tissue_blood_vessel',
    'blood vessel': 'tissue_blood_vessel',
    'tissue_blood_vessel': 'tissue_blood_vessel',
    'tumor': 'tissue_tumor',
    'tumour': 'tissue_tumor',
    'tissue_tumor': 'tissue_tumor',
    'tissue_tumour': 'tissue_tumor',
    'epithelium': 'tissue_epidermis',
    'epidermis': 'tissue_epidermis',
    'tissue_epithelium': 'tissue_epidermis',
    'tissue_epidermis': 'tissue_epidermis',
    'necrosis': 'tissue_necrosis',
    'necrotic': 'tissue_necrosis',
    'tissue_necrosis': 'tissue_necrosis',
    # nuclei aliases
    'nuclei_tumor': 'nuclei_tumor',
    'nucleus_tumor': 'nuclei_tumor',
    'tumor_cell': 'nuclei_tumor',
    'neoplastic': 'nuclei_tumor',
    'nuclei_lymphocyte': 'nuclei_lymphocyte',
    'lymphocyte': 'nuclei_lymphocyte',
    'nuclei_plasma_cell': 'nuclei_plasma_cell',
    'plasma_cell': 'nuclei_plasma_cell',
    'plasma cell': 'nuclei_plasma_cell',
    'nuclei_histiocyte': 'nuclei_histiocyte',
    'histiocyte': 'nuclei_histiocyte',
    'nuclei_melanophage': 'nuclei_melanophage',
    'melanophage': 'nuclei_melanophage',
    'nuclei_neutrophil': 'nuclei_neutrophil',
    'neutrophil': 'nuclei_neutrophil',
    'nuclei_stroma': 'nuclei_stroma',
    'stroma_cell': 'nuclei_stroma',
    'stromal_cell': 'nuclei_stroma',
    'stromal cell': 'nuclei_stroma',
    'connective': 'nuclei_stroma',
    'nuclei_epithelium': 'nuclei_epithelium',
    'epithelial': 'nuclei_epithelium',
    'epithelium_cell': 'nuclei_epithelium',
    'nuclei_endothelium': 'nuclei_endothelium',
    'endothelium': 'nuclei_endothelium',
    'endothelial': 'nuclei_endothelium',
    'nuclei_apoptosis': 'nuclei_apoptosis',
    'apoptosis': 'nuclei_apoptosis',
    'apoptotic': 'nuclei_apoptosis',
    'dead': 'nuclei_apoptosis',
}

IGNORE_INDEX = 255
TISSUE_BACKGROUND_ID = 0
EncoderKind = Literal['single', 'fusion']

@dataclass(frozen=True)
class TrainConfig:
    # Default avoids the sanity-check TypeError when making a temporary config.
    experiment_name: str = 'debug'
    encoder_kind: EncoderKind = 'single'
    encoder_name: str = 'convnextv2_atto.fcmae_ft_in1k'
    aux_encoder_name: str | None = None
    freeze_encoders: bool = True

    # LoRA is used only by E6/E7. The backbone weights stay frozen;
    # only LoRA adapter weights inside the encoder are trainable.
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05

    # PUMA ROI size and foundation-model tile size.
    image_size: int = 1024
    # Crop 1024 ROI into 256x256 tiles, then resize each foundation-model tile
    # to 224 before UNI2-H / Virchow2. This avoids positional-embedding errors.
    foundation_tile_size: int = 256
    foundation_model_input_size: int = 224
    foundation_tile_batch: int = 8

    # Three-fold multi-label cross validation.
    n_folds: int = 3
    fold_id: int = 0
    split_seed: int = 42

    batch_size: int = 4
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    samples_per_train_image: int = 1
    val_samples_per_image: int = 1
    amp: bool = True
    seed: int = 42
    pretrained: bool = True
    decoder_channels: int = 256
    head_channels: int = 128
    early_stop_patience: int = 8

ENCODER_RUNS: Dict[str, TrainConfig] = {
    'E1': TrainConfig(
        experiment_name='E1_UNIv2_frozen',
        encoder_kind='single',
        encoder_name='hf-hub:MahmoodLab/UNI2-h',
    ),
    'E2': TrainConfig(
        experiment_name='E2_ConvNeXtV2_atto_frozen',
        encoder_kind='single',
        encoder_name='convnextv2_atto.fcmae_ft_in1k',
    ),
    'E3': TrainConfig(
        experiment_name='E3_Virchow2_frozen',
        encoder_kind='single',
        encoder_name='hf-hub:paige-ai/Virchow2',
    ),
    'E4': TrainConfig(
        experiment_name='E4_ConvNeXtV2_atto_plus_UNIv2_frozen',
        encoder_kind='fusion',
        encoder_name='convnextv2_atto.fcmae_ft_in1k',
        aux_encoder_name='hf-hub:MahmoodLab/UNI2-h',
    ),
    'E5': TrainConfig(
        experiment_name='E5_ConvNeXtV2_atto_plus_Virchow2_frozen',
        encoder_kind='fusion',
        encoder_name='convnextv2_atto.fcmae_ft_in1k',
        aux_encoder_name='hf-hub:paige-ai/Virchow2',
    ),
    'E6': TrainConfig(
        experiment_name='E6_UNIv2_lora',
        encoder_kind='single',
        encoder_name='hf-hub:MahmoodLab/UNI2-h',
        use_lora=True,
    ),
    'E7': TrainConfig(
        experiment_name='E7_Virchow2_lora',
        encoder_kind='single',
        encoder_name='hf-hub:paige-ai/Virchow2',
        use_lora=True,
    ),
}

ENCODER_RUN_ALIASES = {
    'train_E1': 'E1',
    'train_E2': 'E2',
    'train_E3': 'E3',
    'train_E4': 'E4',
    'train_E5': 'E5',
    'train_E6': 'E6',
    'train_E7': 'E7',
    **{cfg.experiment_name: key for key, cfg in ENCODER_RUNS.items()},
}


def cfg_for_fold(cfg: TrainConfig, fold_id: int) -> TrainConfig:
    if fold_id < 0 or fold_id >= cfg.n_folds:
        raise ValueError(f'fold_id must be in [0, {cfg.n_folds - 1}], got {fold_id}')
    return replace(cfg, fold_id=fold_id)


def normalize_label_name(name: str | None) -> str | None:
    if name is None:
        return None
    s = str(name).strip().lower()
    s = s.replace('-', '_').replace('/', '_')
    s = ' '.join(s.split())
    s = s.replace(' ', '_')
    return CLASS_ALIASES.get(s, s)


def id_to_name(class_map: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in class_map.items()}
