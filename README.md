# SymbioPan — PUMA Track 2 Panoptic Segmentation

Two-stage panoptic segmentation pipeline for the [PUMA Grand Challenge](https://puma.grand-challenge.org/) (Track 2: 10-class nuclei + 5-class tissue), with a strong emphasis on **rare-class performance** via specialized sampling, loss scheduling, and anatomical priors.

## Quick Start

```bash
pip install -e ".[dev]"
```

### Notebook (Colab / Jupyter)

Open [`notebooks/train_model.ipynb`](notebooks/train_model.ipynb) for a fully interactive pipeline with:
- Google Drive mount + git clone + dependency install
- Automatic GPU VRAM detection with batch-size and epoch schedule scaling for 4-hour budget (A6000 96GB → batch 12/24, 30+20 epochs)
- Loss schedule visualization, per-class Dice/IoU plots, S1 vs S2 comparison
- Data leakage prevention verification (group-based split, `val_original_only`)
- Dry-run / full-run Stage 1 + Stage 2 training with checkpoint verification

```bash
jupyter notebook notebooks/train_model.ipynb
```

### Preprocess
```bash
# Expects: Dataset/01_training_dataset_tif_ROIs/*.tif
#          Dataset/01_training_dataset_geojson_tissue/*_tissue.geojson
#          Dataset/01_training_dataset_geojson_nuclei/*_nuclei.geojson
python scripts/run_preprocess.py
```

Configurable via `PreprocessConfig` in `configs/defaults.py`: tile size, rare-crop generation, Cellpose model type (default `cyto3`).

### Train Stage 1
```bash
python scripts/run_stage1.py
# Optional overrides:
#   --epochs 50 --lr 1e-4 --batch-size 12 --val-ratio 0.2 --resume checkpoints/puma_epoch_last_s1.pth
```

Saves `checkpoints/puma_epoch_best_s1.pth` (best selection score) and `puma_epoch_last_s1.pth`.

### Train Stage 2
```bash
python scripts/run_stage2.py
# Optional overrides:
#   --epochs 30 --lr 1e-4 --batch-size 16 --val-ratio 0.2
```

Requires `checkpoints/puma_epoch_best_s1.pth` from Stage 1.
Saves `checkpoints/nuclei_refiner_residual_best.pth` and `nuclei_refiner_residual_last.pth`.

### Inference
```bash
python scripts/run_inference.py \
  --input <tif_dir> --output <out_dir> \
  --cp checkpoints/best_model.pth \
  [--stage2-cp checkpoints/nuclei_refiner_residual_best.pth] \
  [--site-type primary|metastatic] \
  [--cellpose-mode auto|generate|zero] \
  [--tile-size 1024] [--overlap 256] \
  [--np-threshold 0.50]
```

### Docker Inference
```bash
make docker-build
make docker-run
```

### Run Tests
```bash
python -m pytest tests/
```

## Project Structure

```
SymbioPan/
├── COLAB_TRAINING_GUIDE.md      # 4-hour Colab Pro training guide
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project metadata + tooling config
├── .gitignore
├── Dockerfile                   # Multi-stage Docker build for Grand Challenge
├── Makefile                     # Common targets
├── inference.sh                 # Docker entrypoint
│
├── configs/                    # Centralized configuration (dataclass-based)
│   ├── __init__.py             # Re-exports all config classes
│   ├── defaults.py             # All hyperparameters in one place
│   └── serialization.py        # Config serialization utilities
│
├── data/                       # Data layer
│   ├── __init__.py             # Re-exports constants
│   ├── constants.py            # Label mappings, class IDs, normalization constants
│   ├── dataset/                # Dataset + augmentations + sampling
│   │   ├── __init__.py
│   │   ├── puma_dataset.py     # PUMADataset (PyTorch Dataset)
│   │   ├── transforms.py       # Albumentations (vector-safe flips/rotations)
│   │   └── sampling.py         # Rare-weighted sample weight computation
│   └── preprocessing/          # Preprocessing pipeline
│       ├── __init__.py
│       ├── geojson_parser.py   # GeoJSON → rasterized masks
│       ├── flow_generator.py   # Cellpose flow + HV map computation
│       └── preprocess.py       # Orchestrator: TIFF → .npy files
│
├── models/                     # Neural network architectures
│   ├── __init__.py             # Re-exports all model classes
│   ├── encoder.py              # Frozen UNI ViT-L/16 + SpatialInjector bridges
│   ├── backbone.py             # ConvNeXt-Atto backbone
│   ├── fpn_aggregator.py       # 5-level Feature Pyramid Network
│   ├── decoders.py             # ParallelDecoders (ASPP tissue + HoVerNeXt nuclei)
│   ├── cross_attention.py      # SpatialInjector (cross-attention bridge)
│   ├── panoptic_net.py         # UnifiedPanopticNet (main Stage 1 model)
│   └── stage2_refiner.py       # ResidualNucleiRefinerUNet (Stage 2 refiner)
│
├── training/                   # Training pipeline
│   ├── __init__.py             # Re-exports all training functions
│   ├── train_loop.py           # Shared train_one_epoch / validate
│   ├── checkpoint.py           # Safe save/load/extract checkpoint utilities
│   ├── gpu_setup.py            # GPU detection, config overrides, bf16 patching
│   ├── logging_utils.py        # Structured logging
│   ├── cli.py                  # CLI argument parsing
│   ├── stage1_trainer.py       # Stage 1 training (UnifiedPanopticNet)
│   └── stage2_trainer.py       # Stage 2 training (ResidualNucleiRefinerUNet)
│
├── inference/                  # Inference pipeline (broken into modules)
│   ├── __init__.py             # Re-exports inference main
│   ├── model_loader.py         # Stage 1 + Stage 2 model loading
│   ├── tiling.py               # WSI tiling, padding, normalization
│   ├── site_classifier.py      # Primary vs. metastatic site classifier
│   ├── cellpose_flow.py        # Re-exports unified CellposeFlowGenerator
│   ├── postprocessing.py       # HV-instance segmentation + polygons
│   └── infer_wsi.py            # Orchestrator
│
├── utils/                      # Shared utilities
│   ├── __init__.py             # Re-exports losses, metrics, priors, SC-DFA
│   ├── losses.py               # MultiTaskUncertaintyLoss, FocalTverskyLoss, etc.
│   ├── metrics.py              # Dice/IoU accumulators, rare-focused scoring
│   ├── priors.py               # SpatialLogitAdjuster (site-type prior)
│   ├── sc_dfa.py               # SC-DFA: semantic class-dependent feature alignment
│   ├── split_utils.py          # Leakage-safe group-based train/val split
│   └── normalization.py        # Shared image normalization
│
├── scripts/                    # Thin entry-point scripts
│   ├── __init__.py
│   ├── run_preprocess.py
│   ├── run_stage1.py
│   ├── run_stage2.py
│   └── run_inference.py
│
├── notebooks/                  # Development notebooks
│   └── train_model.ipynb
│
├── tests/                      # Unit tests (pytest)
│   ├── __init__.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   └── test_models.py
│
├── checkpoints/                # Training output weights (gitignored, created at runtime)
├── dataset_processed/          # Preprocessed .npy files (gitignored)
├── output/                     # Docker inference output (gitignored)
└── test/                       # Docker test input (TIFF images)
```

## Pipeline Overview

### 1. Data Preprocessing
- Reads raw TIFF ROIs + GeoJSON annotations (tissue & nuclei).
- Rasterizes polygons → semantic/instance masks.
- Computes HoVer distance maps and Cellpose flow maps.
- Generates rare-centered augmented crops for rare classes.
- Writes `.npy` files + `sample_metadata.json`.

### 2. Stage 1 Training — UnifiedPanopticNet
- **Encoder**: Frozen UNI ViT-L/16 (MahmoodLab) + ConvNeXt-Atto (timm) with 4 SpatialInjector cross-attention bridges.
- **Neck**: FPNAggregator — 5-level FPN (P1–P5).
- **Decoders**: ParallelDecoders with ASPP tissue head, HoVer-NeXt-style NP/HV heads, NC head.
- **Auxiliary modules**: SC-DFA, SpatialLogitAdjuster, MutualFeatureExchange.
- **Rare-class focus**: WeightedRandomSampler, Focal Tversky loss with smooth ramp schedule.

### 3. Stage 2 Training — ResidualNucleiRefinerUNet
- 3-level U-Net predicting residual deltas for Stage 1 nuclei logits.
- Zero-initialized final conv → identity initialization.
- Knowledge-distillation preservation loss (masked KL divergence).
- Alpha/keep-lambda schedules for smooth refinement.

### 4. Inference
- Sliding-window tiling with configurable overlap.
- Optional site-type classifier (primary vs. metastatic).
- Optional Stage 2 residual refinement.
- HV watershed instance segmentation → polygon output.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No tissue background channel** | Model predicts 5 classes directly; background is treated as ignore index 255 |
| **SC-DFA** | Learns tissue→nuclei co-occurrence patterns via 5×10 weight matrix |
| **Spatial prior** | Site-type-specific log-prior matrices adjust nuclei logits based on tissue context |
| **Smooth loss schedules** | Focal Tversky, SC-DFA, and spatial prior ramped linearly to avoid training shocks |
| **Rare-class selection score** | 55% weight on rare-class Dice, forcing the model to prioritize minority classes |
| **Group-based split** | All rare crops stay with their source image; validation uses originals only |

## Network Architecture Decisions

### Merged Version 2.2 architecture + Version 4 data/classes
- Tissue model output stays **5 classes**. No trainable tissue background channel.
- Version 4 stored tissue masks are PUMA IDs: 0 background, 1 stroma, 2 blood vessel, 3 tumor, 4 epidermis, 5 necrosis.
- Dataset converts stored background 0 to 255 ignore index.
- Internal tissue order: 0 stroma, 1 blood vessel, 2 tumor, 3 epidermis, 4 necrosis.
- Inference maps internal tissue prediction 0..4 back to PUMA output 1..5.
- SC-DFA and spatial prior stay 5×10.

### Version 6: Ablation 1+2 with Smooth Stage 1
- **Ablation 1**: HoVer-NeXt-style NP/HV decoder heads.
- **Ablation 2**: ASPP tissue head.
- Stage 1 ramps FocalTversky, SC-DFA, and spatial prior smoothly.
- Schedule: FocalTversky epoch 10→16 (max 0.5), SC-DFA epoch 15→22 (max 0.3), Spatial prior epoch 20→28 (max 0.2).

### Cellpose in Inference
- Training data generated with `cyto3` model for better boundary detection.
- Inference uses `nuclei` model (faster, focuses on nuclear boundaries).
- The Stage 1 checkpoint stores the `cellpose_adapter` network inside the PyTorch model.
- Inference can generate fresh Cellpose flows (`--cellpose-mode generate`) or fall back to zero flow (`--cellpose-mode auto`).

## Output Formats

**Tissue mask**: GeoTIFF at `output/images/melanoma-tissue-mask-segmentation/<uuid>.tif`
- Values: 1 (stroma), 2 (blood vessel), 3 (tumor), 4 (epidermis), 5 (necrosis)

**Nuclei JSON**: at `output/melanoma-10-class-nuclei-segmentation.json`
- Multiple polygons with class labels, seed points, and probability scores.

## Google Colab (High-VRAM Setup)

`notebooks/train_model.ipynb` auto-detects available GPU VRAM and scales batch sizes and epoch schedules to fit **within 4 hours total** on Colab Pro (A100 80GB):

| VRAM | Stage 1 batch | Stage 1 epochs | Stage 2 batch | Stage 2 epochs |
|------|--------------|----------------|--------------|----------------|
| A6000 96GB+ (G4) | 12 | 30 | 24 | 20 |
| A100 80GB | 12 | 30 | 24 | 20 |
| A100 40GB | 8 | 30 | 16 | 20 |
| V100 32GB | 4 | 50 | 8 | 30 |
| 16GB | 2 | 50 | 4 | 30 |

Detection logic in `detect_gpu_setup()` overrides `Stage1Config` and `Stage2Config` defaults when ≥40 GB VRAM is detected, scaling milestones proportionally. For full details, see [`COLAB_TRAINING_GUIDE.md`](COLAB_TRAINING_GUIDE.md).

The notebook also:
- Enables **bfloat16** mixed precision on Ampere+ GPUs (more stable than float16)
- Enables **gradient checkpointing** for the UNI ViT encoder
- Enables **multi-GPU** DataParallel when multiple GPUs are detected
- Applies `$object.__setattr__` patches to frozen config dataclasses at runtime
- Saves **entity models** (`torch.save(model, path)`) — no separate state dicts needed

## Verify Data Leakage Prevention

The notebook includes a leakage-prevention verification cell:
```python
python -c "
import numpy as np
data = np.load('checkpoints/split_seed42.npz', allow_pickle=True)
train_src, val_src = set(data['train_sources']), set(data['val_sources'])
assert len(train_src & val_src) == 0, 'LEAKAGE DETECTED!'
print(f'OK: {len(train_src)} train / {len(val_src)} val groups, no overlap')
"
```

## Training Manifest

After training completes (via notebook Section 13b), `checkpoints/training_manifest.json` is generated:
```json
{
  "pipeline": "SymbioPan PUMA Track 2",
  "checkpoints": [...],
  "stage1_config": { "batch_size": 64, "epochs": 50, ... },
  "stage2_config": { "batch_size": 128, "epochs": 30, ... },
  "leakage_prevention": { "split": "group_based", "val_original_only": true }
}
```

## Configuration

All training/inference parameters are centralized in `configs/defaults.py` as `@dataclass(frozen=True)` classes:
- `Stage1Config` — stage 1 hyperparameters
- `Stage2Config` — stage 2 hyperparameters
- `PreprocessConfig` — preprocessing parameters
- `InferenceConfig` — inference parameters

## Inference CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `/input/images/melanoma-whole-slide-image` | TIFF input directory |
| `--output` | `/output` | Output directory |
| `--cp` | `checkpoints/best_model.pth` | Stage 1 panoptic checkpoint |
| `--stage2-cp` | `None` | Stage 2 residual refiner checkpoint |
| `--tile-size` | `1024` | Sliding window tile size |
| `--overlap` | `256` | Tile overlap in pixels |
| `--site-type` | auto-detect | `primary` \| `metastatic` override |
| `--site-classifier-cp` | `checkpoints/site_classifier_atto.pth` | Site classifier checkpoint |
| `--cellpose-mode` | `auto` | `auto` (fallback to zero) \| `generate` (fresh Cellpose) \| `zero` |
| `--np-threshold` | `0.50` | Nuclei probability threshold |
| `--min-nucleus-area` | `20` | Minimum nucleus area in pixels |

## Training CLI Reference

| Flag | Stage 1 (default) | Stage 2 (default) |
|------|-------------------|-------------------|
| `--epochs` | 50 (30 with 4-hr override) | 30 (20 with 4-hr override) |
| `--lr` | 1e-4 | 1e-4 |
| `--batch-size` | 12 | 16 |
| `--val-ratio` | 0.2 | 0.2 |
| `--resume` | `checkpoints/puma_epoch_last_s1.pth` | N/A |

## Docker Checkpoint Layout

For Stage 1 only:
```
checkpoints/best_model.pth
```

For Stage 1 + Stage 2:
```
checkpoints/best_model.pth
checkpoints/nuclei_refiner_residual_best.pth
```

`inference.sh` automatically uses Stage 2 only if `nuclei_refiner_residual_best.pth` exists.

## Version History

### v7.1 — 4-Hour Colab Pro Training + Code Quality
- `notebooks/train_model.ipynb` now 86 cells; `detect_gpu_setup()` auto-tunes for 4-hour budget (A100: 30+20 epochs, batch 64/128).
- **Entity-model saving** (`torch.save(model, path)`) — no separate state dicts for Docker loading.
- **Full codebase refactoring**: 27 files reformatted via `ruff format`, 0 `ruff check` violations.
- **117 Google-style docstrings** added across all modules.
- **`__all__` exports** on all `__init__.py` files.
- `COLAB_TRAINING_GUIDE.md` with time-budget analysis, troubleshooting FAQ.
- Fixed code-quality issues: mutable default args, shadow imports, trailing whitespace, dead code aliases.

### v7 — Training Notebook + Colab Setup
- Comprehensive `notebooks/train_model.ipynb` with 80 cells covering the full pipeline.
- Google Drive mount, git clone, dependency install, GPU VRAM auto-config.
- Loss schedule, per-class Dice/IoU, S1 vs S2 visualization (matplotlib).
- `SemanticMetricAccumulator` multi-batch accumulation demo.
- Data leakage prevention verification (group-based split, `val_original_only`).
- Stage 1 / Stage 2 actual training cells with checkpoint save/verify.
- `training_manifest.json` export after training completion.

### v6 — Smooth Stage 1 Ramp
- Ablation 1: HoVer-NeXt-style NP/HV decoder heads.
- Ablation 2: ASPP tissue head.
- Stage 1 ramps FocalTversky, SC-DFA, and spatial prior smoothly.
- Schedule: FocalTversky epoch 10→16 (max 0.5), SC-DFA epoch 15→22 (max 0.3), Spatial prior epoch 20→28 (max 0.2).
- Rare sampler is gentler: max sample weight 15.0, samples_per_epoch_multiplier 1.0.

### v5 — Rare-Focused
- Rare-centered augmented crops from preprocessing.
- Group-based split for leakage-safe train/val.
- Weighted random sampling with rare-class bonuses.
- Best checkpoint: `checkpoints/puma_epoch_best_s1.pth`.

### v2.2+4 — Merge
- Keeps Version 2.2 panoptic architecture with Version 4 preprocessing.
- Stage 2 input is 21 channels: 3 image + 5 tissue probs + 10 nuclei probs + 1 NP prob + 2 HV.

## References
- [PUMA Grand Challenge](https://puma.grand-challenge.org/)
- [UNI: A Pathology Foundation Model](https://github.com/mahmoodlab/UNI)
- [HoVer-Net](https://github.com/vqdang/hover_net)
- [Cellpose](https://cellpose.readthedocs.io/)
