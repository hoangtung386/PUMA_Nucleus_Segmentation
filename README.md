# SymbioPan — PUMA Track 2 Panoptic Segmentation

Two-stage panoptic segmentation pipeline for the [PUMA Grand Challenge](https://puma.grand-challenge.org/) (Track 2: 10-class nuclei + 5-class tissue), with a strong emphasis on **rare-class performance** via specialized sampling, loss scheduling, and anatomical priors.

## Quick Start

```bash
pip install -e ".[dev]"
```

### Preprocess
```bash
# Expects: Dataset/01_training_dataset_tif_ROIs/*.tif
#          Dataset/01_training_dataset_geojson_tissue/*_tissue.geojson
#          Dataset/01_training_dataset_geojson_nuclei/*_nuclei.geojson
python scripts/run_preprocess.py
```

### Train Stage 1
```bash
python scripts/run_stage1.py
```

### Train Stage 2
```bash
python scripts/run_stage2.py
```

### Inference
```bash
python scripts/run_inference.py --input <tif_dir> --output <out_dir> --cp <checkpoint>
```

### Docker Inference
```bash
make docker-build
make docker-run
```

## Project Structure

```
SymbioPan/
├── configs/                    # Centralized configuration (dataclass-based)
│   ├── __init__.py
│   ├── defaults.py             # All hyperparameters in one place
│   └── serialization.py        # Config serialization utilities
├── data/                       # Data layer
│   ├── constants.py            # Label mappings, class IDs, normalization constants
│   ├── dataset/                # Dataset + augmentations + sampling
│   │   ├── puma_dataset.py     # PUMADataset (PyTorch Dataset)
│   │   ├── transforms.py       # Albumentations (vector-safe flips/rotations)
│   │   └── sampling.py         # Rare-weighted sample weight computation
│   └── preprocessing/          # Preprocessing pipeline
│       ├── geojson_parser.py   # GeoJSON → rasterized masks
│       ├── flow_generator.py   # Cellpose flow + HV map computation
│       └── preprocess.py       # Orchestrator: TIFF → .npy files
├── models/                     # Neural network architectures
│   ├── encoder.py              # Frozen UNI ViT-L/16 + SpatialInjector bridges
│   ├── backbone.py             # ConvNeXt-Atto backbone
│   ├── fpn_aggregator.py       # 5-level Feature Pyramid Network
│   ├── decoders.py             # ParallelDecoders (ASPP tissue + HoVerNeXt nuclei)
│   ├── cross_attention.py      # SpatialInjector (cross-attention bridge)
│   ├── panoptic_net.py         # UnifiedPanopticNet (main Stage 1 model)
│   └── stage2_refiner.py       # ResidualNucleiRefinerUNet (Stage 2 refiner)
├── training/                   # Training pipeline
│   ├── train_loop.py           # Shared train_one_epoch / validate
│   ├── checkpoint.py           # Safe save/load/extract checkpoint utilities
│   ├── logging_utils.py        # Structured logging
│   ├── cli.py                  # CLI argument parsing
│   ├── stage1_trainer.py       # Stage 1 training (UnifiedPanopticNet)
│   └── stage2_trainer.py       # Stage 2 training (ResidualNucleiRefinerUNet)
├── inference/                  # Inference pipeline (broken into modules)
│   ├── model_loader.py         # Stage 1 + Stage 2 model loading
│   ├── tiling.py               # WSI tiling, padding, normalization
│   ├── site_classifier.py      # Primary vs. metastatic site classifier
│   ├── cellpose_flow.py        # Re-exports unified CellposeFlowGenerator
│   ├── postprocessing.py       # HV-instance segmentation + polygons
│   └── infer_wsi.py            # Orchestrator
├── utils/                      # Shared utilities
│   ├── losses.py               # MultiTaskUncertaintyLoss, FocalTverskyLoss, etc.
│   ├── metrics.py              # Dice/IoU accumulators, rare-focused scoring
│   ├── priors.py               # SpatialLogitAdjuster (site-type prior)
│   ├── sc_dfa.py               # SC-DFA: semantic class-dependent feature alignment
│   ├── split_utils.py          # Leakage-safe group-based train/val split
│   └── normalization.py        # Shared image normalization
├── scripts/                    # Thin entry-point scripts
│   ├── run_preprocess.py
│   ├── run_stage1.py
│   ├── run_stage2.py
│   └── run_inference.py
├── notebooks/                  # Development notebooks
├── checkpoints/                # Training output weights + Docker deployment weights
├── dataset_processed/          # Preprocessed .npy files
├── test/                       # Docker test input (TIFF images)
├── output/                     # Docker inference output
├── tests/                      # Unit tests
├── Dockerfile                  # Multi-stage Docker build for Grand Challenge
├── inference.sh                # Docker entrypoint
├── Makefile                    # Common targets
└── pyproject.toml              # Project metadata + tooling config
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

## Configuration

All training/inference parameters are centralized in `configs/defaults.py` as `@dataclass(frozen=True)` classes:
- `Stage1Config` — stage 1 hyperparameters
- `Stage2Config` — stage 2 hyperparameters
- `PreprocessConfig` — preprocessing parameters
- `InferenceConfig` — inference parameters

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

### v6 - Smooth Stage 1 Ramp
- Ablation 1: HoVer-NeXt-style NP/HV decoder heads.
- Ablation 2: ASPP tissue head.
- Stage 1 ramps FocalTversky, SC-DFA, and spatial prior smoothly.
- Schedule: FocalTversky epoch 10→16 (max 0.5), SC-DFA epoch 15→22 (max 0.3), Spatial prior epoch 20→28 (max 0.2).
- Rare sampler is gentler: max sample weight 15.0, samples_per_epoch_multiplier 1.0.

### v5 - Rare-Focused
- Rare-centered augmented crops from preprocessing.
- Group-based split for leakage-safe train/val.
- Weighted random sampling with rare-class bonuses.
- Best checkpoint: `checkpoints/puma_epoch_best_s1.pth`.

### v2.2+4 - Merge
- Keeps Version 2.2 panoptic architecture with Version 4 preprocessing.
- Stage 2 input is 21 channels: 3 image + 5 tissue probs + 10 nuclei probs + 1 NP prob + 2 HV.

## References
- [PUMA Grand Challenge](https://puma.grand-challenge.org/)
- [UNI: A Pathology Foundation Model](https://github.com/mahmoodlab/UNI)
- [HoVer-Net](https://github.com/vqdang/hover_net)
- [Cellpose](https://cellpose.readthedocs.io/)
