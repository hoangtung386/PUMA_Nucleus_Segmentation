# SymbioPan v8 "CellPath" — PUMA Track 2 Panoptic Segmentation

Single-stage panoptic segmentation pipeline for the [PUMA Grand Challenge](https://puma.grand-challenge.org/) (Track 2: 10-class nuclei + 5-class tissue), built on **Virchow2 ViT-H/14 + ConvNeXt-Tiny** with **FiLM site conditioning**, **context ROI encoding**, and **test-time augmentation**.

**Architecture highlights vs. v7:**
- Virchow2 ViT-H/14 (fine-tune last 6 blocks) — replaces frozen UNI ViT-L/16
- ConvNeXt-Tiny (28.6M params) — replaces ConvNeXt-Atto (3.7M params)
- CellViT++ nuclei decoder + DeepLabV3+ tissue decoder — replaces plain ASPP
- FiLM conditioning (9-site) — replaces SpatialLogitAdjuster (2-class prior)
- Context ROI encoder (EfficientNet-B0, 5120×5120→320×320) — new
- TTA (8 augmentations) — replaces Stage 2 refiner
- Stain augmentation (HEStain) + Mixup/CutMix — new training improvements
- Warm-up + cosine decay LR schedule

## Quick Start

```bash
pip install -e ".[dev]"
```

### Preprocess
```bash
# Expects: Dataset/01_training_dataset_tif_ROIs/*.tif
#          Dataset/01_training_dataset_geojson_tissue/*_tissue.geojson
#          Dataset/01_training_dataset_geojson_nuclei/*_nuclei.geojson
#          Dataset/01_training_dataset_tif_context_ROIs/*_context.tif (optional)
python -m data.preprocessing.preprocess
```

Configurable via `PreprocessConfig` in `configs/defaults.py`: tile size, rare-crop generation.

### Train
```bash
python scripts/run_stage1.py
# Optional overrides:
#   --epochs 50 --lr 1e-4 --batch-size 8 --val-ratio 0.2
#   --use-context-encoder --use-stain-aug
#   --resume checkpoints/puma_epoch_last_s1.pth
```

Saves `checkpoints/puma_epoch_best_s1.pth` and `puma_epoch_last_s1.pth`.

### Inference
```bash
python scripts/run_inference.py \
  --input <tif_dir> --output <out_dir> \
  --cp checkpoints/best_model.pth \
  [--site-type primary|metastatic] \
  [--use-tta] \
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
├── README.md
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── Makefile
├── inference.sh
├── Development_Orientation.md    # Architecture design document
├── SYMBIOV8_REFACTOR_PLAN.md     # Refactoring plan
│
├── configs/
│   ├── __init__.py
│   ├── defaults.py               # Stage1Config, PreprocessConfig, InferenceConfig
│   └── serialization.py
│
├── data/
│   ├── __init__.py
│   ├── constants.py               # Label mappings, rare-class IDs, normalization
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── puma_dataset.py        # PUMADataset with context ROI + 9-class site
│   │   ├── transforms.py          # Albumentations (vector-safe, stain aug)
│   │   └── sampling.py            # Rare-weighted sample weights
│   └── preprocessing/
│       ├── __init__.py
│       ├── geojson_parser.py
│       ├── flow_generator.py      # HV map computation only
│       └── preprocess.py
│
├── models/
│   ├── __init__.py
│   ├── encoder.py                 # Virchow2 ViT-H/14 + fine-tune + multi-block features
│   ├── backbone.py                # ConvNeXt-Tiny
│   ├── fpn_aggregator.py          # HierarchicalFPN (multi-scale ViT + CNN)
│   ├── decoders.py                # DeepLabV3+ tissue + CellViT++ nuclei + BoundaryAttn
│   ├── cross_attention.py         # SpatialInjector
│   ├── panoptic_net.py            # UnifiedPanopticNet (main model)
│   ├── stage2_refiner.py          # Deprecated — kept for reference
│   └── components/
│       ├── __init__.py
│       ├── boundary_attention.py
│       ├── context_encoder.py     # EfficientNet-B0 for context ROIs
│       ├── context_fusion.py      # FiLM-style context conditioning
│       ├── film_conditioning.py   # FiLM site conditioning (9-class)
│       └── register_tokens.py     # DINOv2-style register tokens
│
├── training/
│   ├── __init__.py
│   ├── train_loop.py
│   ├── checkpoint.py
│   ├── gpu_setup.py
│   ├── logging_utils.py
│   ├── cli.py
│   └── stage1_trainer.py
│
├── inference/
│   ├── __init__.py
│   ├── infer_wsi.py               # TTA (8 augs), no Stage 2
│   ├── model_loader.py
│   ├── tiling.py
│   ├── site_classifier.py         # 9-class site classifier
│   └── postprocessing.py
│
├── utils/
│   ├── __init__.py
│   ├── losses.py                  # MultiTaskUncertaintyLoss + boundary-aware
│   ├── metrics.py
│   ├── priors.py                  # Deprecated — use film_conditioning
│   ├── sc_dfa.py
│   ├── split_utils.py
│   ├── normalization.py
│   ├── scheduler_utils.py         # Warm-up + cosine decay
│   └── mixup_cutmix.py            # Mixup/CutMix augmentation
│
├── scripts/
│   ├── run_preprocess.py
│   ├── run_stage1.py
│   └── run_inference.py
│
├── notebooks/
│   └── train_model.ipynb
│
├── tests/
│   ├── test_losses.py
│   ├── test_metrics.py
│   └── test_models.py
│
├── checkpoints/                   # Training output (gitignored)
├── dataset_processed/             # Preprocessed .npy files (gitignored)
└── output/                        # Inference output (gitignored)
```

## Pipeline Overview

### 1. Data Preprocessing
- Reads raw TIFF ROIs + GeoJSON annotations
- Rasterizes polygons → semantic/instance masks
- Computes HoVer distance maps (no Cellpose)
- Generates rare-centered augmented crops
- Writes `.npy` files + `sample_metadata.json`

### 2. Training — UnifiedPanopticNet (single stage)
- **Encoder**: Virchow2 ViT-H/14 (fine-tune last 6 blocks) + ConvNeXt-Tiny with 4 SpatialInjector bridges
- **Neck**: HierarchicalFPN — 5-level FPN (P1–P5) with multi-scale ViT features
- **Decoders**: DeepLabV3+ tissue head + CellViT++ nuclei decoder (NC/NP/HV) + BoundaryAttentionModule
- **Site conditioning**: FiLM conditioning (9-site: primary + 8 metastatic sites)
- **Context ROI**: Optional EfficientNet-B0 encoder for 5120×5120 context
- **Rare-class focus**: WeightedRandomSampler, Focal Tversky loss with smooth ramp, stain aug, Mixup/CutMix

### 3. Inference
- Sliding-window tiling with configurable overlap
- Optional site-type classifier (9-class)
- Optional TTA (8 geometric augmentations with inverse transform + averaging)
- HV watershed instance segmentation → polygon output

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No tissue background channel** | 5 classes directly; background is ignore index 255 |
| **Virchow2 ViT-H/14** | SOTA foundation model for pathology (3.1M WSIs, 632M params) — replaces UNI |
| **ConvNeXt-Tiny** | 7× more capacity than Atto for spatial features |
| **Fine-tune last 6 ViT blocks** | Adapts to dense prediction without catastrophic forgetting |
| **FiLM conditioning (9-site)** | Learns per-site feature modulation — replaces SpatialLogitAdjuster |
| **Context ROI** | 5× larger field-of-view for tissue type disambiguation |
| **TTA replaces Stage 2** | 8-aug averaging gives +2-3% without additional training |
| **SC-DFA** | Learns tissue→nuclei co-occurrence patterns via 5×10 weight matrix |
| **Smooth loss schedules** | Focal Tversky + SC-DFA ramped linearly to avoid training shocks |
| **Group-based split** | All rare crops stay with their source image; validation uses originals only |

## Configuration

All parameters in `configs/defaults.py` as `@dataclass(frozen=True)`:
- `Stage1Config` — training hyperparameters
- `PreprocessConfig` — preprocessing parameters
- `InferenceConfig` — inference parameters

## Inference CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `/input/images/...` | TIFF input directory |
| `--output` | `/output` | Output directory |
| `--cp` | `checkpoints/best_model.pth` | Panoptic checkpoint |
| `--tile-size` | `1024` | Sliding window tile size |
| `--overlap` | `256` | Tile overlap in pixels |
| `--site-type` | auto-detect | `primary` \| `metastatic` override |
| `--site-classifier-cp` | `checkpoints/site_classifier_atto.pth` | Site classifier |
| `--use-tta` | `False` | Enable 8-augmentation test-time augmentation |
| `--np-threshold` | `0.50` | Nuclei probability threshold |
| `--min-nucleus-area` | `20` | Minimum nucleus area in pixels |

## VRAM Requirements (RTX 3080 10GB)

| Setting | Batch Size | Notes |
|---------|-----------|-------|
| Virchow2 frozen (inference) | 1-2 | FP16 autocast |
| Virchow2 fine-tune (6 blocks) | 1 | FP16 + grad checkpointing |
| No Virchow2 (ConvNeXt-Tiny only) | 8-16 | Full training |

## Version History

### v8 "CellPath" — Current
- Virchow2 ViT-H/14 encoder with fine-tuning
- ConvNeXt-Tiny backbone
- DeepLabV3+ tissue decoder + CellViT++ nuclei decoder
- FiLM 9-site conditioning
- Context ROI encoder (EfficientNet-B0)
- Stain augmentation (HEStain)
- Warm-up + cosine decay LR
- TTA (8 augmentations)
- Full codebase cleanup: no Cellpose, no Stage 2, all lint clean

### v7 — Previous
- Frozen UNI ViT-L/16 encoder
- ConvNeXt-Atto backbone
- Cellpose flow generation + Stage 2 refinement
- SpatialLogitAdjuster (2-class prior)
- Cosine annealing LR (no warm-up)

## References
- [PUMA Grand Challenge](https://puma.grand-challenge.org/)
- [Virchow2](https://huggingface.co/paige-ai/Virchow2) (Paige AI)
- [CellViT++](https://github.com/TIO-IKIM/CellViT)
- [HoVer-Net](https://github.com/vqdang/hover_net)
- [DeepLabV3+](https://arxiv.org/abs/1802.02611)
