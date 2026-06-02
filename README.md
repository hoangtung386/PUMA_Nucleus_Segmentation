# SymbioPan v9 — PUMA Track 2 Panoptic Segmentation

Single-stage panoptic segmentation pipeline for the [PUMA Grand Challenge](https://puma.grand-challenge.org/) (Track 2: 10-class nuclei + 5-class tissue), built on **Virchow2 ViT-H/14 + ConvNeXt-Tiny** with **context ROI encoding** and **test-time augmentation**.

**Key features:**
- Virchow2 ViT-H/14 encoder (fine-tune last 6 blocks) + ConvNeXt-Tiny backbone
- CellViT++ nuclei decoder + DeepLabV3+ tissue decoder
- Context ROI encoder (EfficientNet-B0, 5120→320)
- TTA (8 geometric augmentations with inverse averaging)
- Stain augmentation, warm-up + cosine decay LR
- Group-based leakage-safe data split

## Quick Start

```bash
pip install -e ".[dev]"
```

### Preprocess
```bash
# Expects data in Dataset/01_training_dataset_tif_ROIs/*.tif + GeoJSON annotations
python -m scripts.preprocess
# Or: symbiopan-preprocess
```

### Train
```bash
python -m scripts.train_stage1
# Or: symbiopan-train
```

### Inference
```bash
python -m scripts.infer_wsi --input <tif_dir> --output <out_dir> --cp checkpoints/best_model.pth
# Or: symbiopan-infer --input <tif_dir> --output <out_dir> --cp checkpoints/best_model.pth
```

### Docker
```bash
make docker-build && make docker-run
```

### Tests
```bash
python -m pytest tests/
```

## Project Structure

```
SymbioPan/
├── symbiopan/                       # Main package
│   ├── common/                      #   logging, device, types, exceptions
│   ├── data/                        #   constants, dataset, transforms, preprocessing
│   ├── models/                      #   encoder, backbone, decoders, FPN, panoptic_net
│   ├── inference/                   #   model_loader, tiling, postproc, TTA, infer_wsi
│   ├── training/                    #   train_loop, checkpoint, GPU setup, stage1 trainer
│   ├── losses/                      #   segmentation (CE/Tversky/Dice/BCE), multitask
│   ├── metrics/                     #   PUMAMetrics, SemanticMetricAccumulator
│   └── modules/                     #   SC-DFA, scheduler, split
├── configs/                         # Frozen dataclass configs (PathsConfig, Stage1Config, ...)
├── scripts/                         # CLI entry points
├── tests/                           # Pytest suite (28 tests)
├── notebooks/                       # Jupyter notebooks
├── docs/                            # Architecture, changelog, refactoring guide
├── Dataset/                         # Raw PUMA data (gitignored)
├── output/                          # Inference results (gitignored)
├── checkpoints/                     # Model weights (gitignored)
├── Dockerfile
├── Makefile
├── inference.sh
├── pyproject.toml
└── LICENSE
```

## Pipeline

### 1. Data Preprocessing
Reads raw TIFF ROIs + GeoJSON annotations → rasterizes masks → computes HoVer distance maps → generates rare-centered augmented crops → writes `.npy` files.

### 2. Training
- **Encoder**: Virchow2 ViT-H/14 (fine-tune last 6 blocks) + ConvNeXt-Tiny with 4 SpatialInjector bridges
- **Neck**: HierarchicalFPN — 5-level FPN (P1–P5) with multi-scale ViT features
- **Decoders**: DeepLabV3+ tissue head + CellViT++ nuclei decoder (NC/NP/HV)
- **Rare-class focus**: WeightedRandomSampler, Focal Tversky loss with smooth ramp, stain augmentation

### 3. Inference
Sliding-window tiling → optional site classifier → optional TTA → HV watershed instance segmentation → polygon GeoJSON output.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Background is class 0** | 6-class tissue_sem (0=background, 1–5=tissue types), background is **not** ignore_index 255 |
| **Virchow2 ViT-H/14** | SOTA foundation model for pathology (3.1M WSIs, 632M params) |
| **ConvNeXt-Tiny** | 7× more capacity than Atto for spatial features |
| **Fine-tune last 6 ViT blocks** | Adapts to dense prediction without catastrophic forgetting |
| **Context ROI** | 5× larger field-of-view for tissue type disambiguation |
| **TTA replaces Stage 2** | 8-aug averaging gives +2-3% without additional training |
| **Group-based split** | All rare crops stay with their source image; never leak between train/val |

## Configuration

All parameters in `configs/defaults.py` as `@dataclass(frozen=True)`:
- `PathsConfig` — data directory paths
- `PreprocessConfig` — preprocessing parameters (tile size, rare-crop generation)
- `Stage1Config` — training hyperparameters (LR, epochs, loss weights, augmentation)
- `InferenceConfig` — inference parameters (tile size, overlap, thresholds)

## Version History

See [docs/CHANGELOG.md](docs/CHANGELOG.md) for full details.

### v9 (current)
- **Major refactor**: restructured into `symbiopan/` package, removed all dead code, fixed inversion of dependency, added CHANGELOG + conftest, updated all tests.

### v8 (previous)
- Virchow2 ViT-H/14 + ConvNeXt-Tiny, CellViT++/DeepLabV3+ decoders, TTA, stain augmentation, context ROI.

### v7 (older)
- Frozen UNI ViT-L/16, ConvNeXt-Atto, Cellpose flow, Stage 2 refinement, cosine annealing.

## References

- [PUMA Grand Challenge](https://puma.grand-challenge.org/)
- [Virchow2](https://huggingface.co/paige-ai/Virchow2) (Paige AI)
- [CellViT++](https://github.com/TIO-IKIM/CellViT)
- [HoVer-Net](https://github.com/vqdang/hover_net)
- [DeepLabV3+](https://arxiv.org/abs/1802.02611)
