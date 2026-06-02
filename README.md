# SymbioPan v9 — PUMA Track 2 Panoptic Segmentation

Single-stage panoptic segmentation pipeline for the [PUMA Grand Challenge](https://puma.grand-challenge.org/) (Track 2: 10-class nuclei + 5-class tissue), built on **Virchow2 ViT-H/14 + ConvNeXt-Tiny** with optional context ROI encoding and test-time augmentation.

**Key features**
- Virchow2 ViT-H/14 encoder (fine-tune last 6 blocks) + ConvNeXt-Tiny backbone
- CellViT++ nuclei decoder + DeepLabV3+ tissue decoder
- Context ROI encoder (EfficientNet-B0)
- TTA with 8 geometric augmentations and inverse averaging
- Stain augmentation, warm-up + cosine LR decay
- Group-based leakage-safe train/val split

## Quick start

```bash
pip install -e ".[dev]"
```

### Preprocess

Expects raw TIFF + GeoJSON annotations under `Dataset/01_training_dataset_tif_ROIs/` and `Dataset/01_training_dataset_geojson_{tissue,nuclei}/`.

```bash
python -m scripts.preprocess
# or
symbiopan-preprocess
```

### Train

```bash
python -m scripts.train_stage1
# or
symbiopan-train
```

### Inference

```bash
python -m scripts.infer_wsi \
    --input  /path/to/wsi-tiles \
    --output /path/to/results \
    --cp     checkpoints/best_model.pth
# or
symbiopan-infer --input … --output … --cp …
```

### Docker

```bash
make docker-build
make docker-run INPUT_DIR=test OUTPUT_DIR=output
```

### Tests & lint

```bash
make test
make lint
```

## Project structure

```
SymbioPan/
├── symbiopan/                    # Main package
│   ├── common/                   #   logging, device, types, exceptions
│   ├── data/                     #   constants, dataset, transforms, preprocessing
│   ├── models/                   #   encoder, backbone, decoders, FPN, panoptic_net
│   ├── inference/                #   model_loader, tiling, postproc, TTA, infer_wsi
│   ├── training/                 #   train_loop, checkpoint, GPU setup, stage1 trainer
│   ├── losses/                   #   segmentation losses + multi-task uncertainty
│   ├── metrics/                  #   panoptic Dice/IoU + selection score
│   └── modules/                  #   SC-DFA, scheduler, group split
├── configs/                      # Frozen dataclass configs (Paths/Stage1/Inference/...)
├── scripts/                      # CLI entry points (-m scripts.preprocess, ...)
├── tests/                        # Pytest suite (subpackages mirror source layout)
│   ├── test_data/
│   ├── test_inference/
│   ├── test_losses/
│   ├── test_metrics/
│   └── test_models/
├── notebooks/                    # Minimal quickstart + visualization notebook
├── docs/                         # Architecture, changelog, refactoring guide
├── .github/workflows/            # CI: lint, test, docker-build
├── Dataset/                      # Raw PUMA data (gitignored)
├── output/                       # Inference results (gitignored)
├── checkpoints/                  # Model weights (gitignored)
├── Dockerfile
├── Makefile
├── inference.sh
└── pyproject.toml
```

## Pipeline overview

### 1. Data preprocessing
TIFF ROIs + GeoJSON → rasterized masks → HoVer distance maps → rare-centered augmented crops → `.npy` files.

### 2. Training
- **Encoder**: Virchow2 ViT-H/14 (fine-tune last 6 blocks) + ConvNeXt-Tiny with 4 SpatialInjector bridges
- **Neck**: HierarchicalFPN (P1–P5) with multi-scale ViT injection
- **Decoders**: DeepLabV3+ tissue head + CellViT++ nuclei decoder (NC/NP/HV)
- **Rare-class focus**: `WeightedRandomSampler`, FocalTversky ramp, stain augmentation

### 3. Inference
Sliding-window tiling → optional site classifier → optional TTA → HV watershed instance segmentation → polygon GeoJSON output.

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| Background is class 0 | Tissue is 6-class (0=background, 1–5=tissue types). Background is **not** an ignore index. |
| Virchow2 ViT-H/14 | SOTA foundation model for pathology (3.1M WSIs). |
| ConvNeXt-Tiny | ~7× more capacity than Atto for spatial features. |
| Fine-tune last 6 ViT blocks | Adapts to dense prediction without catastrophic forgetting. |
| Context ROI | ~5× larger field-of-view for tissue disambiguation. |
| TTA replaces a Stage-2 model | 8-aug averaging yields +2–3% with no extra training. |
| Group-based split | All rare crops stay with their source image — no leakage. |

## Configuration

All parameters are `@dataclass(frozen=True)` in `configs/defaults.py`:
- `PathsConfig` — data / checkpoint / split-file paths
- `PreprocessConfig` — tile size, rare-crop generation parameters
- `Stage1Config` — training hyper-parameters (LR, epochs, loss weights, augmentation)
- `InferenceConfig` — tile size, overlap, thresholds, checkpoint paths (read from `SYMBIOPAN_*` env vars)

## Version history

See [docs/CHANGELOG.md](docs/CHANGELOG.md) for the full history.

### v9 (current)
Major refactor: restructured into `symbiopan/` package, removed all dead code, fixed inversion of dependency, added CI/CD, reorganised tests, fixed `nn.Embedding` bug.

### v8 "CellPath"
Virchow2 ViT-H/14 + ConvNeXt-Tiny, CellViT++/DeepLabV3+ decoders, TTA, stain augmentation, context ROI.

### v7
Frozen UNI ViT-L/16, ConvNeXt-Atto, Cellpose flow, Stage 2 refinement, cosine annealing.

## References
- [PUMA Grand Challenge](https://puma.grand-challenge.org/)
- [Virchow2](https://huggingface.co/paige-ai/Virchow2) (Paige AI)
- [CellViT++](https://github.com/TIO-IKIM/CellViT)
- [HoVer-Net](https://github.com/vqdang/hover_net)
- [DeepLabV3+](https://arxiv.org/abs/1802.02611)
