# PUMA Nucleus Segmentation

Nucleus instance segmentation and classification for the [PUMA dataset](https://puma.grand-challenge.org/) using CP4 (Cellpose-SAM) and ResNet-18.

## Project Structure

```
PUMA_Nucleus_Segmentation/
├── configs/                    # YAML hyperparameter configs
├── data/                       # Local dataset directory (gitignored)
│   ├── raw/                    # Raw images and GeoJSON annotations
│   ├── processed/              # Processed instance masks
│   └── splits/                 # Train/val/test split JSON
├── src/puma_seg/               # Installable package
│   ├── cli/                    # puma-train / puma-eval / puma-predict
│   ├── models/                 # CPTransformer, NucleusClassifier, losses
│   ├── data/                   # Dataset, GeoJSON parser, transforms
│   ├── training/                # Trainer classes and callbacks
│   ├── evaluation/              # Metrics
│   └── utils/                   # I/O and visualization
├── scripts/                     # Script wrappers
│   ├── prepare_data.py         # Convert GeoJSON to instance masks
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   ├── predict.py               # Inference script
│   └── challenge_inference.py   # Grand Challenge submission
├── tests/                       # Unit tests
├── notebooks/                   # EDA and demos
├── configs/                     # YAML configs
│   ├── baseline.yaml           # Segmentation-only config
│   └── multitask.yaml           # Segmentation + classification config
├── pyproject.toml
├── requirements.txt
└── Dockerfile
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or for development with CP4 support
pip install -e ".[dev,cp4]"
```

## Quick Start

```bash
# 1) Prepare data from GeoJSON annotations
python scripts/prepare_data.py --raw-dir data/raw --out-dir data/processed --track 1

# 2) Train segmentation + classification
python scripts/train.py --config configs/multitask.yaml
puma-train --config configs/multitask.yaml

# 3) Evaluate
python scripts/evaluate.py --config configs/multitask.yaml --split val
puma-eval --config configs/multitask.yaml --split val

# 4) Run inference
python scripts/predict.py --image path/to/image.png --output-dir outputs/predictions
puma-predict --image path/to/image.png --output-dir outputs/predictions
```

## Models

- **CPTransformer (CP4)**: SAM ViT-L based segmentation model, loads directly from `cpsam` checkpoint
- **CellposeSegmentor**: Wrapper supporting both CP4 and legacy Cellpose models
- **NucleusClassifier**: ResNet-18 based classifier for nucleus type classification

### Supported Classes

- **Track 1** (3 classes): tumor, lymphocyte, other
- **Track 2** (10 classes): tumor, lymphocyte, plasma cell, histiocyte, melanophage, neutrophil, stroma, endothelium, epithelium, apoptosis

## Configuration

Edit YAML configs in `configs/` to customize:

- `baseline.yaml`: Segmentation only with CP4
- `multitask.yaml`: Segmentation + classification

## Development

```bash
# Run tests
pytest tests/ -v --tb=short --cov=src/puma_seg --cov-report=term-missing

# Lint
ruff check src/ scripts/ tests/
ruff format --check src/ scripts/ tests/
```

## Grand Challenge Submission

```bash
# Build Docker image
docker build -t puma-algorithm:latest .

# Run local test
docker run --rm \
  -v "$(pwd)/data/sample_input:/input" \
  -v "$(pwd)/outputs/challenge:/output" \
  puma-algorithm:latest

# Export
./save.sh puma-algorithm:latest puma-algorithm.tar.gz
```

Output format:
- `images/melanoma-cell-detection/<uuid>.json`
- `images/melanoma-tissue-mask-segmentation/<uuid>.tif`