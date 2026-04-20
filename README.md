# PUMA Nucleus Segmentation

Nucleus instance segmentation and classification for the [PUMA dataset](https://puma.grand-challenge.org/).

## Project Structure

```text
PUMA_Nucleus_Segmentation/
├── configs/                    # YAML hyperparameter configs
├── data/                       # Local dataset directory (gitignored)
│   ├── raw/
│   ├── processed/
│   └── splits/
├── models/                     # Saved checkpoints (gitignored)
├── notebooks/                  # EDA and demos
├── outputs/                    # Inference and evaluation outputs (gitignored)
├── scripts/                    # Script wrappers
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── src/puma_seg/               # Installable package
│   ├── cli/                    # puma-train / puma-eval / puma-predict
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── tests/
├── pyproject.toml
└── README.md
```

## Quick Start

```bash
# 1) Install dependencies
# Option A: install from requirements.txt
pip install -r requirements.txt

# Option B (recommended for development): install package extras
pip install -e ".[dev,cp4]"

# Optional legacy Cellpose support
pip install -e ".[cellpose]"

# 2) Prepare data
python scripts/prepare_data.py --raw-dir data/raw --out-dir data/processed --track 1

# 3) Train (script or CLI)
python scripts/train.py --config configs/baseline.yaml
puma-train --config configs/baseline.yaml

# 4) Evaluate
puma-eval --config configs/multitask.yaml --split val

# 5) Predict
puma-predict --image path/to/image.png --seg-model cyto3 --output-dir outputs/predictions
```

## Dependency Notes

- `requirements.txt` now contains a full dependency set for quick environment setup.
- `pyproject.toml` remains the source of truth for package metadata and extras.
- For editable local development, prefer `pip install -e ".[dev,cp4]"`.

## Development

```bash
ruff check src/ scripts/ tests/
ruff format --check src/ scripts/ tests/
pytest tests/ -v --tb=short --cov=src/puma_seg --cov-report=term-missing
```

## Grand Challenge Submission

This repository now includes a Grand Challenge-compatible entrypoint:

```bash
# Build
docker build -t puma-algorithm:latest .

# Run local smoke test
docker run --rm \
  -v "$(pwd)/data/sample_input:/input" \
  -v "$(pwd)/outputs/challenge:/output" \
  puma-algorithm:latest

# Export compressed container
./save.sh puma-algorithm:latest puma-algorithm.tar.gz
```

The runtime creates outputs under:

- `images/melanoma-cell-detection/<uuid>.json`
- `images/melanoma-tissue-mask-segmentation/<uuid>.tif`

Note: task-1 tissue output is currently a valid placeholder mask and should be
replaced by a trained tissue model before competitive submission.
