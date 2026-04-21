# PUMA Nucleus Segmentation

Nucleus instance segmentation and classification for the [PUMA dataset](https://puma.grand-challenge.org/) using CP4 (Cellpose-SAM) and ResNet-18.

## Tổng quan dự án

### PUMA là gì?

**PUMA (Pan‑Cancer)** là một bài toán segmentation trên Grand Challenge về phân loại tế bào ung thư melanoma từ ảnh histopathology (H&E).

Bài toán có **2 tracks**:
- **Track 1**: 3 lớp (tumor, lymphocyte, other)  
- **Track 2**: 10 lớp (tumor, lymphocyte, plasma cell, histiocyte, melanophage, neutrophil, stroma, endothelium, epithelium, apoptosis)

### Tại sao chúng ta làm dự án này?

1. **Ung thư da** là một trong những loại ung thư phổ biến nhất thế giới
2. **Việc đếm và phân loại tế bào** (cell counting & classification) là bước quan trọng trong chẩn đoán ung thư
3. Giải pháp tự động giúp:
   - Giảm thời gian phân tích (thay vì đếm thủ công hàng giờ)
   - Tăng độ chính xác và nhất quán
   - Hỗ trớ bác sĩ trong chẩn đoán và điều trị

### Contribution của chúng ta

Chúng tôi cung cấp một **pipeline hoàn chỉnh** bao gồm:

| Thành phần | Mô tả |
|-----------|-------|
| **CP4 (CPTransformer)** | Segmentation model dựa trên SAM ViT-L backbone, fine-tuned cho nucleus segmentation |
| **NucleusClassifier** | ResNet-18 classifier cho phân loại loại tế bào |
| **GeoJSON Parser** | Chuyển đổi annotation GeoJSON sang instance masks |
| **Metrics** | PUMA official metric (summed macro F1) |
| **CLI & Notebooks** | Dễ dàng train, evaluate, inference |

### Điểm khác biệt so với các giải pháp khác

- **Direct loading CP4**: Load CPSAM checkpoint trực tiếp, không cần full cellpose library
- **Combined pipeline**: Segmentation + Classification trong một workflow
- **Support 2 tracks**: Cả Track 1 (3 classes) và Track 2 (10 classes)
- **Albumentations augmentation**: Full augmentation pipeline cho histopathology images

---

## Kiến trúc hệ thống

```
PUMA_Nucleus_Segmentation/
├── src/puma_seg/               # Package chính
│   ├── models/
│   │   ├── cp_transformer.py   # CP4 (SAM ViT-L backbone)
│   │   ├── cellpose_wrapper.py # Wrapper cho CP4/Cellpose
│   │   ├── nucleus_classifier.py # ResNet-18 classifier
│   │   ├── losses.py           # Loss functions
│   │   └── cp4_dataset.py      # CP4 dataset & loss
│   ├── data/
│   │   ├── dataset.py         # PUMASegmentationDataset
│   │   ├── geojson_parser.py  # Parse GeoJSON annotations
│   │   └── transforms.py      # Albumentations augmentations
│   ├── training/
│   │   ├── trainer.py        # Training loops
│   │   └── callbacks.py      # EarlyStopping, ModelCheckpoint
│   ├── evaluation/
│   │   └── metrics.py        # PUMA metrics
│   ├── cli/                 # CLI entry points
│   └── utils/               # I/O & visualization
├── scripts/
│   ├── prepare_data.py      # Convert GeoJSON → masks
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── predict.py         # Inference script
├── configs/
│   ├── baseline.yaml      # Segmentation only
│   └── multitask.yaml    # Segmentation + Classification
├── notebooks/
│   └── Train_puma_colab.ipynb  # Colab training notebook
└── tests/
```

---

## Installation

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Hoặc với CP4 support
pip install -e ".[dev,cp4]"
```

---

## Quick Start

```bash
# 1) Chuẩn bị data từ GeoJSON annotations
python scripts/prepare_data.py --raw-dir data/raw --out-dir data/processed --track 1

# 2) Train segmentation + classification
python scripts/train.py --config configs/multitask.yaml

# 3) Evaluate
python scripts/evaluate.py --config configs/multitask.yaml --split val

# 4) Inference
python scripts/predict.py --image path/to/image.png --output-dir outputs/predictions
```

---

## Models

### CP4 (CPTransformer)
- **Architecture**: SAM ViT-L backbone với patch size 8x8
- **Parameters**: ~300M parameters
- **Input**: 512x512 RGB images
- **Output**: Instance segmentation masks

### NucleusClassifier
- **Architecture**: ResNet-18 backbone
- **Input**: 64x64 nucleus crops
- **Output**: Class logits (3 classes cho Track 1, 10 classes cho Track 2)

### Supported Classes

| Track | Classes |
|-------|--------|
| Track 1 (3 classes) | background, tumor, lymphocyte, other |
| Track 2 (10 classes) | background, tumor, lymphocyte, plasma cell, histiocyte, melanophage, neutrophil, stroma, endothelium, epithelium, apoptosis |

---

## Configuration

Edit YAML configs trong `configs/`:

```yaml
# baseline.yaml - Segmentation only
data:
  track: 1
  processed_dir: "data/processed"
segmentation:
  pretrained_model: "cpsam"
  n_epochs: 100
  batch_size: 40
  learning_rate: 1e-5

# multitask.yaml - Segmentation + Classification
data:
  track: 1
  crop_size: 64
segmentation:
  pretrained_model: "cpsam"
classification:
  freeze_backbone: true
  phase1_epochs: 20
  phase2_epochs: 80
```

---

## PUMA Metrics

Theo official PUMA metric:

1. **Instance Matching**: Match predicted nuclei với GT bằng centroid distance ≤ 15px (Hungarian matching)
2. **Detection Metrics**: Precision, Recall, F1
3. **Classification Metrics**: Per-class F1
4. **PUMA Score**: Sum of all per-class F1 scores (không phải average)

---

## Development

```bash
# Chạy tests
pytest tests/ -v --tb=short

# Lint
ruff check src/ scripts/ tests/
ruff format --check src/ scripts/ tests/
```

---

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

**Output format:**
- `images/melanoma-cell-detection/<uuid>.json`
- `images/melanoma-tissue-mask-segmentation/<uuid>.tif`

---

## References

- [PUMA Challenge](https://puma.grand-challenge.org/)
- [Cellpose](https://cellpose.readthedocs.io/)
- [Segment Anything Model (SAM)](https://segment-anything.com/)
- [CP4 Paper](https://arxiv.org/abs/xxxx)