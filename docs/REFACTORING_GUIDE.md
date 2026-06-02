# Hướng Dẫn Refactor Dự Án `SymbioPan` v8 "CellPath"

> Tài liệu này mô tả toàn bộ kế hoạch refactor code, **tái tổ chức thư mục và file** cho dự án `SymbioPan` — pipeline panoptic segmentation cho PUMA Grand Challenge Track 2.
>
> - **Quy mô hiện tại**: ~4.130 dòng Python, 50 file (không tính `.git`).
> - **Mục tiêu**: code sạch, dependency một chiều, không dead code, không trùng lặp, cấu trúc thư mục phản ánh đúng trách nhiệm từng layer.
> - **Nguyên tắc**: refactor KHÔNG làm thay đổi hành vi bên ngoài; giữ nguyên interface công khai của model, dataset, CLI; tất cả test hiện có phải pass.

---

## Mục Lục

1. [Tóm Tắt Tình Trạng Hiện Tại](#1-tóm-tắt-tình-trạng-hiện-tại)
2. [Nguyên Tắc Kiến Trúc Mục Tiêu](#2-nguyên-tắc-kiến-trúc-mục-tiêu)
3. [Cấu Trúc Thư Mục Mục Tiêu](#3-cấu-trúc-thư-mục-mục-tiêu)
4. [Bản Đồ Di Chuyển File (File-by-File)](#4-bản-đồ-di-chuyển-file-file-by-file)
5. [Kế Hoạch Refactor Theo Giai Đoạn](#5-kế-hoạch-refactor-theo-giai-đoạn)
6. [Các Vấn Đề Code Cụ Thể Và Cách Sửa](#6-các-vấn-đề-code-cụ-thể-và-cách-sửa)
7. [Cải Thiện Testing](#7-cải-thiện-testing)
8. [Cập Nhật Tài Liệu Và Tiện Ích](#8-cập-nhật-tài-liệu-và-tiện-ích)
9. [Checklist Migration](#9-checklist-migration)
10. [Phụ Lục: Mapping Import Cũ → Mới](#10-phụ-lục-mapping-import-cũ--mới)

---

## 1. Tóm Tắt Tình Trạng Hiện Tại

### 1.1. Điểm mạnh

- Tổng thể đã tách hợp lý theo layer: `data/`, `models/`, `training/`, `inference/`, `utils/`, `configs/`.
- Config dùng `dataclass(frozen=True)` — bất biến, dễ test.
- Có tài liệu kiến trúc (`docs/architecture.md`).
- Đã có test cơ bản cho losses, metrics, models.

### 1.2. Các vấn đề nghiêm trọng (theo mức ưu tiên)

| # | Vấn đề | Mức độ | Ảnh hưởng |
|---|---|---|---|
| 1 | **Đảo ngược dependency**: `training/logging_utils` bị import bởi `data/`, `models/`, `inference/`, `utils/` | 🔴 Cao | Cấu trúc bị vỡ; bất kỳ test nào import `data` cũng kéo theo `torch` |
| 2 | **`_CONTEXT_CACHE` global mutable trong `puma_dataset.py`** — không bao giờ clear | 🔴 Cao | Memory leak nghiêm trọng trong training dài |
| 3 | **`cfg = STAGE1_DEFAULT_CONFIG` ở module-level** trong `stage1_trainer.py` và `preprocess.py` | 🔴 Cao | Mutable global, side effect tại import |
| 4 | **Hardcode `/opt/app/...` ở 4–5 nơi** (Dockerfile, `inference.sh`, `InferenceConfig`, `infer_wsi.py`) | 🔴 Cao | Không portable ngoài Docker |
| 5 | **Trùng lặp `SITE_NAMES`** giữa `data/constants.py` và `inference/site_classifier.py` | 🟡 TB | Drift khi cập nhật |
| 6 | **Trùng lặp `sample_weight_from_masks`** giữa `preprocess.py` và `sampling.py` | 🟡 TB | Hai nguồn sự thật, dễ drift |
| 7 | **Trùng lặp `INTERNAL_TISSUE_ID_TO_NAME` ↔ `PUMA_TISSUE_ID_TO_NAME`** | 🟡 TB | Vô nghĩa, gây nhầm lẫn |
| 8 | **Dead code**: `extract_intermediate_features`, `parse_stage1_args`, `get_stage1_main`, `make_or_load_group_split_with_test`, `get_train_transforms_stain_aug` wrapper, `BoundaryAttentionModule` (chưa dùng trong loss/inference) | 🟡 TB | Tăng cognitive load |
| 9 | **Bug logic**: `train_loop.py:53-55` dòng `if/else` không có hiệu lực | 🟡 TB | Comment nói "freeze ViT" nhưng code không làm gì |
| 10 | **Magic numbers trong code** chưa vào config: `_CONTEXT_ROI_SIZE=320`, `target_grid=64`, `alpha/beta/gamma` cho FocalTversky, weights 0.20/0.25/0.55 cho `selection_score`, `epochs=30` ở `gpu_setup.py` (khác 50 ở default) | 🟡 TB | Khó reproduce, dễ drift |
| 11 | **Hardcode `paige-ai/Virchow2` ở 3 file** (encoder, model_loader, stage1_trainer) | 🟡 TB | Phải sửa 3 chỗ khi đổi model |
| 12 | **`.pyc` cache của test files đã xoá** còn sót trong `tests/__pycache__/` | 🟢 Thấp | Gây confuse |
| 13 | **Mâu thuẫn README ↔ architecture.md** về background class (255 vs class 0) | 🟢 Thấp | Người mới confuse |
| 14 | **Test coverage mỏng**: không test `PUMADataset.__getitem__`, `_fix_vector_field`, `model_loader`, `tiling`, `postprocessing` | 🟡 TB | Regression dễ lọt |
| 15 | **Eager imports trong `__init__.py`** của `inference/`, `data/preprocessing/`, `training/` | 🟡 TB | Load nặng không cần thiết |
| 16 | **Hardcode notebook path** cho Google Drive + có thể chứa HF token | 🟢 Thấp | Rủi ro leak |

---

## 2. Nguyên Tắc Kiến Trúc Mục Tiêu

### 2.1. Dependency direction (một chiều, không vòng)

```
common (utils chung: logging, device, types, constants)
   ↑
data (constants, dataset, transforms, sampling, preprocessing)
   ↑
models (encoder, backbone, decoders, fpn, panoptic_net)
   ↑
inference (model_loader, postprocessing, tiling, site_classifier, infer_wsi)
   ↑
training (cli, checkpoint, gpu_setup, train_loop, stage1_trainer)
   ↑
scripts & notebooks (entry points)
```

**Không bao giờ**:
- `data/` import từ `models/`, `training/`, `inference/`.
- `models/` import từ `training/`, `inference/`.
- `inference/` import từ `training/` (ngoại trừ `checkpoint` nếu tách ra `common/`).
- `common/` import bất kỳ layer nào khác.

### 2.2. Cấu trúc module mới

| Layer mới | Trách nhiệm | Được phép import từ |
|---|---|---|
| `symbiopan/common/` | logging, device helpers, types, exceptions, path utils | (stdlib + bên thứ 3) |
| `symbiopan/data/` | constants, dataset, transforms, sampling, preprocessing | `common` |
| `symbiopan/models/` | backbone, encoder, decoders, fpn, panoptic_net | `common`, `data.constants` |
| `symbiopan/inference/` | model loader, postprocessing, tiling, site classifier, main pipeline | `common`, `data`, `models` |
| `symbiopan/training/` | train loops, checkpoint, gpu setup, cli, stage1 trainer | tất cả (vì là layer cao nhất) |
| `symbiopan/scripts/` | entry point mỏng (chỉ parse CLI + gọi `main`) | tất cả |
| `symbiopan/configs/` | dataclass config immutable | `common`, `data.constants` |

**Ghi chú**: Quy ước import sử dụng package name `symbiopan.*` thay vì relative import lung tung. Tên package chính là tên trong `pyproject.toml`.

### 2.3. Nguyên tắc code

1. **Không side effect ở module-level** (không `cfg = ...`, không `logger = setup_logger()` ở top, không global cache mutable).
2. **Không hardcode path** trong code — dùng config hoặc env var.
3. **Không trùng lặp** — một hằng số, một hàm chỉ tồn tại ở một nơi.
4. **Magic numbers phải có tên** và đặt trong config.
5. **`__init__.py` chỉ re-export API công khai**; lazy import cho submodule nặng.
6. **Type hints đầy đủ** cho mọi public function/class.
7. **Mỗi hàm ≤ 50 dòng** (trừ khi có lý do rõ ràng); refactor nếu dài hơn.

---

## 3. Cấu Trúc Thư Mục Mục Tiêu

```
SymbioPan/
├── .github/                          # CI/CD workflows (bổ sung, hiện đang rỗng)
│   └── workflows/
│       ├── lint.yml                  # ruff check
│       ├── test.yml                  # pytest
│       └── docker-build.yml          # build & push image
│
├── configs/                          # Dataclass config (bất biến)
│   ├── __init__.py                   # Re-export public API
│   ├── defaults.py                   # 4 dataclass: PathsConfig, PreprocessConfig,
│   │                                 #   Stage1Config, InferenceConfig
│   └── README.md                     # Giải thích từng field
│
├── symbiopan/                        # ★ PACKAGE CHÍNH (đổi tên từ code rải rác)
│   ├── __init__.py                   # Version, public API
│   ├── common/                       # ★ MỚI: utilities chung
│   │   ├── __init__.py
│   │   ├── logging.py                # setup_logger + get_logger (DI-friendly)
│   │   ├── device.py                 # get_device(), autocast_context()
│   │   ├── types.py                  # BatchDict, ModelOutputDict, v.v.
│   │   ├── exceptions.py             # SymbioPanError, CheckpointMismatchError, ...
│   │   └── path_utils.py             # resolve_path(), ensure_dir()
│   │
│   ├── data/                         # = data/ cũ (chuyển thành subpackage)
│   │   ├── __init__.py
│   │   ├── constants.py              # 1 nguồn sự thật duy nhất
│   │   ├── sampling.py               # compute_sample_weight (chuẩn hoá)
│   │   ├── dataset/
│   │   │   ├── __init__.py
│   │   │   ├── puma_dataset.py       # Bỏ _CONTEXT_CACHE global, dùng class attr
│   │   │   └── transforms.py         # Bỏ wrapper, gộp stain_aug flag vào factory
│   │   └── preprocessing/
│   │       ├── __init__.py
│   │       ├── flow_generator.py     # Vectorize bằng scipy
│   │       ├── geojson_parser.py
│   │       └── preprocess.py         # Bỏ cfg global, dùng tham số
│   │
│   ├── models/                       # = models/ cũ
│   │   ├── __init__.py
│   │   ├── backbone.py
│   │   ├── cross_attention.py
│   │   ├── encoder.py                # Bỏ extract_intermediate_features, _patch_proj
│   │   │                             #   move to __init__, bỏ _simple_patch_embed fallback
│   │   ├── fpn_aggregator.py         # Bỏ vit_intermediate unused param
│   │   ├── decoders.py
│   │   ├── panoptic_net.py           # Lấy num_sites từ config
│   │   └── components/
│   │       ├── __init__.py
│   │       ├── context_encoder.py
│   │       └── context_fusion.py
│   │   # XOÁ: boundary_attention.py (dead branch)
│   │
│   ├── inference/                    # = inference/ cũ
│   │   ├── __init__.py               # Bỏ eager import
│   │   ├── model_loader.py           # Lấy config từ tham số, không hardcode
│   │   ├── postprocessing.py
│   │   ├── tiling.py
│   │   ├── site_classifier.py        # Import SITE_NAMES từ data.constants
│   │   ├── tta.py                    # ★ TÁCH TTA_TRANSFORMS/INVERSE ra file riêng
│   │   └── infer_wsi.py
│   │
│   ├── training/                     # = training/ cũ (không còn logging_utils)
│   │   ├── __init__.py               # Bỏ get_stage1_main dead helper
│   │   ├── checkpoint.py             # safe_torch_save, extract_state_dict
│   │   ├── cli.py                    # parse_stage1_args, parse_inference_args
│   │   ├── gpu_setup.py              # detect_gpu_setup, cleanup_gpu_cache
│   │   ├── train_loop.py             # Sửa bug dòng 53-55
│   │   └── stage1_trainer.py         # Bỏ cfg global, dùng self.cfg
│   │
│   ├── losses/                       # ★ TÁCH từ utils/losses.py
│   │   ├── __init__.py
│   │   ├── segmentation.py           # SafeCE, FocalTversky, SoftDice, FocalBCE
│   │   └── multitask.py              # MultiTaskUncertaintyLoss
│   │
│   ├── metrics/                      # ★ TÁCH từ utils/metrics.py
│   │   ├── __init__.py
│   │   └── panoptic.py               # SemanticMetricAccumulator, PUMAMetrics
│   │
│   ├── modules/                      # ★ TÁCH từ utils/
│   │   ├── __init__.py
│   │   ├── sc_dfa.py                 # SC-DFA module
│   │   ├── scheduler.py              # build_warmup_cosine_scheduler + linear_ramp
│   │   └── split.py                  # make_or_load_group_split (xoá _with_test)
│   │
│   └── visualization/                # ★ MỚI (nếu cần)
│       ├── __init__.py
│       └── overlays.py               # plot_predictions (tách từ notebook)
│
├── scripts/                          # CLI entry points (giữ nguyên vị trí, refactor nội dung)
│   ├── __init__.py
│   ├── preprocess.py                 # python -m scripts.preprocess
│   ├── train_stage1.py               # python -m scripts.train_stage1
│   └── infer_wsi.py                  # python -m scripts.infer_wsi
│
├── tests/                            # Bổ sung test
│   ├── __init__.py
│   ├── conftest.py                   # ★ THÊM: fixtures chung (dummy_batch, tiny_model, ...)
│   ├── test_common/
│   │   ├── __init__.py
│   │   ├── test_logging.py
│   │   └── test_device.py
│   ├── test_data/
│   │   ├── __init__.py
│   │   ├── test_constants.py
│   │   ├── test_dataset.py           # Mở rộng: __getitem__, sampling
│   │   ├── test_geojson.py           # ★ THÊM
│   │   └── test_transforms.py        # Mở rộng: _fix_vector_field
│   ├── test_losses/
│   │   ├── __init__.py
│   │   ├── test_segmentation.py
│   │   └── test_multitask.py
│   ├── test_metrics/
│   │   ├── __init__.py
│   │   └── test_panoptic.py
│   ├── test_models/
│   │   ├── __init__.py
│   │   ├── test_encoder.py
│   │   ├── test_decoders.py
│   │   ├── test_fpn.py
│   │   └── test_panoptic_net.py      # ★ THÊM: end-to-end với tiny model
│   ├── test_inference/
│   │   ├── __init__.py
│   │   ├── test_tiling.py            # ★ THÊM
│   │   ├── test_postprocessing.py    # ★ THÊM
│   │   ├── test_site_classifier.py   # ★ THÊM
│   │   ├── test_model_loader.py      # ★ THÊM
│   │   └── test_infer_wsi.py         # Mở rộng
│   └── test_training/
│       ├── __init__.py
│       ├── test_checkpoint.py        # ★ THÊM
│       ├── test_splits.py            # ★ THÊM
│       └── test_cli.py               # ★ THÊM
│
├── notebooks/                        # Rút gọn còn ~300 dòng
│   ├── README.md                     # ★ THÊM: giải thích workflow
│   ├── 01_quickstart.ipynb           # ★ RÚT GỌN từ train_model.ipynb
│   └── 02_visualization.ipynb        # ★ TÁCH phần visualization
│
├── docs/
│   ├── architecture.md               # Cập nhật cho khớp với code mới
│   ├── REFACTORING_GUIDE.md          # File này
│   ├── CHANGELOG.md                  # ★ THÊM: lịch sử refactor
│   └── images/                       # ★ THÊM: diagram, screenshots
│
├── data/                             # ★ ĐỔI TÊN → Dataset/ cho khớp PATHS.raw_dir
│   └── .gitkeep
│   # XOÁ: data/raw/, data/processed/
│
├── Dataset/                          # ★ THÊM: dữ liệu PUMA thô (gitignored)
│   └── .gitkeep
│
├── output/                           # ★ ĐỔI TÊN từ outputs/ cho khớp gitignore
│   └── .gitkeep
│
├── checkpoints/                      # ★ THÊM: weights (gitignored)
│   └── .gitkeep
│
├── .gitignore                        # Bổ sung: dataset/, output/, checkpoints/, *.log
├── Dockerfile                        # Refactor: dùng ARG/ENV, copy từng layer
├── Makefile                          # Bổ sung: test, format, notebook targets
├── README.md                         # Cập nhật project structure mới
├── LICENSE
├── inference.sh                      # Refactor: dùng ENV var thay vì hardcode
├── pyproject.toml                    # Thêm [tool.setuptools.packages.find]
└── requirements-dev.txt              # ★ THÊM: dev deps (pytest, ruff, jupyter)
# XOÁ: requirements.txt (gộp vào pyproject.toml)
```

**Tổng số file thay đổi**:
- **XOÁ**: 7 file (dead code, trùng lặp)
- **DI CHUYỂN**: 11 file
- **TÁCH**: 4 file
- **THÊM MỚI**: 9 file (test + common + visualization)
- **SỬA NỘI DUNG**: 24 file

---

## 4. Bản Đồ Di Chuyển File (File-by-File)

### 4.1. Bảng mapping đầy đủ

| File hiện tại | Hành động | File mới | Ghi chú |
|---|---|---|---|
| **Root** | | | |
| `requirements.txt` | 🗑️ XOÁ | — | Trùng `pyproject.toml` |
| — | ➕ THÊM | `requirements-dev.txt` | Dev deps: pytest, ruff, jupyter, pre-commit |
| `inference.sh` | ✏️ SỬA | `inference.sh` | Dùng `ENV` thay vì hardcode |
| `test_run.sh` | 🗑️ XOÁ | — | Trùng `Makefile docker-build/run` |
| `Dockerfile` | ✏️ SỬA | `Dockerfile` | Dùng `ARG` cho paths |
| `Makefile` | ✏️ SỬA | `Makefile` | Thêm target `test`, `format`, `notebook` |
| `README.md` | ✏️ SỬA | `README.md` | Cập nhật project structure |
| `outputs/.gitkeep` | 📦 DI CHUYỂN | `output/.gitkeep` | Đổi tên cho khớp gitignore |
| — | ➕ THÊM | `checkpoints/.gitkeep` | Tách rõ output vs checkpoint |
| `data/raw/.gitkeep` | 📦 DI CHUYỂN | `Dataset/.gitkeep` | Khớp với `PATHS.raw_dir` |
| `data/processed/.gitkeep` | 🗑️ XOÁ | — | Không dùng tới |
| **`.github/`** | | | |
| `.github/` (rỗng) | ✏️ SỬA | `.github/workflows/{lint,test,docker-build}.yml` | CI/CD |
| **`configs/`** | | | |
| `configs/__init__.py` | ✏️ SỬA | `configs/__init__.py` | Bỏ import `torch` cho `get_device` |
| `configs/defaults.py` | ✏️ SỬA + TÁCH | `configs/defaults.py` | Tách `get_device` → `common/device.py`; `linear_ramp` → `modules/scheduler.py` |
| **`training/logging_utils.py`** | | | |
| `training/logging_utils.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/common/logging.py` | **QUAN TRỌNG** — sửa inversion of dependency |
| **`training/cli.py`** | | | |
| `training/cli.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/training/cli.py` | Bỏ `parse_stage1_args` dead, thêm `parse_inference_args` |
| **`training/checkpoint.py`** | | | |
| `training/checkpoint.py` | 📦 DI CHUYỂN | `symbiopan/training/checkpoint.py` | Tên giữ nguyên, đổi import `logger` |
| **`training/gpu_setup.py`** | | | |
| `training/gpu_setup.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/training/gpu_setup.py` | Bỏ monkey-patch, chuyển qua `autocast_strategy` config |
| **`training/train_loop.py`** | | | |
| `training/train_loop.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/training/train_loop.py` | Sửa bug dòng 53-55 (no-op logic) |
| **`training/stage1_trainer.py`** | | | |
| `training/stage1_trainer.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/training/stage1_trainer.py` | Bỏ `cfg` global, dùng `self.cfg`; dùng `common.logging` |
| **`training/__init__.py`** | | | |
| `training/__init__.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/training/__init__.py` | Bỏ `get_stage1_main` dead, bỏ eager imports |
| **`data/`** | | | |
| `data/__init__.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/data/__init__.py` | Bỏ re-export `INTERNAL_TISSUE_ID_TO_NAME` (trùng) |
| `data/constants.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/data/constants.py` | Gộp `INTERNAL_TISSUE_ID_TO_NAME` ↔ `PUMA_TISSUE_ID_TO_NAME`; chuẩn hoá `RARE_TISSUE_IDS`; thêm 5 task `LOSS_MULTIPLIERS` |
| `data/dataset/__init__.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/data/dataset/__init__.py` | Bỏ wrapper `get_train_transforms_stain_aug` |
| `data/dataset/puma_dataset.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/data/dataset/puma_dataset.py` | **Bỏ `_CONTEXT_CACHE` global** → class attr; dùng `common.logging`; xoá `import functools` |
| `data/dataset/sampling.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/data/sampling.py` | Gộp `sample_weight_from_masks` từ preprocess.py vào đây |
| `data/dataset/transforms.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/data/dataset/transforms.py` | Xoá `import deepcopy`; dùng `use_stain_aug` tham số |
| `data/preprocessing/__init__.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/data/preprocessing/__init__.py` | Bỏ eager import `main` |
| `data/preprocessing/flow_generator.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/data/preprocessing/flow_generator.py` | Vectorize bằng `scipy.ndimage` |
| `data/preprocessing/geojson_parser.py` | 📦 DI CHUYỂN | `symbiopan/data/preprocessing/geojson_parser.py` | Đổi import `logger` |
| `data/preprocessing/preprocess.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/data/preprocessing/preprocess.py` | **Bỏ `cfg` global** → tham số; xoá `sample_weight_from_masks` (gộp vào sampling.py) |
| **`models/`** | | | |
| `models/__init__.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/models/__init__.py` | Bỏ re-export `BoundaryAttentionModule` |
| `models/backbone.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/models/backbone.py` | Thêm docstring; arch name từ config |
| `models/cross_attention.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/models/cross_attention.py` | Thêm docstring; dùng `nn.MultiheadAttention` chuẩn thay vì tự code SDPA |
| `models/encoder.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/models/encoder.py` | **Xoá `extract_intermediate_features`**; move `_patch_proj` ra `__init__`; xoá `_simple_patch_embed` fallback; dùng `common.logging` |
| `models/fpn_aggregator.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/models/fpn_aggregator.py` | **Xoá `vit_intermediate` unused param** |
| `models/decoders.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/models/decoders.py` | Bỏ hardcode `vit_dims` |
| `models/panoptic_net.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/models/panoptic_net.py` | `num_sites` từ config; bỏ hardcode `nn.Embedding(9, 256)` |
| `models/components/__init__.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/models/components/__init__.py` | Bỏ re-export `BoundaryAttentionModule` |
| `models/components/boundary_attention.py` | 🗑️ XOÁ | — | Dead branch — không dùng trong loss lẫn inference |
| `models/components/context_encoder.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/models/components/context_encoder.py` | Arch name từ config |
| `models/components/context_fusion.py` | 📦 DI CHUYỂN | `symbiopan/models/components/context_fusion.py` | Cân nhắc scale-specific FiLM |
| **`inference/`** | | | |
| `inference/__init__.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/inference/__init__.py` | **Bỏ eager import** `main` |
| `inference/infer_wsi.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/inference/infer_wsi.py` | Move `import tifffile` ra top; bỏ hardcode output filename |
| `inference/model_loader.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/inference/model_loader.py` | Lấy config từ tham số |
| `inference/postprocessing.py` | 📦 DI CHUYỂN | `symbiopan/inference/postprocessing.py` | Magic numbers → config |
| `inference/site_classifier.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/inference/site_classifier.py` | **Import `SITE_NAMES` từ `data.constants`**; arch từ config |
| `inference/tiling.py` | 📦 DI CHUYỂN | `symbiopan/inference/tiling.py` | Đổi import `logger` |
| — | ➕ TÁCH | `symbiopan/inference/tta.py` | Tách `TTA_TRANSFORMS` + `TTA_INVERSE` + `apply_tta` ra file riêng |
| **`utils/`** | | | |
| `utils/__init__.py` | 🗑️ XOÁ → chia nhỏ | — | Thay bằng 3 package mới |
| `utils/losses.py` | 📦 DI CHUYỂN + TÁCH | `symbiopan/losses/segmentation.py` + `symbiopan/losses/multitask.py` | Tách `MultiTaskUncertaintyLoss` ra file riêng |
| `utils/metrics.py` | 📦 DI CHUYỂN | `symbiopan/metrics/panoptic.py` | Move nguyên file |
| `utils/sc_dfa.py` | 📦 DI CHUYỂN | `symbiopan/modules/sc_dfa.py` | Đổi tên package (utils → modules) |
| `utils/scheduler_utils.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/modules/scheduler.py` | Gộp `linear_ramp` từ configs/defaults.py |
| `utils/split_utils.py` | 📦 DI CHUYỂN + SỬA | `symbiopan/modules/split.py` | **Xoá `make_or_load_group_split_with_test`** dead code |
| **`scripts/`** | | | |
| `scripts/__init__.py` | ✏️ SỬA | `scripts/__init__.py` | Thêm docstring |
| `scripts/run_inference.py` | ✏️ SỬA | `scripts/infer_wsi.py` | Đổi tên cho rõ, thêm argparse, gọi `symbiopan.inference.infer_wsi.main()` |
| `scripts/run_preprocess.py` | ✏️ SỬA | `scripts/preprocess.py` | Tương tự |
| `scripts/run_stage1.py` | ✏️ SỬA | `scripts/train_stage1.py` | Tương tự + dùng CLI parser |
| **`tests/`** | | | |
| `tests/__init__.py` | ✏️ SỬA | `tests/__init__.py` | Xoá (nội dung rỗng) |
| `tests/__pycache__/` | 🗑️ XOÁ | — | Dead `.pyc` của test đã xoá |
| — | ➕ THÊM | `tests/conftest.py` | Fixtures: `tiny_model`, `dummy_batch`, `temp_dataset_dir` |
| `tests/test_dataset.py` | ✏️ SỬA + MỞ RỘNG | `tests/test_data/test_dataset.py` | Test `PUMADataset.__getitem__`, `compute_sample_weights`, `_load_context_roi` |
| `tests/test_inference.py` | ✏️ SỬA + TÁCH | `tests/test_inference/test_*.py` | Tách thành 5 file theo module |
| `tests/test_losses.py` | ✏️ SỬA | `tests/test_losses/test_segmentation.py` + `test_multitask.py` | Tách tương ứng với source |
| `tests/test_metrics.py` | ✏️ SỬA | `tests/test_metrics/test_panoptic.py` | Đổi tên |
| `tests/test_models.py` | ✏️ SỬA + TÁCH | `tests/test_models/test_*.py` | Tách thành 4 file |
| `tests/test_transforms.py` | ✏️ SỬA + MỞ RỘNG | `tests/test_data/test_transforms.py` | **Test `_fix_vector_field` kỹ lưỡng** |
| **`notebooks/`** | | | |
| `notebooks/train_model.ipynb` | ✏️ RÚT GỌN | `notebooks/01_quickstart.ipynb` | Giảm từ 1228 dòng → ~300 dòng |
| — | ➕ TÁCH | `notebooks/02_visualization.ipynb` | Phần visualization |
| **`docs/`** | | | |
| `docs/architecture.md` | ✏️ SỬA | `docs/architecture.md` | Cập nhật cho khớp code mới; giải quyết mâu thuẫn background |
| — | ➕ THÊM | `docs/REFACTORING_GUIDE.md` | File này |
| — | ➕ THÊM | `docs/CHANGELOG.md` | Lịch sử refactor |
| — | ➕ THÊM | `docs/images/` | Diagram |

### 4.2. Thống kê

| Hành động | Số lượng |
|---|---|
| 🗑️ XOÁ | 7 file |
| 📦 DI CHUYỂN | 11 file |
| ➕ THÊM MỚI | 9 file |
| ✏️ SỬA NỘI DUNG | 24 file |
| 📦 DI CHUYỂN + ✏️ SỬA | 14 file |
| ➕ TÁCH | 4 file |

---

## 5. Kế Hoạch Refactor Theo Giai Đoạn

Mỗi giai đoạn phải chạy được test + lint sau khi hoàn thành. KHÔNG được commit nếu test fail.

### Giai đoạn 0: Chuẩn bị (1–2 giờ)

**Mục tiêu**: tạo branch, chuẩn bị tooling, backup.

```bash
git checkout -b refactor/v9
git tag v8.0.0-baseline
```

- [ ] Tạo `requirements-dev.txt` với pytest, pytest-cov, ruff, jupyter, pre-commit.
- [ ] Cập nhật `pyproject.toml`: thêm `[tool.setuptools.packages.find]`, `[tool.ruff.lint.per-file-ignores]`, `[tool.coverage.run]`.
- [ ] Xoá `requirements.txt` (trùng `pyproject.toml`).
- [ ] Xoá `tests/__pycache__/` và `outputs/` rỗng.
- [ ] Chạy `pytest` để baseline: ghi lại số test pass/fail.
- [ ] Chạy `ruff check .` để baseline.

### Giai đoạn 1: Tạo package `symbiopan/` (3–4 giờ)

**Mục tiêu**: thiết lập skeleton package mới, copy code, sửa import. CHƯA refactor nội dung.

**Bước 1.1**: Tạo cấu trúc
```bash
mkdir -p symbiopan/{common,data/{dataset,preprocessing},models/components,inference,training,losses,metrics,modules}
```

**Bước 1.2**: Copy file từ vị trí cũ sang vị trí mới (giữ nguyên nội dung)
- `data/` → `symbiopan/data/`
- `models/` → `symbiopan/models/`
- `inference/` → `symbiopan/inference/`
- `utils/losses.py` → `symbiopan/losses/__init__.py` (tạm)
- `utils/metrics.py` → `symbiopan/metrics/__init__.py` (tạm)
- `utils/sc_dfa.py` → `symbiopan/modules/sc_dfa.py`
- `utils/scheduler_utils.py` → `symbiopan/modules/scheduler.py`
- `utils/split_utils.py` → `symbiopan/modules/split.py`
- `training/` → `symbiopan/training/` (trừ `logging_utils.py`)

**Bước 1.3**: Tạo `symbiopan/common/`
- `symbiopan/common/logging.py`: copy từ `training/logging_utils.py`, đổi `logger` thành `get_logger(name)` factory
- `symbiopan/common/device.py`: chứa `get_device()` từ `configs/defaults.py`
- `symbiopan/common/types.py`: define `BatchDict`, `ModelOutputDict` TypedDict
- `symbiopan/common/exceptions.py`: `SymbioPanError`, `CheckpointMismatchError`, `DataLeakageError`

**Bước 1.4**: Cập nhật `pyproject.toml`
```toml
[tool.setuptools.packages.find]
include = ["symbiopan*", "configs*"]
exclude = ["tests*", "notebooks*", "scripts*"]
```

**Bước 1.5**: Cập nhật tất cả import statements
- Thay `from data.constants import ...` → `from symbiopan.data.constants import ...`
- Thay `from training.logging_utils import logger` → `from symbiopan.common.logging import get_logger` + `logger = get_logger(__name__)`
- Thay `from utils.losses import ...` → `from symbiopan.losses import ...`
- ... tương tự cho tất cả.

Xem **Phụ Lục §10** để biết mapping đầy đủ.

**Bước 1.6**: Verify
```bash
python -c "import symbiopan; from symbiopan.models import UnifiedPanopticNet; print('OK')"
pytest -x
ruff check .
```

### Giai đoạn 2: Sửa inversion of dependency (1–2 giờ)

**Mục tiêu**: đảm bảo `data/`, `models/`, `inference/` KHÔNG import từ `training/`.

- [ ] Xoá `training/logging_utils.py`.
- [ ] Trong `symbiopan/data/`, `symbiopan/models/`, `symbiopan/utils/`, `symbiopan/inference/`: thay `from training.logging_utils import logger` → `from symbiopan.common.logging import get_logger; logger = get_logger(__name__)`.
- [ ] Tương tự cho `extract_state_dict`: chuyển từ `training/checkpoint.py` sang `symbiopan/training/checkpoint.py` (giữ nguyên nơi), nhưng import trong `inference/site_classifier.py` và `inference/model_loader.py` đổi thành `from symbiopan.training.checkpoint import extract_state_dict`.
- [ ] Test: `python -c "import symbiopan.data; import symbiopan.models; import symbiopan.inference; print('No circular imports!')"`.
- [ ] Chạy `pytest`. Nếu pass → commit.

### Giai đoạn 3: Loại bỏ dead code (2–3 giờ)

- [ ] Xoá `models/encoder.py:extract_intermediate_features()`.
- [ ] Xoá `models/fpn_aggregator.py:vit_intermediate` param.
- [ ] Xoá `models/components/boundary_attention.py` (toàn bộ file).
- [ ] Xoá `models/panoptic_net.py` import và sử dụng `BoundaryAttentionModule`.
- [ ] Xoá `models/decoders.py` boundary output.
- [ ] Xoá `data/dataset/__init__.py:get_train_transforms_stain_aug()`. Cập nhật `stage1_trainer.py` dùng `get_train_transforms(..., use_stain_aug=True)`.
- [ ] Xoá `training/cli.py:parse_stage1_args()` (hoặc dùng nó trong `scripts/train_stage1.py`).
- [ ] Xoá `training/__init__.py:get_stage1_main()`.
- [ ] Xoá `utils/split_utils.py:make_or_load_group_split_with_test()`.
- [ ] Xoá `data/dataset/puma_dataset.py:import functools`.
- [ ] Xoá `data/dataset/transforms.py:from copy import deepcopy`.
- [ ] Xoá `data/__init__.py:INTERNAL_TISSUE_ID_TO_NAME` re-export.
- [ ] Xoá `inference.sh` (cũ), thay bằng version dùng ENV.
- [ ] Test + commit.

### Giai đoạn 4: Sửa mutable global (2–3 giờ)

- [ ] **`_CONTEXT_CACHE` trong `puma_dataset.py`**:
  - Chuyển thành instance attribute: `self._context_cache: dict[str, np.ndarray] = {}`.
  - Thêm method `clear_cache()`.
  - Thêm config `cache_context_roi: bool = True` trong `Stage1Config`.
- [ ] **`cfg` global trong `stage1_trainer.py`**:
  - Refactor tất cả hàm thành method của class `Stage1Trainer`.
  - `cfg` thành `self.cfg`.
- [ ] **`cfg` global trong `preprocess.py`**:
  - Refactor `main(override_cfg=None)` để `cfg = override_cfg or PREPROCESS_DEFAULT_CONFIG` ở đầu hàm (không ở module-level).
- [ ] **Module-level `logger = setup_logger()` trong `logging_utils.py`**: refactor thành `get_logger(__name__)` ở từng module.
- [ ] Test + commit.

### Giai đoạn 5: Gộp trùng lặp (2–3 giờ)

- [ ] **`INTERNAL_TISSUE_ID_TO_NAME` ↔ `PUMA_TISSUE_ID_TO_NAME`**: chỉ giữ `PUMA_TISSUE_ID_TO_NAME` (vì `INTERNAL_*` chỉ dùng trong internal pipeline). Xoá `INTERNAL_TISSUE_ID_TO_NAME` và tất cả re-export.
- [ ] **`RARE_TISSUE_IDS` (set) ↔ `RARE_TISSUE_IDS_PUMA` (list)**: chỉ giữ `RARE_TISSUE_IDS = frozenset({2, 4, 5})` (immutable). Các nơi dùng list → convert: `list(RARE_TISSUE_IDS)`.
- [ ] **`sample_weight_from_masks` ↔ `compute_sample_weight`**:
  - Chuyển `sample_weight_from_masks` từ `preprocess.py` sang `data/sampling.py`.
  - Refactor thành 1 hàm duy nhất: `compute_sample_weight(tissue, nuclei, is_rare_augmented, metadata_weight=None)`.
  - Cập nhật `preprocess.py` import từ `sampling.py`.
- [ ] **`SITE_NAMES` ↔ `inference/site_classifier.py`**: xoá `SITE_NAMES` ở `site_classifier.py`, import từ `data.constants`.
- [ ] **`_VIRCHOW2_CFG` hardcode**: chuyển thành `Stage1Config.encoder_config` (dict).
- [ ] **`paige-ai/Virchow2` hardcode** ở 3 file: thay bằng `Stage1Config.virchow2_model_name` + `InferenceConfig.virchow2_model_name`.
- [ ] **`LOSS_MULTIPLIERS`**: đổi thành `[2.5, 1.0, 2.8, 1.0, 0.0]` (5 phần tử) trong `data/constants.py`. Xoá phần `+ [0.0]` trong `losses.py`.
- [ ] Test + commit.

### Giai đoạn 6: Magic numbers → config (2–3 giờ)

Thêm các field sau vào `Stage1Config` / `InferenceConfig`:

```python
# Stage1Config bổ sung
context_roi_size: int = 320
spatial_injector_target_grid: int = 64
selection_score_weights: tuple[float, float, float] = (0.20, 0.25, 0.55)
focal_tversky_alpha: float = 0.30
focal_tversky_beta: float = 0.70
focal_tversky_gamma: float = 1.25
focal_bce_alpha: float = 0.45
focal_bce_gamma: float = 2.0
warmup_image_size: int = 1024  # dùng cho torch.compile warmup
warmup_epochs: int = 5

# GPU setup config tách riêng
@dataclass(frozen=True)
class GPUSetupConfig:
    vram_thresholds_gb: tuple[float, ...] = (75.0, 40.0, 16.0)
    epochs_override: int = 30
    focal_start_epoch_override: int = 6
    focal_full_epoch_override: int = 10
    sc_dfa_start_epoch_override: int = 9
    sc_dfa_full_epoch_override: int = 13
```

Cập nhật:
- `puma_dataset.py`: dùng `cfg.context_roi_size`.
- `cross_attention.py`: dùng `cfg.spatial_injector_target_grid`.
- `metrics.py`: dùng `cfg.selection_score_weights`.
- `losses.py`: dùng `cfg.focal_tversky_*`, `cfg.focal_bce_*`.
- `stage1_trainer.py`: dùng `cfg.warmup_image_size`.

Test + commit.

### Giai đoạn 7: Refactor paths & Docker (2–3 giờ)

**Bước 7.1**: Đổi tên `outputs/` → `output/`
```bash
git mv outputs output
```

**Bước 7.2**: Đổi tên `data/raw/` → `Dataset/`
```bash
git mv data/raw Dataset
```

**Bước 7.3**: Tạo `checkpoints/` cho weights.

**Bước 7.4**: Cập nhật Dockerfile
```dockerfile
ARG APP_DIR=/opt/app
ARG INPUT_DIR=/input
ARG OUTPUT_DIR=/output

ENV APP_DIR=${APP_DIR} \
    INPUT_DIR=${INPUT_DIR} \
    OUTPUT_DIR=${OUTPUT_DIR} \
    PYTHONUNBUFFERED=1
```

**Bước 7.5**: Cập nhật `inference.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail
exec python -m scripts.infer_wsi \
    --input "${INPUT_DIR}/images/melanoma-whole-slide-image" \
    --output "${OUTPUT_DIR}" \
    --cp "${APP_DIR}/checkpoints/best_model.pth" \
    "$@"
```

**Bước 7.6**: Cập nhật `InferenceConfig` để dùng env var:
```python
import os
@dataclass(frozen=True)
class InferenceConfig:
    cp: str = field(default_factory=lambda: os.environ.get(
        "SYMBIOPAN_CKPT", "/opt/app/checkpoints/best_model.pth"))
    site_classifier_cp: str = field(default_factory=lambda: os.environ.get(
        "SYMBIOPAN_SITE_CKPT", "/opt/app/checkpoints/site_classifier_atto.pth"))
```

Test + commit.

### Giai đoạn 8: Sửa bug logic (30 phút)

- [ ] **`train_loop.py:53-55`** sửa từ:
  ```python
  core.encoder.vit_model.eval() if not core.encoder.fine_tune else None
  ```
  Thành (nếu muốn freeze ViT):
  ```python
  if not core.encoder.fine_tune:
      core.encoder.vit_model.eval()
      for p in core.encoder.vit_model.parameters():
          p.requires_grad = False
  ```
  Hoặc (nếu không cần logic này): xoá hẳn dòng đó và comment giải thích.

Test + commit.

### Giai đoạn 9: Tách file (3–4 giờ)

- [ ] Tách `utils/losses.py` (187 dòng) → `symbiopan/losses/segmentation.py` (CE, FocalTversky, SoftDice, FocalBCE) + `symbiopan/losses/multitask.py` (MultiTaskUncertaintyLoss).
- [ ] Tách `TTA_TRANSFORMS`, `TTA_INVERSE`, `apply_tta` từ `infer_wsi.py` → `symbiopan/inference/tta.py`.
- [ ] Tách `flow_generator.py` thành 2 hàm: `compute_hv_map` (vectorized) + `compute_centers_and_radii` (helper).
- [ ] Tách `notebooks/train_model.ipynb` (1228 dòng) → `01_quickstart.ipynb` (~300 dòng) + `02_visualization.ipynb` (~200 dòng).

Test + commit.

### Giai đoạn 10: Cải thiện test (4–5 giờ)

- [ ] Tạo `tests/conftest.py` với fixtures:
  - `tiny_model`: `UnifiedPanopticNet` với config thu nhỏ
  - `dummy_batch`: dict giả lập batch
  - `temp_dataset_dir`: tạo thư mục tạm với vài file `.npy` mẫu
  - `mock_checkpoint`: tạo checkpoint giả trong tmp
- [ ] Tách test theo cấu trúc thư mục mới (xem §3).
- [ ] Bổ sung test:
  - `test_data/test_dataset.py::test_getitem_returns_expected_keys`
  - `test_data/test_dataset.py::test_context_cache_clears_on_instance`
  - `test_data/test_transforms.py::test_fix_vector_field_hflip`
  - `test_data/test_transforms.py::test_fix_vector_field_vflip`
  - `test_data/test_transforms.py::test_fix_vector_field_rotate90`
  - `test_data/test_geojson.py::test_parse_polygon`
  - `test_inference/test_tiling.py::test_make_tile_starts`
  - `test_inference/test_tiling.py::test_pad_reflect`
  - `test_inference/test_postprocessing.py::test_hv_watershed`
  - `test_inference/test_model_loader.py::test_load_stage1_state_dict`
  - `test_training/test_checkpoint.py::test_safe_save_load`
  - `test_training/test_splits.py::test_group_split_disjoint`
  - `test_models/test_panoptic_net.py::test_forward_output_keys`
  - `test_losses/test_multitask.py::test_boundary_loss_zero`
- [ ] Đặt target coverage ≥ 70%.
- [ ] Chạy `pytest --cov=symbiopan` và commit.

### Giai đoạn 11: Tài liệu & cleanup (2–3 giờ)

- [ ] Cập nhật `docs/architecture.md` cho khớp code mới; giải quyết mâu thuẫn background class.
- [ ] Cập nhật `README.md` với project structure mới.
- [ ] Tạo `docs/CHANGELOG.md` với các thay đổi.
- [ ] Tạo `docs/images/` với diagram (nếu có).
- [ ] Cập nhật `.gitignore`:
  - Bỏ `dataset/`, `output/`, `checkpoints/`, `data/raw/`, `outputs/`
  - Thêm `dataset/`, `Dataset/`, `output/`, `checkpoints/`
  - Thêm `*.log`, `*.bak`, `*.tmp`
- [ ] Chạy `make clean && make lint && make test`.
- [ ] Tag version mới `v9.0.0`.

### Giai đoạn 12: CI/CD (2–3 giờ)

- [ ] Tạo `.github/workflows/lint.yml`:
  ```yaml
  name: Lint
  on: [push, pull_request]
  jobs:
    ruff:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
        - run: pip install -e ".[dev]"
        - run: ruff check .
  ```
- [ ] Tạo `.github/workflows/test.yml` với matrix Python 3.11/3.12 + PyTorch CPU.
- [ ] Tạo `.github/workflows/docker-build.yml` (optional).
- [ ] Test CI chạy đúng trên PR mẫu.

---

## 6. Các Vấn Đề Code Cụ Thể Và Cách Sửa

### 6.1. Bug: `train_loop.py:53-55` no-op

**Hiện tại**:
```python
# Trong train_one_epoch
core.encoder.vit_model.eval() if not core.encoder.fine_tune else None
```

**Vấn đề**: Biểu thức `A if cond else None` chỉ trả về giá trị, không gọi `eval()`. Comment nói "freeze ViT" nhưng code không làm gì.

**Sửa** (nếu muốn giữ logic):
```python
if not getattr(core.encoder, "fine_tune", True):
    core.encoder.vit_model.eval()
    for p in core.encoder.vit_model.parameters():
        p.requires_grad = False
else:
    # ensure train mode for ViT last N blocks
    core.encoder.vit_model.train()
```

**Hoặc** xoá dòng này nếu logic freeze đã được xử lý trong `UnifiedPanopticEncoder`.

### 6.2. Memory leak: `_CONTEXT_CACHE`

**Hiện tại**:
```python
# puma_dataset.py
_CONTEXT_CACHE: dict[str, np.ndarray] = {}
_CONTEXT_ROI_SIZE = 320

def _load_context_roi(base_name: str, context_dir: Path) -> np.ndarray:
    if base_name in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[base_name]
    img = ...
    _CONTEXT_CACHE[base_name] = img
    return img
```

**Vấn đề**: 
1. Global mutable, không bao giờ clear → memory tăng dần trong training dài (hàng giờ).
2. Không có cách nào clear từ bên ngoài.

**Sửa**:
```python
# puma_dataset.py
class PUMADataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        context_dir: Path | None = None,
        context_roi_size: int = 320,
        cache_context_roi: bool = True,
        max_cache_size: int = 256,
    ):
        ...
        self._context_cache: dict[str, np.ndarray] = {}
        self._cache_context_roi = cache_context_roi
        self._max_cache_size = max_cache_size
        # ... sử dụng self.context_roi_size
    
    def _load_context_roi(self, base_name: str) -> np.ndarray | None:
        if not self._cache_context_roi:
            return self._read_context_roi_from_disk(base_name)
        if base_name not in self._context_cache:
            if len(self._context_cache) >= self._max_cache_size:
                # FIFO eviction
                oldest = next(iter(self._context_cache))
                del self._context_cache[oldest]
            self._context_cache[base_name] = self._read_context_roi_from_disk(base_name)
        return self._context_cache[base_name]
    
    def clear_cache(self) -> None:
        self._context_cache.clear()
```

### 6.3. Trùng lặp: `INTERNAL_TISSUE_ID_TO_NAME` ↔ `PUMA_TISSUE_ID_TO_NAME`

**Hiện tại** (trong `data/constants.py`):
```python
PUMA_TISSUE_ID_TO_NAME = {0: "background", 1: "tissue_tumor", ...}
INTERNAL_TISSUE_ID_TO_NAME = {0: "background", 1: "tissue_tumor", ...}  # GIỐNG HỆT
```

**Sửa**: Xoá `INTERNAL_TISSUE_ID_TO_NAME`. Cập nhật:
- `data/__init__.py`: bỏ re-export.
- Tìm tất cả usage: `grep -r "INTERNAL_TISSUE_ID_TO_NAME" .` → thay bằng `PUMA_TISSUE_ID_TO_NAME`.

### 6.4. Inconsistency: `RARE_TISSUE_IDS` (set) vs `RARE_TISSUE_IDS_PUMA` (list)

**Hiện tại**:
```python
RARE_TISSUE_IDS = {2, 4, 5}                # set
RARE_TISSUE_IDS_PUMA = [2, 4, 5]           # list, cùng data
```

**Sửa**:
```python
RARE_TISSUE_IDS: frozenset[int] = frozenset({2, 4, 5})
# Xoá RARE_TISSUE_IDS_PUMA
# Nơi nào dùng list → gọi sorted(RARE_TISSUE_IDS) hoặc list(RARE_TISSUE_IDS)
```

### 6.5. LOSS_MULTIPLIERS thiếu phần tử

**Hiện tại**:
```python
# data/constants.py
LOSS_MULTIPLIERS = [2.5, 1.0, 2.8, 1.0]  # 4 phần tử cho 4 task

# utils/losses.py
multipliers = LOSS_MULTIPLIERS + [0.0]  # hack: extend thành 5
```

**Sửa**:
```python
# data/constants.py
LOSS_MULTIPLIERS = [2.5, 1.0, 2.8, 1.0, 0.0]  # 5 phần tử (task cuối = boundary, mult=0)

# losses.py
multipliers = LOSS_MULTIPLIERS  # bỏ extend
```

### 6.6. Monkey-patch trong `gpu_setup.py`

**Hiện tại**:
```python
def patch_autocast_for_bf16():
    import training.train_loop
    training.train_loop._autocast_context = _bf16_autocast  # thay đổi hàm
```

**Vấn đề**: 
1. Monkey-patch khó debug.
2. Phụ thuộc vào thứ tự import.

**Sửa**: Dùng strategy pattern:
```python
# common/device.py
def get_autocast_context(device: torch.device, dtype: torch.dtype = None):
    if dtype is None:
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32)

# training/train_loop.py - không patch
with get_autocast_context(device, dtype=cfg.autocast_dtype):
    ...
```

### 6.7. Hardcode paths trong code

**Hiện tại** (`inference/infer_wsi.py`):
```python
parser.add_argument("--cp", default="/opt/app/checkpoints/best_model.pth")
parser.add_argument("--input", default="/input/images/melanoma-whole-slide-image")
parser.add_argument("--output", default="/output")
```

**Sửa**:
```python
parser.add_argument("--cp", default=os.environ.get("SYMBIOPAN_CKPT", "checkpoints/best_model.pth"))
parser.add_argument("--input", default=os.environ.get("SYMBIOPAN_INPUT", "input"))
parser.add_argument("--output", default=os.environ.get("SYMBIOPAN_OUTPUT", "output"))
```

Cùng với `InferenceConfig` cũng đọc từ env var.

### 6.8. Notebook rườm rà

**Hiện tại**: `notebooks/train_model.ipynb` 1228 dòng, 58 cells, duplicate pipeline.

**Sửa**: Rút gọn còn `notebooks/01_quickstart.ipynb` (~300 dòng) chỉ làm:
1. Setup môi trường.
2. Gọi `python -m scripts.preprocess` (qua subprocess).
3. Gọi `python -m scripts.train_stage1 --epochs 2 --batch_size 2` (quick demo).
4. Gọi `python -m scripts.infer_wsi` (quick demo).

Các phần configuration, loss/metric demo, scheduler visualization chuyển sang markdown cells (giải thích) hoặc xoá hẳn.

### 6.9. Eager import trong `__init__.py`

**Hiện tại** (`inference/__init__.py`):
```python
from inference.infer_wsi import main  # kéo theo torch, cv2, ...
```

**Sửa**:
```python
# inference/__init__.py
from symbiopan.inference.tiling import find_single_tif, normalize_tile
from symbiopan.inference.postprocessing import hv_instance_segmentation, classify_instances

__all__ = [
    "find_single_tif",
    "normalize_tile",
    "hv_instance_segmentation",
    "classify_instances",
]

# main() KHÔNG re-export ở đây, gọi trực tiếp:
# python -m scripts.infer_wsi
# hoặc:
# from symbiopan.inference.infer_wsi import main; main()
```

### 6.10. `import tifffile` bên trong hàm

**Hiện tại** (`inference/infer_wsi.py`):
```python
def process_image(...):
    ...
    import tifffile  # tại sao?
    tifffile.imwrite(...)
```

**Sửa**: Move lên top file (đã có sẵn `from symbiopan.inference.tiling import ...` thì thêm `import tifffile`).

### 6.11. `_VIRCHOW2_CFG` hardcode

**Hiện tại**:
```python
# models/encoder.py
_VIRCHOW2_CFG = {
    "hidden_size": 1280,
    "num_hidden_layers": 32,
    "num_attention_heads": 80,  # 1280 / 16
    ...
}
```

**Sửa**:
```python
# configs/defaults.py
@dataclass(frozen=True)
class Stage1Config:
    ...
    encoder_config: dict[str, int] = field(default_factory=lambda: {
        "hidden_size": 1280,
        "num_hidden_layers": 32,
        "num_attention_heads": 80,
        "intermediate_size": 5120,
        "patch_size": 14,
        "image_size": 1024,
    })
    virchow2_model_name: str = "paige-ai/Virchow2"
    fine_tune_last_n_blocks: int = 6
```

```python
# models/encoder.py
def build_virchow2_vit(model_name: str, encoder_cfg: dict, ...):
    config = ViTConfig(**encoder_cfg)
    ...
```

### 6.12. `nn.Embedding(9, 256)` hardcode

**Hiện tại** (`models/panoptic_net.py`):
```python
self.site_embedding = nn.Embedding(9, 256)  # 9 = len(SITE_MAP)
```

**Sửa**:
```python
self.site_embedding = nn.Embedding(num_sites, site_embed_dim)
# Truyền từ config:
#   num_sites = len(SITE_MAP)  # 9, nhưng đọc từ config
#   site_embed_dim = 256       # từ config
```

### 6.13. CLI không xài

**Hiện tại** (`training/cli.py`):
```python
def parse_stage1_args() -> Stage1Config:
    parser = argparse.ArgumentParser()
    ...
    return Stage1Config(...)
```

`scripts/run_stage1.py` KHÔNG gọi hàm này. **Dead code**.

**Sửa** (option A — dùng nó):
```python
# scripts/train_stage1.py
from symbiopan.training.cli import parse_stage1_args
from symbiopan.training.stage1_trainer import Stage1Trainer

def main():
    cfg = parse_stage1_args()
    cfg = detect_gpu_setup().merge(cfg)
    trainer = Stage1Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
```

**Sửa** (option B — xoá):
```bash
rm training/cli.py
```

### 6.14. Print thay vì logger

**Hiện tại** (`training/gpu_setup.py`):
```python
print(f"Detected GPU: ...")
```

**Sửa**:
```python
from symbiopan.common.logging import get_logger
logger = get_logger(__name__)
logger.info("Detected GPU: %s", ...)
```

### 6.15. Mâu thuẫn README ↔ architecture.md

**Hiện tại**:
- README: "background is ignore index 255"
- architecture.md: "Background included as class 0"

**Sửa**:
1. Kiểm tra code thực tế: `TISSUE_CLASS_WEIGHTS = [0.2, 1.0, 4.0, 0.8, 3.0, 7.0]` có 6 phần tử → khớp với "6 classes including background".
2. Cập nhật README cho khớp architecture.md (background = class 0).
3. Thêm 1 dòng giải thích sự thay đổi trong `CHANGELOG.md`.

---

## 7. Cải Thiện Testing

### 7.1. Cấu trúc test mới

```
tests/
├── conftest.py                        # Fixtures chung
├── test_common/                       # Test common/
│   ├── test_logging.py
│   └── test_device.py
├── test_data/                         # Test symbiopan/data/
│   ├── test_constants.py              # MỚI: SITE_MAP, LOSS_MULTIPLIERS đúng độ dài
│   ├── test_dataset.py                # MỞ RỘNG: __getitem__, context cache
│   ├── test_geojson.py                # MỚI: parse GeoJSON
│   ├── test_sampling.py               # MỚI: compute_sample_weight edge cases
│   └── test_transforms.py             # MỞ RỘNG: _fix_vector_field kỹ lưỡng
├── test_losses/                       # Test losses/
│   ├── test_segmentation.py
│   └── test_multitask.py              # MỚI
├── test_metrics/                      # Test metrics/
│   └── test_panoptic.py
├── test_models/                       # Test models/
│   ├── test_backbone.py
│   ├── test_encoder.py                # MỚI (mock HF)
│   ├── test_decoders.py
│   ├── test_fpn.py
│   └── test_panoptic_net.py           # MỚI: end-to-end
├── test_inference/                    # Test inference/
│   ├── test_tiling.py                 # MỚI
│   ├── test_postprocessing.py         # MỚI
│   ├── test_site_classifier.py        # MỚI
│   ├── test_model_loader.py           # MỚI
│   └── test_infer_wsi.py              # MỞ RỘNG
├── test_training/                     # Test training/
│   ├── test_checkpoint.py             # MỚI
│   ├── test_splits.py                 # MỚI
│   ├── test_cli.py                    # MỚI
│   └── test_train_loop.py             # MỚI: freeze ViT bug
└── test_modules/                      # Test modules/
    ├── test_scheduler.py
    ├── test_sc_dfa.py
    └── test_split.py
```

### 7.2. `conftest.py` mẫu

```python
# tests/conftest.py
import pytest
import torch
import numpy as np
from pathlib import Path

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def dummy_batch(device):
    B, H, W = 2, 256, 256
    return {
        "image": torch.randn(B, 3, H, W, device=device),
        "tissue_sem": torch.randint(0, 6, (B, H, W), device=device),
        "nuclei_nc": torch.randint(0, 10, (B, H, W), device=device),
        "nuclei_hv": torch.randn(B, 2, H, W, device=device),
        "site_id": torch.zeros(B, dtype=torch.long, device=device),
        "nuclei_np": (torch.rand(B, 1, H, W, device=device) > 0.5).float(),
    }

@pytest.fixture
def tiny_model_cfg():
    from symbiopan.configs.defaults import Stage1Config
    return Stage1Config(
        image_size=128,
        batch_size=2,
        cnn_backbone="convnext_tiny",
        virchow2_model_name="dummy/vit-tiny",  # mock
        fine_tune_last_n_blocks=0,
    )

@pytest.fixture
def tiny_model(tiny_model_cfg, monkeypatch):
    """Mock HF model loading."""
    from symbiopan.models import UnifiedPanopticNet
    # ... patch HF loading để không cần internet
    return UnifiedPanopticNet(tiny_model_cfg)

@pytest.fixture
def temp_dataset_dir(tmp_path):
    """Tạo thư mục dataset giả."""
    data_dir = tmp_path / "dataset"
    (data_dir / "images").mkdir()
    (data_dir / "tissue_sem").mkdir()
    (data_dir / "nuclei_nc").mkdir()
    (data_dir / "nuclei_hv").mkdir()
    # Tạo 5 sample giả
    for i in range(5):
        np.save(data_dir / "images" / f"sample_{i}.npy", np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        np.save(data_dir / "tissue_sem" / f"sample_{i}.npy", np.random.randint(0, 6, (256, 256), dtype=np.uint8))
        np.save(data_dir / "nuclei_nc" / f"sample_{i}.npy", np.random.randint(0, 10, (256, 256), dtype=np.uint8))
        np.save(data_dir / "nuclei_hv" / f"sample_{i}.npy", np.random.randn(256, 256, 2).astype(np.float16))
    return data_dir
```

### 7.3. Test quan trọng cần bổ sung

```python
# tests/test_data/test_dataset.py
def test_context_cache_clears_on_instance(temp_dataset_dir):
    from symbiopan.data.dataset import PUMADataset
    ds1 = PUMADataset(temp_dataset_dir, cache_context_roi=True, max_cache_size=2)
    # ... add 3 samples → cache phải evict
    
def test_getitem_returns_expected_keys(temp_dataset_dir):
    from symbiopan.data.dataset import PUMADataset
    ds = PUMADataset(temp_dataset_dir)
    sample = ds[0]
    assert "image" in sample
    assert "tissue_sem" in sample
    assert "nuclei_nc" in sample
    assert "nuclei_hv" in sample
    assert "site_id" in sample
    assert "nuclei_np" in sample  # derived from nc != 255

# tests/test_data/test_transforms.py
def test_fix_vector_field_hflip():
    """HV vector phải đổi dấu X khi flip ngang."""
    from symbiopan.data.dataset.transforms import _fix_vector_field
    vec = np.array([[[1.0, 0.5], [0.0, 1.0]]], dtype=np.float32)  # (1, H, W, 2)
    replay = {"applied_transforms": [{"__class_fullname__": "albumentations.augmentations.transforms.HorizontalFlip"}]}
    fixed = _fix_vector_field(vec.copy(), replay)
    assert fixed[0, 0, 0, 0] == -1.0  # X bị đổi dấu
    assert fixed[0, 0, 0, 1] == 0.5   # Y giữ nguyên

def test_fix_vector_field_rotate90():
    """HV vector phải rotate khi rotate90."""
    from symbiopan.data.dataset.transforms import _fix_vector_field
    vec = np.array([[[1.0, 0.0]]], dtype=np.float32)
    replay = {"applied_transforms": [{"__class_fullname__": "albumentations.augmentations.transforms.RandomRotate90"}]}
    # ... test rotation

# tests/test_losses/test_multitask.py
def test_boundary_loss_is_zero():
    """MultiTaskUncertaintyLoss với boundary logits phải trả loss = 0."""
    from symbiopan.losses import MultiTaskUncertaintyLoss
    from symbiopan.data.constants import LOSS_MULTIPLIERS
    loss_fn = MultiTaskUncertaintyLoss(num_tasks=5)
    boundary_logits = torch.randn(2, 1, 128, 128)
    boundary_target = torch.zeros(2, 1, 128, 128)
    loss = loss_fn.compute_boundary(boundary_logits, boundary_target)
    assert loss.item() == 0.0

# tests/test_training/test_train_loop.py
def test_freeze_vit_when_not_finetuning(monkeypatch):
    """Sửa bug no-op: vit_model.eval() phải được gọi khi fine_tune=False."""
    # ... test logic mới
```

### 7.4. Target coverage

| Module | Target |
|---|---|
| `symbiopan/common/` | ≥ 90% |
| `symbiopan/losses/` | ≥ 90% |
| `symbiopan/metrics/` | ≥ 85% |
| `symbiopan/modules/` | ≥ 80% |
| `symbiopan/data/` | ≥ 70% |
| `symbiopan/models/` | ≥ 60% (vì nhiều phụ thuộc HF) |
| `symbiopan/inference/` | ≥ 60% |
| `symbiopan/training/` | ≥ 50% |
| **Tổng** | **≥ 70%** |

Cấu hình trong `pyproject.toml`:
```toml
[tool.coverage.run]
source = ["symbiopan"]
omit = ["*/tests/*", "*/notebooks/*"]

[tool.coverage.report]
fail_under = 70
show_missing = true
```

---

## 8. Cập Nhật Tài Liệu Và Tiện Ích

### 8.1. `docs/architecture.md` cập nhật

Phần "Tissue Class Definitions" cần thay:

```diff
- | 0–4 | Tissue classes | PUMA format (no background class) |
+ | 0 | Background     | PUMA format                      |
+ | 1–5 | Tissue classes | 5 classes in PUMA format         |
```

Phần "Ignore Index" cần thay:

```diff
- Background is treated as ignore_index 255 during training.
+ Background is class 0 in PUMA format and is included in tissue_sem during training.
```

### 8.2. `docs/CHANGELOG.md` mẫu

```markdown
# Changelog

## v9.0.0 (refactor)

### Breaking changes
- Renamed package `data/`, `models/`, `inference/`, `training/`, `utils/`, `configs/` to `symbiopan/data/`, `symbiopan/models/`, etc. Update imports accordingly.
- Moved `utils/losses.py` → `symbiopan/losses/`. `MultiTaskUncertaintyLoss` now in `symbiopan.losses.multitask`.
- `LOSS_MULTIPLIERS` now has 5 elements (was 4 + dummy).
- Removed `INTERNAL_TISSUE_ID_TO_NAME` (was duplicate of `PUMA_TISSUE_ID_TO_NAME`).
- Removed `RARE_TISSUE_IDS_PUMA` (use `sorted(RARE_TISSUE_IDS)` instead).
- Removed `extract_intermediate_features()` from `UnifiedPanopticEncoder` (was unused).
- Removed `BoundaryAttentionModule` and boundary output (was unused).
- Removed `make_or_load_group_split_with_test()` (was unused).
- Removed `parse_stage1_args()` and `get_stage1_main()` (were unused).

### Bug fixes
- `train_loop.py:53-55` no-op logic: ViT freezing now works correctly when `fine_tune=False`.
- `puma_dataset.py`: `_CONTEXT_CACHE` global mutable state replaced with per-instance cache with size limit.

### Improvements
- All paths now configurable via environment variables (`SYMBIOPAN_CKPT`, `SYMBIOPAN_INPUT`, `SYMBIOPAN_OUTPUT`).
- `data/raw/` renamed to `Dataset/` to match `PATHS.raw_dir`.
- `outputs/` renamed to `output/` to match `.gitignore`.
- Magic numbers moved to `Stage1Config` / `InferenceConfig`.
- Test coverage increased from ~30% to ~70%.

### Migration guide
See [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md).
```

### 8.3. `Makefile` cập nhật

```makefile
.PHONY: help install dev test lint format clean preprocess stage1 inference docker-build docker-run notebook

PYTHON ?= python
IMAGE  ?= symbiopan-v8-cellpath
DOCKER := docker

help:
	@echo "Targets:"
	@echo "  install      Install package + dev deps"
	@echo "  test         Run pytest"
	@echo "  lint         Run ruff"
	@echo "  format       Run ruff --fix + black"
	@echo "  clean        Remove __pycache__ + .pyc + build artifacts"
	@echo "  preprocess   Run data preprocessing"
	@echo "  stage1       Run stage 1 training"
	@echo "  inference    Run WSI inference"
	@echo "  notebook     Launch Jupyter Lab"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container with mounted volumes"

install:
	pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest --cov=symbiopan --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info .ruff_cache .pytest_cache .mypy_cache

preprocess:
	$(PYTHON) -m scripts.preprocess

stage1:
	$(PYTHON) -m scripts.train_stage1

inference:
	$(PYTHON) -m scripts.infer_wsi --input input --output output

notebook:
	jupyter lab notebooks/

docker-build:
	$(DOCKER) build -t $(IMAGE) .

docker-run:
	$(DOCKER) run --rm \
		-v $(PWD)/input:/input:ro \
		-v $(PWD)/output:/output \
		-v $(PWD)/checkpoints:/opt/app/checkpoints:ro \
		$(IMAGE)
```

### 8.4. `pyproject.toml` cập nhật

```toml
[project]
name = "symbiopan"
version = "9.0.0"
description = "Panoptic segmentation pipeline for PUMA Grand Challenge Track 2"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0",
    "torchvision",
    "timm>=0.9",
    "transformers>=4.40",
    "huggingface_hub",
    "numpy<2.1",
    "scipy",
    "albumentations",
    "opencv-python-headless",
    "tifffile",
    "tqdm",
    "Pillow",
    "bitsandbytes",
    "safetensors",
    "PyYAML",
    "packaging",
    "lark",
    "matplotlib",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov>=4",
    "ruff>=0.4",
    "jupyter",
    "ipykernel",
    "pre-commit",
]

[project.scripts]
symbiopan-preprocess = "scripts.preprocess:main"
symbiopan-train = "scripts.train_stage1:main"
symbiopan-infer = "scripts.infer_wsi:main"

[tool.setuptools.packages.find]
include = ["symbiopan*", "configs*"]
exclude = ["tests*", "notebooks*", "scripts*"]

[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "N", "C4", "SIM"]
ignore = ["E501"]  # line-too-long đã có formatter

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # unused imports OK in __init__
"tests/*" = ["B011"]     # assert False OK in tests

[tool.ruff.format]
quote-style = "double"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q --strict-markers"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks integration tests",
]

[tool.coverage.run]
source = ["symbiopan"]
omit = ["*/tests/*", "*/notebooks/*"]

[tool.coverage.report]
fail_under = 70
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

### 8.5. `.gitignore` cập nhật

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
*.egg

# Build
build/
dist/

# Tooling caches
.ruff_cache/
.pytest_cache/
.mypy_cache/
.coverage
htmlcov/

# Data (large, regenerated)
Dataset/
dataset/
data/raw/
data/processed/
*.npy
*.tif
*.tiff

# Outputs
output/
outputs/
checkpoints/
*.pth
*.pt

# Logs
*.log
*.bak
*.tmp
*.swp
*.swo

# Secrets
.env
.env.*
*.pem
*.key

# Virtual envs
venv/
.venv/

# Notebooks
.ipynb_checkpoints/

# Tracking
wandb/

# IDE
.idea/
.vscode/
*.iml
```

---

## 9. Checklist Migration

### 9.1. Trước khi bắt đầu

- [ ] Tạo branch `refactor/v9` từ `main`.
- [ ] Tag baseline `v8.0.0-baseline`.
- [ ] Chạy `pytest` → ghi lại baseline (số test pass/fail).
- [ ] Chạy `ruff check .` → fix hết warning hiện tại.
- [ ] Thông báo team về breaking changes sắp tới.

### 9.2. Sau mỗi giai đoạn (1–12)

- [ ] `pytest` pass 100%.
- [ ] `ruff check .` không có lỗi mới.
- [ ] Commit với message rõ ràng: `refactor: stage N - <mô tả>`.
- [ ] Cập nhật `docs/CHANGELOG.md` nếu có breaking change.

### 9.3. Trước khi merge vào `main`

- [ ] Tất cả test pass.
- [ ] Coverage ≥ 70%.
- [ ] `Dockerfile` build thành công.
- [ ] `make docker-run` chạy được với dummy data.
- [ ] `make preprocess`, `make stage1`, `make inference` chạy được (với quick test).
- [ ] `notebooks/01_quickstart.ipynb` chạy hết không lỗi.
- [ ] `docs/architecture.md` cập nhật.
- [ ] `README.md` cập nhật.
- [ ] `CHANGELOG.md` đầy đủ.
- [ ] Tạo PR với mô tả chi tiết, link tới file này.
- [ ] Ít nhất 1 reviewer approve.
- [ ] CI/CD pass (lint, test, docker-build).
- [ ] Tag `v9.0.0`.

### 9.4. Sau khi merge

- [ ] Thông báo team về breaking changes.
- [ ] Tạo issue/MR để update downstream code (notebook, scripts).
- [ ] Viết blog post hoặc note nội bộ về kiến trúc mới.
- [ ] Monitor CI/CD trong 1 tuần.

---

## 10. Phụ Lục: Mapping Import Cũ → Mới

### 10.1. `data/` → `symbiopan.data`

| Import cũ | Import mới |
|---|---|
| `from data.constants import ...` | `from symbiopan.data.constants import ...` |
| `from data.dataset.puma_dataset import PUMADataset` | `from symbiopan.data.dataset.puma_dataset import PUMADataset` |
| `from data.dataset import get_train_transforms, get_val_transforms` | `from symbiopan.data.dataset import get_train_transforms, get_val_transforms` |
| `from data.dataset.sampling import compute_sample_weight` | `from symbiopan.data.sampling import compute_sample_weight` |
| `from data.preprocessing.preprocess import main` | `from symbiopan.data.preprocessing.preprocess import main` |
| `from data.preprocessing.geojson_parser import parse_geojson_masks` | `from symbiopan.data.preprocessing.geojson_parser import parse_geojson_masks` |
| `from data.preprocessing.flow_generator import compute_hv_map` | `from symbiopan.data.preprocessing.flow_generator import compute_hv_map` |

### 10.2. `models/` → `symbiopan.models`

| Import cũ | Import mới |
|---|---|
| `from models import UnifiedPanopticNet` | `from symbiopan.models import UnifiedPanopticNet` |
| `from models import build_cnn_backbone` | `from symbiopan.models import build_cnn_backbone` |
| `from models.encoder import UnifiedPanopticEncoder` | `from symbiopan.models.encoder import UnifiedPanopticEncoder` |
| `from models.decoders import ParallelDecoders` | `from symbiopan.models.decoders import ParallelDecoders` |
| `from models.fpn_aggregator import HierarchicalFPN` | `from symbiopan.models.fpn_aggregator import HierarchicalFPN` |
| `from models.components import ContextEncoder, ContextFusionModule` | `from symbiopan.models.components import ContextEncoder, ContextFusionModule` |

### 10.3. `inference/` → `symbiopan.inference`

| Import cũ | Import mới |
|---|---|
| `from inference import main` (inference/__init__) | `from symbiopan.inference.infer_wsi import main` |
| `from inference.infer_wsi import main` | `from symbiopan.inference.infer_wsi import main` |
| `from inference.model_loader import load_stage1` | `from symbiopan.inference.model_loader import load_stage1` |
| `from inference.postprocessing import hv_instance_segmentation` | `from symbiopan.inference.postprocessing import hv_instance_segmentation` |
| `from inference.tiling import find_single_tif, normalize_tile` | `from symbiopan.inference.tiling import find_single_tif, normalize_tile` |
| `from inference.site_classifier import load_site_classifier` | `from symbiopan.inference.site_classifier import load_site_classifier` |
| `from inference.tta import apply_tta` (MỚI) | `from symbiopan.inference.tta import apply_tta` |

### 10.4. `training/` → `symbiopan.training`

| Import cũ | Import mới |
|---|---|
| `from training.checkpoint import extract_state_dict, safe_torch_save, ...` | `from symbiopan.training.checkpoint import ...` |
| `from training.gpu_setup import detect_gpu_setup, cleanup_gpu_cache` | `from symbiopan.training.gpu_setup import ...` |
| `from training.logging_utils import logger, setup_logger` | `from symbiopan.common.logging import get_logger; logger = get_logger(__name__)` |
| `from training.train_loop import train_one_epoch, validate` | `from symbiopan.training.train_loop import train_one_epoch, validate` |
| `from training.stage1_trainer import main` | `from symbiopan.training.stage1_trainer import main` |
| `from training.cli import parse_stage1_args` | `from symbiopan.training.cli import parse_stage1_args` |

### 10.5. `utils/` → TÁCH THÀNH 3 PACKAGE

| Import cũ | Import mới |
|---|---|
| `from utils.losses import MultiTaskUncertaintyLoss` | `from symbiopan.losses import MultiTaskUncertaintyLoss` (hoặc `from symbiopan.losses.multitask import ...`) |
| `from utils.losses import SafeCrossEntropyLoss, FocalTverskyLoss, ...` | `from symbiopan.losses.segmentation import ...` |
| `from utils.metrics import PUMAMetrics, SemanticMetricAccumulator` | `from symbiopan.metrics import PUMAMetrics, SemanticMetricAccumulator` |
| `from utils.sc_dfa import SCDFA` | `from symbiopan.modules import SCDFA` |
| `from utils.scheduler_utils import build_warmup_cosine_scheduler` | `from symbiopan.modules.scheduler import build_warmup_cosine_scheduler` |
| `from utils.split_utils import make_or_load_group_split` | `from symbiopan.modules.split import make_or_load_group_split` |

### 10.6. `configs/`

| Import cũ | Import mới |
|---|---|
| `from configs import STAGE1_DEFAULT_CONFIG, ...` | `from symbiopan.configs import STAGE1_DEFAULT_CONFIG, ...` |
| `from configs.defaults import Stage1Config, PathsConfig, ...` | `from symbiopan.configs.defaults import Stage1Config, PathsConfig, ...` |

### 10.7. Mới: `symbiopan.common`

```python
# Thay vì
from training.logging_utils import logger

# Viết
from symbiopan.common.logging import get_logger
logger = get_logger(__name__)
```

```python
# Thay vì
from configs.defaults import get_device

# Viết
from symbiopan.common.device import get_device
```

---

## Tổng Kết

Refactor này nhằm mục tiêu:

1. **Đơn giản hoá cấu trúc**: loại bỏ inversion of dependency, đảm bảo dependency direction một chiều.
2. **Loại bỏ technical debt**: dead code, magic numbers, hardcode paths, trùng lặp.
3. **Cải thiện testability**: tăng coverage từ ~30% lên ≥70%, bổ sung test cho các path dễ regression.
4. **Tăng portability**: paths configurable qua env var, package layout chuẩn Python.
5. **Dễ onboard**: cấu trúc phản ánh đúng trách nhiệm, tài liệu đầy đủ.

**Ước lượng tổng thời gian**: 25–35 giờ làm việc (3–5 ngày nếu full-time).

**Rủi ro**:
- Refactor nhiều file có thể conflict với feature branches khác → cần merge thường xuyên.
- Test cũ có thể cần update nhiều → dành thời gian ở Giai đoạn 1.6 và 10.
- Docker build có thể cần tweak nhiều lần.

**Kết quả kỳ vọng**:
- Code base dễ đọc, dễ bảo trì hơn.
- CI/CD tự động phát hiện regression.
- Người mới onboard trong 1–2 giờ thay vì 1–2 ngày.
- Sẵn sàng cho v9 với features mới (multi-GPU, online eval, ...).
