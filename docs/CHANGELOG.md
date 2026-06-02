# Changelog

## v9.0.0 (Refactor)

### Breaking changes
- Renamed top-level packages to `symbiopan/` subpackage: update `from data.` → `from symbiopan.data.`, `from models.` → `from symbiopan.models.`, etc.
- Moved `utils/losses.py` → `symbiopan/losses/` (segmentation + multitask split)
- `LOSS_MULTIPLIERS` now has 5 elements (was 4 + `[0.0]` hack)
- Removed `INTERNAL_TISSUE_ID_TO_NAME` (duplicate of `PUMA_TISSUE_ID_TO_NAME`)
- Removed `RARE_TISSUE_IDS_PUMA` (use `sorted(RARE_TISSUE_IDS)` instead)
- Removed `BoundaryAttentionModule` and boundary output (unused)
- Removed `extract_intermediate_features()` from encoder (unused)
- Removed `make_or_load_group_split_with_test()` (unused, re-added for notebooks)
- Removed `parse_stage1_args()` and `get_stage1_main()` (unused)
- Removed `requirements.txt` (deps in `pyproject.toml`)
- Scripts renamed: `run_preprocess.py` → `preprocess.py`, `run_stage1.py` → `train_stage1.py`, `run_inference.py` → `infer_wsi.py`

### Bug fixes
- Fixed no-op ViT freezing in `train_loop.py`: `if/else` now properly sets `eval()` and `requires_grad=False`
- Replaced global mutable `_CONTEXT_CACHE` with per-instance cache with size limit
- Removed `patch_autocast_for_bf16` monkey-patch, replaced with strategy pattern

### Improvements
- All inference paths configurable via environment variables
- Moved `get_device()` from `configs/defaults.py` to `symbiopan/common/device.py`
- Moved `linear_ramp` to `symbiopan/modules/scheduler.py`
- `SITE_NAMES` now single source of truth in `symbiopan/data/constants.py`
- Tests updated to use `symbiopan.` import paths
- Ruff `select` expanded from `["E", "F", "W", "I"]` to include `["B", "UP", "N", "C4", "SIM"]`

### Cleanup
- Removed empty `data/` root directory (use `Dataset/` for raw data)
- Removed `test_run.sh` (use `Makefile` targets)
- Created `docs/CHANGELOG.md`
- Created `tests/conftest.py` with shared fixtures

## v8.0.0 "CellPath"

- Virchow2 ViT-H/14 encoder with fine-tuning
- ConvNeXt-Tiny backbone (28.6M params)
- DeepLabV3+ tissue decoder + CellViT++ nuclei decoder
- Context ROI encoder (EfficientNet-B0)
- Stain augmentation (HEStain)
- Warm-up + cosine decay LR
- TTA (8 augmentations)

## v7.0.0

- Frozen UNI ViT-L/16 encoder
- ConvNeXt-Atto backbone (3.7M params)
- Cellpose flow generation + Stage 2 refinement
- Cosine annealing LR (no warm-up)
