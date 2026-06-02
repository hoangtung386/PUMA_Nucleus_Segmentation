# Changelog

## v9.0.0 (Refactor)

### Breaking changes
- Top-level packages renamed to `symbiopan/` subpackage. Update imports:
  - `from data.…` → `from symbiopan.data.…`
  - `from models.…` → `from symbiopan.models.…`
  - `from inference.…` → `from symbiopan.inference.…`
  - `from training.…` → `from symbiopan.training.…`
  - `from utils.losses` → `from symbiopan.losses`
  - `from utils.metrics` → `from symbiopan.metrics`
  - `from utils.sc_dfa | scheduler_utils | split_utils` → `from symbiopan.modules.…`
- `MultiTaskUncertaintyLoss` constructor now takes `loss_multipliers`, `focal_tversky_tissue`, `focal_tversky_nuclei`, `focal_bce`, `smooth_l1_beta` so all loss hyper-parameters are config-driven.
- `PUMAMetrics(selection_score_weights=...)` now configurable; defaults `(0.20, 0.25, 0.55)`.
- `UnifiedPanopticNet` accepts `num_sites` and `site_embed_dim` (previously hardcoded to 9 / 256).
- `load_stage1(checkpoint_path, device, cfg=None)` now takes an optional `InferenceConfig` instead of hardcoded arguments.
- `Stage1Config` now holds: `virchow2_model_name`, `num_tissue`, `num_nuclei`, `num_sites`, `site_embed_dim`, `loss_multipliers`, `focal_tversky_tissue`, `focal_tversky_nuclei`, `focal_bce`, `smooth_l1_beta`, `selection_score_weights`.
- `InferenceConfig.cp`, `input_dir`, `output_dir`, `site_classifier_cp` read defaults from environment variables (`SYMBIOPAN_*`).
- `LOSS_MULTIPLIERS` now has 5 entries — moved to `Stage1Config.loss_multipliers`.
- Scripts renamed: `run_preprocess.py` → `preprocess.py`, `run_stage1.py` → `train_stage1.py`, `run_inference.py` → `infer_wsi.py`.

### Bug fixes
- **CRITICAL**: Fixed `nn.Embedding(num_sites=9, embedding_dim=256)` → `nn.Embedding(num_embeddings=9, embedding_dim=256)` in `panoptic_net.py`. The previous form raised `TypeError` and prevented model instantiation.
- Fixed misleading `test_hierarchical_fpn` that passed `vit_intermediate` as a positional `img_size` argument.
- Fixed no-op ViT-freeze logic in `train_loop.py` — now properly sets `eval()` and `requires_grad=False`.
- Replaced global `_CONTEXT_CACHE` with a per-`PUMADataset` instance cache (size-bounded FIFO).

### Removed (dead code)
- `INTERNAL_TISSUE_ID_TO_NAME` (duplicate of `PUMA_TISSUE_ID_TO_NAME`)
- `RARE_TISSUE_IDS_PUMA` (use `sorted(RARE_TISSUE_IDS)`)
- `BoundaryAttentionModule` and the boundary decoder output
- `extract_intermediate_features()` on `UnifiedPanopticEncoder`
- `_simple_patch_embed` + `_patch_proj` fallback path
- `make_or_load_group_split_with_test()`
- `parse_stage1_args()` and `get_stage1_main()`
- `requirements.txt` (consolidated into `pyproject.toml`)
- `notebooks/train_model.ipynb` (1232 lines) → replaced with `notebooks/01_quickstart.ipynb`
- `tests/__pycache__/`, `symbiopan.egg-info/`, leftover `{cli_src}` directory, empty `symbiopan/configs/`

### Improvements
- Added `symbiopan/common/` (logging, device, types, exceptions).
- Added `symbiopan/inference/tta.py` (split from `infer_wsi.py`).
- Added `tests/conftest.py` with `device`, `dummy_batch`, `temp_dataset_dir` fixtures.
- Reorganised `tests/` into subpackages mirroring source layout (`test_data/`, `test_inference/`, `test_losses/`, `test_metrics/`, `test_models/`).
- Added GitHub Actions workflows: `lint.yml`, `test.yml` (matrix Python 3.11 + 3.12), `docker-build.yml`.
- `Makefile` now exposes `INPUT_DIR` / `OUTPUT_DIR` env knobs.
- `.gitignore` revised: covers caches, model artifacts, secrets, IDE files.
- Docker `inference.sh` and `Dockerfile` read paths from `SYMBIOPAN_*` env vars.

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
