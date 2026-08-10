# PUMA Version 13

Clean V13 pipeline for Stage-1 nuclei detection and Stage-2 nuclei classification.

## Run order

1. `00_Preprocess.ipynb`
2. `01_Train_Stage1.ipynb`
3. `02_Train_Stage2.ipynb`
4. `03_Evaluate_Infer.ipynb`

## Stage 1

Stage 1 contains one architecture only:

- `A1_IFCRN_PP`

Five outer folds are retained for OOF cross-fitting. For each outer fold:

- the outer fold is kept untouched for OOF prediction;
- one of the remaining folds is used as inner validation;
- the other three folds train A1;
- inner validation selects the checkpoint, heatmap threshold, and local-max radius;
- the selected model predicts the untouched outer fold.

Combining the five outer predictions gives leakage-safe Stage-1 candidates for all labeled ROIs.

Stage-1 outputs are written to `PUMA_stage1_training_outputs/`.

## Stage 2

Stage 2 uses one fixed 80/20 development split. It does not repeat every experiment over the five Stage-1 folds.

The split is created once with:

```python
ensure_v13_split(
    runtime,
    force=False,
    val_fraction=0.20,
    seed=2026,
    check_sources=True,
)
```

Use the same split for every Stage-2 experiment.

### Experiments

| Experiment | Change |
|---|---|
| `V13_01_META_NEW_SPLIT_FROZEN` | V64 + V128 + geometry; frozen UNI2-h |
| `V13_02_META_CONTEXT_NEW_SPLIT_FROZEN` | Adds V256 context |
| `V13_03_META_CONTEXT_CBFOCAL_FROZEN` | Class-balanced focal type loss |
| `V13_04_META_CONTEXT_CBCE_FROZEN` | Class-balanced CE type loss |
| `V13_05_META_CONTEXT_RAREBOOST_FROZEN` | Stronger rare-class sampling |
| `V13_06_META_CONTEXT_LORA_R8_B4` | LoRA rank 8 on the last 4 UNI2-h blocks |

Choose the queue in `02_Train_Stage2.ipynb`:

```python
STAGE2_EXPERIMENTS_TO_RUN = [
    'V13_01_META_NEW_SPLIT_FROZEN',
    'V13_02_META_CONTEXT_NEW_SPLIT_FROZEN',
]
```

Add or remove names as needed.

## RTX 3090 settings

For one RTX 3090 24 GB, use:

```python
STAGE2_PARALLEL_RUNS = 1
LORA_PARALLEL_RUNS = 1
```

V13 targets:

- effective batch: 256
- Stage-2 physical micro-batch: 256
- UNI2-h encoder micro-batch: 256

The CUDA OOM fallback can reduce the physical micro-batch while preserving effective batch 256 through gradient accumulation.

Approximate single-run planning ranges on one RTX 3090:

| Experiment | Peak VRAM | Development time |
|---|---:|---:|
| V13_01 | 7–10 GB | 24–38 h |
| V13_02 | 8–11 GB | 36–55 h |
| V13_03 | 8–11 GB | 36–56 h |
| V13_04 | 8–11 GB | 36–55 h |
| V13_05 | 8–11 GB | 34–53 h |
| V13_06 | 18–23.5 GB at batch 256 | 65–105 h |

These are planning estimates; candidate count, I/O, caching, and OOM fallback can change actual runtime.

## Final model

After reviewing the fixed-split results:

1. lock one Stage-2 experiment;
2. retrain that exact configuration on all labeled ROIs using the complete Stage-1 OOF candidates;
3. keep the five A1 fold models as the deployment detector ensemble;
4. use the single retrained Stage-2 model for final classification.

The final Stage-2 lock is:

`stage2_v13_final_lock.json`

## Main outputs

### Preprocessing

`PUMA_outputs/`

Key artifacts include:

- `puma_rgb_images.npy`
- `puma_instance_maps.npy`
- `puma_class_maps.npy`
- `puma_centroid_heatmaps.npy`
- `puma_centroid_match_disks_15px.npy`
- `puma_roi_manifest.npy`
- `puma_nuclei_centroids.npy`
- `puma_roi_centroid_offsets.npy`
- `puma_fold_assignments.npy`

### Stage 1

`PUMA_stage1_training_outputs/`

Key artifacts:

- five `stage1_best_A1_IFCRN_PP_fold*_seed0.pt` checkpoints
- `stage1_results.csv`
- `stage1_lock.json`
- `stage1_oof_candidates.npy`
- `stage1_oof_candidates_metadata.json`

### Stage 2

`PUMA_stage2_V13_outputs/` when set by the notebooks.

Key artifacts:

- `stage2_v13_results.csv`
- `stage2_v13_lock.json`
- V13 development checkpoints and prediction files
- `stage2_v13_final_<experiment>_seed0.pt`
- `stage2_v13_final_lock.json`
- `puma_final_predictions.npy`
- `puma_final_predictions.csv`

## Colab and local server

The Colab mount/import blocks are intentionally retained. Change only `PROJECT_DIR` for your environment.

Install dependencies with:

```bash
pip install -r requirements_colab.txt
```
