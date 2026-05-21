# SymbioPan — Colab Pro Training Guide (≤4 Hours)

This guide explains how to train both stages of SymbioPan end-to-end on a **Colab Pro A100 80GB GPU** within a **total wall-clock budget of 4 hours**.

---

## 1. Time Budget Overview

| Phase | Est. Time | Notes |
|---|---|---|
| Environment setup | 5 min | Mount Drive, clone repo, install deps |
| GPU detection & config | 1 min | Auto-detect, batch/worker/epoch overrides |
| **Stage 1 training** | **30–45 min** | 30 epochs, batch=64, frozen ViT-L/16 |
| GPU cache clear | 1 min | `gc.collect() + torch.cuda.empty_cache()` |
| **Stage 2 training** | **5–10 min** | 20 epochs, batch=128, small UNet refiner |
| Verify + manifest + Docker export | 2 min | Checkpoint verification, manifest.json, copy |
| **Total** | **~45–65 min** | Well within 4 h budget |

### Why so fast?

- **Stage 1: Frozen ViT encoder.** The UNI ViT-L/16 (307M params) has `requires_grad=False` — no backward pass through it. Only the CNN backbone (~3M params), FPN, and decoders are trained. bf16 + TF32 on A100 further accelerates compute.
- **Stage 2: Small UNet.** `ResidualNucleiRefinerUNet` has tens of thousands of params. 20 epochs at batch=128 finishes in minutes.
- **Optimized DataLoader.** `persistent_workers=True`, `prefetch_factor=2`, `drop_last=True`, capped at 8 workers to avoid OOM on Drive FUSE.
- **Gradient checkpointing on ViT** (enabled in `stage1_trainer.py` via `_enable_grad_ckpt`).

---

## 2. Prerequisites

### 2.1. Google Drive Layout

Before launching the notebook, ensure this Drive structure exists:

```
MyDrive/
├── PUMA/
│   ├── dataset_processed/        ← REQUIRED (preprocessed .npy files)
│   │   ├── images/
│   │   ├── tissue_sem/
│   │   ├── nuclei_nc/
│   │   ├── nuclei_np/
│   │   ├── nuclei_hv/
│   │   ├── cellpose_flow/
│   │   └── site_types.npy
│   └── checkpoints/              ← WILL BE CREATED (model saves)
```

### 2.2. Processed Dataset

Preprocessing must be done **once** before training. You can either:
- Run the preprocessing cells in the notebook (Section 2a) — takes ~30–60 min (required only once).
- Or upload a preprocessed `dataset_processed/` folder to Drive.

> **Important:** Cellpose flow generation is required for training. Set `PREPROCESS_DEFAULT_CONFIG.generate_cellpose_flows = True` (default).

### 2.3. Colab Pro (G4 / A100 80GB)

- Runtime type: **A100 GPU** (High-RAM, 100 GB VRAM tier).
- Make sure bf16 is supported (A100+).

---

## 3. Step-by-Step Notebook Execution

Open `notebooks/train_model.ipynb` in Colab and run cells sequentially.

### Section 0: Colab Setup

- Mount Drive.
- Clone repo (uses `git clone` or `git pull`).
- Install dependencies from `requirements.txt`.
- Override PATHS to point to `dataset_processed` and `checkpoints` on Drive.

> **Troubleshooting:** If Drive mount fails, re-run the mount cell and grant access.

### Section 0b: GPU Auto-Configuration

This calls `detect_gpu_setup()` which:

1. Detects GPU name and VRAM.
2. Sets `batch_size=64` for Stage 1, `batch_size=128` for Stage 2.
3. Caps `num_workers=8`.
4. **Overrides epochs and schedule** to fit the 4-hour budget:

   | Config | Default (50/30) | 4-Hour Mode (30/20) |
   |---|---|---|
   | S1 epochs | 50 | **30** |
   | S1 focal start/full | 10/16 | **6/10** |
   | S1 sc_dfa start/full | 15/22 | **9/13** |
   | S1 prior start/full | 20/28 | **12/17** |
   | S1 samples_per_epoch_multiplier | 1.0 | **3.0** |
   | S2 epochs | 30 | **20** |
   | S2 keep_lambda_decay_epochs | 30 | **20** |
   | S2 alpha_warmup_epochs | 30 | **20** |
   | S2 samples_per_epoch_multiplier | 2.5 | **4.0** |

5. Enables TF32 + high matmul precision + bf16 autocast.

### Sections 1–11: Import & Verification

These cells import all modules and verify the codebase works. They are fast (<1 min total) and do not require a GPU.

### Section 12: Stage 1 Training

- **Duration:** ~30–45 min.
- **Outputs saved to Drive:**
  - `puma_epoch_best_s1.pth` (entity model, best validation selection score)
  - `puma_epoch_last_s1.pth` (entity model, last epoch)
- **What happens:**
  1. Dataset loads, group split is created (if first run).
  2. UnifiedPanopticNet is built + UNI ViT-L weights loaded and frozen.
  3. Per epoch: train → `apply_smooth_schedule` (focal, SC-DFA, prior ramps) → validate.
  4. Best model tracked by `selection_score = 0.20×tissue_dice + 0.25×nuclei_dice + 0.55×rare_macro_dice`.
  5. Entity model saved with `_metadata` dict attached.

> **Monitor:** Watch the validation `selection_score`. If it plateaus early, you could manually stop (not needed for 30 epochs).

### GPU Cache Clear (after Stage 1)

- Frees GPU memory before Stage 2.

### Section 13: Stage 2 Training

- **Duration:** ~5–10 min.
- **Requires:** `puma_epoch_best_s1.pth` from Stage 1.
- **Outputs saved to Drive:**
  - `nuclei_refiner_residual_best.pth` (entity model)
  - `nuclei_refiner_residual_last.pth` (entity model)
- **What happens:**
  1. Stage 1 model is loaded in eval mode (frozen).
  2. ResidualNucleiRefinerUNet initialized (zero-initialized final conv).
  3. Training uses KL divergence + distillation + alpha/keep_lambda schedules.
  4. Best model selected by `improvement_score` over Stage 1 baseline.

### Section 13b: Manifest + Docker Export

- **Duration:** ~2 min.
- Saves `training_manifest.json` to Drive.
- Copies entity models to `/opt/app/checkpoints/` for Docker packaging:
  - `best_model.pth` ← Stage 1 best entity model
  - `nuclei_refiner_residual_best.pth` ← Stage 2 best entity model

---

## 4. Configuration Details

### 4.1. Entity Model Format

All checkpoints are **entity models** (`torch.save(model, path)`), not state dicts. This means:

```python
# Loading (no architecture needed):
model = torch.load("/opt/app/checkpoints/best_model.pth", map_location="cuda")
model.eval()
# model is a fully reconstructed UnifiedPanopticNet, ready for inference.
```

Metadata is embedded in the model object:
```python
meta = getattr(model, "_metadata", {})
# meta = {"epoch": 30, "best_score": 0.75, "inference_config": {...}}
```

### 4.2. Schedule Scaling Rationale

The smooth-schedule milestones (focal start/full, SC-DFA start/full, prior start/full) were designed for 50 epochs. They are linearly scaled to 30 epochs (0.6×) to preserve the same **relative pacing**:

```
Original (50 ep):  focal 10→16 (20-32%)  →  Scaled (30 ep): 6→10 (20-33%)
Original (50 ep):  sc_dfa 15→22 (30-44%) →  Scaled (30 ep): 9→13 (30-43%)
Original (50 ep):  prior 20→28 (40-56%)  →  Scaled (30 ep): 12→17 (40-57%)
```

This means focal loss, SC-DFA, and spatial prior activate at the same relative point in training and reach full strength at the same relative point.

### 4.3. GPU Optimizations Summary

| Optimization | Where | Benefit |
|---|---|---|
| `torch.set_float32_matmul_precision('high')` | Notebook `detect_gpu_setup()` | TF32 on matmuls, ~8× FP32 throughput |
| `cudnn.allow_tf32 = True` | Notebook `detect_gpu_setup()` | TF32 on convolutions |
| bf16 autocast (patched `_autocast_context`) | Notebook after GPU setup | Lower memory, stable mixed precision |
| ViT gradient checkpointing | `stage1_trainer.py` `_enable_grad_ckpt()` | Trade compute for memory, enables larger batch |
| `persistent_workers=True`, `prefetch_factor=2` | Both trainer DataLoaders | Faster data loading |
| `num_workers=8` | Notebook `detect_gpu_setup()` | Balances I/O vs RAM (Drive FUSE limit) |
| `gc.collect()` + `torch.cuda.empty_cache()` | Between stages | Prevents OOM accumulation |

---

## 5. Troubleshooting

### 5.1. "CUDA Out of Memory"

- If Stage 1 crashes with OOM at batch=64, reduce batch size:
  ```python
  detect_gpu_setup(force_batch_size=48)  # or 32
  ```
- If Stage 2 crashes (unlikely), reduce from 128 to 64.

### 5.2. "Killed" / OOM Killer (DataLoader workers)

- If the process is killed during data loading, reduce `num_workers`:
  ```python
  # in detect_gpu_setup()
  n_workers = 4  # instead of min(8, cpu_count)
  ```
- Or pin the notebook: `OMP_NUM_THREADS=1` env var (already set in trainer `main()`).

### 5.3. Slow Data Loading

- The first epoch is always slowest (OS page cache cold). Subsequent epochs are faster.
- If I/O is severely bottlenecked (>30 min/epoch), copy dataset locally:
  ```python
  # !cp -r /content/drive/MyDrive/PUMA/dataset_processed /content/
  # Then update PATHS.data_dir to /content/dataset_processed
  ```

### 5.4. Checkpoint Verification Fails

Entity model saves are CPU-copied by `safe_torch_save_entity`. If you get a CUDA error during load:
```python
obj = torch.load(path, map_location="cpu", weights_only=False)
```
This always works since the model was saved from CPU.

---

## 6. FAQ

**Q: Can I train longer than 4 hours?**
A: Yes. Increase `epochs` in `detect_gpu_setup()` to 50 (S1) / 30 (S2) for the default full schedule. Total time: ~2–3 hours for S1, ~30 min for S2.

**Q: What if I have a V100 (32 GB) instead of A100?**
A: Change `detect_gpu_setup()` to set batch=32, epochs=30 (S1) / 20 (S2). Total time: ~2–3 hours.

**Q: Do I need to re-run preprocessing?**
A: Only once. The processed dataset can be reused across training runs.

**Q: Can I resume training from a checkpoint?**
A: Yes — set `STAGE1_DEFAULT_CONFIG.resume = "/path/to/checkpoint.pth"` or use `force_batch_size` to resume with different settings. Entity models are NOT compatible with resume (they lack optimizer state). Use the state-dict checkpoints for resume (if you change `save_entity_only=True` to `False`).

**Q: How do I package for Docker submission?**
A: Run Section 14 in the notebook. It copies:
- `best_model.pth` (Stage 1 entity model)
- `nuclei_refiner_residual_best.pth` (Stage 2 entity model)

To `/opt/app/checkpoints/`. Then build your Docker image with these files.

---

## 7. Expected Results

With 30 (S1) + 20 (S2) epochs:

| Metric | Expected Range |
|---|---|
| avg_tissue_dice | 0.85–0.92 |
| avg_nuclei_dice | 0.55–0.70 |
| rare_macro_dice | 0.35–0.55 |
| selection_score | 0.48–0.65 |
| S2 improvement (rare macro) | +0.02 to +0.08 |

These are based on the dataset difficulty (class imbalance, rare classes) and reduced epoch count. Running the full 50/30 schedule would yield 1–3% higher scores but takes 2–3× longer.
