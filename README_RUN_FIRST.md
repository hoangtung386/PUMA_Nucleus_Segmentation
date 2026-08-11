# PUMA Version 13.2

V13.2 is the streamlined two-stage PUMA nuclei detection/classification pipeline derived
from V13.1. It keeps the leakage-safe five-fold Stage-1 OOF design and the optimized fixed
Stage-2 train/validation split, then improves detector refitting, rare-class exposure,
curriculum-aware optimization, speed, and Grand-Challenge inference.

> A Vietnamese translation of this document is in [README.md](README.md). The two files
> share the same structure; when editing one, edit the other too.

## Current state

| Stage | Notebook | State |
|---|---|---|
| Preprocess | `00_Preprocess.ipynb` | works, run here |
| Stage 1 | `01_Train_Stage1.ipynb` | V13.2 code, not yet run on this workstation |
| Stage 2 | `02_Train_Stage2.ipynb` | V13.2 code, not yet run on this workstation |
| Evaluate / infer | `03_Evaluate_Infer.ipynb` | V13.2 code, not yet run on this workstation |

The V13.2 core replaced the V13.1 Stage-1/Stage-2 code in this repository. Only
`puma/data/preprocess.py` is local: V13.2 never touched it, and it carries this project's
capacity-constrained fold assignment. See *What has and has not been verified*.

Part 1 is how to run the project on a workstation. Part 2 is the V13.2 pipeline reference.

---

# Part 1 — Running on a workstation

Target: one RTX 3090 Ti 24 GB, Linux, NVIDIA driver 525 or newer.

## 1. Send the whole folder to the workstation

Shipping this folder as-is works, and is the default path in this document. Nothing needs
editing after the transfer: `setup_local.sh` rebuilds the environment, and the notebooks
find the project root themselves through `Path.cwd()`.

**Use `rsync -a` or `tar`, not `scp -r`/`cp -r`/zip.** The one reason, but it matters: the
repo contains a `Dataset -> dataset` symlink. Tools that follow symlinks duplicate the whole
dataset (32 GB becomes 64 GB), or leave `Dataset` as a real directory that `setup_local.sh`
can no longer fix, since it only creates the symlink when `Dataset` does not exist.

```bash
rsync -a --info=progress2 ./ user@workstation:/path/to/SymbioPan/
```

Two directories in there are wasted transfer, but **break nothing**:

| Directory | Size | What happens on the workstation |
|---|---:|---|
| `.venv/` | 7.2 GB | A virtualenv hard-codes absolute paths, so it is broken on another machine. `setup_local.sh` **deletes and rebuilds** it in step 1, so the copy only costs bandwidth. |
| `PUMA_outputs/` | 1.9 GB | May be reused, but the cache key includes source-file mtimes, which most copy tools do not preserve exactly, so `00_Preprocess.ipynb` will likely rebuild anyway — 25 seconds. |

To keep the transfer small, exclude those plus two dataset directories that are not read
anywhere in `puma/` (`tif_context_ROIs` 21 GB — the Stage-2 V256 view is cropped from the
same 1024×1024 ROI; and `geojson_tissue` 74 MB — V13.2 trains no tissue model). That leaves
about **1.1 GB** instead of 32 GB:

```bash
rsync -a --info=progress2 \
  --exclude '.venv' \
  --exclude 'PUMA_outputs' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.ipynb_checkpoints' \
  --exclude 'dataset/01_training_dataset_tif_context_ROIs' \
  --exclude 'dataset/01_training_dataset_geojson_tissue' \
  ./ user@workstation:/path/to/SymbioPan/
```

After the transfer, check three things on the workstation before doing anything else:

```bash
cd /path/to/SymbioPan
ls -l Dataset                                          # must read: Dataset -> dataset
ls dataset/01_training_dataset_tif_ROIs/*.tif | wc -l   # must be 205
ls dataset/01_training_dataset_geojson_nuclei/*.geojson | wc -l   # must be 205
```

If `Dataset` is a real directory rather than a symlink, replace it:
`rm -rf Dataset && ln -s dataset Dataset`.

Stage 2 additionally needs the UNI2-h checkpoint under
`PUMA_pretrained_checkpoints/UNI2-h/`. If the copy already carries it, Stage 2 runs offline;
otherwise `02_Train_Stage2.ipynb` downloads it once (needs `HF_TOKEN`, see section 3).

## 2. Build the environment

Dependencies are managed with [uv](https://docs.astral.sh/uv/), not with `pip` inside
the notebooks. Install uv once if the workstation does not have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Then, from the project root:

```bash
cd /path/to/SymbioPan
bash setup_local.sh
```

The script is idempotent and does all of the following:

1. deletes any `.venv` carried over from another machine;
2. creates a fresh `.venv` on CPython 3.11, downloaded by uv (no system Python, no
   `apt`) — 3.11 rather than 3.13 for wider `rasterio`/`timm` wheel coverage;
3. installs PyTorch with CUDA (`cu128` by default; see below for older drivers);
4. installs `requirements_colab.txt`, JupyterLab, ipykernel, ipywidgets;
5. removes any stale `symbiopan` Jupyter kernel and re-registers it against *this*
   machine's `.venv`;
6. creates the `Dataset -> dataset` symlink if missing;
7. prints the full GPU inventory and which GPU the notebooks will pick, then torch's
   view of the same devices, bf16 support, all dependency imports, and the dataset file
   counts.

`requirements_colab.txt` intentionally omits torch, because Colab preinstalls it. On a
workstation torch must come from step 3.

**Older driver.** `cu128` wheels need driver 525 or newer. Check with `nvidia-smi`, and
if the driver is older, build against CUDA 12.6 instead:

```bash
CUDA_BACKEND=cu126 bash setup_local.sh
```

Expected tail of a good run on a two-GPU workstation:

```
python           3.11.15  (/path/to/SymbioPan/.venv/bin/python)
gpus visible     2
  GPU 0          NVIDIA GeForce RTX 3090 Ti  24 GB
  GPU 1          NVIDIA GeForce RTX 3090 Ti  24 GB
notebooks will use GPU 1 (NVIDIA GeForce RTX 3090 Ti)
                 2 GPUs detected; using preferred GPU 1
torch            2.11.0+cu128
cuda available   True
  torch cuda:0    NVIDIA GeForce RTX 3090 Ti  23.6 GB  sm_86  bf16=True
  torch cuda:1    NVIDIA GeForce RTX 3090 Ti  23.6 GB  sm_86  bf16=True
deps             all 12 imports OK
tif ROIs         205 files  [OK]  .../Dataset/01_training_dataset_tif_ROIs
geojson nuclei   205 files  [OK]  .../Dataset/01_training_dataset_geojson_nuclei
```

The script itself does not mask any GPU — it lists them all so the numbering can be
checked. Masking happens in the notebooks.

## 3. Start JupyterLab

```bash
cd /path/to/SymbioPan
./.venv/bin/jupyter lab
```

Start it **from the project root**. All four notebooks fall back to
`PROJECT_DIR = Path.cwd()` off Colab and refuse to continue if `puma/` is not there, so a
wrong working directory fails immediately with a clear message rather than importing the
wrong package.

All four notebooks select the `SymbioPan (uv .venv)` kernel. `00`, `01` and `02` assert in
their second cell that `sys.executable` really is `.venv/bin/python`, so running on the
wrong kernel fails loudly instead of failing later on a missing import.

There is no `pip` inside the uv venv, so `%pip install` **will not work** in these
notebooks. That is deliberate. To add a package:

```bash
VIRTUAL_ENV=.venv uv pip install <package>
```

No `HF_TOKEN` is needed for preprocessing or Stage 1. It is only required from Stage 2
onwards, where the gated `MahmoodLab/UNI2-h` checkpoint is downloaded: accept the
repository terms on Hugging Face, then `export HF_TOKEN=hf_...` before starting
JupyterLab.

## 4. Which GPU the notebooks use

**On a machine with two or more GPUs, training runs on GPU 1 by default**, leaving GPU 0
for the display and for other jobs. With a single GPU it falls back to GPU 0. The
bootstrap cell prints exactly what it picked:

```
GPUs detected        : 2
  0: NVIDIA GeForce RTX 3090 Ti (24 GB)
  1: NVIDIA GeForce RTX 3090 Ti (24 GB)
CUDA_VISIBLE_DEVICES : 1
CUDA_DEVICE_ORDER    : PCI_BUS_ID
training device      : physical GPU 1 (NVIDIA GeForce RTX 3090 Ti)  -> cuda:0 inside torch
reason               : 2 GPUs detected; using preferred GPU 1
```

To change the default, edit one line in the bootstrap cell of the notebook:

```python
PREFERRED_GPU_INDEX = 1     # 0 for the first GPU, 1 for the second, and so on
```

To override for a whole session without editing anything, set the variable before
launching JupyterLab — an existing `CUDA_VISIBLE_DEVICES` is always respected:

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/jupyter lab
```

Three details are worth knowing, because they are the usual sources of confusion:

- **`CUDA_DEVICE_ORDER=PCI_BUS_ID` is pinned.** `nvidia-smi` numbers GPUs by PCI bus id,
  but CUDA's default is `FAST_FIRST`, so without pinning, "GPU 1" can mean different
  cards in the two tools. With it pinned, the index in the notebook, in
  `CUDA_VISIBLE_DEVICES`, and in `nvidia-smi` all agree.
- **The selected GPU becomes `cuda:0`.** Masking hides the others entirely, so
  `resolve_device()` and every `torch.device('cuda')` in `puma/` pick the right card with
  no further changes. Seeing `cuda:0` in the logs while training on physical GPU 1 is
  expected.
- **Selection must happen before `torch` is imported**, because `CUDA_VISIBLE_DEVICES` is
  only read when the CUDA driver initialises. `puma/gpu.py` therefore imports no torch at
  all, and the bootstrap cell runs before any torch import. If the bootstrap cell is
  re-run in a kernel that has already trained, the switch silently would not apply — so
  it prints a warning in that case, and the environment-check cell raises if torch ends
  up seeing more than one device. In short: **Restart Kernel, then Run All** when
  changing GPUs.

`setup_local.sh` prints the full GPU inventory and states which one the notebooks will
choose, so this can be confirmed before running anything.

The bootstrap cell of `01` and `02` was executed against injected GPU inventories, with these
results:

| GPUs on the machine | `CUDA_VISIBLE_DEVICES` | GPU used for training |
|---:|---|---|
| 2 | `1` | physical GPU 1 |
| 4 | `1` | physical GPU 1 |
| 1 | `0` | physical GPU 0 (fallback) |
| 0 | unset | CPU |

Both notebooks behave identically because they share one bootstrap cell. Nothing in `puma/`
hard-codes `cuda:1`, calls `set_device()`, or uses `DataParallel` — everything goes through
`resolve_device()`, which returns a bare `cuda`, so the masked GPU is the GPU that gets used.

## 5. Run order

1. `00_Preprocess.ipynb`
2. `01_Train_Stage1.ipynb`
3. `02_Train_Stage2.ipynb`
4. `03_Evaluate_Infer.ipynb`

The whole path from a freshly transferred folder to a model:

```bash
cd /path/to/SymbioPan
bash setup_local.sh          # builds .venv + kernel, a few minutes
export HF_TOKEN=hf_...       # Stage 2 only
./.venv/bin/jupyter lab
# In JupyterLab, kernel "SymbioPan (uv .venv)", Run All in order:
#   00_Preprocess.ipynb    ~25 s     -> PUMA_outputs/
#   01_Train_Stage1.ipynb            -> 5 detector checkpoints + OOF candidates
#   02_Train_Stage2.ipynb            -> classifier (50-epoch screening, 100-epoch winner)
#   03_Evaluate_Infer.ipynb          -> final submission model (after enabling switches)
```

`00_Preprocess.ipynb` is **required before** `01_Train_Stage1.ipynb`, even when the copy
brought `PUMA_outputs/` along: the cache key includes source-file mtimes so it will likely
rebuild anyway, and 25 seconds is far cheaper than discovering a missing artifact mid-run.
Stage 1 opens the `.npy` artifacts through `PumaNpyStore.open()`, which raises
`Missing preprocessed artifacts: [...]. Run 00_Preprocess.ipynb first.` when they are
absent. `puma_fold_assignments.npy` in particular is what defines the five folds.

Every notebook opens with the same bootstrap cell, which works both here and on Colab:

```python
try:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_DIR = Path('/content/drive/MyDrive/Research/PUMA')
except ImportError:
    PROJECT_DIR = Path.cwd().resolve()
```

### `00_Preprocess.ipynb` — Run All

Takes about 25 seconds on 12 cores (`preprocessing_workers=0` uses every logical core).
Writes 1.9 GB to `PUMA_outputs/`. Expected:

```
205 GeoJSON / 205 TIFF / 205 matched pairs, 97193 annotated features
all ROIs 1024×1024
fold sizes [41, 41, 41, 41, 41]      size imbalance ratio 1.0
folds missing a class entirely: none
```

`FORCE_PREPROCESS = False` reuses a valid cache. Set it to `True` only to rebuild
deliberately — the cache key covers the data config, the artifact schema version, and
the source TIFF/GeoJSON inventory, so genuine input changes already trigger a rebuild.

### `01_Train_Stage1.ipynb` — Run All, no edits needed

Pick the `SymbioPan (uv .venv)` kernel → Run All. Seven cells in order: bootstrap (GPU
selection) → environment check → runtime + preflight → **fold integrity** → train → OOF +
Stage-1 deployment lock.

The fold-integrity cell prints per-fold ROI counts, melanoma type, and all ten class counts,
and raises on a degenerate split. `run_stage1_a1()` repeats the same check internally, so it
cannot be skipped by running cells out of order.

Trains five `A1_IFCRN_PP` outer folds sequentially. **Each outer fold trains twice**: an
inner-validation pass that selects epoch and post-processing, then a reset and a refit on all
four non-outer folds. That is ten training runs, not five — budget for it.

When it finishes, `PUMA_stage1_training_outputs/` holds:

```
stage1_best_A1_IFCRN_PP_fold{0..4}_seed0.pt    5 detector checkpoints
stage1_results.csv                             per-fold metrics
stage1_lock.json                               Stage-1 deployment lock
stage1_oof_candidates.npy                      REQUIRED input for Stage 2
```

`stage1_oof_candidates.npy` is what connects Stage 1 to Stage 2. The last cell calls
`validate_full_oof(runtime)`, which raises if any ROI still lacks an out-of-fold prediction —
so **keep the full `run_folds=(0, 1, 2, 3, 4)`**; one missing fold means Stage 2 cannot run.

To gauge the cost before committing to all five folds, set `run_folds=(0,)` in the runtime
cell, run it, watch VRAM and wall-clock, then restore all five and rerun.

If the session dies: reopen the notebook and Run All. Every epoch writes a resume checkpoint
and completed folds are skipped when the configuration hash matches, so it continues rather
than restarting.

### `02_Train_Stage2.ipynb` — 50-epoch screening, then the 100-epoch winner

Requires Stage 1 to be finished, since Stage 2 reads `stage1_oof_candidates.npy`.

**Pass 1 — screening.** Keep the default `STAGE2_EPOCHS = 50` and Run All. It creates the
fixed 80/20 split, fetches the UNI2-h checkpoint, then trains all four experiments
sequentially and prints the ranking by `macro_f1`.

**Pass 2 — the winner.** Read the ranking, then edit exactly two lines in the runtime cell:

```python
STAGE2_EPOCHS = 100
WINNER_EXPERIMENT = 'V13_2_02_META_RARE_BS'   # replace with whichever ranked first
```

Restart Kernel, then Run All. At `STAGE2_EPOCHS = 100` the experiment-selection cell switches
itself from "all four" to "the winner only", so nothing else needs changing.
`create_runtime()` accepts only 50 or 100 — anything else is rejected immediately with a
clear message.

The 100-epoch model must be trained **from scratch**, not resumed from the 50-epoch pass:
the curriculum schedules differ (15/15/20 versus 30/30/40).

When it finishes, `PUMA_stage2_training_outputs/` holds each experiment's checkpoints and
metrics. That is the **development model**. For a submission model, go to
`03_Evaluate_Infer.ipynb`: set `CREATE_DEVELOPMENT_LOCK = True` to lock the winner, then
`TRAIN_FINAL_MODEL = True` with `FINAL_EPOCHS = 100` to retrain that configuration on **all**
labeled ROIs. That yields:

```
stage2_v132_final_<experiment>_100ep_seed0_<hash>.pt    final classifier
stage2_v132_final_lock.json                            configuration + validity threshold
stage1_lock.json + the 5 A1 checkpoints                deployment detector ensemble
```

Inference needs **both**: the five Stage-1 checkpoints as the detector, and one Stage-2 model
as the classifier.

### `03_Evaluate_Infer.ipynb` — review, final train, infer

Every step past the ranking table sits behind an explicit switch
(`CREATE_DEVELOPMENT_LOCK`, `TRAIN_FINAL_MODEL`, `RUN_LOCAL_INFERENCE`,
`RUN_GRAND_CHALLENGE_INFERENCE`), all `False` by default, so Run All only prints the
ranking.

## 6. Stage-1 and Stage-2 settings for one RTX 3090 Ti

V13.2 gives each stage its own batch and epoch budget. The notebook defaults are:

```python
stage1_epochs=40,
stage2_epochs=50,                    # 50 screening, 100 winner; nothing else is accepted
stage1_effective_batch_size=16,
stage2_effective_batch_size=256,
stage1_micro_batch_size=16,          # no accumulation
stage2_micro_batch_size=256,         # no accumulation
stage1_early_stopping_enabled=True,
stage1_early_stopping_patience=10,
early_stopping_enabled=False,        # Stage 2: all epochs, for a fair comparison
```

Stage 1 trains on 512×512 tiles (`tile_size=512`, `tile_overlap=128`), which is far
lighter than Stage 2.

Each stage's effective batch must stay divisible by its micro-batch, or `create_runtime()`
rejects the configuration. On CUDA OOM, lower only the micro-batch: Stage 1 falls back
`8 → 4 → 2 → 1` and Stage 2 `128 → 64 → 32 → 16`, both preserving the effective batch
through gradient accumulation, so results are unaffected. The fallback is automatic while
`PUMA_V132_AUTO_OOM_FALLBACK=1`, which the bootstrap cell sets.

bf16 AMP is selected automatically on Ampere. For faster data loading on a many-core
workstation, in the runtime cell:

```python
runtime.training.number_of_workers = 8
```

## 7. What has and has not been verified

Verified on the machine this was prepared on (RTX 3080 10 GB, driver 595, 12 cores),
against code that V13.2 did **not** change:

- `setup_local.sh` end to end: uv venv, torch 2.11.0+cu128, `cuda available: True`, all
  12 dependency imports;
- `00_Preprocess.ipynb` executed head to end, exit 0, 23 s, 205/205 pairs, 97193 nuclei;
- the fold-integrity cell, executed against the real artifacts;
- `tests/test_fold_assignment.py`, 9/9 passing;
- `tests/test_gpu_selection.py`, 14/14 passing — the two-, four-, one-, and zero-GPU
  cases run against an injected inventory, so multi-GPU selection is covered on a
  single-GPU machine;
- that `import puma.gpu` pulls in neither `torch` nor `numpy`, and that setting
  `CUDA_VISIBLE_DEVICES` at that point really does control what torch sees afterwards.

`puma/data/preprocess.py`, `puma/gpu.py` and both test files are unchanged by the V13.2
merge, so the results above still stand for them.

Not verified:

- **Nothing in V13.2's Stage 1 or Stage 2 has been run here.** The merged core compiles
  and every import resolves, but no training, OOF generation, final training, or
  inference has been executed on this workstation.
- Nothing has been run on an RTX 3090 Ti. The 24 GB guidance above is reasoning from
  the tile size and the batch arithmetic, not measurement.
- No machine with two or more physical GPUs was available. The GPU-1 selection is verified by
  unit test, by the masking mechanism, and by executing the real bootstrap cell of `01`/`02`
  against injected GPU inventories (table in section 4) — but not on real multi-GPU hardware.
  Check the bootstrap cell's printout on the first run to confirm.

Run one fold first, by setting `run_folds=(0,)` in the runtime cell, before committing
to all five. Restore `run_folds=(0, 1, 2, 3, 4)` afterwards — complete OOF coverage
requires all five.

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No module named pip` from `%pip install` | Expected: the uv venv has no pip. Use `VIRTUAL_ENV=.venv uv pip install <package>`. |
| `Wrong kernel: /usr/bin/python3` | Kernel → Change Kernel → `SymbioPan (uv .venv)`. If it is absent, re-run `bash setup_local.sh`. |
| `... is not the project root (no puma/ package here)` | JupyterLab was started elsewhere. Start it from the project root. |
| `GeoJSON directory does not exist: .../Dataset/...` | The `Dataset -> dataset` symlink is missing (Linux paths are case-sensitive). Run `ln -s dataset Dataset`. |
| `Missing preprocessed artifacts: [...]` | Run `00_Preprocess.ipynb` first. |
| `Degenerate fold assignment: sizes [...]` | The split cannot support nested validation. Rebuild with `FORCE_PREPROCESS = True`. |
| `V13.2 Stage-2 epochs must be exactly 50 or 100` | `stage2_epochs` accepts only those two profiles. Screen at 50, retrain the winner at 100. |
| `Stage-1 sampling fractions must sum to 1.0` | A `runtime.data.*_fraction` was edited without rebalancing the others. |
| `The wrong PUMA package is loaded` | A stale `puma` in `sys.modules` or on `sys.path`. Restart the kernel and Run All; the bootstrap cell clears both. |
| `torch.cuda.is_available()` is `False` | Driver too old for the wheel. Check `nvidia-smi` and rebuild with `CUDA_BACKEND=cu126 bash setup_local.sh`. |
| CUDA OOM | The automatic fallback lowers the micro-batch and keeps the effective batch. To force it, set `stage1_micro_batch_size=8` or `stage2_micro_batch_size=128`. |
| Missing `PUMA_pretrained_checkpoints/UNI2-h/uni2_h_model.bin` at inference | Inference is offline by design. Run the checkpoint cell of `02_Train_Stage2.ipynb` once, and ship that file in the container. |
| Training landed on the wrong GPU | Read the bootstrap cell's printout. If it says `respected existing`, a `CUDA_VISIBLE_DEVICES` from the shell is winning — unset it and restart the kernel. |
| `Expected exactly one visible GPU after selection` | `CUDA_VISIBLE_DEVICES` lists several devices, or torch initialised CUDA before the bootstrap cell ran. Restart Kernel, then Run All. |
| Bootstrap warns that CUDA was already initialised | The GPU switch did not apply in this kernel. Restart Kernel, then Run All. |

---

# Part 2 — V13.2 pipeline reference

## Stage 1: A1_IFCRN_PP only

V13.2 keeps only `A1_IFCRN_PP`.

For each of the five outer folds:

1. keep the outer fold untouched;
2. train on three folds and use one non-outer fold as inner validation;
3. select the best epoch and post-processing on inner validation;
4. reset A1;
5. refit A1 on **all four non-outer folds** for exactly the selected number of epochs;
6. predict the untouched outer fold with the refit model.

This preserves leakage-safe OOF candidates while increasing the data used by each OOF
detector from 3/5 to 4/5.

### Stage-1 post-processing selection

V13.2 jointly evaluates:

- heatmap threshold;
- local-max radius;
- suppression radius.

It first finds the best oracle macro-F1 region, then among configurations within `0.005`
of the best ceiling prefers higher tail-class recall and then higher overall recall. This
gives a mild recall preference without accepting a materially worse detector ceiling.

### Stage-1 sampling

Tile-origin probabilities:

- density-centered: 30%
- small-nucleus-centered: 20%
- general nucleus-centered: 30%
- rare-nucleus-centered: 15%
- background/random: 5%

Rare-centered sampling only makes rare nuclei visible to the class-agnostic detector;
Stage 1 is still not trained as a class classifier. `create_runtime()` rejects fractions
that do not sum to 1.0.

### Stage-1 batch

Default:

```text
physical batch = 16
effective batch = 16
```

Inner-selection training has early stopping enabled with patience 10; the selected epoch
is then used exactly for the 4/5 refit.

Automatic CUDA-OOM fallback tries `8 -> 4 -> 2 -> 1` while preserving effective batch 16
by gradient accumulation.

### Fold assignment

Folds are built by `multilabel_greedy_folds()` in `puma/data/preprocess.py`, grouped by
`case_id` so no patient spans two folds, and stratified over melanoma type and all ten
classes. Fold size is a hard capacity of `total_rois / number_of_folds` (differing by at
most one), counted in ROIs rather than case groups; class balance is optimised within
that constraint and then refined by swapping equal-ROI-count groups between folds.

On PUMA this yields `[41, 41, 41, 41, 41]`.

Because each fold serves both as an outer OOF fold and as another fold's inner
validation split, a lopsided split invalidates threshold selection and OOF coverage
while still training and reporting without error. `validate_fold_assignments()`
therefore raises on any fold below half the expected size, and is called from
`multilabel_greedy_folds()`, from `run_stage1_a1()`, and from the fold-integrity cell in
notebooks `00` and `01`. `tests/test_fold_assignment.py` covers the balance, grouping,
stratification, determinism, and validator behaviour:

```bash
./.venv/bin/python tests/test_fold_assignment.py
./.venv/bin/python tests/test_gpu_selection.py
```

## GPU selection

`puma/gpu.py` holds the device selection used by the notebook bootstrap cells:
`query_gpu_inventory()` reads `nvidia-smi`, and `select_cuda_device(preferred_index)`
pins `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER`. The module deliberately imports no
`torch` — see *Which GPU the notebooks use* in Part 1 for why the timing matters. Both
functions accept injected inventories and environments, which is how the multi-GPU paths
are tested on a single-GPU machine.

## Stage 2: fixed optimized split

V13.2 creates/reuses one case-grouped optimized 80/20 development split for all Stage-2
experiments. Do not recreate the split between experiments.

The split balances ROI count, nuclei count, class distribution/presence, rare classes,
case grouping, and primary/metastatic composition.

```python
ensure_v132_split(runtime, force=False, val_fraction=0.20, seed=2026, check_sources=True)
```

## Exact epoch profiles

Only two Stage-2 epoch profiles are accepted.

### Screening: 50 epochs

```text
Epoch  1-15: GT_POS
Epoch 16-30: OOF_POS
Epoch 31-50: OOF_ALL
```

### Final/winner: 100 epochs

```text
Epoch   1-30: GT_POS
Epoch  31-60: OOF_POS
Epoch  61-100: OOF_ALL
```

A 100-epoch winner should be trained from scratch with the 100-epoch profile; do not
resume a completed 50-epoch screening run as if it were the same schedule.

## Stage-2 curriculum

The original V13.1 learning logic is deliberately retained:

1. **GT_POS (30%)** — perfect GT centroids; learn clean nucleus phenotype.
2. **OOF_POS (30%)** — matched Stage-1 OOF positives; learn phenotype under real detector
   localization error.
3. **OOF_ALL (40%)** — positives plus REJECT candidates; learn classification and
   candidate validity together.

GT centroids are not jittered. The validity loss is disabled in the first two
positive-only phases and activated only in OOF_ALL.

## Phase-aware learning-rate schedule

A single global cosine is not used. Each curriculum phase has its own schedule.

| Phase | Type/fusion LR | Validity LR |
|---|---:|---:|
| GT_POS | warm up to `1e-4`, cosine to `5e-5` | `0` |
| OOF_POS | `7.5e-5` -> `3e-5` | `0` |
| OOF_ALL | `5e-5` -> `5e-6` | `1e-4` -> `1e-5` |

GT_POS uses a three-epoch warmup. Optimizer is AdamW with `weight_decay=1e-4`; gradient
clipping is `1.0`.

## Strong rare-class exposure

Tail classes:

- plasma cell
- neutrophil
- apoptosis
- melanophage
- endothelium

Main V13.2 sampler target at Stage-2 batch 256:

| Phase | requested guaranteed exposure per tail class |
|---|---:|
| GT_POS | 16 / batch |
| OOF_POS | 12 / batch |
| OOF_ALL | 8 / batch within the positive component |

The sampler is case-aware and unique-first as far as the repeat budget allows. Common
candidates are capped at 4 repeats/epoch; tail candidates at 12 repeats/epoch. If a tail
class is too small for the requested quota without exceeding the repeat cap, the sampler
automatically lowers the effective quota and records it in the sampler statistics rather
than silently overfitting the same nuclei.

Augmentation uses morphology-preserving D4 rotations/flips plus mild stain perturbation.
Rare examples are seen more often, not distorted more aggressively.

## Hard mining

In OOF_ALL:

- hard mining begins at phase epoch 4;
- hard pools refresh every 3 epochs;
- approximately 50% of the reject quota may come from hard rejects;
- approximately 25% of the guaranteed rare quota may come from hard rare positives.

## Stage-2 experiments retained

V13.2 intentionally reduces the old six-way search to four controlled runs:

| Experiment | Purpose |
|---|---|
| `V13_2_01_META_CONTROL_BS` | META V64+V128 control, Balanced Softmax, moderate sampler |
| `V13_2_02_META_RARE_BS` | **Main model**: strong rare exposure + hard mining + Balanced Softmax |
| `V13_2_03_META_RARE_CE` | Same main sampler/training policy but plain CE, to test whether Balanced Softmax still helps after aggressive exposure correction |
| `V13_2_04_META_CONTEXT_RARE_BS` | Same strong rare-exposure/BS policy as experiment 02, but adds V256 context (V64+V128+V256) to isolate the value of larger tissue context |

Removed from V13.2 screening:

- separate CB-Focal branch;
- separate CB-CE branch;
- separate RareBoost branch (rare exposure is now part of the main training policy);
- LoRA.

For a 50-epoch screening run, train all four. For a 100-epoch run, train only the
selected winner.

## Stage-2 batch and speed profile

Default:

```text
Stage-2 effective batch = 256
Stage-2 physical batch  = 256
UNI2-h encoder batch    = 256
```

CUDA-OOM fallback tries smaller physical batches (`128, 64, 32, 16`) while preserving
effective batch 256 through gradient accumulation.

Speed optimizations include:

- BF16 autocast when supported;
- TF32 / cuDNN benchmark in fast non-deterministic mode;
- fused/foreach AdamW where supported;
- channels-last CUDA tensors where beneficial;
- sequential experiments on one GPU (no destructive parallel GPU contention);
- Colab/local hot-array caching;
- per-worker native-crop cache;
- persistent Stage-2 workers for each curriculum phase;
- within-epoch cache of frozen UNI2-h features for repeated oversampled candidates;
- sparse validation: phase boundaries plus every two epochs in OOF_ALL;
- hard-pool refresh every three epochs instead of every epoch.

`02_Train_Stage2.ipynb` enables the fast non-deterministic runtime path for maximum
throughput (`FAST_NONDETERMINISTIC = True`). Use deterministic mode only when exact
replay matters more than speed.

## Checkpoint selection

### 50-epoch screening

Early stopping is disabled. All experiments receive all 50 epochs for fair comparison.
Checkpoint selection begins at OOF_ALL phase epoch 6 and uses pooled macro-F1.

### 100-epoch winner

The notebook supports the exact same 30/30/40 profile at 100 epochs. If early stopping is
enabled manually, apply it only in OOF_ALL; the configured patience is 15 and minimum
delta is `0.001`.

## Primary metric

V13.2 uses pooled TP/FP/FN class F1 and averages over the ten nucleus classes for
macro-F1. Image-wise F1 can be logged as a diagnostic but is not the primary
checkpoint-selection metric.

## Final training

After the 50-epoch screening:

1. lock one Stage-2 experiment;
2. keep the five refit A1 fold checkpoints as the deployment detector ensemble;
3. retrain the selected Stage-2 configuration on all labeled ROIs;
4. use `final_epochs=100` for the intended final model;
5. keep the development-selected validity threshold in the deployment lock.

Main final artifacts:

- `stage1_lock.json`
- `stage2_v132_lock.json`
- `stage2_v132_final_<experiment>_100ep_seed0_<hash>.pt`
- `stage2_v132_final_lock.json`

## Grand-Challenge inference

**Offline submission requirement:** the official baseline runs the container with
networking disabled. The submission/container must therefore include the project-local
`PUMA_pretrained_checkpoints/UNI2-h/uni2_h_model.bin`. V13.2 inference fails clearly if
this local binary is missing instead of attempting an online download.

`puma/pipeline/inference.py` provides both local ROI inference and PUMA Track-2 deployment
inference.

The official PUMA baseline test mounts one TIFF under `/input/images/melanoma-wsi/`. The
challenge/test ROI is 1024×1024, matching the labeled training ROI size. V13.2 therefore
loads the complete ROI directly; there is **no macro-tile/WSI streaming layer**. For
compatibility, the inference wrapper also accepts
`/input/images/melanoma-whole-slide-image/` if a Grand-Challenge interface uses that alias.

The Grand-Challenge path:

- reads exactly one 1024×1024 TIFF from the official input mount;
- runs Stage 1 with its internal 512px tile/overlap logic over that ROI;
- merges/suppresses all Stage-1 candidates for the complete ROI;
- computes the original 7-D Stage-2 geometry over the complete 1024×1024 ROI;
- runs Stage 2 with V64/V128/(optional V256) crops;
- writes `/output/melanoma-10-class-nuclei-segmentation.json`;
- writes `/output/images/melanoma-tissue-mask-segmentation/<uuid>.tif`.

Centroid predictions are serialized as small symmetric polygons. The arithmetic mean of
their polygon vertices is exactly the model centroid, which is the coordinate consumed by
the official nuclei evaluator.

### Tissue-output warning

V13.2 is a nuclei pipeline and does **not** train a tissue segmentation model. If a real
tissue mask is supplied, the inference wrapper validates/writes it. Otherwise it can emit
an all-background mask only to satisfy the Track-2 file contract. That fallback is
structurally valid but is not a competitive tissue prediction.

## Main output directories

- preprocessing: `PUMA_outputs/`
- Stage 1: `PUMA_stage1_training_outputs/`
- Stage 2: `PUMA_stage2_training_outputs/`

Key preprocessing artifacts:

- `puma_rgb_images.npy`
- `puma_instance_maps.npy`
- `puma_class_maps.npy`
- `puma_centroid_heatmaps.npy`
- `puma_centroid_match_disks_15px.npy`
- `puma_roi_manifest.npy`
- `puma_nuclei_centroids.npy`
- `puma_roi_centroid_offsets.npy`
- `puma_fold_assignments.npy`
- `puma_preprocessing_metadata.json`, which includes a `fold_report` section with
  per-fold ROI, melanoma-type, and class counts

Key Stage-1 artifacts:

- five `stage1_best_A1_IFCRN_PP_fold*_seed0.pt` checkpoints
- `stage1_results.csv`
- `stage1_lock.json`
- `stage1_oof_candidates.npy`
- `stage1_oof_candidates_metadata.json`

## Resume and output-directory policy

Training artifacts use exactly these canonical directories:

- `PUMA_stage1_training_outputs/`
- `PUMA_stage2_training_outputs/`

Every training stage saves a resume checkpoint after each completed epoch. If a
Colab/runtime session stops, rerun the same notebook/configuration and training resumes
from the next epoch. If the relevant resume checkpoint/output directory is deleted, that
portion starts again from scratch. Completed-run skipping is allowed only when the current
configuration hash and required checkpoint/prediction artifacts are all present.

Stage-2 screening/final checkpoints embed their semantic configuration hash, and final
deployment also has a separate deployment hash for the validity threshold. Changing only
the final threshold updates the deployment lock without retraining the unchanged weights.
Physical OOM-fallback micro-batch sizes/workers are treated as execution details rather
than a different experiment identity.

## 1024 ROI geometry contract

PUMA labeled training images and challenge test inputs are 1024×1024 ROIs. V13.2 keeps
the original 7-D Stage-2 geometry:

- `log_nearest_distance`
- `local_density`
- `detector_confidence`
- `border_distance_normalized`
- `microns_per_pixel`
- `x_normalized`
- `y_normalized`

`nearest_distance` and `local_density` are computed only after all Stage-1 tiled
predictions have been merged and suppressed for the complete ROI. Therefore internal
Stage-1 tile boundaries do not alter Stage-2 geometry. Grand-Challenge inference validates
the input spatial shape and refuses non-1024×1024 inputs rather than silently changing the
geometry distribution.

## Colab

All four notebooks run unchanged on Colab: the bootstrap cell mounts Drive and uses
`/content/drive/MyDrive/Research/PUMA` as the project root. Dependencies there come from
`pip install -r requirements_colab.txt` in a scratch cell, with torch supplied by the
Colab runtime.
