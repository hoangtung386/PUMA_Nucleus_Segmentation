# Version 5 rare-focused, no-leak package

Run from the project root (`Path.cwd()` must be the PUMA project folder):

```bash
python preprocess.py
rm -f checkpoints/split_seed42.npz
python train_stage1.py
python train_stage2.py
```

Important split rule:

- `preprocess.py` creates original samples and rare-centered samples named `source__rare...`.
- `train_stage1.py` and `train_stage2.py` use a **group-based split** by `source_name`.
- All rare crops stay on the same side as their original ROI.
- Validation uses original samples only, not rare-centered synthetic/translated crops.

Stage 1 config:

- `batch_size = 12`
- `epochs = 50`
- `zero_cellpose_prob = 0.0`
- best checkpoint: `checkpoints/puma_epoch_best_s1.pth`
- prints best epoch at the end.

Stage 2 config:

- best checkpoint: `checkpoints/nuclei_refiner_residual_best.pth`
- prints best epoch at the end.

Docker inference:

- reads `/input/images/melanoma-whole-slide-image/<uuid>.tif`
- writes `/output/melanoma-10-class-nuclei-segmentation.json`
- writes `/output/images/melanoma-tissue-mask-segmentation/<uuid>.tif`

Checkpoint folder for Docker:

```text
checkpoint/
├── best_model.pth                         required; copy from checkpoints/puma_epoch_best_s1.pth
├── nuclei_refiner_residual_best.pth        optional
└── site_classifier_atto.pth                optional
```
