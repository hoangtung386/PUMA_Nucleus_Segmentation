# Merged Version 2.2 architecture + Version 4 data/classes

This package keeps the Version 2.2 panoptic architecture, but uses Version 4 preprocessing and fixed PUMA class names.

## Main decisions

- Tissue model output stays **5 classes**. There is no trainable tissue background channel.
- Version 4 stored tissue masks are PUMA IDs: `0 background, 1 stroma, 2 blood vessel, 3 tumor, 4 epidermis, 5 necrosis`.
- Dataset converts stored background `0` to `255` ignore.
- Internal tissue order is: `0 stroma, 1 blood vessel, 2 tumor, 3 epidermis, 4 necrosis`.
- Inference maps internal tissue prediction `0..4` back to PUMA output `1..5`.
- SC-DFA and spatial prior stay `5 x 10`.
- Nuclei class names use Version 4 fixed names:
  - `nuclei_tumor`
  - `nuclei_lymphocyte`
  - `nuclei_plasma_cell`
  - `nuclei_histiocyte`
  - `nuclei_melanophage`
  - `nuclei_neutrophil`
  - `nuclei_stroma`
  - `nuclei_epithelium`
  - `nuclei_endothelium`
  - `nuclei_apoptosis`

## Stage 1 training

`main.py` now simply calls `train_stage1.py`.

```bash
python preprocess.py

python train_stage1.py \
  --data-dir dataset_processed \
  --uni-weight-dir . \
  --image-size 1024 \
  --batch-size 2 \
  --epochs 80 \
  --split-file checkpoints/split_seed42.npz
```

Stage 1 saves:

```text
checkpoints/puma_epoch_best_s1.pth
checkpoints/split_seed42.npz
```

The checkpoint contains full model weights plus inference config: UNI/ConvNeXt/FPN/decoders/SC-DFA/spatial-prior buffers/cellpose_adapter/class mapping/tile size/SC-DFA flag/lambda.

## Stage 2 training

Stage 2 uses exactly the same split file as Stage 1. Do not generate a new split.

```bash
python train_stage2.py \
  --data-dir dataset_processed \
  --uni-weight-dir . \
  --stage1-ckpt checkpoints/puma_epoch_best_s1.pth \
  --split-file checkpoints/split_seed42.npz \
  --image-size 1024 \
  --batch-size 2 \
  --epochs 60
```

Stage 2 saves:

```text
checkpoints/nuclei_refiner_residual_best.pth
checkpoints/nuclei_refiner_residual_last.pth
```

Stage 2 input is 21 channels, not 22:

```text
3 image + 5 tissue probs + 10 nuclei probs + 1 NP prob + 2 HV = 21
```

## Cellpose in inference

The Stage 1 checkpoint stores the small `cellpose_adapter` network because it is inside the PyTorch model.

It does not store the external Cellpose flow generator. The new `infer_wsi.py` can generate Cellpose flow at inference time if `cellpose` is installed:

```bash
python infer_wsi.py --cellpose-mode generate
```

Default Docker behavior is:

```bash
--cellpose-mode auto
```

This tries to load Cellpose. If Cellpose cannot load, it falls back to zero flow with a warning. Use `--cellpose-mode generate` when you want failure instead of fallback.

## Docker checkpoint layout

For Stage 1 only:

```text
checkpoint/best_model.pth
```

For Stage 1 + Stage 2:

```text
checkpoint/best_model.pth
checkpoint/nuclei_refiner_residual_best.pth
```

`inference.sh` automatically uses Stage 2 only if `nuclei_refiner_residual_best.pth` exists.

## PUMA output paths

The Docker inference script reads:

```text
/input/images/melanoma-whole-slide-image/<uuid>.tif
```

and writes:

```text
/output/melanoma-10-class-nuclei-segmentation.json
/output/images/melanoma-tissue-mask-segmentation/<uuid>.tif
```
