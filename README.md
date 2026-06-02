# PUMA Encoder Probe: 3-Fold Multi-Label Stratified Splits

This is a clean encoder-probe codebase for PUMA Grand Challenge Track 2 style data.

## Experiments

```text
train_E1.py   # UNIv2 frozen
train_E2.py   # ConvNeXt V2 Atto frozen
train_E3.py   # Virchow2 frozen
train_E4.py   # ConvNeXt V2 Atto + UNIv2 frozen
train_E5.py   # ConvNeXt V2 Atto + Virchow2 frozen
```

## Label policy

Tissue is dense 6-class segmentation:

```text
0 tissue_background
1 tissue_stroma
2 tissue_blood_vessel
3 tissue_tumor
4 tissue_epidermis
5 tissue_necrosis
```

Nuclei classification uses 10 foreground classes only. There is no nuclei background class. Pixels outside nuclei use `255` in `nuclei_class` and are ignored by the nuclei-class loss.

```text
0 nuclei_tumor
1 nuclei_lymphocyte
2 nuclei_plasma_cell
3 nuclei_histiocyte
4 nuclei_melanophage
5 nuclei_neutrophil
6 nuclei_stroma
7 nuclei_epithelium
8 nuclei_endothelium
9 nuclei_apoptosis
```

## Data root

The root path is fixed to:

```python
ROOT = Path('/content/drive/MyDrive/Research/PUMA')
```

Expected raw dataset folders:

```text
/content/drive/MyDrive/Research/PUMA/Dataset/01_training_dataset_tif_ROIs
/content/drive/MyDrive/Research/PUMA/Dataset/01_training_dataset_geojson_tissue
/content/drive/MyDrive/Research/PUMA/Dataset/01_training_dataset_geojson_nuclei
```

Processed output:

```text
/content/drive/MyDrive/Research/PUMA/dataset_processed_encoder
```

Training output:

```text
/content/drive/MyDrive/Research/PUMA/checkpoints_encoder
```

## Foundation-model tiling

Original image/mask size stays `1024 x 1024`.

UNIv2 and Virchow2 paths split the ROI into 16 tiles:

```text
1024 x 1024 -> 4 x 4 tiles of 256 x 256
```

The model stitches tile features back spatially before decoding.

ConvNeXt V2 Atto receives the full `1024 x 1024` ROI.

## Run order

```bash
cd /content/drive/MyDrive/Research/PUMA
pip install -r requirements.txt

python preprocess_data.py
python generate_folds.py
python sanity_check.py
```

Optional model-forward sanity check:

```bash
python sanity_check.py --models
```

`--models` creates each E1-E5 model with `pretrained=False` and runs a small synthetic 256x256 forward pass. It may still need internet/Hugging Face access for hub model definitions.

Train all 3 folds for one experiment:

```bash
python train_E1.py
python train_E2.py
python train_E3.py
python train_E4.py
python train_E5.py
```

Train only one fold:

```bash
python train.py --run E1 --fold 0
```

Summarize results:

```bash
python summarize_and_compare.py
```

## Files created by `generate_folds.py`

```text
dataset_processed_encoder/splits_3fold_multilabel/
  fold_assignments.csv
  fold_0_train.csv
  fold_0_val.csv
  fold_1_train.csv
  fold_1_val.csv
  fold_2_train.csv
  fold_2_val.csv
  folds_summary.csv
  folds_metadata.json
```

## Notes

- The split is multi-label stratified using image-level class presence.
- This prevents rare classes such as tissue_necrosis, tissue_epidermis, nuclei_neutrophil, nuclei_plasma_cell, and nuclei_melanophage from disappearing from validation folds when possible.
- Each ROI can contain only a subset of labels. Missing classes are expected.
