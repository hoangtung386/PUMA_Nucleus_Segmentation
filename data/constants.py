"""Centralized constants: label mappings and class IDs."""

PUMA_TISSUE_ID_TO_NAME = {
    0: "background",
    1: "tissue_stroma",
    2: "tissue_blood_vessel",
    3: "tissue_tumor",
    4: "tissue_epidermis",
    5: "tissue_necrosis",
}

INTERNAL_TISSUE_ID_TO_NAME = {
    0: "tissue_stroma",
    1: "tissue_blood_vessel",
    2: "tissue_tumor",
    3: "tissue_epidermis",
    4: "tissue_necrosis",
}

NUM_TISSUE_CLASSES = 5

PUMA_NUCLEI_ID_TO_NAME = {
    0: "nuclei_tumor",
    1: "nuclei_lymphocyte",
    2: "nuclei_plasma_cell",
    3: "nuclei_histiocyte",
    4: "nuclei_melanophage",
    5: "nuclei_neutrophil",
    6: "nuclei_stroma",
    7: "nuclei_epithelium",
    8: "nuclei_endothelium",
    9: "nuclei_apoptosis",
}

NUM_NUCLEI_CLASSES = 10

PUMA_TISSUE_NAME_TO_ID = {v: k for k, v in PUMA_TISSUE_ID_TO_NAME.items()}
PUMA_NUCLEI_NAME_TO_ID = {v: k for k, v in PUMA_NUCLEI_ID_TO_NAME.items()}

RARE_TISSUE_IDS = {1, 3, 4}
RARE_NUCLEI_IDS = {2, 4, 5, 8, 9}
RARE_TISSUE_IDS_PUMA = [2, 4, 5]

RARE_TISSUE_SAMPLE_BONUS = {2: 3.0, 4: 2.0, 5: 6.0}
RARE_NUCLEI_SAMPLE_BONUS = {2: 6.0, 4: 4.0, 5: 8.0, 8: 5.0, 9: 8.0}

TISSUE_CLASS_WEIGHTS = [1.0, 4.0, 0.8, 3.0, 7.0]
NUCLEI_CLASS_WEIGHTS = [0.8, 1.0, 7.0, 2.5, 4.5, 8.0, 2.0, 2.5, 5.5, 8.0]
STAGE2_NUCLEI_WEIGHTS = [0.6, 0.9, 9.0, 2.5, 5.0, 10.0, 2.0, 2.5, 6.0, 10.0]

LOSS_MULTIPLIERS = [2.5, 1.0, 2.8, 1.0]
HV_GRAD_THRESHOLD = 0.35

NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

IGNORE_INDEX = 255
