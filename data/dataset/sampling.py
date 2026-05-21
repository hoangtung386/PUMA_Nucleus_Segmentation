"""Sample weight computation for rare-focused weighted sampling."""

from pathlib import Path
from typing import Sequence

import numpy as np

from data.constants import RARE_NUCLEI_IDS, RARE_NUCLEI_SAMPLE_BONUS, RARE_TISSUE_IDS_PUMA, RARE_TISSUE_SAMPLE_BONUS


def compute_sample_weight(
    tissue: np.ndarray,
    nuclei: np.ndarray,
    is_rare_augmented: bool,
    metadata_weight: float | None = None,
) -> float:
    """Compute a single sample weight based on rare class presence.

    Args:
        tissue: Tissue semantic segmentation mask.
        nuclei: Nuclei classification mask.
        is_rare_augmented: Whether this sample is a rare-class crop.
        metadata_weight: If provided, use this pre-computed weight directly
            instead of computing from masks.

    Returns:
        Float sample weight.
    """
    if metadata_weight is not None:
        return float(metadata_weight)

    weight = 1.0

    rare_tissue_ids = sorted(int(x) for x in np.unique(tissue) if int(x) in RARE_TISSUE_IDS_PUMA)
    rare_nuclei_ids = sorted(int(x) for x in np.unique(nuclei) if int(x) in RARE_NUCLEI_IDS)

    for cls in rare_tissue_ids:
        weight += RARE_TISSUE_SAMPLE_BONUS.get(cls, 0.0)
    for cls in rare_nuclei_ids:
        weight += RARE_NUCLEI_SAMPLE_BONUS.get(cls, 0.0)
    if is_rare_augmented:
        weight *= 1.5
    return float(weight)


def compute_all_sample_weights(
    data_dir: Path,
    base_names: Sequence[str],
    is_rare_augmented: Sequence[bool],
    metadata: dict | None = None,
) -> list[float]:
    """Compute sample weights for a list of samples.

    Uses metadata weights when available, otherwise reads tissue and nuclei
    masks from disk to detect rare classes.

    Args:
        data_dir: Path to the processed data directory.
        base_names: Sequence of sample base names.
        is_rare_augmented: Sequence of booleans indicating rare-augmented status.
        metadata: Optional dict mapping base_name to metadata rows.

    Returns:
        list of float sample weights.
    """
    weights = []
    for i, base_name in enumerate(base_names):
        meta = (metadata or {}).get(str(base_name))
        if meta is not None and "sample_weight" in meta:
            weights.append(float(meta["sample_weight"]))
            continue

        weight = 1.0
        tissue_path = data_dir / "tissue_sem" / f"{base_name}.npy"
        nuclei_path = data_dir / "nuclei_nc" / f"{base_name}.npy"

        if tissue_path.exists():
            tissue = np.load(tissue_path, mmap_mode="r")
            rare_ids = sorted(int(x) for x in np.unique(tissue) if int(x) in RARE_TISSUE_IDS_PUMA)
            for cls in rare_ids:
                weight += RARE_TISSUE_SAMPLE_BONUS.get(cls, 0.0)

        if nuclei_path.exists():
            nuclei = np.load(nuclei_path, mmap_mode="r")
            rare_ids = sorted(int(x) for x in np.unique(nuclei) if int(x) in RARE_NUCLEI_IDS)
            for cls in rare_ids:
                weight += RARE_NUCLEI_SAMPLE_BONUS.get(cls, 0.0)

        if is_rare_augmented[i]:
            weight *= 1.5
        weights.append(float(weight))
    return weights
