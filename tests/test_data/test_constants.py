"""Tests for label-space constants and class ID invariants."""

from symbiopan.data.constants import (
    LOSS_MULTIPLIERS,
    NUCLEI_CLASS_WEIGHTS,
    NUM_NUCLEI_CLASSES,
    NUM_TISSUE_CLASSES,
    PUMA_NUCLEI_ID_TO_NAME,
    PUMA_NUCLEI_NAME_TO_ID,
    PUMA_TISSUE_ID_TO_NAME,
    PUMA_TISSUE_NAME_TO_ID,
    RARE_NUCLEI_IDS,
    RARE_TISSUE_IDS,
    SITE_MAP,
    SITE_NAMES,
    TISSUE_CLASS_WEIGHTS,
)


def test_tissue_id_name_round_trip():
    for tissue_id, name in PUMA_TISSUE_ID_TO_NAME.items():
        assert PUMA_TISSUE_NAME_TO_ID[name] == tissue_id


def test_nuclei_id_name_round_trip():
    for nuc_id, name in PUMA_NUCLEI_ID_TO_NAME.items():
        assert PUMA_NUCLEI_NAME_TO_ID[name] == nuc_id


def test_class_counts():
    assert NUM_TISSUE_CLASSES == 6
    assert NUM_NUCLEI_CLASSES == 10
    assert len(PUMA_TISSUE_ID_TO_NAME) == NUM_TISSUE_CLASSES
    assert len(PUMA_NUCLEI_ID_TO_NAME) == NUM_NUCLEI_CLASSES


def test_class_weights_match_class_counts():
    assert len(TISSUE_CLASS_WEIGHTS) == NUM_TISSUE_CLASSES
    assert len(NUCLEI_CLASS_WEIGHTS) == NUM_NUCLEI_CLASSES


def test_loss_multipliers_length():
    assert len(LOSS_MULTIPLIERS) == 5
    assert LOSS_MULTIPLIERS[-1] == 0.0


def test_site_map_consistent():
    assert len(SITE_NAMES) == len(SITE_MAP)
    for i, name in enumerate(SITE_NAMES):
        assert SITE_MAP[name] == i


def test_rare_ids_immutable():
    assert isinstance(RARE_TISSUE_IDS, frozenset)
    assert {2, 4, 5} == RARE_TISSUE_IDS
    assert {2, 4, 5, 8, 9} == RARE_NUCLEI_IDS
