"""Tests for dataset loading and site-type inference."""

from symbiopan.data.dataset.puma_dataset import infer_site_id


def test_infer_site_id_primary():
    assert infer_site_id("primary_roi_001") == 0


def test_infer_site_id_lymph_node():
    assert infer_site_id("training_set_metastatic_roi_001") == 1


def test_infer_site_id_brain():
    assert infer_site_id("brain_met_001") == 2


def test_infer_site_id_bone():
    assert infer_site_id("bone_met_001") == 3


def test_infer_site_id_soft_tissue():
    assert infer_site_id("soft_tissue_001") == 4


def test_infer_site_id_liver():
    assert infer_site_id("liver_met_001") == 5


def test_infer_site_id_lung():
    assert infer_site_id("lung_met_001") == 6


def test_infer_site_id_gastrointestinal():
    assert infer_site_id("gastrointestinal_001") == 7


def test_infer_site_id_skin_mets():
    assert infer_site_id("skin_met_001") == 8
