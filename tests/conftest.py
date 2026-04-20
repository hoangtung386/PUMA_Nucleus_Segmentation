"""Shared pytest fixtures for the PUMA test suite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def dummy_image() -> np.ndarray:
    """Small RGB image for fast testing."""
    return np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def dummy_instance_mask() -> np.ndarray:
    """Instance mask with two nuclei."""
    mask = np.zeros((128, 128), dtype=np.int32)
    mask[10:30, 10:30] = 1
    mask[70:90, 70:90] = 2
    return mask


@pytest.fixture(scope="session")
def dummy_class_mask() -> np.ndarray:
    """Class mask matching dummy_instance_mask."""
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    mask[70:90, 70:90] = 2
    return mask


@pytest.fixture()
def tmp_processed_dir(tmp_path: Path) -> Path:
    """Create a complete fake processed directory structure."""
    for split in ("train", "val", "test"):
        for subdir in ("images", "masks", "labels"):
            (tmp_path / split / subdir).mkdir(parents=True)
    return tmp_path


@pytest.fixture()
def simple_geojson_path(tmp_path: Path) -> Path:
    """GeoJSON with two non-overlapping nuclei."""
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[10, 10], [30, 10], [30, 30], [10, 30], [10, 10]]],
            },
            "properties": {"classification": {"name": "tumor"}},
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[50, 50], [80, 50], [80, 80], [50, 80], [50, 50]]],
            },
            "properties": {"classification": {"name": "lymphocyte"}},
        },
    ]
    path = tmp_path / "sample.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path
