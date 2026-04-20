"""Tests for GeoJSON parser and dataset utilities."""

from __future__ import annotations

import json

import numpy as np
import pytest

from puma_seg.data.geojson_parser import (
    get_class_map,
    get_nucleus_bboxes,
    get_nucleus_centroids,
    normalize_class_name,
    parse_geojson,
)

# ─── Tests ────────────────────────────────────────────────────────────────────


class TestClassMap:
    def test_track1_has_4_classes(self):
        cmap = get_class_map(1)
        assert len(cmap) == 4  # background + 3 foreground

    def test_track2_has_11_classes(self):
        cmap = get_class_map(2)
        assert len(cmap) == 11  # background + 10 foreground

    def test_background_is_zero(self):
        assert get_class_map(1)["background"] == 0
        assert get_class_map(2)["background"] == 0

    def test_invalid_track_raises(self):
        with pytest.raises(ValueError):
            get_class_map(3)

    def test_normalize_prefixed_labels(self):
        assert normalize_class_name("nuclei_tumor") == "tumor"
        assert normalize_class_name("tissue_blood_vessel") == "blood vessel"
        assert normalize_class_name("tils") == "lymphocyte"


class TestParseGeoJSON:
    def test_returns_correct_shapes(self, simple_geojson_path):
        H, W = 100, 100
        inst_mask, cls_mask, inst_cls = parse_geojson(simple_geojson_path, (H, W), track=1)
        assert inst_mask.shape == (H, W)
        assert cls_mask.shape == (H, W)
        assert inst_mask.dtype == np.int32
        assert cls_mask.dtype == np.uint8

    def test_two_instances_detected(self, simple_geojson_path):
        inst_mask, _, inst_cls = parse_geojson(simple_geojson_path, (100, 100), track=1)
        assert inst_mask.max() == 2
        assert len(inst_cls) == 2

    def test_class_ids_correct(self, simple_geojson_path):
        _, _, inst_cls = parse_geojson(simple_geojson_path, (100, 100), track=1)
        cmap = get_class_map(1)
        # First nucleus = tumor, second = lymphocyte
        assert inst_cls[0] == cmap["tumor"]
        assert inst_cls[1] == cmap["lymphocyte"]

    def test_background_is_zero(self, simple_geojson_path):
        inst_mask, cls_mask, _ = parse_geojson(simple_geojson_path, (100, 100))
        # Pixels far from both annotations should be 0
        assert inst_mask[0, 0] == 0
        assert cls_mask[0, 0] == 0

    def test_empty_geojson(self, tmp_path):
        empty_path = tmp_path / "empty.geojson"
        empty_path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
        inst_mask, cls_mask, inst_cls = parse_geojson(empty_path, (64, 64))
        assert inst_mask.max() == 0
        assert len(inst_cls) == 0

    def test_prefixed_labels_are_mapped(self, tmp_path):
        geojson_path = tmp_path / "prefixed.geojson"
        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[1, 1], [5, 1], [5, 5], [1, 5], [1, 1]]],
                },
                "properties": {"classification": {"name": "nuclei_tumor"}},
            }
        ]
        geojson_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))

        _, _, inst_cls = parse_geojson(geojson_path, (16, 16), track=1)
        assert inst_cls[0] == get_class_map(1)["tumor"]


class TestSpatialHelpers:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mask = np.zeros((100, 100), dtype=np.int32)
        self.mask[10:30, 10:30] = 1
        self.mask[60:80, 60:80] = 2

    def test_centroids_count(self):
        centroids = get_nucleus_centroids(self.mask)
        assert len(centroids) == 2

    def test_centroids_approximate(self):
        centroids = get_nucleus_centroids(self.mask)
        row1, col1 = centroids[1]
        assert abs(row1 - 19.5) < 1.0
        assert abs(col1 - 19.5) < 1.0

    def test_bboxes_count(self):
        bboxes = get_nucleus_bboxes(self.mask, padding=0)
        assert len(bboxes) == 2

    def test_bboxes_bounds(self):
        bboxes = get_nucleus_bboxes(self.mask, padding=0)
        y_min, x_min, y_max, x_max = bboxes[1]
        assert y_min <= 10
        assert x_min <= 10
        assert y_max >= 30
        assert x_max >= 30
