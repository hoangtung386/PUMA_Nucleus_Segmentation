"""Tests for post-processing helpers."""

import numpy as np

from symbiopan.inference.postprocessing import classify_instances, hv_instance_segmentation


def test_hv_instance_segmentation_empty_input():
    h, w = 32, 32
    np_logits = -np.ones((h, w), dtype=np.float32) * 10.0
    hv_map = np.zeros((2, h, w), dtype=np.float32)
    inst = hv_instance_segmentation(np_logits, hv_map, threshold=0.5, min_size=4)
    assert inst.dtype == np.int32
    assert inst.shape == (h, w)
    assert int(inst.sum()) == 0


def test_classify_instances_returns_class_per_id():
    inst = np.zeros((16, 16), dtype=np.int32)
    inst[2:6, 2:6] = 1
    inst[8:12, 8:12] = 2

    nc_logits = np.zeros((10, 16, 16), dtype=np.float32)
    nc_logits[3, 2:6, 2:6] = 5.0
    nc_logits[7, 8:12, 8:12] = 5.0

    out = classify_instances(inst, nc_logits)
    assert out[1][0] == 3
    assert out[2][0] == 7
