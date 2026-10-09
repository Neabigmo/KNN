import numpy as np

from knn_reliability.geometry import (
    ring_support_construction,
    single_prototype_motion_safe,
    verify_ring_separation,
)


def test_ring_construction_has_explicit_separation_for_small_k():
    construction = ring_support_construction(3, epsilon=0.01)
    report = verify_ring_separation(construction)
    assert report["cross_cluster_separation_holds"]
    assert report["center_representatives_are_first_k"]


def test_single_motion_certificate_is_strictly_conservative():
    np.testing.assert_array_equal(
        single_prototype_motion_safe(np.array([1, 2, 3, 5])),
        np.array([False, False, True, True]),
    )
