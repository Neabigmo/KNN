import numpy as np

from knn_reliability.geometry import (
    boundary_motion_construction,
    ring_support_construction,
    single_prototype_motion_safe,
    verify_ring_separation,
)
from knn_reliability.knn import build_neighbor_cache


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


def test_boundary_motion_construction_exchanges_the_kth_neighbor():
    construction = boundary_motion_construction()
    base = build_neighbor_cache(construction.points, construction.query, 5)
    moved = construction.points.copy()
    moved[construction.target_index, 0] += construction.motion_radius
    after = build_neighbor_cache(moved, construction.query, 5)
    assert construction.target_index in base.indices[0]
    assert construction.candidate_index not in base.indices[0]
    assert construction.target_index not in after.indices[0]
    assert construction.candidate_index in after.indices[0]
