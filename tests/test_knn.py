import numpy as np

from knn_reliability.knn import (
    build_neighbor_cache,
    loo_neighbor_cache,
    predict_from_counts,
    stable_topk,
)


def test_stable_topk_breaks_equal_distances_by_original_index():
    distances = np.array([[1.0, 0.0, 0.0, 0.5]])
    np.testing.assert_array_equal(stable_topk(distances, 3), [[1, 2, 3]])


def test_prediction_uses_explicit_tie_priority():
    counts = np.array([[2, 2, 0]])
    prediction, winner = predict_from_counts(
        counts, np.array(["a", "b", "c"]), tie_priority=["b", "a", "c"]
    )
    assert prediction.tolist() == ["b"]
    assert winner.tolist() == [1]


def test_loo_excludes_self_and_keeps_requested_neighbors():
    x_train = np.array([[0.0], [1.0], [3.0]])
    cache = loo_neighbor_cache(x_train, 2)
    assert not np.any(cache.indices == np.arange(3)[:, None])
    np.testing.assert_array_equal(cache.indices[0], [1, 2])


def test_query_cache_is_deterministic():
    x_train = np.array([[0.0], [1.0], [2.0]])
    x_query = np.array([[0.5], [1.5]])
    first = build_neighbor_cache(x_train, x_query, 2)
    second = build_neighbor_cache(x_train, x_query, 2)
    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_allclose(first.distances, second.distances)


def test_weighted_vote_handles_zero_distance_without_division_error():
    from knn_reliability.extensions import weighted_predict_from_neighbors

    prediction, scores = weighted_predict_from_neighbors(
        np.array([0, 1]),
        np.array([[0, 1]]),
        np.array([[0.0, 2.0]]),
        classes=[0, 1],
    )
    assert prediction.tolist() == [0]
    np.testing.assert_allclose(scores, [[1.0, 0.0]])


def test_weighted_power_zero_matches_unweighted_vote_at_zero_distance():
    from knn_reliability.extensions import weighted_predict_from_neighbors

    prediction, scores = weighted_predict_from_neighbors(
        np.array([0, 1, 1]),
        np.array([[0, 1, 2]]),
        np.array([[0.0, 1.0, 1.0]]),
        classes=[0, 1],
        power=0,
    )
    assert prediction.tolist() == [1]
    np.testing.assert_allclose(scores, [[1.0, 2.0]])
