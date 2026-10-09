import numpy as np

from knn_reliability.influence import (
    brute_force_directional_influence,
    compute_global_single_relabel_risk,
    compute_influence,
    compute_neighborhood_exposure,
    compute_signed_error_change,
)


def test_directional_influence_matches_slow_reference_multiclass():
    y_train = np.array([0, 0, 1, 1, 2, 2])
    neighbors = np.array([[0, 2, 4], [1, 3, 5], [0, 1, 2], [3, 4, 5]])
    fast = compute_influence(y_train, neighbors, tie_priority=[0, 1, 2])
    slow = brute_force_directional_influence(
        y_train, neighbors, tie_priority=[0, 1, 2]
    )
    np.testing.assert_array_equal(fast.directional_influence, slow)


def test_all_decisive_prototypes_are_counted_without_early_break():
    y_train = np.array([0, 0, 1])
    neighbors = np.array([[0, 1, 2]])
    result = compute_influence(y_train, neighbors, classes=[0, 1])
    assert result.exact_vulnerable.tolist() == [True]
    assert result.decisive_influence.tolist() == [1, 1, 0]
    assert result.neighborhood_exposure.tolist() == [1, 1, 1]


def test_binary_odd_k_pivotal_identity():
    y_train = np.array([0, 0, 1, 1, 1])
    neighbors = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    result = compute_influence(y_train, neighbors, classes=[0, 1])
    assert result.exact_vulnerable.all()
    assert result.decisive_influence.sum() == 2 * 3


def test_point_global_bound_uses_training_prototype_count():
    # Six disjoint 1-NN queries make every query vulnerable, while each
    # prototype can change only its own query.
    y_train = np.array([0, 1, 0, 1, 0, 1])
    neighbors = np.arange(6, dtype=int)[:, None]
    result = compute_influence(y_train, neighbors, classes=[0, 1])
    point_risk = float(np.mean(result.exact_vulnerable))
    global_risk = compute_global_single_relabel_risk(result)["risk"]
    corrected_lower = ((1 + 1) / (2 * len(y_train))) * point_risk
    assert point_risk == 1.0
    assert global_risk == corrected_lower
    assert global_risk < point_risk
    assert global_risk < point_risk / 1


def test_neighborhood_exposure_preserves_unvisited_training_prototypes():
    neighbors = np.array([[0, 2], [2, 0]])
    exposure = compute_neighborhood_exposure(
        neighbors, np.array([True, False]), n_train=5
    )
    np.testing.assert_array_equal(exposure, np.array([1, 0, 1, 0, 0]))


def test_signed_error_change_distinguishes_helpful_and_harmful_relabels():
    y_train = np.array([0, 0, 1])
    neighbors = np.array([[0, 1, 2]])
    y_query = np.array([1])
    helpful = compute_signed_error_change(y_train, neighbors, y_query, classes=[0, 1])
    assert helpful.signed_error_change[0, 1] == -1.0
    assert helpful.prediction_change_rate[0, 1] == 1.0

    y_train = np.array([0, 1, 1])
    y_query = np.array([1])
    harmful = compute_signed_error_change(y_train, neighbors, y_query, classes=[0, 1])
    assert harmful.signed_error_change[1, 0] == 1.0
    assert harmful.prediction_change_rate[1, 0] == 1.0
