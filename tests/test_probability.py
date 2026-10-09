import itertools

import numpy as np

from knn_reliability.knn import predict_from_counts
from knn_reliability.probability import (
    batch_risk_moments,
    binary_flip_probability,
    first_order_risk,
    monte_carlo_batch_risk,
    poisson_binomial_pmf,
    query_overlap_pairs,
)


def test_poisson_binomial_matches_small_enumeration():
    probabilities = np.array([0.1, 0.4, 0.8])
    expected = np.zeros(4)
    for flips in itertools.product([0, 1], repeat=3):
        mass = np.prod(
            [p if flip else 1.0 - p for p, flip in zip(probabilities, flips)]
        )
        expected[sum(flips)] += mass
    np.testing.assert_allclose(poisson_binomial_pmf(probabilities), expected)


def test_binary_probability_matches_shared_flip_enumeration():
    neighbors = np.array([0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    exact = binary_flip_probability(neighbors, probabilities, classes=[0, 1])
    brute = 0.0
    for flips in itertools.product([0, 1], repeat=len(neighbors)):
        mass = np.prod(
            [p if flip else 1.0 - p for p, flip in zip(probabilities, flips)]
        )
        perturbed = neighbors.copy()
        for index, flip in enumerate(flips):
            if flip:
                perturbed[index] = 1 - perturbed[index]
        base, _ = predict_from_counts(
            np.array([[np.sum(neighbors == 0), np.sum(neighbors == 1)]]),
            np.array([0, 1]),
        )
        changed, _ = predict_from_counts(
            np.array([[np.sum(perturbed == 0), np.sum(perturbed == 1)]]),
            np.array([0, 1]),
        )
        brute += mass * float(base[0] != changed[0])
    assert abs(exact - brute) < 1e-12


def test_first_order_risk_is_directional_derivative():
    assert first_order_risk([2, 1], [0.01, 0.02], 4) == 0.01


def test_batch_moments_keep_overlap_covariance():
    y_train = np.array([0, 0, 1])
    neighbors = np.array([[0, 1, 2], [0, 1, 2]])
    probabilities = np.array([0.1, 0.1, 0.1])
    moments = batch_risk_moments(y_train, neighbors, probabilities, classes=[0, 1])
    assert moments.expectation > 0
    assert moments.variance > 0
    risks = monte_carlo_batch_risk(
        y_train, neighbors, probabilities, repetitions=2000, classes=[0, 1], seed=4
    )
    assert abs(risks.mean() - moments.expectation) < 0.04


def test_overlap_graph_deduplicates_shared_prototype_edges():
    neighbors = np.array([[0, 1], [1, 2], [0, 2], [3, 4]])
    np.testing.assert_array_equal(
        query_overlap_pairs(neighbors),
        np.array([[0, 1], [0, 2], [1, 2]]),
    )


def test_disjoint_batch_pairs_use_marginal_products():
    y_train = np.array([0, 0, 1, 0, 1, 1])
    neighbors = np.array([[0, 1, 2], [3, 4, 5]])
    probabilities = np.full(len(y_train), 0.1)
    moments = batch_risk_moments(y_train, neighbors, probabilities, classes=[0, 1])
    assert moments.overlap_pair_count == 0
    assert moments.total_query_pairs == 1
    p0, p1 = moments.query_probabilities
    expected_variance = (p0 * (1.0 - p0) + p1 * (1.0 - p1)) / 4.0
    np.testing.assert_allclose(moments.variance, expected_variance)
