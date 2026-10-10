"""Exact and simulation-based reliability quantities for label perturbations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .knn import counts_from_neighbors, predict_from_counts, predict_from_neighbors


def poisson_binomial_pmf(probabilities: Iterable[float]) -> np.ndarray:
    """Return the Poisson-binomial mass function for a list of Bernoulli rates."""

    probabilities = np.asarray(list(probabilities), dtype=float)
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    pmf = np.array([1.0], dtype=float)
    for probability in probabilities:
        updated = np.zeros(len(pmf) + 1, dtype=float)
        updated[:-1] += pmf * (1.0 - probability)
        updated[1:] += pmf * probability
        pmf = updated
    return pmf


def odd_k_boundary_probability(
    class_one_probabilities: Iterable[float],
) -> float:
    """Probability that an odd binary vote has a one-vote margin.

    The inputs are conditional class-one probabilities for the fixed
    neighborhood.  This is the conditional vulnerability probability under
    independent labels, not a posterior probability for real annotation
    errors.
    """

    probabilities = np.asarray(list(class_one_probabilities), dtype=float)
    if len(probabilities) == 0 or len(probabilities) % 2 == 0:
        raise ValueError("an odd, non-empty neighborhood is required")
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("class-one probabilities must lie in [0, 1]")
    pmf = poisson_binomial_pmf(probabilities)
    middle = len(probabilities) // 2
    return float(pmf[middle] + pmf[middle + 1])


def _binary_labels(
    y_neighbors: np.ndarray,
    classes: Iterable[object] | None,
) -> np.ndarray:
    values = np.asarray(y_neighbors)
    if values.ndim != 1:
        raise ValueError("y_neighbors must be one-dimensional")
    labels = np.asarray(list(np.unique(values) if classes is None else classes))
    if len(labels) != 2:
        raise ValueError("the exact binary probability requires exactly two classes")
    if not np.all(np.isin(values, labels)):
        raise ValueError("neighbor labels must belong to classes")
    return labels


def binary_flip_probability(
    y_neighbors: np.ndarray,
    flip_probabilities: Iterable[float],
    *,
    classes: Iterable[object] | None = None,
    tie_priority: Iterable[object] | None = None,
) -> float:
    """Exact probability that one binary query prediction changes.

    The local labels are fixed and only their independent replacement events
    are random.  This is a Poisson-binomial calculation and handles even k,
    heterogeneous probabilities, and deterministic tie priority exactly.
    """

    y_neighbors = np.asarray(y_neighbors)
    probabilities = np.asarray(list(flip_probabilities), dtype=float)
    if len(y_neighbors) != len(probabilities):
        raise ValueError("labels and flip_probabilities must have equal length")
    if len(y_neighbors) == 0:
        raise ValueError("at least one neighbor is required")
    labels = _binary_labels(y_neighbors, classes)
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("flip_probabilities must lie in [0, 1]")
    base_counts = np.array([[np.sum(y_neighbors == label) for label in labels]], dtype=int)
    base_prediction, _ = predict_from_counts(base_counts, labels, tie_priority=tie_priority)
    class_zero_probabilities = np.where(
        y_neighbors == labels[0], 1.0 - probabilities, probabilities
    )
    pmf = poisson_binomial_pmf(class_zero_probabilities)
    changed = 0.0
    for count_zero, mass in enumerate(pmf):
        counts = np.array([[count_zero, len(y_neighbors) - count_zero]], dtype=int)
        prediction, _ = predict_from_counts(counts, labels, tie_priority=tie_priority)
        if prediction[0] != base_prediction[0]:
            changed += float(mass)
    return float(changed)


def first_order_risk(
    single_flip_influence: Iterable[float],
    flip_probabilities: Iterable[float],
    n_queries: int,
) -> float:
    """The derivative-at-zero linear risk approximation.

    ``single_flip_influence[i]`` is the number of queries changed when only
    prototype i is relabeled to the other binary class.  The expression is
    exact as the directional derivative at the all-zero probability vector;
    away from zero it is intentionally labelled an approximation.
    """

    influence = np.asarray(list(single_flip_influence), dtype=float)
    probabilities = np.asarray(list(flip_probabilities), dtype=float)
    if len(influence) != len(probabilities):
        raise ValueError("influence and flip_probabilities must have equal length")
    if n_queries <= 0:
        raise ValueError("n_queries must be positive")
    return float(np.dot(influence, probabilities) / n_queries)


def first_order_variance_bounds(
    point_risk: float,
    r_one: float,
    n_train: int,
    k: int,
) -> tuple[float, float]:
    """Return the odd-binary first-order variance bounds.

    For ``c=(k+1)/2``, the pivotal-incidence identity gives
    ``c**2 * point_risk**2 / n_train`` as the lower bound and
    ``c * point_risk * r_one`` as the upper bound.  The result is an
    asymptotic coefficient for independent small-probability flips, not a
    finite-probability variance bound.
    """

    point_risk = float(point_risk)
    r_one = float(r_one)
    n_train = int(n_train)
    k = int(k)
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    if k <= 0 or k % 2 == 0:
        raise ValueError("k must be a positive odd integer")
    if point_risk < 0.0 or r_one < 0.0:
        raise ValueError("risks must be non-negative")
    c = (k + 1) / 2.0
    return (
        float((c**2 / n_train) * point_risk**2),
        float(c * point_risk * r_one),
    )


def enumerate_shared_flip_moments(
    y_train: np.ndarray,
    neighbors: np.ndarray,
    flip_probability: float,
    *,
    classes: Iterable[object] | None = None,
    tie_priority: Iterable[object] | None = None,
    max_prototypes: int = 20,
) -> tuple[float, float]:
    """Enumerate exact batch-risk moments for independent binary flips.

    This small-state audit enumerates every binary flip vector and is intended
    for validating the first-order variance expansion, not for production
    computation.  The production path is ``batch_risk_moments``.
    """

    y_train = np.asarray(y_train)
    neighbors = np.asarray(neighbors, dtype=int)
    probability = float(flip_probability)
    if y_train.ndim != 1 or len(y_train) == 0:
        raise ValueError("y_train must be a non-empty one-dimensional array")
    if neighbors.ndim != 2 or len(neighbors) == 0:
        raise ValueError("neighbors must be a non-empty two-dimensional array")
    if len(y_train) > max_prototypes:
        raise ValueError("state enumeration exceeds max_prototypes")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("flip_probability must lie in [0, 1]")
    if np.any(neighbors < 0) or np.any(neighbors >= len(y_train)):
        raise ValueError("neighbors contain an invalid training index")
    labels, _ = counts_from_neighbors(y_train, neighbors, classes=classes)
    if len(labels) != 2:
        raise ValueError("enumeration currently supports binary labels")
    baseline, _, _ = predict_from_neighbors(
        y_train, neighbors, classes=labels, tie_priority=tie_priority
    )
    probability_mass = 0.0
    first_moment = 0.0
    second_moment = 0.0
    for mask in range(1 << len(y_train)):
        flips = ((mask >> np.arange(len(y_train))) & 1).astype(bool)
        perturbed = y_train.copy()
        perturbed[flips] = np.where(
            perturbed[flips] == labels[0], labels[1], labels[0]
        )
        predictions, _, _ = predict_from_neighbors(
            perturbed, neighbors, classes=labels, tie_priority=tie_priority
        )
        risk = float(np.mean(predictions != baseline))
        mass = probability ** int(flips.sum()) * (1.0 - probability) ** int(
            len(y_train) - flips.sum()
        )
        probability_mass += mass
        first_moment += mass * risk
        second_moment += mass * risk * risk
    if not np.isclose(probability_mass, 1.0, atol=1e-12):
        raise RuntimeError("flip-state masses do not sum to one")
    return float(first_moment), float(max(0.0, second_moment - first_moment**2))


def _postflip_class_zero_probability(label: object, class_zero: object, probability: float) -> float:
    return (1.0 - probability) if label == class_zero else probability


def _joint_binary_flip_probability(
    labels_q: np.ndarray,
    labels_r: np.ndarray,
    indices_q: np.ndarray,
    indices_r: np.ndarray,
    y_train: np.ndarray,
    flip_probabilities: np.ndarray,
    class_labels: np.ndarray,
    tie_priority: Iterable[object] | None,
) -> float:
    """Exact joint event probability for two queries under shared flips."""

    class_zero = class_labels[0]
    base_q, _ = predict_from_counts(
        np.array([[np.sum(labels_q == class_labels[0]), np.sum(labels_q == class_labels[1])]]),
        class_labels,
        tie_priority=tie_priority,
    )
    base_r, _ = predict_from_counts(
        np.array([[np.sum(labels_r == class_labels[0]), np.sum(labels_r == class_labels[1])]]),
        class_labels,
        tie_priority=tie_priority,
    )
    q_positions = {int(index): pos for pos, index in enumerate(indices_q)}
    r_positions = {int(index): pos for pos, index in enumerate(indices_r)}
    union = sorted(set(q_positions) | set(r_positions))
    dp = np.zeros((len(indices_q) + 1, len(indices_r) + 1), dtype=float)
    dp[0, 0] = 1.0
    for train_idx in union:
        q_contribution = 1 if train_idx in q_positions else 0
        r_contribution = 1 if train_idx in r_positions else 0
        p_zero = _postflip_class_zero_probability(
            y_train[train_idx], class_zero, flip_probabilities[train_idx]
        )
        # Each prototype contributes either (0,0), (1,0), (0,1), or (1,1)
        # to the two local class-zero counts.  Array shifts preserve the DP
        # exactly while avoiding a Python loop over every count state.
        updated = dp * (1.0 - p_zero)
        if q_contribution and r_contribution:
            updated[1:, 1:] += dp[:-1, :-1] * p_zero
        elif q_contribution:
            updated[1:, :] += dp[:-1, :] * p_zero
        elif r_contribution:
            updated[:, 1:] += dp[:, :-1] * p_zero
        else:
            updated += dp * p_zero
        dp = updated
    # Precompute the deterministic winner for every possible count once.  A
    # repeated generic classifier call inside the q-by-r state loop made the
    # exact shared-risk audit needlessly expensive without changing results.
    q_counts = np.arange(len(indices_q) + 1)
    r_counts = np.arange(len(indices_r) + 1)
    q_state_predictions, _ = predict_from_counts(
        np.column_stack([q_counts, len(indices_q) - q_counts]),
        class_labels,
        tie_priority=tie_priority,
    )
    r_state_predictions, _ = predict_from_counts(
        np.column_stack([r_counts, len(indices_r) - r_counts]),
        class_labels,
        tie_priority=tie_priority,
    )
    q_changed = q_state_predictions != base_q[0]
    r_changed = r_state_predictions != base_r[0]
    return float(dp[np.ix_(q_changed, r_changed)].sum())


def factorized_joint_binary_flip_probability(
    labels_q: np.ndarray,
    labels_r: np.ndarray,
    indices_q: np.ndarray,
    indices_r: np.ndarray,
    y_train: np.ndarray,
    flip_probabilities: Iterable[float],
    *,
    classes: Iterable[object] | None = None,
    tie_priority: Iterable[object] | None = None,
) -> float:
    """Exact joint change probability using a shared/unique factorization.

    The common neighborhood contributes one Poisson-binomial variable ``U``.
    Conditional on ``U``, the query-specific parts are independent and their
    conditional change probabilities are one-dimensional Poisson-binomial
    tail sums.  This is algebraically equivalent to the two-query DP but uses
    O(k) working memory and O(k**2) arithmetic for one edge.
    """

    y_train = np.asarray(y_train)
    labels_q = np.asarray(labels_q)
    labels_r = np.asarray(labels_r)
    indices_q = np.asarray(indices_q, dtype=int)
    indices_r = np.asarray(indices_r, dtype=int)
    probabilities = np.asarray(list(flip_probabilities), dtype=float)
    class_labels = _binary_labels(
        np.concatenate([labels_q, labels_r]), classes
    )
    if len(probabilities) != len(y_train):
        raise ValueError("flip_probabilities must match y_train")
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("flip_probabilities must lie in [0, 1]")
    if len(labels_q) != len(indices_q) or len(labels_r) != len(indices_r):
        raise ValueError("labels and neighbor indices must have equal lengths")
    if np.any(indices_q < 0) or np.any(indices_q >= len(y_train)):
        raise ValueError("indices_q contains an invalid training index")
    if np.any(indices_r < 0) or np.any(indices_r >= len(y_train)):
        raise ValueError("indices_r contains an invalid training index")
    if len(np.unique(indices_q)) != len(indices_q):
        raise ValueError("indices_q must not repeat a training index")
    if len(np.unique(indices_r)) != len(indices_r):
        raise ValueError("indices_r must not repeat a training index")
    if not np.array_equal(labels_q, y_train[indices_q]):
        raise ValueError("labels_q must agree with y_train at indices_q")
    if not np.array_equal(labels_r, y_train[indices_r]):
        raise ValueError("labels_r must agree with y_train at indices_r")

    return _factorized_joint_binary_flip_probability_local(
        labels_q, labels_r, indices_q, indices_r, y_train, probabilities,
        class_labels, tie_priority,
    )


def _factorized_joint_binary_flip_probability_local(
    labels_q: np.ndarray,
    labels_r: np.ndarray,
    indices_q: np.ndarray,
    indices_r: np.ndarray,
    y_train: np.ndarray,
    probabilities: np.ndarray,
    class_labels: np.ndarray,
    tie_priority: Iterable[object] | None,
) -> float:
    """Factorized edge kernel for arrays validated by the batch entry point."""

    q_set = set(int(index) for index in indices_q)
    r_set = set(int(index) for index in indices_r)
    common = sorted(q_set & r_set)
    q_only = sorted(q_set - r_set)
    r_only = sorted(r_set - q_set)
    class_zero = class_labels[0]

    def class_zero_rates(indices: list[int]) -> np.ndarray:
        return np.asarray(
            [
                _postflip_class_zero_probability(
                    y_train[index], class_zero, probabilities[index]
                )
                for index in indices
            ],
            dtype=float,
        )

    shared_pmf = poisson_binomial_pmf(class_zero_rates(common))
    q_only_pmf = poisson_binomial_pmf(class_zero_rates(q_only))
    r_only_pmf = poisson_binomial_pmf(class_zero_rates(r_only))

    base_q, _ = predict_from_counts(
        np.array([[np.sum(labels_q == class_zero), np.sum(labels_q != class_zero)]]),
        class_labels,
        tie_priority=tie_priority,
    )
    base_r, _ = predict_from_counts(
        np.array([[np.sum(labels_r == class_zero), np.sum(labels_r != class_zero)]]),
        class_labels,
        tie_priority=tie_priority,
    )
    q_changed_given_shared = np.zeros(len(shared_pmf), dtype=float)
    r_changed_given_shared = np.zeros(len(shared_pmf), dtype=float)
    q_total = len(indices_q)
    r_total = len(indices_r)
    for shared_count in range(len(shared_pmf)):
        q_counts = np.column_stack(
            [
                shared_count + np.arange(len(q_only_pmf)),
                q_total - shared_count - np.arange(len(q_only_pmf)),
            ]
        )
        r_counts = np.column_stack(
            [
                shared_count + np.arange(len(r_only_pmf)),
                r_total - shared_count - np.arange(len(r_only_pmf)),
            ]
        )
        q_predictions, _ = predict_from_counts(
            q_counts, class_labels, tie_priority=tie_priority
        )
        r_predictions, _ = predict_from_counts(
            r_counts, class_labels, tie_priority=tie_priority
        )
        q_changed_given_shared[shared_count] = float(
            np.dot(q_only_pmf, q_predictions != base_q[0])
        )
        r_changed_given_shared[shared_count] = float(
            np.dot(r_only_pmf, r_predictions != base_r[0])
        )
    return float(
        np.dot(shared_pmf, q_changed_given_shared * r_changed_given_shared)
    )


@dataclass(frozen=True)
class BatchRiskMoments:
    """Moments of the batch fraction changed by one shared label draw."""

    query_probabilities: np.ndarray
    expectation: float
    second_moment: float
    variance: float
    overlap_pairs: np.ndarray
    pair_scan_count: int = 0
    joint_dp_count: int = 0
    explicit_independent_product_count: int = 0
    aggregated_nonedge_pair_count: int = 0

    @property
    def overlap_pair_count(self) -> int:
        return int(len(self.overlap_pairs))

    @property
    def total_query_pairs(self) -> int:
        n_queries = len(self.query_probabilities)
        return n_queries * (n_queries - 1) // 2

    @property
    def total_nonedge_pair_count(self) -> int:
        return self.total_query_pairs - self.overlap_pair_count


def query_overlap_pairs(neighbors: np.ndarray) -> np.ndarray:
    """Return query pairs sharing at least one training prototype.

    The result is the edge list of the query-neighborhood overlap graph.  It
    is deduplicated and sorted, so only these pairs require a joint dynamic
    program in the exact shared-flip moment calculation.
    """

    neighbors = np.asarray(neighbors, dtype=int)
    if neighbors.ndim != 2:
        raise ValueError("neighbors must be a two-dimensional array")
    memberships: dict[int, list[int]] = {}
    for query_index, row in enumerate(neighbors):
        for prototype_index in np.unique(row):
            memberships.setdefault(int(prototype_index), []).append(query_index)
    pairs: set[tuple[int, int]] = set()
    for query_indices in memberships.values():
        ordered = sorted(query_indices)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1:]:
                pairs.add((left, right))
    if not pairs:
        return np.empty((0, 2), dtype=int)
    return np.asarray(sorted(pairs), dtype=int)


def batch_risk_moments_variant(
    y_train: np.ndarray,
    neighbors: np.ndarray,
    flip_probabilities: Iterable[float],
    *,
    classes: Iterable[object] | None = None,
    tie_priority: Iterable[object] | None = None,
    overlap_graph: bool = True,
    factorized_joint: bool = True,
) -> BatchRiskMoments:
    """Compute exact binary batch moments under an explicit implementation variant.

    ``overlap_graph`` controls whether the pair expansion scans only shared
    neighborhoods or all query pairs.  ``factorized_joint`` selects the
    shared/unique one-dimensional calculation or the retained two-dimensional
    reference calculation for each pair.  The four combinations are used by
    E9 to separate the benefit of graph sparsity from the benefit of the new
    joint-probability factorization.
    """

    y_train = np.asarray(y_train)
    neighbors = np.asarray(neighbors, dtype=int)
    probabilities = np.asarray(list(flip_probabilities), dtype=float)
    if y_train.ndim != 1 or len(y_train) == 0:
        raise ValueError("y_train must be a non-empty one-dimensional array")
    if neighbors.ndim != 2 or len(neighbors) == 0:
        raise ValueError("neighbors must be a non-empty two-dimensional array")
    if len(probabilities) != len(y_train):
        raise ValueError("flip_probabilities must match y_train")
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("flip_probabilities must be finite values in [0, 1]")
    if np.any(neighbors < 0) or np.any(neighbors >= len(y_train)):
        raise ValueError("neighbors contain an invalid training index")
    if any(len(np.unique(row)) != len(row) for row in neighbors):
        raise ValueError("each query neighborhood must not repeat a training index")
    labels, counts = counts_from_neighbors(y_train, neighbors, classes=classes)
    if len(labels) != 2:
        raise ValueError("batch_risk_moments_variant currently supports binary labels")
    query_probabilities = np.array(
        [
            binary_flip_probability(
                y_train[indices],
                probabilities[indices],
                classes=labels,
                tie_priority=tie_priority,
            )
            for indices in neighbors
        ],
        dtype=float,
    )
    expectation = float(np.mean(query_probabilities))
    n_queries = len(neighbors)
    overlap_pairs = query_overlap_pairs(neighbors)
    if overlap_graph:
        pairs = overlap_pairs
    else:
        pairs = np.asarray(
            [(left, right) for left in range(n_queries) for right in range(left + 1, n_queries)],
            dtype=int,
        )
        if len(pairs) == 0:
            pairs = np.empty((0, 2), dtype=int)

    pair_sum = 0.0
    pair_scan_count = 0
    joint_dp_count = 0
    explicit_independent_product_count = 0
    aggregated_nonedge_pair_count = 0
    neighbor_sets = [set(int(index) for index in row) for row in neighbors]
    for left, right in pairs:
        pair_scan_count += 1
        if not (neighbor_sets[left] & neighbor_sets[right]):
            # Both dense variants perform the same independence test. Disjoint
            # events have an exact marginal product and need no joint DP.
            pair_sum += float(query_probabilities[left] * query_probabilities[right])
            explicit_independent_product_count += 1
            continue
        joint_dp_count += 1
        if factorized_joint:
            pair_sum += _factorized_joint_binary_flip_probability_local(
                y_train[neighbors[left]],
                y_train[neighbors[right]],
                neighbors[left],
                neighbors[right],
                y_train,
                probabilities,
                labels,
                tie_priority,
            )
        else:
            pair_sum += _joint_binary_flip_probability(
                y_train[neighbors[left]],
                y_train[neighbors[right]],
                neighbors[left],
                neighbors[right],
                y_train,
                probabilities,
                labels,
                tie_priority,
            )

    if overlap_graph:
        # Nonedges depend on disjoint independent flip coordinates and
        # therefore contribute the product of their marginal probabilities.
        total_pair_product = (
            float(np.sum(query_probabilities)) ** 2
            - float(np.dot(query_probabilities, query_probabilities))
        ) / 2.0
        overlap_product = float(
            sum(query_probabilities[left] * query_probabilities[right] for left, right in overlap_pairs)
        )
        pair_sum = total_pair_product - overlap_product + pair_sum
        aggregated_nonedge_pair_count = n_queries * (n_queries - 1) // 2 - len(overlap_pairs)

    second_numerator = float(np.sum(query_probabilities)) + 2.0 * pair_sum
    second_moment = float(second_numerator / (n_queries * n_queries))
    variance = max(0.0, second_moment - expectation * expectation)
    return BatchRiskMoments(
        query_probabilities,
        expectation,
        second_moment,
        variance,
        overlap_pairs,
        pair_scan_count,
        joint_dp_count,
        explicit_independent_product_count,
        aggregated_nonedge_pair_count,
    )


def batch_risk_moments(
    y_train: np.ndarray,
    neighbors: np.ndarray,
    flip_probabilities: Iterable[float],
    *,
    classes: Iterable[object] | None = None,
    tie_priority: Iterable[object] | None = None,
) -> BatchRiskMoments:
    """Compute exact moments with the production sparse one-dimensional path."""

    return batch_risk_moments_variant(
        y_train,
        neighbors,
        flip_probabilities,
        classes=classes,
        tie_priority=tie_priority,
        overlap_graph=True,
        factorized_joint=True,
    )


def dense_batch_risk_moments(
    y_train: np.ndarray,
    neighbors: np.ndarray,
    flip_probabilities: Iterable[float],
    *,
    classes: Iterable[object] | None = None,
    tie_priority: Iterable[object] | None = None,
) -> BatchRiskMoments:
    """Reference moments with dense pair scanning and the two-dimensional DP."""

    return batch_risk_moments_variant(
        y_train,
        neighbors,
        flip_probabilities,
        classes=classes,
        tie_priority=tie_priority,
        overlap_graph=False,
        factorized_joint=False,
    )


def monte_carlo_batch_risk(
    y_train: np.ndarray,
    neighbors: np.ndarray,
    flip_probabilities: Iterable[float],
    *,
    repetitions: int = 1000,
    classes: Iterable[object] | None = None,
    tie_priority: Iterable[object] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Estimate batch risk by drawing one shared label vector per repetition."""

    y_train = np.asarray(y_train)
    neighbors = np.asarray(neighbors, dtype=int)
    probabilities = np.asarray(list(flip_probabilities), dtype=float)
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if len(probabilities) != len(y_train):
        raise ValueError("flip_probabilities must match y_train")
    labels, counts = counts_from_neighbors(y_train, neighbors, classes=classes)
    baseline, _ = predict_from_counts(counts, labels, tie_priority=tie_priority)
    rng = np.random.default_rng(seed)
    risks = np.empty(repetitions, dtype=float)
    for repeat in range(repetitions):
        flips = rng.random(len(y_train)) < probabilities
        perturbed = y_train.copy()
        if len(labels) != 2:
            raise ValueError("monte_carlo_batch_risk currently supports binary labels")
        perturbed[flips] = np.where(perturbed[flips] == labels[0], labels[1], labels[0])
        changed, _, _ = predict_from_neighbors(
            perturbed, neighbors, classes=labels, tie_priority=tie_priority
        )
        risks[repeat] = np.mean(changed != baseline)
    return risks


def monte_carlo_perfectly_correlated_batch_risk(
    y_train: np.ndarray,
    neighbors: np.ndarray,
    flip_probability: float,
    *,
    classes: Iterable[object] | None = None,
    tie_priority: Iterable[object] | None = None,
    repetitions: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Estimate batch risk when all prototype flips share one Bernoulli draw.

    This deliberately violates the independent-flip assumption while keeping
    the same marginal flip probability for every prototype.  It is used only
    as a model-misspecification stress test for the independent exact engine.
    """

    if not 0.0 <= float(flip_probability) <= 1.0:
        raise ValueError("flip_probability must lie in [0, 1]")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    y_train = np.asarray(y_train)
    neighbors = np.asarray(neighbors, dtype=int)
    labels, _ = counts_from_neighbors(y_train, neighbors, classes=classes)
    if len(labels) != 2:
        raise ValueError("the correlated binary stress test requires two classes")
    baseline, _, _ = predict_from_neighbors(
        y_train, neighbors, classes=labels, tie_priority=tie_priority
    )
    rng = np.random.default_rng(seed)
    risks = np.empty(repetitions, dtype=float)
    for repeat in range(repetitions):
        if rng.random() < flip_probability:
            perturbed = y_train.copy()
            perturbed[:] = np.where(perturbed == labels[0], labels[1], labels[0])
        else:
            perturbed = y_train
        changed, _, _ = predict_from_neighbors(
            perturbed, neighbors, classes=labels, tie_priority=tie_priority
        )
        risks[repeat] = np.mean(changed != baseline)
    return risks
