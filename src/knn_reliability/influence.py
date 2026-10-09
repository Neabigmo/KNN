"""Exact prototype influence under fixed kNN neighborhoods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .knn import counts_from_neighbors, predict_from_counts, predict_from_neighbors


@dataclass(frozen=True)
class InfluenceResult:
    predictions: np.ndarray
    classes: np.ndarray
    counts: np.ndarray
    winner_indices: np.ndarray
    top_two_gap: np.ndarray
    band_flags: np.ndarray
    exact_vulnerable: np.ndarray
    neighborhood_exposure: np.ndarray
    decisive_influence: np.ndarray
    directional_influence: np.ndarray
    decisive_matrix: np.ndarray


@dataclass(frozen=True)
class SignedErrorChangeResult:
    """Directional prediction and error changes for a fixed query batch."""

    signed_error_change: np.ndarray
    changed_error_rate: np.ndarray
    prediction_change_rate: np.ndarray
    baseline_error_rate: float


def compute_band_flags(counts: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Return the conservative vote-margin screening band."""

    return conservative_band(np.asarray(counts), np.asarray(classes))


def compute_exact_prv(result: InfluenceResult) -> np.ndarray:
    """Return the exact per-query single-prototype relabeling indicator."""

    return result.exact_vulnerable.copy()


def compute_neighborhood_exposure(
    neighbors: np.ndarray,
    query_mask: np.ndarray,
    *,
    n_train: int,
) -> np.ndarray:
    """Count query neighborhoods containing each training prototype."""

    neighbors = np.asarray(neighbors, dtype=int)
    query_mask = np.asarray(query_mask, dtype=bool)
    if neighbors.ndim != 2 or len(query_mask) != len(neighbors):
        raise ValueError("neighbors and query_mask have incompatible shapes")
    if not isinstance(n_train, (int, np.integer)) or int(n_train) < 0:
        raise ValueError("n_train must be a non-negative integer")
    n_train = int(n_train)
    if neighbors.size and (np.any(neighbors < 0) or np.any(neighbors >= n_train)):
        raise ValueError("neighbors contain an index outside n_train")
    exposure = np.zeros(n_train, dtype=int)
    for row in neighbors[query_mask]:
        exposure[np.unique(row)] += 1
    return exposure


def compute_decisive_influence(result: InfluenceResult) -> np.ndarray:
    return result.decisive_influence.copy()


def compute_directional_influence(result: InfluenceResult) -> np.ndarray:
    return result.directional_influence.copy()


def compute_global_single_relabel_risk(
    result: InfluenceResult,
    *,
    denominator: int | None = None,
) -> dict[str, object]:
    """Summarize the worst one-prototype, one-direction global change."""

    q = int(denominator if denominator is not None else len(result.predictions))
    if q <= 0:
        raise ValueError("denominator must be positive")
    flat_index = int(np.argmax(result.directional_influence))
    prototype, target = np.unravel_index(
        flat_index, result.directional_influence.shape
    )
    return {
        "prototype_index": prototype,
        "target_class": result.classes[target],
        "changed_queries": int(result.directional_influence[prototype, target]),
        "risk": float(result.directional_influence[prototype, target] / q),
    }


def compute_classwise_influence(
    result: InfluenceResult,
    y_query: np.ndarray,
) -> dict[object, dict[str, float]]:
    """Report exact PRV and directional risk by query true class."""

    y_query = np.asarray(y_query)
    if len(y_query) != len(result.predictions):
        raise ValueError("y_query must match the number of analyzed queries")
    output: dict[object, dict[str, float]] = {}
    for label in np.unique(y_query):
        mask = y_query == label
        denominator = int(mask.sum())
        output[label.item() if hasattr(label, "item") else label] = {
            "n_queries": float(denominator),
            "prv": float(result.exact_vulnerable[mask].mean()) if denominator else float("nan"),
            "mean_decisive_influence": float(result.decisive_matrix[mask].sum(axis=1).mean())
            if denominator
            else float("nan"),
        }
    return output


def compute_signed_error_change(
    y_train: np.ndarray,
    neighbors: np.ndarray,
    y_query: np.ndarray,
    *,
    classes: Iterable[object] | None = None,
    tie_priority: Iterable[object] | None = None,
) -> SignedErrorChangeResult:
    """Compute directional signed test-error changes for every prototype.

    Entry ``(i, c)`` is the change in error rate after replacing prototype
    ``i``'s label with class ``c`` while keeping locations and the query batch
    fixed. Positive values worsen classification and negative values improve it.
    """

    y_train = np.asarray(y_train)
    neighbors = np.asarray(neighbors, dtype=int)
    y_query = np.asarray(y_query)
    if neighbors.ndim != 2 or len(y_query) != len(neighbors):
        raise ValueError("neighbors and y_query have incompatible shapes")
    labels, counts = counts_from_neighbors(y_train, neighbors, classes=classes)
    baseline, _ = predict_from_counts(counts, labels, tie_priority=tie_priority)
    baseline_errors = baseline != y_query
    baseline_error_rate = float(np.mean(baseline_errors))
    class_to_idx = {label: index for index, label in enumerate(labels)}
    signed = np.full((len(y_train), len(labels)), np.nan, dtype=float)
    changed_error = np.full_like(signed, np.nan)
    changed_rate = np.full_like(signed, np.nan)

    for train_index, old_class in enumerate(y_train):
        old_index = class_to_idx[old_class]
        query_mask = np.any(neighbors == train_index, axis=1)
        for target_index, target_class in enumerate(labels):
            if target_class == old_class:
                continue
            if not np.any(query_mask):
                signed[train_index, target_index] = 0.0
                changed_error[train_index, target_index] = baseline_error_rate
                changed_rate[train_index, target_index] = 0.0
                continue
            changed_counts = counts[query_mask].copy()
            changed_counts[:, old_index] -= 1
            changed_counts[:, target_index] += 1
            changed_prediction, _ = predict_from_counts(
                changed_counts, labels, tie_priority=tie_priority
            )
            changed_errors = changed_prediction != y_query[query_mask]
            changed_error_rate = (
                float(np.sum(baseline_errors[~query_mask]))
                + float(np.sum(changed_errors))
            ) / len(y_query)
            signed[train_index, target_index] = changed_error_rate - baseline_error_rate
            changed_error[train_index, target_index] = changed_error_rate
            changed_rate[train_index, target_index] = (
                float(np.sum(changed_prediction != baseline[query_mask])) / len(y_query)
            )

    return SignedErrorChangeResult(
        signed_error_change=signed,
        changed_error_rate=changed_error,
        prediction_change_rate=changed_rate,
        baseline_error_rate=baseline_error_rate,
    )


def top_two_gap(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=int)
    if counts.shape[1] == 1:
        return counts[:, 0].copy()
    ordered = np.sort(counts, axis=1)[:, ::-1]
    return ordered[:, 0] - ordered[:, 1]


def conservative_band(
    counts: np.ndarray,
    classes: np.ndarray,
    *,
    odd_binary_exact: bool = True,
) -> np.ndarray:
    """Return a screening band, not an exact certificate."""

    gap = top_two_gap(counts)
    if len(classes) == 2 and odd_binary_exact:
        k = counts.sum(axis=1)
        # For even k, gap=2 is a conservative candidate band.  Whether the
        # post-change tie flips is resolved by the exact priority-aware test.
        return np.where(k % 2 == 1, gap == 1, gap <= 2)
    return gap <= 2


def _allowed_targets(
    old_class: object,
    classes: np.ndarray,
    allowed_labels: dict[object, Iterable[object]] | None,
) -> list[object]:
    if allowed_labels is None:
        return [c for c in classes if c != old_class]
    values = list(allowed_labels.get(old_class, ()))
    unknown = set(values) - set(classes.tolist())
    if unknown:
        raise ValueError(f"allowed labels not in class set: {unknown}")
    return [c for c in values if c != old_class]


def compute_influence(
    y_train: np.ndarray,
    neighbors: np.ndarray,
    *,
    classes: Iterable[object] | None = None,
    tie_priority: Iterable[object] | None = None,
    allowed_labels: dict[object, Iterable[object]] | None = None,
    exposure_mask: np.ndarray | None = None,
) -> InfluenceResult:
    """Compute all one-label directional effects without an early exit.

    `directional_influence[i, c]` counts query predictions changed by changing
    prototype i to class c. `neighborhood_exposure[i]` counts membership in
    neighborhoods selected by `exposure_mask`; it is intentionally different
    from exact decisive influence.
    """

    y_train = np.asarray(y_train)
    neighbors = np.asarray(neighbors, dtype=int)
    labels, counts = counts_from_neighbors(y_train, neighbors, classes=classes)
    predictions, winner_indices = predict_from_counts(
        counts, labels, tie_priority=tie_priority
    )
    gaps = top_two_gap(counts)
    bands = conservative_band(counts, labels)
    n_queries, n_train = len(neighbors), len(y_train)
    n_classes = len(labels)
    class_to_idx = {label: idx for idx, label in enumerate(labels)}
    directional = np.zeros((n_train, n_classes), dtype=int)
    decisive_matrix = np.zeros((n_queries, n_train), dtype=bool)

    for query_idx, query_neighbors in enumerate(neighbors):
        current_counts = counts[query_idx]
        current_winner = winner_indices[query_idx]
        for train_idx in np.unique(query_neighbors):
            old_class = y_train[train_idx]
            old_idx = class_to_idx[old_class]
            for new_class in _allowed_targets(old_class, labels, allowed_labels):
                new_idx = class_to_idx[new_class]
                changed_counts = current_counts.copy()
                changed_counts[old_idx] -= 1
                changed_counts[new_idx] += 1
                _, changed_winner = predict_from_counts(
                    changed_counts[None, :], labels, tie_priority=tie_priority
                )
                if int(changed_winner[0]) != int(current_winner):
                    directional[train_idx, new_idx] += 1
                    decisive_matrix[query_idx, train_idx] = True

    exact = decisive_matrix.any(axis=1)
    if exposure_mask is None:
        exposure_mask = exact
    exposure_mask = np.asarray(exposure_mask, dtype=bool)
    if len(exposure_mask) != n_queries:
        raise ValueError("exposure_mask must have one value per query")
    exposure = np.zeros(n_train, dtype=int)
    for query_idx in np.flatnonzero(exposure_mask):
        exposure[np.unique(neighbors[query_idx])] += 1

    return InfluenceResult(
        predictions=predictions,
        classes=labels,
        counts=counts,
        winner_indices=winner_indices,
        top_two_gap=gaps,
        band_flags=bands,
        exact_vulnerable=exact,
        neighborhood_exposure=exposure,
        decisive_influence=decisive_matrix.sum(axis=0),
        directional_influence=directional,
        decisive_matrix=decisive_matrix,
    )


def brute_force_directional_influence(
    y_train: np.ndarray,
    neighbors: np.ndarray,
    *,
    classes: Iterable[object] | None = None,
    tie_priority: Iterable[object] | None = None,
) -> np.ndarray:
    """Naive reference implementation used only for correctness and timing."""

    y_train = np.asarray(y_train)
    neighbors = np.asarray(neighbors, dtype=int)
    labels, _ = counts_from_neighbors(y_train, neighbors, classes=classes)
    base, _ = predict_from_counts(
        counts_from_neighbors(y_train, neighbors, classes=labels)[1],
        labels,
        tie_priority=tie_priority,
    )
    class_to_idx = {label: idx for idx, label in enumerate(labels)}
    out = np.zeros((len(y_train), len(labels)), dtype=int)
    for train_idx, old_class in enumerate(y_train):
        for new_class in labels:
            if new_class == old_class:
                continue
            changed = y_train.copy()
            changed[train_idx] = new_class
            changed_pred, _, _ = predict_from_neighbors(
                changed, neighbors, classes=labels, tie_priority=tie_priority
            )
            out[train_idx, class_to_idx[new_class]] = int(
                np.sum(changed_pred != base)
            )
    return out
