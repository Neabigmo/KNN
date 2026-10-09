"""Optional extensions used in robustness and ablation experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def weighted_predict_from_neighbors(
    y_train: np.ndarray,
    neighbors: np.ndarray,
    distances: np.ndarray,
    *,
    classes: Iterable[object] | None = None,
    tie_priority: Iterable[object] | None = None,
    power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict with inverse-distance votes, with exact zero-distance handling.

    ``power=0`` is the unweighted rule, including when a query coincides with
    a training point.  For positive powers, any zero-distance neighbors take
    priority and share equal weight.
    """

    y_train = np.asarray(y_train)
    neighbors = np.asarray(neighbors, dtype=int)
    distances = np.asarray(distances, dtype=float)
    if neighbors.shape != distances.shape or neighbors.ndim != 2:
        raise ValueError("neighbors and distances must have the same 2-D shape")
    if power < 0 or np.any(distances < 0):
        raise ValueError("power and distances must be non-negative")
    labels = np.asarray(list(np.unique(y_train) if classes is None else classes))
    if len(labels) == 0:
        raise ValueError("at least one class is required")
    priority = labels if tie_priority is None else np.asarray(list(tie_priority))
    if set(priority.tolist()) != set(labels.tolist()):
        raise ValueError("tie_priority must contain exactly the class set")
    rank = {label: i for i, label in enumerate(priority)}
    predictions = np.empty(len(neighbors), dtype=labels.dtype)
    scores = np.zeros((len(neighbors), len(labels)), dtype=float)
    for row in range(len(neighbors)):
        row_distances = distances[row]
        if power == 0:
            weights = np.ones_like(row_distances)
        elif np.any(row_distances == 0):
            weights = (row_distances == 0).astype(float)
        else:
            weights = 1.0 / np.power(row_distances, power)
        for col, label in enumerate(labels):
            scores[row, col] = weights[y_train[neighbors[row]] == label].sum()
        best = np.flatnonzero(scores[row] == scores[row].max())
        predictions[row] = min((labels[index] for index in best), key=rank.__getitem__)
    return predictions, scores


def random_tie_distribution(
    counts: np.ndarray,
    classes: Iterable[object],
) -> np.ndarray:
    """Return the uniform prediction distribution over tied winners."""

    counts = np.asarray(counts, dtype=int)
    labels = np.asarray(list(classes))
    if counts.ndim != 2 or counts.shape[1] != len(labels):
        raise ValueError("counts and classes have incompatible shapes")
    distribution = np.zeros_like(counts, dtype=float)
    for row, values in enumerate(counts):
        tied = np.flatnonzero(values == values.max())
        distribution[row, tied] = 1.0 / len(tied)
    return distribution


def class_stratified_change_metrics(
    y_true: np.ndarray,
    baseline_predictions: np.ndarray,
    changed_predictions: np.ndarray,
    classes: Iterable[object],
) -> list[dict[str, object]]:
    """Compute change rate and accuracy delta separately for each true class."""

    y_true = np.asarray(y_true)
    baseline_predictions = np.asarray(baseline_predictions)
    changed_predictions = np.asarray(changed_predictions)
    if not (len(y_true) == len(baseline_predictions) == len(changed_predictions)):
        raise ValueError("all prediction arrays must have equal length")
    rows: list[dict[str, object]] = []
    for label in classes:
        mask = y_true == label
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(
            {
                "class": label,
                "n": n,
                "change_rate": float(np.mean(baseline_predictions[mask] != changed_predictions[mask])),
                "baseline_accuracy": float(np.mean(baseline_predictions[mask] == y_true[mask])),
                "changed_accuracy": float(np.mean(changed_predictions[mask] == y_true[mask])),
                "accuracy_delta": float(
                    np.mean(changed_predictions[mask] == y_true[mask])
                    - np.mean(baseline_predictions[mask] == y_true[mask])
                ),
            }
        )
    return rows
