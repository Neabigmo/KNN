"""Deterministic nearest-neighbor primitives used by every revision experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class NeighborCache:
    """Distances and stable neighbor indices for a fixed query batch."""

    indices: np.ndarray
    distances: np.ndarray


def _validate_xy(x_train: np.ndarray, x_query: np.ndarray) -> None:
    if x_train.ndim != 2 or x_query.ndim != 2:
        raise ValueError("x_train and x_query must be two-dimensional")
    if x_train.shape[1] != x_query.shape[1]:
        raise ValueError("x_train and x_query must have the same feature count")
    if len(x_train) == 0:
        raise ValueError("x_train must not be empty")


def pairwise_squared_distances(
    x_train: np.ndarray,
    x_query: np.ndarray,
    *,
    query_block_size: int = 2048,
) -> np.ndarray:
    """Compute squared Euclidean distances without a giant 3-D temporary."""

    x_train = np.asarray(x_train, dtype=float)
    x_query = np.asarray(x_query, dtype=float)
    _validate_xy(x_train, x_query)
    if query_block_size < 1:
        raise ValueError("query_block_size must be positive")

    train_norm = np.einsum("ij,ij->i", x_train, x_train)
    out = np.empty((len(x_query), len(x_train)), dtype=float)
    for start in range(0, len(x_query), query_block_size):
        stop = min(start + query_block_size, len(x_query))
        query_block = x_query[start:stop]
        out[start:stop] = (
            np.einsum("ij,ij->i", query_block, query_block)[:, None]
            + train_norm[None, :]
            - 2.0 * query_block @ x_train.T
        )
    np.maximum(out, 0.0, out=out)
    return out


def stable_topk(distances: np.ndarray, k: int) -> np.ndarray:
    """Return the first k indices under distance, then original-index order."""

    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 2:
        raise ValueError("distances must be two-dimensional")
    if not 1 <= k <= distances.shape[1]:
        raise ValueError("k must satisfy 1 <= k <= number of training points")
    # Each row begins in index order. Stable sorting therefore implements the
    # stated index tie-break without a second materialized sort key.
    return np.argsort(distances, axis=1, kind="mergesort")[:, :k]


def build_neighbor_cache(
    x_train: np.ndarray,
    x_query: np.ndarray,
    k: int,
    *,
    query_block_size: int = 2048,
) -> NeighborCache:
    distances_sq = pairwise_squared_distances(
        x_train, x_query, query_block_size=query_block_size
    )
    indices = stable_topk(distances_sq, k)
    distances = np.sqrt(np.take_along_axis(distances_sq, indices, axis=1))
    return NeighborCache(indices=indices, distances=distances)


def class_order(
    y_train: np.ndarray,
    classes: Iterable[object] | None = None,
    tie_priority: Iterable[object] | None = None,
) -> tuple[np.ndarray, dict[object, int]]:
    """Return labels and a deterministic priority rank for ties."""

    y_train = np.asarray(y_train)
    labels = list(np.unique(y_train) if classes is None else classes)
    if not labels:
        raise ValueError("at least one class is required")
    if len(set(labels)) != len(labels):
        raise ValueError("classes must be unique")
    if tie_priority is None:
        ordered = labels
    else:
        priority = list(tie_priority)
        if set(priority) != set(labels):
            raise ValueError("tie_priority must contain exactly the class set")
        ordered = priority
    return np.asarray(labels), {label: rank for rank, label in enumerate(ordered)}


def counts_from_neighbors(
    y_train: np.ndarray,
    neighbors: np.ndarray,
    classes: Iterable[object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    y_train = np.asarray(y_train)
    neighbors = np.asarray(neighbors, dtype=int)
    labels, _ = class_order(y_train, classes=classes)
    counts = np.zeros((len(neighbors), len(labels)), dtype=int)
    for col, label in enumerate(labels):
        counts[:, col] = np.sum(y_train[neighbors] == label, axis=1)
    return labels, counts


def predict_from_counts(
    counts: np.ndarray,
    classes: Iterable[object],
    *,
    tie_priority: Iterable[object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=int)
    labels = np.asarray(list(classes))
    if counts.ndim != 2 or counts.shape[1] != len(labels):
        raise ValueError("counts and classes have incompatible shapes")
    _, rank = class_order(labels, classes=labels, tie_priority=tie_priority)
    predictions = np.empty(len(counts), dtype=labels.dtype)
    winner_indices = np.empty(len(counts), dtype=int)
    for row_idx, row in enumerate(counts):
        best = int(row.max())
        candidates = np.flatnonzero(row == best)
        winner_idx = min(candidates, key=lambda idx: rank[labels[idx]])
        winner_indices[row_idx] = winner_idx
        predictions[row_idx] = labels[winner_idx]
    return predictions, winner_indices


def predict_from_neighbors(
    y_train: np.ndarray,
    neighbors: np.ndarray,
    *,
    classes: Iterable[object] | None = None,
    tie_priority: Iterable[object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels, counts = counts_from_neighbors(y_train, neighbors, classes=classes)
    predictions, winners = predict_from_counts(
        counts, labels, tie_priority=tie_priority
    )
    return predictions, counts, winners


def loo_neighbor_cache(x_train: np.ndarray, k: int) -> NeighborCache:
    """Build a cache for LOO with the self-index removed from every row."""

    if len(x_train) < 2:
        raise ValueError("LOO requires at least two training points")
    distances_sq = pairwise_squared_distances(x_train, x_train)
    np.fill_diagonal(distances_sq, np.inf)
    indices = stable_topk(distances_sq, min(k, len(x_train) - 1))
    distances = np.sqrt(np.take_along_axis(distances_sq, indices, axis=1))
    return NeighborCache(indices=indices, distances=distances)
