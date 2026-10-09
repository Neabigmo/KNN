"""Finite two-dimensional Euclidean constructions for theory checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RingConstruction:
    points: np.ndarray
    labels: np.ndarray
    representative_indices: np.ndarray
    support_indices: np.ndarray
    pivotal_representative: int
    epsilon: float


@dataclass(frozen=True)
class BoundaryMotionConstruction:
    """A query-centered kNN boundary with one controllable neighbor swap."""

    points: np.ndarray
    query: np.ndarray
    target_index: int
    candidate_index: int
    target_radius: float
    candidate_radius: float
    motion_radius: float


def single_prototype_motion_safe(top_two_gap: int | np.ndarray) -> np.ndarray:
    """Return the conservative certificate for one moved-and-relabelled point.

    With all other prototype positions and labels fixed, moving one point can
    change the top-k set by at most one member exchange.  Relabeling that
    point changes at most one vote, so the winning margin can shrink by at
    most two.  A strict pre-perturbation gap greater than two therefore
    certifies prediction preservation for any new position and label under
    deterministic unweighted voting.
    """

    gaps = np.asarray(top_two_gap)
    return gaps > 2


def boundary_motion_construction(
    *,
    target_radius: float = 1.0,
    candidate_radius: float = 1.01,
    inner_radius: float = 0.3,
    motion_radius: float = 0.04,
) -> BoundaryMotionConstruction:
    """Build a five-neighbor boundary where outward motion swaps one member.

    Four points lie safely inside the query-centered boundary.  The target is
    the fifth neighbor and the candidate is just outside it.  Moving the
    target radially outward by ``motion_radius`` therefore exchanges the
    target and candidate while leaving the four inner neighbors unchanged.
    """

    values = np.asarray(
        [target_radius, candidate_radius, inner_radius, inner_radius, inner_radius, inner_radius],
        dtype=float,
    )
    if not (0 < inner_radius < target_radius < candidate_radius < target_radius + motion_radius):
        raise ValueError("radii must place the candidate just outside the target boundary")
    angles = np.asarray([0.0, 0.0, 0.0, np.pi / 2, np.pi, 3.0 * np.pi / 2])
    points = np.column_stack([values * np.cos(angles), values * np.sin(angles)])
    return BoundaryMotionConstruction(
        points=points,
        query=np.zeros((1, 2), dtype=float),
        target_index=0,
        candidate_index=1,
        target_radius=target_radius,
        candidate_radius=candidate_radius,
        motion_radius=motion_radius,
    )


def ring_support_construction(k: int, epsilon: float = 0.01) -> RingConstruction:
    """Build the finite ring/support construction for k >= 2.

    Representatives have a one-vote margin in a binary center query.  Each
    representative receives k radial same-label supports, making every LOO
    prediction locally correct when the cross-cluster separation condition is
    satisfied.  The construction is geometric; its verifier checks the
    finite inequalities numerically rather than treating them as assumptions.
    """

    if k < 2 or epsilon <= 0:
        raise ValueError("ring_support_construction requires k >= 2 and epsilon > 0")
    angles = 2.0 * np.pi * np.arange(k) / k
    representatives = np.column_stack([np.cos(angles), np.sin(angles)])
    labels = np.zeros(k, dtype=int)
    labels[: (k + 1) // 2] = 0
    labels[(k + 1) // 2 :] = 1
    if k % 2 == 0:
        labels[: k // 2] = 0
        labels[k // 2 :] = 1
    supports = np.concatenate(
        [
            (1.0 + epsilon * np.arange(1, k + 1)[:, None] / k) * point
            for point in representatives
        ],
        axis=0,
    )
    support_labels = np.repeat(labels, k)
    points = np.vstack([representatives, supports])
    all_labels = np.concatenate([labels, support_labels])
    pivotal = 0
    return RingConstruction(
        points=points,
        labels=all_labels,
        representative_indices=np.arange(k),
        support_indices=np.arange(k, len(points)),
        pivotal_representative=pivotal,
        epsilon=epsilon,
    )


def ring_separation_margin(k: int, epsilon: float) -> float:
    """A conservative lower bound on cross-cluster minus within-cluster distance."""

    if k < 2:
        raise ValueError("k must be at least 2")
    d_cross = 2.0 * np.sin(np.pi / k)
    return d_cross - 3.0 * epsilon


def verify_ring_separation(construction: RingConstruction) -> dict[str, float | bool]:
    """Check the explicit finite distance inequalities of the construction."""

    points = construction.points
    k = len(construction.representative_indices)
    labels = construction.labels
    distance_matrix = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    cluster_ids = np.concatenate([np.arange(k), np.repeat(np.arange(k), k)])
    within = distance_matrix[cluster_ids[:, None] == cluster_ids[None, :]]
    between = distance_matrix[cluster_ids[:, None] != cluster_ids[None, :]]
    within_positive = within[within > 0]
    min_between = float(between.min()) if len(between) else float("inf")
    max_within = float(within_positive.max()) if len(within_positive) else 0.0
    center_distances = np.linalg.norm(points, axis=1)
    rep_cutoff = np.sort(center_distances)[:k].max()
    support_floor = np.sort(center_distances)[k]
    return {
        "k": float(k),
        "max_within_same_label_distance": max_within,
        "min_between_label_distance": min_between,
        "center_representative_cutoff": float(rep_cutoff),
        "center_support_floor": float(support_floor),
        "cross_cluster_separation_holds": bool(min_between > max_within),
        "center_representatives_are_first_k": bool(rep_cutoff < support_floor),
    }
