"""Auditable kNN prototype-reliability analysis."""

from .extensions import (
    class_stratified_change_metrics,
    random_tie_distribution,
    weighted_predict_from_neighbors,
)
from .influence import (
    InfluenceResult,
    SignedErrorChangeResult,
    compute_classwise_influence,
    compute_global_single_relabel_risk,
    compute_influence,
    compute_neighborhood_exposure,
    compute_signed_error_change,
)
from .geometry import (
    ring_support_construction,
    single_prototype_motion_safe,
    verify_ring_separation,
)
from .knn import NeighborCache, build_neighbor_cache, predict_from_neighbors
from .probability import (
    batch_risk_moments,
    binary_flip_probability,
    dense_batch_risk_moments,
    enumerate_shared_flip_moments,
    factorized_joint_binary_flip_probability,
    first_order_risk,
    first_order_variance_bounds,
    monte_carlo_perfectly_correlated_batch_risk,
    monte_carlo_batch_risk,
    odd_k_boundary_probability,
    poisson_binomial_pmf,
    query_overlap_pairs,
)

__all__ = [
    "InfluenceResult",
    "SignedErrorChangeResult",
    "NeighborCache",
    "binary_flip_probability",
    "batch_risk_moments",
    "dense_batch_risk_moments",
    "enumerate_shared_flip_moments",
    "factorized_joint_binary_flip_probability",
    "build_neighbor_cache",
    "class_stratified_change_metrics",
    "compute_classwise_influence",
    "compute_global_single_relabel_risk",
    "compute_influence",
    "compute_neighborhood_exposure",
    "compute_signed_error_change",
    "first_order_risk",
    "first_order_variance_bounds",
    "monte_carlo_batch_risk",
    "monte_carlo_perfectly_correlated_batch_risk",
    "odd_k_boundary_probability",
    "poisson_binomial_pmf",
    "query_overlap_pairs",
    "predict_from_neighbors",
    "random_tie_distribution",
    "ring_support_construction",
    "single_prototype_motion_safe",
    "verify_ring_separation",
    "weighted_predict_from_neighbors",
]
