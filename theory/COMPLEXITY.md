# Computational Complexity

Let `n` be the number of training points, `q` the number of queries, `d` the
feature dimension, `k` the neighborhood size, and `C` the number of classes.

| Operation | Time | Main memory | Notes |
|---|---:|---:|---|
| Blocked squared distances | O(nqd) | O(nq) output plus one query block | No q-by-n-by-d tensor |
| Stable neighbor cache | O(nq log n) after distances | O(qk) | Stable index tie-break |
| Vote counts | O(qk) | O(qC) | Reused by all diagnostics |
| Exact directional influence | O(qkC) | O(qn) boolean matrix plus O(nC) | One count update per allowed direction |
| Naive reference | O(nC qk) plus repeated allocations | O(qk) | Used only for correctness/timing |
| One binary Poisson-binomial query | O(k^2) | O(k) | Heterogeneous probabilities |
| Dense pairwise binary batch moments | O(q^2 k^3) | O(k^2) sequential working state | Runs the joint DP for every query pair |
| Overlap-graph binary batch moments | O(sum_i d_i^2 + |E_Q| k^3) | O(k^2 + |E_Q|) | Marginal products on nonedges; exact joint DP on overlap edges |
| Global Monte Carlo | O(R qk) after cache | O(q) | R shared training-label draws |

The two-query DP stores O(k^2) count states but updates that table for at most
2k prototypes, hence O(k^3) time per pair. E9 directly compares the dense and
overlap-graph exact implementations through 320 queries, alongside shared-label
Monte Carlo. It does not extrapolate either exact method to thousands of dense
overlap queries; larger settings should use a documented approximation or
exploit sparse query--prototype incidence.

The runtime experiment uses warm-up runs, repeated measurements, medians,
quartiles, and a separate traced-peak-memory pass. It never labels a cached
exact computation as an exhaustive enumeration.
