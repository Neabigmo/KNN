# Computational Complexity

Let `n` be the number of training points, `q` the number of queries, `d` the
feature dimension, `k` the neighborhood size, and `C` the number of classes.

| Operation | Time | Main memory | Notes |
|---|---:|---:|---|
| Blocked squared distances | O(nqd) | O(nq) output plus one query block | No q-by-n-by-d tensor |
| Stable neighbor cache | O(nq log n) after distances | O(qk) | Stable index tie-break |
| Vote counts | O(qk) | O(qC) | Reused by all diagnostics |
| Exact directional influence | O(qkC^2) | O(qn) boolean matrix plus O(nC) | Up to C-1 replacement labels; each updated vote vector scans C classes for the winner |
| Naive reference | O(nC qk) plus repeated allocations | O(qk) | Used only for correctness/timing |
| One binary Poisson-binomial query | O(k^2) | O(k) | Heterogeneous probabilities |
| Dense pairwise + reference 2D batch moments | O(q^2 k^3) | O(k^2) sequential working state | Runs the retained two-dimensional joint DP for every query pair |
| Dense pairwise + factorized 1D batch moments | O(q^2 + |E_Q| k^2) | O(k) sequential working state | Scans every pair, uses products on nonedges, and factorizes shared edges |
| Overlap-graph + reference 2D batch moments | O(sum_i d_i^2 + |E_Q| k^3) | O(k^2) | Marginal products on nonedges; reference joint DP on overlap edges |
| Overlap-graph + factorized 1D batch moments | O(sum_i d_i^2 + |E_Q| k^2) | O(k) | Production path: marginal products on nonedges and factorized edges |
| Global Monte Carlo | O(R qk) after cache | O(q) | R shared training-label draws |

The retained two-query DP stores O(k^2) count states but updates that table for
at most 2k prototypes, hence O(k^3) time per pair. The factorized calculation
uses three one-dimensional mass functions and conditional tail sums, giving
O(k^2) time and O(k) additional memory per retained edge. E9 compares all four
exact combinations through 320 queries, alongside shared-label Monte Carlo, so
the overlap-graph and factorization gains are not conflated. It does not extrapolate either exact method to thousands of dense
overlap queries; larger settings should use a documented approximation or
exploit sparse query--prototype incidence.

The runtime experiment uses warm-up runs, repeated measurements, medians,
quartiles, and a separate traced-peak-memory pass. It never labels a cached
exact computation as an exhaustive enumeration.
