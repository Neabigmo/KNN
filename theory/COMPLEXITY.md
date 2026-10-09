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
| Pairwise binary batch moments | O(q^2 k^3) in the current DP | O(k^2) working state | Retains shared-flip covariance |
| Global Monte Carlo | O(R qk) after cache | O(q) | R shared training-label draws |

The E3 scale check therefore reports exact moments at 40 queries for the full
scenario panel and at 80 queries for a reduced independent-flip panel.  It does
not extrapolate exact pairwise computation to thousands of queries; larger
settings should use a documented approximation or exploit sparse query--prototype
incidence rather than silently treating the quadratic routine as scalable.

The runtime experiment times each stage independently.  It never labels a
cached exact computation as an exhaustive enumeration.
