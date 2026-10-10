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
| Dense pairwise + reference 2D batch moments | O(q^2 + |E_Q| k^3) | O(k^2) edge-local, O(q^2) if pairs are materialized | Scans every pair, uses products on nonedges, and runs 2D DP only on overlap edges |
| Dense pairwise + factorized 1D batch moments | O(q^2 + |E_Q| k^2) | O(k) edge-local, O(q^2) if pairs are materialized | Uses the identical nonedge rule and changes only the overlap-edge kernel |
| Overlap-graph + reference 2D batch moments | O(sum_i d_i^2 + |E_Q| k^3) | O(k^2) edge-local plus O(qk+|E_Q|) cached state | Marginal products on nonedges; reference joint DP on overlap edges |
| Overlap-graph + factorized 1D batch moments | O(sum_i d_i^2 + |E_Q| k^2) | O(k) edge-local plus O(qk+|E_Q|) cached state | Production path: marginal products on nonedges and factorized edges |
| Global Monte Carlo | O(R qk) after cache | O(q) | R shared training-label draws |

The public wrapper validates and materializes global arrays for that call. The
batch entry point validates them once before its edge loop. The retained two-query DP
stores O(k^2) count states and updates that table for at most 2k prototypes,
hence O(k^3) time per overlap edge. The factorized local kernel uses three
one-dimensional mass functions and conditional tail sums, giving O(k^2) time
and O(k) edge-local memory without copying the length-n probability vector.

For cached neighborhoods, the complete sparse factorized path conservatively
costs O(n + qk log k + qk^2 + sum_i d_i^2 + |E_Q| log |E_Q| + |E_Q|k^2).
This includes global validation, query marginals, overlap candidates,
deduplication, and edge kernels. It excludes raw-feature distance computation
and neighbor selection, which belong to the end-to-end pipeline.

E9 compares all four exact combinations through 320 queries. Dense 1D and
dense 2D perform the same independence check; sparse 1D and sparse 2D visit the
same overlap edges. The released table records pair_scan_count, joint_dp_count,
explicit_independent_product_count, aggregated_nonedge_pair_count, and
total_nonedge_pair_count. Thus scanning strategy and edge kernel are no
longer conflated. The corrected timings do not show a 1D runtime advantage in
the tested Python workloads.

The runtime experiment uses warm-up runs, repeated measurements, medians,
quartiles, and a separate traced-peak-memory pass. It never labels a cached
exact computation as an exhaustive enumeration.
