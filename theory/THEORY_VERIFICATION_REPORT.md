# Theory Verification Report

Date: 2026-10-09

## Current checks

- `pytest` verifies the exact multiclass directional implementation against a
  slow label-copying reference.
- The binary odd-k identity is checked on a duplicated vulnerable query batch.
- The Poisson-binomial recurrence is checked against exhaustive flip-vector
  enumeration for `k=3`.
- Shared batch-risk moments are checked against Monte Carlo draws in which one
  training-label vector is reused across all queries.
- The corrected binary odd-$k$ lower bound is checked across the E2 panel; the
  corrected bound has zero violations and the withdrawn $R_{\mathrm{point}}/k$
  bound is recorded as failing in the audit file.
- E3 checks 84 dataset-$k$-probability rows across two query scales and
  verifies the independent-model Efron--Stein variance bound row by row;
  the perfectly correlated stress rows are reported as model mismatch rather
  than treated as independent-model validation.
- The ring construction is checked for explicit cross-cluster and
  center-neighborhood distance inequalities at `k=3`.
- The single-prototype motion certificate is checked in 40,000 random
  move-and-relabel trials and on small perturbations of three benchmark
  datasets; no eligible case violates the strict-gap condition.
- The heterogeneous concentration statement is checked in 18 spatial
  conditional settings with a known Lipschitz regression function, including
  an explicit `a=0` branch in the proof.

## Boundary audit

The implementation and proof files explicitly cover or reject: `k=1`, even
`k`, deterministic ties, zero-count competitors, restricted replacement
classes, probabilities 0 and 1, one-class input rejection, repeated feature
distances through stable index order, and the distinction between fixed-position
label replacement and geometric displacement.

## Open empirical questions

The revision does not promote monotone-in-k PRV, a universal finite-p Taylor
error bound, or an unconditional asymptotic rate to theorem status.  These are
reported only through the designated experiments, including negative results
when observed.
