# Batch-dependence reanalysis (proposed revision)

This addendum is a targeted reuse of the **existing E3 inputs**, not a new
benchmark or dataset expansion. In the 84 published E3 rows, the exact batch
variance is computed under independent flips. Sixteen rows are paired with a
**correlated Monte Carlo stress test**; they are excluded from the summary
of the independence-model diagnosis. The analysis therefore reports 68 rows.

## Reproduce

Run these commands in the Online Resource source root. The acquisition step
reconstructs the 14 built-in and synthetic datasets without OpenML access:

```bash
python -m pip install -r requirements.txt
python scripts/acquire_data.py --skip-openml --output-dir data/processed
python analyses/dependence_gap_audit.py
python analyses/theory_counterexamples.py
```

Outputs appear in `results/derived/`, leaving original E3 tables untouched.
The numerical check compares every independently recomputed E3 batch variance
to the release's CSV and aborts if its discrepancy exceeds 1e-11.

## Fixed inputs and definitions

- Four E3 binary datasets: Breast Cancer Wisconsin, Digits-0-vs-8,
  Imbalanced Binary (synthetic), Noisy Binary (synthetic)
- Same `seed=0`, original E3 40/80 query caches, k, and flip probability vectors
- `V_ind = sum_t pi_t*(1-pi_t) / q**2` uses **exact** per-query marginals
- Dependence correction `delta = variance_exact - V_ind`
- Relative absolute error `abs(delta)/variance_exact`
- No statistical independence is attributed to distinct configurations drawn
  from the same dataset or overlapping query batches

Frozen E3 rows and newly derived CSVs are included in the resource; the
processed `.npz` dataset arrays are **not redistributed**. Fresh `.npz` byte
hashes depend on the acquisition environment and must not be assumed to
match the frozen reference; here numerical E3 results are directly checked.

The finite counterexamples enumerate every one of 16 or 32 bit-flip states.
Neither analysis establishes a deployment benefit, an error reduction, or a
new learning algorithm.
