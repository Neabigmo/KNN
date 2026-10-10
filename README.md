# Auditing Prototype Replacement Vulnerability in kNN Classification

This repository provides the reproducible implementation for **Auditing Prototype Replacement Vulnerability in kNN Classification: Exact Influence, Probabilistic Stability, and Label Review**.

The central analysis distinguishes deleted-point leave-one-out recovery error from retained-prototype relabeling vulnerability.  It implements exact directional influence, point-level vulnerability, vote-margin diagnostics, probabilistic stability under label flips, query-neighborhood overlap-graph variance, an exact shared/unique Poisson-binomial factorization, finite-noise variance validation, four-state geometry-change probability curves, randomized tie-order sensitivity, and fixed-query/fixed-overlap-rate runtime audits.

## Repository contents

- `src/knn_reliability/`: deterministic kNN, influence, geometry, probability, and audit utilities.
- `scripts/acquire_data.py`: dataset acquisition and synthetic-data generation with a reproducible manifest.
- `experiments/run_revision_experiments.py`: E0--E9 experiment runner used by the revision protocol.
- `figures/scripts/build_revision_figures.py`: figure-generation script driven by experiment outputs; the released figures use controlled influence/variance panels, class-by-$k$ frequency summaries, within-dataset medians with exact Q25--Q75 intervals, geometry-operation heatmaps, and common-direction runtime ratios.
- `tests/`: 26 regression and exactness tests for the kNN, influence, geometry, and probability implementations.
- `theory/`: theorem proofs, complexity notes, novelty matrix, and theory verification record.

## Data sources

The acquisition script obtains the benchmark panel from:

- scikit-learn built-in loaders: Iris, Wine, Breast Cancer Wisconsin, and Digits;
- scikit-learn generators: two moons, concentric circles, four blobs, and the configured classification panels;
- OpenML datasets: Dermatology (`data_id=32`), Diabetes (`37`), Haberman (`43`), Heart Statlog (`53`), Ionosphere (`59`), Parkinsons (`148`), Segment (`36`), Sonar (`40`), and Vehicle (`54`).

## Reproduce the analyses

From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/acquire_data.py --output-dir data/processed
python -m pytest
python experiments/run_revision_experiments.py
python figures/scripts/build_revision_figures.py
```

The experiment runner writes audit tables and run metadata under `results/`.  The figure builder reads the processed tables and writes panels under `figures/`.  To select a subset of experiments, pass names such as `--experiments e1 e2 e3 e6`.

E0 reports both the same-split implementation audit and the historical comparison status.  To reproduce the historical join, set `KNN_BASELINE_TABLE_DIR` to a directory containing `paa_multiclass_benchmark.csv` before running E0.  The runner records the resulting status in `results/processed/e0_reproduction_status.json` and writes an explicit missing-key audit.  E3 includes exact finite-noise checks at five flip probabilities, a controlled same-$k$ concentration pair, and the factorized overlap calculation; E5 includes randomized ordering within exact score ties; E6 includes the controlled enter/stay/exit audit and a 70-row four-state random-angle displacement curve with two same-label candidate controls on the same prototype.  E9 compares dense--2D, sparse--2D, dense--1D, and sparse--1D exact probability paths on identical inputs and includes a fixed-query, fixed-overlap-edge-fraction sweep over $k=3,5,7,11,15,31$.

The public release corresponds to the manuscript revision tag `paa-revision-2026-10-10-v11`.  The E6 release separates neighborhood membership exchange from within-neighborhood order changes and retains both rates in the audit output.  The experiment runner and figure builder use portable relative paths and acquire benchmark data through the documented acquisition script.

The test suite includes a complete small-state check for the first-order
batch-variance expansion and randomized equivalence checks for the factorized
and dense joint-probability implementations.
GitHub Actions
runs it on Python 3.10 and 3.12.  `environment.yml` records the environment
used for the release verification.
