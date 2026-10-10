# Auditing kNN Predictions under Prototype Relabeling

This repository provides the reproducibility code for directional training-label edits in k-nearest-neighbor classification. It records which query predictions change after an admissible edit and calculates how shared training neighbors couple those changes across a query batch under specified independent binary label-flip probabilities.

## Repository contents

- `src/knn_reliability/`: deterministic kNN, influence, geometry, probability, and audit utilities.
- `scripts/acquire_data.py`: dataset acquisition and synthetic-data generation with a reproducible manifest.
- `experiments/run_revision_experiments.py`: E0--E9 experiment runner used by the revision protocol.
- `figures/scripts/build_revision_figures.py`: figure-generation script driven by experiment outputs; the released figures use controlled influence/variance panels, class-by-$k$ frequency summaries, within-dataset medians with exact Q25--Q75 intervals, geometry-operation heatmaps, and common-direction runtime ratios. Main-panel provenance tables are written alongside the figures, with supplementary validation rows kept separate.
- `tests/`: 30 regression, exactness, and input-contract tests for the kNN, influence, geometry, and probability implementations.
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

Release `paa-revision-2026-10-10-v13` contains the code used for the current revision. The E6 implementation separates neighborhood membership exchange from within-neighborhood order changes, and E9 reports explicit pair scans, overlap-edge joint evaluations, and nonedge contributions aggregated without pairwise iteration. The experiment runner and figure builder use portable relative paths and acquire benchmark data through the documented acquisition script. The associated submission archive supplies frozen result tables, figure panels, and manuscript source.

The test suite includes a complete small-state check for the first-order
batch-variance expansion and randomized equivalence checks for the factorized
and dense joint-probability implementations.
GitHub Actions
runs it on Python 3.10 and 3.12.  `environment.yml` records the environment
used for the release verification.
