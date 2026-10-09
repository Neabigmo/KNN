# Auditing Prototype Replacement Vulnerability in kNN Classification

This repository provides the reproducible implementation for **Auditing Prototype Replacement Vulnerability in kNN Classification: Exact Influence, Probabilistic Stability, and Label Review**.

The central analysis distinguishes deleted-point leave-one-out recovery error from retained-prototype relabeling vulnerability.  It implements exact directional influence, point-level vulnerability, vote-margin diagnostics, probabilistic stability under label flips, query-neighborhood overlap-graph variance, and controlled small-motion checks.

## Repository contents

- `src/knn_reliability/`: deterministic kNN, influence, geometry, probability, and audit utilities.
- `scripts/acquire_data.py`: dataset acquisition and synthetic-data generation with a reproducible manifest.
- `experiments/run_revision_experiments.py`: E0--E9 experiment runner used by the revision protocol.
- `figures/scripts/build_revision_figures.py`: figure-generation script driven by experiment outputs.
- `tests/`: regression tests for the kNN, influence, geometry, and probability implementations.
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

The test suite contains 25 regression and exactness checks, including a
complete small-state check for the first-order batch-variance expansion.
GitHub Actions
runs it on Python 3.10 and 3.12.  `environment.yml` records the environment
used for the release verification.
