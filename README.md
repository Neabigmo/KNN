# Online Resource 1: Reproducible code and processing package

This supplementary archive accompanies the edited manuscript and contains the frozen numerical records, reproducibility code, acquisition instructions, proofs, tests, and regenerated publication-quality figure exports. The figure presentation uses the same experimental measurements as the frozen computational baseline.

This standalone package accompanies the revised manuscript **Leave-One-Out Recovery and Batch Sensitivity to Prototype Relabeling in kNN** by Yiheng Yang and Zhenzhou Feng.

The public source repository is [github.com/Neabigmo/KNN](https://github.com/Neabigmo/KNN).

## Contents

- `src/knn_reliability/`: deterministic kNN, LOO, exact directional influence, probability, geometry, and policy utilities, including the exact query-neighborhood overlap-graph batch-moment engine and shared/unique Poisson-binomial factorization.
- `experiments/run_revision_experiments.py`: E0--E9 experiment runner.
- `results/processed/`: regenerated result tables and audit summaries used by the manuscript.
- `figures/scripts/`, `figures/data/`, and `figures/export/`: figure builder, panel data, and exported figures. Main-panel CSVs contain only plotted rows; supplementary validation rows are kept in explicitly named supplement CSVs.
- `release/generate_frozen_results.py`: deterministic generator for the manuscript result macros from `results/processed/`.
- `release/check_submission_consistency.py`: cross-checks frozen E5 macros, manuscript text, response text, Fig. 2 analytic provenance, Fig. 5 statistics, controlled Fig. 4 source structures, plotted panel methods, Fig. 6 provenance, Fig. 7 configuration labels and ratio direction, and E6 audit fields.
- `release/check_figure_pdf.py`: checks final-size plot text extracted from the compiled manuscript PDF.
- `release/check_figure_collisions.py`: reports overlapping figure words in the compiled manuscript PDF and fails when both the intersection area and the relative overlap of the smaller text box exceed the documented limits.
- `release/make_figure_overview.py`: builds a contact sheet of the seven exported figures for visual review.
- `data_registry/`: the 23-dataset provenance registry and hashes used by the acquisition workflow.
- `scripts/`: acquisition and processing instructions for the sklearn, synthetic, and OpenML sources.
- `legacy_tables/`: archived comparison tables used only by E0.
- `theory/`: proofs, including the all-`k >= 1` finite deletion--relabeling separation with an observed replacement label, claim registry, novelty matrix, and literature registry.
- `tests/`: 30 regression and cross-metric consistency tests.
- `Online_Resource_1_overview.pdf`: the citable overview of this resource.

The probability implementation constructs the query-neighborhood overlap graph,
uses marginal products for disjoint query pairs, and evaluates joint dynamic
programs only on overlap edges.  E3 additionally validates exact finite-noise
variance against the first-order approximation at five flip probabilities on
multiple benchmark neighborhood structures, plus a controlled same-$k$,
same-query-count concentration pair.  E6 includes a controlled low/mid/high
vote-gap enter/stay/exit diagnostic and a 70-row four-state random-angle curve
over displacement radius and boundary gap, with 35 identical geometric settings
under each same-label candidate condition ($0,0$ and $1,1$); these are exact
or conditional audit procedures, not universal robustness claims.  E5 uses
reproducible random ordering within exact score ties and includes a separate
tie-sensitivity table.  E9 includes a fixed-query, fixed-overlap-edge-fraction
sweep over $k=3,5,7,11,15,31$.

- `analyses/`: a pre-specified reinterpretation of existing E3 probability records (68 independent-flip configurations), the query-independence reference, and two exact finite-state witnesses; the source datasets have not been expanded.

## Portable execution

From this directory, use Python >= 3.10 with NumPy, scikit-learn, Matplotlib, and pytest.  The included `pyproject.toml` declares the core dependency; `requirements.txt` lists the complete verification environment.

PowerShell:

```powershell
$data_dir = Join-Path (Get-Location) 'data_generated'
$env:KNN_DATA_DIR = $data_dir
$env:KNN_BASELINE_TABLE_DIR = (Join-Path (Get-Location) 'legacy_tables')
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python scripts/acquire_data.py --output-dir $data_dir
python -m pytest
python experiments/run_revision_experiments.py --experiments e1 e2 e3 e4 e5 e6 e7 e8 e9 e0
python release/generate_frozen_results.py
python figures/scripts/build_revision_figures.py
powershell -ExecutionPolicy Bypass -File .\BUILD_AND_VERIFY.ps1
```

`BUILD_AND_VERIFY.ps1` is a release-integrity check, not a substitute for the
preceding raw-data experiment command: it runs the 30-test suite, validates
recorded metadata and row counts, regenerates numerical macros and figures from
the released processed records, checks manuscript/response consistency, and
compiles the PDFs.  It does not independently rerun E0--E9 from raw data.

The acquisition command writes the processed panel to `data_generated/` and
records the Python, NumPy, scikit-learn, generator seed, and frozen OpenML data
IDs in `acquisition_metadata.json`.  The benchmark runner reads that directory
through `KNN_DATA_DIR`, while `data_registry/` provides the registry and hash
records.  E0 compares the revision tables with the archived CSV tables in
`legacy_tables`; the other experiments read the generated panel through
`KNN_DATA_DIR`.  The exact seed and split protocol is recorded in
`results/raw/run_metadata.json`.

For a run without the archived baseline directory, E0 still produces the
same-split implementation audit and writes `e0_reproduction_status.json` with
an explicit historical-comparison status.  To reproduce that historical join,
set `KNN_BASELINE_TABLE_DIR` to a directory containing
`paa_multiclass_benchmark.csv` before running E0.

To compare a regenerated panel against a private frozen audit cache, use:

```powershell
python scripts/acquire_data.py --output-dir data_generated --verify-dir <frozen_processed_directory>
```

The checker reports both exact array equality and numerical equality at
absolute tolerance `1e-8`; final digits can differ across library builds even
when the data are scientifically identical.

For the additional E3 dependence sensitivity calculation, regenerate the built-in/synthetic datasets with `--skip-openml` and then run `python analyses/dependence_gap_audit.py`; run `python analyses/theory_counterexamples.py` for exhaustive finite-state examples. Both commands write to `results/derived/` without changing frozen E0--E9 tables.

The submitted Online Resource does not contain the processed `.npz` arrays; they are reconstructed locally from the documented sources. Their recorded checksums are reference values, not evidence that a newly acquired cache is byte-identical.

The public implementation is maintained at
`https://github.com/Neabigmo/KNN`, release tag
`paa-revision-2026-10-10-v17-mechanism-decomposition`. The directory manifest and release archive record the exact
resource contents; the journal-hosted supplementary-file location, when
assigned by the publisher, is the authoritative access location for this
resource.
