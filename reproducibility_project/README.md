# PAA Rebuild Project

This project is the full rebuild workspace for turning the previous stability-semantics manuscript into a PAA-oriented method paper:

**Auditing Prototype Replacement Vulnerability in kNN Pattern Classification with Local Vote Margins**

The rebuild is intentionally separated from `paa_submission_package/` so that long-running data acquisition, benchmark execution, literature review, logs, and failure reports do not pollute the final upload package.

## Directory Layout

- `config/`: YAML configuration for environment, proxy, datasets, baselines, and manuscript targets.
- `scripts/`: executable Python utilities with progress bars, lockfiles, structured logging, and error reporting.
- `data/raw/`: downloaded or manually provided datasets.
- `data/processed/`: standardized cached datasets used by experiments.
- `results/tables/`: CSV and LaTeX result tables.
- `results/figures/`: generated figures for the manuscript.
- `logs/`: timestamped execution logs, JSONL events, and failure reports.
- `reports/`: human-readable run summaries, literature notes, and completion audits.
- `manuscript/`: PAA-rebuilt LaTeX source assembled from the results.
- `package/`: final PAA upload package and zip archive.

## Environment

Preferred environment:

```powershell
conda activate E:\anaconda3\envs\pytorch-clean
```

The runner also accepts an explicit Python path through `config/paa_rebuild_config.yaml`.

Recommended proxy for downloads:

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"
```

## Main Commands

Environment check:

```powershell
python scripts/paa_runner.py env-check --config config/paa_rebuild_config.yaml
```

Dataset acquisition:

```powershell
python scripts/paa_runner.py acquire-data --config config/paa_rebuild_config.yaml
```

Full benchmark:

```powershell
python scripts/paa_runner.py run-benchmark --config config/paa_rebuild_config.yaml
```

Report generation:

```powershell
python scripts/paa_runner.py build-reports --config config/paa_rebuild_config.yaml
```

The runner writes both console progress bars and structured logs under `logs/`.
