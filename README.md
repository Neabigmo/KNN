# Leave-One-Out Can Miss Prototype Relabeling Vulnerability in kNN Classification

This repository contains the manuscript source, reproducibility code, and submission materials
for a PAA-targeted study of local reliability in deterministic k-nearest-neighbor classification.

Core claim:

> Deleted-point leave-one-out error is an error-estimation signal, not a certificate of local prototype relabeling reliability; local vote margin provides a practical audit for where relabeling one retained prototype can change nearby decisions.

## Repository layout

- `manuscript_source_flat/`
  - active Springer `sn-jnl` manuscript source
  - current manuscript PDF at `manuscript_source_flat/main.pdf`
- `reproducibility_project/`
  - benchmark runner, experiment scripts, figure builders, and configuration
- `supplementary/`
  - Online Resource preparation files for submission
- `editorial/`
  - cover letter drafts, submission notes, and preflight checks
- `submission_package/`
  - locally assembled upload bundle for journal submission

## Quick start

Recommended environment:

```powershell
conda activate E:\anaconda3\envs\pytorch-clean
```

Optional proxy for downloads:

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"
```

Environment check:

```powershell
python reproducibility_project/scripts/paa_runner.py env-check --config reproducibility_project/config/paa_rebuild_config.yaml
```

Acquire datasets and cache processed copies:

```powershell
python reproducibility_project/scripts/paa_runner.py acquire-data --config reproducibility_project/config/paa_rebuild_config.yaml
```

Run the benchmark:

```powershell
python reproducibility_project/scripts/paa_runner.py run-benchmark --config reproducibility_project/config/paa_rebuild_config.yaml
```

## Data policy

This repository does not publish a bulk copy of all benchmark datasets.
Most datasets are fetched automatically from their original upstream sources or regenerated locally.
See [DATA_ACCESS.md](DATA_ACCESS.md) for acquisition details and fallback instructions.

## Manuscript focus

The current PAA version is organized around:

1. `LOO` as deleted-point recovery error.
2. `PRV` as fixed-location prototype relabeling vulnerability.
3. `EnumVulnRate`, `BandRate`, and `Exposure` as the main audit outputs.
4. margin-aware `k` selection and conservative cleaning as secondary applications.

## Notes

- Cached data, logs, and generated tables/figures are intentionally kept out of version control.
- The public code path is meant to document how results are rebuilt without redistributing third-party datasets in bulk.
- No DOI-based public data deposit is used here because some upstream dataset redistribution rights may be unclear.
