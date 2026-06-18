# Data Access

This repository does not redistribute the benchmark datasets as a public bundle.
Instead, the experiments are designed so that most datasets can be downloaded or regenerated
through the provided scripts, while cached local copies remain outside version control.

## Dataset groups

The benchmark uses three data sources:

1. `sklearn` built-ins
   - `iris`
   - `wine`
   - `breast_cancer`
   - `digits`
   - derived binary subset `digits_0_vs_8`
2. synthetic datasets generated locally with scikit-learn
   - `two_moons`
   - `concentric_circles`
   - `four_blobs`
   - `noisy_binary`
   - `imbalanced_binary`
   - `noisy_multiclass`
   - `low_separation_multiclass`
   - `high_dimensional_sparse`
   - `redundant_multiclass`
3. OpenML-hosted datasets fetched by name
   - `ionosphere`
   - `sonar`
   - `diabetes`
   - `heart-statlog`
   - `liver-disorders`
   - `haberman`
   - `dermatology`
   - `parkinsons`
   - `vehicle`
   - `segment`

## Recommended environment

```powershell
conda activate your_env_name
```

## How to acquire data

From the repository root:

```powershell
python reproducibility_project/scripts/paa_runner.py acquire-data --config reproducibility_project/config/paa_rebuild_config.yaml
```

This command will:

- load the built-in sklearn datasets;
- generate the synthetic datasets locally;
- fetch the configured OpenML datasets by name;
- cache processed copies under `reproducibility_project/data/processed/`;
- write a manifest to `reproducibility_project/data/processed/dataset_manifest.csv`.

## If an upstream download fails

The runner is written to continue when possible. If an upstream source becomes unavailable,
changes access policy, or requires manual intervention, place the recovered raw file under:

```text
reproducibility_project/data/raw/
```

and rerun:

```powershell
python reproducibility_project/scripts/paa_runner.py acquire-data --config reproducibility_project/config/paa_rebuild_config.yaml
```

## Why cached data are not committed here

This repository is intended to keep the public code and manuscript reproducible without
reposting third-party datasets in ways that could conflict with their original licenses
or hosting terms. The scripts document the acquisition pathway, and the benchmark manifest
records which processed files were used in the experiments.
