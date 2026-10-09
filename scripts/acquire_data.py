"""Acquire and process the benchmark panel without redistributing source arrays.

The script writes compressed NumPy files and a manifest to ``--output-dir``.
The public resource ships this processing path, not the resulting third-party
arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_blobs,
    make_circles,
    make_classification,
    make_moons,
)


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--skip-openml", action="store_true")
    parser.add_argument(
        "--verify-dir",
        type=Path,
        help="Optional frozen processed-data directory to compare after acquisition",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    rows: list[dict[str, object]] = []

    def persist(
        name: str,
        group: str,
        source: str,
        X: object,
        y: object,
        *,
        source_id: str,
    ) -> None:
        x_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        filename = f"{safe_name(name)}.npz"
        np.savez_compressed(args.output_dir / filename, X=x_arr, y=y_arr)
        rows.append({"dataset": name, "group": group, "source": source,
                     "source_id": source_id,
                     "rows": len(y_arr), "features": x_arr.shape[1],
                     "classes": len(np.unique(y_arr)), "file": filename})

    iris = load_iris()
    persist("Iris", "tabular", "sklearn", iris.data, iris.target, source_id="sklearn.datasets.load_iris")
    wine = load_wine()
    persist("Wine", "tabular", "sklearn", wine.data, wine.target, source_id="sklearn.datasets.load_wine")
    cancer = load_breast_cancer()
    persist("Breast Cancer Wisconsin", "medical/tabular", "sklearn", cancer.data, cancer.target, source_id="sklearn.datasets.load_breast_cancer")
    digits = load_digits()
    persist("Digits-10", "image", "sklearn", digits.data, digits.target, source_id="sklearn.datasets.load_digits")
    mask = np.isin(digits.target, [0, 8])
    persist("Digits-0-vs-8", "image/binary", "sklearn", digits.data[mask], digits.target[mask], source_id="sklearn.datasets.load_digits[target in {0,8}]")

    X, y = make_moons(n_samples=900, noise=0.18, random_state=rng)
    persist("Two Moons", "synthetic nonlinear", "sklearn synthetic", X, y, source_id="sklearn.datasets.make_moons;seed=20260614")
    X, y = make_circles(n_samples=900, noise=0.12, factor=0.45, random_state=rng)
    persist("Concentric Circles", "synthetic nonlinear", "sklearn synthetic", X, y, source_id="sklearn.datasets.make_circles;seed=20260614")
    X, y = make_blobs(n_samples=1000, centers=4, n_features=8,
                      cluster_std=[1.0, 1.5, 2.0, 1.2], random_state=rng)
    persist("Four Blobs", "synthetic multiclass", "sklearn synthetic", X, y, source_id="sklearn.datasets.make_blobs;seed=20260614")
    synthetic = [
        ("Noisy Binary", dict(n_samples=1000, n_features=20, n_informative=8, n_redundant=4, n_classes=2, flip_y=0.08, class_sep=1.0)),
        ("Imbalanced Binary", dict(n_samples=1000, n_features=18, n_informative=7, n_redundant=3, n_classes=2, weights=[0.82, 0.18], flip_y=0.04, class_sep=0.9)),
        ("Noisy Multiclass", dict(n_samples=1100, n_features=24, n_informative=10, n_redundant=4, n_classes=4, flip_y=0.08, class_sep=1.0)),
        ("Low-Separation Multiclass", dict(n_samples=1100, n_features=16, n_informative=8, n_redundant=2, n_classes=3, flip_y=0.04, class_sep=0.55)),
        ("High-Dimensional Sparse", dict(n_samples=1000, n_features=80, n_informative=10, n_redundant=5, n_classes=3, flip_y=0.05, class_sep=0.9)),
        ("Redundant Multiclass", dict(n_samples=1000, n_features=36, n_informative=8, n_redundant=18, n_classes=3, flip_y=0.03, class_sep=1.0)),
    ]
    for name, spec in synthetic:
        X, y = make_classification(random_state=rng, **spec)
        persist(name, "synthetic tabular", "sklearn synthetic", X, y, source_id=f"sklearn.datasets.make_classification;seed=20260614;spec={name}")

    if not args.skip_openml:
        import pandas as pd
        openml_ids = {
            "dermatology": 32,
            "diabetes": 37,
            "haberman": 43,
            "heart-statlog": 53,
            "ionosphere": 59,
            "parkinsons": 148,
            "segment": 36,
            "sonar": 40,
            "vehicle": 54,
        }
        for name, data_id in openml_ids.items():
            data = fetch_openml(data_id=data_id, as_frame=True)
            frame = data.frame.dropna()
            target_name = data.target_names[0] if isinstance(data.target_names, list) else data.target_names
            y = frame[target_name].astype("category").cat.codes.to_numpy()
            X = pd.get_dummies(frame.drop(columns=[target_name]), dummy_na=False)
            persist(name, "OpenML/UCI", "OpenML", X.to_numpy(dtype=float), y, source_id=f"openml:data_id={data_id}")

    rows.sort(key=lambda row: str(row["dataset"]))
    with (args.output_dir / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "script": "scripts/acquire_data.py",
        "seed": args.seed,
        "numpy": np.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "python": sys.version,
        "openml_source_ids": {
            "dermatology": 32,
            "diabetes": 37,
            "haberman": 43,
            "heart-statlog": 53,
            "ionosphere": 59,
            "parkinsons": 148,
            "segment": 36,
            "sonar": 40,
            "vehicle": 54,
        },
    }
    (args.output_dir / "acquisition_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if args.verify_dir:
        checks = []
        for row in rows:
            generated_path = args.output_dir / row["file"]
            frozen_path = args.verify_dir / row["file"]
            if not frozen_path.exists():
                checks.append({"dataset": row["dataset"], "status": "missing_frozen_file"})
                continue
            generated = np.load(generated_path, allow_pickle=True)
            frozen = np.load(frozen_path, allow_pickle=True)
            feature_delta = float(np.max(np.abs(generated["X"] - frozen["X"])))
            labels_equal = bool(np.array_equal(generated["y"], frozen["y"]))
            checks.append({
                "dataset": row["dataset"],
                "status": "numeric_match" if feature_delta <= 1e-8 and labels_equal else "mismatch",
                "max_abs_feature_delta": feature_delta,
                "labels_equal": labels_equal,
                "exact_array_match": bool(np.array_equal(generated["X"], frozen["X"]) and labels_equal),
            })
        with (args.output_dir / "data_reproduction_check.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
            writer.writeheader()
            writer.writerows(checks)
    print(f"wrote {len(rows)} datasets to {args.output_dir}")


if __name__ == "__main__":
    main()
