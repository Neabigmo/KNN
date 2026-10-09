"""Rebuild the revision experiments from the frozen processed-data cache.

All outputs are revision-only artifacts.  The legacy result tables are read
for E0 provenance but are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import math
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knn_reliability.extensions import (  # noqa: E402
    class_stratified_change_metrics,
    random_tie_distribution,
    weighted_predict_from_neighbors,
)
from knn_reliability.geometry import (  # noqa: E402
    boundary_motion_construction,
    ring_separation_margin,
    ring_support_construction,
    single_prototype_motion_safe,
    verify_ring_separation,
)
from knn_reliability.influence import (  # noqa: E402
    brute_force_directional_influence,
    compute_classwise_influence,
    compute_global_single_relabel_risk,
    compute_influence,
    compute_signed_error_change,
    top_two_gap,
)
from knn_reliability.knn import (  # noqa: E402
    build_neighbor_cache,
    loo_neighbor_cache,
    predict_from_counts,
    predict_from_neighbors,
)
from knn_reliability.probability import (  # noqa: E402
    batch_risk_moments,
    binary_flip_probability,
    dense_batch_risk_moments,
    enumerate_shared_flip_moments,
    first_order_risk,
    first_order_variance_bounds,
    monte_carlo_batch_risk,
    monte_carlo_perfectly_correlated_batch_risk,
    odd_k_boundary_probability,
)


DATA_DIR = Path(os.environ.get("KNN_DATA_DIR", str(ROOT / "data" / "processed")))
BASELINE_TABLE_DIR = Path(os.environ.get("KNN_BASELINE_TABLE_DIR", str(ROOT / "data" / "baseline_tables")))
RESULT_DIR = ROOT / "results"
RAW_DIR = RESULT_DIR / "raw"
PROCESSED_DIR = RESULT_DIR / "processed"
REGISTRY_DIR = ROOT / "data_registry"
ORIGINAL_23 = [
    "breast_cancer_wisconsin",
    "concentric_circles",
    "digits_0_vs_8",
    "digits_10",
    "four_blobs",
    "high_dimensional_sparse",
    "imbalanced_binary",
    "iris",
    "low_separation_multiclass",
    "noisy_binary",
    "noisy_multiclass",
    "redundant_multiclass",
    "two_moons",
    "wine",
    "dermatology",
    "diabetes",
    "haberman",
    "heart_statlog",
    "ionosphere",
    "parkinsons",
    "segment",
    "sonar",
    "vehicle",
]
# Preserve the original seven-value k range so the revision answers the
# reviewer concern about large-k selection rather than silently narrowing it.
K_VALUES = (1, 3, 5, 7, 9, 11, 15)
SEEDS = (0, 1, 2, 3)
ACCURACY_TOLERANCE = 0.01


def ensure_dirs() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, REGISTRY_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def load_manifest() -> dict[str, dict[str, str]]:
    with (DATA_DIR / "dataset_manifest.csv").open(encoding="utf-8") as handle:
        return {row["file"].removesuffix(".npz"): row for row in csv.DictReader(handle)}


def load_dataset(name: str, seed: int, max_samples: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(DATA_DIR / f"{name}.npz", allow_pickle=True)
    x = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"])
    if len(x) > max_samples:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(x), size=max_samples, replace=False))
        x, y = x[selected], y[selected]
    return x, y


def split_and_scale(
    x: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    try:
        x_train, x_rest, y_train, y_rest = train_test_split(
            x, y, test_size=0.50, random_state=seed, stratify=y
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_rest, y_rest, test_size=0.70, random_state=seed + 1000, stratify=y_rest
        )
    except ValueError:
        x_train, x_rest, y_train, y_rest = train_test_split(
            x, y, test_size=0.50, random_state=seed
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_rest, y_rest, test_size=0.70, random_state=seed + 1000
        )
    scaler = StandardScaler().fit(x_train)
    return (
        scaler.transform(x_train),
        y_train,
        scaler.transform(x_val),
        y_val,
        scaler.transform(x_test),
        y_test,
    )


def accuracy(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(prediction)))


def loo_error(x_train: np.ndarray, y_train: np.ndarray, k: int) -> float:
    if len(y_train) < 2:
        return float("nan")
    cache = loo_neighbor_cache(x_train, min(k, len(y_train) - 1))
    prediction, _, _ = predict_from_neighbors(y_train, cache.indices, classes=np.unique(y_train))
    return float(np.mean(prediction != y_train))


def fit_diagnostics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    k: int,
    *,
    y_query: np.ndarray | None = None,
) -> dict[str, object]:
    cache = build_neighbor_cache(x_train, x_query, k)
    influence = compute_influence(y_train, cache.indices, classes=np.unique(y_train))
    prediction = influence.predictions
    result: dict[str, object] = {
        "cache": cache,
        "influence": influence,
        "prediction": prediction,
        "accuracy": accuracy(y_query, prediction) if y_query is not None else float("nan"),
        "prv": float(np.mean(influence.exact_vulnerable)),
        "band_rate": float(np.mean(influence.band_flags)),
        "mean_exposure": float(np.mean(influence.neighborhood_exposure)),
        "max_exposure": int(np.max(influence.neighborhood_exposure)),
        "max_decisive_influence": int(np.max(influence.decisive_influence)),
        "r1": float(np.max(influence.directional_influence) / len(x_query)),
    }
    if y_query is not None:
        result["classwise"] = compute_classwise_influence(influence, y_query)
    return result


def legacy_same_split_diagnostics(
    y_train: np.ndarray,
    neighbors: np.ndarray,
    *,
    classes: np.ndarray,
) -> dict[str, object]:
    """Reproduce the archived early-break logic on the current split.

    This is an implementation audit, not a claim that the archived benchmark
    used the current split.  It deliberately uses the current cached neighbors
    and labels so the comparison isolates the early-break implementation
    change from the historical seed and split protocol.
    """

    reference = compute_influence(
        y_train, neighbors, classes=classes, exposure_mask=np.zeros(len(neighbors), dtype=bool)
    )
    labels, counts = reference.classes, reference.counts
    class_to_idx = {label: index for index, label in enumerate(labels)}
    predictions = labels[np.argmax(counts, axis=1)]
    vulnerable = np.zeros(len(neighbors), dtype=bool)
    exposure = np.zeros(len(y_train), dtype=int)
    band = np.zeros(len(neighbors), dtype=bool)
    gaps = top_two_gap(counts)
    if len(labels) == 2:
        k = counts.sum(axis=1)
        band = np.where(k % 2 == 1, gaps == 1, gaps <= 2)
    else:
        band = gaps <= 2
    for query_index, query_neighbors in enumerate(neighbors):
        current_counts = counts[query_index].copy()
        current_winner = predictions[query_index]
        for train_index in query_neighbors:
            old_index = class_to_idx[y_train[train_index]]
            found = False
            for target_index, target_class in enumerate(labels):
                if target_class == y_train[train_index]:
                    continue
                changed = current_counts.copy()
                changed[old_index] -= 1
                changed[target_index] += 1
                if labels[np.argmax(changed)] != current_winner:
                    vulnerable[query_index] = True
                    exposure[train_index] += 1
                    found = True
                    break
            if found:
                break
    return {
        "point_prv": float(np.mean(vulnerable)),
        "band_rate": float(np.mean(band)),
        "mean_exposure": float(np.mean(exposure)),
    }


def run_e0() -> None:
    baseline_tables = sorted(BASELINE_TABLE_DIR.glob("*.csv"))
    lines = [
        "# LEGACY_REPRODUCTION_AUDIT",
        "",
        "This E0 artifact preserves the archived tables as baseline evidence.",
        "No legacy result is mixed into revision result tables.",
        "",
        "| Artifact | Rows | Status |",
        "|---|---:|---|",
    ]
    for table in baseline_tables:
        with table.open(encoding="utf-8") as handle:
            rows = max(0, sum(1 for _ in handle) - 1)
        lines.append(f"| `{table.name}` | {rows} | archived baseline |")
    same_split_rows: list[dict[str, object]] = []
    historical_rows: list[dict[str, object]] = []
    revision_table = PROCESSED_DIR / "e2_deterministic_influence.csv"
    legacy_table = BASELINE_TABLE_DIR / "paa_multiclass_benchmark.csv"
    if revision_table.exists():
        with revision_table.open(encoding="utf-8", newline="") as handle:
            revised_rows = list(csv.DictReader(handle))
        for current in revised_rows:
            if current["split"] != "test":
                continue
            same_split_rows.append(
                {
                    "dataset": current["dataset"],
                    "seed": int(current["seed"]),
                    "k": int(current["k"]),
                    "comparison_scope": "same_revision_split_legacy_logic",
                    "same_split_indices": True,
                    "legacy_vulnerability_rate": float(current["same_split_legacy_prv"]),
                    "revision_point_prv": float(current["point_prv"]),
                    "delta_point_prv_minus_legacy": float(current["point_prv"])
                    - float(current["same_split_legacy_prv"]),
                    "legacy_flag_rate": float(current["same_split_legacy_band_rate"]),
                    "revision_band_rate": float(current["band_rate"]),
                    "delta_band_rate_minus_legacy": float(current["band_rate"])
                    - float(current["same_split_legacy_band_rate"]),
                    "legacy_mean_exposure": float(current["same_split_legacy_mean_exposure"]),
                    "revision_mean_exposure": float(current["mean_exposure"]),
                }
            )
        save_csv(PROCESSED_DIR / "e0_core_comparison.csv", same_split_rows)
    if revision_table.exists() and legacy_table.exists():
        manifest = load_manifest()
        display_to_stem = {row["dataset"]: stem for stem, row in manifest.items()}
        with legacy_table.open(encoding="utf-8", newline="") as handle:
            legacy_rows = list(csv.DictReader(handle))
        with revision_table.open(encoding="utf-8", newline="") as handle:
            revised_rows = list(csv.DictReader(handle))
        revised = {
            (row["dataset"], int(row["seed"]), int(row["k"])): row
            for row in revised_rows
            if row["split"] == "test"
        }
        for row in legacy_rows:
            key = (display_to_stem.get(row["dataset"], row["dataset"]), int(row["seed"]), int(row["k"]))
            current = revised.get(key)
            if current is None:
                continue
            historical_rows.append(
                {
                    "dataset": key[0],
                    "seed": key[1],
                    "k": key[2],
                    "comparison_scope": "historical_protocol_comparison",
                    "same_split_indices": False,
                    "legacy_vulnerability_rate": float(row["vulnerability_rate"]),
                    "revision_point_prv": float(current["point_prv"]),
                    "delta_point_prv_minus_legacy": float(current["point_prv"]) - float(row["vulnerability_rate"]),
                    "legacy_flag_rate": float(row["flag_rate"]),
                    "revision_band_rate": float(current["band_rate"]),
                    "delta_band_rate_minus_legacy": float(current["band_rate"]) - float(row["flag_rate"]),
                    "legacy_mean_exposure": float(row["mean_exposure"]),
                    "revision_mean_exposure": float(current["mean_exposure"]),
                }
            )
        save_csv(PROCESSED_DIR / "e0_historical_comparison.csv", historical_rows)
    if same_split_rows:
        lines.extend(
            [
                "",
                f"The same-split implementation audit is stored in `e0_core_comparison.csv` for {len(same_split_rows)} test rows.",
                "It fixes the revision train/validation/test indices and compares",
                "the corrected implementation with the archived early-break logic.",
            ]
        )
    if historical_rows:
        lines.extend(
            [
                "",
                f"The historical join is stored separately in `e0_historical_comparison.csv` for {len(historical_rows)} rows.",
                "Because the archived protocol used different random states and",
                "split-call order, these rows are descriptive and not paired evidence.",
            ]
        )
    lines.extend(
        [
            "",
            "The archived benchmark has an exposure early-break and a non-independent",
            "runtime measurement; these are documented in `reports/BASELINE_AUDIT.md`.",
            "The archived environment requires optional `imbalanced-learn`, which is",
            "not installed in the current execution environment.  Therefore E0 does",
            "not claim a successful legacy rerun of those cleaning baselines.",
            "",
            "The revision protocol uses the exact cached 23-dataset panel, four seeds,",
            "a 50/15/35 train/validation/test split, and the independent stage timing",
            "in E9.",
        ]
    )
    (PROCESSED_DIR / "LEGACY_REPRODUCTION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def run_e1() -> None:
    rows: list[dict[str, object]] = []
    band_false_negative = 0
    exact_reference_mismatches = 0
    for classes in range(2, 6):
        for k in range(1, 11):
            for counts in compositions(k, classes):
                labels = np.concatenate(
                    [np.full(count, label, dtype=int) for label, count in enumerate(counts)]
                )
                neighbors = np.arange(k, dtype=int)[None, :]
                result = compute_influence(labels, neighbors, classes=np.arange(classes))
                reference = brute_force_directional_influence(
                    labels, neighbors, classes=np.arange(classes)
                )
                if not np.array_equal(result.directional_influence, reference):
                    exact_reference_mismatches += 1
                if result.exact_vulnerable[0] and not result.band_flags[0]:
                    band_false_negative += 1
                rows.append(
                    {
                        "n_classes": classes,
                        "k": k,
                        "counts": ":".join(map(str, counts)),
                        "exact_vulnerable": int(result.exact_vulnerable[0]),
                        "band_flag": int(result.band_flags[0]),
                    }
                )
    probability_mismatch = 0
    probability_rows: list[dict[str, object]] = []
    for k in range(1, 9):
        for n_one in sorted({k // 2, (k + 1) // 2}):
            labels = np.array([0] * (k - n_one) + [1] * n_one)
            probability_vectors = [
                ("homogeneous_0.01", np.full(k, 0.01)),
                ("homogeneous_0.10", np.full(k, 0.10)),
                ("homogeneous_0.25", np.full(k, 0.25)),
                ("homogeneous_0.50", np.full(k, 0.50)),
                ("heterogeneous_increasing", np.linspace(0.02, 0.40, k)),
                ("heterogeneous_decreasing", np.linspace(0.40, 0.02, k)),
            ]
            for scenario, probabilities in probability_vectors:
                exact = binary_flip_probability(labels, probabilities, classes=[0, 1])
                brute = 0.0
                base_winner = 0 if np.sum(labels == 0) >= np.sum(labels == 1) else 1
                for flips in itertools.product([0, 1], repeat=len(labels)):
                    mass = np.prod([p if flip else 1.0 - p for p, flip in zip(probabilities, flips)])
                    changed = labels.copy()
                    for index, flip in enumerate(flips):
                        if flip:
                            changed[index] = 1 - changed[index]
                    changed_winner = 0 if np.sum(changed == 0) >= np.sum(changed == 1) else 1
                    brute += mass * float(base_winner != changed_winner)
                absolute_error = abs(exact - brute)
                probability_mismatch += int(absolute_error > 1e-12)
                probability_rows.append(
                    {
                        "k": k,
                        "n_class_1": n_one,
                        "scenario": scenario,
                        "exact": exact,
                        "brute": brute,
                        "absolute_error": absolute_error,
                    }
                )
    probability_cases = len(probability_rows)
    save_csv(PROCESSED_DIR / "e1_probability_cases.csv", probability_rows)
    summary = {
        "vote_configurations": len(rows),
        "exact_reference_mismatches": exact_reference_mismatches,
        "band_false_negatives": band_false_negative,
        "probability_cases": probability_cases,
        "probability_mismatches": probability_mismatch,
    }
    (PROCESSED_DIR / "e1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_csv(PROCESSED_DIR / "e1_vote_configurations.csv", rows)


def run_e2() -> None:
    manifest = load_manifest()
    rows: list[dict[str, object]] = []
    signed_rows: list[dict[str, object]] = []
    for dataset in ORIGINAL_23:
        print(f"E2 start {dataset}", flush=True)
        for seed in SEEDS:
            x, y = load_dataset(dataset, seed)
            x_train, y_train, x_val, y_val, x_test, y_test = split_and_scale(x, y, seed)
            loo_errors: dict[int, float] = {}
            for k_loo in K_VALUES:
                loo_cache_k = loo_neighbor_cache(x_train, k_loo)
                loo_predictions_k, _, _ = predict_from_neighbors(
                    y_train, loo_cache_k.indices, classes=np.unique(y_train)
                )
                loo_errors[k_loo] = float(np.mean(loo_predictions_k != y_train))
            for split_name, x_query, y_query in (
                ("validation", x_val, y_val),
                ("test", x_test, y_test),
            ):
                cache_max = build_neighbor_cache(x_train, x_query, max(K_VALUES))
                for k in K_VALUES:
                    neighbor_indices = cache_max.indices[:, : min(k, len(y_train))]
                    diag = compute_influence(
                        y_train, neighbor_indices, classes=np.unique(y_train)
                    )
                    legacy_diag = legacy_same_split_diagnostics(
                        y_train, neighbor_indices, classes=np.unique(y_train)
                    )
                    signed = compute_signed_error_change(
                        y_train, neighbor_indices, y_query, classes=np.unique(y_train)
                    )
                    prediction, _, _ = predict_from_neighbors(
                        y_train, neighbor_indices, classes=np.unique(y_train)
                    )
                    prototype_directional = np.max(diag.directional_influence, axis=1)
                    total_directional = int(np.sum(prototype_directional))
                    rows.append(
                        {
                            "dataset": dataset,
                            "group": manifest[dataset]["group"],
                            "seed": seed,
                            "split": split_name,
                            "n_train": len(y_train),
                            "n_query": len(y_query),
                            "n_classes": len(np.unique(y_train)),
                            "n_features": x_train.shape[1],
                            "k": k,
                            "loo_error": loo_errors[k],
                            "accuracy": accuracy(y_query, prediction),
                            "balanced_accuracy": float(
                                np.mean(
                                    [
                                        np.mean(prediction[y_query == label] == y_query[y_query == label])
                                        for label in np.unique(y_query)
                                        if np.any(y_query == label)
                                    ]
                                )
                            ),
                            "point_prv": float(np.mean(diag.exact_vulnerable)),
                            "band_rate": float(np.mean(diag.band_flags)),
                            "mean_exposure": float(np.mean(diag.neighborhood_exposure)),
                            "max_exposure": int(np.max(diag.neighborhood_exposure)),
                            "max_decisive_influence": int(np.max(diag.decisive_influence)),
                            "r1": float(np.max(diag.directional_influence) / len(y_query)),
                            "influence_concentration": float(
                                np.max(prototype_directional) / total_directional
                                if total_directional
                                else 0.0
                            ),
                            "n_vulnerable_queries": int(np.sum(diag.exact_vulnerable)),
                            "same_split_legacy_prv": legacy_diag["point_prv"],
                            "same_split_legacy_band_rate": legacy_diag["band_rate"],
                            "same_split_legacy_mean_exposure": legacy_diag["mean_exposure"],
                        }
                    )
                    for prototype_index, old_label in enumerate(y_train):
                        for target_index, target_label in enumerate(np.unique(y_train)):
                            if target_label == old_label:
                                continue
                            signed_rows.append(
                                {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "split": split_name,
                                    "k": k,
                                    "n_train": len(y_train),
                                    "n_query": len(y_query),
                                    "prototype_index": prototype_index,
                                    "old_label": old_label,
                                    "new_label": target_label,
                                    "signed_error_change": signed.signed_error_change[prototype_index, target_index],
                                    "baseline_error_rate": signed.baseline_error_rate,
                                    "changed_error_rate": signed.changed_error_rate[prototype_index, target_index],
                                    "prediction_change_rate": signed.prediction_change_rate[prototype_index, target_index],
                                }
                            )
            # Release the distance matrices and influence work arrays before
            # loading the next dataset.  This keeps the full 23-dataset panel
            # reproducible on constrained Windows runners.
            del x, y, x_train, y_train, x_val, y_val, x_test, y_test
            del loo_cache_k, loo_predictions_k, loo_errors, cache_max, neighbor_indices, diag, prediction
            gc.collect()
        print(f"E2 done {dataset}", flush=True)
    save_csv(PROCESSED_DIR / "e2_deterministic_influence.csv", rows)
    save_csv(PROCESSED_DIR / "e2_signed_error_change.csv", signed_rows)
    stats_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        if row["split"] == "test":
            grouped.setdefault((str(row["dataset"]), int(row["k"])), []).append(row)
    rng = np.random.default_rng(20260614)
    for metric in ("point_prv", "r1", "accuracy"):
        differences = []
        for dataset in ORIGINAL_23:
            at3 = grouped.get((dataset, 3), [])
            at_max = grouped.get((dataset, max(K_VALUES)), [])
            if not at3 or not at_max:
                continue
            mean3 = float(np.mean([float(row[metric]) for row in at3]))
            mean_max = float(np.mean([float(row[metric]) for row in at_max]))
            differences.append(mean_max - mean3)
        values = np.asarray(differences, dtype=float)
        bootstrap = values[rng.integers(0, len(values), size=(10000, len(values)))]
        bootstrap_means = bootstrap.mean(axis=1)
        stats_rows.append(
            {
                "comparison": f"k{max(K_VALUES)}_minus_k3",
                "metric": metric,
                "n_dataset_pairs": len(values),
                "mean_difference": float(values.mean()),
                "paired_sd": float(values.std(ddof=1)),
                "paired_cohen_d": float(values.mean() / values.std(ddof=1)) if values.std(ddof=1) else float("nan"),
                "bootstrap_95_low": float(np.quantile(bootstrap_means, 0.025)),
                "bootstrap_95_high": float(np.quantile(bootstrap_means, 0.975)),
            }
        )
    save_csv(PROCESSED_DIR / "e2_paired_statistics.csv", stats_rows)
    consistency = {
        "corrected_lower_bound_violations": 0,
        "old_k_lower_bound_violations": 0,
        "binary_odd_rows": 0,
    }
    for row in rows:
        if int(row["n_classes"]) != 2 or int(row["k"]) % 2 != 1:
            continue
        consistency["binary_odd_rows"] += 1
        point = float(row["point_prv"])
        global_risk = float(row["r1"])
        n_train = int(row["n_train"])
        k = int(row["k"])
        if global_risk + 1e-12 < ((k + 1) / (2 * n_train)) * point:
            consistency["corrected_lower_bound_violations"] += 1
        if global_risk + 1e-12 < point / k:
            consistency["old_k_lower_bound_violations"] += 1
    (PROCESSED_DIR / "e2_consistency_audit.json").write_text(
        json.dumps(consistency, indent=2), encoding="utf-8"
    )


def run_e3() -> None:
    rows: list[dict[str, object]] = []
    h2_rows: list[dict[str, object]] = []
    binary_datasets = ("breast_cancer_wisconsin", "digits_0_vs_8", "imbalanced_binary", "noisy_binary")
    for dataset in binary_datasets:
        x, y = load_dataset(dataset, 0, max_samples=800)
        x_train, y_train, _, _, x_test, _ = split_and_scale(x, y, 0)
        labels = np.unique(y_train)
        if len(labels) != 2:
            continue
        for requested_queries in (40, 80):
            if requested_queries == 80 and dataset != "breast_cancer_wisconsin":
                continue
            for k in ((1, 3, 5, 7) if requested_queries == 40 else (3, 7)):
                query_count = min(requested_queries, len(x_test))
                cache = build_neighbor_cache(x_train, x_test[:query_count], k)
                influence = compute_influence(y_train, cache.indices, classes=labels)
                if k % 2 == 1:
                    pivotal_count = (k + 1) / 2.0
                    point_risk = float(np.mean(influence.exact_vulnerable))
                    influence_counts = influence.decisive_influence.astype(float)
                    total_influence = float(np.sum(influence_counts))
                    h2 = (
                        float(np.sum(influence_counts**2) / total_influence**2)
                        if total_influence > 0.0
                        else 0.0
                    )
                    coefficient = float(np.sum(influence_counts**2) / len(cache.indices) ** 2)
                    predicted_coefficient = float((pivotal_count * point_risk) ** 2 * h2)
                    lower_bound, upper_bound = first_order_variance_bounds(
                        point_risk,
                        float(np.max(influence.directional_influence) / len(cache.indices)),
                        n_train=len(y_train),
                        k=k,
                    )
                    h2_rows.append(
                        {
                            "dataset": dataset,
                            "requested_queries": requested_queries,
                            "n_train": len(y_train),
                            "n_query": len(cache.indices),
                            "k": k,
                            "point_risk": point_risk,
                            "pivotal_count": pivotal_count,
                            "influence_sum": total_influence,
                            "influence_square_concentration_h2": h2,
                            "variance_coefficient_exact": coefficient,
                            "variance_coefficient_from_h2": predicted_coefficient,
                            "identity_absolute_error": abs(coefficient - predicted_coefficient),
                            "lower_bound": lower_bound,
                            "upper_bound": upper_bound,
                            "lower_bound_holds": bool(lower_bound <= coefficient + 1e-12),
                            "upper_bound_holds": bool(coefficient <= upper_bound + 1e-12),
                        }
                    )
                membership = np.bincount(cache.indices.ravel(), minlength=len(y_train))
                overlap_values = []
                for left in range(len(cache.indices)):
                    left_set = set(cache.indices[left].tolist())
                    for right in range(left + 1, len(cache.indices)):
                        right_set = set(cache.indices[right].tolist())
                        overlap_values.append(len(left_set & right_set) / len(left_set | right_set))
                mean_overlap = float(np.mean(overlap_values)) if overlap_values else 0.0
                probability_scenarios = {
                    "low_uniform": np.full(len(y_train), 0.01),
                    "moderate_uniform": np.full(len(y_train), 0.05),
                    "high_uniform": np.full(len(y_train), 0.15),
                    "heterogeneous_rank": np.linspace(0.005, 0.20, len(y_train)),
                    "correlated_all_or_none": np.full(len(y_train), 0.05),
                }
                for scenario, probabilities in probability_scenarios.items():
                    if requested_queries > 40 and scenario not in {"moderate_uniform", "high_uniform"}:
                        continue
                    moments = batch_risk_moments(
                        y_train, cache.indices, probabilities, classes=labels
                    )
                    first_order = first_order_risk(
                        influence.decisive_influence, probabilities, len(cache.indices)
                    )
                    repetitions = 500 if query_count == 40 else 200
                    if scenario == "correlated_all_or_none":
                        monte_carlo = monte_carlo_perfectly_correlated_batch_risk(
                            y_train,
                            cache.indices,
                            0.05,
                            repetitions=repetitions,
                            classes=labels,
                            seed=20260614 + k + query_count,
                        )
                        assumption = "perfectly_correlated_flips"
                    else:
                        monte_carlo = monte_carlo_batch_risk(
                            y_train,
                            cache.indices,
                            probabilities,
                            repetitions=repetitions,
                            classes=labels,
                            seed=20260614 + k + query_count,
                        )
                        assumption = "independent_flips"
                    variance_bound = float(
                        np.sum(probabilities * (1.0 - probabilities) * (membership / len(cache.indices)) ** 2)
                    )
                    ratio = (
                        variance_bound / moments.variance
                        if moments.variance > 1e-15
                        else float("nan")
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "queries": len(cache.indices),
                            "requested_queries": requested_queries,
                            "k": k,
                            "probability_scenario": scenario,
                            "assumption": assumption,
                            "mean_flip_probability": float(np.mean(probabilities)),
                            "mean_query_overlap": mean_overlap,
                            "exact_expectation_independent_model": moments.expectation,
                            "variance_exact_shared_flips": moments.variance,
                            "first_order_approximation": first_order,
                            "monte_carlo_mean": float(monte_carlo.mean()),
                            "monte_carlo_variance": float(monte_carlo.var()),
                            "monte_carlo_repetitions": len(monte_carlo),
                            "overlap_pair_count": moments.overlap_pair_count,
                            "total_query_pairs": moments.total_query_pairs,
                            "overlap_pair_fraction": (
                                moments.overlap_pair_count / moments.total_query_pairs
                                if moments.total_query_pairs
                                else 0.0
                            ),
                            "mc_abs_error_to_independent_expectation": abs(
                                float(monte_carlo.mean()) - moments.expectation
                            ),
                            "efron_stein_variance_bound": variance_bound,
                            "bound_to_exact_variance_ratio": ratio,
                            "exact_within_variance_bound": bool(moments.variance <= variance_bound + 1e-12),
                        }
                    )
    save_csv(PROCESSED_DIR / "e3_probability_risk.csv", rows)
    save_csv(PROCESSED_DIR / "e3_h2_variance_validation.csv", h2_rows)

    # Complete enumeration of a tiny fixed-neighborhood audit.  This is an
    # independent numerical check of the first-order variance expansion.
    audit_y = np.array([0, 0, 0, 1, 0, 0, 0, 0])
    audit_neighbors = np.array([[0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2]])
    audit_epsilon = 0.001
    audit_expectation, audit_variance = enumerate_shared_flip_moments(
        audit_y,
        audit_neighbors,
        audit_epsilon,
        classes=[0, 1],
    )
    audit_influence = compute_influence(
        audit_y, audit_neighbors, classes=[0, 1], tie_priority=[0, 1]
    )
    single_flip_counts = audit_influence.decisive_influence.astype(float)
    n_queries = len(audit_neighbors)
    point_risk = float(np.mean(audit_influence.exact_vulnerable))
    r_one = float(np.max(audit_influence.directional_influence) / n_queries)
    coefficient = float(np.sum(single_flip_counts**2) / n_queries**2)
    lower_bound, upper_bound = first_order_variance_bounds(
        point_risk,
        r_one,
        n_train=len(audit_y),
        k=3,
    )
    save_csv(
        PROCESSED_DIR / "e3_influence_variance_audit.csv",
        [
            {
                "n_train": len(audit_y),
                "n_query": n_queries,
                "k": 3,
                "epsilon": audit_epsilon,
                "state_count": 1 << len(audit_y),
                "single_flip_influence_sum": float(np.sum(single_flip_counts)),
                "single_flip_influence_square_sum": float(np.sum(single_flip_counts**2)),
                "point_prv": point_risk,
                "r_one": r_one,
                "exact_expectation": audit_expectation,
                "exact_variance": audit_variance,
                "variance_over_epsilon": audit_variance / audit_epsilon,
                "first_order_variance_coefficient": coefficient,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "bound_holds": bool(lower_bound <= coefficient <= upper_bound),
            }
        ],
    )


def run_e4() -> None:
    """Audit boundary probabilities under random-label models only.

    Fixed-label flip risks are generated in E3.  Keeping them out of this
    table prevents a class-probability model from being mixed with a
    post-observation label-perturbation model.
    """

    rows: list[dict[str, object]] = []
    class_probability_values = (0.01, 0.10, 0.25, 0.50)
    for k in (1, 3, 5, 7, 9, 11):
        m = (k - 1) // 2
        for class_probability in class_probability_values:
            exact = math.comb(2 * m + 1, m) * (
                class_probability * (1.0 - class_probability)
            ) ** m
            rows.append(
                {
                    "model": "random_training_labels_homogeneous",
                    "probability_event": "odd_k_boundary",
                    "k": k,
                    "class_probability": class_probability,
                    "exact_boundary_probability": exact,
                    "bound": "exact majority boundary mass",
                    "scope": "random training labels; iid class probability",
                }
            )
    # Conditional spatial check with a known Lipschitz regression function.
    # This is a finite-sample verification of the stated assumptions, not a
    # claim about arbitrary dependent labels or real-world label posteriors.
    lipschitz_constant = 0.25
    spatial_rows = []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(20260614 + seed)
        x_train = rng.uniform(-1.0, 1.0, size=(600, 2))
        x_query = rng.uniform(-1.0, 1.0, size=(80, 2))
        eta_train = 0.5 + 0.25 * np.tanh(x_train[:, 0])
        eta_query = 0.5 + 0.25 * np.tanh(x_query[:, 0])
        for k in (1, 3, 5, 7, 9, 11):
            cache = build_neighbor_cache(x_train, x_query, k)
            exact_values = []
            bounds = []
            radii = []
            for query_index, neighbor_indices in enumerate(cache.indices):
                local_probabilities = eta_train[neighbor_indices]
                exact = odd_k_boundary_probability(local_probabilities)
                radius = float(np.max(cache.distances[query_index]))
                a = max(abs(float(eta_query[query_index]) - 0.5) - lipschitz_constant * radius, 0.0)
                bound = math.exp(-2.0 * k * max(a - 1.0 / (2.0 * k), 0.0) ** 2)
                exact_values.append(exact)
                bounds.append(bound)
                radii.append(radius)
            label_mc = []
            for _ in range(200):
                sampled_labels = (rng.random(len(x_train)) < eta_train).astype(int)
                local_counts = np.sum(sampled_labels[cache.indices], axis=1)
                label_mc.append(float(np.mean((local_counts == k // 2) | (local_counts == k // 2 + 1))))
            spatial_rows.append(
                {
                    "model": "random_training_labels_spatial_conditional",
                    "probability_event": "odd_k_boundary",
                    "seed": seed,
                    "k": k,
                    "queries": len(x_query),
                    "lipschitz_constant": lipschitz_constant,
                    "mean_neighbor_radius": float(np.mean(radii)),
                    "mean_exact_boundary_probability": float(np.mean(exact_values)),
                    "mean_bound": float(np.mean(bounds)),
                    "max_exact_minus_bound": float(np.max(np.asarray(exact_values) - np.asarray(bounds))),
                    "label_monte_carlo_mean": float(np.mean(label_mc)),
                    "label_monte_carlo_repetitions": 200,
                    "bound_holds": bool(np.all(np.asarray(exact_values) <= np.asarray(bounds) + 1e-12)),
                    "bound_scope": "conditional feature model; independent labels",
                    "scope": "random training labels conditional on features",
                }
            )
    rows.extend(spatial_rows)
    save_csv(PROCESSED_DIR / "e4_distribution_stability.csv", rows)


def rank_scores(scores: np.ndarray, budget: int) -> np.ndarray:
    order = np.lexsort((np.arange(len(scores)), -scores))
    return order[:budget]


def normalize_score(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if hi <= lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def run_e5() -> None:
    rows: list[dict[str, object]] = []
    binary_datasets = ("breast_cancer_wisconsin", "digits_0_vs_8", "imbalanced_binary")
    for dataset in binary_datasets:
        for seed in SEEDS:
            x, y = load_dataset(dataset, seed, max_samples=800)
            x_train, y_clean, x_val, y_val, x_test, y_test = split_and_scale(x, y, seed)
            labels = np.unique(y_clean)
            for noise_rate in (0.05, 0.10, 0.20):
                rng = np.random.default_rng(seed * 1000 + int(noise_rate * 10000) + 700)
                corrupted = rng.random(len(y_clean)) < noise_rate
                y_noisy = y_clean.copy()
                y_noisy[corrupted] = np.where(y_noisy[corrupted] == labels[0], labels[1], labels[0])
                cache = build_neighbor_cache(x_train, x_val, 5)
                diag = compute_influence(y_noisy, cache.indices, classes=labels)
                loo_cache = loo_neighbor_cache(x_train, 5)
                loo_pred, _, _ = predict_from_neighbors(y_noisy, loo_cache.indices, classes=labels)
                influence_score = normalize_score(diag.decisive_influence.astype(float))
                loo_score = normalize_score((loo_pred != y_noisy).astype(float))
                scores = {
                    "random": rng.random(len(y_clean)),
                    "neighborhood_exposure": diag.neighborhood_exposure.astype(float),
                    "exact_decisive_influence": diag.decisive_influence.astype(float),
                    "loo_error": (loo_pred != y_noisy).astype(float),
                    "combined_influence_loo": 0.5 * influence_score + 0.5 * loo_score,
                }
                base_pred, _, _ = predict_from_neighbors(y_noisy, cache.indices, classes=labels)
                test_cache = build_neighbor_cache(x_train, x_test, 5)
                base_test, _, _ = predict_from_neighbors(y_noisy, test_cache.indices, classes=labels)
                for budget_fraction in (0.05, 0.10, 0.20):
                    budget = max(1, int(budget_fraction * len(y_clean)))
                    for method, score in scores.items():
                        selected = rank_scores(score, budget)
                        y_repaired = y_noisy.copy()
                        y_repaired[selected] = y_clean[selected]
                        repaired_val, _, _ = predict_from_neighbors(y_repaired, cache.indices, classes=labels)
                        repaired_pred, _, _ = predict_from_neighbors(y_repaired, test_cache.indices, classes=labels)
                        baseline_accuracy = accuracy(y_test, base_test)
                        repaired_accuracy = accuracy(y_test, repaired_pred)
                        rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "noise_rate": noise_rate,
                                "method": method,
                                "budget_fraction": budget_fraction,
                                "selected_budget": budget,
                                "injected_error_rate": float(np.mean(corrupted)),
                                "selected_error_rate": float(np.mean(corrupted[selected])),
                                "selected_corrupted_count": int(np.sum(corrupted[selected])),
                                "validation_change_rate": float(np.mean(base_pred != repaired_val)),
                                "baseline_test_accuracy": baseline_accuracy,
                                "oracle_repaired_test_accuracy": repaired_accuracy,
                                "test_accuracy_gain": repaired_accuracy - baseline_accuracy,
                            }
                        )
    save_csv(PROCESSED_DIR / "e5_label_audit.csv", rows)


def run_e6() -> None:
    rows: list[dict[str, object]] = []
    for k in range(2, 10):
        construction = ring_support_construction(k, epsilon=min(0.01, 0.1 / k))
        report = verify_ring_separation(construction)
        query = np.zeros((1, 2))
        cache = build_neighbor_cache(construction.points, query, k)
        base, _, _ = predict_from_neighbors(
            construction.labels, cache.indices, classes=[0, 1], tie_priority=[0, 1]
        )
        changed_labels = construction.labels.copy()
        changed_labels[construction.pivotal_representative] = 1
        changed, _, _ = predict_from_neighbors(
            changed_labels, cache.indices, classes=[0, 1], tie_priority=[0, 1]
        )
        rows.append(
            {
                "k": k,
                "epsilon": construction.epsilon,
                "analytic_separation_margin": ring_separation_margin(k, construction.epsilon),
                "cross_cluster_separation_holds": report["cross_cluster_separation_holds"],
                "center_representatives_are_first_k": report["center_representatives_are_first_k"],
                "baseline_prediction": base[0],
                "after_pivotal_relabel": changed[0],
            }
        )
    displacement_train = np.array([[-0.1, 0.0], [0.1, 0.0]])
    displacement_labels = np.array([0, 1])
    displacement_query = np.array([[0.0, 0.0]])
    baseline_cache = build_neighbor_cache(displacement_train, displacement_query, 1)
    baseline_prediction, _, _ = predict_from_neighbors(
        displacement_labels, baseline_cache.indices, classes=[0, 1]
    )
    for delta in (0.0, 0.05, 0.31, 0.50):
        moved = displacement_train.copy()
        moved[0, 0] += delta
        moved_cache = build_neighbor_cache(moved, displacement_query, 1)
        moved_prediction, _, _ = predict_from_neighbors(
            displacement_labels, moved_cache.indices, classes=[0, 1]
        )
        rows.append(
            {
                "k": 1,
                "epsilon": "",
                "analytic_separation_margin": "",
                "cross_cluster_separation_holds": "",
                "center_representatives_are_first_k": "",
                "baseline_prediction": baseline_prediction[0],
                "after_pivotal_relabel": moved_prediction[0],
                "perturbation": "single_prototype_geometric_displacement",
                "delta": delta,
                "neighbor_exchange": bool(baseline_cache.indices[0, 0] != moved_cache.indices[0, 0]),
                "prediction_changed": bool(baseline_prediction[0] != moved_prediction[0]),
            }
        )
    combined_construction = ring_support_construction(3, epsilon=0.06)
    combined_query = np.zeros((1, 2))
    combined_base_cache = build_neighbor_cache(combined_construction.points, combined_query, 3)
    combined_base, _, _ = predict_from_neighbors(
        combined_construction.labels, combined_base_cache.indices, classes=[0, 1], tie_priority=[0, 1]
    )
    relabeled = combined_construction.labels.copy()
    relabeled[combined_construction.pivotal_representative] = 1
    fixed_label_prediction, _, _ = predict_from_neighbors(
        relabeled, combined_base_cache.indices, classes=[0, 1], tie_priority=[0, 1]
    )
    direction = combined_construction.points[combined_construction.pivotal_representative]
    direction = direction / np.linalg.norm(direction)
    for delta in (0.0, 0.005, 0.015, 0.030, 0.050):
        moved_points = combined_construction.points.copy()
        moved_points[combined_construction.pivotal_representative] += delta * direction
        moved_cache = build_neighbor_cache(moved_points, combined_query, 3)
        combined_prediction, _, _ = predict_from_neighbors(
            relabeled, moved_cache.indices, classes=[0, 1], tie_priority=[0, 1]
        )
        rows.append(
            {
                "k": 3,
                "epsilon": combined_construction.epsilon,
                "perturbation": "relabel_plus_radial_displacement",
                "delta": delta,
                "baseline_prediction": combined_base[0],
                "fixed_label_prediction": fixed_label_prediction[0],
                "combined_prediction": combined_prediction[0],
                "neighbor_exchange": bool(
                    not np.array_equal(combined_base_cache.indices[0], moved_cache.indices[0])
                ),
                "fixed_label_prediction_changed": bool(fixed_label_prediction[0] != combined_base[0]),
                "combined_prediction_changed": bool(combined_prediction[0] != combined_base[0]),
            }
        )
    # A boundary-controlled low/mid/high vote-gap diagnostic.  Each scenario
    # records label-only, geometry-only, and joint interventions separately.
    # The target is deliberately placed so that one condition enters the
    # neighborhood, one exits it, and one remains inside without exchange.
    gap_groups = {
        # Indices 0 and 1 are the target/candidate pair; indices 2--5 are the
        # four strictly inner neighbors.  These label vectors give pre-change
        # vote gaps exactly 1, 3, and 5 whenever the target is inside.
        "low_gap": np.array([0, 1, 0, 0, 1, 1]),
        "mid_gap": np.array([0, 1, 0, 0, 0, 1]),
        "high_gap": np.array([0, 1, 0, 0, 0, 0]),
    }
    outside_gap_groups = {
        "low_gap": np.array([0, 1, 0, 0, 1, 1]),
        "mid_gap": np.array([0, 0, 0, 0, 0, 1]),
        "high_gap": np.array([0, 0, 0, 0, 0, 0]),
    }
    gap_rng = np.random.default_rng(20260615)
    motion_scenarios = {
        "enters": {
            "target_radius": 1.04,
            "candidate_radius": 1.00,
            "motion_radius": 0.06,
            "motion_sign": -1.0,
        },
        "stays_inside": {
            "target_radius": 0.96,
            "candidate_radius": 1.04,
            "motion_radius": 0.03,
            "motion_sign": 1.0,
        },
        "exits": {
            "target_radius": 1.00,
            "candidate_radius": 1.04,
            "motion_radius": 0.06,
            "motion_sign": 1.0,
        },
    }
    for scenario, parameters in motion_scenarios.items():
        boundary = boundary_motion_construction(**{
            key: parameters[key]
            for key in ("target_radius", "candidate_radius", "motion_radius")
        })
        gap_points = boundary.points
        gap_query = boundary.query
        label_groups = outside_gap_groups if scenario == "enters" else gap_groups
        for group, group_labels in label_groups.items():
            base_cache = build_neighbor_cache(gap_points, gap_query, 5)
            base_labels = group_labels.copy()
            base_prediction, base_counts, _ = predict_from_neighbors(
                base_labels, base_cache.indices, classes=[0, 1]
            )
            moved_labels = base_labels.copy()
            moved_labels[boundary.target_index] = 1
            fixed_label_prediction, _, _ = predict_from_neighbors(
                moved_labels, base_cache.indices, classes=[0, 1]
            )
            fixed_changed = int(fixed_label_prediction[0] != base_prediction[0])
            geometry_changed = 0
            combined_changed = 0
            exchanges = 0
            target_inside = 0
            label_effect = 0
            inside_trials = 0
            outside_trials = 0
            label_effect_inside = 0
            label_effect_outside = 0
            joint_minus_sum = 0
            trials = 200
            for _ in range(trials):
                moved_points = gap_points.copy()
                angle = gap_rng.uniform(-0.20, 0.20)
                direction = np.array([np.cos(angle), np.sin(angle)])
                moved_points[boundary.target_index] += (
                    parameters["motion_sign"] * boundary.motion_radius * direction
                )
                moved_cache = build_neighbor_cache(moved_points, gap_query, 5)
                geometry_prediction, _, _ = predict_from_neighbors(
                    base_labels, moved_cache.indices, classes=[0, 1]
                )
                combined_prediction, _, _ = predict_from_neighbors(
                    moved_labels, moved_cache.indices, classes=[0, 1]
                )
                geometry_change = int(geometry_prediction[0] != base_prediction[0])
                combined_change = int(combined_prediction[0] != base_prediction[0])
                is_inside = boundary.target_index in moved_cache.indices[0]
                exchanges += int(not np.array_equal(base_cache.indices[0], moved_cache.indices[0]))
                target_inside += int(is_inside)
                geometry_changed += geometry_change
                combined_changed += combined_change
                label_effect_now = int(combined_prediction[0] != geometry_prediction[0])
                label_effect += label_effect_now
                joint_minus_sum += combined_change - fixed_changed - geometry_change
                if is_inside:
                    inside_trials += 1
                    label_effect_inside += label_effect_now
                else:
                    outside_trials += 1
                    label_effect_outside += label_effect_now
            rows.append(
                {
                    "perturbation": "vote_gap_stratified_boundary_motion",
                    "vote_gap_group": group,
                    "motion_condition": scenario,
                    "pre_change_vote_gap": int(top_two_gap(base_counts)[0]),
                    "baseline_counts": ":".join(map(str, base_counts[0].tolist())),
                    "target_initially_in_neighbor": bool(boundary.target_index in base_cache.indices[0]),
                    "boundary_gap": abs(boundary.candidate_radius - boundary.target_radius),
                    "motion_radius": boundary.motion_radius,
                    "motion_sign": parameters["motion_sign"],
                    "trials": trials,
                    "neighbor_exchange_rate": exchanges / trials,
                    "target_in_neighbor_rate": target_inside / trials,
                    "fixed_label_prediction_changed": bool(fixed_changed),
                    "geometry_prediction_changed_rate": geometry_changed / trials,
                    "combined_prediction_changed_rate": combined_changed / trials,
                    "fixed_label_change_rate": fixed_changed,
                    "label_effect_rate": label_effect / trials,
                    "label_effect_rate_inside": label_effect_inside / inside_trials if inside_trials else 0.0,
                    "label_effect_rate_outside": label_effect_outside / outside_trials if outside_trials else 0.0,
                    "joint_minus_sum_single_effects": joint_minus_sum / trials,
                    "prediction_changes": combined_changed,
                    "change_rate": combined_changed / trials,
                    "certificate_applicable": bool(single_prototype_motion_safe(int(top_two_gap(base_counts)[0]))),
                }
            )

    # Forty thousand random checks of the conservative G>2 certificate.
    rng = np.random.default_rng(20260614)
    eligible = 0
    violations = 0
    for _ in range(40000):
        n = 20
        k = int(rng.choice([1, 3, 5, 7]))
        points = rng.normal(size=(n, 2))
        query = rng.normal(size=(1, 2))
        labels = rng.integers(0, 2, size=n)
        base_cache = build_neighbor_cache(points, query, k)
        base, counts, _ = predict_from_neighbors(labels, base_cache.indices, classes=[0, 1])
        gap = int(top_two_gap(counts)[0])
        if not bool(single_prototype_motion_safe(gap)):
            continue
        eligible += 1
        target = int(rng.integers(n))
        moved_points = points.copy()
        moved_points[target] += rng.normal(scale=2.0, size=2)
        moved_labels = labels.copy()
        moved_labels[target] = 1 - moved_labels[target]
        moved_cache = build_neighbor_cache(moved_points, query, k)
        moved, _, _ = predict_from_neighbors(moved_labels, moved_cache.indices, classes=[0, 1])
        violations += int(moved[0] != base[0])
    rows.append(
        {
            "perturbation": "single_motion_plus_relabel_certificate",
            "certificate": "pre-change top-two gap > 2",
            "random_trials": 40000,
            "eligible_trials": eligible,
            "certificate_violations": violations,
            "certificate_holds": violations == 0,
        }
    )
    # A small perturbation check on three benchmark datasets.  Labels are
    # changed together with the feature position to exercise the stated scope.
    for dataset in ("breast_cancer_wisconsin", "two_moons", "imbalanced_binary"):
        x, y = load_dataset(dataset, 0, max_samples=800)
        x_train, y_train, _, _, x_test, _ = split_and_scale(x, y, 0)
        for k in (3, 5, 7):
            query = x_test[:40]
            base_cache = build_neighbor_cache(x_train, query, k)
            base, counts, _ = predict_from_neighbors(y_train, base_cache.indices, classes=np.unique(y_train))
            candidate_queries = 0
            changed = 0
            local_rng = np.random.default_rng(20260614 + k)
            for query_index in range(len(query)):
                gap = int(top_two_gap(counts[query_index : query_index + 1])[0])
                if not bool(single_prototype_motion_safe(gap)):
                    continue
                candidate_queries += 1
                for target in local_rng.choice(len(x_train), size=min(10, len(x_train)), replace=False):
                    moved = x_train.copy()
                    direction = local_rng.normal(size=x_train.shape[1])
                    direction /= max(np.linalg.norm(direction), 1e-12)
                    moved[target] += 0.01 * direction
                    labels = y_train.copy()
                    labels[target] = 1 - labels[target]
                    moved_prediction, _, _ = predict_from_neighbors(
                        labels,
                        build_neighbor_cache(moved, query[query_index : query_index + 1], k).indices,
                        classes=np.unique(y_train),
                    )
                    changed += int(moved_prediction[0] != base[query_index])
            rows.append(
                {
                    "perturbation": "benchmark_small_motion_plus_relabel",
                    "dataset": dataset,
                    "k": k,
                    "candidate_queries_gap_gt_2": candidate_queries,
                    "tested_motion_relabel_pairs": candidate_queries * min(10, len(x_train)),
                    "prediction_changes": changed,
                    "certificate_holds": changed == 0,
                }
            )
    save_csv(PROCESSED_DIR / "e6_geometry_construction.csv", rows)


def run_e7() -> None:
    from knn_reliability.knn import counts_from_neighbors

    rows: list[dict[str, object]] = []
    for case, y_train, neighbors in (
        ("binary_even_tie", np.array([0, 1, 0, 1]), np.array([[0, 1, 2, 3]])),
        ("binary_even_tie_larger", np.array([0, 0, 0, 1, 1, 1]), np.array([[0, 1, 2, 3, 4, 5]])),
        ("multiclass_three_way_tie", np.array([0, 1, 2, 0, 1, 2]), np.array([[0, 1, 2, 3, 4, 5]])),
    ):
        labels, counts = counts_from_neighbors(y_train, neighbors, classes=np.unique(y_train))
        distribution = random_tie_distribution(counts, labels)[0]
        deterministic, _ = predict_from_counts(counts, labels, tie_priority=labels)
        rows.append(
            {
                "case": case,
                "policy": "random_tie",
                "k": len(neighbors[0]),
                "class_labels": ":".join(map(str, labels.tolist())),
                "tie_probabilities": ":".join(f"{value:.6f}" for value in distribution),
                "deterministic_priority_prediction": int(labels[0]),
            }
        )
        for train_index in range(len(y_train)):
            for replacement in labels:
                if replacement == y_train[train_index]:
                    continue
                changed = y_train.copy()
                changed[train_index] = replacement
                _, changed_counts = counts_from_neighbors(changed, neighbors, classes=labels)
                changed_distribution = random_tie_distribution(changed_counts, labels)[0]
                changed_deterministic, _ = predict_from_counts(
                    changed_counts, labels, tie_priority=labels
                )
                rows.append(
                    {
                        "case": case,
                        "policy": "random_tie_relabel",
                        "k": len(neighbors[0]),
                        "train_index": train_index,
                        "old_label": int(y_train[train_index]),
                        "replacement_label": int(replacement),
                        "before_probabilities": ":".join(
                            f"{value:.6f}" for value in distribution
                        ),
                        "after_probabilities": ":".join(
                            f"{value:.6f}" for value in changed_distribution
                        ),
                        "total_variation_distance": float(
                            0.5 * np.sum(np.abs(distribution - changed_distribution))
                        ),
                        "deterministic_prediction_changed": bool(
                            changed_deterministic[0] != deterministic[0]
                        ),
                    }
                )

    y_train = np.array([0, 1, 0, 1])
    neighbors = np.array([[0, 1, 2, 3]])
    distance_scenarios = {
        "equal_distance": np.array([[1.0, 1.0, 1.0, 1.0]]),
        "mild_distance": np.array([[1.0, 1.1, 1.4, 1.7]]),
        "strong_distance": np.array([[1.0, 1.5, 2.0, 3.0]]),
    }
    labels = np.array([0, 1])
    for distance_name, distances in distance_scenarios.items():
        for power in (0.0, 1.0, 2.0):
            weighted_prediction, weighted_scores = weighted_predict_from_neighbors(
                y_train, neighbors, distances, classes=labels, tie_priority=labels, power=power
            )
            vulnerable = 0
            for train_index in range(len(y_train)):
                changed = y_train.copy()
                changed[train_index] = 1 - changed[train_index]
                changed_prediction, _ = weighted_predict_from_neighbors(
                    changed, neighbors, distances, classes=labels, tie_priority=labels, power=power
                )
                vulnerable += int(changed_prediction[0] != weighted_prediction[0])
            rows.append(
                {
                    "case": distance_name,
                    "policy": "weighted_vote",
                    "power": power,
                    "weighted_prediction": int(weighted_prediction[0]),
                    "weighted_class0_score": weighted_scores[0, 0],
                    "weighted_class1_score": weighted_scores[0, 1],
                    "weighted_margin": abs(weighted_scores[0, 0] - weighted_scores[0, 1]),
                    "vulnerable_prototypes": vulnerable,
                }
            )

    x, y = load_dataset("imbalanced_binary", 0)
    x_train, y_train, _, _, x_test, y_test = split_and_scale(x, y, 0)
    train_labels = np.unique(y_train)
    for k in K_VALUES:
        cache = build_neighbor_cache(x_train, x_test, k)
        diag = compute_influence(y_train, cache.indices, classes=train_labels)
        classwise = compute_classwise_influence(diag, y_test)
        prototype_exposure = diag.neighborhood_exposure
        for label in train_labels:
            prototype_mask = y_train == label
            rows.append(
                {
                    "case": "imbalanced_binary_class_stratified",
                    "policy": "class_stratified",
                    "k": k,
                    "class": int(label),
                    "training_class_fraction": float(np.mean(prototype_mask)),
                    "mean_prototype_exposure": float(np.mean(prototype_exposure[prototype_mask])),
                    "max_prototype_exposure": int(np.max(prototype_exposure[prototype_mask])),
                    "query_class_count": int(classwise[label]["n_queries"]),
                    "query_class_prv": classwise[label]["prv"],
                    "query_class_mean_decisive_influence": classwise[label]["mean_decisive_influence"],
                }
            )
    save_csv(PROCESSED_DIR / "e7_tie_weight_imbalance.csv", rows)


def run_e8() -> None:
    rows: list[dict[str, object]] = []
    for dataset in ORIGINAL_23:
        for seed in SEEDS:
            x, y = load_dataset(dataset, seed)
            x_train, y_train, x_val, y_val, x_test, y_test = split_and_scale(x, y, seed)
            validation_rows = []
            for k in K_VALUES:
                val_diag = fit_diagnostics(x_train, y_train, x_val, k, y_query=y_val)
                test_diag = fit_diagnostics(x_train, y_train, x_test, k, y_query=y_test)
                validation_rows.append((k, val_diag, test_diag))
            k_training_loo = min(
                validation_rows, key=lambda item: (loo_error(x_train, y_train, item[0]), item[0])
            )[0]
            best_validation_accuracy = max(float(item[1]["accuracy"]) for item in validation_rows)
            eligible = [
                item for item in validation_rows
                if float(item[1]["accuracy"]) >= best_validation_accuracy - ACCURACY_TOLERANCE
            ]
            k_validation_accuracy = max(
                validation_rows, key=lambda item: (item[1]["accuracy"], -item[0])
            )[0]
            k_reliability = min(eligible, key=lambda item: (item[1]["prv"], item[0]))[0]
            # Reviewer-requested comparator: among the same accuracy-qualified
            # candidates, choose the largest k without using test labels.
            k_maximum_eligible = max(eligible, key=lambda item: item[0])[0]
            for method, selected_k in (
                ("training_loo", k_training_loo),
                ("validation_accuracy", k_validation_accuracy),
                ("validation_prv", k_reliability),
                ("maximum_eligible_k", k_maximum_eligible),
            ):
                selected_val = next(item[1] for item in validation_rows if item[0] == selected_k)
                test_diag = next(item[2] for item in validation_rows if item[0] == selected_k)
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "selection_method": method,
                        "selected_k": selected_k,
                        "accuracy_tolerance": ACCURACY_TOLERANCE,
                        "best_validation_accuracy": best_validation_accuracy,
                        "selected_validation_accuracy": selected_val["accuracy"],
                        "selected_validation_prv": selected_val["prv"],
                        "test_accuracy": test_diag["accuracy"],
                        "test_prv": test_diag["prv"],
                        "test_r1": test_diag["r1"],
                    }
                )
    save_csv(PROCESSED_DIR / "e8_selection_protocol.csv", rows)


def run_e9() -> None:
    def timed(callable_, *, repeats: int, warmups: int = 1):
        for _ in range(warmups):
            callable_()
        durations = []
        result = None
        for _ in range(repeats):
            start = time.perf_counter()
            result = callable_()
            durations.append(time.perf_counter() - start)
        values = np.asarray(durations, dtype=float)
        return result, {
            "repeats": repeats,
            "warmups": warmups,
            "median": float(np.median(values)),
            "q1": float(np.quantile(values, 0.25)),
            "q3": float(np.quantile(values, 0.75)),
        }

    def peak_bytes(callable_) -> int:
        gc.collect()
        tracemalloc.start()
        try:
            callable_()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return int(peak)

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260614)
    for n in (80, 160, 320):
        q = min(120, n)
        x_train = rng.normal(size=(n, 6))
        x_query = rng.normal(size=(q, 6))
        y_train = rng.integers(0, 2, size=n)
        for k in (3, 7):
            cache, cache_timing = timed(
                lambda: build_neighbor_cache(x_train, x_query, k), repeats=7
            )
            fast, fast_timing = timed(
                lambda: compute_influence(y_train, cache.indices, classes=[0, 1]),
                repeats=7,
            )
            naive_seconds = float("nan")
            naive_q1 = float("nan")
            naive_q3 = float("nan")
            naive_status = "not_executed_budget_policy"
            if n <= 160:
                slow, naive_timing = timed(
                    lambda: brute_force_directional_influence(
                        y_train, cache.indices, classes=[0, 1]
                    ),
                    repeats=5,
                )
                naive_seconds = naive_timing["median"]
                naive_q1 = naive_timing["q1"]
                naive_q3 = naive_timing["q3"]
                if not np.array_equal(fast.directional_influence, slow):
                    raise RuntimeError("E9 fast and reference influence disagree")
                naive_status = "completed"
            rows.append(
                {
                    "benchmark_family": "deterministic_influence",
                    "method": "batched_vs_reference",
                    "n_train": n,
                    "n_query": q,
                    "k": k,
                    "timing_repeats": 7,
                    "timing_warmups": 1,
                    "cache_seconds": cache_timing["median"],
                    "cache_q1_seconds": cache_timing["q1"],
                    "cache_q3_seconds": cache_timing["q3"],
                    "exact_batched_seconds": fast_timing["median"],
                    "exact_batched_q1_seconds": fast_timing["q1"],
                    "exact_batched_q3_seconds": fast_timing["q3"],
                    "naive_exhaustive_seconds": naive_seconds,
                    "naive_exhaustive_q1_seconds": naive_q1,
                    "naive_exhaustive_q3_seconds": naive_q3,
                    "naive_status": naive_status,
                    "speedup": (
                        naive_seconds / fast_timing["median"]
                        if np.isfinite(naive_seconds) and fast_timing["median"] > 0
                        else float("nan")
                    ),
                }
            )

    def append_probability_panel(
        *,
        benchmark_family: str,
        panel: str,
        n: int,
        q: int,
        k: int,
        neighbors: np.ndarray,
        y_train: np.ndarray,
        probabilities: np.ndarray,
        seed: int,
        overlap_mode: str,
        timing_repeats: int = 5,
        dense_repeats: int = 3,
        mc_repeats: int = 5,
    ) -> None:
        """Record exact dense/overlap/Monte Carlo timings for one control panel."""
        overlap, overlap_timing = timed(
            lambda: batch_risk_moments(
                y_train, neighbors, probabilities, classes=[0, 1]
            ),
            repeats=timing_repeats,
        )
        dense, dense_timing = timed(
            lambda: dense_batch_risk_moments(
                y_train, neighbors, probabilities, classes=[0, 1]
            ),
            repeats=dense_repeats,
        )
        if not np.allclose(
            [overlap.expectation, overlap.variance],
            [dense.expectation, dense.variance],
            atol=1e-13,
        ):
            raise RuntimeError(f"E9 dense and overlap paths disagree in {panel}")
        monte_carlo, mc_timing = timed(
            lambda: monte_carlo_batch_risk(
                y_train,
                neighbors,
                probabilities,
                repetitions=500,
                classes=[0, 1],
                seed=seed,
            ),
            repeats=mc_repeats,
        )
        common = {
            "benchmark_family": benchmark_family,
            "panel": panel,
            "overlap_mode": overlap_mode,
            "n_train": n,
            "n_query": q,
            "k": k,
            "overlap_pair_count": overlap.overlap_pair_count,
            "total_query_pairs": overlap.total_query_pairs,
            "overlap_pair_fraction": (
                overlap.overlap_pair_count / overlap.total_query_pairs
                if overlap.total_query_pairs
                else 0.0
            ),
            "exact_expectation": overlap.expectation,
            "exact_variance": overlap.variance,
        }
        rows.extend(
            [
                {
                    **common,
                    "method": "overlap_graph_exact",
                    "timing_repeats": overlap_timing["repeats"],
                    "timing_warmups": overlap_timing["warmups"],
                    "runtime_median_seconds": overlap_timing["median"],
                    "runtime_q1_seconds": overlap_timing["q1"],
                    "runtime_q3_seconds": overlap_timing["q3"],
                    "peak_traced_bytes": peak_bytes(
                        lambda: batch_risk_moments(
                            y_train, neighbors, probabilities, classes=[0, 1]
                        )
                    ),
                    "expectation_estimate": overlap.expectation,
                    "variance_estimate": overlap.variance,
                },
                {
                    **common,
                    "method": "dense_pairwise_exact",
                    "timing_repeats": dense_timing["repeats"],
                    "timing_warmups": dense_timing["warmups"],
                    "runtime_median_seconds": dense_timing["median"],
                    "runtime_q1_seconds": dense_timing["q1"],
                    "runtime_q3_seconds": dense_timing["q3"],
                    "peak_traced_bytes": peak_bytes(
                        lambda: dense_batch_risk_moments(
                            y_train, neighbors, probabilities, classes=[0, 1]
                        )
                    ),
                    "expectation_estimate": dense.expectation,
                    "variance_estimate": dense.variance,
                },
                {
                    **common,
                    "method": "shared_label_monte_carlo",
                    "timing_repeats": mc_timing["repeats"],
                    "timing_warmups": mc_timing["warmups"],
                    "runtime_median_seconds": mc_timing["median"],
                    "runtime_q1_seconds": mc_timing["q1"],
                    "runtime_q3_seconds": mc_timing["q3"],
                    "peak_traced_bytes": peak_bytes(
                        lambda: monte_carlo_batch_risk(
                            y_train,
                            neighbors,
                            probabilities,
                            repetitions=500,
                            classes=[0, 1],
                            seed=seed,
                        )
                    ),
                    "monte_carlo_repetitions": 500,
                    "expectation_estimate": float(np.mean(monte_carlo)),
                    "variance_estimate": float(np.var(monte_carlo)),
                    "expectation_absolute_error": abs(
                        float(np.mean(monte_carlo)) - overlap.expectation
                    ),
                    "variance_absolute_error": abs(
                        float(np.var(monte_carlo)) - overlap.variance
                    ),
                },
            ]
        )

    probability_rng = np.random.default_rng(20261009)
    for q in (40, 80, 160, 320):
        n = max(320, 4 * q)
        k = 5
        x_train = probability_rng.normal(size=(n, 6))
        x_query = probability_rng.normal(size=(q, 6))
        y_train = probability_rng.integers(0, 2, size=n)
        probabilities = np.full(n, 0.05)
        cache = build_neighbor_cache(x_train, x_query, k)

        overlap, overlap_timing = timed(
            lambda: batch_risk_moments(
                y_train, cache.indices, probabilities, classes=[0, 1]
            ),
            repeats=5,
        )
        dense, dense_timing = timed(
            lambda: dense_batch_risk_moments(
                y_train, cache.indices, probabilities, classes=[0, 1]
            ),
            repeats=3,
        )
        if not np.allclose(
            [overlap.expectation, overlap.variance],
            [dense.expectation, dense.variance],
            atol=1e-13,
        ):
            raise RuntimeError("E9 dense and overlap-graph batch moments disagree")
        monte_carlo, mc_timing = timed(
            lambda: monte_carlo_batch_risk(
                y_train,
                cache.indices,
                probabilities,
                repetitions=500,
                classes=[0, 1],
                seed=20261009 + q,
            ),
            repeats=5,
        )
        common = {
            "benchmark_family": "shared_probability",
            "n_train": n,
            "n_query": q,
            "k": k,
            "overlap_pair_count": overlap.overlap_pair_count,
            "total_query_pairs": overlap.total_query_pairs,
            "overlap_pair_fraction": (
                overlap.overlap_pair_count / overlap.total_query_pairs
                if overlap.total_query_pairs
                else 0.0
            ),
            "exact_expectation": overlap.expectation,
            "exact_variance": overlap.variance,
        }
        rows.extend(
            [
                {
                    **common,
                    "method": "overlap_graph_exact",
                    "timing_repeats": overlap_timing["repeats"],
                    "timing_warmups": overlap_timing["warmups"],
                    "runtime_median_seconds": overlap_timing["median"],
                    "runtime_q1_seconds": overlap_timing["q1"],
                    "runtime_q3_seconds": overlap_timing["q3"],
                    "peak_traced_bytes": peak_bytes(
                        lambda: batch_risk_moments(
                            y_train, cache.indices, probabilities, classes=[0, 1]
                        )
                    ),
                    "expectation_estimate": overlap.expectation,
                    "variance_estimate": overlap.variance,
                },
                {
                    **common,
                    "method": "dense_pairwise_exact",
                    "timing_repeats": dense_timing["repeats"],
                    "timing_warmups": dense_timing["warmups"],
                    "runtime_median_seconds": dense_timing["median"],
                    "runtime_q1_seconds": dense_timing["q1"],
                    "runtime_q3_seconds": dense_timing["q3"],
                    "peak_traced_bytes": peak_bytes(
                        lambda: dense_batch_risk_moments(
                            y_train, cache.indices, probabilities, classes=[0, 1]
                        )
                    ),
                    "expectation_estimate": dense.expectation,
                    "variance_estimate": dense.variance,
                },
                {
                    **common,
                    "method": "shared_label_monte_carlo",
                    "timing_repeats": mc_timing["repeats"],
                    "timing_warmups": mc_timing["warmups"],
                    "runtime_median_seconds": mc_timing["median"],
                    "runtime_q1_seconds": mc_timing["q1"],
                    "runtime_q3_seconds": mc_timing["q3"],
                    "peak_traced_bytes": peak_bytes(
                        lambda: monte_carlo_batch_risk(
                            y_train,
                            cache.indices,
                            probabilities,
                            repetitions=500,
                            classes=[0, 1],
                            seed=20261009 + q,
                        )
                    ),
                    "monte_carlo_repetitions": 500,
                    "expectation_estimate": float(np.mean(monte_carlo)),
                    "variance_estimate": float(np.var(monte_carlo)),
                    "expectation_absolute_error": abs(
                        float(np.mean(monte_carlo)) - overlap.expectation
                    ),
                    "variance_absolute_error": abs(
                        float(np.var(monte_carlo)) - overlap.variance
                    ),
                },
            ]
        )

    # Control 1: hold the training size at 1,280 while query count grows.
    # This separates query-count scaling from the variable-n geometry panel.
    fixed_train_rng = np.random.default_rng(20261010)
    fixed_train_n = 1280
    fixed_train_x = fixed_train_rng.normal(size=(fixed_train_n, 6))
    fixed_train_y = fixed_train_rng.integers(0, 2, size=fixed_train_n)
    fixed_train_probabilities = np.full(fixed_train_n, 0.05)
    for q in (40, 80, 160):
        fixed_query_x = fixed_train_rng.normal(size=(q, 6))
        fixed_cache = build_neighbor_cache(fixed_train_x, fixed_query_x, 5)
        append_probability_panel(
            benchmark_family="shared_probability_fixed_train",
            panel="fixed_train_query_scaling",
            n=fixed_train_n,
            q=q,
            k=5,
            neighbors=fixed_cache.indices,
            y_train=fixed_train_y,
            probabilities=fixed_train_probabilities,
            seed=20261010 + q,
            overlap_mode="random_geometry",
        )

    # Control 2: hold (n, q, k) fixed while changing neighborhood sharing.
    # The direct neighbor matrices isolate overlap density from feature geometry.
    overlap_n, overlap_q, overlap_k = 1280, 80, 5
    overlap_y = np.random.default_rng(20261011).integers(0, 2, size=overlap_n)
    overlap_probabilities = np.full(overlap_n, 0.05)
    overlap_modes = {
        "disjoint": np.arange(overlap_q * overlap_k, dtype=int).reshape(overlap_q, overlap_k),
        "chain": np.arange(overlap_k, dtype=int)[None, :]
        + np.arange(overlap_q, dtype=int)[:, None],
        "block_shared": np.empty((overlap_q, overlap_k), dtype=int),
        "fully_shared": np.tile(np.arange(overlap_k, dtype=int), (overlap_q, 1)),
    }
    block_neighbors = overlap_modes["block_shared"]
    for query_idx in range(overlap_q):
        block = query_idx // 8
        start = overlap_q + block * (overlap_k - 1)
        block_neighbors[query_idx] = np.array(
            [block, start, start + 1, start + 2, start + 3], dtype=int
        )
    for mode, neighbors in overlap_modes.items():
        append_probability_panel(
            benchmark_family="shared_probability_overlap_sweep",
            panel="fixed_query_overlap_sweep",
            n=overlap_n,
            q=overlap_q,
            k=overlap_k,
            neighbors=neighbors,
            y_train=overlap_y,
            probabilities=overlap_probabilities,
            seed=20261020 + len(mode),
            overlap_mode=mode,
            timing_repeats=2,
            dense_repeats=1,
            mc_repeats=2,
        )
    save_csv(PROCESSED_DIR / "e9_runtime.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e0"],
    )
    args = parser.parse_args()
    ensure_dirs()
    dispatch = {
        "e0": run_e0,
        "e1": run_e1,
        "e2": run_e2,
        "e3": run_e3,
        "e4": run_e4,
        "e5": run_e5,
        "e6": run_e6,
        "e7": run_e7,
        "e8": run_e8,
        "e9": run_e9,
    }
    failures = []
    for name in args.experiments:
        try:
            dispatch[name]()
            print(f"completed {name}", flush=True)
        except Exception as exc:  # preserve failure evidence and continue only across requested jobs
            failures.append({"experiment": name, "error": repr(exc)})
            print(f"failed {name}: {exc!r}", file=sys.stderr, flush=True)
    (RAW_DIR / "run_metadata.json").write_text(
        json.dumps(
            {
                "seed_protocol": list(SEEDS),
                "split": {"train": 0.50, "validation": 0.15, "test": 0.35},
                "datasets": ORIGINAL_23,
                "python": sys.version,
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
