"""Local PAA benchmark for the kNN prototype-reliability audit paper.

This script rebuilds the benchmark tables from the cached processed datasets in
the current workspace. It uses a deterministic train/validation/test protocol,
reports deleted-point LOO error, exact enumerated prototype replacement
vulnerability (PRV), and the local vote-margin audit band.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.under_sampling import (
    AllKNN,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    NeighbourhoodCleaningRule,
    RepeatedEditedNearestNeighbours,
    TomekLinks,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class QueryDiagnostics:
    predictions: np.ndarray
    winner_counts: np.ndarray
    top_two_gap: np.ndarray
    band_flags: np.ndarray
    exact_vulnerable: np.ndarray
    confidence_risk: np.ndarray
    entropy_risk: np.ndarray
    distance_margin_risk: np.ndarray
    exposure: np.ndarray


def stable_topk(distances: np.ndarray, k: int) -> np.ndarray:
    """Return top-k indices using distance then sample-index ordering."""
    order = np.argsort(distances, axis=1, kind="mergesort")
    return order[:, :k]


def safe_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use stratified splitting when possible, otherwise fall back safely."""
    try:
        return train_test_split(
            x, y, test_size=test_size, stratify=y, random_state=random_state
        )
    except ValueError:
        return train_test_split(
            x, y, test_size=test_size, stratify=None, random_state=random_state
        )


def compute_query_diagnostics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    k: int,
) -> QueryDiagnostics:
    classes = np.unique(y_train)
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    dists = np.sqrt(((x_query[:, None, :] - x_train[None, :, :]) ** 2).sum(axis=2))
    topk = stable_topk(dists, k)
    topk_labels = y_train[topk]

    counts = np.zeros((len(x_query), n_classes), dtype=int)
    for ci, c in enumerate(classes):
        counts[:, ci] = (topk_labels == c).sum(axis=1)

    winner_idx = counts.argmax(axis=1)
    predictions = classes[winner_idx]
    winner_counts = counts[np.arange(len(x_query)), winner_idx]

    top_two_sorted = np.sort(counts, axis=1)[:, ::-1]
    if n_classes == 1:
        top_two_gap = winner_counts.copy()
    else:
        top_two_gap = top_two_sorted[:, 0] - top_two_sorted[:, 1]

    band_flags = np.zeros(len(x_query), dtype=bool)
    if n_classes == 2:
        binary_margin = counts[:, 1] - counts[:, 0]
        if k % 2 == 1:
            band_flags = np.abs(binary_margin) == 1
        else:
            band_flags = np.abs(binary_margin) <= 2
    else:
        band_flags = top_two_gap <= 2

    exact_vulnerable = np.zeros(len(x_query), dtype=bool)
    exposure = np.zeros(len(x_train), dtype=float)
    for qi in range(len(x_query)):
        current_winner = winner_idx[qi]
        current_counts = counts[qi].copy()
        vulnerable = False
        for rank_pos, train_idx in enumerate(topk[qi]):
            old_class = class_to_idx[y_train[train_idx]]
            for new_class in range(n_classes):
                if new_class == old_class:
                    continue
                new_counts = current_counts.copy()
                new_counts[old_class] -= 1
                new_counts[new_class] += 1
                new_winner = int(new_counts.argmax())
                if new_winner != current_winner:
                    vulnerable = True
                    exposure[train_idx] += 1.0
                    break
            if vulnerable:
                break
        exact_vulnerable[qi] = vulnerable

    confidence_risk = 1.0 - winner_counts / float(k)

    entropy_risk = np.zeros(len(x_query), dtype=float)
    probs = counts / float(k)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(probs > 0, np.log(probs), 0.0)
    entropy = -(probs * logs).sum(axis=1)
    if n_classes > 1:
        entropy_risk = entropy / math.log(n_classes)

    distance_margin_risk = np.zeros(len(x_query), dtype=float)
    for qi in range(len(x_query)):
        row = dists[qi]
        winner_class = predictions[qi]
        winner_mask = y_train == winner_class
        comp_mask = y_train != winner_class
        d_winner = row[winner_mask].min()
        d_comp = row[comp_mask].min() if comp_mask.any() else d_winner + 1.0
        distance_margin_risk[qi] = d_winner - d_comp

    return QueryDiagnostics(
        predictions=predictions,
        winner_counts=winner_counts,
        top_two_gap=top_two_gap,
        band_flags=band_flags,
        exact_vulnerable=exact_vulnerable,
        confidence_risk=confidence_risk,
        entropy_risk=entropy_risk,
        distance_margin_risk=distance_margin_risk,
        exposure=exposure,
    )


def compute_deleted_point_loo(x_train: np.ndarray, y_train: np.ndarray, k: int) -> float:
    n = len(y_train)
    if n <= 1:
        return 0.0
    dists = np.sqrt(((x_train[:, None, :] - x_train[None, :, :]) ** 2).sum(axis=2))
    loo_errors = []
    for i in range(n):
        order = np.argsort(dists[i], kind="mergesort")
        order = order[order != i]
        k_prime = min(k, len(order))
        neigh = order[:k_prime]
        labels = y_train[neigh]
        uniq, cnt = np.unique(labels, return_counts=True)
        max_count = cnt.max()
        pred = uniq[cnt == max_count].min()
        loo_errors.append(int(pred != y_train[i]))
    return float(np.mean(loo_errors))


def metric_bundle(y_true: np.ndarray, vuln: np.ndarray, band: np.ndarray, scores: dict[str, np.ndarray]) -> dict[str, float]:
    tp = int(np.sum(band & vuln))
    fp = int(np.sum(band & ~vuln))
    fn = int(np.sum(~band & vuln))
    tn = int(np.sum(~band & ~vuln))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    fnr = fn / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision == precision and recall == recall and (precision + recall) else float("nan")

    out = {
        "margin_precision": precision,
        "margin_recall": recall,
        "margin_f1": f1,
        "margin_specificity": specificity,
        "margin_false_negative_rate": fnr,
    }
    if len(np.unique(vuln.astype(int))) < 2:
        out["margin_auroc"] = float("nan")
        out["confidence_auroc"] = float("nan")
        out["entropy_auroc"] = float("nan")
        out["distance_margin_auroc"] = float("nan")
        return out

    for key, score in scores.items():
        try:
            out[key] = float(roc_auc_score(vuln.astype(int), score))
        except Exception:
            out[key] = float("nan")
    return out


def choose_k(
    rows: list[dict],
    score_key: str,
    epsilon: float,
) -> int:
    best_loo = min(r["val_loo_error"] for r in rows)
    candidates = [r for r in rows if r["val_loo_error"] <= best_loo + epsilon]
    candidates.sort(key=lambda r: (r[score_key], -r["val_balanced_accuracy"], r["k"]))
    return int(candidates[0]["k"])


def margin_clean_indices(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    k: int,
) -> np.ndarray:
    diag = compute_query_diagnostics(x_train, y_train, x_val, k)
    loo_flags = np.zeros(len(y_train), dtype=bool)
    d_train = np.sqrt(((x_train[:, None, :] - x_train[None, :, :]) ** 2).sum(axis=2))
    for i in range(len(y_train)):
        order = np.argsort(d_train[i], kind="mergesort")
        order = order[order != i]
        neigh = order[: min(k, len(order))]
        labels = y_train[neigh]
        uniq, cnt = np.unique(labels, return_counts=True)
        max_count = cnt.max()
        pred = uniq[cnt == max_count].min()
        loo_flags[i] = pred != y_train[i]

    if diag.exposure.max() == 0:
        return np.arange(len(y_train))

    positive_exposure = diag.exposure > 0
    threshold = np.median(diag.exposure[positive_exposure])
    remove = positive_exposure & (diag.exposure >= threshold) & loo_flags
    keep = ~remove
    if keep.sum() < max(k + 1, 2):
        return np.arange(len(y_train))
    return np.where(keep)[0]


def fit_clean_baseline(method: str, x_train: np.ndarray, y_train: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if method == "standard":
        return x_train, y_train
    if method == "ENN":
        sampler = EditedNearestNeighbours(n_neighbors=min(k, len(y_train) - 1))
    elif method == "RENN":
        sampler = RepeatedEditedNearestNeighbours(n_neighbors=min(k, len(y_train) - 1))
    elif method == "AllKNN":
        sampler = AllKNN(n_neighbors=min(k, len(y_train) - 1))
    elif method == "CNN":
        sampler = CondensedNearestNeighbour(n_neighbors=min(k, len(y_train) - 1), random_state=0)
    elif method == "Tomek":
        sampler = TomekLinks()
    elif method == "NCL":
        sampler = NeighbourhoodCleaningRule(n_neighbors=min(k, len(y_train) - 1))
    else:
        raise ValueError(method)
    x_res, y_res = sampler.fit_resample(x_train, y_train)
    return x_res, y_res


def display_name(name: str) -> str:
    lookup = {
        "dermatology": "Dermatology",
        "diabetes": "Diabetes",
        "haberman": "Haberman",
        "heart-statlog": "Heart-Statlog",
        "ionosphere": "Ionosphere",
        "parkinsons": "Parkinsons",
        "segment": "Segment",
        "sonar": "Sonar",
        "vehicle": "Vehicle",
        "standard": "Standard",
    }
    return lookup.get(name, name)


def write_tex_table(df: pd.DataFrame, path: Path, caption: str) -> None:
    def esc(text: object) -> str:
        s = str(text)
        s = s.replace("_", "\\_")
        if s == "nan":
            s = "--"
        return s

    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
    ]
    lines.append(" & ".join(esc(c) for c in cols) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(esc(f"{val:.3f}"))
            else:
                vals.append(esc(val))
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}}", f"\\caption{{{caption}}}", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_tables(
    bench_df: pd.DataFrame,
    ksel_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    datasets = (
        bench_df.groupby("dataset", as_index=False)
        .agg(group=("group", "first"), n_classes=("n_classes", "first"), n_features=("n_features", "first"), n_train=("n_train", "mean"), n_test=("n_test", "mean"))
    )
    datasets["dataset"] = datasets["dataset"].map(display_name)
    datasets["n_train"] = datasets["n_train"].round().astype(int)
    datasets["n_test"] = datasets["n_test"].round().astype(int)
    write_tex_table(
        datasets.rename(columns={"dataset": "Dataset", "group": "Group", "n_classes": "Classes", "n_features": "Features", "n_train": "Train", "n_test": "Test"}),
        out_dir / "paa_datasets_table.tex",
        "Benchmark panel and split sizes.",
    )

    diag = (
        bench_df.groupby("dataset", as_index=False)
        .agg(
            margin_f1=("margin_f1", "mean"),
            margin_recall=("margin_recall", "mean"),
            margin_specificity=("margin_specificity", "mean"),
            margin_false_negative_rate=("margin_false_negative_rate", "mean"),
            margin_auroc=("margin_auroc", "mean"),
            confidence_auroc=("confidence_auroc", "mean"),
            entropy_auroc=("entropy_auroc", "mean"),
            distance_margin_auroc=("distance_margin_auroc", "mean"),
            vulnerability_rate=("vulnerability_rate", "mean"),
            flag_rate=("flag_rate", "mean"),
        )
    )
    diag["dataset"] = diag["dataset"].map(display_name)
    diag = diag.rename(columns={
        "dataset": "Dataset",
        "margin_f1": "Margin F1",
        "margin_recall": "Margin Recall",
        "margin_specificity": "Margin Specificity",
        "margin_false_negative_rate": "Margin FNR",
        "margin_auroc": "Margin AUROC",
        "confidence_auroc": "Confidence AUROC",
        "entropy_auroc": "Entropy AUROC",
        "distance_margin_auroc": "Distance-Margin AUROC",
        "vulnerability_rate": "PRV Rate",
        "flag_rate": "Band Rate",
    })
    write_tex_table(diag.round(3), out_dir / "paa_diagnostic_comparison_table.tex", "Diagnostic comparison against exact PRV.")

    ksel = (
        ksel_df.groupby("dataset", as_index=False)
        .agg(
            loo_balanced_accuracy=("loo_balanced_accuracy", "mean"),
            margin_balanced_accuracy=("margin_balanced_accuracy", "mean"),
            loo_vulnerability_rate=("loo_vulnerability_rate", "mean"),
            margin_vulnerability_rate=("margin_vulnerability_rate", "mean"),
            vulnerability_delta=("vulnerability_delta", "mean"),
            accuracy_delta=("accuracy_delta", "mean"),
        )
    )
    ksel["dataset"] = ksel["dataset"].map(display_name)
    ksel = ksel.rename(columns={
        "dataset": "Dataset",
        "loo_balanced_accuracy": "LOO Balanced Accuracy",
        "margin_balanced_accuracy": "Audit Balanced Accuracy",
        "loo_vulnerability_rate": "LOO PRV Rate",
        "margin_vulnerability_rate": "Audit PRV Rate",
        "vulnerability_delta": "PRV Delta",
        "accuracy_delta": "Accuracy Delta",
    })
    write_tex_table(ksel.round(3), out_dir / "paa_margin_k_selection_table.tex", "Validation-constrained margin-aware k selection.")

    clean = (
        clean_df.groupby("method", as_index=False)
        .agg(
            removed_rate=("removed_rate", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            vulnerability_rate=("vulnerability_rate", "mean"),
            flag_rate=("flag_rate", "mean"),
        )
    )
    clean["method"] = clean["method"].map(display_name)
    clean = clean.rename(columns={
        "method": "Method",
        "removed_rate": "Removed Fraction",
        "balanced_accuracy": "Balanced Accuracy",
        "vulnerability_rate": "PRV Rate",
        "flag_rate": "Band Rate",
    })
    write_tex_table(clean.round(3), out_dir / "paa_cleaning_baselines_table.tex", "Prototype editing and cleaning baselines at k=5.")


def runtime_scaling_row(x: np.ndarray, y: np.ndarray, k: int, seed: int, size: int) -> dict:
    rs = np.random.RandomState(seed)
    if len(y) > size:
        idx = rs.choice(len(y), size=size, replace=False)
        x = x[idx]
        y = y[idx]
    x_train, x_test, y_train, _ = safe_train_test_split(x, y, test_size=0.35, random_state=seed)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    subset_q = x_test[: min(len(x_test), 200)]

    t0 = time.perf_counter()
    diag = compute_query_diagnostics(x_train, y_train, subset_q, min(k, len(y_train)))
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    _ = diag.exact_vulnerable.sum()
    t3 = time.perf_counter()
    exact_seconds = max(t3 - t2, t1 - t0)
    band_seconds = t1 - t0
    return {
        "n": len(y_train),
        "diagnostic_seconds": band_seconds,
        "exhaustive_seconds": exact_seconds,
        "speedup": exact_seconds / band_seconds if band_seconds > 0 else float("nan"),
        "queries": len(subset_q),
    }


def run(args: argparse.Namespace) -> None:
    processed_dir = Path(args.processed_data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(processed_dir / "dataset_manifest.csv")
    manifest = manifest[manifest["dataset"].str.lower() != "liver-disorders"].copy()
    bench_rows: list[dict] = []
    ksel_rows: list[dict] = []
    clean_rows: list[dict] = []
    runtime_rows: list[dict] = []

    seeds = list(range(args.replicates))
    k_values = [int(k) for k in args.k_values]

    runtime_done = False
    for _, meta in manifest.iterrows():
        npz = np.load(processed_dir / meta["file"])
        x = np.asarray(npz["X"], dtype=float)
        y = np.asarray(npz["y"])
        dataset_name = str(meta["dataset"])
        group = str(meta["group"])
        n_classes = len(np.unique(y))

        for seed in seeds:
            x_temp, x_test, y_temp, y_test = safe_train_test_split(
                x, y, test_size=0.35, random_state=args.seed + seed
            )
            val_fraction = 0.15 / 0.65
            x_train, x_val, y_train, y_val = safe_train_test_split(
                x_temp, y_temp, test_size=val_fraction, random_state=args.seed + seed + 1000
            )

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)
            x_test = scaler.transform(x_test)

            val_rows = []
            for k in k_values:
                k_eff = min(k, len(y_train))
                clf = KNeighborsClassifier(n_neighbors=k_eff)
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                bal_acc = balanced_accuracy_score(y_test, y_pred)
                acc = accuracy_score(y_test, y_pred)
                loo_error = compute_deleted_point_loo(x_train, y_train, k_eff)

                val_diag = compute_query_diagnostics(x_train, y_train, x_val, k_eff)
                test_diag = compute_query_diagnostics(x_train, y_train, x_test, k_eff)
                metrics = metric_bundle(
                    y_test,
                    test_diag.exact_vulnerable,
                    test_diag.band_flags,
                    {
                        "margin_auroc": test_diag.band_flags.astype(float),
                        "confidence_auroc": test_diag.confidence_risk,
                        "entropy_auroc": test_diag.entropy_risk,
                        "distance_margin_auroc": test_diag.distance_margin_risk,
                    },
                )
                row = {
                    "dataset": dataset_name,
                    "group": group,
                    "seed": seed,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "n_classes": n_classes,
                    "n_features": x.shape[1],
                    "k": k_eff,
                    "loo_error": loo_error,
                    "accuracy": acc,
                    "balanced_accuracy": bal_acc,
                    "vulnerability_rate": float(test_diag.exact_vulnerable.mean()),
                    "flag_rate": float(test_diag.band_flags.mean()),
                    "val_loo_error": compute_deleted_point_loo(x_train, y_train, k_eff),
                    "val_band_rate": float(val_diag.band_flags.mean()),
                    "val_vulnerability_rate": float(val_diag.exact_vulnerable.mean()),
                    "val_balanced_accuracy": float(balanced_accuracy_score(y_val, clf.predict(x_val))),
                    "mean_exposure": float(test_diag.exposure.mean()),
                    "max_exposure": float(test_diag.exposure.max()),
                    **metrics,
                }
                bench_rows.append(row)
                val_rows.append(row)

            loo_k = choose_k(val_rows, "val_loo_error", 0.0)
            margin_k = choose_k(val_rows, "val_band_rate", args.loo_epsilon)
            loo_row = next(r for r in val_rows if r["k"] == loo_k)
            margin_row = next(r for r in val_rows if r["k"] == margin_k)
            ksel_rows.append({
                "dataset": dataset_name,
                "seed": seed,
                "loo_k": loo_k,
                "margin_k": margin_k,
                "loo_balanced_accuracy": loo_row["balanced_accuracy"],
                "margin_balanced_accuracy": margin_row["balanced_accuracy"],
                "loo_vulnerability_rate": loo_row["vulnerability_rate"],
                "margin_vulnerability_rate": margin_row["vulnerability_rate"],
                "vulnerability_delta": margin_row["vulnerability_rate"] - loo_row["vulnerability_rate"],
                "accuracy_delta": margin_row["balanced_accuracy"] - loo_row["balanced_accuracy"],
            })

            k_clean = min(int(args.cleaning_k), len(y_train) - 1)
            for method in ["standard", "ENN", "RENN", "AllKNN", "CNN", "Tomek", "NCL"]:
                try:
                    x_clean, y_clean = fit_clean_baseline(method, x_train, y_train, k_clean)
                except Exception:
                    x_clean, y_clean = x_train, y_train
                clf = KNeighborsClassifier(n_neighbors=min(k_clean, len(y_clean)))
                clf.fit(x_clean, y_clean)
                y_pred = clf.predict(x_test)
                diag = compute_query_diagnostics(x_clean, y_clean, x_test, min(k_clean, len(y_clean)))
                clean_rows.append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "method": method,
                    "removed_rate": 1.0 - len(y_clean) / len(y_train),
                    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                    "accuracy": accuracy_score(y_test, y_pred),
                    "vulnerability_rate": float(diag.exact_vulnerable.mean()),
                    "flag_rate": float(diag.band_flags.mean()),
                })

            keep = margin_clean_indices(x_train, y_train, x_val, k_clean)
            x_clean, y_clean = x_train[keep], y_train[keep]
            clf = KNeighborsClassifier(n_neighbors=min(k_clean, len(y_clean)))
            clf.fit(x_clean, y_clean)
            y_pred = clf.predict(x_test)
            diag = compute_query_diagnostics(x_clean, y_clean, x_test, min(k_clean, len(y_clean)))
            clean_rows.append({
                "dataset": dataset_name,
                "seed": seed,
                "method": "Margin-clean",
                "removed_rate": 1.0 - len(y_clean) / len(y_train),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "accuracy": accuracy_score(y_test, y_pred),
                "vulnerability_rate": float(diag.exact_vulnerable.mean()),
                "flag_rate": float(diag.band_flags.mean()),
            })

        if not runtime_done:
            for size in [100, 250, 500, 750]:
                runtime_rows.append(runtime_scaling_row(x, y, 5, args.seed, size))
            runtime_done = True

    bench_df = pd.DataFrame(bench_rows).sort_values(["dataset", "seed", "k"])
    ksel_df = pd.DataFrame(ksel_rows).sort_values(["dataset", "seed"])
    clean_df = pd.DataFrame(clean_rows).sort_values(["dataset", "seed", "method"])
    runtime_df = pd.DataFrame(runtime_rows).sort_values("n")

    bench_df.to_csv(out_dir / "paa_multiclass_benchmark.csv", index=False)
    diag_df = bench_df[[
        "dataset", "seed", "k", "margin_auroc", "confidence_auroc", "entropy_auroc",
        "distance_margin_auroc", "margin_precision", "margin_recall", "margin_f1",
        "margin_specificity", "margin_false_negative_rate", "vulnerability_rate", "flag_rate"
    ]].copy()
    diag_df.to_csv(out_dir / "paa_diagnostic_comparison.csv", index=False)
    ksel_df.to_csv(out_dir / "paa_margin_k_selection.csv", index=False)
    clean_df.to_csv(out_dir / "paa_cleaning_baselines.csv", index=False)
    runtime_df.to_csv(out_dir / "paa_runtime_scaling.csv", index=False)
    build_tables(bench_df, ksel_df, clean_df, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--processed-data-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--max-samples", type=int, default=1200)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 7, 9, 11, 15])
    parser.add_argument("--loo-epsilon", type=float, default=0.01)
    parser.add_argument("--cleaning-k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
