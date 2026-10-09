"""Build the six revision figures from the processed experiment tables."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from knn_reliability.knn import build_neighbor_cache, predict_from_neighbors  # noqa: E402
TABLE_DIR = ROOT / "results" / "processed"
FIG_DIR = ROOT / "figures" / "export"
DATA_DIR = ROOT / "figures" / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "figure.dpi": 150})


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.03, f"({label})", transform=ax.transAxes,
            va="bottom", ha="left", fontsize=9, fontweight="bold")


def read_csv(name: str) -> list[dict[str, str]]:
    with (TABLE_DIR / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_panel(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with (DATA_DIR / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig1_framework() -> None:
    """Show correct LOO decisions alongside a retained-prototype flip."""

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    colors = {0: "#377eb8", 1: "#d95f02"}
    points = np.array([
        [-2.00, -0.15], [-1.80, 0.05], [-1.60, -0.05],
        [0.20, 0.05], [0.40, -0.05], [0.60, 0.15],
    ])
    labels = np.array([0, 0, 0, 1, 1, 1])
    query = np.array([[-0.90, 0.00]])
    k = 3
    query_cache = build_neighbor_cache(points, query, k)
    base_prediction, base_counts, _ = predict_from_neighbors(
        labels, query_cache.indices, classes=[0, 1]
    )
    changed_labels = labels.copy()
    changed_labels[2] = 1
    relabel_prediction, relabel_counts, _ = predict_from_neighbors(
        changed_labels, build_neighbor_cache(points, query, k).indices, classes=[0, 1]
    )
    loo_predictions = []
    for index in range(len(points)):
        loo_points = np.delete(points, index, axis=0)
        loo_labels = np.delete(labels, index)
        prediction, _, _ = predict_from_neighbors(
            loo_labels,
            build_neighbor_cache(loo_points, points[index:index + 1], k).indices,
            classes=[0, 1],
        )
        loo_predictions.append(int(prediction[0]))
    loo_correct = int(np.sum(np.asarray(loo_predictions) == labels))
    loo_neighbor_cache = build_neighbor_cache(
        np.delete(points, 2, axis=0), query, k
    )
    _, loo_counts, _ = predict_from_neighbors(
        np.delete(labels, 2), loo_neighbor_cache.indices, classes=[0, 1]
    )
    assert int(base_prediction[0]) == 0
    assert loo_correct == len(points)
    assert int(relabel_prediction[0]) == 1
    axes[0].scatter(points[:, 0], points[:, 1], s=45,
                    c=[colors[int(y)] for y in labels], edgecolor="white", linewidth=0.7)
    axes[0].scatter(points[2, 0], points[2, 1], s=75, facecolors="none",
                    edgecolors="#777777", linewidth=1.2, zorder=3)
    axes[0].scatter(*query[0], marker="*", s=120, c="black", zorder=4)
    for neighbor_index in query_cache.indices[0]:
        axes[0].plot(
            [query[0, 0], points[neighbor_index, 0]],
            [query[0, 1], points[neighbor_index, 1]],
            color="#999999", lw=0.8, ls="--", zorder=0,
        )
    for index, (x, y) in enumerate(points, start=1):
        axes[0].text(x + 0.04, y + 0.04, f"$z_{index}$", fontsize=8)
    axes[0].set_title(f"All training-point LOO predictions correct ({loo_correct}/{len(points)})")
    axes[0].text(
        0.5, 0.05,
        f"independent query votes {base_counts[0].tolist()} $\\rightarrow$ class {int(base_prediction[0])}",
        transform=axes[0].transAxes, ha="center", va="bottom", fontsize=8,
    )
    panel_label(axes[0], "a")

    axes[1].scatter(points[:, 0], points[:, 1], s=45,
                    c=[colors[int(y)] for y in changed_labels], edgecolor="white", linewidth=0.7)
    axes[1].scatter(*query[0], marker="*", s=120, c="black", zorder=4)
    for neighbor_index in query_cache.indices[0]:
        axes[1].plot(
            [query[0, 0], points[neighbor_index, 0]],
            [query[0, 1], points[neighbor_index, 1]],
            color="#999999", lw=0.8, ls="--", zorder=0,
        )
    for index, (x, y) in enumerate(points, start=1):
        axes[1].text(x + 0.04, y + 0.04, f"$z_{index}$", fontsize=8)
    axes[1].annotate("$z_3$: 0 $\\to$ 1", xy=points[2], xytext=(-1.48, 0.34),
                     arrowprops={"arrowstyle": "->", "color": "#984ea3"}, color="#984ea3", fontsize=8)
    axes[1].set_title("One retained-label replacement flips the query")
    axes[1].text(
        0.5, 0.05,
        f"votes {base_counts[0].tolist()} $\\rightarrow$ {relabel_counts[0].tolist()}; "
        f"class {int(base_prediction[0])} $\\rightarrow$ {int(relabel_prediction[0])}",
        transform=axes[1].transAxes, ha="center", va="bottom", fontsize=8,
    )
    panel_label(axes[1], "b")
    xmin = float(points[:, 0].min() - 0.35)
    xmax = float(points[:, 0].max() + 0.35)
    ymin = float(min(points[:, 1].min(), query[:, 1].min()) - 0.35)
    ymax = float(max(points[:, 1].max(), query[:, 1].max()) + 0.35)
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#bbbbbb")
    write_panel("fig1_revision_framework_panels.csv", [
        {"panel": "a", "operation": "LOO at each training location", "k": k,
         "base_votes": ":".join(map(str, base_counts[0])), "changed_votes": "",
         "base_prediction": int(base_prediction[0]), "changed_prediction": "",
         "loo_correct": loo_correct, "loo_total": len(points), "query_type": "independent"},
        {"panel": "b", "operation": "retained prototype relabeling z3", "k": k,
         "base_votes": ":".join(map(str, base_counts[0])), "changed_votes": ":".join(map(str, relabel_counts[0])),
         "base_prediction": int(base_prediction[0]), "changed_prediction": int(relabel_prediction[0]),
         "loo_correct": loo_correct, "loo_total": len(points), "query_type": "independent"},
    ])
    save(fig, "fig1_revision_framework")


def fig2_theory() -> None:
    e1 = read_csv("e1_vote_configurations.csv")
    e4 = read_csv("e4_distribution_stability.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0))

    gaps, exact, band = [], [], []
    for row in e1:
        counts = [int(x) for x in row["counts"].split(":")]
        ordered = sorted(counts, reverse=True)
        gaps.append(ordered[0] - ordered[1])
        exact.append(int(row["exact_vulnerable"]))
        band.append(int(row["band_flag"]))
    axes[0].scatter(gaps, band, s=3, alpha=0.18, label="screening band")
    axes[0].scatter(gaps, exact, s=3, alpha=0.18, label="exact vulnerable")
    axes[0].set(xlabel="top-two vote gap", ylabel="indicator")
    panel_label(axes[0], "a")
    axes[0].legend(frameon=False, markerscale=3, fontsize=7)

    colors = {1: "#b33", 3: "#377eb8", 5: "#4daf4a", 7: "#984ea3", 9: "#ff7f00", 11: "#555"}
    for k, color in colors.items():
        subset = [
            r for r in e4
            if r["model"] == "random_training_labels_homogeneous"
            and int(r["k"]) == k
        ]
        if subset:
            axes[1].plot(
                [float(r["class_probability"]) for r in subset],
                [float(r["exact_boundary_probability"]) for r in subset],
                         "o-", label=f"k={k}", color=color, ms=3)
    axes[1].set(xlabel="local class probability", ylabel="boundary probability")
    panel_label(axes[1], "b")
    axes[1].legend(frameon=False, ncol=2, fontsize=6.5)

    theta = np.linspace(0, 2 * np.pi, 200)
    axes[2].plot(np.cos(theta), np.sin(theta), color="#777", lw=0.8)
    k = 5
    for j in range(k):
        angle = 2 * np.pi * j / k
        color = "#377eb8" if j < 3 else "#d95f02"
        axes[2].scatter([np.cos(angle)], [np.sin(angle)], marker="o", s=28, color=color, zorder=3)
        for t in range(1, k + 1):
            radius = 1 + 0.01 * t / k
            axes[2].scatter([radius * np.cos(angle)], [radius * np.sin(angle)], marker=".", s=12, color=color, alpha=0.65)
    axes[2].scatter([0], [0], marker="*", color="black", s=65, zorder=4)
    axes[2].annotate("query region", xy=(0, 0), xytext=(-0.25, -1.35), fontsize=7,
                     arrowprops={"arrowstyle": "->", "color": "#444"})
    axes[2].set_aspect("equal")
    axes[2].set(xlabel="feature 1", ylabel="feature 2")
    axes[2].text(-0.08, 1.03, "(c)", transform=axes[2].transAxes,
                 va="bottom", ha="left", fontsize=9, fontweight="bold")
    axes[2].annotate("same-label supports", xy=(1.01, 0.0), xytext=(1.32, 0.6),
                     fontsize=6.5, ha="right",
                     arrowprops={"arrowstyle": "->", "color": "#444"})
    axes[2].set_xlim(-1.35, 1.35)
    axes[2].set_ylim(-1.45, 1.25)
    write_panel("fig2_theory_panels.csv", [
        {"panel": "a", "n": len(e1), "exact_reference": "zero mismatches", "band_false_negatives": "zero"},
        {"panel": "b", "rows": len(e4), "model": "random training labels"},
        {"panel": "c", "k": 5, "epsilon": 0.01, "supports_per_representative": 5},
    ])
    save(fig, "fig2_exact_theory")


def fig3_probability() -> None:
    rows = read_csv("e3_probability_risk.csv")
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0))
    independent_rows = [r for r in rows if r["assumption"] == "independent_flips"]
    exact = np.array([float(r["exact_expectation_independent_model"]) for r in independent_rows])
    mc = np.array([float(r["monte_carlo_mean"]) for r in independent_rows])
    first = np.array([float(r["first_order_approximation"]) for r in independent_rows])
    variance = np.array([float(r["variance_exact_shared_flips"]) for r in independent_rows])
    bound = np.array([float(r["efron_stein_variance_bound"]) for r in independent_rows])
    axes[0, 0].scatter(exact, mc, c=[int(r["k"]) for r in independent_rows], cmap="viridis", s=18, alpha=0.75)
    lim = max(exact.max(), mc.max()) * 1.05
    axes[0, 0].plot([0, lim], [0, lim], "k--", lw=0.8)
    axes[0, 0].set(xlabel="exact expectation", ylabel="Monte Carlo mean")
    panel_label(axes[0, 0], "a")
    axes[0, 1].scatter(exact, first, c=[float(r["mean_flip_probability"]) for r in independent_rows], cmap="plasma", s=18, alpha=0.75)
    axes[0, 1].plot([0, lim], [0, lim], "k--", lw=0.8)
    axes[0, 1].set(xlabel="exact expectation", ylabel="first-order approximation")
    panel_label(axes[0, 1], "b")
    axes[1, 0].scatter(variance, bound, c=[int(r["k"]) for r in independent_rows], cmap="viridis", s=18, alpha=0.75)
    lim_v = max(bound.max(), variance.max()) * 1.05
    axes[1, 0].plot([0, lim_v], [0, lim_v], "k--", lw=0.8)
    axes[1, 0].set(xlabel="exact batch variance", ylabel="Efron--Stein bound")
    panel_label(axes[1, 0], "c")
    by_k = {}
    for r in independent_rows:
        by_k.setdefault(int(r["k"]), []).append(
            abs(float(r["exact_expectation_independent_model"]) - float(r["monte_carlo_mean"]))
        )
    axes[1, 1].boxplot([by_k[k] for k in sorted(by_k)], tick_labels=[str(k) for k in sorted(by_k)], showfliers=False)
    axes[1, 1].set(xlabel="k", ylabel="|exact - MC mean|")
    panel_label(axes[1, 1], "d")
    fig.colorbar(axes[0, 0].collections[0], ax=axes[0, 0], label="k", fraction=0.046, pad=0.04)
    write_panel("fig4_probability_panels.csv", [{"panel": "all", **r} for r in rows])
    save(fig, "fig4_probability_stability")


def fig4_influence() -> None:
    rows = [r for r in read_csv("e2_deterministic_influence.csv") if r["split"] == "test"]
    metrics = [
        ("point_prv", "$R_{\\mathrm{point}}$", "point vulnerability"),
        ("r1", "$R_1$", "global single-relabel risk"),
        ("influence_concentration", "influence concentration", "max prototype share"),
        ("loo_error", "LOO error", "deleted-point recovery error"),
    ]
    k_values = sorted({int(row["k"]) for row in rows})
    summary_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.7), sharex=True)
    for axis, (field, ylabel, description), label in zip(
        axes.ravel(), metrics, ("a", "b", "c", "d")
    ):
        means, errors = [], []
        for k_value in k_values:
            grouped = {}
            for row in rows:
                if int(row["k"]) == k_value:
                    grouped.setdefault(row["dataset"], []).append(float(row[field]))
            dataset_means = np.asarray([np.mean(values) for values in grouped.values()])
            value = float(np.mean(dataset_means))
            error = float(1.96 * np.std(dataset_means, ddof=1) / np.sqrt(len(dataset_means)))
            means.append(value)
            errors.append(error)
            summary_rows.append({
                "panel": label, "metric": field, "k": k_value,
                "dataset_count": len(dataset_means), "mean": value,
                "dataset_sd": float(np.std(dataset_means, ddof=1)),
                "ci95_halfwidth": error, "description": description,
            })
        axis.errorbar(k_values, means, yerr=errors, fmt="o-", color="#377eb8",
                      capsize=3, lw=1.2, ms=4)
        axis.set(xlabel="k", ylabel=ylabel, xticks=k_values)
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        panel_label(axis, label)
    write_panel("fig3_deterministic_influence_panels.csv", summary_rows)
    save(fig, "fig3_deterministic_influence")


def fig5_audit() -> None:
    rows = read_csv("e5_label_audit.csv")
    methods = ["random", "neighborhood_exposure", "exact_decisive_influence", "loo_error", "combined_influence_loo"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1))
    selected = [np.mean([float(r["selected_error_rate"]) for r in rows if r["method"] == m]) for m in methods]
    axes[0].bar(range(len(methods)), selected, color=["#999999", "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"])
    axes[0].axhline(np.mean([float(r["injected_error_rate"]) for r in rows]), color="black", ls="--", lw=0.9, label="injected rate")
    axes[0].set(xticks=range(len(methods)), xticklabels=["random", "exposure", "exact\ninfluence", "LOO", "combined"],
                ylabel="selected corrupted fraction")
    panel_label(axes[0], "a")
    axes[0].legend(frameon=False, fontsize=7)
    for method, color in zip(methods, ["#999999", "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]):
        means = []
        for budget in (0.05, 0.10, 0.20):
            values = [float(r["test_accuracy_gain"]) for r in rows if r["method"] == method and abs(float(r["budget_fraction"]) - budget) < 1e-9]
            means.append(np.mean(values))
        axes[1].plot([5, 10, 20], means, "o-", label=method.replace("_", " "), color=color)
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set(xlabel="review budget (%)", ylabel="test accuracy gain")
    panel_label(axes[1], "b")
    axes[1].legend(frameon=False, fontsize=6.5, ncol=2)
    write_panel("fig5_audit_panels.csv", [{"panel": "all", **r} for r in rows])
    save(fig, "fig5_label_audit")


def fig6_operations() -> None:
    runtime = [
        row
        for row in read_csv("e9_runtime.csv")
        if row.get("benchmark_family") == "deterministic_influence"
    ]
    probability = [
        row
        for row in read_csv("e9_runtime.csv")
        if row.get("benchmark_family") in {
            "shared_probability_fixed_train",
            "shared_probability_overlap_sweep",
        }
        and row.get("method") in {"overlap_graph_exact", "dense_pairwise_exact"}
    ]
    ties = read_csv("e7_tie_weight_imbalance.csv")
    geometry = [r for r in read_csv("e6_geometry_construction.csv") if r.get("perturbation") == "relabel_plus_radial_displacement"]
    weighted = [r for r in ties if r.get("policy") == "weighted_vote"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0))
    axes = axes.ravel()
    for row in geometry:
        delta = float(row["delta"])
        fixed = int(row["fixed_label_prediction_changed"] == "True")
        combined = int(row["combined_prediction_changed"] == "True")
        axes[0].scatter(delta, fixed, marker="o", color="#377eb8", s=35)
        axes[0].scatter(delta, combined, marker="x", color="#d95f02", s=40)
    axes[0].set(xlabel="radial displacement", ylabel="prediction change")
    panel_label(axes[0], "a")
    axes[0].set_yticks([0, 1], ["no", "yes"])
    axes[0].legend(["fixed label", "combined"], frameon=False, fontsize=7)
    for case in sorted({r["case"] for r in weighted}):
        subset = [r for r in weighted if r["case"] == case]
        axes[1].plot([float(r["power"]) for r in subset], [float(r["weighted_margin"]) for r in subset], "o-", label=case.replace("_", " "))
    axes[1].set(xlabel="inverse-distance power", ylabel="weighted vote margin")
    panel_label(axes[1], "b")
    axes[1].legend(frameon=False, fontsize=6.5)
    for k in sorted({int(r["k"]) for r in runtime}):
        subset = [
            r
            for r in runtime
            if int(r["k"]) == k and r["speedup"] not in {"", "nan"}
        ]
        axes[2].plot([int(r["n_train"]) for r in subset], [float(r["speedup"]) for r in subset], "o-", label=f"k={k}")
    axes[2].set(xlabel="training prototypes", ylabel="reference / exact time")
    panel_label(axes[2], "c")
    axes[2].legend(frameon=False, fontsize=7)
    fixed = [r for r in probability if r.get("benchmark_family") == "shared_probability_fixed_train"]
    fixed_overlap = {int(r["n_query"]): r for r in fixed if r["method"] == "overlap_graph_exact"}
    fixed_dense = {int(r["n_query"]): r for r in fixed if r["method"] == "dense_pairwise_exact"}
    fixed_q = sorted(set(fixed_overlap) & set(fixed_dense))
    fixed_speedups = [
        float(fixed_dense[q]["runtime_median_seconds"])
        / float(fixed_overlap[q]["runtime_median_seconds"])
        for q in fixed_q
    ]
    sweep = [r for r in probability if r.get("benchmark_family") == "shared_probability_overlap_sweep"]
    sweep_overlap = {r["overlap_mode"]: r for r in sweep if r["method"] == "overlap_graph_exact"}
    sweep_dense = {r["overlap_mode"]: r for r in sweep if r["method"] == "dense_pairwise_exact"}
    if sweep_overlap and sweep_dense:
        modes = ["disjoint", "block_shared", "chain", "fully_shared"]
        edge_values = [float(sweep_overlap[m]["overlap_pair_fraction"]) for m in modes]
        speed_values = [
            float(sweep_dense[m]["runtime_median_seconds"])
            / float(sweep_overlap[m]["runtime_median_seconds"])
            for m in modes
        ]
        labels = [f"q={q}" for q in fixed_q] + [m.replace("_", " ") for m in modes]
        values = fixed_speedups + speed_values
        positions = np.arange(len(values))
        colors = ["#377eb8"] * len(fixed_speedups) + ["#d95f02"] * len(speed_values)
        axes[3].bar(positions, values, color=colors, width=0.72)
        axes[3].set_xticks(positions, labels, rotation=35, ha="right", fontsize=7)
        axes[3].axhline(1.0, color="#555555", lw=0.8, ls=":")
        axes[3].set_xlabel("fixed-training rows (blue) / overlap modes (orange)")
    elif fixed_speedups:
        axes[3].plot(fixed_q, fixed_speedups, "o-", color="#377eb8", label="fixed $n=1280$")
        axes[3].set_xlabel("query count (fixed $n=1280$)")
    else:
        axes[3].set_xlabel("control")
    axes[3].set_ylabel("dense / overlap exact time")
    panel_label(axes[3], "d")
    write_panel("fig6_operations_panels.csv", [
        *[{"panel": "a", **r} for r in geometry],
        *[{"panel": "b", **r} for r in weighted],
        *[{"panel": "c", **r} for r in runtime],
        *[{"panel": "d", **r} for r in probability],
    ])
    save(fig, "fig6_reproducibility_operations")


def main() -> None:
    fig1_framework()
    fig2_theory()
    fig3_probability()
    fig4_influence()
    fig5_audit()
    fig6_operations()
    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
