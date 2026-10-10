"""Build the seven revision figures from the processed experiment tables."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from knn_reliability.influence import compute_influence  # noqa: E402
from knn_reliability.knn import build_neighbor_cache, predict_from_neighbors  # noqa: E402
TABLE_DIR = ROOT / "results" / "processed"
FIG_DIR = ROOT / "figures" / "export"
DATA_DIR = ROOT / "figures" / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
COLORS = {
    "exact": "#0072B2",
    "simulation": "#56B4E9",
    "approximation": "#E69F00",
    "reference": "#777777",
    "class0": "#0072B2",
    "class1": "#D55E00",
    "random": "#7F8790",
    "exposure": "#009E73",
    "influence": "#0072B2",
    "loo": "#D55E00",
    "combined": "#CC79A7",
    "dense_2d": "#4B5563",
    "dense_1d": "#D55E00",
    "sparse_2d": "#0072B2",
    "sparse_1d": "#009E73",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13.0,
    "axes.labelsize": 13.0,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "axes.titlesize": 13.0,
    "legend.fontsize": 12.0,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.25,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.10, f"({label})", transform=ax.transAxes,
            va="bottom", ha="left", fontsize=12, fontweight="bold")


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
    if stem == "fig5_label_audit":
        fig.tight_layout(rect=(0, 0.10, 1, 1))
    elif stem == "fig6_geometry_operations":
        fig.tight_layout(rect=(0, 0.04, 0.96, 1))
    else:
        fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig1_framework() -> None:
    """Show correct LOO decisions alongside a retained-prototype flip."""

    fig = plt.figure(figsize=(6.3, 4.6))
    layout = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.78], hspace=0.45, wspace=0.34)
    axes = [fig.add_subplot(layout[0, 0]), fig.add_subplot(layout[0, 1]),
            fig.add_subplot(layout[1, :])]
    colors = {0: COLORS["class0"], 1: COLORS["class1"]}
    diagonal = float(2**-0.5)
    points = np.array([
        [0.90 * diagonal, 0.90 * diagonal], [-0.90 * diagonal, -0.90 * diagonal],
        [0.00, 0.95], [0.00, -0.95], [0.95, 0.00], [-0.95, 0.00],
        [1.10 * diagonal, 1.10 * diagonal], [1.15 * diagonal, 1.15 * diagonal],
        [-1.10 * diagonal, -1.10 * diagonal], [-1.15 * diagonal, -1.15 * diagonal],
        [0.00, 1.10], [0.00, 1.15], [0.00, -1.10], [0.00, -1.15],
        [1.10, 0.00], [1.15, 0.00], [-1.10, 0.00], [-1.15, 0.00],
    ])
    labels = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    query_batch = np.array([[0.00, 0.02], [0.00, -0.02], [0.02, 0.00], [-0.02, 0.00]])
    query = query_batch[:1]
    k = 3
    query_cache = build_neighbor_cache(points, query_batch, k)
    base_prediction, base_counts, _ = predict_from_neighbors(
        labels, query_cache.indices, classes=[0, 1]
    )
    changed_labels = labels.copy()
    changed_labels[0] = 1
    relabel_prediction, relabel_counts, _ = predict_from_neighbors(
        changed_labels, query_cache.indices, classes=[0, 1]
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
    influence_result = compute_influence(labels, query_cache.indices, classes=[0, 1])
    influence_matrix = influence_result.decisive_matrix.astype(float)
    assert influence_matrix.shape == (4, 18)
    assert np.all(influence_matrix.sum(axis=1) == 2)
    assert np.all(influence_matrix[:, :2] == 1)
    assert np.all(influence_matrix[:, 2:] == 0)
    assert np.all(base_prediction == 0)
    assert loo_correct == len(points)
    assert int(relabel_prediction[0]) == 1
    axes[0].scatter(points[:, 0], points[:, 1], s=60,
                    c=[colors[int(y)] for y in labels], edgecolor="white", linewidth=0.7)
    axes[0].scatter(points[0, 0], points[0, 1], s=95, facecolors="none",
                    edgecolors="#777777", linewidth=1.2, zorder=3)
    axes[0].scatter(*query[0], marker="*", s=145, c="black", zorder=4)
    axes[0].scatter(query_batch[1:, 0], query_batch[1:, 1], marker="x", s=28,
                    color=COLORS["reference"], linewidth=0.9, zorder=3)
    for neighbor_index in query_cache.indices[0]:
        axes[0].plot(
            [query[0, 0], points[neighbor_index, 0]],
            [query[0, 1], points[neighbor_index, 1]],
            color="#999999", lw=0.8, ls="--", zorder=0,
        )
    representative_labels = {1, 2, 3, 5}  # remaining support points share the same class
    for index, (x, y) in enumerate(points, start=1):
        if index in representative_labels:
            axes[0].text(x + 0.04, y + 0.04, f"$z_{index}$", fontsize=12)
    axes[0].text(
        0.5, 0.05,
        f"independent query votes {base_counts[0].tolist()} $\\rightarrow$ class {int(base_prediction[0])}",
        transform=axes[0].transAxes, ha="center", va="bottom", fontsize=12,
    )
    axes[0].text(0.02, 0.97, f"LOO: {loo_correct}/{len(points)} correct",
                 transform=axes[0].transAxes, ha="left", va="top", fontsize=12,
                 bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc"})
    panel_label(axes[0], "a")

    axes[1].scatter(points[:, 0], points[:, 1], s=60,
                    c=[colors[int(y)] for y in changed_labels], edgecolor="white", linewidth=0.7)
    axes[1].scatter(*query[0], marker="*", s=120, c="black", zorder=4)
    axes[1].scatter(query_batch[1:, 0], query_batch[1:, 1], marker="x", s=28,
                    color=COLORS["reference"], linewidth=0.9, zorder=3)
    for neighbor_index in query_cache.indices[0]:
        axes[1].plot(
            [query[0, 0], points[neighbor_index, 0]],
            [query[0, 1], points[neighbor_index, 1]],
            color="#999999", lw=0.8, ls="--", zorder=0,
        )
    for index, (x, y) in enumerate(points, start=1):
        if index in representative_labels:
            axes[1].text(x + 0.04, y + 0.04, f"$z_{index}$", fontsize=12)
    axes[1].annotate("$z_1$: 0 $\\to$ 1", xy=points[0], xytext=(0.18, 1.22),
                     arrowprops={"arrowstyle": "->", "color": "#984ea3"}, color="#984ea3", fontsize=12)
    axes[1].text(
        0.5, 0.05,
        f"votes {base_counts[0].tolist()} $\\rightarrow$ {relabel_counts[0].tolist()}; "
        f"class {int(base_prediction[0])} $\\rightarrow$ {int(relabel_prediction[0])}",
        transform=axes[1].transAxes, ha="center", va="bottom", fontsize=12,
    )
    panel_label(axes[1], "b")
    # The matrix is generated from the same four-query classifier object as
    # panels (a) and (b), rather than entered as a visual schematic.
    # Display the two nonzero columns and the collective all-zero columns.
    # This is a lossless visualization of this particular incidence matrix;
    # the full 4 x 18 matrix is always exported in fig1_influence_matrix.csv.
    other_columns = influence_matrix[:, 2:]
    if not np.all(other_columns == 0):
        raise AssertionError("Figure 1 compressed matrix has a nonzero omitted prototype")
    compressed = np.column_stack([influence_matrix[:, 0], influence_matrix[:, 1],
                                  np.max(other_columns, axis=1)])
    axes[2].imshow(compressed, cmap=ListedColormap(["#f2f2f2", COLORS["exact"]]),
                   vmin=0, vmax=1, aspect="auto")
    axes[2].set_xticks([0, 1, 2], ["$z_1$", "$z_2$", "other 16\n(all zero)"])
    axes[2].set_yticks(range(4), ["$q_1$", "$q_2$", "$q_3$", "$q_4$"])
    axes[2].set_xlabel("training prototype (16 zero columns grouped)")
    axes[2].set_ylabel("query")
    axes[2].tick_params(labelsize=12)
    for row in range(compressed.shape[0]):
        for column in range(compressed.shape[1]):
            axes[2].text(column, row, str(int(compressed[row, column])), ha="center",
                         va="center", fontsize=12,
                         color="white" if compressed[row, column] else "#333333")
    panel_label(axes[2], "c")
    xmin = float(points[:, 0].min() - 0.35)
    xmax = float(points[:, 0].max() + 0.35)
    ymin = float(min(points[:, 1].min(), query[:, 1].min()) - 0.35)
    ymax = float(max(points[:, 1].max(), query[:, 1].max()) + 0.35)
    for ax in axes[:2]:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xticks([-2, -1, 0, 1])
        ax.set_yticks([-0.5, 0, 0.5])
        ax.tick_params(labelsize=12)
        ax.set_xlabel("feature 1", fontsize=12)
        ax.set_ylabel("feature 2", fontsize=12)
        for spine in ax.spines.values():
            spine.set_color("#bbbbbb")
    for spine in axes[2].spines.values():
        spine.set_color("#bbbbbb")
    write_panel("fig1_influence_matrix.csv", [
        {"query_id": query_index + 1, "prototype_id": prototype_index + 1,
         "influence": int(influence_matrix[query_index, prototype_index]),
         "decisive_count": int(influence_matrix[query_index].sum()), "k": k,
         "n_query": len(query_batch), "n_train": len(points)}
        for query_index in range(len(query_batch))
        for prototype_index in range(len(points))
    ])
    write_panel("fig1_revision_framework_panels.csv", [
        {"panel": "a", "operation": "LOO at each training location", "k": k,
         "base_votes": ":".join(map(str, base_counts[0])), "changed_votes": "",
         "base_prediction": int(base_prediction[0]), "changed_prediction": "",
         "loo_correct": loo_correct, "loo_total": len(points), "query_type": "query_1_of_shared_batch"},
        {"panel": "b", "operation": "retained prototype relabeling z1", "k": k,
         "base_votes": ":".join(map(str, base_counts[0])), "changed_votes": ":".join(map(str, relabel_counts[0])),
         "base_prediction": int(base_prediction[0]), "changed_prediction": int(relabel_prediction[0]),
         "loo_correct": loo_correct, "loo_total": len(points), "query_type": "query_1_of_shared_batch"},
        {"panel": "c", "operation": "shared batch influence incidence",
         "k": k, "shared_prototypes": 2, "query_count": len(query_batch),
         "matrix_rows": influence_matrix.shape[0], "matrix_columns": influence_matrix.shape[1],
         "decisive_count_per_query": ":".join(map(str, influence_matrix.sum(axis=1).astype(int))),
         "query_type": "shared_batch_actual"},
    ])
    save(fig, "fig1_revision_framework")


def fig2_theory() -> None:
    e1 = read_csv("e1_vote_configurations.csv")
    e4 = read_csv("e4_distribution_stability.csv")
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.8), gridspec_kw={"hspace": 0.48, "wspace": 0.34})
    axes = axes.ravel()

    classes = sorted({int(row["n_classes"]) for row in e1})
    k_values = sorted({int(row["k"]) for row in e1})
    exact_grid = np.zeros((len(classes), len(k_values)))
    screen_only_grid = np.zeros_like(exact_grid)
    count_rows = []
    for i, n_classes in enumerate(classes):
        for j, k_value in enumerate(k_values):
            subset = [row for row in e1 if int(row["n_classes"]) == n_classes and int(row["k"]) == k_value]
            exact_count = sum(int(row["exact_vulnerable"]) for row in subset)
            screen_count = sum(int(row["band_flag"]) and not int(row["exact_vulnerable"]) for row in subset)
            total = len(subset)
            exact_grid[i, j] = exact_count / total
            screen_only_grid[i, j] = screen_count / total
            count_rows.append({
                "panel": "a/b", "n_classes": n_classes, "k": k_value,
                "total_configurations": total, "exact_vulnerable_count": exact_count,
                "screen_only_count": screen_count,
                "exact_vulnerable_rate": exact_grid[i, j],
                "screen_only_rate": screen_only_grid[i, j],
                "source": "e1_vote_configurations.csv",
            })
    for axis, values, title, cmap, label in (
        (axes[0], exact_grid, "exact vulnerability rate", "Blues", "a"),
        (axes[1], screen_only_grid, "screen-only screening rate", "Oranges", "b"),
    ):
        image = axis.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        axis.set_xticks(range(len(k_values)), k_values)
        axis.set_yticks(range(len(classes)), classes)
        axis.set(xlabel="$k$", ylabel="number of classes", title=title)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                axis.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                          fontsize=12, color="white" if values[i, j] > 0.55 else "#333333")
        fig.colorbar(image, ax=axis, fraction=0.045, pad=0.03, ticks=[0, 0.5, 1])
        panel_label(axis, label)

    theta = np.linspace(0, 2 * np.pi, 200)
    axes[2].plot(np.cos(theta), np.sin(theta), color=COLORS["reference"], lw=1.0)
    k = 5
    for j in range(k):
        angle = 2 * np.pi * j / k
        color = COLORS["class0"] if j < 3 else COLORS["class1"]
        axes[2].scatter([np.cos(angle)], [np.sin(angle)], marker="o", s=32,
                        color=color, zorder=3)
        for t in range(1, k + 1):
            radius = 1 + 0.01 * t / k
            axes[2].scatter([radius * np.cos(angle)], [radius * np.sin(angle)],
                            marker=".", s=18, color=color, alpha=0.65)
    axes[2].scatter([0], [0], marker="*", color="black", s=75, zorder=4)
    axes[2].set_aspect("equal")
    axes[2].set(xlabel="feature 1", ylabel="feature 2", title="verified ring/support construction")
    axes[2].set_xlim(-1.28, 1.28)
    axes[2].set_ylim(-1.28, 1.28)
    panel_label(axes[2], "c")

    support_epsilon = np.arange(1, k + 1) * 0.01 / k
    axes[3].axvline(0, color=COLORS["reference"], lw=0.9, ls="--")
    axes[3].scatter(np.zeros(k), np.arange(1, k + 1), marker="*", s=52,
                    color="black", label="query")
    axes[3].scatter(support_epsilon, np.arange(1, k + 1), marker="o", s=30,
                    color=COLORS["class0"], label="same-label support")
    axes[3].set(xlabel=r"radial separation $\epsilon$", ylabel="support index",
                title="local support separation")
    axes[3].set_xlim(-0.0015, 0.012)
    axes[3].set_xticks([0, 0.002, 0.006, 0.010], ["0", ".002", ".006", ".010"])
    axes[3].legend(frameon=False, loc="upper left", fontsize=12)
    panel_label(axes[3], "d")

    write_panel("fig2_theory_panels.csv", [
        *count_rows,
        {"panel": "c", "n": len(e1), "exact_reference": "zero mismatches", "band_false_negatives": "zero",
         "source": "e1_vote_configurations.csv"},
        {"panel": "d", "k": 5, "epsilon": ".002:.004:.006:.008:.010",
         "supports_per_representative": 5, "source": "constructed from verified geometry"},
        *[{"panel": "supplement", "rows": len(e4), "model": "random training labels",
           "source": "e4_distribution_stability.csv", **row} for row in e4],
    ])
    save(fig, "fig2_exact_theory")


def fig3_probability() -> None:
    rows = read_csv("e3_probability_risk.csv")
    finite_rows = read_csv("e3_finite_noise_validation.csv")
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.0), gridspec_kw={"hspace": 0.58, "wspace": 0.38})
    axes = axes.ravel()
    controlled = [
        r for r in finite_rows if r.get("pair_id") == "controlled_same_k_Rpoint_pair"
    ]
    controlled_by_name = {}
    for row in controlled:
        controlled_by_name.setdefault(row["pair_member"], row)
    shared_y = np.array([0, 0, 1, 1, 1, 1], dtype=int)
    shared_neighbors = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5]], dtype=int)
    distributed_y = np.tile(np.array([0, 0, 1], dtype=int), 4)
    distributed_neighbors = np.arange(12, dtype=int).reshape(4, 3)
    shared_matrix = compute_influence(shared_y, shared_neighbors, classes=[0, 1]).decisive_matrix.astype(int)
    distributed_matrix = compute_influence(
        distributed_y, distributed_neighbors, classes=[0, 1]
    ).decisive_matrix.astype(int)
    assert np.array_equal(shared_matrix, np.array([[1, 1, 0, 0, 0, 0]] * 4))
    expected_distributed = np.zeros((4, 12), dtype=int)
    for query_index in range(4):
        expected_distributed[query_index, 3 * query_index:3 * query_index + 2] = 1
    assert np.array_equal(distributed_matrix, expected_distributed)
    # Match the displayed prototype width without changing any influence value:
    # the six padded columns are never selected and are recorded explicitly.
    shared_display_matrix = np.pad(shared_matrix, ((0, 0), (0, 6)), constant_values=0)
    matrix_rows = [
        *[
            {"panel": "a", "structure": "shared", "query_id": q + 1, "prototype_id": p + 1,
             "influence": int(shared_display_matrix[q, p]), "decisive_count": int(shared_matrix[q].sum()),
             "k": 3, "n_query": 4, "Rpoint": 1.0,
             "unused_padding": int(p >= shared_matrix.shape[1]), "source": "controlled_same_k_Rpoint_pair"}
            for q in range(shared_display_matrix.shape[0]) for p in range(shared_display_matrix.shape[1])
        ],
        *[
            {"panel": "b", "structure": "distributed", "query_id": q + 1, "prototype_id": p + 1,
             "influence": int(distributed_matrix[q, p]), "decisive_count": int(distributed_matrix[q].sum()),
             "k": 3, "n_query": 4, "Rpoint": 1.0, "unused_padding": 0,
             "source": "controlled_same_k_Rpoint_pair"}
            for q in range(distributed_matrix.shape[0]) for p in range(distributed_matrix.shape[1])
        ],
    ]
    write_panel("fig4_influence_matrices.csv", matrix_rows)

    shared_row = controlled_by_name["shared"]
    distributed_row = controlled_by_name["distributed"]
    binary_cmap = ListedColormap(["#f2f2f2", COLORS["exact"]])
    for axis, matrix, title, panel in (
        (axes[0], shared_display_matrix, "shared influence (4 x 12)", "a"),
        (axes[1], distributed_matrix, "distributed influence (4 x 12)", "b"),
    ):
        axis.imshow(matrix, cmap=binary_cmap, vmin=0, vmax=1, aspect="auto")
        coefficient = float(shared_row["variance_coefficient"]) if title.startswith("shared") else float(distributed_row["variance_coefficient"])
        h2_value = float(shared_row["H2"]) if title.startswith("shared") else float(distributed_row["H2"])
        axis.set_title(f"{title}\n$H_2$={h2_value:.3f}; $c^2H_2$={coefficient:.3f}", fontsize=12)
        axis.set_xticks([0, 1, 5, 7, 11], ["1", "2", "6", "8", "12"])
        axis.set_yticks(range(4), ["q1", "q2", "q3", "q4"], fontsize=12)
        axis.set_xlabel("prototype index", fontsize=12)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                if matrix[row_index, column_index]:
                    axis.text(column_index, row_index, "1", ha="center", va="center", fontsize=12)
        panel_label(axis, panel)
    epsilons = sorted({float(row["epsilon"]) for row in controlled})
    curve_rows = []
    markers = {"shared": "o", "distributed": "s"}
    for member, color in (("shared", COLORS["exact"]), ("distributed", COLORS["simulation"])):
        subset = sorted((r for r in controlled if r["pair_member"] == member), key=lambda r: float(r["epsilon"]))
        exact_values = [float(r["exact_variance"]) for r in subset]
        first_values = [float(r["first_order_variance"]) for r in subset]
        axes[2].plot(epsilons, exact_values, marker=markers[member], color=color,
                     label=f"{member} exact")
        axes[2].plot(epsilons, first_values, marker=markers[member], color=COLORS["approximation"],
                     ls="--", label=f"{member} first-order")
        for row_index, row in enumerate(subset, start=1):
            curve_rows.extend([
                {"panel": "c", "pair_member": member, "metric": "exact_variance",
                 "epsilon": row["epsilon"], "value": row["exact_variance"],
                 "variance_coefficient": row["variance_coefficient"],
                 "source_row": f"controlled_{member}_{row_index}", "transformation": "direct frozen CSV field"},
                {"panel": "c", "pair_member": member, "metric": "first_order_variance",
                 "epsilon": row["epsilon"], "value": row["first_order_variance"],
                 "variance_coefficient": row["variance_coefficient"],
                 "source_row": f"controlled_{member}_{row_index}", "transformation": "direct frozen CSV field"},
                {"panel": "d", "pair_member": member, "metric": "relative_approximation_error",
                 "epsilon": row["epsilon"], "value": row["relative_error"],
                 "variance_coefficient": row["variance_coefficient"],
                 "source_row": f"controlled_{member}_{row_index}", "transformation": "frozen relative_error"},
            ])
        axes[3].plot(epsilons, [float(r["relative_error"]) for r in subset],
                     marker=markers[member], color=color, label=member)
    axes[2].set(xlabel="flip probability $\\epsilon$", ylabel="exact batch variance")
    axes[2].set_title("controlled finite-noise validation", fontsize=12)
    axes[2].legend(frameon=False, fontsize=11.5, ncol=2, loc="upper left")
    axes[3].set(xlabel="flip probability $\\epsilon$", ylabel="relative approximation error")
    axes[3].set_title("exact versus first-order approximation", fontsize=12)
    axes[3].set_ylim(bottom=0)
    axes[3].legend(frameon=False, fontsize=11.5, loc="upper left")
    for axis in axes[2:]:
        axis.set_xscale("log")
        axis.set_xticks(epsilons, [f"{value:g}" for value in epsilons])
    panel_label(axes[2], "c")
    panel_label(axes[3], "d")

    finite_real = [r for r in finite_rows if r.get("structure_type") == "real_data"]
    write_panel("fig4_probability_panels.csv", [
        *curve_rows,
        *[{"panel": "supplement_real_data_validation", "dataset": r["dataset"],
           "k": r["k"], "epsilon": r["epsilon"], "exact_variance": r["exact_variance"],
           "first_order_variance": r["first_order_variance"], "source": "e3_finite_noise_validation.csv",
           "transformation": "retained by dataset and k; not averaged into Fig. 4"} for r in finite_real],
    ])
    save(fig, "fig4_probability_stability")


def fig4_influence() -> None:
    """Report means, uncertainty, and a legible paired-dataset distribution."""
    rows = [r for r in read_csv("e2_deterministic_influence.csv") if r["split"] == "test"]
    metrics = [
        ("point_prv", r"$R_{\mathrm{point}}$", "point vulnerability"),
        ("r1", "$R_1$", "global single-relabel risk"),
        ("influence_concentration", "influence concentration", "max prototype share"),
        ("loo_error", "LOO error", "deleted-point recovery error"),
    ]
    k_values = sorted({int(row["k"]) for row in rows})
    summary_rows = []
    fig = plt.figure(figsize=(6.3, 5.7))
    layout = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.85], hspace=0.50, wspace=0.34)
    axes = [fig.add_subplot(layout[0, 0]), fig.add_subplot(layout[0, 1]),
            fig.add_subplot(layout[1, 0]), fig.add_subplot(layout[1, 1]),
            fig.add_subplot(layout[2, :])]
    for axis, (field, ylabel, description), label in zip(
        axes[:4], metrics, ("a", "b", "c", "d")
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
        axis.errorbar(k_values, means, yerr=errors, fmt="o-", color=COLORS["exact"],
                      capsize=3, lw=1.25, ms=4)
        axis.set(xlabel="$k$", ylabel=ylabel, xticks=k_values)
        axis.tick_params(labelsize=16)
        axis.xaxis.label.set_size(16)
        axis.yaxis.label.set_size(17)
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        panel_label(axis, label)
    write_panel("fig3_deterministic_influence_panels.csv", summary_rows)
    paired_rows = []
    dataset_names = sorted({row["dataset"] for row in rows})
    for dataset in dataset_names:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for field, _, _ in metrics:
            at_k3 = [float(row[field]) for row in dataset_rows if int(row["k"]) == 3]
            at_k15 = [float(row[field]) for row in dataset_rows if int(row["k"]) == 15]
            if at_k3 and at_k15:
                paired_rows.append({
                    "dataset": dataset, "metric": field,
                    "comparison": "k15_minus_k3",
                    "delta": float(np.mean(at_k15) - np.mean(at_k3)),
                })
    paired_axis = axes[4]
    metric_colors = {"point_prv": COLORS["exact"], "r1": COLORS["approximation"], "loo_error": COLORS["exposure"]}
    metric_labels = {"point_prv": r"$R_{\mathrm{point}}$", "r1": r"$R_1$", "loo_error": "LOO error"}
    show_fields = ("point_prv", "r1", "loo_error")
    # Plot each of the 23 paired dataset differences without illegible name ticks.
    # The unabridged dataset identities remain in the panel CSV.
    for index, field in enumerate(show_fields):
        values = np.array([next(float(x["delta"]) for x in paired_rows
                                if x["dataset"] == dataset and x["metric"] == field)
                           for dataset in dataset_names])
        deterministic_jitter = 0.12 * np.sin(np.arange(len(values)) * 2.399963)
        paired_axis.scatter(values, np.full(len(values), index) + deterministic_jitter,
                            s=23, alpha=0.8, color=metric_colors[field], edgecolors="white", linewidth=0.3)
        median = float(np.median(values))
        paired_axis.plot([median, median], [index - 0.25, index + 0.25],
                         color="#111111", lw=2.1)
    paired_axis.axvline(0, color="#444444", lw=0.95, ls="--")
    paired_axis.set_yticks(range(3), [metric_labels[f] for f in show_fields])
    paired_axis.tick_params(axis="y", labelsize=16)
    paired_axis.invert_yaxis()
    paired_axis.set_xlabel("paired dataset change, $k=15$ minus $k=3$")
    # Black bars mark medians; dataset identities are in the paired CSV.
    paired_axis.grid(axis="x", color="#dddddd", lw=0.5)
    panel_label(paired_axis, "e")
    write_panel("fig3_deterministic_influence_paired.csv", paired_rows)
    save(fig, "fig3_deterministic_influence")


def fig5_audit() -> None:
    rows = read_csv("e5_label_audit.csv")
    methods = ["random", "neighborhood_exposure", "exact_decisive_influence", "loo_error", "combined_influence_loo"]
    colors = [COLORS["random"], COLORS["exposure"], COLORS["influence"], COLORS["loo"], COLORS["combined"]]
    labels = ["random", "exposure", "exact influence", "LOO", "combined"]
    budgets = (0.05, 0.10, 0.20)
    datasets = sorted({r["dataset"] for r in rows})
    fig, axes = plt.subplots(2, len(datasets), figsize=(6.3, 4.7), squeeze=False,
                             sharex="col", gridspec_kw={"hspace": 0.48, "wspace": 0.28})
    panel_rows = []

    def summarize(dataset: str, method: str, budget: float, field: str) -> tuple[float, float, float, float, int]:
        values = [
            float(r[field]) for r in rows
            if r["dataset"] == dataset
            and r["method"] == method
            and abs(float(r["budget_fraction"]) - budget) < 1e-9
        ]
        if not values:
            return float("nan"), float("nan"), float("nan"), float("nan"), 0
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        return float(np.mean(values)), float(median), float(q25), float(q75), len(values)

    for column, dataset in enumerate(datasets):
        selected_axis = axes[0, column]
        accuracy_axis = axes[1, column]
        for method, label, color in zip(methods, labels, colors):
            selected_medians, selected_low, selected_high = [], [], []
            accuracy_medians, accuracy_low, accuracy_high = [], [], []
            for budget in budgets:
                selected_summary = summarize(dataset, method, budget, "selected_error_rate")
                accuracy_summary = summarize(dataset, method, budget, "test_accuracy_gain")
                selected_medians.append(selected_summary[1])
                selected_low.append(max(0.0, selected_summary[1] - selected_summary[2]))
                selected_high.append(max(0.0, selected_summary[3] - selected_summary[1]))
                accuracy_medians.append(accuracy_summary[1])
                accuracy_low.append(max(0.0, accuracy_summary[1] - accuracy_summary[2]))
                accuracy_high.append(max(0.0, accuracy_summary[3] - accuracy_summary[1]))
                panel_rows.extend([
                    {
                        "panel": chr(ord("a") + column),
                        "dataset": dataset,
                        "metric": "selected_corrupted_fraction",
                        "method": method,
                        "budget_fraction": budget,
                        "mean": selected_summary[0],
                        "median": selected_summary[1],
                        "q25": selected_summary[2],
                        "q75": selected_summary[3],
                        "n_configurations": selected_summary[4],
                        "aggregation": "median over four seeds and three injected noise rates",
                    },
                    {
                        "panel": chr(ord("d") + column),
                        "dataset": dataset,
                        "metric": "test_accuracy_gain",
                        "method": method,
                        "budget_fraction": budget,
                        "mean": accuracy_summary[0],
                        "median": accuracy_summary[1],
                        "q25": accuracy_summary[2],
                        "q75": accuracy_summary[3],
                        "n_configurations": accuracy_summary[4],
                        "aggregation": "median over four seeds and three injected noise rates",
                    },
                ])
            selected_axis.errorbar(
                [5, 10, 20], selected_medians,
                yerr=[selected_low, selected_high], fmt="o-", ms=3.5, capsize=2,
                color=color, label=label,
            )
            accuracy_axis.errorbar(
                [5, 10, 20], accuracy_medians,
                yerr=[accuracy_low, accuracy_high], fmt="o-", ms=3.5, capsize=2,
                color=color, label=label,
            )
        injected = [
            np.mean([
                float(r["injected_error_rate"]) for r in rows
                if r["dataset"] == dataset
                and abs(float(r["budget_fraction"]) - budget) < 1e-9
            ])
            for budget in budgets
        ]
        selected_axis.plot([5, 10, 20], injected, "--", color=COLORS["reference"],
                           lw=1.0, label="injected rate")
        selected_axis.set_title(dataset.replace("_", " "), fontsize=12)
        selected_axis.set_xticks([5, 10, 20])
        selected_axis.set_ylim(bottom=0)
        accuracy_axis.set_xticks([5, 10, 20])
        accuracy_axis.axhline(0, color="black", lw=0.7)
        if column == 0:
            selected_axis.set_ylabel("selected corrupted fraction\nmedian (Q25--Q75)")
            accuracy_axis.set_ylabel("test accuracy gain\nmedian (Q25--Q75)")
        accuracy_axis.set_xlabel("review budget (%)")
        panel_label(selected_axis, chr(ord("a") + column))
        panel_label(accuracy_axis, chr(ord("d") + column))

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), fontsize=11.5)
    fig.subplots_adjust(bottom=0.19)
    write_panel("fig5_audit_panels.csv", panel_rows)
    save(fig, "fig5_label_audit")


def fig6_geometry_operations() -> None:
    ties = read_csv("e7_tie_weight_imbalance.csv")
    e6_rows = read_csv("e6_geometry_construction.csv")
    boundary = [r for r in e6_rows if r.get("perturbation") == "vote_gap_stratified_boundary_motion"]
    curve = [r for r in e6_rows if r.get("perturbation") == "four_state_motion_probability_curve"]
    weighted = [r for r in ties if r.get("policy") == "weighted_vote"]
    fig, axes = plt.subplots(3, 2, figsize=(6.3, 6.8),
                             gridspec_kw={"hspace": 0.72, "wspace": 0.42})
    axes = axes.ravel()
    rows_for_panel = []
    heatmap_cmap = plt.get_cmap("YlGnBu")
    mode_titles = {"0": "candidate labels 0,0", "1": "candidate labels 1,1"}
    mode_panels = {("0", "geometry_prediction_changed_rate"): "a",
                   ("0", "combined_prediction_changed_rate"): "b",
                   ("1", "geometry_prediction_changed_rate"): "c",
                   ("1", "combined_prediction_changed_rate"): "d"}
    for mode in ("0", "1"):
        subset = [r for r in curve if r["candidate_label_mode"] == mode]
        deltas = sorted({float(r["delta"]) for r in subset})
        gaps = sorted({float(r["boundary_gap"]) for r in subset})
        for metric in ("geometry_prediction_changed_rate", "combined_prediction_changed_rate"):
            values = np.full((len(gaps), len(deltas)), np.nan)
            for i, gap in enumerate(gaps):
                for j, delta in enumerate(deltas):
                    matches = [r for r in subset if float(r["boundary_gap"]) == gap and float(r["delta"]) == delta]
                    if matches:
                        values[i, j] = float(matches[0][metric])
                        row = matches[0]
                        rows_for_panel.append({
                            "panel": mode_panels[(mode, metric)], "source_row": f"e6_curve_{e6_rows.index(row) + 2}",
                            "candidate_label_mode": mode, "metric": metric, "delta": delta,
                            "boundary_gap": gap, "value": float(row[metric]),
                            "transformation": "direct frozen CSV field; heatmap cell",
                        })
            axis_index = ord(mode_panels[(mode, metric)]) - ord("a")
            image = axes[axis_index].imshow(values, origin="lower", vmin=0, vmax=1,
                                             cmap=heatmap_cmap, aspect="auto")
            axes[axis_index].set_xticks(range(len(deltas)), ["0", ".02", ".05", ".1", ".2", ".8", "2"])
            axes[axis_index].tick_params(axis="x", labelrotation=28, labelsize=11)
            axes[axis_index].set_yticks(range(len(gaps)), [f"{v:g}" for v in gaps])
            metric_label = "geometry-only" if metric.startswith("geometry") else "joint label + position"
            axes[axis_index].set_title(f"{mode_titles[mode]}\n{metric_label}", fontsize=12)
            axes[axis_index].set(xlabel="$\\delta$", ylabel="boundary gap")
            panel_label(axes[axis_index], mode_panels[(mode, metric)])
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    if not np.isnan(values[i, j]):
                        axes[axis_index].text(j, i, f"{values[i, j]:.1f}", ha="center", va="center",
                                              fontsize=11.5,
                                              color="white" if values[i, j] > 0.55 else "#333333")
    # The same YlGnBu 0--1 scale is used for all four heatmaps; cell labels
    # carry the exact probabilities so a narrow colorbar is unnecessary.
    fig.text(0.99, 0.78, "P(change)\n0--1", rotation=90, ha="center", va="center", fontsize=11,
             color=COLORS["reference"])

    conditions = ["enters", "stays_inside", "exits"]
    tile = np.array([
        [np.mean([float(row["target_in_neighbor_rate"]) for row in boundary if row["motion_condition"] == condition]),
         np.mean([float(row["neighbor_exchange_rate"]) for row in boundary if row["motion_condition"] == condition])]
        for condition in conditions
    ])
    axes[4].imshow(tile, cmap=ListedColormap(["#f2f2f2", COLORS["exact"]]), vmin=0, vmax=1, aspect="auto")
    axes[4].set_xticks([0, 1], ["target inside", "neighbor exchange"], rotation=20, ha="right")
    axes[4].set_yticks(range(3), ["enters", "stays inside", "exits"])
    axes[4].set_title("boundary membership mechanism", fontsize=12)
    axes[4].set_xlabel("binary membership outcome")
    for i in range(tile.shape[0]):
        for j in range(tile.shape[1]):
            axes[4].text(j, i, f"{tile[i, j]:.0f}", ha="center", va="center", fontsize=12)
    panel_label(axes[4], "e")
    for row in boundary:
        rows_for_panel.append({
            "panel": "e", "source_row": f"e6_boundary_{e6_rows.index(row) + 2}",
            "motion_condition": row["motion_condition"], "metric": "target_in_neighbor_rate",
            "value": row["target_in_neighbor_rate"], "neighbor_exchange_rate": row["neighbor_exchange_rate"],
            "transformation": "condition-level mean; binary tile",
        })

    case_order = ["equal_distance", "mild_distance", "strong_distance"]
    case_colors = {"equal_distance": COLORS["exact"], "mild_distance": COLORS["approximation"],
                   "strong_distance": COLORS["loo"]}
    ax_margin = axes[5]
    ax_count = ax_margin.twinx()
    for offset, case in zip((-0.12, 0.0, 0.12), case_order):
        subset = sorted([r for r in weighted if r["case"] == case], key=lambda r: float(r["power"]))
        x = np.array([float(r["power"]) for r in subset]) + offset
        ax_margin.scatter(x, [float(r["weighted_margin"]) for r in subset], s=28,
                          color=case_colors[case], marker="o", label=case.replace("_", " "))
        ax_count.scatter(x, [float(r["vulnerable_prototypes"]) for r in subset], s=28,
                         color=case_colors[case], marker="s", alpha=0.65)
        for row in subset:
            rows_for_panel.extend([
                {"panel": "f", "source_row": f"e7_weighted_{ties.index(row) + 2}",
                 "case": case, "power": row["power"], "metric": "weighted_margin",
                 "value": row["weighted_margin"], "transformation": "direct frozen CSV field"},
                {"panel": "f", "source_row": f"e7_weighted_{ties.index(row) + 2}",
                 "case": case, "power": row["power"], "metric": "vulnerable_prototypes",
                 "value": row["vulnerable_prototypes"], "transformation": "direct frozen CSV field"},
            ])
    ax_margin.axhline(0, color=COLORS["reference"], lw=0.8)
    ax_margin.set(xlabel="inverse-distance power", ylabel="weighted vote margin")
    ax_count.set_ylabel("vulnerable prototypes")
    ax_margin.set_xticks([0, 1, 2])
    ax_margin.set_title("discrete weighted-vote diagnostics", fontsize=12)
    ax_margin.legend(frameon=False, fontsize=11.5, loc="upper left")
    ax_margin.text(0.98, 0.04, "circle: margin\nsquare: count", transform=ax_margin.transAxes,
                   ha="right", va="bottom", fontsize=11, color=COLORS["reference"])
    panel_label(ax_margin, "f")
    write_panel("fig6_geometry_operations_panels.csv", rows_for_panel)
    save(fig, "fig6_geometry_operations")


def fig7_efficiency_operations() -> None:
    runtime_rows = read_csv("e9_runtime.csv")
    deterministic = [r for r in runtime_rows if r.get("benchmark_family") == "deterministic_influence"]
    k_sweep = [r for r in runtime_rows if r.get("benchmark_family") == "shared_probability_k_sweep"]
    fixed = [r for r in runtime_rows if r.get("benchmark_family") == "shared_probability_fixed_train"]
    overlap = [r for r in runtime_rows if r.get("benchmark_family") == "shared_probability_overlap_sweep"]
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.3),
                             gridspec_kw={"hspace": 0.82, "wspace": 0.38})
    axes = axes.ravel()
    config_rows = [r for r in deterministic if r["naive_status"] == "completed"]
    all_configs = sorted(deterministic, key=lambda r: (int(r["n_train"]), int(r["k"])))
    positions = np.arange(len(all_configs))
    exact_cfg = { (int(r["n_train"]), int(r["n_query"]), int(r["k"])): r for r in deterministic }
    axes[0].errorbar(positions, [float(r["exact_batched_seconds"]) for r in all_configs],
                     yerr=[[float(r["exact_batched_seconds"]) - float(r["exact_batched_q1_seconds"]) for r in all_configs],
                           [float(r["exact_batched_q3_seconds"]) - float(r["exact_batched_seconds"]) for r in all_configs]],
                     fmt="o", color=COLORS["exact"], capsize=2.5, label="exact batched")
    valid_positions = [i for i, r in enumerate(all_configs) if r["naive_status"] == "completed"]
    axes[0].errorbar(valid_positions, [float(all_configs[i]["naive_exhaustive_seconds"]) for i in valid_positions],
                     yerr=[[float(all_configs[i]["naive_exhaustive_seconds"]) - float(all_configs[i]["naive_exhaustive_q1_seconds"]) for i in valid_positions],
                           [float(all_configs[i]["naive_exhaustive_q3_seconds"]) - float(all_configs[i]["naive_exhaustive_seconds"]) for i in valid_positions]],
                     fmt="s", color=COLORS["reference"], capsize=2.5, label="label-copying reference")
    labels = [f"({r['n_train']},{r['n_query']},{r['k']})" for r in all_configs]
    axes[0].set_xticks(positions, labels, rotation=42, ha="right", fontsize=11)
    axes[0].set(xlabel="configuration", ylabel="runtime (s)")
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[0].legend(frameon=False, fontsize=11.5, loc="upper left")
    panel_label(axes[0], "a")
    method_rows = {method: {int(r["k"]): r for r in k_sweep if r["method"] == method}
                   for method in ("sparse_1d_exact", "sparse_2d_exact", "dense_1d_exact", "dense_2d_exact")}
    k_values = sorted(method_rows["sparse_1d_exact"])
    sparse_ratio = [float(method_rows["sparse_1d_exact"][k]["runtime_median_seconds"]) / float(method_rows["sparse_2d_exact"][k]["runtime_median_seconds"]) for k in k_values]
    dense_ratio = [float(method_rows["dense_1d_exact"][k]["runtime_median_seconds"]) / float(method_rows["dense_2d_exact"][k]["runtime_median_seconds"]) for k in k_values]
    axes[1].plot(k_values, sparse_ratio, "o", color=COLORS["sparse_1d"], label="sparse T1D / T2D")
    axes[1].plot(k_values, dense_ratio, "s", color=COLORS["dense_1d"], label="dense T1D / T2D")
    axes[1].axhline(1, color=COLORS["reference"], lw=0.9, ls="--", label="equal time")
    axes[1].set(xlabel="$k$", ylabel="T1D / T2D")
    axes[1].set_title(">1: 1D slower", fontsize=11)
    axes[1].legend(frameon=False, fontsize=11.5)
    panel_label(axes[1], "b")
    fixed_methods = {method: {int(r["n_query"]): r for r in fixed if r["method"] == method} for method in ("sparse_1d_exact", "sparse_2d_exact", "dense_1d_exact", "dense_2d_exact")}
    fixed_q = sorted(fixed_methods["sparse_1d_exact"])
    method_order = ["dense_2d_exact", "dense_1d_exact", "sparse_2d_exact", "sparse_1d_exact"]
    method_labels = {"dense_2d_exact": "dense 2D", "dense_1d_exact": "dense 1D",
                     "sparse_2d_exact": "sparse 2D", "sparse_1d_exact": "sparse 1D"}
    method_colors = {"dense_2d_exact": COLORS["dense_2d"], "dense_1d_exact": COLORS["dense_1d"],
                     "sparse_2d_exact": COLORS["sparse_2d"], "sparse_1d_exact": COLORS["sparse_1d"]}
    offsets = np.linspace(-0.24, 0.24, len(method_order))
    for offset, method in zip(offsets, method_order):
        subset = [fixed_methods[method][q] for q in fixed_q]
        axes[2].errorbar(np.arange(len(fixed_q)) + offset, [float(r["runtime_median_seconds"]) for r in subset],
                         yerr=[[float(r["runtime_median_seconds"]) - float(r["runtime_q1_seconds"]) for r in subset],
                               [float(r["runtime_q3_seconds"]) - float(r["runtime_median_seconds"]) for r in subset]],
                         fmt="o", ms=3.2, capsize=2, color=method_colors[method], label=method_labels[method])
    axes[2].set_xticks(range(len(fixed_q)), fixed_q)
    axes[2].set(xlabel="query count (fixed n=1280)", ylabel="runtime (s)")
    axes[2].legend(frameon=False, fontsize=11.5)
    panel_label(axes[2], "c")
    method_order = ["dense_2d_exact", "dense_1d_exact", "sparse_2d_exact", "sparse_1d_exact"]
    modes = ["disjoint", "block_shared", "chain", "fully_shared"]
    overlap_by_method = {method: {r["overlap_mode"]: r for r in overlap if r["method"] == method} for method in method_order}
    positions = np.arange(len(modes))
    baseline = overlap_by_method["sparse_1d_exact"]
    for offset, method in zip((-0.18, -0.06, 0.06, 0.18), method_order):
        values = [float(overlap_by_method[method][mode]["runtime_median_seconds"]) /
                  float(baseline[mode]["runtime_median_seconds"]) for mode in modes]
        axes[3].scatter(positions + offset, values, color=method_colors[method], s=34,
                        label=method_labels[method], zorder=4)
    axes[3].set_yscale("log")
    axes[3].tick_params(labelsize=12)
    axes[3].tick_params(axis="y", labelsize=14)
    axes[3].xaxis.label.set_size(12)
    axes[3].yaxis.label.set_size(13)
    axes[3].set_xticks(positions, ["disjoint", "block", "chain", "shared"], fontsize=10)
    axes[3].set(xlabel="overlap pattern", ylabel="time / sparse--1D")
    axes[3].set_title(">1: slower", fontsize=11)
    axes[3].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[3].axhline(1, color=COLORS["reference"], lw=0.9, ls="--")
    axes[3].grid(axis="y", which="major", color="#dddddd", lw=0.5)
    axes[3].legend(frameon=False, fontsize=9, ncol=1, loc="upper left")
    panel_label(axes[3], "d")
    write_panel("fig7_efficiency_operations_panels.csv", [
        *[{**r, "panel": "a", "source_row": f"e9_deterministic_{deterministic.index(r) + 2}",
           "transformation": "absolute median and Q1--Q3 error bars"} for r in deterministic],
        *[{**r, "panel": "b", "source_row": f"e9_k_sweep_{k_sweep.index(r) + 2}",
           "transformation": "T1D/T2D; >1 means 1D slower"} for r in k_sweep],
        *[{**r, "panel": "c", "source_row": f"e9_fixed_train_{fixed.index(r) + 2}",
           "transformation": "absolute median and Q1--Q3 error bars"} for r in fixed],
        *[{**r, "panel": "d", "source_row": f"e9_overlap_{overlap.index(r) + 2}",
           "transformation": "time divided by sparse--1D; >1 slower than reference"} for r in overlap],
    ])
    save(fig, "fig7_efficiency_operations")


def main() -> None:
    fig1_framework()
    fig2_theory()
    fig3_probability()
    fig4_influence()
    fig5_audit()
    fig6_geometry_operations()
    fig7_efficiency_operations()
    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
