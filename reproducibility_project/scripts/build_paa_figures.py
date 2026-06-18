"""Build the PAA figure set from the local benchmark outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PALETTE = {
    "navy": "#08306B",
    "blue": "#08519C",
    "midblue": "#2171B5",
    "lightblue": "#6BAED6",
    "paleblue": "#C6DBEF",
    "yellow": "#FED976",
    "cream": "#FFF7CC",
    "gray": "#969696",
    "orange": "#D95319",
    "gold": "#EDB120",
    "brightblue": "#0072BD",
    "green": "#77AC30",
    "white": "#FFFFFF",
}


def save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", default="G:/2026/paa_latest_knn/reproducibility_project/results/tables")
    parser.add_argument("--figures-dir", default="G:/2026/paa_latest_knn/reproducibility_project/results/figures")
    parser.add_argument("--processed-dir", default="G:/2026/paa_latest_knn/reproducibility_project/data/processed")
    args = parser.parse_args()

    tables = Path(args.tables_dir)
    figures = Path(args.figures_dir)
    processed = Path(args.processed_dir)
    figures.mkdir(parents=True, exist_ok=True)

    bench = pd.read_csv(tables / "paa_multiclass_benchmark.csv")
    clean = pd.read_csv(tables / "paa_cleaning_baselines.csv")
    mitigate = pd.read_csv(tables / "paa_margin_k_selection.csv")
    runtime = pd.read_csv(tables / "paa_runtime_scaling.csv")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    build_framework(figures)
    build_mechanism(figures)
    build_case_profile(figures, bench, processed)
    build_blind_spot(figures, bench)
    build_diagnostic_comparison(figures, bench)
    build_applications_runtime(figures, clean, mitigate, runtime)


def build_framework(figures: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "manuscript_source_flat" / "fig1_audit_framework.png"
    img = plt.imread(source)
    fig, ax = plt.subplots(figsize=(11.28, 5.07))
    ax.imshow(img)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save(fig, figures / "fig1_audit_framework")


def build_mechanism(figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    panels = [([3, 2], "Before relabeling: gap = 1"), ([2, 3], "After relabeling: winner changes")]
    for ax, (vals, title) in zip(axes, panels):
        ax.bar(["winner", "competitor"], vals, color=[PALETTE["blue"], PALETTE["orange"]], width=0.58)
        ax.set_ylim(0, 4)
        ax.set_ylabel("local votes")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for j, v in enumerate(vals):
            ax.text(j, v + 0.08, str(v), ha="center", color=PALETTE["navy"])
    axes[0].annotate(
        "one fixed-location\nlabel replacement",
        xy=(1.05, 0.55), xycoords="axes fraction",
        xytext=(1.45, 0.55), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.3, color=PALETTE["navy"]),
        ha="center", va="center",
    )
    fig.tight_layout()
    save(fig, figures / "fig2_margin_mechanism")


def build_case_profile(figures: Path, bench: pd.DataFrame, processed: Path) -> None:
    dataset = "Digits-10" if "Digits-10" in set(bench["dataset"]) else bench["dataset"].iloc[0]
    sub = (
        bench[bench["dataset"] == dataset]
        .groupby("k", as_index=False)[["balanced_accuracy", "vulnerability_rate", "flag_rate"]]
        .mean()
    )

    (
        risk_values,
        prototype_images,
        prototype_scores,
        query_images,
        query_neighbor_images,
        selected_k,
    ) = compute_case_study(processed / "digits_10.npz")

    fig = plt.figure(figsize=(9.4, 5.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0], hspace=0.34, wspace=0.28)

    left = gs[:, 0].subgridspec(2, 1, hspace=0.10, height_ratios=[1.0, 1.0])
    ax_a1 = fig.add_subplot(left[0, 0])
    ax_a2 = fig.add_subplot(left[1, 0], sharex=ax_a1)

    ax_a1.plot(sub["k"], sub["balanced_accuracy"], marker="o", color=PALETTE["navy"], linewidth=2.4, markersize=6.8)
    ax_a1.axvline(selected_k, color=PALETTE["gray"], linestyle="--", linewidth=1.2)
    ax_a1.text(selected_k + 0.15, sub["balanced_accuracy"].min() + 0.002, "selected k", color=PALETTE["gray"], fontsize=8)
    label_x = sub["k"].iloc[-1] + 0.28
    ax_a1.text(label_x, sub["balanced_accuracy"].iloc[-1], "Balanced accuracy", color=PALETTE["navy"], va="center", fontsize=8)
    ax_a1.set_ylim(0.93, 0.985)
    ax_a1.set_ylabel("accuracy")
    ax_a1.set_title("(a) Risk--accuracy profile", loc="left")
    ax_a1.grid(axis="y", color=PALETTE["paleblue"], linewidth=0.8)

    ax_a2.plot(
        sub["k"],
        sub["vulnerability_rate"],
        marker="s",
        color=PALETTE["orange"],
        linewidth=2.1,
        markersize=6.0,
        label="EnumVulnRate",
    )
    ax_a2.plot(
        sub["k"],
        sub["flag_rate"],
        marker="^",
        color=PALETTE["green"],
        linewidth=2.1,
        markersize=6.2,
        label="BandRate",
    )
    ax_a2.axvline(selected_k, color=PALETTE["gray"], linestyle="--", linewidth=1.2)
    ax_a2.set_ylabel("risk")
    ax_a2.set_xlabel("k")
    ax_a2.grid(axis="y", color=PALETTE["paleblue"], linewidth=0.8)
    ax_a2.set_xticks(sub["k"])
    leg = ax_a2.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        facecolor=PALETTE["white"],
        edgecolor=PALETTE["paleblue"],
        fontsize=7.5,
        borderpad=0.25,
        handlelength=1.6,
        labelspacing=0.25,
    )
    for text, color in zip(leg.get_texts(), [PALETTE["orange"], PALETTE["green"]]):
        text.set_color(color)

    for ax in (ax_a1, ax_a2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor(PALETTE["white"])
        ax.set_xlim(sub["k"].min() - 0.3, sub["k"].max() + 1.65)

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_title("(b) Exposed prototypes", loc="left")
    draw_ranked_prototypes(ax_b, prototype_images, prototype_scores)

    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.set_title("(c) Flagged query neighborhoods", loc="left")
    draw_query_neighborhoods(ax_c, query_images, query_neighbor_images)

    save(fig, figures / "fig3_case_study_profile")


def build_blind_spot(figures: Path, bench: pd.DataFrame) -> None:
    agg = bench.groupby(["dataset", "k"], as_index=False)[["loo_error", "vulnerability_rate", "balanced_accuracy"]].mean()
    fig, ax = plt.subplots(figsize=(6.1, 4.2))
    cmap = LinearSegmentedColormap.from_list(
        "paa_blues", [PALETTE["paleblue"], PALETTE["lightblue"], PALETTE["midblue"], PALETTE["navy"]]
    )
    sc = ax.scatter(
        agg["loo_error"],
        agg["vulnerability_rate"],
        c=agg["k"],
        s=45 + 90 * agg["balanced_accuracy"].clip(0, 1),
        cmap=cmap,
        alpha=0.85,
        edgecolor=PALETTE["white"],
        linewidth=0.4,
    )
    ax.axvspan(0, 0.10, color=PALETTE["cream"], alpha=0.55)
    ax.axhspan(0.20, 1.0, color=PALETTE["paleblue"], alpha=0.22)
    ax.set_xlabel("LOO error")
    ax.set_ylabel("EnumVulnRate")
    ax.text(0.03, 0.92, "low LOO / high PRV", transform=ax.transAxes, color=PALETTE["navy"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.colorbar(sc, ax=ax, label="k")
    fig.tight_layout()
    save(fig, figures / "fig4_loo_blind_spot")


def build_diagnostic_comparison(figures: Path, bench: pd.DataFrame) -> None:
    long = bench.melt(
        value_vars=["margin_auroc", "confidence_auroc", "entropy_auroc", "distance_margin_auroc"],
        var_name="diagnostic",
        value_name="AUROC",
    ).dropna()
    labels = {
        "margin_auroc": "local margin",
        "confidence_auroc": "vote confidence",
        "entropy_auroc": "label entropy",
        "distance_margin_auroc": "distance margin",
    }
    long["diagnostic"] = long["diagnostic"].map(labels)
    order = ["local margin", "vote confidence", "label entropy", "distance margin"]
    means = long.groupby("diagnostic")["AUROC"].mean().reindex(order)
    stds = long.groupby("diagnostic")["AUROC"].std().reindex(order)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bar_colors = [PALETTE["navy"], PALETTE["midblue"], PALETTE["lightblue"], PALETTE["gold"]]
    ax.bar(order, means, yerr=stds, color=bar_colors, capsize=3)
    ax.set_ylabel("AUROC for exact PRV detection")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save(fig, figures / "fig5_diagnostic_comparison")


def build_applications_runtime(figures: Path, clean: pd.DataFrame, mitigate: pd.DataFrame, runtime: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))

    sel_vals = [
        mitigate["loo_vulnerability_rate"].mean(),
        mitigate["margin_vulnerability_rate"].mean(),
    ]
    sel_err = [
        mitigate["loo_vulnerability_rate"].std(),
        mitigate["margin_vulnerability_rate"].std(),
    ]
    axes[0].bar(["LOO-only", "margin-aware"], sel_vals, yerr=sel_err, color=[PALETTE["gray"], PALETTE["orange"]], capsize=3)
    axes[0].set_ylabel("EnumVulnRate")
    axes[0].set_title("validation-constrained k selection")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    order = ["standard", "ENN", "RENN", "AllKNN", "NCL", "CNN", "Tomek", "Margin-clean"]
    agg = clean.groupby("method")["vulnerability_rate"].agg(["mean", "std"]).reindex(order)
    colors = [PALETTE["gray"]] * len(order)
    colors[-1] = PALETTE["green"]
    axes[1].bar(agg.index, agg["mean"], yerr=agg["std"], color=colors, capsize=2)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_ylabel("EnumVulnRate after cleaning")
    axes[1].set_title("cleaning trade-off")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, figures / "fig6_mitigation_cleaning")


def draw_digit_strip(ax: plt.Axes, images: list[np.ndarray], ncols: int = 5) -> None:
    if not images:
        ax.text(0.5, 0.5, "no flagged examples", ha="center", va="center", color=PALETTE["gray"])
        return
    n = len(images)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    canvas = np.ones((nrows * 8 + (nrows - 1), ncols * 8 + (ncols - 1)))
    for idx, img in enumerate(images):
        r = idx // ncols
        c = idx % ncols
        r0 = r * 9
        c0 = c * 9
        canvas[r0:r0 + 8, c0:c0 + 8] = img.reshape(8, 8)
    ax.imshow(canvas, cmap="gray_r", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_ranked_prototypes(ax: plt.Axes, images: list[np.ndarray], scores: list[float]) -> None:
    ax.axis("off")
    if not images:
        ax.text(0.5, 0.5, "no exposed prototypes", ha="center", va="center", color=PALETTE["gray"])
        return
    cols = len(images)
    for idx, (img, score) in enumerate(zip(images, scores)):
        x0 = 0.03 + idx * (0.94 / cols)
        ax.text(x0 + 0.065, 0.96, f"#{idx + 1}", ha="center", va="top", fontsize=8, color=PALETTE["gray"], transform=ax.transAxes)
        img_ax = ax.inset_axes([x0, 0.28, 0.12, 0.48])
        img_ax.imshow(img.reshape(8, 8), cmap="gray_r", interpolation="nearest")
        img_ax.set_xticks([])
        img_ax.set_yticks([])
        for spine in img_ax.spines.values():
            spine.set_visible(False)
        ax.add_patch(plt.Rectangle((x0 + 0.015, 0.14), 0.09, 0.04, transform=ax.transAxes, facecolor=PALETTE["cream"], edgecolor="none"))
        ax.add_patch(plt.Rectangle((x0 + 0.015, 0.14), 0.09 * score, 0.04, transform=ax.transAxes, facecolor=PALETTE["midblue"], edgecolor="none"))
        ax.text(x0 + 0.06, 0.07, f"{score:.2f}", ha="center", va="center", fontsize=8, color=PALETTE["navy"], transform=ax.transAxes)
    ax.text(0.5, -0.02, "normalized exposure", ha="center", va="top", fontsize=8, color=PALETTE["gray"], transform=ax.transAxes)


def draw_query_neighborhoods(ax: plt.Axes, queries: list[np.ndarray], neighbor_rows: list[list[np.ndarray]]) -> None:
    ax.axis("off")
    if not queries:
        ax.text(0.5, 0.5, "no flagged queries", ha="center", va="center", color=PALETTE["gray"])
        return
    ax.text(0.12, 0.86, "query", fontsize=8, color=PALETTE["gray"], transform=ax.transAxes)
    ax.text(0.50, 0.86, "top-k neighbors", fontsize=8, color=PALETTE["gray"], ha="center", transform=ax.transAxes)
    rows = len(queries)
    for ridx, (query, neighbors) in enumerate(zip(queries, neighbor_rows)):
        y0 = 0.60 - ridx * 0.23
        ax.text(0.02, y0 + 0.07, f"q{ridx + 1}", fontsize=8, color=PALETTE["gray"], transform=ax.transAxes)
        qax = ax.inset_axes([0.08, y0, 0.12, 0.16])
        qax.imshow(query.reshape(8, 8), cmap="gray_r", interpolation="nearest")
        qax.set_xticks([])
        qax.set_yticks([])
        for spine in qax.spines.values():
            spine.set_edgecolor(PALETTE["navy"])
            spine.set_linewidth(0.8)
        for cidx, nbr in enumerate(neighbors[:5]):
            nax = ax.inset_axes([0.26 + cidx * 0.11, y0, 0.09, 0.16])
            nax.imshow(nbr.reshape(8, 8), cmap="gray_r", interpolation="nearest")
            nax.set_xticks([])
            nax.set_yticks([])
            for spine in nax.spines.values():
                spine.set_visible(False)


def compute_case_study(npz_path: Path, k: int = 5) -> tuple[np.ndarray, list[np.ndarray], list[float], list[np.ndarray], list[list[np.ndarray]], int]:
    arr = np.load(npz_path)
    x = StandardScaler().fit_transform(arr["X"].astype(float))
    y = arr["y"]
    raw_x = arr["X"].astype(float)
    x_train, x_query, y_train, _, raw_train, raw_query = train_test_split(
        x, y, raw_x, test_size=0.35, stratify=y, random_state=20260617
    )
    dists = np.sqrt(((x_query[:, None, :] - x_train[None, :, :]) ** 2).sum(axis=2))
    topk = np.argsort(dists, axis=1, kind="mergesort")[:, :k]
    classes = np.unique(y_train)
    exposure_full = np.zeros(len(y_train), dtype=float)
    risk_values = []
    flagged_queries: list[int] = []
    flagged_neighbor_rows: list[list[np.ndarray]] = []
    for qidx, neigh in enumerate(topk):
        counts = np.array([(y_train[neigh] == c).sum() for c in classes])
        gap = np.sort(counts)[::-1]
        risk = 1.0 if (len(gap) > 1 and gap[0] - gap[1] <= 2) else 0.0
        risk_values.append(risk)
        if risk:
            exposure_full[neigh] += 1
            if len(flagged_queries) < 4:
                flagged_queries.append(qidx)
                flagged_neighbor_rows.append([raw_train[n] for n in neigh[:5]])
    top_ids = [i for i in np.argsort(-exposure_full)[:5] if exposure_full[i] > 0]
    top_scores = [float(exposure_full[i]) for i in top_ids]
    if top_scores:
        top_max = max(top_scores)
        top_scores = [score / top_max for score in top_scores]
    top_proto_images = [raw_train[i] for i in top_ids]
    query_images = [raw_query[i] for i in flagged_queries[:3]]
    return np.asarray(risk_values), top_proto_images, top_scores, query_images, flagged_neighbor_rows[:3], k


if __name__ == "__main__":
    main()
