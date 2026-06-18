"""Command runner for the PAA rebuild project.

The runner is designed for long jobs:

- timestamped log files plus JSONL event logs;
- lockfile protection against duplicate runs;
- tqdm progress bars for datasets, replicates, and baselines;
- explicit error reports that tell the user what failed and how to recover;
- proxy-aware dataset download hooks;
- checkpoint CSV files so interrupted runs can continue.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import json
import logging
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def import_or_fail(name: str):
    return importlib.import_module(name)


np = pd = None


@dataclass
class RunContext:
    root: Path
    config_path: Path
    config: dict[str, Any]
    run_id: str
    logger: logging.Logger
    event_log_path: Path
    lock_path: Path

    def event(self, event_type: str, **payload: Any) -> None:
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "event": event_type,
            **payload,
        }
        with self.event_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_yaml(path: Path) -> dict[str, Any]:
    yaml = import_or_fail("yaml")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_context(config_path: Path) -> RunContext:
    config = load_yaml(config_path)
    root = Path(config["project"]["root"])
    for key in ["raw_data_dir", "processed_data_dir", "tables_dir", "figures_dir", "logs_dir", "reports_dir"]:
        (root / config["outputs"][key]).mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = root / config["outputs"]["logs_dir"]
    log_path = logs_dir / f"{run_id}_paa_runner.log"
    event_log_path = logs_dir / f"{run_id}_events.jsonl"
    lock_path = logs_dir / "paa_runner.lock"

    logger = logging.getLogger(f"paa_runner_{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return RunContext(root, config_path, config, run_id, logger, event_log_path, lock_path)


@contextlib.contextmanager
def run_lock(ctx: RunContext, command: str):
    if ctx.lock_path.exists():
        raise RuntimeError(f"Lockfile exists: {ctx.lock_path}. Remove it only after confirming no runner is active.")
    ctx.lock_path.write_text(
        json.dumps({"run_id": ctx.run_id, "command": command, "pid": os.getpid(), "time": datetime.now().isoformat()}, indent=2),
        encoding="utf-8",
    )
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            ctx.lock_path.unlink()


def configure_proxy(ctx: RunContext) -> None:
    net = ctx.config.get("network", {})
    if net.get("use_proxy", False):
        os.environ["HTTP_PROXY"] = net.get("http_proxy", "")
        os.environ["HTTPS_PROXY"] = net.get("https_proxy", "")
        os.environ["http_proxy"] = net.get("http_proxy", "")
        os.environ["https_proxy"] = net.get("https_proxy", "")
        ctx.logger.info("Proxy enabled: %s", net.get("http_proxy"))


def write_error_report(ctx: RunContext, command: str, error: BaseException) -> Path:
    reports_dir = ctx.root / ctx.config["outputs"]["reports_dir"]
    path = reports_dir / f"{ctx.run_id}_{command}_ERROR.md"
    path.write_text(
        "\n".join(
            [
                f"# Error Report: {command}",
                "",
                f"- Run ID: `{ctx.run_id}`",
                f"- Time: `{datetime.now().isoformat(timespec='seconds')}`",
                f"- Python: `{sys.executable}`",
                f"- Platform: `{platform.platform()}`",
                "",
                "## Exception",
                "",
                "```text",
                "".join(traceback.format_exception(error)).strip(),
                "```",
                "",
                "## Suggested Recovery",
                "",
                "- Check the latest `.log` and `_events.jsonl` files under `logs/`.",
                "- If this is a dataset download failure, manually place the dataset under `data/raw/` and rerun `acquire-data`.",
                "- If this is an environment failure, activate `E:/anaconda3/envs/pytorch-clean` and install the missing package.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ctx.logger.error("Wrote error report: %s", path)
    return path


def cmd_env_check(ctx: RunContext) -> None:
    configure_proxy(ctx)
    ctx.logger.info("Python executable: %s", sys.executable)
    ctx.logger.info("Platform: %s", platform.platform())
    rows = []
    for package in ctx.config["environment"]["required_packages"] + ctx.config["environment"].get("optional_packages", []):
        module = package.replace("scikit-learn", "sklearn").replace("pyyaml", "yaml").replace("imbalanced-learn", "imblearn")
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "unknown")
            status = "ok"
        except Exception as exc:
            version = ""
            status = f"missing: {exc.__class__.__name__}"
        rows.append({"package": package, "module": module, "status": status, "version": version})
        ctx.logger.info("Package %-24s %-10s %s", package, status, version)
    out = ctx.root / ctx.config["outputs"]["reports_dir"] / "environment_check.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["package", "module", "status", "version"])
        writer.writeheader()
        writer.writerows(rows)
    ctx.event("env_check_complete", report=str(out))


def ensure_numpy_pandas():
    global np, pd
    if np is None:
        np = import_or_fail("numpy")
    if pd is None:
        pd = import_or_fail("pandas")


def load_dataset_specs(ctx: RunContext):
    ensure_numpy_pandas()
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

    seed = int(ctx.config["execution"]["random_seed"])
    rng = np.random.RandomState(seed)
    specs = []
    processed = ctx.root / ctx.config["outputs"]["processed_data_dir"]
    manifest_path = processed / "dataset_manifest.csv"
    manifest_rows: list[dict[str, Any]] = []
    if manifest_path.exists():
        try:
            manifest_rows = pd.read_csv(manifest_path).to_dict("records")
        except Exception:
            manifest_rows = []

    def persist(name: str, group: str, X: Any, y: Any, source: str) -> None:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        file_name = f"{safe_name(name)}.npz"
        np.savez_compressed(processed / file_name, X=X_arr, y=y_arr)
        row = {
            "dataset": name,
            "group": group,
            "source": source,
            "rows": int(len(y_arr)),
            "features": int(X_arr.shape[1]),
            "classes": int(len(np.unique(y_arr))),
            "file": file_name,
        }
        nonlocal manifest_rows
        manifest_rows = [r for r in manifest_rows if r.get("dataset") != name]
        manifest_rows.append(row)
        pd.DataFrame(manifest_rows).sort_values("dataset").to_csv(manifest_path, index=False)
        ctx.event("dataset_cached", dataset=name, rows=row["rows"], features=row["features"], source=source)

    def add(name, group, X, y, source):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        persist(name, group, X_arr, y_arr, source)
        specs.append({"name": name, "group": group, "X": X_arr, "y": y_arr, "source": source})

    iris = load_iris()
    add("Iris", "tabular", iris.data, iris.target, "sklearn")
    wine = load_wine()
    add("Wine", "tabular", wine.data, wine.target, "sklearn")
    bc = load_breast_cancer()
    add("Breast Cancer Wisconsin", "medical/tabular", bc.data, bc.target, "sklearn")
    digits = load_digits()
    add("Digits-10", "image", digits.data, digits.target, "sklearn")
    mask = np.isin(digits.target, [0, 8])
    add("Digits-0-vs-8", "image/binary", digits.data[mask], digits.target[mask], "sklearn")

    X, y = make_moons(n_samples=900, noise=0.18, random_state=rng)
    add("Two Moons", "synthetic nonlinear", X, y, "sklearn synthetic")
    X, y = make_circles(n_samples=900, noise=0.12, factor=0.45, random_state=rng)
    add("Concentric Circles", "synthetic nonlinear", X, y, "sklearn synthetic")
    X, y = make_blobs(n_samples=1000, centers=4, n_features=8, cluster_std=[1.0, 1.5, 2.0, 1.2], random_state=rng)
    add("Four Blobs", "synthetic multiclass", X, y, "sklearn synthetic")

    synthetic_defs = [
        ("Noisy Binary", dict(n_samples=1000, n_features=20, n_informative=8, n_redundant=4, n_classes=2, flip_y=0.08, class_sep=1.0)),
        ("Imbalanced Binary", dict(n_samples=1000, n_features=18, n_informative=7, n_redundant=3, n_classes=2, weights=[0.82, 0.18], flip_y=0.04, class_sep=0.9)),
        ("Noisy Multiclass", dict(n_samples=1100, n_features=24, n_informative=10, n_redundant=4, n_classes=4, flip_y=0.08, class_sep=1.0)),
        ("Low-Separation Multiclass", dict(n_samples=1100, n_features=16, n_informative=8, n_redundant=2, n_classes=3, flip_y=0.04, class_sep=0.55)),
        ("High-Dimensional Sparse", dict(n_samples=1000, n_features=80, n_informative=10, n_redundant=5, n_classes=3, flip_y=0.05, class_sep=0.9)),
        ("Redundant Multiclass", dict(n_samples=1000, n_features=36, n_informative=8, n_redundant=18, n_classes=3, flip_y=0.03, class_sep=1.0)),
    ]
    for name, kwargs in synthetic_defs:
        X, y = make_classification(random_state=rng, **kwargs)
        add(name, "synthetic tabular", X, y, "sklearn synthetic")

    openml_cfg = ctx.config["datasets"].get("openml", {})
    if openml_cfg.get("enabled", False):
        for name in openml_cfg.get("datasets", []):
            try:
                ctx.logger.info("Downloading OpenML dataset: %s", name)
                data = fetch_openml(name=name, version="active", as_frame=True, parser="auto")
                frame = data.frame.dropna()
                y = frame[data.target_names[0]].astype("category").cat.codes.to_numpy()
                Xdf = frame.drop(columns=data.target_names)
                Xdf = pd.get_dummies(Xdf, dummy_na=False)
                add(name, "OpenML/UCI", Xdf.to_numpy(dtype=float), y, "OpenML")
                ctx.event("openml_download_ok", dataset=name, rows=int(len(y)), features=int(Xdf.shape[1]))
            except Exception as exc:
                ctx.event("openml_download_failed", dataset=name, error=str(exc))
                ctx.logger.warning("OpenML download failed for %s: %s", name, exc)
                if ctx.config["execution"].get("stop_on_download_error", False):
                    raise
    return specs


def cmd_acquire_data(ctx: RunContext) -> None:
    configure_proxy(ctx)
    ensure_numpy_pandas()
    tqdm = import_or_fail("tqdm").tqdm
    specs = load_dataset_specs(ctx)
    processed = ctx.root / ctx.config["outputs"]["processed_data_dir"]
    manifest_path = processed / "dataset_manifest.csv"
    count = len(pd.read_csv(manifest_path)) if manifest_path.exists() else 0
    for _ in tqdm(range(count), desc="Cached datasets"):
        pass
    ctx.event("acquire_data_complete", datasets=count, manifest=str(manifest_path))


def safe_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def cmd_run_benchmark(ctx: RunContext) -> None:
    configure_proxy(ctx)
    ensure_numpy_pandas()
    tqdm = import_or_fail("tqdm").tqdm
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    bench = importlib.import_module("paa_multiclass_benchmark")

    out_dir = ctx.root / ctx.config["outputs"]["tables_dir"]
    args = argparse.Namespace(
        output_dir=str(out_dir),
        processed_data_dir=str(ctx.root / ctx.config["outputs"]["processed_data_dir"]),
        seed=int(ctx.config["execution"]["random_seed"]),
        max_samples=int(ctx.config["execution"]["max_samples_per_dataset"]),
        replicates=int(ctx.config["execution"]["replicates"]),
        k_values=list(ctx.config["execution"]["k_values"]),
        loo_epsilon=float(ctx.config["execution"]["loo_epsilon"]),
        cleaning_k=int(ctx.config["execution"]["cleaning_k"]),
    )
    ctx.logger.info("Running benchmark with args: %s", args)
    bench.run(args)
    ctx.event("benchmark_complete", output_dir=str(out_dir))


def cmd_build_reports(ctx: RunContext) -> None:
    ensure_numpy_pandas()
    tables = ctx.root / ctx.config["outputs"]["tables_dir"]
    reports = ctx.root / ctx.config["outputs"]["reports_dir"]
    report_path = reports / "PAA_REBUILD_RUN_SUMMARY.md"
    files = sorted(p.name for p in tables.glob("*"))
    text = [
        "# PAA Rebuild Run Summary",
        "",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Config: `{ctx.config_path}`",
        "",
        "## Result Files",
        "",
    ]
    text.extend(f"- `{name}`" for name in files)
    text += [
        "",
        "## Manuscript Integration",
        "",
        "- Use `paa_datasets_table.tex` for the benchmark protocol table.",
        "- Use `paa_diagnostic_comparison_table.tex` for diagnostic baselines.",
        "- Use `paa_margin_k_selection_table.tex` for margin-aware k selection.",
        "- Use `paa_cleaning_baselines_table.tex` for prototype editing / cleaning baselines.",
    ]
    report_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    ctx.event("reports_complete", report=str(report_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["env-check", "acquire-data", "run-benchmark", "build-reports"])
    parser.add_argument("--config", default="config/paa_rebuild_config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    ctx = setup_context(config_path)
    try:
        with run_lock(ctx, args.command):
            ctx.event("command_start", command=args.command)
            if args.command == "env-check":
                cmd_env_check(ctx)
            elif args.command == "acquire-data":
                cmd_acquire_data(ctx)
            elif args.command == "run-benchmark":
                cmd_run_benchmark(ctx)
            elif args.command == "build-reports":
                cmd_build_reports(ctx)
            ctx.event("command_success", command=args.command)
    except Exception as exc:
        ctx.event("command_error", command=args.command, error=str(exc))
        write_error_report(ctx, args.command, exc)
        raise


if __name__ == "__main__":
    main()
