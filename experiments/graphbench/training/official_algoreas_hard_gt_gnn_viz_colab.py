#!/usr/bin/env python3
"""Visualise hard-task official GraphBench GT-vs-GNN screening results.

Standalone Colab-friendly script. It scans the Drive results produced by
``official_algoreas_screen_colab.py`` and compares graph-transformer models
against GNN+ baselines for whichever hard tasks are present.

Default comparison:
    GT:  grit
    GNN: gcn_plus,gatedgcn_plus

The script does not require a top-level aggregate CSV. It reconstructs results
from per-run ``summary.json`` files and uses ``metrics.csv`` when available for
learning-curve plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence


RUN_NAME = "official_algoreas_2layer_screen"
BINARY_TYPES = {"edge_binary", "node_binary"}
REGRESSION_TYPES = {"node_regression", "graph_regression"}
DEFAULT_GT_MODELS = ("grit",)
DEFAULT_GNN_MODELS = ("gcn_plus", "gatedgcn_plus")


def install_package_if_missing(import_name: str, package_name: str, log=print) -> None:
    try:
        __import__(import_name)
    except Exception:
        log(f"[deps] Installing {package_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])


def import_plotting():
    install_package_if_missing("matplotlib", "matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def mount_drive(mount_point: Path, enabled: bool, log=print) -> None:
    if not enabled:
        log("[drive] Drive mount disabled.")
        return
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        log("[drive] Not running in Colab; skipping Drive mount.")
        return
    log(f"[drive] Mounting Google Drive at {mount_point}")
    drive.mount(str(mount_point), force_remount=False)


def parse_model_list(value: str) -> tuple[str, ...]:
    models = tuple(part.strip() for part in value.split(",") if part.strip())
    if not models:
        raise argparse.ArgumentTypeError("model list cannot be empty")
    return models


def parse_task_list(value: Optional[str]) -> Optional[tuple[str, ...]]:
    if value is None or not value.strip():
        return None
    tasks = tuple(part.strip() for part in value.split(",") if part.strip())
    return tasks or None


def parse_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def split_task(task: str) -> tuple[str, str]:
    if "_" not in task:
        return task, "unknown"
    return task.rsplit("_", 1)


def task_sort_key(task: str) -> tuple[int, str]:
    base, difficulty = split_task(task)
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}.get(difficulty, 9)
    return difficulty_order, base


def metric_key(task_type: str) -> str:
    if task_type in BINARY_TYPES:
        return "f1"
    if task_type in REGRESSION_TYPES:
        return "mae"
    return "primary"


def metric_label(task_type: str) -> str:
    key = metric_key(task_type)
    return key.upper() if key != "primary" else "primary"


def higher_is_better(task_type: str) -> bool:
    return task_type in BINARY_TYPES or task_type not in REGRESSION_TYPES


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan"), float("nan")
    mean = sum(finite) / len(finite)
    if len(finite) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return mean, math.sqrt(max(0.0, var))


def is_run_summary(data: Mapping[str, object]) -> bool:
    needed = ("task", "task_type", "model", "seed", "train_metrics", "val_metrics", "test_metrics")
    return all(key in data for key in needed)


def discover_candidate_roots(results_dir: Path, drive_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for candidate in (
        results_dir,
        drive_dir / "results" / RUN_NAME,
        drive_dir / "results",
        drive_dir,
    ):
        if candidate not in roots:
            roots.append(candidate)
    return roots


def collect_run_summaries(root: Path) -> list[dict[str, object]]:
    """Collect per-run summaries, preserving run directory when available."""
    if not root.exists():
        return []
    records: dict[tuple[str, str, str], dict[str, object]] = {}
    for path in sorted(root.rglob("summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidates: list[tuple[Mapping[str, object], Optional[Path]]] = []
        if isinstance(data, Mapping) and is_run_summary(data):
            candidates.append((data, path.parent))
        elif isinstance(data, Mapping) and isinstance(data.get("runs"), list):
            for run in data["runs"]:  # type: ignore[index]
                if isinstance(run, Mapping) and is_run_summary(run):
                    candidates.append((run, None))
        for summary, run_dir in candidates:
            key = (str(summary["task"]), str(summary["model"]), str(summary["seed"]))
            row = dict(summary)
            if run_dir is not None:
                row["_run_dir"] = str(run_dir)
            elif key in records and "_run_dir" in records[key]:
                row["_run_dir"] = records[key]["_run_dir"]
            records[key] = row
    return list(records.values())


def collect_from_roots(roots: Sequence[Path]) -> tuple[list[dict[str, object]], Optional[Path]]:
    for root in roots:
        summaries = collect_run_summaries(root)
        if summaries:
            return summaries, root
    return [], None


def available_result_hints(root: Path) -> list[str]:
    if not root.exists():
        return []
    hints = []
    for file_name in ("summary.json", "aggregate_metrics.csv", "metrics.csv"):
        for path in sorted(root.rglob(file_name))[:30]:
            hints.append(str(path))
    return hints[:40]


def filter_summaries(
    summaries: Sequence[Mapping[str, object]],
    difficulty: str,
    tasks: Optional[Sequence[str]],
    models: set[str],
) -> list[dict[str, object]]:
    selected = []
    task_set = set(tasks or ())
    for summary in summaries:
        task = str(summary["task"])
        model = str(summary["model"])
        _, task_difficulty = split_task(task)
        if task_set and task not in task_set:
            continue
        if not task_set and difficulty != "all" and task_difficulty != difficulty:
            continue
        if model not in models:
            continue
        selected.append(dict(summary))
    return selected


def value_from_metrics(metrics: Mapping[str, object], key: str) -> float:
    value = parse_float(metrics.get(key))
    if math.isfinite(value):
        return value
    return parse_float(metrics.get("primary"))


def aggregate_model_rows(summaries: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for summary in summaries:
        grouped[(str(summary["task"]), str(summary["model"]))].append(summary)
    rows: list[dict[str, object]] = []
    for (task, model), runs in sorted(grouped.items(), key=lambda item: (task_sort_key(item[0][0]), item[0][1])):
        task_type = str(runs[0]["task_type"])
        key = metric_key(task_type)
        row: dict[str, object] = {
            "task": task,
            "task_base": split_task(task)[0],
            "difficulty": split_task(task)[1],
            "task_type": task_type,
            "metric": key,
            "metric_label": metric_label(task_type),
            "direction": "higher" if higher_is_better(task_type) else "lower",
            "model": model,
            "n_seeds": len(runs),
        }
        for split in ("train", "val", "test"):
            vals = [value_from_metrics(dict(run[f"{split}_metrics"]), key) for run in runs]
            mean, std = mean_std(vals)
            row[f"{split}_score_mean"] = mean
            row[f"{split}_score_std"] = std
            for extra_key in ("primary", "loss", "f1", "accuracy", "mae", "mse", "r2", "spearman"):
                extra_vals = [
                    parse_float(dict(run[f"{split}_metrics"]).get(extra_key))
                    for run in runs
                ]
                extra_mean, extra_std = mean_std(extra_vals)
                row[f"{split}_{extra_key}_mean"] = extra_mean
                row[f"{split}_{extra_key}_std"] = extra_std
        rows.append(row)
    return rows


def best_family_rows(
    model_rows: Sequence[Mapping[str, object]],
    gt_models: Sequence[str],
    gnn_models: Sequence[str],
) -> list[dict[str, object]]:
    by_task = defaultdict(list)
    for row in model_rows:
        by_task[str(row["task"])].append(row)
    rows: list[dict[str, object]] = []
    for task, task_rows in sorted(by_task.items(), key=lambda item: task_sort_key(item[0])):
        task_type = str(task_rows[0]["task_type"])
        key_higher = higher_is_better(task_type)
        gt_candidates = [row for row in task_rows if str(row["model"]) in gt_models]
        gnn_candidates = [row for row in task_rows if str(row["model"]) in gnn_models]
        if not gt_candidates or not gnn_candidates:
            continue
        choose = max if key_higher else min
        best_gt = choose(gt_candidates, key=lambda row: float(row["test_score_mean"]))
        best_gnn = choose(gnn_candidates, key=lambda row: float(row["test_score_mean"]))
        gt_score = float(best_gt["test_score_mean"])
        gnn_score = float(best_gnn["test_score_mean"])
        advantage = gt_score - gnn_score if key_higher else gnn_score - gt_score
        rows.append(
            {
                "task": task,
                "task_base": split_task(task)[0],
                "difficulty": split_task(task)[1],
                "task_type": task_type,
                "metric": str(best_gt["metric"]),
                "metric_label": str(best_gt["metric_label"]),
                "direction": str(best_gt["direction"]),
                "best_gt_model": str(best_gt["model"]),
                "best_gnn_model": str(best_gnn["model"]),
                "best_gt_test": gt_score,
                "best_gnn_test": gnn_score,
                "best_gt_val": float(best_gt["val_score_mean"]),
                "best_gnn_val": float(best_gnn["val_score_mean"]),
                "best_gt_train": float(best_gt["train_score_mean"]),
                "best_gnn_train": float(best_gnn["train_score_mean"]),
                "gt_advantage": advantage,
                "relative_gt_advantage": advantage / max(1.0e-12, abs(gnn_score)),
                "winner_family": "GT" if advantage > 0 else ("GNN" if advantage < 0 else "tie"),
            }
        )
    return rows


def read_metric_curves(summaries: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen = set()
    for summary in summaries:
        run_dir_value = summary.get("_run_dir")
        if not run_dir_value:
            continue
        metrics_path = Path(str(run_dir_value)) / "metrics.csv"
        if not metrics_path.exists():
            continue
        task = str(summary["task"])
        model = str(summary["model"])
        seed = str(summary["seed"])
        for row in read_csv_rows(metrics_path):
            key = (task, model, seed, row.get("epoch", ""))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "seed": seed,
                    "epoch": int(float(row.get("epoch", 0) or 0)),
                    "train_loss": parse_float(row.get("train_loss")),
                    "val_primary": parse_float(row.get("val_primary")),
                    "val_loss": parse_float(row.get("val_loss")),
                    "val_f1": parse_float(row.get("val_f1")),
                    "val_mae": parse_float(row.get("val_mae")),
                }
            )
    return rows


def summarise(family_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    gt_wins = sum(1 for row in family_rows if row["winner_family"] == "GT")
    gnn_wins = sum(1 for row in family_rows if row["winner_family"] == "GNN")
    ties = len(family_rows) - gt_wins - gnn_wins
    mean_adv = float("nan")
    if family_rows:
        mean_adv = sum(float(row["gt_advantage"]) for row in family_rows) / len(family_rows)
    return {
        "n_tasks": len(family_rows),
        "gt_wins": gt_wins,
        "gnn_wins": gnn_wins,
        "ties": ties,
        "mean_gt_advantage": mean_adv,
        "tasks": [str(row["task"]) for row in family_rows],
    }


def plot_family_advantage(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    if not rows:
        return
    plt = import_plotting()
    ordered = sorted(rows, key=lambda row: float(row["gt_advantage"]))
    labels = [str(row["task"]) for row in ordered]
    values = [float(row["gt_advantage"]) for row in ordered]
    colors = ["#238b45" if value > 0 else "#cb181d" if value < 0 else "#969696" for value in values]
    fig, ax = plt.subplots(figsize=(8.8, max(3.3, 0.48 * len(labels))))
    ax.barh(range(len(labels)), values, color=colors)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("best GT advantage over best GNN\n(F1 diff, or GNN MAE - GT MAE)")
    ax.set_title("Hard GraphBench: GT vs GNN OOD advantage")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(values):
        ha = "left" if value >= 0 else "right"
        offset = 0.006 if value >= 0 else -0.006
        ax.text(value + offset, i, f"{value:+.3f}", va="center", ha=ha, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_paired_scores(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    if not rows:
        return
    plt = import_plotting()
    groups = [
        ("Binary tasks", [row for row in rows if str(row["task_type"]) in BINARY_TYPES], "test F1"),
        ("Regression tasks", [row for row in rows if str(row["task_type"]) in REGRESSION_TYPES], "test MAE"),
    ]
    groups = [group for group in groups if group[1]]
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(5.4 * len(groups), max(3.6, 0.42 * len(rows))), squeeze=False)
    for ax, (title, group, xlabel) in zip(axes[0], groups):
        labels = [str(row["task"]) for row in group]
        y = list(range(len(group)))
        gt = [float(row["best_gt_test"]) for row in group]
        gnn = [float(row["best_gnn_test"]) for row in group]
        ax.scatter(gt, y, label="best GT", color="#2171b5", s=42, zorder=3)
        ax.scatter(gnn, y, label="best GNN", color="#ef3b2c", s=42, zorder=3)
        for yi, a, b in zip(y, gt, gnn):
            ax.plot([a, b], [yi, yi], color="#bdbdbd", linewidth=1.1, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        if "MAE" in xlabel:
            ax.invert_xaxis()
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Hard-task OOD test scores", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_score_heatmaps(model_rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    if not model_rows:
        return
    plt = import_plotting()
    task_types = []
    for kind in ("binary", "regression", "other"):
        if kind == "binary":
            selected = [row for row in model_rows if str(row["task_type"]) in BINARY_TYPES]
        elif kind == "regression":
            selected = [row for row in model_rows if str(row["task_type"]) in REGRESSION_TYPES]
        else:
            selected = [
                row for row in model_rows
                if str(row["task_type"]) not in BINARY_TYPES and str(row["task_type"]) not in REGRESSION_TYPES
            ]
        if selected:
            task_types.append((kind, selected))
    fig, axes = plt.subplots(1, len(task_types), figsize=(5.2 * len(task_types), 3.9), squeeze=False)
    for ax, (kind, rows) in zip(axes[0], task_types):
        tasks = sorted({str(row["task"]) for row in rows}, key=task_sort_key)
        models = sorted({str(row["model"]) for row in rows})
        by_key = {(str(row["model"]), str(row["task"])): row for row in rows}
        data = []
        for model in models:
            data.append([
                parse_float(by_key.get((model, task), {}).get("test_score_mean"))
                for task in tasks
            ])
        finite = [value for row in data for value in row if math.isfinite(value)]
        if finite:
            vmin, vmax = min(finite), max(finite)
            if abs(vmax - vmin) < 1.0e-12:
                vmin, vmax = vmin - 0.5, vmax + 0.5
        else:
            vmin, vmax = 0.0, 1.0
        cmap = "viridis" if kind != "regression" else "viridis_r"
        image = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(tasks, rotation=35, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_title("Binary test F1" if kind == "binary" else "Regression test MAE" if kind == "regression" else "Test score")
        for i, model in enumerate(models):
            for j, task in enumerate(tasks):
                value = data[i][j]
                if math.isfinite(value):
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8, color="white")
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("Hard-task model score heatmaps", y=1.03)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_val_test_gap(model_rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    if not model_rows:
        return
    plt = import_plotting()
    tasks = sorted({str(row["task"]) for row in model_rows}, key=task_sort_key)
    models = sorted({str(row["model"]) for row in model_rows})
    by_key = {(str(row["model"]), str(row["task"])): row for row in model_rows}
    x = list(range(len(tasks)))
    width = 0.78 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(max(8.6, 1.0 * len(tasks)), 4.3))
    colors = ["#2171b5", "#ef3b2c", "#756bb1", "#31a354", "#fd8d3c"]
    for model_index, model in enumerate(models):
        vals = []
        for task in tasks:
            row = by_key.get((model, task))
            if row is None:
                vals.append(float("nan"))
                continue
            if higher_is_better(str(row["task_type"])):
                vals.append(float(row["val_score_mean"]) - float(row["test_score_mean"]))
            else:
                vals.append(float(row["test_score_mean"]) - float(row["val_score_mean"]))
        offsets = [i + (model_index - (len(models) - 1) / 2) * width for i in x]
        ax.bar(offsets, vals, width=width, label=model, color=colors[model_index % len(colors)])
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=35, ha="right")
    ax.set_ylabel("val-to-test degradation\n(F1 drop, or MAE increase)")
    ax.set_title("Hard-task size-generalisation gap")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(curve_rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    if not curve_rows:
        return
    plt = import_plotting()
    tasks = sorted({str(row["task"]) for row in curve_rows}, key=task_sort_key)
    ncols = min(2, len(tasks))
    nrows = math.ceil(len(tasks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.2 * nrows), squeeze=False)
    colors = {"grit": "#2171b5", "gcn_plus": "#ef3b2c", "gatedgcn_plus": "#756bb1"}
    for ax in axes.ravel():
        ax.axis("off")
    for index, task in enumerate(tasks):
        ax = axes[index // ncols][index % ncols]
        ax.axis("on")
        task_rows = [row for row in curve_rows if str(row["task"]) == task]
        models = sorted({str(row["model"]) for row in task_rows})
        for model in models:
            model_rows = sorted([row for row in task_rows if str(row["model"]) == model], key=lambda row: int(row["epoch"]))
            epochs = [int(row["epoch"]) for row in model_rows]
            vals = [float(row["val_primary"]) for row in model_rows]
            ax.plot(epochs, vals, label=model, color=colors.get(model, None), linewidth=1.7)
        ax.set_title(task)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val primary")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Validation curves from metrics.csv", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(
    path: Path,
    family_rows: Sequence[Mapping[str, object]],
    model_rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    lines = [
        "# Hard GraphBench GT vs GNN Summary",
        "",
        "Positive advantage means best GT is better than best GNN. For binary tasks this is `GT F1 - GNN F1`; for regression tasks this is `GNN MAE - GT MAE`.",
        "",
        f"Tasks compared: {summary.get('n_tasks', 0)}",
        f"GT wins: {summary.get('gt_wins', 0)}",
        f"GNN wins: {summary.get('gnn_wins', 0)}",
        f"Ties: {summary.get('ties', 0)}",
        f"Mean GT advantage: {float(summary.get('mean_gt_advantage', float('nan'))):+.4f}",
        "",
        "## Best-Family Comparison",
        "",
        "| Task | Type | Metric | Best GT | GT Test | Best GNN | GNN Test | Advantage | Winner |",
        "|---|---|---:|---|---:|---|---:|---:|---|",
    ]
    for row in sorted(family_rows, key=lambda item: float(item["gt_advantage"]), reverse=True):
        lines.append(
            f"| {row['task']} | {row['task_type']} | {row['metric_label']} | "
            f"{row['best_gt_model']} | {float(row['best_gt_test']):.4f} | "
            f"{row['best_gnn_model']} | {float(row['best_gnn_test']):.4f} | "
            f"{float(row['gt_advantage']):+.4f} | {row['winner_family']} |"
        )
    lines.extend(["", "## Per-Model Test Scores", ""])
    lines.append("| Task | Model | Metric | Test Mean | Test Std | Val Mean | Seeds |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in sorted(model_rows, key=lambda item: (task_sort_key(str(item["task"])), str(item["model"]))):
        lines.append(
            f"| {row['task']} | {row['model']} | {row['metric_label']} | "
            f"{float(row['test_score_mean']):.4f} | {float(row['test_score_std']):.4f} | "
            f"{float(row['val_score_mean']):.4f} | {int(row['n_seeds'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def strip_colab_kernel_args(argv: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in {"-f", "--f", "--file"}:
            skip_next = True
            continue
        if "jupyter/runtime/kernel-" in arg or (arg.endswith(".json") and "kernel-" in arg):
            continue
        cleaned.append(arg)
    return cleaned


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise official GraphBench hard-task GT-vs-GNN results.")
    parser.add_argument("--drive-dir", type=Path, default=Path("/content/drive/MyDrive/graph_specialisation_metrics/official_algoreas_screen"))
    parser.add_argument("--drive-mount", type=Path, default=Path("/content/drive"))
    parser.add_argument("--no-mount-drive", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard", "all"), default="hard")
    parser.add_argument("--tasks", type=str, default=None, help="Optional comma-separated exact task names.")
    parser.add_argument("--gt-models", type=parse_model_list, default=DEFAULT_GT_MODELS)
    parser.add_argument("--gnn-models", type=parse_model_list, default=DEFAULT_GNN_MODELS)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = strip_colab_kernel_args(list(sys.argv[1:] if argv is None else argv))
    args = parse_args(argv)
    mount_drive(args.drive_mount, enabled=not args.no_mount_drive)

    results_dir = args.results_dir or args.drive_dir / "results" / RUN_NAME
    roots = discover_candidate_roots(results_dir, args.drive_dir)
    summaries, resolved_root = collect_from_roots(roots)
    if not summaries:
        hints = available_result_hints(args.drive_dir)
        hint_text = "\n".join(f"  - {hint}" for hint in hints) if hints else "  - no result files found"
        raise FileNotFoundError(
            f"No run summaries found from {results_dir} or {args.drive_dir}.\n"
            f"Discovered files:\n{hint_text}"
        )

    tasks = parse_task_list(args.tasks)
    selected_models = set(args.gt_models) | set(args.gnn_models)
    selected = filter_summaries(summaries, args.difficulty, tasks, selected_models)
    if not selected:
        available_tasks = sorted({str(summary["task"]) for summary in summaries}, key=task_sort_key)
        available_models = sorted({str(summary["model"]) for summary in summaries})
        raise RuntimeError(
            f"No matching summaries for difficulty={args.difficulty}, tasks={tasks}, models={sorted(selected_models)}.\n"
            f"Available tasks: {available_tasks}\nAvailable models: {available_models}"
        )

    model_rows = aggregate_model_rows(selected)
    family_rows = best_family_rows(model_rows, args.gt_models, args.gnn_models)
    if not family_rows:
        available_by_task = defaultdict(list)
        for row in model_rows:
            available_by_task[str(row["task"])].append(str(row["model"]))
        raise RuntimeError(
            "No complete GT-vs-GNN task comparisons found. "
            f"Available model rows by task: {dict(sorted(available_by_task.items()))}"
        )

    output_dir = args.output_dir or (resolved_root or results_dir) / f"visualisations_{args.difficulty}_gt_vs_gnn"
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_rows = read_metric_curves(selected)
    summary = summarise(family_rows)

    write_csv_rows(output_dir / "hard_model_metrics.csv", model_rows)
    write_csv_rows(output_dir / "hard_best_gt_vs_best_gnn.csv", family_rows)
    write_csv_rows(output_dir / "hard_learning_curves.csv", curve_rows)
    write_json(output_dir / "hard_gt_vs_gnn_summary.json", {"summary": summary, "best_family_rows": family_rows})
    write_markdown_summary(output_dir / "hard_gt_vs_gnn_summary.md", family_rows, model_rows, summary)

    plot_family_advantage(family_rows, output_dir / "hard_gt_advantage_over_gnn.png")
    plot_paired_scores(family_rows, output_dir / "hard_paired_ood_test_scores.png")
    plot_score_heatmaps(model_rows, output_dir / "hard_model_score_heatmaps.png")
    plot_val_test_gap(model_rows, output_dir / "hard_val_to_test_generalisation_gap.png")
    plot_learning_curves(curve_rows, output_dir / "hard_validation_curves.png")

    print(f"[data] Using summaries from {resolved_root}", flush=True)
    print(f"[done] Wrote hard-task GT-vs-GNN visualisations to {output_dir}", flush=True)
    print(
        f"[summary] tasks={summary['n_tasks']} gt_wins={summary['gt_wins']} "
        f"gnn_wins={summary['gnn_wins']} ties={summary['ties']} "
        f"mean_gt_advantage={float(summary['mean_gt_advantage']):+.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
