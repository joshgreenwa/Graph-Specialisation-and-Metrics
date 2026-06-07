#!/usr/bin/env python3
"""Visualise official GraphBench AlgoReas GRIT-vs-GCN+ screening results.

This script is standalone and Colab-friendly. It reads outputs produced by
``official_algoreas_screen_colab.py`` and writes compact figures plus summary
tables comparing ``grit`` against ``gcn_plus``.
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
MODEL_A = "grit"
MODEL_B = "gcn_plus"


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
    fieldnames = sorted({field for row in rows for field in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


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
    base, difficulty = task.rsplit("_", 1)
    return base, difficulty


def task_sort_key(task: str) -> tuple[int, str]:
    base, difficulty = split_task(task)
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}.get(difficulty, 9)
    return difficulty_order, base


def metric_name_for_task(task_type: str) -> str:
    if task_type in BINARY_TYPES:
        return "F1"
    if task_type in REGRESSION_TYPES:
        return "MAE"
    return "primary"


def higher_is_better(task_type: str) -> bool:
    return task_type in BINARY_TYPES


def load_aggregate_from_summary(path: Path) -> list[dict[str, object]]:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return []
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    aggregate = data.get("aggregate", [])
    return [dict(row) for row in aggregate]


def is_run_summary(data: Mapping[str, object]) -> bool:
    return all(key in data for key in ("task", "task_type", "model", "seed", "train_metrics", "val_metrics", "test_metrics"))


def collect_run_summaries(root: Path) -> list[dict[str, object]]:
    if not root.exists():
        return []
    summaries = []
    for path in sorted(root.rglob("summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, Mapping) and "runs" in data and isinstance(data["runs"], list):
            for run in data["runs"]:
                if isinstance(run, Mapping) and is_run_summary(run):
                    summaries.append(dict(run))
        elif isinstance(data, Mapping) and is_run_summary(data):
            summaries.append(dict(data))
    unique = {}
    for summary in summaries:
        key = (str(summary["task"]), str(summary["model"]), str(summary["seed"]))
        unique[key] = summary
    return list(unique.values())


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan"), float("nan")
    mean = sum(finite) / len(finite)
    if len(finite) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in finite) / (len(finite) - 1)
    return mean, math.sqrt(max(0.0, var))


def aggregate_from_run_summaries(summaries: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for summary in summaries:
        grouped[(str(summary["task"]), str(summary["model"]))].append(summary)
    rows = []
    for (task, model), runs in sorted(grouped.items(), key=lambda item: (task_sort_key(item[0][0]), item[0][1])):
        row: dict[str, object] = {
            "task": task,
            "task_type": str(runs[0]["task_type"]),
            "model": model,
            "n_seeds": len(runs),
        }
        for split in ("train", "val", "test"):
            metric_keys = sorted({key for run in runs for key in dict(run[f"{split}_metrics"]).keys()})
            for metric in metric_keys:
                vals = [parse_float(dict(run[f"{split}_metrics"]).get(metric)) for run in runs]
                mean, std = mean_std(vals)
                row[f"{split}_{metric}_mean"] = mean
                row[f"{split}_{metric}_std"] = std
        rows.append(row)
    return rows


def load_aggregate(results_dir: Path) -> list[dict[str, object]]:
    csv_rows = read_csv_rows(results_dir / "aggregate_metrics.csv")
    if csv_rows:
        return [dict(row) for row in csv_rows]
    aggregate = load_aggregate_from_summary(results_dir)
    if aggregate:
        return aggregate
    summaries = collect_run_summaries(results_dir)
    if summaries:
        return aggregate_from_run_summaries(summaries)
    return []


def load_aggregate_with_discovery(results_dir: Path, drive_dir: Path) -> tuple[list[dict[str, object]], Path]:
    candidate_roots = []
    for candidate in [
        results_dir,
        drive_dir / "results" / RUN_NAME,
        drive_dir / "results",
        drive_dir,
    ]:
        if candidate not in candidate_roots:
            candidate_roots.append(candidate)
    for candidate in candidate_roots:
        aggregate = load_aggregate(candidate)
        if aggregate:
            return aggregate, candidate
    for root in [candidate for candidate in candidate_roots if candidate.exists()]:
        for file_name in ("aggregate_metrics.csv", "summary.json"):
            for path in sorted(root.rglob(file_name)):
                aggregate = load_aggregate(path.parent)
                if aggregate:
                    return aggregate, path.parent
    return [], results_dir


def available_result_hints(root: Path) -> list[str]:
    if not root.exists():
        return []
    hints = []
    for file_name in ("aggregate_metrics.csv", "summary.json"):
        for path in sorted(root.rglob(file_name))[:20]:
            hints.append(str(path))
    return hints


def complete_pair_rows(aggregate: Sequence[Mapping[str, object]], model_a: str, model_b: str) -> list[dict[str, object]]:
    by_task_model: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in aggregate:
        by_task_model[(str(row["task"]), str(row["model"]))] = row
    tasks = sorted({str(row["task"]) for row in aggregate}, key=task_sort_key)
    rows = []
    for task in tasks:
        a = by_task_model.get((task, model_a))
        b = by_task_model.get((task, model_b))
        if a is None or b is None:
            continue
        task_type = str(a["task_type"])
        base, difficulty = split_task(task)
        if task_type in BINARY_TYPES:
            a_test = parse_float(a.get("test_f1_mean"))
            b_test = parse_float(b.get("test_f1_mean"))
            a_val = parse_float(a.get("val_f1_mean"))
            b_val = parse_float(b.get("val_f1_mean"))
            a_train = parse_float(a.get("train_f1_mean"))
            b_train = parse_float(b.get("train_f1_mean"))
            advantage = a_test - b_test
            relative_advantage = advantage / max(1.0e-12, abs(b_test))
            metric = "f1"
            direction = "higher"
        else:
            a_test = parse_float(a.get("test_mae_mean"))
            b_test = parse_float(b.get("test_mae_mean"))
            a_val = parse_float(a.get("val_mae_mean"))
            b_val = parse_float(b.get("val_mae_mean"))
            a_train = parse_float(a.get("train_mae_mean"))
            b_train = parse_float(b.get("train_mae_mean"))
            advantage = b_test - a_test
            relative_advantage = advantage / max(1.0e-12, abs(b_test))
            metric = "mae"
            direction = "lower"
        rows.append(
            {
                "task": task,
                "task_base": base,
                "difficulty": difficulty,
                "task_type": task_type,
                "metric": metric,
                "direction": direction,
                f"{model_a}_train": a_train,
                f"{model_b}_train": b_train,
                f"{model_a}_val": a_val,
                f"{model_b}_val": b_val,
                f"{model_a}_test": a_test,
                f"{model_b}_test": b_test,
                "grit_advantage": advantage,
                "relative_grit_advantage": relative_advantage,
                "winner": model_a if advantage > 0 else (model_b if advantage < 0 else "tie"),
                "n_seeds": int(float(a.get("n_seeds", 1))),
            }
        )
    return rows


def summarise(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    grit_wins = sum(1 for row in rows if row["winner"] == MODEL_A)
    gcn_wins = sum(1 for row in rows if row["winner"] == MODEL_B)
    ties = len(rows) - grit_wins - gcn_wins
    by_difficulty = defaultdict(list)
    by_task_type = defaultdict(list)
    for row in rows:
        by_difficulty[str(row["difficulty"])].append(float(row["grit_advantage"]))
        by_task_type[str(row["task_type"])].append(float(row["grit_advantage"]))
    return {
        "n_tasks": len(rows),
        "grit_wins": grit_wins,
        "gcn_plus_wins": gcn_wins,
        "ties": ties,
        "mean_grit_advantage": sum(float(row["grit_advantage"]) for row in rows) / len(rows),
        "mean_relative_grit_advantage": sum(float(row["relative_grit_advantage"]) for row in rows) / len(rows),
        "by_difficulty_mean_advantage": {key: sum(vals) / len(vals) for key, vals in sorted(by_difficulty.items())},
        "by_task_type_mean_advantage": {key: sum(vals) / len(vals) for key, vals in sorted(by_task_type.items())},
    }


def plot_advantage_bars(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    if not rows:
        return
    plt = import_plotting()
    labels = [str(row["task"]) for row in rows]
    vals = [float(row["grit_advantage"]) for row in rows]
    colors = ["#2ca25f" if v > 0 else "#de2d26" if v < 0 else "#969696" for v in vals]
    fig, ax = plt.subplots(figsize=(8.2, max(3.2, 0.42 * len(labels))))
    ax.barh(range(len(labels)), vals, color=colors)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("GRIT advantage over GCN+ (F1 diff, or GCN MAE - GRIT MAE)")
    ax.set_title("Official GraphBench: GRIT vs GCN+ test advantage")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_score_pairs(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    if not rows:
        return
    plt = import_plotting()
    groups = [("binary", [row for row in rows if str(row["task_type"]) in BINARY_TYPES]), ("regression", [row for row in rows if str(row["task_type"]) in REGRESSION_TYPES])]
    groups = [(name, group) for name, group in groups if group]
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), max(3.6, 0.34 * len(rows))), squeeze=False)
    for ax, (name, group) in zip(axes[0], groups):
        labels = [str(row["task"]) for row in group]
        y = list(range(len(group)))
        grit = [float(row[f"{MODEL_A}_test"]) for row in group]
        gcn = [float(row[f"{MODEL_B}_test"]) for row in group]
        ax.scatter(grit, y, label=MODEL_A, color="#2b8cbe", s=34)
        ax.scatter(gcn, y, label=MODEL_B, color="#f03b20", s=34)
        for yi, g1, g2 in zip(y, grit, gcn):
            ax.plot([g1, g2], [yi, yi], color="#bdbdbd", linewidth=1.0, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("test F1" if name == "binary" else "test MAE")
        ax.set_title("Binary tasks" if name == "binary" else "Regression tasks")
        ax.grid(axis="x", alpha=0.25)
        if name == "regression":
            ax.invert_xaxis()
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Model test scores by task", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_difficulty_summary(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    if not rows:
        return
    plt = import_plotting()
    diffs = [difficulty for difficulty in ("easy", "medium", "hard", "unknown") if any(str(row["difficulty"]) == difficulty for row in rows)]
    means = []
    wins = []
    for difficulty in diffs:
        group = [row for row in rows if str(row["difficulty"]) == difficulty]
        means.append(sum(float(row["grit_advantage"]) for row in group) / len(group))
        wins.append(sum(1 for row in group if row["winner"] == MODEL_A) / len(group))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    colors = ["#2ca25f" if v > 0 else "#de2d26" if v < 0 else "#969696" for v in means]
    axes[0].bar(diffs, means, color=colors)
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].set_ylabel("mean GRIT advantage")
    axes[0].set_title("Average advantage")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(diffs, wins, color="#756bb1")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("fraction tasks won by GRIT")
    axes[1].set_title("Win rate")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_generalisation(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    if not rows:
        return
    plt = import_plotting()
    labels = [str(row["task"]) for row in rows]
    grit_gap = []
    gcn_gap = []
    for row in rows:
        if str(row["task_type"]) in BINARY_TYPES:
            grit_gap.append(float(row[f"{MODEL_A}_train"]) - float(row[f"{MODEL_A}_test"]))
            gcn_gap.append(float(row[f"{MODEL_B}_train"]) - float(row[f"{MODEL_B}_test"]))
        else:
            grit_gap.append(float(row[f"{MODEL_A}_test"]) - float(row[f"{MODEL_A}_train"]))
            gcn_gap.append(float(row[f"{MODEL_B}_test"]) - float(row[f"{MODEL_B}_train"]))
    x = list(range(len(rows)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(rows)), 4.0))
    ax.bar([i - width / 2 for i in x], grit_gap, width=width, label=MODEL_A, color="#2b8cbe")
    ax.bar([i + width / 2 for i in x], gcn_gap, width=width, label=MODEL_B, color="#f03b20")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("train-test degradation\n(F1 drop, or MAE increase)")
    ax.set_title("Generalisation gap")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(path: Path, rows: Sequence[Mapping[str, object]], summary: Mapping[str, object]) -> None:
    ranked = sorted(rows, key=lambda row: float(row["grit_advantage"]), reverse=True)
    lines = [
        "# GRIT vs GCN+ Official GraphBench Summary",
        "",
        f"Tasks compared: {summary.get('n_tasks', 0)}",
        f"GRIT wins: {summary.get('grit_wins', 0)}",
        f"GCN+ wins: {summary.get('gcn_plus_wins', 0)}",
        f"Ties: {summary.get('ties', 0)}",
        "",
        "Positive advantage means GRIT is better. For binary tasks this is `GRIT F1 - GCN+ F1`; for regression tasks this is `GCN+ MAE - GRIT MAE`.",
        "",
        "| Task | Type | Metric | GRIT Test | GCN+ Test | Advantage | Winner |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['task']} | {row['task_type']} | {row['metric']} | "
            f"{float(row[f'{MODEL_A}_test']):.4f} | {float(row[f'{MODEL_B}_test']):.4f} | "
            f"{float(row['grit_advantage']):+.4f} | {row['winner']} |"
        )
    lines.extend(["", "## Mean Advantage By Difficulty", ""])
    for key, val in dict(summary.get("by_difficulty_mean_advantage", {})).items():
        lines.append(f"- `{key}`: {float(val):+.4f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def strip_colab_kernel_args(argv: Sequence[str]) -> list[str]:
    cleaned = []
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
    parser = argparse.ArgumentParser(description="Visualise official GraphBench GRIT-vs-GCN+ screening results.")
    parser.add_argument("--drive-dir", type=Path, default=Path("/content/drive/MyDrive/graph_specialisation_metrics/official_algoreas_screen"))
    parser.add_argument("--drive-mount", type=Path, default=Path("/content/drive"))
    parser.add_argument("--no-mount-drive", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-a", type=str, default=MODEL_A)
    parser.add_argument("--model-b", type=str, default=MODEL_B)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = strip_colab_kernel_args(list(sys.argv[1:] if argv is None else argv))
    args = parse_args(argv)
    mount_drive(args.drive_mount, enabled=not args.no_mount_drive)
    results_dir = args.results_dir or args.drive_dir / "results" / RUN_NAME
    aggregate, resolved_results_dir = load_aggregate_with_discovery(results_dir, args.drive_dir)
    if not aggregate:
        hints = available_result_hints(args.drive_dir)
        hint_text = "\n".join(f"  - {hint}" for hint in hints[:20]) if hints else "  - no summary/aggregate files found under drive-dir"
        raise FileNotFoundError(
            f"No aggregate metrics found from {results_dir}. Expected aggregate_metrics.csv, top-level summary.json, "
            f"or per-run */summary.json files.\nFiles discovered under {args.drive_dir}:\n{hint_text}"
        )
    print(f"[data] Using results from {resolved_results_dir}", flush=True)
    output_dir = args.output_dir or resolved_results_dir / "visualisations"
    rows = complete_pair_rows(aggregate, args.model_a, args.model_b)
    if not rows:
        available = sorted({str(row.get("model", "")) for row in aggregate})
        raise RuntimeError(f"No complete {args.model_a} vs {args.model_b} task pairs found. Available models: {available}")
    summary = summarise(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "grit_vs_gcn_task_summary.csv", rows)
    write_json(output_dir / "grit_vs_gcn_summary.json", summary)
    write_markdown_summary(output_dir / "grit_vs_gcn_summary.md", rows, summary)
    plot_advantage_bars(rows, output_dir / "grit_advantage_by_task.png")
    plot_score_pairs(rows, output_dir / "grit_gcn_test_scores_by_task.png")
    plot_difficulty_summary(rows, output_dir / "grit_advantage_by_difficulty.png")
    plot_generalisation(rows, output_dir / "train_test_generalisation_gap.png")
    print(f"[done] Wrote visualisations and summaries to {output_dir}", flush=True)
    print(f"[summary] tasks={summary['n_tasks']} grit_wins={summary['grit_wins']} gcn_plus_wins={summary['gcn_plus_wins']} ties={summary['ties']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
