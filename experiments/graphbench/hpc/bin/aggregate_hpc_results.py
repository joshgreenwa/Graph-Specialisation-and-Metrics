#!/usr/bin/env python3
"""Aggregate GraphBench HPC array-job summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_RUN_NAME = "graphbench_algoreas_hpc_base_v1"


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def write_json(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


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


def load_summaries(run_dir: Path) -> list[dict[str, object]]:
    summaries = []
    for path in sorted(run_dir.glob("*/*/seed*/summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] skipping unreadable summary {path}: {exc}", flush=True)
            continue
        summary["_summary_path"] = str(path)
        summaries.append(summary)
    return summaries


def aggregate_rows(summaries: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for summary in summaries:
        if not all(f"{split}_metrics" in summary for split in ("train", "val", "test")):
            continue
        grouped[(str(summary["task"]), str(summary["model"]))].append(summary)

    rows = []
    for (task, model), runs in sorted(grouped.items()):
        task_type = str(runs[0]["task_type"])
        row: dict[str, object] = {
            "task": task,
            "task_type": task_type,
            "model": model,
            "n_seeds": len(runs),
            "trainable_parameters": runs[0].get("trainable_parameters", ""),
        }
        for split in ("train", "val", "test"):
            primary_vals = [float(run[f"{split}_metrics"]["primary"]) for run in runs]
            primary_mean = sum(primary_vals) / len(primary_vals)
            primary_std = 0.0 if len(primary_vals) == 1 else math.sqrt(
                sum((value - primary_mean) ** 2 for value in primary_vals) / (len(primary_vals) - 1)
            )
            row[f"{split}_primary_mean"] = primary_mean
            row[f"{split}_primary_std"] = primary_std
            for metric in (
                "f1",
                "accuracy",
                "precision",
                "recall",
                "mae",
                "raw_mae",
                "target_z_mae",
                "relative_mae",
                "mse",
                "r2",
                "spearman",
                "loss",
            ):
                vals = [float(run[f"{split}_metrics"].get(metric, float("nan"))) for run in runs]
                finite = [value for value in vals if math.isfinite(value)]
                row[f"{split}_{metric}_mean"] = sum(finite) / len(finite) if finite else float("nan")
        rows.append(row)
    return rows


def flat_metric_rows(summaries: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    for summary in summaries:
        for split in ("train", "val", "test"):
            metrics = summary.get(f"{split}_metrics")
            if not isinstance(metrics, Mapping):
                continue
            rows.append(
                {
                    "task": summary["task"],
                    "task_type": summary["task_type"],
                    "model": summary["model"],
                    "seed": summary["seed"],
                    "split": split,
                    "trainable_parameters": summary.get("trainable_parameters", ""),
                    "best_step": summary.get("best_step", ""),
                    **metrics,
                }
            )
    return rows


def missing_rows(summaries: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    for summary in summaries:
        missing = [split for split in ("train", "val", "test") if f"{split}_metrics" not in summary]
        if missing:
            rows.append(
                {
                    "task": summary.get("task", ""),
                    "model": summary.get("model", ""),
                    "seed": summary.get("seed", ""),
                    "missing_metrics": ",".join(missing),
                    "summary_path": summary.get("_summary_path", ""),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate GraphBench HPC result summaries.")
    parser.add_argument("--output-root", type=Path, default=env_path("GRAPHBENCH_OUTPUT_ROOT"))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root is None:
        raise ValueError("--output-root is required or set GRAPHBENCH_OUTPUT_ROOT")
    run_dir = args.output_root.expanduser() / args.run_name
    summaries = load_summaries(run_dir)
    aggregate = aggregate_rows(summaries)
    flat = flat_metric_rows(summaries)
    missing = missing_rows(summaries)

    write_json(run_dir / "hpc_aggregate_summary.json", {"runs": summaries, "aggregate": aggregate, "missing": missing})
    write_csv_rows(run_dir / "hpc_per_run_metrics.csv", flat)
    write_csv_rows(run_dir / "hpc_aggregate_metrics.csv", aggregate)
    write_csv_rows(run_dir / "hpc_missing_final_eval.csv", missing)

    print(f"[aggregate] loaded_summaries={len(summaries)}", flush=True)
    print(f"[aggregate] complete_metric_runs={len(flat)} split rows", flush=True)
    print(f"[aggregate] aggregate_rows={len(aggregate)} missing_runs={len(missing)}", flush=True)
    print(f"[aggregate] output_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
