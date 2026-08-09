#!/usr/bin/env python3
"""Export and summarize recoverable Peptides-func/struct GRIT checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence


BIN_DIR = Path(__file__).resolve().parent
if str(BIN_DIR) not in sys.path:
    sys.path.insert(0, str(BIN_DIR))

from export_best_available import (  # noqa: E402
    jsonable,
    parse_history,
    saved_candidates,
    sha256,
)


VARIANTS = ("dense", "1hop", "1hop_vnode", "2hop", "2hop_vnode")
DISPLAY = {
    "dense": "Dense",
    "1hop": "1-hop",
    "1hop_vnode": "1-hop + VNode",
    "2hop": "2-hop",
    "2hop_vnode": "2-hop + VNode",
}
TASKS = {
    "peptides_func": {"metric": "ap", "direction": "max", "label": "Peptides-func"},
    "peptides_struct": {"metric": "mae", "direction": "min", "label": "Peptides-struct"},
}
EXPECTED_RUNS = tuple(
    f"{task}.{variant}.s{seed}"
    for task in TASKS
    for variant in VARIANTS
    for seed in range(3)
)
EXPECTED_FINAL_EPOCH = 199


def task_and_variant(run_id: str) -> tuple[str, str, int]:
    for task in TASKS:
        prefix = f"{task}."
        if run_id.startswith(prefix):
            body = run_id[len(prefix) :]
            try:
                variant, seed_text = body.rsplit(".s", 1)
                seed = int(seed_text)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid Peptides run ID: {run_id}") from exc
            if variant not in VARIANTS:
                raise ValueError(f"unsupported Peptides variant in {run_id}: {variant}")
            return task, variant, seed
    raise ValueError(f"unsupported Peptides run ID: {run_id}")


def value(stats: dict[str, Any], metric_name: str) -> float:
    try:
        result = float(stats[metric_name])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def selection_key(epoch: int, history: dict[int, dict[str, dict[str, Any]]], metric_name: str, direction: str):
    metric_value = value(history[epoch]["val"], metric_name)
    return ((metric_value if direction == "min" else -metric_value), epoch)


def select_run(run_dir: Path) -> dict[str, Any]:
    task, variant, seed = task_and_variant(run_dir.name)
    spec = TASKS[task]
    metric_name = str(spec["metric"])
    direction = str(spec["direction"])
    log_root = run_dir / "wrapper_logs"
    logs = list(log_root.rglob("*.log")) if log_root.is_dir() else []
    history, parsed_logs = parse_history(logs)
    complete = {
        epoch: splits
        for epoch, splits in history.items()
        if "val" in splits
        and "test" in splits
        and math.isfinite(value(splits["val"], metric_name))
        and math.isfinite(value(splits["test"], metric_name))
    }
    if not complete:
        raise RuntimeError(
            f"{run_dir.name}: no complete val/test {metric_name.upper()} epochs found"
        )
    last_complete_epoch = max(complete)
    if last_complete_epoch < EXPECTED_FINAL_EPOCH:
        raise RuntimeError(
            f"{run_dir.name}: training is unfinished; last complete epoch is "
            f"{last_complete_epoch}, expected {EXPECTED_FINAL_EPOCH}"
        )

    global_epoch = min(
        complete,
        key=lambda epoch: selection_key(epoch, complete, metric_name, direction),
    )
    candidates = saved_candidates(run_dir)
    eligible = {
        epoch: paths
        for epoch, paths in candidates.items()
        if epoch in complete and paths
    }
    if not eligible:
        raise RuntimeError(
            f"{run_dir.name}: no saved checkpoint epoch could be joined to its logs"
        )
    selected_epoch = min(
        eligible,
        key=lambda epoch: selection_key(epoch, complete, metric_name, direction),
    )
    source = eligible[selected_epoch][0]
    global_stats = complete[global_epoch]
    selected_stats = complete[selected_epoch]
    global_val = value(global_stats["val"], metric_name)
    global_test = value(global_stats["test"], metric_name)
    selected_val = value(selected_stats["val"], metric_name)
    selected_test = value(selected_stats["test"], metric_name)

    return {
        "run_id": run_dir.name,
        "task": task,
        "variant": variant,
        "seed": seed,
        "selection_metric": metric_name,
        "selection_direction": direction,
        "last_complete_epoch": last_complete_epoch,
        "global_log_best_epoch": global_epoch,
        "global_log_best_validation": global_val,
        "global_log_best_test": global_test,
        "selected_epoch": selected_epoch,
        "selected_validation": selected_val,
        "selected_test": selected_test,
        "exact_global_best": selected_epoch == global_epoch,
        "epoch_difference": selected_epoch - global_epoch,
        "validation_difference": selected_val - global_val,
        "test_difference": selected_test - global_test,
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_bytes": source.stat().st_size,
        "source_checkpoint_sha256": sha256(source),
        "parsed_logs": [str(path.resolve()) for path in parsed_logs],
        "available_saved_epochs": sorted(eligible),
        "global_best_train": jsonable(global_stats.get("train", {})),
        "global_best_val": jsonable(global_stats.get("val", {})),
        "global_best_test_stats": jsonable(global_stats.get("test", {})),
        "selected_train": jsonable(selected_stats.get("train", {})),
        "selected_val": jsonable(selected_stats.get("val", {})),
        "selected_test_stats": jsonable(selected_stats.get("test", {})),
    }


def format_summary(records: Sequence[dict[str, Any]]) -> str:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["task"], record["variant"])].append(record)

    lines = [
        "# Peptides GRIT performance from exported checkpoints",
        "",
        "Each checkpoint is selected using validation data only. Peptides-func maximizes AP; "
        "Peptides-struct minimizes MAE. Test performance is reported at the selected epoch.",
        "",
    ]
    for task, spec in TASKS.items():
        metric_name = str(spec["metric"]).upper()
        arrow = "↑" if spec["direction"] == "max" else "↓"
        lines.extend(
            [
                f"## {spec['label']}",
                "",
                f"| Model | Exact best | Validation {metric_name} {arrow} | Test {metric_name} {arrow} |",
                "|---|---:|---:|---:|",
            ]
        )
        for variant in VARIANTS:
            group = sorted(grouped[(task, variant)], key=lambda row: row["seed"])
            if len(group) != 3:
                raise RuntimeError(f"expected 3 records for {task}.{variant}, found {len(group)}")
            validation = [float(row["selected_validation"]) for row in group]
            test = [float(row["selected_test"]) for row in group]
            exact = sum(bool(row["exact_global_best"]) for row in group)
            lines.append(
                f"| {DISPLAY[variant]} | {exact}/3 | "
                f"{mean(validation):.6f} ± {stdev(validation):.6f} | "
                f"{mean(test):.6f} ± {stdev(test):.6f} |"
            )
        lines.extend(
            [
                "",
                "### Per-seed selections",
                "",
                f"| Model | Seed | Global log best | Selected epoch | Exact | Validation {metric_name} | Test {metric_name} |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for variant in VARIANTS:
            for row in sorted(grouped[(task, variant)], key=lambda item: item["seed"]):
                lines.append(
                    f"| {DISPLAY[variant]} | {row['seed']} | {row['global_log_best_epoch']} | "
                    f"{row['selected_epoch']} | {'yes' if row['exact_global_best'] else 'no'} | "
                    f"{row['selected_validation']:.6f} | {row['selected_test']:.6f} |"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_runs(input_root: Path, output_dir: Path, run_ids: Sequence[str]) -> list[dict[str, Any]]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    missing = [run_id for run_id in run_ids if not (input_root / run_id).is_dir()]
    if missing:
        raise RuntimeError(f"missing {len(missing)} run directories: {', '.join(missing)}")
    records = [select_run(input_root / run_id) for run_id in run_ids]

    temporary = output_dir.with_name(f".{output_dir.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        for record in records:
            run_out = temporary / "checkpoints" / record["run_id"]
            run_out.mkdir(parents=True)
            source = Path(record["source_checkpoint"])
            destination = run_out / "best_available.ckpt"
            shutil.copy2(source, destination)
            if sha256(destination) != record["source_checkpoint_sha256"]:
                raise RuntimeError(f"checksum mismatch after copying {record['run_id']}")
            record["archive_checkpoint"] = str(
                Path("checkpoints") / record["run_id"] / destination.name
            )
            job_spec = input_root / record["run_id"] / "hpc_job.json"
            if job_spec.is_file():
                shutil.copy2(job_spec, run_out / "hpc_job.json")
            (run_out / "metadata.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        (temporary / "manifest.json").write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        fields = (
            "run_id", "task", "variant", "seed", "selection_metric",
            "selection_direction", "global_log_best_epoch",
            "last_complete_epoch",
            "global_log_best_validation", "global_log_best_test",
            "selected_epoch", "selected_validation", "selected_test",
            "exact_global_best", "epoch_difference", "validation_difference",
            "test_difference", "source_checkpoint", "source_checkpoint_sha256",
            "source_checkpoint_bytes", "archive_checkpoint",
        )
        with (temporary / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        summary = format_summary(records)
        (temporary / "performance.md").write_text(summary, encoding="utf-8")
        exact = sum(bool(record["exact_global_best"]) for record in records)
        (temporary / "README.txt").write_text(
            "GRIT Peptides-func/struct best-available checkpoint corpus\n\n"
            "Selection uses validation data only: maximum AP for Peptides-func and minimum MAE "
            "for Peptides-struct. Test metrics are reported only after selection.\n"
            f"Runs: {len(records)}; exact logged-global-best snapshots: {exact}; "
            f"best-available saved snapshots: {len(records) - exact}.\n",
            encoding="utf-8",
        )
        temporary.rename(output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", default=",".join(EXPECTED_RUNS))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_ids = tuple(item.strip() for item in args.runs.split(",") if item.strip())
    records = export_runs(args.input_root.resolve(), args.output_dir.resolve(), run_ids)
    summary = (args.output_dir.resolve() / "performance.md").read_text(encoding="utf-8")
    print(summary)
    exact = sum(bool(record["exact_global_best"]) for record in records)
    print(f"Exported {len(records)} checkpoints ({exact} exact) to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
