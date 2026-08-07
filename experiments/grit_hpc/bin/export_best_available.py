#!/usr/bin/env python3
"""Export validation-selected, actually recoverable GRIT checkpoints.

This utility is intentionally CPU-only.  It joins the epoch metrics printed in
the wrapper logs to checkpoint files that still exist, selects the saved epoch
with the lowest validation MAE, and writes a provenance-rich export directory.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


EXPECTED_VARIANTS = ("dense", "1hop", "1hop_vnode", "2hop", "2hop_vnode")
EXPECTED_RUNS = tuple(
    f"{task}.{variant}.s{seed}"
    for task in ("zinc", "qm9_gap")
    for variant in EXPECTED_VARIANTS
    for seed in range(3)
)
SPLITS = ("train", "val", "test")


def read_epoch_sidecar(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, TypeError, ValueError):
        return None


def checkpoint_epoch(path: Path) -> int | None:
    text = str(path)
    for pattern in (
        r"period_\d+_epoch(\d+)\.ckpt$",
        r"epoch[=_-]?(\d+).*\.ckpt$",
        r"/ckpt/(\d+)\.ckpt$",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def metric(stats: dict[str, Any]) -> float:
    for key in ("mae", "loss"):
        try:
            value = float(stats[key])
        except (KeyError, TypeError, ValueError):
            continue
        if value == value and value != float("inf"):
            return value
    return float("inf")


def parse_history(log_paths: Iterable[Path]) -> tuple[dict[int, dict[str, dict[str, Any]]], list[Path]]:
    history: dict[int, dict[str, dict[str, Any]]] = {}
    readable: list[Path] = []
    # Newer attempts win if an interrupted epoch appears in more than one log.
    ordered = sorted(set(log_paths), key=lambda path: (path.stat().st_mtime, str(path)))
    for path in ordered:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        readable.append(path)
        for line in lines:
            stripped = line.strip()
            for split in SPLITS:
                prefix = f"{split}: "
                if not stripped.startswith(prefix):
                    continue
                payload = stripped[len(prefix) :]
                if not payload.startswith("{"):
                    continue
                try:
                    stats = ast.literal_eval(payload)
                    epoch = int(stats["epoch"])
                except (KeyError, SyntaxError, TypeError, ValueError):
                    continue
                history.setdefault(epoch, {})[split] = stats
    return history, readable


def add_candidate(candidates: dict[int, list[Path]], epoch: int | None, path: Path) -> None:
    if epoch is None or not path.is_file() or path.stat().st_size <= 0:
        return
    candidates.setdefault(epoch, []).append(path)


def saved_candidates(run_dir: Path) -> dict[int, list[Path]]:
    result_root = run_dir / "results"
    candidates: dict[int, list[Path]] = {}

    for path in result_root.rglob("*.ckpt") if result_root.is_dir() else ():
        add_candidate(candidates, checkpoint_epoch(path), path)

    for sidecar_name, checkpoint_name in (
        ("best_epoch.txt", "best.ckpt"),
        ("latest_epoch.txt", "latest.ckpt"),
    ):
        for sidecar in result_root.rglob(sidecar_name) if result_root.is_dir() else ():
            add_candidate(candidates, read_epoch_sidecar(sidecar), sidecar.with_name(checkpoint_name))

    for epoch, paths in candidates.items():
        candidates[epoch] = sorted(
            set(paths),
            key=lambda path: (
                0 if "_recovery_checkpoints" in str(path) else 1,
                0 if path.name == "best.ckpt" else 1,
                str(path),
            ),
        )
    return candidates


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def select_run(run_dir: Path) -> dict[str, Any]:
    logs = list((run_dir / "wrapper_logs").rglob("*.log")) if (run_dir / "wrapper_logs").is_dir() else []
    history, parsed_logs = parse_history(logs)
    complete = {
        epoch: splits
        for epoch, splits in history.items()
        if "val" in splits and metric(splits["val"]) < float("inf")
    }
    if not complete:
        raise RuntimeError(f"{run_dir.name}: no validation epoch metrics found in wrapper logs")

    global_epoch = min(complete, key=lambda epoch: (metric(complete[epoch]["val"]), epoch))
    candidates = saved_candidates(run_dir)
    eligible = {
        epoch: paths
        for epoch, paths in candidates.items()
        if epoch in complete and paths
    }
    if not eligible:
        raise RuntimeError(f"{run_dir.name}: no saved checkpoint epoch could be joined to validation logs")

    selected_epoch = min(eligible, key=lambda epoch: (metric(complete[epoch]["val"]), epoch))
    source = eligible[selected_epoch][0]
    selected = complete[selected_epoch]
    global_stats = complete[global_epoch]
    return {
        "run_id": run_dir.name,
        "global_log_best_epoch": global_epoch,
        "global_log_best_val_mae": metric(global_stats["val"]),
        "global_log_best_test_mae": metric(global_stats.get("test", {})),
        "selected_epoch": selected_epoch,
        "selected_val_mae": metric(selected["val"]),
        "selected_test_mae": metric(selected.get("test", {})),
        "exact_global_best": selected_epoch == global_epoch,
        "epoch_difference": selected_epoch - global_epoch,
        "validation_mae_difference": metric(selected["val"]) - metric(global_stats["val"]),
        "test_mae_difference": metric(selected.get("test", {})) - metric(global_stats.get("test", {})),
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_bytes": source.stat().st_size,
        "source_checkpoint_sha256": sha256(source),
        "parsed_logs": [str(path.resolve()) for path in parsed_logs],
        "available_saved_epochs": sorted(eligible),
        "global_best_train": jsonable(global_stats.get("train", {})),
        "global_best_val": jsonable(global_stats.get("val", {})),
        "global_best_test": jsonable(global_stats.get("test", {})),
        "selected_train": jsonable(selected.get("train", {})),
        "selected_val": jsonable(selected.get("val", {})),
        "selected_test": jsonable(selected.get("test", {})),
    }


def export_runs(input_root: Path, output_dir: Path, run_ids: Sequence[str]) -> list[dict[str, Any]]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    missing = [run_id for run_id in run_ids if not (input_root / run_id).is_dir()]
    if missing:
        raise RuntimeError(f"missing {len(missing)} expected run directories: {', '.join(missing)}")

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
            copied_hash = sha256(destination)
            if copied_hash != record["source_checkpoint_sha256"]:
                raise RuntimeError(f"checksum mismatch after copying {record['run_id']}")
            record["archive_checkpoint"] = str(Path("checkpoints") / record["run_id"] / destination.name)
            (run_out / "metadata.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        (temporary / "manifest.json").write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        fields = (
            "run_id",
            "global_log_best_epoch",
            "selected_epoch",
            "exact_global_best",
            "global_log_best_val_mae",
            "selected_val_mae",
            "validation_mae_difference",
            "global_log_best_test_mae",
            "selected_test_mae",
            "test_mae_difference",
            "source_checkpoint",
            "source_checkpoint_sha256",
            "source_checkpoint_bytes",
            "archive_checkpoint",
        )
        with (temporary / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

        exact_count = sum(bool(record["exact_global_best"]) for record in records)
        (temporary / "README.txt").write_text(
            "GRIT ZINC/QM9 best-available checkpoint corpus\n"
            "\n"
            "Selection uses validation MAE only. For every run, the exported file is the\n"
            "existing saved checkpoint whose logged epoch has the lowest validation MAE.\n"
            "Test MAE is reported only after that validation-only selection.\n"
            f"Runs: {len(records)}; exact logged-global-best snapshots: {exact_count}; "
            f"best-available saved snapshots: {len(records) - exact_count}.\n",
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
    parser.add_argument(
        "--runs",
        default=",".join(EXPECTED_RUNS),
        help="comma-separated run IDs; defaults to the 30 ZINC/QM9 three-seed runs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_ids = tuple(item.strip() for item in args.runs.split(",") if item.strip())
    if not run_ids:
        raise RuntimeError("no run IDs requested")
    records = export_runs(args.input_root.resolve(), args.output_dir.resolve(), run_ids)
    print("run_id\tglobal_epoch\tselected_epoch\texact\tselected_val\tselected_test")
    for record in records:
        print(
            f"{record['run_id']}\t{record['global_log_best_epoch']}\t"
            f"{record['selected_epoch']}\t{str(record['exact_global_best']).lower()}\t"
            f"{record['selected_val_mae']:.8f}\t{record['selected_test_mae']:.8f}"
        )
    print(f"Exported {len(records)} checkpoints to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
