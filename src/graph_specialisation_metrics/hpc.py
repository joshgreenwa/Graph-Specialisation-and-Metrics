"""Write Slurm templates and copy/paste commands for methodology runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from graph_specialisation_metrics.method_core import ensure_dir


VALIDATION_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=method-validation
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

cd "${PROJECT_ROOT:-/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PYTHON_BIN="${PYTHON_BIN:-$(which python)}"

"${PYTHON_BIN}" -u -m graph_specialisation_metrics.method_validation run-all \\
  --config "${CONFIG_PATH:-experiments/methodology/configs/method_validation_default.yaml}" \\
  --output-root "${OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/method_validation_artifacts}" \\
  --device "${DEVICE:-cuda}" \\
  ${FORCE_FLAG:-}
"""


MAIN_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=main-procedure
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-5

set -euo pipefail

cd "${PROJECT_ROOT:-/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PYTHON_BIN="${PYTHON_BIN:-$(which python)}"
STEP="${STEP_OVERRIDE:-${SLURM_ARRAY_TASK_ID}}"

"${PYTHON_BIN}" -u -m graph_specialisation_metrics.main_procedure run \\
  --config "${CONFIG_PATH:-experiments/methodology/configs/zinc_main_procedure.yaml}" \\
  --steps "${STEP}" \\
  --output-root "${OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/main_procedure_artifacts}" \\
  ${DRY_RUN_FLAG:-} \\
  ${FORCE_FLAG:-}
"""


FIGURES_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=render-figures
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd "${PROJECT_ROOT:-/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-$(which python)}"

"${PYTHON_BIN}" -u -m graph_specialisation_metrics.figures render \\
  --artifact-root "${ARTIFACT_ROOT:?set ARTIFACT_ROOT to an existing artifact directory}"
"""


def write_slurm(output_dir: Path, suite: str) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    selected: list[tuple[str, str]]
    if suite == "validation":
        selected = [("method_validation.sbatch", VALIDATION_TEMPLATE)]
    elif suite == "main":
        selected = [("main_procedure_array.sbatch", MAIN_TEMPLATE)]
    elif suite == "figures":
        selected = [("render_figures.sbatch", FIGURES_TEMPLATE)]
    elif suite == "all":
        selected = [
            ("method_validation.sbatch", VALIDATION_TEMPLATE),
            ("main_procedure_array.sbatch", MAIN_TEMPLATE),
            ("render_figures.sbatch", FIGURES_TEMPLATE),
        ]
    else:
        raise ValueError("suite must be validation, main, figures, or all")
    paths = []
    for name, text in selected:
        path = output_dir / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


def print_submit_commands(output_dir: Path) -> None:
    print("cd /rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics")
    print("git pull --ff-only")
    print('export PYTHON_BIN="$(which python)"')
    print("mkdir -p /rds/user/jgg45/hpc-work/method_validation_artifacts/slurm_logs")
    print(
        "sbatch --export=ALL,PYTHON_BIN,PROJECT_ROOT=$PWD "
        "-A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 "
        "-N 1 --ntasks=1 --gres=gpu:1 "
        f"{output_dir / 'method_validation.sbatch'}"
    )
    print(
        "sbatch --export=ALL,PYTHON_BIN,PROJECT_ROOT=$PWD "
        "-A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 "
        "-N 1 --ntasks=1 --gres=gpu:1 "
        f"{output_dir / 'main_procedure_array.sbatch'}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write-slurm", help="Write Slurm templates.")
    write.add_argument("--suite", choices=["validation", "main", "figures", "all"], default="all")
    write.add_argument("--output-dir", default="experiments/methodology/slurm")
    write.add_argument("--print-submit-commands", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "write-slurm":
        output_dir = Path(args.output_dir)
        paths = write_slurm(output_dir, args.suite)
        for path in paths:
            print(path)
        if args.print_submit_commands:
            print_submit_commands(output_dir)
    else:  # pragma: no cover
        raise ValueError(args.command)


if __name__ == "__main__":  # pragma: no cover
    main()
