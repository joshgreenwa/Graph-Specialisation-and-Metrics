"""Figure-only rerendering for cached methodology artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from graph_specialisation_metrics.method_core import read_yaml


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def render_validation(root: Path, config: Mapping[str, Any]) -> list[Path]:
    from graph_specialisation_metrics.method_validation import (
        plot_carriage,
        plot_interaction,
        plot_patching,
        plot_rank,
    )

    rendered: list[Path] = []
    metrics = root / "metrics"
    if (metrics / "carriage_reconstruction.csv").exists():
        history = read_csv_rows(metrics / "carriage_training.csv") if (metrics / "carriage_training.csv").exists() else []
        plot_carriage(read_csv_rows(metrics / "carriage_reconstruction.csv"), history, root, config)
        rendered.append(root / "figures" / "validation_carriage_check.png")
    if (metrics / "patching_retained.csv").exists():
        plot_patching(read_csv_rows(metrics / "patching_retained.csv"), root, config)
        rendered.append(root / "figures" / "validation_patching_check.png")
    if (metrics / "rank_check.csv").exists():
        plot_rank(read_csv_rows(metrics / "rank_check.csv"), root, config)
        rendered.append(root / "figures" / "validation_rank_check.png")
    if (metrics / "interaction_check.csv").exists():
        plot_interaction(read_csv_rows(metrics / "interaction_check.csv"), root, config)
        rendered.append(root / "figures" / "validation_interaction_check.png")
    return rendered


def render_main(root: Path, config: Mapping[str, Any]) -> list[Path]:
    from graph_specialisation_metrics.main_procedure import render_step_1_figures

    metrics = root / "metrics"
    rendered: list[Path] = []
    metric_rows = read_csv_rows(metrics / "step1_test_metrics.csv") if (metrics / "step1_test_metrics.csv").exists() else []
    history_rows = read_csv_rows(metrics / "step1_training_history.csv") if (metrics / "step1_training_history.csv").exists() else []
    if metric_rows or history_rows:
        render_step_1_figures(metric_rows, history_rows, root, config)
        rendered.extend(
            [
                root / "figures" / "step1_test_error_dense_vs_1hop.png",
                root / "figures" / "step1_training_validation_loss.png",
            ]
        )
    return [path for path in rendered if path.exists()]


def render(root: Path) -> list[Path]:
    config_path = root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"missing artifact config: {config_path}")
    config = read_yaml(config_path)
    metrics = root / "metrics"
    if (metrics / "validation_summary.json").exists():
        return render_validation(root, config)
    if (metrics / "main_status.json").exists() or (metrics / "artifact_discovery.json").exists():
        return render_main(root, config)
    raise RuntimeError(f"could not identify artifact type under {root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rr = sub.add_parser("render", help="Rerender figures from cached metrics.")
    rr.add_argument("--artifact-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "render":
        paths = render(Path(args.artifact_root))
        for path in paths:
            print(path)
    else:  # pragma: no cover
        raise ValueError(args.command)


if __name__ == "__main__":  # pragma: no cover
    main()
