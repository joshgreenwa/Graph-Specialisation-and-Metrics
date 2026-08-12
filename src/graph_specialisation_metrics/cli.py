"""Run the dissertation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .experiments import EXPERIMENTS


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one experiment YAML file."""

    source = Path(path)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"{source} must contain a YAML mapping")
    name = config.get("experiment")
    if name not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {name!r}; choose from {', '.join(EXPERIMENTS)}")
    return config


def _run(args: argparse.Namespace) -> Path | dict[str, Any]:
    config = load_config(args.config)
    experiment = EXPERIMENTS[str(config["experiment"])]
    if args.job_index is not None:
        if not hasattr(experiment, "jobs"):
            raise SystemExit(f"{config['experiment']} does not use job indices")
        config["job_index"] = int(args.job_index)
    output_dir = Path(args.output_dir or "results")
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if args.command == "run" and checkpoint is not None and args.job_index is not None:
        raise SystemExit("--job-index selects a training job; omit it when using --checkpoint")

    if args.command == "train":
        if not hasattr(experiment, "train"):
            raise SystemExit(
                f"{config['experiment']} uses a public checkpoint; there is no train command"
            )
        return experiment.train(config, output_dir=output_dir, fast=args.fast)

    if args.command == "score":
        return experiment.score(
            config,
            checkpoint=checkpoint,
            output_dir=output_dir,
            fast=args.fast,
        )

    if hasattr(experiment, "train"):
        if (
            checkpoint is None
            and hasattr(experiment, "jobs")
            and len(experiment.jobs(config, fast=args.fast)) != 1
        ):
            raise SystemExit(
                f"{config['experiment']} run requires --job-index; "
                "use train to launch every configured job"
            )
        # ``run`` means train then score. Reuse is explicit via ``--checkpoint``;
        # guessing from filenames can silently mix fast and full configurations.
        if checkpoint is not None:
            trained: dict[str, Any] = {}
        else:
            trained = experiment.train(config, output_dir=output_dir, fast=args.fast)
        if checkpoint is None and isinstance(trained, dict):
            paths = trained.get("checkpoints")
            if paths and len(paths) == 1:
                checkpoint = Path(paths[0])
    return experiment.score(
        config,
        checkpoint=checkpoint,
        output_dir=output_dir,
        fast=args.fast,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "train", "score"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True, type=Path)
        subparser.add_argument("--output-dir", type=Path)
        if command != "train":
            subparser.add_argument("--checkpoint", type=Path)
        else:
            subparser.set_defaults(checkpoint=None)
        if command != "score":
            subparser.add_argument("--job-index", type=int)
        else:
            subparser.set_defaults(job_index=None)
        subparser.add_argument("--fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    result = _run(build_parser().parse_args(argv))
    if isinstance(result, Path):
        print(result)
    elif isinstance(result, dict):
        for checkpoint in result.get("checkpoints", ()):
            print(checkpoint)


if __name__ == "__main__":
    main()
