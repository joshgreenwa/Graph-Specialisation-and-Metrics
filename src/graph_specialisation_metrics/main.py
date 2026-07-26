"""Public entry point for the repository's final methodology."""

from __future__ import annotations

from typing import Any

from .methodology.protocol import MethodologyConfig
from .methodology.runner import run_methodology


def main(config: MethodologyConfig | None = None, **overrides: Any):
    """Run the canonical donor-swap methodology.

    Pass a complete :class:`MethodologyConfig`, or keyword arguments accepted by it. Legacy
    experimental runners remain importable from historical modules but are not called here.
    """

    if config is not None and overrides:
        raise ValueError("pass either config=... or MethodologyConfig keyword arguments, not both")
    resolved = config or MethodologyConfig(**overrides)
    return run_methodology(resolved)


def cli() -> None:
    import argparse

    from .methodology.protocol import ExecutionPolicy, parse_csv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="zinc")
    parser.add_argument("--train-seeds", default="42")
    parser.add_argument("--phases", default="scores,causal,carriage,figures")
    parser.add_argument("--output-dir", default=MethodologyConfig().output_dir)
    parser.add_argument("--accelerator", default="cuda:0")
    parser.add_argument("--graphs-per-batch", type=int, default=4)
    parser.add_argument("--no-oom-backoff", action="store_true")
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    checkpoints = {}
    for item in args.checkpoint:
        if "=" not in item:
            parser.error("--checkpoint must be TASK[:SEED]=/path/to/checkpoint")
        key, path = item.split("=", 1)
        checkpoints[key] = path
    config = MethodologyConfig(
        tasks=parse_csv(args.tasks),
        train_seeds=tuple(int(value) for value in parse_csv(args.train_seeds)),
        phases=parse_csv(args.phases),
        output_dir=args.output_dir,
        accelerator=args.accelerator,
        execution=ExecutionPolicy(
            graphs_per_batch=args.graphs_per_batch,
            oom_backoff=not args.no_oom_backoff,
        ),
        checkpoints=checkpoints,
        skip_install=args.skip_install,
        force=args.force,
    )
    run_methodology(config)


if __name__ == "__main__":
    cli()
