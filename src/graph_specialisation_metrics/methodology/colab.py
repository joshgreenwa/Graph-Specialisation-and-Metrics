"""Lightweight Drive-backed Colab front end for the canonical implementation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..carriage import env
from ..carriage.env import log
from .protocol import BootstrapPolicy, MethodologyConfig, RunSizes
from .runner import run_methodology


DEFAULT_DRIVE_OUTPUT = (
    "/content/drive/MyDrive/graph_specialisation_metrics/canonical_methodology"
)


def mount_drive(path: str = "/content/drive") -> None:
    try:
        from google.colab import drive

        log(f"[drive] Mounting at {path}")
        drive.mount(path, force_remount=False)
    except ImportError:
        log("[drive] not running in Colab; using paths as provided")


def run(
    tasks: str | Sequence[str] = ("zinc",),
    *,
    train_seeds: Sequence[int] = (42,),
    phases: Sequence[str] = ("scores", "causal", "carriage", "figures"),
    output_dir: str = DEFAULT_DRIVE_OUTPUT,
    checkpoints: Mapping[str, str] | None = None,
    sizes: RunSizes | Mapping[str, Any] | None = None,
    task_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    figure_overrides: Mapping[str, Any] | None = None,
    analysis_seed: int = 31_415,
    bootstrap_seed: int = 17_071,
    accelerator: str = "cuda:0",
    mount: bool = True,
    skip_install: bool = False,
    force_fresh_grit: bool = False,
    force: bool = False,
    strict_audits: bool = False,
) -> dict[str, Any]:
    """Mount Drive, reuse training caches/checkpoints, and run selected task phases.

    Numerical and estimability audits report into ``audits.json`` and continue by default; pass
    ``strict_audits=True`` for a fail-closed verification run.
    """

    if mount:
        mount_drive()
    if not skip_install:
        env.install_dependencies(pyg_version="2.2.0")
    else:
        log("[deps] using the current runtime (skip_install=True)")
    if isinstance(tasks, str):
        task_names = tuple(part.strip() for part in tasks.split(",") if part.strip())
    else:
        task_names = tuple(str(value) for value in tasks)
    run_sizes = sizes if isinstance(sizes, RunSizes) else RunSizes(**dict(sizes or {}))
    bootstrap = BootstrapPolicy(
        rng_seed=int(bootstrap_seed),
        replicates=run_sizes.bootstrap_replicates,
    )
    config = MethodologyConfig(
        output_dir=output_dir,
        tasks=task_names,
        train_seeds=tuple(int(value) for value in train_seeds),
        phases=tuple(str(value) for value in phases),
        sizes=run_sizes,
        bootstrap=bootstrap,
        analysis_seed=int(analysis_seed),
        accelerator=accelerator,
        checkpoints=dict(checkpoints or {}),
        task_overrides=dict(task_overrides or {}),
        figure_overrides=dict(figure_overrides or {}),
        skip_install=True,
        force=bool(force),
        strict_audits=bool(strict_audits),
    )
    return run_methodology(config, force_fresh_grit=force_fresh_grit)

