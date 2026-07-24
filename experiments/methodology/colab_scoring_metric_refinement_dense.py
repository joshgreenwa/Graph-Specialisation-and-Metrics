"""Standalone Colab frontend for dense ZINC/QM9 scoring-metric refinement.

Upload this file and run:

    %run colab_scoring_metric_refinement_dense.py

The default run mounts Drive, clones the repository branch, verifies both dense
checkpoints, and executes M1--M6 with resumable caches. A Colab secret named
``dissertation_key`` is used for private repository access when present.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence
from urllib.parse import quote


REPOSITORY_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
REPOSITORY_BRANCH = "codex/cfim-grit-experiments"
COLAB_REPOSITORY = Path("/content/Graph-Specialisation-and-Metrics")
SECRET_NAME = "dissertation_key"
DEFAULT_OUTPUT = (
    "/content/drive/MyDrive/graph_specialisation_metrics/"
    "scoring_metric_refinement_dense"
)


def _command(*parts: str) -> None:
    shown = [
        "https://***@github.com/" + part.split("@github.com/", 1)[1]
        if "@github.com/" in part else part
        for part in parts
    ]
    print(f"[cmd] {' '.join(shown)}", flush=True)
    subprocess.run(list(parts), check=True)


def _in_colab() -> bool:
    try:
        available = importlib.util.find_spec("google.colab") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        available = False
    return bool(os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules or available)


def _strip_colab_kernel_args(argv: Sequence[str]) -> list[str]:
    """Discard only Jupyter's injected ``-f kernel-....json`` argument."""

    values = list(argv)
    cleaned = []
    index = 0
    while index < len(values):
        value = values[index]
        if (
            value == "-f"
            and index + 1 < len(values)
            and "kernel-" in values[index + 1]
            and values[index + 1].endswith(".json")
        ):
            print(f"[args] Ignoring Jupyter argument: -f {values[index + 1]}", flush=True)
            index += 2
            continue
        if value.startswith("-f=") and "kernel-" in value and value.endswith(".json"):
            print(f"[args] Ignoring Jupyter argument: {value}", flush=True)
            index += 1
            continue
        cleaned.append(value)
        index += 1
    return cleaned


def bootstrap_repository(*, branch: str, skip_bootstrap: bool) -> Path:
    if skip_bootstrap:
        if "__file__" in globals():
            root = Path(__file__).resolve().parents[2]
        else:
            root = Path.cwd()
        if not (root / "pyproject.toml").exists():
            raise FileNotFoundError(f"not a repository root: {root}")
        return root
    if not _in_colab():
        if "__file__" in globals():
            root = Path(__file__).resolve().parents[2]
            if (root / "pyproject.toml").exists():
                return root
        root = Path.cwd()
        if (root / "pyproject.toml").exists():
            return root
        raise RuntimeError("outside Colab, run this file from the repository")

    from google.colab import drive, userdata  # type: ignore

    drive.mount("/content/drive", force_remount=False)
    try:
        token = userdata.get(SECRET_NAME) or os.environ.get(SECRET_NAME)
    except Exception as exc:  # pragma: no cover - Colab UI dependent
        print(f"[bootstrap] secret unavailable ({exc}); trying public clone", flush=True)
        token = os.environ.get(SECRET_NAME)
    suffix = REPOSITORY_URL.removeprefix("https://github.com/")
    authenticated = REPOSITORY_URL
    if token:
        authenticated = (
            f"https://x-access-token:{quote(str(token).strip(), safe='')}"
            f"@github.com/{suffix}"
        )
    if (COLAB_REPOSITORY / ".git").exists():
        _command(
            "git", "-C", str(COLAB_REPOSITORY), "remote", "set-url", "origin", authenticated
        )
        _command("git", "-C", str(COLAB_REPOSITORY), "fetch", "origin", branch)
        _command("git", "-C", str(COLAB_REPOSITORY), "checkout", branch)
        _command(
            "git", "-C", str(COLAB_REPOSITORY), "reset", "--hard", f"origin/{branch}"
        )
    else:
        if COLAB_REPOSITORY.exists():
            shutil.rmtree(COLAB_REPOSITORY)
        _command(
            "git",
            "clone",
            "--branch",
            branch,
            "--single-branch",
            authenticated,
            str(COLAB_REPOSITORY),
        )
    _command(
        "git", "-C", str(COLAB_REPOSITORY), "remote", "set-url", "origin", REPOSITORY_URL
    )
    _command(sys.executable, "-m", "pip", "install", "-q", "-e", str(COLAB_REPOSITORY))
    return COLAB_REPOSITORY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("all", "scores", "validation", "figures"), default="all"
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=("all", "zinc", "qm9_gap_dense"),
        help="repeat to select tasks; default runs both",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", default="all")
    parser.add_argument("--zinc-checkpoint")
    parser.add_argument("--qm9-checkpoint")
    parser.add_argument("--analysis-seed", type=int, default=1771)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repository-branch", default=REPOSITORY_BRANCH)
    parser.add_argument("--pyg-version", default="2.2.0")
    parser.add_argument("--resume-config", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-fresh-grit", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None):
    supplied = _strip_colab_kernel_args(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(supplied)
    repository = bootstrap_repository(
        branch=args.repository_branch, skip_bootstrap=args.skip_bootstrap
    )
    for path in (repository / "src", repository):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from graph_specialisation_metrics.scoring_refinement import run

    selected = args.task or ["zinc", "qm9_gap_dense"]
    if "all" in selected:
        selected = ["zinc", "qm9_gap_dense"]
    selected = list(dict.fromkeys(selected))
    checkpoints = {}
    if args.zinc_checkpoint:
        checkpoints["zinc"] = args.zinc_checkpoint
    if args.qm9_checkpoint:
        checkpoints["qm9_gap_dense"] = args.qm9_checkpoint
    return run(
        tasks=selected,
        output_dir=args.output_dir,
        methods=args.methods,
        phase=args.phase,
        force=args.force,
        resume_config=args.resume_config,
        fast_dev_run=args.fast_dev_run,
        checkpoints=checkpoints,
        device=args.device,
        skip_install=args.skip_install,
        pyg_version=args.pyg_version,
        force_fresh_grit=args.force_fresh_grit,
        analysis_seed=args.analysis_seed,
    )


if __name__ == "__main__":
    main()
