"""Standalone Colab frontend for the synthetic scoring-method comparison.

Upload this file to Colab and run:

    %run colab_scoring_metric_refinement_synthetic.py

The default command mounts Drive, refreshes the repository branch, loads the
already-trained ``cycle_dual_v2`` checkpoints, resumes the seven-method analysis,
and writes exactly four consolidated PNG/PDF figure families. A Colab secret
named ``dissertation_key`` is used for private repository access when present.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence
from urllib.parse import quote


REPOSITORY_URL = (
    "https://github.com/joshgreenwa/"
    "Graph-Specialisation-and-Metrics.git"
)
REPOSITORY_BRANCH = "codex/cfim-grit-experiments"
COLAB_REPOSITORY = Path("/content/Graph-Specialisation-and-Metrics")
SECRET_NAME = "dissertation_key"


def _command(*parts: str) -> None:
    shown = [
        "https://***@github.com/" + part.split("@github.com/", 1)[1]
        if "@github.com/" in part
        else part
        for part in parts
    ]
    print(f"[cmd] {' '.join(shown)}", flush=True)
    subprocess.run(list(parts), check=True)


def _in_colab() -> bool:
    try:
        available = importlib.util.find_spec("google.colab") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        available = False
    return bool(
        os.environ.get("COLAB_RELEASE_TAG")
        or "google.colab" in sys.modules
        or available
    )


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
            print(
                f"[args] Ignoring Jupyter argument: -f {values[index + 1]}",
                flush=True,
            )
            index += 2
            continue
        if (
            value.startswith("-f=")
            and "kernel-" in value
            and value.endswith(".json")
        ):
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
        print(
            f"[bootstrap] secret unavailable ({exc}); trying public clone",
            flush=True,
        )
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
            "git",
            "-C",
            str(COLAB_REPOSITORY),
            "remote",
            "set-url",
            "origin",
            authenticated,
        )
        _command(
            "git", "-C", str(COLAB_REPOSITORY), "fetch", "origin", branch
        )
        _command("git", "-C", str(COLAB_REPOSITORY), "checkout", branch)
        _command(
            "git",
            "-C",
            str(COLAB_REPOSITORY),
            "reset",
            "--hard",
            f"origin/{branch}",
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
        "git",
        "-C",
        str(COLAB_REPOSITORY),
        "remote",
        "set-url",
        "origin",
        REPOSITORY_URL,
    )
    _command(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-e",
        str(COLAB_REPOSITORY),
    )
    return COLAB_REPOSITORY


def _load_runner(repository: Path) -> object:
    path = (
        repository
        / "experiments"
        / "synthetic"
        / "analysis"
        / "scoring_metric_refinement_synthetic.py"
    )
    name = "_synthetic_scoring_refinement_colab_runner"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import synthetic runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None):
    supplied = _strip_colab_kernel_args(
        sys.argv[1:] if argv is None else argv
    )
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument(
        "--repository-branch", default=REPOSITORY_BRANCH
    )
    bootstrap_parser.add_argument("--skip-bootstrap", action="store_true")
    bootstrap_args, runner_args = bootstrap_parser.parse_known_args(supplied)
    repository = bootstrap_repository(
        branch=bootstrap_args.repository_branch,
        skip_bootstrap=bootstrap_args.skip_bootstrap,
    )
    for path in (repository / "src", repository):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    importlib.invalidate_caches()
    for module_name in list(sys.modules):
        if (
            module_name == "graph_specialisation_metrics"
            or module_name.startswith("graph_specialisation_metrics.")
        ):
            del sys.modules[module_name]
    runner = _load_runner(repository)
    return runner.main(["--repository", str(repository), *runner_args])


if __name__ == "__main__":
    main()
