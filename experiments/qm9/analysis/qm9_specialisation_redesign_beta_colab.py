"""Standalone Colab entry point for the ZINC-redesign analysis on QM9 gap models.

This runner intentionally reuses the complete analysis implementation in
``experiments/zinc/analysis/zinc_specialisation_redesign_beta_colab.py``. It changes
only the dataset/model profile:

* QM9 target 4: HOMO-LUMO energy gap in raw eV;
* the seeded 110000/10000/remainder train/validation/test split used in training;
* dense QM9 GRIT+RRWP versus exact 1-hop QM9 GRIT+RRWP, with the attached runner's
  global VNode enabled by default for the 1-hop model;
* QM9-specific checkpoint roots, cache namespace, provenance labels, and figure names.

All score, causal, mechanism, ablation, topology-reach, gallery, and figure phases are
therefore the same code paths as the ZINC analysis. The default output is persistent:

``/content/drive/MyDrive/graph_specialisation_metrics/qm9_gap_redesign_headline_v2``

The default checkpoints are auto-discovered beneath:

* dense: ``/content/drive/MyDrive/grit_qm9_gap_dense/results``
* 1-hop+VNode: ``/content/drive/MyDrive/grit_qm9_gap_1hop_vnode/results``

Pass ``--dense-checkpoint`` and ``--onehop-checkpoint`` to pin exact files. Pass
``--onehop-no-vnode`` only when analysing the older parameter-matched 1-hop checkpoint
under ``/content/drive/MyDrive/grit_qm9_gap_1hop_real/results``.

Colab use
---------
Upload this file and run ``%run qm9_specialisation_redesign_beta_colab.py``. It mounts
Drive, clones or refreshes the analysis repository, installs it, and resumes cached work.
For a private repository, create a Colab secret named ``dissertation_key``.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote


REPOSITORY_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
REPOSITORY_BRANCH = "codex/cfim-grit-experiments"
COLAB_REPOSITORY = Path("/content/Graph-Specialisation-and-Metrics")
SECRET_NAME = "dissertation_key"
DEFAULT_OUTPUT = Path(
    "/content/drive/MyDrive/graph_specialisation_metrics/"
    "qm9_gap_redesign_headline_v2"
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
        colab_available = importlib.util.find_spec("google.colab") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        colab_available = False
    return bool(
        os.environ.get("COLAB_RELEASE_TAG")
        or "google.colab" in sys.modules
        or colab_available
    )


def bootstrap_repository(*, branch: str = REPOSITORY_BRANCH) -> Path:
    """Mount Drive and return a fresh checkout containing the shared analysis."""

    if not _in_colab():
        if "__file__" in globals():
            candidate = Path(__file__).resolve().parents[3]
            if (candidate / "pyproject.toml").exists():
                return candidate
        candidate = Path.cwd()
        if (candidate / "pyproject.toml").exists():
            return candidate
        raise RuntimeError(
            "Outside Colab, run this file from the Graph-Specialisation-and-Metrics "
            "repository."
        )

    from google.colab import drive, userdata  # type: ignore

    drive.mount("/content/drive", force_remount=False)
    try:
        token = userdata.get(SECRET_NAME) or os.environ.get(SECRET_NAME)
    except Exception as exc:  # pragma: no cover - depends on Colab UI state
        print(
            f"[bootstrap] secret unavailable ({exc}); trying a public clone",
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
            "git", "-C", str(COLAB_REPOSITORY),
            "remote", "set-url", "origin", authenticated,
        )
        _command(
            "git", "-C", str(COLAB_REPOSITORY),
            "fetch", "origin", branch,
        )
        _command(
            "git", "-C", str(COLAB_REPOSITORY),
            "checkout", branch,
        )
        _command(
            "git", "-C", str(COLAB_REPOSITORY),
            "reset", "--hard", f"origin/{branch}",
        )
    else:
        _command(
            "git", "clone", "--branch", branch, "--single-branch",
            authenticated, str(COLAB_REPOSITORY),
        )
    _command(
        "git", "-C", str(COLAB_REPOSITORY),
        "remote", "set-url", "origin", REPOSITORY_URL,
    )
    _command(sys.executable, "-m", "pip", "install", "-q", "-e", str(COLAB_REPOSITORY))
    return COLAB_REPOSITORY


def load_shared_analysis(repository: Path) -> Any:
    """Load the shared implementation without triggering its script entry point."""

    source = (
        repository
        / "experiments/zinc/analysis/zinc_specialisation_redesign_beta_colab.py"
    )
    if not source.exists():
        raise FileNotFoundError(f"shared analysis is missing: {source}")
    module_name = "_qm9_shared_specialisation_redesign"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not construct an import spec for {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _strip_qm9_profile_args(argv: Sequence[str]) -> tuple[list[str], bool]:
    """Inspect the QM9-only switch before invoking the profile-enabled shared parser."""

    cleaned = []
    onehop_vnode = True
    for value in argv:
        if value == "--onehop-no-vnode":
            onehop_vnode = False
        cleaned.append(value)
    return cleaned, onehop_vnode


def _repository_branch(argv: Sequence[str]) -> str:
    """Read the shared parser's branch option early enough for the initial clone."""

    for index, value in enumerate(argv):
        if value == "--repository-branch":
            if index + 1 >= len(argv):
                raise ValueError("--repository-branch requires a value")
            return str(argv[index + 1])
        if value.startswith("--repository-branch="):
            return value.split("=", 1)[1]
    return REPOSITORY_BRANCH


def _strip_colab_kernel_args(argv: Sequence[str]) -> list[str]:
    """Discard only IPython's injected ``-f kernel-....json`` argument pair.

    Pasting this file into a Colab cell leaves ``sys.argv`` owned by
    ``colab_kernel_launcher.py``. Genuine analysis options remain strict so misspelled
    user arguments still fail loudly.
    """

    cleaned: list[str] = []
    index = 0
    values = list(argv)
    while index < len(values):
        value = values[index]
        if (
            value == "-f"
            and index + 1 < len(values)
            and "kernel-" in values[index + 1]
            and values[index + 1].endswith(".json")
        ):
            print(
                "[args] Ignoring Colab/Jupyter kernel argument: "
                f"{value} {values[index + 1]}",
                flush=True,
            )
            index += 2
            continue
        if (
            value.startswith("-f=")
            and "kernel-" in value
            and value.endswith(".json")
        ):
            print(
                f"[args] Ignoring Colab/Jupyter kernel argument: {value}",
                flush=True,
            )
            index += 1
            continue
        cleaned.append(value)
        index += 1
    return cleaned


def configure_qm9_profile(shared: Any, *, onehop_vnode: bool) -> None:
    """Set the dataset profile consumed dynamically by the shared implementation."""

    onehop_task = "qm9_gap_1hop_vnode" if onehop_vnode else "qm9_gap_1hop"
    shared.BETA_VERSION = (
        "qm9-gap-specialisation-headline-v2-vnode"
        if onehop_vnode else
        "qm9-gap-specialisation-headline-v2"
    )
    shared.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    shared.ARCHITECTURES = ("qm9_gap_dense", onehop_task)
    shared.DISPLAY = {
        "qm9_gap_dense": "Dense GRIT",
        onehop_task: "1-hop GRIT + VNode" if onehop_vnode else "1-hop GRIT",
    }
    shared.DATASET_LABEL = "QM9 HOMO-LUMO gap"
    shared.FIGURE_PREFIX = "qm9_gap_headline"
    # PyG QM9 contains 130831 molecules; training consumes 120000.
    shared.TEST_SPLIT_SIZE = 10_831
    # Raw-eV paths use the stricter policy already established by the QM9 carriage run.
    shared.DEFAULT_INTEGRATED_ATOL = 1.0e-4
    shared.DEFAULT_INTEGRATED_RTOL = 1.0e-4
    shared.DEFAULT_INTEGRATED_MAX_INTERVALS = 256
    shared.DEFAULT_INTEGRATED_UNCONVERGED_ERROR_CAP = 1.0e-3
    shared.DEFAULT_INTEGRATED_MAX_UNCONVERGED_FRACTION = 1.0e-2
    shared.ALLOW_ONEHOP_NO_VNODE = True


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    supplied = _strip_colab_kernel_args(
        sys.argv[1:] if argv is None else argv
    )
    repository = bootstrap_repository(branch=_repository_branch(supplied))
    for path in (repository / "src", repository):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    shared = load_shared_analysis(repository)

    supplied, onehop_vnode = _strip_qm9_profile_args(supplied)
    configure_qm9_profile(shared, onehop_vnode=onehop_vnode)

    if not supplied:
        supplied = [
            "--phase", "all",
            "--output-dir", str(DEFAULT_OUTPUT),
            "--resume-config",
        ]
    # The checkout is already fresh and installed. The shared bootstrap still adds src
    # to sys.path, but does not fetch/reset when this flag is present.
    if "--skip-bootstrap" not in supplied:
        supplied.append("--skip-bootstrap")
    return shared.main(supplied)


if __name__ == "__main__":
    main()
