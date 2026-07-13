"""Standalone Colab runner: OFFICIAL dense/1-hop GRIT breakaway + Step-7 carriage on bottleneck retrieval.

PASTE THIS WHOLE FILE INTO ONE COLAB CELL AND RUN IT (the ``main([...])`` at the bottom fires).

Unlike ``bottleneck_retrieval_colab.py`` (pure-torch stand-in, no install), this variant uses the
OFFICIAL LiamMa/GRIT ``GritTransformerLayer`` (dense) and its parameter-matched masked 1-hop control
(``pad_to_full_graph=False``), so it needs the official GRIT repo + torch_geometric. This cell:

  1. mounts Drive and clones the project repo (``dissertation_key`` Colab secret);
  2. clones official GRIT at the pinned commit, installs torch_geometric + GRIT editable, and exports
     ``GRIT_ROOT`` so ``import grit.layer.grit_layer`` works;
  3. runs ``synthetic_bottleneck_retrieval_official.main`` (breakaway sweep + carriage suite).

Trained models cache to Drive; re-running the cell reuses them and only redoes analysis/figures.
Run with ``--fast-dev-run`` first (~a couple of minutes) to check the GRIT install + plumbing.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence
from urllib.parse import quote

REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
BRANCH = "codex/cfim-grit-experiments"
REPO_DIR = "/content/Graph-Specialisation-and-Metrics"
DRIVE_OUT = "/content/drive/MyDrive/graph_specialisation_metrics/bottleneck_retrieval_official"
SECRET_NAME = "dissertation_key"

OFFICIAL_GRIT_URL = "https://github.com/LiamMa/GRIT.git"
OFFICIAL_GRIT_COMMIT = "6c988ea600a606fbb49a2246c64a2d37396b3ab5"
GRIT_DIR = "/content/GRIT"


def _mount_drive() -> None:
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[drive] skipping mount ({exc})")


def _get_token(secret_name: str) -> str | None:
    try:
        from google.colab import userdata  # type: ignore

        tok = userdata.get(secret_name)
        if tok:
            return str(tok)
    except Exception as exc:  # noqa: BLE001
        print(f"[secret] userdata.get({secret_name!r}) failed: {exc}")
    return os.environ.get(secret_name)


def _run(cmd: list[str], *, check: bool = True) -> int:
    print(f"[cmd] {' '.join(map(str, cmd))}", flush=True)
    return subprocess.run(list(map(str, cmd)), check=check).returncode


def _clone_or_update(repo_url: str, branch: str, repo_dir: Path, token: str | None) -> None:
    authed = repo_url.replace("https://", f"https://{quote(token, safe='')}@", 1) if token else repo_url
    if (repo_dir / ".git").exists():
        _run(["git", "-C", str(repo_dir), "remote", "set-url", "origin", authed])
        _run(["git", "-C", str(repo_dir), "fetch", "origin", branch])
        _run(["git", "-C", str(repo_dir), "checkout", branch])
        _run(["git", "-C", str(repo_dir), "reset", "--hard", f"origin/{branch}"])
        _run(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])
    else:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        _run(["git", "clone", "--branch", branch, "--single-branch", authed, str(repo_dir)])
        _run(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])


def _install_pyg_stack() -> None:
    """Install torch_geometric + the compiled PyG extensions GRIT's layers need (torch_scatter/
    torch_sparse), matched to the runtime's torch+CUDA build. Mirrors the ZINC/peptides GRIT colabs:
    `pip install torch_geometric` does NOT pull these C++ extensions, so `import grit.layer.grit_layer`
    fails with `No module named 'torch_scatter'` without this step."""
    import torch  # already present in Colab

    torch_version = str(torch.__version__).split("+")[0]
    cuda = getattr(torch.version, "cuda", None)
    cuda_tag = ("cu" + cuda.replace(".", "")) if cuda else "cpu"
    wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"
    print(f"[deps] torch={torch.__version__} cuda={cuda} | PyG wheel index: {wheel_url}", flush=True)
    _run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric", "yacs", "ogb", "einops"], check=False)
    # Optional accelerators -- best effort (the RRWP GRIT layer does not require them).
    for pkg in ("pyg-lib", "torch-spline-conv", "torch-cluster"):
        _run([sys.executable, "-m", "pip", "install", "-q", pkg, "-f", wheel_url], check=False)
    # Required by GRIT's attention/message passing.
    for pkg in ("torch-scatter", "torch-sparse"):
        rc = _run([sys.executable, "-m", "pip", "install", "-q", pkg, "-f", wheel_url], check=False)
        if rc != 0:
            raise SystemExit(
                f"[deps] Required PyG extension {pkg!r} failed to install from {wheel_url}. "
                "This Colab torch/CUDA build has no matching prebuilt wheel; switch to a runtime with "
                "a PyG-supported torch (e.g. a slightly older torch), then re-run."
            )


def _setup_official_grit(grit_dir: Path, *, install: bool) -> None:
    """Clone official GRIT at the pinned commit and expose it for ``import grit.*``."""
    if not (grit_dir / ".git").exists():
        if grit_dir.exists():
            shutil.rmtree(grit_dir)
        _run(["git", "clone", OFFICIAL_GRIT_URL, str(grit_dir)])
    _run(["git", "-C", str(grit_dir), "checkout", OFFICIAL_GRIT_COMMIT])
    if install:
        _install_pyg_stack()
        _run([sys.executable, "-m", "pip", "install", "-q", "-e", str(grit_dir), "--no-deps"], check=False)
    os.environ["GRIT_ROOT"] = str(grit_dir)
    if str(grit_dir) not in sys.path:
        sys.path.insert(0, str(grit_dir))
    print(f"[grit] GRIT_ROOT={grit_dir} @ {OFFICIAL_GRIT_COMMIT}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Official dense/1-hop GRIT bottleneck retrieval (Colab).")
    ap.add_argument("--repo-url", default=REPO_URL)
    ap.add_argument("--branch", default=BRANCH)
    ap.add_argument("--repo-dir", default=REPO_DIR)
    ap.add_argument("--secret-name", default=SECRET_NAME)
    ap.add_argument("--drive-out", default=DRIVE_OUT)
    ap.add_argument("--grit-dir", default=GRIT_DIR)
    ap.add_argument("--run-name", default="official_first_run")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1])
    ap.add_argument("--graphs", nargs="+", default=["dumbbell", "wellconnected"])
    ap.add_argument("--models", nargs="+", default=["dense", "1hop"])
    ap.add_argument("--distances", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--carriage", action="store_true")
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--force-carriage", action="store_true", help="recompute carriage even if cached cells exist on Drive")
    ap.add_argument("--phase", choices=["all", "train", "analyze"], default="all", help="train=GPU pass (cache all models to Drive); analyze=load models + compute carriage/figures; all=both")
    ap.add_argument("--carriage-graphs", nargs="+", default=["dumbbell", "expander", "wellconnected"])
    ap.add_argument("--carriage-target-distance", type=int, default=3)
    ap.add_argument("--carriage-graphs-count", type=int, default=12)
    ap.add_argument("--channel-start", type=int, default=2)
    ap.add_argument("--ig-steps", type=int, default=24)
    ap.add_argument("--rrwp-replacement", default="donor", choices=["donor", "mean", "zero"])
    ap.add_argument("--donor-samples", type=int, default=4)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--fast-dev-run", action="store_true")
    ap.add_argument("--skip-clone", action="store_true")
    ap.add_argument("--skip-grit-install", action="store_true")
    args = ap.parse_args(argv)

    _mount_drive()
    repo = Path(args.repo_dir)
    if not args.skip_clone:
        _clone_or_update(args.repo_url, args.branch, repo, _get_token(args.secret_name))
    _setup_official_grit(Path(args.grit_dir), install=not args.skip_grit_install)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo), "--no-deps"], check=False)
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    # Drop any already-imported package modules so a warm Colab kernel picks up the freshly reset repo
    # (otherwise `from ... import` returns the stale cached module and repo updates are ignored).
    for _m in [m for m in list(sys.modules) if m == "graph_specialisation_metrics"
               or m.startswith("graph_specialisation_metrics.")]:
        del sys.modules[_m]
    importlib.invalidate_caches()

    try:
        from graph_specialisation_metrics import synthetic_bottleneck_retrieval_official as off
    except ModuleNotFoundError as exc:  # noqa: BLE001
        raise SystemExit(f"[import] failed: {exc}. Check --repo-dir ({repo}) and torch_geometric/GRIT install.")

    inner = [
        "--out-dir", str(args.drive_out), "--run-name", str(args.run_name),
        "--steps", str(args.steps), "--seeds", str(args.seeds), "--dim", str(args.dim),
        "--ranks", *map(str, args.ranks), "--graphs", *map(str, args.graphs),
        "--models", *map(str, args.models),
        "--carriage-graphs", *map(str, args.carriage_graphs),
        "--carriage-target-distance", str(args.carriage_target_distance),
        "--carriage-graphs-count", str(args.carriage_graphs_count),
        "--channel-start", str(args.channel_start), "--ig-steps", str(args.ig_steps),
        "--rrwp-replacement", str(args.rrwp_replacement), "--donor-samples", str(args.donor_samples),
    ]
    if args.distances:
        inner += ["--distances", *map(str, args.distances)]
    if args.carriage:
        inner.append("--carriage")
    if args.skip_sweep:
        inner.append("--skip-sweep")
    if args.force_retrain:
        inner.append("--force-retrain")
    if args.force_carriage:
        inner.append("--force-carriage")
    inner += ["--phase", str(args.phase)]
    if args.fast_dev_run:
        inner.append("--fast-dev-run")

    result = off.main(inner)
    print("\n[done]")
    print(f"  out_dir: {result['out_dir']}")
    if result.get("figure"):
        print(f"  breakaway figure: {result['figure']}")
    carriage = result.get("carriage") or {}
    if carriage.get("figures"):
        print(f"  carriage figures: {len(carriage['figures'])} under {carriage.get('out_dir')}")
    print("  checkpoints cached under <out_dir>/checkpoints (re-run reuses them; --force-retrain to rebuild)")


if __name__ == "__main__":
    main([
        "--run-name", "official_bottleneck_carriage_v1",
        "--steps", "1500", "--seeds", "3", "--ranks", "1",
        "--distances", "1", "2", "3", "4",
        "--graphs", "dumbbell", "wellconnected",
        "--models", "dense", "1hop",
        "--carriage",
        "--carriage-graphs", "dumbbell", "expander", "wellconnected",
        "--carriage-target-distance", "3",
        "--carriage-graphs-count", "12",
        "--ig-steps", "24",
    ])
