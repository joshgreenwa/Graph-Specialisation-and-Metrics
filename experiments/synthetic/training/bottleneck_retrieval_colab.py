"""Standalone Colab runner: dense vs 1-hop attention breakaway on cross-bottleneck retrieval.

PASTE THIS WHOLE FILE INTO ONE COLAB CELL AND RUN IT (the ``main([...])`` at the bottom fires).

It mounts Drive, clones the repo with the ``dissertation_key`` Colab secret, then runs the
self-contained bottleneck-retrieval sweep from
``graph_specialisation_metrics.synthetic_bottleneck_retrieval``. The model is pure torch (no
GRIT / torch_geometric), so there is NOTHING to install beyond what Colab already ships -- this
is the fast first run that validates whether the task separates dense from 1-hop attention.

Outputs (figures + a ``results.json`` cache) are written under Drive at
``--drive-out/--run-name``. If that ``results.json`` already exists it is loaded instead of
retraining, so re-running the cell is cheap.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence
from urllib.parse import quote

REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
BRANCH = "codex/cfim-grit-experiments"
REPO_DIR = "/content/Graph-Specialisation-and-Metrics"
DRIVE_OUT = "/content/drive/MyDrive/graph_specialisation_metrics/bottleneck_retrieval"
SECRET_NAME = "dissertation_key"


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
    import os

    return os.environ.get(secret_name)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _clone_or_update(repo_url: str, branch: str, repo_dir: Path, token: str | None, username: str | None) -> None:
    if token:
        cred = f"{quote(str(username), safe='')}:{quote(token, safe='')}" if username else quote(token, safe="")
        authed = repo_url.replace("https://", f"https://{cred}@", 1)
    else:
        print("[git] no token found; attempting anonymous clone/update (fails on private repos)")
        authed = repo_url
    if (repo_dir / ".git").exists():
        print(f"[git] updating {repo_dir} -> origin/{branch} (hard reset)")
        _run(["git", "-C", str(repo_dir), "remote", "set-url", "origin", authed])
        _run(["git", "-C", str(repo_dir), "fetch", "origin", branch])
        _run(["git", "-C", str(repo_dir), "checkout", branch])
        _run(["git", "-C", str(repo_dir), "reset", "--hard", f"origin/{branch}"])
        _run(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])
    else:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        print(f"[git] cloning {branch} -> {repo_dir}")
        _run(["git", "clone", "--branch", branch, "--single-branch", authed, str(repo_dir)])
        _run(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Dense vs 1-hop attention breakaway on bottleneck retrieval (Colab).")
    ap.add_argument("--repo-url", default=REPO_URL)
    ap.add_argument("--branch", default=BRANCH)
    ap.add_argument("--repo-dir", default=REPO_DIR)
    ap.add_argument("--secret-name", default=SECRET_NAME)
    ap.add_argument("--github-username", default=None)
    ap.add_argument("--drive-out", default=DRIVE_OUT)
    ap.add_argument("--run-name", default="first_run")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1])
    ap.add_argument("--graphs", nargs="+", default=["dumbbell", "wellconnected"])
    ap.add_argument("--addressings", nargs="+", default=["content"])
    ap.add_argument("--models", nargs="+", default=["dense", "1hop", "1hop_vnode"])
    ap.add_argument("--distances", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--fast-dev-run", action="store_true")
    ap.add_argument("--skip-clone", action="store_true")
    args = ap.parse_args(argv)

    _mount_drive()
    repo = Path(args.repo_dir)
    if not args.skip_clone:
        _clone_or_update(args.repo_url, args.branch, repo, _get_token(args.secret_name), args.github_username)

    # Pure-torch module: a path insert is enough. Editable-install + cache invalidation makes it
    # robust in a long-lived kernel; --no-deps so nothing heavy is touched.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo), "--no-deps"], check=False)
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    importlib.invalidate_caches()

    try:
        from graph_specialisation_metrics import synthetic_bottleneck_retrieval as bnr
    except ModuleNotFoundError as exc:  # noqa: BLE001
        raise SystemExit(
            f"[import] failed: {exc}. Check --repo-dir ({repo}) is the clone and that torch/numpy/matplotlib "
            "are importable (all ship with Colab)."
        )

    inner = [
        "--out-dir", str(args.drive_out),
        "--run-name", str(args.run_name),
        "--steps", str(args.steps),
        "--seeds", str(args.seeds),
        "--ranks", *[str(r) for r in args.ranks],
        "--graphs", *[str(g) for g in args.graphs],
        "--addressings", *[str(a) for a in args.addressings],
        "--models", *[str(m) for m in args.models],
    ]
    if args.distances:
        inner += ["--distances", *[str(d) for d in args.distances]]
    if args.fast_dev_run:
        inner.append("--fast-dev-run")

    result = bnr.main(inner)
    print("\n[done]")
    print(f"  figure : {result['figure']}")
    print(f"  cache  : {result['cache']}")
    print(f"  out_dir: {result['out_dir']}")


# --- fires on paste. Default = the DISTANCE-trend figure: content retrieval, all 3 models,
#     query->target distance 1..4 (the reach axis). Expect dense flat at ~1.0, 1-hop high at
#     d=1 and falling as distance grows, and 1-hop+VNode somewhere in between (its failure point
#     is what we're hunting). Pass --fast-dev-run first to check plumbing. ---
if __name__ == "__main__":
    main([
        "--run-name", "distance_trend_v1",
        "--steps", "1500",
        "--seeds", "3",
        "--ranks", "1",
        "--distances", "1", "2", "3", "4",
        "--addressings", "content",
        "--graphs", "dumbbell", "wellconnected",
        "--models", "dense", "1hop", "1hop_vnode",
    ])
