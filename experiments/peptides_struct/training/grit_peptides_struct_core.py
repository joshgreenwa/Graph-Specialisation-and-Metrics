#!/usr/bin/env python3
"""Colab runner for official dense GRIT+RRWP on Peptides-struct.

Default Colab usage:

    from grit_peptides_struct_core import main
    main([])

Default Drive output:

    /content/drive/MyDrive/grit_peptides_struct_official

This runner bootstraps the current project repo from GitHub using the Colab
secret ``dissertation_key`` so it can reuse the tested ZINC GRIT Colab runtime
plumbing while launching the official LiamMa/GRIT Peptides-struct config.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence
from urllib.parse import quote


PROJECT_REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
PROJECT_BRANCH = "codex/cfim-grit-experiments"
PROJECT_REPO_DIR = Path("/content/Graph-Specialisation-and-Metrics")
SECRET_NAME = "dissertation_key"


def _run(cmd: Sequence[str], *, safe: str | None = None) -> None:
    print(f"[cmd] {safe or ' '.join(map(str, cmd))}", flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def _in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _get_secret(name: str) -> str | None:
    try:
        from google.colab import userdata  # type: ignore

        value = userdata.get(name)
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.environ.get(name)


def _auth_url(repo_url: str, token: str | None) -> str:
    if not token or not repo_url.startswith("https://github.com/"):
        return repo_url
    suffix = repo_url.removeprefix("https://github.com/")
    return f"https://x-access-token:{quote(token, safe='')}@github.com/{suffix}"


def _copy_sidecar_common(repo_dir: Path) -> None:
    candidates: list[Path] = []
    file_value = globals().get("__file__")
    if file_value:
        candidates.append(Path(file_value).with_name("grit_peptides_struct_common.py"))
    candidates.extend([
        Path.cwd() / "grit_peptides_struct_common.py",
        Path("/content/grit_peptides_struct_common.py"),
    ])
    sidecar = next((p for p in candidates if p.exists()), None)
    if sidecar is None:
        return
    target = repo_dir / "experiments" / "peptides_struct" / "training" / "grit_peptides_struct_common.py"
    if sidecar.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sidecar, target)
        print(f"[bootstrap] using sidecar Peptides helper: {sidecar} -> {target}", flush=True)


def _bootstrap_project_repo(repo_url: str, branch: str, repo_dir: Path, secret_name: str, *, skip_git: bool) -> None:
    local_root = Path(__file__).resolve().parents[3] if "__file__" in globals() else None
    if local_root and (local_root / "src" / "graph_specialisation_metrics").exists():
        for path in [str(local_root), str(local_root / "src")]:
            if path not in sys.path:
                sys.path.insert(0, path)
        return
    if skip_git:
        _copy_sidecar_common(repo_dir)
        for path in [str(repo_dir), str(repo_dir / "src")]:
            if path not in sys.path:
                sys.path.insert(0, path)
        return
    if not _in_colab():
        raise RuntimeError("Project repo bootstrap needs Colab or an in-repo execution path")
    token = _get_secret(secret_name)
    if not token:
        raise RuntimeError(f"Missing Colab secret {secret_name!r}")
    authed = _auth_url(repo_url, token)
    if (repo_dir / ".git").exists():
        _run(["git", "-C", str(repo_dir), "remote", "set-url", "origin", authed], safe=f"git -C {repo_dir} remote set-url origin <token-authenticated-url>")
        _run(["git", "-C", str(repo_dir), "fetch", "origin", branch])
        _run(["git", "-C", str(repo_dir), "checkout", branch])
        _run(["git", "-C", str(repo_dir), "reset", "--hard", f"origin/{branch}"])
        _run(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])
    else:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        _run(["git", "clone", "--branch", branch, "--single-branch", authed, str(repo_dir)], safe=f"git clone --branch {branch} <token-authenticated-url> {repo_dir}")
        _run(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])
    _copy_sidecar_common(repo_dir)
    for path in [str(repo_dir), str(repo_dir / "src")]:
        if path not in sys.path:
            sys.path.insert(0, path)


def _split_bootstrap_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project-repo-url", default=PROJECT_REPO_URL)
    parser.add_argument("--project-branch", default=PROJECT_BRANCH)
    parser.add_argument("--project-repo-dir", type=Path, default=PROJECT_REPO_DIR)
    parser.add_argument("--secret-name", default=SECRET_NAME)
    parser.add_argument("--skip-project-git", action="store_true")
    raw = list(sys.argv[1:] if argv is None else argv)
    args, rest = parser.parse_known_args(raw)
    return args, rest


def main(argv: Sequence[str] | None = None) -> None:
    bootstrap, rest = _split_bootstrap_args(argv)
    _bootstrap_project_repo(
        bootstrap.project_repo_url,
        bootstrap.project_branch,
        bootstrap.project_repo_dir,
        bootstrap.secret_name,
        skip_git=bootstrap.skip_project_git,
    )
    from experiments.peptides_struct.training.grit_peptides_struct_common import run_training

    run_training("dense", rest)


if __name__ == "__main__":
    main()
