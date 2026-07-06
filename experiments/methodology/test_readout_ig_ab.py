"""Fully standalone A/B test: does full-IG-through-the-readout carriage improve Step-0b
fidelity to the measured one-node-baseline output delta?

PASTE THIS WHOLE FILE INTO ONE COLAB CELL AND RUN IT (the ``main([...])`` at the bottom fires).
No prior runner run is needed. The cell:

  1. mounts Drive,
  2. bootstrap-clones the methodology repo with the ``dissertation_key`` Colab secret,
  3. delegates the ENTIRE environment build to the methodology runner
     (``colab_zinc_main_procedure.main``): it installs torch/PyG + deps, editable-installs
     the package, clones and patches the external GRIT checkout, discovers the trained ZINC
     checkpoints on Drive, and writes the analysis config -- but we stop it right after the
     config is written, before any analysis steps run,
  4. loads that config in-process and compares, per model, the Step-0b R^2 -- predicted
     ``Sum_i C[i,j]`` vs the *measured* one-node-baseline delta y -- for BOTH the frozen-g
     carriage and the readout-IG carriage.

It reads only forward passes and writes no analysis artifacts (the runner does write its
usual config/prepared-pointer files on Drive as a side effect of setup).
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
DRIVE_ROOT = "/content/drive/MyDrive/graph_specialisation_metrics/zinc_main_procedure_colab"
SECRET_NAME = "dissertation_key"


# --------------------------------------------------------------------------------------
# Minimal bootstrap (mount + clone) -- just enough to import the methodology runner, which
# then performs the full, tested environment build for us.
# --------------------------------------------------------------------------------------
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


def _clone_bootstrap(repo_url: str, branch: str, repo_dir: Path, token: str | None, username: str | None) -> None:
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


def _r2(measured: list[float], predicted: list[float]) -> float:
    import numpy as np

    y = np.asarray(measured, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if int(mask.sum()) < 3:
        return float("nan")
    y, p = y[mask], p[mask]
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - float(((y - p) ** 2).sum()) / ss_tot


def _relerr(measured: list[float], predicted: list[float]) -> float:
    """Mean |predicted - measured| / (|measured| + eps) -- reconstruction rel-error per graph."""
    import numpy as np

    y = np.asarray(measured, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if int(mask.sum()) < 1:
        return float("nan")
    y, p = y[mask], p[mask]
    return float(np.mean(np.abs(p - y) / (np.abs(y) + 1e-6)))


def _apply_torch_load_compat() -> None:
    """Match the runner subprocess's py312 compat shim: default ``torch.load`` to
    ``weights_only=False``.

    The trusted GRIT/GIN checkpoints pickle ``torch_geometric.data.Data`` objects, which
    torch>=2.6 refuses under its new ``weights_only=True`` default. The runner applies this
    same patch via a ``sitecustomize.py`` on the subprocess PYTHONPATH; because we run the
    A/B in-process we must apply it here too, or every checkpoint load raises UnpicklingError.
    """
    import torch

    if getattr(torch.load, "__name__", "") == "_compat_torch_load":
        return
    _orig_torch_load = torch.load

    def _compat_torch_load(*args: object, **kwargs: object):  # noqa: ANN202
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _compat_torch_load  # type: ignore[assignment]


class _StopBeforeSteps(Exception):
    """Sentinel used to halt the methodology runner right after it writes the config."""


def _build_environment_and_config(args: argparse.Namespace) -> Path:
    """Set up the full Colab environment by REUSING the methodology runner, then return the
    path to the config it wrote.

    We bootstrap-clone the repo (so the runner module is importable), then run the runner's
    own ``main()`` -- which installs deps, editable-installs the package, clones+patches the
    external GRIT checkout, discovers the ZINC checkpoints, and writes the analysis config --
    but we monkeypatch its step-runner to raise ``_StopBeforeSteps`` so it stops exactly at
    the step boundary (the config is written just before that). This reuses the runner's exact,
    tested setup with zero duplication instead of forking ~700 lines of discovery/config logic.
    """
    repo = Path(args.repo_dir)
    _clone_bootstrap(args.repo_url, args.branch, repo, _get_token(args.secret_name), args.github_username)

    methodology_dir = str(repo / "experiments" / "methodology")
    if methodology_dir not in sys.path:
        sys.path.insert(0, methodology_dir)
    import colab_zinc_main_procedure as runner  # noqa: WPS433  (import from the fresh clone)

    importlib.reload(runner)  # pick up freshly-pulled runner code on re-runs in a long-lived kernel

    def _stop(*_a: object, **_k: object) -> None:
        raise _StopBeforeSteps()

    runner.run_main_procedure = _stop  # type: ignore[assignment]
    runner.run_onehop_locality_preflight = lambda *_a, **_k: None  # type: ignore[assignment]

    runner_argv = [
        "--skip-git",  # reuse the bootstrap clone above (no second clone / token needed)
        "--repo-dir", str(repo),
        "--drive-root", str(args.drive_root),
        "--branch", str(args.branch),
        "--repo-url", str(args.repo_url),
        "--secret-name", str(args.secret_name),
        "--skip-onehop-locality-check",  # read-only A/B: no need to certify 1-hop locality
    ]
    print("[setup] delegating full environment build to the methodology runner (deps + GRIT + config)...", flush=True)
    try:
        runner.main(runner_argv)
    except _StopBeforeSteps:
        pass  # setup finished; the config is on Drive and the steps were intentionally skipped

    config_path = Path(args.drive_root) / "configs" / "zinc_main_procedure_colab.yaml"
    if not config_path.exists():
        raise SystemExit(f"[setup] runner finished but no config was written at {config_path}")
    print(f"[setup] environment ready; using config {config_path}", flush=True)
    return config_path


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Readout-IG vs frozen-g carriage: Step-0b R^2 A/B (fully standalone).")
    ap.add_argument("--repo-url", default=REPO_URL)
    ap.add_argument("--branch", default=BRANCH)
    ap.add_argument("--repo-dir", default=REPO_DIR)
    ap.add_argument("--drive-root", default=DRIVE_ROOT)
    ap.add_argument("--secret-name", default=SECRET_NAME)
    ap.add_argument("--github-username", default=None)
    ap.add_argument(
        "--config",
        default=None,
        help="Skip the runner setup and use this already-written config (e.g. after a prior runner run in the same session).",
    )
    ap.add_argument("--sample-graphs", type=int, default=8)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args(argv)

    _mount_drive()

    repo = Path(args.repo_dir)
    if args.config:
        config_path = Path(args.config)
        print(f"[setup] --config given; assuming the environment is already set up, using {config_path}", flush=True)
    else:
        config_path = _build_environment_and_config(args)

    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    importlib.invalidate_caches()

    # Patch torch.load BEFORE importing the package (which imports PyG). The failing checkpoint
    # load is PyG's *internal* dataset torch.load (it unpickles torch_geometric.data.Data), which
    # under torch>=2.6 defaults to weights_only=True. The runner's sitecustomize shim patches at
    # interpreter startup ahead of any PyG import, so we mirror that ordering here.
    try:
        import torch  # noqa: F401

        _apply_torch_load_compat()
    except ModuleNotFoundError:
        pass  # the package import below will surface a clear deps error

    try:
        from graph_specialisation_metrics.main_procedure import discover_model_artifacts, load_config
        from graph_specialisation_metrics.grit_intervention_procedure import (
            carriage_ig,
            expanded_baseline,
            instantiate_official_models,
            mean_encoded_baseline,
            predict_scalar_from_encoded,
            select_baseline_graphs,
            select_graphs,
        )
    except ModuleNotFoundError as exc:  # noqa: BLE001
        raise SystemExit(
            f"[import] failed after setup: {exc}\n"
            "  The methodology runner should have installed deps + editable-installed the package.\n"
            f"  Check the setup log above for a pip/GRIT error, and that --repo-dir ({repo}) is the clone."
        )

    config = load_config(str(config_path), fast_dev_run=False, output_root=None, analysis_preset="full")
    discovery = [discover_model_artifacts(name, cfg) for name, cfg in config["models"].items()]
    models = instantiate_official_models(config, discovery)

    print("\nreadout-IG vs frozen-g carriage -- TWO different targets:")
    print("  (A) Step-0b PER-SOURCE marginal: predicted Sum_i C[i,j]  vs  measured single-node-baseline dy_j")
    print("      -> tests additivity/necessity; NOT what readout-IG optimises.")
    print("  (B) JOINT completeness: Sum_ij C  vs  measured all-baseline dy  (mean rel-error per graph)")
    print("      -> readout-IG's actual guarantee; should be ~0 for readout-IG on every model.")
    print(f"(ig_steps={args.ig_steps}, sample_graphs={args.sample_graphs})\n")
    print(
        f"{'model':22s} {'n_src':>6s} | {'(A)R2 frozen':>12s} {'(A)R2 rIG':>10s} | "
        f"{'(B)relerr frozen':>16s} {'(B)relerr rIG':>13s}"
    )
    print("-" * 92)
    for model in models:
        try:
            graphs = select_graphs(model.adapter, "test", args.sample_graphs, seed=args.seed)
            baseline = mean_encoded_baseline(
                model.adapter,
                select_baseline_graphs(model.adapter, "test", config, args.sample_graphs, seed=args.seed),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{model.name:22s} skipped ({type(exc).__name__}: {exc})")
            continue
        measured: list[float] = []
        pred_frozen: list[float] = []
        pred_rig: list[float] = []
        joint_meas: list[float] = []
        joint_frozen: list[float] = []
        joint_rig: list[float] = []
        for gi, graph in enumerate(graphs):
            try:
                enc = model.adapter.encoded_node_states(graph).to(model.adapter.device)
                base = expanded_baseline(enc, baseline.to(model.adapter.device))
                clean_pred = float(predict_scalar_from_encoded(model.adapter, graph, enc).detach().cpu().item())
                base_pred = float(predict_scalar_from_encoded(model.adapter, graph, base).detach().cpu().item())
                c_frozen = carriage_ig(model.adapter, graph, baseline, steps=args.ig_steps, readout_ig=False)["carriage"]
                c_rig = carriage_ig(model.adapter, graph, baseline, steps=args.ig_steps, readout_ig=True)["carriage"]
                joint_meas.append(clean_pred - base_pred)  # (B) all-baseline dy
                joint_frozen.append(float(c_frozen.sum().item()))
                joint_rig.append(float(c_rig.sum().item()))
                for j in range(int(enc.size(0))):
                    pert = enc.detach().clone()
                    pert[j] = base[j]
                    pj = float(predict_scalar_from_encoded(model.adapter, graph, pert).detach().cpu().item())
                    measured.append(clean_pred - pj)  # (A) single-node-baseline dy_j
                    pred_frozen.append(float(c_frozen[:, j].sum().item()))
                    pred_rig.append(float(c_rig[:, j].sum().item()))
            except Exception as exc:  # noqa: BLE001
                print(f"  {model.name} graph {gi}: failed ({type(exc).__name__}: {exc})")
        if len(measured) >= 3 and len(joint_meas) >= 1:
            r2f, r2r = _r2(measured, pred_frozen), _r2(measured, pred_rig)
            ef, er = _relerr(joint_meas, joint_frozen), _relerr(joint_meas, joint_rig)
            print(f"{model.name:22s} {len(measured):6d} | {r2f:12.3f} {r2r:10.3f} | {ef:16.3f} {er:13.3f}")
        else:
            print(f"{model.name:22s} insufficient data (src={len(measured)}, graphs={len(joint_meas)})")
    print("\nHow to read this:")
    print("  * (B) relerr rIG ~ 0 everywhere  => readout-IG is computing exactly what it should (exact")
    print("    reconstruction); the negative (A) R2 is therefore NOT a bug.")
    print("  * (A): readout-IG only helps the PER-SOURCE marginal where the model is non-additive (dense).")
    print("    For near-additive models (1-hop) it can hurt (A) -- it integrates the readout gradient")
    print("    through the out-of-distribution mean-baseline region. frozen-g stays at the clean point.")
    print("  => For steps 4-5 (per-node causal usage) frozen-g is the safer default; readout-IG is the")
    print("     right tool when you specifically need exact completeness of the joint reconstruction.")


# --- run it (fires when you paste this file into a Colab cell). First run does the full
#     environment build via the runner (a few minutes); pass --config <path> on later runs
#     in the same session to skip setup. ---
if __name__ == "__main__":
    main([
        "--sample-graphs", "8",
        "--ig-steps", "32",
    ])
