"""One-call orchestration for a carriage notebook: run(task="zinc").

The notebook cell is a tiny bootstrap that clones THIS repo (via the ``dissertation_key``
Colab secret), puts ``<repo>/src`` on sys.path, and calls ``run(...)``. Everything else --
deps, GRIT clone, checkpoint load, the analysis, and the figures -- lives here so central
edits to the methodology apply to every notebook on the next run (each run re-clones).

Kept from the original ZINC runner: local version installs, loading the GRIT checkpoint
from Drive, and figure outputs to Drive. New: all task figures collate under one Drive
root so the different GRIT tasks can be compared side by side.

    from graph_specialisation_metrics.carriage.colab import run
    run(task="zinc")                         # auto-discovers the last checkpoint on Drive
    run(task="zinc", beneficial_denom="integrated")  # signed finite-loss attribution
    run(task="zinc", num_graphs=128, donors=64)
    run(task="zinc", skip_install=True)      # same runtime as a finished training run
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Optional

from . import env, figures
from .env import log
from .grit_runner import CarriageConfig, run_grit_carriage
from .structural_runner import run_grit_structural_carriage
from .tasks import GritTaskSpec, get_task

# All tasks' figures collate here (one subfolder per task) so they can be compared.
DEFAULT_COLLATE_DIR = "/content/drive/MyDrive/graph_specialisation_metrics/carriage_figures"


def _mount_drive(mount_point: str = "/content/drive") -> None:
    try:
        from google.colab import drive
        log(f"[drive] Mounting Google Drive at {mount_point} ...")
        drive.mount(mount_point, force_remount=False)
    except Exception:
        log("[drive] google.colab unavailable; assuming a non-Colab run (paths used as-is).")


def run(
    task: str | GritTaskSpec = "zinc",
    *,
    drive_dir: Optional[str] = None,
    collate_dir: str = DEFAULT_COLLATE_DIR,
    ckpt: Optional[str] = None,
    grit_repo_dir: Optional[str] = None,
    # intervention selector:
    #   "semantic"   (DEFAULT): donor-swap node content, hold structure (the validated path).
    #   "structural" (BETA):    perturb topology-derived encodings, hold content. Two modes via
    #       structural_mode: "transposition" (on-manifold node-role swap u<->v, partner v
    #       marginalised over K matched draws -- primary) or "single_node" (off-manifold
    #       footprint copy onto u only -- a cheap functional baseline). partner_match selects
    #       the transposition partner pool ("degree" | "any").
    intervention: str = "semantic",
    structural_mode: str = "transposition",
    partner_match: str = "degree",
    # analysis knobs
    eval_split: str = "test",
    donor_split: str = "test",
    num_graphs: int = 64,
    donors: int = 32,
    graph_select: str = "random",
    analysis_seed: int = 0,
    seed: int = 42,
    accelerator: str = "cuda:0",
    num_threads: int = 4,
    verify: bool = True,
    verify_graphs: int = 2,
    eval_metric: bool = True,
    allow_param_count_drift: bool = False,
    # beneficial-carriage estimator (argument name retained for compatibility):
    #   "integrated": final-state path integral of the exact task loss through the readout.
    #       Signed, donor-wise, and complete (sum_i B=dL); no ratios and no clipping. Adaptive
    #       quadrature localises L1/ReLU kinks and aborts if its checks do not converge.
    #   "slope" (DEFAULT/back-compatible): clean-gradient tangent scaled by a clipped finite
    #       slope. Fast, but clipping is diagnostic evidence that the tangent is inadequate.
    #   "magnitude": B[i,j] = dL_j * |C[i,j]| / sum_i |C[i,j]|. Convex shares, so
    #       |B| <= |dL_j|, but all carriers inherit the source-level sign.
    #   "signed" (LEGACY):     B[i,j] = dL_j * C[i,j] / sum_i C[i,j]. Keeps the per-carrier
    #       sign but the signed sum can cancel to ~0 and make |B| >> |C| spikes at far
    #       distances. Kept only for comparison; prefer "integrated" for signed inference.
    beneficial_denom: str = "slope",
    integrated_atol: float = 1e-5,
    integrated_rtol: float = 1e-4,
    integrated_max_intervals: int = 64,
    tol: float = 1e-4,
    float_noise_tol: float = 5e-3,
    max_replicas: int = 4096,
    max_pair_edges: int = 12_000_000,
    n_boot: int = 2000,
    boot_seed: int = 1234,
    bd_linthresh: float = 0.0,
    # SPD binning of F/B (fixes huge-diameter x-axes + heavy-tailed per-hop CIs):
    #   bin_strategy: "log" (DEFAULT: {0},{1},{2},{3},{4-7},{8-15},{16-31},...), "hop"
    #       (per-hop; fine for small ZINC), or "equal_count" (quantile bins on d>=4).
    #   central: "trimmed" (DEFAULT 20%-trimmed mean over graphs), "median", or "mean".
    #   min_bin_count: bins with fewer pairs are dropped from F/B (default 50).
    bin_strategy: str = "log",
    central: str = "trimmed",
    min_bin_count: int = 50,
    # environment
    mount: bool = True,
    skip_install: bool = False,
    pyg_version: str = "2.2.0",
    force_fresh_grit: bool = False,
) -> dict:
    """Run the semantic-intervention carriage analysis end to end for one GRIT task.

    Two aggregation choices worth knowing (both default to the improved behaviour):

    * ``beneficial_denom`` -- ``"integrated"`` is the signed finite-loss estimator: it
      integrates the task-loss gradient along each donor's swapped-to-clean final-state path,
      then averages donors. It has no ratio or clipping; completeness and adaptive-quadrature
      diagnostics replace the slope clip. ``"slope"`` remains the back-compatible default,
      while ``"magnitude"`` and ``"signed"`` retain the earlier share estimators.
    * ``bin_strategy`` / ``central`` -- F and B are pooled into adaptive shortest-path
      bins and reported with a robust central tendency + graph-clustered bootstrap CI,
      which is what makes the large-graph (peptides) x-axis legible and the CIs tight.
    """
    spec = task if isinstance(task, GritTaskSpec) else get_task(task)
    drive_dir = drive_dir or spec.drive_dir
    if not drive_dir:
        raise ValueError(f"task {spec.name!r} has no drive_dir; pass drive_dir=...")

    if mount:
        _mount_drive()

    # Figures + artifacts collate under <collate_dir>/<task>/ so tasks sit side by side.
    out_dir = Path(collate_dir) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = str(Path(drive_dir) / "datasets")

    if not skip_install:
        env.install_dependencies(pyg_version=pyg_version)
    else:
        log("[deps] Skipping dependency installation (skip_install=True).")

    # A task that patches GRIT source (e.g. 1-hop) uses its own clone dir so it never
    # collides with a dense clone in the same runtime; caller can still override.
    repo_dir = Path(grit_repo_dir or spec.grit_repo_dir or "/content/GRIT")
    env.clone_grit(repo_dir, spec.grit_repo, spec.grit_commit, force_fresh=force_fresh_grit)
    # Task env hooks (e.g. peptides RDKit + dataset/RRWP patches) MUST run before GRIT is
    # imported, since they patch GRIT source files that would otherwise be import-cached.
    for hook in spec.env_hooks:
        hook(repo_dir)
    env.prepare_inprocess_grit(repo_dir)
    config_file = env.resolve_config(spec, repo_dir, out_dir)

    log("\n[env] Runtime")
    log(f"  platform: {platform.platform()}")
    log(f"  python:   {sys.version.split()[0]}")
    try:
        import torch
        log(f"  torch: {torch.__version__} | cuda avail={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            log(f"  gpu: {p.name} | {p.total_memory/1024**3:.1f} GiB")
    except Exception as exc:  # noqa: BLE001
        log(f"  torch not importable: {exc}")

    chosen_ckpt, _epoch = env.find_checkpoint(Path(drive_dir) / "results", ckpt)

    cc = CarriageConfig(
        ckpt=str(chosen_ckpt), out_dir=str(out_dir), dataset_dir=dataset_dir,
        repo_dir=str(repo_dir), config_file=config_file,
        accelerator=accelerator, seed=seed, num_threads=num_threads,
        eval_split=eval_split, donor_split=donor_split, num_graphs=num_graphs,
        donors=donors, graph_select=graph_select, analysis_seed=analysis_seed,
        verify=verify, verify_graphs=verify_graphs, eval_metric=eval_metric,
        allow_param_count_drift=allow_param_count_drift, beneficial_denom=beneficial_denom,
        integrated_atol=integrated_atol, integrated_rtol=integrated_rtol,
        integrated_max_intervals=integrated_max_intervals,
        intervention=intervention, structural_mode=structural_mode, partner_match=partner_match,
        tol=tol, float_noise_tol=float_noise_tol,
        max_replicas=max_replicas, max_pair_edges=max_pair_edges,
    )
    if intervention == "structural":
        results = run_grit_structural_carriage(spec, cc)
        collate_key = f"{spec.name}__structural_{structural_mode}"
    elif intervention == "semantic":
        results = run_grit_carriage(spec, cc)
        collate_key = spec.name
    else:
        raise ValueError(f"intervention must be 'semantic' or 'structural', got {intervention!r}")
    outputs = figures.make_figures_and_save(
        results, str(out_dir), n_boot=n_boot, boot_seed=boot_seed, bd_linthresh=bd_linthresh,
        bin_strategy=bin_strategy, central=central, min_count=min_bin_count)

    _update_collation_index(Path(collate_dir), spec, results, outputs, key=collate_key)
    log(f"\n[done] Task {spec.name!r} figures + data: {out_dir}")
    log(f"[done] Collation root (all tasks): {collate_dir}")
    return {"task": spec.name, "out_dir": str(out_dir), **outputs, "checks": results["checks"]}


def _update_collation_index(collate_root: Path, spec: GritTaskSpec, results: dict, outputs: dict,
                            key: str = None) -> None:
    """Maintain <collate_root>/index.json: one headline row per task for cross-task comparison.

    ``key`` distinguishes interventions on the same task (semantic vs structural_*), so a
    structural run does not clobber the semantic row for the same checkpoint."""
    key = key or spec.name
    index_path = collate_root / "index.json"
    index = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except Exception:  # noqa: BLE001
            index = {}
    curves = outputs["curves"]
    checks, meta = results["checks"], results["meta"]
    index[key] = {
        "title": spec.title,
        "intervention": meta.get("intervention", "semantic"),
        "structural_mode": meta.get("structural_mode"),
        "checkpoint": meta.get("checkpoint"),
        "checkpoint_epoch": meta.get("checkpoint_epoch"),
        "test_metric": checks.get("test_metric"),
        "num_graphs": meta.get("num_graphs"),
        "donors_K": meta.get("donors_K"),
        "B_far_at_0": (curves["B_far_mean"][0] if curves["B_far_mean"] else None),
        "B_far_at_1": (curves["B_far_mean"][1] if len(curves["B_far_mean"]) > 1 else None),
        "figures": outputs["figures"],
        "summary_json": outputs["json"],
    }
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    log(f"[collate] updated {index_path} ({len(index)} task(s): {sorted(index)})")
