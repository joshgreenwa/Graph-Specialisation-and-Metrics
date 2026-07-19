"""One-call orchestration for the per-head specialisation notebook: run(["zinc", "zinc_1hop"]).

The notebook cell is a tiny bootstrap that clones THIS repo (via the ``dissertation_key`` Colab
secret), puts ``<repo>/src`` on sys.path, and calls ``run(...)``. Everything else -- deps, the
GRIT clone (+ the 1-hop patch), checkpoint load, the per-head scoring, ablation, attention capture,
and the figures -- lives here so central edits propagate to every notebook on the next run.

It processes each model FULLY (score -> ablation -> attention) before switching to the next,
because the 1-hop variant imports a *patched* GRIT clone: ``env.prepare_inprocess_grit`` drops the
stale ``grit`` modules and re-imports from the correct clone between tasks (mirroring how
``carriage.colab.run`` handles zinc vs zinc_1hop across separate invocations).

    from graph_specialisation_metrics.specialisation.colab import run
    run(tasks=["zinc", "zinc_1hop"], num_graphs=200, donors=8, ablation_graphs=256)
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..carriage import env
from ..carriage.env import log
from ..carriage.tasks import GritTaskSpec, get_task
from . import ablation as ablation_mod
from . import attention_viz, figures
from .model import SpecConfig
from .scores import score_model, select_heads

DEFAULT_COLLATE_DIR = "/content/drive/MyDrive/graph_specialisation_metrics/specialisation_figures"


def _mount_drive(mount_point: str = "/content/drive") -> None:
    try:
        from google.colab import drive
        log(f"[drive] Mounting Google Drive at {mount_point} ...")
        drive.mount(mount_point, force_remount=False)
    except Exception:  # noqa: BLE001
        log("[drive] google.colab unavailable; assuming a non-Colab run (paths used as-is).")


def _to_jsonable(o):
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def _pick_attention_molecules(abl, n_mol: int) -> list:
    """A spread of test molecules for the attention grid: span the ring-count range."""
    ids = abl["graph_ids"]
    rings = abl["features"]["n_rings"]
    order = np.argsort(rings)
    picks = np.linspace(0, len(order) - 1, num=min(n_mol, len(order))).round().astype(int)
    return [int(ids[order[p]]) for p in picks]


def run(
    tasks: Sequence[str] = ("zinc", "zinc_1hop"),
    *,
    collate_dir: str = DEFAULT_COLLATE_DIR,
    ckpt: Optional[dict] = None,          # optional {task: explicit_ckpt_path}
    # analysis knobs
    num_graphs: int = 200,
    donors: int = 8,
    max_sources: Optional[int] = None,
    with_attn_routing: bool = True,
    ablation_graphs: int = 256,
    n_random_pairs: int = 300,
    n_attention_molecules: int = 5,
    analysis_seed: int = 0,
    partner_match: str = "degree",
    # environment
    seed: int = 42,
    accelerator: str = "cuda:0",
    num_threads: int = 4,
    mount: bool = True,
    skip_install: bool = False,
    pyg_version: str = "2.2.0",
    force_fresh_grit: bool = False,
) -> dict:
    """Run the full per-head specialisation analysis for each task and write all figures.

    Returns a dict with, per task, the score result (numpy), ablation stats, and figure paths;
    plus the shared figure directory.
    """
    if mount:
        _mount_drive()
    out_dir = Path(collate_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not skip_install:
        env.install_dependencies(pyg_version=pyg_version)
    else:
        log("[deps] Skipping dependency installation (skip_install=True).")

    results: dict = {}
    ablations: dict = {}
    attns: dict = {}
    per_task_meta: dict = {}

    for task_name in tasks:
        spec: GritTaskSpec = get_task(task_name)
        log("\n" + "#" * 84 + f"\n# TASK: {spec.name}  ({spec.title})\n" + "#" * 84)
        drive_dir = spec.drive_dir
        if not drive_dir:
            raise ValueError(f"task {spec.name!r} has no drive_dir; register one in carriage.tasks.")
        task_out = out_dir / spec.name
        task_out.mkdir(parents=True, exist_ok=True)
        dataset_dir = str(Path(drive_dir) / "datasets")

        # --- env: clone GRIT (task-specific dir), apply hooks (1-hop patch), import in-process ---
        # A task with env_hooks PATCHES GRIT source; give it its own clone so two such tasks in one
        # run (e.g. peptides_func + peptides_struct, or a k-hop variant) never mutate a shared clone.
        default_dir = f"/content/GRIT_{spec.name}" if spec.env_hooks else "/content/GRIT"
        repo_dir = Path(spec.grit_repo_dir or default_dir)
        env.clone_grit(repo_dir, spec.grit_repo, spec.grit_commit, force_fresh=force_fresh_grit)
        for hook in spec.env_hooks:
            hook(repo_dir)
        env.prepare_inprocess_grit(repo_dir)
        config_file = env.resolve_config(spec, repo_dir, task_out)

        log("\n[env] Runtime")
        log(f"  platform: {platform.platform()} | python {sys.version.split()[0]}")
        try:
            import torch
            log(f"  torch {torch.__version__} | cuda avail={torch.cuda.is_available()}")
        except Exception as exc:  # noqa: BLE001
            log(f"  torch not importable: {exc}")

        explicit = (ckpt or {}).get(task_name)
        chosen_ckpt, _epoch = env.find_checkpoint(Path(drive_dir) / "results", explicit)

        sc = SpecConfig(
            ckpt=str(chosen_ckpt), out_dir=str(task_out), dataset_dir=dataset_dir,
            config_file=config_file, accelerator=accelerator, seed=seed, num_threads=num_threads,
            num_graphs=num_graphs, donors=donors, ablation_graphs=ablation_graphs,
            analysis_seed=analysis_seed, partner_match=partner_match,
        )

        # --- (1) per-head scores ---
        result = score_model(spec, sc, with_attn_routing=with_attn_routing,
                             max_sources=max_sources, seed=analysis_seed)
        # --- (2) ablation ---
        abl = ablation_mod.run_ablation(result, sc, seed=analysis_seed, n_random_pairs=n_random_pairs)
        # --- (3) attention capture for the interesting heads across molecules ---
        hoi = select_heads(result["S_sem"], result["S_str"])
        mol_ids = _pick_attention_molecules(abl, n_attention_molecules)
        attn = attention_viz.collect_attention(result["gm"], mol_ids, list(hoi.values()),
                                               seed=analysis_seed)

        # --- persist per-task artefacts (Drive) ---
        np.savez(task_out / f"scores_{spec.name}.npz",
                 S_sem=result["S_sem"], S_str=result["S_str"],
                 S_attn_sem=(result["S_attn_sem"] if result["S_attn_sem"] is not None
                             else np.zeros_like(result["S_sem"])),
                 func_mean=abl["func_mean"], loss_mean=abl["loss_mean"])
        stats = {
            "title": result["title"], "test_metric": result["test_metric"],
            "test_metric_name": result["test_metric_name"],
            "num_graphs": result["num_graphs"], "donors_K": result["donors_K"],
            "checks": result["checks"], "heads_of_interest": abl["heads_of_interest"],
            "target_stats": abl["target_stats"], "pair_stats": abl["pair_stats"],
            "score_impact_corr": abl["score_impact_corr"], "feature_corr": abl["feature_corr"],
            "attention_molecules": mol_ids,
        }
        (task_out / f"stats_{spec.name}.json").write_text(
            json.dumps(_to_jsonable(stats), indent=2), encoding="utf-8")

        # drop the model reference before switching GRIT clones (frees GPU + avoids stale import).
        result.pop("gm", None)
        results[task_name] = result
        ablations[task_name] = abl
        attns[task_name] = attn
        per_task_meta[task_name] = stats
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # --- figures (need all models together for the global amplitude normalisation) ---
    figs = figures.make_all_figures(results, ablations, attns, out_dir)
    (out_dir / "specialisation_summary.json").write_text(
        json.dumps(_to_jsonable({"tasks": list(tasks), "figures": figs,
                                 "per_task": per_task_meta}), indent=2), encoding="utf-8")
    log("\n[done] Figures + artefacts under: " + str(out_dir))
    for k, v in figs.items():
        log(f"  {k}: {v}")
    return {"figures": figs, "results": results, "ablations": ablations,
            "out_dir": str(out_dir), "per_task": per_task_meta}
