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
from ..carriage.tasks import GritTaskSpec, get_task, resolve_dataset_dir
from . import ablation as ablation_mod
from . import attention_viz, figures
from . import channel_ablation as channel_ablation_mod
from .model import SpecConfig
from .scores import score_model, select_heads


SCORE_CACHE_VERSION = 2  # v2 includes global-VNode rows in per-head transport scores

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
    with_ablation: bool = True,
    with_attention: bool = True,
    ablation_graphs: int = 256,
    n_random_pairs: int = 300,
    n_attention_molecules: int = 5,
    # channel-split causal ablation (I_sem/I_str, functional + loss); off by default because it
    # is the heavy swap x ablate sweep. The cross-model D/J validation figure needs it.
    with_channel_ablation: bool = False,
    channel_ablation_graphs: int = 48,
    channel_ablation_sources: int = 6,
    channel_ablation_donors: int = 3,
    analysis_seed: int = 0,
    partner_match: str = "degree",
    # environment
    seed: int = 42,
    accelerator: str = "cuda:0",
    num_threads: int = 4,
    resume: bool = True,
    checkpoint_every: int = 4,
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
        dataset_dir = resolve_dataset_dir(spec, drive_dir)

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
            resume=resume, checkpoint_every=checkpoint_every,
        )

        # --- (1) per-head scores (always) ---
        result = score_model(spec, sc, with_attn_routing=with_attn_routing,
                             max_sources=max_sources, seed=analysis_seed)
        # --- (2) ablation (optional; the score-only path for cross-model comparison skips it) ---
        abl = None
        if with_ablation:
            abl = ablation_mod.run_ablation(result, sc, seed=analysis_seed,
                                            n_random_pairs=n_random_pairs)
        # --- (3) attention capture for the interesting heads (optional; needs ablation's
        #         per-graph features to pick a molecule spread) ---
        attn = None
        mol_ids: list = []
        if with_attention and abl is not None:
            hoi = select_heads(result["S_sem"], result["S_str"])
            mol_ids = _pick_attention_molecules(abl, n_attention_molecules)
            attn = attention_viz.collect_attention(result["gm"], mol_ids, list(hoi.values()),
                                                   seed=analysis_seed)

        # --- persist per-task artefacts (Drive). The scores_<task>.npz S_sem/S_str [L,H]
        #     matrices are the per-head cache the cross-model deliverables consume. ---
        savez_kw = dict(
            S_sem=result["S_sem"], S_str=result["S_str"],
            S_attn_sem=(result["S_attn_sem"] if result["S_attn_sem"] is not None
                        else np.zeros_like(result["S_sem"])),
            score_cache_version=np.asarray(SCORE_CACHE_VERSION, dtype=np.int64),
        )
        if abl is not None:
            savez_kw.update(func_mean=abl["func_mean"], loss_mean=abl["loss_mean"])
        np.savez(task_out / f"scores_{spec.name}.npz", **savez_kw)

        # --- (2b) channel-split causal ablation (optional; the D/J validation figure needs it) ---
        chan = None
        if with_channel_ablation:
            chan = channel_ablation_mod.run_channel_ablation(
                result["gm"], sc, num_graphs=channel_ablation_graphs,
                max_sources=channel_ablation_sources, donors=channel_ablation_donors,
                seed=analysis_seed)
            np.savez(
                task_out / f"channel_ablation_{spec.name}.npz",
                I_sem_func=chan["I_sem_func"], I_str_func=chan["I_str_func"],
                I_sem_loss=chan["I_sem_loss"], I_str_loss=chan["I_str_loss"],
                overall_func=chan["overall_func"], overall_loss=chan["overall_loss"])
        stats = {
            "title": result["title"], "test_metric": result["test_metric"],
            "test_metric_name": result["test_metric_name"],
            "val_metric": result.get("val_metric"), "val_metric_name": result.get("val_metric_name"),
            "num_graphs": result["num_graphs"], "donors_K": result["donors_K"],
            "score_cache_version": SCORE_CACHE_VERSION,
            "checks": result["checks"], "attention_molecules": mol_ids,
        }
        if abl is not None:
            stats.update({
                "heads_of_interest": abl["heads_of_interest"],
                "target_stats": abl["target_stats"], "pair_stats": abl["pair_stats"],
                "score_impact_corr": abl["score_impact_corr"], "feature_corr": abl["feature_corr"],
            })
        (task_out / f"stats_{spec.name}.json").write_text(
            json.dumps(_to_jsonable(stats), indent=2), encoding="utf-8")

        # drop the model reference before switching GRIT clones (frees GPU + avoids stale import).
        result.pop("gm", None)
        results[task_name] = result
        if abl is not None:
            ablations[task_name] = abl
        if attn is not None:
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
