"""One-call orchestration for the per-head specialisation notebook: run(["zinc", "zinc_1hop"]).

The notebook cell is a tiny bootstrap that clones THIS repo (via the ``dissertation_key`` Colab
secret), puts ``<repo>/src`` on sys.path, and calls ``run(...)``. Everything else -- deps, the
GRIT clone (+ the 1-hop patch), checkpoint load, per-head scoring, held-out causal mediation,
effective-transport analysis, persistence, and figures -- lives here so central edits propagate to
every notebook on the next run. The original zero-ablation/attention-grid path remains available
through ``causal_extension=False`` (or as appendix output with ``legacy_outputs=True``).

It processes each model fully (score -> mediation -> transport) before switching to the next,
because the 1-hop variant imports a *patched* GRIT clone: ``env.prepare_inprocess_grit`` drops the
stale ``grit`` modules and re-imports from the correct clone between tasks (mirroring how
``carriage.colab.run`` handles zinc vs zinc_1hop across separate invocations).

    from graph_specialisation_metrics.specialisation.colab import run
    run(tasks=["zinc"], num_graphs=64, donors=96, causal_graphs=64)
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import traceback
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
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _write_rows(path: Path, rows) -> None:
    """Write a heterogeneous list of flat-ish dictionaries without requiring pandas."""

    rows = list(rows or [])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = {}
            for key in fields:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple, np.ndarray)):
                    value = json.dumps(_to_jsonable(value), sort_keys=True)
                elif isinstance(value, np.generic):
                    value = value.item()
                encoded[key] = value
            writer.writerow(encoded)


def _persist_causal_extension(task_out: Path, mediation: dict, transport: dict) -> dict:
    """Persist every confirmation-set endpoint needed to regenerate the two paper figures."""

    ext = task_out / "causal_extension"
    ext.mkdir(parents=True, exist_ok=True)

    calibration = mediation["calibration"]
    importance = np.asarray(calibration["importance"])
    preference = np.asarray(calibration["preference"])
    eligible = np.asarray(calibration["eligible"])
    head_rows = []
    for layer in range(importance.shape[0]):
        for head in range(importance.shape[1]):
            head_rows.append({
                "layer": layer,
                "head": head,
                "importance": importance[layer, head],
                "channel_preference": preference[layer, head],
                "eligible": bool(eligible[layer, head]),
                "semantic_calibrated": calibration["semantic_calibrated"][layer, head],
                "rrwp_role_calibrated": calibration["rrwp_role_calibrated"][layer, head],
            })
    _write_rows(ext / "head_selection.csv", head_rows)
    _write_rows(ext / "mediation_single_head.csv", mediation["single_head"]["summary"])
    _write_rows(ext / "mediation_groups.csv", mediation["groups"]["summary"])
    _write_rows(ext / "mediation_topk_contrasts.csv", mediation["contrasts"]["topk"])
    _write_rows(ext / "mediation_group_interactions.csv", mediation["contrasts"]["interactions"])
    _write_rows(ext / "mediation_continuous_selectivity.csv", mediation["contrasts"]["continuous"])
    (ext / "intervention_bank.json").write_text(
        json.dumps(_to_jsonable(mediation["intervention_bank"]), indent=2), encoding="utf-8"
    )

    mp = mediation["predictions"]
    np.savez_compressed(
        ext / "mediation_predictions.npz",
        clean=mp["clean"], counterfactual=mp["counterfactual"], targets=mp["targets"],
        single_noising=mediation["single_head"]["patched_predictions"]["noising"],
        single_denoising=mediation["single_head"]["patched_predictions"]["denoising"],
        group_noising=mediation["groups"]["patched_predictions"]["noising"],
        group_denoising=mediation["groups"]["patched_predictions"]["denoising"],
        head_order=mediation["head_order"],
        discovery_graph_ids=mediation["discovery_graph_ids"],
        confirmation_graph_ids=mediation["confirmation_graph_ids"],
    )

    np.savez_compressed(
        ext / "effective_transport_head_metrics.npz",
        **{key: np.asarray(value) for key, value in transport.get("head_metrics", {}).items()},
        **{
            f"per_graph__{key}": np.asarray(value)
            for key, value in transport.get("head_metrics_per_graph", {}).items()
        },
    )
    transport_summary_rows = []
    transport_graph_rows = []
    graph_ids = np.asarray(transport.get("graph_ids", []))
    for selector, by_k in transport.get("causal", {}).items():
        for k, by_treatment in by_k.items():
            for treatment, effect in by_treatment.items():
                row = {
                    "selector": selector,
                    "k": int(k),
                    "treatment": treatment,
                    "heads": effect.get("heads", []),
                }
                for endpoint in ("delta_mae", "abs_delta_pred"):
                    summary = effect[endpoint]
                    row[f"{endpoint}_mean"] = summary["mean"]
                    row[f"{endpoint}_ci_low"] = summary["ci_low"]
                    row[f"{endpoint}_ci_high"] = summary["ci_high"]
                transport_summary_rows.append(row)
                delta = np.asarray(effect["delta_mae_per_graph"])
                functional = np.asarray(effect["abs_delta_pred_per_graph"])
                for position, graph_id in enumerate(graph_ids):
                    transport_graph_rows.append({
                        "graph_id": int(graph_id),
                        "selector": selector,
                        "k": int(k),
                        "treatment": treatment,
                        "delta_mae": delta[position],
                        "abs_delta_pred": functional[position],
                    })
    _write_rows(ext / "effective_transport_summary.csv", transport_summary_rows)
    _write_rows(ext / "effective_transport_per_graph.csv", transport_graph_rows)

    summary = {
        "schema_version": 1,
        "mediation": {
            "config": mediation["config"],
            "checks": mediation["checks"],
            "calibration": {
                "semantic_scale": calibration["semantic_scale"],
                "rrwp_role_scale": calibration["rrwp_role_scale"],
                "importance_floor": calibration["importance_floor"],
                "rankings": calibration["rankings"],
            },
            "group_specs": mediation["groups"]["specs"],
            "primary_contrasts": mediation["contrasts"]["primary"],
        },
        "effective_transport": {
            "config": transport.get("config", {}),
            "checks": transport.get("checks", {}),
            "head_selection": transport.get("head_selection", {}),
            "clean_mae": transport.get("clean", {}).get("mae"),
        },
    }
    summary_path = ext / "causal_extension_summary.json"
    summary_path.write_text(json.dumps(_to_jsonable(summary), indent=2), encoding="utf-8")
    return {
        "directory": str(ext),
        "summary": str(summary_path),
        "mediation_predictions": str(ext / "mediation_predictions.npz"),
        "transport_head_metrics": str(ext / "effective_transport_head_metrics.npz"),
    }


def _failed_transport_result(
    mediation: dict,
    *,
    topk: Sequence[int],
    primary_k: int,
    residual_permutations: int,
    bootstrap_replicates: int,
    seed: int,
    exc: BaseException,
) -> dict:
    """Serializable failure state: preserve evidence while withholding mechanism estimates."""

    return {
        "schema_version": 1,
        "graph_ids": np.asarray(mediation.get("confirmation_graph_ids", []), dtype=np.int64),
        "topk": np.asarray(sorted({int(k) for k in topk}), dtype=np.int64),
        "primary_k": int(primary_k),
        "primary_available_key": {},
        "head_selection": {},
        "head_metrics": {},
        "head_metrics_per_graph": {},
        "causal": {},
        "clean": {},
        "checks": {
            "passed": False,
            "mechanistic_estimates_withheld": True,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
        },
        "config": {
            "confirmation_only": True,
            "num_confirmation_graphs": int(
                len(np.asarray(mediation.get("confirmation_graph_ids", [])).reshape(-1))
            ),
            "topk": [int(k) for k in sorted({int(value) for value in topk})],
            "primary_k": int(primary_k),
            "residual_permutations": int(residual_permutations),
            "bootstrap_replicates": int(bootstrap_replicates),
            "seed": int(seed),
            "status": "failed_validation",
        },
    }


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
    # default-on, reversible paper analysis
    causal_extension: bool = True,
    causal_graphs: int = 64,
    causal_interventions_per_graph: int = 2,
    causal_topk: Sequence[int] = (1, 2, 4, 8),
    causal_primary_k: int = 4,
    matched_control_draws: int = 8,
    transport_residual_permutations: int = 4,
    bootstrap_replicates: int = 1000,
    legacy_outputs: bool = False,
    # environment
    seed: int = 42,
    accelerator: str = "cuda:0",
    num_threads: int = 4,
    mount: bool = True,
    skip_install: bool = False,
    pyg_version: str = "2.2.0",
    force_fresh_grit: bool = False,
) -> dict:
    """Run the per-head specialisation analysis and write its figures.

    ``causal_extension=True`` is the default paper path: scores are estimated on discovery
    molecules, then channel-specific mediation and effective transport are evaluated on disjoint
    confirmation molecules.  It promotes exactly two paper figures per task.  Set
    ``causal_extension=False`` to restore the previous zero-ablation/attention-grid pipeline;
    ``legacy_outputs=True`` requests those appendix artefacts alongside the extension.
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
    mediations: dict = {}
    transports: dict = {}
    per_task_meta: dict = {}
    extension_artifacts: dict = {}
    do_legacy = (not causal_extension) or bool(legacy_outputs)

    for task_name in tasks:
        spec: GritTaskSpec = get_task(task_name)
        log("\n" + "#" * 84 + f"\n# TASK: {spec.name}  ({spec.title})\n" + "#" * 84)
        drive_dir = spec.drive_dir
        if not drive_dir:
            raise ValueError(
                f"task {spec.name!r} has no drive_dir; register one in carriage.tasks."
            )
        task_out = out_dir / spec.name
        task_out.mkdir(parents=True, exist_ok=True)
        dataset_dir = str(Path(drive_dir) / "datasets")

        # --- env: clone GRIT (task-specific dir), apply hooks (1-hop patch), import in-process ---
        # A task with env_hooks PATCHES GRIT source; give it its own clone so two such tasks in one
        # run (e.g. peptides_func + peptides_struct, or a k-hop variant) never mutate a shared
        # clone.
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
        # Save discovery scores before either optional downstream path begins.
        np.savez_compressed(
            task_out / f"scores_{spec.name}.npz",
            S_sem=result["S_sem"], S_str=result["S_str"],
            S_attn_sem=(result["S_attn_sem"] if result["S_attn_sem"] is not None
                        else np.zeros_like(result["S_sem"])),
            discovery_graph_ids=result["graph_ids"],
        )

        mediation = transport = None
        if causal_extension:
            from .causal_mediation import run_channel_causal_mediation
            from .effective_transport import run_effective_transport

            # --- (2a) held-out, channel-specific noising + denoising ---
            mediation = run_channel_causal_mediation(
                result, sc,
                n_graphs=causal_graphs,
                interventions_per_graph=causal_interventions_per_graph,
                topk=causal_topk,
                primary_k=causal_primary_k,
                matched_control_draws=matched_control_draws,
                bootstrap_replicates=bootstrap_replicates,
                seed=analysis_seed,
            )
            # --- (2b) static routing versus effective projected relational transport ---
            try:
                transport = run_effective_transport(
                    result, sc,
                    graph_ids=mediation["confirmation_graph_ids"],
                    head_selection=mediation["calibration"]["rankings"],
                    topk=causal_topk,
                    primary_k=causal_primary_k,
                    residual_permutations=transport_residual_permutations,
                    bootstrap_replicates=bootstrap_replicates,
                    seed=analysis_seed,
                )
            except Exception as exc:  # noqa: BLE001
                # A failed identity is a scientific result, but no downstream estimate is
                # interpretable.  Preserve the mediation analysis and render a conspicuous
                # failure figure rather than dying before any artefact reaches Drive.
                log("[transport:FAILED] Mechanistic estimates withheld.\n" + traceback.format_exc())
                transport = _failed_transport_result(
                    mediation,
                    topk=causal_topk,
                    primary_k=causal_primary_k,
                    residual_permutations=transport_residual_permutations,
                    bootstrap_replicates=bootstrap_replicates,
                    seed=analysis_seed,
                    exc=exc,
                )
            extension_artifacts[task_name] = _persist_causal_extension(
                task_out, mediation, transport
            )
            mediations[task_name] = mediation
            transports[task_name] = transport

        abl = attn = None
        mol_ids: list[int] = []
        if do_legacy:
            # Exact pre-extension path, retained for reversion and optional appendix output.
            abl = ablation_mod.run_ablation(
                result, sc, seed=analysis_seed, n_random_pairs=n_random_pairs
            )
            hoi = select_heads(result["S_sem"], result["S_str"])
            mol_ids = _pick_attention_molecules(abl, n_attention_molecules)
            attn = attention_viz.collect_attention(
                result["gm"], mol_ids, list(hoi.values()), seed=analysis_seed
            )

        # --- persist per-task artefacts (Drive) ---
        stats = {
            "title": result["title"], "test_metric": result["test_metric"],
            "test_metric_name": result["test_metric_name"],
            "num_graphs": result["num_graphs"], "donors_K": result["donors_K"],
            "checks": result["checks"],
            "causal_extension_enabled": bool(causal_extension),
        }
        if mediation is not None and transport is not None:
            stats["causal_extension"] = {
                "artifacts": extension_artifacts[task_name],
                "mediation_checks": mediation["checks"],
                "primary_contrasts": mediation["contrasts"]["primary"],
                "continuous_selectivity": mediation["contrasts"]["continuous"],
                "effective_transport_checks": transport["checks"],
            }
        if abl is not None:
            stats.update({
                "heads_of_interest": abl["heads_of_interest"],
                "target_stats": abl["target_stats"],
                "pair_stats": abl["pair_stats"],
                "score_impact_corr": abl["score_impact_corr"],
                "feature_corr": abl["feature_corr"],
                "attention_molecules": mol_ids,
            })
            # Exact legacy NPZ layout when the old path is requested.
            np.savez_compressed(
                task_out / f"scores_{spec.name}.npz",
                S_sem=result["S_sem"], S_str=result["S_str"],
                S_attn_sem=(result["S_attn_sem"] if result["S_attn_sem"] is not None
                            else np.zeros_like(result["S_sem"])),
                func_mean=abl["func_mean"], loss_mean=abl["loss_mean"],
            )
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

    # --- figures ---
    legacy_figures = {}
    if do_legacy:
        # Legacy figures need all models together for their global amplitude normalisation.
        legacy_figures = figures.make_all_figures(results, ablations, attns, out_dir)
    main_figures = {}
    if causal_extension:
        from .paper_figures import make_paper_figures
        main_figures = make_paper_figures(results, mediations, transports, out_dir)
        figs = {"main": main_figures}
        if do_legacy:
            figs["legacy"] = legacy_figures
    else:
        # Preserve the pre-extension flat figure dictionary when explicitly reverted.
        figs = legacy_figures
    (out_dir / "specialisation_summary.json").write_text(
        json.dumps(_to_jsonable({"tasks": list(tasks), "figures": figs,
                                 "main_figures": main_figures,
                                 "causal_extension": bool(causal_extension),
                                 "legacy_outputs": bool(do_legacy),
                                 "per_task": per_task_meta}), indent=2), encoding="utf-8")
    log("\n[done] Figures + artefacts under: " + str(out_dir))
    for task_name, task_figures in main_figures.items():
        for figure_name, formats in task_figures.items():
            log(f"  MAIN {task_name}/{figure_name}: {formats['png']}")
    if do_legacy:
        for key, value in legacy_figures.items():
            log(f"  appendix {key}: {value}")
    return {"figures": figs, "results": results, "ablations": ablations,
            "mediations": mediations, "transports": transports,
            "main_figures": main_figures, "extension_artifacts": extension_artifacts,
            "out_dir": str(out_dir), "per_task": per_task_meta}
