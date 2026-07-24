"""End-to-end task runner for the modular scoring-refinement experiment."""

from __future__ import annotations

import gc
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..carriage import env
from ..carriage.structural import full_relabel
from ..carriage.env import log
from ..carriage.tasks import GritTaskSpec, get_task, resolve_dataset_dir
from ..specialisation.model import GritHeadModel, SpecConfig
from .cache import CacheStore, read_csv, read_json, sha256_file, write_csv, write_json
from .config import (
    DEFAULT_OUTPUT_DIR,
    PROTOCOL_VERSION,
    RefinementConfig,
    RunSizes,
    deterministic_splits,
    parse_methods,
)
from .fields import GritFieldCollector, attention_mass_error, reconstruction_error
from .figures import make_all_figures
from .interventions import (
    PE_VARIANTS,
    SEMANTIC_VARIANTS,
    apply_event,
    audit_preservation,
    build_donor_index,
    build_graph_manifest,
    intervention_dose,
)
from .scores import (
    appendix_cosine_group_scores,
    build_score_tables,
    graph_balanced_mean,
    hierarchical_event_mean,
    projected_event_score,
    transposition_permutation,
)
from .validation import (
    build_m1_cross_method_correlations,
    build_m1_arm_agreement,
    build_variant_agreement,
    compare_methods,
    run_causal_patching,
    run_single_head_ablation,
)


SCORE_KEYS = (
    "eg_semantic_single",
    "eg_semantic_transposition",
    "eg_pe_single",
    "eg_pe_transposition",
    "semantic_transport_follow",
    "semantic_transport_invariant",
    "pe_transport_follow",
    "pe_transport_invariant",
    "semantic_attention_follow",
    "semantic_attention_invariant",
    "pe_attention_follow",
    "pe_attention_invariant",
    "topology_eg",
    "clean_throughput",
)


def _ensure_output_layout(task_out: Path) -> None:
    for relative in (
        "cache/manifests",
        "cache/clean_fields",
        "cache/intervention_fields",
        "cache/scores",
        "cache/causal",
        "cache/ablation",
        "tables",
        "figures",
    ):
        (task_out / relative).mkdir(parents=True, exist_ok=True)


def _prepare_task(
    task_name: str,
    cfg: RefinementConfig,
) -> tuple[GritTaskSpec, GritHeadModel, Path, str, str]:
    """Reconstruct the registered GRIT task and load its checkpoint read-only."""

    spec = get_task(task_name)
    task_out = cfg.root / task_name
    _ensure_output_layout(task_out)
    default_repo = f"/content/GRIT_{spec.name}" if spec.env_hooks else "/content/GRIT"
    repo_dir = Path(spec.grit_repo_dir or default_repo)
    env.clone_grit(
        repo_dir,
        spec.grit_repo,
        spec.grit_commit,
        force_fresh=cfg.force_fresh_grit,
    )
    for hook in spec.env_hooks:
        hook(repo_dir)
    env.prepare_inprocess_grit(repo_dir)
    config_file = env.resolve_config(spec, repo_dir, task_out)
    explicit = cfg.checkpoints.get(task_name)
    checkpoint, epoch = env.find_checkpoint(Path(spec.drive_dir) / "results", explicit)
    checkpoint_sha = sha256_file(checkpoint)
    sc = SpecConfig(
        ckpt=str(checkpoint),
        out_dir=str(task_out),
        dataset_dir=resolve_dataset_dir(spec),
        config_file=config_file,
        accelerator=cfg.device,
        seed=cfg.train_seed,
        num_threads=cfg.num_threads,
        eval_split="test",
        donor_split="train",
        eval_metric=True,
        num_graphs=cfg.sizes.score_graphs,
        donors=cfg.sizes.events_per_source,
        ablation_graphs=cfg.sizes.ablation_graphs,
        analysis_seed=cfg.analysis_seed,
        partner_match=cfg.pe_partner_match,
        resume=True,
        checkpoint_every=cfg.checkpoint_every,
    )
    gm = GritHeadModel(spec, sc).load()
    write_json(
        task_out / "model.json",
        {
            "task": task_name,
            "title": spec.title,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_epoch": int(epoch),
            "grit_repository": str(repo_dir),
            "test_metric": gm.test_metric,
            "test_metric_name": gm.checks.get("test_metric_name"),
            "val_metric": gm.val_metric,
            "parameter_count": gm.checks.get("num_parameters"),
        },
    )
    return spec, gm, task_out, str(checkpoint), checkpoint_sha


def _event_group_scores(
    collector: GritFieldCollector,
    gm: Any,
    base: Any,
    clean: Any,
    gradients: Sequence[Any],
    events: Sequence[Mapping[str, Any]],
    *,
    compute_follow: bool,
    cosine_temperature: float | None,
) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]], list[dict[str, Any]]]:
    projected: list[np.ndarray] = []
    agreements: dict[str, list[np.ndarray]] = {}
    diagnostics = []
    variants = [apply_event(base, gm, event) for event in events]
    for event, variant_data in zip(events, variants):
        audit_preservation(base, variant_data, event)
    captures = collector.collect_many(variants, require_grad=False)
    permutations = []
    for event, variant_data, variant in zip(events, variants, captures):
        event_projected = projected_event_score(clean, variant, gradients)
        projected.append(event_projected)
        record = {
            "variant": str(event["variant"]),
            "source": int(event.get("source", -1)),
            "partner": int(event.get("partner", -1)),
            "degree_gap": int(event.get("degree_gap", -1)),
            "dose": intervention_dose(
                base,
                variant_data,
                (
                    "semantic"
                    if str(event["variant"]).startswith("semantic_")
                    else "pe" if str(event["variant"]).startswith("pe_") else "topology"
                ),
            ),
            "prediction_movement": float(
                np.mean(
                    np.abs(
                        clean.prediction.detach().cpu().numpy()
                        - variant.prediction.detach().cpu().numpy()
                    )
                )
            ),
            "eg_mean": float(np.mean(event_projected)),
        }
        diagnostics.append(record)
        if compute_follow:
            permutations.append(
                transposition_permutation(
                int(base.num_nodes),
                int(event["source"]),
                int(event["partner"]),
                device=clean.layers[0].attention.device,
            )
            )
    if compute_follow:
        score = appendix_cosine_group_scores(
            clean,
            captures,
            permutations,
            temperature=cosine_temperature,
        )
        agreements = {key: [value] for key, value in score.items()}
    return projected, agreements, diagnostics


def _score_one_graph(
    gm: Any,
    graph_id: int,
    manifest: Mapping[str, Any],
    *,
    cosine_temperature: float,
) -> dict[str, Any]:
    collector = GritFieldCollector(gm)
    base = gm.eval_ds[int(graph_id)]
    clean, gradients = collector.clean_with_gradients(base)
    reconstruction = max(reconstruction_error(layer) for layer in clean.layers)
    softmax = max(attention_mass_error(layer) for layer in clean.layers)
    if reconstruction > 5.0e-4:
        raise RuntimeError(
            f"complete-message reconstruction failed on graph {graph_id}: "
            f"max error={reconstruction:.3e}"
        )
    if softmax > 1.0e-4:
        raise RuntimeError(
            f"attention rows are not receiver-normalized on graph {graph_id}: "
            f"max error={softmax:.3e}"
        )
    first_group = manifest["sources"][0]
    first_transposition = first_group["semantic_transposition"][0]
    relabelled_data = full_relabel(
        base,
        int(first_transposition["source"]),
        int(first_transposition["partner"]),
    )
    relabelled = collector.collect(relabelled_data, require_grad=False)
    relabel_error = float(
        np.max(
            np.abs(
                clean.prediction.detach().cpu().numpy()
                - relabelled.prediction.detach().cpu().numpy()
            )
        )
    )
    relabel_tolerance = max(
        1.0e-4,
        2.0e-5 * float(np.max(np.abs(clean.prediction.detach().cpu().numpy()))),
    )
    if relabel_error > relabel_tolerance:
        raise RuntimeError(
            f"complete content+structure relabel changed graph prediction on graph "
            f"{graph_id}: error={relabel_error:.3e}, tolerance={relabel_tolerance:.3e}"
        )
    clean_throughput = np.stack(
        [
            np.linalg.norm(
                layer.routed_output.detach().cpu().numpy(), axis=-1
            ).sum(axis=0)
            for layer in clean.layers
        ],
        axis=0,
    )
    event_scores: dict[str, list[list[np.ndarray]]] = {
        name: [] for name in (*SEMANTIC_VARIANTS, *PE_VARIANTS)
    }
    agreement_scores: dict[str, list[list[np.ndarray]]] = {
        "semantic_transport_follow": [],
        "semantic_transport_invariant": [],
        "semantic_attention_follow": [],
        "semantic_attention_invariant": [],
        "semantic_effective_support": [],
        "semantic_attention_mass": [],
        "pe_transport_follow": [],
        "pe_transport_invariant": [],
        "pe_attention_follow": [],
        "pe_attention_invariant": [],
        "pe_effective_support": [],
        "pe_attention_mass": [],
    }
    event_diagnostics = []
    for source_group in manifest["sources"]:
        for variant_name in (*SEMANTIC_VARIANTS, *PE_VARIANTS):
            events = source_group[variant_name]
            compute_follow = variant_name.endswith("transposition")
            projected, agreements, diagnostics = _event_group_scores(
                collector,
                gm,
                base,
                clean,
                gradients,
                events,
                compute_follow=compute_follow,
                cosine_temperature=cosine_temperature,
            )
            event_scores[variant_name].append(projected)
            event_diagnostics.extend(diagnostics)
            if compute_follow:
                prefix = "semantic" if variant_name.startswith("semantic_") else "pe"
                for suffix in (
                    "transport_follow",
                    "transport_invariant",
                    "attention_follow",
                    "attention_invariant",
                    "effective_support",
                    "attention_mass",
                ):
                    agreement_scores[f"{prefix}_{suffix}"].append(agreements[suffix])

    topology_projected, _, topology_diagnostics = _event_group_scores(
        collector,
        gm,
        base,
        clean,
        gradients,
        manifest["topology"],
        compute_follow=False,
        cosine_temperature=cosine_temperature,
    )
    event_diagnostics.extend(topology_diagnostics)
    zero = np.zeros((gm.L, gm.H), dtype=float)
    graph_score = {
        "eg_semantic_single": hierarchical_event_mean(
            event_scores["semantic_single_donor"]
        ),
        "eg_semantic_transposition": hierarchical_event_mean(
            event_scores["semantic_transposition"]
        ),
        "eg_pe_single": hierarchical_event_mean(event_scores["pe_single_donor"]),
        "eg_pe_transposition": hierarchical_event_mean(
            event_scores["pe_transposition"]
        ),
        "topology_eg": (
            np.mean(np.stack(topology_projected), axis=0)
            if topology_projected else zero.copy()
        ),
        "clean_throughput": clean_throughput,
    }
    for key, groups in agreement_scores.items():
        graph_score[key] = hierarchical_event_mean(groups)
    return {
        "graph_id": int(graph_id),
        "scores": graph_score,
        "diagnostics": {
            "reconstruction_error": reconstruction,
            "attention_mass_error": softmax,
            "full_relabel_prediction_error": relabel_error,
            "readout_gradient_norm_min": min(
                float(np.linalg.norm(value.detach().cpu().numpy()))
                for value in gradients
            ),
            "readout_gradient_norm_max": max(
                float(np.linalg.norm(value.detach().cpu().numpy()))
                for value in gradients
            ),
            "topology_available": bool(topology_projected),
            "events": event_diagnostics,
        },
    }


def _load_or_build_donor_index(
    gm: Any,
    cfg: RefinementConfig,
    cache: CacheStore,
) -> dict[str, Any]:
    cached = None if cfg.force else cache.load_torch("manifests", "donor_index")
    if cached is not None:
        return cached
    log(
        f"[donors] indexing up to {cfg.sizes.semantic_pool} semantic graphs and "
        f"{cfg.sizes.topology_pool} topology graphs"
    )
    value = build_donor_index(
        gm.donor_ds,
        semantic_pool=cfg.sizes.semantic_pool,
        topology_pool=cfg.sizes.topology_pool,
        seed=cfg.analysis_seed,
    )
    cache.save_torch("manifests", "donor_index", value)
    return value


def _run_scores(
    task: str,
    gm: Any,
    task_out: Path,
    checkpoint_sha: str,
    cfg: RefinementConfig,
    splits: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    cache = CacheStore(
        task_out,
        protocol_fingerprint=cfg.fingerprint,
        checkpoint_sha=checkpoint_sha,
        task=task,
    )
    donor_index = _load_or_build_donor_index(gm, cfg, cache)
    graph_results = []
    manifests = []
    for position, graph_id in enumerate(splits["score"], start=1):
        manifest = build_graph_manifest(
            gm,
            int(graph_id),
            donor_index,
            sources=cfg.sizes.score_sources,
            events_per_source=cfg.sizes.events_per_source,
            topology_events=cfg.sizes.topology_events,
            seed=cfg.analysis_seed,
            partner_match=cfg.pe_partner_match,
            allow_relaxed_topology=cfg.topology_allow_relaxed,
        )
        manifests.append(manifest)
        manifest_path = task_out / "cache" / "manifests" / f"score_graph_{graph_id}.json"
        write_json(manifest_path, manifest)
        cached = None if cfg.force else cache.load_torch(
            "scores", f"graph_{graph_id}", event_manifest=manifest
        )
        if cached is None:
            log(f"[scores:{task}] graph {position}/{len(splits['score'])} id={graph_id}")
            cached = _score_one_graph(
                gm,
                int(graph_id),
                manifest,
                cosine_temperature=cfg.cosine_temperature,
            )
            cache.save_torch(
                "scores", f"graph_{graph_id}", cached, event_manifest=manifest
            )
        else:
            log(f"[scores:{task}] graph {position}/{len(splits['score'])} id={graph_id} cache hit")
        graph_results.append(cached)

    available_score_keys = tuple(graph_results[0]["scores"])
    missing = sorted(set(SCORE_KEYS) - set(available_score_keys))
    if missing:
        raise RuntimeError(f"score graph cache is missing required fields: {missing}")
    per_graph = {
        key: np.stack([item["scores"][key] for item in graph_results], axis=0)
        for key in available_score_keys
    }
    score = {key: graph_balanced_mean(value) for key, value in per_graph.items()}
    for key in (
        "eg_semantic_single",
        "eg_semantic_transposition",
        "eg_pe_single",
        "eg_pe_transposition",
        "topology_eg",
    ):
        values = np.asarray(score[key], dtype=float)
        finite = values[np.isfinite(values)]
        if not finite.size:
            raise RuntimeError(
                f"{key} has no finite values after aggregation; "
                "refusing to write invalid M1/topology tables"
            )
        nonzero = int(np.count_nonzero(np.abs(finite) > 1.0e-12))
        log(
            f"[score-check:{task}] {key}: max={np.max(np.abs(finite)):.6g}, "
            f"nonzero_heads={nonzero}/{finite.size}"
        )
        if nonzero == 0:
            raise RuntimeError(
                f"{key} is identically zero/non-finite after aggregation; "
                "refusing to write invalid M1/topology tables"
            )
    raw_rows, derived_rows, references = build_score_tables(
        task,
        checkpoint_sha,
        score,
        graph_count=len(graph_results),
        event_count=cfg.sizes.events_per_source,
        methods=cfg.methods,
    )
    agreement_rows = build_variant_agreement(
        score,
        per_graph,
        top_k=cfg.top_k,
        bootstrap_samples=cfg.sizes.bootstrap_samples,
        seed=cfg.analysis_seed,
    )
    agreement_rows.extend(build_m1_arm_agreement(derived_rows, top_k=cfg.top_k))
    cross_method_rows = build_m1_cross_method_correlations(raw_rows)
    diagnostic_rows = []
    for item in graph_results:
        graph_id = int(item["graph_id"])
        diagnostic_rows.extend(
            {"graph_id": graph_id, **row}
            for row in item["diagnostics"]["events"]
        )
    for row in agreement_rows:
        if row["comparison_type"] != "intervention_variant":
            continue
        donor_variant = str(row["donor_variant"])
        transposition_variant = str(row["transposition_variant"])
        donor_events = [
            item for item in diagnostic_rows if item["variant"] == donor_variant
        ]
        transposition_events = [
            item for item in diagnostic_rows if item["variant"] == transposition_variant
        ]
        for label, values in (
            ("donor", donor_events),
            ("transposition", transposition_events),
        ):
            doses = np.asarray([item["dose"] for item in values], dtype=float)
            event_scores = np.asarray([item["eg_mean"] for item in values], dtype=float)
            row[f"{label}_mean_dose"] = (
                float(np.mean(doses)) if len(doses) else float("nan")
            )
            row[f"{label}_noop_rate"] = (
                float(np.mean(doses <= 1.0e-12)) if len(doses) else float("nan")
            )
            row[f"{label}_event_variance"] = (
                float(np.var(event_scores, ddof=1)) if len(event_scores) > 1 else 0.0
            )
    table_dir = task_out / "tables"
    write_csv(table_dir / "raw_head_scores.csv", raw_rows)
    write_csv(table_dir / "derived_head_coordinates.csv", derived_rows)
    write_csv(table_dir / "intervention_variant_agreement.csv", agreement_rows)
    write_csv(table_dir / "m1_cross_method_correlations.csv", cross_method_rows)
    write_csv(table_dir / "intervention_diagnostics.csv", diagnostic_rows)
    cache.save_torch(
        "scores",
        "aggregate",
        {
            "score": score,
            "per_graph": per_graph,
            "references": references,
            "graph_ids": list(splits["score"]),
            "diagnostics": [item["diagnostics"] for item in graph_results],
        },
    )
    return {
        "score": score,
        "per_graph": per_graph,
        "raw_rows": raw_rows,
        "derived_rows": derived_rows,
        "agreement_rows": agreement_rows,
        "cross_method_rows": cross_method_rows,
        "donor_index": donor_index,
    }


def _causal_events(
    gm: Any,
    graph_ids: Sequence[int],
    donor_index: Mapping[str, Any],
    cfg: RefinementConfig,
    task_out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events = []
    manifests = []
    for graph_id in graph_ids:
        manifest = build_graph_manifest(
            gm,
            int(graph_id),
            donor_index,
            sources=cfg.sizes.causal_sources,
            events_per_source=cfg.sizes.causal_events_per_variant,
            topology_events=cfg.sizes.topology_events,
            seed=cfg.analysis_seed + 31,
            partner_match=cfg.pe_partner_match,
            allow_relaxed_topology=cfg.topology_allow_relaxed,
        )
        manifests.append(manifest)
        write_json(
            task_out / "cache" / "manifests" / f"causal_graph_{graph_id}.json",
            manifest,
        )
        base = gm.eval_ds[int(graph_id)]
        for group in manifest["sources"]:
            for name in (*SEMANTIC_VARIANTS, *PE_VARIANTS):
                channel = "semantic" if name.startswith("semantic_") else "pe"
                for event in group[name]:
                    events.append(
                        {
                            "graph_id": int(graph_id),
                            "channel": channel,
                            "intervention": name,
                            "base": base,
                            "variant_data": apply_event(base, gm, event),
                        }
                    )
        for event in manifest["topology"]:
            events.append(
                {
                    "graph_id": int(graph_id),
                    "channel": "topology",
                    "intervention": "topology_matched_donor",
                    "base": base,
                    "variant_data": apply_event(base, gm, event),
                }
            )
    return events, manifests


def _run_validation(
    task: str,
    gm: Any,
    task_out: Path,
    checkpoint_sha: str,
    cfg: RefinementConfig,
    splits: Mapping[str, Sequence[int]],
    score_payload: Mapping[str, Any],
) -> dict[str, Any]:
    cache = CacheStore(
        task_out,
        protocol_fingerprint=cfg.fingerprint,
        checkpoint_sha=checkpoint_sha,
        task=task,
    )
    ablation = None if cfg.force else cache.load_torch("ablation", "single_heads")
    if ablation is None:
        log(f"[validation:{task}] single-head ablation on {len(splits['ablation'])} graphs")
        ablation_rows, ablation_arrays = run_single_head_ablation(
            gm,
            splits["ablation"],
            bootstrap_samples=cfg.sizes.bootstrap_samples,
            seed=cfg.analysis_seed,
        )
        ablation = {"rows": ablation_rows, "arrays": ablation_arrays}
        cache.save_torch("ablation", "single_heads", ablation)
    else:
        log(f"[validation:{task}] ablation cache hit")

    donor_index = score_payload.get("donor_index")
    if donor_index is None:
        donor_index = _load_or_build_donor_index(gm, cfg, cache)
    causal_events, causal_manifests = _causal_events(
        gm, splits["causal"], donor_index, cfg, task_out
    )
    causal = None if cfg.force else cache.load_torch(
        "causal", "head_patching", event_manifest=causal_manifests
    )
    if causal is None:
        log(
            f"[validation:{task}] restore/inject patching over "
            f"{len(causal_events)} held-out events"
        )
        causal_rows, causal_arrays = run_causal_patching(
            gm,
            causal_events,
            bootstrap_samples=cfg.sizes.bootstrap_samples,
            seed=cfg.analysis_seed + 47,
        )
        causal = {"rows": causal_rows, "arrays": causal_arrays}
        cache.save_torch(
            "causal", "head_patching", causal, event_manifest=causal_manifests
        )
    else:
        log(f"[validation:{task}] causal patching cache hit")

    derived_rows = score_payload["derived_rows"]
    ablation_validation, causal_validation, ranking = compare_methods(
        derived_rows,
        ablation["rows"],
        causal["arrays"],
        np.asarray(score_payload["score"]["clean_throughput"]),
        top_k=cfg.top_k,
        seed=cfg.analysis_seed,
    )
    table_dir = task_out / "tables"
    write_csv(table_dir / "ablation_head_effects.csv", ablation["rows"])
    write_csv(table_dir / "causal_head_effects.csv", causal["rows"])
    write_csv(table_dir / "ablation_validation.csv", ablation_validation)
    write_csv(table_dir / "causal_validation.csv", causal_validation)
    write_csv(table_dir / "method_ranking.csv", ranking)
    return {
        "ablation_rows": ablation["rows"],
        "causal_rows": causal["rows"],
        "ablation_validation": ablation_validation,
        "causal_validation": causal_validation,
        "ranking": ranking,
    }


def _read_score_payload(task_out: Path) -> dict[str, Any]:
    protocol_path = task_out / "protocol.json"
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        cached_version = str(protocol.get("protocol_version", "unknown"))
        if cached_version != PROTOCOL_VERSION:
            raise RuntimeError(
                f"score tables under {task_out} use protocol {cached_version!r}, "
                f"but this code requires {PROTOCOL_VERSION!r}; rerun --phase scores "
                "before validation or figure-only rendering"
            )
    raw_rows = read_csv(task_out / "tables" / "raw_head_scores.csv")
    derived_rows = read_csv(task_out / "tables" / "derived_head_coordinates.csv")
    if not raw_rows or not derived_rows:
        raise FileNotFoundError(
            f"score tables are missing under {task_out}; run --phase scores first"
        )
    for name, rows in (
        ("raw_head_scores.csv", raw_rows),
        ("derived_head_coordinates.csv", derived_rows),
    ):
        versions = {str(row.get("protocol_version", "unknown")) for row in rows}
        if versions != {PROTOCOL_VERSION}:
            raise RuntimeError(
                f"{name} under {task_out} contains protocol version(s) "
                f"{sorted(versions)}, expected {PROTOCOL_VERSION!r}; rerun "
                "--phase scores before validation or figure-only rendering"
            )
    return {"raw_rows": raw_rows, "derived_rows": derived_rows}


def _render_task(task_out: Path) -> dict[str, Any]:
    score_payload = _read_score_payload(task_out)
    ablation_rows = read_csv(task_out / "tables" / "ablation_head_effects.csv")
    causal_rows = read_csv(task_out / "tables" / "causal_head_effects.csv")
    figures = make_all_figures(
        score_payload["raw_rows"],
        score_payload["derived_rows"],
        out_dir=task_out / "figures",
        ablation_rows=ablation_rows,
        causal_rows=causal_rows,
    )
    return {
        "figures": figures,
        "validation_available": bool(ablation_rows and causal_rows),
    }


def _release_model(gm: Any) -> None:
    del gm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _cross_task_ranking(cfg: RefinementConfig) -> list[dict[str, Any]]:
    def finite_summary(values: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            return float("nan"), float("nan")
        return float(np.mean(finite)), float(np.max(finite) - np.min(finite))

    by_method: dict[str, list[dict[str, Any]]] = {}
    for task in cfg.tasks:
        for row in read_csv(cfg.root / task / "tables" / "method_ranking.csv"):
            by_method.setdefault(str(row["method"]), []).append(
                {
                    "task": task,
                    "significance_score": float(row["significance_score"]),
                    "role_score": float(row["role_score"]),
                    "significance_rank": int(row["significance_rank"]),
                    "role_rank": int(row["role_rank"]),
                }
            )
    output = []
    for method, rows in sorted(by_method.items()):
        significance = np.asarray([row["significance_score"] for row in rows], dtype=float)
        role = np.asarray([row["role_score"] for row in rows], dtype=float)
        mean_significance, significance_range = finite_summary(significance)
        mean_role, role_range = finite_summary(role)
        output.append(
            {
                "method": method,
                "tasks": len(rows),
                "mean_significance_score": mean_significance,
                "mean_role_score": mean_role,
                "mean_significance_rank": float(
                    np.mean([row["significance_rank"] for row in rows])
                ),
                "mean_role_rank": float(np.mean([row["role_rank"] for row in rows])),
                "significance_score_range": significance_range,
                "role_score_range": role_range,
            }
        )
    return output


def _build_config(
    *,
    tasks: Sequence[str],
    output_dir: str,
    methods: str | Sequence[str],
    phase: str,
    force: bool,
    resume_config: bool,
    fast_dev_run: bool,
    checkpoints: Mapping[str, str] | None,
    device: str,
    skip_install: bool,
    pyg_version: str,
    force_fresh_grit: bool,
    analysis_seed: int,
    cosine_temperature: float,
) -> RefinementConfig:
    protocol_path = Path(output_dir) / "protocol.json"
    if resume_config and protocol_path.exists():
        saved = read_json(protocol_path)
        cfg = RefinementConfig.from_record(
            saved,
            phase=phase,
            output_dir=output_dir,
            checkpoints=checkpoints,
            force=force,
            skip_install=skip_install,
            force_fresh_grit=force_fresh_grit,
        )
        # Runtime selection may narrow tasks/methods without changing cached estimands.
        overrides = {
            **cfg.__dict__,
            "tasks": tuple(tasks),
            "methods": parse_methods(methods),
            "device": device,
            "pyg_version": pyg_version,
        }
        if cosine_temperature is not None:
            overrides["cosine_temperature"] = float(cosine_temperature)
        cfg = RefinementConfig(**overrides)
    else:
        cfg = RefinementConfig(
            output_dir=output_dir,
            tasks=tuple(tasks),
            phase=phase,
            methods=parse_methods(methods),
            sizes=RunSizes.fast() if fast_dev_run else RunSizes(),
            analysis_seed=analysis_seed,
            cosine_temperature=(
                0.1 if cosine_temperature is None else float(cosine_temperature)
            ),
            device=device,
            force=force,
            checkpoints=dict(checkpoints or {}),
            skip_install=skip_install,
            pyg_version=pyg_version,
            force_fresh_grit=force_fresh_grit,
        )
    cfg.validate()
    return cfg


def run(
    tasks: Sequence[str] = ("zinc", "qm9_gap_dense"),
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    methods: str | Sequence[str] = "all",
    phase: str = "all",
    force: bool = False,
    resume_config: bool = False,
    fast_dev_run: bool = False,
    checkpoints: Mapping[str, str] | None = None,
    device: str = "cuda:0",
    skip_install: bool = False,
    pyg_version: str = "2.2.0",
    force_fresh_grit: bool = False,
    analysis_seed: int = 1771,
    cosine_temperature: float | None = None,
) -> dict[str, Any]:
    """Run M1--M6 on any registered dense ``GritTaskSpec`` task."""

    cfg = _build_config(
        tasks=tasks,
        output_dir=output_dir,
        methods=methods,
        phase=phase,
        force=force,
        resume_config=resume_config,
        fast_dev_run=fast_dev_run,
        checkpoints=checkpoints,
        device=device,
        skip_install=skip_install,
        pyg_version=pyg_version,
        force_fresh_grit=force_fresh_grit,
        analysis_seed=analysis_seed,
        cosine_temperature=cosine_temperature,
    )
    cfg.root.mkdir(parents=True, exist_ok=True)
    write_json(cfg.root / "protocol.json", cfg.protocol_record())
    log(
        f"[refinement] protocol={PROTOCOL_VERSION} fingerprint={cfg.fingerprint} "
        f"phase={cfg.phase} tasks={list(cfg.tasks)}"
    )
    if cfg.phase != "figures" and not cfg.skip_install:
        env.install_dependencies(pyg_version=cfg.pyg_version)

    summaries: dict[str, Any] = {}
    for task in cfg.tasks:
        task_out = cfg.root / task
        _ensure_output_layout(task_out)
        if cfg.phase == "figures":
            summaries[task] = _render_task(task_out)
            continue
        log("\n" + "#" * 84)
        log(f"# SCORING REFINEMENT TASK: {task}")
        log("#" * 84)
        spec, gm, task_out, checkpoint, checkpoint_sha = _prepare_task(task, cfg)
        splits = deterministic_splits(len(gm.eval_ds), cfg.sizes, cfg.analysis_seed)
        write_json(task_out / "cache" / "manifests" / "splits.json", splits)
        task_protocol = {
            **cfg.protocol_record(),
            "task": task,
            "task_title": spec.title,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha,
            "splits": splits,
            "runtime": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
            },
        }
        write_json(task_out / "protocol.json", task_protocol)

        score_payload: dict[str, Any]
        if cfg.phase in {"scores", "all"}:
            score_payload = _run_scores(
                task, gm, task_out, checkpoint_sha, cfg, splits
            )
        else:
            table_payload = _read_score_payload(task_out)
            cache = CacheStore(
                task_out,
                protocol_fingerprint=cfg.fingerprint,
                checkpoint_sha=checkpoint_sha,
                task=task,
            )
            aggregate = cache.load_torch("scores", "aggregate")
            if aggregate is None:
                raise FileNotFoundError(
                    f"aggregate score cache missing for validation under {task_out}"
                )
            score_payload = {
                **table_payload,
                "score": aggregate["score"],
                "per_graph": aggregate["per_graph"],
                "donor_index": _load_or_build_donor_index(gm, cfg, cache),
            }

        validation_payload: dict[str, Any] = {}
        if cfg.phase in {"validation", "all"}:
            validation_payload = _run_validation(
                task,
                gm,
                task_out,
                checkpoint_sha,
                cfg,
                splits,
                score_payload,
            )
        if cfg.phase == "all":
            figure_payload = _render_task(task_out)
        else:
            figure_payload = {}
        ranking = validation_payload.get("ranking", [])
        summary = {
            "task": task,
            "checkpoint_sha256": checkpoint_sha,
            "score_graphs": len(splits["score"]),
            "causal_graphs": len(splits["causal"]),
            "ablation_graphs": len(splits["ablation"]),
            "best_significance_method": (
                min(ranking, key=lambda row: int(row["significance_rank"]))["method"]
                if ranking else None
            ),
            "best_role_method": (
                min(ranking, key=lambda row: int(row["role_rank"]))["method"]
                if ranking else None
            ),
            **figure_payload,
        }
        write_json(task_out / "summary.json", summary)
        summaries[task] = summary
        _release_model(gm)
        del gm
    cross_task = _cross_task_ranking(cfg)
    if cross_task:
        write_csv(cfg.root / "tables" / "method_ranking_cross_task.csv", cross_task)
    root_summary = {
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": cfg.fingerprint,
        "tasks": summaries,
        "cross_task_method_ranking": cross_task,
        "best_significance_method_cross_task": (
            min(cross_task, key=lambda row: row["mean_significance_rank"])["method"]
            if cross_task else None
        ),
        "best_role_method_cross_task": (
            min(cross_task, key=lambda row: row["mean_role_rank"])["method"]
            if cross_task else None
        ),
    }
    write_json(cfg.root / "summary.json", root_summary)
    log(f"[done] scoring refinement outputs: {cfg.root}")
    return {"config": cfg.protocol_record(), "tasks": summaries, "output_dir": str(cfg.root)}
