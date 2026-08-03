"""Population-level causal tests for the dense ZINC and QM9 GRIT checkpoints.

This module reuses the preregistered discovery gate, matched head populations,
restoration/injection/necessity estimands, continuous causal-preference test, and
all-head clean-ablation analysis from the PCQM4Mv2 population study.  Only the
registered task runtime and the native routed-output intervention hooks differ.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from ..carriage.env import log
from .cache import (
    CacheContract,
    CanonicalCache,
    StaleCacheError,
    atomic_json,
    load_cache_artifact_file,
)
from .graphormer_causal_analysis import FocusedExecution, run_confidence_gate
from .graphormer_causal_population import (
    PopulationPolicy,
    _population_cache,
    build_population_gate,
    run_population_analysis,
)
from .protocol import (
    BootstrapPolicy,
    ExecutionPolicy,
    FamilyPolicy,
    MethodologyConfig,
    RunSizes,
    stable_hash,
)
from .scores import CONFIDENCE_SPECIALIST_VERSION
from .tasks import get_task

GRIT_POPULATION_CAUSAL_VERSION = "grit-dense-causal-population-v1"
DENSE_GRIT_TASKS = ("zinc", "qm9_gap_dense")
MODEL_LABELS = {
    "zinc": "Dense ZINC GRIT",
    "qm9_gap_dense": "Dense QM9 GRIT",
}


def _normalise_tasks(tasks: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(tasks, str):
        values = tuple(part.strip() for part in tasks.split(",") if part.strip())
    else:
        values = tuple(str(value) for value in tasks)
    if not values:
        raise ValueError("at least one dense GRIT task is required")
    unknown = sorted(set(values) - set(DENSE_GRIT_TASKS))
    if unknown:
        raise ValueError(
            f"dense causal population tasks must be drawn from {DENSE_GRIT_TASKS}; "
            f"got {unknown}"
        )
    if len(set(values)) != len(values):
        raise ValueError("dense GRIT task names must be unique")
    for name in values:
        task = get_task(name)
        if task.backend_kind != "grit" or task.virtual_node:
            raise ValueError(f"task {name!r} is not a registered dense GRIT task")
    return values


def production_config(
    *,
    output_dir: str,
    tasks: str | Sequence[str] = DENSE_GRIT_TASKS,
    train_seed: int = 42,
    checkpoints: Mapping[str, str] | None = None,
    task_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    accelerator: str = "cuda:0",
    force: bool = False,
    graphs_per_batch: int = 16,
    discovery_graphs: int = 256,
    causal_graphs: int = 256,
    clean_ablation_graphs: int = 256,
    semantic_donor_graphs: int = 2_000,
    sources_per_graph: int = 6,
    donors_per_source: int = 8,
) -> MethodologyConfig:
    """Return the paper-scale, disjoint-population dense-GRIT contract."""

    task_names = _normalise_tasks(tasks)
    seed = int(train_seed)
    return MethodologyConfig(
        output_dir=str(output_dir),
        tasks=task_names,
        train_seeds=(seed,),
        task_train_seeds={name: (seed,) for name in task_names},
        phases=("scores",),
        sizes=RunSizes(
            discovery_graphs=int(discovery_graphs),
            causal_graphs=int(causal_graphs),
            clean_ablation_graphs=int(clean_ablation_graphs),
            semantic_donor_graphs=int(semantic_donor_graphs),
            sources_per_graph=int(sources_per_graph),
            donors_per_source=int(donors_per_source),
            bootstrap_replicates=2_000,
        ),
        families=FamilyPolicy(
            activity_floor=0.20,
            specialist_minimum_pairs=3,
            equivalence_half_width=0.10,
        ),
        bootstrap=BootstrapPolicy(replicates=2_000, rng_seed=17_071),
        execution=ExecutionPolicy(
            graphs_per_batch=int(graphs_per_batch),
            oom_backoff=True,
        ),
        accelerator=str(accelerator),
        checkpoints=dict(checkpoints or {}),
        task_overrides={
            str(name): dict(values)
            for name, values in dict(task_overrides or {}).items()
        },
        resume=True,
        force=bool(force),
        skip_install=True,
        compute_beneficial_carriage=False,
    )


def _artifact_paths(
    root: Path,
    task_name: str,
    train_seed: int,
) -> dict[str, Path]:
    base = root / str(task_name) / f"seed_{int(train_seed)}"
    return {
        "model": base / "model.json",
        "scores": base / "cache" / "focused" / "scores" / "raw_inputs_v1.pt",
        "confidence_gate": base / "cache" / "focused" / "gate.pt",
        "population_gate": base / "cache" / "focused_population" / "gate.pt",
        "core": base / "cache" / "focused_population" / "core_tests.pt",
        "figures": base / "figures" / "focused_causal_population",
        "manifest": base / "focused_causal_population_manifest.json",
    }


def _validated_artifact_cache(
    prepared: Any,
    config: MethodologyConfig,
    path: Path,
    *,
    stale_policy: str = "archive",
) -> tuple[Any, CanonicalCache] | None:
    """Load an artifact only when every non-manifest contract field is current."""

    if not path.is_file():
        return None
    try:
        artifact = load_cache_artifact_file(path)
        stored = CacheContract(**dict(artifact.metadata["contract"]))
    except (FileNotFoundError, KeyError, TypeError, StaleCacheError, OSError, RuntimeError):
        log(f"[cache] ignored an unreadable resume artifact: {path}")
        return None
    from .runner import _cache

    expected = _cache(
        prepared,
        config,
        {},
        manifest_hash=str(stored.event_manifest_hash),
    )
    if stored.fingerprint != expected.contract.fingerprint:
        log(f"[cache] ignored an incompatible resume artifact: {path}")
        return None
    return artifact, CanonicalCache(
        config.root,
        expected.contract,
        stale_policy=stale_policy,
    )


def _load_cached_scores(
    prepared: Any,
    config: MethodologyConfig,
    path: Path,
) -> Mapping[str, Any] | None:
    if not config.resume or config.force:
        return None
    loaded = _validated_artifact_cache(prepared, config, path)
    if loaded is None:
        return None
    scores = loaded[0].value
    if not isinstance(scores, Mapping) or scores.get(
        "focused_score_inputs_version"
    ) != "focused-raw-score-inputs-v1":
        return None
    log(f"[cache] fast-loaded focused scores for {prepared.task.name}")
    if prepared.progress is not None:
        prepared.progress.emit(
            "cache_hit",
            phase="scores",
            cache="focused_consolidated_fast",
        )
    return scores


def _load_cached_confidence_gate(
    prepared: Any,
    config: MethodologyConfig,
    path: Path,
    scores: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if not config.resume or config.force:
        return None
    loaded = _validated_artifact_cache(prepared, config, path)
    if loaded is None:
        return None
    gate = loaded[0].value
    if (
        not isinstance(gate, Mapping)
        or gate.get("version") != CONFIDENCE_SPECIALIST_VERSION
        or gate.get("score_manifest_hash") != scores.get("manifest_hash")
    ):
        return None
    log("[cache] fast-loaded focused bootstrap-confidence specialist gate")
    return gate


def _population_gate_record(gate: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(gate)
    record.pop("_cache_lineage", None)
    return record


def _population_manifest(gate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": GRIT_POPULATION_CAUSAL_VERSION,
        "policy": gate["policy"],
        "specialist_pairs": gate["specialist_pairs"],
        "null_pairs": gate["null_pairs"],
        "metrics": ("R_align_adj", "I_align_adj", "N_fraction"),
    }


def _strict_cache_for_manifest(
    prepared: Any,
    config: MethodologyConfig,
    event_manifest_hash: str,
) -> CanonicalCache:
    from .runner import _cache

    base = _cache(prepared, config, {}, manifest_hash=str(event_manifest_hash))
    return CanonicalCache(config.root, base.contract, stale_policy="raise")


def _lineage_caches(
    prepared: Any,
    config: MethodologyConfig,
    gate: Mapping[str, Any],
) -> tuple[CanonicalCache, ...]:
    lineage = gate.get("_cache_lineage", {})
    if not isinstance(lineage, Mapping):
        return ()
    hashes = lineage.get("source_event_manifest_hashes", ())
    return tuple(
        _strict_cache_for_manifest(prepared, config, str(manifest_hash))
        for manifest_hash in dict.fromkeys(str(value) for value in hashes)
    )


def _prepare_population_cache(
    prepared: Any,
    config: MethodologyConfig,
    gate: Mapping[str, Any],
    path: Path,
) -> tuple[
    Mapping[int, Mapping[str, Any]],
    CanonicalCache,
    Mapping[str, Any],
    tuple[CanonicalCache, ...],
]:
    """Resume an exact gate or retarget compatible population event shards."""

    from .runner import _cache

    if config.resume and not config.force:
        loaded = _validated_artifact_cache(
            prepared,
            config,
            path,
            stale_policy="raise",
        )
        if loaded is not None:
            artifact, stored_cache = loaded
            stored_gate = artifact.value
            if (
                isinstance(stored_gate, Mapping)
                and stored_gate.get("version") == GRIT_POPULATION_CAUSAL_VERSION
            ):
                inherited = _lineage_caches(prepared, config, stored_gate)
                if stable_hash(_population_gate_record(stored_gate)) == stable_hash(
                    _population_gate_record(gate)
                ):
                    log("[cache] fast-loaded the exact causal population gate")
                    current_cache = CanonicalCache(
                        config.root,
                        stored_cache.contract,
                        stale_policy="archive",
                    )
                    return {}, current_cache, stored_gate, inherited

                first_graph = int(prepared.splits.causal[0])
                has_population_rows = any(
                    stored_cache.path(
                        f"focused_population/events/{channel}",
                        f"graph_{first_graph:06d}",
                    ).is_file()
                    for channel in ("semantic", "structural")
                )
                if stored_gate.get("status") == "estimable" and has_population_rows:
                    source_caches = (
                        CanonicalCache(
                            config.root,
                            stored_cache.contract,
                            stale_policy="raise",
                        ),
                        *inherited,
                    )
                    source_hashes = tuple(
                        dict.fromkeys(
                            cache.contract.event_manifest_hash
                            for cache in source_caches
                        )
                    )
                    resumed_gate = dict(gate)
                    resumed_gate["_cache_lineage"] = {
                        "version": "population-event-retarget-v1",
                        "source_event_manifest_hashes": source_hashes,
                    }
                    event_manifest_hash = stable_hash(
                        {
                            "derivation": "population-event-retarget-v1",
                            "source_event_manifests": source_hashes,
                            "population_manifest": _population_manifest(resumed_gate),
                        }
                    )
                    resumed_cache = _cache(
                        prepared,
                        config,
                        {},
                        manifest_hash=event_manifest_hash,
                    )
                    resumed_cache.load("focused_population", "gate", strict=True)
                    resumed_cache.save("focused_population", "gate", resumed_gate)
                    log(
                        "[cache] retargeting the new population gate from compatible "
                        "cached causal head rows"
                    )
                    return {}, resumed_cache, resumed_gate, source_caches

    plan, cache = _population_cache(
        prepared,
        config,
        gate,
        analysis_version=GRIT_POPULATION_CAUSAL_VERSION,
    )
    existing_gate = (
        cache.load("focused_population", "gate", strict=True)
        if config.resume and not config.force
        else None
    )
    if existing_gate is None:
        cache.save("focused_population", "gate", gate)
        existing_gate = gate
    return plan, cache, existing_gate, ()


def _load_complete_clean_ablation(
    prepared: Any,
    config: MethodologyConfig,
) -> Sequence[Mapping[str, Any]] | None:
    """Read through a completed gate-independent clean-ablation cache."""

    if not config.resume or config.force:
        return None
    graph_ids = tuple(int(value) for value in prepared.splits.clean_ablation)
    if not graph_ids:
        return None
    stage = "focused/clean_ablation"
    first_name = f"graph_{graph_ids[0]:06d}"
    first_path = (
        prepared.output_dir / "cache" / stage / f"{first_name}.pt"
    )
    loaded = _validated_artifact_cache(
        prepared,
        config,
        first_path,
        stale_policy="raise",
    )
    if loaded is None:
        return None
    source_cache = loaded[1]
    expected_heads = {
        (layer, head)
        for layer in range(int(prepared.grit.L))
        for head in range(int(prepared.grit.H))
    }
    rows: list[Mapping[str, Any]] = []
    for graph_id in graph_ids:
        try:
            payload = source_cache.load(
                stage,
                f"graph_{graph_id:06d}",
                strict=True,
            )
        except StaleCacheError:
            return None
        if payload is None or int(payload.get("graph", -1)) != graph_id:
            return None
        graph_rows = tuple(payload.get("rows", ()))
        if {
            (int(row["head"][0]), int(row["head"][1])) for row in graph_rows
        } != expected_heads:
            return None
        rows.extend(graph_rows)
    log("[cache] reused the complete all-head clean-ablation cache")
    return rows


def render_cached_population_figures(
    output_dir: str | Path,
    *,
    task_name: str,
    train_seed: int = 42,
) -> dict[str, Any]:
    """Render one task entirely from cached CPU artifacts."""

    from .graphormer_causal_population_plots import render_population_figure_suite

    task_name = _normalise_tasks((task_name,))[0]
    paths = _artifact_paths(Path(output_dir), task_name, int(train_seed))
    required = ("scores", "population_gate", "core")
    missing = [name for name in required if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError(
            f"{task_name} population figures require a completed run; missing "
            + ", ".join(str(paths[name]) for name in missing)
        )
    artifacts = {name: load_cache_artifact_file(paths[name]) for name in required}
    scores = artifacts["scores"].value
    gate = artifacts["population_gate"].value
    core = artifacts["core"].value
    if gate.get("version") != GRIT_POPULATION_CAUSAL_VERSION:
        raise RuntimeError(f"{task_name} population gate uses a different analysis version")
    if core.get("version") != GRIT_POPULATION_CAUSAL_VERSION:
        raise RuntimeError(f"{task_name} population core uses a different analysis version")
    model_record = (
        json.loads(paths["model"].read_text(encoding="utf-8"))
        if paths["model"].is_file()
        else {}
    )
    figures = render_population_figure_suite(
        scores,
        gate,
        core,
        output_dir=paths["figures"],
        model_label=MODEL_LABELS[task_name],
        common_metadata={
            "analysis_version": GRIT_POPULATION_CAUSAL_VERSION,
            "task": task_name,
            "task_title": model_record.get("title", get_task(task_name).title),
            "train_seed": int(train_seed),
            "checkpoint_sha256": model_record.get("checkpoint_sha256"),
            "score_cache": str(paths["scores"]),
            "score_cache_sha256": artifacts["scores"].file_sha256,
            "population_gate_cache": str(paths["population_gate"]),
            "population_core_cache": str(paths["core"]),
            "population_core_sha256": artifacts["core"].file_sha256,
        },
    )
    manifest = {
        "analysis_version": GRIT_POPULATION_CAUSAL_VERSION,
        "task": task_name,
        "task_title": model_record.get("title", get_task(task_name).title),
        "train_seed": int(train_seed),
        "sample_sizes": core["sample_sizes"],
        "matching_balance": gate["matching_balance"],
        "selected_heads": gate["heads"],
        "execution_audit": core["execution_audit"],
        "figures": figures,
    }
    atomic_json(paths["manifest"], manifest)
    return manifest


def _run_task(
    prepared: Any,
    config: MethodologyConfig,
    policy: PopulationPolicy,
    execution: FocusedExecution,
    *,
    phase: str,
) -> dict[str, Any]:
    from .runner import run_scores

    progress = prepared.progress
    task_name = str(prepared.task.name)
    train_seed = int(prepared.grit.sc.seed)
    paths = _artifact_paths(config.root, task_name, train_seed)
    if progress is not None:
        progress.update(task=task_name, train_seed=train_seed)
        progress.start()
        progress.emit(
            "population_run_start",
            analysis_version=GRIT_POPULATION_CAUSAL_VERSION,
            phase=phase,
            sample_sizes=dataclasses.asdict(config.sizes),
            population_policy=dataclasses.asdict(policy),
            execution=dataclasses.asdict(execution),
        )
    try:
        context = progress.component("population_scores") if progress else nullcontext()
        with context:
            scores = _load_cached_scores(prepared, config, paths["scores"])
            if scores is None:
                scores = run_scores(
                    prepared,
                    config,
                    focused_only=True,
                )
        context = progress.component("population_selection") if progress else nullcontext()
        with context:
            confidence_gate = _load_cached_confidence_gate(
                prepared,
                config,
                paths["confidence_gate"],
                scores,
            )
            if confidence_gate is None:
                confidence_gate = run_confidence_gate(prepared, config, scores)
            population_gate = build_population_gate(
                scores,
                confidence_gate,
                policy,
                analysis_version=GRIT_POPULATION_CAUSAL_VERSION,
            )
            plan, cache, population_gate, reuse_caches = _prepare_population_cache(
                prepared,
                config,
                population_gate,
                paths["population_gate"],
            )
            if progress is not None:
                progress.emit(
                    "population_selection_complete",
                    status=population_gate["status"],
                    specialist_pairs=len(population_gate["specialist_pairs"]),
                    null_heads=len(population_gate["null_pairs"]),
                    candidate_counts=population_gate["candidate_counts"],
                    matching_balance=population_gate["matching_balance"],
                )
        result = {
            "analysis_version": GRIT_POPULATION_CAUSAL_VERSION,
            "task": task_name,
            "train_seed": train_seed,
            "output_dir": str(config.root),
            "population_status": population_gate["status"],
            "population_status_reason": population_gate["status_reason"],
            "head_pair_count": len(population_gate["specialist_pairs"]),
            "null_head_count": len(population_gate["null_pairs"]),
            "matching_balance": population_gate["matching_balance"],
        }
        if population_gate["status"] != "estimable":
            log(
                f"[population] {task_name} is not estimable under the preregistered gate: "
                f"{population_gate['status_reason']}"
            )
            return result
        context = progress.component("population_causal") if progress else nullcontext()
        with context:
            existing_core = (
                cache.load("focused_population", "core_tests", strict=True)
                if config.resume and not config.force
                else None
            )
            clean_rows = (
                None
                if existing_core is not None
                else _load_complete_clean_ablation(prepared, config)
            )
            core = run_population_analysis(
                prepared,
                config,
                scores,
                confidence_gate,
                population_gate,
                execution=execution,
                analysis_version=GRIT_POPULATION_CAUSAL_VERSION,
                plan=plan,
                cache=cache,
                graph_ids=(
                    tuple(sorted(plan))
                    if plan
                    else tuple(int(value) for value in prepared.splits.causal)
                ),
                reuse_caches=reuse_caches,
                reuse_legacy=False,
                clean_rows=clean_rows,
                clean_cache=cache,
                clean_cache_stage="focused_population/clean_ablation",
            )
        result.update(
            {
                "sample_sizes": core["sample_sizes"],
                "execution_audit": core["execution_audit"],
            }
        )
        if phase == "all":
            context = progress.component("population_figures") if progress else nullcontext()
            with context:
                result["manifest"] = render_cached_population_figures(
                    config.root,
                    task_name=task_name,
                    train_seed=train_seed,
                )
        return result
    finally:
        if progress is not None:
            progress.emit("population_run_stop")
            progress.close()


def run(
    *,
    phase: str,
    output_dir: str,
    tasks: str | Sequence[str] = DENSE_GRIT_TASKS,
    train_seed: int = 42,
    checkpoints: Mapping[str, str] | None = None,
    task_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    accelerator: str = "cuda:0",
    force: bool = False,
    force_fresh_grit: bool = False,
    graphs_per_batch: int = 16,
    head_batch_size: int = 64,
    event_batch_size: int = 48,
    population_head_pairs: int = 8,
    population_minimum_pairs: int = 6,
    discovery_graphs: int = 256,
    causal_graphs: int = 256,
    clean_ablation_graphs: int = 256,
    semantic_donor_graphs: int = 2_000,
    sources_per_graph: int = 6,
    donors_per_source: int = 8,
) -> dict[str, Any]:
    """Run or render the matched causal-population analysis for dense GRIT tasks."""

    phase = str(phase).lower()
    if phase not in {"run", "figures", "all"}:
        raise ValueError("phase must be 'run', 'figures', or 'all'")
    task_names = _normalise_tasks(tasks)
    root = Path(output_dir)
    if phase == "figures":
        return {
            "analysis_version": GRIT_POPULATION_CAUSAL_VERSION,
            "tasks": {
                name: render_cached_population_figures(
                    root,
                    task_name=name,
                    train_seed=int(train_seed),
                )
                for name in task_names
            },
        }

    from .runner import _release_runtime_memory, prepare_task

    config = production_config(
        output_dir=str(root),
        tasks=task_names,
        train_seed=int(train_seed),
        checkpoints=checkpoints,
        task_overrides=task_overrides,
        accelerator=accelerator,
        force=force,
        graphs_per_batch=graphs_per_batch,
        discovery_graphs=discovery_graphs,
        causal_graphs=causal_graphs,
        clean_ablation_graphs=clean_ablation_graphs,
        semantic_donor_graphs=semantic_donor_graphs,
        sources_per_graph=sources_per_graph,
        donors_per_source=donors_per_source,
    )
    policy = PopulationPolicy(
        head_pairs=int(population_head_pairs),
        minimum_pairs=int(population_minimum_pairs),
    )
    policy.validate()
    execution = FocusedExecution(
        head_batch_size=int(head_batch_size),
        event_batch_size=int(event_batch_size),
    )
    execution.validate()
    atomic_json(
        root / "focused_grit_causal_population_protocol.json",
        {
            **config.record(),
            "analysis_version": GRIT_POPULATION_CAUSAL_VERSION,
            "phase": phase,
            "population_policy": dataclasses.asdict(policy),
            "focused_execution": dataclasses.asdict(execution),
        },
    )
    results = {}
    for task_name in task_names:
        prepared = prepare_task(
            config,
            task_name,
            int(train_seed),
            force_fresh_grit=bool(force_fresh_grit),
        )
        try:
            results[task_name] = _run_task(
                prepared,
                config,
                policy,
                execution,
                phase=phase,
            )
        finally:
            del prepared
            _release_runtime_memory()
    return {
        "analysis_version": GRIT_POPULATION_CAUSAL_VERSION,
        "output_dir": str(root),
        "tasks": results,
    }


__all__ = [
    "DENSE_GRIT_TASKS",
    "GRIT_POPULATION_CAUSAL_VERSION",
    "production_config",
    "render_cached_population_figures",
    "run",
]
