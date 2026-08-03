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
from .cache import atomic_json, load_cache_artifact_file
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
)
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
    from .runner import _stage_plan, run_scores

    progress = prepared.progress
    task_name = str(prepared.task.name)
    train_seed = int(prepared.grit.sc.seed)
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
            scores = run_scores(
                prepared,
                config,
                plan=_stage_plan(prepared, config, "scores"),
                focused_only=True,
            )
        context = progress.component("population_selection") if progress else nullcontext()
        with context:
            confidence_gate = run_confidence_gate(prepared, config, scores)
            population_gate = build_population_gate(
                scores,
                confidence_gate,
                policy,
                analysis_version=GRIT_POPULATION_CAUSAL_VERSION,
            )
            _, cache = _population_cache(
                prepared,
                config,
                population_gate,
                analysis_version=GRIT_POPULATION_CAUSAL_VERSION,
            )
            existing_gate = (
                cache.load("focused_population", "gate", strict=True)
                if config.resume and not config.force
                else None
            )
            if existing_gate is None:
                cache.save("focused_population", "gate", population_gate)
            else:
                population_gate = existing_gate
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
            core = run_population_analysis(
                prepared,
                config,
                scores,
                confidence_gate,
                population_gate,
                execution=execution,
                analysis_version=GRIT_POPULATION_CAUSAL_VERSION,
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
    population_head_pairs: int = 16,
    population_minimum_pairs: int = 12,
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
