#!/usr/bin/env python3
"""Run canonical GRIT specialisation/carriage on trained GraphBench AlgoReas models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from graph_specialisation_metrics.methodology.protocol import (  # noqa: E402
    ExecutionPolicy,
    MethodologyConfig,
    RunSizes,
    parse_csv,
)
from graph_specialisation_metrics.methodology.runner import (  # noqa: E402
    render_cached_figures,
    run_methodology,
)


TASK_ALIASES = {
    "bipartite_matching_hard": "graphbench_bipartite_matching_hard",
    "flow_hard": "graphbench_flow_hard",
}


def sizes_for(profile: str) -> RunSizes:
    if profile == "production":
        return RunSizes(
            discovery_graphs=64,
            causal_graphs=32,
            clean_ablation_graphs=96,
            semantic_donor_graphs=2_000,
            sources_per_graph=16,
            donors_per_source=8,
        )
    if profile == "sensitivity":
        return RunSizes(
            discovery_graphs=48,
            causal_graphs=24,
            clean_ablation_graphs=64,
            semantic_donor_graphs=2_000,
            sources_per_graph=6,
            donors_per_source=8,
        )
    if profile == "smoke":
        return RunSizes.smoke()
    raise ValueError(profile)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--tasks",
        default="bipartite_matching_hard,flow_hard",
        help="GraphBench task names (the graphbench_ canonical prefix is optional).",
    )
    value.add_argument("--seeds", default="0,1,2,3")
    value.add_argument(
        "--phases",
        default="scores,causal,carriage,figures",
        help="scores,causal,carriage,figures; figures alone is model-free/cache-only.",
    )
    value.add_argument(
        "--profile",
        choices=("production", "sensitivity", "smoke"),
        default="production",
    )
    value.add_argument(
        "--analysis-output-root",
        type=Path,
        default=Path(
            "/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/"
            "grit_specialisation_graphbench_edge_v1"
        ),
    )
    value.add_argument(
        "--training-output-root",
        type=Path,
        default=Path(
            "/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/"
            "graphbench_algoreas_hpc_base_v1"
        ),
    )
    value.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/rds/user/jgg45/hpc-work/graphbench-algoreas/datasets"),
    )
    value.add_argument(
        "--pe-cache-root",
        type=Path,
        default=Path("/rds/user/jgg45/hpc-work/graphbench-algoreas/pe_cache"),
    )
    value.add_argument("--pe-cache-namespace", default="base_40k4k4k_n64")
    value.add_argument("--pe-cache-dtype", choices=("float32", "float16"), default="float32")
    value.add_argument(
        "--runner-path",
        type=Path,
        default=REPOSITORY_ROOT / "graphbench-algoreas-hpc" / "bin" / "algoreas_hpc.py",
    )
    value.add_argument("--accelerator", default="cuda:0")
    value.add_argument("--graphs-per-batch", type=int, default=4)
    value.add_argument("--replica-pair-budget", type=int, default=200_000)
    value.add_argument("--jacobian-output-chunk", type=int, default=8)
    value.add_argument("--progress-heartbeat-seconds", type=float, default=30.0)
    value.add_argument("--analysis-seed", type=int, default=31_415)
    value.add_argument("--strict-audits", action="store_true")
    value.add_argument("--force", action="store_true")
    value.add_argument("--no-resume", action="store_true")
    value.add_argument("--skip-beneficial-carriage", action="store_true")
    return value


def build_config(args: argparse.Namespace) -> MethodologyConfig:
    tasks = tuple(TASK_ALIASES.get(name, name) for name in parse_csv(args.tasks))
    unknown = sorted(set(tasks) - set(TASK_ALIASES.values()))
    if unknown:
        raise ValueError(f"unsupported GraphBench GRIT task(s): {unknown}")
    seeds = tuple(int(value) for value in parse_csv(args.seeds))
    checkpoints = {
        f"{task}:{seed}": str(
            args.training_output_root
            / task.removeprefix("graphbench_")
            / "grit"
            / f"seed{seed}"
            / "best.pt"
        )
        for task in tasks
        for seed in seeds
    }
    overrides = {
        task: {
            "runner_path": str(args.runner_path),
            "dataset_root": str(args.dataset_root),
            "pe_cache_root": str(args.pe_cache_root),
            "pe_cache_namespace": str(args.pe_cache_namespace),
            "pe_cache_dtype": str(args.pe_cache_dtype),
            "training_output_root": str(args.training_output_root),
            "require_subset_cache": True,
            "require_pe_cache": True,
            "build_missing_pe_cache": False,
            "eval_split": "val",
            "donor_split": "train",
        }
        for task in tasks
    }
    return MethodologyConfig(
        output_dir=str(args.analysis_output_root),
        tasks=tasks,
        train_seeds=seeds,
        phases=parse_csv(args.phases),
        sizes=sizes_for(args.profile),
        execution=ExecutionPolicy(
            graphs_per_batch=int(args.graphs_per_batch),
            oom_backoff=True,
            replica_pair_budget=int(args.replica_pair_budget),
            jacobian_output_chunk=int(args.jacobian_output_chunk),
            progress_heartbeat_seconds=float(args.progress_heartbeat_seconds),
        ),
        analysis_seed=int(args.analysis_seed),
        accelerator=str(args.accelerator),
        checkpoints=checkpoints,
        task_overrides=overrides,
        resume=not bool(args.no_resume),
        force=bool(args.force),
        strict_audits=bool(args.strict_audits),
        compute_beneficial_carriage=not bool(args.skip_beneficial_carriage),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    config = build_config(args)
    config.validate()
    if tuple(config.phases) == ("figures",):
        for task in config.tasks:
            for seed in config.seeds_for(task):
                print(f"[figures-only] {task}:seed{seed}", flush=True)
                render_cached_figures(config, task, seed)
        return
    run_methodology(config)


if __name__ == "__main__":
    main()
