#!/usr/bin/env python3
"""Run the bipartite-matching GRIT structural-PE refinement experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from graph_specialisation_metrics.methodology.graphbench_pe_refinement import (  # noqa: E402
    SCORE_SYSTEMS,
    STRUCTURAL_ARMS,
    PERefinementConfig,
    PERefinementSizes,
    finalize_pe_refinement,
    lock_pe_refinement_selection,
    run_arm_component,
    run_common_component,
)
from graph_specialisation_metrics.methodology.protocol import ExecutionPolicy  # noqa: E402


def _sizes(profile: str) -> PERefinementSizes:
    if profile == "production":
        return PERefinementSizes()
    if profile == "smoke":
        return PERefinementSizes(
            discovery_graphs=3,
            refinement_graphs=2,
            confirmation_graphs=2,
            clean_ablation_graphs=2,
            semantic_donor_graphs=16,
            semantic_sources_per_graph=2,
            donors_per_source=2,
            taylor_graphs=1,
        )
    raise ValueError(profile)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "mode",
        choices=("worker", "finalize", "lock", "preflight"),
    )
    value.add_argument(
        "--component",
        choices=("common-scores", "common-causal", "common-ablation", "arm"),
    )
    value.add_argument("--arm", choices=STRUCTURAL_ARMS)
    value.add_argument("--score-system", choices=SCORE_SYSTEMS)
    value.add_argument("--seed", type=int)
    value.add_argument(
        "--split",
        choices=("refinement", "confirmation"),
        default="refinement",
    )
    value.add_argument(
        "--profile", choices=("production", "smoke"), default="production"
    )
    value.add_argument(
        "--analysis-output-root",
        type=Path,
        default=Path(
            "/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/"
            "grit_specialisation_bipartite_pe_refinement_v1"
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
    value.add_argument("--graphs-per-batch", type=int, default=16)
    value.add_argument("--head-batch-size", type=int, default=24)
    value.add_argument("--replica-pair-budget", type=int, default=2_000_000)
    value.add_argument("--jacobian-output-chunk", type=int, default=64)
    value.add_argument("--progress-heartbeat-seconds", type=float, default=30.0)
    value.add_argument("--analysis-seed", type=int, default=31_415)
    value.add_argument("--strict-audits", action="store_true")
    value.add_argument("--force", action="store_true")
    value.add_argument("--no-resume", action="store_true")
    return value


def build_config(args: argparse.Namespace) -> PERefinementConfig:
    return PERefinementConfig(
        output_dir=str(args.analysis_output_root),
        training_output_root=str(args.training_output_root),
        dataset_root=str(args.dataset_root),
        pe_cache_root=str(args.pe_cache_root),
        runner_path=str(args.runner_path),
        pe_cache_namespace=str(args.pe_cache_namespace),
        pe_cache_dtype=str(args.pe_cache_dtype),
        sizes=_sizes(args.profile),
        execution=ExecutionPolicy(
            graphs_per_batch=int(args.graphs_per_batch),
            oom_backoff=True,
            replica_pair_budget=int(args.replica_pair_budget),
            jacobian_output_chunk=int(args.jacobian_output_chunk),
            progress_heartbeat_seconds=float(args.progress_heartbeat_seconds),
        ),
        analysis_seed=int(args.analysis_seed),
        accelerator=str(args.accelerator),
        head_batch_size=int(args.head_batch_size),
        strict_audits=bool(args.strict_audits),
        resume=not bool(args.no_resume),
        force=bool(args.force),
    )


def _preflight(config: PERefinementConfig) -> None:
    config.validate()
    missing = []
    for seed in config.seeds:
        checkpoint = (
            Path(config.training_output_root)
            / "bipartite_matching_hard"
            / "grit"
            / f"seed{seed}"
            / "best.pt"
        )
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            missing.append(str(checkpoint))
    if missing:
        raise FileNotFoundError("missing checkpoints:\n" + "\n".join(missing))
    for path in (
        Path(config.runner_path),
        Path(config.dataset_root),
        Path(config.pe_cache_root),
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    cache_namespace = (
        Path(config.pe_cache_root)
        / "hpc_base_v1_5task_5k_pe_cache_matched_params"
        / config.pe_cache_namespace
    )
    required_cache_patterns = (
        "bipartite_matching_hard_train_graphs40000_nodes16_"
        f"*_rrwp16_{config.pe_cache_dtype}.pt",
        "bipartite_matching_hard_val_graphs4000_nodes16_"
        f"*_rrwp16_{config.pe_cache_dtype}.pt",
    )
    missing_patterns = [
        str(cache_namespace / pattern)
        for pattern in required_cache_patterns
        if not any(cache_namespace.glob(pattern))
    ]
    if missing_patterns:
        raise FileNotFoundError(
            "missing required bipartite-matching PE cache(s):\n"
            + "\n".join(missing_patterns)
        )
    print("[OK] PE-refinement configuration is valid")
    print("[OK] checkpoints: seeds 0,1,2,3")
    print("[OK] task: graphbench_bipartite_matching_hard / n=16 validation")
    print(f"[OK] PE cache: {cache_namespace}")
    print(f"[OK] output: {config.output_dir}")
    print(f"[OK] fingerprint: {config.fingerprint}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    config = build_config(args)
    config.validate()
    if args.mode == "preflight":
        _preflight(config)
        return
    if args.mode == "worker":
        if args.seed is None or args.component is None:
            raise ValueError("worker mode requires --seed and --component")
        if args.component == "arm":
            if args.arm is None:
                raise ValueError("arm worker requires --arm")
            run_arm_component(config, int(args.seed), str(args.arm))
        else:
            if args.arm is not None:
                raise ValueError("common worker cannot receive --arm")
            run_common_component(config, int(args.seed), str(args.component))
        return
    if args.mode == "finalize":
        finalize_pe_refinement(config, split=str(args.split))
        return
    if args.arm is None or args.score_system is None:
        raise ValueError("lock mode requires --arm and --score-system")
    path = lock_pe_refinement_selection(
        config,
        arm=str(args.arm),
        score_system=str(args.score_system),
    )
    print(f"selection_lock={path}")


if __name__ == "__main__":
    main()
