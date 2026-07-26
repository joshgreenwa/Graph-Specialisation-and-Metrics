"""Protected causal-only completion for score-cached NAR transition cells.

The N=8 and N=32 score artifacts were deliberately computed in a separate, immutable
``transition_scores`` namespace.  This module attaches the repository's official held-out causal
validation to those scores without invoking ``run_scores`` and without writing into either the
canonical N=4/16/64 tree or the v1 score-extension tree.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..methodology.cache import (
    StaleCacheError,
    atomic_json,
    load_cache_artifact_file,
)
from ..methodology.protocol import ExecutionPolicy, MethodologyConfig, stable_hash
from . import nar_grit_fixed as training
from .nar_canonical_analysis import MODEL_ORDER, register_nar_tasks
from .nar_methodology_extension import (
    DEFAULT_EXTENSION_NAME as SOURCE_EXTENSION_NAME,
    _methodology_policies,
    _parse_csv_ints,
    _parse_csv_strings,
    _read_json,
    _safe_extension_layout,
    load_source_binding,
)


CAUSAL_TRANSITION_VERSION = "nar-causal-transition-v1"
DEFAULT_CAUSAL_EXTENSION_NAME = "nar_causal_transition_v1"


def _causal_root(extension_root: Path) -> Path:
    return extension_root / "canonical"


def _causal_path(extension_root: Path, task: str, seed: int) -> Path:
    return (
        _causal_root(extension_root)
        / str(task)
        / f"seed_{int(seed)}"
        / "cache"
        / "causal"
        / "validation.pt"
    )


def _provenance_path(extension_root: Path, task: str, seed: int) -> Path:
    return (
        extension_root
        / "provenance"
        / str(task)
        / f"seed_{int(seed)}"
        / "causal_validation.json"
    )


def _expected_provenance(binding: Any) -> dict[str, Any]:
    return {
        "causal_transition_version": CAUSAL_TRANSITION_VERSION,
        "task": str(binding.task),
        "seed": int(binding.seed),
        "checkpoint_sha256": str(binding.checkpoint_sha256),
        "source_score_path": str(binding.score_artifact.path),
        "source_score_sha256": str(binding.score_artifact.file_sha256),
        "source_score_contract_fingerprint": str(
            binding.source_contract_fingerprint
        ),
        "source_score_families_fingerprint": stable_hash(
            binding.score_artifact.value.get("families", {})
        ),
        "causal_method": (
            "graph_specialisation_metrics.methodology.validation.run_causal_validation"
        ),
        "score_recomputation": False,
    }


def load_transition_causal_artifact(
    *,
    extension_root: Path,
    binding: Any,
) -> Mapping[str, Any]:
    """Load one completed causal artifact and fail closed on provenance mismatch."""

    path = _causal_path(extension_root, binding.task, binding.seed)
    artifact = load_cache_artifact_file(path)
    contract = artifact.metadata.get("contract", {})
    actual = (
        str(contract.get("task")),
        int(contract.get("train_seed", -1)),
        str(contract.get("checkpoint_sha256")),
    )
    expected = (
        str(binding.task),
        int(binding.seed),
        str(binding.checkpoint_sha256),
    )
    if actual != expected:
        raise StaleCacheError(
            f"transition causal/checkpoint provenance mismatch at {path}: "
            f"{actual} != {expected}"
        )
    if artifact.value.get("families", {}) != binding.score_artifact.value.get(
        "families", {}
    ):
        raise StaleCacheError(
            f"transition causal families at {path} do not match the protected score artifact"
        )
    provenance_path = _provenance_path(
        extension_root, binding.task, binding.seed
    )
    if not provenance_path.exists():
        raise StaleCacheError(
            f"transition causal artifact {path} has no protected source-score provenance"
        )
    provenance = _read_json(provenance_path)
    for key, value in _expected_provenance(binding).items():
        if provenance.get(key) != value:
            raise StaleCacheError(
                f"transition causal provenance mismatch for {binding.task}:"
                f"seed{binding.seed} field {key!r}; existing files were left untouched"
            )
    if provenance.get("causal_artifact_sha256") != artifact.file_sha256:
        raise StaleCacheError(
            f"transition causal artifact hash changed at {path}; file left untouched"
        )
    return artifact.value


def _causal_config(
    *,
    base_canonical_root: Path,
    output_dir: Path,
    task: str,
    seed: int,
    accelerator: str,
    graphs_per_batch: int,
    num_threads: int,
) -> MethodologyConfig:
    record = _read_json(base_canonical_root / "protocol.json")
    sizes, numerical, bootstrap, families, analysis_seed = _methodology_policies(
        base_canonical_root
    )
    return MethodologyConfig(
        output_dir=str(output_dir),
        tasks=(str(task),),
        train_seeds=(int(seed),),
        phases=("causal",),
        sizes=sizes,
        numerical=numerical,
        bootstrap=bootstrap,
        families=families,
        execution=ExecutionPolicy(
            graphs_per_batch=int(graphs_per_batch),
            oom_backoff=True,
            replica_pair_budget=None,
            jacobian_output_chunk=8,
            progress_heartbeat_seconds=30.0,
        ),
        analysis_seed=int(analysis_seed),
        accelerator=str(accelerator),
        num_threads=int(num_threads),
        figure_overrides=dict(record.get("figure_overrides", {})),
        skip_install=True,
        resume=True,
        force=False,
        strict_audits=False,
        compute_beneficial_carriage=False,
    )


def run_causal_completion(
    *,
    base_canonical_root: Path,
    source_extension_root: Path,
    causal_extension_root: Path,
    training_run_dir: Path,
    models: Sequence[str],
    ns: Sequence[int],
    cached_ns: Sequence[int],
    seeds: Sequence[int],
    width: int,
    accelerator: str,
    graphs_per_batch: int,
    num_threads: int,
) -> None:
    """Compute only missing causal cells from already-protected score artifacts."""

    from ..methodology.runner import prepare_task
    from ..methodology.validation import run_causal_validation

    register_nar_tasks(
        models=models,
        analysis_ns=ns,
        width=int(width),
        training_run_dir=training_run_dir,
    )
    for records in ns:
        for model in models:
            for seed in seeds:
                binding = load_source_binding(
                    base_canonical_root=base_canonical_root,
                    extension_root=source_extension_root,
                    cached_ns=cached_ns,
                    model_name=str(model),
                    records=int(records),
                    seed=int(seed),
                )
                path = _causal_path(
                    causal_extension_root, binding.task, binding.seed
                )
                if path.exists():
                    load_transition_causal_artifact(
                        extension_root=causal_extension_root,
                        binding=binding,
                    )
                    print(
                        f"[causal-transition] protected cache exists: "
                        f"{binding.task}:seed{seed}",
                        flush=True,
                    )
                    continue
                config = _causal_config(
                    base_canonical_root=base_canonical_root,
                    output_dir=_causal_root(causal_extension_root),
                    task=binding.task,
                    seed=int(seed),
                    accelerator=str(accelerator),
                    graphs_per_batch=int(graphs_per_batch),
                    num_threads=int(num_threads),
                )
                prepared = prepare_task(config, binding.task, int(seed))
                if str(prepared.checkpoint_sha) != str(binding.checkpoint_sha256):
                    raise StaleCacheError(
                        f"loaded checkpoint changed for {binding.task}:seed{seed}; "
                        "protected score and causal artifacts were left untouched"
                    )
                print(
                    f"[causal-transition] {binding.task}:seed{seed} "
                    "(official causal validation; score cache read-only)",
                    flush=True,
                )
                result = run_causal_validation(
                    prepared,
                    config,
                    binding.score_artifact.value,
                )
                if result.get("families", {}) != binding.score_artifact.value.get(
                    "families", {}
                ):
                    raise RuntimeError("official causal result changed frozen score families")
                artifact = load_cache_artifact_file(path)
                provenance = {
                    **_expected_provenance(binding),
                    "causal_artifact_path": str(path),
                    "causal_artifact_sha256": artifact.file_sha256,
                }
                atomic_json(
                    _provenance_path(
                        causal_extension_root, binding.task, binding.seed
                    ),
                    provenance,
                )
                load_transition_causal_artifact(
                    extension_root=causal_extension_root,
                    binding=binding,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Causal-only completion for protected NAR score-transition caches"
    )
    parser.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/graph_specialisation_metrics/nar_grit",
    )
    parser.add_argument("--training-run-name", default="nar_grit_fixed_n_v3")
    parser.add_argument("--base-analysis-name", default="canonical_nar_analysis_d128")
    parser.add_argument("--source-extension-name", default=SOURCE_EXTENSION_NAME)
    parser.add_argument(
        "--causal-extension-name", default=DEFAULT_CAUSAL_EXTENSION_NAME
    )
    parser.add_argument("--models", default="1hop,2hop,dense")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--cached-ns", default="4,16,64")
    parser.add_argument("--causal-ns", default="8,32")
    parser.add_argument("--analysis-width", type=int, default=128)
    parser.add_argument("--graphs-per-batch", type=int, default=48)
    parser.add_argument("--accelerator", default="cuda:0")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--grit-dir", default="/content/GRIT")
    parser.add_argument("--skip-install", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    models = _parse_csv_strings(args.models)
    seeds = _parse_csv_ints(args.seeds)
    cached_ns = _parse_csv_ints(args.cached_ns)
    causal_ns = _parse_csv_ints(args.causal_ns)
    if any(model not in MODEL_ORDER for model in models):
        raise ValueError(f"models must be drawn from {MODEL_ORDER}")
    if int(args.graphs_per_batch) < 1:
        raise ValueError("graphs-per-batch must be positive")
    drive_root = Path(args.drive_root)
    training_run_dir = drive_root / str(args.training_run_name)
    base_analysis_root = training_run_dir / str(args.base_analysis_name)
    base_canonical_root = base_analysis_root / "canonical"
    source_extension_root = (
        base_analysis_root / "extensions" / str(args.source_extension_name)
    )
    causal_extension_root = (
        base_analysis_root / "extensions" / str(args.causal_extension_name)
    )
    _safe_extension_layout(base_analysis_root, causal_extension_root)
    if causal_extension_root.resolve() == source_extension_root.resolve():
        raise ValueError("causal-extension-name must differ from source-extension-name")
    causal_extension_root.mkdir(parents=True, exist_ok=True)
    training.setup_official_grit(
        Path(args.grit_dir), install=not bool(args.skip_install)
    )
    run_causal_completion(
        base_canonical_root=base_canonical_root,
        source_extension_root=source_extension_root,
        causal_extension_root=causal_extension_root,
        training_run_dir=training_run_dir,
        models=models,
        ns=causal_ns,
        cached_ns=cached_ns,
        seeds=seeds,
        width=int(args.analysis_width),
        accelerator=str(args.accelerator),
        graphs_per_batch=int(args.graphs_per_batch),
        num_threads=int(args.num_threads),
    )
    atomic_json(
        causal_extension_root / "run_summary.json",
        {
            "causal_transition_version": CAUSAL_TRANSITION_VERSION,
            "base_canonical_root": str(base_canonical_root),
            "source_extension_root": str(source_extension_root),
            "causal_extension_root": str(causal_extension_root),
            "models": list(models),
            "seeds": list(seeds),
            "causal_N_values": list(causal_ns),
            "score_recomputation": False,
            "cache_policy": (
                "source scores read-only; completed causal cells immutable and validated "
                "against source score hashes"
            ),
        },
    )
    print(
        f"[done] protected causal-transition caches saved under "
        f"{causal_extension_root}",
        flush=True,
    )
    return {"causal_extension_root": str(causal_extension_root)}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    return run(build_parser().parse_args(list(argv) if argv is not None else None))


__all__ = [
    "CAUSAL_TRANSITION_VERSION",
    "DEFAULT_CAUSAL_EXTENSION_NAME",
    "build_parser",
    "load_transition_causal_artifact",
    "main",
    "run",
    "run_causal_completion",
]


if __name__ == "__main__":
    main()
