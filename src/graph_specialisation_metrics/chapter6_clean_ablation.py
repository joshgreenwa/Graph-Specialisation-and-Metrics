"""Focused clean-head ablation measurements for the Chapter 6 comparison.

The canonical 30-run corpus contains score and carriage caches, but it did not run the
causal phase.  This module computes only the independent clean single-head ablation
endpoint needed to validate joint sensitivity.  Canonical per-head shards are reused and
the compact summaries consumed by :mod:`chapter6_multiseed` are written separately.
"""

from __future__ import annotations

import gc
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .chapter6_multiseed import SEEDS, dataset_spec
from .methodology.cache import atomic_json, load_cache_artifact_file

SUMMARY_SCHEMA = "chapter6-clean-head-ablation-v1"


def summary_path(root: Path, task: str, seed: int) -> Path:
    return Path(root) / str(task) / f"seed_{int(seed)}.json"


def _coordinate_array(coordinates: Any, name: str) -> np.ndarray:
    if isinstance(coordinates, Mapping):
        value = coordinates[name]
    else:
        value = getattr(coordinates, name)
    return np.asarray(value, dtype=np.float64)


def _valid_summary(path: Path, *, task: str, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    rows = payload.get("heads")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        return False
    try:
        layers = int(payload["layers"])
        heads = int(payload["heads_per_layer"])
        graphs = int(payload["clean_ablation_graphs"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        payload.get("schema") == SUMMARY_SCHEMA
        and payload.get("task") == str(task)
        and int(payload.get("seed", -1)) == int(seed)
        and graphs > 0
        and len(rows) == layers * heads
        and all(
            isinstance(row, Mapping)
            and np.isfinite(float(row.get("joint_sensitivity", np.nan)))
            and np.isfinite(float(row.get("prediction_movement", np.nan)))
            for row in rows
        )
    )


def missing_runs(
    root: Path,
    *,
    dataset: str,
    seeds: Sequence[int] = SEEDS,
) -> list[tuple[str, int]]:
    spec = dataset_spec(dataset)
    return [
        (task, int(seed))
        for task in spec.tasks
        for seed in seeds
        if not _valid_summary(summary_path(root, task, int(seed)), task=task, seed=int(seed))
    ]


def _summary_payload(
    *,
    task: str,
    seed: int,
    scores: Mapping[str, Any],
    impacts: Mapping[str, Any],
    checkpoint_sha256: str,
    score_cache_sha256: str,
    clean_ablation_graphs: int,
) -> dict[str, Any]:
    joint = _coordinate_array(scores["coordinates"], "joint_sensitivity")
    if joint.ndim != 2:
        raise ValueError("joint sensitivity must have shape [layer, head]")
    rows = []
    for layer in range(joint.shape[0]):
        for head in range(joint.shape[1]):
            name = f"head_L{layer}_H{head}"
            if name not in impacts:
                raise KeyError(f"clean-ablation result is missing {name}")
            rows.append(
                {
                    "layer": int(layer),
                    "head": int(head),
                    "joint_sensitivity": float(joint[layer, head]),
                    "prediction_movement": float(impacts[name]["prediction_movement"]),
                    "loss_change": float(impacts[name]["loss_change"]),
                }
            )
    return {
        "schema": SUMMARY_SCHEMA,
        "task": str(task),
        "seed": int(seed),
        "endpoint": "clean single-head ablation prediction movement",
        "layers": int(joint.shape[0]),
        "heads_per_layer": int(joint.shape[1]),
        "clean_ablation_graphs": int(clean_ablation_graphs),
        "checkpoint_sha256": str(checkpoint_sha256),
        "score_cache_sha256": str(score_cache_sha256),
        "heads": rows,
    }


def consolidate_existing_shards(
    canonical_root: Path,
    summary_root: Path,
    *,
    task: str,
    seed: int,
    clean_ablation_graphs: int = 64,
) -> Path | None:
    """Create the compact summary when every canonical per-head shard already exists."""

    score_path = Path(canonical_root) / task / f"seed_{int(seed)}" / "cache/scores/raw.pt"
    if not score_path.is_file():
        return None
    score_artifact = load_cache_artifact_file(score_path)
    scores = score_artifact.value
    score_contract = dict(score_artifact.metadata["contract"])
    joint = _coordinate_array(scores["coordinates"], "joint_sensitivity")
    impacts: dict[str, Any] = {}
    checkpoint_sha = "unknown"
    observed_graph_counts = set()
    identity_fields = (
        "protocol_fingerprint",
        "task",
        "task_adapter_version",
        "checkpoint_sha256",
        "train_seed",
        "model_geometry",
        "output_representation",
        "sigma",
        "split_fingerprint",
    )
    for layer in range(joint.shape[0]):
        for head in range(joint.shape[1]):
            name = f"head_L{layer}_H{head}"
            path = (
                Path(canonical_root)
                / task
                / f"seed_{int(seed)}"
                / "cache/causal/clean_ablation"
                / f"{name}.pt"
            )
            if not path.is_file():
                return None
            artifact = load_cache_artifact_file(path)
            contract = dict(artifact.metadata["contract"])
            mismatched = [
                field
                for field in identity_fields
                if contract.get(field) != score_contract.get(field)
            ]
            if mismatched:
                raise RuntimeError(
                    f"clean-ablation shard {path} differs from its score cache: {mismatched}"
                )
            impacts[name] = artifact.value
            checkpoint_sha = str(contract.get("checkpoint_sha256", "unknown"))
            observed_graph_counts.add(len(artifact.value.get("graphs", ())))
    if len(observed_graph_counts) != 1:
        raise RuntimeError(
            f"clean-ablation shards disagree on graph count: {sorted(observed_graph_counts)}"
        )
    observed_graphs = observed_graph_counts.pop()
    if observed_graphs != int(clean_ablation_graphs):
        raise RuntimeError(
            f"clean-ablation shards contain {observed_graphs} graphs; "
            f"expected {int(clean_ablation_graphs)}"
        )
    output = summary_path(summary_root, task, seed)
    atomic_json(
        output,
        _summary_payload(
            task=task,
            seed=seed,
            scores=scores,
            impacts=impacts,
            checkpoint_sha256=checkpoint_sha,
            score_cache_sha256=score_artifact.file_sha256,
            clean_ablation_graphs=clean_ablation_graphs,
        ),
    )
    return output


def compute_run(
    config: Any,
    canonical_root: Path,
    summary_root: Path,
    *,
    task: str,
    seed: int,
) -> Path:
    """Compute or resume every single-head clean ablation for one trained model."""

    from .methodology.runner import _release_runtime_memory, prepare_task
    from .methodology.validation import (
        _causal_cache,
        _clean_ablation_stage,
        _targets,
    )

    canonical_root = Path(canonical_root)
    if Path(config.output_dir) != canonical_root:
        raise ValueError("clean ablations must bind to the canonical score-cache root")
    destination = summary_path(summary_root, task, seed)
    if _valid_summary(destination, task=task, seed=seed):
        return destination
    consolidated = consolidate_existing_shards(
        canonical_root,
        summary_root,
        task=task,
        seed=seed,
        clean_ablation_graphs=int(config.sizes.clean_ablation_graphs),
    )
    if consolidated is not None:
        return consolidated

    score_path = canonical_root / task / f"seed_{int(seed)}" / "cache/scores/raw.pt"
    score_artifact = load_cache_artifact_file(score_path)
    scores = score_artifact.value
    prepared = None
    try:
        prepared = prepare_task(config, task, int(seed), force_fresh_grit=False)
        score_checkpoint = str(score_artifact.metadata["contract"].get("checkpoint_sha256", ""))
        if score_checkpoint and score_checkpoint != str(prepared.checkpoint_sha):
            raise RuntimeError("score cache and loaded checkpoint have different SHA-256 values")
        targets = {
            name: family
            for name, family in _targets(prepared, scores).items()
            if name.startswith("head_")
        }
        _, cache = _causal_cache(prepared, config, scores)
        impacts = _clean_ablation_stage(
            prepared,
            config,
            targets,
            scores,
            cache=cache,
        )
        atomic_json(
            destination,
            _summary_payload(
                task=task,
                seed=seed,
                scores=scores,
                impacts=impacts,
                checkpoint_sha256=prepared.checkpoint_sha,
                score_cache_sha256=score_artifact.file_sha256,
                clean_ablation_graphs=int(config.sizes.clean_ablation_graphs),
            ),
        )
    finally:
        if prepared is not None:
            del prepared
        gc.collect()
        _release_runtime_memory()
    return destination


def compute_dataset(
    config: Any,
    canonical_root: Path,
    summary_root: Path,
    *,
    dataset: str,
    seeds: Sequence[int] = SEEDS,
    verbose: bool = True,
) -> list[Path]:
    """Compute the resumable 15-run ablation set for one molecular dataset."""

    spec = dataset_spec(dataset)
    outputs = []
    total = len(spec.tasks) * len(tuple(seeds))
    position = 0
    for task in spec.tasks:
        for seed in seeds:
            position += 1
            if verbose:
                print(
                    f"[chapter6-ablation] {position}/{total}: {task}/seed_{int(seed)}",
                    flush=True,
                )
            outputs.append(
                compute_run(
                    config,
                    canonical_root,
                    summary_root,
                    task=task,
                    seed=int(seed),
                )
            )
    return outputs


def load_rows(
    root: Path,
    *,
    dataset: str,
    seeds: Sequence[int] = SEEDS,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Load compact head-level rows for plotting without constructing a model."""

    rows: list[dict[str, Any]] = []
    missing = []
    spec = dataset_spec(dataset)
    for task in spec.tasks:
        for seed in seeds:
            path = summary_path(root, task, int(seed))
            if not _valid_summary(path, task=task, seed=int(seed)):
                missing.append((task, int(seed)))
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload["heads"]:
                rows.append(
                    {
                        "task": task,
                        "seed": int(seed),
                        "clean_ablation_graphs": int(payload["clean_ablation_graphs"]),
                        **dict(row),
                    }
                )
    if strict and missing:
        detail = ", ".join(f"{task}/seed_{seed}" for task, seed in missing)
        raise FileNotFoundError(f"missing clean-head ablation summaries: {detail}")
    return rows
