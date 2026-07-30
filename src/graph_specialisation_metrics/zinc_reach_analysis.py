"""Bamberger-versus-finite reach analysis for trained ZINC GRIT models.

This extension keeps two comparisons separate:

* The literal Bamberger et al. graph-level proxy is the mean node-level
  pre-pooling Jacobian range.  For categorical ZINC atoms, the input is a
  differentiable one-hot vector followed by the checkpoint's learned embedding.
* Functional carriage is evaluated at final pre-pooling node states for exact
  semantic or structural donor events.
* Semantic profiles additionally expose the raw finite hidden-state response,
  separating finite propagation from task-readout projection.

The Bamberger proxy has no canonical structural-donor counterpart.  It is
therefore shown only for semantic usage; structural usage reports Functional
carriage without inventing a prior-work quantity.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .carriage import env
from .carriage.tasks import get_task as get_grit_task
from .methodology.distance import shortest_path_distances
from .methodology.events import build_channel_events
from .methodology.protocol import (
    BootstrapPolicy,
    ExecutionPolicy,
    MethodologyConfig,
    RunSizes,
    stable_hash,
)
from .methodology.runner import prepare_task
from .methodology.sampling import sample_sources


ANALYSIS_VERSION = "zinc-bamberger-functional-reach-v5"
TASKS = ("zinc_1hop", "zinc_2hop", "zinc_1hop_vnode", "zinc")
TASK_LABELS = {
    "zinc_1hop": "1-hop GRIT",
    "zinc_2hop": "2-hop GRIT",
    "zinc_1hop_vnode": "1-hop GRIT + VN",
    "zinc": "Dense GRIT",
}
CHANNELS = ("semantic", "structural")
DONOR_METHODS = ("functional_carriage",)
DONOR_PROFILE_METHODS = (
    "finite_hidden_response",
    "functional_carriage",
)
DECOMPOSITION_METHODS = (
    "bamberger",
    "finite_hidden_response",
    "functional_carriage",
)
METHOD_LABELS = {
    "bamberger": "Bamberger (coordinatewise Jacobian)",
    "finite_hidden_response": "Finite hidden-state response",
    "functional_carriage": "Functional carriage (task-projected)",
}
METHOD_COLOURS = {
    "bamberger": "#202020",
    "finite_hidden_response": "#0072B2",
    "functional_carriage": "#D55E00",
}
METHOD_MARKERS = {
    "bamberger": "^",
    "finite_hidden_response": "D",
    "functional_carriage": "s",
}
METHOD_LINESTYLES = {
    "bamberger": "-",
    "finite_hidden_response": "--",
    "functional_carriage": ":",
}
MODEL_COLOURS = {
    "zinc_1hop": "#0072B2",
    "zinc_2hop": "#009E73",
    "zinc_1hop_vnode": "#CC79A7",
    "zinc": "#D55E00",
}
MODEL_MARKERS = {
    "zinc_1hop": "o",
    "zinc_2hop": "^",
    "zinc_1hop_vnode": "D",
    "zinc": "s",
}
MODEL_LINESTYLES = {
    "zinc_1hop": "-",
    "zinc_2hop": "--",
    "zinc_1hop_vnode": "-.",
    "zinc": ":",
}


@dataclass(frozen=True)
class ZincReachConfig:
    """Scientific and runtime controls for the standalone analysis."""

    tasks: tuple[str, ...] = TASKS
    seed: int = 0
    graphs: int = 16
    sources_per_graph: int = 6
    donors_per_source: int = 4
    semantic_donor_graphs: int = 256
    bamberger_output_nodes: int = 6
    bamberger_output_channels: int = 8
    effect_floor: float = 1.0e-12
    bootstrap_replicates: int = 2_000
    analysis_seed: int = 91_021
    accelerator: str = "cuda:0"
    num_threads: int = 4

    def validate(self) -> None:
        if not self.tasks or any(task not in TASKS for task in self.tasks):
            raise ValueError(f"tasks must be drawn from {TASKS}")
        for name in (
            "graphs",
            "sources_per_graph",
            "donors_per_source",
            "semantic_donor_graphs",
            "bamberger_output_nodes",
            "bamberger_output_channels",
            "bootstrap_replicates",
            "num_threads",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if float(self.effect_floor) <= 0:
            raise ValueError("effect_floor must be positive")

    @property
    def scientific_record(self) -> dict[str, Any]:
        return {
            "analysis_version": ANALYSIS_VERSION,
            "tasks": list(self.tasks),
            "seed": int(self.seed),
            "graphs": int(self.graphs),
            "sources_per_graph": int(self.sources_per_graph),
            "donors_per_source": int(self.donors_per_source),
            "semantic_donor_graphs": int(self.semantic_donor_graphs),
            "bamberger_output_nodes": int(self.bamberger_output_nodes),
            "bamberger_output_channels": int(self.bamberger_output_channels),
            "effect_floor": float(self.effect_floor),
            "bootstrap_replicates": int(self.bootstrap_replicates),
            "analysis_seed": int(self.analysis_seed),
            "channels": list(CHANNELS),
            "finite_estimand": (
                "clean-minus-donor final pre-pooling node-state change, projected "
                "through the clean graph-output Jacobian"
            ),
            "bamberger_estimand": (
                "mean node-level pre-pooling range from entrywise-absolute Jacobians "
                "with respect to differentiable one-hot atom inputs"
            ),
            "finite_hidden_estimand": (
                "finite clean-minus-donor final pre-pooling hidden-state response"
            ),
            "aggregation": (
                "donor-normalise; donor -> source -> graph; 95% graph bootstrap"
            ),
        }

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.scientific_record)

def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        temporary.write_text("", encoding="utf-8")
        os.replace(temporary, path)
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _repository_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _seed(config: ZincReachConfig, *parts: Any) -> int:
    digest = stable_hash(
        {"analysis_seed": int(config.analysis_seed), "parts": list(parts)},
        length=16,
    )
    return int(digest, 16) % (2**32)


def discover_seed_checkpoint(results_root: str | Path, seed: int = 0) -> Path:
    """Resolve one seed's best checkpoint without relying on modification time."""

    root = Path(results_root)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint results root not found: {root}")

    standard = [
        path
        for path in root.glob("**/ckpt/*.ckpt")
        if path.parent.parent.name == str(int(seed))
    ]
    if standard:
        numeric = [path for path in standard if path.stem.isdigit()]
        return (
            max(numeric, key=lambda path: int(path.stem))
            if numeric
            else sorted(standard)[-1]
        )

    recovery_roots = sorted(
        path
        for path in (root / "_recovery_checkpoints").glob(f"seed{int(seed)}_*")
        if path.is_dir()
    )
    for filename in ("best.ckpt", "latest.ckpt", "first_after_resume.ckpt"):
        matches = [path / filename for path in recovery_roots if (path / filename).is_file()]
        if matches:
            if len(matches) > 1:
                raise RuntimeError(
                    f"ambiguous seed-{seed} recovery checkpoints under {root}: {matches}"
                )
            return matches[0]
    fallbacks = [
        checkpoint
        for directory in recovery_roots
        for checkpoint in sorted(directory.glob("*.ckpt"))
    ]
    if len(fallbacks) == 1:
        return fallbacks[0]
    if fallbacks:
        raise RuntimeError(
            f"ambiguous seed-{seed} recovery checkpoints under {root}: {fallbacks}"
        )
    raise FileNotFoundError(f"no checkpoint for seed {seed} under {root}")


def checkpoint_registry(
    tasks: Sequence[str],
    *,
    seed: int,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return explicit ``task:seed -> path`` entries for the canonical loader."""

    supplied = dict(overrides or {})
    resolved: dict[str, str] = {}
    for task in tasks:
        key = f"{task}:{int(seed)}"
        if key in supplied:
            checkpoint = Path(supplied[key])
            if not checkpoint.is_file():
                raise FileNotFoundError(f"explicit checkpoint does not exist: {checkpoint}")
        else:
            spec = get_grit_task(task)
            checkpoint = discover_seed_checkpoint(
                Path(spec.drive_dir) / "results",
                seed=int(seed),
            )
        resolved[key] = str(checkpoint)
    return resolved


def _methodology_config(
    config: ZincReachConfig,
    *,
    output_dir: Path,
    checkpoints: Mapping[str, str],
) -> MethodologyConfig:
    sizes = RunSizes(
        discovery_graphs=int(config.graphs),
        causal_graphs=1,
        clean_ablation_graphs=1,
        semantic_donor_graphs=int(config.semantic_donor_graphs),
        sources_per_graph=int(config.sources_per_graph),
        donors_per_source=int(config.donors_per_source),
    )
    return MethodologyConfig(
        output_dir=str(output_dir / "prepared"),
        tasks=tuple(config.tasks),
        train_seeds=(int(config.seed),),
        sizes=sizes,
        bootstrap=BootstrapPolicy(),
        execution=ExecutionPolicy(graphs_per_batch=1),
        analysis_seed=int(config.analysis_seed),
        accelerator=str(config.accelerator),
        num_threads=int(config.num_threads),
        checkpoints=dict(checkpoints),
        skip_install=True,
        resume=True,
        strict_audits=False,
        compute_beneficial_carriage=False,
    )


def _atom_embedding(net: Any) -> Any:
    """Find the checkpoint's unique 21-type ZINC atom embedding."""

    import torch

    node_encoder = getattr(getattr(net, "encoder", None), "node_encoder", None)
    candidates = [
        module
        for module in node_encoder.modules()
        if isinstance(module, torch.nn.Embedding) and int(module.num_embeddings) == 21
    ] if node_encoder is not None else []
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one 21-type atom embedding, found {len(candidates)}"
        )
    return candidates[0]


def _after_feature_encoder(prepared: Any, graphs: Sequence[Any]) -> tuple[Any, Any, Any]:
    """Run only the categorical node/edge encoder."""

    from torch_geometric.data import Batch

    if not graphs:
        raise ValueError("at least one graph is required")
    net = prepared.runtime.model.model
    batch = Batch.from_data_list([graph.clone() for graph in graphs]).to(
        prepared.runtime.device
    )
    raw_atoms = batch.x[:, 0].long().detach().clone()
    encoded = net.encoder(batch)
    embedding = _atom_embedding(net)
    expected = embedding(raw_atoms)
    if encoded.x.shape != expected.shape:
        raise RuntimeError("ZINC atom encoder geometry is not the expected single embedding")
    if not bool((encoded.x.detach() - expected.detach()).abs().max() <= 1.0e-6):
        raise RuntimeError("ZINC node encoder is not equivalent to the registered atom embedding")
    return encoded, raw_atoms, embedding


def _finish_encoding(net: Any, after_encoder: Any) -> Any:
    """Run RRWP, optional pre-MP, and optional VNode up to the layer input."""

    data = after_encoder
    if hasattr(net, "rrwp_abs_encoder"):
        data = net.rrwp_abs_encoder(data)
        data = net.rrwp_rel_encoder(data)
    if hasattr(net, "pre_mp"):
        data = net.pre_mp(data)
    global_vnode = getattr(net, "global_vnode", None)
    if global_vnode is not None:
        data = global_vnode(data)
    return data


def _real_mask(data: Any) -> Any:
    import torch

    mask = getattr(data, "real_node_mask", None)
    if mask is None:
        return torch.ones(int(data.x.shape[0]), dtype=torch.bool, device=data.x.device)
    return mask


def _forward_final(net: Any, template: Any, x: Any, edge_attr: Any, real_mask: Any) -> Any:
    data = copy.copy(template)
    data.x = x
    data.edge_attr = edge_attr
    output = net.layers(data)
    return output.x[real_mask]


def _bamberger_rows(
    config: ZincReachConfig,
    prepared: Any,
    *,
    task: str,
    graph_id: int,
    base: Any,
) -> list[dict[str, Any]]:
    """Literal pre-pooling node-level Jacobian proxy for one molecule."""

    import torch
    import torch.nn.functional as functional

    net = prepared.runtime.model.model
    after_encoder, raw_atoms, embedding = _after_feature_encoder(prepared, [base])
    nodes = int(base.num_nodes)
    one_hot = functional.one_hot(
        raw_atoms,
        num_classes=int(embedding.num_embeddings),
    ).to(dtype=embedding.weight.dtype)
    one_hot.requires_grad_(True)

    data = after_encoder.clone()
    data.x = one_hot @ embedding.weight
    layer_input = _finish_encoding(net, data)
    mask = _real_mask(layer_input)
    final = _forward_final(
        net,
        layer_input,
        layer_input.x,
        layer_input.edge_attr,
        mask,
    )
    if int(final.shape[0]) != nodes:
        raise RuntimeError("Bamberger proxy did not return one final state per real node")

    rng = np.random.default_rng(_seed(config, "bamberger", graph_id))
    output_nodes = np.sort(
        rng.choice(
            nodes,
            size=min(nodes, int(config.bamberger_output_nodes)),
            replace=False,
        )
    )
    width = int(final.shape[-1])
    output_channels = np.sort(
        rng.choice(
            width,
            size=min(width, int(config.bamberger_output_channels)),
            replace=False,
        )
    )
    distances = shortest_path_distances(base.edge_index, nodes)
    rows: list[dict[str, Any]] = []
    calls = len(output_nodes) * len(output_channels)
    completed = 0
    for output_node in output_nodes:
        influence = torch.zeros(nodes, device=final.device, dtype=final.dtype)
        for output_channel in output_channels:
            completed += 1
            gradient = torch.autograd.grad(
                final[int(output_node), int(output_channel)],
                one_hot,
                retain_graph=completed < calls,
                allow_unused=False,
            )[0]
            influence += gradient.abs().sum(dim=-1)
        for input_node in range(nodes):
            rows.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "fingerprint": config.fingerprint,
                    "task": task,
                    "model_label": TASK_LABELS[task],
                    "seed": int(config.seed),
                    "graph": int(graph_id),
                    "output_node": int(output_node),
                    "input_node": int(input_node),
                    "distance": int(distances[int(output_node), input_node]),
                    "influence": float(influence[input_node].detach().cpu()),
                    "sampled_output_channels": int(len(output_channels)),
                    "input_space": "differentiable one-hot atom type",
                    "output_space": "final pre-pooling node embedding",
                }
            )
    return rows


def _project_final_change(change: Any, clean_gradient: Any) -> Any:
    """Project ``[E,N,W]`` changes through ``[T,N,W]`` and return ``[E,N]``."""

    import torch

    if change.ndim != 3 or clean_gradient.ndim != 3:
        raise ValueError("change and gradient must be [E,N,W] and [T,N,W]")
    if tuple(change.shape[1:]) != tuple(clean_gradient.shape[1:]):
        raise ValueError("carrier change and output-gradient geometry differ")
    contribution = torch.einsum("enw,tnw->ent", change, clean_gradient)
    return torch.linalg.vector_norm(contribution, dim=-1)


def _donor_rows(
    config: ZincReachConfig,
    prepared: Any,
    *,
    task: str,
    graph_id: int,
    channel: str,
    base: Any,
    sources: Sequence[int],
    variants: Sequence[Any],
    events: Sequence[Any],
    clean_jacobians: Any,
) -> list[dict[str, Any]]:
    """Evaluate finite Functional-carriage mass for one graph/channel."""

    import torch

    capture = prepared.backend.capture_groups([[base, *variants]])[0]
    finite_change = clean_jacobians.capture.final_state.unsqueeze(0) - capture.final_state[1:]
    gradient = clean_jacobians.final_state
    finite_mass = _project_final_change(finite_change, gradient)
    finite_hidden_mass = torch.linalg.vector_norm(finite_change, dim=-1)
    if not bool(
        torch.isfinite(finite_mass).all()
        and torch.isfinite(finite_hidden_mass).all()
    ):
        raise RuntimeError("non-finite final-state reach mass")
    if len(events) != int(finite_mass.shape[0]):
        raise RuntimeError("event manifest and carrier tensor are misaligned")

    distances = shortest_path_distances(base.edge_index, int(base.num_nodes))
    rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        source = int(event.source)
        if source not in sources:
            raise RuntimeError("event source is absent from the frozen source set")
        for carrier in range(int(base.num_nodes)):
            row = {
                "analysis_version": ANALYSIS_VERSION,
                "fingerprint": config.fingerprint,
                "task": task,
                "model_label": TASK_LABELS[task],
                "seed": int(config.seed),
                "graph": int(graph_id),
                "channel": channel,
                "source": source,
                "donor_graph": int(event.donor_graph_id),
                "donor_node": int(event.donor_node),
                "draw": int(event.draw),
                "dose": float(event.dose),
                "carrier": int(carrier),
                "distance": int(distances[source, carrier]),
                "functional_carriage": float(
                    finite_mass[event_index, carrier].detach().cpu()
                ),
            }
            if channel == "semantic":
                row["finite_hidden_response"] = float(
                    finite_hidden_mass[event_index, carrier].detach().cpu()
                )
            rows.append(row)
    return rows


def _measure_graph(
    config: ZincReachConfig,
    prepared: Any,
    *,
    task: str,
    graph_id: int,
) -> dict[str, Any]:
    import torch

    base = prepared.runtime.eval_ds[int(graph_id)]
    sources = tuple(
        int(value)
        for value in sample_sources(
            int(base.num_nodes),
            int(config.sources_per_graph),
            np.random.default_rng(_seed(config, "sources", graph_id)),
        )
    )
    clean_jacobians = prepared.backend.clean_jacobians(base)
    bamberger = _bamberger_rows(
        config,
        prepared,
        task=task,
        graph_id=int(graph_id),
        base=base,
    )
    donor_rows: list[dict[str, Any]] = []
    for channel in CHANNELS:
        variants: list[Any] = []
        events: list[Any] = []
        for source in sources:
            source_variants, source_events = build_channel_events(
                base,
                graph_id=int(graph_id),
                source=int(source),
                channel=channel,
                stage="zinc_reach",
                donors=int(config.donors_per_source),
                rng=np.random.default_rng(
                    _seed(config, "events", graph_id, channel, int(source))
                ),
                task=prepared.task,
                semantic_pool=prepared.donor_pool,
                duplicate_tolerance=1.0e-7,
            )
            variants.extend(source_variants)
            events.extend(source_events)
        if not variants:
            print(
                f"[reach:warning] no estimable {channel} events "
                f"({task}, graph={graph_id})",
                flush=True,
            )
            continue
        rows = _donor_rows(
            config,
            prepared,
            task=task,
            graph_id=int(graph_id),
            channel=channel,
            base=base,
            sources=sources,
            variants=variants,
            events=events,
            clean_jacobians=clean_jacobians,
        )
        donor_rows.extend(rows)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "analysis_version": ANALYSIS_VERSION,
        "fingerprint": config.fingerprint,
        "checkpoint_sha256": str(prepared.checkpoint_sha),
        "task": task,
        "graph": int(graph_id),
        "donor_rows": donor_rows,
        "bamberger_rows": bamberger,
    }


def _shard_path(output_dir: Path, task: str, graph_id: int) -> Path:
    return output_dir / "cache" / task / f"graph_{int(graph_id):06d}.pt"


def _load_shard(
    path: Path,
    *,
    fingerprint: str,
    checkpoint_sha256: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if (
        payload.get("analysis_version") != ANALYSIS_VERSION
        or payload.get("fingerprint") != fingerprint
        or payload.get("checkpoint_sha256") != checkpoint_sha256
        or not {"donor_rows", "bamberger_rows"}.issubset(payload)
    ):
        return None
    return payload


def _save_shard(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def measure(
    config: ZincReachConfig,
    *,
    output_dir: Path,
    checkpoints: Mapping[str, str] | None = None,
    install_dependencies: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Measure all requested models and cache each graph independently on Drive."""

    config.validate()
    if install_dependencies:
        env.install_dependencies(pyg_version="2.2.0")
    env.apply_compat_patches()

    import torch

    torch.set_num_threads(int(config.num_threads))
    resolved = checkpoint_registry(
        config.tasks,
        seed=int(config.seed),
        overrides=checkpoints,
    )
    print("[checkpoints] Explicit checkpoint registry:", flush=True)
    for key, path in resolved.items():
        print(f"[checkpoints] {key}: {path}", flush=True)
    methodology = _methodology_config(
        config,
        output_dir=output_dir,
        checkpoints=resolved,
    )

    donor_rows: list[dict[str, Any]] = []
    bamberger_rows: list[dict[str, Any]] = []
    health: list[dict[str, Any]] = []
    completed = 0
    total = len(config.tasks) * int(config.graphs)
    for task in config.tasks:
        prepared = prepare_task(
            methodology,
            task,
            int(config.seed),
            force_fresh_grit=False,
        )
        health.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "seed": int(config.seed),
                "test_mae": prepared.runtime.test_metric,
                "validation_mae": prepared.runtime.val_metric,
                "checkpoint": str(prepared.checkpoint),
                "checkpoint_sha256": str(prepared.checkpoint_sha),
                "parameters": prepared.runtime.checks.get("num_parameters"),
            }
        )
        graph_ids = tuple(prepared.splits.discovery)[: int(config.graphs)]
        for graph_id in graph_ids:
            path = _shard_path(output_dir, task, int(graph_id))
            shard = _load_shard(
                path,
                fingerprint=config.fingerprint,
                checkpoint_sha256=str(prepared.checkpoint_sha),
            )
            if shard is None:
                shard = _measure_graph(
                    config,
                    prepared,
                    task=task,
                    graph_id=int(graph_id),
                )
                _save_shard(path, shard)
            donor_rows.extend(shard["donor_rows"])
            bamberger_rows.extend(shard["bamberger_rows"])
            completed += 1
            if progress:
                print(
                    f"[reach] {completed}/{total} | {TASK_LABELS[task]} "
                    f"graph={int(graph_id)}",
                    flush=True,
                )

    results_dir = output_dir / "results"
    _write_csv(results_dir / "donor_carrier_mass.csv", donor_rows)
    _write_csv(results_dir / "bamberger_input_output_influence.csv", bamberger_rows)
    _write_csv(results_dir / "model_health.csv", health)
    _write_json(
        results_dir / "measurement_manifest.json",
        {
            **config.scientific_record,
            "fingerprint": config.fingerprint,
            "repository_commit": _repository_commit(),
            "checkpoints": resolved,
            "health": health,
            "completed_graph_shards": completed,
            "donor_rows": len(donor_rows),
            "bamberger_rows": len(bamberger_rows),
            "comparison_scope": {
                "semantic": (
                    "literal Bamberger pre-pooling Jacobian proxy, raw finite "
                    "hidden-state response, and task-projected Functional carriage"
                ),
                "structural": (
                    "finite Functional carriage only; no canonical Bamberger "
                    "structural quantity is claimed"
                ),
                "ground_truth": (
                    "none for learned ZINC range; architecture constrains accessibility "
                    "but does not specify the learned usage distribution"
                ),
            },
            "comparison_fairness": {
                "shared": (
                    "checkpoint, held-out graphs, original-graph SPD, final pre-pooling "
                    "carrier site, and graph-level bootstrap unit"
                ),
                "matched_decomposition": (
                    "finite hidden response and Functional carriage share sources, donor "
                    "draws, source-to-carrier distances, and event-wise normalisation"
                ),
                "literal_bamberger_difference": (
                    "Bamberger remains output-centric, uses all input nodes and sampled "
                    "output nodes/channels, sums absolute coordinatewise derivatives, and "
                    "normalises per output node"
                ),
                "interpretation": (
                    "fair comparison of operational estimands, not an estimator-equivalence "
                    "test; the matched finite pair isolates task-readout filtering from "
                    "the propagated hidden-state response"
                ),
            },
        },
    )
    return {
        "donor_rows": donor_rows,
        "bamberger_rows": bamberger_rows,
        "health": health,
    }


def _float(row: Mapping[str, Any], key: str) -> float:
    return float(row[key])


def _integer(row: Mapping[str, Any], key: str) -> int:
    return int(float(row[key]))


def graph_donor_profiles(
    rows: Sequence[Mapping[str, Any]],
    *,
    effect_floor: float,
) -> list[dict[str, Any]]:
    """Normalise each event, then aggregate donor -> source -> graph."""

    event_fields = (
        "task",
        "graph",
        "channel",
        "source",
        "donor_graph",
        "donor_node",
        "draw",
    )
    events: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        events[tuple(row[field] for field in event_fields)].append(row)

    source_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    task_max_distance: dict[str, int] = defaultdict(int)
    for key, event_rows in events.items():
        common = dict(zip(event_fields, key))
        distances = np.asarray(
            [_integer(row, "distance") for row in event_rows],
            dtype=np.int64,
        )
        task = str(common["task"])
        task_max_distance[task] = max(
            int(task_max_distance[task]),
            int(distances.max(initial=0)),
        )
        for method in DONOR_PROFILE_METHODS:
            availability = [method in row and row[method] != "" for row in event_rows]
            if not any(availability):
                continue
            if not all(availability):
                raise RuntimeError(
                    f"method {method!r} is present for only part of one donor event"
                )
            masses = np.asarray(
                [_float(row, method) for row in event_rows],
                dtype=np.float64,
            )
            total = float(masses.sum())
            if not np.isfinite(total) or total <= float(effect_floor):
                continue
            for distance in np.unique(distances):
                source_key = (
                    task,
                    _integer(common, "graph"),
                    str(common["channel"]),
                    _integer(common, "source"),
                    method,
                    int(distance),
                )
                source_values[source_key].append(
                    float(masses[distances == int(distance)].sum() / total)
                )

    graph_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    present_sources: set[tuple[Any, ...]] = {
        key[:5] for key in source_values
    }
    for task, graph, channel, source, method in present_sources:
        maximum = int(task_max_distance[task])
        for distance in range(maximum + 1):
            values = source_values.get(
                (task, graph, channel, source, method, distance),
                [],
            )
            graph_values[(task, graph, channel, method, distance)].append(
                float(np.mean(values)) if values else 0.0
            )

    output: list[dict[str, Any]] = []
    for (task, graph, channel, method, distance), values in graph_values.items():
        output.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "graph": int(graph),
                "channel": channel,
                "method": method,
                "distance": int(distance),
                "mass": float(np.mean(values)),
            }
        )
    return output


def graph_bamberger_profiles(
    rows: Sequence[Mapping[str, Any]],
    *,
    effect_floor: float,
) -> list[dict[str, Any]]:
    """Normalise each output node's influence, then average output nodes per graph."""

    output_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    task_max_distance: dict[str, int] = defaultdict(int)
    for row in rows:
        task = str(row["task"])
        task_max_distance[task] = max(
            int(task_max_distance[task]),
            _integer(row, "distance"),
        )
        output_groups[
            (task, _integer(row, "graph"), _integer(row, "output_node"))
        ].append(row)

    graph_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for (task, graph, _output_node), values in output_groups.items():
        influence = np.asarray(
            [_float(row, "influence") for row in values],
            dtype=np.float64,
        )
        distances = np.asarray(
            [_integer(row, "distance") for row in values],
            dtype=np.int64,
        )
        total = float(influence.sum())
        if not np.isfinite(total) or total <= float(effect_floor):
            continue
        for distance in range(int(task_max_distance[task]) + 1):
            graph_values[(task, graph, distance)].append(
                float(influence[distances == distance].sum() / total)
            )

    output: list[dict[str, Any]] = []
    for (task, graph, distance), values in graph_values.items():
        output.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "graph": int(graph),
                "channel": "semantic",
                "method": "bamberger",
                "distance": int(distance),
                "mass": float(np.mean(values)),
            }
        )
    return output


def _bootstrap_interval(
    values: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return np.nan, np.nan, np.nan
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(replicates), len(array)))
    draws = array[indices].mean(axis=1)
    return mean, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarise_graph_profiles(
    graph_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarise profiles and their expected distances across held-out graphs."""

    profile_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    by_graph: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in graph_rows:
        profile_groups[
            (
                str(row["task"]),
                str(row["channel"]),
                str(row["method"]),
                _integer(row, "distance"),
            )
        ].append(_float(row, "mass"))
        by_graph[
            (
                str(row["task"]),
                _integer(row, "graph"),
                str(row["channel"]),
                str(row["method"]),
            )
        ].append(row)

    profiles: list[dict[str, Any]] = []
    for key, values in profile_groups.items():
        task, channel, method, distance = key
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed)
            + int(stable_hash({"profile": key}, length=8), 16),
        )
        profiles.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "channel": channel,
                "method": method,
                "distance": int(distance),
                "mean": mean,
                "low": low,
                "high": high,
                "graphs": int(len(values)),
            }
        )

    expected_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for (task, _graph, channel, method), values in by_graph.items():
        total = sum(_float(row, "mass") for row in values)
        if total <= 0:
            continue
        expected_groups[(task, channel, method)].append(
            sum(
                _float(row, "mass") * _integer(row, "distance")
                for row in values
            )
            / total
        )
    expected: list[dict[str, Any]] = []
    for key, values in expected_groups.items():
        task, channel, method = key
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed)
            + int(stable_hash({"expected": key}, length=8), 16),
        )
        expected.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "channel": channel,
                "method": method,
                "mean": mean,
                "low": low,
                "high": high,
                "graphs": int(len(values)),
            }
        )
    return profiles, expected


def summarise_dense_profile_contrasts(
    graph_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Compute paired graph-level profile differences from dense GRIT."""

    profiles: dict[tuple[str, int, str, str], dict[int, float]] = defaultdict(dict)
    for row in graph_rows:
        profiles[
            (
                str(row["task"]),
                _integer(row, "graph"),
                str(row["channel"]),
                str(row["method"]),
            )
        ][_integer(row, "distance")] = _float(row, "mass")

    combinations = sorted(
        {
            (str(row["task"]), str(row["channel"]), str(row["method"]))
            for row in graph_rows
            if str(row["task"]) != "zinc"
        }
    )
    contrasts: list[dict[str, Any]] = []
    for task, channel, method in combinations:
        model_graphs = {
            graph
            for candidate_task, graph, candidate_channel, candidate_method in profiles
            if candidate_task == task
            and candidate_channel == channel
            and candidate_method == method
        }
        dense_graphs = {
            graph
            for candidate_task, graph, candidate_channel, candidate_method in profiles
            if candidate_task == "zinc"
            and candidate_channel == channel
            and candidate_method == method
        }
        paired_graphs = sorted(model_graphs & dense_graphs)
        if not paired_graphs:
            continue
        distances = sorted(
            {
                distance
                for graph in paired_graphs
                for profile in (
                    profiles[(task, graph, channel, method)],
                    profiles[("zinc", graph, channel, method)],
                )
                for distance in profile
            }
        )
        for distance in distances:
            values = [
                profiles[(task, graph, channel, method)].get(distance, 0.0)
                - profiles[("zinc", graph, channel, method)].get(distance, 0.0)
                for graph in paired_graphs
            ]
            mean, low, high = _bootstrap_interval(
                values,
                replicates=int(bootstrap_replicates),
                seed=int(bootstrap_seed)
                + int(
                    stable_hash(
                        {
                            "dense_profile_contrast": (
                                task,
                                channel,
                                method,
                                distance,
                            )
                        },
                        length=8,
                    ),
                    16,
                ),
            )
            contrasts.append(
                {
                    "task": task,
                    "model_label": TASK_LABELS[task],
                    "reference_task": "zinc",
                    "reference_model_label": TASK_LABELS["zinc"],
                    "channel": channel,
                    "method": method,
                    "distance": int(distance),
                    "mean": mean,
                    "low": low,
                    "high": high,
                    "paired_graphs": int(len(values)),
                }
            )
    return contrasts


def _figure_theme() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.7,
            "savefig.dpi": 300,
        }
    )


def _save_figure(fig: Any, figures_dir: Path, name: str) -> dict[str, str]:
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": str(figures_dir / f"{name}.png"),
        "pdf": str(figures_dir / f"{name}.pdf"),
    }
    fig.savefig(paths["png"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def plot_model_profiles(
    rows: Sequence[Mapping[str, Any]],
    contrast_rows: Sequence[Mapping[str, Any]],
    *,
    channel: str,
    method: str,
    title: str,
    filename: str,
    figures_dir: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    _figure_theme()
    fig, (profile_axis, contrast_axis) = plt.subplots(
        2,
        1,
        figsize=(8.8, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": (2.8, 1.25), "hspace": 0.08},
    )
    maximum_distance = 0
    for draw_order, task in enumerate(TASKS):
        values = sorted(
            (
                row
                for row in rows
                if str(row["task"]) == task
                and str(row["channel"]) == channel
                and str(row["method"]) == method
            ),
            key=lambda row: _integer(row, "distance"),
        )
        if not values:
            continue
        x = np.asarray([_integer(value, "distance") for value in values])
        y = np.asarray([_float(value, "mean") for value in values])
        low = np.asarray([_float(value, "low") for value in values])
        high = np.asarray([_float(value, "high") for value in values])
        maximum_distance = max(maximum_distance, int(x.max(initial=0)))
        profile_axis.fill_between(
            x,
            low,
            high,
            color=MODEL_COLOURS[task],
            alpha=0.08,
            linewidth=0,
            zorder=1 + draw_order,
        )
        profile_axis.plot(
            x,
            y,
            color=MODEL_COLOURS[task],
            marker=MODEL_MARKERS[task],
            linestyle=MODEL_LINESTYLES[task],
            markerfacecolor="white",
            markeredgewidth=1.15,
            markersize=5.4,
            linewidth=2.0,
            label=TASK_LABELS[task],
            zorder=5 + draw_order,
        )

    contrast_bound = 0.0
    for draw_order, task in enumerate(TASKS[:-1]):
        values = sorted(
            (
                row
                for row in contrast_rows
                if str(row["task"]) == task
                and str(row["channel"]) == channel
                and str(row["method"]) == method
            ),
            key=lambda row: _integer(row, "distance"),
        )
        if not values:
            continue
        x = np.asarray([_integer(value, "distance") for value in values])
        y = np.asarray([_float(value, "mean") for value in values])
        low = np.asarray([_float(value, "low") for value in values])
        high = np.asarray([_float(value, "high") for value in values])
        maximum_distance = max(maximum_distance, int(x.max(initial=0)))
        contrast_bound = max(
            contrast_bound,
            float(np.max(np.abs(np.concatenate((low, high))))),
        )
        contrast_axis.fill_between(
            x,
            low,
            high,
            color=MODEL_COLOURS[task],
            alpha=0.10,
            linewidth=0,
            zorder=1 + draw_order,
        )
        contrast_axis.plot(
            x,
            y,
            color=MODEL_COLOURS[task],
            marker=MODEL_MARKERS[task],
            linestyle=MODEL_LINESTYLES[task],
            markerfacecolor="white",
            markeredgewidth=1.05,
            markersize=4.7,
            linewidth=1.7,
            zorder=5 + draw_order,
        )

    profile_axis.set_ylabel("Normalised usage mass")
    profile_axis.set_ylim(bottom=0)
    contrast_axis.axhline(0, color="#666666", linewidth=0.9, zorder=0)
    contrast_axis.set_ylabel("Difference from\nDense GRIT")
    contrast_axis.set_xlabel("Shortest-path distance")
    if contrast_bound > 0:
        contrast_bound *= 1.12
        contrast_axis.set_ylim(-contrast_bound, contrast_bound)
    else:
        contrast_axis.set_ylim(-0.01, 0.01)
    contrast_axis.set_xlim(-0.15, maximum_distance + 0.15)
    contrast_axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    contrast_axis.text(
        0.995,
        0.04,
        "Paired by held-out graph",
        transform=contrast_axis.transAxes,
        ha="right",
        va="bottom",
        color="#666666",
        fontsize=8,
    )
    handles, labels = profile_axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
        handlelength=2.7,
        columnspacing=1.4,
    )
    fig.suptitle(title, fontsize=13, y=0.988)
    fig.text(
        0.5,
        0.865,
        "Mean with 95% graph-bootstrap confidence interval",
        ha="center",
        color="#666666",
        fontsize=8.5,
    )
    fig.subplots_adjust(
        left=0.115,
        right=0.985,
        bottom=0.105,
        top=0.82,
    )
    return _save_figure(fig, figures_dir, filename)


def plot_semantic_decomposition(
    rows: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
) -> dict[str, str]:
    """Plot the semantic estimand comparison in model facets."""

    import matplotlib.pyplot as plt

    _figure_theme()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.8, 7.8),
        sharex=True,
        sharey=True,
    )
    maximum_distance = 0
    for axis, task in zip(axes.reshape(-1), TASKS):
        for draw_order, method in enumerate(DECOMPOSITION_METHODS):
            values = sorted(
                (
                    row
                    for row in rows
                    if str(row["task"]) == task
                    and str(row["channel"]) == "semantic"
                    and str(row["method"]) == method
                ),
                key=lambda row: _integer(row, "distance"),
            )
            if not values:
                raise RuntimeError(
                    f"semantic decomposition is missing {method!r} for {task!r}; "
                    "rerun PHASE='all' with the current analysis version"
                )
            x = np.asarray([_integer(value, "distance") for value in values])
            y = np.asarray([_float(value, "mean") for value in values])
            low = np.asarray([_float(value, "low") for value in values])
            high = np.asarray([_float(value, "high") for value in values])
            maximum_distance = max(maximum_distance, int(x.max(initial=0)))
            axis.fill_between(
                x,
                low,
                high,
                color=METHOD_COLOURS[method],
                alpha=0.07,
                linewidth=0,
                zorder=1 + draw_order,
            )
            axis.plot(
                x,
                y,
                color=METHOD_COLOURS[method],
                marker=METHOD_MARKERS[method],
                linestyle=METHOD_LINESTYLES[method],
                markerfacecolor="white",
                markeredgewidth=1.0,
                markersize=4.2,
                linewidth=1.8,
                label=METHOD_LABELS[method],
                zorder=5 + draw_order,
            )
        axis.set_title(TASK_LABELS[task])
        axis.set_ylim(bottom=0)
        axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for axis in axes[:, 0]:
        axis.set_ylabel("Normalised usage mass")
    for axis in axes[-1, :]:
        axis.set_xlabel("Shortest-path distance")
    for axis in axes.reshape(-1):
        axis.set_xlim(-0.15, maximum_distance + 0.15)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.938),
        ncol=3,
        frameon=False,
        handlelength=3.0,
        columnspacing=2.0,
    )
    fig.suptitle("Semantic distance profiles across influence estimands", fontsize=13, y=0.992)
    fig.text(
        0.5,
        0.865,
        (
            "Finite-response and Functional profiles share donor events; "
            "Bamberger retains its output-centric sampling"
        ),
        ha="center",
        color="#666666",
        fontsize=8.5,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.09,
        top=0.81,
        hspace=0.22,
        wspace=0.10,
    )
    return _save_figure(fig, figures_dir, "zinc_semantic_estimand_decomposition")


def plot_expected_distance(
    rows: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    _figure_theme()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), sharey=True)
    positions = np.arange(len(TASKS), dtype=np.float64)
    for axis, channel in zip(axes, CHANNELS):
        methods = (
            ("bamberger", *DONOR_METHODS)
            if channel == "semantic"
            else DONOR_METHODS
        )
        offsets = np.linspace(-0.18, 0.18, len(methods))
        for offset, method in zip(offsets, methods):
            values_by_task = {
                str(row["task"]): row
                for row in rows
                if str(row["channel"]) == channel
                and str(row["method"]) == method
            }
            x_values: list[float] = []
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for index, task in enumerate(TASKS):
                if task not in values_by_task:
                    continue
                value = values_by_task[task]
                x_values.append(float(positions[index] + offset))
                means.append(_float(value, "mean"))
                lows.append(_float(value, "low"))
                highs.append(_float(value, "high"))
            mean_array = np.asarray(means)
            axis.errorbar(
                x_values,
                mean_array,
                yerr=np.vstack(
                    (
                        mean_array - np.asarray(lows),
                        np.asarray(highs) - mean_array,
                    )
                ),
                fmt=METHOD_MARKERS[method],
                markersize=5,
                capsize=2.5,
                linewidth=1.3,
                color=METHOD_COLOURS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_title(f"{channel.capitalize()} perturbations")
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [TASK_LABELS[task] for task in TASKS],
            rotation=22,
            ha="right",
        )
        axis.set_ylabel("Expected shortest-path distance")
        axis.set_ylim(bottom=0)
    axes[1].text(
        0.03,
        0.95,
        "Bamberger structural analogue not defined",
        transform=axes[1].transAxes,
        va="top",
        color="#666666",
        fontsize=8,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Expected distance of model usage on ZINC", fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.27, top=0.72, wspace=0.22)
    return _save_figure(fig, figures_dir, "zinc_expected_reach")


def figures(
    config: ZincReachConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Build every table and paper figure from cached CSV files only."""

    results_dir = output_dir / "results"
    donor_path = results_dir / "donor_carrier_mass.csv"
    bamberger_path = results_dir / "bamberger_input_output_influence.csv"
    if not donor_path.is_file() or not bamberger_path.is_file():
        raise FileNotFoundError(
            "cached measurement CSVs are missing; run PHASE='measure' or 'all' first"
        )
    donor_graph = graph_donor_profiles(
        _read_csv(donor_path),
        effect_floor=float(config.effect_floor),
    )
    bamberger_graph = graph_bamberger_profiles(
        _read_csv(bamberger_path),
        effect_floor=float(config.effect_floor),
    )
    graph_rows = [*donor_graph, *bamberger_graph]
    profile_rows, expected_rows = summarise_graph_profiles(
        graph_rows,
        bootstrap_replicates=int(config.bootstrap_replicates),
        bootstrap_seed=int(config.analysis_seed) + 100,
    )
    contrast_rows = summarise_dense_profile_contrasts(
        graph_rows,
        bootstrap_replicates=int(config.bootstrap_replicates),
        bootstrap_seed=int(config.analysis_seed) + 200,
    )
    _write_csv(results_dir / "graph_distance_profiles.csv", graph_rows)
    _write_csv(results_dir / "distance_profile_summary.csv", profile_rows)
    _write_csv(results_dir / "dense_profile_contrasts.csv", contrast_rows)
    _write_csv(results_dir / "expected_distance_summary.csv", expected_rows)

    figures_dir = output_dir / "figures"
    paths = {
        "semantic_decomposition": plot_semantic_decomposition(
            profile_rows,
            figures_dir=figures_dir,
        ),
        "semantic_functional": plot_model_profiles(
            profile_rows,
            contrast_rows,
            channel="semantic",
            method="functional_carriage",
            title="Semantic usage by Functional carriage",
            filename="zinc_semantic_functional_profiles",
            figures_dir=figures_dir,
        ),
        "semantic_bamberger": plot_model_profiles(
            profile_rows,
            contrast_rows,
            channel="semantic",
            method="bamberger",
            title="Semantic usage by Bamberger Jacobian range",
            filename="zinc_semantic_bamberger_profiles",
            figures_dir=figures_dir,
        ),
        "structural_functional": plot_model_profiles(
            profile_rows,
            contrast_rows,
            channel="structural",
            method="functional_carriage",
            title="Structural usage by Functional carriage",
            filename="zinc_structural_functional_profiles",
            figures_dir=figures_dir,
        ),
        "expected_distance": plot_expected_distance(
            expected_rows,
            figures_dir=figures_dir,
        ),
    }
    _write_json(
        results_dir / "figure_manifest.json",
        {
            "analysis_version": ANALYSIS_VERSION,
            "fingerprint": config.fingerprint,
            "figures": paths,
            "uncertainty": (
                "95% percentile bootstrap over held-out graphs; one trained "
                "checkpoint per architecture, so intervals do not include training-seed "
                "variance. Difference panels use paired graph-level bootstraps against "
                "Dense GRIT."
            ),
        },
    )
    return {
        "figures": paths,
        "profile_rows": profile_rows,
        "contrast_rows": contrast_rows,
        "expected_rows": expected_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "measure", "figures"), default="all")
    parser.add_argument(
        "--output-dir",
        default=(
            "/content/drive/MyDrive/graph_specialisation_metrics/"
            "zinc_bamberger_functional_reach_v5"
        ),
    )
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graphs", type=int, default=16)
    parser.add_argument("--sources-per-graph", type=int, default=6)
    parser.add_argument("--donors-per-source", type=int, default=4)
    parser.add_argument("--semantic-donor-graphs", type=int, default=256)
    parser.add_argument("--bamberger-output-nodes", type=int, default=6)
    parser.add_argument("--bamberger-output-channels", type=int, default=8)
    parser.add_argument("--effect-floor", type=float, default=1.0e-12)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--analysis-seed", type=int, default=91_021)
    parser.add_argument("--accelerator", default="cuda:0")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--skip-dependency-install", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    config = ZincReachConfig(
        tasks=tuple(value.strip() for value in args.tasks.split(",") if value.strip()),
        seed=int(args.seed),
        graphs=int(args.graphs),
        sources_per_graph=int(args.sources_per_graph),
        donors_per_source=int(args.donors_per_source),
        semantic_donor_graphs=int(args.semantic_donor_graphs),
        bamberger_output_nodes=int(args.bamberger_output_nodes),
        bamberger_output_channels=int(args.bamberger_output_channels),
        effect_floor=float(args.effect_floor),
        bootstrap_replicates=int(args.bootstrap_replicates),
        analysis_seed=int(args.analysis_seed),
        accelerator=str(args.accelerator),
        num_threads=int(args.num_threads),
    )
    config.validate()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "analysis_config.json",
        {
            **asdict(config),
            "fingerprint": config.fingerprint,
            "repository_commit": _repository_commit(),
        },
    )
    result: dict[str, Any] = {"config": config, "output_dir": str(output_dir)}
    if args.phase in {"all", "measure"}:
        result["measurement"] = measure(
            config,
            output_dir=output_dir,
            install_dependencies=not bool(args.skip_dependency_install),
            progress=not bool(args.quiet),
        )
    if args.phase in {"all", "figures"}:
        result.update(figures(config, output_dir=output_dir))
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
