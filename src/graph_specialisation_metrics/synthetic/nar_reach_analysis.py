"""Finite-versus-local reach analysis for trained fixed-N NAR GRIT models.

The primary estimands are evaluated at the same native routed-message sites:

* ``Functional carriage`` uses the finite clean-minus-donor change at a carrier.
* ``Local Jacobian`` linearises that same donor direction at the clean input.

Both carrier changes are projected through the clean output Jacobian before taking
the output-vector norm.  The local estimator is a source-conditioned, internal-site
adaptation of Bamberger et al.'s differential premise; it is not presented as a
literal reimplementation of their graph-level task-range statistic.

NAR's classifier reads only the central final-layer state.  Consequently, Beneficial
carriage is reported separately as a signed task-benefit diagnostic at that readout,
not as a distributed reach profile.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from ..methodology.carriage import beneficial_carriage
from ..methodology.distance import shortest_path_distances
from ..methodology.events import build_channel_events
from ..methodology.protocol import (
    BootstrapPolicy,
    ExecutionPolicy,
    MethodologyConfig,
    NumericalPolicy,
    RunSizes,
    stable_hash,
)
from ..methodology.runner import prepare_task
from .nar_canonical_analysis import (
    MODEL_LABELS,
    _to_nar_batch,
    register_nar_tasks,
    task_name,
)
from .nar_grit_fixed import MODEL_RADII, setup_official_grit


ANALYSIS_VERSION = "nar-finite-local-reach-v1"
CHANNELS = ("semantic", "structural")
METHODS = ("local_jacobian", "functional_carriage")
METHOD_LABELS = {
    "local_jacobian": "Local donor-direction Jacobian",
    "functional_carriage": "Functional carriage",
}
METHOD_COLOURS = {
    "local_jacobian": "#0072B2",
    "functional_carriage": "#D55E00",
}
METHOD_MARKERS = {"local_jacobian": "o", "functional_carriage": "s"}


@dataclass(frozen=True)
class ReachConfig:
    """Scientific and execution controls for one reach-analysis extension."""

    models: tuple[str, ...] = ("1hop", "2hop", "dense")
    ns: tuple[int, ...] = (8, 16, 64)
    seeds: tuple[int, ...] = (0, 1, 2)
    width: int = 128
    graphs: int = 16
    donors_per_source: int = 4
    semantic_donor_graphs: int = 256
    analysis_seed: int = 73_021
    accelerator: str = "cuda:0"
    num_threads: int = 4
    effect_floor: float = 1.0e-10
    integrated_atol: float = 1.0e-5
    integrated_rtol: float = 1.0e-4
    integrated_max_intervals: int = 64

    def validate(self) -> None:
        if not self.models or any(model not in MODEL_RADII for model in self.models):
            raise ValueError(f"models must be drawn from {tuple(MODEL_RADII)}")
        if not self.ns or any(int(value) <= 1 for value in self.ns):
            raise ValueError("N values must be greater than one")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        for name in ("width", "graphs", "donors_per_source", "semantic_donor_graphs"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if float(self.effect_floor) <= 0:
            raise ValueError("effect_floor must be positive")

    @property
    def scientific_record(self) -> dict[str, Any]:
        return {
            "analysis_version": ANALYSIS_VERSION,
            "models": list(self.models),
            "ns": list(self.ns),
            "seeds": list(self.seeds),
            "width": int(self.width),
            "graphs": int(self.graphs),
            "donors_per_source": int(self.donors_per_source),
            "semantic_donor_graphs": int(self.semantic_donor_graphs),
            "analysis_seed": int(self.analysis_seed),
            "effect_floor": float(self.effect_floor),
            "integrated_atol": float(self.integrated_atol),
            "integrated_rtol": float(self.integrated_rtol),
            "integrated_max_intervals": int(self.integrated_max_intervals),
            "channels": list(CHANNELS),
            "carrier_site": "native routed-wV output, summed over heads at each node",
            "functional_estimand": (
                "norm over output logits of clean-Jacobian-projected finite carrier change"
            ),
            "local_estimand": (
                "norm over output logits of clean-Jacobian-projected carrier JVP "
                "along the exact clean-minus-donor encoded-input direction; exact "
                "autograd JVP with an epsilon-halving-audited centered fallback"
            ),
            "bamberger_estimand": (
                "original normalized node-level range at the central output: encoded "
                "semantic input-node influence is the entrywise absolute clean Jacobian "
                "summed over input and output channels"
            ),
            "beneficial_estimand": (
                "positive-is-beneficial integrated cross-entropy allocation at the "
                "central final-state readout"
            ),
        }

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.scientific_record)


@dataclass(frozen=True)
class DirectionalTransport:
    values: tuple[Any, ...]
    method: str
    relative_error: float | None


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
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def architectural_reach(model: str, layer: int, *, graph_diameter: int = 3) -> int:
    """Maximum source-to-carrier distance allowed after a one-indexed layer."""

    radius = MODEL_RADII[str(model)]
    if radius is None:
        return int(graph_diameter)
    return min(int(graph_diameter), int(radius) * int(layer))


def nar_known_references() -> dict[str, Any]:
    """Return the genuine NAR references without inventing a learned-route oracle."""

    return {
        "learned_carrier_ground_truth": None,
        "semantic_sources": {
            "query": {
                "distance_to_readout": 2,
                "task_role": "identifies the requested key",
            },
            "requested_record": {
                "distance_to_readout": 1,
                "task_role": "contains the target value",
            },
        },
        "architecture": {
            "1hop": {
                "per_layer_radius": 1,
                "two_layer_maximum": 2,
                "required_query_path": "query -> intermediate -> central",
            },
            "2hop": {"per_layer_radius": 2, "two_layer_maximum": 3},
            "dense": {"per_layer_radius": "all nodes", "two_layer_maximum": 3},
        },
        "structural_caveat": (
            "At fixed N, topology and RRWP are identical across training examples. "
            "RRWP donor swaps therefore probe learned positional/role routing under a "
            "diagnostic counterfactual; they do not have a task-level structural oracle."
        ),
    }


def _source_role(data: Any, source: int) -> str:
    if int(source) == int(data.query_idx):
        return "query"
    if int(source) == int(data.target_idx):
        return "requested_record"
    return "other"


def _event_seed(config: ReachConfig, *parts: Any) -> int:
    digest = stable_hash(
        {"analysis_seed": int(config.analysis_seed), "parts": list(parts)},
        length=16,
    )
    return int(digest, 16) % (2**32)


def _methodology_config(
    config: ReachConfig,
    *,
    output_dir: Path,
    tasks: Sequence[str],
) -> MethodologyConfig:
    # Canonical split construction requires non-empty held-out partitions even though this
    # extension uses only the discovery IDs.
    sizes = RunSizes(
        discovery_graphs=int(config.graphs),
        causal_graphs=1,
        clean_ablation_graphs=1,
        semantic_donor_graphs=int(config.semantic_donor_graphs),
        sources_per_graph=2,
        donors_per_source=int(config.donors_per_source),
    )
    numerical = NumericalPolicy(
        effect_floor=float(config.effect_floor),
        integrated_atol=float(config.integrated_atol),
        integrated_rtol=float(config.integrated_rtol),
        integrated_max_intervals=int(config.integrated_max_intervals),
    )
    return MethodologyConfig(
        output_dir=str(output_dir / "prepared"),
        tasks=tuple(tasks),
        train_seeds=tuple(config.seeds),
        sizes=sizes,
        numerical=numerical,
        bootstrap=BootstrapPolicy(),
        execution=ExecutionPolicy(graphs_per_batch=1),
        analysis_seed=int(config.analysis_seed),
        accelerator=str(config.accelerator),
        num_threads=int(config.num_threads),
        skip_install=True,
        resume=True,
        strict_audits=False,
        compute_beneficial_carriage=True,
    )


def _encoded_transport_jvp(
    model: Any,
    clean_graphs: Sequence[Any],
    event_graphs: Sequence[Any],
) -> DirectionalTransport:
    """JVP of routed transport along exact encoded clean-minus-event directions."""

    import copy

    import torch

    if len(clean_graphs) != len(event_graphs) or not clean_graphs:
        raise ValueError("clean/event graph batches must be non-empty and aligned")
    device = next(model.parameters()).device
    clean_batch = _to_nar_batch(clean_graphs).to(device)
    event_batch = _to_nar_batch(event_graphs).to(device)
    clean_data = model._pyg_batch(clean_batch)
    event_data = model._pyg_batch(event_batch)
    if not torch.equal(clean_data.edge_index, event_data.edge_index):
        raise RuntimeError("NAR donor event changed architectural attention support")

    clean_x = clean_data.x.detach()
    clean_edge = clean_data.edge_attr.detach()
    direction_x = clean_x - event_data.x.detach()
    direction_edge = clean_edge - event_data.edge_attr.detach()
    routed: list[Any] = [None] * int(model.L)
    layer_by_module = {
        id(module): layer for layer, module in enumerate(model.attention_layers)
    }

    def hook(module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        routed[layer_by_module[id(module)]] = output[0]

    handles = [
        module.register_forward_hook(hook) for module in model.attention_layers
    ]

    def forward(encoded_x: Any, encoded_edge: Any) -> tuple[Any, ...]:
        data = copy.copy(clean_data)
        data.x = encoded_x
        data.edge_attr = encoded_edge
        for layer in model.layers:
            data = layer(data)
        if any(value is None for value in routed):
            raise RuntimeError("NAR routed-message hook did not fire during JVP")
        return tuple(routed)

    method = "exact_autograd_jvp"
    relative_error: float | None = None
    try:
        try:
            _, tangent = torch.autograd.functional.jvp(
                forward,
                (clean_x, clean_edge),
                (direction_x, direction_edge),
                create_graph=False,
                strict=False,
            )
        except (NotImplementedError, RuntimeError) as error:
            # Some torch-scatter/PyG builds do not expose the second derivatives used by
            # torch.autograd.functional.jvp. A centered local difference is an auditable
            # numerical Jacobian-vector product, not the finite donor endpoint response.
            method = f"centred_difference_fallback:{type(error).__name__}"

            def centred(epsilon: float) -> tuple[Any, ...]:
                with torch.no_grad():
                    plus = forward(
                        clean_x + epsilon * direction_x,
                        clean_edge + epsilon * direction_edge,
                    )
                    minus = forward(
                        clean_x - epsilon * direction_x,
                        clean_edge - epsilon * direction_edge,
                    )
                return tuple(
                    (positive - negative) / (2.0 * epsilon)
                    for positive, negative in zip(plus, minus)
                )

            coarse = centred(1.0e-3)
            tangent = centred(5.0e-4)
            numerator = torch.sqrt(
                sum((left - right).square().sum() for left, right in zip(coarse, tangent))
            )
            denominator = torch.sqrt(sum(value.square().sum() for value in tangent))
            relative_error = float(
                (numerator / denominator.clamp_min(1.0e-12)).detach().cpu()
            )
    finally:
        for handle in handles:
            handle.remove()

    repetitions = len(clean_graphs)
    nodes = int(clean_graphs[0].num_nodes)
    return DirectionalTransport(
        values=tuple(
            value.reshape(repetitions, nodes, int(model.H), int(model.dh)).detach()
            for value in tangent
        ),
        method=method,
        relative_error=relative_error,
    )


def _clean_reach_jacobians(model: Any, base: Any, device: Any) -> Any:
    """Capture carrier and encoded-input Jacobians in one clean forward/backward pass."""

    import torch

    batch = _to_nar_batch([base]).to(device)
    routed: list[Any] = [None] * int(model.L)
    encoded: dict[str, Any] = {}
    layer_by_module = {
        id(module): layer for layer, module in enumerate(model.attention_layers)
    }

    def attention_hook(module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        routed[layer_by_module[id(module)]] = output[0]

    def input_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        encoded["node"] = inputs[0].x
        encoded["edge"] = inputs[0].edge_attr

    handles = [
        module.register_forward_hook(attention_hook)
        for module in model.attention_layers
    ]
    handles.append(model.layers[0].register_forward_pre_hook(input_hook))
    try:
        with torch.enable_grad():
            prediction = model(batch)
        if any(value is None for value in routed) or set(encoded) != {"node", "edge"}:
            raise RuntimeError("clean NAR reach hooks did not fire")
        targets = tuple(routed) + (encoded["node"], encoded["edge"])
        gradients = []
        outputs = int(prediction.shape[-1])
        for output in range(outputs):
            gradients.append(
                torch.autograd.grad(
                    prediction[:, output].sum(),
                    targets,
                    retain_graph=output + 1 < outputs,
                    allow_unused=False,
                )
            )
    finally:
        for handle in handles:
            handle.remove()

    nodes = int(base.num_nodes)
    clean_transport = tuple(
        value.reshape(nodes, int(model.H), int(model.dh)).detach()
        for value in routed
    )
    transport_gradient = torch.stack(
        [
            torch.stack(
                [
                    row[layer]
                    .reshape(nodes, int(model.H), int(model.dh))
                    .detach()
                    for row in gradients
                ],
                dim=0,
            )
            for layer in range(int(model.L))
        ],
        dim=1,
    )
    input_node_gradient = torch.stack(
        [
            row[-2].reshape(nodes, int(model.width)).detach()
            for row in gradients
        ],
        dim=0,
    )
    input_edge_gradient = torch.stack(
        [row[-1].detach() for row in gradients],
        dim=0,
    )
    if not bool(
        torch.isfinite(transport_gradient).all()
        and torch.isfinite(input_node_gradient).all()
        and torch.isfinite(input_edge_gradient).all()
    ):
        raise RuntimeError("non-finite clean reach Jacobian")
    return SimpleNamespace(
        capture=SimpleNamespace(
            transport=clean_transport,
            prediction=prediction.detach(),
            target=batch.y.detach(),
        ),
        transport=transport_gradient,
        input_node=input_node_gradient,
        input_edge=input_edge_gradient,
    )


def _bamberger_rows(
    *,
    config: ReachConfig,
    model_name: str,
    records: int,
    seed: int,
    graph_id: int,
    base: Any,
    clean_jacobians: Any,
) -> list[dict[str, Any]]:
    """Literal node-level Bamberger input influence for the central NAR output."""

    central = int(base.central_idx)
    pristine = shortest_path_distances(base.edge_index, int(base.num_nodes))
    influence = (
        clean_jacobians.input_node.abs()
        .sum(dim=(0, 2))
        .detach()
        .cpu()
        .numpy()
    )
    return [
        {
            "analysis_version": ANALYSIS_VERSION,
            "fingerprint": config.fingerprint,
            "model": str(model_name),
            "model_label": MODEL_LABELS[str(model_name)],
            "N": int(records),
            "seed": int(seed),
            "graph": int(graph_id),
            "input_node": int(node),
            "input_role": _source_role(base, int(node)),
            "distance_to_central_output": int(pristine[central, node]),
            "influence": float(influence[node]),
            "input_space": (
                "continuous encoded node state with RRWP node/edge encodings held fixed"
            ),
        }
        for node in range(int(base.num_nodes))
    ]


def _project_carrier_change(change: Any, clean_gradient: Any) -> Any:
    """Project ``[E,N,H,D]`` changes and return non-negative ``[E,N]`` mass."""

    import torch

    if change.ndim != 4 or clean_gradient.ndim != 4:
        raise ValueError("change must be [E,N,H,D] and gradient [T,N,H,D]")
    if tuple(change.shape[1:]) != tuple(clean_gradient.shape[1:]):
        raise ValueError("carrier change and clean output gradient geometry differ")
    # Heads are coordinates of one node-level routed message. Summing their signed
    # output contributions before the norm avoids treating heads as separate carriers.
    contribution = torch.einsum("enhd,tnhd->enth", change, clean_gradient)
    node_contribution = contribution.sum(dim=-1)
    return torch.linalg.vector_norm(node_contribution, dim=-1)


def _event_rows(
    *,
    config: ReachConfig,
    model_name: str,
    records: int,
    seed: int,
    graph_id: int,
    channel: str,
    base: Any,
    variants: Sequence[Any],
    events: Sequence[Any],
    capture: Any,
    clean_jacobians: Any,
    local_tangent: Sequence[Any],
) -> list[dict[str, Any]]:
    import torch

    pristine = shortest_path_distances(base.edge_index, int(base.num_nodes))
    rows: list[dict[str, Any]] = []
    clean_transport = clean_jacobians.capture.transport
    for layer, (clean_layer, event_layer, tangent_layer) in enumerate(
        zip(clean_transport, capture.transport, local_tangent),
        start=1,
    ):
        finite_change = clean_layer.unsqueeze(0) - event_layer[1:]
        gradient = clean_jacobians.transport[:, layer - 1]
        finite_mass = _project_carrier_change(finite_change, gradient)
        local_mass = _project_carrier_change(tangent_layer, gradient)
        if not bool(torch.isfinite(finite_mass).all() and torch.isfinite(local_mass).all()):
            raise RuntimeError("non-finite reach mass")
        for event_index, event in enumerate(events):
            source = int(event.source)
            distances = pristine[source]
            for carrier in range(int(base.num_nodes)):
                rows.append(
                    {
                        "analysis_version": ANALYSIS_VERSION,
                        "fingerprint": config.fingerprint,
                        "model": str(model_name),
                        "model_label": MODEL_LABELS[str(model_name)],
                        "N": int(records),
                        "seed": int(seed),
                        "graph": int(graph_id),
                        "channel": str(channel),
                        "source": source,
                        "source_role": _source_role(base, source),
                        "donor_graph": int(event.donor_graph_id),
                        "donor_node": int(event.donor_node),
                        "draw": int(event.draw),
                        "dose": float(event.dose),
                        "layer": int(layer),
                        "carrier": int(carrier),
                        "distance": int(distances[carrier]),
                        "local_jacobian": float(
                            local_mass[event_index, carrier].detach().cpu()
                        ),
                        "functional_carriage": float(
                            finite_mass[event_index, carrier].detach().cpu()
                        ),
                    }
                )
    return rows


def _beneficial_rows(
    *,
    config: ReachConfig,
    prepared: Any,
    model_name: str,
    records: int,
    seed: int,
    graph_id: int,
    channel: str,
    base: Any,
    events: Sequence[Any],
    capture: Any,
) -> list[dict[str, Any]]:
    import torch

    sources = tuple(prepared.backend.eligible_sources(base))
    donors = int(config.donors_per_source)
    expected = len(sources) * donors
    if len(events) != expected:
        raise RuntimeError(f"expected {expected} donor events, found {len(events)}")
    nodes = int(base.num_nodes)
    width = int(prepared.backend.geometry["hidden_width"])
    h_clean = capture.final_state[0]
    h_event = capture.final_state[1:].reshape(len(sources), donors, nodes, width)
    target = capture.target[0:1]
    weights = prepared.backend.carriage_weights(base, h_clean)
    result = beneficial_carriage(
        h_clean,
        h_event,
        prepared.backend.loss_from_pooled(target),
        carrier_weights=weights,
        atol=float(config.integrated_atol),
        rtol=float(config.integrated_rtol),
        max_intervals=int(config.integrated_max_intervals),
    )
    central = int(base.central_idx)
    pristine = shortest_path_distances(base.edge_index, nodes)
    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        for donor_index in range(donors):
            event_index = source_index * donors + donor_index
            event = events[event_index]
            if int(event.source) != int(source):
                raise RuntimeError("beneficial event order no longer matches source-major layout")
            rows.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "fingerprint": config.fingerprint,
                    "model": str(model_name),
                    "model_label": MODEL_LABELS[str(model_name)],
                    "N": int(records),
                    "seed": int(seed),
                    "graph": int(graph_id),
                    "channel": str(channel),
                    "source": int(source),
                    "source_role": _source_role(base, int(source)),
                    "source_to_readout_distance": int(pristine[int(source), central]),
                    "donor_graph": int(event.donor_graph_id),
                    "donor_node": int(event.donor_node),
                    "draw": int(event.draw),
                    "dose": float(event.dose),
                    "carrier": central,
                    "beneficial_carriage": float(
                        result.event_field[source_index, donor_index, central]
                        .detach()
                        .cpu()
                    ),
                    "event_loss_increase": float(
                        result.event_loss_increase[source_index, donor_index]
                        .detach()
                        .cpu()
                    ),
                    "completeness_residual": float(
                        result.completeness_residual[source_index, donor_index]
                        .detach()
                        .cpu()
                    ),
                    "quadrature_error": float(
                        result.quadrature_error[source_index, donor_index]
                        .detach()
                        .cpu()
                    ),
                    "quadrature_intervals": int(
                        result.intervals[source_index, donor_index].detach().cpu()
                    ),
                    "quadrature_converged": bool(
                        result.converged[source_index, donor_index].detach().cpu()
                    ),
                }
            )
    return rows


def _measure_graph(
    config: ReachConfig,
    prepared: Any,
    *,
    model_name: str,
    records: int,
    seed: int,
    graph_id: int,
) -> dict[str, Any]:
    import torch

    base = prepared.runtime.eval_ds[int(graph_id)]
    sources = tuple(prepared.backend.eligible_sources(base))
    if {_source_role(base, source) for source in sources} != {
        "query",
        "requested_record",
    }:
        raise RuntimeError("NAR reach requires query and requested-record sources")

    clean_jacobians = _clean_reach_jacobians(
        prepared.runtime.model,
        base,
        prepared.runtime.device,
    )
    bamberger_rows = _bamberger_rows(
        config=config,
        model_name=model_name,
        records=records,
        seed=seed,
        graph_id=graph_id,
        base=base,
        clean_jacobians=clean_jacobians,
    )
    reach_rows: list[dict[str, Any]] = []
    beneficial_rows: list[dict[str, Any]] = []
    derivative_diagnostics: list[dict[str, Any]] = []
    for channel in CHANNELS:
        variants: list[Any] = []
        events: list[Any] = []
        for source in sources:
            event_variants, event_records = build_channel_events(
                base,
                graph_id=int(graph_id),
                source=int(source),
                channel=str(channel),
                stage="reach",
                donors=int(config.donors_per_source),
                rng=np.random.default_rng(
                    _event_seed(
                        config,
                        model_name,
                        int(records),
                        int(seed),
                        int(graph_id),
                        channel,
                        int(source),
                    )
                ),
                task=prepared.task,
                semantic_pool=prepared.donor_pool,
                duplicate_tolerance=1.0e-7,
            )
            variants.extend(event_variants)
            events.extend(event_records)
        clean_replicas = [base for _ in variants]
        capture = prepared.backend.capture_groups([[base, *variants]])[0]
        local_result = _encoded_transport_jvp(
            prepared.runtime.model,
            clean_replicas,
            variants,
        )
        if (
            local_result.relative_error is not None
            and local_result.relative_error > 5.0e-2
        ):
            print(
                "[reach:warning] local centered-difference JVP changed by "
                f"{local_result.relative_error:.3g} under epsilon halving "
                f"({model_name}, N={records}, seed={seed}, graph={graph_id}, "
                f"{channel})",
                flush=True,
            )
        derivative_diagnostics.append(
            {
                "model": str(model_name),
                "N": int(records),
                "seed": int(seed),
                "graph": int(graph_id),
                "channel": str(channel),
                "method": local_result.method,
                "relative_error": local_result.relative_error,
            }
        )
        reach_rows.extend(
            _event_rows(
                config=config,
                model_name=model_name,
                records=records,
                seed=seed,
                graph_id=graph_id,
                channel=channel,
                base=base,
                variants=variants,
                events=events,
                capture=capture,
                clean_jacobians=clean_jacobians,
                local_tangent=local_result.values,
            )
        )
        beneficial_rows.extend(
            _beneficial_rows(
                config=config,
                prepared=prepared,
                model_name=model_name,
                records=records,
                seed=seed,
                graph_id=graph_id,
                channel=channel,
                base=base,
                events=events,
                capture=capture,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "analysis_version": ANALYSIS_VERSION,
        "fingerprint": config.fingerprint,
        "model": str(model_name),
        "N": int(records),
        "seed": int(seed),
        "graph": int(graph_id),
        "reach": reach_rows,
        "beneficial": beneficial_rows,
        "bamberger": bamberger_rows,
        "derivative_diagnostics": derivative_diagnostics,
    }


def _shard_path(
    output_dir: Path,
    *,
    model_name: str,
    records: int,
    seed: int,
    graph_id: int,
) -> Path:
    return (
        output_dir
        / "cache"
        / f"{model_name}__N{int(records)}__seed_{int(seed)}"
        / f"graph_{int(graph_id):06d}.pt"
    )


def _load_shard(path: Path, fingerprint: str) -> dict[str, Any] | None:
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
        or not {"reach", "beneficial", "bamberger"}.issubset(payload)
        or "derivative_diagnostics" not in payload
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
    config: ReachConfig,
    *,
    output_dir: Path,
    training_run_dir: Path,
    grit_dir: Path,
    install_grit: bool,
    progress: bool,
) -> dict[str, Any]:
    """Measure every requested cell, resuming from graph-level Drive shards."""

    import torch

    config.validate()
    torch.set_num_threads(int(config.num_threads))
    setup_official_grit(Path(grit_dir), install=bool(install_grit))
    tasks = register_nar_tasks(
        models=config.models,
        analysis_ns=config.ns,
        width=int(config.width),
        training_run_dir=training_run_dir,
    )
    methodology = _methodology_config(config, output_dir=output_dir, tasks=tasks)
    reach_rows: list[dict[str, Any]] = []
    beneficial_rows: list[dict[str, Any]] = []
    bamberger_rows: list[dict[str, Any]] = []
    derivative_diagnostics: list[dict[str, Any]] = []
    completed = 0
    total = len(config.models) * len(config.ns) * len(config.seeds) * int(config.graphs)
    checkpoints: list[dict[str, Any]] = []
    for records in config.ns:
        for model_name in config.models:
            task = task_name(model_name, int(records))
            for seed in config.seeds:
                prepared = prepare_task(methodology, task, int(seed))
                checkpoints.append(
                    {
                        "task": task,
                        "model": model_name,
                        "N": int(records),
                        "seed": int(seed),
                        "path": str(prepared.checkpoint),
                        "sha256": str(prepared.checkpoint_sha),
                    }
                )
                graph_ids = tuple(prepared.splits.discovery)[: int(config.graphs)]
                for graph_id in graph_ids:
                    shard_path = _shard_path(
                        output_dir,
                        model_name=model_name,
                        records=records,
                        seed=seed,
                        graph_id=graph_id,
                    )
                    shard = _load_shard(shard_path, config.fingerprint)
                    if shard is None:
                        shard = _measure_graph(
                            config,
                            prepared,
                            model_name=model_name,
                            records=int(records),
                            seed=int(seed),
                            graph_id=int(graph_id),
                        )
                        _save_shard(shard_path, shard)
                    reach_rows.extend(shard["reach"])
                    beneficial_rows.extend(shard["beneficial"])
                    bamberger_rows.extend(shard["bamberger"])
                    derivative_diagnostics.extend(shard["derivative_diagnostics"])
                    completed += 1
                    if progress:
                        print(
                            f"[reach] {completed}/{total} | {model_name} N={records} "
                            f"seed={seed} graph={graph_id}",
                            flush=True,
                        )

    results_dir = output_dir / "results"
    _write_csv(results_dir / "reach_carriers.csv", reach_rows)
    _write_csv(results_dir / "beneficial_carriage.csv", beneficial_rows)
    _write_csv(results_dir / "bamberger_semantic_input.csv", bamberger_rows)
    _write_csv(
        results_dir / "local_derivative_diagnostics.csv",
        derivative_diagnostics,
    )
    _write_json(
        results_dir / "measurement_manifest.json",
        {
            **config.scientific_record,
            "fingerprint": config.fingerprint,
            "repository_commit": _repository_commit(),
            "training_run_dir": str(training_run_dir),
            "checkpoints": checkpoints,
            "reach_rows": len(reach_rows),
            "beneficial_rows": len(beneficial_rows),
            "bamberger_rows": len(bamberger_rows),
            "local_derivative_diagnostics": derivative_diagnostics,
            "completed_graph_shards": completed,
            "known_references": nar_known_references(),
            "comparison_scope": (
                "The original Bamberger node-level semantic input-output range is "
                "reported separately. The donor-direction Jacobian is the matched local "
                "comparator for source-conditioned semantic and structural carrier reach."
            ),
        },
    )
    return {
        "reach_rows": reach_rows,
        "beneficial_rows": beneficial_rows,
        "bamberger_rows": bamberger_rows,
        "derivative_diagnostics": derivative_diagnostics,
        "checkpoints": checkpoints,
    }


def _float(row: Mapping[str, Any], key: str) -> float:
    return float(row[key])


def _integer(row: Mapping[str, Any], key: str) -> int:
    return int(float(row[key]))


def summarise_profiles(
    rows: Sequence[Mapping[str, Any]],
    *,
    effect_floor: float,
    layer: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Event-normalise, then aggregate event -> graph -> seed for profiles/ranges."""

    event_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    event_fields = (
        "model",
        "N",
        "seed",
        "graph",
        "channel",
        "source",
        "donor_graph",
        "donor_node",
        "draw",
        "layer",
    )
    for row in rows:
        if layer is not None and _integer(row, "layer") != int(layer):
            continue
        key = tuple(row[field] for field in event_fields)
        event_groups[key].append(row)

    distance_events: list[dict[str, Any]] = []
    expected_events: list[dict[str, Any]] = []
    for key, event_rows in event_groups.items():
        common = dict(zip(event_fields, key))
        common.update(
            {
                "N": int(float(common["N"])),
                "seed": int(float(common["seed"])),
                "graph": int(float(common["graph"])),
                "source": int(float(common["source"])),
                "layer": int(float(common["layer"])),
                "source_role": str(event_rows[0]["source_role"]),
            }
        )
        for method in METHODS:
            masses = np.asarray([_float(row, method) for row in event_rows])
            distances = np.asarray(
                [_integer(row, "distance") for row in event_rows], dtype=np.float64
            )
            total = float(masses.sum())
            if not np.isfinite(total) or total <= float(effect_floor):
                continue
            expected_events.append(
                {
                    **common,
                    "method": method,
                    "expected_distance": float(np.dot(masses, distances) / total),
                    "total_mass": total,
                }
            )
            for distance in sorted(set(int(value) for value in distances)):
                mask = distances == distance
                distance_events.append(
                    {
                        **common,
                        "method": method,
                        "distance": int(distance),
                        "normalised_mass": float(masses[mask].sum() / total),
                    }
                )

    def hierarchical(
        values: Sequence[Mapping[str, Any]],
        *,
        value_name: str,
        extra_fields: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        graph_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        base_fields = ("model", "N", "seed", "graph", "channel", "layer", "method")
        for row in values:
            key = tuple(row[field] for field in (*base_fields, *extra_fields))
            graph_groups[key].append(float(row[value_name]))
        seed_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        seed_fields = ("model", "N", "seed", "channel", "layer", "method")
        for key, group_values in graph_groups.items():
            mapping = dict(zip((*base_fields, *extra_fields), key))
            seed_key = tuple(mapping[field] for field in (*seed_fields, *extra_fields))
            seed_groups[seed_key].append(float(np.mean(group_values)))
        summary_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        summary_fields = ("model", "N", "channel", "layer", "method")
        for key, group_values in seed_groups.items():
            mapping = dict(zip((*seed_fields, *extra_fields), key))
            summary_key = tuple(
                mapping[field] for field in (*summary_fields, *extra_fields)
            )
            summary_groups[summary_key].append(float(np.mean(group_values)))
        output: list[dict[str, Any]] = []
        for key, seed_values in summary_groups.items():
            mapping = dict(zip((*summary_fields, *extra_fields), key))
            array = np.asarray(seed_values, dtype=np.float64)
            output.append(
                {
                    **mapping,
                    "N": int(float(mapping["N"])),
                    "layer": int(float(mapping["layer"])),
                    "mean": float(array.mean()),
                    "sd_across_seeds": (
                        float(array.std(ddof=1)) if len(array) > 1 else 0.0
                    ),
                    "seeds": int(len(array)),
                }
            )
        return output

    profiles = hierarchical(
        distance_events,
        value_name="normalised_mass",
        extra_fields=("distance",),
    )
    expected = hierarchical(
        expected_events,
        value_name="expected_distance",
        extra_fields=(),
    )
    for row in profiles:
        row["distance"] = int(float(row["distance"]))
    return profiles, expected


def summarise_beneficial(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate signed central-readout benefit event -> graph -> seed."""

    graph_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    fields = ("model", "N", "seed", "graph", "channel", "source_role")
    for row in rows:
        key = tuple(row[field] for field in fields)
        graph_groups[key].append(_float(row, "beneficial_carriage"))
    seed_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for key, values in graph_groups.items():
        mapping = dict(zip(fields, key))
        seed_key = tuple(
            mapping[field]
            for field in ("model", "N", "seed", "channel", "source_role")
        )
        seed_groups[seed_key].append(float(np.mean(values)))
    result_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for key, values in seed_groups.items():
        mapping = dict(
            zip(("model", "N", "seed", "channel", "source_role"), key)
        )
        result_key = tuple(
            mapping[field]
            for field in ("model", "N", "channel", "source_role")
        )
        result_groups[result_key].append(float(np.mean(values)))
    result: list[dict[str, Any]] = []
    for key, values in result_groups.items():
        mapping = dict(zip(("model", "N", "channel", "source_role"), key))
        array = np.asarray(values, dtype=np.float64)
        result.append(
            {
                **mapping,
                "N": int(float(mapping["N"])),
                "mean": float(array.mean()),
                "sd_across_seeds": (
                    float(array.std(ddof=1)) if len(array) > 1 else 0.0
                ),
                "seeds": int(len(array)),
            }
        )
    return result


def summarise_bamberger(
    rows: Sequence[Mapping[str, Any]],
    *,
    effect_floor: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize Bamberger input influence per graph, then aggregate graph -> seed."""

    graph_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["model"]),
            _integer(row, "N"),
            _integer(row, "seed"),
            _integer(row, "graph"),
        )
        graph_groups[key].append(row)
    graph_profiles: list[dict[str, Any]] = []
    graph_ranges: list[dict[str, Any]] = []
    for (model, records, seed, graph), graph_rows in graph_groups.items():
        influence = np.asarray(
            [_float(row, "influence") for row in graph_rows],
            dtype=np.float64,
        )
        distances = np.asarray(
            [_integer(row, "distance_to_central_output") for row in graph_rows],
            dtype=np.float64,
        )
        total = float(influence.sum())
        if not np.isfinite(total) or total <= float(effect_floor):
            continue
        common = {
            "model": model,
            "N": int(records),
            "seed": int(seed),
            "graph": int(graph),
        }
        graph_ranges.append(
            {
                **common,
                "expected_distance": float(np.dot(influence, distances) / total),
            }
        )
        for distance in sorted(set(int(value) for value in distances)):
            graph_profiles.append(
                {
                    **common,
                    "distance": distance,
                    "normalised_influence": float(
                        influence[distances == distance].sum() / total
                    ),
                }
            )

    def aggregate(
        values: Sequence[Mapping[str, Any]],
        *,
        value_name: str,
        include_distance: bool,
    ) -> list[dict[str, Any]]:
        seed_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        for row in values:
            key: tuple[Any, ...] = (
                row["model"],
                int(row["N"]),
                int(row["seed"]),
            )
            if include_distance:
                key += (int(row["distance"]),)
            seed_groups[key].append(float(row[value_name]))
        summary_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        for key, graph_values in seed_groups.items():
            summary_key = (key[0], key[1]) + (key[3:] if include_distance else ())
            summary_groups[summary_key].append(float(np.mean(graph_values)))
        output: list[dict[str, Any]] = []
        for key, seed_values in summary_groups.items():
            array = np.asarray(seed_values, dtype=np.float64)
            row = {
                "model": str(key[0]),
                "N": int(key[1]),
                "mean": float(array.mean()),
                "sd_across_seeds": (
                    float(array.std(ddof=1)) if len(array) > 1 else 0.0
                ),
                "seeds": int(len(array)),
            }
            if include_distance:
                row["distance"] = int(key[2])
            output.append(row)
        return output

    return (
        aggregate(
            graph_profiles,
            value_name="normalised_influence",
            include_distance=True,
        ),
        aggregate(
            graph_ranges,
            value_name="expected_distance",
            include_distance=False,
        ),
    )


def _plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 9,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_figure(fig: Any, stem: Path) -> dict[str, str]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    return {"png": str(png), "pdf": str(pdf)}


def plot_reach_profiles(
    profiles: Sequence[Mapping[str, Any]],
    *,
    channel: str,
    models: Sequence[str],
    ns: Sequence[int],
    layer: int,
    output_stem: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    _plot_style()
    fig, axes = plt.subplots(
        len(ns),
        len(models),
        figsize=(max(5.0, 3.25 * len(models)), max(4.4, 2.35 * len(ns))),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, records in enumerate(ns):
        for column_index, model in enumerate(models):
            ax = axes[row_index, column_index]
            selected = [
                row
                for row in profiles
                if str(row["channel"]) == str(channel)
                and str(row["model"]) == str(model)
                and int(row["N"]) == int(records)
                and int(row["layer"]) == int(layer)
            ]
            for method in METHODS:
                values = sorted(
                    (row for row in selected if row["method"] == method),
                    key=lambda row: int(row["distance"]),
                )
                if not values:
                    continue
                x = np.asarray([int(row["distance"]) for row in values])
                y = np.asarray([float(row["mean"]) for row in values])
                error = np.asarray(
                    [float(row["sd_across_seeds"]) for row in values]
                )
                ax.plot(
                    x,
                    y,
                    color=METHOD_COLOURS[method],
                    marker=METHOD_MARKERS[method],
                    linewidth=1.8,
                    markersize=4.2,
                    label=METHOD_LABELS[method],
                )
                ax.fill_between(
                    x,
                    np.maximum(0.0, y - error),
                    y + error,
                    color=METHOD_COLOURS[method],
                    alpha=0.14,
                    linewidth=0,
                )
            ceiling = architectural_reach(str(model), int(layer))
            if ceiling < 3:
                ax.axvspan(
                    ceiling + 0.5,
                    3.5,
                    color="#BDBDBD",
                    alpha=0.18,
                    linewidth=0,
                )
                ax.text(
                    3.34,
                    0.94,
                    "outside\nsupport",
                    ha="right",
                    va="top",
                    fontsize=7,
                    color="#666666",
                    transform=ax.get_xaxis_transform(),
                )
            ax.set_title(f"N={int(records)} · {MODEL_LABELS[str(model)]}")
            ax.set_xlim(-0.15, 3.15)
            ax.set_ylim(bottom=0)
            ax.set_xticks((0, 1, 2, 3))
            ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
            if row_index == len(ns) - 1:
                ax.set_xlabel("Source–carrier distance")
    fig.supylabel("Fraction of event carriage", x=0.01, fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        frameon=False,
        ncol=2,
    )
    channel_label = (
        "Semantic donor swaps"
        if channel == "semantic"
        else "Structural RRWP donor swaps"
    )
    quantity_label = (
        "finite and local differential reach"
        if int(layer) == 1
        else "finite and local arrival at the readout"
    )
    fig.suptitle(
        f"{channel_label}: {quantity_label}",
        fontsize=13,
        y=0.997,
    )
    subtitle = (
        "Layer 1 routed-message carriers; mean ± SD across trained seeds"
        if int(layer) == 1
        else (
            "Layer 2 routed-message carriers; central readout lies at d=1 from the "
            "record and d=2 from the query"
        )
    )
    fig.text(
        0.5,
        0.925,
        subtitle,
        ha="center",
        va="top",
        fontsize=9,
        color="#4D4D4D",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.79))
    paths = _save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def plot_bamberger_profiles(
    profiles: Sequence[Mapping[str, Any]],
    ranges: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str],
    ns: Sequence[int],
    output_stem: Path,
) -> dict[str, str]:
    """Plot the original clean semantic input-to-central-output influence range."""

    import matplotlib.pyplot as plt

    _plot_style()
    fig, axes = plt.subplots(
        len(ns),
        len(models),
        figsize=(max(5.0, 3.25 * len(models)), max(4.4, 2.35 * len(ns))),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, records in enumerate(ns):
        for column_index, model in enumerate(models):
            ax = axes[row_index, column_index]
            values = sorted(
                (
                    row
                    for row in profiles
                    if str(row["model"]) == str(model)
                    and int(row["N"]) == int(records)
                ),
                key=lambda row: int(row["distance"]),
            )
            if values:
                x = np.asarray([int(row["distance"]) for row in values])
                y = np.asarray([float(row["mean"]) for row in values])
                error = np.asarray(
                    [float(row["sd_across_seeds"]) for row in values]
                )
                ax.plot(
                    x,
                    y,
                    color="#333333",
                    marker="o",
                    linewidth=1.8,
                    markersize=4.2,
                )
                ax.fill_between(
                    x,
                    np.maximum(0.0, y - error),
                    y + error,
                    color="#777777",
                    alpha=0.16,
                    linewidth=0,
                )
            range_row = next(
                (
                    row
                    for row in ranges
                    if str(row["model"]) == str(model)
                    and int(row["N"]) == int(records)
                ),
                None,
            )
            if range_row is not None:
                ax.text(
                    0.97,
                    0.93,
                    f"Expected distance = {float(range_row['mean']):.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7.5,
                    color="#555555",
                )
            ax.axvline(1, color="#CC79A7", linestyle=":", linewidth=0.9, alpha=0.7)
            ax.axvline(2, color="#009E73", linestyle=":", linewidth=0.9, alpha=0.7)
            ax.set_title(f"N={int(records)} · {MODEL_LABELS[str(model)]}")
            ax.set_xlim(-0.1, 2.1)
            ax.set_ylim(bottom=0)
            ax.set_xticks((0, 1, 2))
            ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
            if row_index == len(ns) - 1:
                ax.set_xlabel("Input–central distance")
    fig.supylabel("Fraction of input influence", x=0.01, fontsize=10)
    fig.suptitle("Bamberger semantic input–output range", fontsize=13, y=0.997)
    fig.text(
        0.5,
        0.925,
        (
            "Clean encoded-input Jacobian; dotted anchors mark the requested record "
            "(d=1) and query (d=2)"
        ),
        ha="center",
        va="top",
        fontsize=9,
        color="#4D4D4D",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    paths = _save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def plot_expected_distance(
    expected: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str],
    ns: Sequence[int],
    layer: int,
    output_stem: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    _plot_style()
    fig, axes = plt.subplots(
        2,
        len(models),
        figsize=(max(5.0, 3.3 * len(models)), 5.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, channel in enumerate(CHANNELS):
        for column_index, model in enumerate(models):
            ax = axes[row_index, column_index]
            selected = [
                row
                for row in expected
                if row["channel"] == channel
                and row["model"] == model
                and int(row["layer"]) == int(layer)
            ]
            for method in METHODS:
                by_n = {
                    int(row["N"]): row
                    for row in selected
                    if row["method"] == method
                }
                present = [int(value) for value in ns if int(value) in by_n]
                if not present:
                    continue
                y = np.asarray([float(by_n[value]["mean"]) for value in present])
                error = np.asarray(
                    [float(by_n[value]["sd_across_seeds"]) for value in present]
                )
                ax.errorbar(
                    present,
                    y,
                    yerr=error,
                    color=METHOD_COLOURS[method],
                    marker=METHOD_MARKERS[method],
                    linewidth=1.8,
                    markersize=4.5,
                    capsize=2,
                    label=METHOD_LABELS[method],
                )
            channel_label = "Semantic swaps" if channel == "semantic" else "Structural swaps"
            ax.set_title(f"{MODEL_LABELS[str(model)]}\n{channel_label}")
            ax.set_xticks(tuple(int(value) for value in ns))
            ax.set_ylim(0, 3)
            ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
            if row_index == 1:
                ax.set_xlabel("Number of records, N")
    fig.supylabel("Expected carrier distance", x=0.01, fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        frameon=False,
        ncol=2,
    )
    fig.suptitle("Expected finite and local differential reach", fontsize=13, y=0.995)
    fig.text(
        0.5,
        0.925,
        f"Layer {int(layer)} routed-message carriers; mean ± SD across trained seeds",
        ha="center",
        va="top",
        fontsize=9,
        color="#4D4D4D",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.79), h_pad=2.8, w_pad=2.0)
    paths = _save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def plot_beneficial(
    summary: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str],
    ns: Sequence[int],
    output_stem: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    _plot_style()
    role_style = {
        "query": ("#009E73", "o", "Query source (d=2)"),
        "requested_record": ("#CC79A7", "s", "Requested record (d=1)"),
    }
    fig, axes = plt.subplots(
        2,
        len(models),
        figsize=(max(5.0, 3.3 * len(models)), 5.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, channel in enumerate(CHANNELS):
        for column_index, model in enumerate(models):
            ax = axes[row_index, column_index]
            selected = [
                row
                for row in summary
                if row["channel"] == channel and row["model"] == model
            ]
            for role, (colour, marker, label) in role_style.items():
                by_n = {
                    int(row["N"]): row
                    for row in selected
                    if row["source_role"] == role
                }
                present = [int(value) for value in ns if int(value) in by_n]
                if not present:
                    continue
                y = np.asarray([float(by_n[value]["mean"]) for value in present])
                error = np.asarray(
                    [float(by_n[value]["sd_across_seeds"]) for value in present]
                )
                ax.errorbar(
                    present,
                    y,
                    yerr=error,
                    color=colour,
                    marker=marker,
                    linewidth=1.8,
                    markersize=4.5,
                    capsize=2,
                    label=label,
                )
            ax.axhline(0.0, color="#777777", linewidth=0.8)
            channel_label = "Semantic swaps" if channel == "semantic" else "Structural swaps"
            ax.set_title(f"{MODEL_LABELS[str(model)]}\n{channel_label}")
            ax.set_xticks(tuple(int(value) for value in ns))
            ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
            if row_index == 1:
                ax.set_xlabel("Number of records, N")
    fig.supylabel("Beneficial carriage", x=0.01, fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        frameon=False,
        ncol=2,
    )
    fig.suptitle("Task benefit of clean final-state information", fontsize=13, y=0.995)
    fig.text(
        0.5,
        0.925,
        "Central readout only; positive values mean the clean state reduces cross-entropy loss",
        ha="center",
        va="top",
        fontsize=9,
        color="#4D4D4D",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.79), h_pad=2.8, w_pad=2.0)
    paths = _save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def render_figures(
    config: ReachConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Regenerate all figures from CSV results without loading GRIT or checkpoints."""

    results_dir = output_dir / "results"
    reach_path = results_dir / "reach_carriers.csv"
    beneficial_path = results_dir / "beneficial_carriage.csv"
    bamberger_path = results_dir / "bamberger_semantic_input.csv"
    if (
        not reach_path.is_file()
        or not beneficial_path.is_file()
        or not bamberger_path.is_file()
    ):
        raise FileNotFoundError(
            "figure phase requires reach_carriers.csv, beneficial_carriage.csv, "
            "and bamberger_semantic_input.csv under results/"
        )
    reach_rows = _read_csv(reach_path)
    beneficial_rows = _read_csv(beneficial_path)
    bamberger_rows = _read_csv(bamberger_path)
    profiles, expected = summarise_profiles(
        reach_rows,
        effect_floor=float(config.effect_floor),
        layer=None,
    )
    beneficial = summarise_beneficial(beneficial_rows)
    bamberger_profiles, bamberger_ranges = summarise_bamberger(
        bamberger_rows,
        effect_floor=float(config.effect_floor),
    )
    _write_csv(results_dir / "reach_profiles_summary.csv", profiles)
    _write_csv(results_dir / "expected_distance_summary.csv", expected)
    _write_csv(results_dir / "beneficial_summary.csv", beneficial)
    _write_csv(
        results_dir / "bamberger_semantic_profiles_summary.csv",
        bamberger_profiles,
    )
    _write_csv(
        results_dir / "bamberger_semantic_range_summary.csv",
        bamberger_ranges,
    )

    figure_dir = output_dir / "figures"
    figures: dict[str, Any] = {}
    for channel in CHANNELS:
        figures[f"{channel}_reach_layer1"] = plot_reach_profiles(
            profiles,
            channel=channel,
            models=config.models,
            ns=config.ns,
            layer=1,
            output_stem=figure_dir / f"{channel}_reach_layer1",
        )
        figures[f"{channel}_arrival_layer2"] = plot_reach_profiles(
            profiles,
            channel=channel,
            models=config.models,
            ns=config.ns,
            layer=2,
            output_stem=figure_dir / f"{channel}_arrival_layer2",
        )
    figures["expected_reach_layer1"] = plot_expected_distance(
        expected,
        models=config.models,
        ns=config.ns,
        layer=1,
        output_stem=figure_dir / "expected_reach_layer1",
    )
    figures["bamberger_semantic_range"] = plot_bamberger_profiles(
        bamberger_profiles,
        bamberger_ranges,
        models=config.models,
        ns=config.ns,
        output_stem=figure_dir / "bamberger_semantic_range",
    )
    figures["beneficial"] = plot_beneficial(
        beneficial,
        models=config.models,
        ns=config.ns,
        output_stem=figure_dir / "beneficial_carriage",
    )
    _write_json(
        figure_dir / "nar_reach_analysis.metadata.json",
        {
            **config.scientific_record,
            "fingerprint": config.fingerprint,
            "figures": figures,
            "uncertainty": (
                "mean ± sample SD across trained seeds after "
                "event -> graph -> seed aggregation"
            ),
            "known_references": nar_known_references(),
            "interpretation": {
                "primary": (
                    "Layer-1 full distance curves are the primary learned-reach result; "
                    "expected layer-1 distance is a compact secondary summary."
                ),
                "layer_2": (
                    "Layer-2 curves are an arrival/readout sanity check. Because NAR reads "
                    "only the central final state, output-projected layer-2 mass should "
                    "collapse to the known central-carrier distances."
                ),
                "bamberger": (
                    "The separate Bamberger figure is the original semantic encoded-input "
                    "to central-output Jacobian range. It is not a carrier-path decomposition, "
                    "and the original method has no canonical structural donor-swap analogue."
                ),
                "beneficial": (
                    "Beneficial carriage is a signed final-state task-benefit diagnostic, "
                    "not a distributed reach ground truth."
                ),
            },
        },
    )
    return figures


def run(
    config: ReachConfig,
    *,
    output_dir: str | Path,
    training_run_dir: str | Path | None = None,
    grit_dir: str | Path = "/content/GRIT",
    phase: str = "all",
    install_grit: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Run ``measure``, ``figures``, or both, with figure-only inference isolation."""

    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if phase not in {"all", "measure", "figures"}:
        raise ValueError("phase must be 'all', 'measure', or 'figures'")
    _write_json(
        output / "reach_config.json",
        {
            **asdict(config),
            "analysis_version": ANALYSIS_VERSION,
            "fingerprint": config.fingerprint,
            "known_references": nar_known_references(),
        },
    )
    result: dict[str, Any] = {}
    if phase in {"all", "measure"}:
        if training_run_dir is None:
            raise ValueError("training_run_dir is required for measurement")
        result.update(
            measure(
                config,
                output_dir=output,
                training_run_dir=Path(training_run_dir),
                grit_dir=Path(grit_dir),
                install_grit=bool(install_grit),
                progress=bool(progress),
            )
        )
    if phase in {"all", "figures"}:
        result["figures"] = render_figures(config, output_dir=output)
    return result


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _parse_csv_strings(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "measure", "figures"), default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-run-dir")
    parser.add_argument("--grit-dir", default="/content/GRIT")
    parser.add_argument("--models", default="1hop,2hop,dense")
    parser.add_argument("--ns", default="8,16,64")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--graphs", type=int, default=16)
    parser.add_argument("--donors-per-source", type=int, default=4)
    parser.add_argument("--semantic-donor-graphs", type=int, default=256)
    parser.add_argument("--analysis-seed", type=int, default=73_021)
    parser.add_argument("--accelerator", default="cuda:0")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--effect-floor", type=float, default=1.0e-10)
    parser.add_argument("--skip-grit-install", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    config = ReachConfig(
        models=_parse_csv_strings(args.models),
        ns=_parse_csv_ints(args.ns),
        seeds=_parse_csv_ints(args.seeds),
        width=int(args.width),
        graphs=int(args.graphs),
        donors_per_source=int(args.donors_per_source),
        semantic_donor_graphs=int(args.semantic_donor_graphs),
        analysis_seed=int(args.analysis_seed),
        accelerator=str(args.accelerator),
        num_threads=int(args.num_threads),
        effect_floor=float(args.effect_floor),
    )
    return run(
        config,
        output_dir=args.output_dir,
        training_run_dir=args.training_run_dir,
        grit_dir=args.grit_dir,
        phase=args.phase,
        install_grit=not bool(args.skip_grit_install),
        progress=not bool(args.quiet),
    )


if __name__ == "__main__":
    main()
