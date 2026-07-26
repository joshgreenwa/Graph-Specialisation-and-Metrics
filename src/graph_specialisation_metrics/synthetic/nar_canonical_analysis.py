"""Canonical donor-swap specialisation, causal validation, and carriage for fixed-N NAR.

This is the task adapter for checkpoints trained by :mod:`nar_grit_fixed`.  The estimators,
bootstrap, event cache, causal endpoints, and ordinary per-seed figures remain in the repository's
canonical ``methodology`` package.  This module supplies only the NAR graph/model boundary and the
cross-model dissertation figures requested for the NAR experiment.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from ..carriage.content import FullNodeContentAdapter
from ..methodology.backend import BackendCapture, CleanJacobians
from ..methodology.bootstrap import Observation, nested_percentile_interval, trimmed_mean
from ..methodology.cache import atomic_json, checkpoint_sha256
from ..methodology.distance import display_bins
from ..methodology.protocol import (
    BootstrapPolicy,
    ExecutionPolicy,
    FamilyPolicy,
    MethodologyConfig,
    NumericalPolicy,
    RunSizes,
    deterministic_splits,
)
from ..methodology.sampling import DonorNode, node_degrees
from ..methodology.tasks import CanonicalTask, OutputGeometry, TASKS, register
from . import nar_grit_fixed as training


ANALYSIS_VERSION = "nar-canonical-donor-swap-v1"
MODEL_ORDER = ("1hop", "2hop", "dense")
MODEL_LABELS = {"1hop": "1-hop GRIT", "2hop": "2-hop GRIT", "dense": "Dense GRIT"}
MODEL_COLOURS = {"1hop": "#2F6B9A", "2hop": "#777777", "dense": "#B33A3A"}
MODEL_MARKERS = {"1hop": "o", "2hop": "s", "dense": "D"}
SEED_MARKERS = ("o", "s", "^", "D", "P", "X")


def _cross_entropy_per_graph(prediction, target):
    import torch.nn.functional as F

    return F.cross_entropy(
        prediction.reshape(prediction.shape[0], -1),
        target.reshape(-1).long(),
        reduction="none",
    )


def categorical_accuracy_metric(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return graph-level categorical accuracy for N-way recall logits.

    Canonical validation passes predictions as ``[graphs, classes]`` and may pass labels as
    either ``[graphs]`` or ``[graphs, 1]``.  Normalising both here keeps the registered dataset
    metric identical to the accuracy used by the NAR training frontend.
    """

    logits = np.asarray(predictions)
    labels = np.asarray(targets).reshape(-1)
    if logits.ndim < 2:
        raise ValueError(
            "NAR categorical accuracy requires predictions with a class dimension"
        )
    logits = logits.reshape(-1, int(logits.shape[-1]))
    if int(logits.shape[0]) != int(labels.size):
        raise ValueError(
            "NAR prediction/target graph counts differ: "
            f"{int(logits.shape[0])} != {int(labels.size)}"
        )
    if not labels.size:
        raise ValueError("NAR categorical accuracy requires at least one graph")
    return float(np.mean(np.argmax(logits, axis=-1) == labels))


@dataclass(frozen=True)
class NarTaskSpec:
    name: str
    title: str
    model_name: str
    records: int
    width: int
    training_run_dir: str
    content_adapter: Any = dataclasses.field(default_factory=FullNodeContentAdapter)
    metric_fn: Any = staticmethod(categorical_accuracy_metric)


def task_name(model_name: str, records: int) -> str:
    return f"nar_{str(model_name)}_N{int(records)}"


def register_nar_tasks(
    *,
    models: Sequence[str],
    analysis_ns: Sequence[int],
    width: int,
    training_run_dir: str | Path,
) -> tuple[str, ...]:
    """Register one canonical task per support-by-N checkpoint cell."""

    names: list[str] = []
    for records in analysis_ns:
        for model_name in models:
            name = task_name(model_name, int(records))
            names.append(name)
            if name in TASKS:
                existing = TASKS[name].spec
                expected = (str(model_name), int(records), int(width), str(training_run_dir))
                actual = (
                    str(existing.model_name),
                    int(existing.records),
                    int(existing.width),
                    str(existing.training_run_dir),
                )
                if actual != expected:
                    raise ValueError(
                        f"NAR task {name!r} is already registered with {actual}, expected {expected}"
                    )
                continue
            spec = NarTaskSpec(
                name=name,
                title=f"{MODEL_LABELS[str(model_name)]} associative recall (N={int(records)})",
                model_name=str(model_name),
                records=int(records),
                width=int(width),
                training_run_dir=str(training_run_dir),
            )
            register(
                CanonicalTask(
                    name=name,
                    backend_kind="nar_grit",
                    spec=spec,
                    output=OutputGeometry("logits", None, "unit"),
                    loss_per_graph=_cross_entropy_per_graph,
                    semantic_fields=("x",),
                    immutable_control_fields=(
                        "central_idx",
                        "intermediate_idx",
                        "query_idx",
                        "target_idx",
                        "record_mask",
                        "n_records",
                    ),
                    node_structural_fields=(),
                    pair_structural_fields=(),
                    dense_pair_structural_fields=("rrwp",),
                    fixed_support_fields=("edge_index", "adj", "y"),
                    virtual_node=False,
                    carrier_policy="all_nodes_central_readout",
                    adapter_version=f"{ANALYSIS_VERSION}:role-aligned-donors",
                    extra_known_fields=("num_nodes",),
                )
            )
    return tuple(names)


def _checkpoint_candidates(spec: NarTaskSpec, seed: int) -> list[Path]:
    directory = Path(spec.training_run_dir) / "checkpoints"
    prefix = (
        f"{spec.model_name}__d{int(spec.width)}__N{int(spec.records)}"
        f"__seed_{int(seed)}__"
    )
    return sorted(
        directory.glob(f"{prefix}*.pt"),
        key=lambda item: (item.stat().st_mtime_ns, item.name),
        reverse=True,
    )


def find_checkpoint(spec: NarTaskSpec, seed: int, explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = _checkpoint_candidates(spec, seed)
    if not candidates:
        raise FileNotFoundError(
            f"no checkpoint matching {spec.model_name}, width={spec.width}, "
            f"N={spec.records}, seed={seed} under {spec.training_run_dir}/checkpoints"
        )
    return candidates[0]


def _training_config(payload: Mapping[str, Any], spec: NarTaskSpec) -> training.Config:
    raw = dict(payload.get("config", {}))
    allowed = {field.name for field in dataclasses.fields(training.Config)}
    values = {key: value for key, value in raw.items() if key in allowed}
    for key in ("ns", "mechanistic_ns", "widths", "models", "seeds"):
        if key in values:
            values[key] = tuple(values[key])
    values.setdefault("widths", (int(spec.width),))
    values.setdefault("analysis_width", int(spec.width))
    values.setdefault("models", tuple(MODEL_ORDER))
    values.setdefault("ns", (int(spec.records),))
    values.setdefault("mechanistic_ns", (int(spec.records),))
    values.setdefault("seeds", (int(payload.get("seed", 0)),))
    return training.Config(**values)


def _graph_from_batch(batch: training.NarBatch, index: int = 0):
    import torch
    from torch_geometric.data import Data

    graph = Data()
    graph.x = batch.x[index].clone()
    graph.adj = batch.adj[index].clone()
    graph.rrwp = batch.rrwp[index].clone()
    graph.edge_index = torch.nonzero(graph.adj > 0, as_tuple=False).t().contiguous().long()
    graph.y = batch.y[index : index + 1].clone()
    for name in (
        "central_idx",
        "intermediate_idx",
        "query_idx",
        "target_idx",
        "n_records",
    ):
        setattr(graph, name, getattr(batch, name)[index].clone())
    graph.record_mask = batch.record_mask[index].clone()
    graph.num_nodes = int(graph.x.shape[0])
    return graph


class NarGraphDataset:
    """Deterministic lazy graphs; analysis never mutates or stores training data."""

    def __init__(self, cfg: training.Config, records: int, size: int, seed: int):
        self.cfg = cfg
        self.records = int(records)
        self.size = int(size)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        if not 0 <= int(index) < self.size:
            raise IndexError(index)
        batch = training.make_batch(
            self.cfg,
            1,
            self.records,
            self.seed + int(index) * 1_000_003,
        )
        return _graph_from_batch(batch)


class NarSemanticDonorPool:
    """Graph-balanced, role-aligned NAR donors with no redundant key corruption.

    A query source receives another graph's query row (different queried key, padding value held).
    A requested-record source receives the same-key record from another graph (key held, value
    changed). Each eligible donor graph contributes one typed candidate, after which the canonical
    graph-uniform then node-uniform law is unchanged.
    """

    def __init__(self, donor_graphs: Sequence[tuple[int, Any]], records: int):
        self.records = int(records)
        self.graphs = tuple((int(graph_id), graph) for graph_id, graph in donor_graphs)
        if not self.graphs:
            raise ValueError("NAR semantic donor pool is empty")

    def eligible(
        self,
        source_payload: Any,
        source_degree: int,
        *,
        base_graph_id: int | None = None,
    ) -> dict[int, tuple[DonorNode, ...]]:
        source = np.asarray(source_payload).reshape(-1)
        is_query = int(source[1]) == self.records and int(source[0]) < self.records
        candidates: list[DonorNode] = []
        for graph_id, graph in self.graphs:
            if base_graph_id is not None and int(graph_id) == int(base_graph_id):
                continue
            if is_query:
                nodes = [int(graph.query_idx)]
            else:
                matches = np.flatnonzero(
                    graph.record_mask.detach().cpu().numpy()
                    & (graph.x[:, 0].detach().cpu().numpy() == int(source[0]))
                )
                nodes = [int(value) for value in matches]
            degrees = node_degrees(graph)
            for node in nodes:
                payload = tuple(
                    graph.x[node].detach().cpu().numpy().reshape(-1).tolist()
                )
                if np.array_equal(np.asarray(payload), source):
                    continue
                candidates.append(
                    DonorNode(
                        graph_id=graph_id,
                        node=node,
                        degree=int(degrees[node]),
                        payload=payload,
                    )
                )
        if not candidates:
            return {}
        gap = min(abs(int(item.degree) - int(source_degree)) for item in candidates)
        selected: dict[int, list[DonorNode]] = {}
        for item in candidates:
            if abs(int(item.degree) - int(source_degree)) == gap:
                selected.setdefault(item.graph_id, []).append(item)
        return {graph: tuple(nodes) for graph, nodes in selected.items()}

    def draw(
        self,
        source_payload: Any,
        source_degree: int,
        count: int,
        rng: np.random.Generator,
        *,
        base_graph_id: int | None = None,
    ) -> tuple[DonorNode, ...]:
        eligible = self.eligible(
            source_payload,
            source_degree,
            base_graph_id=base_graph_id,
        )
        graph_ids = np.asarray(sorted(eligible), dtype=np.int64)
        if not graph_ids.size:
            raise ValueError("NAR semantic source has no non-identical typed donor")
        result = []
        for _ in range(int(count)):
            graph_id = int(rng.choice(graph_ids))
            nodes = eligible[graph_id]
            result.append(nodes[int(rng.integers(0, len(nodes)))])
        return tuple(result)


def _to_nar_batch(data_list: Sequence[Any]) -> training.NarBatch:
    import torch

    if not data_list:
        raise ValueError("NAR forward requires at least one graph")
    nodes = {int(data.num_nodes) for data in data_list}
    if len(nodes) != 1:
        raise ValueError("one fixed-N NAR forward cannot mix node counts")

    def stack(name: str):
        return torch.stack([getattr(data, name) for data in data_list], dim=0)

    return training.NarBatch(
        x=stack("x"),
        adj=stack("adj"),
        rrwp=stack("rrwp"),
        central_idx=stack("central_idx").reshape(-1),
        intermediate_idx=stack("intermediate_idx").reshape(-1),
        query_idx=stack("query_idx").reshape(-1),
        target_idx=stack("target_idx").reshape(-1),
        record_mask=stack("record_mask"),
        y=torch.cat([data.y.reshape(-1) for data in data_list], dim=0),
        n_records=stack("n_records").reshape(-1),
    )


class CanonicalNarBackend:
    """Expose the native routed-wV and final-node-state sites of the trained NAR GRIT."""

    def __init__(self, model: Any, task: CanonicalTask, device: Any):
        self.model = model
        self.task = task
        self.device = device

    @property
    def geometry(self) -> dict[str, int]:
        return {
            "layers": int(self.model.L),
            "heads": int(self.model.H),
            "head_width": int(self.model.dh),
            "hidden_width": int(self.model.width),
            "outputs": int(self.model.records),
        }

    @property
    def special_carrier_labels(self) -> tuple[str, ...]:
        return ()

    def _capture_batch(
        self,
        data_list: Sequence[Any],
        *,
        require_grad: bool,
        want_attention: bool = False,
    ) -> tuple[BackendCapture, list[Any] | None]:
        import torch

        batch = _to_nar_batch(data_list).to(self.device)
        routed: list[Any] = [None] * int(self.model.L)
        attention: list[Any] = [None] * int(self.model.L)
        final: dict[str, Any] = {}
        layer_by_module = {
            id(module): layer
            for layer, module in enumerate(self.model.attention_layers)
        }

        def attention_hook(module, inputs, output):
            layer = layer_by_module[id(module)]
            routed[layer] = output[0]
            if want_attention:
                attention[layer] = inputs[0].attn.squeeze(-1)

        def final_hook(_module, _inputs, output):
            final["state"] = output.x

        handles = [
            module.register_forward_hook(attention_hook)
            for module in self.model.attention_layers
        ]
        handles.append(self.model.layers[-1].register_forward_hook(final_hook))
        try:
            context = torch.enable_grad() if require_grad else torch.no_grad()
            with context:
                prediction = self.model(batch)
        finally:
            for handle in handles:
                handle.remove()
        if any(value is None for value in routed) or "state" not in final:
            raise RuntimeError("NAR native transport/final-state hook did not fire")
        repetitions = len(data_list)
        nodes = int(data_list[0].num_nodes)
        transport = tuple(
            value.reshape(repetitions, nodes, self.model.H, self.model.dh)
            for value in routed
        )
        final_state = final["state"].reshape(repetitions, nodes, self.model.width)
        capture = BackendCapture(
            prediction=prediction,
            z=prediction,
            target=batch.y,
            transport=transport,
            final_state=final_state,
            real_mask=None,
        )
        # Jacobians must target the actual tensors consumed downstream, not post-hoc views.
        capture._raw_transport = tuple(routed)
        capture._raw_final_state = final["state"]
        return capture, (attention if want_attention else None)

    def capture(
        self,
        data_list: Sequence[Any],
        *,
        require_grad: bool,
        include_virtual_transport: bool = True,
    ) -> BackendCapture:
        del include_virtual_transport
        if require_grad and len(data_list) != 1:
            raise ValueError("single-graph capture is required for an individual Jacobian")
        capture, _ = self._capture_batch(data_list, require_grad=require_grad)
        if require_grad:
            capture.transport = tuple(value[0] for value in capture.transport)
            capture.final_state = capture.final_state[0]
        return capture

    def capture_groups(
        self,
        groups: Sequence[Sequence[Any]],
        *,
        include_virtual_transport: bool = True,
    ) -> list[BackendCapture]:
        del include_virtual_transport
        groups = [list(group) for group in groups]
        flat = [data for group in groups for data in group]
        capture, _ = self._capture_batch(flat, require_grad=False)
        outputs: list[BackendCapture] = []
        offset = 0
        for group in groups:
            stop = offset + len(group)
            outputs.append(
                BackendCapture(
                    prediction=capture.prediction[offset:stop],
                    z=capture.z[offset:stop],
                    target=capture.target[offset:stop],
                    transport=tuple(value[offset:stop] for value in capture.transport),
                    final_state=capture.final_state[offset:stop],
                    real_mask=None,
                )
            )
            offset = stop
        return outputs

    def _jacobians_from_capture(
        self,
        capture: BackendCapture,
        graph_index: int,
        gradients: Sequence[Sequence[Any]],
    ) -> CleanJacobians:
        import torch

        batch_size, nodes = int(capture.z.shape[0]), int(capture.final_state.shape[1])
        transport = torch.stack(
            [
                torch.stack(
                    [
                        row[layer]
                        .reshape(
                            batch_size,
                            nodes,
                            self.model.H,
                            self.model.dh,
                        )[graph_index]
                        .detach()
                        for row in gradients
                    ],
                    dim=0,
                )
                for layer in range(int(self.model.L))
            ],
            dim=1,
        )
        final_gradient = torch.stack(
            [
                row[-1]
                .reshape(batch_size, nodes, self.model.width)[graph_index]
                .detach()
                for row in gradients
            ],
            dim=0,
        )
        one = BackendCapture(
            prediction=capture.prediction[graph_index : graph_index + 1].detach(),
            z=capture.z[graph_index : graph_index + 1].detach(),
            target=capture.target[graph_index : graph_index + 1].detach(),
            transport=tuple(
                value[graph_index].detach() for value in capture.transport
            ),
            final_state=capture.final_state[graph_index].detach(),
            real_mask=None,
        )
        return CleanJacobians(one, transport, final_gradient)

    def clean_jacobians(self, data: Any) -> CleanJacobians:
        return self.clean_jacobians_many([data])[0]

    def clean_jacobians_many(self, data_list: Sequence[Any]) -> list[CleanJacobians]:
        import torch

        values = list(data_list)
        if not values:
            return []
        capture, _ = self._capture_batch(values, require_grad=True)
        targets = tuple(capture._raw_transport) + (capture._raw_final_state,)
        gradients = []
        outputs = int(capture.z.shape[1])
        for output in range(outputs):
            gradients.append(
                torch.autograd.grad(
                    capture.z[:, output].sum(),
                    targets,
                    retain_graph=output + 1 < outputs,
                    allow_unused=False,
                )
            )
        return [
            self._jacobians_from_capture(capture, graph_index, gradients)
            for graph_index in range(len(values))
        ]

    def eligible_sources(self, data: Any) -> tuple[int, ...]:
        """Use the query and requested record: the two task-defined causal source sites."""

        return tuple(
            sorted({int(data.query_idx), int(data.target_idx)})
        )

    def loss_per_graph(self, prediction, target):
        return self.task.loss_per_graph(prediction, target)

    def loss_from_pooled(self, target):
        def evaluate(pooled):
            prediction = self.model.output_head(pooled)
            return self.loss_per_graph(
                prediction,
                target.reshape(-1).expand(prediction.shape[0]),
            )

        return evaluate

    def carriage_weights(self, data: Any, final_state):
        import torch

        weights = torch.zeros(
            int(data.num_nodes),
            device=final_state.device,
            dtype=final_state.dtype,
        )
        weights[int(data.central_idx)] = 1.0
        return weights

    def transport_distances(
        self, data: Any, source: int, pristine, *, channel: str | None = None
    ) -> list[Any]:
        del data, channel
        return list(pristine[int(source), :])

    def carriage_distance_matrix(
        self,
        data: Any,
        sources: Sequence[int],
        pristine,
        *,
        channel: str | None = None,
    ):
        del data, channel
        return pristine[np.asarray(sources, dtype=np.int64), :].T

    def carriage_carrier_kind(
        self, data: Any, carrier: int, *, channel: str | None = None
    ) -> str:
        del data, carrier, channel
        return "molecular_node"

    def attention_normalization_error(self, data: Any) -> float:
        import torch

        _, attention = self._capture_batch(
            [data], require_grad=False, want_attention=True
        )
        destination = self.model.last_support_destination_local
        error = 0.0
        for values in attention or ():
            mass = torch.zeros(
                int(data.num_nodes),
                int(self.model.H),
                dtype=values.dtype,
                device=values.device,
            )
            mass.index_add_(0, destination, values)
            valid = torch.unique(destination)
            error = max(
                error,
                float(torch.max(torch.abs(mass[valid] - 1.0)).item()),
            )
        return error

    def clean_attention_distance(self, data: Any, pristine, axis: Any):
        _, attention = self._capture_batch(
            [data], require_grad=False, want_attention=True
        )
        source = self.model.last_support_source_local.detach().cpu().numpy()
        destination = self.model.last_support_destination_local.detach().cpu().numpy()
        buckets = np.asarray(
            [axis.index(pristine[int(receiver), int(sender)]) for sender, receiver in zip(source, destination)],
            dtype=np.int64,
        )
        profile = np.zeros(
            (int(self.model.L), int(self.model.H), len(axis.labels)),
            dtype=np.float64,
        )
        for layer, values in enumerate(attention or ()):
            array = values.detach().cpu().numpy()
            for column in range(len(axis.labels)):
                mask = buckets == column
                if mask.any():
                    profile[layer, :, column] = array[mask].sum(axis=0)
            denominator = profile[layer].sum(axis=-1, keepdims=True)
            np.divide(
                profile[layer],
                denominator,
                out=profile[layer],
                where=denominator > 0,
            )
        return profile

    def _native_forward(
        self,
        data_list: Sequence[Any],
        *,
        family: Sequence[tuple[int, int]],
        replacements: Sequence[Any] | None = None,
        ablate: bool = False,
    ):
        import torch

        by_layer: dict[int, list[int]] = {}
        for layer, head in family:
            by_layer.setdefault(int(layer), []).append(int(head))
        handles = []

        def make_hook(layer: int, heads: Sequence[int]):
            def hook(_module, _inputs, output):
                routed, edge = output
                changed = routed.clone()
                if ablate:
                    changed[:, list(heads), :] = 0.0
                else:
                    donor = replacements[layer].to(
                        device=changed.device, dtype=changed.dtype
                    )
                    if donor.shape != changed.shape:
                        raise RuntimeError(
                            f"NAR patch shape mismatch at layer {layer}: "
                            f"{tuple(donor.shape)} != {tuple(changed.shape)}"
                        )
                    changed[:, list(heads), :] = donor[:, list(heads), :]
                return changed, edge

            return hook

        for layer, heads in by_layer.items():
            handles.append(
                self.model.attention_layers[layer].register_forward_hook(
                    make_hook(layer, sorted(set(heads)))
                )
            )
        batch = _to_nar_batch(data_list).to(self.device)
        try:
            with torch.no_grad():
                prediction = self.model(batch)
        finally:
            for handle in handles:
                handle.remove()
        return prediction, prediction, batch.y

    def ablate(self, data_list: Sequence[Any], family: Sequence[tuple[int, int]]):
        return self._native_forward(data_list, family=family, ablate=True)

    def patch(
        self,
        target: Any,
        donor_transport: Sequence[Any],
        family: Sequence[tuple[int, int]],
    ):
        return self._native_forward(
            [target], family=family, replacements=donor_transport
        )

    def patch_many(
        self,
        targets: Sequence[Any],
        donor_transport: Sequence[Any],
        family: Sequence[tuple[int, int]],
    ):
        return self._native_forward(
            targets, family=family, replacements=donor_transport
        )

    def replacement_batch(
        self,
        capture: BackendCapture,
        indices: Sequence[int],
        *,
        repeat_single: bool = False,
    ) -> tuple[Any, ...]:
        result = []
        for layer in capture.transport:
            rows = layer[[int(value) for value in indices]]
            if repeat_single and rows.shape[0] == 1 and len(indices) > 1:
                rows = rows.expand(len(indices), *rows.shape[1:])
            result.append(rows.reshape(-1, rows.shape[-2], rows.shape[-1]).detach())
        return tuple(result)


def prepare_nar_task(
    config: MethodologyConfig,
    task: CanonicalTask,
    train_seed: int,
    task_overrides: Mapping[str, Any],
    *,
    prepared_task_class: type,
    repository_commit: str,
):
    """Load one fixed-N checkpoint read-only and bind it to the canonical runner."""

    import torch

    from ..methodology.audit import audit_scope, log_summary
    from ..methodology.runner import _model_audits

    del task_overrides
    spec: NarTaskSpec = task.spec
    explicit = None
    for key in (f"{task.name}:{train_seed}", f"{task.name}_seed{train_seed}", task.name):
        if key in config.checkpoints:
            explicit = str(config.checkpoints[key])
            break
    checkpoint = find_checkpoint(spec, train_seed, explicit)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = (spec.model_name, int(spec.width), int(spec.records), int(train_seed))
    actual = (
        str(payload.get("model_name")),
        int(payload.get("width", -1)),
        int(payload.get("N", -1)),
        int(payload.get("seed", -1)),
    )
    if actual != expected:
        raise RuntimeError(f"checkpoint metadata {actual} does not match requested cell {expected}")
    cfg = _training_config(payload, spec)
    model_class = training.build_model_class()
    device = torch.device(
        config.accelerator
        if str(config.accelerator).startswith("cpu") or torch.cuda.is_available()
        else "cpu"
    )
    model = model_class(cfg, spec.model_name, int(spec.width), int(spec.records)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    eval_size = (
        int(config.sizes.discovery_graphs)
        + int(config.sizes.causal_graphs)
        + int(config.sizes.clean_ablation_graphs)
    )
    dataset_seed = (
        int(config.analysis_seed)
        + int(spec.records) * 100_003
        + MODEL_ORDER.index(spec.model_name) * 10_000_019
    )
    eval_ds = NarGraphDataset(cfg, spec.records, eval_size, dataset_seed)
    donor_ds = NarGraphDataset(
        cfg,
        spec.records,
        int(config.sizes.semantic_donor_graphs),
        dataset_seed + 900_000_011,
    )
    runtime = SimpleNamespace(
        model=model,
        eval_ds=eval_ds,
        donor_ds=donor_ds,
        sc=SimpleNamespace(seed=int(train_seed)),
        L=int(model.L),
        H=int(model.H),
        dh=int(model.dh),
        dim_h=int(model.width),
        device=device,
        test_metric=payload.get("heldout", {}),
        val_metric=payload.get("best_validation", {}),
        checks={"num_parameters": int(payload.get("parameters", 0))},
        checkpoint_payload=payload,
    )
    sigma = np.ones(int(spec.records), dtype=np.float64)
    backend = CanonicalNarBackend(model, task, device)
    output_dir = config.root / task.name / f"seed_{int(train_seed)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with audit_scope(f"{task.name}:seed{train_seed}:model") as scope:
        audits = _model_audits(runtime, backend, task, config)
    audits["failures"] = log_summary(
        scope, header=f"{task.name}:seed{train_seed} model audits"
    )
    splits = deterministic_splits(
        len(eval_ds),
        len(donor_ds),
        config.sizes,
        int(config.analysis_seed),
        same_index_space=False,
    )
    donor_pool = NarSemanticDonorPool(
        [
            (graph_id, donor_ds[graph_id])
            for graph_id in splits.semantic_donor_pool
        ],
        records=int(spec.records),
    )
    digest = checkpoint_sha256(checkpoint)
    atomic_json(
        output_dir / "model.json",
        {
            "analysis_version": ANALYSIS_VERSION,
            "repository_commit": repository_commit,
            "task": task.name,
            "backend": task.backend_kind,
            "title": task.title,
            "model_family": spec.model_name,
            "N": int(spec.records),
            "width": int(spec.width),
            "train_seed": int(train_seed),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "checkpoint_selection": "lowest validation loss saved by training frontend",
            "output_representation": "N-way logits",
            "sigma": sigma.tolist(),
            "semantic_payload": (
                "complete query or key-value row with the task-irrelevant field held by "
                "role-aligned donor matching"
            ),
            "semantic_donor_pool": (
                "graph-balanced role-aligned donors: query-to-query or same-key record-to-record"
            ),
            "analysis_sources": "query node and requested memory-record node",
            "structural_payload": "dense RRWP source row/column/self on fixed support",
            "splits": dataclasses.asdict(splits),
            "heldout": payload.get("heldout", {}),
            "best_validation": payload.get("best_validation", {}),
            "canonical_audits": audits,
        },
    )
    return prepared_task_class(
        task,
        runtime,
        backend,
        output_dir,
        checkpoint,
        digest,
        sigma,
        splits,
        donor_pool,
    )


def checkpoint_payloads(
    *,
    training_run_dir: str | Path,
    models: Sequence[str],
    ns: Sequence[int],
    seeds: Sequence[int],
    width: int,
) -> list[dict[str, Any]]:
    """Read every performance checkpoint; no model or GRIT installation is needed."""

    import torch

    rows: list[dict[str, Any]] = []
    for records in ns:
        for model_name in models:
            spec = NarTaskSpec(
                name=task_name(model_name, records),
                title="",
                model_name=str(model_name),
                records=int(records),
                width=int(width),
                training_run_dir=str(training_run_dir),
            )
            for seed in seeds:
                checkpoint = find_checkpoint(spec, int(seed))
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                rows.append(
                    {
                        "model": str(model_name),
                        "N": int(records),
                        "seed": int(seed),
                        "width": int(width),
                        "accuracy": float(payload["heldout"]["accuracy"]),
                        "heldout_loss": float(payload["heldout"]["loss"]),
                        "validation_loss": float(payload["best_validation"]["loss"]),
                        "validation_accuracy": float(
                            payload["best_validation"]["accuracy"]
                        ),
                        "checkpoint": str(checkpoint),
                    }
                )
    return rows


def best_seed_by_validation(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    cells = {(str(row["model"]), int(row["N"])) for row in rows}
    for cell in cells:
        selected = [row for row in rows if (str(row["model"]), int(row["N"])) == cell]
        winner = min(
            selected,
            key=lambda row: (
                float(row["validation_loss"]),
                -float(row["validation_accuracy"]),
                int(row["seed"]),
            ),
        )
        result[cell] = int(winner["seed"])
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
        writer.writerows(rows)


def _save_figure(fig, directory: Path, stem: str, metadata: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            directory / f"{stem}.{suffix}",
            dpi=350,
            bbox_inches="tight",
            facecolor="white",
        )
    atomic_json(directory / f"{stem}.metadata.json", dict(metadata))


def _style():
    import matplotlib as mpl

    return mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.55,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_accuracy(
    rows: Sequence[Mapping[str, Any]],
    *,
    figure_dir: Path,
    models: Sequence[str],
    ns: Sequence[int],
) -> None:
    """NeurIPS-style accuracy curve using every N and seed-level 95% t intervals."""

    import matplotlib.pyplot as plt

    table: list[dict[str, Any]] = []
    critical = 4.302652729911275  # two-sided 95% Student-t, df=2
    with _style():
        fig, ax = plt.subplots(figsize=(6.15, 3.85), constrained_layout=True)
        for model_name in models:
            means, errors = [], []
            for records in ns:
                values = np.asarray(
                    [
                        float(row["accuracy"])
                        for row in rows
                        if str(row["model"]) == str(model_name)
                        and int(row["N"]) == int(records)
                    ],
                    dtype=np.float64,
                )
                mean = float(values.mean())
                half = (
                    float(critical * values.std(ddof=1) / math.sqrt(len(values)))
                    if len(values) >= 2
                    else 0.0
                )
                means.append(mean)
                errors.append(half)
                table.append(
                    {
                        "model": model_name,
                        "N": int(records),
                        "seeds": len(values),
                        "mean_accuracy": mean,
                        "ci95_low": mean - half,
                        "ci95_high": mean + half,
                    }
                )
            ax.errorbar(
                ns,
                means,
                yerr=errors,
                color=MODEL_COLOURS[model_name],
                marker=MODEL_MARKERS[model_name],
                linewidth=1.7,
                markersize=5.2,
                markeredgecolor="white",
                markeredgewidth=0.55,
                capsize=2.5,
                label=MODEL_LABELS[model_name],
            )
        ax.set_xlabel("Number of key–value records, $N$")
        ax.set_ylabel("Held-out accuracy")
        ax.set_title("Associative recall accuracy across memory size")
        ax.set_xticks(list(ns))
        ax.set_ylim(-0.02, 1.02)
        ax.legend(ncol=min(3, len(models)), loc="lower left")
    _save_figure(
        fig,
        figure_dir,
        "01_accuracy_vs_N",
        {
            "uncertainty": "mean and two-sided 95% Student-t interval across training seeds",
            "N_values": list(map(int, ns)),
            "models": list(models),
        },
    )
    plt.close(fig)
    _write_csv(figure_dir.parent / "tables" / "accuracy_summary.csv", table)


def _score_result(results: Mapping[str, Any], model_name: str, records: int, seed: int):
    return results[f"{task_name(model_name, records)}:seed{int(seed)}"]["scores"]


def _scatter_model(
    results: Mapping[str, Any],
    *,
    model_name: str,
    analysis_ns: Sequence[int],
    seeds: Sequence[int],
    figure_dir: Path,
    coordinate: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    columns = len(analysis_ns)
    layer_colours = ("#2F6B9A", "#B33A3A", "#6E6E6E", "#8B6BB3")
    with _style():
        fig, axes = plt.subplots(
            1,
            columns,
            figsize=(4.2 * columns, 4.15),
            constrained_layout=False,
            squeeze=False,
        )
        axes = axes[0]
        all_x, all_y = [], []
        for ax, records in zip(axes, analysis_ns):
            for seed_position, seed in enumerate(seeds):
                scores = _score_result(results, model_name, records, seed)
                values = scores["coordinates"]
                if coordinate == "scores":
                    x = np.asarray(values.normalized_structural)
                    y = np.asarray(values.normalized_semantic)
                    x_interval = (
                        np.asarray(scores["intervals"].low[3]),
                        np.asarray(scores["intervals"].high[3]),
                    )
                    y_interval = (
                        np.asarray(scores["intervals"].low[2]),
                        np.asarray(scores["intervals"].high[2]),
                    )
                else:
                    x = np.asarray(values.selectivity)
                    y = np.asarray(values.joint_sensitivity)
                    x_interval = (
                        np.asarray(scores["intervals"].low[5]),
                        np.asarray(scores["intervals"].high[5]),
                    )
                    y_interval = (
                        np.asarray(scores["intervals"].low[4]),
                        np.asarray(scores["intervals"].high[4]),
                    )
                for layer in range(x.shape[0]):
                    xv, yv = x[layer], y[layer]
                    xlo, xhi = x_interval[0][layer], x_interval[1][layer]
                    ylo, yhi = y_interval[0][layer], y_interval[1][layer]
                    ax.errorbar(
                        xv,
                        yv,
                        xerr=np.vstack((xv - xlo, xhi - xv)),
                        yerr=np.vstack((yv - ylo, yhi - yv)),
                        fmt="none",
                        ecolor=layer_colours[layer % len(layer_colours)],
                        alpha=0.15,
                        linewidth=0.45,
                        zorder=1,
                    )
                    ax.scatter(
                        xv,
                        yv,
                        s=32,
                        marker=SEED_MARKERS[seed_position % len(SEED_MARKERS)],
                        color=layer_colours[layer % len(layer_colours)],
                        edgecolor="white",
                        linewidth=0.45,
                        alpha=0.9,
                        zorder=2,
                    )
                    all_x.extend(xv[np.isfinite(xv)].tolist())
                    all_y.extend(yv[np.isfinite(yv)].tolist())
            ax.set_title(f"$N={int(records)}$")
            if coordinate == "scores":
                ax.set_xlabel(r"Structural score  $S_{str}/\overline{S}_{str}$")
                ax.set_ylabel(r"Semantic score  $S_{sem}/\overline{S}_{sem}$")
            else:
                ax.set_xlabel(
                    r"Selectivity $D_{rel}$  (structural $\leftarrow$ 0 "
                    r"$\rightarrow$ semantic)"
                )
                ax.set_ylabel(r"Joint sensitivity $J$")
                ax.axvline(0, color="#777777", linestyle="--", linewidth=0.8)
        if coordinate == "scores" and all_x and all_y:
            upper = max(max(all_x), max(all_y)) * 1.06
            for ax in axes:
                ax.plot([0, upper], [0, upper], color="#777777", linestyle="--", linewidth=0.8)
                ax.set_xlim(0, upper)
                ax.set_ylim(0, upper)
                ax.set_aspect("equal", adjustable="box")
        layer_handles = [
            Line2D(
                [], [], marker="o", linestyle="none", markersize=5,
                markerfacecolor=layer_colours[layer], markeredgecolor="white",
                label=f"Layer {layer + 1}",
            )
            for layer in range(_score_result(results, model_name, analysis_ns[0], seeds[0])[
                "coordinates"
            ].raw_semantic.shape[0])
        ]
        seed_handles = [
            Line2D(
                [], [], marker=SEED_MARKERS[position], linestyle="none", markersize=5,
                markerfacecolor="#777777", markeredgecolor="white", label=f"Seed {seed}",
            )
            for position, seed in enumerate(seeds)
        ]
        fig.legend(
            handles=[*layer_handles, *seed_handles],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=len(layer_handles) + len(seed_handles),
        )
        fig.suptitle(
            (
                f"{MODEL_LABELS[model_name]}: structural and semantic specialisation"
                if coordinate == "scores"
                else f"{MODEL_LABELS[model_name]}: selectivity and joint sensitivity"
            ),
            fontsize=12,
            y=0.96,
        )
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.22, top=0.84, wspace=0.32)
    stem = (
        f"02_{model_name}_structural_vs_semantic"
        if coordinate == "scores"
        else f"03_{model_name}_Drel_vs_J"
    )
    _save_figure(
        fig,
        figure_dir,
        stem,
        {
            "models": [model_name],
            "N_values": list(map(int, analysis_ns)),
            "seeds": list(map(int, seeds)),
            "point": "one attention head",
            "colour": "layer",
            "shape": "training seed",
            "interval": "95% nested percentile bootstrap",
        },
    )
    plt.close(fig)


def _profile_from_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    finite = [int(row["distance"]) for row in rows if np.isfinite(row["distance"])]
    maximum = max(finite, default=0)
    display = display_bins(tuple(range(maximum + 1)), max_points=14)
    bins = tuple((int(group[0]), int(group[-1])) for group in display.groups)
    labels, estimates, lows, highs = [], [], [], []
    for lower, upper in bins:
        labels.append(str(lower) if lower == upper else f"{lower}-{upper}")
        selected = [
            row
            for row in rows
            if np.isfinite(row["distance"])
            and lower <= int(row["distance"]) <= upper
        ]
        graph_ids = {int(row["graph_id"]) for row in selected}
        pairs = {
            (int(row["graph_id"]), int(row["carrier"]), int(row["source"]))
            for row in selected
        }
        if len(graph_ids) < bootstrap.minimum_graphs or len(pairs) < bootstrap.minimum_pairs:
            estimates.append(np.nan)
            lows.append(np.nan)
            highs.append(np.nan)
            continue
        grouped: dict[tuple[int, int, int, int], list[float]] = {}
        for row in selected:
            key = (
                int(row["seed"]),
                int(row["graph_id"]),
                int(row["source"]),
                int(row["donor"]),
            )
            grouped.setdefault(key, []).append(float(row["F_sens"]))
        observations = [
            Observation(
                seed,
                graph,
                source,
                donor,
                np.asarray([np.sum(values), len(values)], dtype=np.float64),
            )
            for (seed, graph, source, donor), values in grouped.items()
        ]

        def reduce(values):
            return trimmed_mean(
                values[:, 0] / values[:, 1],
                bootstrap.trim_fraction,
                axis=0,
            )

        interval = nested_percentile_interval(
            observations,
            bootstrap,
            graph_reduce=reduce,
        )
        estimates.append(float(interval.estimate))
        lows.append(float(interval.low))
        highs.append(float(interval.high))
    return labels, np.asarray(estimates), np.asarray(lows), np.asarray(highs)


def plot_carriage_overlays(
    results: Mapping[str, Any],
    *,
    best_seeds: Mapping[tuple[str, int], int],
    models: Sequence[str],
    analysis_ns: Sequence[int],
    figure_dir: Path,
    config: MethodologyConfig,
) -> None:
    import matplotlib.pyplot as plt

    table: list[dict[str, Any]] = []
    for records in analysis_ns:
        for channel in ("semantic", "structural"):
            curves: dict[str, tuple[list[str], np.ndarray, np.ndarray, np.ndarray]] = {}
            for model_name in models:
                seed = int(best_seeds[(str(model_name), int(records))])
                carriage = results[
                    f"{task_name(model_name, records)}:seed{seed}"
                ]["carriage"]
                if carriage is None:
                    raise RuntimeError(
                        f"missing carriage result for {model_name}, N={records}, seed={seed}"
                    )
                curves[model_name] = _profile_from_pairs(
                    carriage["channels"][channel]["pairs"],
                    bootstrap=config.bootstrap,
                )
            labels = next(iter(curves.values()))[0]
            if any(value[0] != labels for value in curves.values()):
                raise RuntimeError("NAR model carriage profiles disagree on distance bins")
            positions = np.arange(len(labels))
            with _style():
                fig, ax = plt.subplots(figsize=(6.15, 3.85), constrained_layout=True)
                for model_name in models:
                    _, estimate, low, high = curves[model_name]
                    ax.plot(
                        positions,
                        estimate,
                        color=MODEL_COLOURS[model_name],
                        marker=MODEL_MARKERS[model_name],
                        linewidth=1.7,
                        markersize=5.0,
                        markeredgecolor="white",
                        markeredgewidth=0.5,
                        label=MODEL_LABELS[model_name],
                    )
                    ax.fill_between(
                        positions,
                        low,
                        high,
                        color=MODEL_COLOURS[model_name],
                        alpha=0.16,
                        linewidth=0,
                    )
                    for label, point, lo, hi in zip(labels, estimate, low, high):
                        table.append(
                            {
                                "N": int(records),
                                "channel": channel,
                                "model": model_name,
                                "seed": int(best_seeds[(model_name, int(records))]),
                                "distance_bin": label,
                                "functional_carriage": float(point),
                                "ci95_low": float(lo),
                                "ci95_high": float(hi),
                            }
                        )
                ax.set_xticks(positions, labels)
                ax.set_xlabel("Carrier distance from changed node")
                ax.set_ylabel("Functional carriage")
                ax.set_title(
                    f"{channel.capitalize()} functional carriage ($N={int(records)}$)"
                )
                ax.legend(ncol=min(3, len(models)))
            _save_figure(
                fig,
                figure_dir,
                f"05_{channel}_functional_carriage_N{int(records)}",
                {
                    "N": int(records),
                    "channel": channel,
                    "models": list(models),
                    "selection": "lowest validation-loss seed within each model-by-N cell",
                    "best_seeds": {
                        model: int(best_seeds[(model, int(records))]) for model in models
                    },
                    "uncertainty": "95% nested percentile bootstrap",
                    "maximum_distance_groups": 14,
                    "beneficial_carriage": "not computed",
                },
            )
            plt.close(fig)
    _write_csv(
        figure_dir.parent / "tables" / "functional_carriage_profiles.csv",
        table,
    )


def make_nar_figures(
    *,
    results: Mapping[str, Any],
    performance_rows: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    all_ns: Sequence[int],
    analysis_ns: Sequence[int],
    seeds: Sequence[int],
    best_seeds: Mapping[tuple[str, int], int],
    output_dir: str | Path,
    config: MethodologyConfig,
) -> None:
    """Create the cross-seed/model headline figures; canonical causal figures stay per run."""

    output = Path(output_dir)
    figure_dir = output / "figures"
    plot_accuracy(
        performance_rows,
        figure_dir=figure_dir,
        models=models,
        ns=all_ns,
    )
    for model_name in models:
        _scatter_model(
            results,
            model_name=model_name,
            analysis_ns=analysis_ns,
            seeds=seeds,
            figure_dir=figure_dir,
            coordinate="scores",
        )
        _scatter_model(
            results,
            model_name=model_name,
            analysis_ns=analysis_ns,
            seeds=seeds,
            figure_dir=figure_dir,
            coordinate="coordinates",
        )
    plot_carriage_overlays(
        results,
        best_seeds=best_seeds,
        models=models,
        analysis_ns=analysis_ns,
        figure_dir=figure_dir,
        config=config,
    )
    atomic_json(
        output / "nar_figure_index.json",
        {
            "analysis_version": ANALYSIS_VERSION,
            "figures": sorted(path.name for path in figure_dir.glob("*.png")),
            "tables": sorted(path.name for path in (output / "tables").glob("*.csv")),
            "causal_figures": (
                "See <task>/seed_<seed>/figures for the canonical causal calibration, "
                "coordinate, family endpoint, matched-control, and cumulative-prefix panels."
            ),
            "beneficial_carriage": "not computed",
        },
    )


def _parse_csv_ints(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        result = tuple(int(item) for item in value)
    if not result:
        raise ValueError("expected at least one integer")
    return result


def _parse_csv_strings(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        result = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        result = tuple(str(item).strip() for item in value if str(item).strip())
    if not result:
        raise ValueError("expected at least one value")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "performance", "scores", "causal", "carriage", "figures"),
        default="all",
    )
    parser.add_argument(
        "--drive-root",
        default=training.DEFAULT_DRIVE_ROOT,
        help="parent Drive directory containing the NAR training run",
    )
    parser.add_argument("--training-run-name", default="nar_grit_fixed_n_v3")
    parser.add_argument("--analysis-name", default="canonical_nar_analysis_d128")
    parser.add_argument("--models", default="1hop,2hop,dense")
    parser.add_argument("--all-ns", default="4,8,16,32,64,80")
    parser.add_argument("--analysis-ns", default="4,16,64")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--analysis-width", type=int, default=128)
    parser.add_argument("--discovery-graphs", type=int, default=48)
    parser.add_argument("--causal-graphs", type=int, default=48)
    parser.add_argument("--clean-ablation-graphs", type=int, default=64)
    parser.add_argument("--semantic-donor-graphs", type=int, default=256)
    parser.add_argument("--source-nodes-per-graph", type=int, default=2)
    parser.add_argument("--donors-per-source", type=int, default=8)
    parser.add_argument("--graphs-per-batch", type=int, default=8)
    parser.add_argument("--analysis-seed", type=int, default=31_415)
    parser.add_argument("--bootstrap-seed", type=int, default=17_071)
    parser.add_argument("--activity-floor", type=float, default=0.20)
    parser.add_argument("--tail-fraction", type=float, default=0.20)
    parser.add_argument("--central-fraction", type=float, default=0.20)
    parser.add_argument("--accelerator", default="cuda:0")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--grit-dir", default=training.DEFAULT_GRIT_DIR)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict-audits", action="store_true")
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="engineering smoke configuration; never use for dissertation results",
    )
    return parser


def _build_methodology_config(
    args: argparse.Namespace,
    *,
    tasks: Sequence[str],
    seeds: Sequence[int],
    output_dir: Path,
) -> MethodologyConfig:
    if args.fast_dev_run:
        sizes = RunSizes.smoke()
        execution = ExecutionPolicy(graphs_per_batch=1)
    else:
        sizes = RunSizes(
            discovery_graphs=int(args.discovery_graphs),
            causal_graphs=int(args.causal_graphs),
            clean_ablation_graphs=int(args.clean_ablation_graphs),
            semantic_donor_graphs=int(args.semantic_donor_graphs),
            sources_per_graph=int(args.source_nodes_per_graph),
            donors_per_source=int(args.donors_per_source),
        )
        execution = ExecutionPolicy(graphs_per_batch=int(args.graphs_per_batch))
    return MethodologyConfig(
        output_dir=str(output_dir),
        tasks=tuple(tasks),
        train_seeds=tuple(int(seed) for seed in seeds),
        phases=("scores", "causal", "carriage", "figures"),
        sizes=sizes,
        numerical=NumericalPolicy(),
        bootstrap=BootstrapPolicy(rng_seed=int(args.bootstrap_seed)),
        families=FamilyPolicy(
            activity_floor=float(args.activity_floor),
            tail_fraction=float(args.tail_fraction),
            central_fraction=float(args.central_fraction),
        ),
        execution=execution,
        analysis_seed=int(args.analysis_seed),
        accelerator=str(args.accelerator),
        num_threads=int(args.num_threads),
        figure_overrides={
            "layer_cmap": "coolwarm",
            "semantic_color": "#B33A3A",
            "structural_color": "#2F6B9A",
            "central_color": "#777777",
            "functional_color": "#2F6B9A",
            "inactive_color": "#B8B8B8",
            "max_distance_points": 14,
        },
        skip_install=bool(args.skip_install),
        resume=True,
        force=bool(args.force),
        strict_audits=bool(args.strict_audits),
        compute_beneficial_carriage=False,
    )


def _merge_results(
    base: dict[str, Any],
    update: Mapping[str, Any],
) -> dict[str, Any]:
    for key, value in update.items():
        if key not in base:
            base[key] = value
            continue
        for field in ("scores", "carriage", "causal", "figures"):
            if value.get(field) is not None:
                base[key][field] = value[field]
    return base


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the selected resumable phase and return its in-memory result index."""

    from ..methodology.runner import (
        prepare_task,
        run_carriage,
        run_methodology,
    )

    models = _parse_csv_strings(args.models)
    all_ns = _parse_csv_ints(args.all_ns)
    analysis_ns = _parse_csv_ints(args.analysis_ns)
    seeds = _parse_csv_ints(args.seeds)
    if any(model not in MODEL_ORDER for model in models):
        raise ValueError(f"models must be drawn from {MODEL_ORDER}")
    if not set(analysis_ns).issubset(all_ns):
        raise ValueError("analysis-ns must be a subset of all-ns")
    if len(seeds) != 3 and not args.fast_dev_run:
        raise ValueError("the headline NAR analysis requires exactly three training seeds")
    training_run_dir = Path(args.drive_root) / str(args.training_run_name)
    output_dir = training_run_dir / str(args.analysis_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    performance = checkpoint_payloads(
        training_run_dir=training_run_dir,
        models=models,
        ns=all_ns,
        seeds=seeds,
        width=int(args.analysis_width),
    )
    _write_csv(output_dir / "tables" / "checkpoint_performance.csv", performance)
    best_seeds = best_seed_by_validation(performance)
    atomic_json(
        output_dir / "best_seed_manifest.json",
        {
            f"{model}:N{records}": int(seed)
            for (model, records), seed in sorted(best_seeds.items())
        },
    )
    plot_accuracy(
        performance,
        figure_dir=output_dir / "figures",
        models=models,
        ns=all_ns,
    )
    if args.phase == "performance":
        return {
            "performance": performance,
            "best_seeds": best_seeds,
            "output_dir": str(output_dir),
        }

    tasks = register_nar_tasks(
        models=models,
        analysis_ns=analysis_ns,
        width=int(args.analysis_width),
        training_run_dir=training_run_dir,
    )
    config = _build_methodology_config(
        args,
        tasks=tasks,
        seeds=seeds,
        output_dir=output_dir / "canonical",
    )

    # Canonical figure regeneration uses cached estimators but still replays the registered model
    # boundary and checkpoint audit. ``--skip-install`` avoids reinstalling an existing GRIT tree.
    training.setup_official_grit(
        Path(args.grit_dir),
        install=not bool(args.skip_install),
    )

    results: dict[str, Any] = {}
    if args.phase in {"all", "scores", "causal"}:
        requested = (
            ("scores",)
            if args.phase == "scores"
            else ("scores", "causal")
        )
        results = _merge_results(
            results,
            run_methodology(dataclasses.replace(config, phases=requested)),
        )
        if args.phase in {"scores", "causal"}:
            return {
                "runs": results,
                "performance": performance,
                "best_seeds": best_seeds,
                "output_dir": str(output_dir),
            }

    if args.phase in {"all", "carriage"}:
        for records in analysis_ns:
            for model_name in models:
                name = task_name(model_name, records)
                seed = int(best_seeds[(model_name, int(records))])
                prepared = prepare_task(config, name, seed)
                carriage = run_carriage(prepared, config)
                key = f"{name}:seed{seed}"
                if key not in results:
                    results[key] = {
                        "task": name,
                        "seed": seed,
                        "output_dir": str(prepared.output_dir),
                        "scores": None,
                        "carriage": carriage,
                        "causal": None,
                        "figures": None,
                    }
                else:
                    results[key]["carriage"] = carriage
        if args.phase == "carriage":
            return {
                "runs": results,
                "performance": performance,
                "best_seeds": best_seeds,
                "output_dir": str(output_dir),
            }

    # The canonical figure pass loads the already cached score/causal/carriage artifacts. It does
    # not compute missing carriage fields, so only validation-selected seeds have carriage panels.
    figure_results = run_methodology(
        dataclasses.replace(config, phases=("figures",), force=False)
    )
    results = _merge_results(results, figure_results)
    make_nar_figures(
        results=results,
        performance_rows=performance,
        models=models,
        all_ns=all_ns,
        analysis_ns=analysis_ns,
        seeds=seeds,
        best_seeds=best_seeds,
        output_dir=output_dir,
        config=config,
    )
    atomic_json(
        output_dir / "run_summary.json",
        {
            "analysis_version": ANALYSIS_VERSION,
            "training_run_dir": str(training_run_dir),
            "canonical_output_dir": str(config.root),
            "models": list(models),
            "all_N_values": list(all_ns),
            "full_analysis_N_values": list(analysis_ns),
            "seeds": list(seeds),
            "best_seeds": {
                f"{model}:N{records}": int(seed)
                for (model, records), seed in sorted(best_seeds.items())
            },
            "sources": (
                "query node and requested memory record; source cap applies without "
                "introducing irrelevant padding/control nodes"
            ),
            "semantic_donors": (
                "role-aligned query-to-query or same-key record-to-record nodes; "
                "graph-uniform then node-uniform, minimum degree gap, iid with replacement"
            ),
            "beneficial_carriage": "not computed",
            "distance_display": "at most 14 grouped shortest-path columns",
            "cache_policy": "checkpoint- and protocol-fingerprinted atomic Drive caches",
        },
    )
    return {
        "runs": results,
        "performance": performance,
        "best_seeds": best_seeds,
        "output_dir": str(output_dir),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser_argv = list(argv) if argv is not None else None
    args = build_parser().parse_args(parser_argv)
    return run(args)


__all__ = [
    "ANALYSIS_VERSION",
    "CanonicalNarBackend",
    "MODEL_COLOURS",
    "MODEL_LABELS",
    "MODEL_ORDER",
    "NarSemanticDonorPool",
    "NarTaskSpec",
    "best_seed_by_validation",
    "categorical_accuracy_metric",
    "checkpoint_payloads",
    "find_checkpoint",
    "build_parser",
    "main",
    "make_nar_figures",
    "prepare_nar_task",
    "register_nar_tasks",
    "run",
    "task_name",
]


if __name__ == "__main__":
    main()
