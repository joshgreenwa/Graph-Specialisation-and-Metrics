"""Canonical task registrations for every supported model backend."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

import numpy as np

from ..carriage.content import ContentAdapter, FullNodeContentAdapter
from ..carriage.metrics import mae_metric
from ..carriage.tasks import get_task as get_grit_task

SEMANTIC_FIELDS = ("x",)
NODE_STRUCTURAL_FIELDS = ("rrwp", "deg", "log_deg", "abs_pe", "pestat_RRWP")
PAIR_STRUCTURAL_FIELDS = (("rrwp_index", "rrwp_val"),)
DENSE_PAIR_STRUCTURAL_FIELDS: tuple[str, ...] = ()
FIXED_SUPPORT_FIELDS = (
    "edge_index",
    "edge_attr",
    "rrwp_local_edge_index",
    "y",
)

ZINC_FROZEN_KHOP_SUPPORT_TASKS = frozenset(
    {"zinc_2hop", "zinc_1hop_vnode", "zinc_2hop_vnode"}
)
ZINC_FROZEN_KHOP_ADAPTER_VERSION = "canonical-grit-zinc-frozen-khop-support-v2"


def _mae_per_graph(prediction, target):
    import torch

    error = torch.abs(prediction - target).reshape(prediction.shape[0], -1)
    valid = torch.isfinite(error)
    return torch.where(valid, error, torch.zeros_like(error)).sum(dim=1) / valid.sum(
        dim=1
    ).clamp_min(1)


def _bce_per_graph(prediction, target):
    import torch

    target = target.to(prediction.dtype)
    valid = torch.isfinite(target)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    error = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction, safe_target, reduction="none"
    ).reshape(prediction.shape[0], -1)
    valid = valid.reshape(prediction.shape[0], -1)
    return torch.where(valid, error, torch.zeros_like(error)).sum(dim=1) / valid.sum(
        dim=1
    ).clamp_min(1)


@dataclass(frozen=True)
class OutputGeometry:
    """One task's fixed output representation and diagonal scaling."""

    representation: str
    sigma: tuple[float, ...] | None
    sigma_policy: str

    def resolve(self, outputs: int, *, training_targets: np.ndarray | None = None) -> np.ndarray:
        if self.sigma is not None:
            scale = np.asarray(self.sigma, dtype=np.float64).reshape(-1)
        elif self.sigma_policy == "unit":
            scale = np.ones(int(outputs), dtype=np.float64)
        elif self.sigma_policy == "training_target_std":
            if training_targets is None:
                raise ValueError("training_target_std requires training targets")
            values = np.asarray(training_targets, dtype=np.float64).reshape(-1, int(outputs))
            scale = np.nanstd(values, axis=0, ddof=0)
        else:
            raise ValueError(f"unknown sigma policy {self.sigma_policy!r}")
        if scale.shape != (int(outputs),):
            raise ValueError(f"sigma has shape {scale.shape}; expected {(int(outputs),)}")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError(f"all output scales must be finite and positive; got {scale}")
        return scale

    def transform(self, prediction, sigma):
        """Map a prediction tensor/array into registered z-space."""

        if self.representation not in {"evaluation_regression", "logits"}:
            raise ValueError(f"unsupported output representation {self.representation!r}")
        if hasattr(prediction, "new_tensor"):
            scale = prediction.new_tensor(sigma).reshape(1, -1)
        else:
            scale = np.asarray(sigma, dtype=np.float64).reshape(1, -1)
        return prediction.reshape(prediction.shape[0], -1) / scale


@dataclass(frozen=True)
class CanonicalTask:
    """Everything a new task must declare rather than forking the estimators."""

    name: str
    backend_kind: str
    spec: Any
    output: OutputGeometry
    loss_per_graph: Callable
    semantic_fields: tuple[str, ...] = SEMANTIC_FIELDS
    immutable_control_fields: tuple[str, ...] = ()
    node_structural_fields: tuple[str, ...] = NODE_STRUCTURAL_FIELDS
    pair_structural_fields: tuple[tuple[str, str], ...] = PAIR_STRUCTURAL_FIELDS
    dense_pair_structural_fields: tuple[str, ...] = DENSE_PAIR_STRUCTURAL_FIELDS
    fixed_support_fields: tuple[str, ...] = FIXED_SUPPORT_FIELDS
    virtual_node: bool = False
    carrier_policy: str = "real_nodes"
    adapter_version: str = "canonical-grit-v1"
    # Existing node-content tasks draw one shared source set for paired channel inference.
    # GraphBench's edge-semantic extension declares independent edge/node source domains.
    paired_channel_sources: bool = True
    semantic_source_kind: str = "node"
    protocol_extension: str | None = None
    # GraphBench registers coherent carrier aggregation after the PE-refinement
    # experiment; the task-general legacy default remains transport mass.
    raw_score_system: str = "mass"
    bootstrap_seed: int = 17_071
    extra_known_fields: tuple[str, ...] = (
        "num_nodes",
        "batch",
        "ptr",
        "train_mask",
        "val_mask",
        "test_mask",
        "real_node_mask",
    )

    @property
    def title(self) -> str:
        return self.spec.title

    @property
    def content_adapter(self) -> ContentAdapter:
        return self.spec.content_adapter

    @property
    def metric_fn(self) -> Callable:
        return self.spec.metric_fn

    @property
    def grit(self) -> Any:
        """Compatibility alias for older callers; new code should use ``spec``."""

        return self.spec


@dataclass(frozen=True)
class GraphormerTaskSpec:
    """Runtime-only details for an official Hugging Face-compatible Graphormer."""

    name: str
    title: str
    model_id: str
    revision: str | None
    dataset_name: str
    dataset_root: str
    eval_split: str
    donor_split: str
    metric_fn: Callable = staticmethod(mae_metric)
    content_adapter: ContentAdapter = field(default_factory=FullNodeContentAdapter)
    checkpoint_format: str = "auto"


@dataclass(frozen=True)
class GraphBenchTaskSpec:
    """Runtime details for the official-GRIT GraphBench AlgoReas adapter."""

    name: str
    title: str
    graphbench_task: str
    task_type: str
    eval_split: str = "val"
    donor_split: str = "train"
    rrwp_steps: int = 16
    metric_fn: Callable = staticmethod(mae_metric)
    content_adapter: ContentAdapter = field(default_factory=FullNodeContentAdapter)


TASKS: dict[str, CanonicalTask] = {}


def register(task: CanonicalTask) -> CanonicalTask:
    if task.name in TASKS:
        raise ValueError(f"canonical task {task.name!r} is already registered")
    if not callable(task.metric_fn):
        raise ValueError(
            f"canonical task {task.name!r} must register a callable dataset metric"
        )
    if not callable(task.loss_per_graph):
        raise ValueError(
            f"canonical task {task.name!r} must register a callable per-graph loss"
        )
    if task.raw_score_system not in {"mass", "coherent"}:
        raise ValueError(
            f"canonical task {task.name!r} has unknown raw score system "
            f"{task.raw_score_system!r}"
        )
    TASKS[task.name] = task
    return task


def _known_grit_task(name: str) -> CanonicalTask:
    grit = get_grit_task(name)
    if name == "peptides_func":
        output = OutputGeometry("logits", None, "unit")
        loss = _bce_per_graph
    elif name in {"peptides_struct", "peptides_struct_1hop"}:
        output = OutputGeometry("evaluation_regression", None, "training_target_std")
        loss = _mae_per_graph
    else:
        # Scalar regression still registers its geometry explicitly.  Unit scaling avoids
        # claiming cross-task standardisation not required by the protocol.
        output = OutputGeometry("evaluation_regression", (1.0,), "fixed")
        loss = _mae_per_graph
    virtual = name.endswith("_vnode")
    return CanonicalTask(
        name=name,
        backend_kind="grit",
        spec=grit,
        output=output,
        loss_per_graph=loss,
        fixed_support_fields=(
            FIXED_SUPPORT_FIELDS
            + (
                ("pos", "rrwp_attention_edge_index")
                if name.startswith("qm9_")
                else (
                    ("rrwp_attention_edge_index",)
                    if name in ZINC_FROZEN_KHOP_SUPPORT_TASKS
                    else ()
                )
            )
        ),
        virtual_node=virtual,
        carrier_policy=("real_nodes_plus_internal_vnode" if virtual else "real_nodes"),
        adapter_version=(
            ZINC_FROZEN_KHOP_ADAPTER_VERSION
            if name in ZINC_FROZEN_KHOP_SUPPORT_TASKS
            else "canonical-grit-v1"
        ),
    )


for _name in (
    "zinc",
    "zinc_1hop",
    "zinc_1hop_local",
    "zinc_1hop_localrrwp",
    "zinc_2hop",
    "zinc_1hop_vnode",
    "zinc_2hop_vnode",
    "qm9_gap_dense",
    "qm9_gap_1hop",
    "qm9_gap_1hop_local",
    "qm9_gap_2hop",
    "qm9_gap_1hop_vnode",
    "qm9_gap_2hop_vnode",
    "peptides_func",
    "peptides_struct",
    "peptides_struct_1hop",
):
    register(_known_grit_task(_name))


register(
    CanonicalTask(
        name="graphormer_pcqm4mv2",
        backend_kind="graphormer",
        spec=GraphormerTaskSpec(
            name="graphormer_pcqm4mv2",
            title="Official Graphormer PCQM4Mv2",
            model_id="clefourrier/graphormer-base-pcqm4mv2",
            revision="refs/pr/4",
            dataset_name="pcqm4mv2",
            dataset_root="/content/pcqm4mv2",
            eval_split="valid",
            donor_split="train",
        ),
        output=OutputGeometry("evaluation_regression", (1.0,), "fixed"),
        loss_per_graph=_mae_per_graph,
        semantic_fields=("x",),
        node_structural_fields=("in_degree", "out_degree"),
        pair_structural_fields=(),
        dense_pair_structural_fields=("spatial_pos", "attn_edge_type", "input_edges"),
        fixed_support_fields=("edge_index", "edge_attr", "attn_bias", "y", "smiles"),
        virtual_node=False,
        carrier_policy="real_nodes_plus_graph_token",
        adapter_version="canonical-graphormer-hf-v1",
        extra_known_fields=("num_nodes",),
    )
)


def _matching_f1(logits, target) -> float:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    valid = np.isfinite(target)
    prediction = logits[valid] >= 0.0
    truth = target[valid] >= 0.5
    true_positive = float(np.sum(prediction & truth))
    false_positive = float(np.sum(prediction & ~truth))
    false_negative = float(np.sum(~prediction & truth))
    precision = true_positive / max(1.0, true_positive + false_positive)
    recall = true_positive / max(1.0, true_positive + false_negative)
    return 2.0 * precision * recall / max(1.0e-12, precision + recall)


for _name, _title, _task_type, _metric in (
    (
        "graphbench_bipartite_matching_hard",
        "GraphBench Bipartite Matching (hard, n=16)",
        "edge_binary",
        _matching_f1,
    ),
    (
        "graphbench_flow_hard",
        "GraphBench Maximum Flow (hard, n=16)",
        "graph_regression",
        mae_metric,
    ),
):
    _graphbench_name = _name.removeprefix("graphbench_")
    register(
        CanonicalTask(
            name=_name,
            backend_kind="graphbench_grit",
            spec=GraphBenchTaskSpec(
                name=_name,
                title=_title,
                graphbench_task=_graphbench_name,
                task_type=_task_type,
                metric_fn=_metric,
            ),
            output=OutputGeometry(
                "logits" if _task_type == "edge_binary" else "evaluation_regression",
                None,
                "unit" if _task_type == "edge_binary" else "training_target_std",
            ),
            # The GraphBench backend supplies graph-aware padded BCE / normalized MSE.
            loss_per_graph=_bce_per_graph if _task_type == "edge_binary" else _mae_per_graph,
            semantic_fields=("edge_value",),
            immutable_control_fields=("node_type",),
            node_structural_fields=(),
            pair_structural_fields=(),
            dense_pair_structural_fields=("rrwp",),
            fixed_support_fields=("edge_index", "target", "spd", "rwse"),
            carrier_policy=(
                "readout_edges" if _task_type == "edge_binary" else "real_nodes"
            ),
            adapter_version="graphbench-official-grit-complete-pe-coherent-v2",
            paired_channel_sources=False,
            semantic_source_kind="edge",
            protocol_extension="graphbench-complete-pe-coherent-v2",
            raw_score_system="coherent",
            extra_known_fields=("num_nodes", "task_type"),
        )
    )


def get_task(name: str, overrides: Mapping[str, Any] | None = None) -> CanonicalTask:
    if name not in TASKS:
        raise KeyError(f"unknown canonical task {name!r}; known: {sorted(TASKS)}")
    task = TASKS[name]
    if not overrides:
        return task
    allowed = {
        field.name for field in task.__dataclass_fields__.values()
    } - {"name", "backend_kind", "spec"}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"unknown task override fields for {name!r}: {unknown}")
    return replace(task, **dict(overrides))


def training_target_matrix(loader) -> np.ndarray:
    """Collect training labels once for a registered multi-target output scale."""

    rows: list[np.ndarray] = []
    for batch in loader:
        target = batch.y.detach().cpu().numpy()
        rows.append(target.reshape(target.shape[0], -1))
    if not rows:
        raise ValueError("training loader contains no targets")
    return np.concatenate(rows, axis=0)
