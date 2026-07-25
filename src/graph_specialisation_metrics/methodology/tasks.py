"""Canonical task registrations layered over the checkpoint-compatible GRIT registry."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

import numpy as np

from ..carriage.tasks import GritTaskSpec, get_task as get_grit_task


SEMANTIC_FIELDS = ("x",)
NODE_STRUCTURAL_FIELDS = ("rrwp", "deg", "log_deg", "abs_pe", "pestat_RRWP")
PAIR_STRUCTURAL_FIELDS = (("rrwp_index", "rrwp_val"),)
FIXED_SUPPORT_FIELDS = (
    "edge_index",
    "edge_attr",
    "rrwp_local_edge_index",
    "y",
)


def _mae_per_graph(prediction, target):
    import torch

    return torch.abs(prediction - target).reshape(prediction.shape[0], -1).mean(dim=1)


def _bce_per_graph(prediction, target):
    import torch

    return torch.nn.functional.binary_cross_entropy_with_logits(
        prediction, target.to(prediction.dtype), reduction="none"
    ).reshape(prediction.shape[0], -1).mean(dim=1)


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
            scale = np.std(values, axis=0, ddof=0)
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
    grit: GritTaskSpec
    output: OutputGeometry
    loss_per_graph: Callable
    semantic_fields: tuple[str, ...] = SEMANTIC_FIELDS
    immutable_control_fields: tuple[str, ...] = ()
    node_structural_fields: tuple[str, ...] = NODE_STRUCTURAL_FIELDS
    pair_structural_fields: tuple[tuple[str, str], ...] = PAIR_STRUCTURAL_FIELDS
    fixed_support_fields: tuple[str, ...] = FIXED_SUPPORT_FIELDS
    virtual_node: bool = False
    carrier_policy: str = "real_nodes"
    adapter_version: str = "canonical-grit-v1"
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
        return self.grit.title


TASKS: dict[str, CanonicalTask] = {}


def register(task: CanonicalTask) -> CanonicalTask:
    if task.name in TASKS:
        raise ValueError(f"canonical task {task.name!r} is already registered")
    TASKS[task.name] = task
    return task


def _known_task(name: str) -> CanonicalTask:
    grit = get_grit_task(name)
    if name == "peptides_func":
        output = OutputGeometry("logits", None, "unit")
        loss = _bce_per_graph
    elif name == "peptides_struct":
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
        grit=grit,
        output=output,
        loss_per_graph=loss,
        virtual_node=virtual,
        carrier_policy=("real_nodes_plus_internal_vnode" if virtual else "real_nodes"),
    )


for _name in (
    "zinc",
    "zinc_1hop",
    "zinc_1hop_local",
    "zinc_2hop",
    "zinc_1hop_vnode",
    "zinc_2hop_vnode",
    "qm9_gap_dense",
    "qm9_gap_1hop",
    "qm9_gap_1hop_vnode",
    "peptides_func",
    "peptides_struct",
):
    register(_known_task(_name))


def get_task(name: str, overrides: Mapping[str, Any] | None = None) -> CanonicalTask:
    if name not in TASKS:
        raise KeyError(f"unknown canonical task {name!r}; known: {sorted(TASKS)}")
    task = TASKS[name]
    if not overrides:
        return task
    allowed = {field.name for field in task.__dataclass_fields__.values()} - {"name", "grit"}
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
