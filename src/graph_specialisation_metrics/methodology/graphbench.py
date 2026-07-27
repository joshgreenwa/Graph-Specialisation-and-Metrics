"""Official-GRIT adapter for the compact GraphBench AlgoReas checkpoints.

This module deliberately reuses the training runner's data conversion, PE cache, official GRIT
layers, and prediction heads.  It adds only the task boundary needed by the canonical donor-swap
estimators: weighted edges are semantic content, while the fixed-topology RRWP footprint is the
structural channel.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from .audit import audit_check, within_tolerance
from .backend import BackendCapture, CleanJacobians
from .protocol import stable_hash
from .sampling import DonorEvent


def _load_module(path: str | Path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"GraphBench runner does not exist: {resolved}")
    os.environ.setdefault("PROJECT_ROOT", str(resolved.parents[1]))
    name = f"_canonical_graphbench_runner_{stable_hash(str(resolved), length=12)}"
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import GraphBench runner from {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def default_runner_path() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "graphbench-algoreas-hpc" / "bin" / "algoreas_hpc.py",
        *(
            parent / "graphbench-algoreas-hpc" / "bin" / "algoreas_hpc.py"
            for parent in here.parents
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("cannot locate graphbench-algoreas-hpc/bin/algoreas_hpc.py")


def _graphbench_version() -> str:
    for distribution in ("graphbench-lib", "graphbench"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def _replace_graph(graph: Any, **changes: Any) -> Any:
    if dataclasses.is_dataclass(graph):
        return dataclasses.replace(graph, **changes)
    values = {
        name: getattr(graph, name)
        for name in (
            "node_type",
            "edge_index",
            "edge_value",
            "target",
            "task_type",
            "num_nodes",
            "spd",
            "rwse",
            "rrwp",
            "degree_override",
        )
        if hasattr(graph, name)
    }
    values.update(changes)
    return type(graph)(**values)


def _clone_graph(graph: Any) -> Any:
    import torch

    changes = {}
    for name in (
        "node_type",
        "edge_index",
        "edge_value",
        "target",
        "spd",
        "rwse",
        "rrwp",
        "degree_override",
    ):
        value = getattr(graph, name, None)
        if isinstance(value, torch.Tensor):
            changes[name] = value.clone()
    return _replace_graph(graph, **changes)


def edge_units(graph: Any) -> tuple[tuple[int, ...], ...]:
    """Return directed flow edges or reciprocal matching-edge units."""

    edge = graph.edge_index.detach().cpu().numpy()
    if graph.task_type != "edge_binary":
        return tuple((position,) for position in range(int(edge.shape[1])))
    by_coordinate: dict[tuple[int, int], list[int]] = {}
    for position, (source, target) in enumerate(edge.T):
        by_coordinate.setdefault((int(source), int(target)), []).append(int(position))
    used: set[int] = set()
    units: list[tuple[int, ...]] = []
    for position, (source, target) in enumerate(edge.T):
        if position in used:
            continue
        reciprocal = [
            value
            for value in by_coordinate.get((int(target), int(source)), ())
            if value not in used and value != position
        ]
        unit = (position, reciprocal[0]) if reciprocal else (position,)
        used.update(unit)
        units.append(tuple(sorted(unit)))
    return tuple(units)


def _edge_endpoints(graph: Any, source: int) -> tuple[int, int]:
    edge = graph.edge_index
    return int(edge[0, int(source)]), int(edge[1, int(source)])


def _graph_degrees(graph: Any) -> np.ndarray:
    """Match the training collator's binary symmetrized adjacency degree."""

    neighbours = [set() for _ in range(int(graph.num_nodes))]
    for left, right in graph.edge_index.detach().cpu().numpy().T:
        left, right = int(left), int(right)
        neighbours[left].add(right)
        neighbours[right].add(left)
    return np.asarray([len(row) for row in neighbours], dtype=np.int64)


def _degree_signature(graph: Any, source: int) -> tuple[int, int]:
    degree = _graph_degrees(graph)
    left, right = _edge_endpoints(graph, source)
    signature = (int(degree[left]), int(degree[right]))
    return tuple(sorted(signature)) if graph.task_type == "edge_binary" else signature


def semantic_edge_swap(graph: Any, source: int, donor_value: float) -> Any:
    """Replace one ordered flow edge or one reciprocal matching-edge unit."""

    out = _clone_graph(graph)
    unit_by_member = {
        member: unit for unit in edge_units(graph) for member in unit
    }
    unit = unit_by_member.get(int(source), (int(source),))
    out.edge_value[list(unit)] = out.edge_value.new_tensor(float(donor_value))
    return out


def structural_rrwp_swap(graph: Any, source: int, donor: int) -> Any:
    from .interventions import dense_pair_donor_swap

    out = _clone_graph(graph)
    if out.rrwp is None:
        raise RuntimeError("GraphBench structural intervention requires cached RRWP")
    out.rrwp = dense_pair_donor_swap(graph.rrwp, int(source), int(donor))
    return out


def _model_degree(graph: Any):
    """Return the degree vector presented to official GRIT for this graph."""

    import torch

    override = getattr(graph, "degree_override", None)
    if override is not None:
        return override.clone()
    return torch.as_tensor(
        _graph_degrees(graph),
        dtype=torch.float32,
        device=graph.edge_index.device,
    )


def _set_degree_override(graph: Any, degree: Any) -> Any:
    """Set the analysis-only degree field without requiring it on lightweight test graphs."""

    out = _clone_graph(graph)
    if dataclasses.is_dataclass(out) and "degree_override" in {
        field.name for field in dataclasses.fields(out)
    }:
        return dataclasses.replace(out, degree_override=degree.clone())
    setattr(out, "degree_override", degree.clone())
    return out


def structural_pe_intervention(
    graph: Any,
    source: int,
    donor: int,
    *,
    complete_pe: bool,
    transpose: bool,
    rrwp_steps: int | None = None,
) -> Any:
    """Apply one of the four registered RRWP/complete-PE structural interventions.

    ``transpose=False`` copies the donor's incident RRWP role onto the source.  ``transpose=True``
    applies the same source/donor permutation to both RRWP axes.  Complete-PE variants apply the
    corresponding copy/permutation to the explicit degree channel; ``log_deg`` remains exactly
    derived by the official adapter from that degree vector.
    """

    import torch

    source, donor = int(source), int(donor)
    n = int(graph.num_nodes)
    if not 0 <= source < n or not 0 <= donor < n:
        raise IndexError("source/donor is outside the graph")
    if graph.rrwp is None:
        raise RuntimeError("GraphBench structural intervention requires cached RRWP")
    out = _clone_graph(graph)
    clean_rrwp = graph.rrwp
    steps = (
        int(clean_rrwp.shape[-1])
        if rrwp_steps is None
        else int(rrwp_steps)
    )
    if steps < 1 or steps > int(clean_rrwp.shape[-1]):
        raise ValueError(
            f"rrwp_steps={steps} is outside cached width {int(clean_rrwp.shape[-1])}"
        )
    visible_rrwp = clean_rrwp[..., :steps]
    if transpose:
        permutation = torch.arange(n, device=visible_rrwp.device)
        permutation[source], permutation[donor] = (
            permutation[donor].clone(),
            permutation[source].clone(),
        )
        changed_visible = visible_rrwp.index_select(0, permutation).index_select(
            1, permutation
        )
    else:
        from .interventions import dense_pair_donor_swap

        changed_visible = dense_pair_donor_swap(visible_rrwp, source, donor)
    out.rrwp = clean_rrwp.clone()
    out.rrwp[..., :steps] = changed_visible
    if complete_pe:
        degree = _model_degree(graph)
        if transpose:
            changed_degree = degree.clone()
            changed_degree[source], changed_degree[donor] = (
                degree[donor].clone(),
                degree[source].clone(),
            )
        else:
            changed_degree = degree.clone()
            changed_degree[source] = degree[donor]
        out = _set_degree_override(out, changed_degree)
        out.rrwp = clean_rrwp.clone()
        out.rrwp[..., :steps] = changed_visible
    return out


def verify_structural_pe_intervention(
    base: Any,
    event: Any,
    source: int,
    donor: int,
    *,
    complete_pe: bool,
    transpose: bool,
    rrwp_steps: int | None = None,
) -> None:
    """Hard integrity audit for the complete declared GraphBench input boundary."""

    import torch

    expected = structural_pe_intervention(
        base,
        source,
        donor,
        complete_pe=complete_pe,
        transpose=transpose,
        rrwp_steps=rrwp_steps,
    )
    for name in (
        "edge_value",
        "node_type",
        "edge_index",
        "target",
        "spd",
        "rwse",
        "rrwp",
        "degree_override",
    ):
        left, right = getattr(expected, name, None), getattr(event, name, None)
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
                raise RuntimeError(
                    f"GraphBench PE intervention has invalid field {name!r}"
                )
        elif left != right:
            raise RuntimeError(f"GraphBench PE intervention has invalid field {name!r}")


def verify_semantic_edge_swap(
    base: Any,
    event: Any,
    source: int,
    donor_value: float,
) -> None:
    import torch

    expected = semantic_edge_swap(base, source, donor_value)
    for name in ("edge_value", "node_type", "edge_index", "target", "spd", "rwse", "rrwp"):
        left, right = getattr(expected, name, None), getattr(event, name, None)
        if isinstance(left, torch.Tensor):
            if not torch.equal(left, right):
                raise RuntimeError(f"GraphBench semantic swap changed undeclared field {name!r}")
        elif left != right:
            raise RuntimeError(f"GraphBench semantic swap changed undeclared field {name!r}")


def verify_structural_rrwp_swap(base: Any, event: Any, source: int, donor: int) -> None:
    import torch

    expected = structural_rrwp_swap(base, source, donor)
    for name in ("edge_value", "node_type", "edge_index", "target", "spd", "rwse", "rrwp"):
        left, right = getattr(expected, name, None), getattr(event, name, None)
        if isinstance(left, torch.Tensor):
            if not torch.equal(left, right):
                raise RuntimeError(f"GraphBench structural swap has invalid field {name!r}")
        elif left != right:
            raise RuntimeError(f"GraphBench structural swap has invalid field {name!r}")


@dataclass(frozen=True)
class EdgeDonor:
    graph_id: int
    edge: int
    endpoints: tuple[int, int]
    degree_signature: tuple[int, int]
    value: float


@dataclass(frozen=True)
class GraphBenchDonorEvent(DonorEvent):
    source_kind: str = "edge"
    source_endpoints: tuple[int, int] | None = None
    donor_endpoints: tuple[int, int] | None = None
    source_degree_signature: tuple[int, int] | None = None
    donor_degree_signature: tuple[int, int] | None = None


class GraphBenchEdgeDonorPool:
    """Graph-balanced, degree-signature-matched edge donor law."""

    def __init__(self, donor_graphs: Sequence[tuple[int, Any]]):
        self.by_graph: dict[int, tuple[EdgeDonor, ...]] = {}
        for graph_id, graph in donor_graphs:
            rows = []
            for unit in edge_units(graph):
                edge = int(unit[0])
                rows.append(
                    EdgeDonor(
                        graph_id=int(graph_id),
                        edge=edge,
                        endpoints=_edge_endpoints(graph, edge),
                        degree_signature=_degree_signature(graph, edge),
                        value=float(graph.edge_value[edge]),
                    )
                )
            self.by_graph[int(graph_id)] = tuple(rows)
        if not self.by_graph:
            raise ValueError("GraphBench semantic edge donor pool is empty")

    def draw(
        self,
        graph: Any,
        source: int,
        count: int,
        rng: np.random.Generator,
    ) -> tuple[EdgeDonor, ...]:
        source_value = float(graph.edge_value[int(source)])
        signature = _degree_signature(graph, int(source))
        candidates = [
            donor
            for rows in self.by_graph.values()
            for donor in rows
            if not np.isclose(donor.value, source_value, rtol=0.0, atol=0.0)
        ]
        if not candidates:
            return ()
        minimum = min(
            abs(donor.degree_signature[0] - signature[0])
            + abs(donor.degree_signature[1] - signature[1])
            for donor in candidates
        )
        eligible: dict[int, list[EdgeDonor]] = {}
        for donor in candidates:
            gap = abs(donor.degree_signature[0] - signature[0]) + abs(
                donor.degree_signature[1] - signature[1]
            )
            if gap == minimum:
                eligible.setdefault(donor.graph_id, []).append(donor)
        graph_ids = np.asarray(sorted(eligible), dtype=np.int64)
        result = []
        for _ in range(int(count)):
            graph_id = int(rng.choice(graph_ids))
            rows = eligible[graph_id]
            result.append(rows[int(rng.integers(0, len(rows)))])
        return tuple(result)


def _rrwp_footprints(graph: Any) -> tuple[bytes, ...]:
    rrwp = graph.rrwp.detach().cpu().numpy()
    return tuple(
        b"\x1f".join(
            (
                np.ascontiguousarray(rrwp[node, :]).tobytes(),
                np.ascontiguousarray(rrwp[:, node]).tobytes(),
            )
        )
        for node in range(int(graph.num_nodes))
    )


def build_graphbench_channel_events(
    base: Any,
    *,
    graph_id: int,
    source: int,
    channel: str,
    stage: str,
    donors: int,
    rng: np.random.Generator,
    semantic_pool: GraphBenchEdgeDonorPool,
) -> tuple[list[Any], list[DonorEvent]]:
    from .sampling import draw_structural_donors

    variants: list[Any] = []
    records: list[DonorEvent] = []
    if channel == "semantic":
        source_signature = _degree_signature(base, source)
        source_endpoints = _edge_endpoints(base, source)
        for draw, donor in enumerate(
            semantic_pool.draw(base, source, int(donors), rng)
        ):
            event = semantic_edge_swap(base, source, donor.value)
            verify_semantic_edge_swap(base, event, source, donor.value)
            gap = sum(
                abs(left - right)
                for left, right in zip(source_signature, donor.degree_signature)
            )
            variants.append(event)
            records.append(
                GraphBenchDonorEvent(
                    channel=channel,
                    stage=stage,
                    graph_id=int(graph_id),
                    source=int(source),
                    donor_graph_id=int(donor.graph_id),
                    donor_node=int(donor.edge),
                    source_degree=sum(source_signature),
                    donor_degree=sum(donor.degree_signature),
                    degree_gap=int(gap),
                    dose=abs(float(base.edge_value[source]) - float(donor.value)),
                    payload_fingerprint=stable_hash({"edge_value": float(donor.value)}),
                    draw=int(draw),
                    source_kind="edge",
                    source_endpoints=source_endpoints,
                    donor_endpoints=donor.endpoints,
                    source_degree_signature=source_signature,
                    donor_degree_signature=donor.degree_signature,
                )
            )
        return variants, records
    if channel != "structural":
        raise ValueError(f"unknown GraphBench channel {channel!r}")
    footprints = _rrwp_footprints(base)
    degrees = _graph_degrees(base)
    selected = draw_structural_donors(
        footprints,
        degrees,
        int(source),
        int(donors),
        rng,
        equal=lambda left, right: left == right,
    )
    for draw, donor in enumerate(selected):
        donor = int(donor)
        event = structural_rrwp_swap(base, source, donor)
        verify_structural_rrwp_swap(base, event, source, donor)
        delta = event.rrwp.float() - base.rrwp.float()
        variants.append(event)
        records.append(
            DonorEvent(
                channel=channel,
                stage=stage,
                graph_id=int(graph_id),
                source=int(source),
                donor_graph_id=int(graph_id),
                donor_node=donor,
                source_degree=int(degrees[source]),
                donor_degree=int(degrees[donor]),
                degree_gap=abs(int(degrees[source]) - int(degrees[donor])),
                dose=float(delta.square().mean().sqrt()),
                payload_fingerprint=stable_hash(
                    {"structural_footprint": footprints[donor].hex()}
                ),
                draw=int(draw),
            )
        )
    return variants, records


@dataclass
class GraphBenchRuntime:
    runner: Any
    model: Any
    eval_ds: Any
    donor_ds: Any
    splits: Mapping[str, Any]
    device: Any
    seed: int
    task_name: str
    target_stats: Mapping[str, float] | None
    pos_weight: Any
    checkpoint: Mapping[str, Any]
    cfg: Any
    checks: dict[str, Any]
    val_metric: float | None
    test_metric: float | None = None

    def __post_init__(self) -> None:
        self.sc = SimpleNamespace(seed=int(self.seed))
        self.L = len(self.model.layers)
        self.H = int(self.cfg.heads)
        self.dim_h = int(self.cfg.hidden_dim)
        self.dh = self.dim_h // self.H


def build_graphbench_runtime(
    task: Any,
    *,
    checkpoint_path: str | Path,
    train_seed: int,
    accelerator: str,
    overrides: Mapping[str, Any],
    jacobian_output_chunk: int,
) -> tuple[GraphBenchRuntime, str]:
    import torch

    runner = _load_module(overrides.get("runner_path") or default_runner_path())
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint must be a mapping: {checkpoint_path}")
    signature = checkpoint.get("run_signature", {})
    signature_config = signature.get("config", {}) if isinstance(signature, Mapping) else {}
    allowed = {field.name for field in dataclasses.fields(runner.ScreenConfig)}
    config_values = {
        key: value
        for key, value in dict(signature_config).items()
        if key in allowed
    }
    config_values["seed"] = int(train_seed)
    for name in (
        "split_seed",
        "train_size",
        "val_size",
        "test_size",
        "train_node_size",
        "val_node_size",
        "test_node_size",
    ):
        if name in overrides:
            config_values[name] = overrides[name]
    cfg = runner.ScreenConfig(**config_values)
    expected_task = task.spec.graphbench_task
    if isinstance(signature, Mapping):
        for field, expected in (
            ("task", expected_task),
            ("model", "grit"),
            ("model_backend", "official"),
        ):
            observed = signature.get(field)
            if observed is not None and observed != expected:
                raise RuntimeError(
                    f"checkpoint {field}={observed!r}; expected {expected!r}"
                )
    device = torch.device(
        "cuda" if accelerator == "auto" and torch.cuda.is_available() else accelerator
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GraphBench analysis requested CUDA but CUDA is unavailable")
    dataset_root = Path(
        overrides.get(
            "dataset_root",
            os.environ.get(
                "GRAPHBENCH_DATASET_ROOT",
                "/rds/user/jgg45/hpc-work/graphbench-algoreas/datasets",
            ),
        )
    )
    pe_root = Path(
        overrides.get(
            "pe_cache_root",
            os.environ.get(
                "GRAPHBENCH_PE_CACHE_ROOT",
                "/rds/user/jgg45/hpc-work/graphbench-algoreas/pe_cache",
            ),
        )
    )
    splits = runner.load_official_graphbench_task(
        dataset_root,
        expected_task,
        cfg,
        force_reload=bool(overrides.get("force_reload_data", False)),
        require_cache=bool(overrides.get("require_subset_cache", True)),
        log=print,
    )
    splits = runner.attach_or_build_pe_cache(
        splits,
        pe_root,
        cfg,
        namespace=str(overrides.get("pe_cache_namespace", "base_40k4k4k_n64")),
        dtype_name=str(overrides.get("pe_cache_dtype", "float32")),
        pe_workers=int(overrides.get("pe_workers", 1)),
        pe_save_every=int(overrides.get("pe_save_every", 500)),
        force_recompute=False,
        build_missing=bool(overrides.get("build_missing_pe_cache", False)),
        require_present=bool(overrides.get("require_pe_cache", True)),
        log=print,
    )
    model = runner.build_model("grit", cfg, backend="official")
    grit_root = runner.resolve_external_repo_path("GRIT_ROOT", ("GRIT",))
    expected_grit_commit = str(
        overrides.get(
            "expected_grit_commit",
            "6c988ea600a606fbb49a2246c64a2d37396b3ab5",
        )
    )
    try:
        observed_grit_commit = subprocess.run(
            ["git", "-C", str(grit_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"official GRIT source is not a readable git checkout: {grit_root}"
        ) from error
    if observed_grit_commit != expected_grit_commit:
        raise RuntimeError(
            f"official GRIT commit {observed_grit_commit} does not match the registered "
            f"{expected_grit_commit}"
        )
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint state did not match official GRIT exactly: "
            f"missing={missing}, unexpected={unexpected}"
        )
    parameter_count = int(runner.count_parameters(model))
    checkpoint_parameters = checkpoint.get("trainable_parameters")
    if (
        checkpoint_parameters is not None
        and int(checkpoint_parameters) != parameter_count
    ):
        raise RuntimeError(
            f"checkpoint records {int(checkpoint_parameters)} trainable parameters, "
            f"but the official GRIT reconstruction has {parameter_count}"
        )
    model.to(device).eval()
    target_stats = checkpoint.get("target_stats")
    if target_stats is None:
        target_stats = runner.compute_target_stats(splits["train"])
    pos_weight = runner.compute_pos_weight(splits["train"])
    watch_size = int(getattr(cfg, "val_watch_size", len(splits["val"])))
    watch_graphs = runner.deterministic_subset(
        splits["val"],
        min(watch_size, len(splits["val"])),
        int(cfg.split_seed) + 4099,
    )
    watch = runner.OfficialGraphDataset(
        watch_graphs, task_name=expected_task, split="val_watch"
    )
    print(
        f"[graphbench] reproducing checkpoint watch metric "
        f"task={expected_task} seed={train_seed} graphs={len(watch)}",
        flush=True,
    )
    metrics, _ = runner.evaluate_model(
        model,
        watch,
        int(runner.eval_batch_size_for("grit", cfg)),
        device,
        False,
        int(train_seed),
        pos_weight,
        target_stats,
    )
    print(
        f"[graphbench] reproduced watch primary={float(metrics['primary']):.8g}",
        flush=True,
    )
    checkpoint_metric = checkpoint.get("val_metrics", {}).get("primary")
    if checkpoint_metric is not None:
        within_tolerance(
            abs(float(metrics["primary"]) - float(checkpoint_metric)),
            float(overrides.get("metric_reproduction_tolerance", 5.0e-6)),
            "graphbench.checkpoint_metric_reproduction",
            "checkpoint watch-metric reproduction error",
            context={
                "task": expected_task,
                "seed": int(train_seed),
                "reproduced": float(metrics["primary"]),
                "checkpoint": float(checkpoint_metric),
            },
        )
    runtime = GraphBenchRuntime(
        runner=runner,
        model=model,
        eval_ds=splits[str(overrides.get("eval_split", task.spec.eval_split))],
        donor_ds=splits[str(overrides.get("donor_split", task.spec.donor_split))],
        splits=splits,
        device=device,
        seed=int(train_seed),
        task_name=expected_task,
        target_stats=target_stats,
        pos_weight=pos_weight,
        checkpoint=checkpoint,
        cfg=cfg,
        checks={
            "num_parameters": parameter_count,
            "model_geometry": {
                "layers": len(model.layers),
                "heads": int(model.layers[0].attention.num_heads),
                "hidden_width": int(cfg.hidden_dim),
            },
            "metric_reproduction": dict(metrics),
            "jacobian_output_chunk": int(jacobian_output_chunk),
            "official_grit_root": str(grit_root),
            "official_grit_commit": observed_grit_commit,
            "graphbench_package_version": _graphbench_version(),
        },
        val_metric=float(metrics["primary"]),
    )
    return runtime, str(checkpoint_path)


class GraphBenchGritBackend:
    """Canonical transport, readout-carriage, ablation, and patch sites."""

    def __init__(
        self,
        runtime: GraphBenchRuntime,
        task: Any,
        sigma: Sequence[float],
        *,
        jacobian_output_chunk: int = 8,
    ):
        import torch

        self.runtime = runtime
        self.model = runtime.model
        self.runner = runtime.runner
        self.task = task
        self.sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
        self.jacobian_output_chunk = max(1, int(jacobian_output_chunk))
        self.task_type = str(task.spec.task_type)
        self.output_width = (
            max(
                int(runtime.eval_ds[index].edge_index.shape[1])
                for index in range(len(runtime.eval_ds))
            )
            if self.task_type == "edge_binary"
            else 1
        )
        self.device = runtime.device
        self._torch = torch
        # Official GRIT currently reaches PyTorch/PyG scatter kernels whose backwards are not
        # always compatible with ``is_grads_batched=True`` under the pinned PyTorch 2.2 HPC
        # environment.  Probe the faster engine once, then retain an exact sequential-VJP
        # fallback for every later graph if vmap reports an unsupported operator.
        self._batched_vjp_supported: bool | None = None
        self._jacobian_fallback_reason: str | None = None
        # Scores and carriage share the discovery graphs within one worker. Retaining detached
        # clean Jacobians avoids paying the sequential fallback twice while keeping every
        # checkpoint/task/seed boundary isolated in its own backend instance.
        self._clean_jacobian_cache: dict[int, CleanJacobians] = {}
        self._clean_jacobian_cache_hits = 0

    @property
    def geometry(self) -> dict[str, int]:
        return {
            "layers": int(self.runtime.L),
            "heads": int(self.runtime.H),
            "head_width": int(self.runtime.dh),
            "hidden_width": int(self.runtime.dim_h),
            "outputs": int(self.output_width),
        }

    @property
    def special_carrier_labels(self) -> tuple[str, ...]:
        return ()

    @property
    def exhaustive_source_channels(self) -> tuple[str, ...]:
        return ("structural",)

    def eligible_sources(self, data: Any, channel: str = "structural") -> np.ndarray:
        if channel == "semantic":
            return np.asarray([unit[0] for unit in edge_units(data)], dtype=np.int64)
        return np.arange(int(data.num_nodes), dtype=np.int64)

    def event_pair_cost(self, data: Any, sources: int, donors: int) -> int:
        replicas = 1 + int(sources) * int(donors)
        return replicas * int(data.num_nodes) ** 2

    def declared_noop_variants(self, data: Any) -> tuple[Any, Any]:
        semantic = semantic_edge_swap(data, edge_units(data)[0][0], float(data.edge_value[edge_units(data)[0][0]]))
        structural = structural_rrwp_swap(data, 0, 0)
        return semantic, structural

    def _raw_prediction(self, native):
        if self.task_type == "graph_regression":
            return self.runner.denormalize_graph_target(
                native, self.runtime.target_stats
            )
        return native

    def _pad_rows(self, rows: Sequence[Any], *, fill: float = 0.0):
        torch = self._torch
        output = rows[0].new_full((len(rows), self.output_width), float(fill))
        for index, row in enumerate(rows):
            output[index, : int(row.numel())] = row.reshape(-1)
        return output

    def _z(self, prediction):
        scale = prediction.new_tensor(self.sigma)
        if scale.numel() == 1:
            scale = scale.expand(prediction.shape[-1])
        return prediction / scale.reshape(1, -1)

    def _forward_capture(self, data_list: Sequence[Any], *, require_grad: bool):
        torch = self._torch
        batch = self.runner.collate_graphs(list(data_list)).to(self.device)
        transport: list[Any] = [None] * int(self.runtime.L)
        final: dict[str, Any] = {}
        attention: list[tuple[Any, Any, Any]] = []
        handles = []

        def make_attention_hook(layer: int):
            def hook(_module, inputs, output):
                pyg = inputs[0]
                routed = output[0] if isinstance(output, tuple) else output
                if routed.ndim == 2:
                    routed = routed.reshape(routed.shape[0], self.runtime.H, self.runtime.dh)
                transport[layer] = routed
                if getattr(pyg, "attn", None) is not None:
                    attention.append(
                        (
                            layer,
                            pyg.edge_index.detach(),
                            pyg.attn.detach(),
                        )
                    )

            return hook

        for layer, module in enumerate(self.model.layers):
            handles.append(module.attention.register_forward_hook(make_attention_hook(layer)))

        def final_layer_hook(_module, _inputs, output):
            final["nodes"] = output.x
            final["edge_attr"] = output.edge_attr
            final["pyg"] = output

        handles.append(self.model.layers[-1].register_forward_hook(final_layer_hook))

        def edge_head_pre_hook(_module, inputs):
            final["readout"] = inputs[0]

        if self.task_type == "edge_binary":
            handles.append(self.model.heads.edge_head.register_forward_pre_hook(edge_head_pre_hook))
        try:
            if require_grad:
                native = self.model(batch)
            else:
                with torch.no_grad():
                    native = self.model(batch)
        finally:
            for handle in handles:
                handle.remove()
        if any(value is None for value in transport) or "nodes" not in final:
            raise RuntimeError("official GRIT transport/final-state hooks did not fire")
        counts = [int(value) for value in batch.graph_num_nodes.detach().cpu().tolist()]
        edge_counts = [
            int((batch.edge_batch == index).sum().item())
            for index in range(len(data_list))
        ]
        prediction_rows = []
        target_rows = []
        if self.task_type == "edge_binary":
            offset = 0
            for count in edge_counts:
                prediction_rows.append(native[offset : offset + count])
                target_rows.append(batch.edge_target[offset : offset + count])
                offset += count
        else:
            prediction_rows = [native[index : index + 1] for index in range(len(data_list))]
            target_rows = [
                batch.graph_target[index : index + 1] for index in range(len(data_list))
            ]
        prediction = self._pad_rows(
            [self._raw_prediction(row) for row in prediction_rows]
        )
        target = self._pad_rows(target_rows, fill=float("nan"))
        return {
            "batch": batch,
            "native": native,
            "prediction": prediction,
            "z": self._z(prediction),
            "target": target,
            "transport": tuple(transport),
            "nodes": final["nodes"],
            "readout": final.get("readout"),
            "counts": counts,
            "edge_counts": edge_counts,
            "attention": attention,
        }

    def _split_capture(self, captured: Mapping[str, Any]) -> list[BackendCapture]:
        torch = self._torch
        results = []
        node_offset = 0
        edge_offset = 0
        for graph, (nodes, edges) in enumerate(
            zip(captured["counts"], captured["edge_counts"])
        ):
            layers = tuple(
                value[node_offset : node_offset + nodes] for value in captured["transport"]
            )
            if self.task_type == "edge_binary":
                final = captured["readout"][edge_offset : edge_offset + edges]
            else:
                final = captured["nodes"][node_offset : node_offset + nodes]
            results.append(
                BackendCapture(
                    prediction=captured["prediction"][graph : graph + 1],
                    z=captured["z"][graph : graph + 1],
                    target=captured["target"][graph : graph + 1],
                    transport=layers,
                    final_state=final,
                    real_mask=None,
                )
            )
            node_offset += nodes
            edge_offset += edges
        return results

    def capture(
        self,
        data_list: Sequence[Any],
        *,
        require_grad: bool,
        include_virtual_transport: bool = True,
    ) -> BackendCapture:
        del include_virtual_transport
        if not data_list:
            raise ValueError("capture requires at least one GraphBench graph")
        node_counts = {int(data.num_nodes) for data in data_list}
        edge_counts = {int(data.edge_index.shape[1]) for data in data_list}
        if len(node_counts) != 1 or len(edge_counts) != 1:
            raise ValueError("one event capture requires replicas of one base geometry")
        captured = self._forward_capture(data_list, require_grad=require_grad)
        split = self._split_capture(captured)
        return BackendCapture(
            prediction=self._torch.cat([row.prediction for row in split], dim=0),
            z=self._torch.cat([row.z for row in split], dim=0),
            target=self._torch.cat([row.target for row in split], dim=0),
            transport=tuple(
                self._torch.stack([row.transport[layer] for row in split], dim=0)
                for layer in range(int(self.runtime.L))
            )
            if not require_grad
            else tuple(captured["transport"]),
            final_state=(
                self._torch.stack([row.final_state for row in split], dim=0)
                if not require_grad
                else split[0].final_state
            ),
            real_mask=None,
        )

    def capture_groups(
        self,
        groups: Sequence[Sequence[Any]],
        *,
        include_virtual_transport: bool = True,
    ) -> list[BackendCapture]:
        del include_virtual_transport
        flat = [data for group in groups for data in group]
        captured = self._split_capture(
            self._forward_capture(flat, require_grad=False)
        )
        outputs = []
        offset = 0
        for group in groups:
            rows = captured[offset : offset + len(group)]
            outputs.append(
                BackendCapture(
                    prediction=self._torch.cat([row.prediction for row in rows], dim=0),
                    z=self._torch.cat([row.z for row in rows], dim=0),
                    target=self._torch.cat([row.target for row in rows], dim=0),
                    transport=tuple(
                        self._torch.stack(
                            [row.transport[layer] for row in rows], dim=0
                        )
                        for layer in range(int(self.runtime.L))
                    ),
                    final_state=self._torch.stack(
                        [row.final_state for row in rows], dim=0
                    ),
                    real_mask=None,
                )
            )
            offset += len(group)
        return outputs

    @property
    def jacobian_engine(self) -> str:
        if self._batched_vjp_supported is False:
            return "sequential_vjp"
        if self._batched_vjp_supported is True:
            return "batched_vjp"
        return "auto"

    @property
    def clean_jacobian_batch_size(self) -> int:
        # clean_jacobians_many intentionally loops over graph-local output geometries, so a
        # larger outer batch delays per-graph persistence without increasing device concurrency.
        return 1

    def remember_clean_jacobians(self, data: Any, clean: CleanJacobians) -> None:
        """Populate worker-local reuse from a validated persistent shard."""

        self._clean_jacobian_cache[id(data)] = clean

    @staticmethod
    def _is_vmap_compatibility_error(error: RuntimeError) -> bool:
        """Recognise engine limitations without swallowing unrelated autograd failures."""

        message = str(error).lower()
        if "vmap" not in message:
            return False
        return any(
            marker in message
            for marker in (
                "not possible",
                "batching rule",
                "in-place",
                "inplace",
                "does not support",
                "not implemented",
            )
        )

    def _clean_jacobians_impl(
        self,
        data: Any,
        *,
        use_batched_vjp: bool,
    ) -> CleanJacobians:
        torch = self._torch
        raw = self._forward_capture([data], require_grad=True)
        capture = self._split_capture(raw)[0]
        output_count = (
            int(data.edge_index.shape[1]) if self.task_type == "edge_binary" else 1
        )
        z = raw["z"][0, :output_count]
        targets = tuple(raw["transport"]) + (
            raw["readout"] if self.task_type == "edge_binary" else raw["nodes"],
        )
        transport_rows: list[list[Any]] = []
        final_rows: list[Any] = []
        if use_batched_vjp:
            for start in range(0, output_count, self.jacobian_output_chunk):
                stop = min(output_count, start + self.jacobian_output_chunk)
                basis = torch.zeros(
                    stop - start,
                    output_count,
                    device=z.device,
                    dtype=z.dtype,
                )
                basis[
                    torch.arange(stop - start, device=z.device),
                    torch.arange(start, stop, device=z.device),
                ] = 1
                gradients = torch.autograd.grad(
                    z,
                    targets,
                    grad_outputs=basis,
                    is_grads_batched=True,
                    retain_graph=stop < output_count,
                    allow_unused=False,
                )
                transport_rows.append(list(gradients[:-1]))
                if self.task_type == "edge_binary":
                    diagonal = gradients[-1][
                        torch.arange(stop - start, device=z.device),
                        torch.arange(start, stop, device=z.device),
                    ]
                    final_rows.append(diagonal)
                else:
                    final_rows.append(gradients[-1])
        else:
            # This computes the same output-by-intermediate Jacobian as the batched path.  It is
            # slower, but it does not invoke vmap and therefore supports official operators with
            # in-place or missing batching rules.
            for output in range(output_count):
                gradients = torch.autograd.grad(
                    z[output],
                    targets,
                    retain_graph=output + 1 < output_count,
                    allow_unused=False,
                )
                transport_rows.append(
                    [gradient.unsqueeze(0) for gradient in gradients[:-1]]
                )
                if self.task_type == "edge_binary":
                    final_rows.append(gradients[-1][output : output + 1])
                else:
                    final_rows.append(gradients[-1].unsqueeze(0))
        transport = torch.stack(
            [
                torch.cat([chunk[layer] for chunk in transport_rows], dim=0)
                for layer in range(int(self.runtime.L))
            ],
            dim=1,
        )
        final_gradient = torch.cat(final_rows, dim=0)
        audit_check(
            bool(torch.isfinite(transport).all() and torch.isfinite(final_gradient).all()),
            "graphbench.finite_clean_jacobian",
            "non-finite clean GraphBench Jacobian",
        )
        capture.prediction = capture.prediction.detach()
        capture.z = capture.z.detach()
        capture.target = capture.target.detach()
        capture.transport = tuple(value.detach() for value in capture.transport)
        capture.final_state = capture.final_state.detach()
        return CleanJacobians(capture, transport.detach(), final_gradient.detach())

    def clean_jacobians(self, data: Any) -> CleanJacobians:
        cache_key = id(data)
        cached = self._clean_jacobian_cache.get(cache_key)
        if cached is not None:
            self._clean_jacobian_cache_hits += 1
            if self._clean_jacobian_cache_hits == 1:
                print(
                    "[graphbench] reusing detached clean Jacobians across worker "
                    "components",
                    flush=True,
                )
            return cached
        use_batched_vjp = self._batched_vjp_supported is not False
        try:
            clean = self._clean_jacobians_impl(
                data,
                use_batched_vjp=use_batched_vjp,
            )
        except RuntimeError as error:
            if not use_batched_vjp or not self._is_vmap_compatibility_error(error):
                raise
            self._batched_vjp_supported = False
            self._jacobian_fallback_reason = str(error).splitlines()[0]
            print(
                "[graphbench] batched VJP is incompatible with an official model "
                "operator; rebuilding the clean forward and using exact sequential "
                "VJPs for this worker",
                flush=True,
            )
            clean = self._clean_jacobians_impl(data, use_batched_vjp=False)
        else:
            if use_batched_vjp:
                self._batched_vjp_supported = True
        self._clean_jacobian_cache[cache_key] = clean
        return clean

    def clean_jacobians_many(self, data_list: Sequence[Any]) -> list[CleanJacobians]:
        # Output counts differ across matching graphs. Batched VJPs (or the exact sequential
        # fallback selected by the first incompatible official operator) are used inside each
        # graph, while event forwards remain multi-graph. This avoids padding a dense E-by-E
        # Jacobian across unrelated graph geometries.
        return [self.clean_jacobians(data) for data in data_list]

    def functional_carriage_events(self, delta, clean_gradient):
        if self.task_type != "edge_binary":
            from .carriage import functional_carriage_events

            return functional_carriage_events(delta, clean_gradient)
        if delta.ndim != 4 or clean_gradient.ndim != 2:
            raise ValueError("matching carriage expects delta [S,K,E,W], gradient [E,W]")
        return self._torch.einsum("skew,ew->ske", delta, clean_gradient).abs()

    def loss_per_graph(self, prediction, target):
        torch = self._torch
        if self.task_type == "edge_binary":
            valid = torch.isfinite(target)
            safe = torch.where(valid, target, torch.zeros_like(target))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                prediction,
                safe,
                reduction="none",
                pos_weight=(
                    self.runtime.pos_weight.to(prediction.device)
                    if self.runtime.pos_weight is not None
                    else None
                ),
            )
            return (loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        std = max(float(self.runtime.target_stats["std"]), 1.0e-6)
        return ((prediction - target) / std).square().mean(dim=1)

    def loss_from_states(self, data: Any, target):
        target = target[:, : int(data.edge_index.shape[1])]

        def evaluate(states):
            if self.task_type == "edge_binary":
                prediction = self.model.heads.edge_head(states).squeeze(-1)
                return self.loss_per_graph(
                    prediction,
                    target.expand(prediction.shape[0], -1),
                )
            pooled = self._torch.cat(
                (states.mean(dim=1), states.max(dim=1).values), dim=-1
            )
            native = self.model.heads.graph_head(pooled).reshape(-1, 1)
            prediction = self._raw_prediction(native)
            return self.loss_per_graph(
                prediction,
                target.expand(prediction.shape[0], -1),
            )

        return evaluate

    def transport_distances(
        self, data: Any, source: int, pristine, *, channel: str = "structural"
    ) -> list[Any]:
        if channel == "semantic":
            left, right = _edge_endpoints(data, source)
            return list(np.minimum(pristine[left], pristine[right]))
        return list(pristine[int(source), :])

    def carriage_distance_matrix(
        self,
        data: Any,
        sources: Sequence[int],
        pristine,
        *,
        channel: str = "structural",
    ):
        if channel == "semantic":
            source_endpoints = [_edge_endpoints(data, source) for source in sources]
            if self.task_type == "edge_binary":
                carrier_endpoints = [
                    _edge_endpoints(data, edge)
                    for edge in range(int(data.edge_index.shape[1]))
                ]
                return np.asarray(
                    [
                        [
                            min(
                                pristine[c_left, s_left],
                                pristine[c_left, s_right],
                                pristine[c_right, s_left],
                                pristine[c_right, s_right],
                            )
                            for s_left, s_right in source_endpoints
                        ]
                        for c_left, c_right in carrier_endpoints
                    ],
                    dtype=np.float64,
                )
            return np.asarray(
                [
                    [
                        min(pristine[carrier, left], pristine[carrier, right])
                        for left, right in source_endpoints
                    ]
                    for carrier in range(int(data.num_nodes))
                ],
                dtype=np.float64,
            )
        if self.task_type == "edge_binary":
            carrier_endpoints = [
                _edge_endpoints(data, edge)
                for edge in range(int(data.edge_index.shape[1]))
            ]
            return np.asarray(
                [
                    [
                        min(pristine[left, source], pristine[right, source])
                        for source in sources
                    ]
                    for left, right in carrier_endpoints
                ],
                dtype=np.float64,
            )
        return pristine[np.asarray(sources, dtype=np.int64), :].T

    def carriage_carrier_kind(
        self, data: Any, carrier: int, *, channel: str = "structural"
    ) -> str:
        del data, carrier, channel
        return "readout_edge" if self.task_type == "edge_binary" else "algorithm_node"

    def attention_normalization_error(self, data: Any) -> float:
        torch = self._torch
        raw = self._forward_capture([data], require_grad=False)
        error = 0.0
        for _layer, edge_index, attention in raw["attention"]:
            values = attention.squeeze(-1)
            mass = torch.zeros(
                int(data.num_nodes),
                int(self.runtime.H),
                dtype=values.dtype,
                device=values.device,
            )
            mass.index_add_(0, edge_index[1].long(), values)
            valid = torch.unique(edge_index[1].long())
            error = max(
                error, float(torch.max(torch.abs(mass[valid] - 1.0)).item())
            )
        return error

    def clean_attention_distance(self, data: Any, pristine, axis):
        raw = self._forward_capture([data], require_grad=False)
        profile = np.zeros(
            (int(self.runtime.L), int(self.runtime.H), len(axis.labels)),
            dtype=np.float64,
        )
        for layer, edge_index, attention in raw["attention"]:
            edge = edge_index.detach().cpu().numpy()
            values = attention.squeeze(-1).detach().cpu().numpy()
            for position, (sender, receiver) in enumerate(edge.T):
                bucket = axis.index(pristine[int(receiver), int(sender)])
                profile[int(layer), :, bucket] += values[position]
        denominator = profile.sum(axis=-1, keepdims=True)
        np.divide(profile, denominator, out=profile, where=denominator > 0)
        return profile

    def _native_forward(
        self,
        data_list: Sequence[Any],
        *,
        family: Sequence[tuple[int, int]] = (),
        replacements: Sequence[Any] | None = None,
        ablate: bool = False,
    ):
        handles = []
        by_layer: dict[int, list[int]] = {}
        for layer, head in family:
            by_layer.setdefault(int(layer), []).append(int(head))
        for layer, heads in by_layer.items():
            selected = sorted(set(heads))

            def make_hook(layer_index, selected_heads):
                def hook(_module, _inputs, output):
                    routed = output[0] if isinstance(output, tuple) else output
                    view = (
                        routed.reshape(routed.shape[0], self.runtime.H, self.runtime.dh)
                        if routed.ndim == 2
                        else routed
                    )
                    changed = view.clone()
                    if ablate:
                        changed[:, selected_heads, :] = 0.0
                    else:
                        donor = replacements[layer_index].to(
                            device=changed.device, dtype=changed.dtype
                        )
                        if donor.shape != changed.shape:
                            raise RuntimeError(
                                f"patch geometry differs at layer {layer_index}: "
                                f"{tuple(donor.shape)} vs {tuple(changed.shape)}"
                            )
                        changed[:, selected_heads, :] = donor[:, selected_heads, :]
                    routed_changed = changed.reshape_as(routed)
                    if isinstance(output, tuple):
                        return (routed_changed, *output[1:])
                    return routed_changed

                return hook

            handles.append(
                self.model.layers[layer].attention.register_forward_hook(
                    make_hook(layer, selected)
                )
            )
        try:
            raw = self._forward_capture(data_list, require_grad=False)
        finally:
            for handle in handles:
                handle.remove()
        return raw["prediction"], raw["z"], raw["target"]

    def _native_forward_individual_heads(
        self,
        data_list: Sequence[Any],
        heads: Sequence[tuple[int, int]],
        *,
        replacements: Sequence[Any] | None = None,
        ablate: bool = False,
    ):
        """Patch a different single head in each replica within one saturated forward.

        The canonical public patch API applies one family to every replica.  The PE-refinement
        experiment requires the same all-head endpoints but can evaluate independent heads much
        more efficiently by assigning one ``(layer, head)`` pair to each replica.
        """

        torch = self._torch
        if len(data_list) != len(heads) or not data_list:
            raise ValueError("individual-head targets and assignments must align and be non-empty")
        node_counts = {int(data.num_nodes) for data in data_list}
        edge_counts = {int(data.edge_index.shape[1]) for data in data_list}
        if len(node_counts) != 1 or len(edge_counts) != 1:
            raise ValueError("individual-head replicas must share one base geometry")
        if not ablate:
            if replacements is None or len(replacements) != int(self.runtime.L):
                raise ValueError("individual-head patching requires one replacement per layer")
            expected = (
                len(data_list),
                next(iter(node_counts)),
                int(self.runtime.H),
                int(self.runtime.dh),
            )
            normalised_replacements = []
            for layer, replacement in enumerate(replacements):
                if tuple(replacement.shape) == (
                    expected[0] * expected[1],
                    expected[2],
                    expected[3],
                ):
                    replacement = replacement.reshape(expected)
                if tuple(replacement.shape) != expected:
                    raise RuntimeError(
                        f"replacement geometry differs at layer {layer}: "
                        f"{tuple(replacement.shape)} vs {expected}"
                    )
                normalised_replacements.append(replacement)
            replacements = tuple(normalised_replacements)
        assignments = torch.as_tensor(
            heads, dtype=torch.long, device=self.device
        )
        if bool((assignments[:, 0] < 0).any()) or bool(
            (assignments[:, 0] >= int(self.runtime.L)).any()
        ):
            raise IndexError("individual-head layer assignment is outside the model")
        if bool((assignments[:, 1] < 0).any()) or bool(
            (assignments[:, 1] >= int(self.runtime.H)).any()
        ):
            raise IndexError("individual-head index is outside the model")
        replicas = len(data_list)
        nodes = next(iter(node_counts))
        node_index = torch.arange(nodes, device=self.device).reshape(1, -1)
        handles = []
        for layer in range(int(self.runtime.L)):
            selected = torch.nonzero(
                assignments[:, 0] == layer, as_tuple=False
            ).reshape(-1)
            if not int(selected.numel()):
                continue

            def make_hook(layer_index: int, selected_replicas: Any):
                selected_heads = assignments[selected_replicas, 1]

                def hook(_module, _inputs, output):
                    routed = output[0] if isinstance(output, tuple) else output
                    view = routed.reshape(
                        replicas,
                        nodes,
                        int(self.runtime.H),
                        int(self.runtime.dh),
                    )
                    changed = view.clone()
                    replica_grid = selected_replicas.reshape(-1, 1)
                    head_grid = selected_heads.reshape(-1, 1)
                    nodes_grid = node_index.expand(int(selected_replicas.numel()), -1)
                    if ablate:
                        changed[replica_grid, nodes_grid, head_grid, :] = 0.0
                    else:
                        donor = replacements[layer_index].to(
                            device=changed.device, dtype=changed.dtype
                        )
                        changed[replica_grid, nodes_grid, head_grid, :] = donor[
                            replica_grid, nodes_grid, head_grid, :
                        ]
                    routed_changed = changed.reshape_as(routed)
                    if isinstance(output, tuple):
                        return (routed_changed, *output[1:])
                    return routed_changed

                return hook

            handles.append(
                self.model.layers[layer].attention.register_forward_hook(
                    make_hook(layer, selected)
                )
            )
        try:
            raw = self._forward_capture(data_list, require_grad=False)
        finally:
            for handle in handles:
                handle.remove()
        return raw["prediction"], raw["z"], raw["target"]

    def ablate(self, data_list: Sequence[Any], family):
        return self._native_forward(data_list, family=family, ablate=True)

    def patch(self, target: Any, donor_transport: Sequence[Any], family):
        return self._native_forward(
            [target], family=family, replacements=donor_transport
        )

    def patch_many(self, targets: Sequence[Any], donor_transport: Sequence[Any], family):
        return self._native_forward(
            targets, family=family, replacements=donor_transport
        )

    def ablate_individual_heads(
        self,
        data_list: Sequence[Any],
        heads: Sequence[tuple[int, int]],
    ):
        return self._native_forward_individual_heads(
            data_list, heads, ablate=True
        )

    def patch_individual_heads(
        self,
        targets: Sequence[Any],
        donor_transport: Sequence[Any],
        heads: Sequence[tuple[int, int]],
    ):
        return self._native_forward_individual_heads(
            targets, heads, replacements=donor_transport, ablate=False
        )

    def replacement_batch(
        self,
        capture: BackendCapture,
        indices: Sequence[int],
        *,
        repeat_single: bool = False,
    ) -> tuple[Any, ...]:
        del repeat_single
        selected = [int(value) for value in indices]
        return tuple(
            layer[selected].reshape(-1, layer.shape[-2], layer.shape[-1]).detach()
            for layer in capture.transport
        )
