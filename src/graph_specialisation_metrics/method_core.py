"""Reusable methodology primitives for graph specialisation experiments.

This module is intentionally model-agnostic.  It defines the file artifact
contract, a small graph batch view, adapter interfaces, and the core numerical
methods shared by the dissertation procedure and the synthetic validation suite.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

import networkx as nx
import numpy as np
import torch

try:
    import yaml
except Exception as exc:  # pragma: no cover - only hit in incomplete envs.
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


EPS = 1.0e-12
METHODOLOGY_VERSION = "2026-06-28.v1"


@dataclass
class GraphBatchView:
    """Normalized view of one or more graphs.

    Shapes:
      x: [N, F] for a single graph or concatenated mini-batch.
      edge_index: [2, E] directed or undirected COO indices over x.
      batch: [N] graph id per node.  Defaults to one graph.
      y: optional [B, ...] graph labels.
      graph_ids: optional stable graph ids.
      split: train/val/test or synthetic split label.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    batch: Optional[torch.Tensor] = None
    y: Optional[torch.Tensor] = None
    graph_ids: Optional[Sequence[str]] = None
    split: Optional[str] = None
    distances: Optional[torch.Tensor] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.x.dim() != 2:
            raise ValueError(f"x must have shape [N,F], got {tuple(self.x.shape)}")
        if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if self.batch is None:
            self.batch = torch.zeros(self.x.size(0), dtype=torch.long, device=self.x.device)
        if self.batch.numel() != self.x.size(0):
            raise ValueError("batch must contain one graph id per node")

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    @property
    def num_graphs(self) -> int:
        assert self.batch is not None
        return int(self.batch.max().item() + 1) if self.batch.numel() else 0

    def clone_with(self, **updates: Any) -> "GraphBatchView":
        return dataclasses.replace(self, **updates)

    def to(self, device: torch.device | str) -> "GraphBatchView":
        device = torch.device(device)
        return GraphBatchView(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            batch=self.batch.to(device) if self.batch is not None else None,
            y=self.y.to(device) if isinstance(self.y, torch.Tensor) else self.y,
            graph_ids=self.graph_ids,
            split=self.split,
            distances=self.distances.to(device) if isinstance(self.distances, torch.Tensor) else None,
            metadata=dict(self.metadata or {}),
        )


@dataclass
class ForwardCache:
    """Adapter forward output plus optional internal fields."""

    prediction: torch.Tensor
    final_node_states: torch.Tensor
    attention: Optional[list[torch.Tensor]] = None
    channel_fields: Optional[dict[str, Any]] = None
    extras: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    version: str
    implementation: str
    official_repo: Optional[str] = None
    official_commit: Optional[str] = None
    validation_only: bool = False
    dev_only: bool = False


class ModelAdapter(Protocol):
    """Common adapter contract used by the methodology code."""

    info: AdapterInfo

    def forward(self, graph: GraphBatchView) -> ForwardCache:
        ...

    def predict(self, graph: GraphBatchView) -> torch.Tensor:
        ...

    def parameter_count(self) -> int:
        ...

    def attention_maps(self, graph: GraphBatchView) -> Optional[list[torch.Tensor]]:
        ...

    def patch_hidden_states(
        self,
        graph: GraphBatchView,
        clamp_nodes: Sequence[int],
        clean_cache: Optional[ForwardCache] = None,
    ) -> ForwardCache:
        ...


def ensure_dir(path: Path | str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    return value


def write_json(path: Path | str, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def read_yaml(path: Path | str) -> dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError(f"pyyaml is required: {_YAML_IMPORT_ERROR}")
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return dict(payload or {})


def write_yaml(path: Path | str, payload: Mapping[str, Any]) -> Path:
    if yaml is None:  # pragma: no cover
        raise RuntimeError(f"pyyaml is required: {_YAML_IMPORT_ERROR}")
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(jsonable(payload), f, sort_keys=False)
    return path


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key, "")) for key in fieldnames})
    return path


def config_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(cwd: Optional[Path | str] = None) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or Path.cwd()),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip()


def write_manifest(
    artifact_root: Path | str,
    *,
    run_type: str,
    config: Mapping[str, Any],
    adapter: Optional[ModelAdapter | AdapterInfo] = None,
    source_markdowns: Optional[Sequence[Path | str]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    artifact_root = ensure_dir(artifact_root)
    info: Optional[AdapterInfo]
    if adapter is None:
        info = None
    elif isinstance(adapter, AdapterInfo):
        info = adapter
    else:
        info = getattr(adapter, "info", None)
    markdown_hashes = {
        str(Path(path)): file_sha256(path)
        for path in source_markdowns or []
        if Path(path).exists()
    }
    manifest = {
        "run_type": run_type,
        "methodology_version": METHODOLOGY_VERSION,
        "config_hash": config_hash(config),
        "git_sha": git_sha(),
        "adapter": dataclasses.asdict(info) if info is not None else None,
        "parameter_count": adapter.parameter_count() if adapter is not None and not isinstance(adapter, AdapterInfo) else None,
        "source_markdown_sha256": markdown_hashes,
        "config": jsonable(config),
        "extra": jsonable(dict(extra or {})),
    }
    return write_json(artifact_root / "manifest.json", manifest)


def torch_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    gen = torch.Generator(device=torch.device(device).type)
    gen.manual_seed(int(seed))
    return gen


def set_global_seed(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def edge_index_from_edges(num_nodes: int, edges: Sequence[tuple[int, int]], *, undirected: bool = True) -> torch.Tensor:
    out: list[tuple[int, int]] = []
    for u, v in edges:
        if u == v:
            continue
        out.append((int(u), int(v)))
        if undirected:
            out.append((int(v), int(u)))
    if not out:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(out, dtype=torch.long).t().contiguous()


def dense_adjacency_from_edge_index(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=edge_index.device)
    if edge_index.numel():
        adj[edge_index[0].long(), edge_index[1].long()] = True
    return adj


def graph_to_networkx(graph: GraphBatchView, *, undirected: bool = True) -> nx.Graph:
    g = nx.Graph() if undirected else nx.DiGraph()
    g.add_nodes_from(range(graph.num_nodes))
    edges = graph.edge_index.detach().cpu().long().t().tolist()
    g.add_edges_from((int(u), int(v)) for u, v in edges if int(u) != int(v))
    return g


def shortest_path_distance_matrix(graph: GraphBatchView) -> torch.Tensor:
    g = graph_to_networkx(graph, undirected=True)
    n = graph.num_nodes
    dist = torch.full((n, n), float("inf"), dtype=torch.float32)
    for source, lengths in nx.all_pairs_shortest_path_length(g):
        for target, value in lengths.items():
            dist[int(source), int(target)] = float(value)
    return dist.to(graph.x.device)


def minimum_vertex_cut(graph: GraphBatchView, source: int, target: int) -> list[int]:
    """Return one minimum vertex separator excluding source and target."""

    source = int(source)
    target = int(target)
    if source == target:
        return []
    g = graph_to_networkx(graph, undirected=True)
    if not nx.has_path(g, source, target):
        return []
    cut = nx.minimum_node_cut(g, source, target)
    return sorted(int(v) for v in cut if int(v) not in {source, target})


def all_pair_distances_or_compute(graph: GraphBatchView) -> torch.Tensor:
    if graph.distances is not None:
        return graph.distances
    return shortest_path_distance_matrix(graph)


def rank_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def pearson_corr(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[mask]
    bb = bb[mask]
    if len(aa) < 2:
        return float("nan")
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    denom = math.sqrt(float(np.dot(aa, aa) * np.dot(bb, bb)))
    if denom <= EPS:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def spearman_corr(a: Sequence[float], b: Sequence[float]) -> float:
    return pearson_corr(rank_values(np.asarray(a, dtype=np.float64)), rank_values(np.asarray(b, dtype=np.float64)))


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) < 2:
        return float("nan")
    denom = float(np.sum((y - float(np.mean(y))) ** 2))
    if denom <= EPS:
        return float("nan")
    return 1.0 - float(np.sum((y - p) ** 2)) / denom


def auroc_score(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    labels_arr = np.asarray(labels, dtype=bool)
    scores_arr = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(scores_arr)
    labels_arr = labels_arr[mask]
    scores_arr = scores_arr[mask]
    n_pos = int(labels_arr.sum())
    n_neg = int((~labels_arr).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rank_values(scores_arr) + 1.0
    return float((ranks[labels_arr].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_ci(values: Sequence[float], *, seed: int = 0, draws: int = 1000, alpha: float = 0.05) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    means = []
    for _ in range(int(draws)):
        sample = vals[rng.integers(0, len(vals), size=len(vals))]
        means.append(float(np.mean(sample)))
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(np.mean(vals)), float(lo), float(hi)


def effective_rank(matrix: torch.Tensor | np.ndarray, *, energy: float = 0.99) -> int:
    mat = torch.as_tensor(matrix, dtype=torch.float64)
    if mat.numel() == 0:
        return 0
    vals = torch.linalg.svdvals(mat)
    sq = vals.detach().cpu().double() ** 2
    total = float(sq.sum().item())
    if total <= EPS:
        return 0
    cdf = torch.cumsum(sq, dim=0) / total
    return int(torch.searchsorted(cdf, torch.tensor(float(energy), dtype=cdf.dtype)).item() + 1)


def top_singular_share(matrix: torch.Tensor | np.ndarray) -> float:
    mat = torch.as_tensor(matrix, dtype=torch.float64)
    if mat.numel() == 0:
        return 0.0
    vals = torch.linalg.svdvals(mat)
    sq = vals.detach().cpu().double() ** 2
    total = float(sq.sum().item())
    if total <= EPS:
        return 0.0
    return float(sq[0].item() / total)


def permuted_column_top_share(matrix: np.ndarray, rng: np.random.Generator) -> float:
    permuted = np.array(matrix, dtype=np.float64, copy=True)
    for row in range(permuted.shape[0]):
        rng.shuffle(permuted[row, :])
    return top_singular_share(permuted)


def above_null_margin(
    matrix: torch.Tensor | np.ndarray,
    *,
    permutations: int = 32,
    seed: int = 0,
) -> float:
    mat = np.asarray(torch.as_tensor(matrix, dtype=torch.float64).detach().cpu().numpy())
    observed = top_singular_share(mat)
    rng = np.random.default_rng(int(seed))
    null = [permuted_column_top_share(mat, rng) for _ in range(int(permutations))]
    return float(observed - float(np.mean(null)))


def distance_profile(values: torch.Tensor, distances: torch.Tensor, *, max_distance: Optional[int] = None) -> list[dict[str, float]]:
    vals = values.detach().abs().cpu().double()
    dist = distances.detach().cpu()
    finite = torch.isfinite(dist)
    if max_distance is None:
        max_distance = int(dist[finite].max().item()) if bool(finite.any()) else 0
    total = float(vals[finite].sum().item())
    rows = []
    for d in range(int(max_distance) + 1):
        mask = finite & (dist.long() == d)
        mass = float(vals[mask].sum().item())
        rows.append({"distance": d, "mass": mass, "share": mass / max(total, EPS)})
    return rows


def integrated_gradients_output(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    baseline: torch.Tensor,
    *,
    steps: int = 32,
    target_index: Optional[int] = None,
) -> torch.Tensor:
    """Integrated gradients for a scalar model output with respect to x."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    x_detached = x.detach()
    base = baseline.detach().to(device=x_detached.device, dtype=x_detached.dtype)
    if base.shape != x_detached.shape:
        base = base.expand_as(x_detached)
    total_grad = torch.zeros_like(x_detached)
    for step in range(1, int(steps) + 1):
        alpha = float(step) / float(steps)
        point = (base + alpha * (x_detached - base)).detach().requires_grad_(True)
        pred = predict_fn(point)
        scalar = pred.reshape(-1)[0 if target_index is None else int(target_index)]
        (grad,) = torch.autograd.grad(scalar, point, retain_graph=False, create_graph=False)
        total_grad = total_grad + grad.detach()
    return (x_detached - base) * (total_grad / float(steps))


def source_influence_ig(
    adapter: ModelAdapter,
    graph: GraphBatchView,
    baseline: torch.Tensor,
    *,
    steps: int = 32,
    target_index: Optional[int] = None,
) -> torch.Tensor:
    """Return per-source scalar influence estimates summing feature IG."""

    def predict_from_x(x_new: torch.Tensor) -> torch.Tensor:
        return adapter.predict(graph.clone_with(x=x_new))

    ig = integrated_gradients_output(predict_from_x, graph.x, baseline, steps=steps, target_index=target_index)
    return ig.sum(dim=-1)


def pair_carriage_ig(
    adapter: ModelAdapter,
    graph: GraphBatchView,
    baseline: torch.Tensor,
    *,
    steps: int = 32,
    target_index: Optional[int] = None,
) -> torch.Tensor:
    """Approximate C[i,j] by distributing source IG over carriers.

    For adapters that do not expose carrier-level Jacobians, this conservative
    fallback allocates each source's output IG uniformly across carriers.  Official
    adapters can override this in future without changing downstream metrics.
    """

    source = source_influence_ig(adapter, graph, baseline, steps=steps, target_index=target_index)
    n = graph.num_nodes
    return source.view(1, n).expand(n, n) / max(float(n), 1.0)


def swap_source_influence(
    adapter: ModelAdapter,
    graph: GraphBatchView,
    *,
    source: int,
    partner: int,
    normalize: bool = True,
) -> float:
    clean = float(adapter.predict(graph).reshape(-1)[0].detach().cpu().item())
    x_new = graph.x.detach().clone()
    x_new[int(source)] = graph.x[int(partner)]
    swapped = float(adapter.predict(graph.clone_with(x=x_new)).reshape(-1)[0].detach().cpu().item())
    delta = swapped - clean
    if normalize:
        denom = float(torch.linalg.vector_norm(graph.x[int(source)] - graph.x[int(partner)]).detach().cpu().item())
        delta = delta / max(denom, EPS)
    return float(delta)


def non_additivity_ratio(delta_a: float, delta_b: float, delta_ab: float) -> float:
    denom = abs(float(delta_a)) + abs(float(delta_b))
    if denom <= EPS:
        return float("nan")
    return float(abs(float(delta_ab) - float(delta_a) - float(delta_b)) / denom)


def atomic_torch_save(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path
