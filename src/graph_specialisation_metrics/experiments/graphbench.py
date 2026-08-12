"""GraphBench maximum-weight bipartite matching with GRIT."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..distance import shortest_path_distances
from ..runner import (
    align_distance_contributions,
    clean_output_gradients,
    compute_channel_score,
    distance_categories,
    mean_channel_scores,
)
from ..sampling import analysis_indices
from . import ExperimentSetupError, dataset_lock, save_scores
from ._grit import grit_transformer_layer, layer_config


@dataclass(frozen=True)
class Settings:
    layers: int
    heads: int
    hidden_dim: int
    rrwp_steps: int
    dropout: float
    attention_dropout: float
    seeds: tuple[int, ...]
    train_graphs: int
    validation_graphs: int
    batch_size: int
    evaluation_batch_size: int
    max_steps: int
    warmup_steps: int
    evaluate_every: int
    validation_watch_graphs: int
    minimum_checkpoint_step: int
    learning_rate: float
    weight_decay: float
    score_graphs: int
    sources: int
    donor_swaps_per_source: int
    semantic_pool_graphs: int
    score_seed: int


def _settings(config: dict[str, Any], *, fast: bool) -> Settings:
    data, model = config.get("data", {}), config.get("model", {})
    training, scoring = config.get("training", {}), config.get("scoring", {})
    values = Settings(
        layers=int(model.get("layers", 6)),
        heads=int(model.get("heads", 8)),
        hidden_dim=int(model.get("hidden_dim", 112)),
        rrwp_steps=int(model.get("rrwp_steps", 16)),
        dropout=float(model.get("dropout", 0.1)),
        attention_dropout=float(model.get("attention_dropout", 0.1)),
        seeds=tuple(int(seed) for seed in training.get("seeds", (0, 1, 2, 3))),
        train_graphs=int(data.get("train_graphs", 40_000)),
        validation_graphs=int(data.get("validation_graphs", 4_000)),
        batch_size=int(training.get("batch_size", 1024)),
        evaluation_batch_size=int(training.get("evaluation_batch_size", 32)),
        max_steps=int(training.get("steps", 5_000)),
        warmup_steps=int(training.get("warmup_steps", 500)),
        evaluate_every=int(training.get("evaluate_every", 500)),
        validation_watch_graphs=int(training.get("validation_watch_graphs", 1_000)),
        minimum_checkpoint_step=int(training.get("minimum_checkpoint_step", 2_500)),
        learning_rate=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 0.1)),
        score_graphs=int(scoring.get("graphs", 48)),
        sources=int(scoring.get("sources_per_graph", 6)),
        donor_swaps_per_source=int(scoring.get("donor_swaps_per_source", 8)),
        semantic_pool_graphs=int(scoring.get("semantic_pool_graphs", 2_000)),
        score_seed=int(scoring.get("seed", 31_415)),
    )
    if fast:
        values = Settings(
            **{
                **values.__dict__,
                "seeds": values.seeds[:1],
                "train_graphs": min(values.train_graphs, 256),
                "validation_graphs": min(values.validation_graphs, 64),
                "batch_size": min(values.batch_size, 32),
                "evaluation_batch_size": min(values.evaluation_batch_size, 32),
                "max_steps": min(values.max_steps, 4),
                "warmup_steps": 1,
                "evaluate_every": 1,
                "validation_watch_graphs": min(values.validation_watch_graphs, 32),
                "minimum_checkpoint_step": 0,
                "score_graphs": min(values.score_graphs, 2),
                "sources": min(values.sources, 2),
                "donor_swaps_per_source": min(values.donor_swaps_per_source, 2),
                "semantic_pool_graphs": min(values.semantic_pool_graphs, 16),
            }
        )
    if (values.layers, values.heads, values.hidden_dim, values.rrwp_steps) != (6, 8, 112, 16):
        raise ValueError("GraphBench GRIT requires 6 layers, 8 heads, width 112 and RRWP-16")
    if not values.seeds or min(values.train_graphs, values.validation_graphs) < 1:
        raise ValueError("GraphBench needs non-empty splits and at least one seed")
    return values


def jobs(config: dict[str, Any], *, fast: bool = False) -> list[dict[str, Any]]:
    """Return the configured GraphBench training jobs."""

    table = [
        {"experiment": "graphbench", "task": "matching_hard", "model": "grit", "seed": seed}
        for seed in _settings(config, fast=fast).seeds
    ]
    index = config.get("job_index")
    if index is None:
        return table
    index = int(index)
    if index < 0 or index >= len(table):
        raise IndexError(f"job_index {index} is outside 0..{len(table) - 1}")
    return [table[index]]


def _require_runtime() -> None:
    missing = [
        name
        for name in ("graphbench", "torch_geometric", "yacs", "opt_einsum")
        if importlib.util.find_spec(name) is None
    ]
    try:
        importlib.metadata.distribution("graphgps")
    except importlib.metadata.PackageNotFoundError:
        missing.append("GRIT")
    if missing:
        raise ExperimentSetupError(
            "GraphBench dependencies are missing: "
            + ", ".join(missing)
            + ". Install this project with the graphbench extra."
        )


def _seed(value: int) -> None:
    import torch

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _subset(dataset: Sequence[Any], size: int, seed: int) -> list[Any]:
    indices: Sequence[int] = _subset_indices(len(dataset), size, seed)
    return [dataset[index] for index in indices]


def _subset_indices(length: int, size: int, seed: int) -> list[int]:
    if size >= length:
        return list(range(length))
    return sorted(random.Random(seed).sample(range(length), int(size)))


def _data_root(output_dir: Path, config: dict[str, Any]) -> Path:
    configured = config.get("data", {}).get("root")
    return (
        Path(str(configured)).expanduser()
        if configured
        else Path(output_dir).resolve().parent / "data"
    )


def _load_dataset(root: Path) -> dict[str, Sequence[Any]]:
    from graphbench import Loader

    with dataset_lock(root, "graphbench"):
        loaded = Loader(root=root, dataset_names="bipartite_matching_hard").load()
    if len(loaded) != 1:
        raise RuntimeError(f"GraphBench returned {len(loaded)} matching datasets; expected one")
    splits = loaded[0]
    return {name: splits[name] for name in ("train", "val")}


def _load_splits(root: Path, settings: Settings) -> dict[str, list[Any]]:
    splits = _load_dataset(root)
    return {
        "train": _subset(splits["train"], settings.train_graphs, 101),
        "val": _subset(splits["val"], settings.validation_graphs, 211),
    }


def _edge_statistics(graphs: Sequence[Any]) -> tuple[float, float]:
    import torch

    values = torch.cat([graph.edge_attr.detach().float().reshape(-1) for graph in graphs])
    return float(values.mean()), float(values.std(unbiased=False).clamp_min(1e-6))


def _prepare_graph(graph: Any, *, mean: float, std: float, steps: int) -> Any:
    """Add RRWP features without changing attention support."""

    import torch

    data = graph.clone()
    data.edge_index = data.edge_index.long()
    data.orig_edge_index = data.edge_index.clone()
    data.orig_edge_value = (data.edge_attr.detach().float().reshape(-1) - mean) / std
    data.edge_attr = data.orig_edge_value[:, None]
    node_type = getattr(data, "x", None)
    if node_type is None or node_type.numel() != int(data.num_nodes):
        node_type = torch.zeros(int(data.num_nodes), dtype=torch.long)
    data.x = node_type.detach().long().reshape(-1).clamp(0, 7)
    adjacency = torch.zeros(int(data.num_nodes), int(data.num_nodes), dtype=torch.float32)
    adjacency[data.edge_index[0], data.edge_index[1]] = 1.0
    adjacency[data.edge_index[1], data.edge_index[0]] = 1.0
    degree = adjacency.sum(dim=1)
    transition = adjacency / degree.clamp_min(1.0)[:, None]
    powers = [torch.eye(int(data.num_nodes), dtype=torch.float32)]
    for _ in range(1, int(steps)):
        powers.append(powers[-1] @ transition)
    dense = torch.stack(powers, dim=-1)
    nodes = torch.arange(int(data.num_nodes))
    data.rrwp_dense = dense
    data.rrwp = dense[nodes, nodes]
    data.rrwp_index = torch.stack(
        [nodes.repeat_interleave(int(data.num_nodes)), nodes.repeat(int(data.num_nodes))]
    )
    data.rrwp_val = dense.reshape(-1, int(steps))
    data.deg = degree
    data.log_deg = torch.log1p(degree)
    data.y = data.y.detach().float().reshape(-1)
    return data


def _prepared_splits(
    output_dir: Path, config: dict[str, Any], settings: Settings
) -> tuple[dict[str, list[Any]], tuple[float, float]]:
    import torch

    root = _data_root(output_dir, config)
    cache = (
        output_dir
        / "cache"
        / f"matching_{settings.train_graphs}_{settings.validation_graphs}_rrwp{settings.rrwp_steps}.pt"
    )
    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        return payload["splits"], tuple(payload["edge_statistics"])
    raw = _load_splits(root, settings)
    mean, std = _edge_statistics(raw["train"])
    splits = {
        split: [
            _prepare_graph(graph, mean=mean, std=std, steps=settings.rrwp_steps) for graph in graphs
        ]
        for split, graphs in raw.items()
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"splits": splits, "edge_statistics": (mean, std)}, cache)
    return splits, (mean, std)


def _prepared_score_splits(
    output_dir: Path,
    config: dict[str, Any],
    model_settings: Settings,
    score_settings: Settings,
    edge_statistics: Sequence[float],
) -> dict[str, list[Any]]:
    """Load the selected semantic donor and evaluation graphs."""

    root = _data_root(output_dir, config)
    raw = _load_dataset(root)
    mean, std = (float(value) for value in edge_statistics)
    train_indices = _subset_indices(len(raw["train"]), model_settings.train_graphs, 101)
    validation_indices = _subset_indices(len(raw["val"]), model_settings.validation_graphs, 211)
    evaluation_positions, semantic_pool_positions = analysis_indices(
        len(validation_indices),
        min(score_settings.score_graphs, len(validation_indices)),
        len(train_indices),
        score_settings.semantic_pool_graphs,
        score_settings.score_seed,
    )
    selected = {
        "train": [
            raw["train"][train_indices[int(position)]] for position in semantic_pool_positions
        ],
        "val": [raw["val"][validation_indices[int(position)]] for position in evaluation_positions],
    }
    return {
        split: [
            _prepare_graph(graph, mean=mean, std=std, steps=model_settings.rrwp_steps)
            for graph in graphs
        ]
        for split, graphs in selected.items()
    }


def _model(settings: Settings):
    import torch
    from torch import nn

    GritTransformerLayer = grit_transformer_layer()

    class RRWPNodeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(settings.rrwp_steps, settings.hidden_dim, bias=False)
            nn.init.xavier_uniform_(self.fc.weight)

        def forward(self, data):
            data.x = data.x + self.fc(data.rrwp)
            return data

    class RRWPEdgeEncoder(nn.Module):
        """Encode RRWP on the complete pair index."""

        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(settings.rrwp_steps, settings.hidden_dim, bias=False)
            nn.init.xavier_uniform_(self.fc.weight)
            self.register_buffer("padding", torch.zeros(1, settings.hidden_dim))

        def forward(self, data):
            index = data.rrwp_index
            keys = index[0] * int(data.num_nodes) + index[1]
            order = torch.argsort(keys)
            index, keys = index[:, order], keys[order]
            values = self.fc(data.rrwp_val[order])
            query = data.edge_index[0] * int(data.num_nodes) + data.edge_index[1]
            position = torch.searchsorted(keys, query)
            valid = position < keys.numel()
            found = valid & (keys[position.clamp(max=keys.numel() - 1)] == query)
            if not bool(found.all()):
                raise RuntimeError("an original edge is absent from the complete RRWP index")
            values = values.index_add(0, position, data.edge_attr)
            data.edge_index, data.edge_attr = index, values
            return data

    class PredictionHeads(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dim = settings.hidden_dim
            self.graph_head = nn.Sequential(
                nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1)
            )
            self.node_head = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1)
            )
            self.edge_head = nn.Sequential(
                nn.LayerNorm(5 * dim + 1),
                nn.Linear(5 * dim + 1, dim),
                nn.GELU(),
                nn.Linear(dim, 1),
            )

        def edge(self, data, pair):
            source, target = data.orig_edge_index
            left, right = data.x[source], data.x[target]
            features = torch.cat(
                [
                    left,
                    right,
                    torch.abs(left - right),
                    left * right,
                    pair,
                    data.orig_edge_value[:, None],
                ],
                dim=-1,
            )
            return self.edge_head(features).squeeze(-1)

    class MatchingGRIT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dim = settings.hidden_dim
            self.node_encoder = nn.Embedding(8, dim)
            self.edge_encoder = nn.Linear(1, dim)
            self.rrwp_node_encoder = RRWPNodeEncoder()
            self.rrwp_edge_encoder = RRWPEdgeEncoder()
            self.layers = nn.ModuleList(
                GritTransformerLayer(
                    dim,
                    dim,
                    settings.heads,
                    dropout=settings.dropout,
                    attn_dropout=settings.attention_dropout,
                    layer_norm=False,
                    batch_norm=True,
                    residual=True,
                    act="relu",
                    norm_e=True,
                    O_e=True,
                    cfg=layer_config(signed_sqrt=False),
                )
                for _ in range(settings.layers)
            )
            self.heads = PredictionHeads()

        @staticmethod
        def _edge_states(data, source, target):
            keys = data.edge_index[0] * int(data.num_nodes) + data.edge_index[1]
            order = torch.argsort(keys)
            sorted_keys = keys[order]
            query = source * int(data.num_nodes) + target
            position = torch.searchsorted(sorted_keys, query)
            valid = position < sorted_keys.numel()
            found = valid & (sorted_keys[position.clamp(max=sorted_keys.numel() - 1)] == query)
            if not bool(found.all()):
                raise RuntimeError("original edges are absent from GRIT's pair tensor")
            return data.edge_attr[order[position]]

        def forward(self, batch, *, capture: bool = False):
            batch.x = self.node_encoder(batch.x.long().reshape(-1))
            batch.edge_attr = self.edge_encoder(batch.orig_edge_value.float().reshape(-1, 1))
            batch = self.rrwp_node_encoder(batch)
            batch = self.rrwp_edge_encoder(batch)
            head_outputs = []
            handles = []

            def capture_head(_module, _inputs, output):
                head_value, edge_value = output
                recorded = head_value.reshape(
                    int(batch.num_nodes), settings.heads, settings.hidden_dim // settings.heads
                ).clone()
                head_outputs.append(recorded)
                return recorded.reshape_as(head_value), edge_value

            if capture:
                handles = [
                    layer.attention.register_forward_hook(capture_head) for layer in self.layers
                ]
            try:
                for layer in self.layers:
                    batch = layer(batch)
            finally:
                for handle in handles:
                    handle.remove()
            source, target = batch.orig_edge_index
            pair = self._edge_states(batch, source, target)
            logits = self.heads.edge(batch, pair)
            return (logits, tuple(head_outputs)) if capture else logits

    model = MatchingGRIT()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != 956_373:
        raise RuntimeError(f"GraphBench GRIT has {parameters:,} parameters; expected 956,373")
    return model


def _loader(graphs: Sequence[Any], batch_size: int, *, shuffle: bool):
    from torch_geometric.loader import DataLoader

    return DataLoader(list(graphs), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _metrics(logits, targets) -> dict[str, float]:
    import torch

    predictions = torch.sigmoid(logits) >= 0.5
    targets = targets >= 0.5
    tp = int((predictions & targets).sum())
    fp = int((predictions & ~targets).sum())
    fn = int((~predictions & targets).sum())
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "accuracy": float((predictions == targets).float().mean()),
    }


def _evaluate(model, graphs: Sequence[Any], batch_size: int, device) -> dict[str, float]:
    import torch

    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for batch in _loader(graphs, batch_size, shuffle=False):
            batch = batch.to(device)
            logits.append(model(batch).cpu())
            targets.append(batch.y.reshape(-1).cpu())
    return _metrics(torch.cat(logits), torch.cat(targets))


def train(config: dict[str, Any], *, output_dir: Path, fast: bool = False) -> dict[str, Any]:
    """Train the selected seeds and keep the best validation checkpoint."""

    import torch
    import torch.nn.functional as F

    _require_runtime()
    settings = _settings(config, fast=fast)
    output_dir = Path(output_dir).expanduser().resolve()
    table = jobs(config, fast=fast)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "jobs.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    splits, edge_statistics = _prepared_splits(output_dir, config, settings)
    requested = str(config.get("training", {}).get("device", "cuda"))
    device = torch.device(
        requested if requested.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    targets = torch.cat([graph.y.reshape(-1) for graph in splits["train"]]).float()
    positives, negatives = float((targets == 1).sum()), float((targets == 0).sum())
    pos_weight = torch.tensor(negatives / max(1.0, positives), device=device)
    checkpoints = []
    validation = _subset(
        splits["val"], min(settings.validation_watch_graphs, len(splits["val"])), 4_099
    )
    for job in table:
        _seed(int(job["seed"]))
        model = _model(settings).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
        )

        def schedule(step: int) -> float:
            if step < settings.warmup_steps:
                return (step + 1) / max(1, settings.warmup_steps)
            progress = (step - settings.warmup_steps) / max(
                1, settings.max_steps - settings.warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        best_f1, step = -1.0, 0
        checkpoint = output_dir / "checkpoints" / f"seed_{job['seed']}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        while step < settings.max_steps:
            for batch in _loader(splits["train"], settings.batch_size, shuffle=True):
                batch = batch.to(device)
                model.train()
                loss = F.binary_cross_entropy_with_logits(
                    model(batch), batch.y.reshape(-1).float(), pos_weight=pos_weight
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                step += 1
                if step % settings.evaluate_every == 0 or step == settings.max_steps:
                    metrics = _evaluate(model, validation, settings.evaluation_batch_size, device)
                    print(f"matching seed={job['seed']} step={step} val_f1={metrics['f1']:.4f}")
                    if step >= settings.minimum_checkpoint_step and metrics["f1"] > best_f1:
                        best_f1 = metrics["f1"]
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "settings": settings.__dict__,
                                "seed": int(job["seed"]),
                                "edge_statistics": edge_statistics,
                                "validation": metrics,
                            },
                            checkpoint,
                        )
                if step >= settings.max_steps:
                    break
        if not checkpoint.is_file():
            raise RuntimeError("training finished without an eligible finite checkpoint")
        checkpoints.append(checkpoint)
    return {"checkpoints": checkpoints, "jobs": table}


def _reciprocal_unit(graph: Any, edge: int) -> tuple[int, ...]:
    source, target = (int(graph.orig_edge_index[axis, edge]) for axis in (0, 1))
    edge_index = graph.orig_edge_index.detach().cpu().numpy()
    reciprocal = np.flatnonzero((edge_index[0] == target) & (edge_index[1] == source))
    return (edge, int(reciprocal[0])) if reciprocal.size and int(reciprocal[0]) != edge else (edge,)


def _edge_units(graph: Any) -> tuple[tuple[int, ...], ...]:
    units, seen = [], set()
    for edge in range(int(graph.orig_edge_index.shape[1])):
        unit = tuple(sorted(_reciprocal_unit(graph, edge)))
        if unit not in seen:
            seen.add(unit)
            units.append(unit)
    return tuple(units)


def _degrees(graph: Any) -> np.ndarray:
    return graph.deg.detach().cpu().numpy().astype(np.int64, copy=False)


def _degree_signature(graph: Any, edge: int) -> tuple[int, int]:
    degree = _degrees(graph)
    left, right = (int(graph.orig_edge_index[axis, edge]) for axis in (0, 1))
    return tuple(sorted((int(degree[left]), int(degree[right]))))


@dataclass(frozen=True)
class _EdgeDonor:
    graph: int
    edge: int
    value: float
    degree: tuple[int, int]


class _EdgeDonorPool:
    """Edge-weight donors with minimum endpoint-degree matching."""

    def __init__(self, graphs: Sequence[Any]) -> None:
        self.graphs = {
            graph_id: tuple(
                _EdgeDonor(
                    graph_id,
                    unit[0],
                    float(graph.orig_edge_value[unit[0]]),
                    _degree_signature(graph, unit[0]),
                )
                for unit in _edge_units(graph)
            )
            for graph_id, graph in enumerate(graphs)
        }
        if not self.graphs:
            raise ValueError("semantic donor pool is empty")

    def eligible(self, graph: Any, edge: int) -> dict[int, tuple[_EdgeDonor, ...]]:
        value, degree = float(graph.orig_edge_value[edge]), _degree_signature(graph, edge)
        candidates = [
            donor for rows in self.graphs.values() for donor in rows if donor.value != value
        ]
        if not candidates:
            return {}
        gap = lambda donor: sum(abs(a - b) for a, b in zip(degree, donor.degree))
        minimum = min(gap(donor) for donor in candidates)
        by_graph: dict[int, list[_EdgeDonor]] = {}
        for donor in candidates:
            if gap(donor) == minimum:
                by_graph.setdefault(donor.graph, []).append(donor)
        return {graph_id: tuple(rows) for graph_id, rows in by_graph.items()}

    def sample(
        self, graph: Any, edge: int, count: int, rng: np.random.Generator
    ) -> tuple[_EdgeDonor, ...]:
        eligible = self.eligible(graph, edge)
        if not eligible:
            return ()
        graph_ids = tuple(sorted(eligible))
        draws = []
        for _ in range(int(count)):
            graph_id = graph_ids[int(rng.integers(len(graph_ids)))]
            rows = eligible[graph_id]
            draws.append(rows[int(rng.integers(len(rows)))])
        return tuple(draws)


def _semantic_donor_swap(graph: Any, edge: int, value: float) -> Any:
    """Apply a semantic donor-swap to both directions of an edge."""

    donor_swap = graph.clone()
    donor_swap.orig_edge_value[list(_reciprocal_unit(graph, edge))] = float(value)
    donor_swap.edge_attr = donor_swap.orig_edge_value[:, None]
    return donor_swap


def _structural_donor_swap(graph: Any, source: int, donor: int) -> Any:
    """Apply a structural donor-swap to RRWP and degree features."""

    donor_swap = graph.clone()
    dense = graph.rrwp_dense.clone()
    changed_dense = dense.clone()
    changed_dense[source, :] = dense[donor, :]
    changed_dense[:, source] = dense[:, donor]
    changed_dense[source, source] = dense[donor, donor]
    donor_swap.rrwp_dense = changed_dense
    import torch

    nodes = torch.arange(int(graph.num_nodes), device=changed_dense.device)
    donor_swap.rrwp = changed_dense[nodes, nodes]
    donor_swap.rrwp_val = changed_dense.reshape(-1, changed_dense.shape[-1])
    donor_swap.deg[source] = graph.deg[donor]
    donor_swap.log_deg[source] = graph.log_deg[donor]
    return donor_swap


def _structural_candidates(graph: Any, source: int) -> list[int]:
    """Return nodes with different RRWP or degree features."""

    import torch

    source = int(source)
    rrwp = graph.rrwp_dense
    return [
        donor
        for donor in range(int(graph.num_nodes))
        if donor != source
        and (
            not torch.equal(rrwp[source, :], rrwp[donor, :])
            or not torch.equal(rrwp[:, source], rrwp[:, donor])
            or not torch.equal(graph.deg[source], graph.deg[donor])
            or not torch.equal(graph.log_deg[source], graph.log_deg[donor])
        )
    ]


def _structural_donors(
    graph: Any, source: int, count: int, rng: np.random.Generator
) -> tuple[int, ...]:
    """Sample eligible structural donors uniformly without replacement."""

    candidates = _structural_candidates(graph, int(source))
    size = min(int(count), len(candidates))
    if size == len(candidates):
        return tuple(candidates)
    return tuple(int(value) for value in rng.choice(candidates, size=size, replace=False))


def _clean_head_outputs_and_gradients(model, graph, device):
    import torch
    from torch_geometric.data import Batch

    batch = Batch.from_data_list([graph]).to(device)
    logits, head_outputs = model(batch, capture=True)
    return (
        torch.stack(head_outputs).detach(),
        clean_output_gradients(logits, head_outputs).detach(),
    )


def _donor_swap_head_outputs(model, graphs: Sequence[Any], nodes: int, device):
    import torch
    from torch_geometric.data import Batch

    batch = Batch.from_data_list(list(graphs)).to(device)
    with torch.no_grad():
        _logits, head_outputs = model(batch, capture=True)
    return (
        torch.stack(head_outputs)
        .reshape(
            len(head_outputs),
            len(graphs),
            nodes,
            -1,
            head_outputs[0].shape[-1],
        )
        .permute(1, 0, 2, 3, 4)
    )


def score(
    config: dict[str, Any],
    *,
    checkpoint: Path | None,
    output_dir: Path,
    fast: bool = False,
) -> Path:
    """Compute specialisation scores and distance-resolved score contributions."""

    import torch

    _require_runtime()
    settings = _settings(config, fast=fast)
    output_dir = Path(output_dir).expanduser().resolve()
    if checkpoint is None:
        candidates = sorted((output_dir / "checkpoints").glob("seed_*.pt"))
        if len(candidates) != 1:
            raise ExperimentSetupError(
                "could not choose one GraphBench checkpoint; pass --checkpoint"
            )
        checkpoint = candidates[0]
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_settings = Settings(**payload["settings"])
    try:
        edge_statistics = payload["edge_statistics"]
    except KeyError as exc:
        raise ExperimentSetupError(
            "the checkpoint has no GraphBench edge statistics; retrain it with this repository"
        ) from exc
    splits = _prepared_score_splits(output_dir, config, model_settings, settings, edge_statistics)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model(model_settings).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    rng = np.random.default_rng(settings.score_seed + 2)
    semantic_pool = _EdgeDonorPool(
        splits["train"][: min(settings.semantic_pool_graphs, len(splits["train"]))]
    )
    semantic_results, structural_results = [], []
    for graph_id, graph in enumerate(splits["val"][: settings.score_graphs]):
        clean, clean_gradients = _clean_head_outputs_and_gradients(model, graph, device)
        distances = shortest_path_distances(graph.orig_edge_index, int(graph.num_nodes))

        units = [unit for unit in _edge_units(graph) if semantic_pool.eligible(graph, unit[0])]
        if not units:
            raise RuntimeError(f"validation graph {graph_id} has no eligible semantic source")
        semantic_sources = rng.choice(
            len(units), size=min(settings.sources, len(units)), replace=False
        )
        donor_swaps, source_ids, head_output_row_distances = [], [], []
        for source_position in semantic_sources:
            edge = units[int(source_position)][0]
            left, right = (int(graph.orig_edge_index[axis, edge]) for axis in (0, 1))
            for donor in semantic_pool.sample(graph, edge, settings.donor_swaps_per_source, rng):
                donor_swaps.append(_semantic_donor_swap(graph, edge, donor.value))
                source_ids.append(edge)
                head_output_row_distances.append(np.minimum(distances[left], distances[right]))
        if not donor_swaps:
            raise RuntimeError(f"validation graph {graph_id} has no eligible semantic donor-swaps")
        donor_swap_outputs = _donor_swap_head_outputs(
            model, donor_swaps, int(graph.num_nodes), device
        )
        semantic_results.append(
            compute_channel_score(
                clean,
                donor_swap_outputs,
                clean_gradients,
                [graph_id] * len(donor_swaps),
                source_ids,
                head_output_row_distances=head_output_row_distances,
            )
        )

        eligible_sources = [
            node for node in range(int(graph.num_nodes)) if _structural_candidates(graph, node)
        ]
        structural_sources = rng.choice(
            eligible_sources,
            size=min(settings.sources, len(eligible_sources)),
            replace=False,
        )
        donor_swaps, source_ids, head_output_row_distances = [], [], []
        for source in structural_sources:
            selected = _structural_donors(graph, int(source), settings.donor_swaps_per_source, rng)
            for donor in selected:
                donor_swaps.append(_structural_donor_swap(graph, int(source), int(donor)))
                source_ids.append(int(source))
                head_output_row_distances.append(distances[int(source)])
        if not donor_swaps:
            raise RuntimeError(
                f"validation graph {graph_id} has no eligible structural donor-swaps"
            )
        donor_swap_outputs = _donor_swap_head_outputs(
            model, donor_swaps, int(graph.num_nodes), device
        )
        structural_results.append(
            compute_channel_score(
                clean,
                donor_swap_outputs,
                clean_gradients,
                [graph_id] * len(donor_swaps),
                source_ids,
                head_output_row_distances=head_output_row_distances,
            )
        )

    semantic = mean_channel_scores(semantic_results)
    structural = mean_channel_scores(structural_results)
    categories = distance_categories((semantic, structural))

    layer = np.repeat(np.arange(model_settings.layers), model_settings.heads)
    head = np.tile(np.arange(model_settings.heads), model_settings.layers)
    return save_scores(
        output_dir,
        semantic.score,
        structural.score,
        semantic_distance_contributions=align_distance_contributions(semantic, categories).reshape(
            -1, len(categories)
        ),
        structural_distance_contributions=align_distance_contributions(
            structural, categories
        ).reshape(-1, len(categories)),
        distance_categories=categories,
        layer=layer,
        head=head,
        seed=np.full(layer.shape, int(payload["seed"])),
    )
