"""Graphormer PCQM4Mv2 scoring."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..distance import shortest_path_distances
from ..interventions import semantic_donor_swap, structural_donor_swap
from ..runner import (
    align_distance_contributions,
    compute_channel_score,
    distance_categories,
    mean_channel_scores,
    training_target_scale,
)
from ..sampling import (
    SemanticDonorPool,
    analysis_indices,
    eligible_structural_donors,
    sample_structural_donors,
)
from . import ExperimentSetupError, save_scores
from .graphormer_adapter import (
    GraphormerAdapter,
    GraphormerGraph,
    dataset_graph,
    load_public_model,
    semantic_attributes,
)

REQUIRED = {"transformers": "transformers", "ogb": "ogb", "rdkit": "rdkit"}


def _require_dependencies() -> None:
    missing = [
        label for module, label in REQUIRED.items() if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise ExperimentSetupError(
            "Graphormer scoring dependencies are missing: "
            + ", ".join(missing)
            + ". Install this project with the graphormer extra."
        )


@dataclass
class _Assets:
    model: Any
    dataset: Any
    split: dict[str, Any]
    device: Any


def _ogb_split(dataset: Any) -> dict[str, Any]:
    """Load the OGB split file."""

    import torch

    path = Path(dataset.folder) / "split_dict.pt"
    if path.is_file():
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - old PyTorch
            return torch.load(path, map_location="cpu")
    return dataset.get_idx_split()


def _load_assets(config: dict[str, Any]) -> _Assets:
    import torch
    from ogb.lsc import PCQM4Mv2Dataset

    data = config.get("data", {})
    model_config = config.get("model", {})
    scoring = config.get("scoring", {})
    device = torch.device(
        str(
            config.get(
                "device",
                scoring.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            )
        )
    )
    model = load_public_model(
        str(model_config["id"]),
        str(model_config["revision"]),
        device=device,
    )
    root = Path(data.get("root", ".cache/pcqm4mv2")).expanduser()
    dataset = PCQM4Mv2Dataset(root=str(root), only_smiles=True)
    return _Assets(model, dataset, _ogb_split(dataset), device)


def _raw_degrees(graph: GraphormerGraph) -> np.ndarray:
    edges = graph.edge_index.detach().cpu().numpy()
    return np.bincount(edges[0], minlength=graph.num_nodes)


def _structural_fields(graph: GraphormerGraph) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {"in_degree": graph.in_degree, "out_degree": graph.out_degree},
        {
            "spatial_pos": graph.spatial_pos,
            "attn_edge_type": graph.attn_edge_type,
            "input_edges": graph.input_edges,
        },
    )


def _structural_profiles(graph: GraphormerGraph) -> list[tuple[Any, ...]]:
    node_fields, pair_fields = _structural_fields(graph)

    def array(value: Any) -> np.ndarray:
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    nodes = {name: array(value) for name, value in node_fields.items()}
    pairs = {name: array(value) for name, value in pair_fields.items()}
    return [
        tuple(
            [value[node].copy() for value in nodes.values()]
            + [
                part.copy()
                for value in pairs.values()
                for part in (value[node, :], value[:, node], value[node, node])
            ]
        )
        for node in range(graph.num_nodes)
    ]


def _semantic_donor_swap(graph: GraphormerGraph, source: int, attributes: Any) -> GraphormerGraph:
    donor_swap = graph.clone()
    donor_swap.x = semantic_donor_swap(graph.x, source, attributes)
    return donor_swap


def _structural_donor_swap(graph: GraphormerGraph, source: int, donor: int) -> GraphormerGraph:
    node_fields, pair_fields = structural_donor_swap(*_structural_fields(graph), source, donor)
    donor_swap = graph.clone()
    for name, value in {**node_fields, **pair_fields}.items():
        setattr(donor_swap, name, value)
    return donor_swap


def _graph_channel_score(
    adapter: GraphormerAdapter,
    graph: GraphormerGraph,
    graph_id: int,
    sources: np.ndarray,
    donor_swaps_per_source: int,
    rng: np.random.Generator,
    clean: Any,
    clean_gradients: Any,
    *,
    semantic_pool: SemanticDonorPool | None,
    batch_size: int,
):
    donor_swaps: list[GraphormerGraph] = []
    source_ids: list[int] = []
    distances: list[list[Any]] = []
    graph_distances = shortest_path_distances(graph.edge_index, graph.num_nodes)
    degrees = _raw_degrees(graph)
    structural_profiles = _structural_profiles(graph)
    for source_value in sources:
        source = int(source_value)
        if semantic_pool is not None:
            selected = semantic_pool.sample(
                graph.x[source].detach().cpu().numpy(),
                int(degrees[source]),
                donor_swaps_per_source,
                rng,
            )
            source_swaps = [
                _semantic_donor_swap(graph, source, donor.attributes) for donor in selected
            ]
        else:
            selected = sample_structural_donors(
                structural_profiles, source, donor_swaps_per_source, rng
            )
            source_swaps = [_structural_donor_swap(graph, source, int(donor)) for donor in selected]
        donor_swaps.extend(source_swaps)
        source_ids.extend([source] * len(source_swaps))
        distances.extend([["graph-token", *graph_distances[source].tolist()]] * len(source_swaps))
    donor_swap_outputs = adapter.donor_swap_head_outputs(donor_swaps, batch_size=batch_size)
    return compute_channel_score(
        clean,
        donor_swap_outputs,
        clean_gradients,
        [graph_id] * len(donor_swaps),
        source_ids,
        head_output_row_distances=np.asarray(distances, dtype=object),
    )


def _target_scale(dataset: Any, train_indices: np.ndarray) -> np.ndarray:
    if hasattr(dataset, "labels"):
        targets = np.asarray(dataset.labels)[train_indices]
    else:
        targets = np.asarray([dataset[int(index)][1] for index in train_indices])
    return training_target_scale(targets)


def _run(
    config: dict[str, Any],
    assets: _Assets,
    output_dir: Path,
    *,
    fast: bool,
) -> Path:
    scoring = config.get("scoring", {})
    data = config.get("data", {})
    analysis_seed = int(scoring.get("seed", 31415))
    rng = np.random.default_rng(analysis_seed + 2)
    graphs = min(int(scoring.get("graphs", 48)), 1 if fast else 10**9)
    sources_per_graph = min(int(scoring.get("sources_per_graph", 6)), 1 if fast else 10**9)
    donor_swaps_per_source = min(
        int(scoring.get("donor_swaps_per_source", 8)), 1 if fast else 10**9
    )
    semantic_pool_size = min(int(scoring.get("semantic_pool_graphs", 2000)), 8 if fast else 10**9)
    donor_swap_batch_size = int(scoring.get("donor_swap_batch_size", 8))
    if (
        min(
            graphs,
            sources_per_graph,
            donor_swaps_per_source,
            semantic_pool_size,
            donor_swap_batch_size,
        )
        < 1
    ):
        raise ValueError("Graphormer scoring counts must all be positive")

    train_ids = np.asarray(
        assets.split[str(data.get("semantic_pool_split", "train"))], dtype=np.int64
    )
    valid_ids = np.asarray(assets.split[str(data.get("evaluation_split", "valid"))], dtype=np.int64)
    scale = _target_scale(assets.dataset, train_ids)
    try:
        graph_positions, semantic_pool_positions = analysis_indices(
            len(valid_ids), graphs, len(train_ids), semantic_pool_size, analysis_seed
        )
    except ValueError as exc:
        raise ExperimentSetupError(str(exc)) from exc
    selected_graphs = valid_ids[graph_positions]
    semantic_pool_ids = train_ids[semantic_pool_positions]
    attributes: dict[int, np.ndarray] = {}
    degrees: dict[int, np.ndarray] = {}
    for graph_id in semantic_pool_ids:
        attributes[int(graph_id)], degrees[int(graph_id)] = semantic_attributes(
            assets.dataset, int(graph_id)
        )
    semantic_pool = SemanticDonorPool(attributes, degrees)
    adapter = GraphormerAdapter(assets.model, assets.device)

    semantic_results, structural_results, used_graphs = [], [], []
    for graph_id_value in selected_graphs:
        graph_id = int(graph_id_value)
        graph = dataset_graph(assets.dataset, graph_id, assets.model.config)
        structural_profiles = _structural_profiles(graph)
        eligible_sources = np.asarray(
            [
                source
                for source in range(graph.num_nodes)
                if len(eligible_structural_donors(structural_profiles, source))
            ],
            dtype=np.int64,
        )
        if not len(eligible_sources):
            continue
        source_count = min(sources_per_graph, len(eligible_sources))
        sources = np.sort(rng.choice(eligible_sources, size=source_count, replace=False))
        clean, clean_gradients = adapter.clean(graph, scale)
        semantic_results.append(
            _graph_channel_score(
                adapter,
                graph,
                graph_id,
                sources,
                donor_swaps_per_source,
                rng,
                clean,
                clean_gradients,
                semantic_pool=semantic_pool,
                batch_size=donor_swap_batch_size,
            )
        )
        structural_results.append(
            _graph_channel_score(
                adapter,
                graph,
                graph_id,
                sources,
                donor_swaps_per_source,
                rng,
                clean,
                clean_gradients,
                semantic_pool=None,
                batch_size=donor_swap_batch_size,
            )
        )
        used_graphs.append(graph_id)
        if len(used_graphs) == graphs:
            break
    if len(used_graphs) != graphs:
        raise ExperimentSetupError(
            f"only {len(used_graphs)} validation graphs had eligible structural donors; "
            f"{graphs} were requested"
        )

    semantic = mean_channel_scores(semantic_results)
    structural = mean_channel_scores(structural_results)
    categories = distance_categories((semantic, structural))
    layers, heads = semantic.score.shape
    return save_scores(
        output_dir,
        semantic.score.reshape(-1),
        structural.score.reshape(-1),
        semantic_distance_contributions=align_distance_contributions(semantic, categories).reshape(
            semantic.score.size, -1
        ),
        structural_distance_contributions=align_distance_contributions(
            structural, categories
        ).reshape(structural.score.size, -1),
        distance_categories=categories,
        graph_ids=np.asarray(used_graphs, dtype=np.int64),
        model_revision=np.asarray(config["model"]["revision"]),
        layer=np.repeat(np.arange(layers), heads),
        head=np.tile(np.arange(heads), layers),
    )


def score(
    config: dict[str, Any],
    *,
    checkpoint: Path | None,
    output_dir: Path,
    fast: bool = False,
) -> Path:
    """Compute specialisation scores and distance-resolved score contributions."""

    model = config.get("model", {})
    if not model.get("id") or not model.get("revision"):
        raise ValueError("Graphormer config requires model.id and model.revision")
    if checkpoint is not None:
        raise ValueError("Graphormer uses the configured public checkpoint; omit --checkpoint")
    _require_dependencies()
    return _run(config, _load_assets(config), Path(output_dir), fast=fast)
