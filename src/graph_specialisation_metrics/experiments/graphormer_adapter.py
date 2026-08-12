"""Graphormer PCQM4Mv2 adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from ..runner import clean_output_gradients


@dataclass
class GraphormerGraph:
    """Graphormer inputs for one molecule."""

    x: Any
    input_edges: Any
    attn_bias: Any
    in_degree: Any
    out_degree: Any
    spatial_pos: Any
    attn_edge_type: Any
    edge_index: Any

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    def clone(self) -> GraphormerGraph:
        return GraphormerGraph(
            **{
                name: value.clone() if hasattr(value, "clone") else value
                for name, value in vars(self).items()
            }
        )


def _single_embedding(values: Any, offset: int = 512) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if values.ndim == 1:
        values = values[:, None]
    offsets = 1 + np.arange(values.shape[1], dtype=np.int64) * int(offset)
    return values + offsets


def _floyd_warshall(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Graphormer Floyd–Warshall preprocessing in NumPy."""

    nodes = int(adjacency.shape[0])
    distance = adjacency.astype(np.int64, copy=True)
    path = np.full((nodes, nodes), -1, dtype=np.int64)
    for left in range(nodes):
        for right in range(nodes):
            if left == right:
                distance[left, right] = 0
            elif distance[left, right] == 0:
                distance[left, right] = 510
    for middle in range(nodes):
        candidate = distance[:, middle, None] + distance[middle, None, :]
        changed = candidate < distance
        distance[changed] = candidate[changed]
        path[changed] = middle
    unreachable = distance >= 510
    distance[unreachable] = 510
    path[unreachable] = 510
    return distance, path


def _route(path: np.ndarray, source: int, target: int) -> list[int]:
    middle = int(path[source, target])
    if middle in (-1, 510):
        return []
    return _route(path, source, middle) + [middle] + _route(path, middle, target)


def preprocess_graph(graph: Mapping[str, Any], config: Any) -> GraphormerGraph:
    """Apply Hugging Face Graphormer preprocessing."""

    import torch

    nodes = int(graph["num_nodes"])
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    node_features = np.asarray(graph["node_feat"], dtype=np.int64)
    edge_features = np.asarray(graph["edge_feat"], dtype=np.int64)
    if edge_features.ndim == 1:
        edge_features = edge_features[:, None]
    if edge_index.shape[0] != 2 or edge_index.shape[1] != len(edge_features):
        raise ValueError("OGB edge indices and features do not align")

    adjacency = np.zeros((nodes, nodes), dtype=bool)
    adjacency[edge_index[0], edge_index[1]] = True
    distance, path = _floyd_warshall(adjacency)
    maximum = int(distance.max(initial=0))
    if maximum >= 510:
        raise ValueError("Graphormer PCQM4Mv2 scoring expects connected molecules")

    # These shifts match transformers.models.graphormer.preprocess_item exactly.
    input_nodes = _single_embedding(node_features) + 2
    encoded_edges = _single_embedding(edge_features) + 1
    edge_width = int(edge_features.shape[1])
    attention_edges = np.zeros((nodes, nodes, edge_width), dtype=np.int64)
    attention_edges[edge_index[0], edge_index[1]] = encoded_edges

    maximum_hops = min(maximum, int(getattr(config, "multi_hop_max_dist", maximum)))
    input_edges = np.zeros((nodes, nodes, maximum_hops, edge_width), dtype=np.int64)
    for source in range(nodes):
        for target_node in range(nodes):
            if source == target_node:
                continue
            route = [source, *_route(path, source, target_node), target_node]
            for hop, (left, right) in enumerate(pairwise(route)):
                if hop >= maximum_hops:
                    break
                input_edges[source, target_node, hop] = attention_edges[left, right] + 1

    spatial_pos = distance + 1
    attn_bias = np.zeros((nodes + 1, nodes + 1), dtype=np.float32)
    spatial_limit = int(getattr(config, "spatial_pos_max", 1024))
    attn_bias[1:, 1:][spatial_pos >= spatial_limit] = -np.inf
    degree = adjacency.sum(axis=1, dtype=np.int64) + 1
    return GraphormerGraph(
        x=torch.from_numpy(input_nodes).long(),
        input_edges=torch.from_numpy(input_edges).long(),
        attn_bias=torch.from_numpy(attn_bias),
        in_degree=torch.from_numpy(degree).long(),
        out_degree=torch.from_numpy(degree.copy()).long(),
        spatial_pos=torch.from_numpy(spatial_pos).long(),
        attn_edge_type=torch.from_numpy(attention_edges).long(),
        edge_index=torch.from_numpy(edge_index).long(),
    )


def dataset_graph(dataset: Any, index: int, config: Any) -> GraphormerGraph:
    """Load and preprocess one OGB molecule."""

    item = dataset[int(index)]
    if not isinstance(item, (tuple, list)) or len(item) < 2:
        raise TypeError(f"unexpected PCQM4Mv2 item at index {index}")
    graph_or_smiles = item[0]
    if isinstance(graph_or_smiles, str):
        from ogb.utils import smiles2graph

        graph = smiles2graph(graph_or_smiles)
    elif isinstance(graph_or_smiles, Mapping):
        graph = graph_or_smiles
    else:
        raise TypeError(f"unexpected PCQM4Mv2 graph payload: {type(graph_or_smiles)}")
    return preprocess_graph(graph, config)


def semantic_attributes(dataset: Any, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return atom attributes and degrees for semantic donor sampling."""

    graph_or_smiles = dataset[int(index)][0]
    if isinstance(graph_or_smiles, str):
        from ogb.utils import smiles2graph

        graph = smiles2graph(graph_or_smiles)
    else:
        graph = graph_or_smiles
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    degree = np.bincount(edge_index[0], minlength=int(graph["num_nodes"]))
    return _single_embedding(graph["node_feat"]) + 2, degree


class _HeadHooks:
    def __init__(self, model: Any):
        self.layers = model.encoder.graph_encoder.layers
        self.values: list[Any | None] = [None] * len(self.layers)
        self.handles: list[Any] = []

    def _hook(self, layer_index: int):
        def hook(_module: Any, arguments: tuple[Any, ...]):
            value = arguments[0]
            attention = self.layers[layer_index].self_attn
            recorded = value.view(
                value.shape[0], value.shape[1], attention.num_heads, attention.head_dim
            )
            import torch

            if not recorded.requires_grad and torch.is_grad_enabled():
                recorded = recorded.detach().requires_grad_(True)
            self.values[layer_index] = recorded
            # Keep the recorded head output on the model-output path.
            return (recorded.reshape_as(value), *arguments[1:])

        return hook

    def __enter__(self) -> _HeadHooks:  # noqa: PYI034
        for index, layer in enumerate(self.layers):
            self.handles.append(
                layer.self_attn.out_proj.register_forward_pre_hook(self._hook(index))
            )
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self.handles:
            handle.remove()

    def head_outputs(self) -> tuple[Any, ...]:
        if any(value is None for value in self.values):
            raise RuntimeError("a Graphormer head hook did not fire")
        return tuple(self.values)  # type: ignore[return-value]


class GraphormerAdapter:
    """Record each head output before heads are combined."""

    def __init__(self, model: Any, device: Any):
        import torch

        self.model = model.to(device).eval()
        self.device = torch.device(device)

    def _batch(self, graphs: Sequence[GraphormerGraph]) -> dict[str, Any]:
        import torch

        if not graphs:
            raise ValueError("a Graphormer batch cannot be empty")
        maximum_nodes = max(graph.num_nodes for graph in graphs)
        maximum_hops = max(int(graph.input_edges.shape[2]) for graph in graphs)
        exemplar = graphs[0]
        batch = len(graphs)
        values = {
            "input_nodes": torch.zeros(
                batch, maximum_nodes, exemplar.x.shape[1], dtype=exemplar.x.dtype
            ),
            "input_edges": torch.zeros(
                batch,
                maximum_nodes,
                maximum_nodes,
                maximum_hops,
                exemplar.input_edges.shape[-1],
                dtype=exemplar.input_edges.dtype,
            ),
            "attn_bias": torch.zeros(batch, maximum_nodes + 1, maximum_nodes + 1),
            "in_degree": torch.zeros(batch, maximum_nodes, dtype=exemplar.in_degree.dtype),
            "out_degree": torch.zeros(batch, maximum_nodes, dtype=exemplar.out_degree.dtype),
            "spatial_pos": torch.zeros(batch, maximum_nodes, maximum_nodes, dtype=torch.long),
            "attn_edge_type": torch.zeros(
                batch,
                maximum_nodes,
                maximum_nodes,
                exemplar.attn_edge_type.shape[-1],
                dtype=torch.long,
            ),
        }
        for position, graph in enumerate(graphs):
            nodes, hops = graph.num_nodes, int(graph.input_edges.shape[2])
            values["input_nodes"][position, :nodes] = graph.x
            values["input_edges"][position, :nodes, :nodes, :hops] = graph.input_edges
            values["attn_bias"][position, : nodes + 1, : nodes + 1] = graph.attn_bias
            values["in_degree"][position, :nodes] = graph.in_degree
            values["out_degree"][position, :nodes] = graph.out_degree
            values["spatial_pos"][position, :nodes, :nodes] = graph.spatial_pos
            values["attn_edge_type"][position, :nodes, :nodes] = graph.attn_edge_type
        return {name: value.to(self.device) for name, value in values.items()}

    def clean(self, graph: GraphormerGraph, scale: Any) -> tuple[Any, Any]:
        """Return clean head outputs and output gradients."""

        import torch

        self.model.zero_grad(set_to_none=True)
        with _HeadHooks(self.model) as hooks:
            logits = self.model(**self._batch([graph]), return_dict=True).logits[0].reshape(-1)
        head_outputs = hooks.head_outputs()
        gradients = clean_output_gradients(logits, head_outputs, scale)[:, :, :, 0]
        clean_head_outputs = torch.stack([value[:, 0] for value in head_outputs], dim=0)
        return clean_head_outputs.detach().cpu(), gradients.detach().cpu()

    def donor_swap_head_outputs(
        self, graphs: Sequence[GraphormerGraph], batch_size: int = 8
    ) -> Any:
        """Return donor-swap head outputs."""

        import torch

        rows = []
        for start in range(0, len(graphs), max(1, int(batch_size))):
            batch = graphs[start : start + max(1, int(batch_size))]
            with torch.no_grad(), _HeadHooks(self.model) as hooks:
                self.model(**self._batch(batch), return_dict=True)
            head_outputs = torch.stack(
                [value.permute(1, 0, 2, 3) for value in hooks.head_outputs()], dim=1
            )
            rows.append(head_outputs.cpu())
        if not rows:
            raise ValueError("at least one donor-swap graph is required")
        return torch.cat(rows, dim=0)


def load_public_model(
    model_id: str,
    revision: str,
    *,
    device: Any,
) -> Any:
    """Load the public Graphormer checkpoint."""

    from transformers import GraphormerForGraphClassification

    model = GraphormerForGraphClassification.from_pretrained(
        model_id,
        revision=revision,
        use_safetensors=True,
    )
    return model.to(device).eval()


__all__ = [
    "GraphormerAdapter",
    "GraphormerGraph",
    "dataset_graph",
    "load_public_model",
    "preprocess_graph",
    "semantic_attributes",
]
