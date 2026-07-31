"""Official Graphormer runtime, preprocessing, and canonical backend adapter.

Only model-specific mechanics live here. Scientific interventions, score aggregation,
causal validation, and carriage estimators remain in the shared methodology package.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from .audit import audit_check
from .backend import BackendCapture, CleanJacobians
from .protocol import stable_hash

def install_graphormer_dependencies() -> None:
    """Install the checkpoint-compatible Graphormer analysis stack in Colab."""

    if sys.version_info >= (3, 13):
        raise RuntimeError(
            "the official Graphormer checkpoint stack pins transformers==4.40.2/"
            "tokenizers==0.19.1 and requires Python 3.10-3.12"
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "transformers==4.40.2",
            "tokenizers==0.19.1",
            "huggingface-hub==0.36.1",
            "ogb==1.3.6",
            "rdkit",
        ],
        check=True,
    )


@dataclass
class GraphormerGraph:
    """One unpadded molecular graph in Hugging Face Graphormer input coordinates."""

    x: Any
    input_edges: Any
    attn_bias: Any
    in_degree: Any
    out_degree: Any
    spatial_pos: Any
    attn_edge_type: Any
    edge_index: Any
    edge_attr: Any
    y: Any
    smiles: str

    @property
    def input_nodes(self):
        return self.x

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def keys(self) -> tuple[str, ...]:
        return (
            "x",
            "input_edges",
            "attn_bias",
            "in_degree",
            "out_degree",
            "spatial_pos",
            "attn_edge_type",
            "edge_index",
            "edge_attr",
            "y",
            "smiles",
            "num_nodes",
        )

    def clone(self) -> "GraphormerGraph":
        import torch

        values = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            values[name] = value.clone() if isinstance(value, torch.Tensor) else value
        return GraphormerGraph(**values)


def _convert_to_single_emb(value: np.ndarray, offset: int = 512) -> np.ndarray:
    value = np.asarray(value, dtype=np.int64)
    result = value.copy()
    for column in range(int(result.shape[1])):
        result[:, column] += 1 + int(offset) * column
    return result


def _undirected_adjacency(edge_index: np.ndarray, edge_features: np.ndarray, n: int):
    adjacency: dict[int, list[tuple[int, np.ndarray]]] = {node: [] for node in range(int(n))}
    for position in range(int(edge_index.shape[1])):
        left, right = int(edge_index[0, position]), int(edge_index[1, position])
        feature = edge_features[position]
        adjacency[left].append((right, feature))
        adjacency[right].append((left, feature))
    return adjacency


def _all_pairs_shortest_paths(adjacency, n: int):
    infinity = 10**9
    distance = np.full((n, n), infinity, dtype=np.int64)
    parent = np.full((n, n), -1, dtype=np.int64)
    parent_feature = np.empty((n, n), dtype=object)
    for source in range(n):
        distance[source, source] = 0
        queue = [source]
        cursor = 0
        while cursor < len(queue):
            node = queue[cursor]
            cursor += 1
            for neighbour, feature in adjacency.get(node, ()):
                if distance[source, neighbour] > distance[source, node] + 1:
                    distance[source, neighbour] = distance[source, node] + 1
                    parent[source, neighbour] = node
                    parent_feature[source, neighbour] = feature
                    queue.append(neighbour)
    return distance, parent, parent_feature


def _path_features(parent, parent_feature, source: int, target: int):
    if source == target or int(parent[source, target]) < 0:
        return []
    result = []
    cursor = target
    while cursor != source:
        previous = int(parent[source, cursor])
        if previous < 0:
            return []
        result.append(parent_feature[source, cursor])
        cursor = previous
    result.reverse()
    return result


def graphormer_graph_from_ogb(
    graph: Mapping[str, Any],
    target: Any,
    *,
    config: Any,
    smiles: str = "",
    offset: int = 512,
) -> GraphormerGraph:
    """Reproduce the working PCQM preprocessing used by the reference Colab."""

    import torch

    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    node_features = np.asarray(graph["node_feat"], dtype=np.int64)
    edge_features = np.asarray(graph["edge_feat"], dtype=np.int64)
    n = int(graph["num_nodes"])
    edge_width = int(edge_features.shape[1])
    max_hops = int(getattr(config, "multi_hop_max_dist", 5))
    spatial_pos_max = int(getattr(config, "spatial_pos_max", 20))
    model_max_dist = int(getattr(config, "max_dist", max_hops))

    input_nodes = torch.from_numpy(_convert_to_single_emb(node_features, offset)).long()
    encoded_edges = _convert_to_single_emb(edge_features, offset)
    in_degree = np.zeros(n, dtype=np.int64)
    out_degree = np.zeros(n, dtype=np.int64)
    for sender, receiver in zip(edge_index[0], edge_index[1]):
        out_degree[int(sender)] += 1
        in_degree[int(receiver)] += 1

    adjacency = _undirected_adjacency(edge_index, encoded_edges, n)
    distance, parent, parent_feature = _all_pairs_shortest_paths(adjacency, n)
    clipped = np.minimum(distance, spatial_pos_max) + 1
    clipped = np.where(clipped > model_max_dist, model_max_dist + 1, clipped)
    spatial_pos = torch.from_numpy(clipped).long().clamp_min(0)

    attn_edge_type = torch.zeros((n, n, edge_width), dtype=torch.long)
    input_edges = torch.zeros((n, n, max_hops, edge_width), dtype=torch.long)
    for position in range(int(edge_index.shape[1])):
        sender, receiver = int(edge_index[0, position]), int(edge_index[1, position])
        attn_edge_type[sender, receiver] = torch.from_numpy(encoded_edges[position]).long()
    for source in range(n):
        for target_node in range(n):
            features = _path_features(parent, parent_feature, source, target_node)
            for hop, feature in enumerate(features[:max_hops]):
                input_edges[source, target_node, hop] = torch.from_numpy(feature).long()

    label = torch.as_tensor(target, dtype=torch.float32).reshape(-1)
    return GraphormerGraph(
        x=input_nodes,
        input_edges=input_edges,
        attn_bias=torch.zeros((n + 1, n + 1), dtype=torch.float32),
        in_degree=torch.from_numpy(in_degree + 1).long().clamp(max=512),
        out_degree=torch.from_numpy(out_degree + 1).long().clamp(max=512),
        spatial_pos=spatial_pos,
        attn_edge_type=attn_edge_type,
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(edge_features).long(),
        y=label,
        smiles=str(smiles),
    )


class PCQMGraphormerDataset:
    """Lazy split view over OGB PCQM4Mv2 with bounded preprocessing cache."""

    def __init__(self, dataset: Any, indices: Sequence[int], config: Any):
        self.dataset = dataset
        self.indices = tuple(int(value) for value in indices)
        self.config = config

    def __len__(self) -> int:
        return len(self.indices)

    @lru_cache(maxsize=2048)
    def _get(self, position: int) -> GraphormerGraph:
        from ogb.utils import smiles2graph

        dataset_index = self.indices[int(position)]
        item = self.dataset[dataset_index]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise TypeError(f"unexpected PCQM item at {dataset_index}: {type(item)}")
        graph_or_smiles, target = item[0], item[1]
        if isinstance(graph_or_smiles, str):
            smiles = graph_or_smiles
            graph = smiles2graph(smiles)
        elif isinstance(graph_or_smiles, Mapping):
            graph = graph_or_smiles
            smiles = f"pcqm4mv2:{dataset_index}"
        else:
            raise TypeError(
                f"unexpected PCQM graph payload at {dataset_index}: {type(graph_or_smiles)}"
            )
        return graphormer_graph_from_ogb(
            graph,
            target,
            config=self.config,
            smiles=smiles,
        )

    def __getitem__(self, position: int) -> GraphormerGraph:
        return self._get(int(position)).clone()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_files_from_index(index: Path, resolve) -> tuple[Path, ...]:
    record = json.loads(index.read_text())
    names = sorted(set(record.get("weight_map", {}).values()))
    if not names:
        raise RuntimeError(f"Graphormer checkpoint index has no weight_map: {index}")
    return tuple(Path(resolve(name)) for name in names)


def _checkpoint_digest(files: Sequence[Path]) -> str:
    files = tuple(Path(path) for path in files)
    if len(files) == 1:
        return _sha256(files[0])
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _local_hf_weight_files(directory: Path) -> tuple[Path, ...]:
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = directory / name
        if path.is_file():
            return (path,)
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = directory / name
        if index.is_file():
            return _checkpoint_files_from_index(index, lambda shard: directory / shard)
    return ()


def _hub_weight_files(
    model_id: str,
    revision: str | None,
    cache_dir: str | None,
    local_files_only: bool,
) -> tuple[Path, ...]:
    from huggingface_hub import hf_hub_download

    arguments = {
        "repo_id": model_id,
        "revision": revision,
        "cache_dir": cache_dir,
        "local_files_only": bool(local_files_only),
    }
    for name in ("model.safetensors", "pytorch_model.bin"):
        try:
            return (Path(hf_hub_download(filename=name, **arguments)),)
        except (OSError, ValueError):
            pass
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        try:
            index = Path(hf_hub_download(filename=name, **arguments))
        except (OSError, ValueError):
            continue
        return _checkpoint_files_from_index(
            index,
            lambda shard: hf_hub_download(filename=shard, **arguments),
        )
    raise RuntimeError(f"could not resolve checkpoint weight files for {model_id}@{revision}")


def _unwrap_state_dict(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"checkpoint must contain a state dict, got {type(value)}")
    for key in ("state_dict", "model"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            return nested
    return value


def _official_to_hf_state(state: Mapping[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in state.items():
        if key == "encoder.embed_out.weight":
            converted["classifier.classifier.weight"] = value
        elif key == "encoder.lm_output_learned_bias":
            converted["classifier.lm_output_learned_bias"] = value
        elif key.startswith("encoder.masked_lm_pooler"):
            continue
        else:
            converted[key] = value
    return converted


def load_graphormer_model(
    spec: Any,
    *,
    checkpoint: str | None,
    device: Any,
    model_id: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
):
    """Load a public HF checkpoint or overlay a local HF/Fairseq state file."""

    import torch
    from transformers import GraphormerForGraphClassification

    model_id = str(model_id or spec.model_id)
    revision = spec.revision if revision is None else revision
    source = Path(checkpoint).expanduser() if checkpoint else None
    load_source = str(source) if source is not None and source.is_dir() else model_id
    model, loading_info = GraphormerForGraphClassification.from_pretrained(
        load_source,
        revision=None if source is not None and source.is_dir() else revision,
        cache_dir=cache_dir,
        local_files_only=bool(local_files_only),
        use_safetensors=None,
        output_loading_info=True,
    )
    if loading_info.get("mismatched_keys"):
        raise RuntimeError(f"Graphormer checkpoint shape mismatch: {loading_info['mismatched_keys']}")

    if source is not None and source.is_file():
        if source.suffix == ".safetensors":
            from safetensors.torch import load_file

            raw = load_file(str(source), device="cpu")
        else:
            raw = torch.load(str(source), map_location="cpu")
        state = _official_to_hf_state(_unwrap_state_dict(raw))
        target = model.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in target and tuple(value.shape) == tuple(target[key].shape)
        }
        encoder_target = [key for key in target if key.startswith("encoder.")]
        encoder_loaded = [key for key in compatible if key.startswith("encoder.")]
        coverage = len(encoder_loaded) / max(len(encoder_target), 1)
        if coverage < 0.90:
            raise RuntimeError(
                f"local Graphormer checkpoint loaded only {coverage:.1%} of encoder tensors"
            )
        if "classifier.classifier.weight" not in compatible:
            raise RuntimeError("local Graphormer checkpoint does not contain a compatible output head")
        model.load_state_dict(compatible, strict=False)
        descriptor = str(source.resolve())
        digest = _sha256(source)
    elif source is not None and source.is_dir():
        descriptor = str(source.resolve())
        files = _local_hf_weight_files(source)
        digest = (
            _checkpoint_digest(files)
            if files
            else stable_hash({"directory": descriptor})
        )
    else:
        descriptor = f"{model_id}@{revision or 'default'}"
        digest = _checkpoint_digest(
            _hub_weight_files(
                model_id,
                revision,
                cache_dir,
                bool(local_files_only),
            )
        )
    return model.to(device).eval(), descriptor, digest


class GraphormerRuntime:
    """Dataset/model container exposing the geometry expected by canonical orchestration."""

    def __init__(
        self,
        model: Any,
        eval_ds: Any,
        donor_ds: Any,
        *,
        device: Any,
        seed: int,
        metric_fn: Any,
    ):
        layers = model.encoder.graph_encoder.layers
        self.model = model
        self.eval_ds = eval_ds
        self.donor_ds = donor_ds
        self.device = device
        self.sc = SimpleNamespace(seed=int(seed))
        self.L = len(layers)
        self.H = int(layers[0].self_attn.num_heads)
        self.dh = int(layers[0].self_attn.head_dim)
        self.dim_h = int(model.config.embedding_dim)
        self.metric_fn = metric_fn
        self.cfg = SimpleNamespace(model=SimpleNamespace(graph_pooling="graph_token"))
        self.test_metric = None
        self.val_metric = None
        self.checks = {
            "num_parameters": sum(int(parameter.numel()) for parameter in model.parameters())
        }


class _TransportHooks:
    def __init__(
        self,
        model: Any,
        *,
        ablate: Mapping[int, Sequence[int]] | None = None,
        replacements: Sequence[Any] | None = None,
    ):
        self.model = model
        self.ablate = {int(key): tuple(int(value) for value in values) for key, values in (ablate or {}).items()}
        self.replacements = replacements
        self.values: list[Any | None] = [None] * len(model.encoder.graph_encoder.layers)
        self.handles = []

    def _hook(self, layer_index: int):
        def hook(_module, args):
            value = args[0]
            layer = self.model.encoder.graph_encoder.layers[layer_index]
            heads = int(layer.self_attn.num_heads)
            tokens, batch, width = value.shape
            routed = value.view(tokens, batch, heads, width // heads)
            changed = routed
            selected = self.ablate.get(layer_index, ())
            if selected:
                changed = changed.clone()
                changed[:, :, list(selected), :] = 0.0
            if self.replacements is not None and selected:
                donor = self.replacements[layer_index].to(
                    device=changed.device, dtype=changed.dtype
                )
                if tuple(donor.shape) != tuple(changed.shape):
                    raise RuntimeError(
                        f"Graphormer patch geometry differs at layer {layer_index}: "
                        f"{tuple(donor.shape)} vs {tuple(changed.shape)}"
                    )
                if changed is routed:
                    changed = changed.clone()
                changed[:, :, list(selected), :] = donor[:, :, list(selected), :]
            flattened = changed.reshape_as(value)
            # Keep the exact head-shaped tensor that feeds the flattened out-projection
            # input. Creating a new view only after the forward would not itself lie on
            # the prediction graph and therefore could not be differentiated.
            self.values[layer_index] = changed
            if flattened is not value:
                return (flattened, *args[1:])
            return None

        return hook

    def __enter__(self):
        for index, layer in enumerate(self.model.encoder.graph_encoder.layers):
            self.handles.append(
                layer.self_attn.out_proj.register_forward_pre_hook(self._hook(index))
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class _IndividualTransportHooks:
    """Patch or ablate one independently assigned head in each batch replica."""

    def __init__(
        self,
        model: Any,
        assignments: Sequence[tuple[int, int]],
        *,
        replacements: Sequence[Any] | None = None,
        ablate: bool = False,
    ):
        import torch

        if not assignments:
            raise ValueError("individual Graphormer head assignments cannot be empty")
        self.model = model
        self.assignments = torch.as_tensor(assignments, dtype=torch.long)
        self.replacements = replacements
        self.ablate = bool(ablate)
        self.handles = []

        layers = len(model.encoder.graph_encoder.layers)
        heads = int(model.encoder.graph_encoder.layers[0].self_attn.num_heads)
        if bool((self.assignments[:, 0] < 0).any()) or bool(
            (self.assignments[:, 0] >= layers).any()
        ):
            raise IndexError("individual Graphormer layer assignment is outside the model")
        if bool((self.assignments[:, 1] < 0).any()) or bool(
            (self.assignments[:, 1] >= heads).any()
        ):
            raise IndexError("individual Graphormer head assignment is outside the model")
        if not self.ablate and (
            replacements is None or len(replacements) != layers
        ):
            raise ValueError(
                "individual Graphormer patching requires one replacement tensor per layer"
            )

    def _hook(self, layer_index: int):
        def hook(_module, args):
            value = args[0]
            layer = self.model.encoder.graph_encoder.layers[layer_index]
            heads = int(layer.self_attn.num_heads)
            tokens, batch, width = value.shape
            if batch != int(self.assignments.shape[0]):
                raise RuntimeError(
                    "individual Graphormer assignments do not match the forward batch"
                )
            routed = value.view(tokens, batch, heads, width // heads)
            selected = __import__("torch").nonzero(
                self.assignments[:, 0] == int(layer_index), as_tuple=False
            ).reshape(-1)
            if not int(selected.numel()):
                return None
            selected = selected.to(device=routed.device)
            selected_heads = self.assignments[selected.cpu(), 1].to(
                device=routed.device
            )
            changed = routed.clone()
            if self.ablate:
                changed[:, selected, selected_heads, :] = 0.0
            else:
                donor = self.replacements[layer_index].to(
                    device=changed.device, dtype=changed.dtype
                )
                if tuple(donor.shape) != tuple(changed.shape):
                    raise RuntimeError(
                        f"individual Graphormer patch geometry differs at layer "
                        f"{layer_index}: {tuple(donor.shape)} vs {tuple(changed.shape)}"
                    )
                changed[:, selected, selected_heads, :] = donor[
                    :, selected, selected_heads, :
                ]
            return (changed.reshape_as(value), *args[1:])

        return hook

    def __enter__(self):
        for index, layer in enumerate(self.model.encoder.graph_encoder.layers):
            self.handles.append(
                layer.self_attn.out_proj.register_forward_pre_hook(self._hook(index))
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class GraphormerBackend:
    """Canonical backend at Graphormer's native routed-value and graph-token sites."""

    def __init__(self, runtime: GraphormerRuntime, task: Any, sigma: Sequence[float]):
        self.runtime = runtime
        self.model = runtime.model
        self.task = task
        self.sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)

    @property
    def geometry(self) -> dict[str, int]:
        return {
            "layers": int(self.runtime.L),
            "heads": int(self.runtime.H),
            "head_width": int(self.runtime.dh),
            "hidden_width": int(self.runtime.dim_h),
            "outputs": int(len(self.sigma)),
        }

    @property
    def special_carrier_labels(self) -> tuple[str, ...]:
        return ("graph_token",)

    def _z(self, prediction):
        return self.task.output.transform(prediction, self.sigma)

    def _batch(self, data_list: Sequence[GraphormerGraph]):
        import torch

        if not data_list:
            raise ValueError("capture requires at least one graph")
        maximum = max(int(data.num_nodes) for data in data_list)
        batch_size = len(data_list)
        exemplar = data_list[0]
        result = {
            "input_nodes": torch.zeros(
                batch_size,
                maximum,
                *exemplar.x.shape[1:],
                dtype=exemplar.x.dtype,
            ),
            "input_edges": torch.zeros(
                batch_size,
                maximum,
                maximum,
                *exemplar.input_edges.shape[2:],
                dtype=exemplar.input_edges.dtype,
            ),
            "attn_bias": torch.zeros(
                batch_size,
                maximum + 1,
                maximum + 1,
                dtype=exemplar.attn_bias.dtype,
            ),
            "in_degree": torch.zeros(
                batch_size, maximum, dtype=exemplar.in_degree.dtype
            ),
            "out_degree": torch.zeros(
                batch_size, maximum, dtype=exemplar.out_degree.dtype
            ),
            "spatial_pos": torch.zeros(
                batch_size,
                maximum,
                maximum,
                dtype=exemplar.spatial_pos.dtype,
            ),
            "attn_edge_type": torch.zeros(
                batch_size,
                maximum,
                maximum,
                *exemplar.attn_edge_type.shape[2:],
                dtype=exemplar.attn_edge_type.dtype,
            ),
        }
        for index, data in enumerate(data_list):
            nodes = int(data.num_nodes)
            result["input_nodes"][index, :nodes] = data.x
            result["input_edges"][index, :nodes, :nodes] = data.input_edges
            result["attn_bias"][index, : nodes + 1, : nodes + 1] = data.attn_bias
            result["in_degree"][index, :nodes] = data.in_degree
            result["out_degree"][index, :nodes] = data.out_degree
            result["spatial_pos"][index, :nodes, :nodes] = data.spatial_pos
            result["attn_edge_type"][index, :nodes, :nodes] = data.attn_edge_type
        result = {
            name: value.to(self.runtime.device)
            for name, value in result.items()
        }
        target = torch.stack([data.y.reshape(-1) for data in data_list]).to(
            self.runtime.device
        )
        return result, target

    def capture(
        self,
        data_list: Sequence[GraphormerGraph],
        *,
        require_grad: bool,
        include_virtual_transport: bool = True,
        family: Sequence[tuple[int, int]] = (),
        replacements: Sequence[Any] | None = None,
    ) -> BackendCapture:
        del include_virtual_transport
        inputs, target = self._batch(data_list)
        by_layer: dict[int, list[int]] = {}
        for layer, head in family:
            by_layer.setdefault(int(layer), []).append(int(head))
        final: dict[str, Any] = {}

        def final_hook(_module, _args, output):
            final["state"] = output.last_hidden_state

        final_handle = self.model.encoder.register_forward_hook(final_hook)
        context = nullcontext() if require_grad else __import__("torch").no_grad()
        try:
            with _TransportHooks(
                self.model,
                ablate=by_layer,
                replacements=replacements,
            ) as hooks, context:
                output = self.model(**inputs, return_dict=True)
        finally:
            final_handle.remove()
        if any(value is None for value in hooks.values) or "state" not in final:
            raise RuntimeError("Graphormer transport/final-state hook did not fire")
        transport = []
        for routed in hooks.values:
            transport.append(routed if require_grad else routed.permute(1, 0, 2, 3))
        final_state = final["state"]
        return BackendCapture(
            prediction=output.logits,
            z=self._z(output.logits),
            target=target,
            transport=tuple(transport),
            final_state=final_state,
            real_mask=None,
        )

    def capture_groups(
        self,
        groups: Sequence[Sequence[GraphormerGraph]],
        *,
        include_virtual_transport: bool = True,
    ) -> list[BackendCapture]:
        """Pad and batch variable-size event groups, then restore graph-local geometry."""

        del include_virtual_transport
        normalised = [list(group) for group in groups]
        if not normalised or any(not group for group in normalised):
            raise ValueError("capture_groups requires non-empty graph groups")
        for group in normalised:
            counts = {int(data.num_nodes) for data in group}
            if len(counts) != 1:
                raise ValueError("replicas within one Graphormer group must share geometry")
        flat = [data for group in normalised for data in group]
        captured = self.capture(flat, require_grad=False)
        outputs: list[BackendCapture] = []
        offset = 0
        for group in normalised:
            replicas = len(group)
            tokens = int(group[0].num_nodes) + 1
            outputs.append(
                BackendCapture(
                    prediction=captured.prediction[offset : offset + replicas],
                    z=captured.z[offset : offset + replicas],
                    target=captured.target[offset : offset + replicas],
                    transport=tuple(
                        layer[offset : offset + replicas, :tokens]
                        for layer in captured.transport
                    ),
                    final_state=captured.final_state[
                        offset : offset + replicas, :tokens
                    ],
                    real_mask=None,
                )
            )
            offset += replicas
        if offset != len(flat):
            raise RuntimeError("grouped Graphormer capture did not reconstruct every graph")
        return outputs

    def clean_jacobians(self, data: GraphormerGraph) -> CleanJacobians:
        import torch

        self.model.zero_grad(set_to_none=True)
        capture = self.capture([data], require_grad=True)
        targets = tuple(capture.transport) + (capture.final_state,)
        z = capture.z.reshape(-1)
        by_output = []
        for output in range(int(z.numel())):
            by_output.append(
                torch.autograd.grad(
                    z[output],
                    targets,
                    retain_graph=output + 1 < int(z.numel()),
                    allow_unused=False,
                )
            )
        transport = torch.stack(
            [
                torch.stack(
                    [row[layer][:, 0].detach() for row in by_output],
                    dim=0,
                )
                for layer in range(self.runtime.L)
            ],
            dim=1,
        )
        final_gradient = torch.stack(
            [row[-1][0].detach() for row in by_output],
            dim=0,
        )
        audit_check(
            bool(torch.isfinite(transport).all() and torch.isfinite(final_gradient).all()),
            "backend.finite_clean_jacobian",
            "non-finite Graphormer clean z-space Jacobian",
        )
        layer_norms = torch.linalg.vector_norm(
            transport.reshape(transport.shape[0], transport.shape[1], -1),
            dim=(0, 2),
        )
        if bool((layer_norms <= 0).any()):
            missing = torch.nonzero(layer_norms <= 0).reshape(-1).tolist()
            audit_check(
                False,
                "backend.nonzero_clean_transport",
                f"zero clean Graphormer transport Jacobian in layers {missing}; "
                "those layers score zero",
                context={"layers": missing},
            )
        capture.prediction = capture.prediction.detach()
        capture.z = capture.z.detach()
        capture.target = capture.target.detach()
        capture.transport = tuple(value[:, 0].detach() for value in capture.transport)
        capture.final_state = capture.final_state[0].detach()
        return CleanJacobians(capture, transport, final_gradient)

    def clean_jacobians_many(
        self, data_list: Sequence[GraphormerGraph]
    ) -> list[CleanJacobians]:
        """Batch independent variable-size clean Jacobians with native padding."""

        import torch

        values = list(data_list)
        if not values:
            return []
        if len(values) == 1:
            return [self.clean_jacobians(values[0])]
        self.model.zero_grad(set_to_none=True)
        capture = self.capture(values, require_grad=True)
        targets = tuple(capture.transport) + (capture.final_state,)
        by_output = []
        for output in range(int(capture.z.shape[1])):
            by_output.append(
                torch.autograd.grad(
                    capture.z[:, output].sum(),
                    targets,
                    retain_graph=output + 1 < int(capture.z.shape[1]),
                    allow_unused=False,
                )
            )
        outputs: list[CleanJacobians] = []
        for graph_index, data in enumerate(values):
            tokens = int(data.num_nodes) + 1
            transport = torch.stack(
                [
                    torch.stack(
                        [
                            row[layer][:tokens, graph_index].detach()
                            for row in by_output
                        ],
                        dim=0,
                    )
                    for layer in range(self.runtime.L)
                ],
                dim=1,
            )
            final_gradient = torch.stack(
                [row[-1][graph_index, :tokens].detach() for row in by_output],
                dim=0,
            )
            audit_check(
                bool(
                    torch.isfinite(transport).all()
                    and torch.isfinite(final_gradient).all()
                ),
                "backend.finite_clean_jacobian",
                "non-finite grouped Graphormer clean z-space Jacobian",
                context={"graph_batch_index": int(graph_index)},
            )
            layer_norms = torch.linalg.vector_norm(
                transport.reshape(transport.shape[0], transport.shape[1], -1),
                dim=(0, 2),
            )
            if bool((layer_norms <= 0).any()):
                missing = torch.nonzero(layer_norms <= 0).reshape(-1).tolist()
                audit_check(
                    False,
                    "backend.nonzero_clean_transport",
                    f"zero grouped Graphormer transport Jacobian in layers {missing}; "
                    "those layers score zero",
                    context={
                        "graph_batch_index": int(graph_index),
                        "layers": missing,
                    },
                )
            graph_capture = BackendCapture(
                prediction=capture.prediction[
                    graph_index : graph_index + 1
                ].detach(),
                z=capture.z[graph_index : graph_index + 1].detach(),
                target=capture.target[graph_index : graph_index + 1].detach(),
                transport=tuple(
                    value[:tokens, graph_index].detach()
                    for value in capture.transport
                ),
                final_state=capture.final_state[graph_index, :tokens].detach(),
                real_mask=None,
            )
            outputs.append(
                CleanJacobians(graph_capture, transport, final_gradient)
            )
        return outputs

    def loss_per_graph(self, prediction, target):
        return self.task.loss_per_graph(
            prediction.reshape(prediction.shape[0], -1),
            target.reshape(target.shape[0], -1),
        )

    def loss_from_pooled(self, target):
        target = target.reshape(1, -1)

        def evaluate(graph_token):
            prediction = self.model.classifier(graph_token).reshape(graph_token.shape[0], -1)
            return self.loss_per_graph(
                prediction,
                target.expand(prediction.shape[0], -1),
            )

        return evaluate

    def carriage_weights(self, data: GraphormerGraph, final_state):
        weights = final_state.new_zeros(int(final_state.shape[-2]))
        weights[0] = 1.0
        return weights

    def transport_distances(
        self,
        data: GraphormerGraph,
        source: int,
        pristine,
        *,
        channel: str | None = None,
    ):
        del channel
        return ["graph_token", *list(pristine[int(source), :])]

    def carriage_distance_matrix(
        self,
        data: GraphormerGraph,
        sources,
        pristine,
        *,
        channel: str | None = None,
    ):
        del channel
        node = pristine[np.asarray(sources, dtype=np.int64), :].T
        return np.concatenate((np.full((1, len(sources)), np.nan), node), axis=0)

    def carriage_carrier_kind(
        self,
        data: GraphormerGraph,
        carrier: int,
        *,
        channel: str | None = None,
    ) -> str:
        del data, channel
        return "graph_token" if int(carrier) == 0 else "molecular_node"

    def _native_forward(self, data_list, family=(), replacements=None):
        captured = self.capture(
            data_list,
            require_grad=False,
            family=family,
            replacements=replacements,
        )
        return captured.prediction, captured.z, captured.target

    def _native_forward_individual_heads(
        self,
        data_list,
        assignments,
        *,
        replacements=None,
        ablate: bool,
    ):
        values = list(data_list)
        assigned = tuple((int(layer), int(head)) for layer, head in assignments)
        if not values or len(values) != len(assigned):
            raise ValueError(
                "individual Graphormer targets and head assignments must align"
            )
        inputs, target = self._batch(values)
        with _IndividualTransportHooks(
            self.model,
            assigned,
            replacements=replacements,
            ablate=bool(ablate),
        ), __import__("torch").no_grad():
            output = self.model(**inputs, return_dict=True)
        return output.logits, self._z(output.logits), target

    def ablate(self, data_list, family):
        return self._native_forward(data_list, family=family)

    def patch(self, target, donor_transport, family):
        return self._native_forward([target], family=family, replacements=donor_transport)

    def patch_many(self, targets, donor_transport, family):
        return self._native_forward(targets, family=family, replacements=donor_transport)

    def ablate_individual_heads(self, data_list, assignments):
        return self._native_forward_individual_heads(
            data_list,
            assignments,
            ablate=True,
        )

    def patch_individual_heads(self, targets, donor_transport, assignments):
        return self._native_forward_individual_heads(
            targets,
            assignments,
            replacements=donor_transport,
            ablate=False,
        )

    def replacement_batch(
        self,
        capture: BackendCapture,
        indices: Sequence[int],
        *,
        repeat_single: bool = False,
    ):
        del repeat_single
        selected = [int(value) for value in indices]
        return tuple(
            layer[selected].permute(1, 0, 2, 3).detach()
            for layer in capture.transport
        )

    def _attention(self, data: GraphormerGraph):
        import torch

        values: list[Any | None] = [None] * self.runtime.L
        handles = []

        def make_hook(layer_index):
            def hook(_module, _args, output):
                heads = int(self.runtime.H)
                batch = int(output.shape[0]) // heads
                values[layer_index] = output.view(
                    batch, heads, output.shape[-2], output.shape[-1]
                )

            return hook

        for index, layer in enumerate(self.model.encoder.graph_encoder.layers):
            handles.append(
                layer.self_attn.attention_dropout_module.register_forward_hook(
                    make_hook(index)
                )
            )
        try:
            inputs, _ = self._batch([data])
            with torch.no_grad():
                self.model(**inputs, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        if any(value is None for value in values):
            raise RuntimeError("Graphormer attention hook did not fire")
        return tuple(values)

    def attention_normalization_error(self, data: GraphormerGraph) -> float:
        import torch

        return max(
            float(torch.max(torch.abs(value.sum(dim=-1) - 1.0)).item())
            for value in self._attention(data)
        )

    def clean_attention_distance(self, data: GraphormerGraph, pristine, axis):
        profile = np.zeros(
            (self.runtime.L, self.runtime.H, len(axis.labels)),
            dtype=np.float64,
        )
        tokens = int(data.num_nodes) + 1
        bucket = np.empty((tokens, tokens), dtype=np.int64)
        for receiver in range(tokens):
            for sender in range(tokens):
                if receiver == 0 or sender == 0:
                    bucket[receiver, sender] = axis.index("graph_token")
                else:
                    bucket[receiver, sender] = axis.index(
                        pristine[receiver - 1, sender - 1]
                    )
        for layer, attention in enumerate(self._attention(data)):
            values = attention[0].detach().cpu().numpy()
            for distance_index in range(len(axis.labels)):
                profile[layer, :, distance_index] = values[:, bucket == distance_index].sum(
                    axis=-1
                )
            denominator = profile[layer].sum(axis=-1, keepdims=True)
            np.divide(
                profile[layer],
                denominator,
                out=profile[layer],
                where=denominator > 0,
            )
        return profile


def _load_official_pcqm_split(dataset: Any):
    """Load OGB's trusted legacy split metadata under PyTorch 2.6+.

    OGB 1.3.6 calls ``torch.load`` without an explicit ``weights_only`` argument.
    PyTorch 2.6 changed that default to ``True``, but the official PCQM split file
    contains NumPy arrays and is therefore not a weights-only artifact. Limit the
    compatibility opt-out to this one file from the registered official OGB archive.
    """

    import torch

    path = Path(dataset.folder) / "split_dict.pt"
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch releases that predate ``weights_only``.
        return torch.load(path, map_location="cpu")


def _pcqm_datasets(spec: Any, model: Any, overrides: Mapping[str, Any]):
    from ogb.lsc import PCQM4Mv2Dataset

    dataset_root = str(overrides.get("dataset_root", spec.dataset_root))
    try:
        dataset = PCQM4Mv2Dataset(root=dataset_root, only_smiles=True)
    except TypeError:
        dataset = PCQM4Mv2Dataset(root=dataset_root)
    split = _load_official_pcqm_split(dataset)
    eval_split = str(overrides.get("eval_split", spec.eval_split))
    donor_split = str(overrides.get("donor_split", spec.donor_split))
    if eval_split not in split or donor_split not in split:
        raise KeyError(
            f"PCQM split must be one of {sorted(split)}; "
            f"got eval={eval_split!r}, donor={donor_split!r}"
        )
    return (
        PCQMGraphormerDataset(dataset, split[eval_split], model.config),
        PCQMGraphormerDataset(dataset, split[donor_split], model.config),
    )


GRAPHORMER_DATASET_BUILDERS = {"pcqm4mv2": _pcqm_datasets}


def register_graphormer_dataset(name: str, builder) -> None:
    """Register another official Graphormer dataset without changing the estimators."""

    if name in GRAPHORMER_DATASET_BUILDERS:
        raise ValueError(f"Graphormer dataset builder {name!r} is already registered")
    GRAPHORMER_DATASET_BUILDERS[str(name)] = builder


def build_graphormer_runtime(
    task: Any,
    *,
    checkpoint: str | None,
    train_seed: int,
    accelerator: str,
    overrides: Mapping[str, Any],
):
    """Load the official PCQM model and create disjoint validation/train split views."""

    import torch
    device = torch.device(
        accelerator
        if not str(accelerator).startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    spec = task.spec
    model, descriptor, digest = load_graphormer_model(
        spec,
        checkpoint=checkpoint,
        device=device,
        model_id=overrides.get("model_id"),
        revision=overrides.get("revision"),
        cache_dir=overrides.get("cache_dir"),
        local_files_only=bool(overrides.get("local_files_only", False)),
    )
    if spec.dataset_name not in GRAPHORMER_DATASET_BUILDERS:
        raise KeyError(
            f"no Graphormer dataset builder registered for {spec.dataset_name!r}; "
            f"known: {sorted(GRAPHORMER_DATASET_BUILDERS)}"
        )
    eval_ds, donor_ds = GRAPHORMER_DATASET_BUILDERS[spec.dataset_name](
        spec, model, overrides
    )
    runtime = GraphormerRuntime(
        model,
        eval_ds,
        donor_ds,
        device=device,
        seed=int(train_seed),
        metric_fn=task.metric_fn,
    )
    return runtime, descriptor, digest
