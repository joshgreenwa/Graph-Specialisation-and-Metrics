"""Read-only canonical score adapters and additive Graphormer figure diagnostics.

The canonical methodology owns the score definitions and their cache contract.  This
module does not recompute those scores.  It adapts a validated ``scores/raw.pt`` value
for reporting and computes only explicitly additive diagnostics (selected attention,
pooled ``A@V`` vectors, and dot/bias logit spread).  Supplemental caches are immutable
and keyed to the canonical score artifact SHA and the exact model/runtime contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import os
import tempfile

import numpy as np

from .cache import (
    ReadOnlyCacheArtifact,
    StaleCacheError,
    load_cache_artifact_file,
)
from .graphormer import (
    GraphormerBackend,
    GraphormerGraph,
    PCQMGraphormerDataset,
    build_graphormer_runtime,
)
from .protocol import stable_hash
from .tasks import get_task


Head = tuple[int, int]


def _as_numpy(value: Any, *, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def load_graphormer_score_artifact(path: str | Path) -> ReadOnlyCacheArtifact:
    """Load a validated Graphormer score cache for additive figures.

    This compatibility boundary is intentionally narrower than the canonical
    analysis loader: it is read-only and requires the stored task and runtime
    contract to describe the official PCQM4Mv2 Graphormer task.
    """

    artifact = load_cache_artifact_file(path)
    contract = artifact.metadata.get("contract")
    if not isinstance(contract, Mapping):
        raise StaleCacheError(
            f"Graphormer score cache has no valid contract: {artifact.path}"
        )
    if contract.get("task") != "graphormer_pcqm4mv2":
        raise StaleCacheError(
            f"{artifact.path} is for task {contract.get('task')!r}, not "
            "'graphormer_pcqm4mv2'"
        )
    required = {
        "checkpoint_sha256",
        "model_geometry",
        "sigma",
        "split_fingerprint",
        "task_adapter_version",
        "train_seed",
    }
    missing = sorted(required.difference(contract))
    if missing:
        raise StaleCacheError(
            f"Graphormer score cache contract is missing {missing}: {artifact.path}"
        )
    return artifact


@dataclass(frozen=True)
class CanonicalHeadMetrics:
    """Figure-facing arrays copied from a validated canonical score artifact."""

    raw_semantic: np.ndarray
    raw_structural: np.ndarray
    normalized_semantic: np.ndarray
    normalized_structural: np.ndarray
    joint_sensitivity: np.ndarray
    selectivity: np.ndarray
    active: np.ndarray
    estimable: bool
    distance_axis: tuple[str, ...]
    clean_attention_distance: np.ndarray | None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.selectivity.shape)

    @property
    def num_layers(self) -> int:
        return self.shape[0]

    @property
    def num_heads(self) -> int:
        return self.shape[1]

    @classmethod
    def from_scores(cls, scores: Mapping[str, Any]) -> "CanonicalHeadMetrics":
        if "coordinates" not in scores:
            raise KeyError("canonical score payload has no 'coordinates' entry")
        coordinates = scores["coordinates"]
        arrays = {
            "raw_semantic": _as_numpy(
                _field(coordinates, "raw_semantic"), dtype=np.float64
            ),
            "raw_structural": _as_numpy(
                _field(coordinates, "raw_structural"), dtype=np.float64
            ),
            "normalized_semantic": _as_numpy(
                _field(coordinates, "normalized_semantic"), dtype=np.float64
            ),
            "normalized_structural": _as_numpy(
                _field(coordinates, "normalized_structural"), dtype=np.float64
            ),
            "joint_sensitivity": _as_numpy(
                _field(coordinates, "joint_sensitivity"), dtype=np.float64
            ),
            "selectivity": _as_numpy(
                _field(coordinates, "selectivity"), dtype=np.float64
            ),
            "active": _as_numpy(_field(coordinates, "active"), dtype=bool),
        }
        shape = arrays["selectivity"].shape
        if len(shape) != 2:
            raise ValueError(f"expected canonical head arrays [L,H], got {shape}")
        for name, array in arrays.items():
            if array.shape != shape:
                raise ValueError(
                    f"canonical {name} has shape {array.shape}; expected {shape}"
                )

        channels = scores.get("channels", {})
        for channel, coordinate_name in (
            ("semantic", "raw_semantic"),
            ("structural", "raw_structural"),
        ):
            if channel in channels and "raw" in channels[channel]:
                raw = _as_numpy(channels[channel]["raw"], dtype=np.float64)
                if raw.shape != shape or not np.allclose(
                    raw, arrays[coordinate_name], equal_nan=True
                ):
                    raise ValueError(
                        f"canonical coordinate and channel arrays disagree for {channel}"
                    )

        attention_distance = scores.get("clean_attention_distance")
        if attention_distance is not None:
            attention_distance = _as_numpy(attention_distance, dtype=np.float64)
            if (
                attention_distance.ndim != 3
                or attention_distance.shape[:2] != shape
            ):
                raise ValueError(
                    "clean_attention_distance must be [L,H,D] and match head arrays; "
                    f"got {attention_distance.shape} for {shape}"
                )
        axis = tuple(str(label) for label in scores.get("axis", ()))
        if (
            attention_distance is not None
            and len(axis) != attention_distance.shape[-1]
        ):
            raise ValueError(
                f"distance axis has {len(axis)} labels for "
                f"{attention_distance.shape[-1]} bins"
            )
        return cls(
            **arrays,
            estimable=bool(_field(coordinates, "estimable")),
            distance_axis=axis,
            clean_attention_distance=attention_distance,
        )

    def head_record(self, head: Head) -> dict[str, Any]:
        layer, index = validate_head(head, self.shape)
        return {
            "layer": layer,
            "head": index,
            "semantic": float(self.normalized_semantic[layer, index]),
            "structural": float(self.normalized_structural[layer, index]),
            "joint_sensitivity": float(self.joint_sensitivity[layer, index]),
            "selectivity": float(self.selectivity[layer, index]),
            "active": bool(self.active[layer, index]),
        }


def validate_head(head: Head, shape: tuple[int, int]) -> Head:
    layer, index = int(head[0]), int(head[1])
    if not (0 <= layer < shape[0] and 0 <= index < shape[1]):
        raise IndexError(f"head {(layer, index)} is outside canonical shape {shape}")
    return layer, index


def select_specialist_heads(
    metrics: CanonicalHeadMetrics,
    *,
    semantic_head: Head = (1, 24),
    structural_head: Head | None = None,
    active_only: bool = True,
) -> dict[str, Head]:
    """Keep L1 H24 semantic and select the smallest active-head ``D_rel``.

    The structural tie-break is decreasing ``J`` so an exact selectivity tie keeps
    the more causally engaged head.
    """

    semantic_head = validate_head(semantic_head, metrics.shape)
    if structural_head is not None:
        structural_head = validate_head(structural_head, metrics.shape)
    else:
        finite = np.isfinite(metrics.selectivity) & np.isfinite(
            metrics.joint_sensitivity
        )
        eligible = finite & metrics.active if active_only else finite
        eligible = eligible.copy()
        eligible[semantic_head] = False
        layers, heads = np.where(eligible)
        if not len(layers):
            raise ValueError("no eligible head is available for structural selection")
        selectivity = metrics.selectivity[layers, heads]
        joint = metrics.joint_sensitivity[layers, heads]
        order = np.lexsort((-joint, selectivity))
        structural_head = (int(layers[order[0]]), int(heads[order[0]]))
    return {"semantic": semantic_head, "structural": structural_head}


def select_ranked_heads(
    metrics: CanonicalHeadMetrics,
    *,
    semantic_count: int = 2,
    joint_count: int = 2,
    active_only: bool = True,
) -> dict[str, Head]:
    """Rank heads independently by decreasing ``D_rel`` and decreasing ``J``.

    A head may appear in both rankings. Exact ``D_rel`` ties prefer larger
    ``J``; exact ``J`` ties prefer larger ``D_rel``.
    """

    semantic_count = int(semantic_count)
    joint_count = int(joint_count)
    if semantic_count < 0 or joint_count < 0:
        raise ValueError("rank counts must be non-negative")
    finite = np.isfinite(metrics.selectivity) & np.isfinite(
        metrics.joint_sensitivity
    )
    eligible = finite & metrics.active if active_only else finite
    layers, heads = np.where(eligible)
    required = max(semantic_count, joint_count)
    if len(layers) < required:
        raise ValueError(
            f"only {len(layers)} eligible heads are available for a top-{required} ranking"
        )

    selectivity = metrics.selectivity[layers, heads]
    joint = metrics.joint_sensitivity[layers, heads]
    semantic_order = np.lexsort((-joint, -selectivity))
    joint_order = np.lexsort((-selectivity, -joint))

    ranked: dict[str, Head] = {}
    for rank, position in enumerate(semantic_order[:semantic_count], start=1):
        ranked[f"top_semantic_{rank}"] = (
            int(layers[position]),
            int(heads[position]),
        )
    for rank, position in enumerate(joint_order[:joint_count], start=1):
        ranked[f"top_joint_{rank}"] = (
            int(layers[position]),
            int(heads[position]),
        )
    return ranked


class SupplementalCache:
    """Immutable contract-keyed storage for non-canonical diagnostics."""

    SCHEMA_VERSION = "graphormer_pcqm_figure_diagnostics.v1"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str, contract: Mapping[str, Any]) -> Path:
        fingerprint = stable_hash(
            {
                "schema_version": self.SCHEMA_VERSION,
                "name": str(name),
                "contract": dict(contract),
            }
        )
        return self.root / f"{name}-{fingerprint[:16]}.pt"

    def load_or_compute(
        self,
        name: str,
        contract: Mapping[str, Any],
        compute: Callable[[], Any],
        *,
        force: bool = False,
    ) -> tuple[Any, Path, bool]:
        """Return ``(value, path, cache_hit)`` for an additive diagnostic."""

        path = self.path_for(name, contract)
        if path.exists() and not force:
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=False)
            if (
                isinstance(payload, Mapping)
                and payload.get("schema_version") == self.SCHEMA_VERSION
                and payload.get("contract_fingerprint")
                == stable_hash(dict(contract))
            ):
                return payload["value"], path, True

        value = compute()
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "contract": dict(contract),
            "contract_fingerprint": stable_hash(dict(contract)),
            "value": value,
        }
        self._atomic_torch_save(payload, path)
        return value, path, False

    @staticmethod
    def _atomic_torch_save(payload: Any, path: Path) -> None:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".partial", dir=path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class GraphormerFigureRuntime:
    """Official runtime verified against the canonical score cache contract."""

    runtime: Any
    backend: GraphormerBackend
    checkpoint_descriptor: str
    checkpoint_sha256: str


def build_verified_figure_runtime(
    artifact: ReadOnlyCacheArtifact,
    *,
    dataset_root: str,
    accelerator: str = "cuda:0",
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> GraphormerFigureRuntime:
    """Load PCQM Graphormer and fail if it differs from the score artifact."""

    contract = artifact.metadata["contract"]
    if contract.get("task") != "graphormer_pcqm4mv2":
        raise ValueError(
            "figure runtime requires a graphormer_pcqm4mv2 score artifact; "
            f"got {contract.get('task')!r}"
        )
    task = get_task("graphormer_pcqm4mv2")
    if contract.get("task_adapter_version") != task.adapter_version:
        raise ValueError(
            "canonical cache adapter version differs from the current Graphormer task"
        )
    runtime, descriptor, digest = build_graphormer_runtime(
        task,
        checkpoint=None,
        train_seed=int(contract["train_seed"]),
        accelerator=str(accelerator),
        overrides={
            "dataset_root": str(dataset_root),
            "cache_dir": cache_dir,
            "local_files_only": bool(local_files_only),
        },
    )
    expected_digest = str(contract["checkpoint_sha256"])
    if digest != expected_digest:
        raise ValueError(
            "loaded Graphormer checkpoint does not match canonical score cache: "
            f"{digest} != {expected_digest}"
        )
    backend = GraphormerBackend(runtime, task, sigma=contract["sigma"])
    expected_geometry = {
        str(key): int(value) for key, value in contract["model_geometry"].items()
    }
    if backend.geometry != expected_geometry:
        raise ValueError(
            f"loaded Graphormer geometry {backend.geometry} != "
            f"canonical {expected_geometry}"
        )
    return GraphormerFigureRuntime(runtime, backend, descriptor, digest)


@dataclass(frozen=True)
class GraphormerDiagnosticCapture:
    """One-graph tensors at Graphormer's exact attention computation sites."""

    dot: tuple[Any, ...]
    bias: tuple[Any, ...]
    attention: tuple[Any, ...]
    transport: tuple[Any, ...]


class GraphormerDiagnosticExtractor:
    """Capture ``d``, ``b``, softmax attention, and per-head ``A@V`` in one pass."""

    def __init__(self, backend: GraphormerBackend):
        self.backend = backend
        self.model = backend.model

    def extract(self, data: GraphormerGraph) -> GraphormerDiagnosticCapture:
        import torch

        layers = self.model.encoder.graph_encoder.layers
        dot: list[Any | None] = [None] * len(layers)
        bias: list[Any | None] = [None] * len(layers)
        attention: list[Any | None] = [None] * len(layers)
        transport: list[Any | None] = [None] * len(layers)
        handles = []

        def attention_input_hook(layer_index: int):
            def hook(module, args, kwargs):
                query = kwargs.get("query", args[0] if args else None)
                attention_bias = kwargs.get(
                    "attn_bias", args[3] if len(args) > 3 else None
                )
                if query is None or attention_bias is None:
                    raise RuntimeError("Graphormer diagnostic hook missed query/bias")
                tokens, batch, width = query.shape
                heads = int(module.num_heads)
                head_width = width // heads
                q = module.q_proj(query) * module.scaling
                k = module.k_proj(query)
                q = q.view(tokens, batch, heads, head_width).permute(1, 2, 0, 3)
                k = k.view(tokens, batch, heads, head_width).permute(1, 2, 0, 3)
                dot[layer_index] = torch.matmul(q, k.transpose(-1, -2)).detach()
                bias[layer_index] = attention_bias.view(
                    batch, heads, tokens, tokens
                ).detach()

            return hook

        def attention_output_hook(layer_index: int):
            def hook(module, args, output):
                del module, args
                heads = int(layers[layer_index].self_attn.num_heads)
                batch = int(output.shape[0]) // heads
                attention[layer_index] = output.view(
                    batch, heads, output.shape[-2], output.shape[-1]
                ).detach()

            return hook

        def transport_hook(layer_index: int):
            def hook(module, args):
                del module
                value = args[0]
                tokens, batch, width = value.shape
                heads = int(layers[layer_index].self_attn.num_heads)
                transport[layer_index] = (
                    value.view(tokens, batch, heads, width // heads)
                    .permute(1, 2, 0, 3)
                    .detach()
                )

            return hook

        for layer_index, layer in enumerate(layers):
            handles.append(
                layer.self_attn.register_forward_pre_hook(
                    attention_input_hook(layer_index), with_kwargs=True
                )
            )
            handles.append(
                layer.self_attn.attention_dropout_module.register_forward_hook(
                    attention_output_hook(layer_index)
                )
            )
            handles.append(
                layer.self_attn.out_proj.register_forward_pre_hook(
                    transport_hook(layer_index)
                )
            )
        try:
            inputs, _ = self.backend._batch([data])
            with torch.no_grad():
                self.model(**inputs, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        groups = (dot, bias, attention, transport)
        if any(any(value is None for value in group) for group in groups):
            raise RuntimeError("one or more Graphormer diagnostic hooks did not fire")
        tokens = int(data.num_nodes) + 1
        return GraphormerDiagnosticCapture(
            dot=tuple(value[0, :, :tokens, :tokens] for value in dot),
            bias=tuple(value[0, :, :tokens, :tokens] for value in bias),
            attention=tuple(
                value[0, :, :tokens, :tokens] for value in attention
            ),
            transport=tuple(value[0, :, :tokens, :] for value in transport),
        )


def dataset_index(runtime: Any, position: int) -> int:
    indices = getattr(runtime.eval_ds, "indices", None)
    return int(indices[position]) if indices is not None else int(position)


def graph_at_dataset_index(runtime: Any, index: int) -> GraphormerGraph:
    """Build one graph by its global PCQM4Mv2 dataset index."""

    eval_dataset = runtime.eval_ds
    source_dataset = getattr(eval_dataset, "dataset", None)
    config = getattr(eval_dataset, "config", None)
    if source_dataset is None or config is None:
        raise TypeError(
            "global PCQM graph indices require a PCQMGraphormerDataset-backed runtime"
        )
    direct_view = PCQMGraphormerDataset(source_dataset, (int(index),), config)
    return direct_view[0]


def collect_attention_examples(
    figure_runtime: GraphormerFigureRuntime,
    *,
    graph_indices: Sequence[int],
    heads: Mapping[str, Head],
) -> dict[str, Any]:
    extractor = GraphormerDiagnosticExtractor(figure_runtime.backend)
    examples = []
    for graph_index in graph_indices:
        graph = graph_at_dataset_index(figure_runtime.runtime, int(graph_index))
        captured = extractor.extract(graph)
        selected = {
            role: captured.attention[layer][head].float().cpu().numpy()
            for role, (layer, head) in heads.items()
        }
        examples.append(
            {
                "dataset_index": int(graph_index),
                "smiles": str(graph.smiles),
                "n_atoms": int(graph.num_nodes),
                "attention": selected,
            }
        )
    return {"heads": dict(heads), "examples": examples}


def _atom_categories(molecule) -> list[str]:
    """Assign the chemistry-focus categories used by the historical PCA recipe."""

    from rdkit import Chem

    categories: list[str | None] = [None] * molecule.GetNumAtoms()

    def is_carbonyl_carbon(atom) -> bool:
        return atom.GetSymbol() == "C" and any(
            bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetOtherAtom(atom).GetSymbol() == "O"
            for bond in atom.GetBonds()
        )

    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        symbol = atom.GetSymbol()
        if symbol == "O":
            double_carbon = any(
                bond.GetBondType() == Chem.BondType.DOUBLE
                and bond.GetOtherAtom(atom).GetSymbol() == "C"
                for bond in atom.GetBonds()
            )
            single_carbonyl = any(
                bond.GetBondType() == Chem.BondType.SINGLE
                and is_carbonyl_carbon(bond.GetOtherAtom(atom))
                for bond in atom.GetBonds()
            )
            if double_carbon:
                categories[index] = "O: carbonyl"
            elif single_carbonyl:
                categories[index] = "O: ester/carboxyl"
            elif atom.GetTotalNumHs() >= 1:
                categories[index] = "O: hydroxyl"
            else:
                categories[index] = "O: other"
        elif symbol == "N":
            if atom.GetIsAromatic():
                categories[index] = "N: aromatic"
            elif any(
                bond.GetBondType() == Chem.BondType.TRIPLE
                and bond.GetOtherAtom(atom).GetSymbol() == "C"
                for bond in atom.GetBonds()
            ):
                categories[index] = "N: nitrile"
            elif atom.GetFormalCharge() > 0 and sum(
                neighbour.GetSymbol() == "O" for neighbour in atom.GetNeighbors()
            ) >= 2:
                categories[index] = "N: nitro"
            elif any(is_carbonyl_carbon(neighbour) for neighbour in atom.GetNeighbors()):
                categories[index] = "N: amide"
            else:
                categories[index] = "N: other"
        elif symbol == "S":
            categories[index] = "S: sulfur"
        elif symbol == "P":
            categories[index] = "P: phosphorus"
        elif symbol in {"F", "Cl", "Br", "I"}:
            categories[index] = "X: halogen"

    ring_info = molecule.GetRingInfo()
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        if categories[index] is not None:
            continue
        rings = ring_info.NumAtomRings(index)
        if rings >= 2:
            categories[index] = "Ring: junction"
        elif rings == 1:
            categories[index] = (
                "Ring: aromatic" if atom.GetIsAromatic() else "Ring: aliphatic"
            )
        elif atom.GetDegree() >= 3:
            categories[index] = "Branch: degree>=3"
        elif atom.GetFormalCharge() > 0:
            categories[index] = "Charge: +"
        elif atom.GetFormalCharge() < 0:
            categories[index] = "Charge: -"
        else:
            categories[index] = "other/diffuse"
    return [str(value) for value in categories]


def label_attention_focus(
    smiles: str,
    atom_mass: np.ndarray,
    *,
    focus_mass: float = 0.75,
    diffuse_threshold: float = 0.35,
) -> str:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    atom_mass = np.asarray(atom_mass, dtype=np.float64)
    order = np.argsort(-atom_mass)
    selected = []
    cumulative = 0.0
    for index in order:
        selected.append(int(index))
        cumulative += float(atom_mass[index])
        if cumulative >= float(focus_mass):
            break
    categories = _atom_categories(molecule)
    mass_by_category: dict[str, float] = {}
    for index in selected:
        category = categories[index]
        mass_by_category[category] = mass_by_category.get(category, 0.0) + float(
            atom_mass[index]
        )
    best, mass = max(mass_by_category.items(), key=lambda item: item[1])
    return best if mass >= float(diffuse_threshold) else "other/diffuse"


def compute_av_pca_inputs(
    figure_runtime: GraphormerFigureRuntime,
    *,
    head: Head = (1, 24),
    n_graphs: int = 500,
    focus_mass: float = 0.75,
    diffuse_threshold: float = 0.35,
    verbose: bool = True,
) -> dict[str, Any]:
    layer, head_index = head
    extractor = GraphormerDiagnosticExtractor(figure_runtime.backend)
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    positions: list[int] = []
    indices: list[int] = []
    limit = min(int(n_graphs), len(figure_runtime.runtime.eval_ds))
    for position in range(limit):
        try:
            graph = figure_runtime.runtime.eval_ds[position]
            captured = extractor.extract(graph)
            vector = (
                captured.transport[layer][head_index, 1:, :]
                .mean(dim=0)
                .float()
                .cpu()
                .numpy()
            )
            matrix = (
                captured.attention[layer][head_index, 1:, 1:]
                .float()
                .cpu()
                .numpy()
            )
            matrix = matrix / np.clip(
                matrix.sum(axis=-1, keepdims=True), 1e-12, None
            )
            inbound = matrix.mean(axis=0)
            label = label_attention_focus(
                graph.smiles,
                inbound,
                focus_mass=focus_mass,
                diffuse_threshold=diffuse_threshold,
            )
        except Exception as error:
            if verbose:
                print(f"  [A@V PCA] skipped eval position {position}: {error}")
            continue
        vectors.append(vector)
        labels.append(label)
        positions.append(position)
        indices.append(dataset_index(figure_runtime.runtime, position))
        if verbose and (position + 1) % 50 == 0:
            print(f"  [A@V PCA] {position + 1}/{limit}")
    if len(vectors) < 2:
        raise RuntimeError("fewer than two molecules produced valid A@V vectors")
    return {
        "head": tuple(head),
        "vectors": np.stack(vectors),
        "labels": labels,
        "positions": np.asarray(positions, dtype=np.int64),
        "indices": np.asarray(indices, dtype=np.int64),
        "n_requested": int(n_graphs),
        "n_used": len(vectors),
    }


def aggregate_logit_spread(
    figure_runtime: GraphormerFigureRuntime,
    *,
    n_graphs: int = 100,
    verbose: bool = True,
) -> dict[str, Any]:
    """Aggregate key-wise ``std(d)`` and ``std(b)`` over validation molecules."""

    import torch

    extractor = GraphormerDiagnosticExtractor(figure_runtime.backend)
    dot_rows: list[np.ndarray] = []
    bias_rows: list[np.ndarray] = []
    positions: list[int] = []
    indices: list[int] = []
    limit = min(int(n_graphs), len(figure_runtime.runtime.eval_ds))
    for position in range(limit):
        try:
            graph = figure_runtime.runtime.eval_ds[position]
            captured = extractor.extract(graph)
            dot_spread = []
            bias_spread = []
            for dot, bias in zip(captured.dot, captured.bias):
                dot_nodes = dot[:, 1:, 1:].float()
                bias_nodes = bias[:, 1:, 1:].float()
                dot_spread.append(
                    dot_nodes.std(dim=-1, correction=1).mean(dim=-1)
                )
                bias_spread.append(
                    bias_nodes.std(dim=-1, correction=1).mean(dim=-1)
                )
            dot_rows.append(torch.stack(dot_spread).cpu().numpy())
            bias_rows.append(torch.stack(bias_spread).cpu().numpy())
        except Exception as error:
            if verbose:
                print(f"  [logit spread] skipped eval position {position}: {error}")
            continue
        positions.append(position)
        indices.append(dataset_index(figure_runtime.runtime, position))
        if verbose and (position + 1) % 10 == 0:
            print(f"  [logit spread] {position + 1}/{limit}")
    if not dot_rows:
        raise RuntimeError("no molecule produced valid Graphormer logit diagnostics")

    dot_per_graph = np.stack(dot_rows)
    bias_per_graph = np.stack(bias_rows)
    dot_layer_per_graph = dot_per_graph.mean(axis=-1)
    bias_layer_per_graph = bias_per_graph.mean(axis=-1)
    ratio_per_graph = np.log10(
        np.clip(
            bias_per_graph / np.clip(dot_per_graph, 1e-12, None),
            1e-12,
            None,
        )
    )
    ddof = 1 if len(dot_rows) > 1 else 0
    return {
        "dot_std_mean": dot_layer_per_graph.mean(axis=0),
        "dot_std_std": dot_layer_per_graph.std(axis=0, ddof=ddof),
        "bias_std_mean": bias_layer_per_graph.mean(axis=0),
        "bias_std_std": bias_layer_per_graph.std(axis=0, ddof=ddof),
        "log_r_mean": ratio_per_graph.mean(axis=0),
        "log_r_std": ratio_per_graph.std(axis=0, ddof=ddof),
        "dot_per_graph": dot_per_graph,
        "bias_per_graph": bias_per_graph,
        "positions": np.asarray(positions, dtype=np.int64),
        "indices": np.asarray(indices, dtype=np.int64),
        "n_requested": int(n_graphs),
        "n_used": len(dot_rows),
    }


__all__ = [
    "CanonicalHeadMetrics",
    "GraphormerDiagnosticCapture",
    "GraphormerDiagnosticExtractor",
    "GraphormerFigureRuntime",
    "Head",
    "SupplementalCache",
    "aggregate_logit_spread",
    "build_verified_figure_runtime",
    "collect_attention_examples",
    "compute_av_pca_inputs",
    "graph_at_dataset_index",
    "label_attention_focus",
    "load_graphormer_score_artifact",
    "select_ranked_heads",
    "select_specialist_heads",
]
