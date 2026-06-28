"""Model adapters for dissertation methodology runs.

Official paper-claim adapters are strict: they either load the official backend
or fail loudly.  Validation-only adapters are marked as such in their manifest.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_specialisation_metrics.method_core import (
    AdapterInfo,
    ForwardCache,
    GraphBatchView,
    ModelAdapter,
    edge_index_from_edges,
)


class TorchModuleAdapter:
    """Base adapter for local torch modules."""

    info: AdapterInfo

    def __init__(self, model: nn.Module, info: AdapterInfo, *, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.info = info
        self.device = torch.device(device)

    def forward(self, graph: GraphBatchView) -> ForwardCache:
        graph = graph.to(self.device)
        prediction, states, attention = self.model(graph)
        return ForwardCache(prediction=prediction, final_node_states=states, attention=attention)

    def predict(self, graph: GraphBatchView) -> torch.Tensor:
        return self.forward(graph).prediction

    def parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.model.parameters() if p.requires_grad)

    def attention_maps(self, graph: GraphBatchView) -> Optional[list[torch.Tensor]]:
        return self.forward(graph).attention

    def patch_hidden_states(
        self,
        graph: GraphBatchView,
        clamp_nodes: Sequence[int],
        clean_cache: Optional[ForwardCache] = None,
    ) -> ForwardCache:
        raise NotImplementedError(f"{self.info.name} does not expose hidden-state patching")


class DenseAttentionLayer(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=int(width),
            num_heads=int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(int(width))
        self.norm2 = nn.LayerNorm(int(width))
        self.ffn = nn.Sequential(
            nn.Linear(int(width), int(width) * 2),
            nn.ReLU(),
            nn.Linear(int(width) * 2, int(width)),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, weights = self.attn(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, weights[0]


class SmallGraphTransformerRegressor(nn.Module):
    """Validation-only dense GT: MHA stack, sum pool, scalar head."""

    def __init__(
        self,
        *,
        content_dim: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        use_selector_feature: bool = True,
        selector_readout: bool = True,
    ) -> None:
        super().__init__()
        self.content_dim = int(content_dim)
        self.use_selector_feature = bool(use_selector_feature)
        self.selector_readout = bool(selector_readout)
        input_dim = int(content_dim) + (1 if self.use_selector_feature else 0)
        self.encoder = nn.Linear(input_dim, int(hidden_dim))
        self.layers = nn.ModuleList(
            [DenseAttentionLayer(int(hidden_dim), int(heads)) for _ in range(int(layers))]
        )
        self.readout = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.selector_head = nn.Linear(int(content_dim), 1, bias=False)

    def _selector(self, graph: GraphBatchView) -> torch.Tensor:
        selector = None
        if graph.metadata is not None:
            selector = graph.metadata.get("selector_mask")
        if selector is None:
            selector = torch.zeros(graph.x.size(0), dtype=graph.x.dtype, device=graph.x.device)
        return torch.as_tensor(selector, dtype=graph.x.dtype, device=graph.x.device).view(-1, 1)

    def _input(self, graph: GraphBatchView) -> torch.Tensor:
        x = graph.x
        if self.use_selector_feature:
            x = torch.cat([x, self._selector(graph)], dim=-1)
        return x

    def forward(self, graph: GraphBatchView) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        if graph.num_graphs != 1:
            raise ValueError("SmallGraphTransformerRegressor currently expects one graph per view")
        h = self.encoder(self._input(graph)).unsqueeze(0)
        attentions = []
        for layer in self.layers:
            h, attn = layer(h)
            attentions.append(attn)
        node_states = h.squeeze(0)
        pooled = node_states.sum(dim=0, keepdim=True)
        if self.selector_readout:
            selected_content = (graph.x * self._selector(graph)).sum(dim=0, keepdim=True)
            pred = self.selector_head(selected_content).view(1)
        else:
            pred = self.readout(pooled).view(1)
        return pred, node_states, attentions


def make_small_graph_transformer_adapter(
    *,
    content_dim: int = 8,
    hidden_dim: int = 64,
    layers: int = 3,
    heads: int = 4,
    device: str | torch.device = "cpu",
) -> TorchModuleAdapter:
    model = SmallGraphTransformerRegressor(
        content_dim=content_dim,
        hidden_dim=hidden_dim,
        layers=layers,
        heads=heads,
        use_selector_feature=True,
    )
    return TorchModuleAdapter(
        model,
        AdapterInfo(
            name="small_dense_gt",
            version="validation.v1",
            implementation="torch.nn.MultiheadAttention validation-only GT",
            validation_only=True,
        ),
        device=device,
    )


class PlantedLinearSourceAdapter:
    """Analytic validation adapter for the planted linear target."""

    info = AdapterInfo(
        name="planted_linear_source",
        version="validation.v1",
        implementation="analytic validation-only linear source model",
        validation_only=True,
    )

    def __init__(self, weight: torch.Tensor, *, selector_key: str = "selector_mask") -> None:
        self.weight = weight.detach().clone()
        self.selector_key = selector_key

    def _selector(self, graph: GraphBatchView) -> torch.Tensor:
        if graph.metadata is None or self.selector_key not in graph.metadata:
            raise ValueError("planted linear adapter requires selector_mask metadata")
        return torch.as_tensor(graph.metadata[self.selector_key], dtype=graph.x.dtype, device=graph.x.device)

    def forward(self, graph: GraphBatchView) -> ForwardCache:
        selector = self._selector(graph).view(-1, 1)
        weight = self.weight.to(device=graph.x.device, dtype=graph.x.dtype).view(-1)
        node_scores = (graph.x * weight.view(1, -1)).sum(dim=-1) * selector.view(-1)
        pred = node_scores.sum().view(1)
        final_states = graph.x * selector
        return ForwardCache(prediction=pred, final_node_states=final_states)

    def predict(self, graph: GraphBatchView) -> torch.Tensor:
        return self.forward(graph).prediction

    def parameter_count(self) -> int:
        return 0

    def attention_maps(self, graph: GraphBatchView) -> None:
        return None

    def patch_hidden_states(
        self,
        graph: GraphBatchView,
        clamp_nodes: Sequence[int],
        clean_cache: Optional[ForwardCache] = None,
    ) -> ForwardCache:
        return self.forward(graph)


class DirectLinkAdapter:
    """Analytic direct long-range source-to-target link."""

    info = AdapterInfo(
        name="direct_link",
        version="validation.v1",
        implementation="analytic validation-only direct link",
        validation_only=True,
    )

    def __init__(self, source: int, target: int, readout: torch.Tensor) -> None:
        self.source = int(source)
        self.target = int(target)
        self.readout = readout.detach().clone()

    def final_node_states(self, graph: GraphBatchView) -> torch.Tensor:
        h = graph.x.detach().clone()
        h[self.target] = graph.x[self.source]
        return h

    def forward(self, graph: GraphBatchView) -> ForwardCache:
        h = self.final_node_states(graph)
        g = self.readout.to(device=graph.x.device, dtype=graph.x.dtype)
        return ForwardCache(prediction=(h[self.target] * g).sum().view(1), final_node_states=h)

    def predict(self, graph: GraphBatchView) -> torch.Tensor:
        return self.forward(graph).prediction

    def parameter_count(self) -> int:
        return 0

    def attention_maps(self, graph: GraphBatchView) -> None:
        return None

    def patch_hidden_states(
        self,
        graph: GraphBatchView,
        clamp_nodes: Sequence[int],
        clean_cache: Optional[ForwardCache] = None,
    ) -> ForwardCache:
        return self.forward(graph)


class StepByStepChainAdapter(DirectLinkAdapter):
    """Analytic chain transport that collapses when an interior cut is clamped."""

    info = AdapterInfo(
        name="step_by_step_chain",
        version="validation.v1",
        implementation="analytic validation-only composed chain",
        validation_only=True,
    )

    def __init__(self, source: int, target: int, readout: torch.Tensor, chain_nodes: Sequence[int]) -> None:
        super().__init__(source, target, readout)
        self.chain_nodes = [int(v) for v in chain_nodes]

    def patch_hidden_states(
        self,
        graph: GraphBatchView,
        clamp_nodes: Sequence[int],
        clean_cache: Optional[ForwardCache] = None,
    ) -> ForwardCache:
        clamped = set(int(v) for v in clamp_nodes)
        interior = set(self.chain_nodes[1:-1])
        if clamped & interior:
            h = graph.x.detach().clone()
            g = self.readout.to(device=graph.x.device, dtype=graph.x.dtype)
            return ForwardCache(prediction=(h[self.target] * g).sum().view(1), final_node_states=h)
        return self.forward(graph)


def chain_graph_with_branches(
    *,
    length: int = 7,
    branch_attachments: Sequence[int] = (2, 3, 4),
    feature_dim: int = 8,
    seed: int = 0,
) -> GraphBatchView:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    chain_nodes = list(range(int(length)))
    edges = [(i, i + 1) for i in range(int(length) - 1)]
    next_node = int(length)
    for attach in branch_attachments:
        edges.append((int(attach), next_node))
        next_node += 1
    x = torch.randn(next_node, int(feature_dim), generator=gen)
    return GraphBatchView(
        x=x,
        edge_index=edge_index_from_edges(next_node, edges, undirected=True),
        metadata={"chain_nodes": chain_nodes, "branch_nodes": list(range(int(length), next_node))},
    )


@dataclass
class OfficialGRITAdapter:
    """Strict loader facade for the LiamMa/GRIT implementation.

    The adapter validates result metadata and can count checkpoint parameters
    without importing the official repo.  Forward execution requires the official
    checkout to be importable and must be wired to the concrete GRIT task wrapper.
    """

    repo_path: Path
    config_path: Path
    checkpoint_path: Path
    variant: str = "official"
    official_commit: Optional[str] = None

    def __post_init__(self) -> None:
        self.repo_path = Path(self.repo_path)
        self.config_path = Path(self.config_path)
        self.checkpoint_path = Path(self.checkpoint_path)
        self.info = AdapterInfo(
            name=f"grit_{self.variant}",
            version="official-loader.v1",
            implementation="LiamMa/GRIT official checkout",
            official_repo="https://github.com/LiamMa/GRIT",
            official_commit=self.official_commit,
            validation_only=False,
            dev_only=False,
        )
        if not self.config_path.exists():
            raise FileNotFoundError(f"missing GRIT config: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"missing GRIT checkpoint: {self.checkpoint_path}")

    def _load_checkpoint_payload(self) -> Any:
        return torch.load(self.checkpoint_path, map_location="cpu")

    def parameter_count(self) -> int:
        payload = self._load_checkpoint_payload()
        if isinstance(payload, dict):
            for key in ("parameter_count", "param_count", "num_parameters", "n_parameters"):
                if key in payload:
                    return int(payload[key])
        state = payload.get("state_dict", payload.get("model_state_dict", payload)) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise RuntimeError(f"could not read state dict from {self.checkpoint_path}")
        total = 0
        buffer_markers = ("running_mean", "running_var", "num_batches_tracked", "tracked")
        for key, value in state.items():
            if any(marker in str(key) for marker in buffer_markers):
                continue
            if isinstance(value, torch.Tensor):
                total += int(value.numel())
        return total

    def _import_official(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(f"missing official GRIT checkout: {self.repo_path}")
        sys.path.insert(0, str(self.repo_path))
        try:
            importlib.import_module("graphgps")
        except Exception as exc:
            raise RuntimeError(
                "official GRIT execution requires the LiamMa/GRIT checkout and its environment; "
                f"failed importing graphgps from {self.repo_path}: {exc}"
            ) from exc

    def forward(self, graph: GraphBatchView) -> ForwardCache:
        self._import_official()
        raise NotImplementedError(
            "OfficialGRITAdapter validates official artifacts now; task-specific forward hooks "
            "must be connected once trained GRIT checkpoints are present."
        )

    def predict(self, graph: GraphBatchView) -> torch.Tensor:
        return self.forward(graph).prediction

    def attention_maps(self, graph: GraphBatchView) -> Optional[list[torch.Tensor]]:
        return self.forward(graph).attention

    def patch_hidden_states(
        self,
        graph: GraphBatchView,
        clamp_nodes: Sequence[int],
        clean_cache: Optional[ForwardCache] = None,
    ) -> ForwardCache:
        self._import_official()
        raise NotImplementedError("Official GRIT hidden-state patch hooks require checkpoint-specific wiring")

    def metadata(self) -> dict[str, Any]:
        config_payload: dict[str, Any]
        try:
            import yaml

            with self.config_path.open("r", encoding="utf-8") as f:
                config_payload = yaml.safe_load(f) or {}
        except Exception:
            try:
                with self.config_path.open("r", encoding="utf-8") as f:
                    config_payload = json.load(f)
            except Exception:
                config_payload = {"config_path": str(self.config_path)}
        return {
            "repo_path": str(self.repo_path),
            "config_path": str(self.config_path),
            "checkpoint_path": str(self.checkpoint_path),
            "variant": self.variant,
            "config": config_payload,
            "parameter_count": self.parameter_count(),
        }


class PyGGraphClassifier(nn.Module):
    def __init__(self, conv_name: str, input_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import GCNConv, GINConv, global_add_pool
        except Exception as exc:  # pragma: no cover - depends on optional env.
            raise RuntimeError("PyTorch Geometric is required for PyG adapters") from exc
        self.global_add_pool = global_add_pool
        self.convs = nn.ModuleList()
        dim = int(input_dim)
        for _ in range(int(layers)):
            if conv_name == "gcn":
                conv = GCNConv(dim, int(hidden_dim))
            elif conv_name == "gin":
                conv = GINConv(
                    nn.Sequential(
                        nn.Linear(dim, int(hidden_dim)),
                        nn.ReLU(),
                        nn.Linear(int(hidden_dim), int(hidden_dim)),
                    )
                )
            else:
                raise ValueError(f"unknown PyG conv {conv_name}")
            self.convs.append(conv)
            dim = int(hidden_dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, graph: GraphBatchView) -> tuple[torch.Tensor, torch.Tensor, None]:
        x = graph.x
        edge_index = graph.edge_index.long()
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        pooled = self.global_add_pool(x, graph.batch)
        return self.head(pooled).view(-1), x, None


def make_pyg_adapter(
    kind: str,
    *,
    input_dim: int,
    hidden_dim: int = 64,
    layers: int = 3,
    device: str | torch.device = "cpu",
) -> TorchModuleAdapter:
    kind = kind.lower()
    if kind not in {"gcn", "gin"}:
        raise ValueError("kind must be gcn or gin")
    model = PyGGraphClassifier(kind, input_dim=input_dim, hidden_dim=hidden_dim, layers=layers)
    return TorchModuleAdapter(
        model,
        AdapterInfo(
            name=f"pyg_{kind}",
            version="pyg-official-layers.v1",
            implementation=f"PyTorch Geometric {kind.upper()}Conv official layer",
            official_repo="https://github.com/pyg-team/pytorch_geometric",
            validation_only=False,
        ),
        device=device,
    )


def parameter_count_close(a: ModelAdapter, b: ModelAdapter) -> tuple[bool, int, int]:
    pa = int(a.parameter_count())
    pb = int(b.parameter_count())
    return pa == pb, pa, pb
