"""Model adapters for dissertation methodology runs.

Official paper-claim adapters are strict: they either load the official backend
or fail loudly.  Validation-only adapters are marked as such in their manifest.
"""

from __future__ import annotations

import importlib
import contextlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
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

    The adapter keeps all GRIT-specific code behind the model-agnostic
    methodology interface. It imports the official checkout, loads the official
    GraphGym config/model/checkpoint, and uses forward hooks to expose final node
    states, per-layer attention maps, channel fields, and mediator patch points.
    """

    repo_path: Path
    config_path: Path
    checkpoint_path: Path
    variant: str = "official"
    official_commit: Optional[str] = None
    dataset_dir: Optional[Path] = None
    device: str | torch.device = "cpu"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.repo_path = Path(self.repo_path)
        self.config_path = Path(self.config_path)
        self.checkpoint_path = Path(self.checkpoint_path)
        self.dataset_dir = Path(self.dataset_dir) if self.dataset_dir is not None else None
        self.device = torch.device(self.device)
        self._model: Optional[nn.Module] = None
        self._loaders: Optional[Sequence[Any]] = None
        self._official_imported = False
        self.info = AdapterInfo(
            name=f"grit_{self.variant}",
            version="official-hooks.v1",
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
        try:
            return torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(self.checkpoint_path, map_location="cpu")

    @staticmethod
    def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
        if isinstance(payload, dict):
            for key in (
                "state_dict",
                "model_state_dict",
                "model",
                "model_state",
                "module",
                "net",
            ):
                value = payload.get(key)
                if isinstance(value, dict):
                    return {
                        str(k).replace("module.", "", 1): v
                        for k, v in value.items()
                        if isinstance(v, torch.Tensor)
                    }
            if payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
                return {str(k).replace("module.", "", 1): v for k, v in payload.items()}
        raise RuntimeError("could not read a model state_dict from checkpoint payload")

    def parameter_count(self) -> int:
        payload = self._load_checkpoint_payload()
        if isinstance(payload, dict):
            for key in ("parameter_count", "param_count", "num_parameters", "n_parameters"):
                if key in payload:
                    return int(payload[key])
        state = self._extract_state_dict(payload)
        total = 0
        buffer_markers = ("running_mean", "running_var", "num_batches_tracked", "tracked")
        for key, value in state.items():
            if any(marker in str(key) for marker in buffer_markers):
                continue
            if isinstance(value, torch.Tensor):
                total += int(value.numel())
        return total

    def _import_official(self) -> None:
        if self._official_imported:
            return
        if not self.repo_path.exists():
            raise FileNotFoundError(f"missing official GRIT checkout: {self.repo_path}")
        repo_str = str(self.repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        try:
            importlib.import_module("grit")
            importlib.import_module("torch_geometric")
        except Exception as exc:
            raise RuntimeError(
                "official GRIT execution requires the LiamMa/GRIT checkout and its environment; "
                f"failed importing GRIT/PyG from {self.repo_path}: {exc}"
            ) from exc
        self._official_imported = True

    def _configure_official(self) -> Any:
        self._import_official()
        from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg

        set_cfg(cfg)
        cfg.set_new_allowed(True)
        opts: list[str] = []
        if self.dataset_dir is not None:
            opts.extend(["dataset.dir", str(self.dataset_dir)])
        if self.seed is not None:
            opts.extend(["seed", str(int(self.seed))])
        opts.extend(["train.auto_resume", "False"])
        args = SimpleNamespace(
            cfg_file=str(self.config_path),
            opts=opts,
            repeat=1,
            mark_done=False,
        )
        cfg.work_dir = str(self.repo_path)
        load_cfg(cfg, args)
        cfg.cfg_file = str(self.config_path)
        cfg.device = str(self.device)
        cfg.accelerator = str(self.device)
        if "1hop" in str(self.variant).lower() or "one_hop" in str(self.variant).lower():
            attn_cfg = getattr(cfg.gt, "attn", None)
            sparsity = ""
            full_attn = None
            if attn_cfg is not None:
                try:
                    sparsity = str(attn_cfg.get("sparsity", ""))
                except Exception:
                    sparsity = str(getattr(attn_cfg, "sparsity", ""))
                try:
                    full_attn = bool(attn_cfg.get("full_attn"))
                except Exception:
                    full_attn = bool(getattr(attn_cfg, "full_attn", True))
            if sparsity != "one_hop" or full_attn is not False:
                raise RuntimeError(
                    "1-hop GRIT adapter loaded a config that does not declare the 1-hop attention control: "
                    f"gt.attn.sparsity={sparsity!r}, gt.attn.full_attn={full_attn!r}, config={self.config_path}"
                )
        return cfg

    @staticmethod
    def _load_state_dict_into_model(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
        model_state = model.state_dict()
        prefixes = ["", "model.", "module.", "model.module.", "module.model."]
        best_prefix = ""
        best_matches = -1
        for prefix in prefixes:
            matches = sum(1 for key in model_state if prefix + key in state)
            if matches > best_matches:
                best_matches = matches
                best_prefix = prefix
        remapped = {
            key: state[best_prefix + key]
            for key in model_state
            if best_prefix + key in state and tuple(state[best_prefix + key].shape) == tuple(model_state[key].shape)
        }
        if not remapped:
            # Fall back to stripping common wrapper prefixes from checkpoint keys.
            stripped: dict[str, torch.Tensor] = {}
            for key, value in state.items():
                k = str(key)
                for prefix in ("model.", "module.", "model.module.", "module.model."):
                    if k.startswith(prefix):
                        k = k[len(prefix):]
                        break
                if k in model_state and tuple(value.shape) == tuple(model_state[k].shape):
                    stripped[k] = value
            remapped = stripped
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if len(remapped) == 0:
            raise RuntimeError("no checkpoint tensors matched the official GRIT model")
        if len(missing) > max(4, len(model_state) // 10):
            raise RuntimeError(
                f"checkpoint/model mismatch: loaded {len(remapped)} tensors, "
                f"missing {len(missing)}, unexpected {len(unexpected)}"
            )

    def load_model(self) -> nn.Module:
        if self._model is not None:
            return self._model
        cfg = self._configure_official()
        from torch_geometric import seed_everything
        from torch_geometric.graphgym.model_builder import create_model

        if self.seed is not None:
            seed_everything(int(self.seed))
        # Official GRIT/GraphGym creates loaders before the model because loader
        # construction sets shared input/output dimensions in the global cfg.
        self.load_zinc_loaders()
        model = create_model()
        payload = self._load_checkpoint_payload()
        state = self._extract_state_dict(payload)
        self._load_state_dict_into_model(model, state)
        model = model.to(torch.device(cfg.device))
        model.eval()
        self._model = model
        return model

    @staticmethod
    def _clone_pyg_data(data: Any) -> Any:
        if hasattr(data, "clone"):
            return data.clone()
        import copy

        return copy.deepcopy(data)

    def _graph_to_data(self, graph: Any) -> Any:
        self._import_official()
        if isinstance(graph, GraphBatchView):
            from torch_geometric.data import Data

            data = Data(x=graph.x, edge_index=graph.edge_index)
            if graph.y is not None:
                data.y = graph.y
            if graph.batch is not None:
                data.batch = graph.batch
            if graph.distances is not None:
                data.distances = graph.distances
            for key, value in dict(graph.metadata or {}).items():
                if key not in {"x", "edge_index", "batch", "y"}:
                    setattr(data, key, value)
        else:
            data = self._clone_pyg_data(graph)
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(int(data.x.size(0)), dtype=torch.long)
        return data.to(self.device) if hasattr(data, "to") else data

    def load_zinc_loaders(self) -> Sequence[Any]:
        if self._loaders is not None:
            return self._loaders
        self._configure_official()
        from torch_geometric.graphgym.loader import create_loader

        self._loaders = create_loader()
        return self._loaders

    def load_zinc_split(self, split: str, *, limit: Optional[int] = None) -> list[Any]:
        loaders = self.load_zinc_loaders()
        split_map = {"train": 0, "val": 1, "valid": 1, "validation": 1, "test": 2}
        if split not in split_map:
            raise ValueError(f"unknown ZINC split {split!r}; expected train/val/test")
        dataset = loaders[split_map[split]].dataset
        n = len(dataset) if limit is None else min(int(limit), len(dataset))
        return [dataset[i] for i in range(n)]

    @staticmethod
    def _attention_to_dense(edge_index: torch.Tensor, attn: torch.Tensor, num_nodes: int) -> torch.Tensor:
        values = attn.detach()
        if values.dim() == 3 and values.size(-1) == 1:
            values = values.squeeze(-1)
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        if values.dim() != 2:
            raise RuntimeError(f"expected sparse GRIT attention with shape [E,H], got {tuple(values.shape)}")
        if int(values.size(0)) != int(edge_index.size(1)):
            raise RuntimeError(
                f"attention/edge support mismatch: attention has {values.size(0)} rows "
                f"but edge_index has {edge_index.size(1)} edges"
            )
        # GRIT uses edge_index[0]=source, edge_index[1]=destination. Return
        # [heads, destination, source], matching A[i,j] = receiver i reads source j.
        dense = values.new_zeros((values.size(1), int(num_nodes), int(num_nodes)))
        dense[:, edge_index[1].long(), edge_index[0].long()] = values.t()
        return dense

    def _run_with_hooks(
        self,
        graph: Any,
        *,
        content_override: Optional[torch.Tensor] = None,
        clean_cache: Optional[ForwardCache] = None,
        clamp_nodes: Sequence[int] = (),
        clamp_until_layer: Optional[int] = None,
        retain_grad: bool = False,
        capture_attention: bool = True,
        capture_channels: bool = True,
        capture_layer_inputs: bool = True,
        capture_layer_outputs: bool = True,
    ) -> ForwardCache:
        model = self.load_model()
        data = self._graph_to_data(graph)
        captures: dict[str, Any] = {
            "attention": [],
            "attention_edges": [],
            "layer_input_node_states": [],
            "layer_output_node_states": [],
            "layer_input_edge_attr": [],
            "layer_output_edge_attr": [],
            "layer_output_node_state_tensors": [],
            "channel_fields": [],
            "encoded_node_states": None,
            "final_node_states": None,
        }
        handles: list[Any] = []
        clamp = torch.as_tensor(list(clamp_nodes), dtype=torch.long, device=self.device)
        clean_inputs = None
        if clean_cache is not None and clean_cache.extras is not None:
            clean_inputs = clean_cache.extras.get("layer_input_node_states")
        if clamp.numel() > 0 and clean_cache is not None and not clean_inputs:
            raise RuntimeError(
                "mediator patching requested clamped nodes, but the clean GRIT cache "
                "does not contain layer inputs; patching would be a no-op"
            )

        def layer_index_from_name(name: str) -> Optional[int]:
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
            return int(match.group(1)) if match else None

        def feature_encoder_post_hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            batch = output
            if not hasattr(batch, "x") or not isinstance(batch.x, torch.Tensor):
                return output
            if content_override is not None:
                override = content_override.to(device=batch.x.device, dtype=batch.x.dtype)
                if tuple(override.shape) != tuple(batch.x.shape):
                    raise ValueError(
                        f"content_override shape {tuple(override.shape)} does not match encoded node content {tuple(batch.x.shape)}"
                    )
                batch.x = override
            if retain_grad and getattr(batch.x, "requires_grad", False):
                batch.x.retain_grad()
            captures["encoded_node_states"] = batch.x
            return batch

        def make_layer_pre_hook(layer_idx: int):
            def hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
                if not inputs:
                    return
                batch = inputs[0]
                if hasattr(batch, "x") and isinstance(batch.x, torch.Tensor):
                    if capture_layer_inputs:
                        captures["layer_input_node_states"].append(batch.x.detach().clone())
                    if retain_grad and capture_layer_inputs and getattr(batch.x, "requires_grad", False):
                        batch.x.retain_grad()
                    if capture_layer_inputs and batch.get("edge_attr", None) is not None:
                        captures["layer_input_edge_attr"].append(batch.edge_attr.detach().clone())
                should_clamp = (
                    clamp.numel() > 0
                    and clean_inputs is not None
                    and layer_idx < len(clean_inputs)
                    and (clamp_until_layer is None or layer_idx <= int(clamp_until_layer))
                )
                if should_clamp:
                    patched = batch.x.clone()
                    patched[clamp] = clean_inputs[layer_idx].to(device=patched.device, dtype=patched.dtype)[clamp]
                    batch.x = patched
            return hook

        def make_layer_post_hook(layer_idx: int):
            def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                batch = output
                if hasattr(batch, "x") and isinstance(batch.x, torch.Tensor):
                    if retain_grad and capture_layer_outputs and getattr(batch.x, "requires_grad", False):
                        batch.x.retain_grad()
                        captures["layer_output_node_state_tensors"].append(batch.x)
                    if capture_layer_outputs:
                        captures["layer_output_node_states"].append(batch.x.detach().clone())
                if capture_layer_outputs and hasattr(batch, "edge_attr") and isinstance(batch.edge_attr, torch.Tensor):
                    captures["layer_output_edge_attr"].append(batch.edge_attr.detach().clone())
            return hook

        def post_mp_pre_hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if inputs:
                batch = inputs[0]
                if hasattr(batch, "x") and isinstance(batch.x, torch.Tensor):
                    if retain_grad:
                        batch.x.retain_grad()
                    captures["final_node_states"] = batch.x

        def attention_post_hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if not inputs:
                return
            batch = inputs[0]
            if getattr(batch, "attn", None) is not None and (capture_attention or capture_channels):
                edge_index = batch.edge_index.detach().clone()
                attn = batch.attn.detach().clone()
                if capture_attention:
                    captures["attention_edges"].append(edge_index)
                    captures["attention"].append(self._attention_to_dense(edge_index, attn, int(batch.num_nodes)))
                if capture_channels:
                    fields = {
                        "edge_index": edge_index,
                        "attn": attn,
                        "edge_attr": batch.edge_attr.detach().clone() if getattr(batch, "edge_attr", None) is not None else None,
                        "Q_h": batch.Q_h.detach().clone() if getattr(batch, "Q_h", None) is not None else None,
                        "K_h": batch.K_h.detach().clone() if getattr(batch, "K_h", None) is not None else None,
                        "V_h": batch.V_h.detach().clone() if getattr(batch, "V_h", None) is not None else None,
                        "wV": batch.wV.detach().clone() if getattr(batch, "wV", None) is not None else None,
                        "E": batch.E.detach().clone() if getattr(batch, "E", None) is not None else None,
                        "wE": batch.wE.detach().clone() if getattr(batch, "wE", None) is not None else None,
                        "edge_enhance": bool(getattr(module, "edge_enhance", False)),
                        "VeRow": module.VeRow.detach().clone() if getattr(module, "VeRow", None) is not None else None,
                    }
                    captures["channel_fields"].append(fields)

        for name, module in model.named_modules():
            cls_name = module.__class__.__name__
            layer_idx = layer_index_from_name(name)
            if cls_name == "FeatureEncoder":
                handles.append(module.register_forward_hook(feature_encoder_post_hook))
            elif cls_name == "GritTransformerLayer" and layer_idx is not None:
                handles.append(module.register_forward_pre_hook(make_layer_pre_hook(layer_idx)))
                handles.append(module.register_forward_hook(make_layer_post_hook(layer_idx)))
            elif cls_name == "MultiHeadAttentionLayerGritSparse":
                handles.append(module.register_forward_hook(attention_post_hook))
            elif name.endswith("post_mp") or cls_name == "SANGraphHead":
                handles.append(module.register_forward_pre_hook(post_mp_pre_hook))

        try:
            output = model(data)
        finally:
            for handle in handles:
                handle.remove()

        if isinstance(output, tuple):
            prediction = output[0]
        else:
            prediction = output
        if isinstance(prediction, torch.Tensor):
            prediction_tensor = prediction.view(-1)
        else:
            raise RuntimeError(f"official GRIT forward returned unsupported prediction type {type(prediction)!r}")
        if capture_layer_outputs and captures["attention"] and not captures["layer_output_node_states"]:
            raise RuntimeError(
                "captured GRIT attention maps but no GritTransformerLayer outputs; "
                "layer hook registration failed, so layer-resolved and mediator-patching analyses would be invalid"
            )
        if captures["attention"] and len(captures["attention"]) != len(captures["attention_edges"]):
            raise RuntimeError("captured GRIT attention maps without matching sparse edge supports")
        final_states = captures["final_node_states"]
        if final_states is None:
            raise RuntimeError("could not capture final GRIT node states before graph readout")
        extras = {
            "encoded_node_states": captures["encoded_node_states"],
            "layer_input_node_states": captures["layer_input_node_states"],
            "layer_output_node_states": captures["layer_output_node_states"],
            "layer_output_node_state_tensors": captures["layer_output_node_state_tensors"],
            "layer_input_edge_attr": captures["layer_input_edge_attr"],
            "layer_output_edge_attr": captures["layer_output_edge_attr"],
            "attention_edges": captures["attention_edges"],
            "raw_output": output,
        }
        return ForwardCache(
            prediction=prediction_tensor,
            final_node_states=final_states,
            attention=captures["attention"],
            channel_fields={"layers": captures["channel_fields"]},
            extras=extras,
        )

    def forward(self, graph: GraphBatchView) -> ForwardCache:
        with torch.no_grad():
            return self._run_with_hooks(graph)

    def forward_minimal(self, graph: Any) -> ForwardCache:
        with torch.inference_mode():
            return self._run_with_hooks(
                graph,
                capture_attention=False,
                capture_channels=False,
                capture_layer_inputs=False,
                capture_layer_outputs=False,
            )

    def forward_with_grad(self, graph: Any) -> ForwardCache:
        model = self.load_model()
        model.zero_grad(set_to_none=True)
        return self._run_with_hooks(graph, retain_grad=True)

    def encoded_node_states(self, graph: Any) -> torch.Tensor:
        cache = self.forward_minimal(graph)
        encoded = None if cache.extras is None else cache.extras.get("encoded_node_states")
        if not isinstance(encoded, torch.Tensor):
            raise RuntimeError("could not capture encoded GRIT node content after FeatureEncoder")
        return encoded.detach().clone()

    def forward_from_encoded_content(
        self,
        graph: Any,
        encoded_content: torch.Tensor,
        *,
        retain_grad: bool = False,
        capture_attention: bool = False,
        capture_channels: bool = False,
        capture_layer_inputs: bool = False,
        capture_layer_outputs: bool = False,
    ) -> ForwardCache:
        if retain_grad:
            self.load_model().zero_grad(set_to_none=True)
        context = (
            torch.inference_mode()
            if not retain_grad and not bool(getattr(encoded_content, "requires_grad", False))
            else contextlib.nullcontext()
        )
        with context:
            return self._run_with_hooks(
                graph,
                content_override=encoded_content,
                retain_grad=retain_grad,
                capture_attention=capture_attention,
                capture_channels=capture_channels,
                capture_layer_inputs=capture_layer_inputs,
                capture_layer_outputs=capture_layer_outputs,
            )

    def readout_gradient(self, graph: Any, *, target_index: int = 0) -> tuple[ForwardCache, torch.Tensor]:
        """Return ``(cache, d prediction / d final_node_states)`` for one graph.

        Official ZINC GRIT uses sum-pool + MLP, so this gradient is expected to
        be direction-uniform across nodes. We compute it rather than assuming it
        so future official configs with different readouts remain supported.
        """
        cache = self.forward_with_grad(graph)
        grad = self._populate_readout_gradients(cache, target_index=target_index)
        return cache, grad

    def readout_gradient_from_encoded_content(
        self,
        graph: Any,
        encoded_content: torch.Tensor,
        *,
        target_index: int = 0,
        capture_attention: bool = False,
        capture_channels: bool = False,
        capture_layer_inputs: bool = False,
        capture_layer_outputs: bool = False,
    ) -> tuple[ForwardCache, torch.Tensor]:
        cache = self.forward_from_encoded_content(
            graph,
            encoded_content,
            retain_grad=True,
            capture_attention=capture_attention,
            capture_channels=capture_channels,
            capture_layer_inputs=capture_layer_inputs,
            capture_layer_outputs=capture_layer_outputs,
        )
        grad = self._populate_readout_gradients(cache, target_index=target_index)
        return cache, grad

    def _populate_readout_gradients(self, cache: ForwardCache, *, target_index: int = 0) -> torch.Tensor:
        pred = cache.prediction.reshape(-1)[int(target_index)]
        if cache.final_node_states is None:
            raise RuntimeError("GRIT final node states were not captured")
        layer_tensors = []
        if cache.extras is not None:
            layer_tensors = list(cache.extras.get("layer_output_node_state_tensors", []))
        tensors: list[torch.Tensor] = []
        positions: dict[int, int] = {}
        for tensor in [cache.final_node_states, *layer_tensors]:
            if not isinstance(tensor, torch.Tensor):
                continue
            ident = id(tensor)
            if ident not in positions:
                positions[ident] = len(tensors)
                tensors.append(tensor)
        grads = torch.autograd.grad(
            pred,
            tensors,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        by_id = {id(tensor): grad for tensor, grad in zip(tensors, grads)}
        final_grad = by_id.get(id(cache.final_node_states))
        if final_grad is None:
            raise RuntimeError("GRIT final node states did not receive a readout gradient")
        if cache.extras is not None and layer_tensors:
            layer_grads: list[torch.Tensor] = []
            layer_norms: list[float] = []
            missing_layers: list[int] = []
            zero_layers: list[int] = []
            for layer_idx, tensor in enumerate(layer_tensors):
                grad = by_id.get(id(tensor))
                if grad is None:
                    missing_layers.append(layer_idx)
                    continue
                detached = grad.detach().clone()
                norm = float(torch.linalg.vector_norm(detached).detach().cpu().item())
                if norm <= 0.0:
                    zero_layers.append(layer_idx)
                layer_grads.append(detached)
                layer_norms.append(norm)
            if missing_layers:
                raise RuntimeError(
                    "GRIT layer-resolved readout gradients were missing for layer(s) "
                    f"{missing_layers}; refusing to substitute zeros"
                )
            if zero_layers:
                raise RuntimeError(
                    "GRIT layer-resolved readout gradients had zero norm for layer(s) "
                    f"{zero_layers}; layer-resolved channel split would be artificial"
                )
            cache.extras["layer_output_node_gradients"] = layer_grads
            cache.extras["layer_output_node_gradient_norms"] = layer_norms
        return final_grad.detach().clone()

    def predict(self, graph: GraphBatchView) -> torch.Tensor:
        return self.forward_minimal(graph).prediction

    def attention_maps(self, graph: GraphBatchView) -> Optional[list[torch.Tensor]]:
        return self.forward(graph).attention

    def patch_hidden_states(
        self,
        graph: GraphBatchView,
        clamp_nodes: Sequence[int],
        clean_cache: Optional[ForwardCache] = None,
        clamp_until_layer: Optional[int] = None,
    ) -> ForwardCache:
        if clean_cache is None:
            clean_cache = self.forward(graph)
        with torch.no_grad():
            return self._run_with_hooks(
                graph,
                clean_cache=clean_cache,
                clamp_nodes=clamp_nodes,
                clamp_until_layer=clamp_until_layer,
                capture_attention=False,
                capture_channels=False,
                capture_layer_inputs=False,
                capture_layer_outputs=False,
            )

    def patch_hidden_states_from_encoded_content(
        self,
        graph: Any,
        encoded_content: torch.Tensor,
        clamp_nodes: Sequence[int],
        clean_cache: Optional[ForwardCache] = None,
        clamp_until_layer: Optional[int] = None,
        retain_grad: bool = False,
    ) -> ForwardCache:
        if clean_cache is None:
            clean_cache = self.forward(graph)
        return self._run_with_hooks(
            graph,
            content_override=encoded_content,
            clean_cache=clean_cache,
            clamp_nodes=clamp_nodes,
            clamp_until_layer=clamp_until_layer,
            retain_grad=retain_grad,
            capture_attention=False,
            capture_channels=False,
            capture_layer_inputs=False,
            capture_layer_outputs=False,
        )

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
            "dataset_dir": str(self.dataset_dir) if self.dataset_dir is not None else None,
            "device": str(self.device),
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
