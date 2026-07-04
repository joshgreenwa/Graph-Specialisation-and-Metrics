"""True-sparse 1-hop GRIT model.

This module mirrors the official LiamMa/GRIT ZINC RRWP model equations while
keeping attention support sparse from the start.  The strict parity target is
``num_global_tokens=0``: same trainable components as official GRIT with the
1-hop masked RRWP edge encoder used by our current control.  Optional learned
global tokens are an explicit architecture extension and therefore add
parameters.

Official implementation mirrored:
  - ``grit/encoder/type_dict_encoder.py``
  - ``grit/encoder/rrwp_encoder.py``
  - ``grit/layer/grit_layer.py``
  - ``grit/head/san_graph.py``

The attention equations intentionally keep the official GRIT choices: Q/K/V/E
projections, relation-conditioned score scaling/bias, signed sqrt transform,
ReLU score activation, Aw projection, sparse softmax by destination node,
edge-enhanced value rows, degree scaler, residuals, BN, FFN, and add-pooling
graph readout.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional

import torch
from torch import nn
import torch.nn.functional as F


OFFICIAL_ZINC_GRIT_RRWP_PARAMS = 473_473


@dataclass(frozen=True)
class Sparse1HopGRITConfig:
    num_node_types: int = 21
    num_edge_types: int = 4
    rrwp_steps: int = 21
    hidden_dim: int = 64
    layers: int = 10
    heads: int = 8
    dropout: float = 0.0
    attn_dropout: float = 0.2
    clamp: Optional[float] = 5.0
    act: str = "relu"
    batch_norm: bool = True
    layer_norm: bool = False
    bn_momentum: float = 0.1
    bn_no_runner: bool = False
    update_e: bool = True
    norm_e: bool = True
    O_e: bool = True
    edge_enhance: bool = True
    deg_scaler: bool = True
    readout_layers: int = 2
    graph_pooling: str = "add"
    dim_out: int = 1
    num_global_tokens: int = 0
    readout_include_global_tokens: bool = True
    global_tokens_trainable: bool = True
    virtual_edge_features_trainable: bool = True

    def __post_init__(self) -> None:
        if self.hidden_dim % self.heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        if self.graph_pooling != "add":
            raise ValueError("Sparse1HopGRIT currently mirrors official ZINC GRIT add pooling only")


def activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ("identity", "none", ""):
        return nn.Identity()
    raise ValueError(f"unsupported activation {name!r}")


def sparse_softmax_by_dst(src: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Sparse softmax over incoming edges for each destination node.

    ``src`` has shape ``[E, H, 1]`` and ``dst`` has shape ``[E]``.  This mirrors
    official ``pyg_softmax(score, batch.edge_index[1])`` without depending on
    torch_scatter.
    """

    if src.numel() == 0:
        return src
    dst = dst.long()
    flat = src.squeeze(-1)
    heads = flat.size(1)
    max_per_dst = flat.new_full((num_nodes, heads), -torch.inf)
    if hasattr(max_per_dst, "scatter_reduce_"):
        max_per_dst.scatter_reduce_(0, dst[:, None].expand(-1, heads), flat, reduce="amax", include_self=True)
    else:  # pragma: no cover - old torch fallback
        for h in range(heads):
            max_per_dst[:, h].scatter_reduce_(0, dst, flat[:, h], reduce="amax", include_self=True)
    shifted = flat - max_per_dst[dst]
    exp = shifted.exp()
    denom = flat.new_zeros((num_nodes, heads))
    denom.index_add_(0, dst, exp)
    return (exp / (denom[dst] + 1.0e-16)).unsqueeze(-1)


def add_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    out = x.new_zeros((num_graphs, x.size(-1)))
    out.index_add_(0, batch.long(), x)
    return out


class SparseGRITAttention(nn.Module):
    def __init__(self, cfg: Sparse1HopGRITConfig) -> None:
        super().__init__()
        self.out_dim = cfg.hidden_dim // cfg.heads
        self.num_heads = cfg.heads
        self.dropout = nn.Dropout(cfg.attn_dropout)
        self.clamp = abs(float(cfg.clamp)) if cfg.clamp is not None else None
        self.edge_enhance = bool(cfg.edge_enhance)

        self.Q = nn.Linear(cfg.hidden_dim, self.out_dim * self.num_heads, bias=True)
        self.K = nn.Linear(cfg.hidden_dim, self.out_dim * self.num_heads, bias=False)
        self.E = nn.Linear(cfg.hidden_dim, self.out_dim * self.num_heads * 2, bias=True)
        self.V = nn.Linear(cfg.hidden_dim, self.out_dim * self.num_heads, bias=False)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True)
        nn.init.xavier_normal_(self.Aw)
        self.act = activation(cfg.act)
        if self.edge_enhance:
            self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True)
            nn.init.xavier_normal_(self.VeRow)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        src_index = edge_index[0].long()
        dst_index = edge_index[1].long()
        q_h = self.Q(x).view(-1, self.num_heads, self.out_dim)
        k_h = self.K(x).view(-1, self.num_heads, self.out_dim)
        v_h = self.V(x).view(-1, self.num_heads, self.out_dim)

        score = k_h[src_index] + q_h[dst_index]
        e_out = None
        edge_state = None
        if edge_attr is not None:
            edge_state = self.E(edge_attr).view(-1, self.num_heads, self.out_dim * 2)
            e_w, e_b = edge_state[:, :, : self.out_dim], edge_state[:, :, self.out_dim :]
            score = score * e_w
            score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
            score = score + e_b

        score = self.act(score)
        edge_score_state = score
        if edge_attr is not None:
            e_out = score.flatten(1)

        logits = torch.einsum("ehd,dhc->ehc", score, self.Aw)
        if self.clamp is not None:
            logits = torch.clamp(logits, min=-self.clamp, max=self.clamp)
        attn = sparse_softmax_by_dst(logits, dst_index, int(x.size(0)))
        attn = self.dropout(attn)

        msg = v_h[src_index] * attn
        w_v = x.new_zeros((x.size(0), self.num_heads, self.out_dim))
        w_v.index_add_(0, dst_index, msg)

        if self.edge_enhance and edge_attr is not None:
            row_v = x.new_zeros((x.size(0), self.num_heads, self.out_dim))
            row_v.index_add_(0, dst_index, edge_score_state * attn)
            row_v = torch.einsum("nhd,dhc->nhc", row_v, self.VeRow)
            w_v = w_v + row_v

        return w_v, e_out, attn.squeeze(-1)


class SparseGRITLayer(nn.Module):
    def __init__(self, cfg: Sparse1HopGRITConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.dropout = cfg.dropout
        self.residual = True
        self.layer_norm_enabled = cfg.layer_norm
        self.batch_norm_enabled = cfg.batch_norm
        self.update_e = cfg.update_e
        self.deg_scaler = cfg.deg_scaler
        self.act = activation(cfg.act)
        self.attention = SparseGRITAttention(cfg)
        self.O_h = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.O_e = nn.Linear(cfg.hidden_dim, cfg.hidden_dim) if cfg.O_e else nn.Identity()
        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, cfg.hidden_dim, 2))
            nn.init.xavier_normal_(self.deg_coef)
        if self.layer_norm_enabled:
            self.layer_norm1_h = nn.LayerNorm(cfg.hidden_dim)
            self.layer_norm1_e = nn.LayerNorm(cfg.hidden_dim) if cfg.norm_e else nn.Identity()
            self.layer_norm2_h = nn.LayerNorm(cfg.hidden_dim)
        if self.batch_norm_enabled:
            track = not cfg.bn_no_runner
            self.batch_norm1_h = nn.BatchNorm1d(cfg.hidden_dim, track_running_stats=track, eps=1e-5, momentum=cfg.bn_momentum)
            self.batch_norm1_e = nn.BatchNorm1d(cfg.hidden_dim, track_running_stats=track, eps=1e-5, momentum=cfg.bn_momentum) if cfg.norm_e else nn.Identity()
            self.batch_norm2_h = nn.BatchNorm1d(cfg.hidden_dim, track_running_stats=track, eps=1e-5, momentum=cfg.bn_momentum)
        self.FFN_h_layer1 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2)
        self.FFN_h_layer2 = nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        log_deg: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        h_in1 = x
        e_in1 = edge_attr
        h_attn_out, e_attn_out, attn = self.attention(x, edge_index, edge_attr)
        h = h_attn_out.reshape(x.size(0), -1)
        h = F.dropout(h, self.dropout, training=self.training)
        if self.deg_scaler:
            h = torch.stack([h, h * log_deg.view(-1, 1)], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)
        h = self.O_h(h)

        e = None
        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.O_e(e)

        h = h_in1 + h
        if e is not None and e_in1 is not None:
            e = e + e_in1

        if self.layer_norm_enabled:
            h = self.layer_norm1_h(h)
            if e is not None:
                e = self.layer_norm1_e(e)
        if self.batch_norm_enabled:
            h = self.batch_norm1_h(h)
            if e is not None:
                e = self.batch_norm1_e(e)

        h_in2 = h
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)
        h = h_in2 + h
        if self.layer_norm_enabled:
            h = self.layer_norm2_h(h)
        if self.batch_norm_enabled:
            h = self.batch_norm2_h(h)

        return h, e if self.update_e else e_in1, attn


class SANGraphHead(nn.Module):
    def __init__(self, cfg: Sparse1HopGRITConfig) -> None:
        super().__init__()
        layers = [
            nn.Linear(cfg.hidden_dim // (2**l), cfg.hidden_dim // (2 ** (l + 1)), bias=True)
            for l in range(cfg.readout_layers)
        ]
        layers.append(nn.Linear(cfg.hidden_dim // (2**cfg.readout_layers), cfg.dim_out, bias=True))
        self.FC_layers = nn.ModuleList(layers)
        self.L = cfg.readout_layers
        self.activation = activation(cfg.act)

    def forward(self, x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        graph_emb = add_pool(x, batch, num_graphs)
        for layer_idx in range(self.L):
            graph_emb = self.FC_layers[layer_idx](graph_emb)
            graph_emb = self.activation(graph_emb)
        return self.FC_layers[self.L](graph_emb)


class Sparse1HopGRIT(nn.Module):
    """Faithful true-sparse GRIT for ZINC-style batches."""

    def __init__(self, cfg: Sparse1HopGRITConfig = Sparse1HopGRITConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.node_encoder = nn.Embedding(cfg.num_node_types, cfg.hidden_dim)
        self.edge_encoder = nn.Embedding(cfg.num_edge_types, cfg.hidden_dim)
        self.rrwp_abs_encoder = nn.Linear(cfg.rrwp_steps, cfg.hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.rrwp_abs_encoder.weight)
        self.rrwp_rel_encoder = nn.Linear(cfg.rrwp_steps, cfg.hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.rrwp_rel_encoder.weight)
        self.layers = nn.ModuleList([SparseGRITLayer(cfg) for _ in range(cfg.layers)])
        self.post_mp = SANGraphHead(cfg)

        if cfg.num_global_tokens > 0:
            global_token = torch.empty(cfg.num_global_tokens, cfg.hidden_dim)
            nn.init.normal_(global_token)
            if cfg.global_tokens_trainable:
                self.global_token = nn.Parameter(global_token)
            else:
                self.register_buffer("global_token", global_token)
            if cfg.virtual_edge_features_trainable:
                self.virtual_edge_encoder = nn.Embedding(3, cfg.hidden_dim)
                self.register_buffer("virtual_edge_features", None)
            else:
                self.virtual_edge_encoder = None
                virtual_edge_features = torch.empty(3, cfg.hidden_dim)
                nn.init.normal_(virtual_edge_features)
                self.register_buffer("virtual_edge_features", virtual_edge_features)
        else:
            self.register_parameter("global_token", None)
            self.virtual_edge_encoder = None
            self.register_buffer("virtual_edge_features", None)

    @property
    def num_global_tokens(self) -> int:
        return int(self.cfg.num_global_tokens)

    def encode_real_nodes(self, data: Any) -> torch.Tensor:
        x = data.x
        atom = x[:, 0].long() if x.dim() > 1 else x.long()
        h = self.node_encoder(atom)
        if not hasattr(data, "rrwp"):
            raise AttributeError("Sparse1HopGRIT expects data.rrwp absolute RRWP features [num_nodes,k]")
        return h + self.rrwp_abs_encoder(data.rrwp.to(h.device, h.dtype))

    def encode_real_edges(self, data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr
        bond = edge_attr[:, 0].long() if edge_attr.dim() > 1 else edge_attr.long()
        e = self.edge_encoder(bond)
        if not hasattr(data, "edge_rrwp"):
            raise AttributeError("Sparse1HopGRIT expects data.edge_rrwp relative RRWP features [num_edges,k]")
        e = e + self.rrwp_rel_encoder(data.edge_rrwp.to(e.device, e.dtype))

        n = int(data.x.size(0))
        self_index = torch.arange(n, device=edge_index.device, dtype=torch.long)
        self_edge_index = torch.stack([self_index, self_index], dim=0)
        self_e = self.rrwp_rel_encoder(data.rrwp.to(e.device, e.dtype))
        edge_index = torch.cat([edge_index, self_edge_index], dim=1)
        e = torch.cat([e, self_e], dim=0)
        return edge_index, e

    def augment_with_global_tokens(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        log_deg: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        k = self.num_global_tokens
        if k <= 0:
            real_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
            return x, edge_index, edge_attr, batch, log_deg, real_mask

        token_nodes = self.global_token.unsqueeze(0).expand(num_graphs, -1, -1).reshape(num_graphs * k, -1)
        token_batch = torch.arange(num_graphs, device=x.device).repeat_interleave(k)
        token_offset = x.size(0)
        token_index = torch.arange(num_graphs * k, device=x.device, dtype=torch.long) + token_offset
        x_aug = torch.cat([x, token_nodes], dim=0)
        batch_aug = torch.cat([batch.long(), token_batch], dim=0)

        edges: list[torch.Tensor] = [edge_index]
        attrs: list[torch.Tensor] = [edge_attr]
        edge_types: list[torch.Tensor] = []
        real_nodes = torch.arange(x.size(0), device=x.device, dtype=torch.long)
        for graph_id in range(num_graphs):
            real = real_nodes[batch.long() == graph_id]
            toks = token_index[token_batch == graph_id]
            if real.numel() == 0:
                continue
            # real -> global, global -> real, and global -> global.
            r_rep = real.repeat_interleave(k)
            t_tile = toks.repeat(real.numel())
            edges.append(torch.stack([r_rep, t_tile], dim=0))
            edge_types.append(torch.zeros(r_rep.numel(), dtype=torch.long, device=x.device))
            t_rep = toks.repeat_interleave(real.numel())
            r_tile = real.repeat(k)
            edges.append(torch.stack([t_rep, r_tile], dim=0))
            edge_types.append(torch.ones(t_rep.numel(), dtype=torch.long, device=x.device))
            src = toks.repeat_interleave(k)
            dst = toks.repeat(k)
            edges.append(torch.stack([src, dst], dim=0))
            edge_types.append(torch.full((src.numel(),), 2, dtype=torch.long, device=x.device))

        if edge_types:
            vt = torch.cat(edge_types, dim=0)
            if self.virtual_edge_encoder is not None:
                attrs.append(self.virtual_edge_encoder(vt))
            elif self.virtual_edge_features is not None:
                attrs.append(self.virtual_edge_features.to(x.device, x.dtype)[vt])
            else:  # pragma: no cover - defensive; cannot happen for k > 0
                raise RuntimeError("global-token edges require virtual edge features")
        edge_index_aug = torch.cat(edges, dim=1)
        edge_attr_aug = torch.cat(attrs, dim=0)

        token_log_deg = []
        for graph_id in range(num_graphs):
            real_count = int((batch.long() == graph_id).sum().item())
            deg = max(1, real_count + k)
            token_log_deg.extend([math.log(deg + 1.0)] * k)
        log_deg_aug = torch.cat([log_deg.view(-1), x.new_tensor(token_log_deg)], dim=0)
        real_mask = torch.cat(
            [
                torch.ones(x.size(0), dtype=torch.bool, device=x.device),
                torch.zeros(num_graphs * k, dtype=torch.bool, device=x.device),
            ],
            dim=0,
        )
        return x_aug, edge_index_aug, edge_attr_aug, batch_aug, log_deg_aug, real_mask

    def forward(self, data: Any, *, return_cache: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if not hasattr(data, "batch") or data.batch is None:
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        else:
            batch = data.batch.long()
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        x = self.encode_real_nodes(data)
        edge_index, edge_attr = self.encode_real_edges(data)
        if hasattr(data, "log_deg"):
            log_deg = data.log_deg.to(x.device, x.dtype).view(-1)
        else:
            deg = x.new_zeros(x.size(0))
            deg.index_add_(0, edge_index[1].long(), torch.ones(edge_index.size(1), dtype=x.dtype, device=x.device))
            log_deg = torch.log(deg + 1.0)
        x, edge_index, edge_attr, batch, log_deg, real_mask = self.augment_with_global_tokens(
            x, edge_index.to(x.device), edge_attr, batch.to(x.device), log_deg, num_graphs
        )

        attentions: list[torch.Tensor] = []
        for layer in self.layers:
            x, edge_attr, attn = layer(x, edge_index, edge_attr, log_deg)
            if return_cache:
                attentions.append(attn.detach())

        readout_x = x if self.cfg.readout_include_global_tokens else x[real_mask]
        readout_batch = batch if self.cfg.readout_include_global_tokens else batch[real_mask]
        pred = self.post_mp(readout_x, readout_batch, num_graphs)
        if return_cache:
            return pred, {
                "edge_index": edge_index.detach(),
                "attention": attentions,
                "batch": batch.detach(),
                "real_node_mask": real_mask.detach(),
            }
        return pred


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def zinc_sparse1hop_grit_config(
    *,
    num_global_tokens: int = 0,
    parameter_match_global_tokens: bool = False,
) -> Sparse1HopGRITConfig:
    return Sparse1HopGRITConfig(
        num_global_tokens=int(num_global_tokens),
        global_tokens_trainable=not bool(parameter_match_global_tokens),
        virtual_edge_features_trainable=not bool(parameter_match_global_tokens),
    )


def assert_official_parameter_parity(model: nn.Module) -> None:
    count = parameter_count(model)
    if count != OFFICIAL_ZINC_GRIT_RRWP_PARAMS:
        raise AssertionError(
            f"Sparse1HopGRIT K=0 parameter count {count:,} does not match "
            f"official ZINC GRIT RRWP {OFFICIAL_ZINC_GRIT_RRWP_PARAMS:,}"
        )


def _strip_known_prefix(key: str) -> str:
    for prefix in (
        "model.model.",
        "model.",
        "module.model.",
        "module.",
        "net.",
    ):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def translate_official_grit_key(key: str) -> Optional[str]:
    """Translate official GRIT checkpoint keys into ``Sparse1HopGRIT`` keys.

    The translator covers the shared K=0 architecture.  K>0 global-token
    parameters intentionally have no official source and remain randomly
    initialized.
    """

    key = _strip_known_prefix(key)
    replacements = (
        ("encoder.node_encoder.encoder.", "node_encoder."),
        ("encoder.edge_encoder.encoder.", "edge_encoder."),
        ("rrwp_abs_encoder.fc.", "rrwp_abs_encoder."),
        ("rrwp_rel_encoder.fc.", "rrwp_rel_encoder."),
    )
    for old, new in replacements:
        if key.startswith(old):
            return new + key[len(old) :]
    if key.startswith("layers.") or key.startswith("post_mp."):
        return key
    return None


def translated_official_grit_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    model: Sparse1HopGRIT,
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    target = model.state_dict()
    translated: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    shape_mismatch: list[str] = []
    for key, value in state_dict.items():
        new_key = translate_official_grit_key(str(key))
        if new_key is None or new_key not in target:
            skipped.append(str(key))
            continue
        if tuple(value.shape) != tuple(target[new_key].shape):
            shape_mismatch.append(f"{key}->{new_key}: {tuple(value.shape)} != {tuple(target[new_key].shape)}")
            continue
        translated[new_key] = value
    return translated, skipped, shape_mismatch


def load_official_grit_state_dict(
    model: Sparse1HopGRIT,
    state_dict: Mapping[str, torch.Tensor],
    *,
    strict_shared: bool = True,
) -> dict[str, Any]:
    translated, skipped, shape_mismatch = translated_official_grit_state_dict(state_dict, model)
    result = model.load_state_dict(translated, strict=False)
    missing_shared = [
        key
        for key in result.missing_keys
        if not (
            key.startswith("global_token")
            or key.startswith("virtual_edge_encoder.")
            or key.startswith("virtual_edge_features")
        )
    ]
    if strict_shared and (shape_mismatch or missing_shared):
        raise RuntimeError(
            "Official GRIT checkpoint did not cover the shared Sparse1HopGRIT parameters: "
            f"missing_shared={missing_shared[:20]}, shape_mismatch={shape_mismatch[:20]}"
        )
    return {
        "loaded_keys": len(translated),
        "skipped_keys": skipped,
        "shape_mismatch": shape_mismatch,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


__all__ = [
    "OFFICIAL_ZINC_GRIT_RRWP_PARAMS",
    "Sparse1HopGRIT",
    "Sparse1HopGRITConfig",
    "assert_official_parameter_parity",
    "load_official_grit_state_dict",
    "parameter_count",
    "translate_official_grit_key",
    "translated_official_grit_state_dict",
    "zinc_sparse1hop_grit_config",
]
