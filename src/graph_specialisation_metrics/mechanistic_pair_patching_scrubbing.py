"""Paired graph activation patching and pair-set scrubbing for official GRIT.

This runner is deliberately separate from mechanistic_operator_analysis.py.  The
existing operator analysis asks where task-weighted realised transport lands; this
file asks whether solver-derived pair sets can causally rescue a corrupt graph or
damage a clean graph more than matched controls.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from graph_specialisation_metrics import mechanistic_operator_analysis as moa


EPS = 1.0e-12
PATCH_TARGETS = ("routing_logits", "attention", "pair_value", "pair_state", "head_output")
EXPERIMENTS = ("pair_patching", "pair_scrubbing")
ANALYSIS_PRESETS = ("custom", "pilot", "core_fast", "confirmatory_1h", "h2_validation")
FLOW_CORE_OPERATORS = (
    "saturated_edge",
    "min_cut_crossing_edge",
    "shortest_st_path_edge",
)
BIPARTITE_CORE_OPERATORS = (
    "optimum_matching_edge",
    "alternating_forest_edge",
    "unmatched_feasible_edge",
)
CORE_PATCH_TARGETS = ("routing_logits", "pair_value", "pair_state")
PILOT_PATCH_TARGETS = ("pair_value", "pair_state")
H2_FLOW_OPERATORS = ("saturated_edge", "min_cut_crossing_edge")
H2_BIPARTITE_OPERATORS = ("optimum_matching_edge", "alternating_forest_edge")


@dataclass(frozen=True)
class GraphScore:
    position: int
    graph_index: int
    num_nodes: int
    pred_score: float
    target_score: float
    primary: float


@dataclass(frozen=True)
class PairSpec:
    pair_id: int
    clean_position: int
    corrupt_position: int
    clean_graph_index: int
    corrupt_graph_index: int
    num_nodes: int
    clean_pred_score: float
    corrupt_pred_score: float
    pred_delta: float
    clean_target_score: float
    corrupt_target_score: float
    target_delta: float
    wrong_source_position: int = -1


@dataclass
class LayerActivation:
    layer: int
    src: torch.Tensor
    dst: torch.Tensor
    local_src: torch.Tensor
    local_dst: torch.Tensor
    attention: torch.Tensor
    logits: torch.Tensor
    node_message: torch.Tensor
    pair_message: torch.Tensor
    message: torch.Tensor
    pair_e: Optional[torch.Tensor]
    pair_state_input: Optional[torch.Tensor]
    pair_state_output: Optional[torch.Tensor]
    heads: int
    message_dim: int


@dataclass(frozen=True)
class BatchScores:
    scores: list[float]
    primary: list[float]
    targets: list[float]


@dataclass(frozen=True)
class ControlMask:
    control_type: str
    control_id: str
    control_operator: str
    source_type: str
    dense_mask: torch.Tensor


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def parse_csv(raw: str, *, all_values: Sequence[str]) -> tuple[str, ...]:
    if raw == "all":
        return tuple(all_values)
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = sorted(set(values) - set(all_values))
    if unknown:
        raise ValueError(f"unknown values {unknown}; expected {list(all_values)} or all")
    return values


def cli_supplied(argv: Sequence[str], *names: str) -> bool:
    return any(item == name or item.startswith(f"{name}=") for item in argv for name in names)


def task_core_operators(task: str) -> str:
    if "flow" in task:
        return ",".join(FLOW_CORE_OPERATORS)
    if "bipartite" in task or "matching" in task:
        return ",".join(BIPARTITE_CORE_OPERATORS)
    return "directed_edge,global_nonedge"


def task_h2_operators(task: str) -> str:
    if "flow" in task:
        return ",".join(H2_FLOW_OPERATORS)
    if "bipartite" in task or "matching" in task:
        return ",".join(H2_BIPARTITE_OPERATORS)
    return task_core_operators(task)


def apply_analysis_preset(args: argparse.Namespace, argv: Sequence[str]) -> None:
    preset = getattr(args, "analysis_preset", "custom")
    if preset == "custom":
        return
    if preset == "pilot":
        defaults = {
            "experiments": "pair_patching",
            "num_graphs": 384,
            "num_pairs": 24,
            "patch_targets": ",".join(PILOT_PATCH_TARGETS),
            "operator_masks": task_core_operators(args.task),
            "matched_random_controls": 1,
            "include_outside_operator_control": False,
            "include_wrong_operator_control": False,
            "include_wrong_source_control": False,
        }
    elif preset == "core_fast":
        defaults = {
            "experiments": "pair_patching,pair_scrubbing",
            "num_graphs": 768,
            "num_pairs": 64,
            "patch_targets": ",".join(CORE_PATCH_TARGETS),
            "operator_masks": task_core_operators(args.task),
            "matched_random_controls": 2,
            "include_outside_operator_control": False,
            "include_wrong_operator_control": False,
            "include_wrong_source_control": False,
        }
    elif preset == "confirmatory_1h":
        defaults = {
            "experiments": "pair_patching,pair_scrubbing",
            "num_graphs": 1536,
            "num_pairs": 128,
            "patch_targets": ",".join(CORE_PATCH_TARGETS),
            "operator_masks": task_core_operators(args.task),
            "matched_random_controls": 4,
            "include_outside_operator_control": False,
            "include_wrong_operator_control": False,
            "include_wrong_source_control": False,
        }
    elif preset == "h2_validation":
        defaults = {
            "experiments": "pair_scrubbing",
            "num_graphs": 2048,
            "num_pairs": 256,
            "patch_targets": "pair_value,pair_state",
            "operator_masks": task_h2_operators(args.task),
            "matched_random_controls": 4,
            "include_outside_operator_control": False,
            "include_wrong_operator_control": False,
            "include_wrong_source_control": False,
        }
    else:
        raise ValueError(f"unknown analysis preset {preset!r}")

    flag_by_dest = {
        "experiments": ("--experiments",),
        "num_graphs": ("--num-graphs",),
        "num_pairs": ("--num-pairs",),
        "patch_targets": ("--patch-targets",),
        "operator_masks": ("--operator-masks",),
        "matched_random_controls": ("--matched-random-controls",),
        "include_outside_operator_control": (
            "--include-outside-operator-control",
            "--no-include-outside-operator-control",
        ),
        "include_wrong_operator_control": (
            "--include-wrong-operator-control",
            "--no-include-wrong-operator-control",
        ),
        "include_wrong_source_control": (
            "--include-wrong-source-control",
            "--no-include-wrong-source-control",
        ),
    }
    for dest, value in defaults.items():
        if not cli_supplied(argv, *flag_by_dest[dest]):
            setattr(args, dest, value)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def git_sha(cwd: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def safe_restoration(clean: float, corrupt: float, patched: float, min_abs_delta: float) -> float:
    denom = clean - corrupt
    if not math.isfinite(denom) or abs(denom) < min_abs_delta:
        raise RuntimeError(
            f"degenerate clean/corrupt score delta for restoration: "
            f"clean={clean:.6g} corrupt={corrupt:.6g} min_abs_delta={min_abs_delta:.6g}"
        )
    return (patched - corrupt) / denom


def score_prediction(
    runner: Any,
    pred: torch.Tensor,
    batch: Any,
    target_stats: Optional[Mapping[str, float]],
) -> BatchScores:
    scores: list[float] = []
    primary: list[float] = []
    targets: list[float] = []
    if batch.task_type == "graph_regression":
        pred_raw = runner.denormalize_graph_target(pred.detach().cpu(), target_stats)
        target = batch.graph_target.detach().cpu().reshape(-1)
        error = (pred_raw.reshape(-1) - target).abs()
        scores.extend(float(v) for v in pred_raw.reshape(-1))
        primary.extend(float(v) for v in error.reshape(-1))
        targets.extend(float(v) for v in target.reshape(-1))
        return BatchScores(scores, primary, targets)
    if batch.task_type == "edge_binary":
        pred_cpu = pred.detach().cpu()
        target_cpu = batch.edge_target.detach().cpu()
        edge_batch_cpu = batch.edge_batch.detach().cpu()
        for graph_idx in range(batch.num_graphs):
            mask = edge_batch_cpu == graph_idx
            if not bool(mask.any()):
                scores.append(float("nan"))
                primary.append(float("nan"))
                targets.append(float("nan"))
                continue
            signed = (2.0 * target_cpu[mask].float() - 1.0) * pred_cpu[mask].float()
            metrics = runner.binary_metrics(pred_cpu[mask], target_cpu[mask])
            scores.append(float(signed.mean()))
            primary.append(float(metrics["f1"]))
            targets.append(float(target_cpu[mask].float().mean()))
        return BatchScores(scores, primary, targets)
    if batch.task_type == "node_binary":
        pred_cpu = pred.detach().cpu()
        target_cpu = batch.node_target.detach().cpu()
        node_mask_cpu = batch.node_mask.detach().cpu()
        for graph_idx in range(batch.num_graphs):
            mask = node_mask_cpu[graph_idx]
            signed = (2.0 * target_cpu[graph_idx, mask].float() - 1.0) * pred_cpu[graph_idx, mask].float()
            metrics = runner.binary_metrics(pred_cpu[graph_idx, mask], target_cpu[graph_idx, mask])
            scores.append(float(signed.mean()))
            primary.append(float(metrics["f1"]))
            targets.append(float(target_cpu[graph_idx, mask].float().mean()))
        return BatchScores(scores, primary, targets)
    raise ValueError(f"unsupported task type for paired patching: {batch.task_type}")


@torch.no_grad()
def score_selected_graphs(args: argparse.Namespace, loaded: moa.LoadedExperiment) -> list[GraphScore]:
    rows: list[GraphScore] = []
    index_to_position = {graph_index: pos for pos, graph_index in enumerate(loaded.selected_indices)}
    model = loaded.model
    model.eval()
    if loaded.analysis_batches:
        iterator = loaded.analysis_batches
        cached = True
    else:
        batch_size = max(1, int(args.analysis_cache_batch_size or args.batch_size))
        iterator = moa.graph_batches(loaded.graphs, loaded.selected_indices, batch_size)
        cached = False
    for graph_indices, item in iterator:
        batch = item if cached else loaded.runner.collate_graphs(item)
        batch = moa.batch_to_device(batch, loaded.device)
        with moa.autocast_context(loaded.device, args.autocast_dtype):
            pred = model(batch)
        scored = score_prediction(loaded.runner, pred, batch, loaded.target_stats)
        for local_idx, graph_index in enumerate(graph_indices):
            position = index_to_position[int(graph_index)]
            rows.append(
                GraphScore(
                    position=position,
                    graph_index=int(graph_index),
                    num_nodes=int(batch.graph_num_nodes[local_idx].detach().cpu()),
                    pred_score=scored.scores[local_idx],
                    target_score=scored.targets[local_idx],
                    primary=scored.primary[local_idx],
                )
            )
    rows.sort(key=lambda row: row.position)
    return rows


def adaptive_delta(scores: Sequence[GraphScore], field: str, quantile: float) -> float:
    by_n: dict[int, list[float]] = defaultdict(list)
    for row in scores:
        value = getattr(row, field)
        if math.isfinite(value):
            by_n[row.num_nodes].append(float(value))
    deltas: list[float] = []
    for values in by_n.values():
        values = sorted(values)
        for lo in range(len(values)):
            for hi in range(lo + 1, len(values)):
                delta = abs(values[hi] - values[lo])
                if delta > EPS:
                    deltas.append(delta)
    if not deltas:
        return 0.0
    deltas.sort()
    idx = min(len(deltas) - 1, max(0, int(round(float(quantile) * (len(deltas) - 1)))))
    return float(deltas[idx])


def select_pairs(
    scores: Sequence[GraphScore],
    *,
    num_pairs: int,
    min_pred_delta: Optional[float],
    min_target_delta: Optional[float],
    adaptive_quantile: float,
    seed: int,
) -> list[PairSpec]:
    if num_pairs <= 0:
        raise ValueError("--num-pairs must be positive")
    pred_threshold = (
        adaptive_delta(scores, "pred_score", adaptive_quantile)
        if min_pred_delta is None
        else float(min_pred_delta)
    )
    target_threshold = (
        adaptive_delta(scores, "target_score", adaptive_quantile)
        if min_target_delta is None
        else float(min_target_delta)
    )
    by_n: dict[int, list[GraphScore]] = defaultdict(list)
    for row in scores:
        if math.isfinite(row.pred_score) and math.isfinite(row.target_score):
            by_n[row.num_nodes].append(row)
    rng = random.Random(seed)
    node_sizes = sorted(by_n)
    rng.shuffle(node_sizes)
    pairs: list[PairSpec] = []
    used: set[int] = set()
    for n in node_sizes:
        group = sorted(by_n[n], key=lambda row: (row.pred_score, row.target_score, row.graph_index))
        lo = 0
        hi = len(group) - 1
        while lo < hi and len(pairs) < num_pairs:
            corrupt = group[lo]
            clean = group[hi]
            lo += 1
            hi -= 1
            if clean.position in used or corrupt.position in used:
                continue
            pred_delta = clean.pred_score - corrupt.pred_score
            target_delta = clean.target_score - corrupt.target_score
            if abs(pred_delta) < pred_threshold and abs(target_delta) < target_threshold:
                continue
            pair_id = len(pairs)
            pairs.append(
                PairSpec(
                    pair_id=pair_id,
                    clean_position=clean.position,
                    corrupt_position=corrupt.position,
                    clean_graph_index=clean.graph_index,
                    corrupt_graph_index=corrupt.graph_index,
                    num_nodes=n,
                    clean_pred_score=clean.pred_score,
                    corrupt_pred_score=corrupt.pred_score,
                    pred_delta=pred_delta,
                    clean_target_score=clean.target_score,
                    corrupt_target_score=corrupt.target_score,
                    target_delta=target_delta,
                )
            )
            used.add(clean.position)
            used.add(corrupt.position)
        if len(pairs) >= num_pairs:
            break
    if len(pairs) < num_pairs:
        raise RuntimeError(
            f"only found {len(pairs)} valid same-node-count clean/corrupt pairs; "
            f"requested {num_pairs}. Candidate thresholds were "
            f"pred_delta>={pred_threshold:.6g} or target_delta>={target_threshold:.6g}."
        )
    return attach_wrong_sources(pairs)


def attach_wrong_sources(pairs: Sequence[PairSpec]) -> list[PairSpec]:
    by_n: dict[int, list[PairSpec]] = defaultdict(list)
    for pair in pairs:
        by_n[pair.num_nodes].append(pair)
    out: list[PairSpec] = []
    for pair in pairs:
        candidates = [other for other in by_n[pair.num_nodes] if other.clean_position != pair.clean_position]
        wrong = candidates[0].clean_position if candidates else -1
        out.append(
            PairSpec(
                pair_id=pair.pair_id,
                clean_position=pair.clean_position,
                corrupt_position=pair.corrupt_position,
                clean_graph_index=pair.clean_graph_index,
                corrupt_graph_index=pair.corrupt_graph_index,
                num_nodes=pair.num_nodes,
                clean_pred_score=pair.clean_pred_score,
                corrupt_pred_score=pair.corrupt_pred_score,
                pred_delta=pair.pred_delta,
                clean_target_score=pair.clean_target_score,
                corrupt_target_score=pair.corrupt_target_score,
                target_delta=pair.target_delta,
                wrong_source_position=wrong,
            )
        )
    return out


def pair_rows(pairs: Sequence[PairSpec]) -> list[dict[str, Any]]:
    return [
        {
            "pair_id": pair.pair_id,
            "clean_position": pair.clean_position,
            "corrupt_position": pair.corrupt_position,
            "clean_graph_index": pair.clean_graph_index,
            "corrupt_graph_index": pair.corrupt_graph_index,
            "num_nodes": pair.num_nodes,
            "clean_pred_score": pair.clean_pred_score,
            "corrupt_pred_score": pair.corrupt_pred_score,
            "pred_delta": pair.pred_delta,
            "clean_target_score": pair.clean_target_score,
            "corrupt_target_score": pair.corrupt_target_score,
            "target_delta": pair.target_delta,
            "wrong_source_position": pair.wrong_source_position,
        }
        for pair in pairs
    ]


class GRITActivationCapture:
    def __init__(self, model: nn.Module) -> None:
        if not hasattr(model, "layers"):
            raise TypeError("paired patching expects an official GRIT/static-GRIT model with layers")
        self.model = model
        self.records: dict[int, LayerActivation] = {}
        self.handles: list[Any] = []

    def __enter__(self) -> "GRITActivationCapture":
        self.records = {}
        self.handles = []
        for layer_idx, layer in enumerate(self.model.layers):
            self.handles.append(layer.attention.register_forward_hook(self._make_hook(layer_idx)))
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _make_hook(self, layer_idx: int):
        def hook(module: nn.Module, inputs: tuple[Any, ...], outputs: Any) -> None:
            del outputs
            pyg_batch = inputs[0]
            node_msg, pair_msg, logits, _edge_state = moa.grit_attention_components(module, pyg_batch)
            attn = pyg_batch.attn.squeeze(-1)
            if attn.dim() != 2:
                raise RuntimeError(f"expected sparse attention [E,H], got {tuple(attn.shape)}")
            edge_index = pyg_batch.edge_index.long()
            _graph, local_src, local_dst = moa.edge_local_coordinates(
                edge_index[0],
                edge_index[1],
                [int(v) for v in pyg_batch.graph_num_nodes.detach().cpu().tolist()],
            )
            pair_e = None
            if getattr(pyg_batch, "E", None) is not None:
                pair_e = pyg_batch.E.detach().float()
            self.records[layer_idx] = LayerActivation(
                layer=layer_idx,
                src=edge_index[0].detach(),
                dst=edge_index[1].detach(),
                local_src=local_src.detach(),
                local_dst=local_dst.detach(),
                attention=attn.detach().float(),
                logits=logits.detach().float(),
                node_message=node_msg.detach().float(),
                pair_message=pair_msg.detach().float(),
                message=(node_msg + pair_msg).detach().float(),
                pair_e=pair_e,
                pair_state_input=None
                if getattr(pyg_batch, "edge_attr", None) is None
                else pyg_batch.edge_attr.detach().float(),
                pair_state_output=None
                if getattr(pyg_batch, "wE", None) is None
                else pyg_batch.wE.detach().float(),
                heads=int(attn.size(1)),
                message_dim=int(node_msg.size(-1)),
            )

        return hook


def capture_prediction(
    args: argparse.Namespace,
    loaded: moa.LoadedExperiment,
    batch: Any,
) -> tuple[torch.Tensor, dict[int, LayerActivation]]:
    loaded.model.eval()
    with GRITActivationCapture(loaded.model) as capture:
        with torch.no_grad():
            with moa.autocast_context(loaded.device, args.autocast_dtype):
                pred = loaded.model(batch)
    if not capture.records:
        raise RuntimeError("no GRIT activation records were captured")
    return pred.detach(), capture.records


def sparse_positions_for_mask(record: LayerActivation, dense_mask: torch.Tensor) -> torch.Tensor:
    if dense_mask.dim() == 3:
        if dense_mask.size(0) != 1:
            raise ValueError("paired patching currently processes one graph at a time")
        dense_mask = dense_mask[0]
    selected = dense_mask[record.local_dst.long(), record.local_src.long()] > 0
    return selected.to(device=record.local_src.device)


def assert_nonzero_sparse_coverage(record: LayerActivation, dense_mask: torch.Tensor, label: str) -> int:
    selected = sparse_positions_for_mask(record, dense_mask)
    count = int(selected.sum().detach().cpu())
    if count <= 0:
        raise RuntimeError(f"zero sparse pair coverage for {label}")
    return count


def align_source_positions(source: LayerActivation, current_local_src: torch.Tensor, current_local_dst: torch.Tensor, n: int) -> torch.Tensor:
    source_keys = source.local_dst.long() * int(n) + source.local_src.long()
    current_keys = current_local_dst.long() * int(n) + current_local_src.long()
    order = torch.argsort(source_keys)
    sorted_keys = source_keys[order]
    pos = torch.searchsorted(sorted_keys, current_keys)
    max_pos = max(int(sorted_keys.numel()) - 1, 0)
    clamped = pos.clamp(max=max_pos)
    found = (pos < sorted_keys.numel()) & (sorted_keys[clamped] == current_keys)
    if not bool(found.all()):
        missing = int((~found).sum().detach().cpu())
        raise RuntimeError(f"could not align {missing} sparse pair positions by local (dst,src)")
    return order[clamped]


def renormalize_attention(attention: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if attention.dim() != 2:
        raise ValueError(f"expected attention [E,H], got {tuple(attention.shape)}")
    denom = attention.new_zeros((int(num_nodes), int(attention.size(1))))
    denom.index_add_(0, dst.long(), attention)
    return attention / denom[dst.long()].clamp_min(EPS)


def replace_selected(
    current: torch.Tensor,
    replacement: torch.Tensor,
    selected: torch.Tensor,
    head: int,
) -> torch.Tensor:
    out = current.clone()
    replacement = replacement.to(device=current.device, dtype=current.dtype)
    if head < 0:
        out[selected] = replacement[selected]
    else:
        out[selected, head] = replacement[selected, head]
    return out


def matched_replacement(
    values: torch.Tensor,
    strata: torch.Tensor,
    *,
    exclude: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if exclude is None:
        if values.dim() == 2:
            return moa.matched_mean_sparse(values.unsqueeze(-1), strata).squeeze(-1)
        return moa.matched_mean_sparse(values, strata)
    out = torch.empty_like(values)
    for code in torch.unique(strata.detach().cpu()).tolist():
        target = strata == int(code)
        source = target & ~exclude
        if not bool(source.any()):
            source = target
        if values.dim() == 2:
            out[target] = values[source].mean(dim=0, keepdim=True)
        else:
            out[target] = values[source].mean(dim=0, keepdim=True)
    return out


class GRITInterventionContext:
    def __init__(
        self,
        model: nn.Module,
        *,
        official_batch: Any,
        layer: int,
        head: int,
        target: str,
        dense_mask: torch.Tensor,
        mode: str,
        source: Optional[LayerActivation] = None,
    ) -> None:
        if target not in PATCH_TARGETS:
            raise ValueError(f"unknown patch target {target!r}")
        if mode not in {"patch", "scrub"}:
            raise ValueError(f"unknown intervention mode {mode!r}")
        self.model = model
        self.official_batch = official_batch
        self.layer = int(layer)
        self.head = int(head)
        self.target = target
        self.dense_mask = dense_mask
        self.mode = mode
        self.source = source
        self.original: Optional[Any] = None

    def __enter__(self) -> "GRITInterventionContext":
        attention = self.model.layers[self.layer].attention
        self.original = attention.propagate_attention
        attention.propagate_attention = self._make_propagate(attention)
        return self

    def __exit__(self, *_exc: object) -> None:
        attention = self.model.layers[self.layer].attention
        if self.original is not None:
            attention.propagate_attention = self.original
        self.original = None

    def _source_aligned(self, pyg_batch: Any, n: int) -> Optional[dict[str, torch.Tensor]]:
        if self.source is None:
            return None
        edge_index = pyg_batch.edge_index.long()
        _graph, local_src, local_dst = moa.edge_local_coordinates(
            edge_index[0],
            edge_index[1],
            [int(v) for v in pyg_batch.graph_num_nodes.detach().cpu().tolist()],
        )
        pos = align_source_positions(self.source, local_src, local_dst, n).to(device=edge_index.device)
        out: dict[str, torch.Tensor] = {
            "attention": self.source.attention.to(edge_index.device)[pos],
            "logits": self.source.logits.to(edge_index.device)[pos],
            "pair_message": self.source.pair_message.to(edge_index.device)[pos],
            "message": self.source.message.to(edge_index.device)[pos],
        }
        if self.source.pair_e is not None:
            out["pair_e"] = self.source.pair_e.to(edge_index.device)[pos]
        return out

    def _selected(self, pyg_batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_index = pyg_batch.edge_index.long()
        _graph, local_src, local_dst = moa.edge_local_coordinates(
            edge_index[0],
            edge_index[1],
            [int(v) for v in pyg_batch.graph_num_nodes.detach().cpu().tolist()],
        )
        dense = self.dense_mask
        if dense.dim() == 3:
            dense = dense[0]
        selected = dense.to(device=edge_index.device)[local_dst.long(), local_src.long()] > 0
        if int(selected.sum().detach().cpu()) <= 0:
            raise RuntimeError(
                f"zero sparse coverage during {self.mode} "
                f"layer={self.layer} head={self.head} target={self.target}"
            )
        return selected, local_src, local_dst

    def _patch_or_scrub(self, current: torch.Tensor, selected: torch.Tensor, strata: torch.Tensor, source_value: Optional[torch.Tensor]) -> torch.Tensor:
        if self.mode == "patch":
            if source_value is None:
                raise RuntimeError("patch intervention requires source activations")
            return replace_selected(current, source_value, selected, self.head)
        return replace_selected(
            current,
            matched_replacement(current, strata, exclude=selected),
            selected,
            self.head,
        )

    def _patch_or_scrub_pair_e(
        self,
        pair_e: torch.Tensor,
        selected: torch.Tensor,
        strata: torch.Tensor,
        source_pair_e: Optional[torch.Tensor],
        heads: int,
    ) -> torch.Tensor:
        view = pair_e.view(pair_e.size(0), heads, -1)
        if self.mode == "patch":
            if source_pair_e is None:
                raise RuntimeError("pair_state patch requires source pair-E activations")
            replacement = source_pair_e.to(device=pair_e.device, dtype=pair_e.dtype).view(pair_e.size(0), heads, -1)
        else:
            replacement = matched_replacement(view, strata, exclude=selected).to(dtype=pair_e.dtype)
        patched = replace_selected(view, replacement, selected, self.head)
        return patched.reshape_as(pair_e)

    def _make_propagate(self, attention_module: nn.Module):
        def propagate(pyg_batch: Any) -> None:
            from torch_scatter import scatter

            selected, local_src, _local_dst = self._selected(pyg_batch)
            n = int(self.official_batch.graph_num_nodes[0].detach().cpu())
            source = self._source_aligned(pyg_batch, n)
            strata = moa.sparse_pair_strata(pyg_batch, self.official_batch)

            if self.target == "pair_state":
                if getattr(pyg_batch, "E", None) is None:
                    raise RuntimeError("pair_state intervention requested but pyg_batch.E is missing")
                src_e = None if source is None else source.get("pair_e")
                pyg_batch.E = self._patch_or_scrub_pair_e(
                    pyg_batch.E,
                    selected,
                    strata,
                    src_e,
                    heads=int(pyg_batch.V_h.size(1)),
                )

            node_msg, pair_msg, logits, edge_state = moa.grit_attention_components(
                attention_module,
                pyg_batch,
            )
            if self.target == "routing_logits":
                logits = self._patch_or_scrub(
                    logits,
                    selected,
                    strata,
                    None if source is None else source["logits"],
                )
            if self.target == "pair_value":
                pair_msg = self._patch_or_scrub(
                    pair_msg,
                    selected,
                    strata,
                    None if source is None else source["pair_message"],
                )

            score = moa.pyg_sparse_softmax(
                logits.unsqueeze(-1),
                pyg_batch.edge_index[1],
                pyg_batch.num_nodes,
            ).squeeze(-1)
            if self.target == "attention":
                score = self._patch_or_scrub(
                    score,
                    selected,
                    strata,
                    None if source is None else source["attention"],
                )
                score = renormalize_attention(score.clamp_min(0.0), pyg_batch.edge_index[1], pyg_batch.num_nodes)

            score = attention_module.dropout(score.unsqueeze(-1))
            pyg_batch.attn = score
            if getattr(pyg_batch, "E", None) is not None:
                pyg_batch.wE = edge_state.flatten(1)

            message = node_msg + pair_msg
            weighted = message * score
            if self.target == "head_output":
                source_weighted = None
                if source is not None:
                    source_weighted = source["message"] * source["attention"].unsqueeze(-1)
                weighted = self._patch_or_scrub(weighted, selected, strata, source_weighted)

            pyg_batch.wV = torch.zeros_like(pyg_batch.V_h)
            scatter(weighted, pyg_batch.edge_index[1], dim=0, out=pyg_batch.wV, reduce="add")

        return propagate


def single_graph_batch(loaded: moa.LoadedExperiment, position: int) -> Any:
    batch = loaded.runner.collate_graphs([loaded.graphs[position]])
    return moa.batch_to_device(batch, loaded.device)


@torch.no_grad()
def predict_with_context(
    args: argparse.Namespace,
    loaded: moa.LoadedExperiment,
    batch: Any,
    context: Optional[GRITInterventionContext] = None,
) -> torch.Tensor:
    loaded.model.eval()
    manager = context if context is not None else nullcontext()
    with manager:
        with moa.autocast_context(loaded.device, args.autocast_dtype):
            pred = loaded.model(batch)
    if not torch.isfinite(pred.detach()).all():
        raise RuntimeError("nonfinite model prediction during paired patching/scrubbing")
    return pred.detach()


def primary_operator_names(masks: Mapping[str, torch.Tensor], raw: str) -> list[str]:
    available = sorted(name for name in masks if moa.is_primary_operator(name))
    if raw == "all":
        return available
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"unknown operator masks {unknown}; available={available}")
    return requested


def sample_stratified_from_allowed(
    mask: torch.Tensor,
    allowed: torch.Tensor,
    strata: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    device = mask.device
    sampled = torch.zeros_like(mask)
    positive = torch.nonzero(mask > 0, as_tuple=False).detach().cpu()
    if positive.numel() == 0:
        return sampled
    allowed_positions = torch.nonzero(allowed, as_tuple=False).detach().cpu()
    strata_cpu = strata.detach().cpu()
    candidates: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row, col in allowed_positions.tolist():
        candidates[int(strata_cpu[row, col])].append((row, col))
    positives_by_stratum: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row, col in positive.tolist():
        positives_by_stratum[int(strata_cpu[row, col])].append((row, col))
    for code, positives in positives_by_stratum.items():
        pool = candidates.get(code, [])
        if len(pool) < len(positives):
            continue
        order = torch.randperm(len(pool), generator=generator)
        chosen = [pool[int(i)] for i in order[: len(positives)]]
        weights = mask[
            torch.tensor([row for row, _col in positives], device=device),
            torch.tensor([col for _row, col in positives], device=device),
        ].detach().cpu()
        weight_order = torch.randperm(weights.numel(), generator=generator)
        weights = weights[weight_order].to(device=device, dtype=mask.dtype)
        rows = torch.tensor([row for row, _col in chosen], device=device)
        cols = torch.tensor([col for _row, col in chosen], device=device)
        sampled[rows, cols] = weights
    return sampled


def outside_control_mask(operator: torch.Tensor, batch: Any, seed: int) -> torch.Tensor:
    valid = moa.valid_pair_mask(batch)
    strata = moa.random_control_strata(batch, valid)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    allowed = valid[0] & ~(operator[0] > 0)
    sampled = sample_stratified_from_allowed(operator[0], allowed, strata[0], generator)
    return sampled.unsqueeze(0)


def wrong_operator_control_mask(
    operator: torch.Tensor,
    other: torch.Tensor,
    batch: Any,
    seed: int,
) -> torch.Tensor:
    valid = moa.valid_pair_mask(batch)
    strata = moa.random_control_strata(batch, valid)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    allowed = valid[0] & (other[0] > 0)
    sampled = sample_stratified_from_allowed(operator[0], allowed, strata[0], generator)
    return sampled.unsqueeze(0)


def build_controls(
    args: argparse.Namespace,
    batch: Any,
    masks: Mapping[str, torch.Tensor],
    operator: str,
    *,
    seed: int,
    wrong_source_available: bool,
) -> list[ControlMask]:
    operator_mask = masks[operator].float()
    operator_count = int((operator_mask > 0).sum().detach().cpu())
    controls = [
        ControlMask(
            control_type="operator",
            control_id="operator",
            control_operator=operator,
            source_type="clean",
            dense_mask=operator_mask,
        )
    ]
    valid = moa.valid_pair_mask(batch)
    for control_idx in range(max(0, int(args.matched_random_controls))):
        sampled = moa.rate_matched_random_masks(
            {operator: operator_mask},
            batch,
            valid,
            seed + 1009 * (control_idx + 1),
        )[f"random__{operator}"]
        controls.append(
            ControlMask(
                control_type="matched_random",
                control_id=f"random_{control_idx}",
                control_operator=operator,
                source_type="clean",
                dense_mask=sampled,
            )
        )
    if args.include_outside_operator_control:
        sampled = outside_control_mask(operator_mask, batch, seed + 17)
        if int((sampled > 0).sum().detach().cpu()) == operator_count:
            controls.append(
                ControlMask(
                    control_type="outside_operator",
                    control_id="outside_operator",
                    control_operator=operator,
                    source_type="clean",
                    dense_mask=sampled,
                )
            )
    if args.include_wrong_operator_control:
        others = [name for name in masks if moa.is_primary_operator(name) and name != operator and bool((masks[name] > 0).any())]
        if others:
            other = sorted(others)[0]
            sampled = wrong_operator_control_mask(operator_mask, masks[other].float(), batch, seed + 31)
            if int((sampled > 0).sum().detach().cpu()) == operator_count:
                controls.append(
                    ControlMask(
                        control_type="wrong_operator",
                        control_id=f"wrong_operator__{other}",
                        control_operator=other,
                        source_type="clean",
                        dense_mask=sampled,
                    )
                )
    if args.include_wrong_source_control and wrong_source_available:
        controls.append(
            ControlMask(
                control_type="wrong_source",
                control_id="wrong_clean_source",
                control_operator=operator,
                source_type="wrong_clean",
                dense_mask=operator_mask,
            )
        )
    return controls


def layer_head_items(records: Mapping[int, LayerActivation], head_mode: str) -> list[tuple[int, int]]:
    items: list[tuple[int, int]] = []
    for layer in sorted(records):
        if head_mode == "layer":
            items.append((layer, -1))
        elif head_mode == "per_head":
            for head in range(records[layer].heads):
                items.append((layer, head))
        else:
            raise ValueError(f"unknown head mode {head_mode!r}")
    return items


def estimated_controls_per_operator(args: argparse.Namespace) -> int:
    count = 1 + max(0, int(args.matched_random_controls))
    count += int(bool(args.include_outside_operator_control))
    count += int(bool(args.include_wrong_operator_control))
    count += int(bool(args.include_wrong_source_control))
    return count


def estimate_intervention_forwards(
    *,
    pairs: int,
    layers: int,
    heads: int,
    head_mode: str,
    targets: int,
    operators: int,
    controls_per_operator: int,
    experiments: int,
) -> int:
    head_factor = int(heads) if head_mode == "per_head" else 1
    return (
        int(pairs)
        * int(layers)
        * head_factor
        * int(targets)
        * int(operators)
        * int(controls_per_operator)
        * int(experiments)
    )


def append_patch_rows(
    args: argparse.Namespace,
    loaded: moa.LoadedExperiment,
    pair: PairSpec,
    clean_batch: Any,
    corrupt_batch: Any,
    clean_scores: BatchScores,
    corrupt_scores: BatchScores,
    clean_records: Mapping[int, LayerActivation],
    wrong_records: Optional[Mapping[int, LayerActivation]],
    masks: Mapping[str, torch.Tensor],
    patch_rows_out: list[dict[str, Any]],
) -> None:
    operators = primary_operator_names(masks, args.operator_masks)
    for operator in operators:
        if not bool((masks[operator] > 0).any()):
            continue
        controls = build_controls(
            args,
            clean_batch,
            masks,
            operator,
            seed=args.random_seed + pair.pair_id * 100_003,
            wrong_source_available=wrong_records is not None,
        )
        for layer, head in layer_head_items(clean_records, args.head_mode):
            for target in parse_csv(args.patch_targets, all_values=PATCH_TARGETS):
                for control in controls:
                    source_records = wrong_records if control.source_type == "wrong_clean" else clean_records
                    if source_records is None:
                        continue
                    source = source_records[layer]
                    coverage = assert_nonzero_sparse_coverage(source, control.dense_mask, f"{operator}/{control.control_type}")
                    context = GRITInterventionContext(
                        loaded.model,
                        official_batch=corrupt_batch,
                        layer=layer,
                        head=head,
                        target=target,
                        dense_mask=control.dense_mask,
                        mode="patch",
                        source=source,
                    )
                    pred = predict_with_context(args, loaded, corrupt_batch, context)
                    scored = score_prediction(loaded.runner, pred, corrupt_batch, loaded.target_stats)
                    restoration = safe_restoration(
                        clean_scores.scores[0],
                        corrupt_scores.scores[0],
                        scored.scores[0],
                        args.min_restoration_denominator,
                    )
                    patch_rows_out.append(
                        {
                            "pair_id": pair.pair_id,
                            "clean_graph_index": pair.clean_graph_index,
                            "corrupt_graph_index": pair.corrupt_graph_index,
                            "num_nodes": pair.num_nodes,
                            "layer": layer,
                            "head": head,
                            "operator": operator,
                            "patch_target": target,
                            "control_type": control.control_type,
                            "control_id": control.control_id,
                            "control_operator": control.control_operator,
                            "source_type": control.source_type,
                            "coverage_pairs": coverage,
                            "coverage_fraction": coverage / max(1, int(moa.valid_pair_mask(clean_batch).sum().detach().cpu())),
                            "clean_score": clean_scores.scores[0],
                            "corrupt_score": corrupt_scores.scores[0],
                            "patched_score": scored.scores[0],
                            "restoration": restoration,
                            "clean_primary": clean_scores.primary[0],
                            "corrupt_primary": corrupt_scores.primary[0],
                            "patched_primary": scored.primary[0],
                        }
                    )


def append_scrub_rows(
    args: argparse.Namespace,
    loaded: moa.LoadedExperiment,
    pair: PairSpec,
    clean_batch: Any,
    corrupt_scores: BatchScores,
    clean_scores: BatchScores,
    clean_records: Mapping[int, LayerActivation],
    masks: Mapping[str, torch.Tensor],
    scrub_rows_out: list[dict[str, Any]],
) -> None:
    operators = primary_operator_names(masks, args.operator_masks)
    for operator in operators:
        if not bool((masks[operator] > 0).any()):
            continue
        controls = build_controls(
            args,
            clean_batch,
            masks,
            operator,
            seed=args.random_seed + pair.pair_id * 100_003 + 53,
            wrong_source_available=False,
        )
        controls = [control for control in controls if control.source_type == "clean"]
        for layer, head in layer_head_items(clean_records, args.head_mode):
            record = clean_records[layer]
            for target in parse_csv(args.patch_targets, all_values=PATCH_TARGETS):
                for control in controls:
                    coverage = assert_nonzero_sparse_coverage(record, control.dense_mask, f"{operator}/{control.control_type}")
                    context = GRITInterventionContext(
                        loaded.model,
                        official_batch=clean_batch,
                        layer=layer,
                        head=head,
                        target=target,
                        dense_mask=control.dense_mask,
                        mode="scrub",
                        source=None,
                    )
                    pred = predict_with_context(args, loaded, clean_batch, context)
                    scored = score_prediction(loaded.runner, pred, clean_batch, loaded.target_stats)
                    if clean_batch.task_type == "graph_regression":
                        primary_drop = scored.primary[0] - clean_scores.primary[0]
                    else:
                        primary_drop = clean_scores.primary[0] - scored.primary[0]
                    toward_corrupt = safe_restoration(
                        corrupt_scores.scores[0],
                        clean_scores.scores[0],
                        scored.scores[0],
                        args.min_restoration_denominator,
                    )
                    scrub_rows_out.append(
                        {
                            "pair_id": pair.pair_id,
                            "clean_graph_index": pair.clean_graph_index,
                            "corrupt_graph_index": pair.corrupt_graph_index,
                            "num_nodes": pair.num_nodes,
                            "layer": layer,
                            "head": head,
                            "operator": operator,
                            "scrub_target": target,
                            "control_type": control.control_type,
                            "control_id": control.control_id,
                            "control_operator": control.control_operator,
                            "coverage_pairs": coverage,
                            "coverage_fraction": coverage / max(1, int(moa.valid_pair_mask(clean_batch).sum().detach().cpu())),
                            "clean_score": clean_scores.scores[0],
                            "corrupt_score": corrupt_scores.scores[0],
                            "scrubbed_score": scored.scores[0],
                            "toward_corrupt_fraction": toward_corrupt,
                            "clean_primary": clean_scores.primary[0],
                            "scrubbed_primary": scored.primary[0],
                            "drop": primary_drop,
                        }
                    )


def summarise_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    keys: Sequence[str],
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = finite_float(row.get(metric))
        if math.isfinite(value):
            grouped[tuple(row.get(key, "") for key in keys)].append(value)
    out: list[dict[str, Any]] = []
    for group_key, values in sorted(grouped.items(), key=lambda item: item[0]):
        mean = sum(values) / len(values)
        lo, hi = moa.bootstrap_ci(values, seed + len(out), samples=1000)
        row = {key: value for key, value in zip(keys, group_key)}
        row.update(
            {
                "metric": metric,
                "samples": len(values),
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "positive_fraction": sum(1 for value in values if value > 0) / len(values),
                "support_status": "strong_support"
                if mean > 0 and lo > 0
                else ("directional_support" if mean > 0 else "not_supported"),
            }
        )
        out.append(row)
    return out


def add_control_lift(summary: list[dict[str, Any]], *, target_field: str) -> list[dict[str, Any]]:
    controls: dict[tuple[Any, ...], float] = {}
    for row in summary:
        if row.get("control_type") != "matched_random":
            continue
        key = (
            row.get("operator"),
            row.get(target_field),
            row.get("layer"),
            row.get("head"),
        )
        controls[key] = float(row["mean"])
    for row in summary:
        key = (
            row.get("operator"),
            row.get(target_field),
            row.get("layer"),
            row.get("head"),
        )
        control = controls.get(key)
        row["matched_random_mean"] = control if control is not None else float("nan")
        row["control_normalised_lift"] = (
            float(row["mean"]) - control if control is not None else float("nan")
        )
    return summary


def mean_or_nan(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def support_status(values: Sequence[float], seed: int) -> tuple[float, float, float, str]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan"), float("nan"), float("nan"), "not_run"
    mean = sum(finite) / len(finite)
    lo, hi = moa.bootstrap_ci(finite, seed, samples=1000)
    status = (
        "strong_support"
        if mean > 0 and lo > 0
        else ("directional_support" if mean > 0 else "not_supported")
    )
    return mean, lo, hi, status


def pair_control_means(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    target_field: str,
    operators: Sequence[str],
    targets: Sequence[str],
    control_type: str,
) -> dict[int, float]:
    operator_set = set(operators)
    target_set = set(targets)
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("control_type") != control_type:
            continue
        if str(row.get("operator")) not in operator_set:
            continue
        if str(row.get(target_field)) not in target_set:
            continue
        value = finite_float(row.get(metric))
        if math.isfinite(value):
            grouped[int(row["pair_id"])].append(value)
    return {pair_id: mean_or_nan(values) for pair_id, values in grouped.items()}


def pair_lift_series(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    target_field: str,
    operators: Sequence[str],
    targets: Sequence[str],
) -> dict[int, float]:
    real = pair_control_means(
        rows,
        metric=metric,
        target_field=target_field,
        operators=operators,
        targets=targets,
        control_type="operator",
    )
    random_control = pair_control_means(
        rows,
        metric=metric,
        target_field=target_field,
        operators=operators,
        targets=targets,
        control_type="matched_random",
    )
    out: dict[int, float] = {}
    for pair_id in sorted(set(real) & set(random_control)):
        out[pair_id] = real[pair_id] - random_control[pair_id]
    return out


def hypothesis_operator_groups(task: str) -> dict[str, tuple[str, ...]]:
    core = tuple(task_core_operators(task).split(","))
    if "flow" in task:
        return {
            "core": core,
            "bottleneck": ("saturated_edge", "min_cut_crossing_edge"),
            "path": ("shortest_st_path_edge",),
        }
    if "bipartite" in task or "matching" in task:
        return {
            "core": core,
            "solution": ("optimum_matching_edge",),
            "search": ("alternating_forest_edge", "unmatched_feasible_edge"),
        }
    return {"core": core}


def requested_hypothesis_operators(args: argparse.Namespace) -> tuple[str, ...]:
    if args.operator_masks == "all":
        return tuple(task_core_operators(args.task).split(","))
    return tuple(item.strip() for item in str(args.operator_masks).split(",") if item.strip())


def keep_requested(values: Sequence[str], requested: Sequence[str]) -> tuple[str, ...]:
    requested_set = set(requested)
    return tuple(value for value in values if value in requested_set)


def make_hypothesis_outputs(
    args: argparse.Namespace,
    patch_rows: Sequence[Mapping[str, Any]],
    scrub_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = hypothesis_operator_groups(args.task)
    requested_operators = requested_hypothesis_operators(args)
    core = keep_requested(groups["core"], requested_operators)
    if not core:
        core = requested_operators
    requested_targets = set(parse_csv(args.patch_targets, all_values=PATCH_TARGETS))
    content_targets = tuple(target for target in ("pair_value", "pair_state") if target in requested_targets)
    routing_targets = tuple(target for target in ("routing_logits",) if target in requested_targets)
    pair_rows_out: list[dict[str, Any]] = []

    def add_lift_hypothesis(
        *,
        hypothesis_id: str,
        claim: str,
        rows: Sequence[Mapping[str, Any]],
        metric: str,
        target_field: str,
        operators: Sequence[str],
        targets: Sequence[str],
        seed_offset: int,
    ) -> dict[str, Any]:
        series = pair_lift_series(
            rows,
            metric=metric,
            target_field=target_field,
            operators=operators,
            targets=targets,
        )
        for pair_id, effect in sorted(series.items()):
            pair_rows_out.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "pair_id": pair_id,
                    "pair_effect": effect,
                    "metric": metric,
                    "comparison": "operator_minus_matched_random",
                }
            )
        mean, lo, hi, status = support_status(list(series.values()), args.random_seed + seed_offset)
        return {
            "hypothesis_id": hypothesis_id,
            "claim": claim,
            "experiment": "patching" if target_field == "patch_target" else "scrubbing",
            "metric": metric,
            "comparison": "operator minus matched random",
            "operators": ",".join(operators),
            "targets": ",".join(targets),
            "aggregation": "pre-specified pair mean over requested layers/operators/targets",
            "effect": mean,
            "ci_low": lo,
            "ci_high": hi,
            "pairs": len(series),
            "status": status,
            "confirmatory": True,
        }

    def add_difference_hypothesis(
        *,
        hypothesis_id: str,
        claim: str,
        rows: Sequence[Mapping[str, Any]],
        metric: str,
        target_field: str,
        positive_operators: Sequence[str],
        positive_targets: Sequence[str],
        negative_operators: Sequence[str],
        negative_targets: Sequence[str],
        seed_offset: int,
    ) -> dict[str, Any]:
        positive = pair_lift_series(
            rows,
            metric=metric,
            target_field=target_field,
            operators=positive_operators,
            targets=positive_targets,
        )
        negative = pair_lift_series(
            rows,
            metric=metric,
            target_field=target_field,
            operators=negative_operators,
            targets=negative_targets,
        )
        common = sorted(set(positive) & set(negative))
        values = {pair_id: positive[pair_id] - negative[pair_id] for pair_id in common}
        for pair_id, effect in values.items():
            pair_rows_out.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "pair_id": pair_id,
                    "pair_effect": effect,
                    "metric": metric,
                    "comparison": "difference_of_control_normalised_effects",
                }
            )
        mean, lo, hi, status = support_status(list(values.values()), args.random_seed + seed_offset)
        return {
            "hypothesis_id": hypothesis_id,
            "claim": claim,
            "experiment": "patching" if target_field == "patch_target" else "scrubbing",
            "metric": metric,
            "comparison": "positive effect minus negative effect after matched-random subtraction",
            "operators": ",".join(positive_operators),
            "targets": ",".join(positive_targets),
            "negative_operators": ",".join(negative_operators),
            "negative_targets": ",".join(negative_targets),
            "aggregation": "pre-specified pair mean over requested layers/operators/targets",
            "effect": mean,
            "ci_low": lo,
            "ci_high": hi,
            "pairs": len(values),
            "status": status,
            "confirmatory": True,
        }

    hypotheses: list[dict[str, Any]] = []
    if content_targets:
        hypotheses.append(
            add_lift_hypothesis(
                hypothesis_id="H1_patch_solver_content_rescue",
                claim="Clean solver-pair content activations causally rescue corrupt graph predictions.",
                rows=patch_rows,
                metric="restoration",
                target_field="patch_target",
                operators=core,
                targets=content_targets,
                seed_offset=101,
            )
        )
        hypotheses.append(
            add_lift_hypothesis(
                hypothesis_id="H2_scrub_solver_content_necessity",
                claim="Matched scrubbing of solver-pair content damages clean predictions.",
                rows=scrub_rows,
                metric="drop",
                target_field="scrub_target",
                operators=core,
                targets=content_targets,
                seed_offset=211,
            )
        )
    if content_targets and routing_targets:
        hypotheses.append(
            add_difference_hypothesis(
                hypothesis_id="H3_patch_content_exceeds_routing",
                claim="Relation-conditioned content rescue exceeds scalar routing rescue.",
                rows=patch_rows,
                metric="restoration",
                target_field="patch_target",
                positive_operators=core,
                positive_targets=content_targets,
                negative_operators=core,
                negative_targets=routing_targets,
                seed_offset=307,
            )
        )
        hypotheses.append(
            add_difference_hypothesis(
                hypothesis_id="H4_scrub_content_exceeds_routing",
                claim="Relation-conditioned content scrubbing damage exceeds scalar routing scrubbing damage.",
                rows=scrub_rows,
                metric="drop",
                target_field="scrub_target",
                positive_operators=core,
                positive_targets=content_targets,
                negative_operators=core,
                negative_targets=routing_targets,
                seed_offset=401,
            )
        )
    if "flow" in args.task and content_targets:
        bottleneck = keep_requested(groups["bottleneck"], requested_operators)
        path = keep_requested(groups["path"], requested_operators)
    else:
        bottleneck = ()
        path = ()
    if "flow" in args.task and content_targets and bottleneck and path:
        hypotheses.append(
            add_difference_hypothesis(
                hypothesis_id="H5_patch_bottleneck_exceeds_path",
                claim="Max-flow bottleneck/cut pairs rescue more than shortest-path pairs.",
                rows=patch_rows,
                metric="restoration",
                target_field="patch_target",
                positive_operators=bottleneck,
                positive_targets=content_targets,
                negative_operators=path,
                negative_targets=content_targets,
                seed_offset=509,
            )
        )
    return hypotheses, pair_rows_out


def write_hypothesis_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Confirmatory Mechanistic Hypothesis Summary",
        "",
        "Primary effects are pre-specified pair-level means over requested layers, operators, and targets. "
        "Matched-random controls are evaluated on the same graph pairs. Layer-localized maxima are not used "
        "for support calls.",
        "",
        "| Hypothesis | Direct Test | Effect | 95% CI | Pairs | Status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        effect = finite_float(row.get("effect"))
        lo = finite_float(row.get("ci_low"))
        hi = finite_float(row.get("ci_high"))
        lines.append(
            f"| {row.get('hypothesis_id')} | {row.get('claim')} | "
            f"{effect:.4g} | [{lo:.4g}, {hi:.4g}] | {int(row.get('pairs', 0))} | "
            f"{row.get('status')} |"
        )
    lines.extend(
        [
            "",
            "Status rule: strong support requires a positive mean effect and positive bootstrap lower bound. "
            "Directional support means the mean has the predicted sign but the lower bound does not.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_claim_summary(out_dir: Path, patch_summary: Sequence[Mapping[str, Any]], scrub_summary: Sequence[Mapping[str, Any]]) -> None:
    best_patch = max(
        (row for row in patch_summary if row.get("control_type") == "operator"),
        key=lambda row: finite_float(row.get("control_normalised_lift")),
        default=None,
    )
    best_scrub = max(
        (row for row in scrub_summary if row.get("control_type") == "operator"),
        key=lambda row: finite_float(row.get("control_normalised_lift")),
        default=None,
    )
    lines = [
        "# Paired Patching And Pair-Set Scrubbing Summary",
        "",
        "| Claim | Evidence | Effect | Status |",
        "|---|---|---:|---|",
    ]
    if best_patch is not None:
        lines.append(
            "| Operator-specific clean activations causally rescue corrupt graphs. "
            f"| best restoration lift: {best_patch.get('operator')} / {best_patch.get('patch_target')} "
            f"L{best_patch.get('layer')} H{best_patch.get('head')} "
            f"| {finite_float(best_patch.get('control_normalised_lift')):.4g} "
            f"| {best_patch.get('support_status')} |"
        )
    if best_scrub is not None:
        lines.append(
            "| Operator-specific pair information is necessary in clean graphs. "
            f"| best scrub drop lift: {best_scrub.get('operator')} / {best_scrub.get('scrub_target')} "
            f"L{best_scrub.get('layer')} H{best_scrub.get('head')} "
            f"| {finite_float(best_scrub.get('control_normalised_lift')):.4g} "
            f"| {best_scrub.get('support_status')} |"
        )
    lines.extend(
        [
            "",
            "Interpretation rule: strong support requires a positive paired mean and a positive bootstrap lower bound. "
            "The lift column subtracts matched-random controls evaluated on the same graph pairs.",
        ]
    )
    (out_dir / "claim_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def matrix_rows_from_raw(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    target_field: str,
) -> list[dict[str, Any]]:
    operators = sorted({str(row.get("operator")) for row in rows if row.get("control_type") == "operator"})
    targets = sorted({str(row.get(target_field)) for row in rows if row.get("control_type") == "operator"})
    out: list[dict[str, Any]] = []
    for operator in operators:
        for target in targets:
            series = pair_lift_series(
                rows,
                metric=metric,
                target_field=target_field,
                operators=(operator,),
                targets=(target,),
            )
            if not series:
                continue
            mean, lo, hi, status = support_status(list(series.values()), seed=len(out) + 17)
            out.append(
                {
                    "operator": operator,
                    target_field: target,
                    "effect": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "pairs": len(series),
                    "status": status,
                }
            )
    return out


def plot_hypothesis_dashboard(
    out_dir: Path,
    hypothesis_rows: Sequence[Mapping[str, Any]],
    patch_matrix_rows: Sequence[Mapping[str, Any]],
    scrub_matrix_rows: Sequence[Mapping[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        print(f"[plot] skipped hypothesis dashboard because pandas/matplotlib import failed: {exc}", flush=True)
        return

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]})
    hyp = pd.DataFrame(list(hypothesis_rows))
    ax = axes[0]
    if hyp.empty:
        ax.text(0.5, 0.5, "No confirmatory hypotheses available", ha="center", va="center")
        ax.axis("off")
    else:
        hyp = hyp.copy()
        hyp["short"] = hyp["hypothesis_id"].str.replace("_", "\n", regex=False)
        y = range(len(hyp))
        effects = hyp["effect"].astype(float).fillna(0.0)
        lo = hyp["ci_low"].astype(float).fillna(effects)
        hi = hyp["ci_high"].astype(float).fillna(effects)
        xerr = [effects - lo, hi - effects]
        colors = hyp["status"].map(
            {
                "strong_support": "#2ca02c",
                "directional_support": "#ffbf00",
                "not_supported": "#d62728",
                "not_run": "#8c8c8c",
            }
        ).fillna("#8c8c8c")
        ax.barh(list(y), effects, color=colors)
        ax.errorbar(effects, list(y), xerr=xerr, fmt="none", color="black", linewidth=1.0, capsize=3)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_yticks(list(y))
        ax.set_yticklabels(hyp["short"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("pre-specified paired effect")
        ax.set_title("Confirmatory hypothesis tests")

    def heatmap(ax, matrix_rows: Sequence[Mapping[str, Any]], target_field: str, title: str) -> None:
        df = pd.DataFrame(list(matrix_rows))
        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            return
        pivot = df.pivot(index="operator", columns=target_field, values="effect").fillna(0.0)
        values = pivot.to_numpy(dtype=float)
        vmax = max(1.0e-9, abs(values).max())
        im = ax.imshow(values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(title)
        for row_idx in range(pivot.shape[0]):
            for col_idx in range(pivot.shape[1]):
                value = values[row_idx, col_idx]
                color = "white" if abs(value) > 0.5 * vmax else "black"
                ax.text(col_idx, row_idx, f"{value:.2g}", ha="center", va="center", fontsize=7, color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    heatmap(axes[1], patch_matrix_rows, "patch_target", "Patching: operator minus matched random")
    heatmap(axes[2], scrub_matrix_rows, "scrub_target", "Scrubbing: operator minus matched random")
    fig.suptitle("Paired graph patching and pair-set scrubbing: pre-specified tests", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "main_pair_patch_scrub_dashboard.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_outputs(out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        print(f"[plot] skipped figures because pandas/matplotlib import failed: {exc}", flush=True)
        return

    patch_path = out_dir / "patching_summary.csv"
    if patch_path.exists() and patch_path.stat().st_size > 0:
        patch = pd.read_csv(patch_path)
        primary = patch[patch["control_type"].eq("operator")].copy()
        if not primary.empty:
            primary["label"] = (
                primary["operator"].astype(str)
                + "\n"
                + primary["patch_target"].astype(str)
                + " L"
                + primary["layer"].astype(str)
                + " H"
                + primary["head"].astype(str)
            )
            primary = primary.sort_values("control_normalised_lift", ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(primary)), 4.5))
            ax.bar(primary["label"], primary["control_normalised_lift"])
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_ylabel("restoration minus matched random")
            ax.set_title("Paired activation patching")
            ax.tick_params(axis="x", rotation=60)
            fig.tight_layout()
            fig.savefig(out_dir / "patching_restoration_lift.png", dpi=200)
            plt.close(fig)

    scrub_path = out_dir / "scrubbing_summary.csv"
    if scrub_path.exists() and scrub_path.stat().st_size > 0:
        scrub = pd.read_csv(scrub_path)
        primary = scrub[scrub["control_type"].eq("operator")].copy()
        if not primary.empty:
            primary["label"] = (
                primary["operator"].astype(str)
                + "\n"
                + primary["scrub_target"].astype(str)
                + " L"
                + primary["layer"].astype(str)
                + " H"
                + primary["head"].astype(str)
            )
            primary = primary.sort_values("control_normalised_lift", ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(primary)), 4.5))
            ax.bar(primary["label"], primary["control_normalised_lift"])
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_ylabel("drop minus matched random")
            ax.set_title("Pair-set scrubbing")
            ax.tick_params(axis="x", rotation=60)
            fig.tight_layout()
            fig.savefig(out_dir / "scrubbing_drop_lift.png", dpi=200)
            plt.close(fig)

    result_path = out_dir / "patching_results.csv"
    if result_path.exists() and result_path.stat().st_size > 0:
        patch = pd.read_csv(result_path)
        primary = patch[patch["control_type"].eq("operator")].copy()
        if not primary.empty:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.hist(primary["restoration"].dropna(), bins=40)
            ax.axvline(0.0, color="black", linewidth=0.8)
            ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("restoration fraction")
            ax.set_ylabel("interventions")
            ax.set_title("Clean-to-corrupt restoration distribution")
            fig.tight_layout()
            fig.savefig(out_dir / "patching_restoration_distribution.png", dpi=200)
            plt.close(fig)


def validate_args(args: argparse.Namespace) -> None:
    if args.model_backend != "official":
        raise RuntimeError("paired patching/scrubbing currently supports --model-backend official only")
    if args.model not in {"grit", "static_grit"}:
        raise RuntimeError("paired patching/scrubbing currently supports --model grit or --model static_grit only")
    if args.analysis_preset not in ANALYSIS_PRESETS:
        raise RuntimeError(f"unknown --analysis-preset {args.analysis_preset!r}")
    parse_csv(args.experiments, all_values=EXPERIMENTS)
    parse_csv(args.patch_targets, all_values=PATCH_TARGETS)
    if args.autocast_dtype != "none" and not args.allow_autocast_for_interventions:
        raise RuntimeError(
            "patching/scrubbing defaults to exact fp32/no-autocast reliability; "
            "pass --allow-autocast-for-interventions to use --autocast-dtype"
        )


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    start = time.time()
    loaded = moa.load_experiment(args)
    if loaded.dataset[0].task_type not in {"graph_regression", "edge_binary", "node_binary"}:
        raise RuntimeError(f"unsupported task type: {loaded.dataset[0].task_type}")

    graph_scores = score_selected_graphs(args, loaded)
    pairs = select_pairs(
        graph_scores,
        num_pairs=args.num_pairs,
        min_pred_delta=args.min_pred_delta,
        min_target_delta=args.min_target_delta,
        adaptive_quantile=args.adaptive_delta_quantile,
        seed=args.graph_seed,
    )
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    moa.write_csv(out_dir / "paired_graphs.csv", pair_rows(pairs))

    experiments = set(parse_csv(args.experiments, all_values=EXPERIMENTS))
    first_batch = single_graph_batch(loaded, pairs[0].clean_position)
    _first_pred, first_records = capture_prediction(args, loaded, first_batch)
    first_masks = moa.make_operator_masks(args.task, first_batch, random_controls=False, seed=args.random_seed)
    requested_operators = primary_operator_names(first_masks, args.operator_masks)
    nonempty_first_operators = [
        name for name in requested_operators if bool((first_masks[name] > 0).any())
    ]
    if not nonempty_first_operators:
        raise RuntimeError(f"no nonempty operator masks matched --operator-masks {args.operator_masks!r}")
    first_layer = next(iter(first_records.values()))
    planned_forwards = estimate_intervention_forwards(
        pairs=len(pairs),
        layers=len(first_records),
        heads=first_layer.heads,
        head_mode=args.head_mode,
        targets=len(parse_csv(args.patch_targets, all_values=PATCH_TARGETS)),
        operators=len(requested_operators),
        controls_per_operator=estimated_controls_per_operator(args),
        experiments=len(experiments),
    )
    print(
        "[plan] "
        f"preset={args.analysis_preset} pairs={len(pairs)} layers={len(first_records)} "
        f"heads={first_layer.heads} head_mode={args.head_mode} "
        f"targets={args.patch_targets} operators={','.join(requested_operators)} "
        f"nonempty_first_graph={','.join(nonempty_first_operators)} "
        f"controls_per_operator<={estimated_controls_per_operator(args)} "
        f"intervention_forwards<={planned_forwards}",
        flush=True,
    )
    if planned_forwards > args.max_interventions and not args.allow_large_sweep:
        raise RuntimeError(
            f"planned intervention forwards ({planned_forwards}) exceed --max-interventions "
            f"({args.max_interventions}). Use --analysis-preset pilot/core_fast, reduce "
            "--num-pairs/--matched-random-controls/targets/operators, or pass "
            "--allow-large-sweep deliberately."
        )
    del first_batch, _first_pred, first_records, first_masks

    patch_rows: list[dict[str, Any]] = []
    scrub_rows: list[dict[str, Any]] = []
    for pair_idx, pair in enumerate(pairs):
        clean_batch = single_graph_batch(loaded, pair.clean_position)
        corrupt_batch = single_graph_batch(loaded, pair.corrupt_position)
        clean_n = int(clean_batch.graph_num_nodes[0].detach().cpu())
        corrupt_n = int(corrupt_batch.graph_num_nodes[0].detach().cpu())
        if clean_n != corrupt_n:
            raise RuntimeError(f"pair {pair.pair_id} node mismatch: clean={clean_n} corrupt={corrupt_n}")

        clean_pred, clean_records = capture_prediction(args, loaded, clean_batch)
        corrupt_pred = predict_with_context(args, loaded, corrupt_batch)
        clean_scores = score_prediction(loaded.runner, clean_pred, clean_batch, loaded.target_stats)
        corrupt_scores = score_prediction(loaded.runner, corrupt_pred, corrupt_batch, loaded.target_stats)
        safe_restoration(
            clean_scores.scores[0],
            corrupt_scores.scores[0],
            (clean_scores.scores[0] + corrupt_scores.scores[0]) / 2.0,
            args.min_restoration_denominator,
        )
        masks = moa.make_operator_masks(args.task, clean_batch, random_controls=False, seed=args.random_seed + pair.pair_id)

        wrong_records: Optional[dict[int, LayerActivation]] = None
        if args.include_wrong_source_control and pair.wrong_source_position >= 0:
            wrong_batch = single_graph_batch(loaded, pair.wrong_source_position)
            wrong_n = int(wrong_batch.graph_num_nodes[0].detach().cpu())
            if wrong_n == clean_n:
                _wrong_pred, wrong_records = capture_prediction(args, loaded, wrong_batch)

        if "pair_patching" in experiments:
            append_patch_rows(
                args,
                loaded,
                pair,
                clean_batch,
                corrupt_batch,
                clean_scores,
                corrupt_scores,
                clean_records,
                wrong_records,
                masks,
                patch_rows,
            )
        if "pair_scrubbing" in experiments:
            append_scrub_rows(
                args,
                loaded,
                pair,
                clean_batch,
                corrupt_scores,
                clean_scores,
                clean_records,
                masks,
                scrub_rows,
            )
        print(
            f"[pair] {pair_idx + 1}/{len(pairs)} id={pair.pair_id} "
            f"clean={pair.clean_graph_index} corrupt={pair.corrupt_graph_index} "
            f"patch_rows={len(patch_rows)} scrub_rows={len(scrub_rows)}",
            flush=True,
        )
        if args.empty_cache_every_batch and loaded.device.type == "cuda":
            torch.cuda.empty_cache()

    moa.write_csv(out_dir / "patching_results.csv", patch_rows)
    moa.write_csv(out_dir / "scrubbing_results.csv", scrub_rows)
    patch_summary = add_control_lift(
        summarise_rows(
            patch_rows,
            metric="restoration",
            keys=("operator", "patch_target", "layer", "head", "control_type", "source_type"),
            seed=args.random_seed,
        ),
        target_field="patch_target",
    )
    scrub_summary = add_control_lift(
        summarise_rows(
            scrub_rows,
            metric="drop",
            keys=("operator", "scrub_target", "layer", "head", "control_type"),
            seed=args.random_seed + 17,
        ),
        target_field="scrub_target",
    )
    moa.write_csv(out_dir / "patching_summary.csv", patch_summary)
    moa.write_csv(out_dir / "scrubbing_summary.csv", scrub_summary)
    hypothesis_rows, hypothesis_pair_rows = make_hypothesis_outputs(args, patch_rows, scrub_rows)
    patch_matrix_rows = matrix_rows_from_raw(
        patch_rows,
        metric="restoration",
        target_field="patch_target",
    )
    scrub_matrix_rows = matrix_rows_from_raw(
        scrub_rows,
        metric="drop",
        target_field="scrub_target",
    )
    moa.write_csv(out_dir / "hypothesis_summary.csv", hypothesis_rows)
    moa.write_csv(out_dir / "hypothesis_effects_by_pair.csv", hypothesis_pair_rows)
    moa.write_csv(out_dir / "patching_operator_target_matrix.csv", patch_matrix_rows)
    moa.write_csv(out_dir / "scrubbing_operator_target_matrix.csv", scrub_matrix_rows)
    write_hypothesis_markdown(out_dir / "hypothesis_summary.md", hypothesis_rows)
    write_claim_summary(out_dir, patch_summary, scrub_summary)
    metadata = {
        "task": args.task,
        "split": args.split,
        "model": args.model,
        "model_backend": args.model_backend,
        "checkpoint": str(args.checkpoint),
        "num_candidate_graphs": len(loaded.graphs),
        "num_pairs": len(pairs),
        "selected_graph_indices": loaded.selected_indices,
        "device": str(loaded.device),
        "cuda_name": torch.cuda.get_device_name(loaded.device) if loaded.device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "torch_runtime": dict(loaded.runtime_settings),
        "analysis_batch_cache_mode": loaded.batch_cache_mode,
        "analysis_batch_cache_estimated_gib": loaded.batch_cache_estimated_bytes / (1024**3),
        "args": jsonable(vars(args)),
        "git_sha": git_sha(Path.cwd()),
        "elapsed_seconds": time.time() - start,
        "interpretation": (
            "Patching support means clean activations restore corrupt scores more than matched controls; "
            "scrubbing support means matched replacement damages clean primary performance more than controls."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    plot_hypothesis_dashboard(out_dir, hypothesis_rows, patch_matrix_rows, scrub_matrix_rows)
    plot_outputs(out_dir)
    print(
        f"[done] wrote {len(patch_rows)} patch rows and {len(scrub_rows)} scrub rows to {out_dir}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = moa.build_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == "experiments":
            action.help = "Comma list: pair_patching,pair_scrubbing,all"
    parser.set_defaults(
        experiments="pair_patching,pair_scrubbing",
        output_dir=Path("outputs/mechanistic_pair_patching_scrubbing"),
        num_graphs=1024,
        batch_size=4,
        analysis_cache_batch_size=0,
        autocast_dtype="none",
    )
    parser.add_argument("--num-pairs", type=int, default=128)
    parser.add_argument(
        "--analysis-preset",
        default="custom",
        choices=ANALYSIS_PRESETS,
        help=(
            "custom preserves explicit flags; pilot is a very small patching-only "
            "proof-of-life run; core_fast tests the core patching+scrubbing hypotheses "
            "with task-aware operators, three patch targets, and matched-random controls; "
            "confirmatory_1h roughly triples core_fast evidence while staying under the "
            "default intervention cap; h2_validation runs only the scrub-necessity test "
            "on content fields and causal bottleneck operators."
        ),
    )
    parser.add_argument(
        "--pair-batch-size",
        type=int,
        default=1,
        help="Reserved for future grouped same-size pair execution; v1 processes pairs sequentially.",
    )
    parser.add_argument(
        "--patch-targets",
        default="all",
        help="Comma list: routing_logits,attention,pair_value,pair_state,head_output,all",
    )
    parser.add_argument(
        "--operator-masks",
        default="all",
        help="Comma list of solver/operator masks to patch/scrub, or all.",
    )
    parser.add_argument("--matched-random-controls", type=int, default=4)
    parser.add_argument("--min-pred-delta", type=float, default=None)
    parser.add_argument("--min-target-delta", type=float, default=None)
    parser.add_argument("--adaptive-delta-quantile", type=float, default=0.25)
    parser.add_argument("--min-restoration-denominator", type=float, default=1.0e-6)
    parser.add_argument(
        "--max-interventions",
        type=int,
        default=50_000,
        help="Fail before the expensive loop if the planned intervention forward count exceeds this cap.",
    )
    parser.add_argument(
        "--allow-large-sweep",
        action="store_true",
        help="Allow runs above --max-interventions. Use only for deliberate exhaustive sweeps.",
    )
    parser.add_argument("--head-mode", choices=("layer", "per_head"), default="layer")
    parser.add_argument("--include-outside-operator-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-wrong-operator-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-wrong-source-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-autocast-for-interventions",
        action="store_true",
        help="Allow non-none --autocast-dtype for speed. Default is disabled for reliability.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    apply_analysis_preset(args, sys.argv[1:])
    if args.pair_batch_size != 1:
        print("[warn] --pair-batch-size is currently reserved; processing pairs sequentially", flush=True)
    run(args)


if __name__ == "__main__":
    main()
