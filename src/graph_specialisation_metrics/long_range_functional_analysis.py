"""Functional long-range analysis for graph-transformer teacher-student models.

This module implements the experiment suite specified in
``long_range_gt_functional_analysis.md``.  The implementation is deliberately
functional: it perturbs graph inputs, caches final node states and outputs, and
then computes demand/usage/source-map/interaction/address-mode quantities from
those cached input-output effects.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch

from graph_specialisation_metrics.counterfactual_interchange_mediation import (
    TASKS,
    artifact_root,
    choose_device,
    clone_record_with,
    collate_records,
    configure_runtime,
    data_dir,
    default_model_config_path,
    graph_sum_from_node_output,
    graph_sum_teacher,
    load_config,
    load_model_from_checkpoint,
    load_records,
    read_csv_dicts,
    write_csv,
    write_json,
)


DEFAULT_BUCKET_TEXT = "1,2,3,4,5-6,7-9,10+"
EPS = 1.0e-12


@dataclass(frozen=True)
class DistanceBucket:
    name: str
    low: int
    high: int

    def contains(self, distance: int) -> bool:
        return int(self.low) <= int(distance) <= int(self.high)


def lr_root(cfg: Mapping[str, Any]) -> Path:
    return artifact_root(cfg) / "long_range_functional"


def lr_task_root(cfg: Mapping[str, Any], task: str, seed: int) -> Path:
    return lr_root(cfg) / task / f"seed_{int(seed)}"


def lr_graph_dir(cfg: Mapping[str, Any], task: str, seed: int) -> Path:
    return lr_task_root(cfg, task, seed) / "graphs"


def lr_metrics_dir(cfg: Mapping[str, Any], task: str, seed: int) -> Path:
    return lr_task_root(cfg, task, seed) / "metrics"


def lr_figures_dir(cfg: Mapping[str, Any], task: str, seed: int) -> Path:
    return lr_task_root(cfg, task, seed) / "figures"


def lr_manifest_path(cfg: Mapping[str, Any], task: str, seed: int) -> Path:
    return lr_task_root(cfg, task, seed) / "manifest.json"


def lr_combined_figures_dir(cfg: Mapping[str, Any], seed: int) -> Path:
    return lr_root(cfg) / f"seed_{int(seed)}" / "figures"


def parse_distance_buckets(text: str) -> list[DistanceBucket]:
    buckets: list[DistanceBucket] = []
    for raw in text.split(","):
        value = raw.strip()
        if not value:
            continue
        if value.endswith("+"):
            low = int(value[:-1])
            high = 10_000
            name = f"{low}+"
        elif "-" in value:
            low_s, high_s = value.split("-", 1)
            low = int(low_s)
            high = int(high_s)
            name = f"{low}-{high}"
        else:
            low = high = int(value)
            name = str(low)
        if low <= 0 or high < low:
            raise ValueError(f"invalid distance bucket {value!r}")
        buckets.append(DistanceBucket(name=name, low=low, high=high))
    if not buckets:
        raise ValueError("at least one distance bucket is required")
    return buckets


def bucket_for_distance(distance: int, buckets: Sequence[DistanceBucket]) -> str | None:
    for bucket in buckets:
        if bucket.contains(int(distance)):
            return bucket.name
    return None


def graph_neighbors(record: Mapping[str, Any]) -> list[list[int]]:
    adj = record["struct"]["adjacency"].long()
    n = int(record["n"])
    return [torch.nonzero(adj[i, :n] > 0, as_tuple=False).reshape(-1).tolist() for i in range(n)]


def betweenness_centrality_unweighted(record: Mapping[str, Any]) -> np.ndarray:
    neighbors = graph_neighbors(record)
    n = len(neighbors)
    cb = np.zeros(n, dtype=np.float64)
    for s in range(n):
        stack: list[int] = []
        pred = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=np.float64)
        sigma[s] = 1.0
        dist = -np.ones(n, dtype=np.int64)
        dist[s] = 0
        queue: deque[int] = deque([s])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in neighbors[v]:
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = np.zeros(n, dtype=np.float64)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]
    if n > 2:
        cb /= 2.0
    return cb


def focal_nodes_for_graph(record: Mapping[str, Any], *, count: int, rng: random.Random) -> list[int]:
    n = int(record["n"])
    centrality = betweenness_centrality_unweighted(record)
    ranked = list(np.argsort(-centrality))
    selected = [int(idx) for idx in ranked[: max(1, min(int(count), n))]]
    if len(selected) < int(count):
        remaining = [idx for idx in range(n) if idx not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: int(count) - len(selected)])
    return selected


def type_compatibility_class(record: Mapping[str, Any], node: int) -> str:
    degree = int(record["struct"]["degree"][int(node)].item())
    return f"degree={degree}"


def content_feature_distance(record: Mapping[str, Any], u: int, v: int) -> float:
    diff = record["x"][int(u)].float() - record["x"][int(v)].float()
    return float(torch.linalg.vector_norm(diff).item())


def content_swap_record(base: Mapping[str, Any], u: int, v: int, cfg: Mapping[str, Any], *, graph_id: str | None = None) -> dict[str, Any]:
    """Swap all symbolic node features while holding topology/structural tensors fixed."""

    u = int(u)
    v = int(v)
    payload = base["payload"].clone()
    payload[[u, v]] = payload[[v, u]]
    anchor_indicator = None
    anchor_priority = None
    if base.get("anchor_indicator") is not None:
        anchor_indicator = base["anchor_indicator"].clone()
        anchor_priority = base["anchor_priority"].clone()
        anchor_indicator[[u, v]] = anchor_indicator[[v, u]]
        anchor_priority[[u, v]] = anchor_priority[[v, u]]
    return clone_record_with(
        base,
        cfg,
        graph_id=graph_id or f"{base['graph_id']}__lr_content_{u}_{v}",
        payload=payload,
        anchor_indicator=anchor_indicator,
        anchor_priority=anchor_priority,
    )


def content_relayout_record(
    base: Mapping[str, Any],
    old_to_new: Mapping[int, int],
    cfg: Mapping[str, Any],
    *,
    graph_id: str,
) -> dict[str, Any]:
    payload = base["payload"].clone()
    new_payload = payload.clone()
    for old, new in old_to_new.items():
        new_payload[int(new)] = payload[int(old)]
    anchor_indicator = None
    anchor_priority = None
    if base.get("anchor_indicator") is not None:
        anchor_indicator = base["anchor_indicator"].clone()
        anchor_priority = base["anchor_priority"].clone()
        new_indicator = anchor_indicator.clone()
        new_priority = anchor_priority.clone()
        for old, new in old_to_new.items():
            new_indicator[int(new)] = anchor_indicator[int(old)]
            new_priority[int(new)] = anchor_priority[int(old)]
        anchor_indicator = new_indicator
        anchor_priority = new_priority
    return clone_record_with(
        base,
        cfg,
        graph_id=graph_id,
        payload=new_payload,
        anchor_indicator=anchor_indicator,
        anchor_priority=anchor_priority,
    )


def load_lr_model_config(base_cfg: Mapping[str, Any], task: str, config_path: Path | None, fast_dev_run: bool = False) -> dict[str, Any]:
    resolved = config_path or default_model_config_path(task, "grit")
    if resolved is not None:
        cfg = load_config(resolved, task=task, fast_dev_run=fast_dev_run)
    else:
        cfg = copy.deepcopy(dict(base_cfg))
        cfg["task"] = task
    cfg["artifacts"]["root"] = str(artifact_root(base_cfg))
    return cfg


@torch.no_grad()
def model_states_and_outputs(model: torch.nn.Module, records: Sequence[Mapping[str, Any]], *, batch_size: int, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    model.eval()
    states: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    for start in range(0, len(records), int(batch_size)):
        chunk = list(records[start : start + int(batch_size)])
        batch = collate_records(chunk).to(device)
        if not hasattr(model, "node_states") or not hasattr(model, "readout_from_states"):
            raise TypeError("long-range analysis requires models exposing node_states/readout_from_states")
        h = model.node_states(batch).detach()
        y = model.readout_from_states(batch, h).detach()
        for idx, record in enumerate(chunk):
            n = int(record["n"])
            states.append(h[idx, :n].cpu().clone())
            outputs.append(y[idx, :n].cpu().clone())
    return states, outputs


def norm_vec(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float()).item())


def rms(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(math.sqrt(float(np.nanmean(arr * arr))))


def participation_ratio(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return float("nan")
    return float((vals.sum() ** 2) / np.square(vals).sum().clip(min=EPS))


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x = x[mask]
    y = y[mask]
    w = w[mask]
    if x.size < 2:
        return float("nan")
    w = w / w.sum()
    mx = float((w * x).sum())
    my = float((w * y).sum())
    vx = float((w * np.square(x - mx)).sum())
    vy = float((w * np.square(y - my)).sum())
    if vx <= EPS or vy <= EPS:
        return float("nan")
    return float((w * (x - mx) * (y - my)).sum() / math.sqrt(vx * vy))


def sample_lr_swaps(
    record: Mapping[str, Any],
    *,
    focal_nodes: Sequence[int],
    buckets: Sequence[DistanceBucket],
    broad_random: int,
    targeted_per_bucket: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    n = int(record["n"])
    spd = record["struct"]["shortest_path_distance"]
    by_class: dict[str, list[int]] = defaultdict(list)
    for node in range(n):
        by_class[type_compatibility_class(record, node)].append(node)
    candidates = [(u, v) for nodes in by_class.values() for idx, u in enumerate(nodes) for v in nodes[idx + 1 :]]
    rng.shuffle(candidates)
    selected: dict[tuple[int, int], dict[str, Any]] = {}

    def add_pair(u: int, v: int, source: str, focal: int | None = None, bucket: str | None = None) -> None:
        key = tuple(sorted((int(u), int(v))))
        if key in selected:
            selected[key]["sample_source"] = selected[key]["sample_source"] + f"+{source}"
            return
        selected[key] = {
            "u": key[0],
            "v": key[1],
            "sample_source": source,
            "target_focal": "" if focal is None else int(focal),
            "target_bucket": "" if bucket is None else bucket,
        }

    for u, v in candidates[: int(broad_random)]:
        add_pair(u, v, "broad_random")

    for focal in focal_nodes:
        for bucket in buckets:
            pool = []
            for u, v in candidates:
                bu = bucket.contains(int(spd[int(focal), int(u)]))
                bv = bucket.contains(int(spd[int(focal), int(v)]))
                if bu and bv:
                    pool.append((u, v))
            rng.shuffle(pool)
            for u, v in pool[: int(targeted_per_bucket)]:
                add_pair(u, v, "targeted_bucket_equidistant", int(focal), bucket.name)

    rows = []
    for swap_idx, row in enumerate(selected.values()):
        u = int(row["u"])
        v = int(row["v"])
        rows.append(
            {
                **row,
                "swap_id": f"s{swap_idx:05d}_{u}_{v}",
                "d_uv": int(spd[u, v]),
                "feature_l2": content_feature_distance(record, u, v),
                "type_class_u": type_compatibility_class(record, u),
                "type_class_v": type_compatibility_class(record, v),
            }
        )
    return rows


def readout_graph_sum(output: torch.Tensor) -> torch.Tensor:
    return graph_sum_from_node_output(output)


def node_state_patched_graph_delta(
    model: torch.nn.Module,
    record: Mapping[str, Any],
    h_background: torch.Tensor,
    h_insert: torch.Tensor,
    y_background_sum: torch.Tensor,
    *,
    node: int,
    device: torch.device,
) -> torch.Tensor:
    batch = collate_records([record]).to(device)
    states = h_background.unsqueeze(0).to(device).clone()
    states[0, int(node)] = h_insert[int(node)].to(device)
    with torch.no_grad():
        pred = model.readout_from_states(batch, states).detach().cpu()[0, : int(record["n"])]
    return readout_graph_sum(pred) - y_background_sum


def per_node_linear_readout_deltas(y_clean: torch.Tensor, y_source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact P1/P2 vectors for the repository's node-head + sum-pool readout."""

    p1 = y_source.float() - y_clean.float()
    p2 = y_clean.float() - y_source.float()
    return p1, p2


def build_source_maps(
    n: int,
    swaps: Sequence[Mapping[str, Any]],
    values_by_swap: Mapping[str, np.ndarray],
) -> np.ndarray:
    buckets: list[list[list[float]]] = [[[] for _ in range(n)] for _ in range(n)]
    for swap in swaps:
        sid = str(swap["swap_id"])
        if sid not in values_by_swap:
            continue
        vals = np.asarray(values_by_swap[sid], dtype=np.float64)
        for source in (int(swap["u"]), int(swap["v"])):
            for receiver in range(n):
                buckets[receiver][source].append(float(vals[receiver]))
    out = np.full((n, n), np.nan, dtype=np.float64)
    for receiver in range(n):
        for source in range(n):
            if buckets[receiver][source]:
                out[receiver, source] = rms(buckets[receiver][source])
    return out


def far_mask_from_spd(spd: torch.Tensor, r: int) -> np.ndarray:
    return (spd.cpu().numpy().astype(np.int64) > int(r))


def compute_influence_stats(matrix: np.ndarray, spd: torch.Tensor, r: int) -> dict[str, float]:
    far = far_mask_from_spd(spd, r)
    mat = np.asarray(matrix, dtype=np.float64)
    full_mass = float(np.nansum(mat))
    far_values = np.where(far, mat, np.nan)
    far_mass = float(np.nansum(far_values))
    clean_far = np.nan_to_num(far_values, nan=0.0)
    try:
        singular = np.linalg.svd(clean_far, compute_uv=False)
    except np.linalg.LinAlgError:
        singular = np.asarray([], dtype=np.float64)
    col = np.nansum(far_values, axis=0)
    row = np.nansum(far_values, axis=1)
    valid_pair = np.isfinite(far_values) & np.isfinite(far_values.T) & far
    reciprocity = float(np.corrcoef(far_values[valid_pair], far_values.T[valid_pair])[0, 1]) if int(valid_pair.sum()) > 2 else float("nan")
    d = spd.cpu().numpy()
    near_diag = far & (d <= int(r) + 2)
    col_top = set(np.argsort(-np.nan_to_num(col, nan=0.0))[: max(1, len(col) // 5)])
    row_top = set(np.argsort(-np.nan_to_num(row, nan=0.0))[: max(1, len(row) // 5)])
    return {
        "effective_rank": participation_ratio(singular),
        "top_singular_fraction": float(singular[0] / singular.sum()) if singular.size and singular.sum() > EPS else float("nan"),
        "column_concentration": float(np.nanmax(col) / np.nansum(col)) if np.nansum(col) > EPS else float("nan"),
        "row_concentration": float(np.nanmax(row) / np.nansum(row)) if np.nansum(row) > EPS else float("nan"),
        "matching_concentration": float(np.nansum(np.where(near_diag, mat, 0.0)) / far_mass) if far_mass > EPS else float("nan"),
        "reciprocity": reciprocity,
        "far_mass_fraction": float(far_mass / full_mass) if full_mass > EPS else float("nan"),
        "effective_sources_per_receiver": float(np.nanmedian([participation_ratio(row_vals[np.isfinite(row_vals)]) for row_vals in far_values])),
        "hub_overlap_jaccard": float(len(col_top & row_top) / max(1, len(col_top | row_top))),
    }


def compute_graph_cache(
    cfg: Mapping[str, Any],
    task: str,
    record: Mapping[str, Any],
    model: torch.nn.Module,
    *,
    graph_index: int,
    seed: int,
    buckets: Sequence[DistanceBucket],
    r: int,
    focal_count: int,
    broad_random: int,
    targeted_per_bucket: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    rng = random.Random(int(seed) + 104729 * int(graph_index))
    focal_nodes = focal_nodes_for_graph(record, count=focal_count, rng=rng)
    swaps = sample_lr_swaps(
        record,
        focal_nodes=focal_nodes,
        buckets=buckets,
        broad_random=broad_random,
        targeted_per_bucket=targeted_per_bucket,
        rng=rng,
    )
    clean_states, clean_outputs = model_states_and_outputs(model, [record], batch_size=1, device=device)
    h_clean = clean_states[0]
    y_clean = clean_outputs[0]
    y_clean_sum = readout_graph_sum(y_clean)
    teacher_clean = record["teacher"]["Y"].float()
    teacher_clean_sum = graph_sum_teacher(record)

    source_records = [
        content_swap_record(record, int(swap["u"]), int(swap["v"]), cfg, graph_id=f"{record['graph_id']}__{swap['swap_id']}")
        for swap in swaps
    ]
    source_states, source_outputs = model_states_and_outputs(model, source_records, batch_size=batch_size, device=device)

    node_model_values: dict[str, np.ndarray] = {}
    node_oracle_values: dict[str, np.ndarray] = {}
    p1_values: dict[str, np.ndarray] = {}
    p2_values: dict[str, np.ndarray] = {}
    failure_values: dict[str, np.ndarray] = {}
    effect_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    n = int(record["n"])
    spd = record["struct"]["shortest_path_distance"]

    for idx, swap in enumerate(swaps):
        sid = str(swap["swap_id"])
        src_record = source_records[idx]
        h_s = source_states[idx]
        y_s = source_outputs[idx]
        y_s_sum = readout_graph_sum(y_s)
        teacher_s = src_record["teacher"]["Y"].float()
        teacher_s_sum = graph_sum_teacher(src_record)
        delta_model_nodes = y_s - y_clean
        delta_oracle_nodes = teacher_s - teacher_clean
        node_model = torch.linalg.vector_norm(delta_model_nodes.float(), dim=1).numpy()
        node_oracle = torch.linalg.vector_norm(delta_oracle_nodes.float(), dim=1).numpy()
        failure = torch.linalg.vector_norm((delta_model_nodes - delta_oracle_nodes).float(), dim=1).numpy()
        node_model_values[sid] = node_model
        node_oracle_values[sid] = node_oracle
        failure_values[sid] = failure

        p1_vec, p2_vec = per_node_linear_readout_deltas(y_clean, y_s)
        p1 = torch.linalg.vector_norm(p1_vec.float(), dim=1).numpy().astype(np.float64)
        p2 = torch.linalg.vector_norm(p2_vec.float(), dim=1).numpy().astype(np.float64)
        near_mask = np.zeros(n, dtype=bool)
        dist_to_swap = torch.minimum(spd[:, int(swap["u"])], spd[:, int(swap["v"])])
        near_mask[dist_to_swap.numpy() <= int(r)] = True
        p1_values[sid] = p1
        p2_values[sid] = p2
        sum_p1 = p1_vec.sum(dim=0)
        full_delta = y_s_sum - y_clean_sum
        ledger_residual = norm_vec(full_delta - sum_p1)
        ledger_rows.append(
            {
                "task": task,
                "graph_id": record["graph_id"],
                "swap_id": sid,
                "full_delta_norm": norm_vec(full_delta),
                "sum_p1_norm": norm_vec(sum_p1),
                "near_p1_sum": float(np.nansum(p1[near_mask])),
                "far_p1_sum": float(np.nansum(p1[~near_mask])),
                "interaction_residual_norm": ledger_residual,
                "ledger_relative_residual": ledger_residual / max(norm_vec(full_delta), EPS),
            }
        )

        feature_l2 = max(float(swap["feature_l2"]), EPS)
        graph_model_delta = norm_vec(y_s_sum - y_clean_sum)
        graph_oracle_delta = norm_vec(teacher_s_sum - teacher_clean_sum)
        graph_delta_res = norm_vec((y_s_sum - y_clean_sum) - (teacher_s_sum - teacher_clean_sum))
        for focal in focal_nodes:
            du = int(spd[int(focal), int(swap["u"])])
            dv = int(spd[int(focal), int(swap["v"])])
            bucket = bucket_for_distance(du, buckets) if bucket_for_distance(du, buckets) == bucket_for_distance(dv, buckets) else None
            if bucket is None:
                continue
            effect_rows.append(
                {
                    "task": task,
                    "graph_id": record["graph_id"],
                    "focal_node": int(focal),
                    "swap_id": sid,
                    "u": int(swap["u"]),
                    "v": int(swap["v"]),
                    "distance_bucket": bucket,
                    "d_u_focal": du,
                    "d_v_focal": dv,
                    "feature_l2": feature_l2,
                    "oracle_node_delta": float(node_oracle[int(focal)]),
                    "model_node_delta": float(node_model[int(focal)]),
                    "oracle_graph_delta": graph_oracle_delta,
                    "model_graph_delta": graph_model_delta,
                    "delta_residual_graph": graph_delta_res,
                    "oracle_node_delta_per_feature": float(node_oracle[int(focal)] / feature_l2),
                    "model_node_delta_per_feature": float(node_model[int(focal)] / feature_l2),
                    "oracle_graph_delta_per_feature": graph_oracle_delta / feature_l2,
                    "model_graph_delta_per_feature": graph_model_delta / feature_l2,
                }
            )

    source_map_direct = build_source_maps(n, swaps, node_model_values)
    source_map_oracle = build_source_maps(n, swaps, node_oracle_values)
    source_map_p1 = build_source_maps(n, swaps, p1_values)
    source_map_p2 = build_source_maps(n, swaps, p2_values)
    failure_map = build_source_maps(n, swaps, failure_values)
    influence_stats = compute_influence_stats(source_map_direct, spd, r)

    reached_rows = []
    for swap, h_s in zip(swaps, source_states):
        sid = str(swap["swap_id"])
        dist_to_swap = torch.minimum(spd[:, int(swap["u"])], spd[:, int(swap["v"])])
        h_delta = torch.linalg.vector_norm((h_s - h_clean).float(), dim=1).numpy()
        mattered = node_model_values[sid]
        for node in range(n):
            bucket = bucket_for_distance(int(dist_to_swap[node]), buckets)
            if bucket is None:
                continue
            reached_rows.append(
                {
                    "task": task,
                    "graph_id": record["graph_id"],
                    "receiver": node,
                    "swap_id": sid,
                    "distance_bucket": bucket,
                    "reached": int(float(h_delta[node]) > 1.0e-8),
                    "mattered": int(float(mattered[node]) > 1.0e-8),
                    "state_delta_norm": float(h_delta[node]),
                    "output_delta_norm": float(mattered[node]),
                }
            )

    return {
        "task": task,
        "graph_id": str(record["graph_id"]),
        "graph_index": int(graph_index),
        "n": n,
        "r": int(r),
        "focal_nodes": [int(v) for v in focal_nodes],
        "buckets": [bucket.name for bucket in buckets],
        "bucket_specs": [{"name": bucket.name, "low": int(bucket.low), "high": int(bucket.high)} for bucket in buckets],
        "swaps": swaps,
        "effect_rows": effect_rows,
        "ledger_rows": ledger_rows,
        "reached_rows": reached_rows,
        "influence_stats": {"task": task, "graph_id": str(record["graph_id"]), **influence_stats},
        "source_map_direct": source_map_direct,
        "source_map_oracle": source_map_oracle,
        "source_map_p1": source_map_p1,
        "source_map_p2": source_map_p2,
        "failure_map": failure_map,
        "spd": spd.numpy().astype(np.int64),
        "adjacency": record["struct"]["adjacency"].cpu().numpy().astype(np.int8),
        "centrality": betweenness_centrality_unweighted(record),
    }


def build_long_range_cache(
    cfg: Mapping[str, Any],
    task: str,
    *,
    seed: int = 9701,
    split: str = "test_id",
    num_graphs: int = 200,
    distance_buckets: str = DEFAULT_BUCKET_TEXT,
    receptive_radius: int = 2,
    focal_nodes: int = 3,
    broad_random_swaps: int = 160,
    targeted_swaps_per_bucket: int = 12,
    batch_size: int = 512,
    device_name: str = "auto",
    config_path: Path | None = None,
    checkpoint: Path | None = None,
    backend: str | None = "official",
    progress_every_graphs: int = 1,
    force: bool = False,
    fast_dev_run: bool = False,
) -> Path:
    device = choose_device(device_name)
    configure_runtime(cfg, device)
    model_cfg = load_lr_model_config(cfg, task, config_path, fast_dev_run=fast_dev_run)
    model, _ = load_model_from_checkpoint(model_cfg, task, checkpoint, device, backend=backend)
    records = load_records(data_dir(cfg, task) / f"{split}.pt")[: int(4 if fast_dev_run else num_graphs)]
    buckets = parse_distance_buckets(distance_buckets)
    print(
        f"[long-range-cache] start task={task} split={split} graphs={len(records)} "
        f"broad_random={int(broad_random_swaps)} targeted_per_bucket={int(targeted_swaps_per_bucket)} "
        f"batch_size={int(batch_size)} device={device}",
        flush=True,
    )
    graph_dir = lr_graph_dir(cfg, task, seed)
    graph_dir.mkdir(parents=True, exist_ok=True)
    effect_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    reached_rows: list[dict[str, Any]] = []
    influence_rows: list[dict[str, Any]] = []
    manifest_graphs = []
    for idx, record in enumerate(records):
        out_path = graph_dir / f"{idx:05d}_{record['graph_id']}.pt"
        if out_path.exists() and not force:
            if (idx + 1) % max(1, int(progress_every_graphs)) == 0 or idx == 0 or idx + 1 == len(records):
                print(f"[long-range-cache] using cached graph={idx + 1}/{len(records)} id={record['graph_id']}", flush=True)
            cache = torch.load(out_path, map_location="cpu", weights_only=False)
        else:
            print(f"[long-range-cache] task={task} graph={idx + 1}/{len(records)} id={record['graph_id']}", flush=True)
            cache = compute_graph_cache(
                cfg,
                task,
                record,
                model,
                graph_index=idx,
                seed=seed,
                buckets=buckets,
                r=receptive_radius,
                focal_count=focal_nodes,
                broad_random=broad_random_swaps,
                targeted_per_bucket=targeted_swaps_per_bucket,
                batch_size=batch_size,
                device=device,
            )
            torch.save(cache, out_path)
        effect_rows.extend(cache["effect_rows"])
        ledger_rows.extend(cache["ledger_rows"])
        reached_rows.extend(cache["reached_rows"])
        influence_rows.append(cache["influence_stats"])
        manifest_graphs.append({"graph_id": cache["graph_id"], "path": str(out_path), "swaps": len(cache["swaps"])})
    metrics = lr_metrics_dir(cfg, task, seed)
    write_csv(metrics / "rq0_rq1_effects.csv", effect_rows)
    write_csv(metrics / "rq1_ledger.csv", ledger_rows)
    write_csv(metrics / "rq1_reached_mattered.csv", reached_rows)
    write_csv(metrics / "rq2_influence_stats.csv", influence_rows)
    write_json(
        lr_manifest_path(cfg, task, seed),
        {
            "task": task,
            "seed": int(seed),
            "split": split,
            "num_graphs": len(records),
            "distance_buckets": [bucket.name for bucket in buckets],
            "receptive_field_radius_r": int(receptive_radius),
            "centrality_measure": "betweenness_centrality",
            "normaliser": "symbolic_feature_l2",
            "type_compatibility_class": "degree",
            "readout_definition": "sum over node-regression outputs for pooled P1/P2 diagnostics; direct node outputs for node-level primary deltas",
            "checkpoint": str(checkpoint) if checkpoint is not None else "default best.pt",
            "graphs": manifest_graphs,
        },
    )
    print(f"[long-range-cache] wrote {lr_task_root(cfg, task, seed)}", flush=True)
    return lr_manifest_path(cfg, task, seed)


def group_numeric(rows: Sequence[Mapping[str, Any]], key: str, value: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        try:
            out[str(row[key])].append(float(row[value]))
        except Exception:
            pass
    return out


def mean_band(values: Sequence[float]) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


def plot_rq0_rq1(cfg: Mapping[str, Any], task: str, *, seed: int = 9701) -> None:
    metrics = lr_metrics_dir(cfg, task, seed)
    fig_dir = lr_figures_dir(cfg, task, seed)
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv_dicts(metrics / "rq0_rq1_effects.csv")
    if not rows:
        raise FileNotFoundError("missing rq0_rq1_effects.csv; run build-cache first")
    buckets = list(dict.fromkeys(row["distance_bucket"] for row in rows))
    x = np.arange(len(buckets))

    def profile(column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        by = group_numeric(rows, "distance_bucket", column)
        vals = [mean_band(by.get(bucket, [])) for bucket in buckets]
        return tuple(np.asarray([v[idx] for v in vals], dtype=np.float64) for idx in range(3))  # type: ignore[return-value]

    oracle_mean, oracle_q1, oracle_q3 = profile("oracle_node_delta_per_feature")
    model_mean, model_q1, model_q3 = profile("model_node_delta_per_feature")

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(x, oracle_mean, marker="o", label="oracle demand", color="#2b6cb0")
    ax.fill_between(x, oracle_q1, oracle_q3, color="#2b6cb0", alpha=0.18)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel("distance bucket from focal node")
    ax.set_ylabel("oracle Δ per unit symbolic change")
    ax.set_title("F0.1 Task demand distance profile")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "F0.1_task_demand_distance_profile.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(x, oracle_mean, marker="o", label="oracle demand", color="#2b6cb0")
    ax.fill_between(x, oracle_q1, oracle_q3, color="#2b6cb0", alpha=0.14)
    ax.plot(x, model_mean, marker="s", label="model usage", color="#1b7f79")
    ax.fill_between(x, model_q1, model_q3, color="#1b7f79", alpha=0.14)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel("distance bucket from focal node")
    ax.set_ylabel("Δ per unit symbolic change")
    ax.set_title("F1.2 Model usage vs task demand")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "F1.2_model_usage_vs_task_demand.pdf")
    plt.close(fig)

    per_graph: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        per_graph[str(row["graph_id"])][str(row["distance_bucket"])].append(float(row["oracle_node_delta_per_feature"]))
    with PdfPages(fig_dir / "F0.A1_per_graph_demand_profiles.pdf") as pdf:
        graph_ids = list(per_graph)[:40]
        for start in range(0, len(graph_ids), 12):
            subset = graph_ids[start : start + 12]
            fig, axes = plt.subplots(3, 4, figsize=(12, 8), sharex=True, sharey=True)
            for ax, gid in zip(axes.reshape(-1), subset):
                vals = [np.mean(per_graph[gid].get(bucket, [np.nan])) for bucket in buckets]
                ax.plot(x, vals, marker="o", color="#2b6cb0")
                ax.set_title(gid, fontsize=8)
                ax.grid(axis="y", alpha=0.2)
            for ax in axes.reshape(-1)[len(subset) :]:
                ax.axis("off")
            fig.suptitle("F0.A1 Per-graph demand profiles")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    data = [group_numeric(rows, "distance_bucket", "oracle_node_delta_per_feature").get(bucket, []) for bucket in buckets]
    ax.violinplot(data, positions=x, showmedians=True, showextrema=False)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_title("F0.A2 Per-bucket demand distributions")
    ax.set_ylabel("oracle Δ per unit symbolic change")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "F0.A2_per_bucket_demand_distributions.pdf")
    plt.close(fig)

    ledger_rows = read_csv_dicts(metrics / "rq1_ledger.csv")
    reached_rows = read_csv_dicts(metrics / "rq1_reached_mattered.csv")
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    p1 = [float(row["near_p1_sum"]) + float(row["far_p1_sum"]) for row in ledger_rows]
    full = [float(row["full_delta_norm"]) for row in ledger_rows]
    ax.scatter(p1, full, s=8, alpha=0.45, color="#4a5568")
    lim = max([0.0] + p1 + full)
    ax.plot([0, lim], [0, lim], color="black", lw=0.8)
    ax.set_xlabel("sum per-node |P1|")
    ax.set_ylabel("full graph Δ")
    ax.set_title("F1.A5 Ledger check")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "F1.A5_ledger_check.pdf")
    plt.close(fig)

    by_bucket = defaultdict(lambda: {"n": 0, "reached": 0, "mattered": 0})
    for row in reached_rows:
        b = by_bucket[str(row["distance_bucket"])]
        b["n"] += 1
        b["reached"] += int(row["reached"])
        b["mattered"] += int(row["mattered"])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    reached = [by_bucket[b]["reached"] / max(1, by_bucket[b]["n"]) for b in buckets]
    mattered = [by_bucket[b]["mattered"] / max(1, by_bucket[b]["n"]) for b in buckets]
    width = 0.36
    ax.bar(x - width / 2, reached, width, label="reached h_j", color="#805ad5")
    ax.bar(x + width / 2, mattered, width, label="mattered output", color="#1b7f79")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylim(0, 1.0)
    ax.set_title("F1.A2 Reached vs mattered")
    ax.set_ylabel("fraction of receiver-swap observations")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "F1.A2_reached_vs_mattered.pdf")
    plt.close(fig)


def plot_graph_map(ax: Any, matrix: np.ndarray, spd: np.ndarray, focal: int, *, title: str) -> None:
    values = np.asarray(matrix[int(focal)], dtype=np.float64)
    order = np.argsort(spd[int(focal)])
    ax.bar(np.arange(len(order)), values[order], color="#1b7f79")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([str(int(i)) for i in order], rotation=90, fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("source node ordered by distance")
    ax.set_ylabel("RMS influence")
    ax.grid(axis="y", alpha=0.25)


def spectral_layout(adjacency: np.ndarray) -> np.ndarray:
    adj = np.asarray(adjacency, dtype=np.float64)
    n = int(adj.shape[0])
    if n <= 2:
        theta = np.linspace(0.0, 2.0 * math.pi, max(n, 3), endpoint=False)[:n]
        return np.column_stack([np.cos(theta), np.sin(theta)])
    deg = np.diag(adj.sum(axis=1))
    lap = deg - adj
    try:
        vals, vecs = np.linalg.eigh(lap)
        order = np.argsort(vals)
        coords = vecs[:, order[1:3]]
        if coords.shape[1] < 2:
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        theta = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        coords = np.column_stack([np.cos(theta), np.sin(theta)])
    coords = coords - np.nanmean(coords, axis=0, keepdims=True)
    scale = np.nanmax(np.abs(coords))
    return coords / max(float(scale), EPS)


def draw_source_graph(ax: Any, values: np.ndarray, adjacency: np.ndarray, focal: int, *, title: str, cmap: str = "viridis") -> Any:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    adj = np.asarray(adjacency)
    coords = spectral_layout(adj)
    n = int(adj.shape[0])
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] > 0:
                ax.plot([coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]], color="#a0aec0", lw=0.7, alpha=0.55, zorder=1)
    vmax = float(np.nanmax(values)) if values.size else 0.0
    sizes = 60.0 + 260.0 * (values / max(vmax, EPS))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=sizes, cmap=cmap, edgecolors="#1a202c", linewidths=0.35, zorder=2)
    ax.scatter([coords[int(focal), 0]], [coords[int(focal), 1]], s=360, facecolors="none", edgecolors="#c53030", linewidths=2.0, zorder=3)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    return sc


def plot_rq1_source_maps(cfg: Mapping[str, Any], task: str, *, seed: int = 9701) -> None:
    graph_files = sorted(lr_graph_dir(cfg, task, seed).glob("*.pt"))
    if not graph_files:
        raise FileNotFoundError("missing graph cache files; run build-cache first")
    fig_dir = lr_figures_dir(cfg, task, seed)
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache = torch.load(graph_files[0], map_location="cpu", weights_only=False)
    focal = int(cache["focal_nodes"][0])
    spd = np.asarray(cache["spd"])
    plot_graph_map_pdf = fig_dir / "F1.1_source_map_exemplar.pdf"
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    if "adjacency" in cache:
        sc = draw_source_graph(
            ax,
            np.asarray(cache["source_map_p1"], dtype=np.float64)[focal],
            np.asarray(cache["adjacency"]),
            focal,
            title=f"F1.1 Source map exemplar: focal {focal}",
        )
        fig.colorbar(sc, ax=ax, shrink=0.75, label="P1 RMS influence")
    else:
        plot_graph_map(ax, cache["source_map_p1"], spd, focal, title=f"F1.1 Source map exemplar: focal {focal}")
    fig.tight_layout()
    fig.savefig(plot_graph_map_pdf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    p1_vals = []
    p2_vals = []
    for graph_file in graph_files:
        graph = torch.load(graph_file, map_location="cpu", weights_only=False)
        p1 = np.asarray(graph["source_map_p1"])
        p2 = np.asarray(graph["source_map_p2"])
        far = far_mask_from_spd(torch.as_tensor(graph["spd"]), int(graph["r"]))
        p1_vals.extend(p1[far & np.isfinite(p1)].tolist())
        p2_vals.extend(p2[far & np.isfinite(p2)].tolist())
    ax.hist2d(p1_vals, p2_vals, bins=45, cmap="viridis")
    ax.set_xlabel("|P1| carriage")
    ax.set_ylabel("|P2| uniqueness / monopoly")
    ax.set_title("F1.3 Carriage vs uniqueness")
    fig.tight_layout()
    fig.savefig(fig_dir / "F1.3_carriage_vs_uniqueness.pdf")
    plt.close(fig)

    with PdfPages(fig_dir / "F1.A1_source_map_gallery.pdf") as pdf:
        for graph_file in graph_files[:24]:
            graph = torch.load(graph_file, map_location="cpu", weights_only=False)
            spd = np.asarray(graph["spd"])
            for focal in graph["focal_nodes"][:1]:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
                if "adjacency" in graph:
                    draw_source_graph(axes[0], np.asarray(graph["source_map_p1"])[int(focal)], np.asarray(graph["adjacency"]), int(focal), title=f"{graph['graph_id']} P1")
                    draw_source_graph(axes[1], np.asarray(graph["source_map_p2"])[int(focal)], np.asarray(graph["adjacency"]), int(focal), title=f"{graph['graph_id']} P2")
                else:
                    plot_graph_map(axes[0], graph["source_map_p1"], spd, int(focal), title=f"{graph['graph_id']} P1")
                    plot_graph_map(axes[1], graph["source_map_p2"], spd, int(focal), title=f"{graph['graph_id']} P2")
                fig.suptitle("F1.A1 Source-map gallery")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    cache = torch.load(graph_files[0], map_location="cpu", weights_only=False)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    if "adjacency" in cache:
        sc = draw_source_graph(
            ax,
            np.asarray(cache["failure_map"])[int(cache["focal_nodes"][0])],
            np.asarray(cache["adjacency"]),
            int(cache["focal_nodes"][0]),
            title="F1.A3 Failure map",
            cmap="magma",
        )
        fig.colorbar(sc, ax=ax, shrink=0.75, label="delta residual RMS")
    else:
        plot_graph_map(ax, cache["failure_map"], np.asarray(cache["spd"]), int(cache["focal_nodes"][0]), title="F1.A3 Failure map")
    fig.tight_layout()
    fig.savefig(fig_dir / "F1.A3_failure_map.pdf")
    plt.close(fig)

    rows = read_csv_dicts(lr_metrics_dir(cfg, task, seed) / "rq0_rq1_effects.csv")
    buckets = list(dict.fromkeys(row["distance_bucket"] for row in rows))
    x = np.arange(len(buckets))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for col, label, color in [
        ("model_node_delta_per_feature", "model", "#1b7f79"),
        ("oracle_node_delta_per_feature", "oracle", "#2b6cb0"),
    ]:
        grouped = group_numeric(rows, "distance_bucket", col)
        vals = [mean_band(grouped.get(bucket, []))[0] for bucket in buckets]
        ax.plot(x, vals, marker="o", label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_title("F1.A4 Usage profile by dataset / model variant")
    ax.set_ylabel("Δ per unit symbolic change")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(lr_figures_dir(cfg, task, seed) / "F1.A4_usage_profile_by_dataset_model_variant.pdf")
    plt.close(fig)


def plot_rq2(cfg: Mapping[str, Any], task: str, *, seed: int = 9701) -> None:
    graph_files = sorted(lr_graph_dir(cfg, task, seed).glob("*.pt"))
    fig_dir = lr_figures_dir(cfg, task, seed)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not graph_files:
        raise FileNotFoundError("missing graph cache files; run build-cache first")
    first = torch.load(graph_files[0], map_location="cpu", weights_only=False)
    mat = np.asarray(first["source_map_direct"], dtype=np.float64)
    spd = np.asarray(first["spd"])
    order = np.argsort(first["centrality"])[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(mat[np.ix_(order, order)], cmap="magma", aspect="auto")
    ax.set_title("F2.1 Influence-matrix heatmap")
    ax.set_xlabel("source")
    ax.set_ylabel("receiver")
    fig.colorbar(im, ax=ax, shrink=0.8, label="RMS influence")
    fig.tight_layout()
    fig.savefig(fig_dir / "F2.1_influence_matrix_heatmap.pdf")
    plt.close(fig)

    spectra = []
    rows = []
    for graph_file in graph_files:
        graph = torch.load(graph_file, map_location="cpu", weights_only=False)
        far = far_mask_from_spd(torch.as_tensor(graph["spd"]), int(graph["r"]))
        clean = np.nan_to_num(np.where(far, graph["source_map_direct"], np.nan), nan=0.0)
        s = np.linalg.svd(clean, compute_uv=False)
        if s.sum() > EPS:
            spectra.append(s / s.sum())
        rows.append(graph["influence_stats"])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    if spectra:
        max_len = max(len(s) for s in spectra)
        spec = np.asarray([np.pad(s, (0, max_len - len(s)), constant_values=np.nan) for s in spectra])
        mean = np.nanmean(spec, axis=0)
        q1 = np.nanpercentile(spec, 25, axis=0)
        q3 = np.nanpercentile(spec, 75, axis=0)
        xs = np.arange(1, len(mean) + 1)
        ax.plot(xs, mean, marker="o", color="#2b6cb0")
        ax.fill_between(xs, q1, q3, color="#2b6cb0", alpha=0.18)
    else:
        ax.text(0.5, 0.5, "no nonzero far-block spectra", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("singular value rank")
    ax.set_ylabel("normalised singular value")
    ax.set_title("F2.2 Spectrum / rank")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "F2.2_spectrum_rank.pdf")
    plt.close(fig)

    stats = read_csv_dicts(lr_metrics_dir(cfg, task, seed) / "rq2_influence_stats.csv")
    motif_cols = ["effective_rank", "column_concentration", "row_concentration", "matching_concentration"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    data = [[float(row[col]) for row in stats if row.get(col, "") not in {"", "nan"}] for col in motif_cols]
    ax.boxplot(data, labels=[c.replace("_", "\n") for c in motif_cols], showfliers=False)
    ax.set_title("F2.3 Motif summary")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "F2.3_motif_summary.pdf")
    plt.close(fig)

    with PdfPages(fig_dir / "F2.A1_influence_matrix_gallery.pdf") as pdf:
        for graph_file in graph_files[:24]:
            graph = torch.load(graph_file, map_location="cpu", weights_only=False)
            mat = np.asarray(graph["source_map_direct"], dtype=np.float64)
            order = np.argsort(graph["centrality"])[::-1]
            fig, ax = plt.subplots(figsize=(5, 4.4))
            im = ax.imshow(mat[np.ix_(order, order)], cmap="magma", aspect="auto")
            ax.set_title(graph["graph_id"], fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.75)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    col_strength = []
    row_strength = []
    far_mass = []
    reciprocity = []
    eff_sources = []
    for graph_file in graph_files:
        graph = torch.load(graph_file, map_location="cpu", weights_only=False)
        far = far_mask_from_spd(torch.as_tensor(graph["spd"]), int(graph["r"]))
        mat = np.asarray(graph["source_map_direct"], dtype=np.float64)
        far_mat = np.where(far, mat, np.nan)
        col_strength.extend(np.nansum(far_mat, axis=0).tolist())
        row_strength.extend(np.nansum(far_mat, axis=1).tolist())
        far_mass.append(graph["influence_stats"]["far_mass_fraction"])
        reciprocity.append(graph["influence_stats"]["reciprocity"])
        eff_sources.append(graph["influence_stats"]["effective_sources_per_receiver"])
    with PdfPages(fig_dir / "F2.A2_column_row_strength_distributions.pdf") as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        axes[0].hist(col_strength, bins=35, alpha=0.75, label="columns / broadcasters", color="#2b6cb0")
        axes[1].hist(row_strength, bins=35, alpha=0.75, label="rows / receivers", color="#805ad5")
        axes[0].set_title("F2.A2 Column strength")
        axes[1].set_title("F2.A2 Row strength")
        for ax in axes:
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        for graph_file in graph_files[:12]:
            graph = torch.load(graph_file, map_location="cpu", weights_only=False)
            if "adjacency" not in graph:
                continue
            far = far_mask_from_spd(torch.as_tensor(graph["spd"]), int(graph["r"]))
            mat = np.asarray(graph["source_map_direct"], dtype=np.float64)
            far_mat = np.where(far, mat, np.nan)
            col = np.nan_to_num(np.nansum(far_mat, axis=0), nan=0.0)
            row = np.nan_to_num(np.nansum(far_mat, axis=1), nan=0.0)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
            sc0 = draw_source_graph(axes[0], col, np.asarray(graph["adjacency"]), int(np.nanargmax(row)), title=f"{graph['graph_id']} broadcasters", cmap="Blues")
            sc1 = draw_source_graph(axes[1], row, np.asarray(graph["adjacency"]), int(np.nanargmax(row)), title=f"{graph['graph_id']} receivers", cmap="Purples")
            fig.colorbar(sc0, ax=axes[0], shrink=0.7, label="column strength")
            fig.colorbar(sc1, ax=axes[1], shrink=0.7, label="row strength")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(far_mass, bins=25, color="#1b7f79")
    axes[1].hist(reciprocity, bins=25, color="#b24c3d")
    axes[0].set_title("F2.A3 Far-mass fraction")
    axes[1].set_title("F2.A3 Reciprocity")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "F2.A3_far_mass_and_reciprocity.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(eff_sources, bins=25, color="#4a5568")
    ax.set_title("F2.A4 Effective sources per receiver")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "F2.A4_effective_sources_per_receiver.pdf")
    plt.close(fig)


def sample_disjoint_far_pairs(record: Mapping[str, Any], focal: int, *, r: int, max_pairs: int, rng: random.Random) -> list[dict[str, Any]]:
    n = int(record["n"])
    spd = record["struct"]["shortest_path_distance"]
    far_nodes = [i for i in range(n) if int(spd[int(focal), i]) > int(r)]
    pairs = [(u, v) for idx, u in enumerate(far_nodes) for v in far_nodes[idx + 1 :] if type_compatibility_class(record, u) == type_compatibility_class(record, v)]
    combos = []
    for idx, a in enumerate(pairs):
        aset = set(a)
        for b in pairs[idx + 1 :]:
            if aset.isdisjoint(b):
                sep = min(int(spd[x, y]) for x in a for y in b)
                a_dist = min(int(spd[int(focal), x]) for x in a)
                b_dist = min(int(spd[int(focal), x]) for x in b)
                combos.append((a, b, sep, a_dist, b_dist))
    rng.shuffle(combos)
    return [
        {
            "pair_id": f"p{idx:04d}",
            "a_u": a[0],
            "a_v": a[1],
            "b_u": b[0],
            "b_v": b[1],
            "ab_separation": sep,
            "a_distance_from_focal": a_dist,
            "b_distance_from_focal": b_dist,
            "mean_distance_from_focal": 0.5 * (float(a_dist) + float(b_dist)),
        }
        for idx, (a, b, sep, a_dist, b_dist) in enumerate(combos[: int(max_pairs)])
    ]


def run_rq3(
    cfg: Mapping[str, Any],
    task: str,
    *,
    seed: int = 9701,
    max_pairs_per_graph: int = 32,
    batch_size: int = 512,
    device_name: str = "auto",
    config_path: Path | None = None,
    checkpoint: Path | None = None,
    backend: str | None = "official",
    progress_every_graphs: int = 1,
) -> Path:
    device = choose_device(device_name)
    configure_runtime(cfg, device)
    model_cfg = load_lr_model_config(cfg, task, config_path, fast_dev_run=False)
    model, _ = load_model_from_checkpoint(model_cfg, task, checkpoint, device, backend=backend)
    graph_files = sorted(lr_graph_dir(cfg, task, seed).glob("*.pt"))
    records_by_id = {str(row["graph_id"]): row for row in load_records(data_dir(cfg, task) / "test_id.pt")}
    rows = []
    rng = random.Random(int(seed) + 303)
    print(
        f"[long-range-rq3] start task={task} graphs={len(graph_files)} max_pairs_per_graph={int(max_pairs_per_graph)} "
        f"batch_size={int(batch_size)} device={device}",
        flush=True,
    )
    for graph_idx, graph_file in enumerate(graph_files, start=1):
        graph = torch.load(graph_file, map_location="cpu", weights_only=False)
        record = records_by_id[graph["graph_id"]]
        focal = int(graph["focal_nodes"][0])
        pairs = sample_disjoint_far_pairs(record, focal, r=int(graph["r"]), max_pairs=max_pairs_per_graph, rng=rng)
        if graph_idx == 1 or graph_idx == len(graph_files) or graph_idx % max(1, int(progress_every_graphs)) == 0:
            print(
                f"[long-range-rq3] graph={graph_idx}/{len(graph_files)} id={graph['graph_id']} "
                f"focal={focal} pairs={len(pairs)}",
                flush=True,
            )
        clean_states, clean_outputs = model_states_and_outputs(model, [record], batch_size=1, device=device)
        h_clean = clean_states[0]
        y_clean = clean_outputs[0]
        y_clean_sum = readout_graph_sum(y_clean)
        teacher_clean = record["teacher"]["Y"].float()
        teacher_clean_sum = graph_sum_teacher(record)
        for pair in pairs:
            rec_a = content_swap_record(record, pair["a_u"], pair["a_v"], cfg, graph_id=f"{record['graph_id']}__rq3A_{pair['pair_id']}")
            rec_b = content_swap_record(record, pair["b_u"], pair["b_v"], cfg, graph_id=f"{record['graph_id']}__rq3B_{pair['pair_id']}")
            rec_ab0 = content_swap_record(record, pair["a_u"], pair["a_v"], cfg, graph_id=f"{record['graph_id']}__rq3AB0_{pair['pair_id']}")
            rec_ab = content_swap_record(rec_ab0, pair["b_u"], pair["b_v"], cfg, graph_id=f"{record['graph_id']}__rq3AB_{pair['pair_id']}")
            _states, outs = model_states_and_outputs(model, [rec_a, rec_b, rec_ab], batch_size=batch_size, device=device)
            y_a, y_b, y_ab = [readout_graph_sum(out) for out in outs]
            delta_a = y_a - y_clean_sum
            delta_b = y_b - y_clean_sum
            delta_ab = y_ab - y_clean_sum
            output_interaction = delta_ab - (delta_a + delta_b)
            p1_a = outs[0][focal] - y_clean[focal]
            p1_b = outs[1][focal] - y_clean[focal]
            p1_ab = outs[2][focal] - y_clean[focal]
            p1_interaction = p1_ab - (p1_a + p1_b)
            oracle_a = graph_sum_teacher(rec_a) - teacher_clean_sum
            oracle_b = graph_sum_teacher(rec_b) - teacher_clean_sum
            oracle_ab = graph_sum_teacher(rec_ab) - teacher_clean_sum
            oracle_interaction = oracle_ab - (oracle_a + oracle_b)
            node_a = rec_a["teacher"]["Y"][focal] - teacher_clean[focal]
            node_b = rec_b["teacher"]["Y"][focal] - teacher_clean[focal]
            node_ab = rec_ab["teacher"]["Y"][focal] - teacher_clean[focal]
            rows.append(
                {
                    "task": task,
                    "graph_id": graph["graph_id"],
                    "focal_node": focal,
                    **pair,
                    "delta_ab_norm": norm_vec(delta_ab),
                    "delta_a_plus_b_norm": norm_vec(delta_a + delta_b),
                    "output_interaction_norm": norm_vec(output_interaction),
                    "p1_interaction_norm": norm_vec(p1_interaction),
                    "pooling_induced_interaction_norm": norm_vec(output_interaction - p1_interaction),
                    "oracle_ab_norm": norm_vec(oracle_ab),
                    "oracle_a_plus_b_norm": norm_vec(oracle_a + oracle_b),
                    "oracle_interaction_norm": norm_vec(oracle_interaction),
                    "oracle_node_ab_norm": norm_vec(node_ab),
                    "oracle_node_a_plus_b_norm": norm_vec(node_a + node_b),
                    "oracle_node_interaction_norm": norm_vec(node_ab - (node_a + node_b)),
                }
            )
    out = lr_metrics_dir(cfg, task, seed) / "rq3_interactions.csv"
    write_csv(out, rows)
    print(f"[long-range-rq3] wrote rows={len(rows)} path={out}", flush=True)
    return out


def plot_rq3(cfg: Mapping[str, Any], task: str, *, seed: int = 9701) -> None:
    rows = read_csv_dicts(lr_metrics_dir(cfg, task, seed) / "rq3_interactions.csv")
    if not rows:
        raise FileNotFoundError("missing rq3_interactions.csv; run run-rq3 first")
    fig_dir = lr_figures_dir(cfg, task, seed)
    fig_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray([float(r["delta_a_plus_b_norm"]) for r in rows])
    y = np.asarray([float(r["delta_ab_norm"]) for r in rows])
    sep = np.asarray([float(r["ab_separation"]) for r in rows])
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    sc = ax.scatter(x, y, c=sep, cmap="viridis", s=18, alpha=0.75)
    lim = max(float(np.nanmax(x)), float(np.nanmax(y)), EPS)
    ax.plot([0, lim], [0, lim], color="black", lw=0.8)
    ax.set_xlabel("||δ_A + δ_B||")
    ax.set_ylabel("||δ_AB||")
    ax.set_title("F3.1 Additivity scatter")
    fig.colorbar(sc, ax=ax, label="A-B separation")
    fig.tight_layout()
    fig.savefig(fig_dir / "F3.1_additivity_scatter.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(sep, [float(r["output_interaction_norm"]) for r in rows], label="model output", alpha=0.7, s=18)
    ax.scatter(sep, [float(r["oracle_interaction_norm"]) for r in rows], label="oracle", alpha=0.7, s=18)
    ax.set_xlabel("A-B separation")
    ax.set_ylabel("|interaction|")
    ax.set_title("F3.2 Interaction vs distance")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "F3.2_interaction_vs_distance.pdf")
    plt.close(fig)

    with PdfPages(fig_dir / "F3.A1_per_graph_interaction_maps.pdf") as pdf:
        for graph_id in list(dict.fromkeys(row["graph_id"] for row in rows))[:24]:
            subset = [r for r in rows if r["graph_id"] == graph_id]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter([float(r["ab_separation"]) for r in subset], [float(r["output_interaction_norm"]) for r in subset], s=18)
            ax.set_title(f"F3.A1 {graph_id}", fontsize=8)
            ax.set_xlabel("A-B separation")
            ax.set_ylabel("output interaction")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    axes[0].hist([float(r["output_interaction_norm"]) for r in rows], bins=35, color="#1b7f79")
    axes[1].hist([float(r["oracle_interaction_norm"]) for r in rows], bins=35, color="#2b6cb0")
    axes[0].set_title("F3.A2 Model additivity")
    axes[1].set_title("F3.A2 Oracle additivity")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "F3.A2_model_vs_oracle_additivity.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter([float(r["p1_interaction_norm"]) for r in rows], [float(r["output_interaction_norm"]) for r in rows], s=18, alpha=0.7)
    lim = max([float(r["output_interaction_norm"]) for r in rows] + [float(r["p1_interaction_norm"]) for r in rows] + [EPS])
    ax.plot([0, lim], [0, lim], color="black", lw=0.8)
    ax.set_xlabel("P1-level interaction")
    ax.set_ylabel("output-level interaction")
    ax.set_title("F3.A3 P1-level vs output-level interaction")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "F3.A3_p1_level_vs_output_level_interaction.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ox = np.asarray([float(r["oracle_a_plus_b_norm"]) for r in rows])
    oy = np.asarray([float(r["oracle_ab_norm"]) for r in rows])
    sc = ax.scatter(ox, oy, c=sep, cmap="viridis", s=18, alpha=0.75)
    lim = max(float(np.nanmax(ox)), float(np.nanmax(oy)), EPS)
    ax.plot([0, lim], [0, lim], color="black", lw=0.8)
    ax.set_xlabel("||δ_A^oracle + δ_B^oracle||")
    ax.set_ylabel("||δ_AB^oracle||")
    ax.set_title("F0.A3 Oracle additivity preview")
    fig.colorbar(sc, ax=ax, label="A-B separation")
    fig.tight_layout()
    fig.savefig(fig_dir / "F0.A3_oracle_additivity_preview.pdf")
    plt.close(fig)
    print(f"[long-range-plot] wrote RQ3 figures to {fig_dir}", flush=True)


def type_compatible_far_relayout(record: Mapping[str, Any], far_nodes: Sequence[int], rng: random.Random) -> dict[int, int]:
    by_class: dict[str, list[int]] = defaultdict(list)
    for node in far_nodes:
        by_class[type_compatibility_class(record, int(node))].append(int(node))
    old_to_new: dict[int, int] = {}
    for nodes in by_class.values():
        targets = list(nodes)
        if len(targets) > 1:
            for _ in range(20):
                rng.shuffle(targets)
                if any(int(old) != int(new) for old, new in zip(nodes, targets)):
                    break
        for old, new in zip(nodes, targets):
            old_to_new[int(old)] = int(new)
    return old_to_new


def sample_far_block_partner_swaps(
    record: Mapping[str, Any],
    far_nodes: Sequence[int],
    *,
    partners_per_source: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    by_class: dict[str, list[int]] = defaultdict(list)
    for node in far_nodes:
        by_class[type_compatibility_class(record, int(node))].append(int(node))
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for source in far_nodes:
        partners = [node for node in by_class[type_compatibility_class(record, int(source))] if int(node) != int(source)]
        rng.shuffle(partners)
        for partner in partners[: int(partners_per_source)]:
            u, v = sorted((int(source), int(partner)))
            key = (u, v)
            if key not in selected:
                selected[key] = {
                    "swap_id": f"rq4_s{len(selected):05d}_{u}_{v}",
                    "u": u,
                    "v": v,
                    "sample_source": "far_block_partner_marginalisation",
                    "target_focal": "",
                    "target_bucket": "",
                    "d_uv": int(record["struct"]["shortest_path_distance"][u, v]),
                    "feature_l2": content_feature_distance(record, u, v),
                    "type_class_u": type_compatibility_class(record, u),
                    "type_class_v": type_compatibility_class(record, v),
                }
    return list(selected.values())


def focal_source_map_after_relayout(
    cfg: Mapping[str, Any],
    record: Mapping[str, Any],
    model: torch.nn.Module,
    *,
    focal: int,
    far_nodes: Sequence[int],
    partners_per_source: int,
    batch_size: int,
    device: torch.device,
    rng: random.Random,
    graph_id_prefix: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    swaps = sample_far_block_partner_swaps(record, far_nodes, partners_per_source=partners_per_source, rng=rng)
    n = int(record["n"])
    if not swaps:
        return np.full(n, np.nan, dtype=np.float64), []
    clean_states, clean_outputs = model_states_and_outputs(model, [record], batch_size=1, device=device)
    y_clean = clean_outputs[0]
    source_records = [
        content_swap_record(record, int(swap["u"]), int(swap["v"]), cfg, graph_id=f"{graph_id_prefix}__{swap['swap_id']}")
        for swap in swaps
    ]
    _source_states, source_outputs = model_states_and_outputs(model, source_records, batch_size=batch_size, device=device)
    values_by_swap: dict[str, np.ndarray] = {}
    for swap, y_s in zip(swaps, source_outputs):
        values_by_swap[str(swap["swap_id"])] = torch.linalg.vector_norm((y_s - y_clean).float(), dim=1).numpy()
    source_map = build_source_maps(n, swaps, values_by_swap)
    return np.asarray(source_map[int(focal)], dtype=np.float64), swaps


def run_rq4(
    cfg: Mapping[str, Any],
    task: str,
    *,
    seed: int = 9701,
    num_pi: int = 5,
    partners_per_source: int = 8,
    batch_size: int = 512,
    device_name: str = "auto",
    config_path: Path | None = None,
    checkpoint: Path | None = None,
    backend: str | None = "official",
    progress_every_graphs: int = 1,
) -> Path:
    device = choose_device(device_name)
    configure_runtime(cfg, device)
    model_cfg = load_lr_model_config(cfg, task, config_path, fast_dev_run=False)
    model, _ = load_model_from_checkpoint(model_cfg, task, checkpoint, device, backend=backend)
    records_by_id = {str(row["graph_id"]): row for row in load_records(data_dir(cfg, task) / "test_id.pt")}
    graph_files = sorted(lr_graph_dir(cfg, task, seed).glob("*.pt"))
    rows = []
    exemplars = []
    rng = random.Random(int(seed) + 404)
    print(
        f"[long-range-rq4] start task={task} graphs={len(graph_files)} num_pi={int(num_pi)} "
        f"partners_per_source={int(partners_per_source)} batch_size={int(batch_size)} device={device}",
        flush=True,
    )
    for graph_idx, graph_file in enumerate(graph_files, start=1):
        graph = torch.load(graph_file, map_location="cpu", weights_only=False)
        record = records_by_id[str(graph["graph_id"])]
        spd = np.asarray(graph["spd"])
        for focal in graph["focal_nodes"][:1]:
            focal = int(focal)
            far_nodes = [idx for idx in range(graph["n"]) if int(spd[focal, idx]) > int(graph["r"])]
            if graph_idx == 1 or graph_idx == len(graph_files) or graph_idx % max(1, int(progress_every_graphs)) == 0:
                print(
                    f"[long-range-rq4] graph={graph_idx}/{len(graph_files)} id={graph['graph_id']} "
                    f"focal={focal} far_nodes={len(far_nodes)}",
                    flush=True,
                )
            if len(far_nodes) < 3:
                continue
            s0 = np.asarray(graph["source_map_p1"][focal], dtype=np.float64)
            alpha = np.nan_to_num(s0[far_nodes], nan=0.0)
            if alpha.sum() <= EPS:
                continue
            alpha = alpha / alpha.sum()
            for pi_idx in range(int(num_pi)):
                old_to_new = type_compatible_far_relayout(record, far_nodes, rng)
                moved_fraction = float(np.mean([int(old) != int(new) for old, new in old_to_new.items()])) if old_to_new else 0.0
                if moved_fraction <= 0.0:
                    continue
                relayout = content_relayout_record(
                    record,
                    old_to_new,
                    cfg,
                    graph_id=f"{record['graph_id']}__rq4_relayout_{pi_idx:03d}",
                )
                s_pi_full, sampled_swaps = focal_source_map_after_relayout(
                    cfg,
                    relayout,
                    model,
                    focal=focal,
                    far_nodes=far_nodes,
                    partners_per_source=partners_per_source,
                    batch_size=batch_size,
                    device=device,
                    rng=rng,
                    graph_id_prefix=f"{record['graph_id']}__rq4_pi{pi_idx:03d}",
                )
                s_struct = s0[far_nodes]
                s_symbolic_full = np.full_like(s0, np.nan)
                for old, new in old_to_new.items():
                    s_symbolic_full[int(new)] = s0[int(old)]
                s_symbolic = s_symbolic_full[far_nodes]
                s_pi = s_pi_full[far_nodes]
                structural_score = weighted_corr(s_pi, s_struct, alpha)
                symbolic_score = weighted_corr(s_pi, s_symbolic, alpha)
                if graph_idx == 1 or graph_idx == len(graph_files) or graph_idx % max(1, int(progress_every_graphs)) == 0:
                    print(
                        f"[long-range-rq4] graph={graph_idx}/{len(graph_files)} pi={pi_idx + 1}/{int(num_pi)} "
                        f"sampled_swaps={len(sampled_swaps)} moved_fraction={moved_fraction:.3f} "
                        f"structural={structural_score:.4f} symbolic={symbolic_score:.4f}",
                        flush=True,
                    )
                if len(exemplars) < 12 and np.isfinite(structural_score) and np.isfinite(symbolic_score):
                    exemplars.append(
                        {
                            "task": task,
                            "graph_id": graph["graph_id"],
                            "focal_node": focal,
                            "pi_idx": pi_idx,
                            "far_nodes": far_nodes,
                            "old_to_new": old_to_new,
                            "s0": s0,
                            "s_pi": s_pi_full,
                            "s_struct": s0.copy(),
                            "s_symbolic": s_symbolic_full,
                            "structural_score": structural_score,
                            "symbolic_score": symbolic_score,
                        }
                    )
                cached_buckets = [
                    DistanceBucket(str(item["name"]), int(item["low"]), int(item["high"]))
                    for item in graph.get("bucket_specs", [])
                ] or parse_distance_buckets(DEFAULT_BUCKET_TEXT)
                for bucket_name in graph["buckets"]:
                    bucket_nodes = [node for node in far_nodes if bucket_for_distance(int(spd[focal, node]), cached_buckets) == bucket_name]
                    if len(bucket_nodes) < 2:
                        continue
                    idxs = [far_nodes.index(node) for node in bucket_nodes]
                    weights = alpha[idxs]
                    cv = float(np.nanstd(s_struct[idxs]) / max(float(np.nanmean(np.abs(s_struct[idxs]))), EPS))
                    finite_n = int(np.isfinite(s_pi[idxs]).sum())
                    rows.append(
                        {
                            "task": task,
                            "graph_id": graph["graph_id"],
                            "focal_node": focal,
                            "pi_idx": pi_idx,
                            "distance_bucket": bucket_name,
                            "structural_score": weighted_corr(s_pi[idxs], s_struct[idxs], weights),
                            "symbolic_score": weighted_corr(s_pi[idxs], s_symbolic[idxs], weights),
                            "clean_influence_cv": cv,
                            "sampled_swaps": len(sampled_swaps),
                            "finite_sources": finite_n,
                            "moved_fraction": moved_fraction,
                            "power_flag": "low_power" if cv < 0.05 or finite_n < 3 else "ok",
                        }
                    )
                rows.append(
                    {
                        "task": task,
                        "graph_id": graph["graph_id"],
                        "focal_node": focal,
                        "pi_idx": pi_idx,
                        "distance_bucket": "far_block",
                        "structural_score": structural_score,
                        "symbolic_score": symbolic_score,
                        "clean_influence_cv": float(np.nanstd(s_struct) / max(float(np.nanmean(np.abs(s_struct))), EPS)),
                        "sampled_swaps": len(sampled_swaps),
                        "finite_sources": int(np.isfinite(s_pi).sum()),
                        "moved_fraction": moved_fraction,
                        "power_flag": "ok" if int(np.isfinite(s_pi).sum()) >= 3 else "low_power",
                    }
                )
    out = lr_metrics_dir(cfg, task, seed) / "rq4_address_mode.csv"
    write_csv(out, rows)
    if exemplars:
        torch.save(exemplars, lr_metrics_dir(cfg, task, seed) / "rq4_address_mode_exemplars.pt")
    print(f"[long-range-rq4] wrote rows={len(rows)} path={out}", flush=True)
    return out


def plot_rq4(cfg: Mapping[str, Any], task: str, *, seed: int = 9701) -> None:
    rows = read_csv_dicts(lr_metrics_dir(cfg, task, seed) / "rq4_address_mode.csv")
    if not rows:
        raise FileNotFoundError("missing rq4_address_mode.csv; run run-rq4 first")
    fig_dir = lr_figures_dir(cfg, task, seed)
    fig_dir.mkdir(parents=True, exist_ok=True)
    far = [r for r in rows if r["distance_bucket"] == "far_block"]
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter([float(r["structural_score"]) for r in far], [float(r["symbolic_score"]) for r in far], s=22, alpha=0.75, color="#1b7f79")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("structural score")
    ax.set_ylabel("symbolic score")
    ax.set_title("F4.1 Address-mode plane")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "F4.1_address_mode_plane.pdf")
    plt.close(fig)

    buckets = [b for b in dict.fromkeys(r["distance_bucket"] for r in rows) if b != "far_block"]
    x = np.arange(len(buckets))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for col, label, color in [("structural_score", "structural", "#2b6cb0"), ("symbolic_score", "symbolic", "#805ad5")]:
        vals = [mean_band([float(r[col]) for r in rows if r["distance_bucket"] == bucket])[0] for bucket in buckets]
        ax.plot(x, vals, marker="o", label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel("distance bucket")
    ax.set_ylabel("weighted correlation")
    ax.set_title("F4.2 Address mode vs distance")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "F4.2_address_mode_vs_distance.pdf")
    plt.close(fig)

    with PdfPages(fig_dir / "F4.A1_per_graph_score_maps.pdf") as pdf:
        for graph_id in list(dict.fromkeys(r["graph_id"] for r in far))[:24]:
            subset = [r for r in far if r["graph_id"] == graph_id]
            fig, ax = plt.subplots(figsize=(5, 4.2))
            ax.scatter([float(r["structural_score"]) for r in subset], [float(r["symbolic_score"]) for r in subset], s=24)
            ax.set_title(f"F4.A1 {graph_id}", fontsize=8)
            ax.set_xlabel("structural")
            ax.set_ylabel("symbolic")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    exemplar_path = lr_metrics_dir(cfg, task, seed) / "rq4_address_mode_exemplars.pt"
    if exemplar_path.exists():
        exemplars = torch.load(exemplar_path, map_location="cpu", weights_only=False)
        scored = [
            (
                abs(float(ex["structural_score"]) - float(ex["symbolic_score"])),
                ex,
            )
            for ex in exemplars
            if np.isfinite(float(ex["structural_score"])) and np.isfinite(float(ex["symbolic_score"]))
        ]
        scored.sort(reverse=True, key=lambda item: item[0])
        if scored:
            ex = scored[0][1]
            s0 = np.asarray(ex["s0"], dtype=np.float64)
            s_pi = np.asarray(ex["s_pi"], dtype=np.float64)
            s_symbolic = np.asarray(ex["s_symbolic"], dtype=np.float64)
            far_nodes = [int(v) for v in ex["far_nodes"]]
            fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
            order = far_nodes
            axes[0].bar(np.arange(len(order)), s0[order], color="#2b6cb0")
            axes[1].bar(np.arange(len(order)), s_pi[order], color="#1b7f79")
            axes[2].bar(np.arange(len(order)), s_symbolic[order], color="#805ad5")
            axes[0].set_title("clean source map")
            axes[1].set_title("measured after relayout")
            axes[2].set_title("symbolic-follow prediction")
            for ax in axes:
                ax.set_xticks(np.arange(len(order)))
                ax.set_xticklabels([str(v) for v in order], rotation=90, fontsize=7)
                ax.grid(axis="y", alpha=0.25)
            fig.suptitle("F4.A2 Follow-vs-stay exemplar")
            fig.tight_layout()
            fig.savefig(fig_dir / "F4.A2_follow_vs_stay_exemplars.pdf")
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.hist([float(r["symbolic_score"]) - float(r["structural_score"]) for r in far], bins=30, color="#4a5568")
    ax.set_title("F4.A3 Scores by dataset / model variant")
    ax.set_xlabel("symbolic - structural")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "F4.A3_scores_by_dataset_model_variant.pdf")
    plt.close(fig)
    print(f"[long-range-plot] wrote RQ4 figures to {fig_dir}", flush=True)


def plot_all_from_cache(cfg: Mapping[str, Any], task: str, *, seed: int = 9701) -> None:
    plot_rq0_rq1(cfg, task, seed=seed)
    plot_rq1_source_maps(cfg, task, seed=seed)
    plot_rq2(cfg, task, seed=seed)
    print(f"[long-range-plot] wrote RQ0/RQ1/RQ2 figures to {lr_figures_dir(cfg, task, seed)}", flush=True)


def plot_combined_from_cache(cfg: Mapping[str, Any], tasks: Sequence[str], *, seed: int = 9701) -> None:
    fig_dir = lr_combined_figures_dir(cfg, seed)
    fig_dir.mkdir(parents=True, exist_ok=True)
    task_rows: dict[str, list[Mapping[str, Any]]] = {}
    for task in tasks:
        path = lr_metrics_dir(cfg, task, seed) / "rq0_rq1_effects.csv"
        if path.exists():
            rows = read_csv_dicts(path)
            if rows:
                task_rows[task] = rows
    if not task_rows:
        raise FileNotFoundError("no completed long-range rq0/rq1 metrics found for combined plotting")

    buckets = [bucket.name for bucket in parse_distance_buckets(DEFAULT_BUCKET_TEXT)]
    x = np.arange(len(buckets))
    palette = {"local_mean_gcn": "#4a5568", "ppr_diffusion": "#2b6cb0", "nearest_anchor_voronoi": "#1b7f79"}

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for task, rows in task_rows.items():
        grouped = group_numeric(rows, "distance_bucket", "oracle_node_delta_per_feature")
        vals = [mean_band(grouped.get(bucket, [])) for bucket in buckets]
        mean = np.asarray([v[0] for v in vals], dtype=np.float64)
        q1 = np.asarray([v[1] for v in vals], dtype=np.float64)
        q3 = np.asarray([v[2] for v in vals], dtype=np.float64)
        color = palette.get(task, None)
        ax.plot(x, mean, marker="o", label=task, color=color)
        ax.fill_between(x, q1, q3, color=color, alpha=0.12)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel("distance bucket from focal node")
    ax.set_ylabel("oracle Δ per unit symbolic change")
    ax.set_title("F0.1 Task demand distance profile")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "F0.1_task_demand_distance_profile.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharex=True)
    for task, rows in task_rows.items():
        color = palette.get(task, None)
        for ax, column, label in [
            (axes[0], "oracle_node_delta_per_feature", "oracle"),
            (axes[1], "model_node_delta_per_feature", "model"),
        ]:
            grouped = group_numeric(rows, "distance_bucket", column)
            vals = [mean_band(grouped.get(bucket, []))[0] for bucket in buckets]
            ax.plot(x, vals, marker="o", label=task, color=color)
            ax.set_title(label)
            ax.grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(buckets)
        ax.set_xlabel("distance bucket")
    axes[0].set_ylabel("Δ per unit symbolic change")
    axes[1].legend(frameon=False)
    fig.suptitle("F1.A4 Usage profile by dataset / model variant")
    fig.tight_layout()
    fig.savefig(fig_dir / "F1.A4_usage_profile_by_dataset_model_variant.pdf")
    plt.close(fig)

    rq4_rows: list[Mapping[str, Any]] = []
    for task in tasks:
        path = lr_metrics_dir(cfg, task, seed) / "rq4_address_mode.csv"
        if path.exists():
            rq4_rows.extend(read_csv_dicts(path))
    far_rows = [row for row in rq4_rows if row.get("distance_bucket") == "far_block"]
    if far_rows:
        labels = [task for task in tasks if any(row["task"] == task for row in far_rows)]
        data = [
            [float(row["symbolic_score"]) - float(row["structural_score"]) for row in far_rows if row["task"] == task]
            for task in labels
        ]
        fig, ax = plt.subplots(figsize=(7.0, 4.3))
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("symbolic score - structural score")
        ax.set_title("F4.A3 Scores by dataset / model variant")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "F4.A3_scores_by_dataset_model_variant.pdf")
        plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", type=Path)
        p.add_argument("--task", choices=TASKS, required=True)
        p.add_argument("--seed", type=int, default=9701)
        p.add_argument("--fast-dev-run", action="store_true")

    p = sub.add_parser("build-cache")
    common(p)
    p.add_argument("--split", default="test_id")
    p.add_argument("--num-graphs", type=int, default=200)
    p.add_argument("--distance-buckets", default=DEFAULT_BUCKET_TEXT)
    p.add_argument("--receptive-radius", type=int, default=2)
    p.add_argument("--focal-nodes", type=int, default=3)
    p.add_argument("--broad-random-swaps", type=int, default=160)
    p.add_argument("--targeted-swaps-per-bucket", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--model-config", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--backend", default="official")
    p.add_argument("--progress-every-graphs", type=int, default=1)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("plot-rq0-rq1-rq2")
    common(p)

    p = sub.add_parser("run-rq3")
    common(p)
    p.add_argument("--max-pairs-per-graph", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--model-config", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--backend", default="official")
    p.add_argument("--progress-every-graphs", type=int, default=1)

    p = sub.add_parser("plot-rq3")
    common(p)

    p = sub.add_parser("run-rq4")
    common(p)
    p.add_argument("--num-pi", type=int, default=5)
    p.add_argument("--partners-per-source", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--model-config", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--backend", default="official")
    p.add_argument("--progress-every-graphs", type=int, default=1)

    p = sub.add_parser("plot-rq4")
    common(p)

    p = sub.add_parser("plot-combined")
    p.add_argument("--config", type=Path)
    p.add_argument("--tasks", default="local_mean_gcn,ppr_diffusion,nearest_anchor_voronoi")
    p.add_argument("--seed", type=int, default=9701)
    p.add_argument("--fast-dev-run", action="store_true")

    p = sub.add_parser("run-all")
    common(p)
    p.add_argument("--split", default="test_id")
    p.add_argument("--num-graphs", type=int, default=200)
    p.add_argument("--distance-buckets", default=DEFAULT_BUCKET_TEXT)
    p.add_argument("--receptive-radius", type=int, default=2)
    p.add_argument("--focal-nodes", type=int, default=3)
    p.add_argument("--broad-random-swaps", type=int, default=160)
    p.add_argument("--targeted-swaps-per-bucket", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--model-config", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--backend", default="official")
    p.add_argument("--max-pairs-per-graph", type=int, default=32)
    p.add_argument("--num-pi", type=int, default=5)
    p.add_argument("--partners-per-source", type=int, default=8)
    p.add_argument("--progress-every-graphs", type=int, default=1)
    p.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "plot-combined":
        combined_tasks = [value.strip() for value in str(args.tasks).split(",") if value.strip()]
        config_task = combined_tasks[0] if combined_tasks else "ppr_diffusion"
    else:
        combined_tasks = []
        config_task = args.task
    cfg = load_config(args.config, task=config_task, fast_dev_run=bool(args.fast_dev_run))
    task = str(cfg["task"])
    if args.command == "build-cache":
        build_long_range_cache(
            cfg,
            task,
            seed=args.seed,
            split=args.split,
            num_graphs=args.num_graphs,
            distance_buckets=args.distance_buckets,
            receptive_radius=args.receptive_radius,
            focal_nodes=args.focal_nodes,
            broad_random_swaps=args.broad_random_swaps,
            targeted_swaps_per_bucket=args.targeted_swaps_per_bucket,
            batch_size=args.batch_size,
            device_name=args.device,
            config_path=args.model_config,
            checkpoint=args.checkpoint,
            backend=args.backend,
            progress_every_graphs=args.progress_every_graphs,
            force=args.force,
            fast_dev_run=bool(args.fast_dev_run),
        )
    elif args.command == "plot-rq0-rq1-rq2":
        plot_all_from_cache(cfg, task, seed=args.seed)
    elif args.command == "run-rq3":
        run_rq3(
            cfg,
            task,
            seed=args.seed,
            max_pairs_per_graph=args.max_pairs_per_graph,
            batch_size=args.batch_size,
            device_name=args.device,
            config_path=args.model_config,
            checkpoint=args.checkpoint,
            backend=args.backend,
            progress_every_graphs=args.progress_every_graphs,
        )
    elif args.command == "plot-rq3":
        plot_rq3(cfg, task, seed=args.seed)
    elif args.command == "run-rq4":
        run_rq4(
            cfg,
            task,
            seed=args.seed,
            num_pi=args.num_pi,
            partners_per_source=args.partners_per_source,
            batch_size=args.batch_size,
            device_name=args.device,
            config_path=args.model_config,
            checkpoint=args.checkpoint,
            backend=args.backend,
            progress_every_graphs=args.progress_every_graphs,
        )
    elif args.command == "plot-rq4":
        plot_rq4(cfg, task, seed=args.seed)
    elif args.command == "plot-combined":
        plot_combined_from_cache(cfg, combined_tasks, seed=args.seed)
    elif args.command == "run-all":
        build_long_range_cache(
            cfg,
            task,
            seed=args.seed,
            split=args.split,
            num_graphs=args.num_graphs,
            distance_buckets=args.distance_buckets,
            receptive_radius=args.receptive_radius,
            focal_nodes=args.focal_nodes,
            broad_random_swaps=args.broad_random_swaps,
            targeted_swaps_per_bucket=args.targeted_swaps_per_bucket,
            batch_size=args.batch_size,
            device_name=args.device,
            config_path=args.model_config,
            checkpoint=args.checkpoint,
            backend=args.backend,
            progress_every_graphs=args.progress_every_graphs,
            force=args.force,
            fast_dev_run=bool(args.fast_dev_run),
        )
        plot_all_from_cache(cfg, task, seed=args.seed)
        run_rq3(
            cfg,
            task,
            seed=args.seed,
            max_pairs_per_graph=args.max_pairs_per_graph,
            batch_size=args.batch_size,
            device_name=args.device,
            config_path=args.model_config,
            checkpoint=args.checkpoint,
            backend=args.backend,
            progress_every_graphs=args.progress_every_graphs,
        )
        plot_rq3(cfg, task, seed=args.seed)
        run_rq4(
            cfg,
            task,
            seed=args.seed,
            num_pi=args.num_pi,
            partners_per_source=args.partners_per_source,
            batch_size=args.batch_size,
            device_name=args.device,
            config_path=args.model_config,
            checkpoint=args.checkpoint,
            backend=args.backend,
            progress_every_graphs=args.progress_every_graphs,
        )
        plot_rq4(cfg, task, seed=args.seed)
    else:  # pragma: no cover
        raise ValueError(args.command)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
