"""Intervention implementation for the dissertation GRIT procedure.

The functions here implement the paper methodology on adapters that expose the
official GRIT hooks.  They are intentionally conservative: they operate on
encoded atom content after GRIT's official FeatureEncoder, keeping topology and
RRWP fixed, and they write file artifacts after every step so expensive
interventions are restartable.
"""

from __future__ import annotations

import math
import random
import re
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from graph_specialisation_metrics.method_adapters import (
    OfficialBenchmarkingGNNsGINAdapter,
    OfficialGRITAdapter,
    OfficialPyGGINAdapter,
)
from graph_specialisation_metrics.method_core import (
    EPS,
    all_pair_distances_or_compute,
    atomic_torch_save,
    bootstrap_ci,
    distance_profile,
    effective_rank,
    ensure_dir,
    graph_to_networkx,
    minimum_vertex_cut,
    r2_score,
    spearman_corr,
    shortest_path_distance_matrix,
    top_singular_share,
    write_csv,
    write_json,
)


@dataclass
class ModelRun:
    name: str
    adapter: Any
    role: str
    variant: str


MODEL_LABELS = {
    "dense_grit": "dense GRIT",
    "grit_1hop": "1-hop GRIT\n(global RRWP)",
    "grit_1hop_localrrwp": "1-hop GRIT\n(local RRWP)",
    "gin": "GIN",
}

MODEL_PLOT_ORDER = ["dense_grit", "grit_1hop_localrrwp", "grit_1hop", "ginplus", "gin", "gcn"]
MODEL_PALETTE = {
    "dense_grit": "#4c78a8",
    "grit_1hop_localrrwp": "#54a24b",
    "grit_1hop": "#f58518",
    "ginplus": "#b279a2",
    "gin": "#e45756",
    "gcn": "#72b7b2",
}


def model_label(model_name: str) -> str:
    return MODEL_LABELS.get(str(model_name), str(model_name).replace("_", " "))


def ordered_model_names(names: Sequence[str]) -> list[str]:
    preferred = ["dense_grit", "grit_1hop", "grit_1hop_localrrwp", "gin"]
    present = {str(name) for name in names}
    out = [name for name in preferred if name in present]
    out.extend(sorted(present - set(out)))
    return out


def progress(message: str) -> None:
    print(f"[intervention:{time.strftime('%H:%M:%S')}] {message}", flush=True)


def progress_interval(total: int, target_messages: int = 10) -> int:
    total = max(1, int(total))
    return max(1, total // max(1, target_messages))


def progress_graph(step: str, model: str, graph_idx: int, total: int, *, split: str = "test") -> None:
    interval = progress_interval(total)
    if graph_idx == 0 or graph_idx + 1 == total or (graph_idx + 1) % interval == 0:
        progress(f"{step} {model} {split} graph {graph_idx + 1}/{total}")


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def nanmean_or_nan(values: Sequence[Any]) -> float:
    arr = np.asarray([safe_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def stable_seed(*parts: Any, base: int = 0) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return (int(digest[:8], 16) + int(base)) % (2**32 - 1)


def graph_label(graph: Any) -> float:
    y = getattr(graph, "y", None)
    if isinstance(y, torch.Tensor) and y.numel():
        return float(y.reshape(-1)[0].detach().cpu().item())
    return float("nan")


def graph_num_nodes(graph: Any) -> int:
    if hasattr(graph, "num_nodes") and graph.num_nodes is not None:
        return int(graph.num_nodes)
    return int(graph.x.size(0))


def graph_identity(split: str, index: int, graph: Any) -> str:
    for key in ("graph_id", "idx", "id", "name"):
        if hasattr(graph, key):
            value = getattr(graph, key)
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                return f"{split}:{int(value.item())}"
            return f"{split}:{value}"
    return f"{split}:{index}"


def pyg_graph_view(graph: Any):
    from graph_specialisation_metrics.method_core import GraphBatchView

    return GraphBatchView(
        x=graph.x.detach().float().cpu() if isinstance(graph.x, torch.Tensor) else torch.as_tensor(graph.x).float(),
        edge_index=graph.edge_index.detach().long().cpu(),
        y=graph.y.detach().cpu() if isinstance(getattr(graph, "y", None), torch.Tensor) else None,
    )


def distance_matrix(graph: Any) -> torch.Tensor:
    """Return molecular hop distance, not GRIT structural/RRWP fields.

    Official GRIT data objects may contain a ``distances`` tensor used by the
    model preprocessing. The dissertation figures are explicitly by molecular
    hop distance, so the canonical source is always the molecular ``edge_index``
    when it is available.
    """
    if hasattr(graph, "edge_index") and isinstance(graph.edge_index, torch.Tensor):
        return shortest_path_distance_matrix(pyg_graph_view(graph)).detach().cpu().float()
    if hasattr(graph, "distances") and isinstance(graph.distances, torch.Tensor):
        return graph.distances.detach().cpu().float()
    return all_pair_distances_or_compute(pyg_graph_view(graph)).cpu()


def graph_distance_summary(dist: torch.Tensor) -> dict[str, float]:
    values = dist.detach().cpu().float()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {"graph_diameter": float("nan"), "graph_mean_distance": float("nan")}
    finite = finite[finite > 0]
    if finite.numel() == 0:
        return {"graph_diameter": 0.0, "graph_mean_distance": 0.0}
    return {
        "graph_diameter": float(finite.max().item()),
        "graph_mean_distance": float(finite.mean().item()),
    }


def far_pairs(dist: torch.Tensor, tau: int, *, max_pairs: Optional[int] = None, seed: int = 0) -> list[tuple[int, int]]:
    dist_cpu = dist.detach().cpu().float()
    mask = torch.isfinite(dist_cpu) & (dist_cpu > float(tau))
    if dist_cpu.dim() != 2:
        return []
    diag = torch.eye(dist_cpu.size(0), dist_cpu.size(1), dtype=torch.bool)
    mask = mask & ~diag
    pairs = [(int(i), int(j)) for i, j in mask.nonzero(as_tuple=False).tolist()]
    rng = random.Random(int(seed))
    rng.shuffle(pairs)
    return pairs[: int(max_pairs)] if max_pairs is not None else pairs


def distance_pairs(
    dist: torch.Tensor,
    *,
    min_distance: int = 2,
    max_distance: Optional[int] = None,
    max_pairs: Optional[int] = None,
    seed: int = 0,
    stratify_by_distance: bool = True,
) -> list[tuple[int, int]]:
    dist_cpu = dist.detach().cpu().float()
    if dist_cpu.dim() != 2:
        return []
    mask = torch.isfinite(dist_cpu) & (dist_cpu >= float(min_distance))
    if max_distance is not None:
        mask = mask & (dist_cpu <= float(max_distance))
    diag = torch.eye(dist_cpu.size(0), dist_cpu.size(1), dtype=torch.bool)
    mask = mask & ~diag
    pairs = [(int(i), int(j)) for i, j in mask.nonzero(as_tuple=False).tolist()]
    rng = random.Random(int(seed))
    if max_pairs is not None and stratify_by_distance and pairs:
        by_distance: dict[int, list[tuple[int, int]]] = {}
        for i, j in pairs:
            by_distance.setdefault(int(round(float(dist_cpu[i, j].item()))), []).append((i, j))
        for values in by_distance.values():
            rng.shuffle(values)
        distances = sorted(by_distance)
        rng.shuffle(distances)
        quota = max(1, int(max_pairs) // max(1, len(distances)))
        selected: list[tuple[int, int]] = []
        leftovers: list[tuple[int, int]] = []
        for distance in distances:
            values = by_distance[distance]
            selected.extend(values[:quota])
            leftovers.extend(values[quota:])
        rng.shuffle(leftovers)
        selected.extend(leftovers)
        return selected[: int(max_pairs)]
    rng.shuffle(pairs)
    return pairs[: int(max_pairs)] if max_pairs is not None else pairs


def optional_pair_limit(raw: Any, default: Optional[int]) -> Optional[int]:
    value = default if raw is None else raw
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"all", "full", "none", "null", ""}:
        return None
    limit = int(value)
    return None if limit <= 0 else limit


def split_seed_offset(split: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(split)))


def stable_int_hash(value: Any) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def node_type_signatures(graph: Any) -> Optional[list[Any]]:
    x = getattr(graph, "x", None)
    if not isinstance(x, torch.Tensor) or x.ndim == 0:
        return None
    x_cpu = x.detach().cpu()
    if x_cpu.ndim == 1:
        return [int(v.item()) if float(v.item()).is_integer() else float(v.item()) for v in x_cpu]
    if x_cpu.ndim == 2 and x_cpu.size(1) == 1:
        return [int(v.item()) if float(v.item()).is_integer() else float(v.item()) for v in x_cpu[:, 0]]
    signatures = []
    for row in x_cpu:
        values = row.tolist()
        signatures.append(tuple(int(v) if float(v).is_integer() else float(v) for v in values))
    return signatures


def deterministic_partners(
    n: int,
    source: int,
    partners: int,
    seed: int,
    *,
    type_signatures: Optional[Sequence[Any]] = None,
    require_different_type: bool = False,
) -> list[int]:
    choices = [p for p in range(int(n)) if p != int(source)]
    if require_different_type and type_signatures is not None and int(source) < len(type_signatures):
        source_type = type_signatures[int(source)]
        different = [p for p in choices if p < len(type_signatures) and type_signatures[p] != source_type]
        if different:
            choices = different
    rng = random.Random(int(seed) + int(source) * 1009)
    rng.shuffle(choices)
    return choices[: max(1, min(int(partners), len(choices)))]


def direct_fraction_value(direct: float, unclamped: float, *, min_effect_abs: float) -> tuple[float, bool]:
    effect = abs(float(unclamped))
    if not math.isfinite(effect) or effect < float(min_effect_abs):
        return float("nan"), False
    return abs(float(direct)) / effect, True


def row_is_nontrivial(row: Mapping[str, Any], *, default: bool = True) -> bool:
    value = row.get("nontrivial_effect", default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def weighted_direct_fraction(rows: Sequence[Mapping[str, Any]], *, seed: int, draws: int = 500) -> dict[str, Any]:
    clean = [
        r
        for r in rows
        if row_is_nontrivial(r)
        and math.isfinite(safe_float(r.get("direct")))
        and math.isfinite(safe_float(r.get("unclamped")))
        and abs(safe_float(r.get("unclamped"))) > EPS
    ]
    if not clean:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "pairs": 0}

    def ratio(items: Sequence[Mapping[str, Any]]) -> float:
        numerator = sum(abs(safe_float(item.get("direct"))) for item in items)
        denominator = sum(abs(safe_float(item.get("unclamped"))) for item in items)
        return float(numerator / max(denominator, EPS))

    point = ratio(clean)
    if len(clean) < 2:
        return {"mean": point, "ci_low": point, "ci_high": point, "pairs": len(clean)}
    rng = np.random.default_rng(int(seed))
    samples = []
    for _ in range(int(draws)):
        idx = rng.integers(0, len(clean), size=len(clean))
        samples.append(ratio([clean[int(i)] for i in idx]))
    lo, hi = np.percentile(np.asarray(samples, dtype=float), [2.5, 97.5])
    return {"mean": point, "ci_low": float(lo), "ci_high": float(hi), "pairs": len(clean)}


def bounded_direct_far_mass_share(
    direct: torch.Tensor,
    carriage: torch.Tensor,
    dist: torch.Tensor,
    threshold: int,
) -> dict[str, Any]:
    """Observed, bounded direct far mass for the rung-funnel summary.

    `r_nc_size_scaled_sum_estimate` is useful as a size-controlled estimate, but it
    is not a share. For the funnel, count only measured far pairs and bound each
    pair's direct mass by its original carriage mass so the displayed rung is a
    conservative surviving-mass share.
    """
    far_mask = torch.isfinite(dist) & (dist > int(threshold))
    measured_mask = far_mask & torch.isfinite(direct)
    total_pairs = int(far_mask.sum().item())
    measured_pairs = int(measured_mask.sum().item())
    total_carriage_mass = float(carriage.detach().abs().sum().item())
    if measured_pairs == 0 or total_carriage_mass <= EPS:
        return {
            "direct_far_share_bounded": float("nan"),
            "direct_far_share_raw": float("nan"),
            "direct_far_fraction_measured_unclamped": float("nan"),
            "direct_far_mass_bounded": float("nan"),
            "direct_far_mass_raw": float("nan"),
            "measured_unclamped_far_mass": float("nan"),
            "measured_far_pairs": measured_pairs,
            "total_far_pairs": total_pairs,
            "measured_pair_coverage": float(measured_pairs / total_pairs) if total_pairs else float("nan"),
        }
    direct_abs = direct.detach().abs()[measured_mask]
    carriage_abs = carriage.detach().abs()[measured_mask]
    raw_direct_mass = float(direct_abs.sum().item())
    bounded_direct_mass = float(torch.minimum(direct_abs, carriage_abs).sum().item())
    measured_unclamped_mass = float(carriage_abs.sum().item())
    return {
        "direct_far_share_bounded": bounded_direct_mass / max(total_carriage_mass, EPS),
        "direct_far_share_raw": raw_direct_mass / max(total_carriage_mass, EPS),
        "direct_far_fraction_measured_unclamped": bounded_direct_mass / max(measured_unclamped_mass, EPS),
        "direct_far_mass_bounded": bounded_direct_mass,
        "direct_far_mass_raw": raw_direct_mass,
        "measured_unclamped_far_mass": measured_unclamped_mass,
        "measured_far_pairs": measured_pairs,
        "total_far_pairs": total_pairs,
        "measured_pair_coverage": float(measured_pairs / total_pairs) if total_pairs else float("nan"),
    }


def signal_gate_enabled(config: Mapping[str, Any], step: str) -> bool:
    return bool(config.get("steps", {}).get(str(step), {}).get("signal_gate", True))


def signal_gate_quantile(config: Mapping[str, Any], step: str) -> float:
    return float(config.get("steps", {}).get(str(step), {}).get("signal_gate_quantile", 0.90))


def signal_gate_reference_floor_cap_fraction(config: Mapping[str, Any], step: str) -> float:
    """Cap global reference floors so they cannot silently remove all signal.

    The empirical 1-hop floor is useful as a noise reference, but it is a
    single absolute value pooled across graphs. On small molecules it can be
    larger than a graph's entire carriage scale, which makes every dense pair
    fail the gate and empties Step 4/5. The cap keeps the floor local to the
    current graph while preserving a small absolute minimum.
    """

    return float(config.get("steps", {}).get(str(step), {}).get("signal_gate_reference_floor_cap_fraction", 0.05))


def distance_bin_signal_floor(
    carriage: torch.Tensor,
    dist: torch.Tensor,
    carrier: int,
    source: int,
    *,
    quantile: float,
    min_floor: float,
    reference_floor: float = float("nan"),
    reference_floor_cap_fraction: float = 0.05,
) -> float:
    d = dist.detach().cpu().float()[int(carrier), int(source)]
    if not torch.isfinite(d):
        return float("inf")
    # Gate against the NOISE floor (empirical 1-hop-beyond-receptive-field level,
    # passed in as ``reference_floor``), NOT the observed carriage distribution at
    # this distance. A quantile of the signal itself rejects ~90% of real pairs by
    # construction and perversely keeps only sparse outliers, so it is not used.
    floors = [float(min_floor)]
    if math.isfinite(float(reference_floor)):
        abs_values = carriage.detach().abs().float()
        abs_values = abs_values[torch.isfinite(abs_values)]
        if abs_values.numel():
            local_cap = float(reference_floor_cap_fraction) * float(abs_values.max().item())
            floors.append(min(float(reference_floor), max(float(min_floor), local_cap)))
        else:
            floors.append(float(reference_floor))
    return max(floors)


def pair_passes_signal_gate(
    carriage: torch.Tensor,
    dist: torch.Tensor,
    carrier: int,
    source: int,
    *,
    enabled: bool,
    quantile: float,
    min_floor: float,
    reference_floor: float = float("nan"),
    reference_floor_cap_fraction: float = 0.05,
) -> tuple[bool, float, float]:
    effect = abs(float(carriage[int(carrier), int(source)].detach().cpu().item()))
    floor = distance_bin_signal_floor(
        carriage,
        dist,
        carrier,
        source,
        quantile=quantile,
        min_floor=min_floor,
        reference_floor=reference_floor,
        reference_floor_cap_fraction=reference_floor_cap_fraction,
    )
    if not enabled:
        return True, floor, effect
    return bool(math.isfinite(effect) and effect > floor), floor, effect


def model_message_passing_depth(adapter: Any) -> Optional[int]:
    try:
        model = adapter.load_model()
    except Exception:
        return None
    if hasattr(model, "n_layers"):
        try:
            return int(getattr(model, "n_layers"))
        except Exception:
            return None
    count = 0
    try:
        for module in model.modules():
            if module.__class__.__name__ == "GritTransformerLayer":
                count += 1
    except Exception:
        return None
    return count or None


def empirical_onehop_noise_floor(
    models: Sequence[ModelRun],
    artifact_root: Path,
    config: Mapping[str, Any],
    *,
    sample_graphs: int,
    split: str,
    ig_steps: int,
    seed: int,
    batched_vjp: bool,
    min_floor: float,
    quantile: float,
) -> float:
    onehop = next((m for m in models if "1hop" in m.name.lower() or "1hop" in str(m.variant).lower()), None)
    if onehop is None:
        return float(min_floor)
    depth = model_message_passing_depth(onehop.adapter)
    if depth is None or depth <= 0:
        return float(min_floor)
    graphs = select_graphs(onehop.adapter, split, sample_graphs, seed=seed)
    if not graphs:
        return float(min_floor)
    baseline = mean_encoded_baseline(onehop.adapter, select_baseline_graphs(onehop.adapter, split, config, sample_graphs, seed=seed))
    values: list[torch.Tensor] = []
    for graph_idx, graph in enumerate(graphs):
        gid = graph_identity(split, graph_idx, graph)
        dist = distance_matrix(graph)
        result = carriage_ig_cached(
            onehop,
            graph,
            baseline,
            artifact_root,
            config,
            split=split,
            graph_id=gid,
            steps=ig_steps,
            batched_vjp=batched_vjp,
        )
        mask = torch.isfinite(dist) & (dist > float(depth))
        if dist.dim() == 2:
            diag = torch.eye(dist.size(0), dist.size(1), dtype=torch.bool)
            mask = mask & ~diag
        vals = result["carriage"].detach().abs()[mask]
        vals = vals[torch.isfinite(vals)]
        if vals.numel():
            values.append(vals.float().cpu())
    if not values:
        return float(min_floor)
    joined = torch.cat(values)
    q = min(max(float(quantile), 0.0), 1.0)
    return max(float(min_floor), float(torch.quantile(joined, q).item()))


def r_nc_estimate(direct: torch.Tensor, dist: torch.Tensor, threshold: int) -> dict[str, Any]:
    far_mask = torch.isfinite(dist) & (dist > int(threshold))
    measured_mask = far_mask & torch.isfinite(direct)
    total_pairs = int(far_mask.sum().item())
    measured_pairs = int(measured_mask.sum().item())
    values = direct.detach().abs()[measured_mask]
    raw_sum = float(values.sum().item()) if values.numel() else float("nan")
    mean_abs = float(values.mean().item()) if values.numel() else float("nan")
    scaled_sum = mean_abs * float(total_pairs) if math.isfinite(mean_abs) else float("nan")
    coverage = float(measured_pairs / total_pairs) if total_pairs else float("nan")
    return {
        "r_nc": mean_abs,
        "r_nc_raw_sample_sum": raw_sum,
        "r_nc_mean_abs_sampled_pair": mean_abs,
        "r_nc_size_scaled_sum_estimate": scaled_sum,
        "r_nc_sampled_pairs": measured_pairs,
        "r_nc_total_far_pairs": total_pairs,
        "r_nc_pair_coverage": coverage,
        "r_nc_scaled_from_sample": measured_pairs != total_pairs,
        "r_nc_definition": "mean_abs_direct_carriage_per_measured_far_pair",
    }


def far_thresholds(config: Mapping[str, Any]) -> list[int]:
    values = [int(v) for v in config.get("far_thresholds", [config.get("primary_tau", 3)])]
    primary = int(config.get("primary_tau", 3))
    if primary not in values:
        values.append(primary)
    return sorted(set(values))


def carriage_ig_uses_batched_vjp(config: Mapping[str, Any]) -> bool:
    return bool(config.get("perturbation", {}).get("batched_vjp", True))


def carriage_ig_uses_readout_ig(config: Mapping[str, Any]) -> bool:
    """Whether to integrate the readout gradient along the IG path (full IG through the readout)
    instead of freezing it at the clean value."""
    return bool(config.get("perturbation", {}).get("carriage_readout_ig", False))


def swap_partner_policy(config: Mapping[str, Any]) -> str:
    return str(config.get("perturbation", {}).get("swap_partner_policy", "different_type")).strip().lower()


def baseline_sample_graphs(config: Mapping[str, Any], fallback: int) -> int:
    raw = config.get("perturbation", {}).get("baseline_sample_graphs", fallback)
    value = int(raw)
    return int(fallback) if value <= 0 else value


def select_baseline_graphs(adapter: Any, split: str, config: Mapping[str, Any], fallback: int, *, seed: int) -> list[Any]:
    return select_graphs(adapter, split, baseline_sample_graphs(config, fallback), seed=seed)


def mean_encoded_baseline(adapter: OfficialGRITAdapter, graphs: Sequence[Any], *, max_graphs: int = 32) -> torch.Tensor:
    chunks = []
    for graph in list(graphs)[: int(max_graphs)]:
        chunks.append(adapter.encoded_node_states(graph).detach().cpu())
    if not chunks:
        raise ValueError("cannot build baseline from an empty graph sample")
    mean = torch.cat(chunks, dim=0).mean(dim=0, keepdim=True)
    return mean


def expanded_baseline(encoded: torch.Tensor, mean_baseline: torch.Tensor) -> torch.Tensor:
    base = mean_baseline.to(device=encoded.device, dtype=encoded.dtype)
    return base.expand_as(encoded)


def tensor_digest(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(cpu.shape)).encode("utf-8"))
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()[:20]


def cache_safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def adapter_cache_fingerprint(adapter: Any) -> str:
    h = hashlib.sha256()
    for attr in ("checkpoint_path", "config_path"):
        path = getattr(adapter, attr, None)
        if path is None:
            continue
        p = Path(str(path))
        h.update(str(p).encode("utf-8"))
        try:
            stat = p.stat()
        except OSError:
            continue
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(stat.st_mtime_ns).encode("utf-8"))
    return h.hexdigest()[:16]


def carriage_cache_enabled(config: Mapping[str, Any]) -> bool:
    runtime = config.get("runtime", {})
    if bool(runtime.get("force_recompute_carriage", runtime.get("force_recompute", False))):
        return False
    return bool(runtime.get("reuse_intervention_carriage", True))


def carriage_cache_file(
    artifact_root: Path,
    *,
    model: str,
    split: str,
    graph_id: str,
    steps: int,
    clean_encoded: torch.Tensor,
    baseline: torch.Tensor,
    model_fingerprint: str = "",
) -> Path:
    clean_hash = tensor_digest(clean_encoded)
    baseline_hash = tensor_digest(baseline)
    model_hash = cache_safe_name(model_fingerprint or "model-unknown")
    filename = (
        f"{cache_safe_name(graph_id)}__ig{int(steps)}__"
        f"{model_hash}__"
        f"clean-{clean_hash}__base-{baseline_hash}.pt"
    )
    return artifact_root / "tensors" / "carriage_cache" / cache_safe_name(model) / cache_safe_name(split) / filename


def load_cached_carriage(
    artifact_root: Path,
    config: Mapping[str, Any],
    *,
    model: str,
    split: str,
    graph_id: str,
    steps: int,
    clean_encoded: torch.Tensor,
    baseline: torch.Tensor,
    model_fingerprint: str = "",
) -> Optional[torch.Tensor]:
    if not carriage_cache_enabled(config):
        return None
    path = carriage_cache_file(
        artifact_root,
        model=model,
        split=split,
        graph_id=graph_id,
        steps=steps,
        clean_encoded=clean_encoded,
        baseline=baseline,
        model_fingerprint=model_fingerprint,
    )
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    carriage = payload.get("carriage") if isinstance(payload, Mapping) else None
    if not isinstance(carriage, torch.Tensor):
        return None
    return carriage.detach().cpu()


def write_cached_carriage(
    artifact_root: Path,
    *,
    model: str,
    split: str,
    graph_id: str,
    steps: int,
    clean_encoded: torch.Tensor,
    baseline: torch.Tensor,
    carriage: torch.Tensor,
    model_fingerprint: str = "",
) -> None:
    path = carriage_cache_file(
        artifact_root,
        model=model,
        split=split,
        graph_id=graph_id,
        steps=steps,
        clean_encoded=clean_encoded,
        baseline=baseline,
        model_fingerprint=model_fingerprint,
    )
    atomic_torch_save(
        path,
        {
            "model": model,
            "split": split,
            "graph_id": graph_id,
            "ig_steps": int(steps),
            "model_fingerprint": model_fingerprint,
            "clean_encoded_sha256": tensor_digest(clean_encoded),
            "baseline_sha256": tensor_digest(baseline),
            "carriage": carriage.detach().cpu(),
        },
    )


def encoded_baseline(
    encoded: torch.Tensor,
    mean_baseline: torch.Tensor,
    *,
    kind: str,
) -> torch.Tensor:
    if kind == "mean_node_embedding":
        return expanded_baseline(encoded, mean_baseline)
    if kind == "zero_embedding":
        return torch.zeros_like(encoded)
    raise ValueError(f"unsupported Step 0 IG baseline {kind!r}; expected mean_node_embedding or zero_embedding")


def predict_scalar_from_encoded(adapter: OfficialGRITAdapter, graph: Any, encoded: torch.Tensor) -> torch.Tensor:
    return adapter.forward_from_encoded_content(graph, encoded, retain_grad=False).prediction.reshape(-1)[0]


def output_ig_source_endpoint(
    adapter: OfficialGRITAdapter,
    graph: Any,
    start_encoded: torch.Tensor,
    end_encoded: torch.Tensor,
    *,
    source: int,
    steps: int,
) -> float:
    """Output IG for one source along the path from ``start`` to ``end``."""

    start = start_encoded.detach().to(adapter.device)
    end = end_encoded.detach().to(adapter.device)
    delta = end - start
    total = 0.0
    for alpha_idx in range(1, int(steps) + 1):
        alpha = float(alpha_idx) / float(steps)
        point = (start + alpha * delta).detach().requires_grad_(True)
        pred = predict_scalar_from_encoded(adapter, graph, point)
        (grad,) = torch.autograd.grad(pred, point, retain_graph=False, create_graph=False)
        total += float((grad[int(source)].detach() * delta[int(source)]).sum().cpu().item()) / float(steps)
    return total


def clean_readout_state_from_baseline(
    adapter: OfficialGRITAdapter,
    graph: Any,
    mean_baseline: torch.Tensor,
    *,
    target_index: int = 0,
    baseline_override: Optional[torch.Tensor] = None,
    capture_attention: bool = False,
    capture_channels: bool = False,
    capture_layer_inputs: bool = False,
    capture_layer_outputs: bool = False,
) -> dict[str, Any]:
    clean_encoded = adapter.encoded_node_states(graph).detach()
    base = (
        baseline_override.to(device=clean_encoded.device, dtype=clean_encoded.dtype)
        if baseline_override is not None
        else expanded_baseline(clean_encoded, mean_baseline)
    )
    if tuple(base.shape) != tuple(clean_encoded.shape):
        base = base.expand_as(clean_encoded)
    clean_cache, readout_grad = adapter.readout_gradient_from_encoded_content(
        graph,
        clean_encoded.detach().clone(),
        target_index=target_index,
        capture_attention=capture_attention,
        capture_channels=capture_channels,
        capture_layer_inputs=capture_layer_inputs,
        capture_layer_outputs=capture_layer_outputs,
    )
    pred_clean = float(clean_cache.prediction.reshape(-1)[target_index].detach().cpu().item())
    return {
        "clean_encoded": clean_encoded.detach().cpu(),
        "baseline": base.detach().cpu(),
        "readout_gradient": readout_grad.detach().cpu(),
        "prediction": pred_clean,
        "clean_cache": clean_cache,
    }


def carriage_ig(
    adapter: OfficialGRITAdapter,
    graph: Any,
    mean_baseline: torch.Tensor,
    *,
    steps: int,
    target_index: int = 0,
    baseline_override: Optional[torch.Tensor] = None,
    batched_vjp: bool = True,
    readout_ig: bool = False,
    loss_label: Optional[float] = None,
    capture_attention: bool = False,
    capture_channels: bool = False,
    capture_layer_inputs: bool = False,
    capture_layer_outputs: bool = False,
    clean_state: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Compute markdown carriage C[i,j] using encoded-content IG.

    For each carrier ``i`` this integrates the gradient of ``g_i · h_i^L`` with respect to each
    encoded source ``j`` along the baseline-to-input path. With ``readout_ig=False`` (default) the
    readout gradient ``g_i`` is frozen at the clean value (a linearised readout). With
    ``readout_ig=True`` the readout gradient is re-evaluated (and detached) at every path point --
    full IG through the readout -- which makes the reconstruction ``Sum C == y_clean - y_base``
    exact instead of first-order, at the cost of one extra backward pass per IG step.
    """
    state = (
        dict(clean_state)
        if clean_state is not None
        else clean_readout_state_from_baseline(
            adapter,
            graph,
            mean_baseline,
            target_index=target_index,
            baseline_override=baseline_override,
            capture_attention=capture_attention,
            capture_channels=capture_channels,
            capture_layer_inputs=capture_layer_inputs,
            capture_layer_outputs=capture_layer_outputs,
        )
    )
    clean_encoded = state["clean_encoded"].to(adapter.device)
    base = state["baseline"].to(adapter.device)
    delta = clean_encoded - base
    clean_cache = state["clean_cache"]
    readout_grad = state["readout_gradient"].to(adapter.device)
    n = int(clean_encoded.size(0))
    carriage = clean_encoded.new_zeros((n, n))
    use_batched_vjp = bool(batched_vjp)
    batched_vjp_error: str | None = None
    def _step_readout_grad(cache: Any) -> torch.Tensor:
        # Frozen readout gradient (clean) unless readout_ig: then re-evaluate g at this path point,
        # detached so it acts as a constant cotangent for the h->e VJP below.
        if not readout_ig:
            g = readout_grad
        else:
            pred_alpha = cache.prediction.reshape(-1)[int(target_index)]
            (g_alpha,) = torch.autograd.grad(pred_alpha, cache.final_node_states, retain_graph=True, create_graph=False)
            g = g_alpha.detach()
        if loss_label is not None:
            # Loss-carriage: attribute L=|y_hat-y| instead of y_hat, i.e. project by dL/dy_hat =
            # sign(y_hat(alpha)-y). Sum C_L then telescopes to L(clean)-L(base): a beneficial
            # (loss-reducing) carriage is negative. Same IG/baseline/readout/transport axis as C(d).
            pa = float(cache.prediction.reshape(-1)[int(target_index)].detach().cpu().item())
            g = g * (1.0 if pa >= float(loss_label) else -1.0)
        return g

    for alpha_idx in range(1, int(steps) + 1):
        alpha = float(alpha_idx) / float(steps)
        point = (base + alpha * delta).detach().requires_grad_(True)
        cache = adapter.forward_from_encoded_content(graph, point, retain_grad=False)
        step_grad = _step_readout_grad(cache)
        carrier_scores = (cache.final_node_states * step_grad).sum(dim=-1)
        if use_batched_vjp:
            try:
                eye = torch.eye(n, device=carrier_scores.device, dtype=carrier_scores.dtype)
                (batched_grad,) = torch.autograd.grad(
                    carrier_scores,
                    point,
                    grad_outputs=eye,
                    is_grads_batched=True,
                    retain_graph=False,
                    create_graph=False,
                )
                carriage += torch.einsum("csd,sd->cs", batched_grad.detach(), delta) / float(steps)
                continue
            except (RuntimeError, TypeError) as exc:
                use_batched_vjp = False
                batched_vjp_error = str(exc)
                point = (base + alpha * delta).detach().requires_grad_(True)
                cache = adapter.forward_from_encoded_content(graph, point, retain_grad=False)
                step_grad = _step_readout_grad(cache)
                carrier_scores = (cache.final_node_states * step_grad).sum(dim=-1)
        for carrier in range(n):
            scalar = carrier_scores[carrier]
            (grad,) = torch.autograd.grad(scalar, point, retain_graph=carrier < n - 1, create_graph=False)
            carriage[carrier] += (grad.detach() * delta).sum(dim=-1) / float(steps)
    pred_base = float(predict_scalar_from_encoded(adapter, graph, base).detach().cpu().item())
    return {
        "carriage": carriage.detach().cpu(),
        "clean_encoded": clean_encoded.detach().cpu(),
        "baseline": base.detach().cpu(),
        "readout_gradient": readout_grad.detach().cpu(),
        "prediction": state["prediction"],
        "baseline_prediction": pred_base,
        "clean_cache": clean_cache,
        "batched_vjp_used": use_batched_vjp,
        "batched_vjp_error": batched_vjp_error,
    }


def carriage_ig_cached(
    model: ModelRun,
    graph: Any,
    mean_baseline: torch.Tensor,
    artifact_root: Path,
    config: Mapping[str, Any],
    *,
    split: str,
    graph_id: str,
    steps: int,
    target_index: int = 0,
    batched_vjp: bool = True,
    capture_attention: bool = False,
    capture_channels: bool = False,
    capture_layer_inputs: bool = False,
    capture_layer_outputs: bool = False,
) -> dict[str, Any]:
    state = clean_readout_state_from_baseline(
        model.adapter,
        graph,
        mean_baseline,
        target_index=target_index,
        capture_attention=capture_attention,
        capture_channels=capture_channels,
        capture_layer_inputs=capture_layer_inputs,
        capture_layer_outputs=capture_layer_outputs,
    )
    model_fingerprint = adapter_cache_fingerprint(model.adapter)
    readout_ig = carriage_ig_uses_readout_ig(config)
    if readout_ig:
        # Distinct cache key: readout-IG carriage differs from the frozen-g carriage.
        model_fingerprint = f"{model_fingerprint}-readoutig"
    cached = load_cached_carriage(
        artifact_root,
        config,
        model=model.name,
        split=split,
        graph_id=graph_id,
        steps=steps,
        clean_encoded=state["clean_encoded"],
        baseline=state["baseline"],
        model_fingerprint=model_fingerprint,
    )
    if cached is not None:
        return {
            "carriage": cached,
            "clean_encoded": state["clean_encoded"],
            "baseline": state["baseline"],
            "readout_gradient": state["readout_gradient"],
            "prediction": state["prediction"],
            "baseline_prediction": float("nan"),
            "clean_cache": state["clean_cache"],
            "batched_vjp_used": bool(batched_vjp),
            "batched_vjp_error": None,
            "carriage_cache_hit": True,
        }
    result = carriage_ig(
        model.adapter,
        graph,
        mean_baseline,
        steps=steps,
        target_index=target_index,
        batched_vjp=batched_vjp,
        readout_ig=readout_ig,
        capture_attention=capture_attention,
        capture_channels=capture_channels,
        capture_layer_inputs=capture_layer_inputs,
        capture_layer_outputs=capture_layer_outputs,
        clean_state=state,
    )
    result["carriage_cache_hit"] = False
    write_cached_carriage(
        artifact_root,
        model=model.name,
        split=split,
        graph_id=graph_id,
        steps=steps,
        clean_encoded=result["clean_encoded"],
        baseline=result["baseline"],
        carriage=result["carriage"],
        model_fingerprint=model_fingerprint,
    )
    return result


def carriage_swap(
    adapter: OfficialGRITAdapter,
    graph: Any,
    clean_encoded: torch.Tensor,
    readout_grad: torch.Tensor,
    *,
    partners: int,
    seed: int,
    partner_policy: str = "different_type",
) -> torch.Tensor:
    n = int(clean_encoded.size(0))
    if int(partners) <= 0:
        return clean_encoded.new_zeros((n, n))
    clean_cache = adapter.forward_from_encoded_content(graph, clean_encoded.to(adapter.device), retain_grad=False)
    clean_h = clean_cache.final_node_states.detach()
    out = clean_encoded.new_zeros((n, n), device=adapter.device)
    encoded_device = clean_encoded.to(adapter.device)
    grad_device = readout_grad.to(adapter.device)
    type_signatures = node_type_signatures(graph)
    require_different_type = str(partner_policy).strip().lower() in {"different_type", "different-type", "different_atom_type"}
    for source in range(n):
        choices = deterministic_partners(
            n,
            source,
            partners,
            seed,
            type_signatures=type_signatures,
            require_different_type=require_different_type,
        )
        sum_delta_h = clean_h.new_zeros(clean_h.shape)
        for partner in choices:
            pert = encoded_device.detach().clone()
            pert[source] = encoded_device[partner]
            cache = adapter.forward_from_encoded_content(graph, pert, retain_grad=False)
            delta_h = cache.final_node_states.detach() - clean_h
            sum_delta_h += delta_h
        mean_delta_h = sum_delta_h / float(len(choices))
        out[:, source] = (mean_delta_h * grad_device).sum(dim=-1)
    return out.detach().cpu()


def attention_profiles(cache: Any, dist: torch.Tensor, model: str) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    rows: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}
    if not cache.attention:
        return rows, tensors
    first = cache.attention[0].detach().cpu().mean(dim=0)
    last = cache.attention[-1].detach().cpu().mean(dim=0)
    tensors["attention_first"] = first
    tensors["attention_last"] = last
    for row in distance_profile(last.abs(), dist):
        rows.append({"model": model, "quantity": "attention_last", **row})
    return rows, tensors


def attention_graph(cache: Any, *, aggregation: str = "mean") -> Optional[torch.Tensor]:
    """Attention graph (El, Choudhury, Lio & Joshi, 2025, arXiv:2502.12352): aggregate the
    per-layer, head-averaged attention of a graph transformer into a single [receiver, source]
    information-flow matrix over the input nodes.

    Works for any adapter that captures per-layer attention (GRIT dense and masked 1-hop; the 1-hop
    graph is bond-local by construction, so its far-range entries are ~0). ``aggregation``:
      * ``mean`` (default) / ``sum`` / ``max`` -- per-edge statistic across layers (robust, no
        rollout heuristics);
      * ``rollout`` -- Abnar & Zuidema multi-hop flow: product of row-normalised (0.5*A + 0.5*I)
        across layers, modelling information propagation through the stack.
    Returns [N,N] with entry [i,j] = aggregated attention flow from source j to receiver i, or
    ``None`` if no attention was captured.
    """
    layers = list(getattr(cache, "attention", None) or [])
    if not layers:
        return None
    mats: list[torch.Tensor] = []
    for layer in layers:
        t = layer.detach().cpu().float()
        if t.dim() == 3:  # [heads, N, N] -> head-average
            t = t.mean(dim=0)
        if t.dim() != 2 or t.size(0) != t.size(1):
            return None
        mats.append(t)
    if any(m.shape != mats[0].shape for m in mats):
        return None
    n = int(mats[0].size(0))
    agg = str(aggregation).lower()
    if agg in ("mean", "sum", "max"):
        stack = torch.stack(mats, dim=0)
        if agg == "mean":
            return stack.mean(dim=0)
        if agg == "sum":
            return stack.sum(dim=0)
        return stack.amax(dim=0)
    if agg == "rollout":
        eye = torch.eye(n)
        g: Optional[torch.Tensor] = None
        for a in mats:
            m = 0.5 * a + 0.5 * eye
            m = m / m.sum(dim=1, keepdim=True).clamp_min(EPS)
            g = m if g is None else m @ g
        return g
    raise ValueError(f"unknown attention-graph aggregation {aggregation!r}; expected mean/sum/max/rollout")


def sparse_attention_layer_weights(cache: Any, layer_idx: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Return ``(edge_index, abs_attention_by_edge_head)`` for one layer if available."""

    extras = cache.extras or {}
    edges = list(extras.get("attention_edges", []) or [])
    weights = list(extras.get("attention_edge_weights", []) or [])
    if layer_idx >= len(edges) or layer_idx >= len(weights):
        return None
    edge_index = edges[layer_idx].detach().cpu().long()
    values = weights[layer_idx].detach().abs().cpu().float()
    if values.dim() == 3 and values.size(-1) == 1:
        values = values.squeeze(-1)
    if values.dim() == 1:
        values = values.unsqueeze(-1)
    if values.dim() != 2:
        raise RuntimeError(f"attention layer {layer_idx} has unsupported sparse weight shape {tuple(values.shape)}")
    if int(values.size(0)) != int(edge_index.size(1)):
        raise RuntimeError(
            f"attention layer {layer_idx} has {values.size(0)} attention rows but "
            f"{edge_index.size(1)} sparse edges"
        )
    return edge_index, values


def attention_mean_distance_rows(cache: Any, dist: torch.Tensor, model: str, graph_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dist_cpu = dist.detach().cpu().float()
    layer_count = max(len(list(cache.attention or [])), len(list((cache.extras or {}).get("attention_edges", []) or [])))
    for layer_idx in range(layer_count):
        sparse = sparse_attention_layer_weights(cache, layer_idx)
        if sparse is not None:
            edge_index, values = sparse
            src = edge_index[0].long()
            dst = edge_index[1].long()
            edge_dist = dist_cpu[dst, src]
            finite_edges = torch.isfinite(edge_dist)
            values = values * finite_edges.float().unsqueeze(-1)
            # Average first over each receiver/head query, then over heads.
            by_head = values.t().contiguous()
            heads = by_head.size(0)
            num_nodes = int(dist_cpu.size(0))
            dst_index = dst.unsqueeze(0).expand(heads, -1)
            denom = by_head.new_zeros((heads, num_nodes))
            numerator = by_head.new_zeros((heads, num_nodes))
            denom.scatter_add_(1, dst_index, by_head)
            numerator.scatter_add_(1, dst_index, by_head * edge_dist.nan_to_num(0.0).unsqueeze(0))
            valid = denom > EPS
            mean_distance = float((numerator[valid] / denom[valid]).mean().item()) if bool(valid.any()) else float("nan")
            head_query_count = int(valid.sum().item())
        else:
            matrix = list(cache.attention or [])[layer_idx]
            weights = matrix.detach().abs().cpu().float()
            finite = torch.isfinite(dist_cpu)
            if weights.dim() == 2:
                weights = weights.unsqueeze(0)
            if weights.dim() != 3 or tuple(weights.shape[-2:]) != tuple(dist_cpu.shape):
                raise RuntimeError(
                    f"attention mean-distance requires attention shaped [heads,n,n]; "
                    f"layer {layer_idx} has shape {tuple(weights.shape)} for distance shape {tuple(dist_cpu.shape)}"
                )
            masked_weights = weights * finite.unsqueeze(0)
            denom = masked_weights.sum(dim=-1)
            numerator = (masked_weights * dist_cpu.unsqueeze(0)).sum(dim=-1)
            valid = denom > EPS
            mean_distance = float((numerator[valid] / denom[valid]).mean().item()) if bool(valid.any()) else float("nan")
            head_query_count = int(valid.sum().item())
        rows.append(
            {
                "model": model,
                "graph_id": graph_id,
                "layer": layer_idx,
                "mean_attention_distance": mean_distance,
                "head_query_count": head_query_count,
            }
        )
    return rows


def attention_support_audit_rows(
    cache: Any,
    dist: torch.Tensor,
    model: str,
    graph_id: str,
    *,
    expected_max_direct_distance: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Audit the sparse support used by each captured GRIT attention layer.

    For the 1-hop control, a single attention layer should only contain
    self-pairs and molecular-neighbour pairs.
    """

    attention = list(cache.attention or [])
    edges = list((cache.extras or {}).get("attention_edges", []))
    rows: list[dict[str, Any]] = []
    dist_cpu = dist.detach().cpu()
    far_mask_gt1 = torch.isfinite(dist_cpu) & (dist_cpu > 1)
    layer_count = max(len(attention), len(edges))
    for layer_idx in range(layer_count):
        sparse = sparse_attention_layer_weights(cache, layer_idx)
        if sparse is not None:
            edge_index, values = sparse
            edge_mass = values.reshape(values.size(0), -1).sum(dim=-1)
            src = edge_index[0].long()
            dst = edge_index[1].long()
            edge_dist = dist_cpu[dst, src]
            finite = torch.isfinite(edge_dist)
            support = finite & (edge_mass > EPS)
            total_mass = float(edge_mass[finite].sum().item()) if bool(finite.any()) else 0.0
            far_mass_gt1 = float(edge_mass[finite & (edge_dist > 1)].sum().item() / max(total_mass, EPS))
            max_edge_distance = float(edge_dist[support].max().item()) if bool(support.any()) else float("nan")
            edge_count = int(support.sum().item())
            far_edge_count_gt1 = int((support & (edge_dist > 1)).sum().item())
            expected_violation_edges = (
                int((support & (edge_dist > int(expected_max_direct_distance))).sum().item())
                if expected_max_direct_distance is not None
                else 0
            )
        else:
            matrix = attention[layer_idx]
            mat = matrix.detach().abs().cpu()
            if mat.dim() == 3:
                mat_for_mass = mat.sum(dim=0)
            elif mat.dim() == 2:
                mat_for_mass = mat
            else:
                raise RuntimeError(f"attention layer {layer_idx} has unsupported shape {tuple(mat.shape)}")
            total_mass = float(mat_for_mass.sum().item())
            far_mass_gt1 = float(mat_for_mass[far_mask_gt1].sum().item() / max(total_mass, EPS))
            support = mat_for_mass > EPS
            support_dist = dist_cpu[support]
            finite = torch.isfinite(support_dist)
            max_edge_distance = float(support_dist[finite].max().item()) if bool(finite.any()) else float("nan")
            edge_count = int(support.sum().item())
            far_edge_count_gt1 = int(((support_dist > 1) & finite).sum().item())
            expected_violation_edges = (
                int(((support_dist > int(expected_max_direct_distance)) & finite).sum().item())
                if expected_max_direct_distance is not None
                else 0
            )
        rows.append(
            {
                "model": model,
                "graph_id": graph_id,
                "layer": layer_idx,
                "edge_count": edge_count,
                "max_direct_attention_distance": max_edge_distance,
                "direct_edges_distance_gt1": far_edge_count_gt1,
                "direct_attention_mass_distance_gt1": far_mass_gt1,
                "expected_max_direct_distance": "" if expected_max_direct_distance is None else int(expected_max_direct_distance),
                "expected_distance_violating_edges": expected_violation_edges,
                "passes_expected_direct_support": expected_violation_edges == 0,
            }
        )
    return rows


def attention_support_failures(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        r
        for r in rows
        if str(r.get("expected_max_direct_distance", "")) != ""
        and math.isfinite(safe_float(r.get("expected_distance_violating_edges", 0)))
        and safe_float(r.get("expected_distance_violating_edges", 0)) > 0
    ]


def assert_attention_support_audit(rows: Sequence[Mapping[str, Any]]) -> None:
    failures = attention_support_failures(rows)
    if failures:
        first = failures[0]
        raise RuntimeError(
            "1-hop GRIT locality audit failed: direct attention support contains "
            f"distance>{first.get('expected_max_direct_distance')} pairs. "
            f"First failure: model={first.get('model')} graph={first.get('graph_id')} "
            f"layer={first.get('layer')} violating_edges={first.get('expected_distance_violating_edges')} "
            f"max_direct_distance={first.get('max_direct_attention_distance')}. "
            "This means Step 2 attention plots cannot be interpreted as a 1-hop control until "
            "the checkpoint/config or attention extraction is fixed."
        )


def attention_faithfulness_rows(
    model: str,
    graph_id: str,
    attention_tensors: Mapping[str, torch.Tensor],
    carriage: torch.Tensor,
    dist: torch.Tensor,
    tau: int,
    *,
    carriage_estimator: str = "ig",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    c = carriage.abs().reshape(-1).numpy()
    d = dist.reshape(-1).numpy()
    finite = np.isfinite(d)
    for quantity, tensor in attention_tensors.items():
        if quantity not in {"attention_first", "attention_last"}:
            continue
        a = tensor.abs().reshape(-1).numpy()
        far = finite & (d > tau)
        rows.append(
            {
                "model": model,
                "graph_id": graph_id,
                "carriage_estimator": carriage_estimator,
                "attention_quantity": quantity,
                "distance_bin": "overall",
                "spearman": spearman_corr(a[finite], c[finite]),
                "attention_far_mass": far_mass(tensor, dist, tau),
                "carriage_far_mass": far_mass(carriage, dist, tau),
            }
        )
        rows.append(
            {
                "model": model,
                "graph_id": graph_id,
                "carriage_estimator": carriage_estimator,
                "attention_quantity": quantity,
                "distance_bin": f">{tau}",
                "spearman": spearman_corr(a[far], c[far]) if far.any() else float("nan"),
                "attention_far_mass": far_mass(tensor, dist, tau),
                "carriage_far_mass": far_mass(carriage, dist, tau),
            }
        )
        for distance in sorted({int(v) for v in d[finite]}):
            mask = finite & (d == distance)
            rows.append(
                {
                    "model": model,
                    "graph_id": graph_id,
                    "carriage_estimator": carriage_estimator,
                    "attention_quantity": quantity,
                    "distance_bin": distance,
                    "spearman": spearman_corr(a[mask], c[mask]) if mask.any() else float("nan"),
                    "attention_far_mass": far_mass(tensor, dist, tau),
                    "carriage_far_mass": far_mass(carriage, dist, tau),
                }
            )
    return rows


def _attention_erasure_mask_from_ranking(
    *,
    edge_index: torch.Tensor,
    dist: torch.Tensor,
    carriage: torch.Tensor,
    attention_values: torch.Tensor,
    fraction: float,
    tau: int,
    region: str,
    ranker: str,
    seed: int,
) -> tuple[torch.Tensor, int, int]:
    """Build a dense [dst,src] mask for removing ranked sparse attention edges."""

    dist_cpu = dist.detach().cpu().float()
    src = edge_index[0].detach().cpu().long()
    dst = edge_index[1].detach().cpu().long()
    edge_dist = dist_cpu[dst, src]
    finite = torch.isfinite(edge_dist)
    if region == "far":
        candidates = finite & (edge_dist > float(tau))
    elif region == "near":
        candidates = finite & (edge_dist > 0.0) & (edge_dist <= float(tau))
    else:
        raise ValueError(f"unknown attention-erasure region {region!r}")
    candidate_indices = candidates.nonzero(as_tuple=False).flatten()
    available = int(candidate_indices.numel())
    remove_count = int(round(max(0.0, min(1.0, float(fraction))) * available))
    dense_mask = torch.zeros_like(dist_cpu, dtype=torch.bool)
    if available == 0 or remove_count <= 0:
        return dense_mask, available, 0
    remove_count = min(remove_count, available)
    if ranker == "attention":
        values = attention_values.detach().abs().cpu().float()
        if values.dim() == 2:
            score = values.mean(dim=1)
        elif values.dim() == 1:
            score = values
        else:
            raise RuntimeError(f"unsupported attention values for erasure ranking: {tuple(values.shape)}")
        ranking_scores = score[candidate_indices]
        order = torch.argsort(ranking_scores, descending=True, stable=True)
        selected = candidate_indices[order[:remove_count]]
    elif ranker == "carriage":
        c = carriage.detach().abs().cpu().float()
        ranking_scores = c[dst[candidate_indices], src[candidate_indices]]
        order = torch.argsort(ranking_scores, descending=True, stable=True)
        selected = candidate_indices[order[:remove_count]]
    elif ranker == "random":
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        order = torch.randperm(available, generator=generator)
        selected = candidate_indices[order[:remove_count]]
    else:
        raise ValueError(f"unknown attention-erasure ranker {ranker!r}")
    dense_mask[dst[selected], src[selected]] = True
    return dense_mask, available, int(selected.numel())


def attention_erasure_rows(
    model: ModelRun,
    graph: Any,
    graph_id: str,
    cache: Any,
    carriage: torch.Tensor,
    dist: torch.Tensor,
    *,
    tau: int,
    fractions: Sequence[float],
    include_near_control: bool,
    seed: int,
) -> list[dict[str, Any]]:
    if not hasattr(model.adapter, "forward_with_attention_erasure"):
        return []
    edges = list((cache.extras or {}).get("attention_edges", []) or [])
    if edges:
        last_layer = len(edges) - 1
    else:
        last_layer = len(list(cache.attention or [])) - 1
    if last_layer < 0:
        return []
    sparse = sparse_attention_layer_weights(cache, last_layer)
    if sparse is None:
        return []
    edge_index, attention_values = sparse
    clean_pred = float(cache.prediction.reshape(-1)[0].detach().cpu().item())
    rows: list[dict[str, Any]] = []
    regions = ["far"] + (["near"] if include_near_control else [])
    for region in regions:
        for fraction in fractions:
            for ranker in ("attention", "carriage", "random"):
                mask, available, removed = _attention_erasure_mask_from_ranking(
                    edge_index=edge_index,
                    dist=dist,
                    carriage=carriage,
                    attention_values=attention_values,
                    fraction=float(fraction),
                    tau=tau,
                    region=region,
                    ranker=ranker,
                    seed=seed + 7919 * stable_int_hash(graph_id) + 101 * int(round(float(fraction) * 1000)),
                )
                if available == 0:
                    erased_pred = float("nan")
                    delta = float("nan")
                elif removed == 0:
                    erased_pred = clean_pred
                    delta = 0.0
                else:
                    erased = model.adapter.forward_with_attention_erasure(graph, {int(last_layer): mask})
                    erased_pred = float(erased.prediction.reshape(-1)[0].detach().cpu().item())
                    delta = abs(erased_pred - clean_pred)
                rows.append(
                    {
                        "model": model.name,
                        "graph_id": graph_id,
                        "attention_layer": last_layer,
                        "region": region,
                        "ranker": ranker,
                        "tau": tau,
                        "fraction_removed": float(fraction),
                        "edges_available": available,
                        "edges_removed": removed,
                        "prediction_clean": clean_pred,
                        "prediction_erased": erased_pred,
                        "abs_delta_prediction": delta,
                    }
                )
    return rows


def _clone_graph_corrupt_sources(graph: Any, sources: Sequence[int], rng: random.Random) -> Optional[Any]:
    """Clone ``graph`` with the raw content of every node in ``sources`` swapped for a donor node's
    (drawn from the non-corrupted set). Topology is untouched, so RRWP recomputes from the same
    structure -> a pure, routing-agnostic content ablation of those source nodes.
    """
    x = getattr(graph, "x", None)
    if not isinstance(x, torch.Tensor) or not hasattr(graph, "clone"):
        return None
    n = int(x.size(0))
    src_set = {int(s) for s in sources}
    donor_pool = [k for k in range(n) if k not in src_set] or list(range(n))
    try:
        clone = graph.clone()
    except Exception:
        return None
    base = x.detach()
    new_x = base.clone()
    for j in src_set:
        donor = donor_pool[rng.randrange(len(donor_pool))]
        new_x[j] = base[donor]
    clone.x = new_x
    return clone


def attention_graph_vs_carriage_ablation_rows(
    model: ModelRun,
    graph: Any,
    graph_id: str,
    cache: Any,
    c_ig: torch.Tensor,
    dist: torch.Tensor,
    *,
    tau: int,
    fractions: Sequence[float],
    include_near_control: bool,
    aggregation: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Fair attention-vs-carriage comparison: rank *source nodes* by attention-graph importance vs
    carriage importance vs random, ablate the top fraction at the INPUT (content corruption), and
    record task MAE against the true label.

    The ablation modality is content (routing-agnostic), so it is not rigged toward attention the
    way attention-edge erasure is; the attention side uses the whole-network attention graph (El et
    al. 2025) rather than one layer, matching carriage's global, all-layers nature. All selectors
    ablate the SAME number of source nodes at each fraction, so the budget is controlled.
    """
    if not hasattr(model.adapter, "forward_minimal") or not isinstance(getattr(graph, "x", None), torch.Tensor):
        return []
    graph_attn = attention_graph(cache, aggregation=aggregation)
    if graph_attn is None:
        return []
    n = min(int(graph_attn.size(0)), int(c_ig.size(0)), int(dist.size(0)))
    if n < 2:
        return []
    g_attn = graph_attn[:n, :n].detach().abs().cpu()
    carr = c_ig[:n, :n].detach().abs().cpu()
    d = dist[:n, :n].detach().cpu()
    y = graph_label(graph)
    if not math.isfinite(y):
        return []
    try:
        clean_pred = float(model.adapter.forward_minimal(graph).prediction.reshape(-1)[0].detach().cpu().item())
    except Exception:
        return []
    clean_mae = abs(clean_pred - y)
    rng = random.Random(f"{seed}:{graph_id}")
    rows: list[dict[str, Any]] = []
    regions = ["far"] + (["near"] if include_near_control else [])
    for region in regions:
        if region == "far":
            region_mask = torch.isfinite(d) & (d > float(tau))
        else:
            region_mask = torch.isfinite(d) & (d > 0.0) & (d <= float(tau))
        region_f = region_mask.float()
        att_src = (g_attn * region_f).sum(dim=0)  # sum over receivers i -> per-source importance
        carr_src = (carr * region_f).sum(dim=0)
        candidates = torch.nonzero(region_mask.any(dim=0), as_tuple=False).flatten().tolist()
        if len(candidates) < 2:
            continue
        rankings = {
            "attention_graph": sorted(candidates, key=lambda j: float(att_src[int(j)]), reverse=True),
            "carriage": sorted(candidates, key=lambda j: float(carr_src[int(j)]), reverse=True),
        }
        random_order = list(candidates)
        rng.shuffle(random_order)
        rankings["random"] = random_order
        for fraction in fractions:
            k = int(round(max(0.0, min(1.0, float(fraction))) * len(candidates)))
            for selector, order in rankings.items():
                if k <= 0:
                    ablated_pred = clean_pred
                    task_mae = clean_mae
                    n_src = 0
                else:
                    corrupted = _clone_graph_corrupt_sources(graph, order[:k], rng)
                    if corrupted is None:
                        continue
                    try:
                        ablated_pred = float(model.adapter.forward_minimal(corrupted).prediction.reshape(-1)[0].detach().cpu().item())
                    except Exception:
                        continue
                    task_mae = abs(ablated_pred - y)
                    n_src = int(k)
                rows.append(
                    {
                        "model": model.name,
                        "graph_id": graph_id,
                        "region": region,
                        "selector": selector,
                        "aggregation": aggregation,
                        "fraction_ablated": float(fraction),
                        "sources_ablated": n_src,
                        "candidate_sources": len(candidates),
                        "prediction_clean": clean_pred,
                        "prediction_ablated": ablated_pred,
                        "clean_mae": clean_mae,
                        "task_mae": task_mae,
                        "delta_mae": task_mae - clean_mae,
                    }
                )
    return rows


def carriage_profile_rows(model: str, graph_id: str, carriage: torch.Tensor, dist: torch.Tensor, quantity: str) -> list[dict[str, Any]]:
    rows = []
    for row in distance_profile(carriage.abs(), dist):
        rows.append({"model": model, "graph_id": graph_id, "quantity": quantity, **row})
    return rows


def far_mass(matrix: torch.Tensor, dist: torch.Tensor, tau: int) -> float:
    mask = torch.isfinite(dist) & (dist > float(tau))
    total = float(matrix.detach().abs().sum().item())
    return float(matrix.detach().abs()[mask].sum().item() / max(total, EPS))


def select_graphs(adapter: Any, split: str, sample_graphs: int, *, seed: int = 0) -> list[Any]:
    graphs = adapter.load_zinc_split(split, limit=None)
    if int(sample_graphs) <= 0 or int(sample_graphs) >= len(graphs):
        return list(graphs)
    indices = list(range(len(graphs)))
    rng = random.Random(int(seed) + split_seed_offset(split))
    rng.shuffle(indices)
    selected = sorted(indices[: int(sample_graphs)])
    return [graphs[idx] for idx in selected]


def instantiate_official_models(config: Mapping[str, Any], discovery: Sequence[Mapping[str, Any]]) -> list[ModelRun]:
    out: list[ModelRun] = []
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(config.get("seeds", [0])[0])
    runtime_cfg = config.get("runtime", {}) if isinstance(config.get("runtime", {}), Mapping) else {}
    skip_failed_optional = bool(runtime_cfg.get("skip_failed_optional_adapters", True))
    for entry in discovery:
        if not entry.get("checkpoint_candidates") or not entry.get("config_candidates"):
            continue
        name = str(entry["model"])
        model_cfg = config["models"][name]
        adapter_kind = str(entry.get("adapter", model_cfg.get("adapter", ""))).strip().lower()
        model_device = str(model_cfg.get("device", device))
        role = str(model_cfg.get("role", ""))
        is_optional_reference = any(token in role.lower() for token in ("reference", "validation")) or name.lower() in {"gin", "gcn"}
        try:
            if adapter_kind == "official_grit":
                adapter = OfficialGRITAdapter(
                    repo_path=Path(str(model_cfg.get("repo_path", "external/GRIT"))),
                    config_path=Path(str(model_cfg.get("config_path") or entry["config_candidates"][0])),
                    checkpoint_path=Path(str(model_cfg.get("checkpoint_path") or entry["checkpoint_candidates"][0])),
                    variant=str(model_cfg.get("variant", name)),
                    official_commit=str(model_cfg.get("official_commit", "")) or None,
                    dataset_dir=Path(str(model_cfg["dataset_dir"])) if model_cfg.get("dataset_dir") else None,
                    device=model_device,
                    seed=seed,
                )
            elif adapter_kind in {"pyg_gin", "official_pyg_gin"}:
                adapter = OfficialPyGGINAdapter(
                    config_path=Path(str(model_cfg.get("config_path") or entry["config_candidates"][0])),
                    checkpoint_path=Path(str(model_cfg.get("checkpoint_path") or entry["checkpoint_candidates"][0])),
                    dataset_dir=Path(str(model_cfg["dataset_dir"])) if model_cfg.get("dataset_dir") else None,
                    device=model_device,
                    seed=seed,
                )
            elif adapter_kind in {"benchmarking_gnns_gin", "official_benchmarking_gnns_gin", "official_dgl_gin"}:
                adapter = OfficialBenchmarkingGNNsGINAdapter(
                    repo_path=Path(str(model_cfg.get("repo_path", "external/benchmarking-gnns"))),
                    config_path=Path(str(model_cfg.get("config_path") or entry["config_candidates"][0])),
                    checkpoint_path=Path(str(model_cfg.get("checkpoint_path") or entry["checkpoint_candidates"][0])),
                    dataset_dir=Path(str(model_cfg["dataset_dir"])) if model_cfg.get("dataset_dir") else None,
                    device=model_device,
                    seed=seed,
                    official_commit=str(model_cfg.get("official_commit", "")) or None,
                )
            else:
                continue
            if is_optional_reference and bool(model_cfg.get("validate_on_load", True)):
                _ = adapter.parameter_count()
                if adapter_kind in {"benchmarking_gnns_gin", "official_benchmarking_gnns_gin", "official_dgl_gin"}:
                    smoke_graphs = adapter.load_zinc_split("test", limit=1)
                    if smoke_graphs:
                        _ = adapter.forward(smoke_graphs[0])
        except Exception as exc:
            if is_optional_reference and skip_failed_optional:
                progress(f"skipping optional reference adapter {name}: {type(exc).__name__}: {exc}")
                continue
            raise
        out.append(ModelRun(name=name, adapter=adapter, role=role, variant=str(model_cfg.get("variant", ""))))
    return out


def render_distance_profile(
    rows: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    filename: str,
    title: str,
    *,
    ylabel: str = "Share of own mass by distance (each series sums to 1)",
    dpi: int = 180,
) -> None:
    numeric_rows = [r for r in rows if math.isfinite(safe_float(r.get("distance"))) and math.isfinite(safe_float(r.get("share")))]
    if not numeric_rows:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    grouped_values: dict[tuple[str, str, int], list[float]] = {}
    for row in numeric_rows:
        key = (str(row.get("model")), str(row.get("quantity")), int(safe_float(row.get("distance"))))
        grouped_values.setdefault(key, []).append(safe_float(row.get("share")))
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    for (model, quantity, distance), values in grouped_values.items():
        mean, lo, hi = bootstrap_ci(values, seed=17 + distance, draws=500)
        grouped.setdefault((model, quantity), []).append({"distance": float(distance), "share": mean, "lo": lo, "hi": hi})
    for (model, quantity), items in sorted(grouped.items()):
        items = sorted(items, key=lambda r: safe_float(r.get("distance")))
        x = np.asarray([safe_float(r.get("distance")) for r in items], dtype=float)
        y = np.asarray([safe_float(r.get("share")) for r in items], dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.5,
            label=f"{model} {quantity}",
        )
        lo = np.asarray([safe_float(r.get("lo")) for r in items], dtype=float)
        hi = np.asarray([safe_float(r.get("hi")) for r in items], dtype=float)
        if np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.fill_between(x, lo, hi, alpha=0.12)
    ax.set_title(title)
    ax.set_xlabel("Molecular hop distance")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / f"{filename}.png", dpi=dpi)
    fig.savefig(figures / f"{filename}.pdf")
    plt.close(fig)


def render_bar(
    rows: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    filename: str,
    title: str,
    *,
    x_key: str,
    y_key: str,
    ylabel: str,
    dpi: int = 180,
) -> None:
    clean = [r for r in rows if math.isfinite(safe_float(r.get(y_key)))]
    if not clean:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    labels = [str(r.get(x_key)) for r in clean]
    values = [safe_float(r.get(y_key)) for r in clean]
    ax.bar(labels, values, color="#4c78a8")
    ax.set_title(title)
    ax.set_xlabel(x_key.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / f"{filename}.png", dpi=dpi)
    fig.savefig(figures / f"{filename}.pdf")
    plt.close(fig)


def run_step0(models: Sequence[ModelRun], artifact_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    progress("Step 0 start: measurement-model validation")
    cfg = config["steps"]["0"]
    sample_graphs = int(cfg.get("sample_graphs", 200))
    ig_steps = int(config["perturbation"].get("ig_steps", 32))
    swap_partners = int(config["perturbation"].get("swap_partners", 8))
    batched_vjp = carriage_ig_uses_batched_vjp(config)
    partner_policy = swap_partner_policy(config)
    seed = int(config.get("seeds", [0])[0])
    dpi = int(config["figures"]["dpi"])
    recon_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    sweep_summary_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    matched_summary_rows: list[dict[str, Any]] = []
    pair_agreement_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    tensors: dict[str, Any] = {}
    for model in models:
        graphs = select_graphs(model.adapter, "test", sample_graphs, seed=seed)
        progress(f"Step 0 {model.name}: selected {len(graphs)} test graph(s), IG steps={ig_steps}, swap_partners={swap_partners}")
        baseline = mean_encoded_baseline(model.adapter, select_baseline_graphs(model.adapter, "test", config, sample_graphs, seed=seed))
        pred_vals: list[float] = []
        measured_vals: list[float] = []
        cache_hits = 0
        for graph_idx, graph in enumerate(graphs):
            progress_graph("Step 0", model.name, graph_idx, len(graphs))
            gid = graph_identity("test", graph_idx, graph)
            dist = distance_matrix(graph)
            result = carriage_ig_cached(
                model,
                graph,
                baseline,
                artifact_root,
                config,
                split="test",
                graph_id=gid,
                steps=ig_steps,
                batched_vjp=batched_vjp,
            )
            cache_hits += int(bool(result.get("carriage_cache_hit")))
            c_ig = result["carriage"]
            encoded = result["clean_encoded"].to(model.adapter.device)
            base = result["baseline"].to(model.adapter.device)
            clean_cache = result["clean_cache"]
            clean_h = clean_cache.final_node_states.detach()
            readout_grad = result["readout_gradient"].to(model.adapter.device)
            clean_pred = float(result["prediction"])
            c_swap = carriage_swap(
                model.adapter,
                graph,
                result["clean_encoded"],
                result["readout_gradient"],
                partners=swap_partners,
                seed=seed + graph_idx,
                partner_policy=partner_policy,
            )
            tensors[f"step0/{model.name}/{gid}/carriage_ig"] = c_ig
            tensors[f"step0/{model.name}/{gid}/carriage_swap"] = c_swap
            pair_agreement_rows.extend(pairwise_ig_swap_rows(model.name, gid, c_ig, c_swap, dist))
            for source in range(c_ig.size(1)):
                pert = encoded.detach().clone()
                pert[source] = base[source]
                pred_pert = float(predict_scalar_from_encoded(model.adapter, graph, pert).detach().cpu().item())
                measured = clean_pred - pred_pert
                predicted = float(c_ig[:, source].sum().item())
                residual_rows.extend(
                    step0_residual_diagnostic_rows(
                        model=model.name,
                        graph_id=gid,
                        dist=dist,
                        carriage=c_ig,
                        source=source,
                        predicted=predicted,
                        measured=measured,
                        tau=int(config.get("primary_tau", 3)),
                    )
                )
                recon_rows.append(
                    {
                        "model": model.name,
                        "graph_id": gid,
                        "source": source,
                        "partner": "",
                        "perturbation": "ig_baseline_replacement",
                        "predicted_delta": predicted,
                        "measured_delta": measured,
                    }
                )
                pred_vals.append(predicted)
                measured_vals.append(measured)
                type_signatures = node_type_signatures(graph)
                for partner in deterministic_partners(
                    c_ig.size(1),
                    source,
                    swap_partners,
                    seed + graph_idx,
                    type_signatures=type_signatures,
                    require_different_type=partner_policy in {"different_type", "different-type", "different_atom_type"},
                ):
                    swap_pert = encoded.detach().clone()
                    swap_pert[source] = encoded[int(partner)]
                    swap_cache = model.adapter.forward_from_encoded_content(graph, swap_pert, retain_grad=False)
                    delta_h = swap_cache.final_node_states.detach() - clean_h
                    predicted_swap = float((delta_h * readout_grad).sum().detach().cpu().item())
                    measured_swap = float(swap_cache.prediction.reshape(-1)[0].detach().cpu().item()) - clean_pred
                    recon_rows.append(
                        {
                            "model": model.name,
                            "graph_id": gid,
                            "source": source,
                            "partner": int(partner),
                            "perturbation": "finite_content_swap",
                            "predicted_delta": predicted_swap,
                            "measured_delta": measured_swap,
                        }
                )
            profile_rows.extend(carriage_profile_rows(model.name, gid, c_ig, dist, "ig_carriage"))
            profile_rows.extend(carriage_profile_rows(model.name, gid, c_swap, dist, "swap_carriage"))
        model_sweep_rows, model_sweep_summary = step0_ig_step_sweep(model, graphs, baseline, cfg, seed=seed, batched_vjp=batched_vjp)
        sweep_rows.extend(model_sweep_rows)
        sweep_summary_rows.extend(model_sweep_summary)
        model_matched_rows, model_matched_summary = step0_matched_swap_target(
            model,
            graphs,
            baseline,
            cfg,
            seed=seed,
            partner_policy=partner_policy,
        )
        matched_rows.extend(model_matched_rows)
        matched_summary_rows.extend(model_matched_summary)
        finite_rows = [r for r in recon_rows if r.get("model") == model.name and r.get("perturbation") == "finite_content_swap"]
        summaries.append(
            {
                "model": model.name,
                "ig_baseline_reconstruction_r2": r2_score(measured_vals, pred_vals),
                "finite_swap_reconstruction_r2": r2_score(
                    [safe_float(r["measured_delta"]) for r in finite_rows],
                    [safe_float(r["predicted_delta"]) for r in finite_rows],
                ),
                "graphs": len(graphs),
            }
        )
        progress(f"Step 0 {model.name}: complete, reconstruction_rows={len([r for r in recon_rows if r.get('model') == model.name])}")
        if cache_hits:
            progress(f"Step 0 {model.name}: reused cached IG carriage for {cache_hits}/{len(graphs)} graph(s)")
    write_csv(artifact_root / "metrics" / "step0_reconstruction.csv", recon_rows)
    write_csv(artifact_root / "metrics" / "step0_profile_agreement.csv", profile_rows)
    write_csv(artifact_root / "metrics" / "step0_reconstruction_residual_diagnostics.csv", residual_rows)
    write_csv(artifact_root / "metrics" / "step0_ig_step_count_sweep.csv", sweep_rows)
    write_csv(artifact_root / "metrics" / "step0_ig_step_count_sweep_summary.csv", sweep_summary_rows)
    write_csv(artifact_root / "metrics" / "step0_matched_swap_target.csv", matched_rows)
    write_csv(artifact_root / "metrics" / "step0_matched_swap_target_summary.csv", matched_summary_rows)
    write_csv(artifact_root / "metrics" / "step0_pairwise_ig_vs_swap_carriage.csv", pair_agreement_rows)
    write_csv(artifact_root / "metrics" / "step0_summary.csv", summaries)
    atomic_torch_save(artifact_root / "tensors" / "step0_carriage.pt", tensors)
    endpoint_swap_rows = [r for r in matched_rows if str(r.get("estimator")) == "endpoint_ig"]
    finite_linearization_rows = [r for r in recon_rows if r.get("perturbation") == "finite_content_swap"]
    render_step0_reconstruction(
        endpoint_swap_rows if endpoint_swap_rows else finite_linearization_rows,
        artifact_root,
        filename="step0_carriage_reconstruction",
        title="Step 0a: matched endpoint IG vs finite-swap Δŷ" if endpoint_swap_rows else "Step 0a diagnostic: clean readout linearization vs finite-swap Δŷ",
        xlabel="Predicted Δŷ from endpoint IG (prediction units)" if endpoint_swap_rows else "Predicted Δŷ from clean readout linearization",
        ylabel="Measured finite-swap Δŷ (prediction units)",
        dpi=dpi,
    )
    if endpoint_swap_rows and finite_linearization_rows:
        render_step0_reconstruction(
            finite_linearization_rows,
            artifact_root,
            filename="step0_clean_readout_linearization_swap",
            title="Step 0 diagnostic: clean readout linearization vs finite-swap Δŷ",
            xlabel="Predicted Δŷ from clean readout linearization",
            ylabel="Measured finite-swap Δŷ (prediction units)",
            dpi=dpi,
        )
    render_step0_reconstruction(
        [r for r in recon_rows if r.get("perturbation") == "ig_baseline_replacement"],
        artifact_root,
        filename="step0_ig_baseline_reconstruction",
        title="Step 0b diagnostic: all-baseline carriage vs one-node baseline Δŷ",
        xlabel="Predicted Δŷ = Σᵢ C[i,j] from all-baseline path",
        ylabel="Measured one-node baseline Δŷ (prediction units)",
        dpi=dpi,
    )
    render_distance_profile(profile_rows, artifact_root, "step0_swap_vs_ig_profiles", "Step 0: estimator agreement, swap vs IG", dpi=dpi)
    render_step0_pairwise_ig_vs_swap(pair_agreement_rows, artifact_root, dpi=dpi)
    render_step0_diagnostics(sweep_summary_rows, matched_summary_rows, residual_rows, artifact_root, dpi=dpi)
    progress("Step 0 complete: metrics, tensors, and figures written")
    return {
        "status": "complete",
        "models": [m.name for m in models],
        "summary": summaries,
        "ig_step_sweep_rows": len(sweep_rows),
        "matched_swap_target_rows": len(matched_rows),
        "pairwise_ig_swap_rows": len(pair_agreement_rows),
        "residual_diagnostic_rows": len(residual_rows),
    }


def pairwise_ig_swap_rows(
    model: str,
    graph_id: str,
    c_ig: torch.Tensor,
    c_swap: torch.Tensor,
    dist: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = min(int(c_ig.size(0)), int(c_ig.size(1)), int(c_swap.size(0)), int(c_swap.size(1)), int(dist.size(0)), int(dist.size(1)))
    for carrier in range(n):
        for source in range(n):
            d = safe_float(dist[carrier, source].item())
            ig = safe_float(c_ig[carrier, source].item())
            swap = safe_float(c_swap[carrier, source].item())
            if not (math.isfinite(d) and math.isfinite(ig) and math.isfinite(swap)):
                continue
            rows.append(
                {
                    "model": model,
                    "graph_id": graph_id,
                    "carrier": carrier,
                    "source": source,
                    "distance": d,
                    "same_node": carrier == source,
                    "c_ig": ig,
                    "c_swap": swap,
                }
            )
    return rows


def render_step0_reconstruction(
    rows: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    *,
    filename: str,
    title: str,
    dpi: int,
    xlabel: str = "Predicted Δŷ = Σᵢ C[i,j] (prediction units)",
    ylabel: str = "Measured Δŷ (prediction units)",
) -> None:
    if not rows:
        return
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)
    models = sorted(grouped)
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 5.0), squeeze=False, constrained_layout=True)
    for ax, model in zip(axes[0], models):
        items = grouped[model]
        x = [safe_float(r["predicted_delta"]) for r in items]
        y = [safe_float(r["measured_delta"]) for r in items]
        finite_vals = [v for v in [*x, *y] if math.isfinite(v)]
        lim = max([abs(v) for v in finite_vals] + [1.0e-6])
        ax.scatter(x, y, s=10, alpha=0.45, edgecolors="none")
        ax.plot([-lim, lim], [-lim, lim], "--", color="#555555", linewidth=1)
        ax.set_title(f"{model} R2={r2_score(y, x):.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.suptitle(title)
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / f"{filename}.png", dpi=dpi)
    fig.savefig(figures / f"{filename}.pdf")
    plt.close(fig)


def render_step0_pairwise_ig_vs_swap(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    clean = [
        r
        for r in rows
        if math.isfinite(safe_float(r.get("c_ig")))
        and math.isfinite(safe_float(r.get("c_swap")))
        and math.isfinite(safe_float(r.get("distance")))
    ]
    if not clean:
        return
    models = sorted({str(r.get("model")) for r in clean})
    fig, axes = plt.subplots(1, len(models), figsize=(5.4 * len(models), 5.0), squeeze=False, constrained_layout=True)
    all_vals = [safe_float(r.get("c_ig")) for r in clean] + [safe_float(r.get("c_swap")) for r in clean]
    lim = max([abs(v) for v in all_vals if math.isfinite(v)] + [1.0e-6])
    vmax = max([safe_float(r.get("distance")) for r in clean if math.isfinite(safe_float(r.get("distance")))] + [1.0])
    scatter = None
    for ax, model in zip(axes[0], models):
        items = [r for r in clean if str(r.get("model")) == model]
        x = np.asarray([safe_float(r.get("c_ig")) for r in items], dtype=float)
        y = np.asarray([safe_float(r.get("c_swap")) for r in items], dtype=float)
        color = np.asarray([safe_float(r.get("distance")) for r in items], dtype=float)
        scatter = ax.scatter(x, y, c=color, cmap="viridis", vmin=0.0, vmax=vmax, s=6, alpha=0.35, edgecolors="none")
        ax.plot([-lim, lim], [-lim, lim], "--", color="#555555", linewidth=1)
        ax.set_title(model)
        ax.set_xlabel("C_ig[i,j]")
        ax.set_ylabel("C_swap[i,j]")
    if scatter is not None:
        fig.colorbar(scatter, ax=list(axes[0]), label="Molecular hop distance")
    fig.suptitle("Step 0c: pair-level IG vs swap carriage")
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step0_pairwise_ig_vs_swap_carriage.png", dpi=dpi)
    fig.savefig(figures / "step0_pairwise_ig_vs_swap_carriage.pdf")
    plt.close(fig)


def step0_diagnostic_graphs(graphs: Sequence[Any], cfg: Mapping[str, Any]) -> list[Any]:
    limit = int(cfg.get("diagnostic_sample_graphs", 0))
    if limit <= 0:
        return []
    return list(graphs)[: min(limit, len(graphs))]


def step0_ig_step_sweep(
    model: ModelRun,
    graphs: Sequence[Any],
    mean_baseline: torch.Tensor,
    cfg: Mapping[str, Any],
    *,
    seed: int,
    batched_vjp: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step_values = [int(v) for v in cfg.get("ig_step_sweep", [])]
    baseline_kinds = [str(v) for v in cfg.get("baseline_sweep", ["mean_node_embedding"])]
    if not step_values:
        return [], []
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for graph_idx, graph in enumerate(step0_diagnostic_graphs(graphs, cfg)):
        gid = graph_identity("test", graph_idx, graph)
        clean_encoded = model.adapter.encoded_node_states(graph).detach()
        measured_cache = model.adapter.forward_from_encoded_content(graph, clean_encoded.to(model.adapter.device), retain_grad=False)
        clean_pred = float(measured_cache.prediction.reshape(-1)[0].detach().cpu().item())
        for baseline_kind in baseline_kinds:
            base = encoded_baseline(clean_encoded, mean_baseline, kind=baseline_kind)
            measured_by_source: dict[int, float] = {}
            for source in range(clean_encoded.size(0)):
                pert = clean_encoded.to(model.adapter.device).detach().clone()
                pert[source] = base.to(model.adapter.device)[source]
                pred_pert = float(predict_scalar_from_encoded(model.adapter, graph, pert).detach().cpu().item())
                measured_by_source[int(source)] = clean_pred - pred_pert
            for steps in step_values:
                result = carriage_ig(
                    model.adapter,
                    graph,
                    mean_baseline,
                    steps=steps,
                    baseline_override=base,
                    batched_vjp=batched_vjp,
                )
                c = result["carriage"]
                for source in range(c.size(1)):
                    predicted = float(c[:, source].sum().item())
                    measured = measured_by_source[int(source)]
                    rows.append(
                        {
                            "model": model.name,
                            "graph_id": gid,
                            "baseline": baseline_kind,
                            "ig_steps": int(steps),
                            "source": int(source),
                            "predicted_delta": predicted,
                            "measured_delta": measured,
                            "residual": measured - predicted,
                        }
                    )
    for model_name in sorted(set(str(r["model"]) for r in rows)):
        model_rows = [r for r in rows if str(r["model"]) == model_name]
        for baseline_kind in sorted(set(str(r["baseline"]) for r in model_rows)):
            base_rows = [r for r in model_rows if str(r["baseline"]) == baseline_kind]
            for steps in sorted(set(int(r["ig_steps"]) for r in base_rows)):
                step_rows = [r for r in base_rows if int(r["ig_steps"]) == steps]
                measured = [safe_float(r["measured_delta"]) for r in step_rows]
                predicted = [safe_float(r["predicted_delta"]) for r in step_rows]
                summaries.append(
                    {
                        "model": model_name,
                        "baseline": baseline_kind,
                        "ig_steps": int(steps),
                        "r2": r2_score(measured, predicted),
                        "mean_abs_residual": float(np.nanmean(np.abs(np.asarray(measured) - np.asarray(predicted)))) if step_rows else float("nan"),
                        "rows": len(step_rows),
                    }
                )
    return rows, summaries


def step0_matched_swap_target(
    model: ModelRun,
    graphs: Sequence[Any],
    mean_baseline: torch.Tensor,
    cfg: Mapping[str, Any],
    *,
    seed: int,
    partner_policy: str = "different_type",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matched_steps = int(cfg.get("matched_target_ig_steps", 0))
    sources_per_graph = int(cfg.get("matched_target_sources_per_graph", 0))
    partners_per_source = int(cfg.get("matched_target_partners_per_source", 0))
    if matched_steps <= 0 or sources_per_graph <= 0 or partners_per_source <= 0:
        return [], []
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for graph_idx, graph in enumerate(step0_diagnostic_graphs(graphs, cfg)):
        gid = graph_identity("test", graph_idx, graph)
        encoded = model.adapter.encoded_node_states(graph).detach().to(model.adapter.device)
        clean_cache, readout_grad = model.adapter.readout_gradient_from_encoded_content(graph, encoded)
        clean_h = clean_cache.final_node_states.detach()
        readout_grad = readout_grad.to(model.adapter.device)
        clean_pred = float(clean_cache.prediction.reshape(-1)[0].detach().cpu().item())
        sources = list(range(encoded.size(0)))
        rng = random.Random(int(seed) + 7919 * (graph_idx + 1))
        rng.shuffle(sources)
        type_signatures = node_type_signatures(graph)
        require_different_type = partner_policy in {"different_type", "different-type", "different_atom_type"}
        for source in sources[: min(sources_per_graph, len(sources))]:
            partners = deterministic_partners(
                encoded.size(0),
                source,
                partners_per_source,
                seed + 3571 * graph_idx,
                type_signatures=type_signatures,
                require_different_type=require_different_type,
            )
            for partner in partners:
                endpoint = encoded.detach().clone()
                endpoint[int(source)] = encoded[int(partner)]
                swap_cache = model.adapter.forward_from_encoded_content(graph, endpoint, retain_grad=False)
                measured = float(swap_cache.prediction.reshape(-1)[0].detach().cpu().item()) - clean_pred
                delta_h = swap_cache.final_node_states.detach() - clean_h
                first_order = float((delta_h * readout_grad).sum().detach().cpu().item())
                endpoint_ig = output_ig_source_endpoint(
                    model.adapter,
                    graph,
                    encoded,
                    endpoint,
                    source=int(source),
                    steps=matched_steps,
                )
                for estimator, predicted in [
                    ("clean_readout_linearization", first_order),
                    ("endpoint_ig", endpoint_ig),
                ]:
                    rows.append(
                        {
                            "model": model.name,
                            "graph_id": gid,
                            "source": int(source),
                            "partner": int(partner),
                            "estimator": estimator,
                            "ig_steps": matched_steps if estimator == "endpoint_ig" else "",
                            "predicted_delta": predicted,
                            "measured_delta": measured,
                            "residual": measured - predicted,
                        }
                    )
    for model_name in sorted(set(str(r["model"]) for r in rows)):
        model_rows = [r for r in rows if str(r["model"]) == model_name]
        for estimator in sorted(set(str(r["estimator"]) for r in model_rows)):
            est_rows = [r for r in model_rows if str(r["estimator"]) == estimator]
            measured = [safe_float(r["measured_delta"]) for r in est_rows]
            predicted = [safe_float(r["predicted_delta"]) for r in est_rows]
            summaries.append(
                {
                    "model": model_name,
                    "estimator": estimator,
                    "r2": r2_score(measured, predicted),
                    "mean_abs_residual": float(np.nanmean(np.abs(np.asarray(measured) - np.asarray(predicted)))) if est_rows else float("nan"),
                    "rows": len(est_rows),
                }
            )
    return rows, summaries


def step0_residual_diagnostic_rows(
    *,
    model: str,
    graph_id: str,
    dist: torch.Tensor,
    carriage: torch.Tensor,
    source: int,
    predicted: float,
    measured: float,
    tau: int,
) -> list[dict[str, Any]]:
    residual = float(measured) - float(predicted)
    source_mass = carriage.detach().abs()[:, int(source)].cpu()
    finite_dist = dist[:, int(source)].detach().cpu()
    finite = torch.isfinite(finite_dist)
    total_mass = float(source_mass[finite].sum().item())
    distance_masses: dict[int, float] = {}
    for d in sorted({int(v.item()) for v in finite_dist[finite]}):
        mask = finite & (finite_dist.long() == int(d))
        distance_masses[int(d)] = float(source_mass[mask].sum().item())
    dominant_distance = max(distance_masses.items(), key=lambda item: item[1])[0] if distance_masses else -1
    far_mass = float(source_mass[finite & (finite_dist > tau)].sum().item())
    rows = [
        {
            "model": model,
            "graph_id": graph_id,
            "source": int(source),
            "bin_type": "source",
            "bin": "source_total",
            "residual": residual,
            "abs_residual": abs(residual),
            "predicted_delta": float(predicted),
            "measured_delta": float(measured),
            "source_abs_carriage": total_mass,
            "source_far_mass_share": far_mass / max(total_mass, EPS),
            "dominant_distance": dominant_distance,
            "weight": 1.0,
        }
    ]
    for distance, mass in distance_masses.items():
        weight = mass / max(total_mass, EPS)
        rows.append(
            {
                "model": model,
                "graph_id": graph_id,
                "source": int(source),
                "bin_type": "distance_weighted",
                "bin": int(distance),
                "residual": residual,
                "abs_residual": abs(residual),
                "predicted_delta": float(predicted),
                "measured_delta": float(measured),
                "source_abs_carriage": total_mass,
                "source_far_mass_share": far_mass / max(total_mass, EPS),
                "dominant_distance": dominant_distance,
                "weight": weight,
            }
        )
    return rows


def render_step0_diagnostics(
    sweep_summary: Sequence[Mapping[str, Any]],
    matched_summary: Sequence[Mapping[str, Any]],
    residual_rows: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    *,
    dpi: int,
) -> None:
    figures = ensure_dir(artifact_root / "figures")
    if sweep_summary:
        fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
        for (model, baseline), rows in sorted(
            {
                (str(r["model"]), str(r["baseline"])): [
                    item for item in sweep_summary if str(item["model"]) == str(r["model"]) and str(item["baseline"]) == str(r["baseline"])
                ]
                for r in sweep_summary
            }.items()
        ):
            rows = sorted(rows, key=lambda r: safe_float(r["ig_steps"]))
            ax.plot([safe_float(r["ig_steps"]) for r in rows], [safe_float(r["r2"]) for r in rows], marker="o", label=f"{model} {baseline}")
        ax.set_xscale("log", base=2)
        ax.set_title("Step 0: IG step-count sweep")
        ax.set_xlabel("IG steps")
        ax.set_ylabel("Baseline-replacement R2")
        ax.legend(frameon=False, fontsize=7)
        fig.savefig(figures / "step0_ig_step_count_sweep.png", dpi=dpi)
        fig.savefig(figures / "step0_ig_step_count_sweep.pdf")
        plt.close(fig)

    if matched_summary:
        fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        labels = [f"{r['model']} {str(r['estimator']).replace('_', ' ')}" for r in matched_summary]
        values = [safe_float(r["r2"]) for r in matched_summary]
        clipped_values = [min(1.0, max(-1.0, v)) if math.isfinite(v) else float("nan") for v in values]
        ax.barh(np.arange(len(labels)), clipped_values, color="#4c78a8")
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=7)
        ax.axvline(0, color="#555555", linewidth=1)
        ax.set_title("Step 0: matched swap-target reconstruction")
        ax.set_xlabel("R2 against measured swap effect (display clipped to [-1, 1])")
        for idx, (value, shown) in enumerate(zip(values, clipped_values)):
            if math.isfinite(value) and value != shown:
                ax.text(
                    shown,
                    idx,
                    f" actual {value:.1f}",
                    va="center",
                    ha="right" if shown < 0 else "left",
                    fontsize=6,
                    color="#333333",
                )
        fig.savefig(figures / "step0_matched_swap_target.png", dpi=dpi)
        fig.savefig(figures / "step0_matched_swap_target.pdf")
        plt.close(fig)

    source_rows = [r for r in residual_rows if str(r.get("bin_type")) == "source"]
    if source_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), constrained_layout=True)
        by_distance: dict[tuple[str, int], list[float]] = {}
        for row in source_rows:
            if math.isfinite(safe_float(row.get("dominant_distance"))):
                by_distance.setdefault((str(row["model"]), int(safe_float(row["dominant_distance"]))), []).append(safe_float(row["abs_residual"]))
        for model in sorted({key[0] for key in by_distance}):
            xs = sorted(key[1] for key in by_distance if key[0] == model)
            axes[0].plot(xs, [float(np.nanmean(by_distance[(model, x)])) for x in xs], marker="o", label=model)
        axes[0].set_title("Residual by dominant distance")
        axes[0].set_xlabel("Dominant carriage distance")
        axes[0].set_ylabel("Mean absolute residual")
        axes[0].legend(frameon=False, fontsize=7)

        for model in sorted(set(str(r["model"]) for r in source_rows)):
            vals = [safe_float(r["source_abs_carriage"]) for r in source_rows if str(r["model"]) == model]
            finite_vals = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
            if finite_vals.size == 0:
                continue
            qs = np.quantile(finite_vals, [0.25, 0.5, 0.75])
            bucket_values: dict[int, list[float]] = {idx: [] for idx in range(4)}
            for row in source_rows:
                if str(row["model"]) != model:
                    continue
                bucket = int(np.searchsorted(qs, safe_float(row["source_abs_carriage"]), side="right"))
                bucket_values[bucket].append(safe_float(row["abs_residual"]))
            axes[1].plot(
                [1, 2, 3, 4],
                [float(np.nanmean(bucket_values[idx])) if bucket_values[idx] else float("nan") for idx in range(4)],
                marker="o",
                label=model,
            )
        axes[1].set_title("Residual by carriage size")
        axes[1].set_xlabel("|C| quartile")
        axes[1].set_ylabel("Mean absolute residual")
        axes[1].legend(frameon=False, fontsize=7)
        fig.savefig(figures / "step0_reconstruction_residual_diagnostics.png", dpi=dpi)
        fig.savefig(figures / "step0_reconstruction_residual_diagnostics.pdf")
        plt.close(fig)


def render_step2_profiles(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    direct_rows = [r for r in rows if str(r.get("quantity")) in {"attention_last", "carriage", "swap_carriage"}]
    figures = ensure_dir(artifact_root / "figures")
    deprecated_stems = ("step2_" + "roll" + "out_vs_carriage_distance",)
    for stem in deprecated_stems:
        for suffix in (".png", ".pdf"):
            path = figures / f"{stem}{suffix}"
            if path.exists():
                path.unlink()
    render_distance_profile(
        direct_rows,
        artifact_root,
        "step2_direct_attention_distance",
        "Step 2: last-layer attention and carriage by molecular distance",
        dpi=dpi,
    )

    numeric_rows = [
        r
        for r in rows
        if str(r.get("quantity")) in {"attention_last", "carriage", "swap_carriage"}
        and math.isfinite(safe_float(r.get("distance")))
        and math.isfinite(safe_float(r.get("share")))
    ]
    if not numeric_rows:
        return

    def grouped_items(quantities: set[str]) -> dict[tuple[str, str], list[dict[str, float]]]:
        grouped_values: dict[tuple[str, str, int], list[float]] = {}
        for row in numeric_rows:
            quantity = str(row.get("quantity"))
            if quantity not in quantities:
                continue
            key = (str(row.get("model")), quantity, int(safe_float(row.get("distance"))))
            grouped_values.setdefault(key, []).append(safe_float(row.get("share")))
        grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
        for (model, quantity, distance), values in grouped_values.items():
            mean, lo, hi = bootstrap_ci(values, seed=517 + distance, draws=500)
            grouped.setdefault((model, quantity), []).append({"distance": float(distance), "share": mean, "lo": lo, "hi": hi})
        return grouped

    label_map = {
        "attention_last": "last-layer attention",
        "carriage": "IG carriage",
        "swap_carriage": "swap carriage",
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    for (model, quantity), items in sorted(grouped_items({"attention_last", "carriage", "swap_carriage"}).items()):
        items = sorted(items, key=lambda r: safe_float(r.get("distance")))
        x = np.asarray([safe_float(r.get("distance")) for r in items], dtype=float)
        y = np.asarray([safe_float(r.get("share")) for r in items], dtype=float)
        linestyle = "--" if quantity.startswith("attention") else "-"
        ax.plot(x, y, marker="o", linewidth=1.35, linestyle=linestyle, label=f"{model} {label_map.get(quantity, quantity)}")
        lo = np.asarray([safe_float(r.get("lo")) for r in items], dtype=float)
        hi = np.asarray([safe_float(r.get("hi")) for r in items], dtype=float)
        if np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.fill_between(x, lo, hi, alpha=0.10)
    ax.set_title("Step 2: last-layer attention vs carriage by molecular distance")
    ax.set_xlabel("Molecular hop distance")
    ax.set_ylabel("Share of own mass by distance (each series sums to 1)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(figures / "step2_carriage_vs_attention_distance.png", dpi=dpi)
    fig.savefig(figures / "step2_carriage_vs_attention_distance.pdf")
    plt.close(fig)


def render_attention_mean_distance(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    clean = [
        r
        for r in rows
        if math.isfinite(safe_float(r.get("layer"))) and math.isfinite(safe_float(r.get("mean_attention_distance")))
    ]
    if not clean:
        return
    grouped_values: dict[tuple[str, int], list[float]] = {}
    for row in clean:
        grouped_values.setdefault(
            (str(row.get("model")), int(safe_float(row.get("layer")))),
            [],
        ).append(safe_float(row.get("mean_attention_distance")))
    grouped: dict[str, list[dict[str, float]]] = {}
    for (model, layer), values in grouped_values.items():
        mean, lo, hi = bootstrap_ci(values, seed=971 + layer, draws=500)
        grouped.setdefault(model, []).append({"layer": float(layer), "mean": mean, "lo": lo, "hi": hi})
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    for model, items in sorted(grouped.items()):
        items = sorted(items, key=lambda r: safe_float(r["layer"]))
        x = np.asarray([safe_float(r["layer"]) for r in items], dtype=float)
        y = np.asarray([safe_float(r["mean"]) for r in items], dtype=float)
        lo = np.asarray([safe_float(r["lo"]) for r in items], dtype=float)
        hi = np.asarray([safe_float(r["hi"]) for r in items], dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.5, label=model)
        ax.fill_between(x, lo, hi, alpha=0.12)
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1, label="1-hop")
    ax.set_title("Step 2: mean attention distance by layer")
    ax.set_xlabel("GRIT attention layer")
    ax.set_ylabel("Mean attention distance (molecular hops)")
    ax.legend(frameon=False, fontsize=8)
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step2_mean_attention_distance_by_layer.png", dpi=dpi)
    fig.savefig(figures / "step2_mean_attention_distance_by_layer.pdf")
    plt.close(fig)


def render_attention_support_audit(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    clean = [
        r
        for r in rows
        if math.isfinite(safe_float(r.get("layer")))
        and math.isfinite(safe_float(r.get("max_direct_attention_distance")))
        and math.isfinite(safe_float(r.get("direct_attention_mass_distance_gt1")))
    ]
    if not clean:
        return
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in clean:
        grouped.setdefault((str(row.get("model")), int(safe_float(row.get("layer")))), []).append(row)
    summary: dict[str, list[dict[str, float]]] = {}
    for (model, layer), items in grouped.items():
        max_distance_values = [safe_float(r.get("max_direct_attention_distance")) for r in items]
        far_mass_values = [safe_float(r.get("direct_attention_mass_distance_gt1")) for r in items]
        summary.setdefault(model, []).append(
            {
                "layer": float(layer),
                "max_distance": float(np.nanmax(max_distance_values)),
                "far_mass": float(np.nanmean(far_mass_values)),
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for model, items in sorted(summary.items()):
        items = sorted(items, key=lambda r: safe_float(r["layer"]))
        x = np.asarray([safe_float(r["layer"]) for r in items], dtype=float)
        max_distance = np.asarray([safe_float(r["max_distance"]) for r in items], dtype=float)
        far_mass = np.asarray([safe_float(r["far_mass"]) for r in items], dtype=float)
        axes[0].plot(x, max_distance, marker="o", linewidth=1.5, label=model)
        axes[1].plot(x, far_mass, marker="o", linewidth=1.5, label=model)
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=1, label="1-hop limit")
    axes[0].set_title("Per-layer attention support")
    axes[0].set_xlabel("GRIT attention layer")
    axes[0].set_ylabel("Max molecular hop distance")
    axes[1].axhline(0.0, color="#555555", linestyle="--", linewidth=1, label="1-hop expected")
    axes[1].set_title("Non-local per-layer attention mass")
    axes[1].set_xlabel("GRIT attention layer")
    axes[1].set_ylabel("Mass at distance > 1")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step2_attention_support_audit.png", dpi=dpi)
    fig.savefig(figures / "step2_attention_support_audit.pdf")
    plt.close(fig)


def run_step2(models: Sequence[ModelRun], artifact_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    progress("Step 2 start: usage vs causal usage")
    cfg = config["steps"]["2"]
    sample_graphs = int(cfg.get("sample_graphs", 200))
    ig_steps = int(config["perturbation"].get("ig_steps", 32))
    swap_partners = int(config["perturbation"].get("swap_partners", 8))
    batched_vjp = carriage_ig_uses_batched_vjp(config)
    partner_policy = swap_partner_policy(config)
    include_swap_attention_check = bool(cfg.get("compare_attention_to_swaps", True))
    run_layer_channel_split = bool(cfg.get("run_layer_channel_split", True))
    run_attention_erasure = bool(cfg.get("run_attention_erasure", True))
    erasure_models = {str(name) for name in cfg.get("erasure_models", ["dense_grit"])}
    erasure_sample_graphs = int(cfg.get("erasure_sample_graphs", min(sample_graphs, 64)))
    erasure_fractions = [float(value) for value in cfg.get("erasure_fractions", [0.0, 0.05, 0.10, 0.20, 0.35, 0.50])]
    erasure_include_near_control = bool(cfg.get("erasure_include_near_control", True))
    run_attention_graph_ablation = bool(cfg.get("run_attention_graph_ablation", True))
    attention_graph_aggregation = str(cfg.get("attention_graph_aggregation", "mean"))
    attention_graph_ablation_models = {str(name) for name in cfg.get("attention_graph_ablation_models", cfg.get("erasure_models", ["dense_grit"]))}
    attention_graph_ablation_sample_graphs = int(cfg.get("attention_graph_ablation_sample_graphs", 16))
    attention_graph_ablation_fractions = [float(v) for v in cfg.get("attention_graph_ablation_fractions", [0.0, 0.1, 0.2, 0.3, 0.5])]
    tau = int(config.get("primary_tau", 3))
    seed = int(config.get("seeds", [0])[0])
    dpi = int(config["figures"]["dpi"])
    profile_rows: list[dict[str, Any]] = []
    faith_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    erasure_rows: list[dict[str, Any]] = []
    ag_ablation_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    mean_distance_rows: list[dict[str, Any]] = []
    tensors: dict[str, Any] = {}
    for model in models:
        graphs = select_graphs(model.adapter, "test", sample_graphs, seed=seed)
        progress(
            f"Step 2 {model.name}: selected {len(graphs)} test graph(s), "
            f"IG steps={ig_steps}, swap_check={include_swap_attention_check}, "
            f"layer_channel_split={run_layer_channel_split}, tau={tau}"
        )
        baseline = mean_encoded_baseline(model.adapter, select_baseline_graphs(model.adapter, "test", config, sample_graphs, seed=seed))
        cache_hits = 0
        for graph_idx, graph in enumerate(graphs):
            progress_graph("Step 2", model.name, graph_idx, len(graphs))
            gid = graph_identity("test", graph_idx, graph)
            dist = distance_matrix(graph)
            result = carriage_ig_cached(
                model,
                graph,
                baseline,
                artifact_root,
                config,
                split="test",
                graph_id=gid,
                steps=ig_steps,
                batched_vjp=batched_vjp,
                capture_attention=True,
                capture_channels=run_layer_channel_split,
                capture_layer_outputs=run_layer_channel_split,
            )
            cache_hits += int(bool(result.get("carriage_cache_hit")))
            c_ig = result["carriage"]
            cache = result["clean_cache"]
            tensors[f"step2/{model.name}/{gid}/carriage"] = c_ig
            profile_rows.extend(carriage_profile_rows(model.name, gid, c_ig, dist, "carriage"))
            attn_rows, attn_tensors = attention_profiles(cache, dist, model.name)
            for row in attn_rows:
                row["graph_id"] = gid
            profile_rows.extend(attn_rows)
            mean_distance_rows.extend(attention_mean_distance_rows(cache, dist, model.name, gid))
            tensors.update({f"step2/{model.name}/{gid}/{k}": v for k, v in attn_tensors.items()})
            expected_direct_distance = 1 if ("1hop" in model.name or "1hop" in model.variant) else None
            support_rows.extend(
                attention_support_audit_rows(
                    cache,
                    dist,
                    model.name,
                    gid,
                    expected_max_direct_distance=expected_direct_distance,
                )
            )
            c_swap = None
            if include_swap_attention_check:
                c_swap = carriage_swap(
                    model.adapter,
                    graph,
                    result["clean_encoded"].to(model.adapter.device),
                    result["readout_gradient"].to(model.adapter.device),
                    partners=swap_partners,
                    seed=seed + 1009 * graph_idx,
                    partner_policy=partner_policy,
                )
                tensors[f"step2/{model.name}/{gid}/swap_carriage"] = c_swap
                profile_rows.extend(carriage_profile_rows(model.name, gid, c_swap, dist, "swap_carriage"))
            if any(quantity in attn_tensors for quantity in ("attention_last", "attention_first")):
                faith_rows.extend(attention_faithfulness_rows(model.name, gid, attn_tensors, c_ig, dist, tau, carriage_estimator="ig"))
                if c_swap is not None:
                    faith_rows.extend(
                        attention_faithfulness_rows(
                            model.name,
                            gid,
                            attn_tensors,
                            c_swap,
                            dist,
                            tau,
                            carriage_estimator="swap",
                        )
                    )
                for threshold in far_thresholds(config):
                    for quantity in ["attention_last", "attention_first"]:
                        if quantity in attn_tensors:
                            threshold_rows.append(
                                {
                                    "model": model.name,
                                    "graph_id": gid,
                                    "attention_quantity": quantity,
                                    "tau": threshold,
                                    "attention_far_mass": far_mass(attn_tensors[quantity], dist, threshold),
                                    "carriage_far_mass": far_mass(c_ig, dist, threshold),
                                }
                            )
            if (
                run_attention_graph_ablation
                and model.name in attention_graph_ablation_models
                and graph_idx < max(0, attention_graph_ablation_sample_graphs)
                and cache.attention
            ):
                try:
                    ag_ablation_rows.extend(
                        attention_graph_vs_carriage_ablation_rows(
                            model,
                            graph,
                            gid,
                            cache,
                            c_ig,
                            dist,
                            tau=tau,
                            fractions=attention_graph_ablation_fractions,
                            include_near_control=erasure_include_near_control,
                            aggregation=attention_graph_aggregation,
                            seed=seed + 5501 * graph_idx,
                        )
                    )
                except Exception as exc:
                    progress(f"  {model.name} graph {graph_idx}: attention-graph ablation failed ({exc})")
            if run_layer_channel_split:
                channel_rows.extend(layer_channel_split_rows(model, graph, result, dist, gid, tau))
            if (
                run_attention_erasure
                and model.name in erasure_models
                and graph_idx < max(0, erasure_sample_graphs)
                and any(quantity in attn_tensors for quantity in ("attention_last",))
            ):
                try:
                    erasure_rows.extend(
                        attention_erasure_rows(
                            model,
                            graph,
                            gid,
                            cache,
                            c_ig,
                            dist,
                            tau=tau,
                            fractions=erasure_fractions,
                            include_near_control=erasure_include_near_control,
                            seed=seed + 9973 * graph_idx,
                        )
                    )
                except Exception as exc:
                    erasure_rows.append(
                        {
                            "model": model.name,
                            "graph_id": gid,
                            "region": "error",
                            "ranker": "error",
                            "tau": tau,
                            "fraction_removed": float("nan"),
                            "edges_available": 0,
                            "edges_removed": 0,
                            "prediction_clean": float("nan"),
                            "prediction_erased": float("nan"),
                            "abs_delta_prediction": float("nan"),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        if cache_hits:
            progress(f"Step 2 {model.name}: reused cached IG carriage for {cache_hits}/{len(graphs)} graph(s)")
    write_csv(artifact_root / "metrics" / "step2_profiles.csv", profile_rows)
    write_csv(artifact_root / "metrics" / "step2_attention_faithfulness.csv", faith_rows)
    write_csv(artifact_root / "metrics" / "step2_far_threshold_sensitivity.csv", threshold_rows)
    write_csv(artifact_root / "metrics" / "step2_channel_split.csv", channel_rows)
    head_rows = [r for r in channel_rows if str(r.get("quantity")) == "head_far_carriage"]
    write_csv(artifact_root / "metrics" / "step2_head_resolved_carriage.csv", head_rows)
    write_csv(artifact_root / "metrics" / "step2_attention_erasure.csv", erasure_rows)
    write_csv(artifact_root / "metrics" / "step2_attention_support_audit.csv", support_rows)
    write_csv(artifact_root / "metrics" / "step2_attention_mean_distance.csv", mean_distance_rows)
    atomic_torch_save(artifact_root / "tensors" / "step2_usage_carriage.pt", tensors)
    support_failures = attention_support_failures(support_rows)
    write_json(
        artifact_root / "metrics" / "step2_attention_support_summary.json",
        {
            "status": "pass" if not support_failures else "failed_attention_support_audit",
            "violating_rows": len(support_failures),
            "checked_rows": len(support_rows),
            "first_failure": dict(support_failures[0]) if support_failures else None,
        },
    )
    render_attention_support_audit(support_rows, artifact_root, dpi=dpi)
    if support_failures:
        # Non-fatal: this is an ATTENTION-FIGURE diagnostic only. Standalone validation confirmed the
        # 1-hop model's attention_edges are d<=1 (the model is genuinely local), so violations here
        # reflect a capture/read discrepancy in the pipeline audit path, NOT a broken control — and
        # the carriage that drives Steps 3/4/5 is independent of this audit. Record it in the Step 2
        # status and continue; do not abort the whole procedure on an attention-support figure.
        first = support_failures[0]
        progress(
            f"[WARN] Step 2 attention-support audit: {len(support_failures)} violating row(s) "
            f"(first: {first.get('model')} layer={first.get('layer')} "
            f"max_direct_distance={first.get('max_direct_attention_distance')}). NON-FATAL — recorded in "
            "step2 status; Steps 3/4/5 (carriage-based) are unaffected. Investigate attention capture "
            "separately if the Step 2 attention figures are needed."
        )
    render_step2_profiles(profile_rows, artifact_root, dpi=dpi)
    render_attention_mean_distance(mean_distance_rows, artifact_root, dpi=dpi)
    render_step2_faithfulness(faith_rows, artifact_root, dpi=dpi)
    render_step2_faithfulness(
        faith_rows,
        artifact_root,
        dpi=dpi,
        attention_quantity="attention_first",
        attention_label="first-layer attention",
        faithfulness_stem="step2_first_layer_attention_faithfulness",
        far_mass_stem="step2_far_mass_first_layer_attention_vs_carriage",
    )
    final_channel_rows = [r for r in channel_rows if bool(r.get("headline_final_layer"))]
    render_distance_profile(final_channel_rows, artifact_root, "step2_channel_split_distance", "Step 2: final-layer carriage by channel and distance", dpi=dpi)
    render_layer_resolved_channel_split(channel_rows, artifact_root, dpi=dpi)
    render_head_resolved_carriage(head_rows, artifact_root, dpi=dpi)
    render_step2_attention_erasure(erasure_rows, artifact_root, dpi=dpi)
    write_csv(artifact_root / "metrics" / "step2_attention_graph_vs_carriage_ablation.csv", ag_ablation_rows)
    render_step2_attention_graph_ablation(ag_ablation_rows, artifact_root, dpi=dpi)
    progress("Step 2 complete: metrics, tensors, and figures written")
    return {
        "status": "complete" if not support_failures else "complete_with_attention_support_violations",
        "models": [m.name for m in models],
        "attention_policy": "rollout_omitted_by_design; reads=last_layer_head_averaged_attention plus first_layer/per_layer_attention_diagnostics; carries=carriage",
        "profile_rows": len(profile_rows),
        "faithfulness_rows": len(faith_rows),
        "attention_erasure_rows": len(erasure_rows),
        "attention_graph_ablation_rows": len(ag_ablation_rows),
        "support_audit_rows": len(support_rows),
        "head_resolved_rows": len(head_rows),
        "attention_mean_distance_rows": len(mean_distance_rows),
        "attention_support_violations": len(support_failures),
    }


def layer_channel_split_rows(model: ModelRun, graph: Any, result: Mapping[str, Any], dist: torch.Tensor, graph_id: str, tau: int) -> list[dict[str, Any]]:
    clean_cache = result["clean_cache"]
    clean_layers = (clean_cache.channel_fields or {}).get("layers", [])
    if not clean_layers:
        return []
    layer_grads = []
    if clean_cache.extras is not None:
        layer_grads = list(clean_cache.extras.get("layer_output_node_gradients", []))
    if len(layer_grads) < len(clean_layers):
        raise RuntimeError(
            "layer-resolved channel split requires gradients for every captured GRIT layer; "
            f"captured {len(clean_layers)} layer field(s) but only {len(layer_grads)} layer gradient tensor(s). "
            "The layer-resolved channel figure would otherwise show artificial zeros."
        )
    zero_norm_layers = [
        idx
        for idx, grad in enumerate(layer_grads[: len(clean_layers)])
        if float(torch.linalg.vector_norm(grad.detach()).detach().cpu().item()) <= 0.0
    ]
    if zero_norm_layers:
        raise RuntimeError(
            "layer-resolved channel split received zero-norm readout gradients for GRIT layer(s) "
            f"{zero_norm_layers}; the layer-resolved channel figure would show artificial zeros."
        )
    readout_grad = result["readout_gradient"].to(model.adapter.device)
    baseline = result["baseline"].to(model.adapter.device)
    encoded = result["clean_encoded"].to(model.adapter.device)
    n = int(encoded.size(0))
    last_layer = len(clean_layers) - 1
    masses_by_layer = {
        layer_idx: {"routing": torch.zeros((n, n)), "transport": torch.zeros((n, n)), "cross": torch.zeros((n, n))}
        for layer_idx in range(len(clean_layers))
    }
    head_masses_by_layer: dict[int, dict[int, torch.Tensor]] = {}
    for source in range(n):
        pert = encoded.detach().clone()
        pert[source] = baseline[source]
        cache = model.adapter.forward_from_encoded_content(
            graph,
            pert,
            retain_grad=False,
            capture_channels=True,
        )
        pert_layers = (cache.channel_fields or {}).get("layers", [])
        for layer_idx, clean in enumerate(clean_layers):
            if layer_idx >= len(pert_layers):
                continue
            fields = pert_layers[layer_idx]
            edge_index = clean["edge_index"]
            attn = clean["attn"].detach()
            if attn.dim() == 3 and attn.size(-1) == 1:
                attn = attn.squeeze(-1)
            pert_attn = fields["attn"].detach()
            if pert_attn.dim() == 3 and pert_attn.size(-1) == 1:
                pert_attn = pert_attn.squeeze(-1)
            clean_msg = relation_enhanced_edge_messages(clean, edge_index)
            pert_msg = relation_enhanced_edge_messages(fields, edge_index)
            if pert_attn.shape != attn.shape or pert_msg.shape != clean_msg.shape:
                continue
            da = pert_attn - attn
            dv = pert_msg - clean_msg
            src = edge_index[0].long()
            dst = edge_index[1].long()
            if layer_idx < len(layer_grads):
                grad = layer_grads[layer_idx].to(model.adapter.device)
            elif layer_idx == last_layer:
                grad = readout_grad
            else:
                continue
            for e in range(edge_index.size(1)):
                receiver = int(dst[e].item())
                routing_vec = (clean_msg[e] * da[e].view(-1, 1)).reshape(-1)
                transport_vec = (dv[e] * attn[e].view(-1, 1)).reshape(-1)
                cross_vec = (dv[e] * da[e].view(-1, 1)).reshape(-1)
                heads = int(clean_msg.size(1))
                head_dim = int(clean_msg.size(2))
                if layer_idx not in head_masses_by_layer:
                    head_masses_by_layer[layer_idx] = {head: torch.zeros((n, n)) for head in range(heads)}
                g = grad[receiver].reshape(-1)
                if g.numel() == routing_vec.numel():
                    masses_by_layer[layer_idx]["routing"][receiver, source] += float(torch.dot(g, routing_vec).detach().cpu().item())
                    masses_by_layer[layer_idx]["transport"][receiver, source] += float(torch.dot(g, transport_vec).detach().cpu().item())
                    masses_by_layer[layer_idx]["cross"][receiver, source] += float(torch.dot(g, cross_vec).detach().cpu().item())
                    g_heads = g.view(heads, head_dim)
                    for head in range(heads):
                        head_vec = (
                            clean_msg[e, head] * da[e, head]
                            + dv[e, head] * attn[e, head]
                            + dv[e, head] * da[e, head]
                        )
                        head_masses_by_layer[layer_idx][head][receiver, source] += float(
                            torch.dot(g_heads[head], head_vec).detach().cpu().item()
                        )
                else:
                    # Fallback if GRIT's output projection changes hidden shape.
                    masses_by_layer[layer_idx]["routing"][receiver, source] += float(torch.linalg.vector_norm(routing_vec).detach().cpu().item())
                    masses_by_layer[layer_idx]["transport"][receiver, source] += float(torch.linalg.vector_norm(transport_vec).detach().cpu().item())
                    masses_by_layer[layer_idx]["cross"][receiver, source] += float(torch.linalg.vector_norm(cross_vec).detach().cpu().item())
                    for head in range(heads):
                        head_vec = (
                            clean_msg[e, head] * da[e, head]
                            + dv[e, head] * attn[e, head]
                            + dv[e, head] * da[e, head]
                        )
                        head_masses_by_layer[layer_idx][head][receiver, source] += float(
                            torch.linalg.vector_norm(head_vec).detach().cpu().item()
                        )
    rows: list[dict[str, Any]] = []
    for layer_idx, masses in masses_by_layer.items():
        for quantity, matrix in masses.items():
            for row in carriage_profile_rows(model.name, graph_id, matrix, dist, quantity):
                row["layer"] = layer_idx
                row["headline_final_layer"] = layer_idx == last_layer
                rows.append(row)
            rows.append(
                {
                    "model": model.name,
                    "graph_id": graph_id,
                    "quantity": f"{quantity}_far_mass",
                    "layer": layer_idx,
                    "headline_final_layer": layer_idx == last_layer,
                    "distance": f">{tau}",
                    "mass": float(matrix.abs()[torch.isfinite(dist) & (dist > tau)].sum().item()),
                    "share": far_mass(matrix, dist, tau),
                }
            )
    far_mask = torch.isfinite(dist) & (dist > tau)
    for layer_idx, by_head in sorted(head_masses_by_layer.items()):
        total_far = sum(float(matrix.abs()[far_mask].sum().item()) for matrix in by_head.values())
        for head, matrix in sorted(by_head.items()):
            head_far = float(matrix.abs()[far_mask].sum().item())
            rows.append(
                {
                    "model": model.name,
                    "graph_id": graph_id,
                    "quantity": "head_far_carriage",
                    "layer": layer_idx,
                    "headline_final_layer": layer_idx == last_layer,
                    "head": int(head),
                    "distance": f">{tau}",
                    "mass": head_far,
                    "share": head_far / max(total_far, EPS),
                    "far_mass_within_head": far_mass(matrix, dist, tau),
                }
            )
    return rows


def relation_enhanced_edge_messages(fields: Mapping[str, Any], edge_index: torch.Tensor) -> torch.Tensor:
    src = edge_index[0].long()
    msg = fields["V_h"].detach()[src]
    if bool(fields.get("edge_enhance")) and fields.get("wE") is not None and fields.get("VeRow") is not None:
        edge_state = fields["wE"].detach()
        if edge_state.dim() == 2:
            edge_state = edge_state.view(edge_state.size(0), msg.size(1), msg.size(2))
        pair_msg = torch.einsum("ehd,dhc->ehc", edge_state, fields["VeRow"].detach().to(edge_state.device, edge_state.dtype))
        msg = msg + pair_msg.to(device=msg.device, dtype=msg.dtype)
    return msg


def render_layer_resolved_channel_split(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    far_rows = [
        r
        for r in rows
        if str(r.get("quantity")) in {"routing_far_mass", "transport_far_mass", "cross_far_mass"}
        and math.isfinite(safe_float(r.get("share")))
        and math.isfinite(safe_float(r.get("layer")))
    ]
    if not far_rows:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    grouped_values: dict[tuple[str, str, int], list[float]] = {}
    for row in far_rows:
        key = (str(row.get("model")), str(row.get("quantity")).replace("_far_mass", ""), int(safe_float(row.get("layer"))))
        grouped_values.setdefault(key, []).append(safe_float(row.get("share")))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (model, quantity, layer), values in grouped_values.items():
        grouped.setdefault((model, quantity), []).append({"layer": layer, "share": float(np.nanmean(values))})
    for (model, quantity), items in sorted(grouped.items()):
        items = sorted(items, key=lambda r: safe_float(r.get("layer")))
        ax.plot(
            [safe_float(r.get("layer")) for r in items],
            [safe_float(r.get("share")) for r in items],
            marker="o",
            linewidth=1.5,
            label=f"{model} {quantity}",
        )
    ax.set_title("Step 2: layer-resolved far carriage by channel")
    ax.set_xlabel("GRIT attention layer")
    ax.set_ylabel("Far-mass share")
    ax.legend(frameon=False, fontsize=8)
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step2_layer_resolved_channel_split.png", dpi=dpi)
    fig.savefig(figures / "step2_layer_resolved_channel_split.pdf")
    plt.close(fig)


def render_head_resolved_carriage(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    final_rows = [
        r
        for r in rows
        if bool(r.get("headline_final_layer"))
        and str(r.get("quantity")) == "head_far_carriage"
        and math.isfinite(safe_float(r.get("head")))
        and math.isfinite(safe_float(r.get("mass")))
    ]
    if not final_rows:
        return
    models = sorted({str(r.get("model")) for r in final_rows})
    heads = sorted({int(safe_float(r.get("head"))) for r in final_rows})
    if not models or not heads:
        return
    x = np.arange(len(heads), dtype=float)
    width = min(0.8 / max(1, len(models)), 0.35)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for model_idx, model in enumerate(models):
        masses = []
        for head in heads:
            vals = [
                safe_float(r.get("mass"))
                for r in final_rows
                if str(r.get("model")) == model and int(safe_float(r.get("head"))) == head
            ]
            vals = [v for v in vals if math.isfinite(v)]
            masses.append(float(np.nansum(vals)) if vals else 0.0)
        total = max(float(np.nansum(masses)), EPS)
        values = [v / total for v in masses]
        offset = (model_idx - (len(models) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=model)
    ax.set_title("Which heads carry the long-range signal: far carriage by attention head")
    ax.set_xlabel("Final GRIT attention head")
    ax.set_ylabel("Share of final-layer far channel carriage")
    ax.set_xticks(x)
    ax.set_xticklabels([str(head) for head in heads])
    ax.legend(frameon=False, fontsize=8)
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step2_head_resolved_far_carriage.png", dpi=dpi)
    fig.savefig(figures / "step2_head_resolved_far_carriage.pdf")
    plt.close(fig)


def render_step2_faithfulness(
    rows: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    *,
    dpi: int,
    attention_quantity: str = "attention_last",
    attention_label: str = "last-layer attention",
    faithfulness_stem: str = "step2_attention_faithfulness",
    far_mass_stem: str = "step2_far_mass_attention_vs_carriage",
) -> None:
    rows = [r for r in rows if str(r.get("attention_quantity", "attention_last")) == attention_quantity]
    if not rows:
        return
    far_bins = sorted({str(r.get("distance_bin")) for r in rows if str(r.get("distance_bin", "")).startswith(">")})
    far_text = f"d > {far_bins[0][1:].strip()}" if far_bins else "d > τ"
    summary: list[dict[str, Any]] = []
    for model in sorted(set(str(r["model"]) for r in rows)):
        model_rows = [r for r in rows if str(r["model"]) == model]
        for estimator in sorted(set(str(r.get("carriage_estimator", "ig")) for r in model_rows)):
            estimator_rows = [r for r in model_rows if str(r.get("carriage_estimator", "ig")) == estimator]
            for quantity in sorted(set(str(r.get("attention_quantity", "attention_last")) for r in estimator_rows)):
                items = [r for r in estimator_rows if str(r.get("attention_quantity", "attention_last")) == quantity]
                overall = [r for r in items if str(r.get("distance_bin")) == "overall"]
                far = [r for r in items if str(r.get("distance_bin", "")).startswith(">")]
                if not overall:
                    continue
                summary.append(
                    {
                        "model": model,
                        "carriage_estimator": estimator,
                        "attention_quantity": quantity,
                        "label": f"{model} {attention_label.replace(' attention', '')} vs {estimator}",
                        "spearman_overall": float(np.nanmean([safe_float(r["spearman"]) for r in overall])),
                        "spearman_far": float(np.nanmean([safe_float(r["spearman"]) for r in far])),
                        "attention_far_mass": float(np.nanmean([safe_float(r["attention_far_mass"]) for r in overall])),
                        "carriage_far_mass": float(np.nanmean([safe_float(r["carriage_far_mass"]) for r in overall])),
                    }
                )
    if not summary:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2), constrained_layout=True)
    for ax, key, title in [
        (axes[0], "spearman_overall", "Overall"),
        (axes[1], "spearman_far", "Far bin"),
    ]:
        labels = [str(r["label"]) for r in summary]
        values = [safe_float(r[key]) for r in summary]
        y = np.arange(len(labels), dtype=float)
        ax.barh(y, values, color="#4c78a8")
        ax.axvline(0, color="#555555", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(title)
        ax.set_xlabel(f"Spearman({attention_label}, |C|), per-molecule mean")
    figures = ensure_dir(artifact_root / "figures")
    fig.suptitle(f"Step 2: {attention_label} faithfulness to carriage")
    fig.savefig(figures / f"{faithfulness_stem}.png", dpi=dpi)
    fig.savefig(figures / f"{faithfulness_stem}.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    for r in summary:
        marker = "o" if str(r.get("carriage_estimator")) == "ig" else "s"
        ax.scatter([r["attention_far_mass"]], [r["carriage_far_mass"]], s=55, marker=marker, label=r["label"])
    ax.plot([0, 1], [0, 1], "--", color="#555555")
    ax.set_title(f"Step 2: far-mass, {attention_label} vs carriage")
    ax.set_xlabel(f"{attention_label.capitalize()} far-mass ({far_text})")
    ax.set_ylabel(f"Carriage far-mass ({far_text})")
    ax.legend(frameon=False, fontsize=6, loc="best")
    fig.savefig(figures / f"{far_mass_stem}.png", dpi=dpi)
    fig.savefig(figures / f"{far_mass_stem}.pdf")
    plt.close(fig)


def render_step2_attention_erasure(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    clean = [
        r
        for r in rows
        if math.isfinite(safe_float(r.get("fraction_removed")))
        and math.isfinite(safe_float(r.get("abs_delta_prediction")))
        and int(safe_float(r.get("edges_available"))) > 0
    ]
    if not clean:
        return
    figures = ensure_dir(artifact_root / "figures")
    regions = ["far"]
    if any(str(r.get("region")) == "near" for r in clean):
        regions.append("near")
    fig, axes = plt.subplots(1, len(regions), figsize=(6.4 * len(regions), 4.6), squeeze=False, constrained_layout=True)
    palette = {"attention": "#4c78a8", "carriage": "#f58518", "random": "#6b7280"}
    labels = {"attention": "attention-ranked", "carriage": "carriage-ranked", "random": "random"}
    for ax, region in zip(axes[0], regions):
        region_rows = [r for r in clean if str(r.get("region")) == region]
        for ranker in ("attention", "carriage", "random"):
            rank_rows = [r for r in region_rows if str(r.get("ranker")) == ranker]
            fractions = sorted({safe_float(r.get("fraction_removed")) for r in rank_rows})
            xs: list[float] = []
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for fraction in fractions:
                values = [
                    safe_float(r.get("abs_delta_prediction"))
                    for r in rank_rows
                    if abs(safe_float(r.get("fraction_removed")) - fraction) < 1e-9
                ]
                values = [v for v in values if math.isfinite(v)]
                if not values:
                    continue
                mean, lo, hi = bootstrap_ci(
                    values,
                    seed=9200 + int(round(fraction * 1000)) + 31 * len(ranker) + len(region),
                    draws=500,
                )
                xs.append(fraction)
                means.append(mean)
                lows.append(lo)
                highs.append(hi)
            if xs:
                x = np.asarray(xs, dtype=float)
                y = np.asarray(means, dtype=float)
                lo = np.asarray(lows, dtype=float)
                hi = np.asarray(highs, dtype=float)
                ax.plot(x, y, marker="o", linewidth=1.8, color=palette[ranker], label=labels[ranker])
                ax.fill_between(x, lo, hi, color=palette[ranker], alpha=0.16, linewidth=0)
        title = "Far edges (d > tau)" if region == "far" else "Near edges (0 < d <= tau)"
        ax.set_title(title)
        ax.set_xlabel("Fraction of edges removed")
        ax.set_ylabel("Mean absolute prediction change")
        ax.set_xlim(-0.01, 0.51)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Faithfulness by erasure: prediction change vs edges removed - dense GRIT", fontsize=13)
    fig.savefig(figures / "step2_attention_erasure_faithfulness.png", dpi=dpi)
    fig.savefig(figures / "step2_attention_erasure_faithfulness.pdf")
    plt.close(fig)


def run_step3(models: Sequence[ModelRun], artifact_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    progress("Step 3 start: train/test distance-resolved overfitting")
    analysis_models = [m for m in models if "grit" in m.name.lower()] or list(models)
    if not analysis_models:
        return {"status": "skipped_no_models"}
    cfg = config["steps"]["3"]
    sample_graphs = int(cfg.get("sample_graphs", 200))
    ig_steps = int(config["perturbation"].get("ig_steps", 32))
    batched_vjp = carriage_ig_uses_batched_vjp(config)
    dpi = int(config["figures"]["dpi"])
    seed = int(config.get("seeds", [0])[0])
    rows: list[dict[str, Any]] = []
    for model in analysis_models:
        for split in ["train", "test"]:
            graphs = select_graphs(model.adapter, split, sample_graphs, seed=seed)
            progress(f"Step 3 {model.name}: selected {len(graphs)} {split} graph(s), IG steps={ig_steps}")
            baseline = mean_encoded_baseline(model.adapter, select_baseline_graphs(model.adapter, split, config, sample_graphs, seed=seed))
            cache_hits = 0
            for graph_idx, graph in enumerate(graphs):
                progress_graph("Step 3", model.name, graph_idx, len(graphs), split=split)
                gid = graph_identity(split, graph_idx, graph)
                dist = distance_matrix(graph)
                result = carriage_ig_cached(
                    model,
                    graph,
                    baseline,
                    artifact_root,
                    config,
                    split=split,
                    graph_id=gid,
                    steps=ig_steps,
                    batched_vjp=batched_vjp,
                )
                cache_hits += int(bool(result.get("carriage_cache_hit")))
                c = result["carriage"]
                for row in distance_profile(c.abs(), dist):
                    rows.append({"model": model.name, "split": split, "graph_id": gid, **row})
            if cache_hits:
                progress(f"Step 3 {model.name} {split}: reused cached IG carriage for {cache_hits}/{len(graphs)} graph(s)")
    write_csv(artifact_root / "metrics" / "step3_train_test_profiles.csv", rows)
    gap_rows = train_test_gap_rows(rows, int(config.get("primary_tau", 3)))
    write_csv(artifact_root / "metrics" / "step3_train_minus_test_gap.csv", gap_rows)
    render_step3(rows, gap_rows, artifact_root, dpi=dpi)
    progress("Step 3 complete: metrics and figures written")
    return {"status": "complete", "models": [m.name for m in analysis_models], "profile_rows": len(rows)}


def train_test_gap_rows(rows: Sequence[Mapping[str, Any]], tau: int) -> list[dict[str, Any]]:
    out = []
    for model in sorted({str(r.get("model")) for r in rows}):
        model_rows = [r for r in rows if str(r.get("model")) == model]
        distances = sorted({int(safe_float(r["distance"])) for r in model_rows if math.isfinite(safe_float(r["distance"]))})
        for d in distances:
            train = [
                safe_float(r["share"])
                for r in model_rows
                if r.get("split") == "train" and int(safe_float(r["distance"])) == d
            ]
            test = [
                safe_float(r["share"])
                for r in model_rows
                if r.get("split") == "test" and int(safe_float(r["distance"])) == d
            ]
            if train and test:
                gap, lo, hi = bootstrap_gap_ci(train, test, seed=1000 + int(d) + 97 * len(out))
                out.append(
                    {
                        "model": model,
                        "distance": d,
                        "train_share": float(np.nanmean(train)),
                        "test_share": float(np.nanmean(test)),
                        "gap": gap,
                        "gap_ci_low": lo,
                        "gap_ci_high": hi,
                    }
                )
        far_train = [safe_float(r["share"]) for r in model_rows if r.get("split") == "train" and safe_float(r["distance"]) > tau]
        far_test = [safe_float(r["share"]) for r in model_rows if r.get("split") == "test" and safe_float(r["distance"]) > tau]
        if far_train and far_test:
            gap, lo, hi = bootstrap_gap_ci(far_train, far_test, seed=2000 + int(tau) + 97 * len(out))
            out.append(
                {
                    "model": model,
                    "distance": f">{tau}",
                    "train_share": float(np.nanmean(far_train)),
                    "test_share": float(np.nanmean(far_test)),
                    "gap": gap,
                    "gap_ci_low": lo,
                    "gap_ci_high": hi,
                }
            )
    return out


def bootstrap_gap_ci(train: Sequence[float], test: Sequence[float], *, seed: int, draws: int = 1000) -> tuple[float, float, float]:
    train_arr = np.asarray(train, dtype=np.float64)
    test_arr = np.asarray(test, dtype=np.float64)
    train_arr = train_arr[np.isfinite(train_arr)]
    test_arr = test_arr[np.isfinite(test_arr)]
    if train_arr.size == 0 or test_arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(draws)):
        train_sample = train_arr[rng.integers(0, train_arr.size, size=train_arr.size)]
        test_sample = test_arr[rng.integers(0, test_arr.size, size=test_arr.size)]
        vals.append(float(np.mean(train_sample) - np.mean(test_sample)))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(np.mean(train_arr) - np.mean(test_arr)), float(lo), float(hi)


def render_step2_attention_graph_ablation(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    clean = [
        r
        for r in rows
        if str(r.get("selector")) in ("attention_graph", "carriage", "random")
        and math.isfinite(safe_float(r.get("task_mae")))
        and math.isfinite(safe_float(r.get("fraction_ablated")))
    ]
    if not clean:
        return
    models = sorted({str(r.get("model")) for r in clean})
    model = "dense_grit" if "dense_grit" in models else models[0]
    msub = [r for r in clean if str(r.get("model")) == model]
    regions = sorted({str(r.get("region")) for r in msub}, reverse=True)  # far, then near
    aggregation = str(msub[0].get("aggregation", "mean")) if msub else "mean"
    colors = {"carriage": "#f58518", "attention_graph": "#4c78a8", "random": "#7f7f7f"}
    labels = {"carriage": "carriage", "attention_graph": "attention graph", "random": "random"}
    fig, axes = plt.subplots(1, len(regions), figsize=(6.6 * len(regions), 4.6), constrained_layout=True, squeeze=False)
    for ax, region in zip(axes[0], regions):
        rsub = [r for r in msub if str(r.get("region")) == region]
        for selector in ("attention_graph", "carriage", "random"):
            pts: dict[float, list[float]] = {}
            for r in rsub:
                if str(r.get("selector")) != selector:
                    continue
                pts.setdefault(safe_float(r.get("fraction_ablated")), []).append(safe_float(r.get("task_mae")))
            fracs = sorted(f for f in pts if math.isfinite(f))
            if not fracs:
                continue
            means = [float(np.nanmean(pts[f])) for f in fracs]
            los: list[float] = []
            his: list[float] = []
            for f in fracs:
                arr = np.asarray([v for v in pts[f] if math.isfinite(v)], dtype=float)
                if arr.size >= 5:
                    rng = np.random.default_rng(13)
                    bs = [float(np.mean(rng.choice(arr, size=arr.size, replace=True))) for _ in range(300)]
                    lo, hi = np.percentile(bs, [2.5, 97.5])
                    los.append(float(lo))
                    his.append(float(hi))
                else:
                    los.append(float("nan"))
                    his.append(float("nan"))
            ax.plot(fracs, means, marker="o", linewidth=1.8, color=colors[selector], label=labels[selector])
            lo_arr = np.asarray(los)
            hi_arr = np.asarray(his)
            m = np.isfinite(lo_arr) & np.isfinite(hi_arr)
            if m.any():
                ax.fill_between(np.asarray(fracs)[m], lo_arr[m], hi_arr[m], color=colors[selector], alpha=0.15)
        clean_mae_vals = [safe_float(r.get("clean_mae")) for r in rsub if math.isfinite(safe_float(r.get("clean_mae")))]
        if clean_mae_vals:
            ax.axhline(float(np.mean(clean_mae_vals)), color="#555555", linestyle="--", linewidth=1, label="clean MAE")
        ax.set_title(f"{region} sources")
        ax.set_xlabel("fraction of source nodes corrupted")
        ax.set_ylabel("task MAE  |pred - y|")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Attention-graph vs carriage selection — input-content ablation vs task MAE "
        f"({model}, agg={aggregation}; higher = selected content more task-critical)"
    )
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step2_attention_graph_vs_carriage_ablation.png", dpi=dpi)
    fig.savefig(figures / "step2_attention_graph_vs_carriage_ablation.pdf")
    plt.close(fig)


def render_step3(rows: Sequence[Mapping[str, Any]], gap_rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    render_distance_profile(
        [{**r, "model": f"{r['model']} {r['split']}", "quantity": "carriage"} for r in rows],
        artifact_root,
        "step3_train_vs_test_carriage",
        "Step 3: carriage by distance, train vs test",
        dpi=dpi,
    )
    numeric = [r for r in gap_rows if isinstance(r.get("distance"), int)]
    if numeric:
        fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
        for model in sorted({str(r.get("model")) for r in numeric}):
            model_rows = sorted([r for r in numeric if str(r.get("model")) == model], key=lambda r: safe_float(r["distance"]))
            x = np.asarray([safe_float(r["distance"]) for r in model_rows], dtype=float)
            y = np.asarray([safe_float(r["gap"]) for r in model_rows], dtype=float)
            lo = np.asarray([safe_float(r.get("gap_ci_low")) for r in model_rows], dtype=float)
            hi = np.asarray([safe_float(r.get("gap_ci_high")) for r in model_rows], dtype=float)
            err_low = np.maximum(0.0, y - lo)
            err_high = np.maximum(0.0, hi - y)
            ax.errorbar(x, y, yerr=np.vstack([err_low, err_high]), marker="o", capsize=3, label=model)
        ax.axhline(0, color="#555555", linestyle="--", linewidth=1)
        ax.set_title("Step 3: train-minus-test carriage gap")
        ax.set_xlabel("Molecular hop distance")
        ax.set_ylabel("Train share - test share")
        ax.legend(frameon=False, fontsize=8)
        figures = ensure_dir(artifact_root / "figures")
        fig.savefig(figures / "step3_train_minus_test_gap.png", dpi=dpi)
        fig.savefig(figures / "step3_train_minus_test_gap.pdf")
        plt.close(fig)


def mediator_cut(graph: Any, target: int, source: int) -> list[int]:
    return minimum_vertex_cut(pyg_graph_view(graph), int(target), int(source))


def cut_disconnects_pair(graph: Any, target: int, source: int, cut: Sequence[int]) -> bool:
    """Return whether removing ``cut`` disconnects source from target on bonds."""

    target = int(target)
    source = int(source)
    cut_nodes = [int(v) for v in cut if int(v) not in {target, source}]
    if target == source or not cut_nodes:
        return False
    g = graph_to_networkx(pyg_graph_view(graph), undirected=True)
    g.remove_nodes_from(cut_nodes)
    if target not in g or source not in g:
        return True
    return not nx.has_path(g, source, target)


def mediator_cut_class(cut: Sequence[int], disconnects_pair: bool) -> str:
    if not cut:
        return "empty_cut"
    if not disconnects_pair:
        return "cut_failed"
    if len(cut) == 1:
        return "single_node_cut"
    return "multi_node_cut"


def random_off_path_node(
    dist: torch.Tensor,
    carrier: int,
    source: int,
    cut: Sequence[int],
    *,
    seed: int,
) -> int | None:
    dist_cpu = dist.detach().cpu().float()
    endpoint_distance = dist_cpu[int(carrier), int(source)]
    if not bool(torch.isfinite(endpoint_distance)):
        return None
    blocked = {int(carrier), int(source), *[int(v) for v in cut]}
    candidates: list[int] = []
    for node in range(int(dist_cpu.size(0))):
        if node in blocked:
            continue
        left = dist_cpu[int(carrier), node]
        right = dist_cpu[node, int(source)]
        on_shortest_path = (
            bool(torch.isfinite(left))
            and bool(torch.isfinite(right))
            and abs(float(left.item() + right.item() - endpoint_distance.item())) <= 1.0e-6
        )
        if not on_shortest_path:
            candidates.append(node)
    if not candidates:
        return None
    rng = random.Random(int(seed))
    return int(rng.choice(candidates))


def patched_ig_pair(
    adapter: OfficialGRITAdapter,
    graph: Any,
    mean_baseline: torch.Tensor,
    clean_cache: Any,
    readout_grad: torch.Tensor,
    *,
    carrier: int,
    source: int,
    clamp_nodes: Sequence[int],
    steps: int,
    clamp_until_layer: Optional[int] = None,
    clamp_mode: str = "detach",
    clean_encoded_override: Optional[torch.Tensor] = None,
    baseline_override: Optional[torch.Tensor] = None,
) -> float:
    clean_encoded = (
        clean_encoded_override.detach().to(adapter.device)
        if clean_encoded_override is not None
        else adapter.encoded_node_states(graph).detach()
    )
    base = (
        baseline_override.detach().to(device=adapter.device, dtype=clean_encoded.dtype)
        if baseline_override is not None
        else expanded_baseline(clean_encoded, mean_baseline)
    )
    if tuple(base.shape) != tuple(clean_encoded.shape):
        base = base.expand_as(clean_encoded)
    delta = clean_encoded - base
    total = 0.0
    for alpha_idx in range(1, int(steps) + 1):
        point = (base + (float(alpha_idx) / float(steps)) * delta).detach().requires_grad_(True)
        cache = adapter.patch_hidden_states_from_encoded_content(
            graph,
            point,
            clamp_nodes=clamp_nodes,
            clean_cache=clean_cache,
            clamp_until_layer=clamp_until_layer,
            clamp_mode=clamp_mode,
            retain_grad=False,
        )
        scalar = (cache.final_node_states[int(carrier)] * readout_grad.to(adapter.device)[int(carrier)]).sum()
        (grad,) = torch.autograd.grad(scalar, point, retain_graph=False, create_graph=False)
        total += float((grad[int(source)].detach() * delta.to(adapter.device)[int(source)]).sum().cpu().item()) / float(steps)
    return total


def run_step4(models: Sequence[ModelRun], artifact_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    progress("Step 4 start: mediator patching")
    cfg = config["steps"]["4"]
    sample_graphs = int(cfg.get("sample_graphs", 100))
    max_pairs = optional_pair_limit(cfg.get("max_far_pairs_per_graph", 64), 64)
    depth_pairs = int(cfg.get("depth_pairs_per_graph", 8))
    min_effect_abs = float(cfg.get("min_effect_abs", 1.0e-6))
    use_signal_gate = signal_gate_enabled(config, "4")
    gate_quantile = signal_gate_quantile(config, "4")
    gate_floor_cap_fraction = signal_gate_reference_floor_cap_fraction(config, "4")
    clamp_mode = str(cfg.get("clamp_mode", "detach")).strip().lower()
    run_clamp_negative_control = bool(cfg.get("run_clamp_negative_control", True))
    run_clamp_mode_comparison = bool(cfg.get("run_clamp_mode_comparison", False))
    clamp_mode_comparison_modes = [
        str(mode).strip().lower()
        for mode in cfg.get("clamp_mode_comparison_modes", ["detach", "overwrite"])
        if str(mode).strip()
    ]
    clamp_mode_comparison_modes = [mode for mode in clamp_mode_comparison_modes if mode in {"detach", "overwrite"}]
    if not clamp_mode_comparison_modes:
        clamp_mode_comparison_modes = [clamp_mode]
    clamp_mode_comparison_max_pairs_per_model = optional_pair_limit(
        cfg.get("clamp_mode_comparison_max_pairs_per_model", 16),
        16,
    )
    composed_reference_max_direct_fraction = float(cfg.get("composed_reference_max_direct_fraction", 0.20))
    tau = int(config.get("primary_tau", 3))
    if bool(cfg.get("all_distance", True)):
        min_distance = 2
    else:
        min_distance = int(cfg.get("min_distance", 2))
    min_distance = max(2, int(min_distance))
    stratify_by_distance = bool(cfg.get("stratify_by_distance", True))
    max_distance_raw = cfg.get("max_distance")
    max_distance = (
        None
        if max_distance_raw is None or str(max_distance_raw).strip().lower() in {"", "none", "all", "null"}
        else int(max_distance_raw)
    )
    onset_fraction_threshold = float(cfg.get("onset_fraction_threshold", 0.50))
    ig_steps = int(config["perturbation"].get("ig_steps", 32))
    batched_vjp = carriage_ig_uses_batched_vjp(config)
    seed = int(config.get("seeds", [0])[0])
    dpi = int(config["figures"]["dpi"])
    rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    clamp_mode_rows: list[dict[str, Any]] = []
    tensors: dict[str, Any] = {}
    signal_gate_rows: list[dict[str, Any]] = []
    analytic_patching_check: dict[str, Any] = {"status": "not_run"}
    if bool(cfg.get("run_analytic_patching_check", True)):
        try:
            from graph_specialisation_metrics.method_validation import DEFAULT_CONFIG as VALIDATION_DEFAULT_CONFIG
            from graph_specialisation_metrics.method_validation import run_patching_check

            analytic_metrics = run_patching_check(VALIDATION_DEFAULT_CONFIG, artifact_root, seed)
            analytic_patching_check = {"status": "complete", **analytic_metrics}
        except Exception as exc:
            analytic_patching_check = {"status": "failed", "error": str(exc)}
        write_json(artifact_root / "metrics" / "step4_analytic_patching_check_summary.json", analytic_patching_check)
    if not bool(cfg.get("run_mediator_patching", True)):
        progress("Step 4 mediator patching skipped by config; running only symbolic/structural RRWP probes")
        write_csv(artifact_root / "metrics" / "step4_mediator_patching.csv", rows)
        write_csv(artifact_root / "metrics" / "step4_signal_gate.csv", signal_gate_rows)
        write_csv(artifact_root / "metrics" / "step4_depth_schedule.csv", depth_rows)
        write_csv(artifact_root / "metrics" / "step4_clamp_mode_comparison.csv", clamp_mode_rows)
        write_csv(artifact_root / "metrics" / "step4_mediator_validation_summary.csv", [])
        write_csv(artifact_root / "metrics" / "step4_noncomposable_excess_over_composed.csv", [])
        write_csv(artifact_root / "metrics" / "step4_clamp_negative_control_summary.csv", [])
        write_csv(artifact_root / "metrics" / "step4_verified_separating_cuts_summary.csv", [])
        write_csv(artifact_root / "metrics" / "step4_clamp_mode_comparison_summary.csv", [])
        write_csv(artifact_root / "metrics" / "step4_depth_magnitude_summary.csv", [])
        write_csv(artifact_root / "metrics" / "step4_pathway_interference_summary.csv", [])
        write_csv(artifact_root / "metrics" / "step4_clamp_validation_d2_summary.csv", [])
        write_csv(artifact_root / "metrics" / "step4_mediator_cut_class_summary.csv", [])
        write_csv(artifact_root / "metrics" / "step4_single_cut_composed_reference_diagnostic.csv", [])
        atomic_torch_save(artifact_root / "tensors" / "step4_mediator_patching.pt", tensors)
        rrwp_ablation_status: dict[str, Any] = {"status": "skipped"}
        if bool(cfg.get("run_symbolic_structural_carriage", False)):
            try:
                run_symbolic_structural_probe(models, artifact_root, config)
            except Exception as exc:
                progress(f"Step 4 symbolic/structural probe failed: {exc}")
        if bool(cfg.get("run_rrwp_distance_ablation", False)):
            try:
                rrwp_ablation_status = run_rrwp_distance_ablation_probe(models, artifact_root, config)
            except Exception as exc:
                rrwp_ablation_status = {"status": "failed", "error": str(exc)}
                progress(f"Step 4 RRWP distance-bin ablation failed: {exc}")
        progress("Step 4 complete: mediator patching skipped; structural/RRWP figures written")
        return {
            "status": "complete",
            "mediator_patching_skipped": True,
            "patch_rows": 0,
            "depth_rows": 0,
            "signal_gate_rows": 0,
            "signal_gate_pass_rows": 0,
            "signal_gate_enabled": use_signal_gate,
            "analytic_patching_check": analytic_patching_check,
            "rrwp_distance_ablation": rrwp_ablation_status,
        }
    onehop_floor = empirical_onehop_noise_floor(
        models,
        artifact_root,
        config,
        sample_graphs=sample_graphs,
        split="test",
        ig_steps=ig_steps,
        seed=seed,
        batched_vjp=batched_vjp,
        min_floor=min_effect_abs,
        quantile=gate_quantile,
    ) if use_signal_gate else float(min_effect_abs)
    progress(
        f"Step 4 signal gate: enabled={use_signal_gate}, quantile={gate_quantile:.2f}, "
        f"onehop_empirical_floor={onehop_floor:.3g}, "
        f"reference_floor_cap_fraction={gate_floor_cap_fraction:.3g}"
    )
    for model in models:
        graphs = select_graphs(model.adapter, "test", sample_graphs, seed=seed)
        max_pairs_label = "all" if max_pairs is None else str(max_pairs)
        progress(
            f"Step 4 {model.name}: selected {len(graphs)} test graph(s), "
            f"distance>= {min_distance}, max_pairs_per_graph={max_pairs_label}, IG steps={ig_steps}"
        )
        baseline = mean_encoded_baseline(model.adapter, select_baseline_graphs(model.adapter, "test", config, sample_graphs, seed=seed))
        cache_hits = 0
        clamp_mode_pairs_used = 0
        for graph_idx, graph in enumerate(graphs):
            progress_graph("Step 4", model.name, graph_idx, len(graphs))
            gid = graph_identity("test", graph_idx, graph)
            dist = distance_matrix(graph)
            result = carriage_ig_cached(
                model,
                graph,
                baseline,
                artifact_root,
                config,
                split="test",
                graph_id=gid,
                steps=ig_steps,
                batched_vjp=batched_vjp,
                capture_layer_inputs=True,
            )
            cache_hits += int(bool(result.get("carriage_cache_hit")))
            c = result["carriage"]
            clean_cache = result["clean_cache"]
            readout_grad = result["readout_gradient"]
            clean_encoded = result["clean_encoded"].to(model.adapter.device)
            base_encoded = result["baseline"].to(model.adapter.device)
            selected = distance_pairs(
                dist,
                min_distance=min_distance,
                max_distance=max_distance,
                max_pairs=max_pairs,
                seed=seed + graph_idx,
                stratify_by_distance=stratify_by_distance,
            )
            direct_matrix = torch.full_like(c, float("nan"))
            accepted_pair_idx = 0
            for pair_idx, (carrier, source) in enumerate(selected):
                pair_interval = progress_interval(len(selected), target_messages=4)
                if pair_idx == 0 or pair_idx + 1 == len(selected) or (pair_idx + 1) % pair_interval == 0:
                    progress(f"Step 4 {model.name} graph {graph_idx + 1}/{len(graphs)}: patched pair {pair_idx + 1}/{len(selected)}")
                gate_pass, signal_floor, effect_abs = pair_passes_signal_gate(
                    c,
                    dist,
                    carrier,
                    source,
                    enabled=use_signal_gate,
                    quantile=gate_quantile,
                    min_floor=min_effect_abs,
                    reference_floor=onehop_floor,
                    reference_floor_cap_fraction=gate_floor_cap_fraction,
                )
                signal_gate_rows.append(
                    {
                        "model": model.name,
                        "graph_id": gid,
                        "carrier": carrier,
                        "source": source,
                        "distance": float(dist[carrier, source].item()),
                        "carriage": float(c[carrier, source].item()),
                        "effect_abs": effect_abs,
                        "signal_floor": signal_floor,
                        "signal_gate_pass": gate_pass,
                        "signal_gate_quantile": gate_quantile,
                        "onehop_empirical_floor": onehop_floor,
                        "reference_floor_cap_fraction": gate_floor_cap_fraction,
                        "patch_min_distance": min_distance,
                        "patch_max_distance": max_distance if max_distance is not None else "",
                    }
                )
                # Do NOT hard-gate on the noise floor. Patch every selected (distance-stratified,
                # capped) pair whose unclamped carriage is above the trivial-zero threshold, and let
                # the ratio-of-sums aggregation (weighted_direct_fraction) down-weight noise by
                # magnitude. Skipping only truly-zero pairs (|C| < min_effect_abs) avoids 0/0 without
                # discarding real-but-small signal, and keeps the estimator on ALL measured samples.
                if abs(float(c[carrier, source].item())) < float(min_effect_abs):
                    continue
                cut = mediator_cut(graph, carrier, source)
                if not cut:
                    continue
                disconnects_pair = cut_disconnects_pair(graph, carrier, source, cut)
                cut_class = mediator_cut_class(cut, disconnects_pair)
                direct = patched_ig_pair(
                    model.adapter,
                    graph,
                    baseline,
                    clean_cache,
                    readout_grad,
                    carrier=carrier,
                    source=source,
                    clamp_nodes=cut,
                    steps=ig_steps,
                    clamp_mode=clamp_mode,
                    clean_encoded_override=clean_encoded,
                    baseline_override=base_encoded,
                )
                direct_matrix[carrier, source] = direct
                unclamped = float(c[carrier, source].item())
                direct_fraction, nontrivial_effect = direct_fraction_value(direct, unclamped, min_effect_abs=min_effect_abs)
                rows.append(
                    {
                        "model": model.name,
                        "graph_id": gid,
                        "clamp_type": "cut",
                        "carrier": carrier,
                        "source": source,
                        "distance": float(dist[carrier, source].item()),
                        "cut_size": len(cut),
                        "cut_disconnects_pair": disconnects_pair,
                        "cut_class": cut_class,
                        "clamp_nodes": ",".join(str(v) for v in cut),
                        "clamp_mode": clamp_mode,
                        "unclamped": unclamped,
                        "direct": direct,
                        "composed": unclamped - direct,
                        "direct_fraction": direct_fraction,
                        "effect_abs": abs(unclamped),
                        "min_effect_abs": min_effect_abs,
                        "signal_floor": signal_floor,
                        "signal_gate_pass": gate_pass,
                        "signal_gate_quantile": gate_quantile,
                        "onehop_empirical_floor": onehop_floor,
                        "reference_floor_cap_fraction": gate_floor_cap_fraction,
                        "patch_min_distance": min_distance,
                        "patch_max_distance": max_distance if max_distance is not None else "",
                        "nontrivial_effect": nontrivial_effect,
                    }
                )
                off_path: int | None = None
                primary_off_path_direct: Optional[float] = None
                if run_clamp_negative_control:
                    off_path = random_off_path_node(
                        dist,
                        carrier,
                        source,
                        cut,
                        seed=seed + 4099 * (graph_idx + 1) + 131 * (pair_idx + 1),
                    )
                    if off_path is not None:
                        control_direct = patched_ig_pair(
                            model.adapter,
                            graph,
                            baseline,
                            clean_cache,
                            readout_grad,
                            carrier=carrier,
                            source=source,
                            clamp_nodes=[off_path],
                            steps=ig_steps,
                            clamp_mode=clamp_mode,
                            clean_encoded_override=clean_encoded,
                            baseline_override=base_encoded,
                        )
                        primary_off_path_direct = control_direct
                        control_fraction, control_nontrivial = direct_fraction_value(
                            control_direct,
                            unclamped,
                            min_effect_abs=min_effect_abs,
                        )
                        rows.append(
                            {
                                "model": model.name,
                                "graph_id": gid,
                                "clamp_type": "random_off_path",
                                "carrier": carrier,
                                "source": source,
                                "distance": float(dist[carrier, source].item()),
                                "cut_size": 1,
                                "original_cut_size": len(cut),
                                "original_cut_disconnects_pair": disconnects_pair,
                                "original_cut_class": cut_class,
                                "cut_disconnects_pair": False,
                                "cut_class": "random_off_path_control",
                                "clamp_nodes": str(off_path),
                                "clamp_mode": clamp_mode,
                                "unclamped": unclamped,
                                "direct": control_direct,
                                "composed": unclamped - control_direct,
                                "direct_fraction": control_fraction,
                                "effect_abs": abs(unclamped),
                                "min_effect_abs": min_effect_abs,
                                "signal_floor": signal_floor,
                                "signal_gate_pass": gate_pass,
                                "signal_gate_quantile": gate_quantile,
                                "onehop_empirical_floor": onehop_floor,
                                "reference_floor_cap_fraction": gate_floor_cap_fraction,
                                "patch_min_distance": min_distance,
                                "patch_max_distance": max_distance if max_distance is not None else "",
                                "nontrivial_effect": control_nontrivial,
                            }
                        )
                if run_clamp_mode_comparison and (
                    clamp_mode_comparison_max_pairs_per_model is None
                    or clamp_mode_pairs_used < int(clamp_mode_comparison_max_pairs_per_model)
                ):
                    comparison_specs: list[tuple[str, Sequence[int], Optional[float]]] = [
                        ("cut", cut, direct if clamp_mode in clamp_mode_comparison_modes else None)
                    ]
                    if run_clamp_negative_control and off_path is not None:
                        comparison_specs.append(
                            (
                                "random_off_path",
                                [off_path],
                                primary_off_path_direct if clamp_mode in clamp_mode_comparison_modes else None,
                            )
                        )
                    for compare_mode in clamp_mode_comparison_modes:
                        for compare_clamp_type, compare_nodes, cached_value in comparison_specs:
                            if compare_mode == clamp_mode and cached_value is not None:
                                compare_direct = float(cached_value)
                            else:
                                compare_direct = patched_ig_pair(
                                    model.adapter,
                                    graph,
                                    baseline,
                                    clean_cache,
                                    readout_grad,
                                    carrier=carrier,
                                    source=source,
                                    clamp_nodes=compare_nodes,
                                    steps=ig_steps,
                                    clamp_mode=compare_mode,
                                    clean_encoded_override=clean_encoded,
                                    baseline_override=base_encoded,
                                )
                            compare_fraction, compare_nontrivial = direct_fraction_value(
                                compare_direct,
                                unclamped,
                                min_effect_abs=min_effect_abs,
                            )
                            clamp_mode_rows.append(
                                {
                                    "model": model.name,
                                    "graph_id": gid,
                                    "clamp_type": compare_clamp_type,
                                    "carrier": carrier,
                                    "source": source,
                                    "distance": float(dist[carrier, source].item()),
                                    "cut_size": len(compare_nodes),
                                    "original_cut_size": len(cut),
                                    "original_cut_disconnects_pair": disconnects_pair,
                                    "original_cut_class": cut_class,
                                    "cut_disconnects_pair": disconnects_pair if compare_clamp_type == "cut" else False,
                                    "cut_class": cut_class if compare_clamp_type == "cut" else "random_off_path_control",
                                    "clamp_nodes": ",".join(str(v) for v in compare_nodes),
                                    "clamp_mode": compare_mode,
                                    "unclamped": unclamped,
                                    "direct": compare_direct,
                                    "composed": unclamped - compare_direct,
                                    "direct_fraction": compare_fraction,
                                    "effect_abs": abs(unclamped),
                                    "min_effect_abs": min_effect_abs,
                                    "signal_floor": signal_floor,
                                    "signal_gate_pass": gate_pass,
                                    "nontrivial_effect": compare_nontrivial,
                                }
                            )
                    clamp_mode_pairs_used += 1
                if depth_pairs > 0 and accepted_pair_idx < depth_pairs:
                    layer_count = len((clean_cache.extras or {}).get("layer_input_node_states", []))
                    for layer in range(layer_count):
                        depth_direct = patched_ig_pair(
                            model.adapter,
                            graph,
                            baseline,
                            clean_cache,
                            readout_grad,
                            carrier=carrier,
                            source=source,
                            clamp_nodes=cut,
                            steps=ig_steps,
                            clamp_until_layer=layer,
                            clamp_mode=clamp_mode,
                            clean_encoded_override=clean_encoded,
                            baseline_override=base_encoded,
                        )
                        depth_fraction, depth_nontrivial = direct_fraction_value(
                            depth_direct,
                            unclamped,
                            min_effect_abs=min_effect_abs,
                        )
                        depth_rows.append(
                            {
                                "model": model.name,
                                "graph_id": gid,
                                "carrier": carrier,
                                "source": source,
                                "distance": float(dist[carrier, source].item()),
                                "cut_size": len(cut),
                                "cut_disconnects_pair": disconnects_pair,
                                "cut_class": cut_class,
                                "clamp_until_layer": layer,
                                "clamp_mode": clamp_mode,
                                "unclamped": unclamped,
                                "direct": depth_direct,
                                "composed": unclamped - depth_direct,
                                "direct_fraction": depth_fraction,
                                "effect_abs": abs(unclamped),
                                "min_effect_abs": min_effect_abs,
                                "signal_floor": signal_floor,
                                "signal_gate_pass": gate_pass,
                                "signal_gate_quantile": gate_quantile,
                                "onehop_empirical_floor": onehop_floor,
                                "reference_floor_cap_fraction": gate_floor_cap_fraction,
                                "patch_min_distance": min_distance,
                                "patch_max_distance": max_distance if max_distance is not None else "",
                                "nontrivial_effect": depth_nontrivial,
                            }
                        )
                accepted_pair_idx += 1
            tensors[f"step4/{model.name}/{gid}/unclamped_carriage"] = c
            tensors[f"step4/{model.name}/{gid}/direct_carriage"] = direct_matrix
        if cache_hits:
            progress(f"Step 4 {model.name}: reused cached IG carriage for {cache_hits}/{len(graphs)} graph(s)")
    write_csv(artifact_root / "metrics" / "step4_mediator_patching.csv", rows)
    write_csv(artifact_root / "metrics" / "step4_signal_gate.csv", signal_gate_rows)
    write_csv(artifact_root / "metrics" / "step4_depth_schedule.csv", depth_rows)
    write_csv(artifact_root / "metrics" / "step4_clamp_mode_comparison.csv", clamp_mode_rows)
    # Report the informational noise-floor pass-rate per model. NOTE: this gate no longer
    # FILTERS anything — all patched pairs enter the ratio-of-sums estimator regardless — so a
    # 0% pass-rate does NOT empty the figures. It only flags that a model's far carriage is small
    # relative to the (cross-model, IG-step-sensitive) 1-hop floor; confirm signal vs noise via the
    # composed reference (GIN) and more IG steps, not this rate.
    gate_by_model: dict[str, list[bool]] = {}
    for gate_row in signal_gate_rows:
        gate_by_model.setdefault(str(gate_row.get("model")), []).append(bool(gate_row.get("signal_gate_pass")))
    gate_pass_by_model: dict[str, dict[str, Any]] = {}
    for model in models:
        flags = gate_by_model.get(model.name, [])
        n_pass = sum(1 for f in flags if f)
        rate = (n_pass / len(flags)) if flags else float("nan")
        gate_pass_by_model[model.name] = {"considered": len(flags), "passed": n_pass, "pass_rate": rate}
        progress(
            f"Step 4 noise-floor pass-rate (informational only, does NOT filter) {model.name}: "
            f"{n_pass}/{len(flags)} above floor={onehop_floor:.3g} ({rate:.1%})"
        )
        if flags and n_pass == 0:
            progress(
                f"[INFO] Step 4 {model.name}: 0 pairs exceed the informational noise floor, but its "
                "figures are NOT empty (all patched pairs enter the estimator). This flags small far "
                "carriage relative to that floor; judge signal vs noise via GIN (composed ~0) and IG steps."
            )
    validation_summary = mediator_validation_summary(rows)
    write_csv(artifact_root / "metrics" / "step4_mediator_validation_summary.csv", validation_summary)
    excess_summary = noncomposable_excess_summary(validation_summary)
    write_csv(artifact_root / "metrics" / "step4_noncomposable_excess_over_composed.csv", excess_summary)
    render_step4_noncomposable_excess(excess_summary, artifact_root, dpi=dpi)
    for excess_row in excess_summary:
        progress(
            f"Step 4 non-composable excess {excess_row['model']} vs {excess_row['composed_reference']}: "
            f"direct_fraction={safe_float(excess_row['direct_fraction']):.3f} - "
            f"composed_floor={safe_float(excess_row['composed_floor']):.3f} = "
            f"excess={safe_float(excess_row['excess_over_composed']):.3f} "
            f"[{safe_float(excess_row['excess_ci_low']):.3f}, {safe_float(excess_row['excess_ci_high']):.3f}] "
            f"-> {excess_row['verdict']}"
        )
    clamp_control_summary = clamp_negative_control_summary(rows)
    write_csv(artifact_root / "metrics" / "step4_clamp_negative_control_summary.csv", clamp_control_summary)
    verified_separating_summary = verified_separating_cut_summary(rows)
    write_csv(artifact_root / "metrics" / "step4_verified_separating_cuts_summary.csv", verified_separating_summary)
    clamp_mode_summary = clamp_mode_comparison_summary(clamp_mode_rows)
    write_csv(artifact_root / "metrics" / "step4_clamp_mode_comparison_summary.csv", clamp_mode_summary)
    depth_magnitude_summary = depth_magnitude_summary_rows(depth_rows)
    write_csv(artifact_root / "metrics" / "step4_depth_magnitude_summary.csv", depth_magnitude_summary)
    interference_summary = pathway_interference_summary(rows)
    write_csv(artifact_root / "metrics" / "step4_pathway_interference_summary.csv", interference_summary)
    clamp_d2_summary = clamp_negative_control_summary(rows, distance=2)
    write_csv(artifact_root / "metrics" / "step4_clamp_validation_d2_summary.csv", clamp_d2_summary)
    cut_class_summary = mediator_cut_class_summary(rows)
    write_csv(artifact_root / "metrics" / "step4_mediator_cut_class_summary.csv", cut_class_summary)
    single_cut_summary = mediator_single_cut_diagnostic(rows, models)
    write_csv(artifact_root / "metrics" / "step4_single_cut_composed_reference_diagnostic.csv", single_cut_summary)
    atomic_torch_save(artifact_root / "tensors" / "step4_mediator_patching.pt", tensors)
    onset_rows = onset_depth_rows(depth_rows, threshold=onset_fraction_threshold)
    write_csv(artifact_root / "metrics" / "step4_onset_depth_by_distance.csv", onset_rows)
    render_step4(
        rows,
        depth_rows,
        validation_summary,
        artifact_root,
        dpi=dpi,
        tau=tau,
        onset_rows=onset_rows,
        signal_gate_rows=signal_gate_rows,
        verified_separating_summary=verified_separating_summary,
        clamp_mode_summary=clamp_mode_summary,
        depth_magnitude_summary=depth_magnitude_summary,
        interference_summary=interference_summary,
    )
    render_step4_cut_class_summary(cut_class_summary, artifact_root, dpi=dpi)
    composed_reference_failures = [
        dict(row)
        for row in validation_summary
        if is_composed_reference_model(str(row.get("model", "")), models)
        and math.isfinite(safe_float(row.get("mean_direct_fraction")))
        and safe_float(row.get("mean_direct_fraction")) > composed_reference_max_direct_fraction
    ]
    single_cut_composed_reference_failures = [
        dict(row)
        for row in single_cut_summary
        if str(row.get("validation_role")) == "composed_reference"
        and math.isfinite(safe_float(row.get("mean_direct_fraction")))
        and safe_float(row.get("mean_direct_fraction")) > composed_reference_max_direct_fraction
    ]
    cut_disconnect_failures = [
        dict(row)
        for row in rows
        if str(row.get("clamp_type", "cut")) == "cut"
        and not bool(row.get("cut_disconnects_pair"))
    ]
    status = "complete"
    if analytic_patching_check.get("status") == "failed":
        status = "complete_with_failed_analytic_patching_check"
    elif single_cut_composed_reference_failures:
        status = "complete_with_failed_single_cut_composed_reference_check"
    elif composed_reference_failures:
        status = "complete_with_failed_composed_reference_check"
    if bool(cfg.get("run_symbolic_structural_carriage", False)):
        try:
            run_symbolic_structural_probe(models, artifact_root, config)
        except Exception as exc:
            progress(f"Step 4 symbolic/structural probe failed: {exc}")
    rrwp_ablation_status: dict[str, Any] = {"status": "skipped"}
    if bool(cfg.get("run_rrwp_distance_ablation", False)):
        try:
            rrwp_ablation_status = run_rrwp_distance_ablation_probe(models, artifact_root, config)
        except Exception as exc:
            rrwp_ablation_status = {"status": "failed", "error": str(exc)}
            progress(f"Step 4 RRWP distance-bin ablation failed: {exc}")
    progress(f"Step 4 complete: status={status}, metrics, tensors, and figures written")
    return {
        "status": status,
        "patch_rows": len(rows),
        "depth_rows": len(depth_rows),
        "signal_gate_rows": len(signal_gate_rows),
        "signal_gate_pass_rows": len([r for r in signal_gate_rows if bool(r.get("signal_gate_pass"))]),
        "signal_gate_pass_by_model": gate_pass_by_model,
        "signal_gate_enabled": use_signal_gate,
        "signal_gate_quantile": gate_quantile,
        "onehop_empirical_floor": onehop_floor,
        "patch_min_distance": min_distance,
        "patch_max_distance": max_distance if max_distance is not None else "",
        "stratify_by_distance": stratify_by_distance,
        "onset_fraction_threshold": onset_fraction_threshold,
        "onset_rows": len(onset_rows),
        "clamp_negative_control_rows": len([r for r in rows if str(r.get("clamp_type")) == "random_off_path"]),
        "clamp_mode_comparison_rows": len(clamp_mode_rows),
        "clamp_mode_comparison_enabled": run_clamp_mode_comparison,
        "verified_separating_cut_rows": len(verified_separating_summary),
        "depth_magnitude_rows": len(depth_magnitude_summary),
        "pathway_interference_rows": len(interference_summary),
        "clamp_validation_d2_rows": len(clamp_d2_summary),
        "analytic_patching_check": analytic_patching_check,
        "composed_reference_failures": composed_reference_failures,
        "single_cut_composed_reference_failures": single_cut_composed_reference_failures,
        "cut_disconnect_failure_rows": len(cut_disconnect_failures),
        "cut_class_summary_rows": len(cut_class_summary),
        "single_cut_diagnostic_rows": len(single_cut_summary),
        "composed_reference_max_direct_fraction": composed_reference_max_direct_fraction,
        "clamp_mode": clamp_mode,
        "rrwp_distance_ablation": rrwp_ablation_status,
    }


def is_composed_reference_model(model_name: str, models: Sequence[ModelRun]) -> bool:
    """Return whether a trained model should be used as a hard clamp validator.

    Trained 1-hop/GIN/GCN references are useful local-mechanism context, but
    their far carriage can be at the measurement floor on ZINC. The hard
    composed-reference validator is therefore the analytic patching check, not
    a noisy trained reference ratio.
    """

    _ = model_name, models
    return False


def mediator_validation_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model in sorted(set(str(r.get("model")) for r in rows)):
        model_rows = [
            r
            for r in rows
            if str(r.get("model")) == model
            and str(r.get("clamp_type", "cut")) == "cut"
            and row_is_nontrivial(r)
        ]
        stats = weighted_direct_fraction(model_rows, seed=3000 + len(out), draws=500)
        if int(stats.get("pairs", 0)) <= 0:
            continue
        out.append(
            {
                "model": model,
                "mean_direct_fraction": stats["mean"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "pairs": stats["pairs"],
                "aggregation": "carriage_weighted",
                "validation_role": (
                    "local_reference_context_not_hard_validator"
                    if any(token in model.lower() for token in ("1hop", "gin", "gcn"))
                    else "dense_signal_context"
                ),
            }
        )
    return out


def noncomposable_excess_summary(validation_summary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Excess direct (non-composable) carriage fraction over the composed reference.

    The composed models (GIN / 1-hop), whose far carriage is composed by
    construction, give the empirical "fully composed" direct-fraction level (~0). A
    treatment model (dense GRIT) with genuine attention shortcuts sits ABOVE it;
    one at the composed level has no non-composable transport (clean null). This is
    the per-model-appropriate noise floor for direct carriage, and it turns the
    outcome decision into a single signed margin with a CI.
    """

    def is_composed(name: str) -> bool:
        return any(tok in name.lower() for tok in ("gin", "gcn", "1hop", "one_hop"))

    composed = [r for r in validation_summary if is_composed(str(r.get("model")))]
    treatment = [r for r in validation_summary if not is_composed(str(r.get("model")))]
    if not composed or not treatment:
        return []
    matched = [r for r in composed if str(r.get("model")) == "grit_1hop"]
    if matched:
        floor_row = matched[0]
    else:
        onehop_like = [r for r in composed if "1hop" in str(r.get("model")).lower() or "one_hop" in str(r.get("model")).lower()]
        floor_row = onehop_like[0] if onehop_like else min(composed, key=lambda r: safe_float(r.get("mean_direct_fraction")))
    floor = safe_float(floor_row.get("mean_direct_fraction"))
    floor_hi = safe_float(floor_row.get("ci_high"))
    floor_ref = floor_hi if math.isfinite(floor_hi) else floor
    out: list[dict[str, Any]] = []
    for r in treatment:
        dense_df = safe_float(r.get("mean_direct_fraction"))
        excess = dense_df - floor
        excess_low = safe_float(r.get("ci_low")) - floor_ref
        excess_high = safe_float(r.get("ci_high")) - floor
        out.append(
            {
                "model": str(r.get("model")),
                "composed_reference": str(floor_row.get("model")),
                "direct_fraction": dense_df,
                "composed_floor": floor,
                "excess_over_composed": excess,
                "excess_ci_low": excess_low,
                "excess_ci_high": excess_high,
                "pairs": int(r.get("pairs", 0)),
                "verdict": (
                    "non_composable_transport_above_composed_floor"
                    if math.isfinite(excess_low) and excess_low > 0
                    else "at_or_below_composed_floor_consistent_with_clean_null"
                ),
            }
        )
    return out


def render_step4_noncomposable_excess(summary: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    if not summary:
        return
    labels = [str(r["model"]) for r in summary]
    excess = np.asarray([safe_float(r["excess_over_composed"]) for r in summary], dtype=float)
    lo = np.asarray([safe_float(r["excess_ci_low"]) for r in summary], dtype=float)
    hi = np.asarray([safe_float(r["excess_ci_high"]) for r in summary], dtype=float)
    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    ax.bar(
        labels,
        excess,
        yerr=np.vstack([np.maximum(0.0, excess - lo), np.maximum(0.0, hi - excess)]),
        capsize=4,
        color="#4c78a8",
    )
    ax.axhline(0.0, color="#555555", linewidth=1, label="Matched 1-hop floor")
    ref = str(summary[0].get("composed_reference", "composed"))
    ax.set_title("Excess non-composable carriage over the matched 1-hop control")
    ax.set_xlabel("Treatment model")
    ax.set_ylabel(f"Excess direct fraction over {ref}")
    ax.legend(frameon=False, fontsize=8)
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)
        tick.set_ha("right")
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step4_noncomposable_excess.png", dpi=dpi)
    fig.savefig(figures / "step4_noncomposable_excess.pdf")
    plt.close(fig)


def clamp_negative_control_summary(rows: Sequence[Mapping[str, Any]], *, distance: Optional[int] = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    clamp_types = ["cut", "random_off_path"]
    for model in sorted(set(str(r.get("model")) for r in rows)):
        for clamp_type in clamp_types:
            model_rows = [
                r
                for r in rows
                if str(r.get("model")) == model
                and str(r.get("clamp_type", "cut")) == clamp_type
                and row_is_nontrivial(r)
                and (distance is None or int(round(safe_float(r.get("distance")))) == int(distance))
            ]
            stats = weighted_direct_fraction(model_rows, seed=3700 + len(out), draws=500)
            if int(stats.get("pairs", 0)) <= 0:
                continue
            out.append(
                {
                    "model": model,
                    "clamp_type": clamp_type,
                    "mean_direct_fraction": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "pairs": stats["pairs"],
                    "aggregation": "carriage_weighted",
                    "distance": distance if distance is not None else "",
                }
            )
    return out


def verified_separating_cut_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in sorted(set(str(r.get("model")) for r in rows)):
        model_rows = [
            r
            for r in rows
            if str(r.get("model")) == model
            and str(r.get("clamp_type", "cut")) == "cut"
            and bool(r.get("cut_disconnects_pair"))
            and row_is_nontrivial(r)
        ]
        stats = weighted_direct_fraction(model_rows, seed=3900 + len(out), draws=500)
        if int(stats.get("pairs", 0)) <= 0:
            continue
        out.append(
            {
                "model": model,
                "condition": "verified_separating_cuts_only",
                "mean_direct_fraction": stats["mean"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "pairs": stats["pairs"],
                "aggregation": "carriage_weighted",
                "interpretation": "For a bond-local composed model, faithful cut clamps should be near zero on verified separators.",
            }
        )
    return out


def clamp_mode_comparison_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not rows:
        return out
    models = sorted({str(r.get("model")) for r in rows})
    clamp_modes = sorted({str(r.get("clamp_mode")) for r in rows})
    clamp_types = [t for t in ["cut", "random_off_path"] if any(str(r.get("clamp_type")) == t for r in rows)]
    for model in models:
        for clamp_mode in clamp_modes:
            for clamp_type in clamp_types:
                model_rows = [
                    r
                    for r in rows
                    if str(r.get("model")) == model
                    and str(r.get("clamp_mode")) == clamp_mode
                    and str(r.get("clamp_type")) == clamp_type
                    and row_is_nontrivial(r)
                ]
                stats = weighted_direct_fraction(model_rows, seed=3950 + len(out), draws=500)
                if int(stats.get("pairs", 0)) <= 0:
                    continue
                out.append(
                    {
                        "model": model,
                        "clamp_mode": clamp_mode,
                        "clamp_type": clamp_type,
                        "mean_direct_fraction": stats["mean"],
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                        "pairs": stats["pairs"],
                        "aggregation": "carriage_weighted",
                    }
                )
    return out


def depth_magnitude_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        layer = safe_float(row.get("clamp_until_layer"))
        direct = safe_float(row.get("direct"))
        unclamped = safe_float(row.get("unclamped"))
        distance = safe_float(row.get("distance"))
        if not (math.isfinite(layer) and math.isfinite(direct) and math.isfinite(unclamped) and math.isfinite(distance)):
            continue
        if not row_is_nontrivial(row):
            continue
        band = depth_distance_band(distance)
        grouped.setdefault((str(row.get("model")), band, int(layer)), []).append(row)
    out: list[dict[str, Any]] = []
    for (model, band, layer), group in sorted(grouped.items(), key=lambda item: (item[0][0], depth_band_sort_key(item[0][1]), item[0][2])):
        direct_abs = [abs(safe_float(r.get("direct"))) for r in group]
        unclamped_abs = [abs(safe_float(r.get("unclamped"))) for r in group]
        direct_signed = [safe_float(r.get("direct")) for r in group]
        unclamped_signed = [safe_float(r.get("unclamped")) for r in group]
        out.append(
            {
                "model": model,
                "distance_band": band,
                "clamp_until_layer": layer,
                "mean_abs_direct_clamped": float(np.nanmean(direct_abs)),
                "mean_abs_unclamped": float(np.nanmean(unclamped_abs)),
                "mean_signed_direct_clamped": float(np.nanmean(direct_signed)),
                "mean_signed_unclamped": float(np.nanmean(unclamped_signed)),
                "pairs": len(group),
            }
        )
    return out


def pathway_interference_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get("clamp_type", "cut")) != "cut" or not row_is_nontrivial(row):
            continue
        direct = safe_float(row.get("direct"))
        through = safe_float(row.get("composed"))
        distance = safe_float(row.get("distance"))
        if not (math.isfinite(direct) and math.isfinite(through) and math.isfinite(distance)):
            continue
        grouped.setdefault((str(row.get("model")), depth_distance_band(distance)), []).append(row)
    out: list[dict[str, Any]] = []
    for (model, band), group in sorted(grouped.items(), key=lambda item: (item[0][0], depth_band_sort_key(item[0][1]))):
        direct_vals = np.asarray([safe_float(r.get("direct")) for r in group], dtype=float)
        through_vals = np.asarray([safe_float(r.get("composed")) for r in group], dtype=float)
        unclamped_vals = np.asarray([safe_float(r.get("unclamped")) for r in group], dtype=float)
        opposite = np.asarray([float(d * t < 0.0) for d, t in zip(direct_vals, through_vals)], dtype=float)
        out.append(
            {
                "model": model,
                "distance_band": band,
                "mean_around_cut_C_clamp": float(np.nanmean(direct_vals)),
                "mean_through_cut_C_through": float(np.nanmean(through_vals)),
                "mean_unclamped_total": float(np.nanmean(unclamped_vals)),
                "mean_abs_around_cut": float(np.nanmean(np.abs(direct_vals))),
                "mean_abs_through_cut": float(np.nanmean(np.abs(through_vals))),
                "opposite_sign_share": float(np.nanmean(opposite)) if opposite.size else float("nan"),
                "pairs": len(group),
            }
        )
    return out


def mediator_cut_class_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cut_rows = [
        r
        for r in rows
        if str(r.get("clamp_type", "cut")) == "cut"
        and row_is_nontrivial(r)
    ]
    for model in sorted({str(r.get("model")) for r in cut_rows}):
        for cut_class in ["single_node_cut", "multi_node_cut", "cut_failed", "empty_cut"]:
            model_rows = [
                r
                for r in cut_rows
                if str(r.get("model")) == model and str(r.get("cut_class", "")) == cut_class
            ]
            stats = weighted_direct_fraction(model_rows, seed=4100 + len(out), draws=500)
            if int(stats.get("pairs", 0)) <= 0:
                continue
            out.append(
                {
                    "model": model,
                    "cut_class": cut_class,
                    "mean_direct_fraction": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "pairs": stats["pairs"],
                    "mean_cut_size": float(np.nanmean([safe_float(r.get("cut_size")) for r in model_rows])),
                    "cut_disconnect_failures": sum(not bool(r.get("cut_disconnects_pair")) for r in model_rows),
                    "aggregation": "carriage_weighted",
                }
            )
    return out


def mediator_single_cut_diagnostic(rows: Sequence[Mapping[str, Any]], models: Sequence[ModelRun]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in sorted({str(r.get("model")) for r in rows}):
        model_rows = [
            r
            for r in rows
            if str(r.get("model")) == model
            and str(r.get("clamp_type", "cut")) == "cut"
            and str(r.get("cut_class")) == "single_node_cut"
            and bool(r.get("cut_disconnects_pair"))
            and row_is_nontrivial(r)
        ]
        stats = weighted_direct_fraction(model_rows, seed=4300 + len(out), draws=500)
        role = "composed_reference" if is_composed_reference_model(model, models) else "treatment_context"
        out.append(
            {
                "model": model,
                "validation_role": role,
                "diagnostic": "single_node_cut_pairs",
                "status": "no_pairs" if int(stats.get("pairs", 0)) <= 0 else "complete",
                "mean_direct_fraction": stats["mean"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "pairs": stats["pairs"],
                "aggregation": "carriage_weighted",
                "interpretation": (
                    "If a bond-local composed reference remains near 1 on this subset, "
                    "the clamp is not severing path-composed carriage even when the cut is valid."
                ),
            }
        )
    return out


def render_step4_cut_class_summary(summary: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    clean = [
        r
        for r in summary
        if math.isfinite(safe_float(r.get("mean_direct_fraction")))
        and safe_float(r.get("pairs")) > 0
    ]
    if not clean:
        return
    models = sorted({str(r.get("model")) for r in clean})
    classes = [c for c in ["single_node_cut", "multi_node_cut", "cut_failed"] if any(str(r.get("cut_class")) == c for r in clean)]
    if not models or not classes:
        return
    x = np.arange(len(models), dtype=float)
    width = min(0.8 / max(1, len(classes)), 0.28)
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    labels = {
        "single_node_cut": "Single-node cut",
        "multi_node_cut": "Multi-node cut",
        "cut_failed": "Cut did not separate",
    }
    for idx, cut_class in enumerate(classes):
        values = []
        err_low = []
        err_high = []
        for model in models:
            row = next((r for r in clean if str(r.get("model")) == model and str(r.get("cut_class")) == cut_class), None)
            mean = safe_float(row.get("mean_direct_fraction")) if row else float("nan")
            lo = safe_float(row.get("ci_low")) if row else float("nan")
            hi = safe_float(row.get("ci_high")) if row else float("nan")
            values.append(mean)
            err_low.append(max(0.0, mean - lo) if math.isfinite(mean) and math.isfinite(lo) else 0.0)
            err_high.append(max(0.0, hi - mean) if math.isfinite(mean) and math.isfinite(hi) else 0.0)
        offset = (idx - (len(classes) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            [0.0 if not math.isfinite(v) else v for v in values],
            width=width,
            yerr=np.vstack([err_low, err_high]),
            capsize=3,
            label=labels.get(cut_class, cut_class),
        )
    ax.axhline(0.0, color="#555555", linewidth=1)
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1, label="No path blocked")
    ax.set_title("Step 4: direct fraction by verified cut class")
    ax.set_xlabel("Model")
    ax.set_ylabel("Carriage-weighted direct fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(frameon=False, fontsize=8)
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step4_direct_fraction_by_cut_class.png", dpi=dpi)
    fig.savefig(figures / "step4_direct_fraction_by_cut_class.pdf")
    plt.close(fig)


def render_step4_clamp_negative_control(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    summary = clamp_negative_control_summary(rows)
    if not summary:
        return
    models = sorted({str(r.get("model")) for r in summary})
    clamp_types = ["cut", "random_off_path"]
    labels = {"cut": "Cut clamp", "random_off_path": "Random off-path clamp"}
    x = np.arange(len(models), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for idx, clamp_type in enumerate(clamp_types):
        values = []
        err_low = []
        err_high = []
        for model in models:
            row = next(
                (r for r in summary if str(r.get("model")) == model and str(r.get("clamp_type")) == clamp_type),
                None,
            )
            mean = safe_float(row.get("mean_direct_fraction")) if row else float("nan")
            lo = safe_float(row.get("ci_low")) if row else float("nan")
            hi = safe_float(row.get("ci_high")) if row else float("nan")
            values.append(mean)
            err_low.append(max(0.0, mean - lo) if math.isfinite(mean) and math.isfinite(lo) else 0.0)
            err_high.append(max(0.0, hi - mean) if math.isfinite(mean) and math.isfinite(hi) else 0.0)
        offset = (idx - 0.5) * width
        ax.bar(
            x + offset,
            [0.0 if not math.isfinite(v) else v for v in values],
            width=width,
            yerr=np.vstack([err_low, err_high]),
            capsize=4,
            label=labels.get(clamp_type, clamp_type),
        )
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1, label="No effect from clamp")
    ax.set_title("Step 4: clamp negative control")
    ax.set_xlabel("Model")
    ax.set_ylabel("Carriage-weighted direct fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(frameon=False, fontsize=8)
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step4_clamp_negative_control.png", dpi=dpi)
    fig.savefig(figures / "step4_clamp_negative_control.pdf")
    plt.close(fig)


def render_step4_clamp_validation_d2(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    summary = clamp_negative_control_summary(rows, distance=2)
    if not summary:
        return
    models = sorted({str(r.get("model")) for r in summary})
    clamp_types = ["cut", "random_off_path"]
    labels = {"cut": "Cut clamp", "random_off_path": "Random off-path clamp"}
    x = np.arange(len(models), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for idx, clamp_type in enumerate(clamp_types):
        values = []
        err_low = []
        err_high = []
        for model in models:
            row = next((r for r in summary if str(r.get("model")) == model and str(r.get("clamp_type")) == clamp_type), None)
            mean = safe_float(row.get("mean_direct_fraction")) if row else float("nan")
            lo = safe_float(row.get("ci_low")) if row else float("nan")
            hi = safe_float(row.get("ci_high")) if row else float("nan")
            values.append(mean)
            err_low.append(max(0.0, mean - lo) if math.isfinite(mean) and math.isfinite(lo) else 0.0)
            err_high.append(max(0.0, hi - mean) if math.isfinite(mean) and math.isfinite(hi) else 0.0)
        offset = (idx - 0.5) * width
        ax.bar(
            x + offset,
            [0.0 if not math.isfinite(v) else v for v in values],
            width=width,
            yerr=np.vstack([err_low, err_high]),
            capsize=4,
            label=labels.get(clamp_type, clamp_type),
        )
    ax.axhline(0.0, color="#777777", linewidth=1, label="Fully composed")
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1, label="No clamp effect")
    ax.set_title("Clamp validation on real signal: direct fraction on d=2 pairs (cut vs off-path)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Carriage-weighted direct fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(frameon=False, fontsize=8)
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step4_clamp_validation_d2.png", dpi=dpi)
    fig.savefig(figures / "step4_clamp_validation_d2.pdf")
    plt.close(fig)


def render_step4_verified_separating_cuts(summary: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    clean = [r for r in summary if math.isfinite(safe_float(r.get("mean_direct_fraction")))]
    if not clean:
        return
    labels = [str(r.get("model")) for r in clean]
    y = np.asarray([safe_float(r.get("mean_direct_fraction")) for r in clean], dtype=float)
    lo = np.asarray([safe_float(r.get("ci_low")) for r in clean], dtype=float)
    hi = np.asarray([safe_float(r.get("ci_high")) for r in clean], dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.bar(
        labels,
        y,
        yerr=np.vstack([np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)]),
        capsize=4,
        color="#4c78a8",
    )
    ax.axhline(0.0, color="#777777", linewidth=1, label="Fully composed")
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1, label="No clamp effect")
    ax.set_title("Direct fraction on verified separating cuts — 1-hop must be ≈ 0 if the clamp is faithful")
    ax.set_xlabel("Model")
    ax.set_ylabel("Direct fraction = |C^clamp| / |C^unclamp|")
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)
        tick.set_ha("right")
    ax.legend(frameon=False, fontsize=8)
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step4_verified_separating_cuts_direct_fraction.png", dpi=dpi)
    fig.savefig(figures / "step4_verified_separating_cuts_direct_fraction.pdf")
    plt.close(fig)


def render_step4_clamp_mode_comparison(summary: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    clean = [r for r in summary if math.isfinite(safe_float(r.get("mean_direct_fraction")))]
    if not clean:
        return
    labels = []
    y = []
    lo = []
    hi = []
    colors = []
    color_by_type = {"cut": "#4c78a8", "random_off_path": "#f58518"}
    for row in clean:
        clamp_type = str(row.get("clamp_type"))
        clamp_label = "cut" if clamp_type == "cut" else "off-path"
        labels.append(f"{row.get('model')}\n{row.get('clamp_mode')} {clamp_label}")
        mean = safe_float(row.get("mean_direct_fraction"))
        y.append(mean)
        lo.append(safe_float(row.get("ci_low")))
        hi.append(safe_float(row.get("ci_high")))
        colors.append(color_by_type.get(clamp_type, "#999999"))
    y_arr = np.asarray(y, dtype=float)
    lo_arr = np.asarray(lo, dtype=float)
    hi_arr = np.asarray(hi, dtype=float)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.62 * len(labels)), 5.2), constrained_layout=True)
    ax.bar(
        np.arange(len(labels)),
        y_arr,
        yerr=np.vstack([np.maximum(0.0, y_arr - lo_arr), np.maximum(0.0, hi_arr - y_arr)]),
        capsize=3,
        color=colors,
    )
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1, label="No clamp effect")
    ax.set_title("Clamp specificity: cut vs off-path clamp, and detach vs overwrite")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Carriage-weighted direct fraction")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=color_by_type["cut"], label="Cut clamp"),
        Patch(facecolor=color_by_type["random_off_path"], label="Random off-path clamp"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8)
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step4_clamp_specificity_detach_vs_overwrite.png", dpi=dpi)
    fig.savefig(figures / "step4_clamp_specificity_detach_vs_overwrite.pdf")
    plt.close(fig)


def render_step4_depth_magnitude(summary: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    clean = [
        r
        for r in summary
        if math.isfinite(safe_float(r.get("clamp_until_layer")))
        and math.isfinite(safe_float(r.get("mean_abs_direct_clamped")))
        and math.isfinite(safe_float(r.get("mean_abs_unclamped")))
    ]
    if not clean:
        return
    models = sorted({str(r.get("model")) for r in clean})
    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(8.6, max(4.6, 3.4 * len(models))),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, model in zip(axes[:, 0], models):
        model_rows = [r for r in clean if str(r.get("model")) == model]
        bands = sorted({str(r.get("distance_band")) for r in model_rows}, key=depth_band_sort_key)
        for band in bands:
            band_rows = sorted(
                [r for r in model_rows if str(r.get("distance_band")) == band],
                key=lambda r: safe_float(r.get("clamp_until_layer")),
            )
            xs = np.asarray([safe_float(r.get("clamp_until_layer")) for r in band_rows], dtype=float)
            direct = np.asarray([safe_float(r.get("mean_abs_direct_clamped")) for r in band_rows], dtype=float)
            unclamped = np.asarray([safe_float(r.get("mean_abs_unclamped")) for r in band_rows], dtype=float)
            line = ax.plot(xs, direct, marker="o", linewidth=1.8, label=f"{band}: |C^clamp|")[0]
            ax.plot(xs, unclamped, linestyle="--", linewidth=1.4, color=line.get_color(), label=f"{band}: |C^unclamp|")
        ax.set_title(str(model))
        ax.set_xlabel("Clamp through layer")
        ax.set_ylabel("Mean absolute carriage")
        ax.set_ylim(bottom=0.0)
        ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Direct vs total carriage magnitude by clamp depth (is the ratio driven by |C^clamp| rising?)")
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step4_direct_vs_total_magnitude_by_clamp_depth.png", dpi=dpi)
    fig.savefig(figures / "step4_direct_vs_total_magnitude_by_clamp_depth.pdf")
    plt.close(fig)


def render_step4_pathway_interference(
    rows: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    *,
    dpi: int,
) -> None:
    clean_rows = [
        r
        for r in rows
        if str(r.get("clamp_type", "cut")) == "cut"
        and row_is_nontrivial(r)
        and math.isfinite(safe_float(r.get("direct")))
        and math.isfinite(safe_float(r.get("composed")))
    ]
    if not clean_rows and not summary:
        return
    figures = ensure_dir(artifact_root / "figures")
    if clean_rows:
        fig, ax = plt.subplots(figsize=(6.8, 5.6), constrained_layout=True)
        models = sorted({str(r.get("model")) for r in clean_rows})
        for model in models:
            model_rows = [r for r in clean_rows if str(r.get("model")) == model]
            x = np.asarray([safe_float(r.get("direct")) for r in model_rows], dtype=float)
            y = np.asarray([safe_float(r.get("composed")) for r in model_rows], dtype=float)
            ax.scatter(x, y, s=18, alpha=0.68, label=model)
        ax.axhline(0.0, color="#777777", linewidth=1)
        ax.axvline(0.0, color="#777777", linewidth=1)
        ax.set_title("Through-cut vs around-cut carriage: opposite signs ⇒ ratio > 1")
        ax.set_xlabel("Around-cut carriage C^clamp")
        ax.set_ylabel("Through-cut carriage C^through = C^unclamp - C^clamp")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(figures / "step4_pathway_interference_scatter.png", dpi=dpi)
        fig.savefig(figures / "step4_pathway_interference_scatter.pdf")
        plt.close(fig)
    clean_summary = [r for r in summary if math.isfinite(safe_float(r.get("mean_around_cut_C_clamp")))]
    if clean_summary:
        labels = [f"{r.get('model')} {r.get('distance_band')}" for r in clean_summary]
        around = np.asarray([safe_float(r.get("mean_around_cut_C_clamp")) for r in clean_summary], dtype=float)
        through = np.asarray([safe_float(r.get("mean_through_cut_C_through")) for r in clean_summary], dtype=float)
        x = np.arange(len(labels), dtype=float)
        width = 0.36
        fig, ax = plt.subplots(figsize=(max(8.0, 0.5 * len(labels)), 5.0), constrained_layout=True)
        ax.bar(x - width / 2.0, around, width=width, label="Around-cut C^clamp")
        ax.bar(x + width / 2.0, through, width=width, label="Through-cut C^through")
        ax.axhline(0.0, color="#555555", linewidth=1)
        ax.set_title("Through-cut vs around-cut carriage by distance band")
        ax.set_xlabel("Model and distance band")
        ax.set_ylabel("Signed carriage")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.legend(frameon=False, fontsize=8)
        for idx, row in enumerate(clean_summary):
            share = safe_float(row.get("opposite_sign_share"))
            if math.isfinite(share):
                ax.text(idx, max(around[idx], through[idx], 0.0), f"opp={share:.0%}", fontsize=7, ha="center", va="bottom")
        fig.savefig(figures / "step4_pathway_interference_bars.png", dpi=dpi)
        fig.savefig(figures / "step4_pathway_interference_bars.pdf")
        plt.close(fig)


def render_step4_signal_magnitude_by_distance(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    clean = [
        r
        for r in rows
        if math.isfinite(safe_float(r.get("distance")))
        and math.isfinite(safe_float(r.get("effect_abs")))
        and safe_float(r.get("effect_abs")) >= 0.0
    ]
    if not clean:
        return
    graph_max: dict[tuple[str, str], float] = {}
    for row in clean:
        key = (str(row.get("model")), str(row.get("graph_id")))
        graph_max[key] = max(graph_max.get(key, 0.0), safe_float(row.get("effect_abs")))
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in clean:
        key = (str(row.get("model")), str(row.get("graph_id")))
        denom = max(graph_max.get(key, 0.0), EPS)
        distance = int(round(safe_float(row.get("distance"))))
        grouped.setdefault((str(row.get("model")), distance), []).append(safe_float(row.get("effect_abs")) / denom)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    half_rows: list[dict[str, Any]] = []
    for model in sorted({model for model, _ in grouped}):
        distances = sorted(distance for m, distance in grouped if m == model)
        values = [float(np.nanmean(grouped[(model, distance)])) for distance in distances]
        ax.plot(distances, values, marker="o", linewidth=1.8, label=model)
        value_arr = np.asarray(values, dtype=float)
        finite_mask = np.isfinite(value_arr)
        if distances and values and finite_mask.any():
            finite_indices = np.flatnonzero(finite_mask)
            peak_idx = int(finite_indices[int(np.nanargmax(value_arr[finite_mask]))])
            peak = float(value_arr[peak_idx])
            threshold = 0.25 * peak
            half_distance = float("nan")
            for distance, value in zip(distances[peak_idx:], values[peak_idx:]):
                if math.isfinite(value) and value <= threshold:
                    half_distance = float(distance)
                    break
            half_rows.append(
                {
                    "model": model,
                    "peak_distance": distances[peak_idx],
                    "peak_normalized_carriage": peak,
                    "threshold_fraction_of_peak": 0.25,
                    "half_distance": half_distance,
                    "max_observed_distance": max(distances),
                }
            )
            if math.isfinite(half_distance):
                ax.axvline(half_distance, color=ax.lines[-1].get_color(), linestyle=":", linewidth=0.9, alpha=0.65)
                ax.text(
                    half_distance,
                    threshold,
                    f"{model} hd={half_distance:.0f}",
                    fontsize=7,
                    rotation=90,
                    va="bottom",
                    ha="right",
                    color=ax.lines[-1].get_color(),
                )
            else:
                ax.text(
                    distances[-1],
                    values[-1],
                    f"{model} hd>{distances[-1]}",
                    fontsize=7,
                    va="bottom",
                    ha="right",
                    color=ax.lines[-1].get_color(),
                )
    # No gate-floor line: the hard signal gate was removed; noise is handled by the
    # ratio-of-sums aggregation and the composed reference (GIN), not a per-pair floor.
    ax.set_title("Over-squashing: carriage magnitude vs distance by model (half-distance annotated)")
    ax.set_xlabel("Molecular hop distance")
    ax.set_ylabel("|C| / per-graph max")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=8)
    figures = ensure_dir(artifact_root / "figures")
    write_csv(artifact_root / "metrics" / "step4_carriage_half_distance.csv", half_rows)
    fig.savefig(figures / "step4_carriage_signal_by_distance.png", dpi=dpi)
    fig.savefig(figures / "step4_carriage_signal_by_distance.pdf")
    plt.close(fig)


def _clone_graph_with_swapped_content(graph: Any, source: int, donor: int) -> Optional[Any]:
    """Clone ``graph`` with node ``source``'s raw content replaced by node ``donor``'s.

    Only node content (x) changes; edge_index/topology is untouched, so RRWP is recomputed from
    the same structure -> a pure content perturbation.
    """
    x = getattr(graph, "x", None)
    if not isinstance(x, torch.Tensor) or not hasattr(graph, "clone"):
        return None
    try:
        clone = graph.clone()
    except Exception:
        return None
    base = x.detach()
    new_x = base.clone()
    new_x[int(source)] = base[int(donor)]
    clone.x = new_x
    return clone


def _emit_carriage_rows(
    rows: list[dict[str, Any]],
    model: ModelRun,
    gid: str,
    source: int,
    contribs: torch.Tensor,
    dist: torch.Tensor,
    *,
    min_distance: int,
    factor: str,
    benefit_sign: Optional[float] = None,
    mode: str = "functional",
) -> None:
    """Emit per-(carrier, source) carriage rows.

    Always emits ``mode="functional"`` (the readout-projected effect). When ``benefit_sign`` is
    given (= sign(y_hat_clean - y)), also emits ``mode="beneficial"`` = benefit_sign * effect, whose
    sum telescopes (first order) to L(clean) - L(corrupt): the SAME loss projection used by the
    content loss-carriage, so content and structural beneficial carriage share one convention
    (signed value < 0 = loss-reducing = beneficial). This is the shared symbolic/structural hook.
    """
    c = contribs.detach().cpu()
    n = int(c.numel())
    for carrier in range(n):
        if carrier == source:
            continue
        d = float(dist[carrier, source].item())
        if not math.isfinite(d) or d < float(min_distance):
            continue
        val = float(c[carrier].item())
        base = {
            "model": model.name,
            "role": model.role,
            "graph_id": gid,
            "source": int(source),
            "carrier": int(carrier),
            "distance": int(round(d)),
            "factor": factor,
        }
        rows.append({**base, "effect_abs": abs(val), "effect_signed": val, "mode": mode})
        if benefit_sign is not None:
            bval = val * float(benefit_sign)
            rows.append({**base, "effect_abs": abs(bval), "effect_signed": bval, "mode": "beneficial"})


def structural_carriage_ig(
    adapter: Any,
    graph: Any,
    *,
    target_kind: str,          # "node" (node-RRWP) or "pair" (pair-RRWP)
    steps: int,
    target_index: int = 0,
    readout_ig: bool = False,
    loss_label: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """IG structural carriage over the RRWP input -- the headline analog of ``carriage_ig``.

    Identical construction to the content carriage (integrate g_i.h_i^L along a baseline->clean path,
    VJP onto carriers, project by g), but the integration variable is the RAW RRWP (node or pair)
    instead of the encoded content, and the baseline is the graph-mean RRWP. ``loss_label`` gives the
    beneficial version (project by sign(y_hat(alpha)-y) => sum telescopes to L(clean)-L(base)),
    exactly like B_IG. Returns C_struct[carrier, source] plus a completeness check.
    """
    if not hasattr(adapter, "readout_gradient") or not hasattr(adapter, "_run_with_hooks"):
        return None
    try:
        clean_cache, g = adapter.readout_gradient(graph, target_index=target_index)
    except Exception:  # noqa: BLE001
        return None
    h_clean = getattr(clean_cache, "final_node_states", None)
    if not isinstance(h_clean, torch.Tensor) or not isinstance(g, torch.Tensor):
        return None
    h_clean = h_clean.detach()
    g = g.detach().to(h_clean)
    n = int(h_clean.size(0))
    extras = getattr(clean_cache, "extras", None) or {}
    raw_node = extras.get("raw_rrwp")
    raw_val = extras.get("raw_rrwp_val")
    raw_index = extras.get("raw_rrwp_index")

    # --- resolve the RRWP integration variable into a common FLAT form so one code path serves the
    # sparse node ([n,K]), sparse pair ([E,K]+index) AND dense ([n,n,K]) representations. For the
    # dense tensor, node RRWP = the diagonal, pair RRWP = the off-diagonal (mirrors the sparse split);
    # only the selected slots are perturbed (delta zeroed elsewhere), and attribution maps slot (a,b)
    # to source a. ``active`` also drops padded slots (a>=n or b>=n) so a padded [maxN,maxN,K] is safe.
    if isinstance(raw_node, torch.Tensor) and raw_node.dim() == 3:
        dp = raw_node.detach()
        nn, kk = int(dp.size(0)), int(dp.size(-1))
        rrwp_flat = dp.reshape(nn * nn, kk)
        a_idx = torch.arange(nn).repeat_interleave(nn)
        b_idx = torch.arange(nn).repeat(nn)
        row_src = a_idx.clone()
        diag = a_idx == b_idx
        active = (diag if target_kind == "node" else ~diag) & (a_idx < n) & (b_idx < n)
        dense_shape = tuple(dp.shape)
        override_kw = "rrwp_node_override"

        def to_override(pf: torch.Tensor) -> dict[str, torch.Tensor]:
            return {override_kw: pf.reshape(dense_shape)}
    elif target_kind == "node":
        if not isinstance(raw_node, torch.Tensor) or raw_node.dim() != 2:
            return None
        rrwp_flat = raw_node.detach()
        row_src = torch.arange(int(rrwp_flat.size(0)))
        active = row_src < n
        override_kw = "rrwp_node_override"

        def to_override(pf: torch.Tensor) -> dict[str, torch.Tensor]:
            return {override_kw: pf}
    else:  # pair -- raw sparse pair (rrwp_val + rrwp_index) if exposed, else the encoded-edge channel
        if (isinstance(raw_val, torch.Tensor) and raw_val.dim() == 2
                and isinstance(raw_index, torch.Tensor) and raw_index.dim() == 2):
            rrwp_flat = raw_val.detach()
            row_src = raw_index[0].detach().cpu().long()
            override_kw = "rrwp_val_override"
        else:
            # These GRIT models consume RRWP inside their encoders and do NOT expose a raw RRWP tensor
            # at hook time. batch.edge_attr (the encoded relative structural encoding, post RRWP-edge-
            # encoder) is the adapter's DESIGNED pair-RRWP intervention point -- the same substrate the
            # finite swap perturbs, applied differentiably -- so IG and swap stay directly comparable.
            # It perturbs pair structure while leaving node content intact.
            enc_attr = extras.get("encoded_edge_attr")
            enc_index = extras.get("encoded_edge_index")
            if (not isinstance(enc_attr, torch.Tensor) or enc_attr.dim() != 2
                    or not isinstance(enc_index, torch.Tensor) or enc_index.dim() != 2
                    or int(enc_index.size(1)) != int(enc_attr.size(0))):
                return None
            rrwp_flat = enc_attr.detach()
            row_src = enc_index[0].detach().cpu().long()
            override_kw = "edge_attr_override"
        active = row_src < n

        def to_override(pf: torch.Tensor) -> dict[str, torch.Tensor]:
            return {override_kw: pf}

    if int(rrwp_flat.size(0)) == 0 or not bool(active.any()):
        return None
    base = rrwp_flat.mean(dim=0, keepdim=True).expand_as(rrwp_flat)  # graph-mean RRWP baseline
    delta = (rrwp_flat - base) * active.to(rrwp_flat.dtype).unsqueeze(-1)  # perturb active slots only
    active_idx = active.nonzero(as_tuple=False).reshape(-1)
    add_src = row_src.to(h_clean.device)[active_idx]
    y = None if loss_label is None else float(loss_label)
    yhat_clean = float(clean_cache.prediction.reshape(-1)[target_index].detach().cpu().item())
    carriage = h_clean.new_zeros((n, n))
    yhat_endpoint = None
    for alpha_idx in range(1, int(steps) + 1):
        alpha = float(alpha_idx) / float(steps)
        point = (base + alpha * delta).detach().requires_grad_(True)
        cache = adapter._run_with_hooks(
            graph, capture_attention=False, capture_channels=False,
            capture_layer_inputs=False, capture_layer_outputs=False, **to_override(point),
        )
        h = getattr(cache, "final_node_states", None)
        if not isinstance(h, torch.Tensor):
            return None
        pred_a = cache.prediction.reshape(-1)[target_index]
        if alpha_idx == int(steps):
            yhat_endpoint = float(pred_a.detach().cpu().item())
        if readout_ig:
            (g_a,) = torch.autograd.grad(pred_a, h, retain_graph=True, create_graph=False)
            g_step = g_a.detach()
        else:
            g_step = g
        if y is not None:
            g_step = g_step * (1.0 if float(pred_a.detach().cpu().item()) >= y else -1.0)
        carrier_scores = (h * g_step).sum(dim=-1)
        for i in range(n):
            (grad,) = torch.autograd.grad(carrier_scores[i], point, retain_graph=(i < n - 1), create_graph=False)
            per_row = (grad.detach() * delta).sum(dim=-1)
            carriage[i].index_add_(0, add_src, per_row[active_idx].to(carriage.device))
    carriage = carriage / float(steps)
    # completeness: sum carriage telescopes to y_hat(endpoint) - y_hat(base), where the endpoint is
    # the alpha=1 point (active slots at clean, others at baseline) -- exact with readout_ig.
    yhat_base = None
    try:
        base_cache = adapter._run_with_hooks(
            graph, capture_attention=False, capture_channels=False,
            capture_layer_inputs=False, capture_layer_outputs=False, **to_override(base.detach()),
        )
        yhat_base = float(base_cache.prediction.reshape(-1)[target_index].detach().cpu().item())
    except Exception:  # noqa: BLE001
        yhat_base = None
    recon = float(carriage.sum().item())
    target = None if (yhat_base is None or yhat_endpoint is None) else (yhat_endpoint - yhat_base)
    return {
        "carriage": carriage.detach().cpu(),
        "target_kind": target_kind,
        "yhat_clean": yhat_clean,
        "yhat_base": yhat_base,
        "yhat_endpoint": yhat_endpoint,
        "reconstruction_sum": recon,
        "completeness_target": target,
    }


def symbolic_structural_carriage_rows(
    model: ModelRun,
    graph: Any,
    gid: str,
    *,
    seed: int,
    min_distance: int = 1,
    max_sources: Optional[int] = None,
    rrwp_channel_start: int = 2,
    rrwp_replacement: str = "zero",
    donor_samples: int = 4,
    ig_baseline: Optional[torch.Tensor] = None,
    artifact_root: Optional[Path] = None,
    config: Optional[Mapping[str, Any]] = None,
    completeness_out: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Levelled discrete content (symbolic) vs structural (RRWP) carriage for one graph.

    When ``ig_baseline``/``artifact_root``/``config`` are supplied, also emits the content
    carriage via the existing IG method (factor ``content_ig``) as a validation reference for the
    swap-based content panel (does moving to swaps distort the by-distance profile via OOD shift?).

    Both axes use the same readout-projected node-delta ``g_i . (h_i^clean - h_i^corrupt)`` as the
    carriage, but with a discrete on-manifold patch instead of IG so they are directly comparable:
      * content   : swap node j's raw content for a donor node's (topology/RRWP unchanged) -> pure
        content perturbation.
      * node_rrwp : remove long-range raw node-RRWP channels at source j only.
      * pair_rrwp : remove long-range raw pair-RRWP channels on directed pair entries sourced at j.
      * both_rrwp : apply both node and pair RRWP source perturbations together.

    Channels before ``rrwp_channel_start`` are preserved by default, so this is a
    global-structural-information perturbation rather than a deletion of self/one-hop identity.
    Emits one row per (carrier i, source j) with the pair's molecular distance.
    """
    adapter = model.adapter
    if not hasattr(adapter, "readout_gradient") or not hasattr(adapter, "forward_minimal"):
        return []
    try:
        clean_cache, g = adapter.readout_gradient(graph)
    except Exception:
        return []
    h_clean = getattr(clean_cache, "final_node_states", None)
    if not isinstance(h_clean, torch.Tensor) or not isinstance(g, torch.Tensor):
        return []
    h_clean = h_clean.detach()
    g = g.detach().to(device=h_clean.device, dtype=h_clean.dtype)
    n = int(h_clean.size(0))
    if n < 2:
        return []
    dist = distance_matrix(graph).detach().cpu()
    rng = random.Random(f"{seed}:{gid}")
    # Shared loss projection for beneficial carriage (content + structure): sign(y_hat_clean - y).
    # ``yv`` is the label itself, reused by the IG loss-carriage (content_ig + structural IG) so
    # their beneficial carriage is the proper integrated loss-carriage, not the first-order shortcut.
    benefit_sign: Optional[float] = None
    yv: Optional[float] = None
    _yl = getattr(graph, "y", None)
    _yhat = getattr(clean_cache, "prediction", None)
    if _yl is not None:
        try:
            yv = float(torch.as_tensor(_yl).reshape(-1)[0].item())
        except Exception:  # noqa: BLE001
            yv = None
    if yv is not None and isinstance(_yhat, torch.Tensor):
        try:
            _yhatv = float(_yhat.reshape(-1)[0].item())
            benefit_sign = 1.0 if _yhatv >= yv else -1.0
        except Exception:  # noqa: BLE001
            benefit_sign = None
    sources = list(range(n))
    if max_sources is not None and len(sources) > int(max_sources):
        sources = sorted(rng.sample(sources, int(max_sources)))
    rows: list[dict[str, Any]] = []
    donor_k = max(1, int(donor_samples))

    def _distinct_donors(focal: int, k: int) -> list[int]:
        """Up to k distinct donor nodes != focal (for K-donor averaged swaps)."""
        out: list[int] = []
        seen: set[int] = set()
        tries = 0
        while len(out) < min(k, n - 1) and tries < 8 * max(1, k):
            cand = rng.randrange(n)
            if cand != focal and cand not in seen:
                seen.add(cand)
                out.append(cand)
            tries += 1
        return out

    # --- content (symbolic) carriage: swap raw node content for donor nodes' (K-donor averaged) ---
    # Averaging the readout-projected effect over K distinct donors de-noises the single-donor swap
    # and makes it less hostage to one random donor -- tightening agreement with the IG estimator.
    if isinstance(getattr(graph, "x", None), torch.Tensor):
        for j in sources:
            acc: Optional[torch.Tensor] = None
            got = 0
            for donor in _distinct_donors(j, donor_k):
                corrupted = _clone_graph_with_swapped_content(graph, j, donor)
                if corrupted is None:
                    continue
                try:
                    cache = adapter.forward_minimal(corrupted)
                except Exception:
                    continue
                h = getattr(cache, "final_node_states", None)
                if not isinstance(h, torch.Tensor) or tuple(h.shape) != tuple(h_clean.shape):
                    continue
                c = (g * (h_clean - h.detach())).sum(dim=-1)
                acc = c if acc is None else acc + c
                got += 1
            if acc is not None and got:
                _emit_carriage_rows(rows, model, gid, j, acc / float(got), dist, min_distance=min_distance, benefit_sign=benefit_sign, factor="content")

    # --- structural carriage: source-specific raw RRWP perturbations before RRWP encoding ---
    # Node-RRWP (diagonal, fed to the node encoder) and pair-RRWP (off-diagonal, fed to the edge/
    # attention encoder) are SEPARATE model inputs, so they are perturbed independently. Crucially the
    # node swap is NOT gated behind pair exposure: sparse models expose pair RRWP as ``edge_rrwp``
    # rather than ``rrwp_val``/``rrwp_index``, which previously skipped the whole branch and dropped
    # the Node RRWP column. Donor-mode swaps are averaged over ``struct_k`` random donors to de-noise,
    # matching the multi-donor content swap. (The primary structural estimator remains the IG method
    # below; these swaps are the on-manifold cross-check.)
    extras = getattr(clean_cache, "extras", None) or {}
    raw_rrwp0 = extras.get("raw_rrwp")
    raw_rrwp_val0 = extras.get("raw_rrwp_val")
    raw_rrwp_index0 = extras.get("raw_rrwp_index")
    struct_k = donor_k if rrwp_replacement == "donor" else 1

    def _swap_contribs(**override_kwargs: Any) -> Optional[torch.Tensor]:
        try:
            cache = adapter.forward_minimal(graph, **override_kwargs)
        except Exception:  # noqa: BLE001
            return None
        h = getattr(cache, "final_node_states", None)
        if not isinstance(h, torch.Tensor) or tuple(h.shape) != tuple(h_clean.shape):
            return None
        return (g * (h_clean - h.detach())).sum(dim=-1)

    def _avg_swap(make_override: Any, k: int) -> Optional[torch.Tensor]:
        """Average readout-projected swap effect over up to k (re-drawn) overrides."""
        acc: Optional[torch.Tensor] = None
        got = 0
        for _ in range(max(1, int(k))):
            ov = make_override()
            if ov is None:
                continue
            c = _swap_contribs(**ov)
            if c is None:
                continue
            acc = c if acc is None else acc + c
            got += 1
        return (acc / float(got)) if (acc is not None and got) else None

    # node-RRWP swap: independent of pair exposure (gated only on the raw node RRWP tensor)
    if isinstance(raw_rrwp0, torch.Tensor) and raw_rrwp0.dim() == 2:
        rrwp_node0 = raw_rrwp0.detach()
        for j in sources:
            if int(j) >= int(rrwp_node0.size(0)):
                continue
            src_row = torch.as_tensor([int(j)], dtype=torch.long)
            c = _avg_swap(
                lambda sr=src_row: {"rrwp_node_override": _rrwp_replace_channels(
                    rrwp_node0, sr, channel_start=rrwp_channel_start, replacement=rrwp_replacement, rng=rng)},
                struct_k,
            )
            if c is not None:
                _emit_carriage_rows(rows, model, gid, j, c, dist, min_distance=min_distance, benefit_sign=benefit_sign, factor="node_rrwp")

    # pair-RRWP swap: sparse (val+index) -> dense (3-D) -> legacy encoded-edge fallback
    used_pair = False
    if (
        isinstance(raw_rrwp_val0, torch.Tensor)
        and isinstance(raw_rrwp_index0, torch.Tensor)
        and raw_rrwp_val0.dim() == 2
        and raw_rrwp_index0.dim() == 2
        and int(raw_rrwp_val0.size(0)) == int(raw_rrwp_index0.size(1))
    ):
        used_pair = True
        rrwp_val0 = raw_rrwp_val0.detach()
        src_e = raw_rrwp_index0.detach()[0]
        for j in sources:
            outgoing_rows = torch.nonzero((src_e == int(j)).detach().cpu().bool(), as_tuple=False).reshape(-1).long()
            if outgoing_rows.numel() == 0:
                continue
            c = _avg_swap(
                lambda orows=outgoing_rows: {"rrwp_val_override": _rrwp_replace_channels(
                    rrwp_val0, orows, channel_start=rrwp_channel_start, replacement=rrwp_replacement, rng=rng)},
                struct_k,
            )
            if c is not None:
                _emit_carriage_rows(rows, model, gid, j, c, dist, min_distance=min_distance, benefit_sign=benefit_sign, factor="pair_rrwp")
    elif isinstance(raw_rrwp0, torch.Tensor) and raw_rrwp0.dim() == 3:
        used_pair = True
        dense_pair = raw_rrwp0.detach()
        rrwp_index, _ = _dense_pair_rrwp_to_sparse(dense_pair.detach().cpu())
        src_e = rrwp_index[0]
        flat = dense_pair.reshape(-1, int(dense_pair.size(-1)))
        for j in sources:
            outgoing_rows = torch.nonzero((src_e == int(j)).detach().cpu().bool(), as_tuple=False).reshape(-1).long()
            if outgoing_rows.numel() == 0:
                continue
            c = _avg_swap(
                lambda orows=outgoing_rows: {"rrwp_node_override": _rrwp_replace_channels(
                    flat, orows, channel_start=rrwp_channel_start, replacement=rrwp_replacement, rng=rng).reshape_as(dense_pair)},
                struct_k,
            )
            if c is not None:
                _emit_carriage_rows(rows, model, gid, j, c, dist, min_distance=min_distance, benefit_sign=benefit_sign, factor="pair_rrwp")
    if not used_pair:
        # Legacy fallback: perturb the already-encoded edge state if raw pair RRWP is not exposed
        # (mean-replacement is deterministic, so no donor averaging).
        edge_attr0 = extras.get("encoded_edge_attr")
        edge_index0 = extras.get("encoded_edge_index")
        if (
            isinstance(edge_attr0, torch.Tensor)
            and isinstance(edge_index0, torch.Tensor)
            and edge_attr0.dim() == 2
            and edge_index0.dim() == 2
            and int(edge_attr0.size(0)) == int(edge_index0.size(1))
        ):
            edge_attr0 = edge_attr0.detach()
            ei = edge_index0.detach()
            mean_ea = edge_attr0.mean(dim=0, keepdim=True)
            src_e, dst_e = ei[0], ei[1]
            for j in sources:
                incident = (src_e == int(j)) | (dst_e == int(j))
                if not bool(incident.any()):
                    continue
                ea = edge_attr0.clone()
                ea[incident] = mean_ea.to(ea.dtype)
                c = _swap_contribs(edge_attr_override=ea)
                if c is not None:
                    _emit_carriage_rows(rows, model, gid, j, c, dist, min_distance=min_distance, benefit_sign=benefit_sign, factor="pair_rrwp")

    # --- IG-aligned structural carriage (HEADLINE; mirrors carriage_ig, completeness-checked) ---
    # Functional = structural_carriage_ig; beneficial = same with loss_label (the proper integrated
    # loss-carriage, exactly like content B_IG -- not the first-order benefit_sign shortcut).
    if config is not None:
        cfg_sub = (config.get("steps", {}) or {}).get("7") or (config.get("steps", {}) or {}).get("4") or {}
        if bool(cfg_sub.get("run_structural_carriage_ig", True)):
            ig_steps = int(config["perturbation"].get("ig_steps", 32))
            struct_readout_ig = carriage_ig_uses_readout_ig(config)
            for tkind, fac in (("node", "node_rrwp_ig"), ("pair", "pair_rrwp_ig")):
                res = structural_carriage_ig(adapter, graph, target_kind=tkind, steps=ig_steps, readout_ig=struct_readout_ig)
                if res is None:
                    def _shp(key: str) -> Any:
                        t = extras.get(key)
                        return tuple(t.shape) if isinstance(t, torch.Tensor) else None
                    progress(
                        f"  {model.name} graph {gid}: {fac} unavailable -- raw_rrwp={_shp('raw_rrwp')} "
                        f"raw_rrwp_val={_shp('raw_rrwp_val')} raw_rrwp_index={_shp('raw_rrwp_index')} "
                        f"encoded_edge_attr={_shp('encoded_edge_attr')} hooks={hasattr(adapter, '_run_with_hooks')} "
                        f"keys={sorted(k for k, v in extras.items() if v is not None)[:12]}"
                    )
                    continue
                if completeness_out is not None and res.get("completeness_target") is not None:
                    rec = float(res["reconstruction_sum"])
                    tgt = float(res["completeness_target"])
                    completeness_out.append({
                        "model": model.name, "graph_id": gid, "target_kind": tkind,
                        "reconstruction_sum": rec, "completeness_target": tgt, "abs_error": abs(rec - tgt),
                    })
                c_f = res["carriage"]
                for j in sources:
                    _emit_carriage_rows(rows, model, gid, j, c_f[:, int(j)], dist, min_distance=min_distance, factor=fac, mode="functional")
                if yv is not None:
                    resb = structural_carriage_ig(adapter, graph, target_kind=tkind, steps=ig_steps, readout_ig=struct_readout_ig, loss_label=yv)
                    if resb is not None:
                        c_b = resb["carriage"]
                        for j in sources:
                            _emit_carriage_rows(rows, model, gid, j, c_b[:, int(j)], dist, min_distance=min_distance, factor=fac, mode="beneficial")

    # --- content carriage via the IG method (HEADLINE; same estimator/convention as structural IG) ---
    # Functional = carriage_ig; beneficial = carriage_ig(loss_label=yv) -> the proper integrated
    # loss-carriage (sum telescopes to L(clean)-L(base); <0 = beneficial), EXACTLY like the structural
    # IG and Step-6 content B_IG -- NOT the first-order benefit_sign shortcut (that is reserved for the
    # finite-swap factors, which cannot be path-integrated). Failures are logged, never swallowed, so a
    # model dropping out of content_ig is visible instead of silently producing a single-model panel.
    if ig_baseline is not None and config is not None:
        ig_steps = int(config["perturbation"].get("ig_steps", 32))
        r_ig = carriage_ig_uses_readout_ig(config)
        b_vjp = carriage_ig_uses_batched_vjp(config)
        try:
            res_f = carriage_ig(adapter, graph, ig_baseline, steps=ig_steps, readout_ig=r_ig, batched_vjp=b_vjp)
            c_ig = res_f.get("carriage")
            if isinstance(c_ig, torch.Tensor) and c_ig.dim() == 2 and int(c_ig.size(0)) == n:
                for j in sources:
                    _emit_carriage_rows(rows, model, gid, j, c_ig[:, int(j)], dist, min_distance=min_distance, factor="content_ig", mode="functional")
                if completeness_out is not None:
                    # content-IG completeness: Sum C == y_hat_clean - y_hat_base (exact with readout_ig)
                    try:
                        rec = float(c_ig.sum().item())
                        pred_clean = float(torch.as_tensor(res_f.get("prediction")).reshape(-1)[0].item())
                        pred_base = float(res_f.get("baseline_prediction"))
                        completeness_out.append({
                            "model": model.name, "graph_id": gid, "target_kind": "content",
                            "reconstruction_sum": rec, "completeness_target": pred_clean - pred_base,
                            "abs_error": abs(rec - (pred_clean - pred_base)),
                        })
                    except Exception:  # noqa: BLE001
                        pass
            else:
                progress(f"  {model.name} graph {gid}: content_ig functional produced no valid carriage")
        except Exception as exc:  # noqa: BLE001
            progress(f"  {model.name} graph {gid}: content_ig functional FAILED ({type(exc).__name__}: {exc})")
        if yv is not None:
            try:
                res_b = carriage_ig(adapter, graph, ig_baseline, steps=ig_steps, readout_ig=r_ig, batched_vjp=b_vjp, loss_label=yv)
                c_igb = res_b.get("carriage")
                if isinstance(c_igb, torch.Tensor) and c_igb.dim() == 2 and int(c_igb.size(0)) == n:
                    for j in sources:
                        _emit_carriage_rows(rows, model, gid, j, c_igb[:, int(j)], dist, min_distance=min_distance, factor="content_ig", mode="beneficial")
            except Exception as exc:  # noqa: BLE001
                progress(f"  {model.name} graph {gid}: content_ig beneficial FAILED ({type(exc).__name__}: {exc})")
    elif ig_baseline is None:
        progress(f"  {model.name} graph {gid}: content_ig skipped (no IG baseline available for this model)")
    return rows


def run_symbolic_structural_probe(models: Sequence[ModelRun], artifact_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Step-7 substrate probe: symbolic (content) vs structural (RRWP) carriage by distance."""
    cfg = config["steps"].get("7") or config["steps"].get("4") or {}
    sample_graphs = int(cfg.get("symbolic_structural_sample_graphs", min(8, int(cfg.get("sample_graphs", 8)))))
    raw_max_sources = cfg.get("symbolic_structural_max_sources", None)
    max_sources = None if raw_max_sources in (None, "all", "", "None") else int(raw_max_sources)
    min_distance = int(cfg.get("symbolic_structural_min_distance", 1))
    rrwp_channel_start = int(cfg.get("symbolic_structural_rrwp_channel_start", cfg.get("rrwp_ablation_channel_start", 2)))
    # Default to on-manifold DONOR replacement (a real other row's channels) rather than zeroing, so
    # the structural swap is the on-manifold cross-check aligned with the content donor-swap.
    rrwp_replacement = str(cfg.get("symbolic_structural_rrwp_replacement", "donor"))
    donor_samples = int(cfg.get("symbolic_structural_donor_samples", 4))
    reach_tau = int(cfg.get("reach_far_distance_tau", 3))
    seed = int(config.get("seeds", [0])[0])
    dpi = int(config["figures"]["dpi"])
    rows: list[dict[str, Any]] = []
    completeness_rows: list[dict[str, Any]] = []
    for model in models:
        try:
            graphs = select_graphs(model.adapter, "test", sample_graphs, seed=seed)
        except Exception as exc:
            progress(f"Step 7 symbolic/structural: skip {model.name} ({exc})")
            continue
        progress(f"Step 7 symbolic/structural probe {model.name}: {len(graphs)} graph(s), max_sources={max_sources}")
        try:
            ig_baseline = mean_encoded_baseline(
                model.adapter, select_baseline_graphs(model.adapter, "test", config, sample_graphs, seed=seed)
            )
        except Exception as exc:
            ig_baseline = None
            progress(f"  {model.name}: IG content reference panel unavailable ({exc})")
        for graph_idx, graph in enumerate(graphs):
            gid = graph_identity("test", graph_idx, graph)
            try:
                rows.extend(
                    symbolic_structural_carriage_rows(
                        model,
                        graph,
                        gid,
                        seed=seed,
                        min_distance=min_distance,
                        max_sources=max_sources,
                        rrwp_channel_start=rrwp_channel_start,
                        rrwp_replacement=rrwp_replacement,
                        donor_samples=donor_samples,
                        ig_baseline=ig_baseline,
                        artifact_root=artifact_root,
                        config=config,
                        completeness_out=completeness_rows,
                    )
                )
            except Exception as exc:
                progress(f"  {model.name} graph {graph_idx}: symbolic/structural failed ({exc})")
    if not rows:
        progress("Step 7 symbolic/structural: no rows produced")
        return {"status": "no_rows", "rows": 0}
    write_csv(artifact_root / "metrics" / "step7_symbolic_structural_carriage.csv", rows)
    completeness_note: Optional[str] = None
    if completeness_rows:
        write_csv(artifact_root / "metrics" / "step7_structural_carriage_ig_completeness.csv", completeness_rows)

        def _rel(kinds: set[str]) -> Optional[float]:
            e = [safe_float(r["abs_error"]) for r in completeness_rows
                 if str(r.get("target_kind")) in kinds and math.isfinite(safe_float(r.get("abs_error")))]
            t = [abs(safe_float(r["completeness_target"])) for r in completeness_rows
                 if str(r.get("target_kind")) in kinds]
            return None if not e else float(np.mean(e)) / (float(np.mean(t)) + 1e-9)

        def _fmt(x: Optional[float]) -> str:
            return "n/a" if x is None else f"{x:.1e}"

        rel_c, rel_s = _rel({"content"}), _rel({"node", "pair"})
        completeness_note = (
            f"IG completeness rel-err -- content: {_fmt(rel_c)}   structural: {_fmt(rel_s)}   "
            "(Sum C == y_hat_clean - y_hat_base; ~0 certifies IG as the anchor)"
        )
        progress(f"Step 7 IG completeness: content rel~{_fmt(rel_c)}, structural rel~{_fmt(rel_s)} "
                 "(should be ~0 -- the IG self-test on real GRIT)")
    # --- coverage diagnostics: which (model, factor, mode) combos actually produced rows? ---
    # This is the antidote to silent single-model panels: the CSV + log say exactly which models
    # are present/missing per factor, so a dropout is a data-availability fact, not a mystery.
    coverage: dict[tuple[str, str, str], dict[str, float]] = {}
    for r in rows:
        key = (str(r.get("model")), str(r.get("factor")), str(r.get("mode")))
        d = safe_float(r.get("distance"))
        cur = coverage.setdefault(key, {"count": 0.0, "dmin": math.inf, "dmax": -math.inf})
        cur["count"] += 1.0
        if math.isfinite(d):
            cur["dmin"] = min(cur["dmin"], d)
            cur["dmax"] = max(cur["dmax"], d)
    cov_rows = [
        {
            "model": m, "factor": f, "mode": md, "rows": int(v["count"]),
            "distance_min": (None if not math.isfinite(v["dmin"]) else int(v["dmin"])),
            "distance_max": (None if not math.isfinite(v["dmax"]) else int(v["dmax"])),
        }
        for (m, f, md), v in sorted(coverage.items())
    ]
    write_csv(artifact_root / "metrics" / "step7_carriage_coverage.csv", cov_rows)
    all_models = ordered_model_names(sorted({str(r.get("model")) for r in rows}))
    for fac in ("content_ig", "node_rrwp_ig", "pair_rrwp_ig", "content", "node_rrwp", "pair_rrwp", "both_rrwp"):
        have = [m for m in all_models if (m, fac, "functional") in coverage]
        miss = [m for m in all_models if m not in have]
        if have or miss:
            progress(f"Step 7 coverage [{fac} functional]: have {have or '-'}; MISSING {miss or '-'}")
    write_csv(artifact_root / "metrics" / "step7_carriage_by_distance_summary.csv", step7_by_distance_summary_rows(rows))
    write_csv(artifact_root / "metrics" / "step7_carriage_reach_summary.csv", step7_reach_summary_rows(rows, tau=reach_tau))
    write_csv(artifact_root / "metrics" / "step7_method_agreement_summary.csv", step7_method_agreement_rows(rows))
    render_symbolic_structural_by_distance(rows, artifact_root, dpi=dpi)
    render_step7_funcbenef_grid(rows, artifact_root, dpi=dpi, method="ig")
    render_step7_funcbenef_grid(rows, artifact_root, dpi=dpi, method="swap")
    render_carriage_functional_vs_beneficial(rows, artifact_root, dpi=dpi, completeness_note=completeness_note)
    render_step7_reach_summary(rows, artifact_root, dpi=dpi, tau=reach_tau)
    render_step7_method_agreement(rows, artifact_root, dpi=dpi, completeness_note=completeness_note)
    n_struct = len([r for r in rows if str(r.get("factor")) in {"node_rrwp", "pair_rrwp", "both_rrwp", "structure"}])
    progress(f"Step 7 symbolic/structural: wrote {len(rows)} rows ({n_struct} structural)")
    return {"status": "complete", "rows": len(rows), "structural_rows": n_struct}


def _bootstrap_mean_ci(
    values: Sequence[float], rng: np.random.Generator, *, n_boot: int = 1000, alpha: float = 0.05
) -> tuple[float, float, float, int]:
    """(mean, ci_lo, ci_hi, n) for one bucket via percentile bootstrap. n<2 -> degenerate CI = mean."""
    x = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    n = int(x.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(x.mean())
    if n == 1:
        return mean, mean, mean, 1
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot = x[idx].mean(axis=1)
    lo = float(np.percentile(boot, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return mean, lo, hi, n


def _step7_series(
    rows: Sequence[Mapping[str, Any]],
    factor: str,
    model: str,
    mode: str,
    value_key: str,
    *,
    rng: Optional[np.random.Generator] = None,
    n_boot: int = 1000,
) -> tuple[list[int], list[float], list[float], list[float], list[int]]:
    """Per-distance (distances, means, ci_lo, ci_hi, counts) for one (factor, model, mode).

    Means are over all (carrier, source) pairs at each integer carrier<->source distance; CIs are
    percentile bootstrap over those pairs, so the band WIDTH is the visual per-distance sample-size
    signal (wide = few pairs = noisy, e.g. the far tail). Empty lists if the combo produced no rows.
    """
    by: dict[int, list[float]] = {}
    for r in rows:
        if str(r.get("factor")) != factor or str(r.get("model")) != model or str(r.get("mode")) != mode:
            continue
        d = safe_float(r.get("distance"))
        v = safe_float(r.get(value_key))
        if not (math.isfinite(d) and math.isfinite(v)):
            continue
        by.setdefault(int(round(d)), []).append(v)
    ds = sorted(by)
    if rng is None:
        rng = np.random.default_rng(0)
    means: list[float] = []
    los: list[float] = []
    his: list[float] = []
    ns: list[int] = []
    for d in ds:
        m_, lo_, hi_, n_ = _bootstrap_mean_ci(by[d], rng, n_boot=n_boot)
        means.append(m_)
        los.append(lo_)
        his.append(hi_)
        ns.append(n_)
    return ds, means, los, his, ns


def step7_by_distance_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flat (factor, model, mode, distance, mean, ci_lo, ci_hi, n) table backing the curve figures --
    the quantitative per-distance sample counts + CIs so divergences can be read off numerically."""
    rng = np.random.default_rng(12345)
    factors = ["content_ig", "node_rrwp_ig", "pair_rrwp_ig", "content", "node_rrwp", "pair_rrwp", "both_rrwp"]
    models = ordered_model_names(sorted({str(r.get("model")) for r in rows}))
    out: list[dict[str, Any]] = []
    for factor in factors:
        for model in models:
            for mode, vk in (("functional", "effect_abs"), ("beneficial", "effect_signed")):
                ds, means, los, his, ns = _step7_series(rows, factor, model, mode, vk, rng=rng)
                for d, m_, lo_, hi_, n_ in zip(ds, means, los, his, ns):
                    out.append({
                        "factor": factor, "model": model, "mode": mode, "distance": int(d),
                        "mean": m_, "ci_lo": lo_, "ci_hi": hi_, "n_pairs": int(n_),
                    })
    return out


# Row layout shared by the Step-7 core curve figures: (mode, value column, y-label, panel prefix).
_STEP7_FUNCBENEF_ROWS = [
    ("functional", "effect_abs", "mean |carriage|", "Functional"),
    ("beneficial", "effect_signed", "mean carriage  (<0 = beneficial)", "Beneficial"),
]


def render_carriage_functional_vs_beneficial(
    rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int, completeness_note: Optional[str] = None
) -> None:
    """HEADLINE method comparison (the core Step-7 figure).

    Functional (top row) vs beneficial (bottom row) carriage by distance, for content and node/pair
    structural carriage (columns), overlaying the IG estimator (solid + shaded 95% bootstrap CI) and
    the finite-swap estimator (dashed line + 95% bootstrap CI error bars) for every model (colour). IG
    is the anchor; where the swap error bars OVERLAP the IG band the two estimators agree at that
    distance, where they are disjoint the divergence is significant (both carry uncertainty, so a gap
    is only meaningful when the intervals do not overlap). Shows at a glance that transport is
    functionally far-reaching yet beneficial only short-range. Missing models annotated in red.
    """
    from matplotlib.lines import Line2D

    triples = [
        ("content_ig", "content", "Content"),
        ("node_rrwp_ig", "node_rrwp", "Node RRWP"),
        ("pair_rrwp_ig", "pair_rrwp", "Pair RRWP"),
    ]
    triples = [(fi, fs, t) for fi, fs, t in triples if any(str(r.get("factor")) in (fi, fs) for r in rows)]
    if not triples:
        return
    involved = {f for fi, fs, _ in triples for f in (fi, fs)}
    models = ordered_model_names(sorted({str(r["model"]) for r in rows if str(r.get("factor")) in involved}))
    if not models:
        return
    rng = np.random.default_rng(7)
    palette = plt.cm.tab10.colors
    color = {m: palette[i % len(palette)] for i, m in enumerate(models)}
    fig, axes = plt.subplots(2, len(triples), figsize=(4.9 * len(triples), 8.0), squeeze=False)
    for c, (fi, fs, title) in enumerate(triples):
        for ridx, (mode, vk, ylab, sub) in enumerate(_STEP7_FUNCBENEF_ROWS):
            ax = axes[ridx][c]
            missing: list[str] = []
            nmax = 0
            for m in models:
                ds_i, ys_i, lo_i, hi_i, ns_i = _step7_series(rows, fi, m, mode, vk, rng=rng)
                ds_s, ys_s, lo_s, hi_s, ns_s = _step7_series(rows, fs, m, mode, vk, rng=rng)
                if ds_i:
                    # IG = anchor: solid line + shaded 95% CI band
                    ax.fill_between(ds_i, lo_i, hi_i, color=color[m], alpha=0.15, lw=0)
                    ax.plot(ds_i, ys_i, "-o", ms=3.5, lw=1.8, color=color[m])
                    nmax = max([nmax, *ns_i])
                if ds_s:
                    # finite-swap: dashed line + 95% CI error bars (overlap with the IG band => agree)
                    yerr = np.clip(np.array([[y - lo for y, lo in zip(ys_s, lo_s)],
                                             [hi - y for y, hi in zip(ys_s, hi_s)]]), 0.0, None)
                    ax.errorbar(ds_s, ys_s, yerr=yerr, fmt="--s", ms=3.0, lw=1.3, alpha=0.85,
                                color=color[m], elinewidth=0.9, capsize=2.0)
                    nmax = max([nmax, *ns_s])
                if not ds_i and not ds_s:
                    missing.append(model_label(m))
            if mode == "beneficial":
                ax.axhline(0, color="k", lw=0.7, ls=":")
            ax.set_title(f"{sub}: {title}")
            ax.set_xlabel("carrier<->source distance (hops)")
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)
            if missing:
                ax.text(0.98, 0.03, "no data: " + ", ".join(missing), transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=6.5, color="crimson")
    handles = [Line2D([0], [0], color=color[m], lw=2, label=model_label(m)) for m in models]
    handles += [
        Line2D([0], [0], color="0.25", ls="-", marker="o", ms=4, label="IG (shaded = 95% CI)"),
        Line2D([0], [0], color="0.25", ls="--", marker="s", ms=4, label="finite-swap (bars = 95% CI)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Functional vs beneficial carriage: IG vs finite-swap, content + node/pair structural", y=0.965)
    if completeness_note:
        fig.text(0.5, 0.005, completeness_note, ha="center", va="bottom", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step7_functional_vs_beneficial_carriage.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(figures / "step7_functional_vs_beneficial_carriage.pdf", bbox_inches="tight")
    plt.close(fig)


def render_step7_funcbenef_grid(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int, method: str) -> None:
    """Core Step-7 per-estimator figure: functional (top) vs beneficial (bottom) carriage by distance,
    for content and node/pair structural carriage (columns), models overlaid, with shaded 95%
    bootstrap CIs (band width = per-distance sample size). ``method="ig"`` uses the IG factors
    (content_ig / node_rrwp_ig / pair_rrwp_ig); ``method="swap"`` uses the finite-swap factors
    (content / node_rrwp / pair_rrwp). Missing models are annotated in red.
    """
    fac_by_method = {
        "ig": [("content_ig", "Content"), ("node_rrwp_ig", "Node RRWP"), ("pair_rrwp_ig", "Pair RRWP")],
        "swap": [("content", "Content"), ("node_rrwp", "Node RRWP"), ("pair_rrwp", "Pair RRWP")],
    }
    cols = [(f, t) for f, t in fac_by_method.get(method, []) if any(str(r.get("factor")) == f for r in rows)]
    if not cols:
        return
    col_factors = {f for f, _ in cols}
    models = ordered_model_names(sorted({str(r["model"]) for r in rows if str(r.get("factor")) in col_factors}))
    if not models:
        return
    rng = np.random.default_rng(11)
    palette = plt.cm.tab10.colors
    color = {m: palette[i % len(palette)] for i, m in enumerate(models)}
    method_name = "IG" if method == "ig" else "finite-swap"
    fig, axes = plt.subplots(2, len(cols), figsize=(4.9 * len(cols), 8.0), squeeze=False)
    for c, (factor, title) in enumerate(cols):
        for ridx, (mode, vk, ylab, sub) in enumerate(_STEP7_FUNCBENEF_ROWS):
            ax = axes[ridx][c]
            missing: list[str] = []
            n_lo = math.inf
            n_hi = 0
            for m in models:
                ds, ys, lo, hi, ns = _step7_series(rows, factor, m, mode, vk, rng=rng)
                if ds:
                    ax.fill_between(ds, lo, hi, color=color[m], alpha=0.18, lw=0)
                    ax.plot(ds, ys, marker="o", ms=4, lw=1.8, color=color[m], label=model_label(m))
                    n_lo = min(n_lo, min(ns))
                    n_hi = max(n_hi, max(ns))
                else:
                    missing.append(model_label(m))
            if mode == "beneficial":
                ax.axhline(0, color="k", lw=0.7, ls=":")
            ax.set_title(f"{sub}: {title}")
            ax.set_xlabel("carrier<->source distance (hops)")
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)
            if math.isfinite(n_lo) and n_hi:
                ax.text(0.98, 0.97, f"n/hop: {int(n_lo)}–{int(n_hi)}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=6.5, color="0.4")
            if missing:
                ax.text(0.98, 0.03, "no data: " + ", ".join(missing), transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=6.5, color="crimson")
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Functional vs beneficial carriage by distance -- {method_name} method "
                 f"(content + node/pair structural; shaded = 95% bootstrap CI)")
    fig.tight_layout()
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / f"step7_functional_vs_beneficial_{method}.png", dpi=dpi)
    fig.savefig(figures / f"step7_functional_vs_beneficial_{method}.pdf")
    plt.close(fig)


# IG structural factors (the completeness-checked anchor) used by the reach + agreement summaries.
_STEP7_IG_FACTORS = [("content_ig", "Content"), ("node_rrwp_ig", "Node RRWP"), ("pair_rrwp_ig", "Pair RRWP")]
_STEP7_IG_SWAP_PAIRS = [("content_ig", "content", "Content"),
                        ("node_rrwp_ig", "node_rrwp", "Node RRWP"),
                        ("pair_rrwp_ig", "pair_rrwp", "Pair RRWP")]


def _step7_pairs(rows: Sequence[Mapping[str, Any]], factor: str, model: str, mode: str, value_key: str) -> list[tuple[int, float]]:
    """Raw (integer distance, value) pairs for one (factor, model, mode) -- backs the reach bootstrap."""
    out: list[tuple[int, float]] = []
    for r in rows:
        if str(r.get("factor")) != factor or str(r.get("model")) != model or str(r.get("mode")) != mode:
            continue
        d = safe_float(r.get("distance"))
        v = safe_float(r.get(value_key))
        if math.isfinite(d) and math.isfinite(v):
            out.append((int(round(d)), v))
    return out


def _reach_stats(pairs: Sequence[tuple[int, float]], tau: int, rng: np.random.Generator, *, n_boot: int = 800) -> Optional[dict[str, float]]:
    """Mass-weighted carriage reach + long-range fraction (>= tau) with percentile-bootstrap CIs.

    reach = sum(d * |c|) / sum(|c|)  (mean carrier<->source distance weighted by |carriage| mass);
    far_frac = fraction of |carriage| mass at distance >= tau. Both are computed over the SAME
    (carrier, source) pair set for every model (common graph geometry), so model differences are
    model-driven, not geometry -- and neither involves an ablation, so both are confound-free.
    """
    if not pairs:
        return None
    d = np.asarray([p[0] for p in pairs], dtype=float)
    w = np.abs(np.asarray([p[1] for p in pairs], dtype=float))
    tot = float(w.sum())
    if tot <= 0.0:
        return None
    reach = float((d * w).sum() / tot)
    far = float(w[d >= tau].sum() / tot)
    n = len(pairs)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    dd, ww = d[idx], w[idx]
    tt = ww.sum(axis=1)
    tt[tt <= 0] = np.nan
    rb = (dd * ww).sum(axis=1) / tt
    fb = (ww * (dd >= tau)).sum(axis=1) / tt

    def _ci(a: np.ndarray) -> tuple[float, float]:
        a = a[np.isfinite(a)]
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if a.size else (float("nan"), float("nan"))

    r_lo, r_hi = _ci(rb)
    f_lo, f_hi = _ci(fb)
    return {"reach": reach, "reach_lo": r_lo, "reach_hi": r_hi,
            "far_frac": far, "far_lo": f_lo, "far_hi": f_hi, "n_pairs": float(n), "total_mass": tot}


def step7_reach_summary_rows(rows: Sequence[Mapping[str, Any]], *, tau: int = 3) -> list[dict[str, Any]]:
    """Per (model, IG factor) carriage reach + long-range fraction -- the quantitative backing for the
    dense-vs-1hop and global-vs-local contrasts (uses the completeness-checked IG functional carriage)."""
    rng = np.random.default_rng(2024)
    models = ordered_model_names(sorted({str(r.get("model")) for r in rows}))
    out: list[dict[str, Any]] = []
    for factor, _ in _STEP7_IG_FACTORS:
        for m in models:
            st = _reach_stats(_step7_pairs(rows, factor, m, "functional", "effect_abs"), tau, rng)
            if st is not None:
                out.append({"factor": factor, "model": m, "mode": "functional", "tau": int(tau), **st})
    return out


def render_step7_reach_summary(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int, tau: int = 3) -> None:
    """CORE contrast figure: how far each model transports content vs node/pair structure.

    Left: mean carriage distance (reach). Right: long-range fraction (share of |carriage| at
    distance >= tau). Grouped by IG factor (Content / Node RRWP / Pair RRWP), bars per model, with
    95% bootstrap CIs. Answers at a glance: (i) dense GT reaches further than 1-hop (the core
    difference), and (ii) global-RRWP 1-hop vs local-RRWP 1-hop differ specifically on PAIR RRWP
    reach -- the node/pair split isolating what the global RRWP buys. Confound-free (no ablation).
    """
    stats = step7_reach_summary_rows(rows, tau=tau)
    if not stats:
        return
    factors = [(f, t) for f, t in _STEP7_IG_FACTORS if any(s["factor"] == f for s in stats)]
    models = ordered_model_names(sorted({str(s["model"]) for s in stats}))
    if not factors or not models:
        return
    by = {(str(s["factor"]), str(s["model"])): s for s in stats}
    palette = plt.cm.tab10.colors
    color = {m: palette[i % len(palette)] for i, m in enumerate(models)}
    fig, axes = plt.subplots(1, 2, figsize=(6.6 + 1.2 * len(factors), 5.0), squeeze=False)
    metrics = [("reach", "reach_lo", "reach_hi", "mean carriage distance (hops)"),
               ("far_frac", "far_lo", "far_hi", f"long-range fraction (|carriage| at d>={tau})")]
    x = np.arange(len(factors))
    width = 0.8 / max(1, len(models))
    for ax, (mkey, lo_k, hi_k, ylab) in zip(axes[0], metrics):
        for i, m in enumerate(models):
            xs = x + (i - (len(models) - 1) / 2.0) * width
            heights, yerr_lo, yerr_hi = [], [], []
            for f, _ in factors:
                s = by.get((f, m))
                v = float(s[mkey]) if s else 0.0
                heights.append(v)
                yerr_lo.append(0.0 if not s else max(0.0, v - float(s[lo_k])))
                yerr_hi.append(0.0 if not s else max(0.0, float(s[hi_k]) - v))
            ax.bar(xs, heights, width=width * 0.95, color=color[m], label=model_label(m),
                   yerr=[yerr_lo, yerr_hi], capsize=2.5, error_kw={"lw": 0.9, "alpha": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels([t for _, t in factors])
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle("Carriage reach: how far each model transports content vs node/pair structure (IG functional; 95% CI)")
    fig.tight_layout()
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step7_carriage_reach_summary.png", dpi=dpi)
    fig.savefig(figures / "step7_carriage_reach_summary.pdf")
    plt.close(fig)


def step7_method_agreement_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pearson r + OLS slope of finite-swap vs IG, per (factor, mode), pooled over (model, distance)
    means -- the quantitative IG<->swap consistency check (r~1, slope~1 => the two estimators agree)."""
    models = ordered_model_names(sorted({str(r.get("model")) for r in rows}))
    out: list[dict[str, Any]] = []
    for fi, fs, _ in _STEP7_IG_SWAP_PAIRS:
        for mode, vk in (("functional", "effect_abs"), ("beneficial", "effect_signed")):
            xs: list[float] = []
            ys: list[float] = []
            for m in models:
                di, mi, *_ = _step7_series(rows, fi, m, mode, vk)
                dsw, msw, *_ = _step7_series(rows, fs, m, mode, vk)
                ig_map = dict(zip(di, mi))
                sw_map = dict(zip(dsw, msw))
                for d in sorted(set(ig_map) & set(sw_map)):
                    xs.append(ig_map[d])
                    ys.append(sw_map[d])
            rec: dict[str, Any] = {"factor": fi, "mode": mode, "n_points": len(xs)}
            if len(xs) >= 3 and np.std(xs) > 0 and np.std(ys) > 0:
                rec["pearson_r"] = float(np.corrcoef(xs, ys)[0, 1])
                rec["slope"] = float(np.polyfit(xs, ys, 1)[0])
            else:
                rec["pearson_r"] = float("nan")
                rec["slope"] = float("nan")
            out.append(rec)
    return out


def render_step7_method_agreement(
    rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int, completeness_note: Optional[str] = None
) -> None:
    """Q1 -- IG vs finite-swap consistency. Scatter of swap mean (y) against IG mean (x) across every
    (model, distance) point, per factor (columns) and mode (rows), with the y=x line and Pearson r +
    OLS slope annotated. Points on y=x with r~1 => the two estimators agree; systematic departures
    localise where (which factor / near vs far) they diverge. IG is the anchor (it is completeness-
    exact, Sum C == y_hat_clean - y_hat_base); the swap is the on-manifold corroboration.
    """
    triples = [(fi, fs, t) for fi, fs, t in _STEP7_IG_SWAP_PAIRS
               if any(str(r.get("factor")) == fi for r in rows) and any(str(r.get("factor")) == fs for r in rows)]
    if not triples:
        return
    models = ordered_model_names(sorted({str(r["model"]) for r in rows}))
    palette = plt.cm.tab10.colors
    color = {m: palette[i % len(palette)] for i, m in enumerate(models)}
    # Precompute matched (IG, swap) points per (mode, factor, model); drop mode-rows with no data.
    all_modes = [("functional", "effect_abs", "Functional"), ("beneficial", "effect_signed", "Beneficial")]
    pts: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for mode, vk, _ in all_modes:
        for fi, fs, _t in triples:
            for m in models:
                di, mi, *_ = _step7_series(rows, fi, m, mode, vk)
                dsw, msw, *_ = _step7_series(rows, fs, m, mode, vk)
                ig_map, sw_map = dict(zip(di, mi)), dict(zip(dsw, msw))
                common = sorted(set(ig_map) & set(sw_map))
                if common:
                    pts[(mode, fi, m)] = [(ig_map[d], sw_map[d]) for d in common]
    modes = [(mode, vk, sub) for mode, vk, sub in all_modes
             if any((mode, fi, m) in pts for fi, _fs, _t in triples for m in models)]
    if not modes:
        return
    fig, axes = plt.subplots(len(modes), len(triples), figsize=(4.6 * len(triples), 4.2 * len(modes)), squeeze=False)
    for c, (fi, fs, title) in enumerate(triples):
        for ridx, (mode, vk, sub) in enumerate(modes):
            ax = axes[ridx][c]
            xs: list[float] = []
            ys: list[float] = []
            for m in models:
                mpts = pts.get((mode, fi, m))
                if mpts:
                    ax.scatter([p[0] for p in mpts], [p[1] for p in mpts],
                               s=22, color=color[m], alpha=0.8, edgecolors="none",
                               label=(model_label(m) if ridx == 0 and c == 0 else None))
                    xs.extend(p[0] for p in mpts)
                    ys.extend(p[1] for p in mpts)
            if xs:
                lo, hi = min(xs + ys), max(xs + ys)
                pad = 0.05 * (hi - lo + 1e-9)
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k:", lw=0.8)
                if len(xs) >= 3 and np.std(xs) > 0 and np.std(ys) > 0:
                    r = float(np.corrcoef(xs, ys)[0, 1])
                    slope = float(np.polyfit(xs, ys, 1)[0])
                    ax.text(0.03, 0.97, f"r = {r:.2f}\nslope = {slope:.2f}\nn = {len(xs)}",
                            transform=ax.transAxes, va="top", ha="left", fontsize=8,
                            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
            ax.set_title(f"{sub}: {title}")
            ax.set_xlabel("IG mean (anchor)")
            ax.set_ylabel("finite-swap mean")
            ax.grid(alpha=0.3)
    if any(axes[0][0].get_legend_handles_labels()[1]):
        axes[0][0].legend(fontsize=7, loc="lower right")
    fig.suptitle("IG vs finite-swap agreement (points on y=x, r~1 => estimators consistent)", y=0.99)
    if completeness_note:
        fig.text(0.5, 0.005, completeness_note, ha="center", va="bottom", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step7_ig_vs_swap_agreement.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(figures / "step7_ig_vs_swap_agreement.pdf", bbox_inches="tight")
    plt.close(fig)


def render_symbolic_structural_by_distance(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    clean = [
        r
        for r in rows
        if math.isfinite(safe_float(r.get("distance"))) and math.isfinite(safe_float(r.get("effect_abs")))
        and str(r.get("mode", "functional")) == "functional"
    ]
    if not clean:
        return
    panels = [
        ("content_ig", "Content carriage (IG)"),
        ("node_rrwp_ig", "Node RRWP carriage (IG)"),
        ("pair_rrwp_ig", "Pair RRWP carriage (IG)"),
        ("node_rrwp", "Node RRWP carriage (swap)"),
        ("pair_rrwp", "Pair RRWP carriage (swap)"),
        ("both_rrwp", "Node + pair RRWP carriage (swap)"),
    ]
    figures = ensure_dir(artifact_root / "figures")
    available_panels = [(factor, title) for factor, title in panels if any(str(r.get("factor")) == factor for r in clean)]
    if not available_panels:
        return
    ncols = min(2, len(available_panels))
    nrows = int(math.ceil(len(available_panels) / max(ncols, 1)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.7 * ncols, 4.2 * nrows), constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for ax, (factor, title) in zip(axes_flat, available_panels):
        sub = [r for r in clean if str(r.get("factor")) == factor]
        graph_distance_sums: dict[tuple[str, str, int], float] = {}
        graph_totals: dict[tuple[str, str], float] = {}
        for r in sub:
            key = (str(r.get("model")), str(r.get("graph_id")))
            distance = int(round(safe_float(r.get("distance"))))
            value = safe_float(r.get("effect_abs"))
            if not math.isfinite(value):
                continue
            graph_distance_sums[(key[0], key[1], distance)] = graph_distance_sums.get((key[0], key[1], distance), 0.0) + value
            graph_totals[key] = graph_totals.get(key, 0.0) + value
        grouped: dict[tuple[str, int], list[float]] = {}
        for (model_name, graph_id, distance), value in graph_distance_sums.items():
            denom = max(graph_totals.get((model_name, graph_id), 0.0), EPS)
            grouped.setdefault((model_name, distance), []).append(value / denom)
        for model in sorted({m for m, _ in grouped}):
            distances = sorted(distance for m, distance in grouped if m == model)
            values = [float(np.nanmean(grouped[(model, distance)])) for distance in distances]
            ax.plot(distances, values, marker="o", linewidth=1.8, label=model_label(model))
        ax.set_title(title)
        ax.set_xlabel("Molecular hop distance")
        ax.set_ylabel("Share of own |carriage| by distance")
        ax.set_ylim(bottom=0.0, top=1.0)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes_flat[len(available_panels):]:
        ax.set_axis_off()
    fig.suptitle("Source-specific carriage by distance: content vs long-RRWP structure")
    fig.savefig(figures / "step7_symbolic_structural_carriage_by_distance.png", dpi=dpi)
    fig.savefig(figures / "step7_symbolic_structural_carriage_by_distance.pdf")
    plt.close(fig)

    fig_abs, axes_abs = plt.subplots(nrows, ncols, figsize=(6.7 * ncols, 4.2 * nrows), constrained_layout=True)
    axes_abs_flat = np.asarray(axes_abs).reshape(-1)
    for ax, (factor, title) in zip(axes_abs_flat, available_panels):
        sub = [r for r in clean if str(r.get("factor")) == factor]
        grouped: dict[tuple[str, int], list[float]] = {}
        for r in sub:
            distance = int(round(safe_float(r.get("distance"))))
            grouped.setdefault((str(r.get("model")), distance), []).append(safe_float(r.get("effect_abs")))
        for model in sorted({m for m, _ in grouped}):
            distances = sorted(distance for m, distance in grouped if m == model)
            values = [float(np.nanmean(grouped[(model, distance)])) for distance in distances]
            ax.plot(distances, values, marker="o", linewidth=1.8, label=model_label(model))
        ax.set_title(title)
        ax.set_xlabel("Molecular hop distance")
        ax.set_ylabel("Mean |carriage| (prediction units)")
        ax.set_ylim(bottom=0.0)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes_abs_flat[len(available_panels):]:
        ax.set_axis_off()
    fig_abs.suptitle("Source-specific carriage by distance: absolute content and long-RRWP effects")
    fig_abs.savefig(figures / "step7_symbolic_structural_carriage_absolute_by_distance.png", dpi=dpi)
    fig_abs.savefig(figures / "step7_symbolic_structural_carriage_absolute_by_distance.pdf")
    plt.close(fig_abs)

    totals: dict[tuple[str, str], list[float]] = {}
    for factor, _ in panels:
        sub = [r for r in clean if str(r.get("factor")) == factor]
        per_graph: dict[tuple[str, str], float] = {}
        for r in sub:
            key = (str(r.get("model")), str(r.get("graph_id")))
            value = safe_float(r.get("effect_abs"))
            if math.isfinite(value):
                per_graph[key] = per_graph.get(key, 0.0) + value
        for (model_name, _graph_id), value in per_graph.items():
            totals.setdefault((factor, model_name), []).append(value)
    total_rows: list[dict[str, Any]] = []
    fig_tot, ax_tot = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    factors = [factor for factor, _ in panels if any(key[0] == factor for key in totals)]
    model_names = ordered_model_names([model for _factor, model in totals])
    x = np.arange(len(factors), dtype=float)
    width = 0.8 / max(len(model_names), 1)
    for idx, model_name in enumerate(model_names):
        means = []
        lows = []
        highs = []
        for factor in factors:
            vals = [v for v in totals.get((factor, model_name), []) if math.isfinite(v)]
            if vals:
                mean, lo, hi = bootstrap_ci(vals, seed=stable_seed("structural_total", factor, model_name), draws=500)
            else:
                mean, lo, hi = float("nan"), float("nan"), float("nan")
            means.append(mean)
            lows.append(lo)
            highs.append(hi)
            total_rows.append(
                {
                    "factor": factor,
                    "model": model_name,
                    "mean_total_abs_carriage": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "graphs": len(vals),
                }
            )
        pos = x - 0.4 + width / 2 + idx * width
        y = np.asarray(means, dtype=float)
        yerr = np.vstack([
            np.maximum(0.0, y - np.asarray(lows, dtype=float)),
            np.maximum(0.0, np.asarray(highs, dtype=float) - y),
        ])
        ax_tot.bar(pos, y, width=width, yerr=yerr, capsize=2.5, label=model_label(model_name), alpha=0.9)
    ax_tot.set_xticks(x)
    ax_tot.set_xticklabels([dict(panels).get(factor, factor) for factor in factors], rotation=15, ha="right")
    ax_tot.set_ylabel("Total |carriage| per graph (prediction units)")
    ax_tot.set_title("Diagnostic total carriage: content and pair-RRWP effects")
    ax_tot.legend(frameon=False, fontsize=8)
    write_csv(artifact_root / "metrics" / "step7_symbolic_structural_component_totals.csv", total_rows)
    fig_tot.savefig(figures / "step7_symbolic_structural_component_totals.png", dpi=dpi)
    fig_tot.savefig(figures / "step7_symbolic_structural_component_totals.pdf")
    plt.close(fig_tot)

    global_model = "grit_1hop"
    local_model = "grit_1hop_localrrwp"
    contrast_factors = [
        ("node_rrwp", "Node RRWP"),
        ("pair_rrwp", "Pair RRWP"),
        ("both_rrwp", "Node + pair RRWP"),
    ]
    has_global = any(str(r.get("model")) == global_model for r in clean)
    has_local = any(str(r.get("model")) == local_model for r in clean)
    if has_global and has_local:
        per_graph_distance: dict[tuple[str, str, str, int], float] = {}
        for r in clean:
            factor = str(r.get("factor"))
            if factor not in {f for f, _ in contrast_factors}:
                continue
            model = str(r.get("model"))
            if model not in {global_model, local_model}:
                continue
            distance = int(round(safe_float(r.get("distance"))))
            value = safe_float(r.get("effect_abs"))
            if not math.isfinite(value):
                continue
            key = (model, str(r.get("graph_id")), factor, distance)
            per_graph_distance[key] = per_graph_distance.get(key, 0.0) + value
        contrast_rows: list[dict[str, Any]] = []
        panel_payload: list[tuple[str, str, dict[int, list[float]]]] = []
        for factor, title in contrast_factors:
            paired_values: dict[int, list[float]] = {}
            graph_distance_keys = {
                (graph_id, distance)
                for model, graph_id, f, distance in per_graph_distance
                if f == factor and model in {global_model, local_model}
            }
            for graph_id, distance in sorted(graph_distance_keys):
                g_val = per_graph_distance.get((global_model, graph_id, factor, distance))
                l_val = per_graph_distance.get((local_model, graph_id, factor, distance))
                if g_val is None or l_val is None:
                    continue
                paired_values.setdefault(distance, []).append(g_val - l_val)
            if paired_values:
                panel_payload.append((factor, title, paired_values))
        if not panel_payload:
            return
        fig_width = 6.8 if len(panel_payload) == 1 else 4.8 * len(panel_payload)
        fig_con, axes_con = plt.subplots(1, len(panel_payload), figsize=(fig_width, 4.6), constrained_layout=True)
        axes_con_flat = np.asarray(axes_con).reshape(-1)
        for ax, (factor, title, paired_values) in zip(axes_con_flat, panel_payload):
            distances = sorted(paired_values)
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for distance in distances:
                vals = [v for v in paired_values[distance] if math.isfinite(v)]
                if vals:
                    mean, lo, hi = bootstrap_ci(
                        vals,
                        seed=stable_seed("symbolic_global_local_contrast", factor, distance),
                        draws=500,
                    )
                else:
                    mean, lo, hi = float("nan"), float("nan"), float("nan")
                means.append(mean)
                lows.append(lo)
                highs.append(hi)
                contrast_rows.append(
                    {
                        "factor": factor,
                        "distance": distance,
                        "mean_global_minus_local_abs_carriage": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "graphs": len(vals),
                    }
                )
            x = np.asarray(distances, dtype=float)
            y = np.asarray(means, dtype=float)
            lo_arr = np.asarray(lows, dtype=float)
            hi_arr = np.asarray(highs, dtype=float)
            ax.plot(x, y, marker="o", linewidth=1.8)
            mask = np.isfinite(y) & np.isfinite(lo_arr) & np.isfinite(hi_arr)
            if bool(mask.any()):
                ax.fill_between(x[mask], lo_arr[mask], hi_arr[mask], alpha=0.14)
            ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1)
            ax.set_title(title)
            ax.set_xlabel("Molecular hop distance")
            ax.set_ylabel("Global 1-hop - local 1-hop\nmean |RRWP carriage|")
        if len(panel_payload) == 1:
            fig_con.suptitle("Diagnostic: global-minus-local RRWP carriage contrast")
        else:
            fig_con.suptitle("Where global RRWP changes structural carriage relative to local RRWP")
        write_csv(artifact_root / "metrics" / "step7_symbolic_global_vs_local_rrwp_contrast.csv", contrast_rows)
        fig_con.savefig(figures / "step7_symbolic_global_vs_local_rrwp_contrast.png", dpi=dpi)
        fig_con.savefig(figures / "step7_symbolic_global_vs_local_rrwp_contrast.pdf")
        plt.close(fig_con)


def rrwp_distance_ablation_types(raw: Any) -> list[str]:
    if raw is None:
        raw = ["node", "pair", "both"]
    if isinstance(raw, str):
        values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, Sequence):
        values = [str(part).strip().lower() for part in raw if str(part).strip()]
    else:
        values = [str(raw).strip().lower()]
    allowed = {"node", "pair", "both"}
    out = [value for value in values if value in allowed]
    return out or ["node", "pair", "both"]


def _rrwp_replace_channels(
    tensor: torch.Tensor, rows: torch.Tensor, *, channel_start: int, replacement: str,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    out = tensor.detach().clone()
    if out.dim() != 2 or int(out.size(0)) == 0 or int(out.size(1)) <= int(channel_start):
        return out
    row_idx = rows.detach().cpu().long().reshape(-1)
    row_idx = row_idx[(row_idx >= 0) & (row_idx < int(out.size(0)))]
    if row_idx.numel() == 0:
        return out
    start = max(0, min(int(channel_start), int(out.size(1))))
    mode = str(replacement).strip().lower()
    if mode == "donor":
        # On-manifold swap: replace long-range channels [start:] with a random OTHER row's real
        # values -- the structural analog of the content donor swap, so content and structure use
        # the same on-manifold perturbation and are directly comparable. Falls back to graph-mean
        # for a singleton row set.
        total = int(out.size(0))
        if total <= 1:
            out[row_idx, start:] = out[:, start:].mean(dim=0, keepdim=True).to(out)
            return out
        r = rng or random.Random(0)
        for ridx in row_idx.tolist():
            donor = r.randrange(total)
            tries = 0
            while donor == int(ridx) and tries < 8:
                donor = r.randrange(total)
                tries += 1
            out[int(ridx), start:] = out[donor, start:]
        return out
    if mode in {"mean", "dataset_mean", "graph_mean"}:
        repl = out[:, start:].mean(dim=0, keepdim=True)
        out[row_idx, start:] = repl.to(dtype=out.dtype, device=out.device)
    else:
        out[row_idx, start:] = 0
    return out


def _rrwp_distance_bin_pair_mask(
    rrwp_index: torch.Tensor,
    dist: torch.Tensor,
    spec: Mapping[str, Any],
) -> torch.Tensor:
    if rrwp_index.dim() != 2 or int(rrwp_index.size(0)) != 2:
        return torch.zeros(0, dtype=torch.bool)
    pair_index = rrwp_index.detach().cpu().long()
    src = pair_index[0]
    dst = pair_index[1]
    n = int(dist.size(0))
    valid = (src >= 0) & (dst >= 0) & (src < n) & (dst < n)
    pair_dist = torch.full((int(pair_index.size(1)),), float("inf"), dtype=torch.float32)
    if bool(valid.any()):
        pair_dist[valid] = dist.detach().cpu().float()[dst[valid], src[valid]]
    low = int(spec.get("min", 0))
    high_raw = spec.get("max")
    high = None if high_raw in (None, "", "none", "None") else int(high_raw)
    mask = torch.isfinite(pair_dist) & (pair_dist >= float(low))
    if high is not None:
        mask = mask & (pair_dist <= float(high))
    return mask


def _rrwp_endpoint_nodes(rrwp_index: torch.Tensor, pair_mask: torch.Tensor, n_nodes: int) -> torch.Tensor:
    if pair_mask.numel() == 0 or not bool(pair_mask.any()):
        return torch.zeros(0, dtype=torch.long)
    pair_index = rrwp_index.detach().cpu().long()
    endpoints = torch.unique(pair_index[:, pair_mask].reshape(-1))
    endpoints = endpoints[(endpoints >= 0) & (endpoints < int(n_nodes))]
    return endpoints.long()


def _distance_bin_endpoint_nodes(dist: torch.Tensor, spec: Mapping[str, Any]) -> torch.Tensor:
    dist_cpu = dist.detach().cpu().float()
    low = int(spec.get("min", 0))
    high_raw = spec.get("max")
    high = None if high_raw in (None, "", "none", "None") else int(high_raw)
    mask = torch.isfinite(dist_cpu) & (dist_cpu >= float(low))
    if high is not None:
        mask = mask & (dist_cpu <= float(high))
    mask = mask & (dist_cpu > 0)
    if not bool(mask.any()):
        return torch.zeros(0, dtype=torch.long)
    endpoints = torch.unique(torch.nonzero(mask, as_tuple=False).reshape(-1))
    n_nodes = int(dist_cpu.size(0))
    return endpoints[(endpoints >= 0) & (endpoints < n_nodes)].long()


def _available_graph_keys(graph: Any) -> list[str]:
    keys: list[str] = []
    try:
        keys.extend([str(key) for key in graph.keys()])
    except Exception:
        pass
    try:
        keys.extend([str(key) for key in graph.to_dict().keys()])
    except Exception:
        pass
    if hasattr(graph, "__dict__"):
        keys.extend([str(key) for key in vars(graph).keys() if not str(key).startswith("_")])
    return sorted(set(keys))


def _graph_tensor_field(graph: Any, names: Sequence[str]) -> tuple[Optional[str], Optional[torch.Tensor]]:
    keys = set(_available_graph_keys(graph))
    for name in names:
        value = None
        if name in keys:
            try:
                value = graph[name]
            except Exception:
                value = None
        if value is None and hasattr(graph, name):
            try:
                value = getattr(graph, name)
            except Exception:
                value = None
        if isinstance(value, torch.Tensor):
            return name, value
    return None, None


def _dense_pair_rrwp_to_sparse(rrwp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dense = rrwp.detach()
    if dense.dim() != 3:
        raise ValueError(f"expected dense pair RRWP [N,N,K], got {tuple(dense.shape)}")
    n = int(dense.size(0))
    rows = torch.arange(n, dtype=torch.long)
    dst, src = torch.meshgrid(rows, rows, indexing="ij")
    # Official RRWP sparse index convention is [source, destination].
    index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)
    values = dense.reshape(n * n, int(dense.size(-1)))
    return index, values


def rrwp_distance_ablation_rows_for_graph(
    model: ModelRun,
    graph: Any,
    gid: str,
    *,
    pair_id: Optional[str] = None,
    bins: Sequence[Mapping[str, Any]],
    ablation_types: Sequence[str],
    channel_start: int,
    replacement: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Ablate raw GRIT RRWP channels by distance bin before official RRWP encoders run.

    The intervention is split into node RRWP, pair RRWP, and both.  Channels before
    ``channel_start`` are left untouched so the default preserves self and one-hop
    terms while removing longer-range structural encodings.
    """

    adapter = model.adapter
    stable_pair_id = str(pair_id or gid)
    if not hasattr(adapter, "forward_minimal"):
        return [], None
    try:
        clean_cache = adapter.forward_minimal(graph)
    except Exception as exc:
        return [], {
            "status": "failed_clean_forward",
            "model": model.name,
            "graph_id": gid,
            "pair_id": stable_pair_id,
            "error": str(exc),
        }
    extras = getattr(clean_cache, "extras", None) or {}
    raw_rrwp = extras.get("raw_rrwp")
    raw_rrwp_val = extras.get("raw_rrwp_val")
    raw_rrwp_index = extras.get("raw_rrwp_index")
    raw_rrwp_key = extras.get("raw_rrwp_key")
    raw_rrwp_val_key = extras.get("raw_rrwp_val_key")
    raw_rrwp_index_key = extras.get("raw_rrwp_index_key")
    data_keys = list(extras.get("data_keys") or [])
    graph_keys = _available_graph_keys(graph)
    if not isinstance(raw_rrwp, torch.Tensor):
        raw_rrwp_key, raw_rrwp = _graph_tensor_field(graph, ("rrwp", "pestat_RRWP", "pestat_rrwp", "RWSE", "rwse"))
    if not isinstance(raw_rrwp_val, torch.Tensor):
        raw_rrwp_val_key, raw_rrwp_val = _graph_tensor_field(
            graph,
            ("rrwp_val", "rrwp_values", "rrwp_value", "pestat_RRWP_val", "pestat_rrwp_val"),
        )
    if not isinstance(raw_rrwp_index, torch.Tensor):
        raw_rrwp_index_key, raw_rrwp_index = _graph_tensor_field(
            graph,
            ("rrwp_index", "rrwp_idx", "pestat_RRWP_index", "pestat_rrwp_index"),
        )
    dense_pair_rrwp = False
    raw_rrwp_dense_pair: Optional[torch.Tensor] = None
    if (
        not isinstance(raw_rrwp_val, torch.Tensor)
        and not isinstance(raw_rrwp_index, torch.Tensor)
        and isinstance(raw_rrwp, torch.Tensor)
        and raw_rrwp.dim() == 3
    ):
        raw_rrwp_dense_pair = raw_rrwp.detach()
        raw_rrwp_index, raw_rrwp_val = _dense_pair_rrwp_to_sparse(raw_rrwp.detach().cpu())
        raw_rrwp_val_key = raw_rrwp_key
        raw_rrwp_index_key = f"{raw_rrwp_key}_dense_index"
        dense_pair_rrwp = True
    if not (
        isinstance(raw_rrwp_val, torch.Tensor)
        and isinstance(raw_rrwp_index, torch.Tensor)
        and raw_rrwp_val.dim() == 2
        and raw_rrwp_index.dim() == 2
        and int(raw_rrwp_index.size(0)) == 2
        and int(raw_rrwp_val.size(0)) == int(raw_rrwp_index.size(1))
    ):
        return [], {
            "status": "missing_raw_rrwp_pair_fields",
            "model": model.name,
            "graph_id": gid,
            "pair_id": stable_pair_id,
            "cache_data_keys": ",".join(data_keys),
            "graph_keys": ",".join(graph_keys),
            "raw_rrwp_key": raw_rrwp_key or "",
            "raw_rrwp_val_key": raw_rrwp_val_key or "",
            "raw_rrwp_index_key": raw_rrwp_index_key or "",
            "raw_rrwp_shape": tuple(raw_rrwp.shape) if isinstance(raw_rrwp, torch.Tensor) else "",
            "raw_rrwp_val_shape": tuple(raw_rrwp_val.shape) if isinstance(raw_rrwp_val, torch.Tensor) else "",
            "raw_rrwp_index_shape": tuple(raw_rrwp_index.shape) if isinstance(raw_rrwp_index, torch.Tensor) else "",
        }
    if not isinstance(raw_rrwp, torch.Tensor) or raw_rrwp.dim() != 2 or dense_pair_rrwp:
        raw_rrwp = None
    clean_pred = safe_float(clean_cache.prediction.detach().reshape(-1)[0].cpu().item())
    target = graph_label(graph)
    dist = distance_matrix(graph).detach().cpu().float()
    n_nodes = graph_num_nodes(graph)
    rows: list[dict[str, Any]] = []
    for spec in bins:
        label = str(spec.get("label") or "")
        pair_mask = _rrwp_distance_bin_pair_mask(raw_rrwp_index, dist, spec)
        pair_rows = torch.nonzero(pair_mask, as_tuple=False).reshape(-1).long()
        endpoint_nodes = _rrwp_endpoint_nodes(raw_rrwp_index, pair_mask, n_nodes)
        if endpoint_nodes.numel() == 0:
            endpoint_nodes = _distance_bin_endpoint_nodes(dist, spec)
        if pair_rows.numel() == 0 and endpoint_nodes.numel() == 0:
            continue
        for ablation_type in ablation_types:
            node_override = None
            pair_override = None
            if dense_pair_rrwp and isinstance(raw_rrwp_dense_pair, torch.Tensor):
                if ablation_type == "node":
                    continue
                flat = raw_rrwp_dense_pair.detach().reshape(-1, int(raw_rrwp_dense_pair.size(-1)))
                flat_override = _rrwp_replace_channels(
                    flat,
                    pair_rows,
                    channel_start=channel_start,
                    replacement=replacement,
                )
                node_override = flat_override.reshape_as(raw_rrwp_dense_pair)
            elif ablation_type in {"node", "both"} and isinstance(raw_rrwp, torch.Tensor):
                node_override = _rrwp_replace_channels(
                    raw_rrwp,
                    endpoint_nodes,
                    channel_start=channel_start,
                    replacement=replacement,
                )
            if (not dense_pair_rrwp) and ablation_type in {"pair", "both"}:
                pair_override = _rrwp_replace_channels(
                    raw_rrwp_val,
                    pair_rows,
                    channel_start=channel_start,
                    replacement=replacement,
                )
            if node_override is None and pair_override is None:
                continue
            try:
                ablated_cache = adapter.forward_minimal(
                    graph,
                    rrwp_node_override=node_override,
                    rrwp_val_override=pair_override,
                )
                ablated_pred = safe_float(ablated_cache.prediction.detach().reshape(-1)[0].cpu().item())
            except Exception as exc:
                rows.append(
                    {
                        "model": model.name,
                        "role": model.role,
                        "graph_id": gid,
                        "pair_id": stable_pair_id,
                        "distance_bin": label,
                        "distance_min": int(spec.get("min", 0)),
                        "distance_max": spec.get("max") if spec.get("max") is not None else "",
                        "ablation_type": ablation_type,
                        "status": "failed_ablation_forward",
                        "error": str(exc),
                        "n_pair_entries": int(pair_rows.numel()),
                        "n_endpoint_nodes": int(endpoint_nodes.numel()),
                    }
                )
                continue
            clean_mae = abs(clean_pred - target) if math.isfinite(target) else float("nan")
            ablated_mae = abs(ablated_pred - target) if math.isfinite(target) else float("nan")
            rows.append(
                {
                    "model": model.name,
                    "role": model.role,
                    "graph_id": gid,
                    "pair_id": stable_pair_id,
                    "distance_bin": label,
                    "distance_min": int(spec.get("min", 0)),
                    "distance_max": spec.get("max") if spec.get("max") is not None else "",
                    "ablation_type": ablation_type,
                    "status": "complete",
                    "channel_start": int(channel_start),
                    "replacement": replacement,
                    "n_pair_entries": int(pair_rows.numel()),
                    "n_endpoint_nodes": int(endpoint_nodes.numel()),
                    "clean_prediction": clean_pred,
                    "ablated_prediction": ablated_pred,
                    "delta_pred": ablated_pred - clean_pred,
                    "abs_delta_pred": abs(ablated_pred - clean_pred),
                    "raw_rrwp_key": raw_rrwp_key or "",
                    "raw_rrwp_val_key": raw_rrwp_val_key or "",
                    "raw_rrwp_index_key": raw_rrwp_index_key or "",
                    "dense_pair_rrwp": dense_pair_rrwp,
                    "target": target,
                    "clean_mae": clean_mae,
                    "ablated_mae": ablated_mae,
                    "delta_mae": ablated_mae - clean_mae if math.isfinite(clean_mae) and math.isfinite(ablated_mae) else float("nan"),
                }
            )
    return rows, {"status": "complete", "model": model.name, "graph_id": gid, "pair_id": stable_pair_id, "rows": len(rows)}


def graph_rrwp_contrast_metric_row(
    graph: Any,
    gid: str,
    *,
    pair_id: Optional[str] = None,
    tau: int,
    seed: int,
    max_cut_pairs: int,
) -> dict[str, Any]:
    dist = distance_matrix(graph).detach().cpu().float()
    n_nodes = graph_num_nodes(graph)
    edge_index = getattr(graph, "edge_index", None)
    undirected_edges: set[tuple[int, int]] = set()
    if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2:
        for u, v in edge_index.detach().cpu().long().t().tolist():
            if int(u) == int(v):
                continue
            a, b = sorted((int(u), int(v)))
            undirected_edges.add((a, b))
    summary = graph_distance_summary(dist)
    try:
        nx_graph = graph_to_networkx(pyg_graph_view(graph), undirected=True)
        articulation_count = len(list(nx.articulation_points(nx_graph))) if n_nodes else 0
    except Exception:
        articulation_count = 0
    sampled = far_pairs(dist, tau, max_pairs=max_cut_pairs, seed=seed)
    cut_sizes: list[int] = []
    cut_disconnects: list[bool] = []
    for carrier, source in sampled:
        try:
            cut = minimum_vertex_cut(pyg_graph_view(graph), int(carrier), int(source))
            cut_sizes.append(len(cut))
            cut_disconnects.append(cut_disconnects_pair(graph, int(carrier), int(source), cut))
        except Exception:
            continue
    finite = dist[torch.isfinite(dist) & (dist > 0)]
    far_possible = int(((torch.isfinite(dist) & (dist > float(tau))).sum()).item())
    return {
        "graph_id": gid,
        "pair_id": str(pair_id or gid),
        "n_nodes": n_nodes,
        "n_edges_undirected": len(undirected_edges),
        "mean_degree": float((2 * len(undirected_edges)) / max(n_nodes, 1)),
        "graph_diameter": summary["graph_diameter"],
        "graph_mean_distance": summary["graph_mean_distance"],
        "distance_std": float(finite.std(unbiased=False).item()) if finite.numel() else float("nan"),
        "articulation_fraction": float(articulation_count / max(n_nodes, 1)),
        "far_pair_count": far_possible,
        "sampled_cut_pairs": len(cut_sizes),
        "mean_far_cut_size": float(np.mean(cut_sizes)) if cut_sizes else float("nan"),
        "single_cut_fraction": float(np.mean([size == 1 for size in cut_sizes])) if cut_sizes else float("nan"),
        "verified_cut_fraction": float(np.mean(cut_disconnects)) if cut_disconnects else float("nan"),
    }


def summarize_rrwp_ablation_for_contrast(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("status")) != "complete":
            continue
        pair_id = str(row.get("pair_id") or row.get("graph_id"))
        key = (str(row.get("model")), pair_id)
        entry = out.setdefault(
            key,
            {
                "model": str(row.get("model")),
                "graph_id": str(row.get("graph_id")),
                "pair_id": pair_id,
                "clean_prediction": safe_float(row.get("clean_prediction")),
                "target": safe_float(row.get("target")),
                "clean_mae": safe_float(row.get("clean_mae")),
            },
        )
        ablation_type = str(row.get("ablation_type"))
        abs_delta = safe_float(row.get("abs_delta_pred"))
        delta_mae = safe_float(row.get("delta_mae"))
        prefix = f"{ablation_type}_rrwp"
        entry[f"{prefix}_effect_sum"] = safe_float(entry.get(f"{prefix}_effect_sum", 0.0)) + (abs_delta if math.isfinite(abs_delta) else 0.0)
        entry[f"{prefix}_delta_mae_sum"] = safe_float(entry.get(f"{prefix}_delta_mae_sum", 0.0)) + (delta_mae if math.isfinite(delta_mae) else 0.0)
        entry[f"{prefix}_rows"] = int(entry.get(f"{prefix}_rows", 0)) + 1
    for entry in out.values():
        for ablation_type in ("node", "pair", "both"):
            prefix = f"{ablation_type}_rrwp"
            rows_n = max(1, int(entry.get(f"{prefix}_rows", 0)))
            entry[f"{prefix}_effect_mean"] = safe_float(entry.get(f"{prefix}_effect_sum", 0.0)) / rows_n
            entry[f"{prefix}_delta_mae_mean"] = safe_float(entry.get(f"{prefix}_delta_mae_sum", 0.0)) / rows_n
    return out


def global_vs_local_rrwp_contrast_rows(
    ablation_rows: Sequence[Mapping[str, Any]],
    graph_metric_rows: Sequence[Mapping[str, Any]],
    *,
    global_model: str,
    local_model: str,
) -> list[dict[str, Any]]:
    summary = summarize_rrwp_ablation_for_contrast(ablation_rows)
    metrics = {str(row.get("pair_id") or row.get("graph_id")): dict(row) for row in graph_metric_rows}
    global_ids = {pair_id for model, pair_id in summary if model == global_model}
    local_ids = {pair_id for model, pair_id in summary if model == local_model}
    paired_ids = sorted(global_ids & local_ids)
    rows: list[dict[str, Any]] = []
    for pair_id in paired_ids:
        glob = summary[(global_model, pair_id)]
        loc = summary[(local_model, pair_id)]
        global_mae = safe_float(glob.get("clean_mae"))
        local_mae = safe_float(loc.get("clean_mae"))
        if not (math.isfinite(global_mae) and math.isfinite(local_mae)):
            continue
        row = {
            "pair_id": pair_id,
            "global_graph_id": glob.get("graph_id", ""),
            "local_graph_id": loc.get("graph_id", ""),
            "graph_id": glob.get("graph_id", pair_id),
            "global_model": global_model,
            "local_model": local_model,
            "global_clean_mae": global_mae,
            "local_clean_mae": local_mae,
            "local_minus_global_mae": local_mae - global_mae,
            "global_prediction": safe_float(glob.get("clean_prediction")),
            "local_prediction": safe_float(loc.get("clean_prediction")),
            "target": safe_float(glob.get("target")),
        }
        for ablation_type in ("node", "pair", "both"):
            for suffix in ("effect_sum", "effect_mean", "delta_mae_sum", "delta_mae_mean", "rows"):
                row[f"global_{ablation_type}_rrwp_{suffix}"] = glob.get(f"{ablation_type}_rrwp_{suffix}", 0)
                row[f"local_{ablation_type}_rrwp_{suffix}"] = loc.get(f"{ablation_type}_rrwp_{suffix}", 0)
                if suffix != "rows":
                    row[f"global_minus_local_{ablation_type}_rrwp_{suffix}"] = (
                        safe_float(glob.get(f"{ablation_type}_rrwp_{suffix}", 0.0))
                        - safe_float(loc.get(f"{ablation_type}_rrwp_{suffix}", 0.0))
                    )
        row.update(metrics.get(pair_id, {}))
        rows.append(row)
    return rows


def render_step7_rrwp_distance_bin_ablation(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    clean = [
        r
        for r in rows
        if str(r.get("status")) == "complete"
        and str(r.get("ablation_type")) == "pair"
        and math.isfinite(safe_float(r.get("abs_delta_pred")))
        and str(r.get("distance_bin"))
    ]
    figures = ensure_dir(artifact_root / "figures")
    if not clean:
        fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
        ax.text(0.5, 0.5, "No pair-RRWP distance-bin ablation rows were available.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(figures / "step7_rrwp_distance_bin_ablation.png", dpi=dpi)
        fig.savefig(figures / "step7_rrwp_distance_bin_ablation.pdf")
        plt.close(fig)
        return
    labels = sorted(
        {str(r.get("distance_bin")) for r in clean},
        key=lambda label: min([int(safe_float(r.get("distance_min"))) for r in clean if str(r.get("distance_bin")) == label] or [999]),
    )
    x = np.arange(len(labels), dtype=float)
    summary_rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    for model in ordered_model_names([str(r.get("model")) for r in clean]):
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for label in labels:
            vals = [
                safe_float(r.get("abs_delta_pred"))
                for r in clean
                if str(r.get("model")) == model and str(r.get("distance_bin")) == label
            ]
            vals = [v for v in vals if math.isfinite(v)]
            if not vals:
                means.append(float("nan"))
                lows.append(float("nan"))
                highs.append(float("nan"))
                continue
            mean, lo, hi = bootstrap_ci(
                vals,
                seed=stable_seed("rrwp_pair_distance_ablation", model, label),
                draws=500,
            )
            means.append(mean)
            lows.append(lo)
            highs.append(hi)
            summary_rows.append(
                {
                    "model": model,
                    "distance_bin": label,
                    "mean_abs_delta_pred": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "graphs": len(vals),
                }
            )
        y = np.asarray(means, dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.8, label=model_label(model))
        lo_arr = np.asarray(lows, dtype=float)
        hi_arr = np.asarray(highs, dtype=float)
        mask = np.isfinite(y) & np.isfinite(lo_arr) & np.isfinite(hi_arr)
        if bool(mask.any()):
            ax.fill_between(x[mask], lo_arr[mask], hi_arr[mask], alpha=0.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("Molecular hop distance bin")
    ax.set_ylabel("Mean |Δŷ| from pair-RRWP long-channel removal")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Pair-RRWP structural sensitivity by distance")
    ax.text(
        0.01,
        -0.24,
        "Node RRWP is graph/node-level, so distance-binned node ablation is intentionally omitted.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#555555",
    )
    write_csv(artifact_root / "metrics" / "step7_pair_rrwp_distance_bin_ablation_summary.csv", summary_rows)
    fig.savefig(figures / "step7_rrwp_distance_bin_ablation.png", dpi=dpi)
    fig.savefig(figures / "step7_rrwp_distance_bin_ablation.pdf")
    plt.close(fig)


def render_step7_global_vs_local_rrwp_paired_contrast(
    rows: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    *,
    dpi: int,
    global_model: str,
    local_model: str,
) -> None:
    figures = ensure_dir(artifact_root / "figures")
    predictors = [
        ("global_both_rrwp_effect_sum", "Global-RRWP ablation effect\n(both, sum |Δŷ|)"),
        ("global_pair_rrwp_effect_sum", "Pair-RRWP ablation effect\n(sum |Δŷ|)"),
        ("global_node_rrwp_effect_sum", "Node-RRWP ablation effect\n(sum |Δŷ|)"),
        ("graph_diameter", "Graph diameter"),
        ("articulation_fraction", "Articulation-point fraction"),
        ("single_cut_fraction", "Single-cut far-pair fraction"),
    ]
    y_key = "local_minus_global_mae"
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), constrained_layout=True)
    clean_rows = [r for r in rows if math.isfinite(safe_float(r.get(y_key)))]
    if not clean_rows:
        axes = np.asarray(axes).reshape(-1)
        axes[0].text(
            0.5,
            0.5,
            f"No paired {global_model} vs {local_model} rows were available.",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        for ax in axes:
            ax.set_axis_off()
        fig.savefig(figures / "step7_global_vs_local_rrwp_paired_contrast.png", dpi=dpi)
        fig.savefig(figures / "step7_global_vs_local_rrwp_paired_contrast.pdf")
        plt.close(fig)
        return
    for ax, (x_key, label) in zip(np.asarray(axes).reshape(-1), predictors):
        xs = np.asarray([safe_float(r.get(x_key)) for r in clean_rows], dtype=float)
        ys = np.asarray([safe_float(r.get(y_key)) for r in clean_rows], dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() == 0:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        ax.scatter(xs[mask], ys[mask], s=28, alpha=0.75)
        r_text = "r=n/a"
        if mask.sum() >= 3 and np.nanstd(xs[mask]) > 0 and np.nanstd(ys[mask]) > 0:
            corr = float(np.corrcoef(xs[mask], ys[mask])[0, 1])
            coef = np.polyfit(xs[mask], ys[mask], deg=1)
            x_line = np.linspace(float(xs[mask].min()), float(xs[mask].max()), 100)
            y_line = coef[0] * x_line + coef[1]
            ax.plot(x_line, y_line, color="#444444", linestyle="--", linewidth=1)
            r_text = f"r={corr:.2f}"
        ax.axhline(0, color="#777777", linewidth=1, linestyle=":")
        ax.set_title(label)
        ax.set_xlabel(label.replace("\n", " "))
        ax.set_ylabel(f"{local_model} MAE - {global_model} MAE")
        ax.text(0.03, 0.95, r_text, transform=ax.transAxes, ha="left", va="top", fontsize=9)
    fig.suptitle("Global vs local RRWP: paired molecule contrast")
    fig.savefig(figures / "step7_global_vs_local_rrwp_paired_contrast.png", dpi=dpi)
    fig.savefig(figures / "step7_global_vs_local_rrwp_paired_contrast.pdf")
    plt.close(fig)


def rrwp_global_channel_ablation_rows_for_graph(
    model: ModelRun,
    graph: Any,
    gid: str,
    *,
    pair_id: Optional[str],
    ablation_types: Sequence[str],
    channel_start: int,
    replacement: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Ablate all long RRWP channels in a graph, split into node/pair/both components.

    This is the cleanest test of the 1-hop global-RRWP vs local-RRWP difference:
    channels before ``channel_start`` are kept, while longer random-walk channels
    are zeroed/mean-replaced. For the local-RRWP control this should be nearly a
    no-op; for the global-RRWP 1-hop model, any nonzero effect is direct evidence
    that the model uses the extra structural signal.
    """

    adapter = model.adapter
    stable_pair_id = str(pair_id or gid)
    if not hasattr(adapter, "forward_minimal"):
        return [], None
    try:
        clean_cache = adapter.forward_minimal(graph)
    except Exception as exc:
        return [], {
            "status": "failed_clean_forward",
            "model": model.name,
            "graph_id": gid,
            "pair_id": stable_pair_id,
            "error": str(exc),
        }
    extras = getattr(clean_cache, "extras", None) or {}
    raw_rrwp = extras.get("raw_rrwp")
    raw_rrwp_val = extras.get("raw_rrwp_val")
    raw_rrwp_index = extras.get("raw_rrwp_index")
    raw_rrwp_key = extras.get("raw_rrwp_key")
    raw_rrwp_val_key = extras.get("raw_rrwp_val_key")
    raw_rrwp_index_key = extras.get("raw_rrwp_index_key")
    if not isinstance(raw_rrwp, torch.Tensor):
        raw_rrwp_key, raw_rrwp = _graph_tensor_field(graph, ("rrwp", "pestat_RRWP", "pestat_rrwp", "RWSE", "rwse"))
    if not isinstance(raw_rrwp_val, torch.Tensor):
        raw_rrwp_val_key, raw_rrwp_val = _graph_tensor_field(
            graph,
            ("rrwp_val", "rrwp_values", "rrwp_value", "pestat_RRWP_val", "pestat_rrwp_val"),
        )
    if not isinstance(raw_rrwp_index, torch.Tensor):
        raw_rrwp_index_key, raw_rrwp_index = _graph_tensor_field(
            graph,
            ("rrwp_index", "rrwp_idx", "pestat_RRWP_index", "pestat_rrwp_index"),
        )

    dense_pair_rrwp = False
    raw_rrwp_dense_pair: Optional[torch.Tensor] = None
    if (
        not isinstance(raw_rrwp_val, torch.Tensor)
        and not isinstance(raw_rrwp_index, torch.Tensor)
        and isinstance(raw_rrwp, torch.Tensor)
        and raw_rrwp.dim() == 3
    ):
        dense_pair_rrwp = True
        raw_rrwp_dense_pair = raw_rrwp.detach()

    node_rrwp = raw_rrwp.detach() if isinstance(raw_rrwp, torch.Tensor) and raw_rrwp.dim() == 2 else None
    pair_rrwp = raw_rrwp_val.detach() if isinstance(raw_rrwp_val, torch.Tensor) and raw_rrwp_val.dim() == 2 else None
    if node_rrwp is None and pair_rrwp is None and raw_rrwp_dense_pair is None:
        return [], {
            "status": "missing_raw_rrwp_fields",
            "model": model.name,
            "graph_id": gid,
            "pair_id": stable_pair_id,
            "raw_rrwp_key": raw_rrwp_key or "",
            "raw_rrwp_val_key": raw_rrwp_val_key or "",
            "raw_rrwp_index_key": raw_rrwp_index_key or "",
            "raw_rrwp_shape": tuple(raw_rrwp.shape) if isinstance(raw_rrwp, torch.Tensor) else "",
            "raw_rrwp_val_shape": tuple(raw_rrwp_val.shape) if isinstance(raw_rrwp_val, torch.Tensor) else "",
        }

    clean_pred = safe_float(clean_cache.prediction.detach().reshape(-1)[0].cpu().item())
    target = graph_label(graph)
    clean_mae = abs(clean_pred - target) if math.isfinite(target) else float("nan")
    rows: list[dict[str, Any]] = []
    for ablation_type in ablation_types:
        node_override = None
        pair_override = None
        removed_abs = 0.0
        removed_count = 0
        if dense_pair_rrwp and isinstance(raw_rrwp_dense_pair, torch.Tensor):
            if ablation_type == "node":
                continue
            flat = raw_rrwp_dense_pair.reshape(-1, int(raw_rrwp_dense_pair.size(-1)))
            all_rows = torch.arange(int(flat.size(0)), dtype=torch.long)
            flat_override = _rrwp_replace_channels(
                flat,
                all_rows,
                channel_start=channel_start,
                replacement=replacement,
            )
            node_override = flat_override.reshape_as(raw_rrwp_dense_pair)
            removed_abs = float((flat[:, int(channel_start):] - flat_override[:, int(channel_start):]).abs().sum().item())
            removed_count = int(flat[:, int(channel_start):].numel())
        else:
            if ablation_type in {"node", "both"} and isinstance(node_rrwp, torch.Tensor):
                all_rows = torch.arange(int(node_rrwp.size(0)), dtype=torch.long)
                node_override = _rrwp_replace_channels(
                    node_rrwp,
                    all_rows,
                    channel_start=channel_start,
                    replacement=replacement,
                )
                removed_abs += float((node_rrwp[:, int(channel_start):] - node_override[:, int(channel_start):]).abs().sum().item())
                removed_count += int(node_rrwp[:, int(channel_start):].numel())
            if ablation_type in {"pair", "both"} and isinstance(pair_rrwp, torch.Tensor):
                all_rows = torch.arange(int(pair_rrwp.size(0)), dtype=torch.long)
                pair_override = _rrwp_replace_channels(
                    pair_rrwp,
                    all_rows,
                    channel_start=channel_start,
                    replacement=replacement,
                )
                removed_abs += float((pair_rrwp[:, int(channel_start):] - pair_override[:, int(channel_start):]).abs().sum().item())
                removed_count += int(pair_rrwp[:, int(channel_start):].numel())
        if node_override is None and pair_override is None:
            continue
        try:
            ablated_cache = adapter.forward_minimal(
                graph,
                rrwp_node_override=node_override,
                rrwp_val_override=pair_override,
            )
            ablated_pred = safe_float(ablated_cache.prediction.detach().reshape(-1)[0].cpu().item())
        except Exception as exc:
            rows.append(
                {
                    "status": "failed_ablation_forward",
                    "model": model.name,
                    "role": model.role,
                    "graph_id": gid,
                    "pair_id": stable_pair_id,
                    "ablation_type": ablation_type,
                    "error": str(exc),
                }
            )
            continue
        ablated_mae = abs(ablated_pred - target) if math.isfinite(target) else float("nan")
        rows.append(
            {
                "status": "complete",
                "model": model.name,
                "role": model.role,
                "graph_id": gid,
                "pair_id": stable_pair_id,
                "ablation_type": ablation_type,
                "channel_start": int(channel_start),
                "replacement": replacement,
                "clean_prediction": clean_pred,
                "ablated_prediction": ablated_pred,
                "delta_pred": ablated_pred - clean_pred,
                "abs_delta_pred": abs(ablated_pred - clean_pred),
                "target": target,
                "clean_mae": clean_mae,
                "ablated_mae": ablated_mae,
                "delta_mae": ablated_mae - clean_mae if math.isfinite(clean_mae) and math.isfinite(ablated_mae) else float("nan"),
                "removed_abs_rrwp": removed_abs,
                "removed_rrwp_values": removed_count,
                "mean_abs_removed_rrwp": removed_abs / max(removed_count, 1),
                "raw_rrwp_key": raw_rrwp_key or "",
                "raw_rrwp_val_key": raw_rrwp_val_key or "",
                "raw_rrwp_index_key": raw_rrwp_index_key or "",
                "dense_pair_rrwp": dense_pair_rrwp,
            }
        )
    return rows, {"status": "complete", "model": model.name, "graph_id": gid, "pair_id": stable_pair_id, "rows": len(rows)}


def _global_rrwp_channel_order(clean: Sequence[Mapping[str, Any]]) -> list[str]:
    order = sorted(
        {str(r.get("model")) for r in clean},
        key=lambda name: (MODEL_PLOT_ORDER.index(name) if name in MODEL_PLOT_ORDER else len(MODEL_PLOT_ORDER), name),
    )
    if "grit_1hop" in order and "grit_1hop_localrrwp" not in order:
        insert_at = order.index("grit_1hop")
        order.insert(insert_at, "grit_1hop_localrrwp")
    return order


def _render_step4_global_rrwp_channel_ablation_main(
    rows: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    *,
    dpi: int,
) -> None:
    figures = ensure_dir(artifact_root / "figures")
    clean = [r for r in rows if str(r.get("status")) == "complete"]
    if not clean:
        return
    factors = [f for f in ("node", "pair", "both") if any(str(r.get("ablation_type")) == f for r in clean)]
    if not factors:
        return
    order = _global_rrwp_channel_order(clean)
    labels = {"node": "node\nRRWP", "pair": "pair\nRRWP", "both": "node + pair\nRRWP"}
    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    x = np.arange(len(factors), dtype=float)
    width = 0.8 / max(len(order), 1)
    all_values: list[float] = []
    for idx, model_name in enumerate(order):
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        missing: list[bool] = []
        for factor in factors:
            vals = [
                safe_float(r.get("abs_delta_pred"))
                for r in clean
                if str(r.get("model")) == model_name and str(r.get("ablation_type")) == factor
            ]
            vals = [v for v in vals if math.isfinite(v)]
            if vals:
                mean, lo, hi = bootstrap_ci(vals, seed=stable_seed("global_rrwp_channel_main", model_name, factor), draws=500)
                missing.append(False)
                all_values.extend([mean, lo, hi])
            else:
                mean, lo, hi = 0.0, 0.0, 0.0
                missing.append(True)
            means.append(mean)
            lows.append(lo)
            highs.append(hi)
        pos = x - 0.4 + width / 2 + idx * width
        y = np.asarray(means, dtype=float)
        yerr = np.vstack([
            np.maximum(0.0, y - np.asarray(lows, dtype=float)),
            np.maximum(0.0, np.asarray(highs, dtype=float) - y),
        ])
        bars = ax.bar(
            pos,
            y,
            width=width,
            yerr=yerr,
            capsize=2.5,
            label=model_label(model_name),
            color=MODEL_PALETTE.get(model_name),
            alpha=0.9,
        )
        for bar, is_missing in zip(bars, missing):
            if is_missing:
                bar.set_facecolor("white")
                bar.set_edgecolor(MODEL_PALETTE.get(model_name, "#888888"))
                bar.set_hatch("//")
                bar.set_linewidth(1.2)
    ax.axhline(0.0, color="#555555", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(f, f) for f in factors])
    ax.set_ylabel("Prediction disruption, mean |Δŷ|")
    ax.set_title("Global RRWP dependence in 1-hop GRIT")
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        0.01,
        -0.20,
        "Long RRWP channels are removed beyond self/1-hop; hatched zero bars mean no matching long channel was available.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#555555",
    )
    ax.set_ylim(bottom=0.0)
    fig.savefig(figures / "step7_global_to_local_rrwp_ablation_main.png", dpi=dpi)
    fig.savefig(figures / "step7_global_to_local_rrwp_ablation_main.pdf")
    plt.close(fig)


def render_step4_global_rrwp_channel_ablation(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    figures = ensure_dir(artifact_root / "figures")
    clean = [r for r in rows if str(r.get("status")) == "complete"]
    if not clean:
        fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
        ax.text(0.5, 0.5, "No long-RRWP channel ablation rows were available.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(figures / "step7_global_to_local_rrwp_ablation.png", dpi=dpi)
        fig.savefig(figures / "step7_global_to_local_rrwp_ablation.pdf")
        plt.close(fig)
        return
    order = _global_rrwp_channel_order(clean)
    factors = [f for f in ("node", "pair", "both") if any(str(r.get("ablation_type")) == f for r in clean)]
    labels = {"node": "node\nRRWP", "pair": "pair\nRRWP", "both": "node+pair\nRRWP"}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    x = np.arange(len(factors), dtype=float)
    width = 0.8 / max(len(order), 1)
    summary_rows: list[dict[str, Any]] = []
    for ax, value_key, ylabel, title in (
        (axes[0], "delta_mae", "Estimated Δ test MAE (positive = worse)", "Does long-RRWP removal damage the prediction?"),
        (axes[1], "abs_delta_pred", "Prediction disruption, mean |Δŷ|", "Sensitivity to long RRWP"),
    ):
        for idx, model_name in enumerate(order):
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for factor in factors:
                vals = [
                    safe_float(r.get(value_key))
                    for r in clean
                    if str(r.get("model")) == model_name and str(r.get("ablation_type")) == factor
                ]
                vals = [v for v in vals if math.isfinite(v)]
                if vals:
                    mean, lo, hi = bootstrap_ci(vals, seed=stable_seed("global_rrwp_channel", value_key, model_name, factor), draws=500)
                else:
                    mean, lo, hi = (0.0, 0.0, 0.0) if model_name == "grit_1hop_localrrwp" else (float("nan"), float("nan"), float("nan"))
                means.append(mean)
                lows.append(lo)
                highs.append(hi)
                if value_key == "delta_mae":
                    summary_rows.append(
                        {
                            "model": model_name,
                            "ablation_type": factor,
                            "mean_delta_mae": mean,
                            "delta_mae_ci_low": lo,
                            "delta_mae_ci_high": hi,
                            "graphs": len(vals),
                        }
                    )
            pos = x - 0.4 + width / 2 + idx * width
            y = np.asarray(means, dtype=float)
            yerr = np.vstack([
                np.maximum(0.0, y - np.asarray(lows, dtype=float)),
                np.maximum(0.0, np.asarray(highs, dtype=float) - y),
            ])
            bars = ax.bar(
                pos,
                y,
                width=width,
                yerr=yerr,
                capsize=2.5,
                label=model_label(model_name),
                color=MODEL_PALETTE.get(model_name),
                alpha=0.9,
            )
            for bar, factor, mean in zip(bars, factors, means):
                has_rows = any(str(r.get("model")) == model_name and str(r.get("ablation_type")) == factor for r in clean)
                if not has_rows and math.isfinite(float(mean)):
                    bar.set_facecolor("white")
                    bar.set_edgecolor(MODEL_PALETTE.get(model_name, "#888888"))
                    bar.set_hatch("//")
                    bar.set_linewidth(1.2)
        ax.axhline(0, color="#555555", linewidth=1, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(f, f) for f in factors])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Global-to-local RRWP ablation: which long structural channels matter?")
    write_csv(artifact_root / "metrics" / "step7_global_to_local_rrwp_ablation_summary.csv", summary_rows)
    fig.savefig(figures / "step7_global_to_local_rrwp_ablation.png", dpi=dpi)
    fig.savefig(figures / "step7_global_to_local_rrwp_ablation.pdf")
    plt.close(fig)
    _render_step4_global_rrwp_channel_ablation_main(rows, artifact_root, dpi=dpi)


def run_rrwp_distance_ablation_probe(models: Sequence[ModelRun], artifact_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Step-7 raw RRWP ablation probe for global-vs-local structural reasoning."""

    cfg = config["steps"].get("7") or config["steps"].get("4") or {}
    sample_graphs = int(cfg.get("rrwp_ablation_sample_graphs", min(24, int(cfg.get("sample_graphs", 24)))))
    global_channel_sample_graphs = int(cfg.get("global_rrwp_channel_ablation_sample_graphs", sample_graphs))
    run_global_channel_ablation = bool(cfg.get("run_global_rrwp_channel_ablation", True))
    channel_start = int(cfg.get("rrwp_ablation_channel_start", 2))
    replacement = str(cfg.get("rrwp_ablation_replacement", "zero"))
    distance_ablation_types = rrwp_distance_ablation_types(cfg.get("rrwp_distance_ablation_types", ["pair"]))
    global_channel_ablation_types = rrwp_distance_ablation_types(
        cfg.get("global_rrwp_channel_ablation_types", cfg.get("rrwp_ablation_types"))
    )
    bins = distance_ablation_bins(cfg.get("rrwp_ablation_distance_bins"))
    seed = int(config.get("seeds", [0])[0])
    tau = int(config.get("primary_tau", 3))
    dpi = int(config["figures"]["dpi"])
    max_cut_pairs = int(cfg.get("rrwp_graph_metric_cut_pairs", 32))
    global_model = str(cfg.get("rrwp_contrast_global_model", "grit_1hop"))
    local_model = str(cfg.get("rrwp_contrast_local_model", "grit_1hop_localrrwp"))
    rows: list[dict[str, Any]] = []
    global_channel_rows: list[dict[str, Any]] = []
    global_channel_skip_rows: list[dict[str, Any]] = []
    graph_metric_rows_by_gid: dict[str, dict[str, Any]] = {}
    skip_rows: list[dict[str, Any]] = []
    progress(
        f"Step 4 RRWP distance-bin ablation: sample_graphs={sample_graphs}, "
        f"types={','.join(distance_ablation_types)}, channel_start={channel_start}, replacement={replacement}"
    )
    model_complete_counts: dict[str, int] = {}
    for model in models:
        try:
            graphs = select_graphs(model.adapter, "test", sample_graphs, seed=seed)
        except Exception as exc:
            skip_rows.append({"model": model.name, "status": "failed_graph_load", "error": str(exc)})
            continue
        progress(f"Step 4 RRWP ablation {model.name}: {len(graphs)} graph(s)")
        for graph_idx, graph in enumerate(graphs):
            gid = graph_identity("test", graph_idx, graph)
            pair_id = f"test_sample:{graph_idx}"
            if pair_id not in graph_metric_rows_by_gid:
                graph_metric_rows_by_gid[pair_id] = graph_rrwp_contrast_metric_row(
                    graph,
                    gid,
                    pair_id=pair_id,
                    tau=tau,
                    seed=seed + graph_idx,
                    max_cut_pairs=max_cut_pairs,
                )
            graph_rows, status = rrwp_distance_ablation_rows_for_graph(
                model,
                graph,
                gid,
                pair_id=pair_id,
                bins=bins,
                ablation_types=distance_ablation_types,
                channel_start=channel_start,
                replacement=replacement,
            )
            rows.extend(graph_rows)
            model_complete_counts[model.name] = model_complete_counts.get(model.name, 0) + len(
                [row for row in graph_rows if str(row.get("status")) == "complete"]
            )
            if status is not None and status.get("status") != "complete":
                skip_rows.append(status)
            progress_graph("Step 4 RRWP ablation", model.name, graph_idx, len(graphs))

    if run_global_channel_ablation:
        progress(
            f"Step 4 global-to-local RRWP channel ablation: sample_graphs={global_channel_sample_graphs}, "
            f"types={','.join(global_channel_ablation_types)}, channel_start={channel_start}, replacement={replacement}"
        )
        for model in models:
            try:
                graphs = select_graphs(model.adapter, "test", global_channel_sample_graphs, seed=seed)
            except Exception as exc:
                global_channel_skip_rows.append({"model": model.name, "status": "failed_graph_load", "error": str(exc)})
                continue
            progress(f"Step 4 global-to-local RRWP {model.name}: {len(graphs)} graph(s)")
            for graph_idx, graph in enumerate(graphs):
                gid = graph_identity("test", graph_idx, graph)
                pair_id = f"test_sample:{graph_idx}"
                graph_rows, status = rrwp_global_channel_ablation_rows_for_graph(
                    model,
                    graph,
                    gid,
                    pair_id=pair_id,
                    ablation_types=global_channel_ablation_types,
                    channel_start=channel_start,
                    replacement=replacement,
                )
                global_channel_rows.extend(graph_rows)
                if status is not None and status.get("status") != "complete":
                    global_channel_skip_rows.append(status)
                progress_graph("Step 4 global-to-local RRWP", model.name, graph_idx, len(graphs))

    graph_metric_rows = list(graph_metric_rows_by_gid.values())
    contrast_rows = global_vs_local_rrwp_contrast_rows(
        rows,
        graph_metric_rows,
        global_model=global_model,
        local_model=local_model,
    )
    write_csv(artifact_root / "metrics" / "step7_rrwp_distance_bin_ablation.csv", rows)
    write_csv(artifact_root / "metrics" / "step7_rrwp_distance_bin_ablation_skips.csv", skip_rows)
    write_csv(artifact_root / "metrics" / "step7_rrwp_graph_metrics.csv", graph_metric_rows)
    write_csv(artifact_root / "metrics" / "step7_global_vs_local_rrwp_paired_contrast.csv", contrast_rows)
    write_csv(artifact_root / "metrics" / "step7_global_to_local_rrwp_ablation.csv", global_channel_rows)
    write_csv(artifact_root / "metrics" / "step7_global_to_local_rrwp_ablation_skips.csv", global_channel_skip_rows)
    render_step7_rrwp_distance_bin_ablation(rows, artifact_root, dpi=dpi)
    render_step7_global_vs_local_rrwp_paired_contrast(
        contrast_rows,
        artifact_root,
        dpi=dpi,
        global_model=global_model,
        local_model=local_model,
    )
    if run_global_channel_ablation:
        render_step4_global_rrwp_channel_ablation(global_channel_rows, artifact_root, dpi=dpi)
    complete_rows = [r for r in rows if str(r.get("status")) == "complete"]
    complete_global_channel_rows = [r for r in global_channel_rows if str(r.get("status")) == "complete"]
    progress(
        f"Step 4 RRWP distance-bin ablation: wrote {len(complete_rows)} complete rows, "
        f"{len(contrast_rows)} paired contrast rows, {len(skip_rows)} skips; "
        f"complete rows by model={model_complete_counts}"
    )
    if run_global_channel_ablation:
        progress(
            f"Step 4 global-to-local RRWP channel ablation: wrote "
            f"{len(complete_global_channel_rows)} complete rows, {len(global_channel_skip_rows)} skips"
        )
    return {
        "status": "complete" if complete_rows else "no_rows",
        "rows": len(rows),
        "complete_rows": len(complete_rows),
        "skip_rows": len(skip_rows),
        "global_channel_rows": len(global_channel_rows),
        "global_channel_complete_rows": len(complete_global_channel_rows),
        "global_channel_skip_rows": len(global_channel_skip_rows),
        "global_channel_enabled": run_global_channel_ablation,
        "graph_metric_rows": len(graph_metric_rows),
        "paired_contrast_rows": len(contrast_rows),
        "global_model": global_model,
        "local_model": local_model,
    }


def render_step4_mean_abs_carriage_by_distance(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    clean = [
        r
        for r in rows
        if math.isfinite(safe_float(r.get("distance")))
        and math.isfinite(safe_float(r.get("effect_abs")))
        and safe_float(r.get("effect_abs")) >= 0.0
    ]
    if not clean:
        return
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in clean:
        distance = int(round(safe_float(row.get("distance"))))
        grouped.setdefault((str(row.get("model")), distance), []).append(safe_float(row.get("effect_abs")))
    summary_rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for model in sorted({model for model, _ in grouped}):
        distances = sorted(distance for m, distance in grouped if m == model)
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        counts: list[int] = []
        for distance in distances:
            values = np.asarray(grouped[(model, distance)], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                means.append(float("nan"))
                lows.append(float("nan"))
                highs.append(float("nan"))
                counts.append(0)
                continue
            mean, low, high = bootstrap_ci(values, seed=7300 + stable_int_hash(f"{model}:{distance}"), draws=500)
            means.append(mean)
            lows.append(low)
            highs.append(high)
            counts.append(int(values.size))
            summary_rows.append(
                {
                    "model": model,
                    "distance": distance,
                    "mean_abs_carriage": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n_pairs": int(values.size),
                }
            )
        ax.plot(distances, means, marker="o", linewidth=1.8, label=model)
        mean_arr = np.asarray(means, dtype=float)
        low_arr = np.asarray(lows, dtype=float)
        high_arr = np.asarray(highs, dtype=float)
        finite = np.isfinite(mean_arr) & np.isfinite(low_arr) & np.isfinite(high_arr)
        if finite.any():
            ax.fill_between(
                np.asarray(distances, dtype=float)[finite],
                low_arr[finite],
                high_arr[finite],
                alpha=0.12,
            )
    ax.set_title("Mean carriage magnitude vs molecular distance")
    ax.set_xlabel("Molecular hop distance")
    ax.set_ylabel("Mean |C[i,j]| (prediction units)")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=8)
    figures = ensure_dir(artifact_root / "figures")
    write_csv(artifact_root / "metrics" / "step4_mean_abs_carriage_by_distance.csv", summary_rows)
    fig.savefig(figures / "step4_mean_abs_carriage_by_distance.png", dpi=dpi)
    fig.savefig(figures / "step4_mean_abs_carriage_by_distance.pdf")
    plt.close(fig)


def _pearson_corr_values(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 3:
        return float("nan")
    if float(np.std(x_arr)) <= EPS or float(np.std(y_arr)) <= EPS:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def render_step4_dense_onehop_carriage_agreement(
    rows: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    *,
    dpi: int,
    tau: int,
) -> None:
    records: dict[str, dict[tuple[str, int, int], Mapping[str, Any]]] = {"dense_grit": {}, "grit_1hop": {}}
    for row in rows:
        model = str(row.get("model"))
        if model not in records:
            continue
        distance = safe_float(row.get("distance"))
        if not math.isfinite(distance) or distance <= float(tau):
            continue
        carriage = safe_float(row.get("carriage", row.get("unclamped")))
        if not math.isfinite(carriage):
            continue
        key = (str(row.get("graph_id")), int(row.get("carrier")), int(row.get("source")))
        records[model][key] = row

    matched_keys = sorted(set(records["dense_grit"]).intersection(records["grit_1hop"]))
    if not matched_keys:
        return

    pair_rows: list[dict[str, Any]] = []
    for key in matched_keys:
        dense_row = records["dense_grit"][key]
        hop_row = records["grit_1hop"][key]
        dense_c = safe_float(dense_row.get("carriage", dense_row.get("unclamped")))
        hop_c = safe_float(hop_row.get("carriage", hop_row.get("unclamped")))
        distance = int(round(safe_float(dense_row.get("distance"))))
        if not (math.isfinite(dense_c) and math.isfinite(hop_c) and math.isfinite(float(distance))):
            continue
        pair_rows.append(
            {
                "graph_id": key[0],
                "carrier": key[1],
                "source": key[2],
                "distance": distance,
                "dense_carriage": dense_c,
                "onehop_carriage": hop_c,
                "abs_difference": abs(dense_c - hop_c),
            }
        )
    if not pair_rows:
        return

    distances = np.asarray([int(r["distance"]) for r in pair_rows], dtype=int)
    dense = np.asarray([safe_float(r["dense_carriage"]) for r in pair_rows], dtype=float)
    onehop = np.asarray([safe_float(r["onehop_carriage"]) for r in pair_rows], dtype=float)
    finite = np.isfinite(dense) & np.isfinite(onehop) & np.isfinite(distances.astype(float))
    dense = dense[finite]
    onehop = onehop[finite]
    distances = distances[finite]
    pair_rows = [r for r, keep in zip(pair_rows, finite.tolist()) if keep]
    if dense.size < 3:
        return

    summary_rows: list[dict[str, Any]] = []
    for distance in sorted(set(int(d) for d in distances.tolist())):
        mask = distances == distance
        d_dense = dense[mask]
        d_onehop = onehop[mask]
        summary_rows.append(
            {
                "distance": distance,
                "n_pairs": int(mask.sum()),
                "pearson": _pearson_corr_values(d_dense, d_onehop),
                "spearman": spearman_corr(d_dense.tolist(), d_onehop.tolist()) if int(mask.sum()) >= 3 else float("nan"),
                "mean_abs_dense": float(np.mean(np.abs(d_dense))) if d_dense.size else float("nan"),
                "mean_abs_onehop": float(np.mean(np.abs(d_onehop))) if d_onehop.size else float("nan"),
                "mean_abs_difference": float(np.mean(np.abs(d_dense - d_onehop))) if d_dense.size else float("nan"),
            }
        )
    overall = {
        "distance": "all",
        "n_pairs": int(dense.size),
        "pearson": _pearson_corr_values(dense, onehop),
        "spearman": spearman_corr(dense.tolist(), onehop.tolist()),
        "mean_abs_dense": float(np.mean(np.abs(dense))),
        "mean_abs_onehop": float(np.mean(np.abs(onehop))),
        "mean_abs_difference": float(np.mean(np.abs(dense - onehop))),
    }

    metrics = ensure_dir(artifact_root / "metrics")
    figures = ensure_dir(artifact_root / "figures")
    write_csv(metrics / "step4_dense_vs_1hop_carriage_agreement_pairs.csv", pair_rows)
    write_csv(metrics / "step4_dense_vs_1hop_carriage_agreement_by_distance.csv", [overall, *summary_rows])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    ax = axes[0]
    scatter = ax.scatter(
        dense,
        onehop,
        c=distances.astype(float),
        cmap="viridis",
        s=18,
        alpha=0.68,
        edgecolors="none",
    )
    lo = float(np.nanmin([np.nanmin(dense), np.nanmin(onehop)]))
    hi = float(np.nanmax([np.nanmax(dense), np.nanmax(onehop)]))
    if math.isfinite(lo) and math.isfinite(hi):
        pad = 0.05 * max(abs(lo), abs(hi), EPS)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="#555555", linewidth=1.2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    ax.axhline(0.0, color="#888888", linewidth=0.8)
    ax.axvline(0.0, color="#888888", linewidth=0.8)
    ax.set_title("Per-pair carriage: dense vs 1-hop")
    ax.set_xlabel("Dense GRIT C[i,j] (prediction units)")
    ax.set_ylabel("1-hop GRIT C[i,j] (prediction units)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Molecular hop distance")

    ax = axes[1]
    distance_values = [int(r["distance"]) for r in summary_rows if isinstance(r.get("distance"), int)]
    pearson_values = [safe_float(r.get("pearson")) for r in summary_rows if isinstance(r.get("distance"), int)]
    spearman_values = [safe_float(r.get("spearman")) for r in summary_rows if isinstance(r.get("distance"), int)]
    ax.plot(distance_values, pearson_values, marker="o", linewidth=1.8, label="Pearson")
    ax.plot(distance_values, spearman_values, marker="s", linewidth=1.8, label="Spearman")
    ax.axhline(0.0, color="#777777", linewidth=0.9)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Agreement by distance")
    ax.set_xlabel("Molecular hop distance")
    ax.set_ylabel("Correlation of signed C[i,j]")
    ax.legend(frameon=False)
    fig.suptitle(
        "Dense vs 1-hop carriage agreement: same computation or alternative solution?",
        fontsize=15,
    )
    fig.savefig(figures / "step4_dense_vs_1hop_carriage_agreement.png", dpi=dpi)
    fig.savefig(figures / "step4_dense_vs_1hop_carriage_agreement.pdf")
    plt.close(fig)


def render_step4_direct_fraction_by_distance(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get("clamp_type", "cut")) != "cut" or not row_is_nontrivial(row):
            continue
        distance = safe_float(row.get("distance"))
        if not math.isfinite(distance):
            continue
        grouped.setdefault((str(row.get("model")), int(round(distance))), []).append(row)
    summaries: list[dict[str, Any]] = []
    for idx, ((model, distance), group_rows) in enumerate(sorted(grouped.items())):
        stats = weighted_direct_fraction(group_rows, seed=6100 + idx, draws=500)
        mean = safe_float(stats.get("mean"))
        if not math.isfinite(mean):
            continue
        summaries.append(
            {
                "model": model,
                "distance": distance,
                "mean": mean,
                "ci_low": safe_float(stats.get("ci_low")),
                "ci_high": safe_float(stats.get("ci_high")),
                "pairs": int(stats.get("pairs", 0)),
            }
        )
    if not summaries:
        return
    models = sorted({str(r["model"]) for r in summaries})
    offsets = np.linspace(-0.12, 0.12, num=max(1, len(models))) if len(models) > 1 else np.asarray([0.0])
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    for model_idx, model in enumerate(models):
        model_rows = [r for r in summaries if str(r["model"]) == model]
        xs = np.asarray([safe_float(r["distance"]) + float(offsets[model_idx]) for r in model_rows], dtype=float)
        means = np.asarray([safe_float(r["mean"]) for r in model_rows], dtype=float)
        lows = np.asarray([safe_float(r["ci_low"]) for r in model_rows], dtype=float)
        highs = np.asarray([safe_float(r["ci_high"]) for r in model_rows], dtype=float)
        yerr = np.vstack([np.maximum(0.0, means - lows), np.maximum(0.0, highs - means)])
        ax.errorbar(xs, means, yerr=yerr, fmt="o", capsize=3, label=model)
        for x_val, y_val, row in zip(xs, means, model_rows):
            if int(row.get("pairs", 0)) < 3:
                ax.text(x_val, y_val, f"n={int(row.get('pairs', 0))}", fontsize=7, ha="center", va="bottom")
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1, label="No clamp effect")
    ax.set_title("Composed vs direct carriage by molecular distance (d >= 2)")
    ax.set_xlabel("Molecular hop distance")
    ax.set_ylabel("Direct fraction = |C^clamp| / |C^unclamp|")
    ax.legend(frameon=False, fontsize=8)
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step4_direct_fraction_by_distance.png", dpi=dpi)
    fig.savefig(figures / "step4_direct_fraction_by_distance.pdf")
    plt.close(fig)


def render_step4(
    rows: Sequence[Mapping[str, Any]],
    depth_rows: Sequence[Mapping[str, Any]],
    validation_summary: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    *,
    dpi: int,
    tau: int = 3,
    onset_rows: Sequence[Mapping[str, Any]] = (),
    signal_gate_rows: Sequence[Mapping[str, Any]] = (),
    verified_separating_summary: Sequence[Mapping[str, Any]] = (),
    clamp_mode_summary: Sequence[Mapping[str, Any]] = (),
    depth_magnitude_summary: Sequence[Mapping[str, Any]] = (),
    interference_summary: Sequence[Mapping[str, Any]] = (),
) -> None:
    if validation_summary:
        labels = [str(r["model"]) for r in validation_summary]
        y = np.asarray([safe_float(r["mean_direct_fraction"]) for r in validation_summary], dtype=float)
        lo = np.asarray([safe_float(r["ci_low"]) for r in validation_summary], dtype=float)
        hi = np.asarray([safe_float(r["ci_high"]) for r in validation_summary], dtype=float)
        fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
        ax.bar(labels, y, yerr=np.vstack([np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)]), capsize=4, color="#4c78a8")
        ax.axhline(0, color="#555555", linewidth=1)
        ax.set_title("Non-composable long-range carriage is dense-only (dense > 1-hop ≈ local-RRWP ≈ GIN ≈ 0)")
        ax.set_xlabel("Model")
        ax.set_ylabel("Carriage-weighted direct fraction")
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_ha("right")
        figures = ensure_dir(artifact_root / "figures")
        fig.savefig(figures / "step4_mediator_patching_validation.png", dpi=dpi)
        fig.savefig(figures / "step4_mediator_patching_validation.pdf")
        plt.close(fig)
    if signal_gate_rows:
        render_step4_signal_magnitude_by_distance(signal_gate_rows, artifact_root, dpi=dpi)
        render_step4_mean_abs_carriage_by_distance(signal_gate_rows, artifact_root, dpi=dpi)
        render_step4_dense_onehop_carriage_agreement(signal_gate_rows, artifact_root, dpi=dpi, tau=tau)
    if rows:
        render_step4_clamp_validation_d2(rows, artifact_root, dpi=dpi)
        render_step4_direct_fraction_by_distance(rows, artifact_root, dpi=dpi)
        render_step4_clamp_negative_control(rows, artifact_root, dpi=dpi)
        render_step4_verified_separating_cuts(verified_separating_summary, artifact_root, dpi=dpi)
        render_step4_clamp_mode_comparison(clamp_mode_summary, artifact_root, dpi=dpi)
        render_step4_pathway_interference(rows, interference_summary, artifact_root, dpi=dpi)
    if depth_rows:
        render_step4_depth_magnitude(depth_magnitude_summary, artifact_root, dpi=dpi)
        by_layer: dict[tuple[str, int], list[float]] = {}
        for row in depth_rows:
            value = safe_float(row["direct_fraction"])
            if row_is_nontrivial(row) and math.isfinite(value):
                by_layer.setdefault((str(row["model"]), int(row["clamp_until_layer"])), []).append(value)
        if not by_layer:
            return
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        for model in sorted(set(k[0] for k in by_layer)):
            xs = sorted(k[1] for k in by_layer if k[0] == model)
            ax.plot(xs, [float(np.nanmean(by_layer[(model, x)])) for x in xs], marker="o", label=model)
        ax.set_title("Step 4: depth-resolved direct carriage")
        ax.set_xlabel("Clamp through layer")
        ax.set_ylabel("Direct fraction = |C^clamp| / |C^unclamp|")
        ax.legend(frameon=False)
        figures = ensure_dir(artifact_root / "figures")
        fig.savefig(figures / "step4_depth_resolved_direct_carriage.png", dpi=dpi)
        fig.savefig(figures / "step4_depth_resolved_direct_carriage.pdf")
        plt.close(fig)
        render_step4_depth_by_distance(depth_rows, artifact_root, dpi=dpi)
        render_step4_onset_depth_by_distance(onset_rows, artifact_root, dpi=dpi)


def onset_depth_rows(rows: Sequence[Mapping[str, Any]], *, threshold: float = 0.50) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[float]] = {}
    for row in rows:
        if not row_is_nontrivial(row):
            continue
        distance = safe_float(row.get("distance"))
        layer = safe_float(row.get("clamp_until_layer"))
        value = safe_float(row.get("direct_fraction"))
        if not (math.isfinite(distance) and math.isfinite(layer) and math.isfinite(value)):
            continue
        grouped.setdefault((str(row.get("model")), int(round(distance)), int(layer)), []).append(value)
    out: list[dict[str, Any]] = []
    for model, distance in sorted({(model, distance) for model, distance, _ in grouped}):
        layer_values = {
            layer: float(np.nanmean(grouped[(model, distance, layer)]))
            for layer in sorted(layer for m, d, layer in grouped if m == model and d == distance)
        }
        onset = next((layer for layer, value in layer_values.items() if math.isfinite(value) and value >= float(threshold)), None)
        out.append(
            {
                "model": model,
                "distance": distance,
                "onset_layer": onset if onset is not None else float("nan"),
                "threshold": float(threshold),
                "status": "complete" if onset is not None else "no_onset_above_threshold",
                "layers_evaluated": len(layer_values),
                "max_direct_fraction": max([v for v in layer_values.values() if math.isfinite(v)] or [float("nan")]),
            }
        )
    return out


def depth_distance_band(distance: float) -> str:
    if not math.isfinite(distance):
        return "unknown"
    d = int(round(float(distance)))
    if d <= 3:
        return "d=2-3"
    if d <= 6:
        return "d=4-6"
    return "d>=7"


def depth_band_sort_key(band: str) -> int:
    if band == "d=2-3":
        return 2
    if band == "d=4-6":
        return 4
    if band == "d>=7":
        return 7
    match = re.search(r"\d+", str(band))
    return int(match.group(0)) if match else 99


def render_step4_onset_depth_by_distance(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    clean = [
        r
        for r in rows
        if math.isfinite(safe_float(r.get("distance")))
        and math.isfinite(safe_float(r.get("onset_layer")))
    ]
    if not clean:
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    all_distances: list[float] = []
    all_onsets: list[float] = []
    for model in sorted({str(r.get("model")) for r in clean}):
        model_rows = sorted([r for r in clean if str(r.get("model")) == model], key=lambda r: safe_float(r.get("distance")))
        x = np.asarray([safe_float(r.get("distance")) for r in model_rows], dtype=float)
        y = np.asarray([safe_float(r.get("onset_layer")) for r in model_rows], dtype=float)
        all_distances.extend([float(v) for v in x if math.isfinite(float(v))])
        all_onsets.extend([float(v) for v in y if math.isfinite(float(v))])
        ax.plot(x, y, marker="o", linewidth=1.6, label=model)
    if all_distances:
        lim_min = max(0.0, min(all_distances) - 0.5)
        lim_max = max(max(all_distances + all_onsets), 1.0) + 0.5
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="#555555", linewidth=1.0, label="y = distance")
        ax.set_xlim(lim_min, lim_max)
    ax.set_title("Onset depth vs distance: composition staircase vs direct routing")
    ax.set_xlabel("Molecular hop distance")
    ax.set_ylabel("First clamp depth with surviving direct fraction >= threshold")
    ax.legend(frameon=False, fontsize=8)
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step4_onset_depth_vs_distance.png", dpi=dpi)
    fig.savefig(figures / "step4_onset_depth_vs_distance.pdf")
    plt.close(fig)


def render_step4_depth_by_distance(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    clean_rows = [
        r
        for r in rows
        if row_is_nontrivial(r)
        and math.isfinite(safe_float(r.get("direct_fraction")))
        and math.isfinite(safe_float(r.get("clamp_until_layer")))
        and math.isfinite(safe_float(r.get("distance")))
    ]
    if not clean_rows:
        return
    models = sorted({str(r.get("model")) for r in clean_rows})
    if not models:
        return
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(max(5.6, 4.4 * len(models)), 4.8),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, model in zip(axes[0], models):
        model_rows = [r for r in clean_rows if str(r.get("model")) == model]
        grouped: dict[tuple[str, int], list[float]] = {}
        for row in model_rows:
            band = depth_distance_band(safe_float(row.get("distance")))
            layer = int(safe_float(row.get("clamp_until_layer")))
            grouped.setdefault((band, layer), []).append(safe_float(row.get("direct_fraction")))
        bands = sorted({band for band, _ in grouped}, key=depth_band_sort_key)
        for band in bands:
            layers = sorted(layer for b, layer in grouped if b == band)
            y = [float(np.nanmean(grouped[(band, layer)])) for layer in layers]
            ax.plot(layers, y, marker="o", linewidth=1.4, label=band)
        ax.set_title(model)
        ax.set_xlabel("Clamp through layer")
        ax.set_ylabel("Direct fraction = |C^clamp| / |C^unclamp|")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("How reach is built: surviving direct carriage vs clamp depth, by hop distance")
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step4_depth_resolved_direct_carriage_by_distance.png", dpi=dpi)
    fig.savefig(figures / "step4_depth_resolved_direct_carriage_by_distance.pdf")
    plt.close(fig)


def run_step5(models: Sequence[ModelRun], artifact_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    progress("Step 5 start: non-composable gap attribution")
    dense = next((m for m in models if m.name == "dense_grit"), None)
    onehop = next((m for m in models if m.name == "grit_1hop"), None)
    if dense is None or onehop is None:
        return {"status": "skipped_requires_dense_and_onehop"}
    cfg = config["steps"]["5"]
    sample_graphs = int(cfg.get("sample_graphs", 200))
    max_pairs = optional_pair_limit(cfg.get("max_far_pairs_per_graph", config["steps"]["4"].get("max_far_pairs_per_graph", 64)), 64)
    interaction_pairs = int(cfg.get("interaction_pairs", 1000))
    min_effect_abs = float(cfg.get("min_effect_abs", 1.0e-6))
    use_signal_gate = signal_gate_enabled(config, "5")
    gate_quantile = signal_gate_quantile(config, "5")
    gate_floor_cap_fraction = signal_gate_reference_floor_cap_fraction(config, "5")
    clamp_mode = str(cfg.get("clamp_mode", config["steps"]["4"].get("clamp_mode", "detach"))).strip().lower()
    partner_policy = swap_partner_policy(config)
    tau = int(config.get("primary_tau", 3))
    thresholds = far_thresholds(config)
    selection_tau = min(thresholds)
    ig_steps = int(config["perturbation"].get("ig_steps", 32))
    batched_vjp = carriage_ig_uses_batched_vjp(config)
    seed = int(config.get("seeds", [0])[0])
    dpi = int(config["figures"]["dpi"])
    requested_models = list(cfg.get("structure_models", cfg.get("reference_models", ["dense_grit", "grit_1hop", "gin"])))
    if "dense_grit" not in requested_models:
        requested_models.insert(0, "dense_grit")
    by_name = {model.name: model for model in models}
    analysis_models: list[ModelRun] = []
    seen_names: set[str] = set()
    for name in requested_models:
        model = by_name.get(str(name))
        if model is not None and model.name not in seen_names:
            analysis_models.append(model)
            seen_names.add(model.name)
    if dense.name not in seen_names:
        analysis_models.insert(0, dense)
    missing_requested = [str(name) for name in requested_models if str(name) not in by_name]
    if missing_requested:
        progress(f"Step 5: optional reference model(s) not loaded and skipped: {', '.join(missing_requested)}")
    gap_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    rung_rows: list[dict[str, Any]] = []
    signal_gate_rows: list[dict[str, Any]] = []
    total_carriage_ablation_rows: list[dict[str, Any]] = []
    component_carriage_ablation_rows: list[dict[str, Any]] = []
    distance_binned_ablation_rows: list[dict[str, Any]] = []
    dense_control_excess_ablation_rows: list[dict[str, Any]] = []
    graph_carriage_records: dict[tuple[str, str], dict[str, Any]] = {}
    worked_example: Optional[dict[str, Any]] = None
    worked_example_score = 0.0
    rng = random.Random(seed)
    load_bearing_fractions = ablation_fractions(cfg.get("load_bearing_ablation_fractions"))
    load_bearing_random_draws = int(cfg.get("load_bearing_random_draws", 8))
    distance_bins = distance_ablation_bins(cfg.get("distance_binned_ablation_bins"))
    dense_control_model = str(cfg.get("dense_excess_control_model", "grit_1hop"))
    dense_excess_bins = far_only_distance_bins(
        distance_ablation_bins(cfg.get("dense_excess_distance_bins", cfg.get("distance_binned_ablation_bins"))),
        tau=tau,
    )
    onehop_floor = (
        empirical_onehop_noise_floor(
            analysis_models,
            artifact_root,
            config,
            sample_graphs=sample_graphs,
            split="test",
            ig_steps=ig_steps,
            seed=seed,
            batched_vjp=batched_vjp,
            min_floor=min_effect_abs,
            quantile=gate_quantile,
        )
        if use_signal_gate
        else float(min_effect_abs)
    )
    non_additivity_effect_floor = float(cfg.get("non_additivity_min_effect_abs", min_effect_abs))
    progress(
        f"Step 5 signal gate: enabled={use_signal_gate}, quantile={gate_quantile:.2f}, "
        f"onehop_empirical_floor={onehop_floor:.3g}, "
        f"reference_floor_cap_fraction={gate_floor_cap_fraction:.3g}, "
        f"non_additivity_effect_floor={non_additivity_effect_floor:.3g}"
    )
    progress(
        "Step 5 load-bearing ablation: IG-completeness removal fractions="
        f"{', '.join(f'{f:g}' for f in load_bearing_fractions)}, "
        f"random_draws={load_bearing_random_draws}"
    )
    progress(
        "Step 5 distance-binned ablation bins="
        f"{', '.join(str(b.get('label')) for b in distance_bins)}"
    )
    progress(
        "Step 5 dense-minus-control distance ablation: "
        f"control={dense_control_model}, bins={', '.join(str(b.get('label')) for b in dense_excess_bins)}"
    )
    for model in analysis_models:
        graphs = select_graphs(model.adapter, "test", sample_graphs, seed=seed)
        max_pairs_label = "all" if max_pairs is None else str(max_pairs)
        progress(
            f"Step 5 {model.name}: selected {len(graphs)} test graph(s), "
            f"max_far_pairs_per_graph={max_pairs_label}, IG steps={ig_steps}"
        )
        baseline = mean_encoded_baseline(model.adapter, select_baseline_graphs(model.adapter, "test", config, sample_graphs, seed=seed))
        cache_hits = 0
        for graph_idx, graph in enumerate(graphs):
            progress_graph("Step 5", model.name, graph_idx, len(graphs))
            gid = graph_identity("test", graph_idx, graph)
            dist = distance_matrix(graph)
            result = carriage_ig_cached(
                model,
                graph,
                baseline,
                artifact_root,
                config,
                split="test",
                graph_id=gid,
                steps=ig_steps,
                batched_vjp=batched_vjp,
                capture_layer_inputs=True,
                capture_attention=("grit" in model.name.lower()),
            )
            cache_hits += int(bool(result.get("carriage_cache_hit")))
            c = result["carriage"]
            clean_cache = result["clean_cache"]
            readout_grad = result["readout_gradient"]
            clean_encoded = result["clean_encoded"].to(model.adapter.device)
            base_encoded = result["baseline"].to(model.adapter.device)
            direct = torch.full_like(c, float("nan"))
            selected = far_pairs(dist, selection_tau, max_pairs=max_pairs, seed=seed + graph_idx)
            for pair_idx, (carrier, source) in enumerate(selected):
                pair_interval = progress_interval(len(selected), target_messages=4)
                if pair_idx == 0 or pair_idx + 1 == len(selected) or (pair_idx + 1) % pair_interval == 0:
                    progress(f"Step 5 {model.name} graph {graph_idx + 1}/{len(graphs)}: patched pair {pair_idx + 1}/{len(selected)}")
                gate_pass, signal_floor, effect_abs = pair_passes_signal_gate(
                    c,
                    dist,
                    carrier,
                    source,
                    enabled=use_signal_gate,
                    quantile=gate_quantile,
                    min_floor=min_effect_abs,
                    reference_floor=onehop_floor,
                    reference_floor_cap_fraction=gate_floor_cap_fraction,
                )
                signal_gate_rows.append(
                    {
                        "model": model.name,
                        "graph_id": gid,
                        "carrier": carrier,
                        "source": source,
                        "distance": float(dist[carrier, source].item()),
                        "effect_abs": effect_abs,
                        "signal_floor": signal_floor,
                        "signal_gate_pass": gate_pass,
                        "signal_gate_quantile": gate_quantile,
                        "onehop_empirical_floor": onehop_floor,
                        "reference_floor_cap_fraction": gate_floor_cap_fraction,
                    }
                )
                # Patch all measured samples (see Step 4): skip only truly-zero unclamped carriage.
                if abs(float(c[carrier, source].item())) < float(min_effect_abs):
                    continue
                cut = mediator_cut(graph, carrier, source)
                if not cut:
                    continue
                direct[carrier, source] = patched_ig_pair(
                    model.adapter,
                    graph,
                    baseline,
                    clean_cache,
                    readout_grad,
                    carrier=carrier,
                    source=source,
                    clamp_nodes=cut,
                    steps=ig_steps,
                    clamp_mode=clamp_mode,
                    clean_encoded_override=clean_encoded,
                    baseline_override=base_encoded,
                )
            r_nc_by_tau = {int(threshold): r_nc_estimate(direct, dist, int(threshold)) for threshold in thresholds}
            attention_far = float("nan")
            if clean_cache.attention:
                attention_last = clean_cache.attention[-1].detach().cpu().mean(dim=0)
                attention_far = far_mass(attention_last, dist, tau)
            carriage_far = far_mass(c, dist, tau)
            total_carriage_mass = float(c.detach().abs().sum().item())
            direct_share_stats = bounded_direct_far_mass_share(direct, c, dist, tau)
            direct_far_share = safe_float(direct_share_stats.get("direct_far_share_bounded"))
            for rung_index, (rung, value, denominator) in enumerate(
                [
                    ("attention", attention_far, "own_attention_mass"),
                    ("carriage", carriage_far, "own_carriage_mass"),
                    ("direct_carriage", direct_far_share, "bounded_measured_direct_mass_over_total_carriage_mass"),
                ]
            ):
                rung_rows.append(
                    {
                        "model": model.name,
                        "role": model.role,
                        "graph_id": gid,
                        "tau": tau,
                        "rung_index": rung_index,
                        "rung": rung,
                        "far_mass": value,
                        "denominator": denominator,
                        "r_nc": r_nc_by_tau[int(tau)].get("r_nc"),
                        "total_carriage_mass": total_carriage_mass,
                        "direct_far_share_raw": direct_share_stats.get("direct_far_share_raw"),
                        "direct_far_share_bounded": direct_share_stats.get("direct_far_share_bounded"),
                        "direct_far_fraction_measured_unclamped": direct_share_stats.get("direct_far_fraction_measured_unclamped"),
                        "direct_far_mass_raw": direct_share_stats.get("direct_far_mass_raw"),
                        "direct_far_mass_bounded": direct_share_stats.get("direct_far_mass_bounded"),
                        "measured_unclamped_far_mass": direct_share_stats.get("measured_unclamped_far_mass"),
                        "measured_far_pairs": direct_share_stats.get("measured_far_pairs"),
                        "total_far_pairs": direct_share_stats.get("total_far_pairs"),
                        "measured_pair_coverage": direct_share_stats.get("measured_pair_coverage"),
                        "attention_available": bool(clean_cache.attention),
                        "signal_gate_enabled": use_signal_gate,
                    }
                )
            if model.name == dense.name and clean_cache.attention and torch.isfinite(direct).any():
                finite_direct = direct.detach().abs()
                finite_direct[~torch.isfinite(direct)] = 0.0
                carrier, source = (int(v) for v in torch.nonzero(finite_direct == finite_direct.max(), as_tuple=False)[0].tolist())
                score = float(finite_direct[carrier, source].item())
                if score > worked_example_score and torch.isfinite(dist[carrier, source]):
                    cut = mediator_cut(graph, carrier, source)
                    unclamped = float(c[carrier, source].item())
                    direct_fraction, nontrivial = direct_fraction_value(
                        float(direct[carrier, source].item()),
                        unclamped,
                        min_effect_abs=min_effect_abs,
                    )
                    worked_example_score = score
                    worked_example = {
                        "model": model.name,
                        "graph_id": gid,
                        "edge_index": graph.edge_index.detach().cpu().clone(),
                        "attention": clean_cache.attention[-1].detach().cpu().mean(dim=0),
                        "carriage": c.detach().cpu().clone(),
                        "direct": direct.detach().cpu().clone(),
                        "dist": dist.detach().cpu().clone(),
                        "carrier": carrier,
                        "source": source,
                        "cut": cut,
                        "distance": float(dist[carrier, source].item()),
                        "direct_value": float(direct[carrier, source].item()),
                        "unclamped_value": unclamped,
                        "direct_fraction": direct_fraction,
                        "nontrivial_effect": nontrivial,
                    }
            y = graph_label(graph)
            model_pred = float(result["prediction"])
            append_total_carriage_ablation_rows(
                total_carriage_ablation_rows,
                model=model,
                graph_id=gid,
                prediction=model_pred,
                target=y,
                carriage=c,
                dist=dist,
                tau=tau,
                fractions=load_bearing_fractions,
                random_draws=load_bearing_random_draws,
                seed=seed,
            )
            append_component_carriage_ablation_rows(
                component_carriage_ablation_rows,
                model=model,
                graph_id=gid,
                prediction=model_pred,
                target=y,
                carriage=c,
                direct=direct,
                dist=dist,
                tau=tau,
                fractions=load_bearing_fractions,
            )
            append_distance_binned_total_carriage_ablation_rows(
                distance_binned_ablation_rows,
                model=model,
                graph_id=gid,
                prediction=model_pred,
                target=y,
                carriage=c,
                dist=dist,
                bins=distance_bins,
            )
            graph_carriage_records[(model.name, gid)] = {
                "model": model.name,
                "graph_id": gid,
                "prediction": float(model_pred),
                "target": float(y),
                "carriage": c.detach().cpu().clone(),
                "dist": dist.detach().cpu().clone(),
            }
            if model.name == dense.name:
                graph_stats = graph_distance_summary(dist)
                r_nc = safe_float(r_nc_by_tau[int(tau)].get("r_nc"))
                # result["prediction"] is dense's clean forward from encoded content; by
                # construction it equals dense.adapter.predict(graph) (content_override is injected
                # at the FeatureEncoder capture point in _run_with_hooks, so predict and this
                # reconstruction share the exact same forward). This is dense's true model
                # prediction -- there is no reconstruction gap to correct for here.
                dense_pred = model_pred
                try:
                    onehop_pred = float(onehop.adapter.predict(graph).reshape(-1)[0].detach().cpu().item())
                except Exception:
                    onehop_pred = float("nan")
                dense_err = abs(dense_pred - y) if math.isfinite(y) else float("nan")
                onehop_err = abs(onehop_pred - y) if math.isfinite(y) and math.isfinite(onehop_pred) else float("nan")
                optional_reference_errors: dict[str, float] = {}
                for ref_name in ("gin", "grit_1hop_localrrwp"):
                    ref_model = by_name.get(ref_name)
                    if ref_model is None:
                        continue
                    try:
                        ref_pred = float(ref_model.adapter.predict(graph).reshape(-1)[0].detach().cpu().item())
                    except Exception:
                        ref_pred = float("nan")
                    optional_reference_errors[ref_name] = (
                        abs(ref_pred - y)
                        if math.isfinite(y) and math.isfinite(ref_pred)
                        else float("nan")
                    )
                # Self-ablation (dense as its own control): remove the measured non-composable
                # (direct) far carriage from dense's own prediction, keeping everything a GNN could
                # compose. Reconstruction identity ŷ = ŷ_base + Σ C[i,j] => the direct far
                # contribution is the SIGNED sum of C^clamp[i,j] over d>τ pairs. If ablating it
                # pushes dense's error toward 1-hop's, the non-composable transport is the causal
                # source of the advantage; if error barely moves, it is present but not load-bearing.
                far_signed_mask = torch.isfinite(dist) & (dist > int(tau)) & torch.isfinite(direct)
                signed_direct_far = float(direct[far_signed_mask].sum().item()) if bool(far_signed_mask.any()) else 0.0
                dense_ablated_pred = dense_pred - signed_direct_far
                dense_ablated_err = abs(dense_ablated_pred - y) if math.isfinite(y) else float("nan")
                ablation_minus_dense_error = (
                    dense_ablated_err - dense_err
                    if math.isfinite(dense_ablated_err) and math.isfinite(dense_err)
                    else float("nan")
                )
                gap_row = {
                    "graph_id": gid,
                    "tau": tau,
                    "r_nc_model": dense.name,
                    "r_nc": r_nc,
                    "r_nc_raw_sample_sum": r_nc_by_tau[int(tau)].get("r_nc_raw_sample_sum"),
                    "r_nc_mean_abs_sampled_pair": r_nc_by_tau[int(tau)].get("r_nc_mean_abs_sampled_pair"),
                    "r_nc_size_scaled_sum_estimate": r_nc_by_tau[int(tau)].get("r_nc_size_scaled_sum_estimate"),
                    "r_nc_sampled_pairs": r_nc_by_tau[int(tau)].get("r_nc_sampled_pairs"),
                    "r_nc_total_far_pairs": r_nc_by_tau[int(tau)].get("r_nc_total_far_pairs"),
                    "r_nc_pair_coverage": r_nc_by_tau[int(tau)].get("r_nc_pair_coverage"),
                    "r_nc_scaled_from_sample": r_nc_by_tau[int(tau)].get("r_nc_scaled_from_sample"),
                    "r_nc_definition": r_nc_by_tau[int(tau)].get("r_nc_definition"),
                    "graph_num_nodes": graph_num_nodes(graph),
                    "graph_diameter": graph_stats.get("graph_diameter"),
                    "graph_mean_distance": graph_stats.get("graph_mean_distance"),
                    "signal_gate_enabled": use_signal_gate,
                    "signal_gate_quantile": gate_quantile,
                    "onehop_empirical_floor": onehop_floor,
                    "clamp_mode": clamp_mode,
                    "dense_error": dense_err,
                    "onehop_error": onehop_err,
                    "onehop_minus_dense_error": onehop_err - dense_err if math.isfinite(onehop_err) and math.isfinite(dense_err) else float("nan"),
                    "gap_treatment": dense.name,
                    "gap_control": onehop.name,
                    "signed_direct_far_carriage": signed_direct_far,
                    "dense_prediction": dense_pred,
                    "dense_ablated_prediction": dense_ablated_pred,
                    "dense_ablated_error": dense_ablated_err,
                    "ablation_minus_dense_error": ablation_minus_dense_error,
                }
                for ref_name, ref_err in optional_reference_errors.items():
                    gap_row[f"{ref_name}_error"] = ref_err
                    gap_row[f"{ref_name}_minus_dense_error"] = (
                        ref_err - dense_err
                        if math.isfinite(ref_err) and math.isfinite(dense_err)
                        else float("nan")
                    )
                    gap_row[f"{ref_name}_minus_onehop_error"] = (
                        ref_err - onehop_err
                        if math.isfinite(ref_err) and math.isfinite(onehop_err)
                        else float("nan")
                    )
                for threshold, stats in r_nc_by_tau.items():
                    gap_row[f"r_nc_tau_{threshold}"] = stats.get("r_nc")
                    gap_row[f"r_nc_tau_{threshold}_coverage"] = stats.get("r_nc_pair_coverage")
                    gap_row[f"r_nc_tau_{threshold}_size_scaled_sum_estimate"] = stats.get("r_nc_size_scaled_sum_estimate")
                gap_rows.append(gap_row)
            for threshold in thresholds:
                far_mask = torch.isfinite(dist) & (dist > threshold)
                measured_mask = far_mask & torch.isfinite(direct)
                total_pairs = int(far_mask.sum().item())
                sampled_pairs = int(measured_mask.sum().item())
                coverage = float(sampled_pairs / total_pairs) if total_pairs else float("nan")
                row = {
                    "model": model.name,
                    "role": model.role,
                    "graph_id": gid,
                    "tau": threshold,
                    "primary_tau": threshold == tau,
                    "r_nc": r_nc_by_tau[int(threshold)].get("r_nc"),
                    "r_nc_raw_sample_sum": r_nc_by_tau[int(threshold)].get("r_nc_raw_sample_sum"),
                    "r_nc_mean_abs_sampled_pair": r_nc_by_tau[int(threshold)].get("r_nc_mean_abs_sampled_pair"),
                    "r_nc_size_scaled_sum_estimate": r_nc_by_tau[int(threshold)].get("r_nc_size_scaled_sum_estimate"),
                    "r_nc_definition": r_nc_by_tau[int(threshold)].get("r_nc_definition"),
                    "sampled_pairs": sampled_pairs,
                    "total_far_pairs": total_pairs,
                    "pair_coverage": coverage,
                    "signal_gate_enabled": use_signal_gate,
                    "signal_gate_quantile": gate_quantile,
                    "onehop_empirical_floor": onehop_floor,
                    "clamp_mode": clamp_mode,
                }
                if total_pairs > 0 and sampled_pairs > 0:
                    far_matrix = torch.zeros_like(direct)
                    far_matrix[measured_mask] = direct[measured_mask]
                    row.update(
                        {
                            "status": "complete_sampled_signal_pairs",
                            "effective_rank": effective_rank(far_matrix),
                            "top_singular_share": top_singular_share(far_matrix),
                            "above_null_margin": distance_preserving_above_null_margin(far_matrix, dist, seed=seed + graph_idx + threshold),
                        }
                    )
                else:
                    row.update(
                        {
                            "status": "skipped_no_signal_pairs",
                            "effective_rank": float("nan"),
                            "top_singular_share": float("nan"),
                            "above_null_margin": float("nan"),
                        }
                    )
                rank_rows.append(row)
            per_graph_interactions = max(1, interaction_pairs // max(1, sample_graphs))
            candidate_pairs = far_pairs(dist, tau, max_pairs=per_graph_interactions * 4, seed=seed + 1000 + graph_idx)
            pairs: list[tuple[int, int]] = []
            for carrier, source in candidate_pairs:
                # Measure non-additivity only on pairs with real (non-trivial) carriage, using the
                # same trivial-zero threshold as patching rather than the aggressive noise gate.
                if abs(float(c[carrier, source].item())) >= float(min_effect_abs):
                    pairs.append((carrier, source))
                if len(pairs) >= per_graph_interactions:
                    break
            interaction_rows.extend(
                non_additivity_rows(
                    model,
                    graph,
                    baseline,
                    pairs,
                    gid,
                    rng,
                    min_effect_abs=non_additivity_effect_floor,
                    partner_policy=partner_policy,
                )
            )
        if cache_hits:
            progress(f"Step 5 {model.name}: reused cached IG carriage for {cache_hits}/{len(graphs)} graph(s)")
    write_csv(artifact_root / "metrics" / "step5_gap_vs_rnc.csv", gap_rows)
    write_csv(artifact_root / "metrics" / "step5_far_carriage_rank.csv", rank_rows)
    write_csv(artifact_root / "metrics" / "step5_non_additivity.csv", interaction_rows)
    write_csv(artifact_root / "metrics" / "step5_rung_funnel.csv", rung_rows)
    write_csv(artifact_root / "metrics" / "step5_signal_gate.csv", signal_gate_rows)
    write_csv(artifact_root / "metrics" / "step5_total_carriage_ablation.csv", total_carriage_ablation_rows)
    write_csv(artifact_root / "metrics" / "step5_component_carriage_ablation.csv", component_carriage_ablation_rows)
    write_csv(artifact_root / "metrics" / "step5_distance_binned_total_carriage_ablation.csv", distance_binned_ablation_rows)
    append_dense_control_excess_distance_ablation_rows(
        dense_control_excess_ablation_rows,
        records=graph_carriage_records,
        dense_model=dense.name,
        control_model=dense_control_model,
        bins=dense_excess_bins,
    )
    write_csv(artifact_root / "metrics" / "step5_dense_minus_control_distance_ablation.csv", dense_control_excess_ablation_rows)
    total_carriage_ablation_summary = carriage_ablation_summary_rows(
        total_carriage_ablation_rows,
        extra_keys=["ranker", "component"],
        seed=9100,
    )
    component_carriage_ablation_summary = carriage_ablation_summary_rows(
        component_carriage_ablation_rows,
        extra_keys=["component"],
        seed=9300,
    )
    distance_binned_ablation_summary = distance_binned_carriage_ablation_summary_rows(
        distance_binned_ablation_rows,
        seed=9500,
    )
    dense_control_excess_ablation_summary = dense_control_excess_distance_ablation_summary_rows(
        dense_control_excess_ablation_rows,
        seed=9700,
    )
    write_csv(artifact_root / "metrics" / "step5_total_carriage_ablation_summary.csv", total_carriage_ablation_summary)
    write_csv(artifact_root / "metrics" / "step5_component_carriage_ablation_summary.csv", component_carriage_ablation_summary)
    write_csv(artifact_root / "metrics" / "step5_distance_binned_total_carriage_ablation_summary.csv", distance_binned_ablation_summary)
    write_csv(
        artifact_root / "metrics" / "step5_dense_minus_control_distance_ablation_summary.csv",
        dense_control_excess_ablation_summary,
    )
    write_json(artifact_root / "metrics" / "step5_gap_regression.json", gap_regression_summary(gap_rows))
    ablation = self_ablation_summary(gap_rows)
    write_json(artifact_root / "metrics" / "step5_self_ablation.json", ablation)
    render_step5_self_ablation(ablation, gap_rows, artifact_root, dpi=dpi)
    render_step5_load_bearing_carriage_ablation(
        total_carriage_ablation_summary,
        component_carriage_ablation_summary,
        artifact_root,
        dpi=dpi,
    )
    render_step5_distance_binned_total_carriage_ablation(
        distance_binned_ablation_summary,
        artifact_root,
        dpi=dpi,
    )
    render_step5_dense_control_excess_distance_ablation(
        dense_control_excess_ablation_summary,
        artifact_root,
        dpi=dpi,
    )
    render_step5_gap_structural_severity(gap_rows, artifact_root, dpi=dpi)
    if int(ablation.get("n", 0)) > 0:
        order_sig = bool(ablation.get("dense_onehop_order_significant", False))
        progress(
            f"Step 5 self-ablation (n={int(ablation.get('n', 0))} molecules): dense_err="
            f"{safe_float(ablation.get('mean_dense_error')):.4f}, ablated_err="
            f"{safe_float(ablation.get('mean_dense_ablated_error')):.4f}, onehop_err="
            f"{safe_float(ablation.get('mean_onehop_error')):.4f}; ablation_penalty="
            f"{safe_float(ablation.get('ablation_penalty')):.4f} "
            f"[{safe_float(ablation.get('ablation_penalty_ci_low')):.4f}, {safe_float(ablation.get('ablation_penalty_ci_high')):.4f}]; "
            f"onehop-dense={safe_float(ablation.get('onehop_minus_dense_error_mean')):.4f} "
            f"[{safe_float(ablation.get('onehop_minus_dense_error_ci_low')):.4f}, {safe_float(ablation.get('onehop_minus_dense_error_ci_high')):.4f}] "
            f"({'order significant' if order_sig else 'order NOT separable at this n'}) "
            f"-> {ablation.get('verdict')}"
        )
    write_csv(artifact_root / "metrics" / "step5_far_carriage_rank_summary.csv", rank_summary_rows(rank_rows, models=[dense.name]))
    write_csv(artifact_root / "metrics" / "step5_non_additivity_summary.csv", non_additivity_summary_rows(interaction_rows, models=[dense.name]))
    vnode_decision = summarize_vnode_decision(rank_rows, interaction_rows, model=dense.name)
    write_json(artifact_root / "metrics" / "step5_vnode_decision.json", vnode_decision)
    write_csv(artifact_root / "metrics" / "step5_vnode_decision.csv", [vnode_decision])
    signal_vs_noise = render_step5_signal_vs_noise(rank_rows, artifact_root, dpi=dpi)
    write_json(artifact_root / "metrics" / "step5_signal_vs_noise.json", signal_vs_noise)
    if signal_vs_noise.get("status") == "computed":
        progress(
            "Step 5 signal-vs-noise (dense far carriage): observed_top_share="
            f"{safe_float(signal_vs_noise.get('observed_top_share_mean')):.3f} vs null="
            f"{safe_float(signal_vs_noise.get('null_top_share_mean')):.3f}; above-null margin="
            f"{safe_float(signal_vs_noise.get('above_null_margin_mean')):.3f} "
            f"[{safe_float(signal_vs_noise.get('above_null_margin_ci_low')):.3f}, "
            f"{safe_float(signal_vs_noise.get('above_null_margin_ci_high')):.3f}] -> {signal_vs_noise.get('verdict')}"
        )
    render_step5(gap_rows, rank_rows, interaction_rows, artifact_root, dpi=dpi)
    render_step5_vnode_decision(vnode_decision, artifact_root, dpi=dpi)
    render_step5_rung_funnel(rung_rows, artifact_root, dpi=dpi)
    if worked_example is not None:
        render_worked_molecule_example(worked_example, artifact_root, dpi=dpi)
        write_json(
            artifact_root / "metrics" / "step5_worked_molecule_example.json",
            {
                key: value
                for key, value in worked_example.items()
                if key
                not in {
                    "edge_index",
                    "attention",
                    "carriage",
                    "direct",
                    "dist",
                }
            },
        )
    progress("Step 5 complete: metrics, tensors, and figures written")
    rank_complete = sum(1 for row in rank_rows if str(row.get("status")).startswith("complete"))
    return {
        "status": "complete",
        "models": [model.name for model in analysis_models],
        "missing_optional_reference_models": missing_requested,
        "gap_rows": len(gap_rows),
        "rank_rows": len(rank_rows),
        "rank_complete_rows": rank_complete,
        "rank_skipped_rows": len(rank_rows) - rank_complete,
        "interaction_rows": len(interaction_rows),
        "rung_rows": len(rung_rows),
        "signal_gate_rows": len(signal_gate_rows),
        "signal_gate_pass_rows": len([r for r in signal_gate_rows if bool(r.get("signal_gate_pass"))]),
        "total_carriage_ablation_rows": len(total_carriage_ablation_rows),
        "component_carriage_ablation_rows": len(component_carriage_ablation_rows),
        "distance_binned_ablation_rows": len(distance_binned_ablation_rows),
        "dense_control_excess_ablation_rows": len(dense_control_excess_ablation_rows),
        "dense_control_excess_control_model": dense_control_model,
        "signal_gate_enabled": use_signal_gate,
        "signal_gate_quantile": gate_quantile,
        "onehop_empirical_floor": onehop_floor,
        "worked_molecule_example": worked_example is not None,
    }


def non_additivity_rows(
    model: ModelRun,
    graph: Any,
    mean_baseline: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
    graph_id: str,
    rng: random.Random,
    *,
    min_effect_abs: float = 1.0e-6,
    partner_policy: str = "different_type",
) -> list[dict[str, Any]]:
    encoded = model.adapter.encoded_node_states(graph).detach().to(model.adapter.device)
    clean = float(predict_scalar_from_encoded(model.adapter, graph, encoded).detach().cpu().item())
    rows = []
    seen: set[tuple[int, int]] = set()
    type_signatures = node_type_signatures(graph)
    require_different_type = str(partner_policy).strip().lower() in {"different_type", "different-type", "different_atom_type"}

    def partner_candidates(source: int, blocked: set[int]) -> list[int]:
        choices = [idx for idx in range(encoded.size(0)) if idx not in blocked]
        if require_different_type and type_signatures is not None and int(source) < len(type_signatures):
            source_type = type_signatures[int(source)]
            different = [idx for idx in choices if idx < len(type_signatures) and type_signatures[idx] != source_type]
            if different:
                return different
        return choices

    for a, b in pairs:
        if a == b:
            continue
        key = tuple(sorted((int(a), int(b))))
        if key in seen:
            continue
        seen.add(key)
        candidates_a = partner_candidates(int(a), {int(a), int(b)})
        candidates_b = partner_candidates(int(b), {int(a), int(b)})
        if not candidates_a or not candidates_b:
            continue
        partner_a = rng.choice(candidates_a)
        partner_b = rng.choice(candidates_b)
        xa = encoded.detach().clone()
        xb = encoded.detach().clone()
        xab = encoded.detach().clone()
        xa[a] = encoded[partner_a]
        xb[b] = encoded[partner_b]
        xab[a] = encoded[partner_a]
        xab[b] = encoded[partner_b]
        da = float(predict_scalar_from_encoded(model.adapter, graph, xa).detach().cpu().item()) - clean
        db = float(predict_scalar_from_encoded(model.adapter, graph, xb).detach().cpu().item()) - clean
        dab = float(predict_scalar_from_encoded(model.adapter, graph, xab).detach().cpu().item()) - clean
        denom = abs(da) + abs(db)
        nontrivial = abs(da) >= float(min_effect_abs) and abs(db) >= float(min_effect_abs) and denom >= float(min_effect_abs)
        ratio = abs(dab - da - db) / denom if nontrivial else float("nan")
        rows.append(
            {
                "model": model.name,
                "graph_id": graph_id,
                "source_a": a,
                "source_b": b,
                "partner_a": partner_a,
                "partner_b": partner_b,
                "delta_a": da,
                "delta_b": db,
                "delta_ab": dab,
                "denominator": denom,
                "non_additivity": ratio,
                "min_effect_abs": min_effect_abs,
                "nontrivial_effect": nontrivial,
                "perturbation": "finite_content_swap",
                "partner_policy": partner_policy,
            }
        )
    return rows


def distance_preserving_above_null_margin(
    matrix: torch.Tensor,
    distances: torch.Tensor,
    *,
    permutations: int = 32,
    seed: int = 0,
) -> float:
    """Top singular-share margin over a distance-preserving shuffle null."""

    mat = matrix.detach().cpu().double()
    dist = distances.detach().cpu()
    observed = top_singular_share(mat)
    finite = torch.isfinite(dist)
    bins = sorted({int(v.item()) for v in dist[finite].reshape(-1)})
    rng = np.random.default_rng(int(seed))
    null_values = []
    for _ in range(int(permutations)):
        shuffled = mat.clone()
        for d in bins:
            mask = finite & (dist.long() == int(d))
            vals = shuffled[mask].detach().cpu().numpy()
            if vals.size > 1:
                rng.shuffle(vals)
                shuffled[mask] = torch.as_tensor(vals, dtype=shuffled.dtype)
        null_values.append(top_singular_share(shuffled))
    if not null_values:
        return float("nan")
    return float(observed - float(np.mean(null_values)))


def summarize_vnode_decision(
    rank_rows: Sequence[Mapping[str, Any]],
    interaction_rows: Sequence[Mapping[str, Any]],
    *,
    model: str = "dense_grit",
) -> dict[str, Any]:
    primary_rank_rows = [
        r
        for r in rank_rows
        if bool(r.get("primary_tau", True))
        and str(r.get("model")) == str(model)
        and str(r.get("status", "")).startswith("complete")
        and int(safe_float(r.get("sampled_pairs")) if math.isfinite(safe_float(r.get("sampled_pairs"))) else 0) >= 3
    ]
    if primary_rank_rows:
        rank_rows = primary_rank_rows
    else:
        rank_rows = [
            r
            for r in rank_rows
            if str(r.get("model")) == str(model)
            and str(r.get("status", "")).startswith("complete")
            and int(safe_float(r.get("sampled_pairs")) if math.isfinite(safe_float(r.get("sampled_pairs"))) else 0) >= 3
        ]
    non_add = np.asarray(
        [
            safe_float(r.get("non_additivity"))
            for r in interaction_rows
            if str(r.get("model")) == str(model) and row_is_nontrivial(r, default=False)
        ],
        dtype=float,
    )
    non_add = non_add[np.isfinite(non_add)]
    top_share = np.asarray([safe_float(r.get("top_singular_share")) for r in rank_rows], dtype=float)
    top_share = top_share[np.isfinite(top_share)]
    effective = np.asarray([safe_float(r.get("effective_rank")) for r in rank_rows], dtype=float)
    effective = effective[np.isfinite(effective)]
    above_null = np.asarray([safe_float(r.get("above_null_margin")) for r in rank_rows], dtype=float)
    above_null = above_null[np.isfinite(above_null)]
    mean_non_add = float(np.mean(non_add)) if non_add.size else float("nan")
    median_non_add = float(np.median(non_add)) if non_add.size else float("nan")
    mean_top_share = float(np.mean(top_share)) if top_share.size else float("nan")
    mean_effective_rank = float(np.mean(effective)) if effective.size else float("nan")
    mean_above_null = float(np.mean(above_null)) if above_null.size else float("nan")
    threshold = 0.10
    rank1_top_share_threshold = 0.90
    if not math.isfinite(mean_non_add):
        decision = "insufficient_non_additivity_data"
    elif not (math.isfinite(mean_top_share) or math.isfinite(mean_effective_rank)):
        decision = "insufficient_far_carriage_rank_data"
    elif mean_non_add <= threshold and (mean_top_share >= rank1_top_share_threshold or mean_effective_rank <= 1.25):
        decision = "rank1_broadcast_plausible_by_additivity"
    else:
        decision = "source_specific_interactions_exceed_rank1_broadcast"
    return {
        "decision": decision,
        "model": model,
        "non_additivity_threshold": threshold,
        "mean_non_additivity": mean_non_add,
        "median_non_additivity": median_non_add,
        "rank1_top_share_threshold": rank1_top_share_threshold,
        "mean_effective_rank": mean_effective_rank,
        "mean_top_singular_share": mean_top_share,
        "mean_distance_preserving_above_null_margin": mean_above_null,
        "interaction_pairs": int(non_add.size),
        "rank_graphs": int(top_share.size),
        "interpretation": (
            "VNode/rank-1 broadcast is plausible only when distant-source effects are close to additive; "
            "high non-additivity means a single pooled global node cannot reproduce source-specific pair coupling."
        ),
    }


def ablation_fractions(values: Any) -> list[float]:
    if values is None:
        values = [0.0, 0.05, 0.10, 0.25, 0.50, 1.0]
    if isinstance(values, str):
        raw = [part.strip() for part in values.split(",") if part.strip()]
    elif isinstance(values, Sequence):
        raw = list(values)
    else:
        raw = [values]
    out = []
    for value in raw:
        f = safe_float(value)
        if not math.isfinite(f):
            continue
        out.append(min(1.0, max(0.0, f)))
    out = sorted(set(out))
    if 0.0 not in out:
        out.insert(0, 0.0)
    return out or [0.0, 0.05, 0.10, 0.25, 0.50, 1.0]


def distance_ablation_bins(values: Any = None) -> list[dict[str, Any]]:
    if values is None:
        return [
            {"label": "d=2-3", "min": 2, "max": 3},
            {"label": "d=4-6", "min": 4, "max": 6},
            {"label": "d=7-10", "min": 7, "max": 10},
            {"label": "d=11-14", "min": 11, "max": 14},
            {"label": "d>14", "min": 15, "max": None},
        ]
    out: list[dict[str, Any]] = []
    for item in values if isinstance(values, Sequence) and not isinstance(values, str) else [values]:
        if isinstance(item, Mapping):
            low = int(item.get("min", item.get("low", 0)))
            high_raw = item.get("max", item.get("high"))
            high = None if high_raw in (None, "", "none", "None") else int(high_raw)
            label = str(item.get("label") or (f"d>{low - 1}" if high is None else f"d={low}-{high}"))
            out.append({"label": label, "min": low, "max": high})
        elif isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            if ">" in text:
                low = int(re.findall(r"\d+", text)[0]) + 1
                out.append({"label": text, "min": low, "max": None})
            else:
                nums = [int(v) for v in re.findall(r"\d+", text)]
                if len(nums) == 1:
                    out.append({"label": f"d={nums[0]}", "min": nums[0], "max": nums[0]})
                elif len(nums) >= 2:
                    out.append({"label": f"d={nums[0]}-{nums[1]}", "min": nums[0], "max": nums[1]})
    return out or distance_ablation_bins(None)


def far_only_distance_bins(bins: Sequence[Mapping[str, Any]], *, tau: int) -> list[dict[str, Any]]:
    """Keep only distance bins that sit strictly beyond the dissertation far threshold."""

    out: list[dict[str, Any]] = []
    threshold = int(tau)
    for spec in bins:
        low = int(spec.get("min", 0))
        high_raw = spec.get("max")
        high = None if high_raw in (None, "", "none", "None") else int(high_raw)
        if high is not None and high <= threshold:
            continue
        clipped_low = max(low, threshold + 1)
        label = str(spec.get("label") or (f"d>{clipped_low - 1}" if high is None else f"d={clipped_low}-{high}"))
        if clipped_low != low:
            label = f"d>{threshold}" if high is None else f"d={clipped_low}-{high}"
        out.append({"label": label, "min": clipped_low, "max": high})
    return out or [{"label": f"d>{threshold}", "min": threshold + 1, "max": None}]


def top_fraction_count(total: int, fraction: float) -> int:
    total = max(0, int(total))
    f = min(1.0, max(0.0, float(fraction)))
    if total == 0 or f <= 0.0:
        return 0
    if f >= 1.0:
        return total
    return max(1, int(math.ceil(total * f)))


def append_total_carriage_ablation_rows(
    rows: list[dict[str, Any]],
    *,
    model: ModelRun,
    graph_id: str,
    prediction: float,
    target: float,
    carriage: torch.Tensor,
    dist: torch.Tensor,
    tau: int,
    fractions: Sequence[float],
    random_draws: int,
    seed: int,
) -> None:
    """Attribution-space removal of total long-range carriage C_unclamp.

    This is the cross-model "is long-range usage load-bearing?" test. It does not
    use the mediator clamp: IG completeness gives yhat - yhat_base = sum C, so
    subtracting selected signed C entries is the on-attribution counterfactual for
    removing those pair contributions while leaving all other contributions intact.
    """
    if not (math.isfinite(prediction) and math.isfinite(target)):
        return
    c = carriage.detach().cpu()
    d = dist.detach().cpu()
    far_mask = torch.isfinite(d) & (d > int(tau)) & torch.isfinite(c)
    pair_index = torch.nonzero(far_mask, as_tuple=False)
    if pair_index.numel() == 0:
        return
    values = c[far_mask].reshape(-1)
    order = torch.argsort(values.abs(), descending=True)
    ordered_values = values[order].detach().cpu().numpy().astype(float)
    all_values = values.detach().cpu().numpy().astype(float)
    total_pairs = int(all_values.size)
    base_error = abs(float(prediction) - float(target))
    for fraction in fractions:
        k = top_fraction_count(total_pairs, float(fraction))
        ranked_sum = float(np.sum(ordered_values[:k])) if k else 0.0
        ranked_abs = float(np.sum(np.abs(ordered_values[:k]))) if k else 0.0
        ablated_pred = float(prediction) - ranked_sum
        ablated_error = abs(ablated_pred - float(target))
        rows.append(
            {
                "model": model.name,
                "role": model.role,
                "graph_id": graph_id,
                "tau": int(tau),
                "ranker": "carriage",
                "component": "total_carriage",
                "fraction_removed": float(fraction),
                "actual_fraction_removed": float(k / total_pairs) if total_pairs else float("nan"),
                "removed_pairs": int(k),
                "total_far_pairs": int(total_pairs),
                "prediction": float(prediction),
                "target": float(target),
                "baseline_error": float(base_error),
                "ablated_prediction": float(ablated_pred),
                "ablated_error": float(ablated_error),
                "delta_mae": float(ablated_error - base_error),
                "signed_removed_carriage": ranked_sum,
                "abs_removed_carriage": ranked_abs,
                "random_draws": 0,
                "ablation_type": "ig_completeness_total_carriage",
            }
        )
        if random_draws <= 0:
            continue
        rng = np.random.default_rng(stable_seed(model.name, graph_id, tau, fraction, "random_total_carriage", base=seed))
        random_deltas: list[float] = []
        random_preds: list[float] = []
        random_abs: list[float] = []
        for _ in range(int(random_draws)):
            if k == 0:
                selected = np.asarray([], dtype=int)
            else:
                selected = rng.choice(total_pairs, size=k, replace=False)
            removed = float(np.sum(all_values[selected])) if k else 0.0
            removed_abs = float(np.sum(np.abs(all_values[selected]))) if k else 0.0
            random_pred = float(prediction) - removed
            random_error = abs(random_pred - float(target))
            random_deltas.append(float(random_error - base_error))
            random_preds.append(random_pred)
            random_abs.append(removed_abs)
        rows.append(
            {
                "model": model.name,
                "role": model.role,
                "graph_id": graph_id,
                "tau": int(tau),
                "ranker": "random_far_pairs",
                "component": "total_carriage",
                "fraction_removed": float(fraction),
                "actual_fraction_removed": float(k / total_pairs) if total_pairs else float("nan"),
                "removed_pairs": int(k),
                "total_far_pairs": int(total_pairs),
                "prediction": float(prediction),
                "target": float(target),
                "baseline_error": float(base_error),
                "ablated_prediction": float(np.mean(random_preds)) if random_preds else float(prediction),
                "ablated_error": float(base_error + np.mean(random_deltas)) if random_deltas else float(base_error),
                "delta_mae": float(np.mean(random_deltas)) if random_deltas else 0.0,
                "signed_removed_carriage": float("nan"),
                "abs_removed_carriage": float(np.mean(random_abs)) if random_abs else 0.0,
                "random_draws": int(random_draws),
                "ablation_type": "ig_completeness_total_carriage_random_control",
            }
        )


def append_distance_binned_total_carriage_ablation_rows(
    rows: list[dict[str, Any]],
    *,
    model: ModelRun,
    graph_id: str,
    prediction: float,
    target: float,
    carriage: torch.Tensor,
    dist: torch.Tensor,
    bins: Sequence[Mapping[str, Any]],
) -> None:
    """Remove all signed total carriage in each molecular-distance bin."""
    if not (math.isfinite(prediction) and math.isfinite(target)):
        return
    c = carriage.detach().cpu()
    d = dist.detach().cpu()
    finite = torch.isfinite(d) & torch.isfinite(c)
    base_error = abs(float(prediction) - float(target))
    for bin_index, spec in enumerate(bins):
        low = int(spec.get("min", 0))
        high_raw = spec.get("max")
        high = None if high_raw in (None, "", "none", "None") else int(high_raw)
        label = str(spec.get("label") or (f"d>{low - 1}" if high is None else f"d={low}-{high}"))
        mask = finite & (d >= low)
        if high is not None:
            mask = mask & (d <= high)
        pair_count = int(mask.sum().item())
        if pair_count:
            values = c[mask].reshape(-1).detach().cpu().numpy().astype(float)
            signed_removed = float(np.sum(values))
            abs_removed = float(np.sum(np.abs(values)))
            mean_abs_pair = float(np.mean(np.abs(values)))
        else:
            signed_removed = 0.0
            abs_removed = 0.0
            mean_abs_pair = float("nan")
        ablated_pred = float(prediction) - signed_removed
        ablated_error = abs(ablated_pred - float(target))
        rows.append(
            {
                "model": model.name,
                "role": model.role,
                "graph_id": graph_id,
                "distance_bin": label,
                "distance_bin_index": int(bin_index),
                "distance_min": low,
                "distance_max": "" if high is None else high,
                "pair_count": pair_count,
                "prediction": float(prediction),
                "target": float(target),
                "baseline_error": float(base_error),
                "ablated_prediction": float(ablated_pred),
                "ablated_error": float(ablated_error),
                "delta_mae": float(ablated_error - base_error),
                "signed_removed_carriage": signed_removed,
                "abs_removed_carriage": abs_removed,
                "mean_abs_carriage_per_pair": mean_abs_pair,
                "ablation_type": "ig_completeness_distance_binned_total_carriage",
            }
        )


def append_dense_control_excess_distance_ablation_rows(
    rows: list[dict[str, Any]],
    *,
    records: Mapping[tuple[str, str], Mapping[str, Any]],
    dense_model: str,
    control_model: str,
    bins: Sequence[Mapping[str, Any]],
) -> None:
    """Remove dense-specific excess carriage over a matched 1-hop control by distance bin.

    This is the distance-localised version of the performance-gap question.  It does not ask
    whether dense uses long-range carriage in isolation; it asks whether the part of dense's
    carriage that is absent from the matched 1-hop solution improves dense's own test error.
    """

    dense_ids = {gid for model, gid in records if model == dense_model}
    control_ids = {gid for model, gid in records if model == control_model}
    for gid in sorted(dense_ids.intersection(control_ids)):
        dense_record = records.get((dense_model, gid))
        control_record = records.get((control_model, gid))
        if dense_record is None or control_record is None:
            continue
        dense_c = dense_record.get("carriage")
        control_c = control_record.get("carriage")
        dist = dense_record.get("dist")
        if not isinstance(dense_c, torch.Tensor) or not isinstance(control_c, torch.Tensor) or not isinstance(dist, torch.Tensor):
            continue
        if tuple(dense_c.shape) != tuple(control_c.shape):
            continue
        prediction = safe_float(dense_record.get("prediction"))
        target = safe_float(dense_record.get("target"))
        if not (math.isfinite(prediction) and math.isfinite(target)):
            continue
        dense_c = dense_c.detach().cpu()
        control_c = control_c.detach().cpu()
        dist = dist.detach().cpu()
        excess = dense_c - control_c
        finite = torch.isfinite(dist) & torch.isfinite(excess)
        base_error = abs(prediction - target)
        for bin_index, spec in enumerate(bins):
            low = int(spec.get("min", 0))
            high_raw = spec.get("max")
            high = None if high_raw in (None, "", "none", "None") else int(high_raw)
            label = str(spec.get("label") or (f"d>{low - 1}" if high is None else f"d={low}-{high}"))
            mask = finite & (dist >= low)
            if high is not None:
                mask = mask & (dist <= high)
            pair_count = int(mask.sum().item())
            if pair_count:
                dense_values = dense_c[mask].reshape(-1).detach().cpu().numpy().astype(float)
                control_values = control_c[mask].reshape(-1).detach().cpu().numpy().astype(float)
                excess_values = excess[mask].reshape(-1).detach().cpu().numpy().astype(float)
                signed_dense = float(np.sum(dense_values))
                signed_control = float(np.sum(control_values))
                signed_excess = float(np.sum(excess_values))
                abs_excess = float(np.sum(np.abs(excess_values)))
                mean_abs_excess_pair = float(np.mean(np.abs(excess_values)))
            else:
                signed_dense = 0.0
                signed_control = 0.0
                signed_excess = 0.0
                abs_excess = 0.0
                mean_abs_excess_pair = float("nan")
            ablated_prediction = prediction - signed_excess
            ablated_error = abs(ablated_prediction - target)
            rows.append(
                {
                    "dense_model": dense_model,
                    "control_model": control_model,
                    "graph_id": gid,
                    "distance_bin": label,
                    "distance_bin_index": int(bin_index),
                    "distance_min": low,
                    "distance_max": "" if high is None else high,
                    "pair_count": pair_count,
                    "dense_prediction": prediction,
                    "target": target,
                    "dense_error": base_error,
                    "ablated_prediction": ablated_prediction,
                    "ablated_error": ablated_error,
                    "delta_mae": ablated_error - base_error,
                    "signed_dense_carriage": signed_dense,
                    "signed_control_carriage": signed_control,
                    "signed_dense_minus_control_carriage": signed_excess,
                    "abs_dense_minus_control_carriage": abs_excess,
                    "mean_abs_dense_minus_control_carriage_per_pair": mean_abs_excess_pair,
                    "ablation_type": "ig_completeness_dense_minus_control_distance_binned_carriage",
                }
            )


def append_component_carriage_ablation_rows(
    rows: list[dict[str, Any]],
    *,
    model: ModelRun,
    graph_id: str,
    prediction: float,
    target: float,
    carriage: torch.Tensor,
    direct: torch.Tensor,
    dist: torch.Tensor,
    tau: int,
    fractions: Sequence[float],
) -> None:
    """Decompose total carriage into non-composable and composable components.

    The direct matrix is C_clamp from the detach mediator clamp. It is used only
    for this mechanism decomposition; the cross-model load-bearing test uses the
    total C_unclamp matrix above.
    """
    if not (math.isfinite(prediction) and math.isfinite(target)):
        return
    c = carriage.detach().cpu()
    direct_c = direct.detach().cpu()
    d = dist.detach().cpu()
    measured_mask = torch.isfinite(d) & (d > int(tau)) & torch.isfinite(c) & torch.isfinite(direct_c)
    if not bool(measured_mask.any()):
        return
    total_values = c[measured_mask].reshape(-1)
    direct_values = direct_c[measured_mask].reshape(-1)
    composed_values = total_values - direct_values
    order = torch.argsort(total_values.abs(), descending=True)
    arrays = {
        "total_measured_carriage": total_values[order].detach().cpu().numpy().astype(float),
        "non_composable_carriage": direct_values[order].detach().cpu().numpy().astype(float),
        "composable_carriage": composed_values[order].detach().cpu().numpy().astype(float),
    }
    total_pairs = int(total_values.numel())
    base_error = abs(float(prediction) - float(target))
    for fraction in fractions:
        k = top_fraction_count(total_pairs, float(fraction))
        for component, values in arrays.items():
            removed = float(np.sum(values[:k])) if k else 0.0
            removed_abs = float(np.sum(np.abs(values[:k]))) if k else 0.0
            ablated_pred = float(prediction) - removed
            ablated_error = abs(ablated_pred - float(target))
            rows.append(
                {
                    "model": model.name,
                    "role": model.role,
                    "graph_id": graph_id,
                    "tau": int(tau),
                    "component": component,
                    "fraction_removed": float(fraction),
                    "actual_fraction_removed": float(k / total_pairs) if total_pairs else float("nan"),
                    "removed_pairs": int(k),
                    "measured_far_pairs": int(total_pairs),
                    "prediction": float(prediction),
                    "target": float(target),
                    "baseline_error": float(base_error),
                    "ablated_prediction": float(ablated_pred),
                    "ablated_error": float(ablated_error),
                    "delta_mae": float(ablated_error - base_error),
                    "signed_removed_carriage": removed,
                    "abs_removed_carriage": removed_abs,
                    "ranked_by": "abs_total_measured_carriage",
                    "ablation_type": "ig_completeness_component_carriage",
                }
            )


def carriage_ablation_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    extra_keys: Sequence[str],
    seed: int = 8800,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not rows:
        return out
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    keys = ["model", *extra_keys, "fraction_removed"]
    for row in rows:
        value = safe_float(row.get("delta_mae"))
        if not math.isfinite(value):
            continue
        key = tuple(row.get(k) for k in keys)
        groups.setdefault(key, []).append(row)
    for idx, (key, group_rows) in enumerate(sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0]))):
        values = [safe_float(r.get("delta_mae")) for r in group_rows]
        values = [v for v in values if math.isfinite(v)]
        if not values:
            continue
        mean, lo, hi = bootstrap_ci(values, seed=seed + idx, draws=1000)
        row = {name: value for name, value in zip(keys, key)}
        row.update(
            {
                "mean_delta_mae": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n_molecules": len(values),
                "mean_removed_pairs": float(np.mean([safe_float(r.get("removed_pairs")) for r in group_rows])),
                "mean_abs_removed_carriage": float(np.nanmean([safe_float(r.get("abs_removed_carriage")) for r in group_rows])),
                "mean_baseline_error": float(np.nanmean([safe_float(r.get("baseline_error")) for r in group_rows])),
            }
        )
        out.append(row)
    return out


def distance_binned_carriage_ablation_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 9500,
) -> list[dict[str, Any]]:
    clean = [
        r
        for r in rows
        if math.isfinite(safe_float(r.get("delta_mae")))
        and math.isfinite(safe_float(r.get("distance_bin_index")))
    ]
    if not clean:
        return []
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for row in clean:
        grouped.setdefault(
            (
                str(row.get("model")),
                int(safe_float(row.get("distance_bin_index"))),
                str(row.get("distance_bin")),
            ),
            [],
        ).append(row)
    out: list[dict[str, Any]] = []
    for idx, ((model, bin_index, label), group_rows) in enumerate(sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))):
        deltas = [safe_float(r.get("delta_mae")) for r in group_rows]
        deltas = [v for v in deltas if math.isfinite(v)]
        if not deltas:
            continue
        mean_delta, lo, hi = bootstrap_ci(deltas, seed=seed + idx, draws=1000)
        out.append(
            {
                "model": model,
                "distance_bin_index": bin_index,
                "distance_bin": label,
                "distance_min": group_rows[0].get("distance_min"),
                "distance_max": group_rows[0].get("distance_max"),
                "mean_delta_mae": mean_delta,
                "ci_low": lo,
                "ci_high": hi,
                "n_molecules": len(deltas),
                "mean_pair_count": float(np.nanmean([safe_float(r.get("pair_count")) for r in group_rows])),
                "mean_signed_removed_carriage": float(np.nanmean([safe_float(r.get("signed_removed_carriage")) for r in group_rows])),
                "mean_abs_removed_carriage": float(np.nanmean([safe_float(r.get("abs_removed_carriage")) for r in group_rows])),
                "mean_abs_carriage_per_pair": float(np.nanmean([safe_float(r.get("mean_abs_carriage_per_pair")) for r in group_rows])),
                "mean_baseline_error": float(np.nanmean([safe_float(r.get("baseline_error")) for r in group_rows])),
            }
        )
    return out


def dense_control_excess_distance_ablation_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 9700,
) -> list[dict[str, Any]]:
    clean = [
        r
        for r in rows
        if math.isfinite(safe_float(r.get("delta_mae")))
        and math.isfinite(safe_float(r.get("distance_bin_index")))
    ]
    if not clean:
        return []
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in clean:
        grouped.setdefault(
            (int(safe_float(row.get("distance_bin_index"))), str(row.get("distance_bin"))),
            [],
        ).append(row)
    out: list[dict[str, Any]] = []
    for idx, ((bin_index, label), group_rows) in enumerate(sorted(grouped.items(), key=lambda item: item[0][0])):
        deltas = [safe_float(r.get("delta_mae")) for r in group_rows]
        deltas = [v for v in deltas if math.isfinite(v)]
        if not deltas:
            continue
        mean_delta, lo, hi = bootstrap_ci(deltas, seed=seed + idx, draws=1000)
        out.append(
            {
                "dense_model": str(group_rows[0].get("dense_model", "dense_grit")),
                "control_model": str(group_rows[0].get("control_model", "grit_1hop")),
                "distance_bin_index": bin_index,
                "distance_bin": label,
                "distance_min": group_rows[0].get("distance_min"),
                "distance_max": group_rows[0].get("distance_max"),
                "mean_delta_mae": mean_delta,
                "ci_low": lo,
                "ci_high": hi,
                "n_molecules": len(deltas),
                "mean_pair_count": nanmean_or_nan([r.get("pair_count") for r in group_rows]),
                "mean_signed_dense_carriage": nanmean_or_nan([r.get("signed_dense_carriage") for r in group_rows]),
                "mean_signed_control_carriage": nanmean_or_nan([r.get("signed_control_carriage") for r in group_rows]),
                "mean_signed_dense_minus_control_carriage": nanmean_or_nan(
                    [r.get("signed_dense_minus_control_carriage") for r in group_rows]
                ),
                "mean_abs_dense_minus_control_carriage": nanmean_or_nan(
                    [r.get("abs_dense_minus_control_carriage") for r in group_rows]
                ),
                "mean_abs_dense_minus_control_carriage_per_pair": nanmean_or_nan(
                    [r.get("mean_abs_dense_minus_control_carriage_per_pair") for r in group_rows]
                ),
                "mean_dense_error": nanmean_or_nan([r.get("dense_error") for r in group_rows]),
            }
        )
    return out


def self_ablation_summary(gap_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Causal self-ablation: does removing dense's non-composable far carriage hurt test error?

    Compares three per-molecule errors: dense (intact), dense with the measured
    non-composable far carriage subtracted from its own prediction, and the trained
    1-hop control. dense is its own control, so the ablation isolates the shortcuts
    (no cross-model optimisation confound). If the ablation penalty recovers the
    trained-1hop gap, the non-composable transport is the causal source of the
    advantage; if it is ~0, the transport is present but not load-bearing.
    """
    dense = np.asarray([safe_float(r.get("dense_error")) for r in gap_rows], dtype=float)
    ablated = np.asarray([safe_float(r.get("dense_ablated_error")) for r in gap_rows], dtype=float)
    onehop = np.asarray([safe_float(r.get("onehop_error")) for r in gap_rows], dtype=float)
    rnc = np.asarray([safe_float(r.get("r_nc")) for r in gap_rows], dtype=float)
    effect = np.asarray([safe_float(r.get("ablation_minus_dense_error")) for r in gap_rows], dtype=float)
    mask = np.isfinite(dense) & np.isfinite(ablated) & np.isfinite(onehop)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "status": "no_scored_molecules"}
    d, a, o = dense[mask], ablated[mask], onehop[mask]
    ablation_penalty = float(np.mean(a) - np.mean(d))
    onehop_penalty = float(np.mean(o) - np.mean(d))
    # Paired bootstrap over molecules: resample once per draw and recompute every mean on the
    # same resample, so the three bars and their contrasts share sampling noise. At the small
    # sample sizes used by the lighter presets (medium = 16 molecules) this is what tells the
    # dense-vs-1hop order apart from noise instead of over-reading three bare means.
    rng = np.random.default_rng(9090)
    boot_d: list[float] = []
    boot_a: list[float] = []
    boot_o: list[float] = []
    pen_samples: list[float] = []
    gap_samples: list[float] = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        md, ma, mo = float(np.mean(d[idx])), float(np.mean(a[idx])), float(np.mean(o[idx]))
        boot_d.append(md)
        boot_a.append(ma)
        boot_o.append(mo)
        pen_samples.append(ma - md)
        gap_samples.append(mo - md)  # 1-hop minus dense; > 0 means dense is better

    def _ci(samples: list[float]) -> tuple[float, float]:
        lo, hi = np.percentile(samples, [2.5, 97.5])
        return float(lo), float(hi)

    pen_lo, pen_hi = _ci(pen_samples)
    dense_ci = _ci(boot_d)
    ablated_ci = _ci(boot_a)
    onehop_ci = _ci(boot_o)
    gap_lo, gap_hi = _ci(gap_samples)
    dense_onehop_order_significant = bool(gap_lo > 0.0 or gap_hi < 0.0)
    reg_mask = np.isfinite(rnc) & np.isfinite(effect)
    slope = corr = float("nan")
    if int(reg_mask.sum()) >= 2 and float(np.std(rnc[reg_mask])) > EPS:
        slope = float(np.polyfit(rnc[reg_mask], effect[reg_mask], 1)[0])
        corr = float(np.corrcoef(rnc[reg_mask], effect[reg_mask])[0, 1])
    recovers = float(ablation_penalty / onehop_penalty) if abs(onehop_penalty) > EPS else float("nan")
    if pen_lo <= 0.0:
        verdict = "ablation_penalty_ci_includes_zero_not_load_bearing"
    elif math.isfinite(recovers) and recovers >= 0.5:
        verdict = "ablation_recovers_onehop_gap_noncomposable_transport_is_causal"
    else:
        verdict = "ablation_hurts_but_below_onehop_gap_partial_contribution"
    return {
        "n": n,
        "mean_dense_error": float(np.mean(d)),
        "mean_dense_error_ci_low": dense_ci[0],
        "mean_dense_error_ci_high": dense_ci[1],
        "mean_dense_ablated_error": float(np.mean(a)),
        "mean_dense_ablated_error_ci_low": ablated_ci[0],
        "mean_dense_ablated_error_ci_high": ablated_ci[1],
        "mean_onehop_error": float(np.mean(o)),
        "mean_onehop_error_ci_low": onehop_ci[0],
        "mean_onehop_error_ci_high": onehop_ci[1],
        "onehop_minus_dense_error_mean": float(np.mean(o) - np.mean(d)),
        "onehop_minus_dense_error_ci_low": gap_lo,
        "onehop_minus_dense_error_ci_high": gap_hi,
        "dense_onehop_order_significant": dense_onehop_order_significant,
        "ablation_penalty": ablation_penalty,
        "ablation_penalty_ci_low": pen_lo,
        "ablation_penalty_ci_high": pen_hi,
        "onehop_penalty": onehop_penalty,
        "ablation_recovers_onehop_fraction": recovers,
        "ablation_effect_vs_rnc_slope": slope,
        "ablation_effect_vs_rnc_pearson": corr,
        "verdict": verdict,
    }


def render_step5_self_ablation(summary: Mapping[str, Any], gap_rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    if int(summary.get("n", 0)) <= 0:
        return
    figures = ensure_dir(artifact_root / "figures")
    labels = ["dense", "dense\n(shortcuts ablated)", "1-hop (trained)"]
    means = [safe_float(summary.get("mean_dense_error")), safe_float(summary.get("mean_dense_ablated_error")), safe_float(summary.get("mean_onehop_error"))]
    ci_low = [safe_float(summary.get("mean_dense_error_ci_low")), safe_float(summary.get("mean_dense_ablated_error_ci_low")), safe_float(summary.get("mean_onehop_error_ci_low"))]
    ci_high = [safe_float(summary.get("mean_dense_error_ci_high")), safe_float(summary.get("mean_dense_ablated_error_ci_high")), safe_float(summary.get("mean_onehop_error_ci_high"))]
    yerr_lo = [max(0.0, m - lo) if math.isfinite(lo) else 0.0 for m, lo in zip(means, ci_low)]
    yerr_hi = [max(0.0, hi - m) if math.isfinite(hi) else 0.0 for m, hi in zip(means, ci_high)]
    n = int(summary.get("n", 0))
    gap_mean = safe_float(summary.get("onehop_minus_dense_error_mean"))
    if bool(summary.get("dense_onehop_order_significant", False)):
        order_note = "dense < 1-hop (95% CI)" if gap_mean > 0 else "1-hop < dense (95% CI)"
    else:
        order_note = "dense vs 1-hop NOT separable (95% CI overlap)"
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    ax.bar(
        labels,
        means,
        yerr=[yerr_lo, yerr_hi],
        capsize=5,
        color=["#4c78a8", "#f58518", "#54a24b"],
        error_kw={"ecolor": "#333333", "elinewidth": 1.2},
    )
    ax.axhline(safe_float(summary.get("mean_dense_error")), color="#555555", linestyle="--", linewidth=1, label="dense baseline")
    ax.set_title(
        "Self-ablation: test error with non-composable transport removed\n"
        f"n={n} molecules (bootstrap 95% CI); {order_note}",
        fontsize=10,
    )
    ax.set_ylabel("Mean |prediction - target| (test)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(figures / "step5_self_ablation_error.png", dpi=dpi)
    fig.savefig(figures / "step5_self_ablation_error.pdf")
    plt.close(fig)

    x = np.asarray([safe_float(r.get("r_nc")) for r in gap_rows], dtype=float)
    yv = np.asarray([safe_float(r.get("ablation_minus_dense_error")) for r in gap_rows], dtype=float)
    m = np.isfinite(x) & np.isfinite(yv)
    if int(m.sum()) >= 2:
        fig, ax = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)
        ax.scatter(x[m], yv[m], s=18, alpha=0.65)
        if float(np.std(x[m])) > EPS:
            coef = np.polyfit(x[m], yv[m], 1)
            xs = np.linspace(float(x[m].min()), float(x[m].max()), 100)
            ax.plot(xs, coef[0] * xs + coef[1], color="#f58518", label=f"pearson={safe_float(summary.get('ablation_effect_vs_rnc_pearson')):.2f}")
            ax.legend(frameon=False)
        ax.axhline(0.0, color="#555555", linewidth=1)
        ax.set_title("Self-ablation penalty vs non-composable carriage")
        ax.set_xlabel("R_nc (per molecule)")
        ax.set_ylabel("Ablated error - dense error")
        fig.savefig(figures / "step5_self_ablation_vs_rnc.png", dpi=dpi)
        fig.savefig(figures / "step5_self_ablation_vs_rnc.pdf")
        plt.close(fig)


def _model_is_grit_family(model: str) -> bool:
    text = str(model).lower()
    return "grit" in text


def _apply_grit_focused_ylim(
    ax: Any,
    *,
    focus_values: Sequence[float],
    all_values: Sequence[float],
    positive_floor_zero: bool = False,
    pad_fraction: float = 0.14,
) -> None:
    """Keep plot scale readable for GRIT-family comparisons while allowing GIN to clip."""

    focus = np.asarray([safe_float(v) for v in focus_values], dtype=float)
    focus = focus[np.isfinite(focus)]
    all_arr = np.asarray([safe_float(v) for v in all_values], dtype=float)
    all_arr = all_arr[np.isfinite(all_arr)]
    if focus.size == 0:
        return
    lo = float(np.nanmin(focus))
    hi = float(np.nanmax(focus))
    if positive_floor_zero:
        lo = min(0.0, lo)
    else:
        lo = min(0.0, lo)
        hi = max(0.0, hi)
    span = max(hi - lo, abs(hi), abs(lo), 1.0e-6)
    lo_plot = lo - pad_fraction * span
    hi_plot = hi + pad_fraction * span
    if positive_floor_zero:
        lo_plot = min(0.0, lo_plot)
    if not (math.isfinite(lo_plot) and math.isfinite(hi_plot) and hi_plot > lo_plot):
        return
    ax.set_ylim(lo_plot, hi_plot)
    clipped_high = bool(all_arr.size and np.nanmax(all_arr) > hi_plot)
    clipped_low = bool(all_arr.size and np.nanmin(all_arr) < lo_plot)
    if clipped_high:
        ax.text(
            0.985,
            0.965,
            "non-GRIT clipped above axis",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#555555",
        )
    if clipped_low:
        ax.text(
            0.985,
            0.035,
            "non-GRIT clipped below axis",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )


def _ordered_models_for_rows(rows: Sequence[Mapping[str, Any]], *, grit_only: bool = False) -> list[str]:
    models = {str(r.get("model")) for r in rows if str(r.get("model")) not in {"", "None"}}
    if grit_only:
        models = {model for model in models if _model_is_grit_family(model)}
    return sorted(
        models,
        key=lambda name: (MODEL_PLOT_ORDER.index(name) if name in MODEL_PLOT_ORDER else len(MODEL_PLOT_ORDER), name),
    )


def _render_step5_load_bearing_main(
    total_summary: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    *,
    dpi: int,
) -> None:
    """Clean headline panel for cross-model load-bearing long-range carriage.

    The two-panel diagnostic figure is still written by
    ``render_step5_load_bearing_carriage_ablation``. This figure keeps only the
    interpretable cross-model test: removing total far carriage ranked by |C|,
    with random far-pair removal as the ERASER-style control.
    """

    rows = [
        r
        for r in total_summary
        if str(r.get("ranker")) in {"carriage", "random_far_pairs"}
        and _model_is_grit_family(str(r.get("model")))
        and math.isfinite(safe_float(r.get("fraction_removed")))
        and math.isfinite(safe_float(r.get("mean_delta_mae")))
    ]
    if not rows:
        return
    figures = ensure_dir(artifact_root / "figures")
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    all_values: list[float] = []
    focus_values: list[float] = []
    for model in _ordered_models_for_rows(rows, grit_only=True):
        for ranker, linestyle, linewidth, marker in (
            ("carriage", "-", 2.2, "o"),
            ("random_far_pairs", "--", 1.6, "s"),
        ):
            sub = sorted(
                [r for r in rows if str(r.get("model")) == model and str(r.get("ranker")) == ranker],
                key=lambda r: safe_float(r.get("fraction_removed")),
            )
            if not sub:
                continue
            x = np.asarray([safe_float(r.get("fraction_removed")) for r in sub], dtype=float)
            y = np.asarray([safe_float(r.get("mean_delta_mae")) for r in sub], dtype=float)
            lo = np.asarray([safe_float(r.get("ci_low")) for r in sub], dtype=float)
            hi = np.asarray([safe_float(r.get("ci_high")) for r in sub], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if not mask.any():
                continue
            color = MODEL_PALETTE.get(model)
            label_suffix = "|C|-ranked" if ranker == "carriage" else "random"
            ax.plot(
                x[mask],
                y[mask],
                marker=marker,
                linestyle=linestyle,
                linewidth=linewidth,
                color=color,
                label=f"{model_label(model)} {label_suffix}",
            )
            values = [float(v) for v in list(y[mask]) + list(lo[mask]) + list(hi[mask]) if math.isfinite(float(v))]
            all_values.extend(values)
            focus_values.extend(values)
            ci_mask = mask & np.isfinite(lo) & np.isfinite(hi)
            if ci_mask.any() and ranker == "carriage":
                ax.fill_between(x[ci_mask], lo[ci_mask], hi[ci_mask], color=color, alpha=0.13, linewidth=0)
    ax.axhline(0.0, color="#555555", linewidth=1)
    ax.set_title("Load-bearing far carriage: ranked IG-completeness ablation")
    ax.set_xlabel("Fraction of far pairs removed (d > τ)")
    ax.set_ylabel("Estimated Δ test MAE (positive = worse)")
    _apply_grit_focused_ylim(ax, focus_values=focus_values, all_values=all_values)
    ax.legend(frameon=False, fontsize=8, ncol=1)
    ax.text(
        0.01,
        -0.20,
        "Solid lines remove the largest |C[i,j]| far pairs first; dashed lines remove random far pairs.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#555555",
    )
    fig.savefig(figures / "step5_load_bearing_carriage_ablation_main.png", dpi=dpi)
    fig.savefig(figures / "step5_load_bearing_carriage_ablation_main.pdf")
    plt.close(fig)


def _render_step5_distance_binned_main(
    summary_rows: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    *,
    dpi: int,
) -> None:
    rows = [
        r
        for r in summary_rows
        if _model_is_grit_family(str(r.get("model")))
        and math.isfinite(safe_float(r.get("distance_bin_index")))
        and math.isfinite(safe_float(r.get("mean_delta_mae")))
    ]
    if not rows:
        return
    figures = ensure_dir(artifact_root / "figures")
    bin_rows = sorted(
        {
            (int(safe_float(r.get("distance_bin_index"))), str(r.get("distance_bin")))
            for r in rows
        },
        key=lambda item: item[0],
    )
    x = np.arange(len(bin_rows), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)
    effect_values: list[float] = []
    mass_values: list[float] = []
    for model in _ordered_models_for_rows(rows, grit_only=True):
        model_rows = {int(safe_float(r.get("distance_bin_index"))): r for r in rows if str(r.get("model")) == model}
        y = np.asarray([safe_float(model_rows.get(idx, {}).get("mean_delta_mae")) for idx, _ in bin_rows], dtype=float)
        lo = np.asarray([safe_float(model_rows.get(idx, {}).get("ci_low")) for idx, _ in bin_rows], dtype=float)
        hi = np.asarray([safe_float(model_rows.get(idx, {}).get("ci_high")) for idx, _ in bin_rows], dtype=float)
        mass = np.asarray([safe_float(model_rows.get(idx, {}).get("mean_abs_removed_carriage")) for idx, _ in bin_rows], dtype=float)
        color = MODEL_PALETTE.get(model)
        mask = np.isfinite(y)
        if mask.any():
            axes[0].plot(x[mask], y[mask], marker="o", linewidth=2.0, color=color, label=model_label(model))
            effect_values.extend([float(v) for v in list(y[mask]) + list(lo[mask]) + list(hi[mask]) if math.isfinite(float(v))])
            ci_mask = mask & np.isfinite(lo) & np.isfinite(hi)
            if ci_mask.any():
                axes[0].fill_between(x[ci_mask], lo[ci_mask], hi[ci_mask], color=color, alpha=0.13, linewidth=0)
        mass_mask = np.isfinite(mass)
        if mass_mask.any():
            axes[1].plot(x[mass_mask], mass[mass_mask], marker="o", linewidth=2.0, color=color, label=model_label(model))
            mass_values.extend([float(v) for v in mass[mass_mask] if math.isfinite(float(v))])
    labels = [label for _, label in bin_rows]
    axes[0].axhline(0.0, color="#555555", linewidth=1)
    axes[0].set_title("Prediction penalty by distance band")
    axes[0].set_xlabel("Molecular hop distance bin")
    axes[0].set_ylabel("Estimated Δ test MAE after removing band")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    _apply_grit_focused_ylim(axes[0], focus_values=effect_values, all_values=effect_values)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_title("How much carriage was removed")
    axes[1].set_xlabel("Molecular hop distance bin")
    axes[1].set_ylabel("Mean Σ|C[i,j]| removed")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    _apply_grit_focused_ylim(axes[1], focus_values=mass_values, all_values=mass_values, positive_floor_zero=True)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Distance-binned load-bearing carriage in the GRIT family")
    fig.text(
        0.02,
        0.01,
        "d=2-3 is near/mid-range; dissertation long-range claims should focus on d>=4.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.savefig(figures / "step5_distance_binned_total_carriage_ablation_grit_only.png", dpi=dpi)
    fig.savefig(figures / "step5_distance_binned_total_carriage_ablation_grit_only.pdf")
    plt.close(fig)


def render_step5_load_bearing_carriage_ablation(
    total_summary: Sequence[Mapping[str, Any]],
    component_summary: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    *,
    dpi: int,
) -> None:
    if not total_summary and not component_summary:
        return
    figures = ensure_dir(artifact_root / "figures")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    ax = axes[0]
    model_order = MODEL_PLOT_ORDER
    present_models = sorted(
        {str(r.get("model")) for r in total_summary},
        key=lambda name: (model_order.index(name) if name in model_order else len(model_order), name),
    )
    palette = MODEL_PALETTE
    ranker_labels = {"carriage": "carriage-ranked", "random_far_pairs": "random far pairs"}
    panel_a_all_values: list[float] = []
    panel_a_focus_values: list[float] = []
    for model in present_models:
        for ranker, linestyle in [("carriage", "-"), ("random_far_pairs", "--")]:
            rows = [
                r
                for r in total_summary
                if str(r.get("model")) == model and str(r.get("ranker")) == ranker
            ]
            rows = sorted(rows, key=lambda r: safe_float(r.get("fraction_removed")))
            if not rows:
                continue
            x = np.asarray([safe_float(r.get("fraction_removed")) for r in rows], dtype=float)
            y = np.asarray([safe_float(r.get("mean_delta_mae")) for r in rows], dtype=float)
            lo = np.asarray([safe_float(r.get("ci_low")) for r in rows], dtype=float)
            hi = np.asarray([safe_float(r.get("ci_high")) for r in rows], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if not mask.any():
                continue
            label = f"{model} {ranker_labels.get(ranker, ranker)}"
            color = palette.get(model)
            ax.plot(x[mask], y[mask], marker="o", linewidth=2.0 if ranker == "carriage" else 1.4, linestyle=linestyle, color=color, label=label)
            panel_a_all_values.extend([float(v) for v in y[mask] if math.isfinite(float(v))])
            panel_a_all_values.extend([float(v) for v in lo[mask] if math.isfinite(float(v))])
            panel_a_all_values.extend([float(v) for v in hi[mask] if math.isfinite(float(v))])
            if _model_is_grit_family(model):
                panel_a_focus_values.extend([float(v) for v in y[mask] if math.isfinite(float(v))])
                panel_a_focus_values.extend([float(v) for v in lo[mask] if math.isfinite(float(v))])
                panel_a_focus_values.extend([float(v) for v in hi[mask] if math.isfinite(float(v))])
            ci_mask = mask & np.isfinite(lo) & np.isfinite(hi)
            if ci_mask.any() and ranker == "carriage":
                ax.fill_between(x[ci_mask], lo[ci_mask], hi[ci_mask], color=color, alpha=0.14, linewidth=0)
    ax.axhline(0.0, color="#555555", linewidth=1)
    ax.set_title("Panel A: total long-range carriage removal")
    ax.set_xlabel("Fraction of far pairs removed (d > τ)")
    ax.set_ylabel("Δ test MAE (positive = worse)")
    _apply_grit_focused_ylim(ax, focus_values=panel_a_focus_values, all_values=panel_a_all_values)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=7, ncol=1)

    ax = axes[1]
    endpoint = [
        r
        for r in component_summary
        if abs(safe_float(r.get("fraction_removed")) - 1.0) < 1.0e-9
        and str(r.get("component")) in {"composable_carriage", "non_composable_carriage"}
    ]
    if endpoint:
        models = sorted(
            {str(r.get("model")) for r in endpoint},
            key=lambda name: (model_order.index(name) if name in model_order else len(model_order), name),
        )
        components = ["composable_carriage", "non_composable_carriage"]
        component_labels = {
            "composable_carriage": "composable\n(C - C^clamp)",
            "non_composable_carriage": "non-composable\n(C^clamp)",
        }
        x = np.arange(len(models), dtype=float)
        width = 0.36
        panel_b_all_values: list[float] = []
        panel_b_focus_values: list[float] = []
        for offset, component in zip([-width / 2, width / 2], components):
            rows = [next((r for r in endpoint if str(r.get("model")) == model and str(r.get("component")) == component), None) for model in models]
            means = np.asarray([safe_float(r.get("mean_delta_mae")) if r is not None else float("nan") for r in rows], dtype=float)
            lows = np.asarray([safe_float(r.get("ci_low")) if r is not None else float("nan") for r in rows], dtype=float)
            highs = np.asarray([safe_float(r.get("ci_high")) if r is not None else float("nan") for r in rows], dtype=float)
            mask = np.isfinite(means)
            yerr = np.vstack([np.maximum(0.0, means - lows), np.maximum(0.0, highs - means)])
            ax.bar(x[mask] + offset, means[mask], width=width, yerr=yerr[:, mask], capsize=3, label=component_labels[component])
            panel_b_all_values.extend([float(v) for v in means[mask] if math.isfinite(float(v))])
            panel_b_all_values.extend([float(v) for v in lows[mask] if math.isfinite(float(v))])
            panel_b_all_values.extend([float(v) for v in highs[mask] if math.isfinite(float(v))])
            for model_name, mean, low, high in zip(models, means, lows, highs):
                if _model_is_grit_family(model_name):
                    for value in (mean, low, high):
                        if math.isfinite(float(value)):
                            panel_b_focus_values.append(float(value))
        ax.axhline(0.0, color="#555555", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=18, ha="right")
        ax.set_ylabel("Δ test MAE at all measured far pairs")
        ax.set_title("Panel B: composable vs non-composable component")
        _apply_grit_focused_ylim(ax, focus_values=panel_b_focus_values, all_values=panel_b_all_values)
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            "No detach-clamped component rows were available.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.40,
            "Panel B needs measured C^clamp values from Step 5 patching.",
            ha="center",
            va="center",
            color="#555555",
            fontsize=9,
            transform=ax.transAxes,
        )
    fig.suptitle("Step 5: load-bearing long-range carriage by IG completeness ablation")
    fig.savefig(figures / "step5_load_bearing_carriage_ablation.png", dpi=dpi)
    fig.savefig(figures / "step5_load_bearing_carriage_ablation.pdf")
    plt.close(fig)
    _render_step5_load_bearing_main(total_summary, artifact_root, dpi=dpi)


def render_step5_distance_binned_total_carriage_ablation(
    summary_rows: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    *,
    dpi: int,
) -> None:
    if not summary_rows:
        return
    rows = [
        r
        for r in summary_rows
        if math.isfinite(safe_float(r.get("distance_bin_index")))
        and math.isfinite(safe_float(r.get("mean_delta_mae")))
    ]
    if not rows:
        return
    model_order = MODEL_PLOT_ORDER
    models = sorted(
        {str(r.get("model")) for r in rows},
        key=lambda name: (model_order.index(name) if name in model_order else len(model_order), name),
    )
    bin_rows = sorted(
        {
            (int(safe_float(r.get("distance_bin_index"))), str(r.get("distance_bin")))
            for r in rows
        },
        key=lambda item: item[0],
    )
    bin_labels = [label for _, label in bin_rows]
    x = np.arange(len(bin_rows), dtype=float)
    palette = MODEL_PALETTE
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    effect_all_values: list[float] = []
    effect_focus_values: list[float] = []
    mass_all_values: list[float] = []
    mass_focus_values: list[float] = []
    for model in models:
        model_rows = {int(safe_float(r.get("distance_bin_index"))): r for r in rows if str(r.get("model")) == model}
        means = np.asarray([safe_float(model_rows.get(idx, {}).get("mean_delta_mae")) for idx, _ in bin_rows], dtype=float)
        lows = np.asarray([safe_float(model_rows.get(idx, {}).get("ci_low")) for idx, _ in bin_rows], dtype=float)
        highs = np.asarray([safe_float(model_rows.get(idx, {}).get("ci_high")) for idx, _ in bin_rows], dtype=float)
        finite = np.isfinite(means)
        if finite.any():
            color = palette.get(model)
            axes[0].plot(x[finite], means[finite], marker="o", linewidth=2.0, label=model_label(model), color=color)
            effect_all_values.extend([float(v) for v in means[finite] if math.isfinite(float(v))])
            effect_all_values.extend([float(v) for v in lows[finite] if math.isfinite(float(v))])
            effect_all_values.extend([float(v) for v in highs[finite] if math.isfinite(float(v))])
            if _model_is_grit_family(model):
                effect_focus_values.extend([float(v) for v in means[finite] if math.isfinite(float(v))])
                effect_focus_values.extend([float(v) for v in lows[finite] if math.isfinite(float(v))])
                effect_focus_values.extend([float(v) for v in highs[finite] if math.isfinite(float(v))])
            ci = finite & np.isfinite(lows) & np.isfinite(highs)
            if ci.any():
                axes[0].fill_between(x[ci], lows[ci], highs[ci], color=color, alpha=0.13, linewidth=0)
        abs_mass = np.asarray([safe_float(model_rows.get(idx, {}).get("mean_abs_removed_carriage")) for idx, _ in bin_rows], dtype=float)
        finite_mass = np.isfinite(abs_mass)
        if finite_mass.any():
            axes[1].plot(x[finite_mass], abs_mass[finite_mass], marker="o", linewidth=2.0, label=model_label(model), color=palette.get(model))
            mass_all_values.extend([float(v) for v in abs_mass[finite_mass] if math.isfinite(float(v))])
            if _model_is_grit_family(model):
                mass_focus_values.extend([float(v) for v in abs_mass[finite_mass] if math.isfinite(float(v))])
    axes[0].axhline(0.0, color="#555555", linewidth=1)
    axes[0].set_title("Prediction effect by distance band")
    axes[0].set_xlabel("Molecular hop distance bin")
    axes[0].set_ylabel("Δ test MAE after removing bin (positive = worse)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bin_labels, rotation=15, ha="right")
    _apply_grit_focused_ylim(axes[0], focus_values=effect_focus_values, all_values=effect_all_values)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_title("Total absolute carriage removed")
    axes[1].set_xlabel("Molecular hop distance bin")
    axes[1].set_ylabel("Mean Σ|C[i,j]| removed (prediction units)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bin_labels, rotation=15, ha="right")
    _apply_grit_focused_ylim(axes[1], focus_values=mass_focus_values, all_values=mass_all_values, positive_floor_zero=True)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Step 5: distance-binned total-carriage ablation")
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step5_distance_binned_total_carriage_ablation.png", dpi=dpi)
    fig.savefig(figures / "step5_distance_binned_total_carriage_ablation.pdf")
    plt.close(fig)
    _render_step5_distance_binned_main(summary_rows, artifact_root, dpi=dpi)


def render_step5_dense_control_excess_distance_ablation(
    summary_rows: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    *,
    dpi: int,
) -> None:
    if not summary_rows:
        return
    rows = [
        r
        for r in summary_rows
        if math.isfinite(safe_float(r.get("distance_bin_index")))
        and math.isfinite(safe_float(r.get("mean_delta_mae")))
    ]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: int(safe_float(r.get("distance_bin_index"))))
    labels = [str(r.get("distance_bin")) for r in rows]
    x = np.arange(len(rows), dtype=float)
    means = np.asarray([safe_float(r.get("mean_delta_mae")) for r in rows], dtype=float)
    lows = np.asarray([safe_float(r.get("ci_low")) for r in rows], dtype=float)
    highs = np.asarray([safe_float(r.get("ci_high")) for r in rows], dtype=float)
    abs_excess = np.asarray([safe_float(r.get("mean_abs_dense_minus_control_carriage")) for r in rows], dtype=float)
    pair_counts = np.asarray([safe_float(r.get("mean_pair_count")) for r in rows], dtype=float)
    control = str(rows[0].get("control_model", "grit_1hop"))
    dense_model = str(rows[0].get("dense_model", "dense_grit"))

    figures = ensure_dir(artifact_root / "figures")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), constrained_layout=True)
    ax = axes[0]
    yerr = np.vstack([np.maximum(0.0, means - lows), np.maximum(0.0, highs - means)])
    ax.bar(x, means, yerr=yerr, capsize=4, color="#4c78a8")
    ax.axhline(0.0, color="#555555", linewidth=1)
    ax.set_title("MAE effect of removing dense-only carriage")
    ax.set_xlabel("Molecular hop distance bin")
    ax.set_ylabel("Δ dense test MAE (positive = worse)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    for xi, yi, n_pairs in zip(x, means, pair_counts):
        if math.isfinite(float(n_pairs)):
            ax.text(
                float(xi),
                float(yi) if math.isfinite(float(yi)) else 0.0,
                f"pairs≈{n_pairs:.0f}",
                ha="center",
                va="bottom" if safe_float(yi) >= 0 else "top",
                fontsize=7,
                color="#444444",
            )

    ax = axes[1]
    ax.bar(x, abs_excess, color="#f58518")
    ax.set_title("Magnitude of dense-minus-control carriage removed")
    ax.set_xlabel("Molecular hop distance bin")
    ax.set_ylabel("Mean Σ|C_dense - C_control| (prediction units)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(bottom=0.0)

    fig.suptitle(
        f"Step 5: distance-localised dense advantage ({dense_model} minus {control})",
        fontsize=14,
    )
    fig.savefig(figures / "step5_dense_minus_control_distance_ablation.png", dpi=dpi)
    fig.savefig(figures / "step5_dense_minus_control_distance_ablation.pdf")
    # Alias with the default control in the filename for easier notebook discovery.
    if control == "grit_1hop":
        fig.savefig(figures / "step5_dense_minus_1hop_distance_ablation.png", dpi=dpi)
        fig.savefig(figures / "step5_dense_minus_1hop_distance_ablation.pdf")
    plt.close(fig)


def render_step5_gap_structural_severity(gap_rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    if not gap_rows:
        return
    diameter = np.asarray([safe_float(r.get("graph_diameter")) for r in gap_rows], dtype=float)
    gin_minus_onehop = np.asarray([safe_float(r.get("gin_minus_onehop_error")) for r in gap_rows], dtype=float)
    r_nc = np.asarray([safe_float(r.get("r_nc")) for r in gap_rows], dtype=float)
    onehop_minus_dense = np.asarray([safe_float(r.get("onehop_minus_dense_error")) for r in gap_rows], dtype=float)

    figures = ensure_dir(artifact_root / "figures")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)

    def scatter_with_fit(ax: Any, x: np.ndarray, y: np.ndarray, *, xlabel: str, ylabel: str, title: str) -> None:
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            ax.text(0.5, 0.5, "Not available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            ax.set_title(title)
            return
        ax.scatter(x[mask], y[mask], s=22, alpha=0.68)
        ax.axhline(0.0, color="#777777", linewidth=1)
        if mask.sum() >= 2 and float(np.nanmax(x[mask]) - np.nanmin(x[mask])) > 0.0:
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
            coef = np.polyfit(x[mask], y[mask], 1)
            xs = np.linspace(float(np.nanmin(x[mask])), float(np.nanmax(x[mask])), 100)
            ax.plot(xs, coef[0] * xs + coef[1], color="#f58518", label=f"r={corr:.2f}")
            ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    scatter_with_fit(
        axes[0],
        diameter,
        gin_minus_onehop,
        xlabel="Graph diameter",
        ylabel="GIN error - 1-hop error",
        title="RRWP-tier advantage vs graph bottleneck",
    )
    scatter_with_fit(
        axes[1],
        r_nc,
        onehop_minus_dense,
        xlabel="R_nc = mean |C^clamp| over far pairs",
        ylabel="1-hop error - dense error",
        title="Dense advantage vs long-range demand",
    )
    fig.suptitle("Where each gap lives: RRWP-tier advantage vs graph bottleneck; dense advantage vs long-range demand")
    fig.savefig(figures / "step5_gap_structural_severity.png", dpi=dpi)
    fig.savefig(figures / "step5_gap_structural_severity.pdf")
    plt.close(fig)


def gap_regression_summary(gap_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    x = np.asarray([safe_float(r.get("r_nc")) for r in gap_rows], dtype=float)

    def regress(key: str) -> dict[str, Any]:
        y = np.asarray([safe_float(r.get(key)) for r in gap_rows], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return {"n": int(mask.sum()), "pearson_r": float("nan"), "slope": float("nan"), "intercept": float("nan")}
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
        return {"n": int(mask.sum()), "pearson_r": corr, "slope": float(slope), "intercept": float(intercept)}

    standard = regress("onehop_minus_dense_error")
    out = {
        "n": standard["n"],
        "pearson_r": standard["pearson_r"],
        "slope": standard["slope"],
        "intercept": standard["intercept"],
        "reference": "grit_1hop",
        "matched_1hop": standard,
    }
    local_key = "grit_1hop_localrrwp_minus_dense_error"
    if any(math.isfinite(safe_float(r.get(local_key))) for r in gap_rows):
        out["local_rrwp_1hop"] = regress(local_key)
    return out


def rank_summary_rows(
    rank_rows: Sequence[Mapping[str, Any]],
    *,
    min_sampled_pairs: int = 3,
    models: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    primary = [r for r in rank_rows if bool(r.get("primary_tau", True))]
    rows = primary or list(rank_rows)
    if models is not None:
        allowed = {str(model) for model in models}
        rows = [r for r in rows if str(r.get("model")) in allowed]
    out = []
    for model in sorted(set(str(r.get("model")) for r in rows)):
        model_rows = [
            r
            for r in rows
            if str(r.get("model")) == model
            and str(r.get("status", "")).startswith("complete")
            and int(safe_float(r.get("sampled_pairs")) if math.isfinite(safe_float(r.get("sampled_pairs"))) else 0) >= int(min_sampled_pairs)
        ]
        for metric in ["effective_rank", "top_singular_share", "above_null_margin"]:
            values = [safe_float(r.get(metric)) for r in model_rows]
            values = [v for v in values if math.isfinite(v)]
            if not values:
                continue
            mean, lo, hi = bootstrap_ci(values, seed=4100 + len(out), draws=1000)
            out.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": len(values),
                    "tau": model_rows[0].get("tau", "") if model_rows else "",
                    "min_sampled_pairs": int(min_sampled_pairs),
                }
            )
    return out


def non_additivity_summary_rows(
    interaction_rows: Sequence[Mapping[str, Any]],
    *,
    models: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    out = []
    allowed = {str(model) for model in models} if models is not None else None
    for model in sorted(set(str(r.get("model")) for r in interaction_rows)):
        if allowed is not None and model not in allowed:
            continue
        values = [safe_float(r.get("non_additivity")) for r in interaction_rows if str(r.get("model")) == model and row_is_nontrivial(r, default=False)]
        values = [v for v in values if math.isfinite(v)]
        if not values:
            continue
        mean, lo, hi = bootstrap_ci(values, seed=5100 + len(out), draws=1000)
        out.append({"model": model, "mean_non_additivity": mean, "ci_low": lo, "ci_high": hi, "pairs": len(values)})
    return out


def render_step5_signal_vs_noise(rank_rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> dict[str, Any]:
    """Standalone signal-vs-noise verdict for dense far carriage.

    Compares the observed far-carriage structure (top singular-value share) against
    a distance-preserving shuffle null (source labels permuted within each distance
    bin). If the observed structure exceeds the null with a CI that excludes zero,
    the far carriage carries resolvable signal above chance; if the margin CI
    includes zero, what is there is at the noise floor — "nothing above noise".
    """
    rows = [r for r in rank_rows if bool(r.get("primary_tau", True))] or list(rank_rows)
    pairs = [
        (safe_float(r.get("top_singular_share")), safe_float(r.get("above_null_margin")))
        for r in rows
        if math.isfinite(safe_float(r.get("top_singular_share"))) and math.isfinite(safe_float(r.get("above_null_margin")))
    ]
    if len(pairs) < 2:
        return {"status": "insufficient_far_carriage_rows", "n": len(pairs)}
    obs_v = [o for o, _ in pairs]
    null_v = [max(0.0, o - m) for o, m in pairs]  # null share = observed - margin
    marg_v = [m for _, m in pairs]
    o_mean, o_lo, o_hi = bootstrap_ci(obs_v, seed=8123, draws=1000)
    n_mean, n_lo, n_hi = bootstrap_ci(null_v, seed=8124, draws=1000)
    m_mean, m_lo, m_hi = bootstrap_ci(marg_v, seed=8125, draws=1000)
    signal = bool(math.isfinite(m_lo) and m_lo > 0.0)
    verdict = (
        "low-rank structure above distance-preserving null"
        if signal
        else "no low-rank structure above distance-preserving null"
    )
    figures = ensure_dir(artifact_root / "figures")
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    labels = ["observed\nfar carriage", "distance-preserving\nshuffle null"]
    means = [o_mean, n_mean]
    errs = np.vstack(
        [
            [max(0.0, o_mean - o_lo), max(0.0, n_mean - n_lo)],
            [max(0.0, o_hi - o_mean), max(0.0, n_hi - n_mean)],
        ]
    )
    ax.bar(labels, means, yerr=errs, capsize=4, color=["#4c78a8", "#999999"])
    ax.set_ylabel("Top singular-value share (structure)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Dense far carriage structure vs shuffle null\nhigh-rank ≠ noise; see self-ablation — {verdict}", fontsize=10)
    ax.text(
        0.5,
        min(1.02, max(means) + 0.08),
        f"above-null margin = {m_mean:.3f} [{m_lo:.3f}, {m_hi:.3f}]  (n={len(pairs)} graphs)",
        ha="center",
        fontsize=9,
        transform=ax.get_xaxis_transform(),
    )
    fig.savefig(figures / "step5_signal_vs_noise.png", dpi=dpi)
    fig.savefig(figures / "step5_signal_vs_noise.pdf")
    plt.close(fig)
    return {
        "status": "computed",
        "n_graphs": len(pairs),
        "observed_top_share_mean": o_mean,
        "null_top_share_mean": n_mean,
        "above_null_margin_mean": m_mean,
        "above_null_margin_ci_low": m_lo,
        "above_null_margin_ci_high": m_hi,
        "signal_above_noise": signal,
        "verdict": verdict,
    }


def render_step5(
    gap_rows: Sequence[Mapping[str, Any]],
    rank_rows: Sequence[Mapping[str, Any]],
    interaction_rows: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    *,
    dpi: int,
) -> None:
    figures = ensure_dir(artifact_root / "figures")
    primary_rank_rows = [r for r in rank_rows if bool(r.get("primary_tau", True))]
    if primary_rank_rows:
        rank_rows = primary_rank_rows
    if gap_rows:
        x = np.asarray([safe_float(r["r_nc"]) for r in gap_rows], dtype=float)
        series = [
            ("1-hop global RRWP - dense", np.asarray([safe_float(r["onehop_minus_dense_error"]) for r in gap_rows], dtype=float), "#4c78a8"),
        ]
        local_y = np.asarray([safe_float(r.get("grit_1hop_localrrwp_minus_dense_error")) for r in gap_rows], dtype=float)
        if np.isfinite(local_y).any():
            series.append(("1-hop local PE - dense", local_y, "#54a24b"))
        if any((np.isfinite(x) & np.isfinite(y)).any() for _, y, _ in series):
            fig, ax = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)
            for label, y, color in series:
                mask = np.isfinite(x) & np.isfinite(y)
                if not mask.any():
                    continue
                ax.scatter(x[mask], y[mask], s=18, alpha=0.65, color=color, label=label)
                corr = float(np.corrcoef(x[mask], y[mask])[0, 1]) if mask.sum() >= 2 else float("nan")
                if mask.sum() >= 2:
                    coef = np.polyfit(x[mask], y[mask], 1)
                    xs = np.linspace(float(x[mask].min()), float(x[mask].max()), 100)
                    ax.plot(xs, coef[0] * xs + coef[1], color=color, linestyle="--", linewidth=1.0, label=f"{label} fit r={corr:.2f}")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=False)
            ax.set_title("Step 5: performance gap vs non-composable carriage")
            ax.set_xlabel("R_nc = mean |C^clamp| over signal-gated far pairs")
            ax.set_ylabel("Reference error - dense error")
            fig.savefig(figures / "step5_gap_vs_rnc.png", dpi=dpi)
            fig.savefig(figures / "step5_gap_vs_rnc.pdf")
            plt.close(fig)
    if rank_rows:
        summary = rank_summary_rows(rank_rows, models=["dense_grit"])
        if summary:
            metric_labels = {
                "effective_rank": ("Effective rank", "Effective rank (count)"),
                "top_singular_share": ("Top singular share", "Share (0-1)"),
                "above_null_margin": ("Above-null margin", "Share margin (0-1)"),
            }
            metrics = [m for m in ["effective_rank", "top_singular_share", "above_null_margin"] if any(str(r.get("metric")) == m for r in summary)]
            fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.4), constrained_layout=True)
            axes_arr = np.atleast_1d(axes)
            for ax, metric in zip(axes_arr, metrics):
                label, ylabel = metric_labels.get(metric, (metric.replace("_", " "), "Value"))
                metric_rows = [r for r in summary if str(r.get("metric")) == metric]
                models = [str(r.get("model")) for r in metric_rows]
                means = np.asarray([safe_float(r.get("mean")) for r in metric_rows], dtype=float)
                lows = np.asarray([safe_float(r.get("ci_low")) for r in metric_rows], dtype=float)
                highs = np.asarray([safe_float(r.get("ci_high")) for r in metric_rows], dtype=float)
                yerr = np.vstack([np.maximum(0.0, means - lows), np.maximum(0.0, highs - means)])
                ax.bar(models, means, yerr=yerr, capsize=4, color="#4c78a8")
                ax.axhline(0, color="#555555", linewidth=1)
                ax.set_title(label)
                ax.set_ylabel(ylabel)
                for tick in ax.get_xticklabels():
                    tick.set_rotation(18)
                    tick.set_ha("right")
            fig.suptitle("Step 5: dense structure of non-composable long-range carriage")
            fig.savefig(figures / "step5_structure_non_composable_carriage.png", dpi=dpi)
            fig.savefig(figures / "step5_structure_non_composable_carriage.pdf")
            plt.close(fig)
        else:
            dense_rows = [r for r in rank_rows if str(r.get("model")) == "dense_grit"]
            complete_rows = sum(1 for r in dense_rows if str(r.get("status", "")).startswith("complete"))
            fig, ax = plt.subplots(figsize=(8.0, 3.8), constrained_layout=True)
            ax.axis("off")
            ax.text(
                0.5,
                0.58,
                "Dense GRIT did not have enough signal-gated far pairs for a rank summary.",
                ha="center",
                va="center",
                fontsize=13,
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                0.38,
                f"Complete dense rows before pair-count gate: {complete_rows}. 1-hop/GIN are floor references, not rank targets.",
                ha="center",
                va="center",
                fontsize=10,
                color="#555555",
                transform=ax.transAxes,
            )
            fig.suptitle("Step 5: dense structure of non-composable long-range carriage")
            fig.savefig(figures / "step5_structure_non_composable_carriage.png", dpi=dpi)
            fig.savefig(figures / "step5_structure_non_composable_carriage.pdf")
            plt.close(fig)
    if interaction_rows:
        summary = non_additivity_summary_rows(interaction_rows, models=["dense_grit"])
        if summary:
            labels = [str(r["model"]) for r in summary]
            y = np.asarray([safe_float(r["mean_non_additivity"]) for r in summary], dtype=float)
            lo = np.asarray([safe_float(r["ci_low"]) for r in summary], dtype=float)
            hi = np.asarray([safe_float(r["ci_high"]) for r in summary], dtype=float)
            fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
            ax.bar(labels, y, yerr=np.vstack([np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)]), capsize=4, color="#f58518")
            ax.set_title("Step 5: dense distant-source non-additivity after signal gate")
            ax.set_ylabel("Non-additivity ratio")
            for tick in ax.get_xticklabels():
                tick.set_rotation(15)
                tick.set_ha("right")
            fig.savefig(figures / "step5_non_additivity.png", dpi=dpi)
            fig.savefig(figures / "step5_non_additivity.pdf")
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.4, 5.0), constrained_layout=True)
        scatter_rows = [r for r in interaction_rows if row_is_nontrivial(r)]
        sx_all: list[float] = []
        sy_all: list[float] = []
        for model in sorted(set(str(r.get("model")) for r in scatter_rows)):
            model_rows = [r for r in scatter_rows if str(r.get("model")) == model]
            sx = [safe_float(r["delta_a"]) + safe_float(r["delta_b"]) for r in model_rows]
            sy = [safe_float(r["delta_ab"]) for r in model_rows]
            sx_all.extend(sx)
            sy_all.extend(sy)
            ax.scatter(sx, sy, s=12, alpha=0.5, label=model)
        lim = max([abs(v) for v in sx_all + sy_all if math.isfinite(v)] + [1.0e-6])
        ax.plot([-lim, lim], [-lim, lim], "--", color="#555555")
        ax.set_title("Step 5: additivity of distant sources")
        ax.set_xlabel("delta_A + delta_B")
        ax.set_ylabel("delta_AB")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(figures / "step5_additivity_scatter.png", dpi=dpi)
        fig.savefig(figures / "step5_additivity_scatter.pdf")
        plt.close(fig)


def render_step5_rung_funnel(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    clean_rows = [
        r
        for r in rows
        if math.isfinite(safe_float(r.get("rung_index")))
    ]
    if not clean_rows:
        return
    rung_order = ["attention", "carriage", "direct_carriage"]
    rung_labels = {
        "attention": "attention",
        "carriage": "carriage",
        "direct_carriage": "direct carriage",
    }
    models = sorted({str(r.get("model")) for r in clean_rows})
    x = np.arange(len(rung_order), dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    all_means: list[float] = []
    for model in models:
        means = []
        for rung in rung_order:
            vals = [
                safe_float(r.get("far_mass"))
                for r in clean_rows
                if str(r.get("model")) == model and str(r.get("rung")) == rung
            ]
            vals = [v for v in vals if math.isfinite(v)]
            means.append(float(np.nanmean(vals)) if vals else float("nan"))
        all_means.extend([v for v in means if math.isfinite(v)])
        ax.plot(x, means, marker="o", linewidth=2.0, label=model)
    ax.set_title("From reading to non-composable transport: far-mass surviving each rung")
    ax.set_xlabel("Rung")
    ax.set_ylabel("Far-mass share (direct rung is observed and bounded)")
    ax.set_xticks(x)
    ax.set_xticklabels([rung_labels[r] for r in rung_order])
    upper = max(all_means + [1.0])
    ax.set_ylim(0, 1.05 if upper <= 1.05 else upper * 1.08)
    ax.legend(frameon=False, fontsize=8)
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step5_rung_funnel.png", dpi=dpi)
    fig.savefig(figures / "step5_rung_funnel.pdf")
    plt.close(fig)


def render_worked_molecule_example(example: Mapping[str, Any], artifact_root: Path, *, dpi: int) -> None:
    try:
        import networkx as nx
    except Exception:
        return
    edge_index = example.get("edge_index")
    attention = example.get("attention")
    carriage = example.get("carriage")
    direct = example.get("direct")
    if not isinstance(edge_index, torch.Tensor) or not isinstance(carriage, torch.Tensor):
        return
    n = int(carriage.size(0))
    carrier = int(example.get("carrier", 0))
    source = int(example.get("source", 0))
    cut = [int(v) for v in example.get("cut", [])]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for src, dst in edge_index.t().detach().cpu().long().tolist():
        graph.add_edge(int(src), int(dst))
    pos = nx.spring_layout(graph, seed=17)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), constrained_layout=True)

    ax = axes[0]
    if isinstance(attention, torch.Tensor):
        mat = attention.detach().cpu().float().numpy()
        im = ax.imshow(mat, cmap="magma", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="attention")
    else:
        ax.text(0.5, 0.5, "No attention map", ha="center", va="center")
    ax.set_title("Direct attention map")
    ax.set_xlabel("Source atom")
    ax.set_ylabel("Receiving atom")

    ax = axes[1]
    source_strength = carriage.detach().abs().cpu()[carrier].float().numpy()
    node_colors = [float(source_strength[idx]) for idx in range(n)]
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#999999", width=1.2)
    nodes = nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, cmap="viridis", node_size=260)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
    fig.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04, label=f"|C[{carrier}, source]|")
    ax.set_title(f"Carriage source-map into atom {carrier}")
    ax.set_axis_off()

    ax = axes[2]
    colors = []
    for node in range(n):
        if node == carrier:
            colors.append("#4c78a8")
        elif node == source:
            colors.append("#e45756")
        elif node in cut:
            colors.append("#f58518")
        else:
            colors.append("#dddddd")
    widths = []
    for u, v in graph.edges():
        on_pair_edge = {u, v}.issubset({carrier, source})
        widths.append(2.4 if on_pair_edge else 1.2)
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#999999", width=widths)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colors, edgecolors="#333333", linewidths=0.6, node_size=300)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
    direct_value = safe_float(example.get("direct_value"))
    direct_fraction = safe_float(example.get("direct_fraction"))
    distance = safe_float(example.get("distance"))
    ax.set_title(f"Far pair d={distance:.0f}, direct={direct_value:.3g}, fraction={direct_fraction:.2g}")
    ax.set_axis_off()

    fig.suptitle("A single molecule: attention, carriage, and a surviving long-range shortcut")
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step5_worked_molecule_shortcut.png", dpi=dpi)
    fig.savefig(figures / "step5_worked_molecule_shortcut.pdf")
    plt.close(fig)


def render_step5_vnode_decision(decision: Mapping[str, Any], artifact_root: Path, *, dpi: int) -> None:
    share_values = [
        safe_float(decision.get("mean_non_additivity")),
        safe_float(decision.get("mean_top_singular_share")),
        safe_float(decision.get("mean_distance_preserving_above_null_margin")),
    ]
    rank_value = safe_float(decision.get("mean_effective_rank"))
    if not any(math.isfinite(v) for v in share_values + [rank_value]):
        return
    labels = ["Non-additivity", "Top singular share", "Above-null margin"]
    if math.isfinite(rank_value):
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), constrained_layout=True)
        ax = axes[0]
        rank_ax = axes[1]
    else:
        fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
        rank_ax = None
    ax.bar(labels, [0.0 if not math.isfinite(v) else v for v in share_values], color=["#f58518", "#4c78a8", "#54a24b"])
    ax.axhline(safe_float(decision.get("non_additivity_threshold")), color="#555555", linestyle="--", linewidth=1, label="Non-additivity threshold")
    ax.set_title("Share and ratio summaries")
    ax.set_ylabel("Share / ratio / margin")
    ax.legend(frameon=False, fontsize=8)
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)
        tick.set_ha("right")
    if rank_ax is not None:
        rank_ax.bar(["Effective rank"], [rank_value], color="#4c78a8")
        rank_ax.set_title("Effective rank")
        rank_ax.set_ylabel("Effective rank (count)")
    fig.suptitle("Step 5: virtual-node sufficiency decision")
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step5_vnode_decision.png", dpi=dpi)
    fig.savefig(figures / "step5_vnode_decision.pdf")
    plt.close(fig)


class _GRITBeneficialBackend:
    """Adapts an official GRIT ModelRun to the beneficial_carriage.FBCBackend protocol.

    Resampling is at the encoded-content level (same unit as carriage): replace one node's encoded
    state with a donor's and forward through the frozen readout. Distances are cached per graph.
    """

    def __init__(self, model: ModelRun) -> None:
        self.adapter = model.adapter
        self._dist: dict[int, torch.Tensor] = {}

    def _dm(self, graph: Any) -> torch.Tensor:
        key = id(graph)
        if key not in self._dist:
            self._dist[key] = torch.as_tensor(distance_matrix(graph)).long()
        return self._dist[key]

    def encoded(self, graph: Any) -> torch.Tensor:
        return self.adapter.encoded_node_states(graph).detach()

    def predict(self, graph: Any, encoded: torch.Tensor) -> float:
        return float(predict_scalar_from_encoded(self.adapter, graph, encoded).detach().cpu().item())

    def label(self, graph: Any) -> float:
        y = getattr(graph, "y", None)
        if y is None:
            return float("nan")
        return float(torch.as_tensor(y).reshape(-1)[0].item())

    def distances(self, graph: Any) -> torch.Tensor:
        return self._dm(graph)

    def degree(self, graph: Any) -> torch.Tensor:
        return (self._dm(graph) == 1).sum(dim=1).float()

    @staticmethod
    def _atoms(graph: Any) -> Optional[torch.Tensor]:
        x = getattr(graph, "x", None)
        if x is None:
            return None
        xt = torch.as_tensor(x)
        return xt.reshape(int(xt.shape[0]), -1)[:, 0].long()

    def signature(self, graph: Any, node: int) -> tuple[int, tuple[int, ...]]:
        # ENVIRONMENT only (degree + neighbour-atom multiset), NOT the node's own atom -- matched
        # donors must share context while their content is free to vary, else the resample is a no-op.
        neighbours = (self._dm(graph)[node] == 1).nonzero(as_tuple=True)[0].tolist()
        atoms = self._atoms(graph)
        neigh_atoms = tuple(sorted(int(atoms[k].item()) for k in neighbours)) if atoms is not None else ()
        return (len(neighbours), neigh_atoms)

    def donor_token(self, graph: Any, node: int) -> int:
        atoms = self._atoms(graph)
        return int(atoms[node].item()) if atoms is not None else 0

    def apply_donor(self, graph: Any, encoded: torch.Tensor, source: int, donor_atom: int) -> torch.Tensor:
        # Symbolic-only resample: swap the source atom to a matched-environment donor's and re-encode.
        # Structure is untouched, so node-RRWP is unchanged -- only the source's atom embedding moves.
        clone = graph.clone()
        xt = torch.as_tensor(clone.x).clone()
        if xt.dim() == 1:
            xt[source] = int(donor_atom)
        else:
            xt[source, 0] = int(donor_atom)
        clone.x = xt
        re_enc = self.adapter.encoded_node_states(clone).detach().to(encoded)
        pert = encoded.clone()
        pert[source] = re_enc[source]
        return pert


def render_step6(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = artifact_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    models = sorted({str(r["model"]) for r in rows})

    def sel(model: str, split: str, mode: str, scope: str = "single_source") -> list[Mapping[str, Any]]:
        return sorted(
            [r for r in rows if r["model"] == model and r["split"] == split
             and r["resampler"] == mode and r.get("scope", "single_source") == scope],
            key=lambda r: r["distance"],
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    for model in models:
        pts = sel(model, "test", "matched")
        if not pts:
            continue
        d = [p["distance"] for p in pts]
        b = [p["B"] for p in pts]
        yerr = [[p["B"] - p["B_lo"] for p in pts], [p["B_hi"] - p["B"] for p in pts]]
        ax.errorbar(d, b, yerr=yerr, marker="o", capsize=3, label=model)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_title("Beneficial carriage B(d): test, matched resampler")
    ax.set_xlabel("distance from focal node (hops)")
    ax.set_ylabel("B(d): loss increase when true content is resampled")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    m0 = models[0]
    for mode, ls in (("matched", "-"), ("marginal", "--")):
        pts = sel(m0, "test", mode)
        if pts:
            ax.plot([p["distance"] for p in pts], [p["B"] for p in pts], ls, marker="s", label=f"B {mode}")
    ptsf = sel(m0, "test", "matched")
    if ptsf:
        ax.plot([p["distance"] for p in ptsf], [p["F"] for p in ptsf], ":", marker="^", color="gray", label="F (functional)")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_title(f"{m0}: functional F(d) vs beneficial B(d) (matched vs marginal = the artifact)")
    ax.set_xlabel("distance from focal node (hops)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle("Step 6: functional (F) vs beneficial (B) carriage by distance")
    fig.tight_layout()
    fig.savefig(figures / "step6_beneficial_carriage.png", dpi=dpi)
    fig.savefig(figures / "step6_beneficial_carriage.pdf")
    plt.close(fig)


def render_step6_ig(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = artifact_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    models = sorted({str(r["model"]) for r in rows})
    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 4.4), squeeze=False)
    for ax, model in zip(axes[0], models):
        pts = sorted([r for r in rows if r["model"] == model], key=lambda r: r["distance"])
        d = [p["distance"] for p in pts]
        l1 = ax.plot(d, [p["C_functional_share"] for p in pts], "o-", color="#1f77b4",
                     label="C(d) functional (|C| share)")
        ax.set_xlabel("transport distance d(i,j) (hops)")
        ax.set_ylabel("functional carriage share", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax.set_ylim(bottom=0)
        ax2 = ax.twinx()
        l2 = ax2.plot(d, [p["B_ig_loss_carriage"] for p in pts], "s--", color="#d62728",
                      label="B_IG(d) loss-carriage (signed)")
        ax2.axhline(0, color="k", lw=0.7, ls=":")
        ax2.set_ylabel("loss-carriage  (<0 = beneficial)", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax.set_title(str(model))
        ax.legend(l1 + l2, [ln.get_label() for ln in l1 + l2], loc="upper right", fontsize=8)
    fig.suptitle("Step 6 (IG-aligned): functional C(d) vs loss-carriage B_IG(d) by transport distance")
    fig.tight_layout()
    fig.savefig(figures / "step6_ig_loss_carriage.png", dpi=dpi)
    fig.savefig(figures / "step6_ig_loss_carriage.pdf")
    plt.close(fig)


def run_step6(models: Sequence[ModelRun], artifact_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Beneficial carriage B(d): does distance-d content help the task, not just move the output?"""
    from . import beneficial_carriage as _bc

    progress("Step 6 start: beneficial carriage B(d)")
    analysis_models = [m for m in models if "grit" in m.name.lower()] or list(models)
    if not analysis_models:
        return {"status": "skipped_no_models"}
    cfg = dict(config["steps"].get("6", {}))
    sample_graphs = int(cfg.get("sample_graphs", 60))
    donors = int(cfg.get("donors", 4))
    max_d = int(cfg.get("max_distance", 6))
    modes = list(cfg.get("resamplers", ["matched", "marginal"]))
    whole_band = bool(cfg.get("whole_band", False))
    splits = list(cfg.get("splits", ["test", "train"]))
    dpi = int(config["figures"]["dpi"])
    seed = int(config.get("seeds", [0])[0])
    rows: list[dict[str, Any]] = []
    for model in analysis_models:
        backend = _GRITBeneficialBackend(model)
        for split in splits:
            graphs = select_graphs(model.adapter, split, sample_graphs, seed=seed)
            progress(f"Step 6 {model.name} {split}: {len(graphs)} graphs, donors={donors}, resamplers={modes}")
            rows.extend(
                _bc.beneficial_carriage_rows(
                    backend, graphs, model=model.name, split=split, modes=modes,
                    donors=donors, max_d=max_d, whole_band=whole_band, seed=seed + 7,
                )
            )
            if whole_band:
                rows.extend(
                    _bc.beneficial_carriage_rows(
                        backend, graphs, model=model.name, split=split, modes=["matched"],
                        donors=donors, max_d=max_d, whole_band=True, seed=seed + 11,
                    )
                )
    write_csv(artifact_root / "metrics" / "step6_beneficial_carriage.csv", rows)
    render_step6(rows, artifact_root, dpi=dpi)

    # IG-aligned pair: functional carriage C(d) vs loss-carriage B_IG(d), BOTH on the transport
    # (carrier<->source) distance axis, same IG/baseline/readout -- directly comparable to Step 3.
    ig_rows: list[dict[str, Any]] = []
    if bool(cfg.get("run_loss_carriage", True)):
        ig_sample = int(cfg.get("ig_sample_graphs", 8))
        ig_steps = int(config["perturbation"].get("ig_steps", 16))
        readout_ig = carriage_ig_uses_readout_ig(config)
        baseline_mode = str(cfg.get("loss_carriage_baseline", "matched"))  # in-distribution by default
        for model in analysis_models:
            backend = _GRITBeneficialBackend(model)
            graphs = select_graphs(model.adapter, "test", ig_sample, seed=seed)
            base_graphs = select_baseline_graphs(model.adapter, "test", config, ig_sample, seed=seed)
            baseline = mean_encoded_baseline(model.adapter, base_graphs)
            sig_means = glob_mean = None
            if baseline_mode == "matched":
                sig_means, glob_mean = _bc.build_signature_means(backend, base_graphs)
            progress(f"Step 6 {model.name}: loss-carriage on {len(graphs)} graph(s), baseline={baseline_mode}")
            func_acc: dict[int, float] = {}
            loss_acc: dict[int, float] = {}
            ng = 0
            for graph in graphs:
                yl = getattr(graph, "y", None)
                if yl is None:
                    continue
                y = float(torch.as_tensor(yl).reshape(-1)[0].item())
                dist = torch.as_tensor(distance_matrix(graph)).long()
                base_override = (_bc.matched_baseline(backend, graph, sig_means, glob_mean)
                                 if baseline_mode == "matched" else None)
                cf = carriage_ig(model.adapter, graph, baseline, steps=ig_steps, readout_ig=readout_ig,
                                 baseline_override=base_override)["carriage"]
                cl = carriage_ig(model.adapter, graph, baseline, steps=ig_steps, readout_ig=readout_ig,
                                 baseline_override=base_override, loss_label=y)["carriage"]
                for d in range(0, max_d + 1):
                    m = dist == d
                    if bool(m.any()):
                        func_acc[d] = func_acc.get(d, 0.0) + float(cf.abs()[m].sum().item())
                        loss_acc[d] = loss_acc.get(d, 0.0) + float(cl[m].sum().item())
                ng += 1
            tot_func = sum(func_acc.values()) or 1.0
            for d in sorted(set(func_acc) | set(loss_acc)):
                ig_rows.append({
                    "model": model.name, "distance": int(d), "n_graphs": ng,
                    "C_functional_share": func_acc.get(d, 0.0) / tot_func,      # |C| share (like Step 3)
                    "B_ig_loss_carriage": loss_acc.get(d, 0.0) / max(ng, 1),    # signed: <0 = loss-reducing (beneficial)
                })
        if ig_rows:
            write_csv(artifact_root / "metrics" / "step6_ig_loss_carriage.csv", ig_rows)
            render_step6_ig(ig_rows, artifact_root, dpi=dpi)

    progress("Step 6 complete: metrics and figures written")
    return {"status": "complete", "models": [m.name for m in analysis_models],
            "rows": len(rows), "ig_rows": len(ig_rows)}


def run_step7(models: Sequence[ModelRun], artifact_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Step 7: symbolic (content) vs structural (RRWP) carriage -- the SUBSTRATE axis.

    Moved out of Step 4 (which is composability / mediator patching): content vs node/pair RRWP,
    local vs global RRWP, functional + beneficial, IG (headline) + on-manifold swap (cross-check).
    Runs the symbolic/structural probe and the raw-RRWP ablations (distance-binned + global-channel).
    """
    cfg = config["steps"].get("7", {})
    status: dict[str, Any] = {}
    if bool(cfg.get("run_symbolic_structural_carriage", True)):
        try:
            status["symbolic_structural"] = run_symbolic_structural_probe(models, artifact_root, config)
        except Exception as exc:  # noqa: BLE001
            progress(f"Step 7 symbolic/structural probe failed: {exc}")
            status["symbolic_structural"] = {"status": "failed", "error": str(exc)}
    if bool(cfg.get("run_rrwp_distance_ablation", True)):
        try:
            status["rrwp_ablation"] = run_rrwp_distance_ablation_probe(models, artifact_root, config)
        except Exception as exc:  # noqa: BLE001
            progress(f"Step 7 RRWP ablation probe failed: {exc}")
            status["rrwp_ablation"] = {"status": "failed", "error": str(exc)}
    progress("Step 7 complete: symbolic vs structural carriage (substrate)")
    return {
        "status": "complete",
        **{k: (v.get("status") if isinstance(v, dict) else v) for k, v in status.items()},
    }


def run_intervention_steps(
    config: Mapping[str, Any],
    discovery: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    steps: Sequence[str],
    *,
    models: Optional[Sequence[ModelRun]] = None,
) -> dict[str, Any]:
    if models is None:
        progress(f"instantiating official GRIT adapters for steps: {','.join(steps)}")
        models = instantiate_official_models(config, discovery)
    else:
        progress(f"reusing official GRIT adapters for steps: {','.join(steps)}")
    if not models:
        progress("no official GRIT model artifacts available for intervention steps")
        return {
            step: {
                "status": "waiting_for_official_grit_artifacts",
                "step": step,
                "name": config["steps"][step]["name"],
                "required_adapter": "official_grit",
                "artifact_cache_contract": ["metrics/*.csv", "metrics/*.json", "tensors/*.pt", "figures/*.png", "figures/*.pdf"],
            }
            for step in steps
            if step != "1"
        }
    status: dict[str, Any] = {}
    runners = {
        "0": lambda: run_step0(models, artifact_root, config),
        "2": lambda: run_step2(models, artifact_root, config),
        "3": lambda: run_step3(models, artifact_root, config),
        "4": lambda: run_step4(models, artifact_root, config),
        "5": lambda: run_step5(models, artifact_root, config),
        "6": lambda: run_step6(models, artifact_root, config),
        "7": lambda: run_step7(models, artifact_root, config),
    }
    for step in steps:
        if step == "1":
            continue
        if step not in runners:
            continue
        try:
            progress(f"running Step {step}: {config['steps'][step]['name']}")
            status[step] = runners[step]()
            progress(f"finished Step {step}: {status[step].get('status')}")
        except Exception as exc:
            progress(f"failed Step {step}: {exc}")
            status[step] = {
                "status": "failed",
                "step": step,
                "name": config["steps"][step]["name"],
                "error": str(exc),
                "methodology_core_available": True,
            }
    return status
