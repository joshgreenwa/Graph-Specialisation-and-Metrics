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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from graph_specialisation_metrics.method_adapters import OfficialGRITAdapter
from graph_specialisation_metrics.method_core import (
    EPS,
    all_pair_distances_or_compute,
    atomic_torch_save,
    attention_rollout,
    bootstrap_ci,
    distance_profile,
    effective_rank,
    ensure_dir,
    graph_to_networkx,
    minimum_vertex_cut,
    r2_score,
    spearman_corr,
    top_singular_share,
    write_csv,
    write_json,
)


@dataclass
class ModelRun:
    name: str
    adapter: OfficialGRITAdapter
    role: str
    variant: str


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
    if hasattr(graph, "distances") and isinstance(graph.distances, torch.Tensor):
        return graph.distances.detach().cpu().float()
    return all_pair_distances_or_compute(pyg_graph_view(graph)).cpu()


def far_pairs(dist: torch.Tensor, tau: int, *, max_pairs: Optional[int] = None, seed: int = 0) -> list[tuple[int, int]]:
    pairs = [
        (int(i), int(j))
        for i in range(dist.size(0))
        for j in range(dist.size(1))
        if i != j and bool(torch.isfinite(dist[i, j])) and float(dist[i, j].item()) > float(tau)
    ]
    rng = random.Random(int(seed))
    rng.shuffle(pairs)
    return pairs[: int(max_pairs)] if max_pairs is not None else pairs


def deterministic_partners(n: int, source: int, partners: int, seed: int) -> list[int]:
    choices = [p for p in range(int(n)) if p != int(source)]
    rng = random.Random(int(seed) + int(source) * 1009)
    rng.shuffle(choices)
    return choices[: max(1, min(int(partners), len(choices)))]


def far_thresholds(config: Mapping[str, Any]) -> list[int]:
    values = [int(v) for v in config.get("far_thresholds", [config.get("primary_tau", 3)])]
    primary = int(config.get("primary_tau", 3))
    if primary not in values:
        values.append(primary)
    return sorted(set(values))


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


def predict_scalar_from_encoded(adapter: OfficialGRITAdapter, graph: Any, encoded: torch.Tensor) -> torch.Tensor:
    return adapter.forward_from_encoded_content(graph, encoded, retain_grad=False).prediction.reshape(-1)[0]


def carriage_ig(
    adapter: OfficialGRITAdapter,
    graph: Any,
    mean_baseline: torch.Tensor,
    *,
    steps: int,
    target_index: int = 0,
) -> dict[str, Any]:
    """Compute markdown carriage C[i,j] using encoded-content IG.

    For each carrier ``i`` this integrates the gradient of
    ``g_i · h_i^L`` with respect to each encoded source ``j`` along the
    baseline-to-input path, where ``g_i`` is the clean readout gradient.
    """
    clean_encoded = adapter.encoded_node_states(graph).detach()
    base = expanded_baseline(clean_encoded, mean_baseline)
    delta = clean_encoded - base
    clean_cache, readout_grad = adapter.readout_gradient_from_encoded_content(
        graph,
        clean_encoded.detach().clone(),
        target_index=target_index,
    )
    n = int(clean_encoded.size(0))
    carriage = clean_encoded.new_zeros((n, n))
    for alpha_idx in range(1, int(steps) + 1):
        alpha = float(alpha_idx) / float(steps)
        point = (base + alpha * delta).detach().requires_grad_(True)
        cache = adapter.forward_from_encoded_content(graph, point, retain_grad=False)
        for carrier in range(n):
            scalar = (cache.final_node_states[carrier] * readout_grad[carrier]).sum()
            (grad,) = torch.autograd.grad(scalar, point, retain_graph=carrier < n - 1, create_graph=False)
            carriage[carrier] += (grad.detach() * delta).sum(dim=-1) / float(steps)
    pred_clean = float(clean_cache.prediction.reshape(-1)[target_index].detach().cpu().item())
    pred_base = float(predict_scalar_from_encoded(adapter, graph, base).detach().cpu().item())
    return {
        "carriage": carriage.detach().cpu(),
        "clean_encoded": clean_encoded.detach().cpu(),
        "baseline": base.detach().cpu(),
        "readout_gradient": readout_grad.detach().cpu(),
        "prediction": pred_clean,
        "baseline_prediction": pred_base,
        "clean_cache": clean_cache,
    }


def carriage_swap(
    adapter: OfficialGRITAdapter,
    graph: Any,
    clean_encoded: torch.Tensor,
    readout_grad: torch.Tensor,
    *,
    partners: int,
    seed: int,
) -> torch.Tensor:
    n = int(clean_encoded.size(0))
    clean_cache = adapter.forward_from_encoded_content(graph, clean_encoded.to(adapter.device), retain_grad=False)
    clean_h = clean_cache.final_node_states.detach()
    rng = random.Random(int(seed))
    out = clean_encoded.new_zeros((n, n), device=adapter.device)
    encoded_device = clean_encoded.to(adapter.device)
    grad_device = readout_grad.to(adapter.device)
    for source in range(n):
        choices = deterministic_partners(n, source, partners, seed)
        acc = clean_encoded.new_zeros(n, device=adapter.device)
        for partner in choices:
            pert = encoded_device.detach().clone()
            pert[source] = encoded_device[partner]
            cache = adapter.forward_from_encoded_content(graph, pert, retain_grad=False)
            delta_h = cache.final_node_states.detach() - clean_h
            projected = (delta_h * grad_device).sum(dim=-1)
            denom = torch.linalg.vector_norm(encoded_device[source] - encoded_device[partner]).clamp_min(EPS)
            acc += projected.abs() / denom
        out[:, source] = acc / float(len(choices))
    return out.detach().cpu()


def attention_profiles(cache: Any, dist: torch.Tensor, model: str) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    rows: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}
    if not cache.attention:
        return rows, tensors
    last = cache.attention[-1].detach().cpu().mean(dim=0)
    rollout = attention_rollout([a.detach().cpu() for a in cache.attention])
    tensors["attention_last"] = last
    tensors["attention_rollout"] = rollout
    for kind, matrix in [("attention_last", last), ("attention_rollout", rollout)]:
        for row in distance_profile(matrix.abs(), dist):
            rows.append({"model": model, "quantity": kind, **row})
    return rows, tensors


def attention_faithfulness_rows(
    model: str,
    graph_id: str,
    attention_tensors: Mapping[str, torch.Tensor],
    carriage: torch.Tensor,
    dist: torch.Tensor,
    tau: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    c = carriage.abs().reshape(-1).numpy()
    d = dist.reshape(-1).numpy()
    finite = np.isfinite(d)
    for quantity, tensor in attention_tensors.items():
        if quantity not in {"attention_rollout", "attention_last"}:
            continue
        a = tensor.abs().reshape(-1).numpy()
        far = finite & (d > tau)
        rows.append(
            {
                "model": model,
                "graph_id": graph_id,
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
                    "attention_quantity": quantity,
                    "distance_bin": distance,
                    "spearman": spearman_corr(a[mask], c[mask]) if mask.any() else float("nan"),
                    "attention_far_mass": far_mass(tensor, dist, tau),
                    "carriage_far_mass": far_mass(carriage, dist, tau),
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


def select_graphs(adapter: OfficialGRITAdapter, split: str, sample_graphs: int) -> list[Any]:
    return adapter.load_zinc_split(split, limit=int(sample_graphs))


def instantiate_official_models(config: Mapping[str, Any], discovery: Sequence[Mapping[str, Any]]) -> list[ModelRun]:
    out: list[ModelRun] = []
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(config.get("seeds", [0])[0])
    for entry in discovery:
        if entry.get("adapter") != "official_grit":
            continue
        if not entry.get("checkpoint_candidates") or not entry.get("config_candidates"):
            continue
        name = str(entry["model"])
        model_cfg = config["models"][name]
        adapter = OfficialGRITAdapter(
            repo_path=Path(str(model_cfg.get("repo_path", "external/GRIT"))),
            config_path=Path(str(model_cfg.get("config_path") or entry["config_candidates"][0])),
            checkpoint_path=Path(str(model_cfg.get("checkpoint_path") or entry["checkpoint_candidates"][0])),
            variant=str(model_cfg.get("variant", name)),
            official_commit=str(model_cfg.get("official_commit", "")) or None,
            dataset_dir=Path(str(model_cfg["dataset_dir"])) if model_cfg.get("dataset_dir") else None,
            device=device,
            seed=seed,
        )
        out.append(ModelRun(name=name, adapter=adapter, role=str(model_cfg.get("role", "")), variant=str(model_cfg.get("variant", ""))))
    return out


def render_distance_profile(rows: Sequence[Mapping[str, Any]], artifact_root: Path, filename: str, title: str, *, ylabel: str = "Share of mass", dpi: int = 180) -> None:
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


def render_bar(rows: Sequence[Mapping[str, Any]], artifact_root: Path, filename: str, title: str, *, x_key: str, y_key: str, ylabel: str, dpi: int = 180) -> None:
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
    seed = int(config.get("seeds", [0])[0])
    dpi = int(config["figures"]["dpi"])
    recon_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    tensors: dict[str, Any] = {}
    for model in models:
        graphs = select_graphs(model.adapter, "test", sample_graphs)
        progress(f"Step 0 {model.name}: selected {len(graphs)} test graph(s), IG steps={ig_steps}, swap_partners={swap_partners}")
        baseline = mean_encoded_baseline(model.adapter, graphs)
        pred_vals: list[float] = []
        measured_vals: list[float] = []
        for graph_idx, graph in enumerate(graphs):
            progress_graph("Step 0", model.name, graph_idx, len(graphs))
            gid = graph_identity("test", graph_idx, graph)
            dist = distance_matrix(graph)
            result = carriage_ig(model.adapter, graph, baseline, steps=ig_steps)
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
            )
            tensors[f"step0/{model.name}/{gid}/carriage_ig"] = c_ig
            tensors[f"step0/{model.name}/{gid}/carriage_swap"] = c_swap
            for source in range(c_ig.size(1)):
                pert = encoded.detach().clone()
                pert[source] = base[source]
                pred_pert = float(predict_scalar_from_encoded(model.adapter, graph, pert).detach().cpu().item())
                measured = clean_pred - pred_pert
                predicted = float(c_ig[:, source].sum().item())
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
                for partner in deterministic_partners(c_ig.size(1), source, swap_partners, seed + graph_idx):
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
    write_csv(artifact_root / "metrics" / "step0_reconstruction.csv", recon_rows)
    write_csv(artifact_root / "metrics" / "step0_profile_agreement.csv", profile_rows)
    write_csv(artifact_root / "metrics" / "step0_summary.csv", summaries)
    atomic_torch_save(artifact_root / "tensors" / "step0_carriage.pt", tensors)
    render_step0_reconstruction(
        [r for r in recon_rows if r.get("perturbation") == "finite_content_swap"],
        artifact_root,
        filename="step0_carriage_reconstruction",
        ylabel="Measured finite-swap delta",
        dpi=dpi,
    )
    render_step0_reconstruction(
        [r for r in recon_rows if r.get("perturbation") == "ig_baseline_replacement"],
        artifact_root,
        filename="step0_ig_baseline_reconstruction",
        ylabel="Measured baseline-replacement delta",
        dpi=dpi,
    )
    render_distance_profile(profile_rows, artifact_root, "step0_swap_vs_ig_profiles", "Step 0: estimator agreement, swap vs IG", dpi=dpi)
    progress("Step 0 complete: metrics, tensors, and figures written")
    return {"status": "complete", "models": [m.name for m in models], "summary": summaries}


def render_step0_reconstruction(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, filename: str, ylabel: str, dpi: int) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)
    for model, items in sorted(grouped.items()):
        x = [safe_float(r["predicted_delta"]) for r in items]
        y = [safe_float(r["measured_delta"]) for r in items]
        ax.scatter(x, y, s=10, alpha=0.45, label=f"{model} R2={r2_score(y, x):.2f}", edgecolors="none")
    vals = [safe_float(r[k]) for r in rows for k in ("predicted_delta", "measured_delta")]
    lim = max([abs(v) for v in vals if math.isfinite(v)] + [1.0e-6])
    ax.plot([-lim, lim], [-lim, lim], "--", color="#555555", linewidth=1)
    ax.set_title("Step 0: carriage reconstruction of output sensitivity")
    ax.set_xlabel("First-order sum_i C[i,j]")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / f"{filename}.png", dpi=dpi)
    fig.savefig(figures / f"{filename}.pdf")
    plt.close(fig)


def run_step2(models: Sequence[ModelRun], artifact_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    progress("Step 2 start: usage vs causal usage")
    cfg = config["steps"]["2"]
    sample_graphs = int(cfg.get("sample_graphs", 200))
    ig_steps = int(config["perturbation"].get("ig_steps", 32))
    tau = int(config.get("primary_tau", 3))
    dpi = int(config["figures"]["dpi"])
    profile_rows: list[dict[str, Any]] = []
    faith_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    tensors: dict[str, Any] = {}
    for model in models:
        graphs = select_graphs(model.adapter, "test", sample_graphs)
        progress(f"Step 2 {model.name}: selected {len(graphs)} test graph(s), IG steps={ig_steps}, tau={tau}")
        baseline = mean_encoded_baseline(model.adapter, graphs)
        for graph_idx, graph in enumerate(graphs):
            progress_graph("Step 2", model.name, graph_idx, len(graphs))
            gid = graph_identity("test", graph_idx, graph)
            dist = distance_matrix(graph)
            result = carriage_ig(model.adapter, graph, baseline, steps=ig_steps)
            c_ig = result["carriage"]
            cache = result["clean_cache"]
            tensors[f"step2/{model.name}/{gid}/carriage"] = c_ig
            profile_rows.extend(carriage_profile_rows(model.name, gid, c_ig, dist, "carriage"))
            attn_rows, attn_tensors = attention_profiles(cache, dist, model.name)
            for row in attn_rows:
                row["graph_id"] = gid
            profile_rows.extend(attn_rows)
            tensors.update({f"step2/{model.name}/{gid}/{k}": v for k, v in attn_tensors.items()})
            if "attention_rollout" in attn_tensors:
                faith_rows.extend(attention_faithfulness_rows(model.name, gid, attn_tensors, c_ig, dist, tau))
                for threshold in far_thresholds(config):
                    for quantity in ["attention_rollout", "attention_last"]:
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
            channel_rows.extend(layer_channel_split_rows(model, graph, result, dist, gid, tau))
    write_csv(artifact_root / "metrics" / "step2_profiles.csv", profile_rows)
    write_csv(artifact_root / "metrics" / "step2_attention_faithfulness.csv", faith_rows)
    write_csv(artifact_root / "metrics" / "step2_far_threshold_sensitivity.csv", threshold_rows)
    write_csv(artifact_root / "metrics" / "step2_channel_split.csv", channel_rows)
    atomic_torch_save(artifact_root / "tensors" / "step2_usage_carriage.pt", tensors)
    render_distance_profile(profile_rows, artifact_root, "step2_carriage_vs_attention_distance", "Step 2: carriage vs attention by molecular distance", dpi=dpi)
    render_step2_faithfulness(faith_rows, artifact_root, dpi=dpi)
    final_channel_rows = [r for r in channel_rows if bool(r.get("headline_final_layer"))]
    render_distance_profile(final_channel_rows, artifact_root, "step2_channel_split_distance", "Step 2: final-layer carriage by channel and distance", dpi=dpi)
    render_layer_resolved_channel_split(channel_rows, artifact_root, dpi=dpi)
    progress("Step 2 complete: metrics, tensors, and figures written")
    return {"status": "complete", "models": [m.name for m in models], "profile_rows": len(profile_rows), "faithfulness_rows": len(faith_rows)}


def layer_channel_split_rows(model: ModelRun, graph: Any, result: Mapping[str, Any], dist: torch.Tensor, graph_id: str, tau: int) -> list[dict[str, Any]]:
    clean_cache = result["clean_cache"]
    clean_layers = (clean_cache.channel_fields or {}).get("layers", [])
    if not clean_layers:
        return []
    layer_grads = []
    if clean_cache.extras is not None:
        layer_grads = list(clean_cache.extras.get("layer_output_node_gradients", []))
    readout_grad = result["readout_gradient"].to(model.adapter.device)
    baseline = result["baseline"].to(model.adapter.device)
    encoded = result["clean_encoded"].to(model.adapter.device)
    n = int(encoded.size(0))
    last_layer = len(clean_layers) - 1
    masses_by_layer = {
        layer_idx: {"routing": torch.zeros((n, n)), "transport": torch.zeros((n, n)), "cross": torch.zeros((n, n))}
        for layer_idx in range(len(clean_layers))
    }
    for source in range(n):
        pert = encoded.detach().clone()
        pert[source] = baseline[source]
        cache = model.adapter.forward_from_encoded_content(graph, pert, retain_grad=False)
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
                g = grad[receiver].reshape(-1)
                if g.numel() == routing_vec.numel():
                    masses_by_layer[layer_idx]["routing"][receiver, source] += float(torch.dot(g, routing_vec).detach().cpu().item())
                    masses_by_layer[layer_idx]["transport"][receiver, source] += float(torch.dot(g, transport_vec).detach().cpu().item())
                    masses_by_layer[layer_idx]["cross"][receiver, source] += float(torch.dot(g, cross_vec).detach().cpu().item())
                else:
                    # Fallback if GRIT's output projection changes hidden shape.
                    masses_by_layer[layer_idx]["routing"][receiver, source] += float(torch.linalg.vector_norm(routing_vec).detach().cpu().item())
                    masses_by_layer[layer_idx]["transport"][receiver, source] += float(torch.linalg.vector_norm(transport_vec).detach().cpu().item())
                    masses_by_layer[layer_idx]["cross"][receiver, source] += float(torch.linalg.vector_norm(cross_vec).detach().cpu().item())
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


def render_step2_faithfulness(rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    if not rows:
        return
    summary: list[dict[str, Any]] = []
    for model in sorted(set(str(r["model"]) for r in rows)):
        for quantity in sorted(set(str(r.get("attention_quantity", "attention_rollout")) for r in rows if str(r["model"]) == model)):
            items = [r for r in rows if str(r["model"]) == model and str(r.get("attention_quantity", "attention_rollout")) == quantity]
            overall = [r for r in items if str(r.get("distance_bin")) == "overall"]
            far = [r for r in items if str(r.get("distance_bin", "")).startswith(">")]
            if not overall:
                continue
            summary.append(
                {
                    "model": model,
                    "attention_quantity": quantity,
                    "label": f"{model} {quantity.replace('attention_', '')}",
                    "spearman_overall": float(np.nanmean([safe_float(r["spearman"]) for r in overall])),
                    "spearman_far": float(np.nanmean([safe_float(r["spearman"]) for r in far])),
                    "attention_far_mass": float(np.nanmean([safe_float(r["attention_far_mass"]) for r in overall])),
                    "carriage_far_mass": float(np.nanmean([safe_float(r["carriage_far_mass"]) for r in overall])),
                }
            )
    if not summary:
        return
    rollout_summary = [r for r in summary if r["attention_quantity"] == "attention_rollout"] or summary
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), constrained_layout=True)
    for ax, key, title in [
        (axes[0], "spearman_overall", "Overall"),
        (axes[1], "spearman_far", "Far bin"),
    ]:
        labels = [str(r["model"]) for r in rollout_summary]
        values = [safe_float(r[key]) for r in rollout_summary]
        ax.bar(labels, values, color="#4c78a8")
        ax.axhline(0, color="#555555", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel("Spearman(rollout, |C|)")
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_ha("right")
    figures = ensure_dir(artifact_root / "figures")
    fig.suptitle("Step 2: attention faithfulness to carriage")
    fig.savefig(figures / "step2_attention_faithfulness.png", dpi=dpi)
    fig.savefig(figures / "step2_attention_faithfulness.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)
    for r in summary:
        ax.scatter([r["attention_far_mass"]], [r["carriage_far_mass"]], s=60, label=r["label"])
        ax.annotate(r["label"], (r["attention_far_mass"], r["carriage_far_mass"]))
    ax.plot([0, 1], [0, 1], "--", color="#555555")
    ax.set_title("Step 2: far-mass, attention vs carriage")
    ax.set_xlabel("Attention far-mass")
    ax.set_ylabel("Carriage far-mass")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(figures / "step2_far_mass_attention_vs_carriage.png", dpi=dpi)
    fig.savefig(figures / "step2_far_mass_attention_vs_carriage.pdf")
    plt.close(fig)

def run_step3(models: Sequence[ModelRun], artifact_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    progress("Step 3 start: train/test distance-resolved overfitting")
    dense_models = [m for m in models if m.name == "dense_grit"] or list(models[:1])
    if not dense_models:
        return {"status": "skipped_no_dense_grit"}
    model = dense_models[0]
    cfg = config["steps"]["3"]
    sample_graphs = int(cfg.get("sample_graphs", 200))
    ig_steps = int(config["perturbation"].get("ig_steps", 32))
    dpi = int(config["figures"]["dpi"])
    rows: list[dict[str, Any]] = []
    for split in ["train", "test"]:
        graphs = select_graphs(model.adapter, split, sample_graphs)
        progress(f"Step 3 {model.name}: selected {len(graphs)} {split} graph(s), IG steps={ig_steps}")
        baseline = mean_encoded_baseline(model.adapter, graphs)
        for graph_idx, graph in enumerate(graphs):
            progress_graph("Step 3", model.name, graph_idx, len(graphs), split=split)
            gid = graph_identity(split, graph_idx, graph)
            dist = distance_matrix(graph)
            c = carriage_ig(model.adapter, graph, baseline, steps=ig_steps)["carriage"]
            for row in distance_profile(c.abs(), dist):
                rows.append({"model": model.name, "split": split, "graph_id": gid, **row})
    write_csv(artifact_root / "metrics" / "step3_train_test_profiles.csv", rows)
    gap_rows = train_test_gap_rows(rows, int(config.get("primary_tau", 3)))
    write_csv(artifact_root / "metrics" / "step3_train_minus_test_gap.csv", gap_rows)
    render_step3(rows, gap_rows, artifact_root, dpi=dpi)
    progress("Step 3 complete: metrics and figures written")
    return {"status": "complete", "model": model.name, "profile_rows": len(rows)}


def train_test_gap_rows(rows: Sequence[Mapping[str, Any]], tau: int) -> list[dict[str, Any]]:
    distances = sorted({int(safe_float(r["distance"])) for r in rows if math.isfinite(safe_float(r["distance"]))})
    out = []
    for d in distances:
        train = [safe_float(r["share"]) for r in rows if r.get("split") == "train" and int(safe_float(r["distance"])) == d]
        test = [safe_float(r["share"]) for r in rows if r.get("split") == "test" and int(safe_float(r["distance"])) == d]
        if train and test:
            gap, lo, hi = bootstrap_gap_ci(train, test, seed=1000 + int(d))
            out.append(
                {
                    "distance": d,
                    "train_share": float(np.nanmean(train)),
                    "test_share": float(np.nanmean(test)),
                    "gap": gap,
                    "gap_ci_low": lo,
                    "gap_ci_high": hi,
                }
            )
    far_train = [safe_float(r["share"]) for r in rows if r.get("split") == "train" and safe_float(r["distance"]) > tau]
    far_test = [safe_float(r["share"]) for r in rows if r.get("split") == "test" and safe_float(r["distance"]) > tau]
    if far_train and far_test:
        gap, lo, hi = bootstrap_gap_ci(far_train, far_test, seed=2000 + int(tau))
        out.append(
            {
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


def render_step3(rows: Sequence[Mapping[str, Any]], gap_rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    render_distance_profile(
        [{"model": str(r["split"]), "quantity": "carriage", **r} for r in rows],
        artifact_root,
        "step3_train_vs_test_carriage",
        "Step 3: carriage by distance, train vs test",
        dpi=dpi,
    )
    numeric = [r for r in gap_rows if isinstance(r.get("distance"), int)]
    if numeric:
        fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
        x = np.asarray([safe_float(r["distance"]) for r in numeric], dtype=float)
        y = np.asarray([safe_float(r["gap"]) for r in numeric], dtype=float)
        lo = np.asarray([safe_float(r.get("gap_ci_low")) for r in numeric], dtype=float)
        hi = np.asarray([safe_float(r.get("gap_ci_high")) for r in numeric], dtype=float)
        err_low = np.maximum(0.0, y - lo)
        err_high = np.maximum(0.0, hi - y)
        ax.errorbar(x, y, yerr=np.vstack([err_low, err_high]), marker="o", capsize=3)
        ax.axhline(0, color="#555555", linestyle="--", linewidth=1)
        ax.set_title("Step 3: train-minus-test carriage gap")
        ax.set_xlabel("Molecular hop distance")
        ax.set_ylabel("Train share - test share")
        figures = ensure_dir(artifact_root / "figures")
        fig.savefig(figures / "step3_train_minus_test_gap.png", dpi=dpi)
        fig.savefig(figures / "step3_train_minus_test_gap.pdf")
        plt.close(fig)


def mediator_cut(graph: Any, target: int, source: int) -> list[int]:
    return minimum_vertex_cut(pyg_graph_view(graph), int(target), int(source))


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
) -> float:
    clean_encoded = adapter.encoded_node_states(graph).detach()
    base = expanded_baseline(clean_encoded, mean_baseline)
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
    max_pairs = int(cfg.get("max_far_pairs_per_graph", 64))
    tau = int(config.get("primary_tau", 3))
    selection_tau = min(far_thresholds(config))
    ig_steps = int(config["perturbation"].get("ig_steps", 32))
    seed = int(config.get("seeds", [0])[0])
    dpi = int(config["figures"]["dpi"])
    rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    tensors: dict[str, Any] = {}
    for model in models:
        graphs = select_graphs(model.adapter, "test", sample_graphs)
        progress(f"Step 4 {model.name}: selected {len(graphs)} test graph(s), max_far_pairs_per_graph={max_pairs}, IG steps={ig_steps}")
        baseline = mean_encoded_baseline(model.adapter, graphs)
        for graph_idx, graph in enumerate(graphs):
            progress_graph("Step 4", model.name, graph_idx, len(graphs))
            gid = graph_identity("test", graph_idx, graph)
            dist = distance_matrix(graph)
            result = carriage_ig(model.adapter, graph, baseline, steps=ig_steps)
            c = result["carriage"]
            clean_cache = result["clean_cache"]
            readout_grad = result["readout_gradient"]
            selected = far_pairs(dist, selection_tau, max_pairs=max_pairs, seed=seed + graph_idx)
            direct_matrix = torch.zeros_like(c)
            for pair_idx, (carrier, source) in enumerate(selected):
                pair_interval = progress_interval(len(selected), target_messages=4)
                if pair_idx == 0 or pair_idx + 1 == len(selected) or (pair_idx + 1) % pair_interval == 0:
                    progress(f"Step 4 {model.name} graph {graph_idx + 1}/{len(graphs)}: patched pair {pair_idx + 1}/{len(selected)}")
                cut = mediator_cut(graph, carrier, source)
                if not cut:
                    continue
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
                )
                direct_matrix[carrier, source] = direct
                unclamped = float(c[carrier, source].item())
                rows.append(
                    {
                        "model": model.name,
                        "graph_id": gid,
                        "carrier": carrier,
                        "source": source,
                        "distance": float(dist[carrier, source].item()),
                        "cut_size": len(cut),
                        "unclamped": unclamped,
                        "direct": direct,
                        "composed": unclamped - direct,
                        "direct_fraction": abs(direct) / max(abs(unclamped), EPS),
                    }
                )
                if pair_idx < max(1, min(8, max_pairs)):
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
                            steps=max(2, min(ig_steps, 8)),
                            clamp_until_layer=layer,
                        )
                        depth_rows.append(
                            {
                                "model": model.name,
                                "graph_id": gid,
                                "carrier": carrier,
                                "source": source,
                                "distance": float(dist[carrier, source].item()),
                                "clamp_until_layer": layer,
                                "direct": depth_direct,
                                "direct_fraction": abs(depth_direct) / max(abs(unclamped), EPS),
                            }
                        )
            tensors[f"step4/{model.name}/{gid}/unclamped_carriage"] = c
            tensors[f"step4/{model.name}/{gid}/direct_carriage"] = direct_matrix
    write_csv(artifact_root / "metrics" / "step4_mediator_patching.csv", rows)
    write_csv(artifact_root / "metrics" / "step4_depth_schedule.csv", depth_rows)
    validation_summary = mediator_validation_summary(rows)
    write_csv(artifact_root / "metrics" / "step4_mediator_validation_summary.csv", validation_summary)
    atomic_torch_save(artifact_root / "tensors" / "step4_mediator_patching.pt", tensors)
    render_step4(rows, depth_rows, validation_summary, artifact_root, dpi=dpi)
    progress("Step 4 complete: metrics, tensors, and figures written")
    return {"status": "complete", "patch_rows": len(rows), "depth_rows": len(depth_rows)}


def mediator_validation_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model in sorted(set(str(r.get("model")) for r in rows)):
        vals = [safe_float(r.get("direct_fraction")) for r in rows if str(r.get("model")) == model]
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            continue
        mean, lo, hi = bootstrap_ci(vals, seed=3000 + len(out), draws=500)
        out.append(
            {
                "model": model,
                "mean_direct_fraction": mean,
                "ci_low": lo,
                "ci_high": hi,
                "pairs": len(vals),
                "validation_role": "secondary_composed_reference" if "1hop" in model else "treatment_context",
            }
        )
    return out


def render_step4(rows: Sequence[Mapping[str, Any]], depth_rows: Sequence[Mapping[str, Any]], validation_summary: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    if validation_summary:
        labels = [str(r["model"]) for r in validation_summary]
        y = np.asarray([safe_float(r["mean_direct_fraction"]) for r in validation_summary], dtype=float)
        lo = np.asarray([safe_float(r["ci_low"]) for r in validation_summary], dtype=float)
        hi = np.asarray([safe_float(r["ci_high"]) for r in validation_summary], dtype=float)
        fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
        ax.bar(labels, y, yerr=np.vstack([np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)]), capsize=4, color="#4c78a8")
        ax.axhline(0, color="#555555", linewidth=1)
        ax.set_title("Step 4: mediator-patching validation")
        ax.set_xlabel("Model")
        ax.set_ylabel("Mean direct fraction on far pairs")
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_ha("right")
        figures = ensure_dir(artifact_root / "figures")
        fig.savefig(figures / "step4_mediator_patching_validation.png", dpi=dpi)
        fig.savefig(figures / "step4_mediator_patching_validation.pdf")
        plt.close(fig)
    if rows:
        by_model_distance: dict[tuple[str, int], list[float]] = {}
        for row in rows:
            by_model_distance.setdefault((str(row["model"]), int(safe_float(row["distance"]))), []).append(safe_float(row["direct_fraction"]))
        prof = [
            {"model": model, "quantity": "direct_fraction", "distance": dist, "share": float(np.nanmean(vals)), "mass": float(np.nanmean(vals))}
            for (model, dist), vals in sorted(by_model_distance.items())
        ]
        render_distance_profile(prof, artifact_root, "step4_direct_fraction_by_distance", "Step 4: direct fraction of far carriage by distance", ylabel="Direct fraction", dpi=dpi)
    if depth_rows:
        by_layer: dict[tuple[str, int], list[float]] = {}
        for row in depth_rows:
            by_layer.setdefault((str(row["model"]), int(row["clamp_until_layer"])), []).append(safe_float(row["direct_fraction"]))
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        for model in sorted(set(k[0] for k in by_layer)):
            xs = sorted(k[1] for k in by_layer if k[0] == model)
            ax.plot(xs, [float(np.nanmean(by_layer[(model, x)])) for x in xs], marker="o", label=model)
        ax.set_title("Step 4: depth-resolved direct carriage")
        ax.set_xlabel("Clamp through layer")
        ax.set_ylabel("Surviving direct fraction")
        ax.legend(frameon=False)
        figures = ensure_dir(artifact_root / "figures")
        fig.savefig(figures / "step4_depth_resolved_direct_carriage.png", dpi=dpi)
        fig.savefig(figures / "step4_depth_resolved_direct_carriage.pdf")
        plt.close(fig)


def run_step5(models: Sequence[ModelRun], artifact_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    progress("Step 5 start: non-composable gap attribution")
    dense = next((m for m in models if m.name == "dense_grit"), None)
    onehop = next((m for m in models if m.name == "grit_1hop"), None)
    if dense is None or onehop is None:
        return {"status": "skipped_requires_dense_and_onehop"}
    cfg = config["steps"]["5"]
    sample_graphs = int(cfg.get("sample_graphs", 200))
    max_pairs = int(config["steps"]["4"].get("max_far_pairs_per_graph", 64))
    interaction_pairs = int(cfg.get("interaction_pairs", 1000))
    tau = int(config.get("primary_tau", 3))
    thresholds = far_thresholds(config)
    selection_tau = min(thresholds)
    ig_steps = int(config["perturbation"].get("ig_steps", 32))
    seed = int(config.get("seeds", [0])[0])
    dpi = int(config["figures"]["dpi"])
    graphs = select_graphs(dense.adapter, "test", sample_graphs)
    dense_base = mean_encoded_baseline(dense.adapter, graphs)
    onehop_base = mean_encoded_baseline(onehop.adapter, select_graphs(onehop.adapter, "test", sample_graphs))
    gap_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for graph_idx, graph in enumerate(graphs):
        progress_graph("Step 5", dense.name, graph_idx, len(graphs))
        gid = graph_identity("test", graph_idx, graph)
        dist = distance_matrix(graph)
        result = carriage_ig(dense.adapter, graph, dense_base, steps=ig_steps)
        c = result["carriage"]
        clean_cache = result["clean_cache"]
        readout_grad = result["readout_gradient"]
        direct = torch.zeros_like(c)
        selected = far_pairs(dist, selection_tau, max_pairs=max_pairs, seed=seed + graph_idx)
        for pair_idx, (carrier, source) in enumerate(selected):
            pair_interval = progress_interval(len(selected), target_messages=4)
            if pair_idx == 0 or pair_idx + 1 == len(selected) or (pair_idx + 1) % pair_interval == 0:
                progress(f"Step 5 dense_grit graph {graph_idx + 1}/{len(graphs)}: patched pair {pair_idx + 1}/{len(selected)}")
            cut = mediator_cut(graph, carrier, source)
            if not cut:
                continue
            direct[carrier, source] = patched_ig_pair(
                dense.adapter,
                graph,
                dense_base,
                clean_cache,
                readout_grad,
                carrier=carrier,
                source=source,
                clamp_nodes=cut,
                steps=ig_steps,
            )
        r_nc_by_tau = {
            int(threshold): float(direct.abs()[torch.isfinite(dist) & (dist > int(threshold))].sum().item())
            for threshold in thresholds
        }
        r_nc = r_nc_by_tau[int(tau)]
        y = graph_label(graph)
        dense_pred = float(result["prediction"])
        try:
            onehop_pred = float(onehop.adapter.predict(graph).reshape(-1)[0].detach().cpu().item())
        except Exception:
            onehop_pred = float("nan")
        dense_err = abs(dense_pred - y) if math.isfinite(y) else float("nan")
        onehop_err = abs(onehop_pred - y) if math.isfinite(y) and math.isfinite(onehop_pred) else float("nan")
        gap_row = {
            "graph_id": gid,
            "tau": tau,
            "r_nc": r_nc,
            "dense_error": dense_err,
            "onehop_error": onehop_err,
            "onehop_minus_dense_error": onehop_err - dense_err if math.isfinite(onehop_err) and math.isfinite(dense_err) else float("nan"),
        }
        for threshold, value in r_nc_by_tau.items():
            gap_row[f"r_nc_tau_{threshold}"] = value
        gap_rows.append(gap_row)
        for threshold in thresholds:
            far_mask = torch.isfinite(dist) & (dist > threshold)
            far_matrix = torch.zeros_like(direct)
            far_matrix[far_mask] = direct[far_mask]
            rank_rows.append(
                {
                    "model": dense.name,
                    "graph_id": gid,
                    "tau": threshold,
                    "primary_tau": threshold == tau,
                    "effective_rank": effective_rank(far_matrix),
                    "top_singular_share": top_singular_share(far_matrix),
                    "above_null_margin": distance_preserving_above_null_margin(far_matrix, dist, seed=seed + graph_idx + threshold),
                    "r_nc": r_nc_by_tau[int(threshold)],
                }
            )
        pairs = far_pairs(dist, tau, max_pairs=max(1, interaction_pairs // max(1, sample_graphs)), seed=seed + 1000 + graph_idx)
        interaction_rows.extend(non_additivity_rows(dense, graph, dense_base, pairs, gid, rng))
    write_csv(artifact_root / "metrics" / "step5_gap_vs_rnc.csv", gap_rows)
    write_csv(artifact_root / "metrics" / "step5_far_carriage_rank.csv", rank_rows)
    write_csv(artifact_root / "metrics" / "step5_non_additivity.csv", interaction_rows)
    write_json(artifact_root / "metrics" / "step5_gap_regression.json", gap_regression_summary(gap_rows))
    write_csv(artifact_root / "metrics" / "step5_far_carriage_rank_summary.csv", rank_summary_rows(rank_rows))
    write_csv(artifact_root / "metrics" / "step5_non_additivity_summary.csv", non_additivity_summary_rows(interaction_rows))
    vnode_decision = summarize_vnode_decision(rank_rows, interaction_rows)
    write_json(artifact_root / "metrics" / "step5_vnode_decision.json", vnode_decision)
    write_csv(artifact_root / "metrics" / "step5_vnode_decision.csv", [vnode_decision])
    render_step5(gap_rows, rank_rows, interaction_rows, artifact_root, dpi=dpi)
    render_step5_vnode_decision(vnode_decision, artifact_root, dpi=dpi)
    progress("Step 5 complete: metrics, tensors, and figures written")
    return {"status": "complete", "gap_rows": len(gap_rows), "rank_rows": len(rank_rows), "interaction_rows": len(interaction_rows)}


def non_additivity_rows(model: ModelRun, graph: Any, mean_baseline: torch.Tensor, pairs: Sequence[tuple[int, int]], graph_id: str, rng: random.Random) -> list[dict[str, Any]]:
    encoded = model.adapter.encoded_node_states(graph).detach().to(model.adapter.device)
    clean = float(predict_scalar_from_encoded(model.adapter, graph, encoded).detach().cpu().item())
    rows = []
    seen: set[tuple[int, int]] = set()
    for a, b in pairs:
        if a == b:
            continue
        key = tuple(sorted((int(a), int(b))))
        if key in seen:
            continue
        seen.add(key)
        candidates_a = [idx for idx in range(encoded.size(0)) if idx not in {int(a), int(b)}]
        candidates_b = [idx for idx in range(encoded.size(0)) if idx not in {int(a), int(b)}]
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
        ratio = abs(dab - da - db) / denom if denom > EPS else float("nan")
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
                "non_additivity": ratio,
                "perturbation": "finite_content_swap",
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


def summarize_vnode_decision(rank_rows: Sequence[Mapping[str, Any]], interaction_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary_rank_rows = [r for r in rank_rows if bool(r.get("primary_tau", True))]
    if primary_rank_rows:
        rank_rows = primary_rank_rows
    non_add = np.asarray([safe_float(r.get("non_additivity")) for r in interaction_rows], dtype=float)
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
    elif mean_non_add <= threshold and (mean_top_share >= rank1_top_share_threshold or mean_effective_rank <= 1.25):
        decision = "rank1_broadcast_plausible_by_additivity"
    else:
        decision = "source_specific_interactions_exceed_rank1_broadcast"
    return {
        "decision": decision,
        "non_additivity_threshold": threshold,
        "mean_non_additivity": mean_non_add,
        "median_non_additivity": median_non_add,
        "rank1_top_share_threshold": rank1_top_share_threshold,
        "mean_effective_rank": mean_effective_rank,
        "mean_top_singular_share": mean_top_share,
        "mean_distance_preserving_above_null_margin": mean_above_null,
        "interaction_pairs": int(non_add.size),
        "rank_graphs": int(top_share.size),
        "interpretation": "VNode/rank-1 broadcast is plausible only when distant-source effects are close to additive; high non-additivity means a single pooled global node cannot reproduce source-specific pair coupling.",
    }


def gap_regression_summary(gap_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    x = np.asarray([safe_float(r.get("r_nc")) for r in gap_rows], dtype=float)
    y = np.asarray([safe_float(r.get("onehop_minus_dense_error")) for r in gap_rows], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return {"n": int(mask.sum()), "pearson_r": float("nan"), "slope": float("nan"), "intercept": float("nan")}
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
    return {"n": int(mask.sum()), "pearson_r": corr, "slope": float(slope), "intercept": float(intercept)}


def rank_summary_rows(rank_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    primary = [r for r in rank_rows if bool(r.get("primary_tau", True))]
    rows = primary or list(rank_rows)
    out = []
    for metric in ["effective_rank", "top_singular_share", "above_null_margin"]:
        values = [safe_float(r.get(metric)) for r in rows]
        values = [v for v in values if math.isfinite(v)]
        if not values:
            continue
        mean, lo, hi = bootstrap_ci(values, seed=4100 + len(out), draws=1000)
        out.append({"metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi, "n": len(values), "tau": rows[0].get("tau", "") if rows else ""})
    return out


def non_additivity_summary_rows(interaction_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model in sorted(set(str(r.get("model")) for r in interaction_rows)):
        values = [safe_float(r.get("non_additivity")) for r in interaction_rows if str(r.get("model")) == model]
        values = [v for v in values if math.isfinite(v)]
        if not values:
            continue
        mean, lo, hi = bootstrap_ci(values, seed=5100 + len(out), draws=1000)
        out.append({"model": model, "mean_non_additivity": mean, "ci_low": lo, "ci_high": hi, "pairs": len(values)})
    return out


def render_step5(gap_rows: Sequence[Mapping[str, Any]], rank_rows: Sequence[Mapping[str, Any]], interaction_rows: Sequence[Mapping[str, Any]], artifact_root: Path, *, dpi: int) -> None:
    figures = ensure_dir(artifact_root / "figures")
    primary_rank_rows = [r for r in rank_rows if bool(r.get("primary_tau", True))]
    if primary_rank_rows:
        rank_rows = primary_rank_rows
    if gap_rows:
        x = np.asarray([safe_float(r["r_nc"]) for r in gap_rows], dtype=float)
        y = np.asarray([safe_float(r["onehop_minus_dense_error"]) for r in gap_rows], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            fig, ax = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)
            ax.scatter(x[mask], y[mask], s=18, alpha=0.65)
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1]) if mask.sum() >= 2 else float("nan")
            if mask.sum() >= 2:
                coef = np.polyfit(x[mask], y[mask], 1)
                xs = np.linspace(float(x[mask].min()), float(x[mask].max()), 100)
                ax.plot(xs, coef[0] * xs + coef[1], color="#f58518", label=f"r={corr:.2f}")
                ax.legend(frameon=False)
            ax.set_title("Step 5: performance gap vs non-composable carriage")
            ax.set_xlabel("R_nc")
            ax.set_ylabel("1-hop error - dense error")
            fig.savefig(figures / "step5_gap_vs_rnc.png", dpi=dpi)
            fig.savefig(figures / "step5_gap_vs_rnc.pdf")
            plt.close(fig)
    if rank_rows:
        summary = rank_summary_rows(rank_rows)
        if summary:
            labels = [str(r["metric"]).replace("_", " ") for r in summary]
            y = np.asarray([safe_float(r["mean"]) for r in summary], dtype=float)
            lo = np.asarray([safe_float(r["ci_low"]) for r in summary], dtype=float)
            hi = np.asarray([safe_float(r["ci_high"]) for r in summary], dtype=float)
            fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
            ax.bar(labels, y, yerr=np.vstack([np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)]), capsize=4, color="#4c78a8")
            ax.axhline(0, color="#555555", linewidth=1)
            ax.set_title("Step 5: structure of non-composable long-range carriage")
            ax.set_ylabel("Summary value")
            for tick in ax.get_xticklabels():
                tick.set_rotation(18)
                tick.set_ha("right")
            fig.savefig(figures / "step5_structure_non_composable_carriage.png", dpi=dpi)
            fig.savefig(figures / "step5_structure_non_composable_carriage.pdf")
            plt.close(fig)
    if interaction_rows:
        summary = non_additivity_summary_rows(interaction_rows)
        if summary:
            labels = [str(r["model"]) for r in summary]
            y = np.asarray([safe_float(r["mean_non_additivity"]) for r in summary], dtype=float)
            lo = np.asarray([safe_float(r["ci_low"]) for r in summary], dtype=float)
            hi = np.asarray([safe_float(r["ci_high"]) for r in summary], dtype=float)
            fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
            ax.bar(labels, y, yerr=np.vstack([np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)]), capsize=4, color="#f58518")
            ax.set_title("Step 5: distant-source non-additivity")
            ax.set_ylabel("Non-additivity ratio")
            for tick in ax.get_xticklabels():
                tick.set_rotation(15)
                tick.set_ha("right")
            fig.savefig(figures / "step5_non_additivity.png", dpi=dpi)
            fig.savefig(figures / "step5_non_additivity.pdf")
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.4, 5.0), constrained_layout=True)
        sx = [safe_float(r["delta_a"]) + safe_float(r["delta_b"]) for r in interaction_rows]
        sy = [safe_float(r["delta_ab"]) for r in interaction_rows]
        ax.scatter(sx, sy, s=12, alpha=0.5)
        lim = max([abs(v) for v in sx + sy if math.isfinite(v)] + [1.0e-6])
        ax.plot([-lim, lim], [-lim, lim], "--", color="#555555")
        ax.set_title("Step 5: additivity of distant sources")
        ax.set_xlabel("delta_A + delta_B")
        ax.set_ylabel("delta_AB")
        fig.savefig(figures / "step5_additivity_scatter.png", dpi=dpi)
        fig.savefig(figures / "step5_additivity_scatter.pdf")
        plt.close(fig)


def render_step5_vnode_decision(decision: Mapping[str, Any], artifact_root: Path, *, dpi: int) -> None:
    values = [
        safe_float(decision.get("mean_non_additivity")),
        safe_float(decision.get("mean_top_singular_share")),
        safe_float(decision.get("mean_distance_preserving_above_null_margin")),
    ]
    if not any(math.isfinite(v) for v in values):
        return
    labels = ["Non-additivity", "Top singular share", "Above-null margin"]
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    ax.bar(labels, [0.0 if not math.isfinite(v) else v for v in values], color=["#f58518", "#4c78a8", "#54a24b"])
    ax.axhline(safe_float(decision.get("non_additivity_threshold")), color="#555555", linestyle="--", linewidth=1, label="Non-additivity threshold")
    ax.set_title("Step 5: virtual-node sufficiency decision")
    ax.set_ylabel("Summary value")
    ax.legend(frameon=False, fontsize=8)
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)
        tick.set_ha("right")
    figures = ensure_dir(artifact_root / "figures")
    fig.savefig(figures / "step5_vnode_decision.png", dpi=dpi)
    fig.savefig(figures / "step5_vnode_decision.pdf")
    plt.close(fig)


def run_intervention_steps(config: Mapping[str, Any], discovery: Sequence[Mapping[str, Any]], artifact_root: Path, steps: Sequence[str]) -> dict[str, Any]:
    progress(f"instantiating official GRIT adapters for steps: {','.join(steps)}")
    models = instantiate_official_models(config, discovery)
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
