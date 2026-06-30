"""Synthetic validation suite for the dissertation methodology.

The suite writes the shared artifact contract:
  manifest.json, config.yaml, metrics/*.csv/json, tensors/*.pt, figures/*.png/pdf.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from graph_specialisation_metrics.method_adapters import (
    DirectLinkAdapter,
    StepByStepChainAdapter,
    chain_graph_with_branches,
    make_small_graph_transformer_adapter,
)
from graph_specialisation_metrics.method_core import (
    AdapterInfo,
    GraphBatchView,
    above_null_margin,
    atomic_torch_save,
    auroc_score,
    bootstrap_ci,
    config_hash,
    effective_rank,
    edge_index_from_edges,
    ensure_dir,
    integrated_gradients_output,
    minimum_vertex_cut,
    non_additivity_ratio,
    r2_score,
    read_yaml,
    set_global_seed,
    top_singular_share,
    write_csv,
    write_json,
    write_manifest,
    write_yaml,
)


METHOD_VALIDATION_MD = Path("/Users/joshgreen/Downloads/method_validation.md")
MAIN_PROCEDURE_MD = Path("/Users/joshgreen/Downloads/dissertation_core_procedure.md")
SOURCE_MARKDOWN_EXPECTED_SHA256 = {
    "method_validation.md": "58e85e0897199e03b8f757f92eb0cb6f13ab3c8f571498834732bb3cb464165a",
    "dissertation_core_procedure.md": "cf567856da55d780b6eb5a10f51e94e8448ac0c4ad1959dcea8839a5127c525f",
}


DEFAULT_CONFIG: dict[str, Any] = {
    "artifact_root": "artifacts/method_validation",
    "seed": 41,
    "device": "auto",
    "carriage": {
        "num_graphs": 500,
        "num_nodes": 20,
        "feature_dim": 8,
        "planted_set_size": 3,
        "min_planted_pair_distance": 3,
        "analysis_graphs": 200,
        "ig_steps": 32,
        "swap_partners": 8,
        "train_epochs": 250,
        "learning_rate": 0.001,
        "small_gt": {"layers": 3, "hidden_dim": 64, "heads": 4},
        "fit_mse_fraction_target_variance": 0.01,
    },
    "patching": {
        "chain_lengths": [4, 5, 6, 7, 8],
        "feature_dim": 8,
        "branch_attachments": [2, 3, 4],
    },
    "rank": {
        "matrix_size": 40,
        "planted_ranks": [1, 2, 3, 5, 8],
        "noise_levels": [0.05, 0.2],
        "matrices_per_setting": 50,
        "energy": 0.99,
        "null_permutations": 32,
    },
    "interaction": {
        "feature_dim": 8,
        "pairs": 1000,
        "min_denominator": 1.0e-8,
    },
    "figures": {"dpi": 180, "bootstrap_draws": 1000},
}


FAST_DEV_OVERRIDES: dict[str, Any] = {
    "artifact_root": "artifacts/method_validation_fast_dev",
    "carriage": {
        "num_graphs": 48,
        "num_nodes": 14,
        "analysis_graphs": 12,
        "ig_steps": 8,
        "swap_partners": 3,
        "train_epochs": 45,
        "small_gt": {"layers": 2, "hidden_dim": 32, "heads": 4},
    },
    "patching": {"chain_lengths": [4, 5, 6]},
    "rank": {
        "matrix_size": 24,
        "planted_ranks": [1, 2, 3],
        "noise_levels": [0.05, 0.2],
        "matrices_per_setting": 8,
        "null_permutations": 8,
    },
    "interaction": {"pairs": 128},
    "figures": {"bootstrap_draws": 200},
}


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def connected_edges(num_nodes: int, rng: np.random.Generator, extra_edges: int) -> list[tuple[int, int]]:
    edges = [(i, i + 1) for i in range(num_nodes - 1)]
    existing = {tuple(sorted(e)) for e in edges}
    attempts = 0
    while len(existing) < num_nodes - 1 + extra_edges and attempts < 10000:
        u, v = rng.choice(num_nodes, size=2, replace=False)
        key = tuple(sorted((int(u), int(v))))
        if key not in existing:
            existing.add(key)
            edges.append(key)
        attempts += 1
    return edges


def shortest_distances_numpy(num_nodes: int, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    dist = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    for u, v in edges:
        dist[int(u), int(v)] = 1.0
        dist[int(v), int(u)] = 1.0
    for k in range(num_nodes):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    return dist


def select_planted_set(
    num_nodes: int,
    size: int,
    min_pair_distance: int,
    dist: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    for _ in range(2000):
        chosen = sorted(int(v) for v in rng.choice(num_nodes, size=size, replace=False))
        if max(dist[u, v] for u in chosen for v in chosen if u != v) >= min_pair_distance:
            return chosen
    return sorted(int(v) for v in rng.choice(num_nodes, size=size, replace=False))


def make_carriage_dataset(cfg: Mapping[str, Any], seed: int) -> tuple[list[GraphBatchView], torch.Tensor]:
    rng = np.random.default_rng(int(seed))
    num_graphs = int(cfg["num_graphs"])
    num_nodes = int(cfg["num_nodes"])
    feature_dim = int(cfg["feature_dim"])
    planted_size = int(cfg["planted_set_size"])
    min_distance = int(cfg["min_planted_pair_distance"])
    weight = torch.randn(feature_dim, generator=torch.Generator(device="cpu").manual_seed(int(seed) + 99))
    graphs: list[GraphBatchView] = []
    for graph_idx in range(num_graphs):
        edges = connected_edges(num_nodes, rng, extra_edges=max(1, num_nodes // 3))
        dist = shortest_distances_numpy(num_nodes, edges)
        planted = select_planted_set(num_nodes, planted_size, min_distance, dist, rng)
        x = torch.randn(num_nodes, feature_dim, generator=torch.Generator(device="cpu").manual_seed(int(seed) + 1000 + graph_idx))
        selector = torch.zeros(num_nodes, dtype=torch.float32)
        selector[planted] = 1.0
        y = (x[planted] * weight.view(1, -1)).sum().view(1)
        graphs.append(
            GraphBatchView(
                x=x,
                edge_index=edge_index_from_edges(num_nodes, edges, undirected=True),
                y=y,
                graph_ids=[f"synthetic_carriage_{graph_idx:05d}"],
                split="validation",
                distances=torch.as_tensor(dist, dtype=torch.float32),
                metadata={"planted_nodes": planted, "selector_mask": selector},
            )
        )
    return graphs, weight


def graph_to_device(graph: GraphBatchView, device: torch.device) -> GraphBatchView:
    meta = dict(graph.metadata or {})
    if "selector_mask" in meta:
        meta["selector_mask"] = torch.as_tensor(meta["selector_mask"], dtype=torch.float32, device=device)
    return graph.to(device).clone_with(metadata=meta)


def train_small_gt(
    graphs: Sequence[GraphBatchView],
    cfg: Mapping[str, Any],
    *,
    device: torch.device,
    seed: int,
) -> tuple[Any, list[dict[str, float]]]:
    set_global_seed(seed)
    spec = cfg["small_gt"]
    adapter = make_small_graph_transformer_adapter(
        content_dim=int(cfg["feature_dim"]),
        hidden_dim=int(spec["hidden_dim"]),
        layers=int(spec["layers"]),
        heads=int(spec["heads"]),
        device=device,
    )
    opt = torch.optim.AdamW(adapter.model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=1.0e-5)
    history: list[dict[str, float]] = []
    train_graphs = list(graphs)
    target_values = torch.cat([g.y.view(-1) for g in train_graphs if g.y is not None])
    target_var = float(torch.var(target_values).item())
    target_mse = float(cfg["fit_mse_fraction_target_variance"]) * max(target_var, 1.0e-8)
    rng = np.random.default_rng(int(seed) + 1234)
    for epoch in range(1, int(cfg["train_epochs"]) + 1):
        order = rng.permutation(len(train_graphs))
        losses = []
        for idx in order:
            graph = graph_to_device(train_graphs[int(idx)], device)
            assert graph.y is not None
            opt.zero_grad(set_to_none=True)
            pred = adapter.predict(graph)
            loss = F.mse_loss(pred.view(-1), graph.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        mean_loss = float(np.mean(losses))
        history.append({"epoch": float(epoch), "mse": mean_loss, "target_mse": target_mse})
        if mean_loss <= target_mse and epoch >= 20:
            break
    return adapter, history


def baseline_for_graphs(graphs: Sequence[GraphBatchView], device: torch.device) -> torch.Tensor:
    all_x = torch.cat([g.x for g in graphs], dim=0)
    mean = all_x.mean(dim=0, keepdim=True).to(device)
    return mean


def carriage_reconstruction_rows(
    adapter: Any,
    graphs: Sequence[GraphBatchView],
    *,
    baseline: torch.Tensor,
    cfg: Mapping[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    rows: list[dict[str, Any]] = []
    tensors: dict[str, Any] = {"source_influences": [], "measured_influences": [], "labels": []}
    for graph_idx, raw_graph in enumerate(graphs[: int(cfg["analysis_graphs"])]):
        graph = graph_to_device(raw_graph, device)
        planted = set(int(v) for v in (graph.metadata or {}).get("planted_nodes", []))
        base = baseline.expand_as(graph.x).to(device=device, dtype=graph.x.dtype)

        def predict_from_x(x_new: torch.Tensor) -> torch.Tensor:
            return adapter.predict(graph.clone_with(x=x_new))

        ig = integrated_gradients_output(
            predict_from_x,
            graph.x,
            base,
            steps=int(cfg["ig_steps"]),
        )
        source_influence = ig.sum(dim=-1).detach().cpu()
        clean = float(adapter.predict(graph).detach().cpu().reshape(-1)[0].item())
        measured = []
        for node in range(graph.num_nodes):
            x_new = graph.x.detach().clone()
            x_new[node] = base[node]
            pred = float(adapter.predict(graph.clone_with(x=x_new)).detach().cpu().reshape(-1)[0].item())
            measured.append(clean - pred)
        measured_t = torch.tensor(measured, dtype=torch.float32)
        abs_scores = source_influence.abs()
        norm = float(abs_scores.sum().item())
        for node in range(graph.num_nodes):
            label = int(node in planted)
            rows.append(
                {
                    "graph_index": graph_idx,
                    "node": node,
                    "planted": label,
                    "predicted_influence": float(source_influence[node].item()),
                    "measured_influence": float(measured_t[node].item()),
                    "influence_score": float(abs_scores[node].item()),
                    "normalised_influence_score": float(abs_scores[node].item() / max(norm, 1.0e-12)),
                }
            )
        tensors["source_influences"].append(source_influence)
        tensors["measured_influences"].append(measured_t)
        tensors["labels"].append(torch.tensor([int(n in planted) for n in range(graph.num_nodes)]))
    tensor_payload = {key: torch.stack(value) for key, value in tensors.items()}
    return rows, tensor_payload


def plot_carriage(rows: Sequence[Mapping[str, Any]], history: Sequence[Mapping[str, Any]], out_dir: Path, cfg: Mapping[str, Any]) -> dict[str, float]:
    predicted = np.asarray([float(r["predicted_influence"]) for r in rows], dtype=np.float64)
    measured = np.asarray([float(r["measured_influence"]) for r in rows], dtype=np.float64)
    labels = np.asarray([int(r["planted"]) for r in rows], dtype=bool)
    scores = np.asarray([float(r["normalised_influence_score"]) for r in rows], dtype=np.float64)
    r2 = r2_score(measured, predicted)
    auc = auroc_score(labels, scores)
    figures = ensure_dir(out_dir / "figures")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    axes[0].scatter(predicted, measured, s=12, alpha=0.55, color="#1f77b4", edgecolors="none")
    lim = float(np.nanmax(np.abs(np.concatenate([predicted, measured])))) if len(predicted) else 1.0
    lim = max(lim, 1.0e-6)
    axes[0].plot([-lim, lim], [-lim, lim], linestyle="--", color="#555555", linewidth=1.2)
    axes[0].set_title(f"Carriage reconstructs source effects (R2={r2:.2f})")
    axes[0].set_xlabel("IG predicted influence")
    axes[0].set_ylabel("Measured baseline-replacement influence")
    planted_scores = scores[labels]
    other_scores = scores[~labels]
    axes[1].boxplot([planted_scores, other_scores], showfliers=False)
    axes[1].set_xticks([1, 2], ["Planted atoms", "Other atoms"])
    rng = np.random.default_rng(17)
    for xpos, vals in [(1, planted_scores), (2, other_scores)]:
        jitter = rng.normal(0.0, 0.035, size=len(vals))
        axes[1].scatter(np.full(len(vals), xpos) + jitter, vals, s=9, alpha=0.45, edgecolors="none")
    axes[1].set_title(f"Planted atoms rank first (AUROC={auc:.2f})")
    axes[1].set_ylabel("Normalised absolute influence")
    fig.suptitle("Validation 1: carriage check", fontsize=13)
    fig.savefig(figures / "validation_carriage_check.png", dpi=int(cfg["figures"]["dpi"]))
    fig.savefig(figures / "validation_carriage_check.pdf")
    plt.close(fig)

    if history:
        fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
        ax.plot([h["epoch"] for h in history], [h["mse"] for h in history], color="#1f77b4", label="train MSE")
        ax.plot([h["epoch"] for h in history], [h["target_mse"] for h in history], color="#777777", linestyle="--", label="1% target variance")
        ax.set_title("Small GT training fit for carriage validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean squared error")
        ax.legend(frameon=False)
        fig.savefig(figures / "validation_carriage_training.png", dpi=int(cfg["figures"]["dpi"]))
        fig.savefig(figures / "validation_carriage_training.pdf")
        plt.close(fig)
    return {"r2": float(r2), "auroc": float(auc)}


def carriage_estimator_bakeoff_rows(
    adapter: Any,
    graphs: Sequence[GraphBatchView],
    *,
    baseline: torch.Tensor,
    cfg: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    partners = int(cfg.get("swap_partners", 8))
    for graph_idx, raw_graph in enumerate(graphs[: int(cfg["analysis_graphs"])]):
        graph = graph_to_device(raw_graph, device)
        planted = set(int(v) for v in (graph.metadata or {}).get("planted_nodes", []))
        base = baseline.expand_as(graph.x).to(device=device, dtype=graph.x.dtype)

        def predict_from_x(x_new: torch.Tensor) -> torch.Tensor:
            return adapter.predict(graph.clone_with(x=x_new))

        ig = integrated_gradients_output(
            predict_from_x,
            graph.x,
            base,
            steps=int(cfg["ig_steps"]),
        )
        ig_scores = ig.sum(dim=-1).detach().abs().cpu()
        clean = float(adapter.predict(graph).detach().cpu().reshape(-1)[0].item())
        swap_scores = []
        for node in range(graph.num_nodes):
            choices = [idx for idx in range(graph.num_nodes) if idx != node]
            rng = random.Random(int(seed) + graph_idx * 1009 + node * 9173)
            rng.shuffle(choices)
            vals = []
            for partner in choices[: max(1, min(partners, len(choices)))]:
                x_new = graph.x.detach().clone()
                x_new[node] = graph.x[int(partner)]
                pred = float(adapter.predict(graph.clone_with(x=x_new)).detach().cpu().reshape(-1)[0].item())
                denom = float(torch.linalg.vector_norm(graph.x[node] - graph.x[int(partner)]).detach().cpu().item())
                vals.append((pred - clean) / max(denom, 1.0e-12))
            swap_scores.append(float(np.sqrt(np.mean(np.square(vals)))) if vals else 0.0)
        score_by_estimator = {
            "ig": ig_scores.numpy().astype(float),
            "swap": np.asarray(swap_scores, dtype=float),
        }
        for estimator, scores in score_by_estimator.items():
            norm = float(np.sum(np.abs(scores)))
            for node, score in enumerate(scores):
                rows.append(
                    {
                        "graph_index": graph_idx,
                        "node": int(node),
                        "estimator": estimator,
                        "planted": int(node in planted),
                        "influence_score": float(score),
                        "normalised_influence_score": float(abs(score) / max(norm, 1.0e-12)),
                    }
                )
    return rows


def plot_carriage_estimator_bakeoff(rows: Sequence[Mapping[str, Any]], out_dir: Path, cfg: Mapping[str, Any]) -> list[dict[str, float]]:
    summary: list[dict[str, float]] = []
    for estimator in sorted(set(str(r["estimator"]) for r in rows)):
        est_rows = [r for r in rows if str(r["estimator"]) == estimator]
        labels = np.asarray([int(r["planted"]) for r in est_rows], dtype=bool)
        scores = np.asarray([float(r["normalised_influence_score"]) for r in est_rows], dtype=np.float64)
        summary.append({"estimator": estimator, "auroc": float(auroc_score(labels, scores)), "rows": float(len(est_rows))})
    if summary:
        figures = ensure_dir(out_dir / "figures")
        fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
        labels = [str(r["estimator"]).upper() for r in summary]
        values = [float(r["auroc"]) for r in summary]
        ax.bar(labels, values, color=["#4c78a8", "#f58518"][: len(labels)])
        ax.set_ylim(0.0, 1.0)
        ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
        ax.set_title("Validation 1b: known-source ranking, IG vs swap")
        ax.set_xlabel("Estimator")
        ax.set_ylabel("AUROC for planted atoms")
        fig.savefig(figures / "validation_carriage_estimator_bakeoff.png", dpi=int(cfg["figures"]["dpi"]))
        fig.savefig(figures / "validation_carriage_estimator_bakeoff.pdf")
        plt.close(fig)
    return summary


def run_carriage_check(cfg: Mapping[str, Any], out_dir: Path, device: torch.device, seed: int) -> dict[str, float]:
    graphs, weight = make_carriage_dataset(cfg["carriage"], seed)
    adapter, history = train_small_gt(graphs, cfg["carriage"], device=device, seed=seed)
    baseline = baseline_for_graphs(graphs, device)
    rows, tensor_payload = carriage_reconstruction_rows(
        adapter,
        graphs,
        baseline=baseline,
        cfg=cfg["carriage"],
        device=device,
    )
    write_csv(out_dir / "metrics" / "carriage_reconstruction.csv", rows)
    write_csv(out_dir / "metrics" / "carriage_training.csv", history)
    metrics = plot_carriage(rows, history, out_dir, cfg)
    bakeoff_rows = carriage_estimator_bakeoff_rows(
        adapter,
        graphs,
        baseline=baseline,
        cfg=cfg["carriage"],
        device=device,
        seed=seed,
    )
    bakeoff_summary = plot_carriage_estimator_bakeoff(bakeoff_rows, out_dir, cfg)
    write_csv(out_dir / "metrics" / "carriage_estimator_bakeoff.csv", bakeoff_rows)
    write_csv(out_dir / "metrics" / "carriage_estimator_bakeoff_summary.csv", bakeoff_summary)
    for row in bakeoff_summary:
        metrics[f"{row['estimator']}_source_ranking_auroc"] = float(row["auroc"])
    write_json(out_dir / "metrics" / "carriage_summary.json", metrics)
    tensor_payload["target_weight"] = weight
    atomic_torch_save(out_dir / "tensors" / "carriage_check.pt", tensor_payload)
    return metrics


def run_patching_check(cfg: Mapping[str, Any], out_dir: Path, seed: int) -> dict[str, float]:
    rows: list[dict[str, Any]] = []
    feature_dim = int(cfg["patching"]["feature_dim"])
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 77)
    readout = torch.randn(feature_dim, generator=gen)
    for length in cfg["patching"]["chain_lengths"]:
        attachments = [min(max(1, int(a)), int(length) - 2) for a in cfg["patching"]["branch_attachments"]]
        graph = chain_graph_with_branches(
            length=int(length),
            branch_attachments=attachments,
            feature_dim=feature_dim,
            seed=int(seed) + int(length),
        )
        source = 0
        target = int(length) - 1
        chain_nodes = list(range(int(length)))
        cut = minimum_vertex_cut(graph, source, target)
        cut_node = cut[len(cut) // 2] if cut else int(length) // 2
        branch_nodes = list((graph.metadata or {}).get("branch_nodes", []))
        branch_node = branch_nodes[0] if branch_nodes else int(length)
        direct = DirectLinkAdapter(source, target, readout)
        step = StepByStepChainAdapter(source, target, readout, chain_nodes)
        for condition, adapter, clamp in [
            ("direct_clamp_cut", direct, [cut_node]),
            ("step_clamp_cut", step, [cut_node]),
            ("step_clamp_branch", step, [branch_node]),
        ]:
            clean = float(adapter.predict(graph).item())
            clamped = float(adapter.patch_hidden_states(graph, clamp).prediction.item())
            baseline_graph = graph.clone_with(x=graph.x.detach().clone())
            baseline_graph.x[source] = torch.zeros_like(baseline_graph.x[source])
            base_clean = float(adapter.predict(baseline_graph).item())
            base_clamped = float(adapter.patch_hidden_states(baseline_graph, clamp).prediction.item())
            unclamped_dep = clean - base_clean
            clamped_dep = clamped - base_clamped
            retained = abs(clamped_dep) / max(abs(unclamped_dep), 1.0e-12)
            rows.append(
                {
                    "chain_length": int(length),
                    "condition": condition,
                    "cut_node": int(cut_node),
                    "clamp_node": int(clamp[0]),
                    "unclamped_dependence": unclamped_dep,
                    "clamped_dependence": clamped_dep,
                    "retained": retained,
                }
            )
    write_csv(out_dir / "metrics" / "patching_retained.csv", rows)
    summary = plot_patching(rows, out_dir, cfg)
    write_json(out_dir / "metrics" / "patching_summary.json", summary)
    return summary


def plot_patching(rows: Sequence[Mapping[str, Any]], out_dir: Path, cfg: Mapping[str, Any]) -> dict[str, float]:
    order = ["direct_clamp_cut", "step_clamp_cut", "step_clamp_branch"]
    labels = ["Direct: clamp cut", "Composed: clamp cut", "Composed: clamp branch"]
    values = {key: [float(r["retained"]) for r in rows if r["condition"] == key] for key in order}
    means = [float(np.mean(values[key])) for key in order]
    ci = [bootstrap_ci(values[key], seed=3, draws=int(cfg["figures"]["bootstrap_draws"])) for key in order]
    err = [[m - lo for m, lo, hi in ci], [hi - m for m, lo, hi in ci]]
    fig, ax = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    ax.bar(labels, means, yerr=err, color=["#4c78a8", "#f58518", "#54a24b"], capsize=4)
    ax.axhline(0, color="#555555", linewidth=1.0)
    ax.axhline(1, color="#555555", linewidth=1.0, linestyle="--")
    ax.set_ylim(0, max(1.2, max(means) * 1.2))
    ax.set_ylabel("Dependence retained after clamp")
    ax.set_title("Validation 2: mediator patching separates direct and composed paths")
    for tick in ax.get_xticklabels():
        tick.set_rotation(10)
        tick.set_ha("right")
    figures = ensure_dir(out_dir / "figures")
    fig.savefig(figures / "validation_patching_check.png", dpi=int(cfg["figures"]["dpi"]))
    fig.savefig(figures / "validation_patching_check.pdf")
    plt.close(fig)
    return {f"{key}_mean": means[idx] for idx, key in enumerate(order)}


def structured_matrix(size: int, rank: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    mat = np.zeros((size, size), dtype=np.float64)
    for _ in range(int(rank)):
        a = rng.normal(size=size)
        b = rng.normal(size=size)
        a = a / max(float(np.linalg.norm(a)), 1.0e-12)
        b = b / max(float(np.linalg.norm(b)), 1.0e-12)
        mat += np.outer(a, b)
    mat += rng.normal(scale=float(sigma), size=(size, size))
    return mat


def run_rank_check(cfg: Mapping[str, Any], out_dir: Path, seed: int) -> dict[str, float]:
    rank_cfg = cfg["rank"]
    rng = np.random.default_rng(int(seed) + 500)
    rows: list[dict[str, Any]] = []
    size = int(rank_cfg["matrix_size"])
    for sigma in rank_cfg["noise_levels"]:
        for planted in rank_cfg["planted_ranks"]:
            for draw in range(int(rank_cfg["matrices_per_setting"])):
                mat = structured_matrix(size, int(planted), float(sigma), rng)
                rows.append(
                    {
                        "condition": "structured",
                        "planted_rank": int(planted),
                        "noise": float(sigma),
                        "draw": draw,
                        "effective_rank": effective_rank(mat, energy=float(rank_cfg["energy"])),
                        "top_share": top_singular_share(mat),
                        "above_null_margin": above_null_margin(
                            mat,
                            permutations=int(rank_cfg["null_permutations"]),
                            seed=int(seed) + draw,
                        ),
                    }
                )
        for draw in range(int(rank_cfg["matrices_per_setting"])):
            mat = structured_matrix(size, 0, float(sigma), rng)
            rows.append(
                {
                    "condition": "pure_noise",
                    "planted_rank": 0,
                    "noise": float(sigma),
                    "draw": draw,
                    "effective_rank": effective_rank(mat, energy=float(rank_cfg["energy"])),
                    "top_share": top_singular_share(mat),
                    "above_null_margin": above_null_margin(
                        mat,
                        permutations=int(rank_cfg["null_permutations"]),
                        seed=int(seed) + 900 + draw,
                    ),
                }
            )
    write_csv(out_dir / "metrics" / "rank_check.csv", rows)
    summary = plot_rank(rows, out_dir, cfg)
    write_json(out_dir / "metrics" / "rank_summary.json", summary)
    return summary


def plot_rank(rows: Sequence[Mapping[str, Any]], out_dir: Path, cfg: Mapping[str, Any]) -> dict[str, float]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    structured = [r for r in rows if r["condition"] == "structured"]
    noises = sorted({float(r["noise"]) for r in structured})
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for idx, noise in enumerate(noises):
        sub = [r for r in structured if float(r["noise"]) == noise]
        ranks = sorted({int(r["planted_rank"]) for r in sub})
        means = [float(np.mean([float(r["effective_rank"]) for r in sub if int(r["planted_rank"]) == rank])) for rank in ranks]
        axes[0].plot(ranks, means, marker="o", linewidth=1.8, label=f"noise {noise:g}", color=colors[idx % len(colors)])
    all_ranks = sorted({int(r["planted_rank"]) for r in structured})
    axes[0].plot(all_ranks, all_ranks, linestyle="--", color="#555555", label="ideal")
    axes[0].set_xlabel("Planted rank")
    axes[0].set_ylabel("Recovered effective rank (99% energy)")
    axes[0].set_title("Rank recovery")
    axes[0].legend(frameon=False)

    groups = ["structured", "pure_noise"]
    labels = ["Structured", "Pure noise"]
    values = {group: [float(r["above_null_margin"]) for r in rows if r["condition"] == group] for group in groups}
    means = [float(np.mean(values[group])) for group in groups]
    ci = [bootstrap_ci(values[group], seed=7, draws=int(cfg["figures"]["bootstrap_draws"])) for group in groups]
    err = [[m - lo for m, lo, hi in ci], [hi - m for m, lo, hi in ci]]
    axes[1].bar(labels, means, yerr=err, color=["#4c78a8", "#bab0ac"], capsize=4)
    axes[1].axhline(0, color="#555555", linewidth=1.0)
    axes[1].set_ylabel("Top singular share above column-permutation null")
    axes[1].set_title("Structure above noise")
    fig.suptitle("Validation 3: rank check", fontsize=13)
    figures = ensure_dir(out_dir / "figures")
    fig.savefig(figures / "validation_rank_check.png", dpi=int(cfg["figures"]["dpi"]))
    fig.savefig(figures / "validation_rank_check.pdf")
    plt.close(fig)
    return {
        "structured_above_null_mean": means[0],
        "pure_noise_above_null_mean": means[1],
    }


def interaction_targets(
    x_a: np.ndarray,
    x_b: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
) -> tuple[float, float]:
    a = float(np.dot(w_a, x_a))
    b = float(np.dot(w_b, x_b))
    return a + b, a * b


def run_interaction_check(cfg: Mapping[str, Any], out_dir: Path, seed: int) -> dict[str, float]:
    inter_cfg = cfg["interaction"]
    rng = np.random.default_rng(int(seed) + 800)
    dim = int(inter_cfg["feature_dim"])
    w_a = rng.normal(size=dim)
    w_b = rng.normal(size=dim)
    rows: list[dict[str, Any]] = []
    for draw in range(int(inter_cfg["pairs"])):
        x_a = rng.normal(size=dim)
        x_b = rng.normal(size=dim)
        new_a = rng.normal(size=dim)
        new_b = rng.normal(size=dim)
        for target_name in ["additive", "interacting"]:
            y = interaction_targets(x_a, x_b, w_a, w_b)[0 if target_name == "additive" else 1]
            y_a = interaction_targets(new_a, x_b, w_a, w_b)[0 if target_name == "additive" else 1]
            y_b = interaction_targets(x_a, new_b, w_a, w_b)[0 if target_name == "additive" else 1]
            y_ab = interaction_targets(new_a, new_b, w_a, w_b)[0 if target_name == "additive" else 1]
            delta_a = y_a - y
            delta_b = y_b - y
            delta_ab = y_ab - y
            ratio = non_additivity_ratio(delta_a, delta_b, delta_ab)
            rows.append(
                {
                    "draw": draw,
                    "target": target_name,
                    "delta_a": delta_a,
                    "delta_b": delta_b,
                    "delta_ab": delta_ab,
                    "sum_individual": delta_a + delta_b,
                    "non_additivity_ratio": ratio,
                }
            )
    write_csv(out_dir / "metrics" / "interaction_check.csv", rows)
    summary = plot_interaction(rows, out_dir, cfg)
    write_json(out_dir / "metrics" / "interaction_summary.json", summary)
    return summary


def plot_interaction(rows: Sequence[Mapping[str, Any]], out_dir: Path, cfg: Mapping[str, Any]) -> dict[str, float]:
    targets = ["additive", "interacting"]
    values = {
        target: [
            float(r["non_additivity_ratio"])
            for r in rows
            if r["target"] == target and math.isfinite(float(r["non_additivity_ratio"]))
        ]
        for target in targets
    }
    means = [float(np.mean(values[target])) for target in targets]
    ci = [bootstrap_ci(values[target], seed=11, draws=int(cfg["figures"]["bootstrap_draws"])) for target in targets]
    err = [[m - lo for m, lo, hi in ci], [hi - m for m, lo, hi in ci]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    axes[0].bar(["Additive", "Interacting"], means, yerr=err, color=["#4c78a8", "#e45756"], capsize=4)
    axes[0].set_ylabel("Non-additivity ratio")
    axes[0].set_title("Interacting targets do not add up")
    colors = {"additive": "#4c78a8", "interacting": "#e45756"}
    for target in targets:
        sub = [r for r in rows if r["target"] == target]
        x = np.asarray([float(r["sum_individual"]) for r in sub], dtype=np.float64)
        y = np.asarray([float(r["delta_ab"]) for r in sub], dtype=np.float64)
        axes[1].scatter(x, y, s=10, alpha=0.35, color=colors[target], label=target.capitalize(), edgecolors="none")
    all_xy = np.asarray(
        [float(r["sum_individual"]) for r in rows] + [float(r["delta_ab"]) for r in rows],
        dtype=np.float64,
    )
    lim = float(np.nanpercentile(np.abs(all_xy), 98)) if len(all_xy) else 1.0
    lim = max(lim, 1.0)
    axes[1].plot([-lim, lim], [-lim, lim], linestyle="--", color="#555555", linewidth=1.0)
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    axes[1].set_xlabel("delta_A + delta_B")
    axes[1].set_ylabel("delta_AB")
    axes[1].set_title("Joint effect vs additive prediction")
    axes[1].legend(frameon=False)
    fig.suptitle("Validation 4: interaction check", fontsize=13)
    figures = ensure_dir(out_dir / "figures")
    fig.savefig(figures / "validation_interaction_check.png", dpi=int(cfg["figures"]["dpi"]))
    fig.savefig(figures / "validation_interaction_check.pdf")
    plt.close(fig)
    return {"additive_mean": means[0], "interacting_mean": means[1]}


def run_all(config: Mapping[str, Any], *, force: bool = False) -> Path:
    seed = int(config["seed"])
    set_global_seed(seed)
    device = resolve_device(str(config.get("device", "auto")))
    artifact_root = ensure_dir(Path(str(config["artifact_root"])) / config_hash(config))
    if (artifact_root / "manifest.json").exists() and not force:
        return artifact_root
    ensure_dir(artifact_root / "metrics")
    ensure_dir(artifact_root / "tensors")
    ensure_dir(artifact_root / "figures")
    write_yaml(artifact_root / "config.yaml", config)

    summaries = {
        "carriage": run_carriage_check(config, artifact_root, device, seed),
        "patching": run_patching_check(config, artifact_root, seed),
        "rank": run_rank_check(config, artifact_root, seed),
        "interaction": run_interaction_check(config, artifact_root, seed),
    }
    write_json(artifact_root / "metrics" / "validation_summary.json", summaries)
    write_manifest(
        artifact_root,
        run_type="method_validation",
        config=config,
        adapter=AdapterInfo(
            name="method_validation_suite",
            version="validation.v1",
            implementation="mixed trained validation GT and analytic controls",
            validation_only=True,
        ),
        source_markdowns=[METHOD_VALIDATION_MD, MAIN_PROCEDURE_MD],
        extra={
            "summaries": summaries,
            "device": str(device),
            "source_markdown_expected_sha256": SOURCE_MARKDOWN_EXPECTED_SHA256,
        },
    )
    return artifact_root


def load_config(path: str | None, *, fast_dev_run: bool, output_root: str | None, device: str | None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        cfg = deep_update(cfg, read_yaml(path))
    if fast_dev_run:
        cfg = deep_update(cfg, FAST_DEV_OVERRIDES)
    if output_root:
        cfg["artifact_root"] = output_root
    if device:
        cfg["device"] = device
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-all", help="Run all four method validation checks.")
    run.add_argument("--config", type=str, default=None)
    run.add_argument("--output-root", type=str, default=None)
    run.add_argument("--device", type=str, default=None)
    run.add_argument("--fast-dev-run", action="store_true")
    run.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run-all":
        cfg = load_config(
            args.config,
            fast_dev_run=bool(args.fast_dev_run),
            output_root=args.output_root,
            device=args.device,
        )
        root = run_all(cfg, force=bool(args.force))
        print(f"[done] method validation artifacts: {root}", flush=True)
    else:  # pragma: no cover
        raise ValueError(args.command)


if __name__ == "__main__":  # pragma: no cover
    main()
