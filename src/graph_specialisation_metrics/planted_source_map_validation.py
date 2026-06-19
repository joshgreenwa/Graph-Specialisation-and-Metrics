"""Planted-dependency validation for graph permutation source maps.

This is a lightweight synthetic validation of the source-map estimator from the
long-range functional-analysis protocol.  It plants a known functional support
for every focal node, then asks whether content-swap source maps recover that
support from black-box output perturbations alone.

The important non-tautological check is used-vs-unused near-node AUROC: both
classes are topologically reachable, so a topology-only baseline is exactly
chance, while a functional source map should separate them.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1.0e-12


@dataclass(frozen=True)
class ValidationConfig:
    n_nodes: int = 48
    feature_dim: int = 8
    local_radius: int = 2
    geometric_radius: float = 0.30
    seed: int = 0
    backgrounds: int = 8
    partners_per_source: int = 24
    used_near_fraction: float = 0.5
    output_root: Path = Path("artifacts/planted_source_map_validation")
    variants: str = "tanh:0,tanh:0.03,linear:0"


@dataclass
class PlantedTeacher:
    graph: nx.Graph
    distances: np.ndarray
    weights: np.ndarray
    functional_mask: np.ndarray
    weight_norms: np.ndarray
    planted_far: dict[int, int]
    config: ValidationConfig


@dataclass(frozen=True)
class DistanceProfileConfig:
    n_nodes: int = 48
    feature_dim: int = 8
    local_radius: int = 2
    far_offset: int = 8
    heads: int = 4
    seed: int = 0
    train_size: int = 4096
    val_size: int = 1024
    test_size: int = 1024
    batch_size: int = 512
    max_epochs: int = 500
    patience: int = 40
    lr: float = 3.0e-3
    weight_decay: float = 1.0e-5
    source_map_backgrounds: int = 8
    source_map_partners: int = 24
    device: str = "auto"
    output_root: Path = Path("artifacts/planted_source_map_validation")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def connected_geometric_graph(config: ValidationConfig) -> nx.Graph:
    """Generate a connected graph with finite far nodes for every focal."""

    rng = np.random.default_rng(int(config.seed) + 11)
    for attempt in range(500):
        graph_seed = int(rng.integers(0, 2**31 - 1))
        graph = nx.random_geometric_graph(
            int(config.n_nodes),
            float(config.geometric_radius),
            seed=graph_seed,
        )
        if not nx.is_connected(graph):
            continue
        distances = distance_matrix(graph)
        has_far = np.all(np.sum(distances > int(config.local_radius), axis=1) > 0)
        has_near = np.all(np.sum((distances > 0) & (distances <= int(config.local_radius)), axis=1) > 0)
        if has_far and has_near:
            graph.graph["seed"] = graph_seed
            return graph
    raise RuntimeError(
        "could not sample a connected graph with both near and far candidates for every node"
    )


def distance_matrix(graph: nx.Graph) -> np.ndarray:
    n = graph.number_of_nodes()
    out = np.full((n, n), np.inf, dtype=float)
    for source, lengths in nx.all_pairs_shortest_path_length(graph):
        for target, length in lengths.items():
            out[int(source), int(target)] = float(length)
    return out


def plant_teacher(config: ValidationConfig) -> PlantedTeacher:
    rng = np.random.default_rng(int(config.seed) + 101)
    graph = connected_geometric_graph(config)
    distances = distance_matrix(graph)
    n = int(config.n_nodes)
    d = int(config.feature_dim)
    weights = np.zeros((n, n, d, d), dtype=float)
    mask = np.zeros((n, n), dtype=bool)
    planted_far: dict[int, int] = {}
    for focal in range(n):
        near = [
            node
            for node in range(n)
            if node != focal and 0 < distances[focal, node] <= int(config.local_radius)
        ]
        used_count = max(1, int(round(float(config.used_near_fraction) * len(near))))
        if len(near) > 1:
            used_count = min(used_count, len(near) - 1)
        used_near = list(rng.choice(near, size=used_count, replace=False)) if near else []
        far = [
            node
            for node in range(n)
            if node != focal and np.isfinite(distances[focal, node]) and distances[focal, node] > int(config.local_radius)
        ]
        if not far:
            raise RuntimeError(f"focal node {focal} has no finite far candidate")
        far_node = int(rng.choice(far))
        planted_far[focal] = far_node
        for source in [*used_near, far_node]:
            strength = float(rng.lognormal(mean=0.0, sigma=0.35))
            weights[focal, source] = strength * rng.standard_normal((d, d)) / math.sqrt(float(d))
            mask[focal, source] = True
    norms = np.linalg.norm(weights.reshape(n, n, -1), axis=2)
    return PlantedTeacher(
        graph=graph,
        distances=distances,
        weights=weights,
        functional_mask=mask,
        weight_norms=norms,
        planted_far=planted_far,
        config=config,
    )


def evaluate_teacher(
    teacher: PlantedTeacher,
    x: np.ndarray,
    *,
    nonlinearity: str,
    output_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    pre = np.einsum("ijod,bjd->bio", teacher.weights, x)
    if nonlinearity == "linear":
        y = pre
    elif nonlinearity == "tanh":
        y = np.tanh(pre)
    else:
        raise ValueError(f"unknown nonlinearity {nonlinearity!r}")
    if float(output_noise) > 0.0:
        y = y + float(output_noise) * rng.standard_normal(y.shape)
    return y


def sample_partner_records(
    n_nodes: int,
    partners_per_source: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    records: list[tuple[int, int]] = []
    nodes = np.arange(int(n_nodes))
    for source in range(int(n_nodes)):
        candidates = nodes[nodes != int(source)]
        count = min(int(partners_per_source), int(candidates.size))
        partners = rng.choice(candidates, size=count, replace=False)
        for partner in partners:
            records.append((int(source), int(partner)))
    return records


def compute_swap_source_map(
    teacher: PlantedTeacher,
    *,
    nonlinearity: str,
    output_noise: float,
    backgrounds: int,
    partners_per_source: int,
    seed: int,
) -> np.ndarray:
    """Compute partner-marginalised RMS swap source map for all focals.

    The returned matrix has shape [receiver, source].  Entry (i,w) is the RMS
    direct output response at focal i to swaps (w,p), divided by the swapped
    content distance and averaged over clean backgrounds and partners.
    """

    rng = np.random.default_rng(int(seed))
    n = int(teacher.config.n_nodes)
    d = int(teacher.config.feature_dim)
    accum = np.zeros((n, n), dtype=float)
    counts = np.zeros(n, dtype=float)
    for _background in range(int(backgrounds)):
        x0 = rng.standard_normal((n, d))
        eval_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        y0 = evaluate_teacher(
            teacher,
            x0[None, :, :],
            nonlinearity=nonlinearity,
            output_noise=output_noise,
            rng=eval_rng,
        )[0]
        pair_records = sample_partner_records(n, int(partners_per_source), rng)
        swapped = np.repeat(x0[None, :, :], len(pair_records), axis=0)
        distances = np.zeros(len(pair_records), dtype=float)
        for idx, (source, partner) in enumerate(pair_records):
            swapped[idx, source] = x0[partner]
            swapped[idx, partner] = x0[source]
            distances[idx] = max(float(np.linalg.norm(x0[source] - x0[partner])), EPS)
        eval_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        y_swap = evaluate_teacher(
            teacher,
            swapped,
            nonlinearity=nonlinearity,
            output_noise=output_noise,
            rng=eval_rng,
        )
        deltas = np.linalg.norm(y_swap - y0[None, :, :], axis=2) / distances[:, None]
        for idx, (source, _partner) in enumerate(pair_records):
            accum[:, source] += deltas[idx] ** 2
            counts[source] += 1.0
    return np.sqrt(accum / np.maximum(counts[None, :], 1.0))


def average_ranks(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    npos = int(labels.sum())
    nneg = int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    return float((ranks[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    ar = average_ranks(a[valid])
    br = average_ranks(b[valid])
    if np.std(ar) <= EPS or np.std(br) <= EPS:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def reciprocal_rank(scores: np.ndarray, target: int, candidates: Sequence[int]) -> float:
    ordered = sorted(candidates, key=lambda node: float(scores[node]), reverse=True)
    for rank, node in enumerate(ordered, start=1):
        if int(node) == int(target):
            return 1.0 / float(rank)
    return 0.0


def summarise_source_map(
    teacher: PlantedTeacher,
    source_map: np.ndarray,
    *,
    variant: str,
    nonlinearity: str,
    output_noise: float,
) -> dict[str, Any]:
    n = int(teacher.config.n_nodes)
    off_diag = ~np.eye(n, dtype=bool)
    near = (teacher.distances > 0) & (teacher.distances <= int(teacher.config.local_radius))
    functional_auroc = auroc(teacher.functional_mask[off_diag], source_map[off_diag])
    near_auroc = auroc(teacher.functional_mask[near], source_map[near])
    topology_near_baseline = auroc(teacher.functional_mask[near], np.ones(int(near.sum())))
    far_top1_hits = []
    far_mrr = []
    for focal, target in teacher.planted_far.items():
        candidates = [
            node
            for node in range(n)
            if node != focal
            and np.isfinite(teacher.distances[focal, node])
            and teacher.distances[focal, node] > int(teacher.config.local_radius)
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda node: float(source_map[focal, node]))
        far_top1_hits.append(float(int(best) == int(target)))
        far_mrr.append(reciprocal_rank(source_map[focal], target, candidates))
    used = teacher.functional_mask
    return {
        "variant": variant,
        "nonlinearity": nonlinearity,
        "output_noise": float(output_noise),
        "functional_mask_auroc": functional_auroc,
        "used_vs_unused_near_auroc": near_auroc,
        "topology_only_near_auroc": topology_near_baseline,
        "planted_far_top1": float(np.mean(far_top1_hits)) if far_top1_hits else float("nan"),
        "planted_far_mrr": float(np.mean(far_mrr)) if far_mrr else float("nan"),
        "graded_spearman_used_vs_weight_norm": spearman(source_map[used], teacher.weight_norms[used]),
        "interpret_graded_metric": bool(nonlinearity == "linear" and float(output_noise) == 0.0),
    }


def parse_variants(text: str) -> list[tuple[str, str, float]]:
    variants = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            nonlinearity, noise_text = raw.split(":", 1)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("variants must use format 'tanh:0,tanh:0.03,linear:0'") from exc
        nonlinearity = nonlinearity.strip()
        if nonlinearity not in {"tanh", "linear"}:
            raise argparse.ArgumentTypeError("variant nonlinearity must be tanh or linear")
        noise = float(noise_text)
        suffix = "deterministic" if noise == 0.0 else f"noise_{noise:g}".replace(".", "p")
        variants.append((f"{nonlinearity}_{suffix}", nonlinearity, noise))
    if not variants:
        raise argparse.ArgumentTypeError("at least one variant is required")
    return variants


def import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def plot_metrics(root: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    plt = import_plotting()
    labels = [str(row["variant"]).replace("_", "\n") for row in rows]
    x = np.arange(len(rows), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.55))

    width = 0.32
    axes[0].bar(
        x - width / 2,
        [float(row["functional_mask_auroc"]) for row in rows],
        width=width,
        label="all used vs unused",
        color="#4f78b5",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].bar(
        x + width / 2,
        [float(row["used_vs_unused_near_auroc"]) for row in rows],
        width=width,
        label="near used vs reachable-unused",
        color="#c2473f",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].axhline(0.5, color="#555555", linewidth=1.0, linestyle="--", label="topology-only chance")
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0.45, 1.02)
    axes[0].set_title("Functional support recovery")
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].bar(
        x - width / 2,
        [float(row["planted_far_top1"]) for row in rows],
        width=width,
        label="top-1",
        color="#7a5aa6",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].bar(
        x + width / 2,
        [float(row["planted_far_mrr"]) for row in rows],
        width=width,
        label="MRR",
        color="#b89ad9",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].set_ylabel("score")
    axes[1].set_ylim(0.0, 1.03)
    axes[1].set_title("Planted far-source recovery")
    axes[1].legend(frameon=False, loc="lower right")

    spearman_values = [float(row["graded_spearman_used_vs_weight_norm"]) for row in rows]
    colours = ["#4f8f5b" if str(row["interpret_graded_metric"]) == "True" else "#bdbdbd" for row in rows]
    axes[2].bar(x, spearman_values, color=colours, edgecolor="black", linewidth=0.5)
    axes[2].axhline(0.0, color="#555555", linewidth=1.0)
    axes[2].set_ylabel("Spearman")
    axes[2].set_ylim(-0.1, 1.0)
    axes[2].set_title("Graded magnitude check")
    axes[2].text(
        0.02,
        0.04,
        "green: interpretable linear variant\n grey: reported, not load-bearing",
        transform=axes[2].transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color="#444444",
    )

    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.suptitle("Planted-Dependency Validation of Graph Permutation Source Maps", y=1.04, fontsize=13)
    fig.tight_layout()
    path = ensure_dir(root / "figures") / "planted_source_map_validation_metrics.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def ring_distances(n_nodes: int) -> np.ndarray:
    idx = np.arange(int(n_nodes))
    diff = np.abs(idx[:, None] - idx[None, :])
    return np.minimum(diff, int(n_nodes) - diff).astype(float)


def plant_ring_profile_teacher(config: DistanceProfileConfig) -> PlantedTeacher:
    rng = np.random.default_rng(int(config.seed) + 301)
    n = int(config.n_nodes)
    d = int(config.feature_dim)
    graph = nx.cycle_graph(n)
    distances = ring_distances(n)
    weights = np.zeros((n, n, d, d), dtype=float)
    mask = np.zeros((n, n), dtype=bool)
    planted_far: dict[int, int] = {}
    validation_config = ValidationConfig(
        n_nodes=n,
        feature_dim=d,
        local_radius=int(config.local_radius),
        seed=int(config.seed),
        backgrounds=int(config.source_map_backgrounds),
        partners_per_source=int(config.source_map_partners),
        output_root=Path(config.output_root),
    )
    for focal in range(n):
        near = [
            node
            for node in range(n)
            if node != focal and 0 < distances[focal, node] <= int(config.local_radius)
        ]
        # Use a deterministic alternating subset so every focal has reachable
        # unused neighbours; this keeps the source-map task functional rather
        # than topological.
        used_near = [node for idx, node in enumerate(sorted(near)) if idx % 2 == 0]
        far_node = int((focal + int(config.far_offset)) % n)
        if distances[focal, far_node] <= int(config.local_radius):
            raise ValueError("--far-offset must be beyond --local-radius on the ring")
        planted_far[focal] = far_node
        for source in [*used_near, far_node]:
            if source == far_node:
                strength = 1.4
            else:
                strength = float(rng.uniform(0.65, 1.0))
            weights[focal, source] = strength * rng.standard_normal((d, d)) / math.sqrt(float(d))
            mask[focal, source] = True
    norms = np.linalg.norm(weights.reshape(n, n, -1), axis=2)
    return PlantedTeacher(
        graph=graph,
        distances=distances,
        weights=weights,
        functional_mask=mask,
        weight_norms=norms,
        planted_far=planted_far,
        config=validation_config,
    )


def torch_teacher_output(weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ijod,bjd->bio", weights, x)


class FullTransportProfileModel(nn.Module):
    def __init__(self, n_nodes: int, feature_dim: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(int(n_nodes), int(n_nodes), int(feature_dim), int(feature_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch_teacher_output(self.weights, x)


class LocalRoutingOnlyProfileModel(nn.Module):
    def __init__(self, distances: np.ndarray, *, feature_dim: int, heads: int, local_radius: int) -> None:
        super().__init__()
        n = int(distances.shape[0])
        d = int(feature_dim)
        h = int(heads)
        support = (distances > 0) & (distances <= int(local_radius))
        self.register_buffer("support", torch.as_tensor(support, dtype=torch.bool))
        self.logits = nn.Parameter(torch.zeros(h, n, n))
        self.values = nn.Parameter(torch.randn(h, d, d) / math.sqrt(float(d)))
        self.output = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.output.weight.copy_(torch.eye(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = int(self.values.shape[0])
        logits = self.logits.masked_fill(~self.support.unsqueeze(0), -1.0e9)
        attn = torch.softmax(logits, dim=-1)
        msg = torch.einsum("bjd,hdo->bhjo", x, self.values)
        mixed = torch.einsum("hij,bhjo->bio", attn, msg)
        return self.output(mixed / math.sqrt(float(h)))


def resolve_torch_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_torch_dataset(
    teacher: PlantedTeacher,
    *,
    size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn(int(size), int(teacher.config.n_nodes), int(teacher.config.feature_dim), generator=gen)
    weights = torch.as_tensor(teacher.weights, dtype=torch.float32)
    y = torch_teacher_output(weights, x)
    return x.to(device), y.to(device)


@torch.no_grad()
def rel_mse_torch(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
    model.eval()
    mse = 0.0
    denom = 0.0
    for start in range(0, int(x.shape[0]), int(batch_size)):
        pred = model(x[start : start + int(batch_size)])
        target = y[start : start + int(batch_size)]
        mse += float(((pred - target) ** 2).sum().detach().cpu())
        denom += float((target**2).sum().detach().cpu())
    return mse / max(denom, EPS)


def train_profile_model(
    *,
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    config: DistanceProfileConfig,
    device: torch.device,
    checkpoint_path: Path,
) -> dict[str, Any]:
    complete_path = checkpoint_path.with_suffix(".json")
    if checkpoint_path.exists() and complete_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
        return {"skipped": True, **read_json(complete_path)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    gen = torch.Generator(device="cpu").manual_seed(int(config.seed) + 9021)
    for epoch in range(1, int(config.max_epochs) + 1):
        model.train()
        order = torch.randperm(int(train_x.shape[0]), generator=gen).to(train_x.device)
        for start in range(0, int(train_x.shape[0]), int(config.batch_size)):
            idx = order[start : start + int(config.batch_size)]
            pred = model(train_x[idx])
            loss = F.mse_loss(pred, train_y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val = rel_mse_torch(model, val_x, val_y, int(config.batch_size))
        if val < best_val - 1.0e-8:
            best_val = val
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 25 == 0:
            print(f"[profile-train] epoch={epoch:03d} val_rel_mse={val:.5g} best={best_val:.5g}@{best_epoch}")
        if epoch - best_epoch >= int(config.patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    ensure_dir(checkpoint_path.parent)
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)
    summary = {"best_epoch": int(best_epoch), "best_val_rel_mse": float(best_val)}
    write_json(complete_path, summary)
    return summary


def fit_full_transport_closed_form(
    *,
    model: FullTransportProfileModel,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    checkpoint_path: Path,
) -> dict[str, Any]:
    complete_path = checkpoint_path.with_suffix(".json")
    if checkpoint_path.exists() and complete_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
        return {"skipped": True, **read_json(complete_path)}
    n = int(train_x.shape[1])
    d = int(train_x.shape[2])
    x_flat = train_x.detach().cpu().numpy().reshape(int(train_x.shape[0]), n * d)
    y_flat = train_y.detach().cpu().numpy().reshape(int(train_y.shape[0]), n * d)
    coef, *_ = np.linalg.lstsq(x_flat, y_flat, rcond=None)
    weights = coef.reshape(n, d, n, d).transpose(2, 0, 3, 1).copy()
    with torch.no_grad():
        model.weights.copy_(torch.as_tensor(weights, dtype=torch.float32, device=device))
    val = rel_mse_torch(model, val_x, val_y, int(batch_size))
    ensure_dir(checkpoint_path.parent)
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)
    summary = {"best_epoch": 0, "best_val_rel_mse": float(val), "fit_method": "least_squares"}
    write_json(complete_path, summary)
    print(f"[profile-fit] full relation transport least-squares val_rel_mse={val:.5g}")
    return summary


def read_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    import json

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


@torch.no_grad()
def torch_model_predictor(model: nn.Module, device: torch.device, batch_size: int):
    def predict(x: np.ndarray) -> np.ndarray:
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, int(x.shape[0]), int(batch_size)):
                xb = torch.as_tensor(x[start : start + int(batch_size)], dtype=torch.float32, device=device)
                preds.append(model(xb).detach().cpu().numpy())
        return np.concatenate(preds, axis=0)

    return predict


def compute_swap_source_map_from_predictor(
    *,
    n_nodes: int,
    feature_dim: int,
    predictor: Any,
    backgrounds: int,
    partners_per_source: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    accum = np.zeros((int(n_nodes), int(n_nodes)), dtype=float)
    counts = np.zeros(int(n_nodes), dtype=float)
    for _background in range(int(backgrounds)):
        x0 = rng.standard_normal((int(n_nodes), int(feature_dim)))
        y0 = predictor(x0[None, :, :])[0]
        pair_records = sample_partner_records(int(n_nodes), int(partners_per_source), rng)
        swapped = np.repeat(x0[None, :, :], len(pair_records), axis=0)
        distances = np.zeros(len(pair_records), dtype=float)
        for idx, (source, partner) in enumerate(pair_records):
            swapped[idx, source] = x0[partner]
            swapped[idx, partner] = x0[source]
            distances[idx] = max(float(np.linalg.norm(x0[source] - x0[partner])), EPS)
        y_swap = predictor(swapped)
        deltas = np.linalg.norm(y_swap - y0[None, :, :], axis=2) / distances[:, None]
        for idx, (source, _partner) in enumerate(pair_records):
            accum[:, source] += deltas[idx] ** 2
            counts[source] += 1.0
    return np.sqrt(accum / np.maximum(counts[None, :], 1.0))


def floor_correct_source_map(source_map: np.ndarray) -> np.ndarray:
    values = np.asarray(source_map, dtype=float) ** 2
    corrected = np.zeros_like(values)
    n = values.shape[0]
    for focal in range(n):
        mask = np.ones(n, dtype=bool)
        mask[focal] = False
        floor = float(np.min(values[focal, mask]))
        corrected[focal, mask] = np.maximum(values[focal, mask] - floor, 0.0)
    return corrected


def distance_mass_profile(source_map: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corrected = floor_correct_source_map(source_map)
    off_diag = distances > 0
    total = float(corrected[off_diag].sum())
    max_dist = int(np.nanmax(distances[np.isfinite(distances)]))
    xs = np.arange(1, max_dist + 1)
    ys = []
    for dist in xs:
        mask = distances == float(dist)
        ys.append(float(corrected[mask].sum() / max(total, EPS)))
    return xs, np.asarray(ys, dtype=float)


def plot_distance_profile(
    root: Path,
    teacher: PlantedTeacher,
    profiles: Mapping[str, tuple[np.ndarray, np.ndarray]],
    metrics: Sequence[Mapping[str, Any]],
    config: DistanceProfileConfig,
) -> Path:
    plt = import_plotting()
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.05))
    styles = {
        "Oracle": ("#111111", "-", 2.4),
        "Full relation transport": ("#c2473f", "-", 2.2),
        "Local routing-only": ("#4f78b5", "--", 2.2),
    }
    for label, (xs, ys) in profiles.items():
        color, linestyle, linewidth = styles[label]
        ax.plot(xs, ys, marker="o", color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    ax.axvline(int(config.far_offset), color="#777777", linestyle=":", linewidth=1.2)
    ax.text(
        int(config.far_offset) + 0.12,
        0.96,
        "planted far source",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8,
        color="#555555",
    )
    ax.set_xlabel("hop distance from receiver")
    ax.set_ylabel("source-map mass (baseline-corrected)")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("GPI Source Maps Reveal Learned Long-Range Dependence")
    text = ""
    if metrics:
        validation = metrics[0]
        text = (
            f"near AUROC={float(validation['used_vs_unused_near_auroc']):.2f} "
            f"(topology={float(validation['topology_only_near_auroc']):.1f}); "
            f"far top-1={float(validation['planted_far_top1']):.2f}"
        )
        ax.text(0.02, 0.02, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="#333333")
    fig.tight_layout()
    path = ensure_dir(root / "figures") / "gpi_distance_profile_trained_models.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def run_distance_profile(config: DistanceProfileConfig) -> None:
    root = ensure_dir(Path(config.output_root))
    device = resolve_torch_device(str(config.device))
    teacher = plant_ring_profile_teacher(config)
    train_x, train_y = make_torch_dataset(teacher, size=int(config.train_size), seed=int(config.seed) + 1, device=device)
    val_x, val_y = make_torch_dataset(teacher, size=int(config.val_size), seed=int(config.seed) + 2, device=device)
    test_x, test_y = make_torch_dataset(teacher, size=int(config.test_size), seed=int(config.seed) + 3, device=device)
    full = FullTransportProfileModel(int(config.n_nodes), int(config.feature_dim)).to(device)
    local = LocalRoutingOnlyProfileModel(
        teacher.distances,
        feature_dim=int(config.feature_dim),
        heads=int(config.heads),
        local_radius=int(config.local_radius),
    ).to(device)
    ckpt_root = ensure_dir(root / "checkpoints" / "distance_profile")
    full_summary = fit_full_transport_closed_form(
        model=full,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        batch_size=int(config.batch_size),
        device=device,
        checkpoint_path=ckpt_root / "full_relation_transport.pt",
    )
    local_summary = train_profile_model(
        model=local,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        config=config,
        device=device,
        checkpoint_path=ckpt_root / "local_routing_only.pt",
    )
    full_test = rel_mse_torch(full, test_x, test_y, int(config.batch_size))
    local_test = rel_mse_torch(local, test_x, test_y, int(config.batch_size))
    print(f"[profile] full_transport test_rel_mse={full_test:.5g}")
    print(f"[profile] local_routing_only test_rel_mse={local_test:.5g}")

    oracle_predict = lambda x: evaluate_teacher(
        teacher,
        x,
        nonlinearity="linear",
        output_noise=0.0,
        rng=np.random.default_rng(0),
    )
    oracle_map = compute_swap_source_map_from_predictor(
        n_nodes=int(config.n_nodes),
        feature_dim=int(config.feature_dim),
        predictor=oracle_predict,
        backgrounds=int(config.source_map_backgrounds),
        partners_per_source=int(config.source_map_partners),
        seed=int(config.seed) + 7001,
    )
    full_map = compute_swap_source_map_from_predictor(
        n_nodes=int(config.n_nodes),
        feature_dim=int(config.feature_dim),
        predictor=torch_model_predictor(full, device, int(config.batch_size)),
        backgrounds=int(config.source_map_backgrounds),
        partners_per_source=int(config.source_map_partners),
        seed=int(config.seed) + 7001,
    )
    local_map = compute_swap_source_map_from_predictor(
        n_nodes=int(config.n_nodes),
        feature_dim=int(config.feature_dim),
        predictor=torch_model_predictor(local, device, int(config.batch_size)),
        backgrounds=int(config.source_map_backgrounds),
        partners_per_source=int(config.source_map_partners),
        seed=int(config.seed) + 7001,
    )
    profiles = {
        "Oracle": distance_mass_profile(oracle_map, teacher.distances),
        "Full relation transport": distance_mass_profile(full_map, teacher.distances),
        "Local routing-only": distance_mass_profile(local_map, teacher.distances),
    }
    metric_rows = []
    for model, summary, test, source_map in [
        ("full_relation_transport", full_summary, full_test, full_map),
        ("local_routing_only", local_summary, local_test, local_map),
    ]:
        row = summarise_source_map(
            teacher,
            source_map,
            variant=model,
            nonlinearity="linear",
            output_noise=0.0,
        )
        row.update(
            {
                "model": model,
                "test_rel_mse": float(test),
                "best_epoch": int(summary["best_epoch"]),
                "best_val_rel_mse": float(summary["best_val_rel_mse"]),
            }
        )
        metric_rows.append(row)
    oracle_metrics = [
        summarise_source_map(
            teacher,
            oracle_map,
            variant="oracle",
            nonlinearity="linear",
            output_noise=0.0,
        )
    ]
    write_csv(root / "metrics" / "gpi_distance_profile_metrics.csv", [*oracle_metrics, *metric_rows])
    np.savez_compressed(
        ensure_dir(root / "arrays") / "gpi_distance_profile_arrays.npz",
        distances=teacher.distances,
        functional_mask=teacher.functional_mask,
        oracle_source_map=oracle_map,
        full_transport_source_map=full_map,
        local_routing_source_map=local_map,
    )
    plot_distance_profile(root, teacher, profiles, oracle_metrics, config)


def run_validation(config: ValidationConfig) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    root = ensure_dir(Path(config.output_root))
    teacher = plant_teacher(config)
    variants = parse_variants(config.variants)
    rows: list[dict[str, Any]] = []
    maps: dict[str, np.ndarray] = {}
    for index, (variant, nonlinearity, output_noise) in enumerate(variants):
        source_map = compute_swap_source_map(
            teacher,
            nonlinearity=nonlinearity,
            output_noise=output_noise,
            backgrounds=int(config.backgrounds),
            partners_per_source=int(config.partners_per_source),
            seed=int(config.seed) + 1009 * (index + 1),
        )
        maps[variant] = source_map
        row = summarise_source_map(
            teacher,
            source_map,
            variant=variant,
            nonlinearity=nonlinearity,
            output_noise=output_noise,
        )
        row.update(
            {
                "n_nodes": int(config.n_nodes),
                "feature_dim": int(config.feature_dim),
                "local_radius": int(config.local_radius),
                "backgrounds": int(config.backgrounds),
                "partners_per_source": int(config.partners_per_source),
                "graph_seed": teacher.graph.graph.get("seed", ""),
            }
        )
        rows.append(row)
        print(
            f"[validation] {variant}: near_AUROC={row['used_vs_unused_near_auroc']:.3f} "
            f"topology={row['topology_only_near_auroc']:.3f} "
            f"far_top1={row['planted_far_top1']:.3f} "
            f"spearman={row['graded_spearman_used_vs_weight_norm']:.3f}"
        )
    write_csv(root / "metrics" / "planted_source_map_validation_metrics.csv", rows)
    np.savez_compressed(
        ensure_dir(root / "arrays") / "planted_source_map_validation_arrays.npz",
        distances=teacher.distances,
        functional_mask=teacher.functional_mask,
        weight_norms=teacher.weight_norms,
        **{f"source_map_{name}": value for name, value in maps.items()},
    )
    plot_metrics(root, rows)
    return rows, maps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-distance-profile", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ValidationConfig.output_root)
    parser.add_argument("--n-nodes", type=int, default=ValidationConfig.n_nodes)
    parser.add_argument("--feature-dim", type=int, default=ValidationConfig.feature_dim)
    parser.add_argument("--local-radius", type=int, default=ValidationConfig.local_radius)
    parser.add_argument("--geometric-radius", type=float, default=ValidationConfig.geometric_radius)
    parser.add_argument("--seed", type=int, default=ValidationConfig.seed)
    parser.add_argument("--backgrounds", type=int, default=ValidationConfig.backgrounds)
    parser.add_argument("--partners-per-source", type=int, default=ValidationConfig.partners_per_source)
    parser.add_argument("--used-near-fraction", type=float, default=ValidationConfig.used_near_fraction)
    parser.add_argument("--variants", type=str, default=ValidationConfig.variants)
    parser.add_argument("--far-offset", type=int, default=DistanceProfileConfig.far_offset)
    parser.add_argument("--heads", type=int, default=DistanceProfileConfig.heads)
    parser.add_argument("--train-size", type=int, default=DistanceProfileConfig.train_size)
    parser.add_argument("--val-size", type=int, default=DistanceProfileConfig.val_size)
    parser.add_argument("--test-size", type=int, default=DistanceProfileConfig.test_size)
    parser.add_argument("--batch-size", type=int, default=DistanceProfileConfig.batch_size)
    parser.add_argument("--max-epochs", type=int, default=DistanceProfileConfig.max_epochs)
    parser.add_argument("--patience", type=int, default=DistanceProfileConfig.patience)
    parser.add_argument("--lr", type=float, default=DistanceProfileConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=DistanceProfileConfig.weight_decay)
    parser.add_argument("--source-map-backgrounds", type=int, default=DistanceProfileConfig.source_map_backgrounds)
    parser.add_argument("--source-map-partners", type=int, default=DistanceProfileConfig.source_map_partners)
    parser.add_argument("--device", type=str, default=DistanceProfileConfig.device)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_distance_profile:
        config = DistanceProfileConfig(
            n_nodes=int(args.n_nodes),
            feature_dim=int(args.feature_dim),
            local_radius=int(args.local_radius),
            far_offset=int(args.far_offset),
            heads=int(args.heads),
            seed=int(args.seed),
            train_size=int(args.train_size),
            val_size=int(args.val_size),
            test_size=int(args.test_size),
            batch_size=int(args.batch_size),
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            source_map_backgrounds=int(args.source_map_backgrounds),
            source_map_partners=int(args.source_map_partners),
            device=str(args.device),
            output_root=Path(args.output_root),
        )
        run_distance_profile(config)
    else:
        config = ValidationConfig(
            n_nodes=int(args.n_nodes),
            feature_dim=int(args.feature_dim),
            local_radius=int(args.local_radius),
            geometric_radius=float(args.geometric_radius),
            seed=int(args.seed),
            backgrounds=int(args.backgrounds),
            partners_per_source=int(args.partners_per_source),
            used_near_fraction=float(args.used_near_fraction),
            output_root=Path(args.output_root),
            variants=str(args.variants),
        )
        run_validation(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
