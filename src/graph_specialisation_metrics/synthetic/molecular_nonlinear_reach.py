"""Semi-synthetic nonlinear reach on real molecular graph supports.

The experiment uses ZINC molecular topologies and deterministic binary semantic
payloads.  A node-level target combines a local linear channel with a distant
saturated channel::

    y_v = a x_v^(local) + b mean_{u: d(u,v)=R} phi_k(x_u^(far)),
    phi_k(x) = tanh(k x) / tanh(k).

All observed payloads are in ``{-1, +1}``, so ``phi_k(x) == x`` for every
finite ``k``.  Consequently the clean predictions, full donor swaps, and exact
target are invariant to ``k``.  The local derivative of the distant channel is
not: it vanishes as ``k`` grows.  This gives a behaviour-preserving test of
whether a differential range estimator depends on an arbitrary off-manifold
extension of categorical molecular inputs.

The learned model is a lightweight shortest-path radial graph filter.  It is
trained rather than assigned the target coefficients, and its weights span all
distances up to ``max_distance``.  The exact oracle, literal coordinatewise
Jacobian range, dose-matched directional tangent, and finite donor-swap response
are then evaluated on held-out molecular graphs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn


PROTOCOL_VERSION = "molecular-nonlinear-reach-v1"
METHOD_LABELS = {
    "oracle": "Known target",
    "bamberger": "Bamberger Jacobian",
    "tangent": "Dose-matched tangent",
    "finite": "Functional carriage",
}
METHOD_COLOURS = {
    "oracle": "#222222",
    "bamberger": "#0072B2",
    "tangent": "#009E73",
    "finite": "#D55E00",
}
METHOD_STYLES = {
    "oracle": ("-", "o"),
    "bamberger": ("--", "^"),
    "tangent": ("-.", "D"),
    "finite": (":", "s"),
}


@dataclass(frozen=True)
class Config:
    output_dir: Path
    data_root: Path
    train_graphs: int = 512
    val_graphs: int = 96
    test_graphs: int = 96
    sources_per_graph: int = 0
    target_distance: int = 4
    max_distance: int = 8
    local_weight: float = 0.5
    far_weight: float = 1.0
    train_kappa: float = 4.0
    kappas: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    train_steps: int = 600
    learning_rate: float = 0.05
    weight_decay: float = 1.0e-6
    bootstrap_replicates: int = 2_000
    analysis_seed: int = 91_021

    @property
    def fingerprint(self) -> str:
        payload = asdict(self)
        payload.pop("output_dir")
        payload.pop("data_root")
        payload["protocol_version"] = PROTOCOL_VERSION
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]

    def validate(self) -> None:
        if min(self.train_graphs, self.val_graphs, self.test_graphs) < 1:
            raise ValueError("every split must contain at least one graph")
        if self.sources_per_graph < 0:
            raise ValueError("sources_per_graph must be non-negative; zero means all nodes")
        if not (1 <= self.target_distance <= self.max_distance):
            raise ValueError("target_distance must lie in [1, max_distance]")
        if self.train_steps < 1 or self.learning_rate <= 0:
            raise ValueError("invalid optimisation settings")
        if not self.kappas or min(self.kappas) < 0:
            raise ValueError("kappas must be non-negative")


@dataclass
class MolecularSample:
    graph_id: int
    atom_types: torch.Tensor
    spd: torch.Tensor
    semantic: torch.Tensor
    target: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.semantic.shape[0])


class RadialGraphRegressor(nn.Module):
    """A learned shortest-path graph filter with one weight per distance/channel."""

    def __init__(self, max_distance: int):
        super().__init__()
        self.max_distance = int(max_distance)
        self.radial_weight = nn.Parameter(torch.empty(max_distance + 1, 2))
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.radial_weight, std=0.08)

    def forward(
        self,
        semantic: torch.Tensor,
        spd: torch.Tensor,
        *,
        kappa: float,
    ) -> torch.Tensor:
        design = radial_design(semantic, spd, self.max_distance, kappa=kappa)
        return torch.einsum("ndc,dc->n", design, self.radial_weight) + self.bias


def saturated_channel(value: torch.Tensor, kappa: float) -> torch.Tensor:
    """Normalised tanh with an exact linear limit at ``kappa == 0``."""

    if float(kappa) == 0.0:
        return value
    scale = torch.tanh(value.new_tensor(float(kappa)))
    return torch.tanh(float(kappa) * value) / scale


def shortest_path_matrix(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    """Unweighted all-pairs shortest paths for a small molecular graph."""

    neighbours: list[list[int]] = [[] for _ in range(int(num_nodes))]
    for left, right in edge_index.detach().cpu().t().tolist():
        neighbours[int(left)].append(int(right))
    unreachable = int(num_nodes) + 1
    distance = torch.full((num_nodes, num_nodes), unreachable, dtype=torch.long)
    for source in range(num_nodes):
        distance[source, source] = 0
        frontier = [source]
        for depth in range(1, num_nodes + 1):
            following: list[int] = []
            for node in frontier:
                for target in neighbours[node]:
                    if int(distance[source, target]) == unreachable:
                        distance[source, target] = depth
                        following.append(target)
            if not following:
                break
            frontier = following
    return distance


def radial_design(
    semantic: torch.Tensor,
    spd: torch.Tensor,
    max_distance: int,
    *,
    kappa: float,
) -> torch.Tensor:
    """Return receiver-wise shell means with shape ``[node, distance, channel]``."""

    encoded = torch.stack(
        (semantic[:, 0], saturated_channel(semantic[:, 1], kappa)), dim=-1
    )
    rows: list[torch.Tensor] = []
    for distance in range(int(max_distance) + 1):
        mask = (spd == distance).to(encoded.dtype)
        count = mask.sum(dim=-1, keepdim=True)
        mean = mask @ encoded / count.clamp_min(1.0)
        mean = torch.where(count > 0, mean, torch.zeros_like(mean))
        rows.append(mean)
    return torch.stack(rows, dim=1)


def exact_target(
    semantic: torch.Tensor,
    spd: torch.Tensor,
    *,
    target_distance: int,
    local_weight: float,
    far_weight: float,
    kappa: float,
) -> torch.Tensor:
    design = radial_design(
        semantic,
        spd,
        max_distance=int(target_distance),
        kappa=kappa,
    )
    return (
        float(local_weight) * design[:, 0, 0]
        + float(far_weight) * design[:, int(target_distance), 1]
    )


def _semantic_payload(graph_id: int, nodes: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(int(seed) + 104_729 * int(graph_id))
    values = rng.choice(np.asarray([-1.0, 1.0]), size=(int(nodes), 2))
    if nodes >= 2:
        # Prevent degenerate single-class graphs without changing reproducibility.
        for channel in range(2):
            if np.all(values[:, channel] == values[0, channel]):
                values[0, channel] = -1.0
                values[1, channel] = 1.0
    return torch.tensor(values, dtype=torch.float64)


def _select_indices(length: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(length), size=min(int(count), int(length)), replace=False))


def load_zinc_samples(
    config: Config,
    *,
    split: str,
    count: int,
    split_seed: int,
) -> list[MolecularSample]:
    """Load only the selected ZINC subset graphs and attach deterministic payloads."""

    try:
        from torch_geometric.datasets import ZINC
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise ImportError(
            "The molecular benchmark requires torch_geometric; install it before measurement."
        ) from error

    dataset = ZINC(root=str(config.data_root), subset=True, split=split)
    indices = _select_indices(len(dataset), count, split_seed)
    offset = {"train": 0, "val": 20_000, "test": 40_000}[split]
    samples: list[MolecularSample] = []
    for local_id, dataset_index in enumerate(indices.tolist()):
        graph = dataset[int(dataset_index)]
        nodes = int(graph.num_nodes)
        graph_id = offset + int(dataset_index)
        semantic = _semantic_payload(graph_id, nodes, config.analysis_seed)
        spd = shortest_path_matrix(nodes, graph.edge_index)
        target = exact_target(
            semantic,
            spd,
            target_distance=config.target_distance,
            local_weight=config.local_weight,
            far_weight=config.far_weight,
            kappa=config.train_kappa,
        )
        samples.append(
            MolecularSample(
                graph_id=graph_id,
                atom_types=graph.x.view(-1).to(torch.long),
                spd=spd,
                semantic=semantic,
                target=target,
            )
        )
    return samples


def _concatenated_design(
    samples: Sequence[MolecularSample],
    max_distance: int,
    kappa: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    design = torch.cat(
        [
            radial_design(sample.semantic, sample.spd, max_distance, kappa=kappa)
            for sample in samples
        ],
        dim=0,
    )
    target = torch.cat([sample.target for sample in samples], dim=0)
    return design, target


def train_model(
    config: Config,
    train_samples: Sequence[MolecularSample],
    val_samples: Sequence[MolecularSample],
    *,
    seed: int,
) -> tuple[RadialGraphRegressor, list[dict[str, float]]]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    model = RadialGraphRegressor(config.max_distance).to(dtype=torch.float64)
    train_design, train_target = _concatenated_design(
        train_samples, config.max_distance, config.train_kappa
    )
    val_design, val_target = _concatenated_design(
        val_samples, config.max_distance, config.train_kappa
    )
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_val = math.inf
    log: list[dict[str, float]] = []
    report_every = max(1, int(config.train_steps) // 20)
    for step in range(1, int(config.train_steps) + 1):
        optimiser.zero_grad(set_to_none=True)
        prediction = (
            torch.einsum("ndc,dc->n", train_design, model.radial_weight) + model.bias
        )
        loss = torch.mean((prediction - train_target) ** 2)
        loss.backward()
        optimiser.step()
        if step == 1 or step % report_every == 0 or step == int(config.train_steps):
            with torch.no_grad():
                val_prediction = (
                    torch.einsum("ndc,dc->n", val_design, model.radial_weight)
                    + model.bias
                )
                val_mae = torch.mean(torch.abs(val_prediction - val_target)).item()
                train_mae = torch.mean(torch.abs(prediction - train_target)).item()
            log.append(
                {"step": float(step), "train_mae": train_mae, "val_mae": val_mae}
            )
            if val_mae < best_val:
                best_val = val_mae
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, log


def _event_profile(
    mass: torch.Tensor,
    distances: torch.Tensor,
    max_distance: int,
) -> np.ndarray:
    values = np.zeros(int(max_distance) + 1, dtype=np.float64)
    mass_np = mass.detach().cpu().to(torch.float64).numpy()
    distance_np = distances.detach().cpu().numpy()
    for distance in range(int(max_distance) + 1):
        values[distance] = float(mass_np[distance_np == distance].sum())
    total = float(values.sum())
    return values / total if total > 1.0e-15 else values


def _source_indices(sample: MolecularSample, count: int, seed: int) -> np.ndarray:
    if int(count) == 0 or int(count) >= sample.num_nodes:
        return np.arange(sample.num_nodes, dtype=np.int64)
    rng = np.random.default_rng(int(seed) + 65_537 * int(sample.graph_id))
    return np.sort(
        rng.choice(sample.num_nodes, size=min(int(count), sample.num_nodes), replace=False)
    )


def measure_model(
    config: Config,
    model: RadialGraphRegressor,
    test_samples: Sequence[MolecularSample],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    profile_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    for kappa in config.kappas:
        for sample in test_samples:
            semantic = sample.semantic.detach().clone().requires_grad_(True)

            def output_fn(value: torch.Tensor) -> torch.Tensor:
                return model(value, sample.spd, kappa=float(kappa))

            clean = output_fn(semantic)
            oracle_clean = exact_target(
                semantic.detach(),
                sample.spd,
                target_distance=config.target_distance,
                local_weight=config.local_weight,
                far_weight=config.far_weight,
                kappa=float(kappa),
            )
            performance_rows.append(
                {
                    "seed": seed,
                    "graph_id": sample.graph_id,
                    "kappa": float(kappa),
                    "node_mae": float(torch.mean(torch.abs(clean.detach() - oracle_clean))),
                    "prediction_mean": float(clean.detach().mean()),
                }
            )
            jacobian = torch.autograd.functional.jacobian(
                output_fn,
                semantic,
                vectorize=True,
            ).detach()  # [output, source, channel]
            sources = _source_indices(
                sample, config.sources_per_graph, config.analysis_seed + 1_009 * int(seed)
            )
            bamberger_matrix = jacobian[:, sources, :].abs().sum(dim=-1)
            tangent_matrix = torch.zeros_like(bamberger_matrix)
            finite_matrix = torch.zeros_like(bamberger_matrix)
            oracle_matrix = torch.zeros_like(bamberger_matrix)
            for source_index, source in enumerate(sources.tolist()):
                source = int(source)
                donor = semantic.detach().clone()
                donor[source] = -donor[source]
                with torch.no_grad():
                    finite_mass = torch.abs(clean.detach() - output_fn(donor))
                    oracle_donor = exact_target(
                        donor,
                        sample.spd,
                        target_distance=config.target_distance,
                        local_weight=config.local_weight,
                        far_weight=config.far_weight,
                        kappa=float(kappa),
                    )
                    oracle_mass = torch.abs(oracle_clean - oracle_donor)
                derivative = jacobian[:, source, :]
                dose = donor[source] - semantic.detach()[source]
                tangent_matrix[:, source_index] = torch.abs(derivative @ dose)
                finite_matrix[:, source_index] = finite_mass
                oracle_matrix[:, source_index] = oracle_mass
            mass_matrices = {
                "oracle": oracle_matrix,
                "bamberger": bamberger_matrix,
                "tangent": tangent_matrix,
                "finite": finite_matrix,
            }
            # Match Bamberger et al.'s native orientation: for each output/carrier
            # node, normalise influence over input/source nodes, then average
            # carrier-level profiles within graph.
            carrier_profiles: dict[str, list[np.ndarray]] = {
                method: [] for method in mass_matrices
            }
            carrier_totals: dict[str, list[float]] = {
                method: [] for method in mass_matrices
            }
            for carrier in range(sample.num_nodes):
                masses = {
                    method: matrix[carrier]
                    for method, matrix in mass_matrices.items()
                }
                distances = sample.spd[carrier, sources]
                for method, mass in masses.items():
                    carrier_profiles[method].append(
                        _event_profile(mass, distances, config.max_distance)
                    )
                    carrier_totals[method].append(float(mass.sum()))
            for method, values in carrier_profiles.items():
                graph_profile = np.mean(values, axis=0)
                expected_distance = float(
                    np.dot(np.arange(config.max_distance + 1), graph_profile)
                )
                for distance, normalised_mass in enumerate(graph_profile.tolist()):
                    profile_rows.append(
                        {
                            "seed": seed,
                            "graph_id": sample.graph_id,
                            "kappa": float(kappa),
                            "method": method,
                            "distance": distance,
                            "normalised_mass": normalised_mass,
                            "expected_distance": expected_distance,
                            "mean_raw_total": float(np.mean(carrier_totals[method])),
                            "carriers": sample.num_nodes,
                            "sources": int(len(sources)),
                        }
                    )
    return profile_rows, performance_rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _graph_level_profiles(rows: Sequence[dict[str, Any]]) -> dict[tuple, np.ndarray]:
    max_distance = max(int(row["distance"]) for row in rows)
    grouped: dict[tuple, np.ndarray] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            int(row["graph_id"]),
            float(row["kappa"]),
            str(row["method"]),
        )
        grouped.setdefault(key, np.zeros(max_distance + 1, dtype=np.float64))[
            int(row["distance"])
        ] = float(row["normalised_mass"])
    return grouped


def _paired_graph_units(
    graph_profiles: dict[tuple, np.ndarray],
) -> dict[tuple[int, float, str], np.ndarray]:
    by_graph: dict[tuple[int, float, str], list[np.ndarray]] = {}
    for (seed, graph_id, kappa, method), profile in graph_profiles.items():
        by_graph.setdefault((graph_id, kappa, method), []).append(profile)
    return {key: np.mean(values, axis=0) for key, values in by_graph.items()}


def _bootstrap_mean(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = np.mean(values, axis=0)
    if values.shape[0] < 2 or replicates < 2:
        return mean, mean, mean
    rng = np.random.default_rng(int(seed))
    boot = np.empty((int(replicates),) + mean.shape, dtype=np.float64)
    for index in range(int(replicates)):
        selected = rng.integers(0, values.shape[0], size=values.shape[0])
        boot[index] = np.mean(values[selected], axis=0)
    return mean, np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0)


def summarise(
    config: Config,
    profile_rows: Sequence[dict[str, Any]],
    performance_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    graph_profiles = _paired_graph_units(_graph_level_profiles(profile_rows))
    profile_summary: list[dict[str, Any]] = []
    expected_summary: list[dict[str, Any]] = []
    for kappa in config.kappas:
        for method in METHOD_LABELS:
            values = np.stack(
                [
                    profile
                    for (graph_id, row_kappa, row_method), profile in graph_profiles.items()
                    if row_kappa == float(kappa) and row_method == method
                ]
            )
            mean, low, high = _bootstrap_mean(
                values,
                replicates=config.bootstrap_replicates,
                seed=config.analysis_seed + int(round(100 * kappa)) + 17 * len(method),
            )
            distance = np.arange(values.shape[1], dtype=np.float64)
            expected_values = values @ distance
            exp_mean, exp_low, exp_high = _bootstrap_mean(
                expected_values,
                replicates=config.bootstrap_replicates,
                seed=config.analysis_seed + int(round(100 * kappa)) + 31 * len(method),
            )
            expected_summary.append(
                {
                    "kappa": float(kappa),
                    "method": method,
                    "mean": float(exp_mean),
                    "low": float(exp_low),
                    "high": float(exp_high),
                    "graphs": int(values.shape[0]),
                }
            )
            for d in range(values.shape[1]):
                profile_summary.append(
                    {
                        "kappa": float(kappa),
                        "method": method,
                        "distance": d,
                        "mean": float(mean[d]),
                        "low": float(low[d]),
                        "high": float(high[d]),
                        "graphs": int(values.shape[0]),
                    }
                )

    performance_by_graph: dict[tuple[int, float], list[float]] = {}
    for row in performance_rows:
        performance_by_graph.setdefault(
            (int(row["graph_id"]), float(row["kappa"])), []
        ).append(float(row["node_mae"]))
    performance_summary: list[dict[str, Any]] = []
    for kappa in config.kappas:
        values = np.asarray(
            [
                np.mean(rows)
                for (graph_id, row_kappa), rows in performance_by_graph.items()
                if row_kappa == float(kappa)
            ],
            dtype=np.float64,
        )
        mean, low, high = _bootstrap_mean(
            values,
            replicates=config.bootstrap_replicates,
            seed=config.analysis_seed + 7 + int(round(100 * kappa)),
        )
        performance_summary.append(
            {
                "kappa": float(kappa),
                "mean": float(mean),
                "low": float(low),
                "high": float(high),
                "graphs": int(values.size),
            }
        )
    return profile_summary, expected_summary, performance_summary


def _select_rows(rows: Sequence[dict[str, Any]], **selection: Any) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if all(
            (float(row[key]) == float(value) if key == "kappa" else row[key] == value)
            for key, value in selection.items()
        )
    ]


def plot_headline(
    config: Config,
    profile_summary: Sequence[dict[str, Any]],
    expected_summary: Sequence[dict[str, Any]],
    performance_summary: Sequence[dict[str, Any]],
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.08, top=0.79, wspace=0.25, hspace=0.42)
    kappas = np.asarray([float(row["kappa"]) for row in performance_summary])
    mae = np.asarray([float(row["mean"]) for row in performance_summary])
    mae_low = np.asarray([float(row["low"]) for row in performance_summary])
    mae_high = np.asarray([float(row["high"]) for row in performance_summary])
    axes[0, 0].plot(kappas, mae, color="#555555", marker="o", linewidth=2)
    axes[0, 0].fill_between(kappas, mae_low, mae_high, color="#777777", alpha=0.16)
    axes[0, 0].set_title("Held-out prediction is unchanged")
    axes[0, 0].set_xlabel(r"Saturation $\kappa$")
    axes[0, 0].set_ylabel("Node MAE")
    axes[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[0, 0].set_ylim(0.0, max(float(mae_high.max()) * 1.25, 1.0e-8))
    axes[0, 0].text(
        0.03,
        0.93,
        "Clean input--output function is identical for every $\\kappa$",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="top",
        color="#666666",
        fontsize=8,
    )

    for method in METHOD_LABELS:
        selected = _select_rows(expected_summary, method=method)
        selected.sort(key=lambda row: float(row["kappa"]))
        x = np.asarray([float(row["kappa"]) for row in selected])
        y = np.asarray([float(row["mean"]) for row in selected])
        low = np.asarray([float(row["low"]) for row in selected])
        high = np.asarray([float(row["high"]) for row in selected])
        style, marker = METHOD_STYLES[method]
        axes[0, 1].plot(
            x,
            y,
            linestyle=style,
            marker=marker,
            color=METHOD_COLOURS[method],
            linewidth=2,
            label=METHOD_LABELS[method],
        )
        axes[0, 1].fill_between(x, low, high, color=METHOD_COLOURS[method], alpha=0.12)
    axes[0, 1].set_title("Estimated semantic reach")
    axes[0, 1].set_xlabel(r"Saturation $\kappa$")
    axes[0, 1].set_ylabel("Expected distance")

    endpoint_kappas = (min(config.kappas), max(config.kappas))
    endpoint_titles = ("Linear control", "Locally saturated pathway")
    for axis, kappa, title in zip(axes[1], endpoint_kappas, endpoint_titles):
        for method in METHOD_LABELS:
            selected = _select_rows(profile_summary, method=method, kappa=float(kappa))
            selected.sort(key=lambda row: int(row["distance"]))
            distance = np.asarray([int(row["distance"]) for row in selected])
            mean = np.asarray([float(row["mean"]) for row in selected])
            low = np.asarray([float(row["low"]) for row in selected])
            high = np.asarray([float(row["high"]) for row in selected])
            style, marker = METHOD_STYLES[method]
            axis.plot(
                distance,
                mean,
                linestyle=style,
                marker=marker,
                color=METHOD_COLOURS[method],
                linewidth=2,
                label=METHOD_LABELS[method],
            )
            axis.fill_between(distance, low, high, color=METHOD_COLOURS[method], alpha=0.10)
        axis.set_title(rf"{title} ($\kappa={kappa:g}$)")
        axis.set_xlabel("Shortest-path distance")
        axis.set_ylabel("Normalised usage mass")
        axis.set_xlim(-0.15, config.max_distance + 0.15)
        axis.set_ylim(bottom=0)
        axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.875),
    )
    fig.suptitle("Finite and differential semantic reach on molecular graphs", fontsize=15, y=0.985)
    fig.text(
        0.5,
        0.932,
        (
            f"ZINC topology; known target uses d=0 and d={config.target_distance}; "
            "mean with 95% paired-graph bootstrap interval"
        ),
        ha="center",
        va="top",
        color="#666666",
        fontsize=9,
    )
    figure_dir = config.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "molecular_nonlinear_reach.png"
    pdf = figure_dir / "molecular_nonlinear_reach.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def measure(config: Config) -> dict[str, Any]:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_samples = load_zinc_samples(
        config,
        split="train",
        count=config.train_graphs,
        split_seed=config.analysis_seed + 1,
    )
    val_samples = load_zinc_samples(
        config,
        split="val",
        count=config.val_graphs,
        split_seed=config.analysis_seed + 2,
    )
    test_samples = load_zinc_samples(
        config,
        split="test",
        count=config.test_graphs,
        split_seed=config.analysis_seed + 3,
    )
    all_profiles: list[dict[str, Any]] = []
    all_performance: list[dict[str, Any]] = []
    health: list[dict[str, Any]] = []
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for seed in config.seeds:
        model, training_log = train_model(
            config,
            train_samples,
            val_samples,
            seed=int(seed),
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "seed": int(seed),
                "fingerprint": config.fingerprint,
            },
            checkpoint_dir / f"seed_{seed}.pt",
        )
        _write_csv(config.output_dir / "training" / f"seed_{seed}.csv", training_log)
        profiles, performance = measure_model(
            config,
            model,
            test_samples,
            seed=int(seed),
        )
        all_profiles.extend(profiles)
        all_performance.extend(performance)
        final = training_log[-1]
        learned = model.radial_weight.detach().cpu().numpy()
        other_weight = learned.copy()
        other_weight[0, 0] = 0.0
        other_weight[config.target_distance, 1] = 0.0
        health.append(
            {
                "seed": int(seed),
                "train_mae": float(final["train_mae"]),
                "val_mae": float(final["val_mae"]),
                "learned_local_weight": float(learned[0, 0]),
                "learned_far_weight": float(learned[config.target_distance, 1]),
                "largest_other_weight": float(np.max(np.abs(other_weight))),
            }
        )
        print(
            f"[seed {seed}] val MAE={final['val_mae']:.3e}; "
            f"w_local={learned[0, 0]:.3f}; "
            f"w_far={learned[config.target_distance, 1]:.3f}"
        )
    _write_csv(config.output_dir / "results" / "event_profiles.csv", all_profiles)
    _write_csv(config.output_dir / "results" / "performance.csv", all_performance)
    _write_csv(config.output_dir / "results" / "model_health.csv", health)
    return {
        "profile_rows": len(all_profiles),
        "performance_rows": len(all_performance),
        "health": health,
    }


def figures(config: Config) -> dict[str, Any]:
    profile_rows = _read_csv(config.output_dir / "results" / "event_profiles.csv")
    performance_rows = _read_csv(config.output_dir / "results" / "performance.csv")
    profile_summary, expected_summary, performance_summary = summarise(
        config, profile_rows, performance_rows
    )
    _write_csv(config.output_dir / "results" / "profile_summary.csv", profile_summary)
    _write_csv(config.output_dir / "results" / "expected_distance_summary.csv", expected_summary)
    _write_csv(config.output_dir / "results" / "performance_summary.csv", performance_summary)
    paths = plot_headline(config, profile_summary, expected_summary, performance_summary)
    _write_json(
        config.output_dir / "results" / "summary.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "fingerprint": config.fingerprint,
            "figure": paths,
            "interpretation": (
                "Clean predictions and finite donor effects are invariant to kappa. "
                "Any kappa-dependent movement of a Jacobian range is therefore estimator "
                "dependence on the off-manifold continuous extension, not a change in learned use."
            ),
        },
    )
    return {
        "figures": paths,
        "profile_summary": profile_summary,
        "expected_summary": expected_summary,
        "performance_summary": performance_summary,
    }


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "measure", "figures"), default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/molecular_nonlinear_reach_v1"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / ".cache" / "graph_specialisation_metrics" / "zinc_subset",
    )
    parser.add_argument("--train-graphs", type=int, default=512)
    parser.add_argument("--val-graphs", type=int, default=96)
    parser.add_argument("--test-graphs", type=int, default=96)
    parser.add_argument(
        "--sources-per-graph",
        type=int,
        default=0,
        help="Input/source nodes per graph; 0 uses all nodes for literal Bamberger range.",
    )
    parser.add_argument("--target-distance", type=int, default=4)
    parser.add_argument("--max-distance", type=int, default=8)
    parser.add_argument("--local-weight", type=float, default=0.5)
    parser.add_argument("--far-weight", type=float, default=1.0)
    parser.add_argument("--train-kappa", type=float, default=4.0)
    parser.add_argument("--kappas", default="0,0.5,1,2,4,8")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--train-steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--analysis-seed", type=int, default=91_021)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    config = Config(
        output_dir=args.output_dir,
        data_root=args.data_root,
        train_graphs=int(args.train_graphs),
        val_graphs=int(args.val_graphs),
        test_graphs=int(args.test_graphs),
        sources_per_graph=int(args.sources_per_graph),
        target_distance=int(args.target_distance),
        max_distance=int(args.max_distance),
        local_weight=float(args.local_weight),
        far_weight=float(args.far_weight),
        train_kappa=float(args.train_kappa),
        kappas=_parse_floats(args.kappas),
        seeds=_parse_ints(args.seeds),
        train_steps=int(args.train_steps),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        bootstrap_replicates=int(args.bootstrap_replicates),
        analysis_seed=int(args.analysis_seed),
    )
    config.validate()
    _write_json(
        config.output_dir / "config.json",
        {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "data_root": str(config.data_root),
            "fingerprint": config.fingerprint,
            "protocol_version": PROTOCOL_VERSION,
        },
    )
    result: dict[str, Any] = {"config": config, "output_dir": str(config.output_dir)}
    if args.phase in {"all", "measure"}:
        result["measurement"] = measure(config)
    if args.phase in {"all", "figures"}:
        result.update(figures(config))
    if "figures" in result:
        print(f"[figure] {result['figures']['png']}")
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
