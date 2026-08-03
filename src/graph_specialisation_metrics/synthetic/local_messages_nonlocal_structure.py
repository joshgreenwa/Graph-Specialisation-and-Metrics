"""Lightweight Chapter 6 control for local messages and non-local structure.

The experiment separates semantic transport from structural preprocessing.  Two
degree-two candidate nodes carry independent standard-normal values.  One lies
on a cycle and one lies in the interior of a long chain; the regression target
is the value on the cycle.  Their radius-two rooted neighbourhoods are exactly
isomorphic.

A shared candidate-local gate receives only the candidate value and its
diagonal RRWP vector.  The graph prediction is the RRWP-gated sum of the two
values.  All horizon conditions have the same encoder width and parameter
count; unavailable higher-order RRWP channels are set to zero.  Consequently,
the short-horizon model is forced to average the two values (population MSE
0.5), while a longer horizon can solve the task without transporting either
semantic value away from its candidate.

The module also performs a deterministic remote-edge check: adding an edge six
hops from a selected node leaves its I and P entries unchanged but alters
higher-order RRWP on the selected self pair and an adjacent bond.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

plt.switch_backend("Agg")

DTYPE = torch.float64
BLUE = "#4477AA"
ORANGE = "#EE7733"
GREEN = "#228833"
PURPLE = "#AA3377"
RED = "#CC6677"
GREY = "#777777"
LIGHT_GREY = "#D7D7D7"
TEXT = "#222222"


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path
    cycle_sizes: tuple[int, ...] = (6, 8, 10)
    local_depth: int = 2
    max_horizon: int = 16
    seeds: tuple[int, ...] = tuple(range(8))
    training_steps: int = 5_000
    learning_rate: float = 0.2
    test_examples: int = 8_192
    data_seed: int = 60_613
    numerical_tolerance: float = 1.0e-14

    def validate(self) -> None:
        if not self.cycle_sizes or len(self.cycle_sizes) != len(set(self.cycle_sizes)):
            raise ValueError("cycle_sizes must be non-empty and unique")
        if tuple(sorted(self.cycle_sizes)) != self.cycle_sizes:
            raise ValueError("cycle_sizes must be increasing")
        if any(size < 2 * self.local_depth + 2 for size in self.cycle_sizes):
            raise ValueError("each cycle must extend beyond the local receptive field")
        if max(self.cycle_sizes) > self.max_horizon:
            raise ValueError("max_horizon must reach every registered cycle length")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be non-empty and unique")
        if min(self.training_steps, self.test_examples) < 1:
            raise ValueError("training_steps and test_examples must be positive")
        if self.learning_rate <= 0 or self.numerical_tolerance <= 0:
            raise ValueError("learning_rate and numerical_tolerance must be positive")


def _transition_matrix(graph: nx.Graph) -> tuple[np.ndarray, list[int]]:
    nodes = sorted(int(node) for node in graph.nodes)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=np.float64)
    degree = adjacency.sum(axis=1, keepdims=True)
    if np.any(degree == 0):
        raise ValueError("RRWP requires a graph without isolated nodes")
    return adjacency / degree, nodes


def rrwp_powers(graph: nx.Graph, max_horizon: int) -> tuple[np.ndarray, list[int]]:
    """Return ``[I, P, ..., P^K]`` for an undirected graph."""

    transition, nodes = _transition_matrix(graph)
    powers = [np.eye(len(nodes), dtype=np.float64)]
    for _ in range(int(max_horizon)):
        powers.append(powers[-1] @ transition)
    return np.stack(powers, axis=0), nodes


def make_cycle_chain_graph(
    cycle_size: int,
    *,
    max_horizon: int,
) -> tuple[nx.Graph, int, int]:
    """Build one connected graph with a cycle and a long-chain candidate."""

    cycle_size = int(cycle_size)
    chain_length = 2 * int(max_horizon) + 5
    graph = nx.Graph()
    cycle_nodes = list(range(cycle_size))
    graph.add_edges_from(
        (cycle_nodes[index], cycle_nodes[(index + 1) % cycle_size])
        for index in range(cycle_size)
    )
    chain_nodes = list(range(cycle_size, cycle_size + chain_length))
    graph.add_edges_from(pairwise(chain_nodes))
    hub = cycle_size + chain_length
    graph.add_edge(cycle_nodes[cycle_size // 2], hub)
    graph.add_edge(chain_nodes[0], hub)
    return graph, cycle_nodes[0], chain_nodes[chain_length // 2]


def _rooted_ball(graph: nx.Graph, root: int, radius: int) -> nx.Graph:
    distances = nx.single_source_shortest_path_length(graph, int(root), cutoff=int(radius))
    ball = graph.subgraph(distances).copy()
    nx.set_node_attributes(
        ball,
        {node: bool(int(node) == int(root)) for node in ball.nodes},
        "root",
    )
    return ball


def rooted_balls_are_isomorphic(
    graph: nx.Graph,
    first_root: int,
    second_root: int,
    *,
    radius: int,
) -> bool:
    first = _rooted_ball(graph, first_root, radius)
    second = _rooted_ball(graph, second_root, radius)
    return bool(
        nx.is_isomorphic(
            first,
            second,
            node_match=lambda left, right: left["root"] == right["root"],
        )
    )


def candidate_rrwp_features(
    cycle_size: int,
    *,
    max_horizon: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    graph, cycle_candidate, chain_candidate = make_cycle_chain_graph(
        cycle_size,
        max_horizon=max_horizon,
    )
    powers, nodes = rrwp_powers(graph, max_horizon)
    positions = {node: index for index, node in enumerate(nodes)}
    features = np.stack(
        (
            powers[:, positions[cycle_candidate], positions[cycle_candidate]],
            powers[:, positions[chain_candidate], positions[chain_candidate]],
        ),
        axis=0,
    )
    return features, {
        "graph": graph,
        "cycle_candidate": int(cycle_candidate),
        "chain_candidate": int(chain_candidate),
    }


def remote_edge_sensitivity(max_horizon: int = 20) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Measure RRWP changes caused by one edge outside a radius-two ball."""

    base = nx.path_graph(17)
    altered = base.copy()
    candidate = 8
    neighbour = 9
    remote_edge = (2, 14)
    altered.add_edge(*remote_edge)
    base_powers, nodes = rrwp_powers(base, max_horizon)
    altered_powers, altered_nodes = rrwp_powers(altered, max_horizon)
    if nodes != altered_nodes:
        raise RuntimeError("remote-edge graph node order changed")
    position = {node: index for index, node in enumerate(nodes)}
    c = position[candidate]
    n = position[neighbour]
    rows: list[dict[str, float | int]] = []
    for horizon in range(int(max_horizon) + 1):
        base_self = float(base_powers[horizon, c, c])
        altered_self = float(altered_powers[horizon, c, c])
        base_bond = float(base_powers[horizon, c, n])
        altered_bond = float(altered_powers[horizon, c, n])
        rows.append(
            {
                "horizon": int(horizon),
                "base_self": base_self,
                "altered_self": altered_self,
                "delta_self": abs(altered_self - base_self),
                "base_bond": base_bond,
                "altered_bond": altered_bond,
                "delta_bond": abs(altered_bond - base_bond),
            }
        )
    table = pd.DataFrame(rows)
    return table, {
        "base_graph": base,
        "altered_graph": altered,
        "candidate": candidate,
        "neighbour": neighbour,
        "remote_edge": remote_edge,
        "remote_edge_distance_from_candidate": min(
            nx.shortest_path_length(base, candidate, endpoint)
            for endpoint in remote_edge
        ),
    }


def _first_nonzero(values: np.ndarray, tolerance: float) -> int | None:
    indices = np.flatnonzero(np.abs(values) > float(tolerance))
    return int(indices[0]) if indices.size else None


def train_horizon_probes(
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    """Fit all parameter-matched local gates in one vectorised optimisation."""

    config.validate()
    feature_bank: dict[int, np.ndarray] = {}
    metadata: dict[int, dict[str, Any]] = {}
    delta_rows: list[np.ndarray] = []
    row_keys: list[tuple[int, int, int]] = []
    dimension = int(config.max_horizon) + 1
    for cycle_size in config.cycle_sizes:
        features, item_metadata = candidate_rrwp_features(
            cycle_size,
            max_horizon=config.max_horizon,
        )
        feature_bank[int(cycle_size)] = features
        metadata[int(cycle_size)] = item_metadata
        for horizon in range(1, int(config.max_horizon) + 1):
            difference = (features[0] - features[1]).copy()
            difference[horizon + 1 :] = 0.0
            for seed in config.seeds:
                delta_rows.append(difference.copy())
                row_keys.append((int(cycle_size), int(horizon), int(seed)))

    deltas = torch.tensor(np.stack(delta_rows), dtype=DTYPE)
    generator = torch.Generator().manual_seed(int(config.data_seed))
    weights = torch.randn(deltas.shape, generator=generator, dtype=DTYPE) * 0.02
    weights.requires_grad_(True)
    optimiser = torch.optim.Adam([weights], lr=float(config.learning_rate))
    for _ in range(int(config.training_steps)):
        score_difference = torch.sum(weights * deltas, dim=-1)
        cycle_mass = torch.sigmoid(score_difference)
        # For independent N(0,1) values, E[(y_hat-y_cycle)^2] is exact.
        population_mse = 2.0 * (1.0 - cycle_mass).square()
        loss = population_mse.sum()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        score_difference = torch.sum(weights * deltas, dim=-1)
        cycle_masses = torch.sigmoid(score_difference).cpu().numpy()

    test_values: dict[tuple[int, int], np.ndarray] = {}
    for cycle_size in config.cycle_sizes:
        for seed in config.seeds:
            rng = np.random.default_rng(
                int(config.data_seed) + 10_000 * int(cycle_size) + int(seed)
            )
            test_values[(int(cycle_size), int(seed))] = rng.standard_normal(
                (int(config.test_examples), 2)
            )

    rows: list[dict[str, float | int]] = []
    for row_index, ((cycle_size, horizon, seed), cycle_mass) in enumerate(
        zip(row_keys, cycle_masses, strict=True)
    ):
        values = test_values[(cycle_size, seed)]
        prediction = cycle_mass * values[:, 0] + (1.0 - cycle_mass) * values[:, 1]
        error = prediction - values[:, 0]
        difference = deltas[row_index].cpu().numpy()
        rows.append(
            {
                "cycle_size": int(cycle_size),
                "horizon": int(horizon),
                "seed": int(seed),
                "parameter_count": int(dimension + 1),
                "rrwp_difference_l2": float(np.linalg.norm(difference)),
                "cycle_gate_mass": float(cycle_mass),
                "population_mse": float(2.0 * (1.0 - cycle_mass) ** 2),
                "test_mse": float(np.mean(error**2)),
                "test_mae": float(np.mean(np.abs(error))),
            }
        )
    return pd.DataFrame(rows), feature_bank, metadata


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _clean_axis(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(width=0.8)


def _panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        color=TEXT,
    )


def plot_remote_edge_figure(
    table: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    _configure_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.4, 3.8),
        gridspec_kw={"width_ratios": (1.05, 1.35)},
    )
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.18, top=0.78, wspace=0.28)
    figure.suptitle(
        "A remote edge changes RRWP on a locally attended pair",
        fontsize=14.5,
        y=0.96,
        color=TEXT,
    )
    figure.text(
        0.5,
        0.875,
        "The selected node and bond are unchanged; only the dashed edge six hops away is added",
        ha="center",
        fontsize=9.2,
        color=GREY,
    )

    axis = axes[0]
    _panel_label(axis, "a")
    candidate = int(metadata["candidate"])
    neighbour = int(metadata["neighbour"])
    remote_left, remote_right = metadata["remote_edge"]
    x_positions = {node: float(node - candidate) for node in metadata["base_graph"].nodes}
    local_nodes = {node for node in metadata["base_graph"].nodes if abs(node - candidate) <= 2}
    for left, right in metadata["base_graph"].edges:
        axis.plot(
            (x_positions[left], x_positions[right]),
            (0.0, 0.0),
            color=LIGHT_GREY,
            linewidth=1.6,
            zorder=1,
        )
    arc_x = np.linspace(x_positions[remote_left], x_positions[remote_right], 160)
    radius = (x_positions[remote_right] - x_positions[remote_left]) / 2.0
    centre = (x_positions[remote_left] + x_positions[remote_right]) / 2.0
    arc_y = np.sqrt(np.maximum(radius**2 - (arc_x - centre) ** 2, 0.0)) * 0.28
    axis.plot(arc_x, arc_y, color=RED, linewidth=1.8, linestyle="--", zorder=2)
    for node in metadata["base_graph"].nodes:
        if node == candidate:
            face, edge, size = BLUE, BLUE, 75
        elif node == neighbour:
            face, edge, size = ORANGE, ORANGE, 58
        elif node in local_nodes:
            face, edge, size = "white", BLUE, 48
        elif node in (remote_left, remote_right):
            face, edge, size = "white", RED, 52
        else:
            face, edge, size = "white", GREY, 34
        axis.scatter(
            x_positions[node],
            0.0,
            s=size,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.2,
            zorder=3,
        )
    axis.text(x_positions[candidate], -0.42, "selected node c", ha="center", color=BLUE)
    axis.text(
        (x_positions[candidate] + x_positions[neighbour]) / 2.0,
        0.18,
        "local bond",
        ha="center",
        color=ORANGE,
        fontsize=8.5,
    )
    axis.text(centre, float(arc_y.max()) + 0.14, "added remote edge", ha="center", color=RED)
    axis.text(0.0, -0.73, "blue outline = radius-2 neighbourhood", ha="center", color=GREY)
    axis.set_xlim(-8.7, 8.7)
    axis.set_ylim(-0.9, 2.15)
    axis.axis("off")

    axis = axes[1]
    _panel_label(axis, "b")
    axis.axvspan(-0.2, 1.2, color=LIGHT_GREY, alpha=0.35, linewidth=0)
    axis.plot(
        table["horizon"],
        table["delta_self"],
        marker="o",
        markersize=4.0,
        linewidth=1.8,
        color=BLUE,
        label=r"self pair $|\Delta(P^k)_{cc}|$",
    )
    axis.plot(
        table["horizon"],
        table["delta_bond"],
        marker="s",
        markersize=3.8,
        linewidth=1.6,
        color=ORANGE,
        label=r"local bond $|\Delta(P^k)_{c,c+1}|$",
    )
    first_self = _first_nonzero(table["delta_self"].to_numpy(), 1.0e-15)
    first_bond = _first_nonzero(table["delta_bond"].to_numpy(), 1.0e-15)
    if first_bond is not None:
        y_value = float(table.loc[table["horizon"] == first_bond, "delta_bond"].iloc[0])
        axis.annotate(
            f"first local-pair change: k={first_bond}",
            xy=(first_bond, y_value),
            xytext=(first_bond - 5.2, 0.00105),
            arrowprops={"arrowstyle": "->", "color": GREY, "linewidth": 0.9},
            color=TEXT,
            fontsize=8.5,
        )
    if first_self is not None:
        y_value = float(table.loc[table["horizon"] == first_self, "delta_self"].iloc[0])
        axis.annotate(
            f"self changes: k={first_self}",
            xy=(first_self, y_value),
            xytext=(first_self + 0.8, 0.00058),
            arrowprops={"arrowstyle": "->", "color": GREY, "linewidth": 0.9},
            color=TEXT,
            fontsize=8.5,
        )
    axis.annotate(
        "I, P unchanged",
        xy=(0.55, 0.0),
        xytext=(2.0, 0.00032),
        arrowprops={"arrowstyle": "->", "color": GREY, "linewidth": 0.8},
        ha="left",
        va="center",
        color=GREY,
    )
    axis.set_xlabel("RRWP order k")
    axis.set_ylabel("absolute change after adding the edge")
    axis.set_xlim(-0.3, float(table["horizon"].max()) + 0.3)
    axis.set_ylim(-0.00008, 0.00365)
    axis.set_xticks(np.arange(0, int(table["horizon"].max()) + 1, 2))
    axis.legend(frameon=False, loc="upper left")
    _clean_axis(axis)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / "01_remote_edge_rrwp_sensitivity.png"
    pdf = figures_dir / "01_remote_edge_rrwp_sensitivity.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _mean_ci(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=np.float64)
    mean = float(array.mean())
    if array.size < 2:
        return mean, 0.0
    return mean, float(1.96 * array.std(ddof=1) / np.sqrt(array.size))


def plot_horizon_figure(
    table: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: Path,
) -> tuple[Path, Path]:
    _configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.78, wspace=0.28)
    figure.suptitle(
        "Local semantic selection begins when RRWP reaches the distinguishing structure",
        fontsize=14.0,
        y=0.96,
        color=TEXT,
    )
    figure.text(
        0.5,
        0.875,
        "Same gate width and parameter count at every horizon; unavailable RRWP channels are zeroed",
        ha="center",
        fontsize=9.2,
        color=GREY,
    )
    colours = (GREEN, BLUE, PURPLE, ORANGE)

    axis = axes[0]
    _panel_label(axis, "a")
    for colour, cycle_size in zip(colours, config.cycle_sizes, strict=False):
        subset = table[table["cycle_size"] == int(cycle_size)]
        grouped = subset.groupby("horizon", as_index=False)["rrwp_difference_l2"].mean()
        axis.plot(
            grouped["horizon"],
            grouped["rrwp_difference_l2"],
            color=colour,
            linewidth=1.9,
            marker="o",
            markersize=3.6,
            label=f"cycle length {cycle_size}",
        )
        axis.scatter([cycle_size], [0.0], marker="^", s=42, color=colour, zorder=4)
    axis.set_title("Structural information at the candidate")
    axis.set_xlabel("RRWP horizon K")
    axis.set_ylabel(r"candidate separation $\|r_K^{cycle}-r_K^{chain}\|_2$")
    axis.set_xlim(0.7, config.max_horizon + 0.3)
    axis.set_xticks(np.arange(2, config.max_horizon + 1, 2))
    axis.set_ylim(bottom=-0.002)
    axis.legend(frameon=False, loc="upper left")
    _clean_axis(axis)

    axis = axes[1]
    axis.text(
        -0.20,
        1.08,
        "b",
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        color=TEXT,
    )
    for colour, cycle_size in zip(colours, config.cycle_sizes, strict=False):
        subset = table[table["cycle_size"] == int(cycle_size)]
        horizons: list[int] = []
        means: list[float] = []
        cis: list[float] = []
        for horizon, values in subset.groupby("horizon")["test_mse"]:
            mean, ci = _mean_ci(values)
            horizons.append(int(horizon))
            means.append(mean)
            cis.append(ci)
        horizons_array = np.asarray(horizons)
        means_array = np.asarray(means)
        cis_array = np.asarray(cis)
        axis.plot(
            horizons_array,
            means_array,
            color=colour,
            linewidth=2.0,
            marker="o",
            markersize=3.6,
            label=f"cycle length {cycle_size}",
        )
        axis.fill_between(
            horizons_array,
            means_array - cis_array,
            means_array + cis_array,
            color=colour,
            alpha=0.14,
            linewidth=0,
        )
    axis.axhline(0.5, color=GREY, linewidth=1.1, linestyle="--")
    axis.text(config.max_horizon - 0.2, 0.512, "indistinguishable candidates", ha="right", color=GREY)
    axis.axhline(0.0, color=LIGHT_GREY, linewidth=0.9)
    axis.set_title("Held-out semantic prediction (8 paired seeds)")
    axis.set_xlabel("RRWP horizon K")
    axis.set_ylabel("test MSE")
    axis.set_xlim(0.7, config.max_horizon + 0.3)
    axis.set_xticks(np.arange(2, config.max_horizon + 1, 2))
    axis.set_ylim(-0.015, 0.555)
    _clean_axis(axis)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / "02_local_semantic_selection_by_rrwp_horizon.png"
    pdf = figures_dir / "02_local_semantic_selection_by_rrwp_horizon.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _paired_summary(
    table: pd.DataFrame,
    *,
    cycle_size: int,
    short_horizon: int,
    long_horizon: int,
) -> dict[str, float]:
    subset = table[table["cycle_size"] == int(cycle_size)]
    short = subset[subset["horizon"] == int(short_horizon)].set_index("seed")
    long = subset[subset["horizon"] == int(long_horizon)].set_index("seed")
    paired = short["test_mse"] - long["test_mse"]
    mean, ci = _mean_ci(paired)
    return {
        "short_mean_test_mse": float(short["test_mse"].mean()),
        "long_mean_test_mse": float(long["test_mse"].mean()),
        "paired_mse_reduction": mean,
        "paired_mse_reduction_95ci_half_width": ci,
        "long_mean_cycle_gate_mass": float(long["cycle_gate_mass"].mean()),
    }


def run(config: ExperimentConfig) -> dict[str, Any]:
    config.validate()
    torch.set_num_threads(1)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    remote_table, remote_metadata = remote_edge_sensitivity(max_horizon=20)
    probe_table, feature_bank, graph_metadata = train_horizon_probes(config)
    remote_table.to_csv(config.output_dir / "remote_edge_sensitivity.csv", index=False)
    probe_table.to_csv(config.output_dir / "horizon_probe_results.csv", index=False)
    remote_figures = plot_remote_edge_figure(remote_table, remote_metadata, config.output_dir)
    horizon_figures = plot_horizon_figure(probe_table, config, config.output_dir)

    remote_self_first = _first_nonzero(
        remote_table["delta_self"].to_numpy(), config.numerical_tolerance
    )
    remote_bond_first = _first_nonzero(
        remote_table["delta_bond"].to_numpy(), config.numerical_tolerance
    )
    cycle_summaries: dict[str, Any] = {}
    for cycle_size in config.cycle_sizes:
        features = feature_bank[int(cycle_size)]
        difference_by_horizon = []
        for horizon in range(config.max_horizon + 1):
            difference = (features[0] - features[1]).copy()
            difference[horizon + 1 :] = 0.0
            difference_by_horizon.append(float(np.linalg.norm(difference)))
        first_horizon = _first_nonzero(
            np.asarray(difference_by_horizon), config.numerical_tolerance
        )
        item_metadata = graph_metadata[int(cycle_size)]
        cycle_summaries[str(cycle_size)] = {
            "radius_local_depth_balls_isomorphic": rooted_balls_are_isomorphic(
                item_metadata["graph"],
                item_metadata["cycle_candidate"],
                item_metadata["chain_candidate"],
                radius=config.local_depth,
            ),
            "first_distinguishing_rrwp_horizon": first_horizon,
            **_paired_summary(
                probe_table,
                cycle_size=int(cycle_size),
                short_horizon=1,
                long_horizon=config.max_horizon,
            ),
        }

    result = {
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
        },
        "remote_edge_check": {
            "edge_distance_from_candidate": int(
                remote_metadata["remote_edge_distance_from_candidate"]
            ),
            "identity_and_one_step_self_unchanged": bool(
                np.all(
                    remote_table.loc[
                        remote_table["horizon"].isin((0, 1)), "delta_self"
                    ].to_numpy()
                    <= config.numerical_tolerance
                )
            ),
            "identity_and_one_step_bond_unchanged": bool(
                np.all(
                    remote_table.loc[
                        remote_table["horizon"].isin((0, 1)), "delta_bond"
                    ].to_numpy()
                    <= config.numerical_tolerance
                )
            ),
            "first_changed_self_order": remote_self_first,
            "first_changed_local_bond_order": remote_bond_first,
        },
        "cycle_chain_controls": cycle_summaries,
        "figures": [str(path) for path in (*remote_figures, *horizon_figures)],
    }
    with (config.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Run the lightweight local-messages/non-local-structure pilot."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/chapter6_local_nonlocal_synthetic_v1"),
    )
    parser.add_argument("--cycle-sizes", default="6,8,10")
    parser.add_argument("--local-depth", type=int, default=2)
    parser.add_argument("--max-horizon", type=int, default=16)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--training-steps", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--test-examples", type=int, default=8_192)
    args = parser.parse_args(argv)
    config = ExperimentConfig(
        output_dir=args.output_dir,
        cycle_sizes=tuple(int(value) for value in args.cycle_sizes.split(",") if value),
        local_depth=int(args.local_depth),
        max_horizon=int(args.max_horizon),
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        training_steps=int(args.training_steps),
        learning_rate=float(args.learning_rate),
        test_examples=int(args.test_examples),
    )
    result = run(config)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
