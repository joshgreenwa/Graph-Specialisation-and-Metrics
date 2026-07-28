"""Focused publication plots shared by the ZINC and QM9 GRIT analyses."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

from .grit_figure_data import CanonicalHeadMetrics, Head


NAVY = "#17324D"
BLUE = "#2878B5"
TEAL = "#087E8B"
GOLD = "#E6A700"
ORANGE = "#D97706"
SLATE = "#607080"
LIGHT_GRID = "#DCE3E8"
ATTENTION_CMAP = plt.get_cmap("Blues")
SELECTIVITY_CMAP = plt.get_cmap("coolwarm")
HEAD_STYLES = {
    "semantic": {"color": GOLD, "label": "Semantic specialist"},
    "structural": {"color": TEAL, "label": "Structural specialist"},
    "structural_alternate": {
        "color": "#2A9D8F",
        "label": "Alternate structural specialist",
    },
}
ELEMENT_COLORS = {
    "H-focused": "#A7B0B8",
    "C-focused": "#2878B5",
    "N-focused": "#008E72",
    "O-focused": "#D97706",
    "F-focused": "#7B61A8",
    "P-focused": "#8B6F47",
    "S-focused": "#CCB000",
    "Cl-focused": "#6B8E23",
    "Br-focused": "#9C4F96",
    "I-focused": "#C44E52",
    "other/diffuse": "#B8C2CA",
    "Other / rare": "#4B5563",
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": NAVY,
            "axes.linewidth": 0.9,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": LIGHT_GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _head_label(head: Head) -> str:
    return f"L{head[0]} H{head[1]}"


def _heatmap(ax, values, *, title: str, cmap, norm):
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    layers, heads = values.shape
    ax.set_title(title, pad=10)
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")
    ax.set_xticks(np.arange(heads))
    ax.set_xticklabels(np.arange(heads), fontsize=7)
    ax.set_yticks(np.arange(layers))
    ax.set_yticklabels(np.arange(layers))
    ax.tick_params(length=0)
    return image


def plot_score_heatmaps(
    metrics: CanonicalHeadMetrics,
    selected_heads: Mapping[str, Head] | None = None,
    *,
    title: str,
):
    del selected_heads
    apply_publication_style()
    values = np.concatenate(
        (
            metrics.normalized_semantic.reshape(-1),
            metrics.normalized_structural.reshape(-1),
        )
    )
    vmax = max(float(np.nanpercentile(values, 99.5)), 1.0)
    norm = Normalize(vmin=0.0, vmax=vmax)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 5.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    semantic = _heatmap(
        axes[0],
        metrics.normalized_semantic,
        title=r"Semantic score $S_{\rm sem}/\overline{S}_{\rm sem}$",
        cmap="viridis",
        norm=norm,
    )
    _heatmap(
        axes[1],
        metrics.normalized_structural,
        title=r"Structural score $S_{\rm str}/\overline{S}_{\rm str}$",
        cmap="viridis",
        norm=norm,
    )
    colorbar = fig.colorbar(semantic, ax=axes, shrink=0.86, pad=0.015)
    colorbar.set_label("Normalised score")
    fig.suptitle(title, fontsize=18, y=1.04)
    return fig


def plot_coordinate_heatmaps(
    metrics: CanonicalHeadMetrics,
    selected_heads: Mapping[str, Head] | None = None,
    *,
    title: str,
):
    del selected_heads
    apply_publication_style()
    active_values = metrics.selectivity[
        metrics.active & np.isfinite(metrics.selectivity)
    ]
    d_max = max(
        0.1,
        min(
            1.0,
            float(np.max(np.abs(active_values))) if active_values.size else 0.1,
        ),
    )
    finite_joint = metrics.joint_sensitivity[
        np.isfinite(metrics.joint_sensitivity)
    ]
    j_max = max(
        1.0,
        float(np.percentile(finite_joint, 99.5)) if finite_joint.size else 1.0,
    )
    selectivity_cmap = SELECTIVITY_CMAP.copy()
    selectivity_cmap.set_bad("#D3D9DE")
    active_selectivity = np.ma.array(metrics.selectivity, mask=~metrics.active)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 5.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    selectivity = _heatmap(
        axes[0],
        active_selectivity,
        title=r"Relative selectivity $D_{\rm rel}$ (active heads)",
        cmap=selectivity_cmap,
        norm=TwoSlopeNorm(vmin=-d_max, vcenter=0.0, vmax=d_max),
    )
    joint = _heatmap(
        axes[1],
        metrics.joint_sensitivity,
        title=r"Joint sensitivity $J$",
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=j_max),
    )
    d_bar = fig.colorbar(selectivity, ax=axes[0], shrink=0.86, pad=0.015)
    d_bar.set_label(
        r"$D_{\rm rel}$  (structural $\leftarrow$  $\rightarrow$ semantic)"
    )
    j_bar = fig.colorbar(joint, ax=axes[1], shrink=0.86, pad=0.015)
    j_bar.set_label(r"$J$")
    fig.suptitle(title, fontsize=18, y=1.04)
    return fig


def _scatter_heads(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    active: np.ndarray | None = None,
):
    layers, heads = x.shape
    layer_ids = np.repeat(np.arange(layers), heads)
    finite = np.isfinite(x.reshape(-1)) & np.isfinite(y.reshape(-1))
    if active is not None:
        inactive = finite & ~active.reshape(-1)
        if np.any(inactive):
            ax.scatter(
                x.reshape(-1)[inactive],
                y.reshape(-1)[inactive],
                s=24,
                color="#C7D0D6",
                alpha=0.55,
                linewidth=0,
                label="Inactive",
                zorder=1,
            )
        finite &= active.reshape(-1)
    return ax.scatter(
        x.reshape(-1)[finite],
        y.reshape(-1)[finite],
        c=layer_ids[finite],
        cmap="viridis",
        norm=Normalize(0, max(layers - 1, 1)),
        s=42,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.92,
        zorder=2,
    )


def _annotate_selected_scatter(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    selected_heads: Mapping[str, Head] | None,
) -> None:
    if not selected_heads:
        return
    for role, head in selected_heads.items():
        layer, index = head
        style = HEAD_STYLES.get(role, {"color": NAVY, "label": role.title()})
        point_x, point_y = float(x[layer, index]), float(y[layer, index])
        x_limits, y_limits = ax.get_xlim(), ax.get_ylim()
        x_fraction = (point_x - x_limits[0]) / max(
            x_limits[1] - x_limits[0], 1e-12
        )
        y_fraction = (point_y - y_limits[0]) / max(
            y_limits[1] - y_limits[0], 1e-12
        )
        x_offset = -10 if x_fraction > 0.72 else 10
        y_offset = -14 if y_fraction > 0.76 else 10
        ax.scatter(
            [point_x],
            [point_y],
            s=110,
            facecolors="none",
            edgecolors=style["color"],
            linewidths=2.1,
            zorder=5,
        )
        ax.annotate(
            f"{style['label']} ({_head_label(head)})",
            xy=(point_x, point_y),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            color=style["color"],
            fontsize=9,
            fontweight="bold",
            ha="right" if x_offset < 0 else "left",
            va="top" if y_offset < 0 else "bottom",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": style["color"],
                "linewidth": 0.8,
                "alpha": 0.94,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": style["color"],
                "linewidth": 1.1,
            },
            zorder=6,
        )


def plot_score_plane(
    metrics: CanonicalHeadMetrics,
    selected_heads: Mapping[str, Head] | None = None,
    *,
    title: str,
):
    apply_publication_style()
    x, y = metrics.normalized_structural, metrics.normalized_semantic
    fig, ax = plt.subplots(figsize=(7.7, 6.5), constrained_layout=True)
    scatter = _scatter_heads(ax, x, y)
    maximum = max(float(np.nanmax(x)), float(np.nanmax(y))) * 1.06
    ax.plot([0, maximum], [0, maximum], color=SLATE, linestyle="--", linewidth=1.2)
    ax.set(xlim=(0, maximum), ylim=(0, maximum))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Structural score $S_{\rm str}/\overline{S}_{\rm str}$")
    ax.set_ylabel(r"Semantic score $S_{\rm sem}/\overline{S}_{\rm sem}$")
    ax.set_title(title, fontsize=17, pad=12)
    ax.grid(True)
    _annotate_selected_scatter(ax, x, y, selected_heads)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Layer")
    colorbar.set_ticks(np.arange(metrics.num_layers))
    return fig


def automatic_selectivity_limits(
    selectivity: np.ndarray,
    active: np.ndarray | None = None,
) -> tuple[float, float]:
    finite = np.isfinite(selectivity)
    if active is not None:
        finite &= active
    values = selectivity[finite]
    if not values.size:
        return -0.1, 0.1
    lower, upper = np.nanpercentile(values, [0.25, 99.75])
    span = max(float(upper - lower), 0.1)
    lower = min(float(lower - 0.08 * span), -0.03)
    upper = max(float(upper + 0.08 * span), 0.03)
    return max(-1.0, lower), min(1.0, upper)


def plot_selectivity_joint_plane(
    metrics: CanonicalHeadMetrics,
    selected_heads: Mapping[str, Head] | None = None,
    *,
    xlim: tuple[float, float] | None = None,
    active_only: bool = True,
    title: str,
):
    apply_publication_style()
    x, y = metrics.selectivity, metrics.joint_sensitivity
    fig, ax = plt.subplots(figsize=(8.0, 5.9), constrained_layout=True)
    scatter = _scatter_heads(
        ax, x, y, active=metrics.active if active_only else None
    )
    ax.axvline(0, color=SLATE, linestyle="--", linewidth=1.1)
    ax.set_xlim(
        *(xlim or automatic_selectivity_limits(x, metrics.active if active_only else None))
    )
    ax.set_ylim(bottom=0, top=float(np.nanmax(y)) * 1.08)
    ax.set_xlabel(
        r"Relative selectivity $D_{\rm rel}$"
        "\n(structural $\\leftarrow$   $\\rightarrow$ semantic)"
    )
    ax.set_ylabel(r"Joint sensitivity $J$")
    ax.set_title(title, fontsize=17, pad=12)
    ax.grid(True)
    _annotate_selected_scatter(ax, x, y, selected_heads)
    if active_only and np.any(~metrics.active):
        ax.legend(loc="upper left", fontsize=9)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Layer")
    colorbar.set_ticks(np.arange(metrics.num_layers))
    return fig


def _node_conditioned_attention(attention: np.ndarray) -> np.ndarray:
    attention = np.asarray(attention, dtype=np.float64)
    if attention.ndim != 2 or attention.shape[0] != attention.shape[1]:
        raise ValueError(f"expected square attention matrix, got {attention.shape}")
    denominator = np.clip(attention.sum(axis=-1, keepdims=True), 1e-12, None)
    return attention / denominator


def _graph_layout(example: Mapping[str, Any]) -> tuple[dict[int, Any], list[tuple[int, int]]]:
    import networkx as nx

    n = int(example["n_atoms"])
    edge_index = np.asarray(example["edge_index"], dtype=np.int64)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(
        (int(source), int(target))
        for source, target in edge_index.T
        if int(source) != int(target)
    )
    edges = sorted(
        {
            tuple(sorted((int(source), int(target))))
            for source, target in edge_index.T
            if int(source) != int(target)
        }
    )
    positions = nx.spring_layout(graph, seed=42, weight=None)
    return positions, edges


def _draw_graph(
    ax,
    example: Mapping[str, Any],
    *,
    positions: Mapping[int, Any],
    edges: list[tuple[int, int]],
    attention_mass: np.ndarray | None = None,
    vmax: float | None = None,
) -> None:
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(int(example["n_atoms"])))
    graph.add_edges_from(edges)
    labels = {
        index: f"{index}\n{element}"
        for index, element in enumerate(example["node_labels"])
    }
    if attention_mass is None:
        colours = ["#EDF2F5"] * len(labels)
        sizes = [620] * len(labels)
        label_colours = {index: NAVY for index in labels}
    else:
        values = np.asarray(attention_mass, dtype=np.float64)
        norm = Normalize(vmin=0.0, vmax=max(float(vmax or values.max()), 1e-12))
        colours = [ATTENTION_CMAP(0.18 + 0.72 * float(norm(value))) for value in values]
        sizes = [520 + 620 * np.sqrt(float(norm(value))) for value in values]
        label_colours = {
            index: "white" if float(norm(value)) >= 0.48 else NAVY
            for index, value in enumerate(values)
        }
    nx.draw_networkx_edges(
        graph, positions, ax=ax, edge_color="#8B98A1", width=1.4, alpha=0.85
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color=colours,
        node_size=sizes,
        edgecolors=NAVY,
        linewidths=0.8,
    )
    for index, label in labels.items():
        nx.draw_networkx_labels(
            graph,
            positions,
            labels={index: label},
            ax=ax,
            font_size=7,
            font_color=label_colours[index],
        )
    ax.set_axis_off()


def plot_attention_grid(
    examples_payload: Mapping[str, Any],
    *,
    role: str,
    head: Head,
    per_graph_coordinates: Mapping[int, Mapping[str, float]],
    net_d_rel: float,
    net_joint_sensitivity: float,
    title_label: str | None = None,
):
    """GRIT attention views with aggregate and graph-local coordinates."""

    apply_publication_style()
    examples = list(examples_payload["examples"])
    if not examples:
        raise ValueError("the attention grid requires at least one example")
    matrices = [
        _node_conditioned_attention(example["attention"][role])
        for example in examples
    ]
    matrix_max = max(
        max(float(np.nanpercentile(matrix, 99)), 1e-6) for matrix in matrices
    )
    inbound = [matrix.mean(axis=0) for matrix in matrices]
    inbound_max = max(
        max(float(np.nanpercentile(values, 99)), 1e-6) for values in inbound
    )
    num_rows = len(examples)
    fig = plt.figure(
        figsize=(13.2, 1.45 + 3.25 * num_rows), constrained_layout=True
    )
    grid = fig.add_gridspec(
        num_rows + 1,
        3,
        height_ratios=[0.15, *([1.0] * num_rows)],
        width_ratios=[1.0, 1.08, 1.12],
    )
    title_axis = fig.add_subplot(grid[0, :])
    title_axis.axis("off")
    axes = np.asarray(
        [
            [fig.add_subplot(grid[row + 1, column]) for column in range(3)]
            for row in range(num_rows)
        ],
        dtype=object,
    )
    task = str(examples_payload["task"])
    for row, (example, matrix) in enumerate(zip(examples, matrices)):
        graph_index = int(example["dataset_index"])
        graph_coordinates = per_graph_coordinates.get(
            graph_index, per_graph_coordinates.get(str(graph_index))
        )
        if graph_coordinates is None:
            raise KeyError(
                f"no graph-local coordinates were supplied for eval index {graph_index}"
            )
        d_rel = float(
            graph_coordinates.get(
                "D_rel", graph_coordinates.get("selectivity", np.nan)
            )
        )
        joint = float(
            graph_coordinates.get(
                "J", graph_coordinates.get("joint_sensitivity", np.nan)
            )
        )
        positions, edges = _graph_layout(example)
        _draw_graph(
            axes[row, 0], example, positions=positions, edges=edges
        )
        lower, upper = axes[row, 0].get_ylim()
        axes[row, 0].set_ylim(lower, upper + 0.32 * (upper - lower))
        axes[row, 0].text(
            0.01,
            0.99,
            f"{task} eval index {graph_index}\n"
            rf"Graph-local $D_{{\rm rel}}={d_rel:+.3f}$"
            "\n"
            rf"Graph-local $J={joint:.3f}$",
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color=NAVY,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.87},
        )
        _draw_graph(
            axes[row, 1],
            example,
            positions=positions,
            edges=edges,
            attention_mass=inbound[row],
            vmax=inbound_max,
        )
        image = axes[row, 2].imshow(
            matrix,
            cmap=ATTENTION_CMAP,
            vmin=0,
            vmax=matrix_max,
            interpolation="nearest",
            aspect="equal",
            rasterized=True,
        )
        axes[row, 2].set_xlabel("Key atom")
        axes[row, 2].set_ylabel("Query atom")
        axes[row, 2].set_xticks(np.arange(matrix.shape[0]))
        axes[row, 2].set_yticks(np.arange(matrix.shape[0]))
        axes[row, 2].tick_params(labelsize=6, length=2)
    for column, label in enumerate(
        ["Molecular graph", "Attention inflow", "Node-conditioned attention"]
    ):
        axes[0, column].set_title(label, fontsize=12, pad=8)
    style = HEAD_STYLES.get(role, {"label": role.replace("_", " ").title()})
    title_axis.text(
        0.5,
        0.76,
        f"{task}: {title_label or style['label']} — {_head_label(head)}",
        ha="center",
        va="center",
        fontsize=18,
        color=NAVY,
    )
    title_axis.text(
        0.5,
        0.16,
        rf"Aggregate: $D_{{\rm rel}}={float(net_d_rel):+.3f};\quad "
        rf"J={float(net_joint_sensitivity):.3f}$",
        ha="center",
        va="center",
        fontsize=14,
        color=NAVY,
    )
    colorbar = fig.colorbar(
        image, ax=axes[:, 2], location="right", shrink=0.72, pad=0.02
    )
    colorbar.set_label("Attention weight")
    return fig


def _pca(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float64)
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    if np.linalg.norm(centered) < 1e-12:
        return np.zeros((len(vectors), 2)), np.zeros(2)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    components = min(2, len(right))
    coordinates = np.zeros((len(vectors), 2), dtype=np.float64)
    coordinates[:, :components] = centered @ right[:components].T
    variance = singular**2
    explained = np.zeros(2, dtype=np.float64)
    explained[:components] = variance[:components] / np.clip(
        variance.sum(), 1e-12, None
    )
    return coordinates, explained


def _group_pca_labels(
    labels: list[str], *, maximum_categories: int = 9, minimum_count: int = 5
) -> list[str]:
    counts = Counter(labels)
    retained = {
        label
        for label, count in counts.most_common(maximum_categories - 1)
        if count >= minimum_count
    }
    return [label if label in retained else "Other / rare" for label in labels]


def plot_av_pca(
    payload: Mapping[str, Any],
    *,
    maximum_categories: int = 12,
    minimum_count: int = 1,
    title_label: str | None = None,
    d_rel: float | None = None,
    joint_sensitivity: float | None = None,
):
    apply_publication_style()
    coordinates, explained = _pca(np.asarray(payload["vectors"]))
    labels = _group_pca_labels(
        list(payload["labels"]),
        maximum_categories=maximum_categories,
        minimum_count=minimum_count,
    )
    counts = Counter(labels)
    categories = [
        label for label in ELEMENT_COLORS if label in counts and label != "Other / rare"
    ]
    categories.extend(
        sorted(
            label
            for label in counts
            if label not in ELEMENT_COLORS and label != "Other / rare"
        )
    )
    if "Other / rare" in counts:
        categories.append("Other / rare")
    dynamic = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)
    labels_array = np.asarray(labels)
    for category_index, category in enumerate(categories):
        color = ELEMENT_COLORS.get(
            category, dynamic(category_index / max(1, len(categories) - 1))
        )
        mask = labels_array == category
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=34,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.78,
            label=f"{category} (n={int(mask.sum())})",
        )
    ax.axhline(0, color=LIGHT_GRID, linewidth=0.8, zorder=0)
    ax.axvline(0, color=LIGHT_GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel(f"PC1 ({100 * explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100 * explained[1]:.1f}% variance)")
    head = tuple(payload["head"])
    metric_line = ""
    if d_rel is not None and joint_sensitivity is not None:
        metric_line = (
            "\n"
            + rf"$D_{{\rm rel}}={float(d_rel):+.3f};\quad "
            + rf"J={float(joint_sensitivity):.3f}$"
        )
    descriptor = (
        f"{title_label} — {_head_label(head)}"
        if title_label
        else _head_label(head)
    )
    ax.set_title(
        f"PCA of native GRIT routed head output — {descriptor}"
        f"{metric_line}\n"
        f"$n={int(payload['n_used'])}$ {payload['task']} graphs",
        fontsize=15,
    )
    ax.grid(False)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        handletextpad=0.5,
    )
    return fig


def plot_hop_attention_mass(
    metrics: CanonicalHeadMetrics,
    head: Head,
    *,
    title: str | None = None,
):
    if metrics.clean_attention_distance is None:
        raise ValueError("score cache has no clean_attention_distance profile")
    apply_publication_style()
    layer, index = head
    values = np.asarray(metrics.clean_attention_distance[layer, index])
    labels = list(metrics.distance_axis)
    special = np.asarray(
        [
            str(label).lower().replace(" ", "_") in {"graph_token", "virtual"}
            for label in labels
        ]
    )
    x = np.arange(len(values), dtype=np.float64)
    x[special] += 0.8
    colors = [GOLD if value else TEAL for value in special]
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    bars = ax.bar(
        x,
        values,
        width=0.72,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(label).replace("_", "\n") for label in labels]
    )
    ax.set_xlabel("Shortest-path distance")
    ax.set_ylabel("Mean clean attention mass")
    ax.set_ylim(0, max(float(values.max()) * 1.18, 0.05))
    ax.set_title(
        title or f"Attention mass by hop distance — {_head_label(head)}",
        fontsize=15,
    )
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        if value >= 0.025:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=NAVY,
            )
    return fig


def plot_logit_spread(
    logit_payload: Mapping[str, Any],
    *,
    title: str = "GRIT raw attention-logit spread",
):
    """GRIT node-only counterfactual and actual relation-conditioned spread."""

    apply_publication_style()
    node = np.asarray(logit_payload["node_std_mean"])
    node_error = np.asarray(logit_payload["node_std_std"])
    relation = np.asarray(logit_payload["relation_std_mean"])
    relation_error = np.asarray(logit_payload["relation_std_std"])
    layers = np.arange(len(node))
    fig, ax = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)
    ax.fill_between(
        layers,
        np.clip(node - node_error, 0.0, None),
        node + node_error,
        color=BLUE,
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(
        layers,
        node,
        color=BLUE,
        marker="o",
        linewidth=2.0,
        markersize=5,
        label="Node-only counterfactual: mean ±1 SD",
    )
    ax.fill_between(
        layers,
        np.clip(relation - relation_error, 0.0, None),
        relation + relation_error,
        color=ORANGE,
        alpha=0.16,
        linewidth=0,
    )
    ax.plot(
        layers,
        relation,
        color=ORANGE,
        marker="s",
        linewidth=2.0,
        markersize=5,
        label="Relation-conditioned actual: mean ±1 SD",
    )
    ax.set_xticks(layers)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean key-wise raw-logit standard deviation")
    ax.set_title(
        f"{title}\n$n={int(logit_payload['n_used'])}$ "
        f"{logit_payload['task']} graphs",
        fontsize=15,
    )
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend()
    return fig


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank, right_rank = _average_ranks(left), _average_ranks(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def plot_selectivity_vs_logit_ratio(
    metrics: CanonicalHeadMetrics,
    logit_payload: Mapping[str, Any],
    selected_heads: Mapping[str, Head] | None = None,
    *,
    active_only: bool = True,
):
    """``D_rel`` versus GRIT node-only/relation-conditioned logit balance."""

    apply_publication_style()
    ratio = np.asarray(logit_payload["log_r_mean"], dtype=np.float64)
    if ratio.shape != metrics.shape:
        raise ValueError(
            f"logit ratio shape {ratio.shape} != head metric grid {metrics.shape}"
        )
    selectivity = metrics.selectivity
    finite = np.isfinite(ratio) & np.isfinite(selectivity)
    if active_only:
        finite &= metrics.active
    correlation = (
        _spearman_correlation(ratio[finite], selectivity[finite])
        if finite.sum() >= 3
        else float("nan")
    )
    correlation_label = (
        rf"; active-head Spearman $\rho={correlation:.2f}$"
        if np.isfinite(correlation)
        else ""
    )
    fig, ax = plt.subplots(figsize=(8.1, 5.8), constrained_layout=True)
    scatter = _scatter_heads(
        ax,
        ratio,
        selectivity,
        active=metrics.active if active_only else None,
    )
    ax.axhline(0, color=SLATE, linestyle="--", linewidth=1.0)
    ax.axvline(0, color=SLATE, linestyle="--", linewidth=1.0)
    ax.set_xlabel(
        r"$\log_{10}[\mathrm{std}(\mathrm{node\ only})/"
        r"\mathrm{std}(\mathrm{relation\ conditioned})]$"
    )
    ax.set_ylabel(r"Relative selectivity $D_{\rm rel}$")
    ax.set_title(
        "Relative selectivity versus GRIT logit balance\n"
        f"$n={int(logit_payload['n_used'])}$ {logit_payload['task']} graphs"
        f"{correlation_label}",
        fontsize=15,
    )
    ax.grid(True)
    _annotate_selected_scatter(ax, ratio, selectivity, selected_heads)
    if active_only and np.any(~metrics.active):
        ax.legend(loc="lower left", fontsize=9)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Layer")
    colorbar.set_ticks(np.arange(metrics.num_layers))
    return fig


def save_figure_bundle(
    figure,
    output_directory: str | Path,
    stem: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    dpi: int = 300,
) -> dict[str, Path]:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": directory / f"{stem}.png",
        "pdf": directory / f"{stem}.pdf",
        "metadata": directory / f"{stem}.json",
    }
    figure.savefig(paths["png"], dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    paths["metadata"].write_text(
        json.dumps(
            {"figure": stem, **dict(metadata or {})},
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


__all__ = [
    "apply_publication_style",
    "automatic_selectivity_limits",
    "plot_attention_grid",
    "plot_av_pca",
    "plot_coordinate_heatmaps",
    "plot_hop_attention_mass",
    "plot_logit_spread",
    "plot_score_heatmaps",
    "plot_score_plane",
    "plot_selectivity_joint_plane",
    "plot_selectivity_vs_logit_ratio",
    "save_figure_bundle",
]
