"""Generate a Matplotlib recreation of the range-measure Figure 1 schematic.

The paper PDF embeds the original asset as a Matplotlib PDF
(``combined_task_range_node_graph.pdf``). This script uses the same likely
recipe: NetworkX for the grid graph and shortest-path distances, then manual
Matplotlib patches for the actual visual design.

The script is intentionally coordinate-driven. For schematic paper figures this
is more controllable than automatic NetworkX drawing.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "graph_specialisation_metrics_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROWS = 3
COLS = 4
U_NODE = (2, 0)
V_NODE = (1, 2)

PANEL_FILL = "#dbe9f4"
PANEL_EDGE = "#0d3040"
EDGE_COLOR = "#111111"
NODE_EDGE = "#111111"
PANEL_LINEWIDTH = 1.45
MINI_EDGE_LINEWIDTH = 1.42
MINI_NODE_LINEWIDTH = 1.25
RANGE_EDGE_LINEWIDTH = 1.78
RANGE_NODE_LINEWIDTH = 1.62
ARROW_LINEWIDTH = 1.55


def make_grid_graph() -> nx.Graph:
    return nx.grid_2d_graph(ROWS, COLS)


def mini_positions(origin: tuple[float, float], spacing: float = 0.62) -> dict[tuple[int, int], tuple[float, float]]:
    ox, oy = origin
    return {(r, c): (ox + c * spacing, oy + (ROWS - 1 - r) * spacing) for r in range(ROWS) for c in range(COLS)}


def graph_distances(graph: nx.Graph, source: tuple[int, int]) -> dict[tuple[int, int], float]:
    return {node: float(dist) for node, dist in nx.single_source_shortest_path_length(graph, source).items()}


def influence_values(target: tuple[int, int]) -> dict[tuple[int, int], float]:
    """Hand-tuned values chosen to match the paper's schematic distributions."""

    if target == V_NODE:
        values = np.array(
            [
                [0.00, 0.52, 0.00, 0.55],
                [0.45, 0.16, 0.90, 0.02],
                [0.00, 0.70, 0.00, 0.50],
            ],
            dtype=float,
        )
    elif target == U_NODE:
        values = np.array(
            [
                [0.62, 0.40, 0.16, 0.00],
                [0.76, 0.52, 0.40, 0.14],
                [1.00, 0.70, 0.57, 0.34],
            ],
            dtype=float,
        )
    else:
        raise ValueError(target)
    return {(r, c): float(values[r, c]) for r in range(ROWS) for c in range(COLS)}


def range_values() -> dict[tuple[int, int], float]:
    """Node-level range opacities for the right-hand graph."""

    values = np.array(
        [
            [0.48, 0.00, 0.77, 0.64],
            [0.42, 0.66, 0.72, 0.26],
            [0.82, 0.00, 0.75, 0.62],
        ],
        dtype=float,
    )
    return {(r, c): float(values[r, c]) for r in range(ROWS) for c in range(COLS)}


def draw_panel(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.08,rounding_size=0.38",
            facecolor=PANEL_FILL,
            edgecolor=PANEL_EDGE,
            linewidth=PANEL_LINEWIDTH,
            zorder=0,
        )
    )


def draw_grid_edges(
    ax: plt.Axes,
    graph: nx.Graph,
    positions: dict[tuple[int, int], tuple[float, float]],
    linewidth: float,
) -> None:
    for a, b in graph.edges:
        ax.plot(
            [positions[a][0], positions[b][0]],
            [positions[a][1], positions[b][1]],
            color=EDGE_COLOR,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=1,
        )


def draw_nodes(
    ax: plt.Axes,
    positions: dict[tuple[int, int], tuple[float, float]],
    values: dict[tuple[int, int], float],
    cmap_name: str,
    norm: Normalize,
    radius: float,
    target: tuple[int, int] | None = None,
    target_label: str | None = None,
    target_is_white: bool = False,
    labels: dict[tuple[int, int], str] | None = None,
    linewidth: float = MINI_NODE_LINEWIDTH,
    target_fontsize: float = 10.0,
    label_fontsize: float = 11.0,
) -> None:
    cmap = plt.get_cmap(cmap_name)
    labels = labels or {}
    for node, (x, y) in positions.items():
        facecolor = "white" if target_is_white and node == target else cmap(norm(values[node]))
        circle = Circle(
            (x, y),
            radius=radius,
            facecolor=facecolor,
            edgecolor=NODE_EDGE,
            linewidth=linewidth,
            zorder=3,
        )
        ax.add_patch(circle)

        if node == target and target_label is not None:
            ax.text(
                x,
                y - 0.005,
                rf"${target_label}$",
                ha="center",
                va="center",
                fontsize=target_fontsize,
                zorder=4,
            )
        if node in labels:
            ax.text(x, y, labels[node], ha="center", va="center", fontsize=label_fontsize, zorder=4)


def draw_mini_graph(
    ax: plt.Axes,
    graph: nx.Graph,
    origin: tuple[float, float],
    values: dict[tuple[int, int], float],
    cmap_name: str,
    norm: Normalize,
    target: tuple[int, int],
    target_label: str,
    target_is_white: bool,
) -> None:
    positions = mini_positions(origin)
    draw_grid_edges(ax, graph, positions, linewidth=MINI_EDGE_LINEWIDTH)
    draw_nodes(
        ax,
        positions,
        values,
        cmap_name,
        norm,
        radius=0.165,
        target=target,
        target_label=target_label,
        target_is_white=target_is_white,
        linewidth=MINI_NODE_LINEWIDTH,
        target_fontsize=10.0,
    )


def draw_range_graph(ax: plt.Axes, graph: nx.Graph, origin: tuple[float, float]) -> dict[tuple[int, int], tuple[float, float]]:
    positions = mini_positions(origin, spacing=1.00)
    labels = {
        U_NODE: r"$\rho_u$",
        V_NODE: r"$\rho_v$",
    }
    draw_grid_edges(ax, graph, positions, linewidth=RANGE_EDGE_LINEWIDTH)
    draw_nodes(
        ax,
        positions,
        range_values(),
        "Purples",
        Normalize(vmin=0.0, vmax=1.0),
        radius=0.245,
        labels=labels,
        linewidth=RANGE_NODE_LINEWIDTH,
        label_fontsize=11.0,
    )
    return positions


def draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    rad: float = 0.0,
    color: str = "#0d3040",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=ARROW_LINEWIDTH,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=6,
            zorder=2,
        )
    )


def make_figure() -> plt.Figure:
    graph = make_grid_graph()
    distance_norm = Normalize(vmin=0, vmax=5)
    influence_norm = Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=(10.7, 3.75))
    ax.set_xlim(0, 13.9)
    ax.set_ylim(0.36, 5.20)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_panel(ax, 0.25, 3.02, 6.95, 2.05)
    draw_panel(ax, 0.25, 0.56, 6.95, 2.05)

    ax.text(2.20, 4.87, r"Distance $d_G(v,\cdot)$", ha="center", va="center", fontsize=9.9)
    ax.text(5.18, 4.87, r"Influence $I_v$", ha="center", va="center", fontsize=9.9)
    ax.text(2.20, 2.41, r"Distance $d_G(u,\cdot)$", ha="center", va="center", fontsize=9.9)
    ax.text(5.18, 2.41, r"Influence $I_u$", ha="center", va="center", fontsize=9.9)

    draw_mini_graph(
        ax,
        graph,
        origin=(1.06, 3.33),
        values=graph_distances(graph, V_NODE),
        cmap_name="Reds",
        norm=distance_norm,
        target=V_NODE,
        target_label="v",
        target_is_white=True,
    )
    draw_mini_graph(
        ax,
        graph,
        origin=(4.14, 3.33),
        values=influence_values(V_NODE),
        cmap_name="Blues",
        norm=influence_norm,
        target=V_NODE,
        target_label="v",
        target_is_white=False,
    )
    draw_mini_graph(
        ax,
        graph,
        origin=(1.06, 0.87),
        values=graph_distances(graph, U_NODE),
        cmap_name="Reds",
        norm=distance_norm,
        target=U_NODE,
        target_label="u",
        target_is_white=True,
    )
    draw_mini_graph(
        ax,
        graph,
        origin=(4.14, 0.87),
        values=influence_values(U_NODE),
        cmap_name="Blues",
        norm=influence_norm,
        target=U_NODE,
        target_label="u",
        target_is_white=False,
    )

    ax.text(9.92, 4.42, "Node-level Range", ha="center", va="center", fontsize=14.2)
    range_positions = draw_range_graph(ax, graph, origin=(8.30, 1.58))

    draw_arrow(ax, start=(7.20, 4.02), end=(range_positions[V_NODE][0] - 0.05, range_positions[V_NODE][1] + 0.02), rad=0.32)
    draw_arrow(ax, start=(7.20, 1.78), end=(range_positions[U_NODE][0] - 0.05, range_positions[U_NODE][1] + 0.02), rad=0.0)

    ax.text(12.38, 2.55, "}", ha="center", va="center", fontsize=98, color="black")
    ax.text(13.00, 2.55, r"$\rho_G$", ha="left", va="center", fontsize=24, color="black")

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("visualisations/generated"),
        help="Directory where the PDF and PNG will be written.",
    )
    parser.add_argument("--stem", default="range_measure_figure1", help="Output filename stem.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG export DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig = make_figure()
    pdf_path = args.out_dir / f"{args.stem}.pdf"
    png_path = args.out_dir / f"{args.stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.03, dpi=args.dpi)
    plt.close(fig)

    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
