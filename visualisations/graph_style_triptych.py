"""Render three publication-style graph motifs in cohesive palette variants."""

from __future__ import annotations

import argparse
import itertools
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import TypedDict

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "graph_specialisation_metrics_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch


class Palette(TypedDict):
    ink: str
    edge: str
    shadow: str
    nodes: tuple[str, ...]
    accent: str


PALETTES: dict[str, Palette] = {
    "atlantic": {
        "ink": "#18313B",
        "edge": "#405A64",
        "shadow": "#20323A",
        "nodes": ("#3F738F", "#5D91A8", "#79AFAE", "#A5C8BC", "#D1E0D6"),
        "accent": "#E58B68",
    },
    "mulberry": {
        "ink": "#29263F",
        "edge": "#57536D",
        "shadow": "#2B2938",
        "nodes": ("#504B79", "#69658F", "#8580AA", "#AAA5C5", "#D0CDE0"),
        "accent": "#D9A24F",
    },
}


def degrees(nodes: tuple[int, ...], edges: tuple[tuple[int, int], ...]) -> Counter[int]:
    result: Counter[int] = Counter({node: 0 for node in nodes})
    for a, b in edges:
        result[a] += 1
        result[b] += 1
    return result


def draw_graph(
    ax: plt.Axes,
    *,
    nodes: tuple[int, ...],
    edges: tuple[tuple[int, int], ...],
    positions: dict[int, tuple[float, float]],
    color_indices: dict[int, int | str],
    palette: Palette,
    curved_edges: dict[tuple[int, int], float] | None = None,
    edge_width: float = 1.12,
    bottleneck_edges: set[tuple[int, int]] | None = None,
) -> None:
    curved_edges = curved_edges or {}
    bottleneck_edges = bottleneck_edges or set()
    node_degree = degrees(nodes, edges)

    for a, b in edges:
        rad = curved_edges.get((a, b), curved_edges.get((b, a), 0.0))
        is_bottleneck = (a, b) in bottleneck_edges or (b, a) in bottleneck_edges
        width = edge_width + (0.28 if is_bottleneck else 0.0)
        ax.add_patch(
            FancyArrowPatch(
                positions[a],
                positions[b],
                arrowstyle="-",
                connectionstyle=f"arc3,rad={rad}",
                linewidth=width,
                color=palette["ink"] if is_bottleneck else palette["edge"],
                alpha=0.90 if is_bottleneck else 0.84,
                shrinkA=7,
                shrinkB=7,
                capstyle="round",
                joinstyle="round",
                zorder=1,
            )
        )

    for node in nodes:
        x, y = positions[node]
        radius = 0.110 + 0.013 * math.sqrt(max(1, node_degree[node]))
        color_key = color_indices[node]
        fill = palette["accent"] if color_key == "accent" else palette["nodes"][int(color_key)]

        ax.add_patch(
            Circle(
                (x + 0.022, y - 0.030),
                radius * 1.04,
                facecolor=palette["shadow"],
                edgecolor="none",
                alpha=0.10,
                zorder=2,
            )
        )
        ax.add_patch(
            Circle(
                (x, y),
                radius * 1.09,
                facecolor="#ffffff",
                edgecolor="none",
                zorder=3,
            )
        )
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor=fill,
                edgecolor=palette["ink"],
                linewidth=0.82,
                zorder=4,
            )
        )


def representative_graph() -> tuple[
    tuple[int, ...],
    tuple[tuple[int, int], ...],
    dict[int, tuple[float, float]],
    dict[int, int | str],
]:
    """An irregular, cyclic graph with degrees ranging from one to six."""
    nodes = tuple(range(12))
    edges = (
        (0, 1),
        (0, 8),
        (0, 10),
        (1, 2),
        (1, 8),
        (2, 3),
        (2, 8),
        (2, 9),
        (3, 4),
        (3, 9),
        (4, 5),
        (4, 9),
        (5, 6),
        (5, 9),
        (5, 11),
        (6, 7),
        (6, 8),
        (6, 9),
        (7, 8),
        (7, 10),
        (8, 9),
    )
    positions = {
        0: (0.26, 1.66),
        1: (0.72, 2.39),
        2: (1.52, 2.70),
        3: (2.32, 2.35),
        4: (2.91, 1.66),
        5: (2.58, 0.82),
        6: (1.77, 0.38),
        7: (0.88, 0.59),
        8: (1.09, 1.43),
        9: (1.98, 1.52),
        10: (0.18, 0.70),
        11: (3.03, 0.57),
    }
    color_indices: dict[int, int | str] = {
        0: 2,
        1: 1,
        2: 0,
        3: 2,
        4: 3,
        5: 1,
        6: 3,
        7: 2,
        8: "accent",
        9: 0,
        10: 4,
        11: 4,
    }
    return nodes, edges, positions, color_indices


def cyclic_graph() -> tuple[
    tuple[int, ...],
    tuple[tuple[int, int], ...],
    dict[int, tuple[float, float]],
    dict[int, int | str],
    dict[tuple[int, int], float],
]:
    nodes = tuple(range(12))
    centre = (4.90, 1.50)
    radius_x, radius_y = 1.18, 1.15
    positions = {}
    for node in nodes:
        angle = math.pi / 2 - 2 * math.pi * node / len(nodes)
        positions[node] = (
            centre[0] + radius_x * math.cos(angle),
            centre[1] + radius_y * math.sin(angle),
        )
    ring = tuple((node, (node + 1) % len(nodes)) for node in nodes)
    chords = ((0, 5), (1, 7), (3, 9), (4, 10), (6, 11))
    edges = ring + chords
    color_indices: dict[int, int | str] = {
        node: (0, 2, 3, 1, 4, 2)[node % 6] for node in nodes
    }
    color_indices[0] = "accent"
    curved = {
        (0, 5): 0.10,
        (1, 7): -0.12,
        (3, 9): 0.10,
        (4, 10): -0.12,
        (6, 11): 0.10,
    }
    return nodes, edges, positions, color_indices, curved


def barbell_graph() -> tuple[
    tuple[int, ...],
    tuple[tuple[int, int], ...],
    dict[int, tuple[float, float]],
    dict[int, int | str],
    set[tuple[int, int]],
]:
    """Two five-node cliques joined by a two-node bottleneck path."""
    left = tuple(range(5))
    right = tuple(range(5, 10))
    bridge_nodes = (10, 11)
    nodes = left + right + bridge_nodes
    positions = {
        0: (7.05, 2.30),
        1: (6.69, 1.57),
        2: (6.98, 0.78),
        3: (7.62, 0.66),
        4: (7.83, 1.50),
        5: (9.44, 1.50),
        6: (9.66, 2.31),
        7: (10.28, 2.17),
        8: (10.56, 1.42),
        9: (10.06, 0.72),
        10: (8.36, 1.50),
        11: (8.91, 1.50),
    }
    left_edges = tuple(itertools.combinations(left, 2))
    right_edges = tuple(itertools.combinations(right, 2))
    bridge_path = ((4, 10), (10, 11), (11, 5))
    edges = left_edges + right_edges + bridge_path
    color_indices: dict[int, int | str] = {
        0: 0,
        1: 2,
        2: 3,
        3: 1,
        4: 0,
        5: 0,
        6: 1,
        7: 3,
        8: 2,
        9: 4,
        10: "accent",
        11: "accent",
    }
    return nodes, edges, positions, color_indices, set(bridge_path)


def make_figure(palette_name: str) -> plt.Figure:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    palette = PALETTES[palette_name]
    fig, ax = plt.subplots(figsize=(10.0, 3.15))
    ax.set_xlim(0.00, 10.75)
    ax.set_ylim(0.05, 3.08)
    ax.set_aspect("equal")
    ax.axis("off")

    nodes, edges, positions, colors = representative_graph()
    draw_graph(
        ax,
        nodes=nodes,
        edges=edges,
        positions=positions,
        color_indices=colors,
        palette=palette,
        edge_width=1.02,
    )

    nodes, edges, positions, colors, curved = cyclic_graph()
    draw_graph(
        ax,
        nodes=nodes,
        edges=edges,
        positions=positions,
        color_indices=colors,
        palette=palette,
        curved_edges=curved,
        edge_width=1.04,
    )

    nodes, edges, positions, colors, bottleneck = barbell_graph()
    draw_graph(
        ax,
        nodes=nodes,
        edges=edges,
        positions=positions,
        color_indices=colors,
        palette=palette,
        edge_width=0.78,
        bottleneck_edges=bottleneck,
    )

    return fig


def save_variant(
    *,
    palette_name: str,
    out_dir: Path,
    stem: str,
    dpi: int,
) -> tuple[Path, Path]:
    fig = make_figure(palette_name)
    pdf_path = out_dir / f"{stem}_{palette_name}.pdf"
    png_path = out_dir / f"{stem}_{palette_name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.05, dpi=dpi)
    plt.close(fig)
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("visualisations/generated"),
        help="Directory where the PDFs and PNGs will be written.",
    )
    parser.add_argument("--stem", default="graph_connectivity_triptych", help="Output filename stem.")
    parser.add_argument(
        "--palette",
        choices=("all", *PALETTES),
        default="all",
        help="Palette variant to render.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG export DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    palette_names = tuple(PALETTES) if args.palette == "all" else (args.palette,)
    for palette_name in palette_names:
        pdf_path, png_path = save_variant(
            palette_name=palette_name,
            out_dir=args.out_dir,
            stem=args.stem,
            dpi=args.dpi,
        )
        print(pdf_path)
        print(png_path)


if __name__ == "__main__":
    main()
