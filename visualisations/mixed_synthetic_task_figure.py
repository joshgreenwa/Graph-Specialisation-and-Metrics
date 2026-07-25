"""Draw a compact schematic of the balanced semantic/structural C16 task."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "graph_specialisation_metrics_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


INK = "#172630"
EDGE = "#87969F"
MUTED = "#647480"
GUIDE = "#DCE3E7"
NODE = "#E7EFF1"
NODE_ALT = "#DCE8EA"
QUERY = "#3F7893"
QUERY_LIGHT = "#E8F1F5"
SEM = "#DF6B59"
SEM_DARK = "#AE4438"
SEM_LIGHT = "#FCEDE9"
STR = "#3F8B78"
STR_DARK = "#286956"
STR_LIGHT = "#EAF5F0"
KEY = "#6B73B5"

N = 16
QUERY_NODE = 8
SOURCE_NODE = 3


def cycle_positions(
    centre: tuple[float, float],
    radius: float,
) -> dict[int, tuple[float, float]]:
    cx, cy = centre
    return {
        node: (
            cx + radius * math.cos(math.pi / 2 - 2 * math.pi * node / N),
            cy + radius * math.sin(math.pi / 2 - 2 * math.pi * node / N),
        )
        for node in range(N)
    }


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linewidth: float = 1.25,
    mutation: float = 11,
    rad: float = 0.0,
    zorder: int = 7,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=linewidth,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=3,
            zorder=zorder,
        )
    )


def mode_badge(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    face: str,
    edge: str,
) -> None:
    width = 1.30
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            0.30,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.75,
            zorder=8,
        )
    )
    ax.text(x + width / 2, y + 0.15, "50% of examples", ha="center", va="center", fontsize=6.4, color=INK, zorder=9)


def output_box(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    formula: str,
    face: str,
    edge: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            1.12,
            0.62,
            boxstyle="round,pad=0.035,rounding_size=0.10",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            zorder=5,
        )
    )
    ax.text(x + 0.56, y + 0.39, formula, ha="center", va="center", fontsize=9.2, color=INK, zorder=6)
    ax.text(x + 0.56, y + 0.14, r"$\in\{0,\ldots,7\}$", ha="center", va="center", fontsize=6.5, color=MUTED, zorder=6)


def draw_cycle(
    ax: plt.Axes,
    centre: tuple[float, float],
    *,
    source_color: str,
    source_ring: str,
    highlighted_path: tuple[int, ...] | None = None,
) -> dict[int, tuple[float, float]]:
    positions = cycle_positions(centre, 1.02)
    highlighted_edges = set()
    if highlighted_path:
        highlighted_edges = {
            tuple(sorted((highlighted_path[idx], highlighted_path[idx + 1])))
            for idx in range(len(highlighted_path) - 1)
        }

    for node in range(N):
        nxt = (node + 1) % N
        is_highlighted = tuple(sorted((node, nxt))) in highlighted_edges
        x0, y0 = positions[node]
        x1, y1 = positions[nxt]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=STR if is_highlighted else EDGE,
            linewidth=2.25 if is_highlighted else 0.92,
            alpha=1.0 if is_highlighted else 0.78,
            solid_capstyle="round",
            zorder=1 if not is_highlighted else 2,
        )

    for node in range(N):
        x, y = positions[node]
        fill = NODE if node % 2 == 0 else NODE_ALT
        edge = "#71828C"
        radius = 0.084
        if node == QUERY_NODE:
            ax.add_patch(Circle((x, y), 0.145, facecolor=QUERY_LIGHT, edgecolor=QUERY, linewidth=1.15, zorder=3))
            fill = QUERY
            edge = INK
            radius = 0.112
        elif node == SOURCE_NODE:
            ax.add_patch(Circle((x, y), 0.145, facecolor="white", edgecolor=source_ring, linewidth=1.35, zorder=3))
            fill = source_color
            edge = INK
            radius = 0.112

        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.78,
                zorder=4,
            )
        )
        if node == QUERY_NODE:
            ax.text(x, y - 0.002, r"$q$", ha="center", va="center", fontsize=8.1, fontstyle="italic", color="white", zorder=5)
        elif node == SOURCE_NODE:
            ax.text(x, y - 0.002, r"$s$", ha="center", va="center", fontsize=8.1, fontstyle="italic", color="white", zorder=5)

    return positions


def draw_semantic_panel(ax: plt.Axes) -> None:
    ax.text(0.40, 3.31, "Semantic mode", ha="left", va="top", fontsize=10.0, weight=500, color=SEM_DARK)
    ax.text(0.40, 3.00, "retrieve the marked source value", ha="left", va="top", fontsize=7.2, color=MUTED)
    mode_badge(ax, 3.38, 3.00, face=SEM_LIGHT, edge="#E9B8AE")

    positions = draw_cycle(ax, (1.72, 1.65), source_color=SEM, source_ring=SEM)
    qx, qy = positions[QUERY_NODE]
    sx, sy = positions[SOURCE_NODE]

    arrow(
        ax,
        (qx + 0.07, qy + 0.14),
        (sx - 0.13, sy - 0.12),
        color=KEY,
        linewidth=1.42,
        mutation=11,
        rad=0.0,
    )
    ax.text(1.66, 1.47, r"query key $k_s$", ha="center", va="center", fontsize=7.0, color=KEY)
    ax.text(sx + 0.18, sy + 0.32, "source", ha="left", va="center", fontsize=6.8, color=SEM_DARK)

    output_box(ax, 3.34, 1.73, formula=r"$y_{\rm sem}=v_s$", face=SEM_LIGHT, edge="#E7A99D")
    arrow(ax, (sx + 0.16, sy), (3.30, sy), color=SEM_DARK, linewidth=1.35, mutation=11)

    ax.text(
        2.42,
        0.28,
        r"graph distance does not predict $v_s$",
        ha="center",
        va="center",
        fontsize=6.9,
        color=MUTED,
    )


def draw_structural_panel(ax: plt.Axes) -> None:
    ax.text(5.36, 3.31, "Structural mode", ha="left", va="top", fontsize=10.0, weight=500, color=STR_DARK)
    ax.text(5.36, 3.00, "recover the query-source distance", ha="left", va="top", fontsize=7.2, color=MUTED)
    mode_badge(ax, 8.34, 3.00, face=STR_LIGHT, edge="#ADD2C4")

    shortest_path = (8, 7, 6, 5, 4, 3)
    positions = draw_cycle(
        ax,
        (6.68, 1.65),
        source_color=STR,
        source_ring=STR,
        highlighted_path=shortest_path,
    )
    sx, sy = positions[SOURCE_NODE]
    ax.text(6.68, 1.46, r"$d$ hops", ha="center", va="center", fontsize=8.2, color=STR_DARK)

    output_box(ax, 8.30, 1.73, formula=r"$y_{\rm str}=d-1$", face=STR_LIGHT, edge="#9FCAB9")
    arrow(ax, (sx + 0.16, sy), (8.25, sy), color=STR_DARK, linewidth=1.35, mutation=11)

    ax.text(
        7.38,
        0.28,
        r"node keys and values do not predict $d$",
        ha="center",
        va="center",
        fontsize=6.9,
        color=MUTED,
    )


def make_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(8.0, 3.45))
    ax.set_xlim(0.08, 9.88)
    ax.set_ylim(0.05, 4.02)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(4.98, 3.91, "Mixed semantic-structural task", ha="center", va="top", fontsize=9.8, weight=500, color=INK)
    ax.text(
        4.98,
        3.63,
        r"Each node receives a unique key $k_i$ and a random value $v_i\in\{0,\ldots,7\}$; $q$ and $s$ are marked on every $C_{16}$.",
        ha="center",
        va="top",
        fontsize=6.9,
        color=MUTED,
    )
    ax.plot([0.38, 9.58], [3.46, 3.46], color=GUIDE, linewidth=0.85)
    ax.plot([4.98, 4.98], [0.23, 3.33], color=GUIDE, linewidth=0.8)

    draw_semantic_panel(ax)
    draw_structural_panel(ax)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("visualisations/generated"),
        help="Directory where the PDF and PNG will be written.",
    )
    parser.add_argument("--stem", default="mixed_synthetic_task", help="Output filename stem.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG export DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    pdf_path = args.out_dir / f"{args.stem}.pdf"
    png_path = args.out_dir / f"{args.stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.05, dpi=args.dpi)
    plt.close(fig)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
