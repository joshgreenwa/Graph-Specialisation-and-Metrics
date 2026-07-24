"""Draw a compact dissertation overview of the measurement programme.

The figure follows one visual argument from graph input, through controlled
intervention and transported node-state change, to per-head responses and the
resulting semantic/structural head roles.
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
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Wedge


INK = "#18222d"
EDGE = "#34424e"
MUTED = "#657483"
GUIDE = "#dfe5ea"
SEM = "#dd5c53"
SEM_DARK = "#aa4039"
STR = "#388071"
STR_DARK = "#286255"
REPR = "#7770c5"
REPR_LIGHT = "#d9d5f2"
JOINT = "#7564a8"

NODE_COLORS = (
    "#78aee0",
    "#f0a05a",
    "#73c59d",
    "#e68191",
    "#b789da",
    "#f1d365",
    "#8fc4ca",
)
NODES = tuple(range(7))
EDGES = (
    (0, 1),
    (1, 2),
    (0, 3),
    (1, 4),
    (2, 4),
    (2, 6),
    (3, 4),
    (4, 5),
    (5, 6),
)
POS = {
    0: (0.00, 1.12),
    1: (0.95, 1.75),
    2: (1.95, 1.18),
    3: (0.20, 0.15),
    4: (1.12, 0.68),
    5: (2.08, 0.18),
    6: (2.88, 1.08),
}


def stage_heading(
    ax: plt.Axes,
    x: float,
    title: str,
    subtitle: str,
    *,
    width: float,
) -> None:
    ax.text(x, 3.82, title, ha="left", va="top", fontsize=8.7, weight=500, color=INK)
    ax.text(x, 3.49, subtitle, ha="left", va="top", fontsize=6.2, color=MUTED)
    ax.plot([x, x + width], [3.25, 3.25], color=GUIDE, linewidth=0.8, solid_capstyle="round")


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    linewidth: float = 1.15,
    mutation: float = 11,
    rad: float = 0.0,
    style: str = "-|>",
    zorder: int = 8,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=mutation,
            linewidth=linewidth,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1,
            shrinkB=2,
            zorder=zorder,
        )
    )


def graph_positions(origin: tuple[float, float], scale: float) -> dict[int, tuple[float, float]]:
    ox, oy = origin
    return {node: (ox + scale * px, oy + scale * py) for node, (px, py) in POS.items()}


def draw_input_graph(
    ax: plt.Axes,
    origin: tuple[float, float],
    *,
    scale: float,
) -> dict[int, tuple[float, float]]:
    positions = graph_positions(origin, scale)
    radius = 0.13
    for a, b in EDGES:
        x0, y0 = positions[a]
        x1, y1 = positions[b]
        ax.plot([x0, x1], [y0, y1], color=EDGE, linewidth=1.05, solid_capstyle="round", zorder=1)
    for node in NODES:
        ax.add_patch(
            Circle(
                positions[node],
                radius,
                facecolor=NODE_COLORS[node],
                edgecolor=INK,
                linewidth=0.9,
                zorder=3,
            )
        )
    return positions


def draw_donor_intervention(ax: plt.Axes) -> None:
    donor = (3.03, 2.54)
    target = (4.18, 2.54)
    ax.add_patch(Circle(donor, 0.19, facecolor=SEM, edgecolor=INK, linewidth=0.9, zorder=5))
    ax.text(*donor, r"$\tilde{x}$", ha="center", va="center", fontsize=9.0, color=INK, zorder=6)
    ax.add_patch(Circle(target, 0.19, facecolor=NODE_COLORS[6], edgecolor=INK, linewidth=0.9, zorder=5))
    ax.text(*target, r"$j$", ha="center", va="center", fontsize=8.8, fontstyle="italic", color=INK, zorder=6)
    arrow(ax, (3.27, 2.54), (3.94, 2.54), color=SEM_DARK, linewidth=1.45, mutation=12)
    ax.text(3.61, 2.17, "donor replacement", ha="center", va="top", fontsize=6.8, color=SEM_DARK)


def draw_transposition_intervention(ax: plt.Axes) -> None:
    left = (3.18, 1.18)
    right = (4.03, 1.18)
    ax.add_patch(Circle(left, 0.17, facecolor=NODE_COLORS[0], edgecolor=INK, linewidth=0.9, zorder=5))
    ax.add_patch(Circle(right, 0.17, facecolor=NODE_COLORS[5], edgecolor=INK, linewidth=0.9, zorder=5))
    ax.text(*left, r"$u$", ha="center", va="center", fontsize=8.2, fontstyle="italic", zorder=6)
    ax.text(*right, r"$v$", ha="center", va="center", fontsize=8.2, fontstyle="italic", zorder=6)
    arrow(
        ax,
        (left[0] + 0.19, left[1] + 0.11),
        (right[0] - 0.19, right[1] + 0.11),
        color=STR_DARK,
        linewidth=1.2,
        mutation=9,
        rad=-0.25,
    )
    arrow(
        ax,
        (right[0] - 0.19, right[1] - 0.11),
        (left[0] + 0.19, left[1] - 0.11),
        color=STR_DARK,
        linewidth=1.2,
        mutation=9,
        rad=-0.25,
    )
    ax.text(3.61, 0.78, "node transposition", ha="center", va="top", fontsize=6.8, color=STR_DARK)


def draw_transported_state(ax: plt.Axes) -> None:
    origin = (5.54, 1.20)
    scale = 0.56
    positions = graph_positions(origin, scale)
    response = (0.30, 0.47, 0.78, 0.25, 1.00, 0.66, 0.42)
    radius = 0.105

    for a, b in EDGES:
        x0, y0 = positions[a]
        x1, y1 = positions[b]
        ax.plot([x0, x1], [y0, y1], color="#9aa7b1", linewidth=0.8, solid_capstyle="round", zorder=1)

    for node in NODES:
        x, y = positions[node]
        halo_radius = radius * (1.30 + 0.48 * response[node])
        ax.add_patch(
            Circle(
                (x, y),
                halo_radius,
                facecolor=REPR,
                edgecolor="none",
                alpha=0.10 + 0.24 * response[node],
                zorder=2,
            )
        )
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor="#f6f7f9",
                edgecolor=REPR,
                linewidth=0.75 + 1.0 * response[node],
                zorder=3,
            )
        )

    carrier = positions[4]
    ax.text(carrier[0] + 0.23, carrier[1] - 0.02, r"$\Delta h_i$", ha="left", va="center", fontsize=8.8, color=REPR)
    arrow(
        ax,
        (carrier[0] + 0.16, carrier[1] + 0.02),
        (carrier[0] + 0.06, carrier[1] + 0.01),
        color=REPR,
        linewidth=1.0,
        mutation=8,
    )

    # A short latent-state vector makes the representation interpretation explicit.
    vx, vy = 6.45, 0.72
    heights = (0.19, 0.35, 0.13, 0.28, 0.22)
    for idx, height in enumerate(heights):
        ax.add_patch(
            Rectangle(
                (vx + 0.12 * idx, vy),
                0.075,
                height,
                facecolor=REPR,
                edgecolor="none",
                alpha=0.42 + 0.10 * idx,
                zorder=4,
            )
        )
    ax.text(vx + 0.25, 0.55, "carrier-state change", ha="center", va="top", fontsize=7.2, color=MUTED)


def draw_head_response_grid(ax: plt.Axes) -> None:
    # Fixed illustrative response strengths: rows are layers, columns are heads.
    sem = (
        (0.18, 0.63, 0.30, 0.22, 0.42),
        (0.25, 0.80, 0.36, 0.20, 0.54),
        (0.38, 0.58, 0.92, 0.28, 0.48),
        (0.20, 0.42, 0.68, 0.74, 0.34),
    )
    structural = (
        (0.50, 0.22, 0.35, 0.70, 0.32),
        (0.64, 0.31, 0.44, 0.82, 0.30),
        (0.54, 0.40, 0.28, 0.66, 0.52),
        (0.76, 0.37, 0.48, 0.24, 0.62),
    )
    x0, y0 = 8.56, 2.70
    cell, gap = 0.25, 0.10
    row_gap = 0.16

    ax.add_patch(Rectangle((8.38, 3.00), 0.11, 0.11, facecolor=SEM, edgecolor="none"))
    ax.text(8.55, 3.055, r"$S_{\mathrm{sem}}$", ha="left", va="center", fontsize=7.1, color=MUTED)
    ax.add_patch(Rectangle((9.31, 3.00), 0.11, 0.11, facecolor=STR, edgecolor="none"))
    ax.text(9.48, 3.055, r"$S_{\mathrm{str}}$", ha="left", va="center", fontsize=7.1, color=MUTED)

    for layer in range(4):
        y = y0 - layer * (cell + row_gap)
        ax.text(x0 - 0.20, y + cell / 2, rf"$\ell_{layer + 1}$", ha="right", va="center", fontsize=6.9, color=MUTED)
        for head in range(5):
            x = x0 + head * (cell + gap)
            ax.add_patch(
                Rectangle(
                    (x, y),
                    cell / 2,
                    cell,
                    facecolor=SEM,
                    edgecolor="none",
                    alpha=0.10 + 0.80 * sem[layer][head],
                    zorder=2,
                )
            )
            ax.add_patch(
                Rectangle(
                    (x + cell / 2, y),
                    cell / 2,
                    cell,
                    facecolor=STR,
                    edgecolor="none",
                    alpha=0.10 + 0.80 * structural[layer][head],
                    zorder=2,
                )
            )
            ax.add_patch(
                Rectangle(
                    (x, y),
                    cell,
                    cell,
                    facecolor="none",
                    edgecolor="#5f6d78",
                    linewidth=0.55,
                    zorder=3,
                )
            )
    ax.text(9.31, 0.74, "heads", ha="center", va="top", fontsize=7.2, color=MUTED)


def draw_role_plane(ax: plt.Axes) -> None:
    x0, y0 = 11.12, 0.78
    w, h = 2.15, 2.22
    arrow(ax, (x0, y0), (x0 + w, y0), color="#788692", linewidth=0.9, mutation=8)
    arrow(ax, (x0, y0), (x0, y0 + h), color="#788692", linewidth=0.9, mutation=8)
    ax.plot(
        [x0 + 0.16, x0 + w - 0.10],
        [y0 + 0.16, y0 + h - 0.10],
        color=GUIDE,
        linewidth=0.8,
        linestyle=(0, (2, 2)),
        zorder=1,
    )

    points = (
        (0.35, 1.74, SEM, "o"),
        (0.55, 1.38, SEM, "o"),
        (0.78, 1.82, SEM, "o"),
        (1.70, 0.35, STR, "s"),
        (1.44, 0.56, STR, "s"),
        (1.84, 0.78, STR, "s"),
        (1.32, 1.55, JOINT, "D"),
        (1.02, 1.10, JOINT, "D"),
        (0.55, 0.48, "#8f9aa3", "o"),
    )
    for dx, dy, color, marker in points:
        ax.scatter(
            [x0 + dx],
            [y0 + dy],
            s=27,
            marker=marker,
            facecolor=color,
            edgecolor=INK,
            linewidth=0.55,
            zorder=4,
        )

    ax.text(x0 + 0.31, y0 + h - 0.12, "semantic", ha="left", va="top", fontsize=7.5, color=SEM_DARK)
    ax.text(x0 + w - 0.08, y0 + 0.26, "structural", ha="right", va="bottom", fontsize=7.5, color=STR_DARK)
    ax.text(x0 + 1.32, y0 + 1.67, "joint", ha="left", va="bottom", fontsize=7.1, color=JOINT)
    ax.text(x0 + w / 2, y0 - 0.24, "structural response", ha="center", va="top", fontsize=6.8, color=MUTED)
    ax.text(
        x0 - 0.28,
        y0 + h / 2,
        "semantic response",
        ha="center",
        va="center",
        fontsize=6.8,
        color=MUTED,
        rotation=90,
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
    # Sized close to a full-width conference/dissertation figure so typography
    # remains legible without large downstream scaling.
    fig, ax = plt.subplots(figsize=(8.1, 2.45))
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 4.08)
    ax.axis("off")

    stage_heading(ax, 0.30, "Graph input", r"content $X$; structure $S$", width=1.72)
    stage_heading(ax, 2.72, "Interventions", "replace / transpose", width=1.98)
    stage_heading(ax, 5.35, "Transport", r"final-state change $\Delta h_i$", width=2.05)
    stage_heading(ax, 8.18, "Head responses", "response of each head", width=1.94)
    stage_heading(ax, 10.88, "Head roles", "semantic / structural role", width=2.72)

    draw_input_graph(ax, (0.48, 1.10), scale=0.55)
    ax.text(1.28, 0.63, "attributes + relations", ha="center", va="top", fontsize=6.6, color=MUTED)

    draw_donor_intervention(ax)
    draw_transposition_intervention(ax)
    draw_transported_state(ax)
    draw_head_response_grid(ax)
    draw_role_plane(ax)

    # Quiet stage-to-stage connectors.
    connectors = (
        ((2.12, 1.95), (2.60, 1.95)),
        ((4.82, 1.95), (5.23, 1.95)),
        ((7.54, 1.95), (8.06, 1.95)),
        ((10.24, 1.95), (10.74, 1.95)),
    )
    for start, end in connectors:
        arrow(ax, start, end, color="#7f8c97", linewidth=1.05, mutation=10)

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("visualisations/generated"),
        help="Directory where the PDF and PNG will be written.",
    )
    parser.add_argument("--stem", default="dissertation_overview", help="Output filename stem.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG export DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    pdf_path = args.out_dir / f"{args.stem}.pdf"
    png_path = args.out_dir / f"{args.stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.06, dpi=args.dpi)
    plt.close(fig)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
