"""Draw a compact paper schematic of the two carriage interventions.

Both branches begin from the same clean graph:

* donor replacement inserts an external donor at a chosen target ``j``;
* node transposition exchanges two selected nodes ``u`` and ``v``.

The drawing is intentionally concerned with the intervention mechanics only.
Downstream transport deltas and carriage summaries can be appended to the two
output lanes in a later figure revision.
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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


INK = "#17212b"
MUTED = "#617181"
EDGE = "#34414d"
PANEL_EDGE = "#d3dde5"
PANEL_FILL = "#f8fafc"
DONOR = "#df5b52"
DONOR_DARK = "#ad3f38"
SWAP = "#357b6c"
SWAP_DARK = "#286254"

CONTENT_COLORS = {
    0: "#78aee0",
    1: "#f0a05a",
    2: "#73c59d",
    3: "#e68191",
    4: "#bc8ddd",
    5: "#f1d365",
    6: "#8fc4ca",
    "donor": DONOR,
}

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

# The three intervention sites are deliberately well separated in the graph.
TARGET_J = 6
ANCHOR_U = 0
PARTNER_V = 5


def rounded_panel(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    edge: str = PANEL_EDGE,
    face: str = PANEL_FILL,
    linewidth: float = 1.0,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.035,rounding_size=0.16",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            zorder=0,
        )
    )


def graph_positions(origin: tuple[float, float], scale: float) -> dict[int, tuple[float, float]]:
    ox, oy = origin
    return {node: (ox + scale * px, oy + scale * py) for node, (px, py) in POS.items()}


def draw_graph(
    ax: plt.Axes,
    origin: tuple[float, float],
    *,
    scale: float,
    overrides: dict[int, str] | None = None,
    labels: dict[int, str] | None = None,
    emphasis: dict[int, str] | None = None,
) -> dict[int, tuple[float, float]]:
    """Draw the shared graph grammar and return node centres."""
    positions = graph_positions(origin, scale)
    overrides = overrides or {}
    labels = labels or {}
    emphasis = emphasis or {}

    for a, b in EDGES:
        x0, y0 = positions[a]
        x1, y1 = positions[b]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=EDGE,
            linewidth=1.15,
            solid_capstyle="round",
            zorder=1,
        )

    radius = 0.145 * scale / 0.72
    for node in NODES:
        x, y = positions[node]
        if node in emphasis:
            ax.add_patch(
                Circle(
                    (x, y),
                    radius * 1.42,
                    facecolor="none",
                    edgecolor=emphasis[node],
                    linewidth=1.55,
                    zorder=3,
                )
            )
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor=overrides.get(node, CONTENT_COLORS[node]),
                edgecolor=INK,
                linewidth=1.0,
                zorder=4,
            )
        )
        if node in labels:
            ax.text(
                x,
                y - 0.003,
                labels[node],
                ha="center",
                va="center",
                fontsize=9.2,
                fontstyle="italic",
                color=INK,
                zorder=5,
            )
    return positions


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linewidth: float = 1.35,
    mutation: float = 13,
    rad: float = 0.0,
    style: str = "-|>",
    zorder: int = 6,
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
            shrinkB=3,
            zorder=zorder,
        )
    )


def lane_label(
    ax: plt.Axes,
    x: float,
    y: float,
    letter: str,
    title: str,
    subtitle: str,
    color: str,
) -> None:
    ax.text(x, y, letter, ha="left", va="top", fontsize=11.5, weight=600, color=INK)
    ax.text(x + 0.34, y, title, ha="left", va="top", fontsize=11.5, weight=500, color=INK)
    ax.text(x + 0.34, y - 0.33, subtitle, ha="left", va="top", fontsize=8.5, color=MUTED)
    ax.plot([x + 0.34, x + 0.90], [y - 0.57, y - 0.57], color=color, linewidth=2.2, solid_capstyle="round")


def draw_swap_glyph(ax: plt.Axes, centre: tuple[float, float]) -> None:
    """A compact, unmistakable two-way exchange mark."""
    cx, cy = centre
    left = (cx - 0.34, cy)
    right = (cx + 0.34, cy)
    ax.add_patch(Circle(left, 0.15, facecolor=CONTENT_COLORS[ANCHOR_U], edgecolor=INK, linewidth=0.9, zorder=7))
    ax.add_patch(Circle(right, 0.15, facecolor=CONTENT_COLORS[PARTNER_V], edgecolor=INK, linewidth=0.9, zorder=7))
    ax.text(left[0], left[1], r"$u$", ha="center", va="center", fontsize=8.2, fontstyle="italic", zorder=8)
    ax.text(right[0], right[1], r"$v$", ha="center", va="center", fontsize=8.2, fontstyle="italic", zorder=8)
    arrow(
        ax,
        (left[0] + 0.16, cy + 0.11),
        (right[0] - 0.16, cy + 0.11),
        color=SWAP_DARK,
        linewidth=1.25,
        mutation=10,
        rad=-0.26,
        zorder=8,
    )
    arrow(
        ax,
        (right[0] - 0.16, cy - 0.11),
        (left[0] + 0.16, cy - 0.11),
        color=SWAP_DARK,
        linewidth=1.25,
        mutation=10,
        rad=-0.26,
        zorder=8,
    )


def make_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.8, 5.5))
    ax.set_xlim(0, 11.7)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # Shared input graph.
    rounded_panel(ax, 0.42, 1.23, 3.12, 3.12)
    ax.text(0.68, 4.08, "Shared clean graph", ha="left", va="top", fontsize=10.5, weight=500, color=INK)
    draw_graph(
        ax,
        (0.77, 1.64),
        scale=0.76,
        labels={TARGET_J: r"$j$", ANCHOR_U: r"$u$", PARTNER_V: r"$v$"},
        emphasis={TARGET_J: DONOR, ANCHOR_U: SWAP, PARTNER_V: SWAP},
    )

    # Lane headings.
    lane_label(
        ax,
        4.05,
        5.26,
        "A",
        "Donor replacement",
        r"replace target $j$ with a node from another graph",
        DONOR,
    )
    lane_label(
        ax,
        4.05,
        2.54,
        "B",
        "Node transposition",
        r"exchange the selected nodes $u$ and $v$",
        SWAP,
    )

    # Output panels share the same geometry and alignment.
    panel_x, panel_w, panel_h = 8.03, 3.22, 2.05
    top_y, bottom_y = 3.10, 0.30
    rounded_panel(ax, panel_x, top_y, panel_w, panel_h, edge="#e4c5c0", face="#fffafa")
    rounded_panel(ax, panel_x, bottom_y, panel_w, panel_h, edge="#c7ddd6", face="#f8fcfa")
    ax.text(panel_x + 0.23, top_y + panel_h - 0.20, "After replacement", ha="left", va="top", fontsize=9.6, weight=500, color=INK)
    ax.text(panel_x + 0.23, bottom_y + panel_h - 0.20, "After transposition", ha="left", va="top", fontsize=9.6, weight=500, color=INK)

    out_scale = 0.63
    top_pos = draw_graph(
        ax,
        (panel_x + 0.39, top_y + 0.35),
        scale=out_scale,
        overrides={TARGET_J: CONTENT_COLORS["donor"]},
        labels={TARGET_J: r"$j$"},
        emphasis={TARGET_J: DONOR},
    )
    bottom_pos = draw_graph(
        ax,
        (panel_x + 0.39, bottom_y + 0.35),
        scale=out_scale,
        overrides={
            ANCHOR_U: CONTENT_COLORS[PARTNER_V],
            PARTNER_V: CONTENT_COLORS[ANCHOR_U],
        },
        labels={ANCHOR_U: r"$u$", PARTNER_V: r"$v$"},
        emphasis={ANCHOR_U: SWAP, PARTNER_V: SWAP},
    )

    # The common input splits orthogonally so process lines never cross labels.
    split_x = 3.88
    clean_y = 2.78
    top_lane_y = 3.64
    bottom_lane_y = 1.08
    ax.plot(
        [3.56, split_x, split_x],
        [clean_y, clean_y, top_lane_y],
        color="#8a98a5",
        linewidth=1.15,
        solid_capstyle="round",
        zorder=2,
    )
    ax.plot(
        [split_x, split_x],
        [clean_y, bottom_lane_y],
        color="#8a98a5",
        linewidth=1.15,
        solid_capstyle="round",
        zorder=2,
    )
    arrow(ax, (split_x, top_lane_y), (5.13, top_lane_y), color="#8a98a5", linewidth=1.15, mutation=12)
    arrow(ax, (split_x, bottom_lane_y), (5.02, bottom_lane_y), color="#8a98a5", linewidth=1.15, mutation=12)

    # Each operation is encoded directly on its branch.
    target_icon = (5.75, top_lane_y)
    ax.add_patch(
        Circle(
            target_icon,
            0.18,
            facecolor=CONTENT_COLORS[TARGET_J],
            edgecolor=INK,
            linewidth=1.0,
            zorder=7,
        )
    )
    ax.text(*target_icon, r"$j$", ha="center", va="center", fontsize=8.8, fontstyle="italic", color=INK, zorder=8)
    donor_icon = (5.56, 4.18)
    ax.add_patch(
        Circle(
            donor_icon,
            0.19,
            facecolor=CONTENT_COLORS["donor"],
            edgecolor=INK,
            linewidth=1.0,
            zorder=8,
        )
    )
    ax.text(*donor_icon, r"$\tilde{x}$", ha="center", va="center", fontsize=9.2, color=INK, zorder=9)
    ax.text(donor_icon[0], donor_icon[1] + 0.27, "external donor", ha="center", va="bottom", fontsize=7.9, color=MUTED)
    arrow(
        ax,
        (donor_icon[0] + 0.05, donor_icon[1] - 0.20),
        (target_icon[0] - 0.04, target_icon[1] + 0.20),
        color=DONOR_DARK,
        linewidth=1.55,
        mutation=12,
        rad=0.02,
        zorder=9,
    )
    arrow(
        ax,
        (target_icon[0] + 0.23, top_lane_y),
        (panel_x - 0.08, top_lane_y),
        color=DONOR_DARK,
        linewidth=1.45,
        mutation=13,
    )

    draw_swap_glyph(ax, centre=(5.75, bottom_lane_y))
    arrow(
        ax,
        (6.25, bottom_lane_y),
        (panel_x - 0.08, bottom_lane_y),
        color=SWAP_DARK,
        linewidth=1.45,
        mutation=13,
    )

    # Small endpoint cues reinforce that only the selected site(s) change.
    swap_mid_x = (bottom_pos[ANCHOR_U][0] + bottom_pos[PARTNER_V][0]) / 2
    ax.text(
        swap_mid_x,
        bottom_y + 0.17,
        r"$u \leftrightarrow v$",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=SWAP_DARK,
    )

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("visualisations/generated"),
        help="Directory where the PDF and PNG will be written.",
    )
    parser.add_argument("--stem", default="carriage_interventions", help="Output filename stem.")
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
