from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


OUTPUT_DIR = Path(__file__).resolve().parents[3] / "output" / "pdf"
QUERY_COLOR = "#6A3D9A"
SOURCE_COLOR = "#D95F02"
TEXT_COLOR = "#17212B"
MUTED_TEXT = "#52606D"
PALETTES = {
    "option_1_reference_exact": (
        "#E0F3E5",
        "#7DCBAE",
        "#5195A6",
        "#415C97",
        "#362B51",
        "#0B0606",
    ),
    "option_2_reference_dark_background": (
        "#0B0606",
        "#362B51",
        "#415C97",
        "#5195A6",
        "#7DCBAE",
        "#E0F3E5",
    ),
    "option_3_navy_cyan": (
        "#030712",
        "#101D42",
        "#164E63",
        "#0891B2",
        "#67E8F9",
        "#ECFEFF",
    ),
}


def synthetic_examples() -> list[dict[str, object]]:
    rng = np.random.default_rng(20260729)
    specifications = (
        ("Semantic specialist", 2, 1, 1.47, +0.97, 7, 10),
        ("Structural specialist", 2, 2, 2.12, -0.88, 11, 2),
        ("High-J generalist", 3, 4, 2.71, +0.15, 8, 3),
        ("First-layer head", 1, 7, 0.55, +0.38, 8, 13),
    )
    examples: list[dict[str, object]] = []
    for index, (role, layer, head, joint, selectivity, query, source) in enumerate(
        specifications
    ):
        matrix = rng.gamma(0.65, 1.0, size=(16, 16))
        matrix += 0.025
        matrix /= matrix.sum(axis=0, keepdims=True)
        if index == 0:
            matrix[:, query] *= 0.05
            matrix[source, query] = 0.95
            matrix[:, query] /= matrix[:, query].sum()
        elif index == 1:
            matrix[source, :] *= 6.5
            matrix /= matrix.sum(axis=0, keepdims=True)
        elif index == 2:
            matrix[:, query] *= 0.28
            matrix[source, query] = 0.72
            matrix[:, query] /= matrix[:, query].sum()
        else:
            matrix = 0.72 * matrix + 0.28 * np.eye(16)
            matrix /= matrix.sum(axis=0, keepdims=True)
        examples.append(
            {
                "role": role,
                "layer_display": layer,
                "head_display": head,
                "joint_score_J": joint,
                "selectivity_D_rel": selectivity,
                "query_node": query,
                "source_node": source,
                "attention_matrix": matrix,
                "query_attention": matrix[:, query],
            }
        )
    return examples


def build_figure(
    examples: list[dict[str, object]],
    *,
    palette_name: str,
    palette_colors: tuple[str, ...],
) -> mpl.figure.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
        }
    )
    matrices = [
        np.asarray(example["attention_matrix"], dtype=float) for example in examples
    ]
    global_max = max(float(matrix.max()) for matrix in matrices)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=global_max)
    attention_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        palette_name,
        palette_colors,
    )

    fig = plt.figure(figsize=(7.4, 5.45), facecolor="white")
    outer = fig.add_gridspec(
        2,
        2,
        left=0.045,
        right=0.985,
        bottom=0.155,
        top=0.975,
        hspace=0.09,
        wspace=0.15,
    )
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 16, endpoint=False)
    positions = np.column_stack((np.cos(angles), np.sin(angles)))
    ticks = np.array((0, 3, 6, 9, 12, 15))

    for panel, (example, matrix) in enumerate(zip(examples, matrices)):
        tile = outer[panel // 2, panel % 2].subgridspec(
            2,
            2,
            height_ratios=(0.10, 1.0),
            width_ratios=(1.0, 1.05),
            hspace=0.03,
            wspace=0.21,
        )
        title_ax = fig.add_subplot(tile[0, :])
        graph_ax = fig.add_subplot(tile[1, 0])
        matrix_ax = fig.add_subplot(tile[1, 1])

        title_ax.set_axis_off()
        title_ax.text(
            0.0,
            0.53,
            rf"$\bf{{{chr(97 + panel)}}}$  {example['role']}",
            fontsize=9.5,
            ha="left",
            va="center",
        )
        title_ax.text(
            1.0,
            0.53,
            (
                f"L{int(example['layer_display'])} H{int(example['head_display'])}"
                f"   J = {float(example['joint_score_J']):.2f}"
                rf"   $D_{{\rm rel}}$ = {float(example['selectivity_D_rel']):+.2f}"
            ),
            fontsize=7.2,
            color=MUTED_TEXT,
            ha="right",
            va="center",
        )

        for node in range(16):
            neighbour = (node + 1) % 16
            graph_ax.plot(
                positions[[node, neighbour], 0],
                positions[[node, neighbour], 1],
                color="#AAB4BE",
                linewidth=0.9,
                zorder=1,
            )

        query = int(example["query_node"])
        source = int(example["source_node"])
        query_attention = np.asarray(example["query_attention"], dtype=float)
        graph_ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=74.0,
            c=query_attention,
            cmap=attention_cmap,
            norm=norm,
            edgecolors="#65727E",
            linewidths=0.55,
            zorder=3,
        )
        for selected_node, color in (
            (query, QUERY_COLOR),
            (source, SOURCE_COLOR),
        ):
            graph_ax.scatter(
                [positions[selected_node, 0]],
                [positions[selected_node, 1]],
                s=145.0,
                facecolors="none",
                edgecolors="white",
                linewidths=2.2,
                zorder=5,
            )
            graph_ax.scatter(
                [positions[selected_node, 0]],
                [positions[selected_node, 1]],
                s=145.0,
                facecolors="none",
                edgecolors=color,
                linewidths=1.25,
                zorder=6,
            )
        for node, (x, y) in enumerate(positions):
            rgba = attention_cmap(norm(query_attention[node]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            graph_ax.text(
                x,
                y,
                str(node + 1),
                ha="center",
                va="center",
                fontsize=5.5,
                color=TEXT_COLOR if luminance > 0.56 else "white",
                zorder=7,
            )
        graph_ax.set(
            xlim=(-1.24, 1.24),
            ylim=(-1.22, 1.22),
            aspect="equal",
        )
        graph_ax.set_axis_off()
        graph_ax.set_title(
            "Attention to query",
            fontsize=7.7,
            color=MUTED_TEXT,
            pad=2,
        )

        matrix_ax.imshow(
            matrix,
            cmap=attention_cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        for coordinate, extent, color in (
            ((query - 0.5, -0.5), (1.0, 16), QUERY_COLOR),
            ((-0.5, source - 0.5), (16, 1.0), SOURCE_COLOR),
        ):
            matrix_ax.add_patch(
                Rectangle(
                    coordinate,
                    *extent,
                    fill=False,
                    edgecolor="white",
                    linewidth=1.5,
                )
            )
            matrix_ax.add_patch(
                Rectangle(
                    coordinate,
                    *extent,
                    fill=False,
                    edgecolor=color,
                    linewidth=0.75,
                )
            )
        matrix_ax.set_xticks(ticks, [str(value + 1) for value in ticks])
        matrix_ax.set_yticks(ticks, [str(value + 1) for value in ticks])
        matrix_ax.tick_params(labelsize=6.6, length=2.2, pad=1.5)
        matrix_ax.set_xlabel("Destination node", fontsize=7.2, labelpad=2)
        matrix_ax.set_ylabel("Source node", fontsize=7.2, labelpad=2)
        matrix_ax.set_title(
            "Raw attention matrix",
            fontsize=7.7,
            color=MUTED_TEXT,
            pad=2,
        )

    query_handle = Line2D(
        [],
        [],
        marker="o",
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor=QUERY_COLOR,
        markeredgewidth=2.0,
        markersize=7,
        label="Query node / destination column",
    )
    source_handle = Line2D(
        [],
        [],
        marker="o",
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor=SOURCE_COLOR,
        markeredgewidth=2.0,
        markersize=7,
        label="Source node / source row",
    )
    fig.legend(
        handles=(query_handle, source_handle),
        loc="lower left",
        bbox_to_anchor=(0.045, 0.026),
        frameon=False,
        ncol=2,
        fontsize=7.2,
        handletextpad=0.45,
        columnspacing=1.05,
        borderaxespad=0,
    )
    colorbar_ax = fig.add_axes((0.69, 0.052, 0.27, 0.017))
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=attention_cmap),
        cax=colorbar_ax,
        orientation="horizontal",
    )
    colorbar.ax.tick_params(labelsize=6.5, length=2.0, pad=1.5)
    colorbar.ax.set_title(
        "Attention weight (linear colour scale)",
        fontsize=7.2,
        color=MUTED_TEXT,
        pad=3,
    )
    return fig


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    example_data = synthetic_examples()
    for name, colors in PALETTES.items():
        figure = build_figure(
            example_data,
            palette_name=name,
            palette_colors=colors,
        )
        png = OUTPUT_DIR / f"attention_palette_{name}.png"
        pdf = OUTPUT_DIR / f"attention_palette_{name}.pdf"
        figure.savefig(png, dpi=300, facecolor="white", bbox_inches=None)
        figure.savefig(
            pdf,
            facecolor="white",
            bbox_inches=None,
            metadata={
                "Creator": "Graph Specialisation and Metrics",
                "Title": f"Synthetic attention palette comparison - {name}",
            },
        )
        plt.close(figure)
        print(png)
        print(pdf)
