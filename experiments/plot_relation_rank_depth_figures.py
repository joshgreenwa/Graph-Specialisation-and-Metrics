#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Make depth-focused figures from a saved relation-rank sweep.

This script does not train models. It loads a completed
graphormer_relation_rank_sweep.py run, usually:

    /content/drive/MyDrive/graph_operator_distillation/relation_rank_depth

and writes a compact paper figure set:

* depth_paper_figures/depth_core.pdf
* depth_appendix_figures/depth_gain_heatmaps.pdf
* depth_appendix_figures/all_depth_curves.pdf
* depth_appendix_figures/purity_rank_vs_depth.pdf
* depth_findings_summary.md

Colab:

    !python plot_relation_rank_depth_figures.py \
      --root /content/drive/MyDrive/graph_operator_distillation/relation_rank_depth
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


STUDENT_ORDER = (
    "graphormer_struct_support",
    "graphormer_support",
    "csa_bias_support",
    "rel_value_support",
    "edge_gnn",
)

STUDENT_LABELS = {
    "graphormer_struct_support": "Graphormer\nstructural routing",
    "graphormer_support": "Graphormer\nrouting + QK",
    "csa_bias_support": "CSA bias-only\nrouting",
    "rel_value_support": "Pair-value\ntransport",
    "edge_gnn": "Edge-GNN\ntransport",
}

STUDENT_COLORS = {
    "graphormer_struct_support": "#4063A3",
    "graphormer_support": "#5F83C2",
    "csa_bias_support": "#B56B45",
    "rel_value_support": "#2F8A5B",
    "edge_gnn": "#277C8E",
}

STUDENT_MARKERS = {
    "graphormer_struct_support": "o",
    "graphormer_support": "s",
    "csa_bias_support": "^",
    "rel_value_support": "D",
    "edge_gnn": "P",
}


def import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def is_number(value: object) -> bool:
    return math.isfinite(fnum(value))


def mean_sem(values: Sequence[float]) -> tuple[float, float]:
    vals = np.array([v for v in values if math.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals.mean()), 0.0
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(vals.size))


def unique_ints(rows: Sequence[Mapping[str, object]], key: str) -> list[int]:
    return sorted({int(fnum(row.get(key))) for row in rows if is_number(row.get(key))})


def available_students(rows: Sequence[Mapping[str, object]]) -> list[str]:
    return [s for s in STUDENT_ORDER if any(row.get("student") == s for row in rows)]


def group_metric(
    rows: Sequence[Mapping[str, object]],
    student: str,
    x_key: str,
    metric: str,
    filters: Mapping[str, object] | None = None,
) -> tuple[list[float], list[float], list[float]]:
    grouped: dict[float, list[float]] = {}
    for row in rows:
        if row.get("student") != student or not is_number(row.get(x_key)) or not is_number(row.get(metric)):
            continue
        if filters is not None and any(str(row.get(k)) != str(v) for k, v in filters.items()):
            continue
        grouped.setdefault(float(row[x_key]), []).append(float(row[metric]))
    xs = sorted(grouped)
    means: list[float] = []
    sems: list[float] = []
    for x in xs:
        mean, sem = mean_sem(grouped[x])
        means.append(mean)
        sems.append(sem)
    return xs, means, sems


def metric_mean(
    rows: Sequence[Mapping[str, object]],
    student: str,
    relation: int,
    heads: int,
    layer: int,
    metric: str = "relative_mse",
) -> float:
    vals = [
        fnum(row.get(metric))
        for row in rows
        if row.get("student") == student
        and int(fnum(row.get("relation_types"))) == relation
        and int(fnum(row.get("heads"))) == heads
        and int(fnum(row.get("layers"))) == layer
        and is_number(row.get(metric))
    ]
    return mean_sem(vals)[0]


def save_figure(fig, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")


def add_legend_if_any(ax, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(**kwargs)


def hardest_setting(rows: Sequence[Mapping[str, object]]) -> tuple[int, int]:
    relations = unique_ints(rows, "relation_types")
    heads = unique_ints(rows, "heads")
    if not relations or not heads:
        raise ValueError("summary.csv does not contain relation_types/heads")
    return max(relations), min(heads)


def median_head_setting(rows: Sequence[Mapping[str, object]]) -> tuple[int, int]:
    relations = unique_ints(rows, "relation_types")
    heads = unique_ints(rows, "heads")
    if not relations or not heads:
        raise ValueError("summary.csv does not contain relation_types/heads")
    return max(relations), int(min(heads, key=lambda h: abs(h - np.median(heads))))


def plot_depth_core(rows: Sequence[Mapping[str, object]], root: Path, relation: int | None, heads: int | None) -> None:
    plt = import_plotting()
    paper_dir = root / "depth_paper_figures"
    relations = unique_ints(rows, "relation_types")
    head_values = unique_ints(rows, "heads")
    layers = unique_ints(rows, "layers")
    if len(layers) < 2:
        raise ValueError("depth figures need at least two layer values")

    hard_r, hard_h = hardest_setting(rows)
    main_r = relation if relation is not None else hard_r
    main_h = heads if heads is not None else hard_h
    max_layer = max(layers)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3), gridspec_kw={"width_ratios": [1.25, 1.0, 1.1]})

    ax = axes[0]
    filters = {"relation_types": main_r, "heads": main_h}
    for student in available_students(rows):
        xs, ys, sems = group_metric(rows, student, "layers", "relative_mse", filters)
        if not xs:
            continue
        ax.errorbar(
            xs,
            ys,
            yerr=sems,
            marker=STUDENT_MARKERS.get(student, "o"),
            color=STUDENT_COLORS.get(student),
            linewidth=1.7,
            markersize=4.5,
            capsize=2,
            label=STUDENT_LABELS.get(student, student).replace("\n", " "),
        )
    ax.set_xlabel("layers")
    ax.set_ylabel("relative MSE")
    ax.set_yscale("log")
    ax.set_title(f"A. Depth rescue at high pressure\nR={main_r}, H={main_h}, R/H={main_r/main_h:.1f}")
    ax.grid(alpha=0.22)
    add_legend_if_any(ax, frameon=False)

    ax = axes[1]
    gains = []
    labels = []
    colors = []
    for student in available_students(rows):
        y1 = metric_mean(rows, student, main_r, main_h, min(layers), "relative_mse")
        yk = metric_mean(rows, student, main_r, main_h, max_layer, "relative_mse")
        if math.isfinite(y1) and math.isfinite(yk) and yk > 0:
            gains.append(y1 / yk)
            labels.append(STUDENT_LABELS.get(student, student).replace("\n", " "))
            colors.append(STUDENT_COLORS.get(student, "#777777"))
    y_pos = np.arange(len(gains))
    ax.barh(y_pos, gains, color=colors)
    ax.axvline(1.0, color="#303030", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_pos, labels)
    ax.set_xscale("log")
    ax.set_xlabel(f"error improvement L={min(layers)} / L={max_layer}")
    ax.set_title("B. Depth gain")
    ax.grid(axis="x", alpha=0.22)

    ax = axes[2]
    graph_x, graph_y, _ = group_metric(
        rows, "graphormer_struct_support", "basis_pressure_R_over_H", "relative_mse", {"layers": max_layer}
    )
    rel_x, rel_y, _ = group_metric(rows, "rel_value_support", "basis_pressure_R_over_H", "relative_mse", {"layers": max_layer})
    edge_x, edge_y, _ = group_metric(rows, "edge_gnn", "basis_pressure_R_over_H", "relative_mse", {"layers": max_layer})
    xs_common = sorted(set(graph_x) & set(rel_x))
    if xs_common:
        graph = dict(zip(graph_x, graph_y))
        rel = dict(zip(rel_x, rel_y))
        ax.plot(xs_common, [graph[x] / max(rel[x], 1.0e-12) for x in xs_common], marker="o", color="#4063A3", label="vs pair-value")
    xs_common = sorted(set(graph_x) & set(edge_x))
    if xs_common:
        graph = dict(zip(graph_x, graph_y))
        edge = dict(zip(edge_x, edge_y))
        ax.plot(xs_common, [graph[x] / max(edge[x], 1.0e-12) for x in xs_common], marker="P", color="#277C8E", label="vs Edge-GNN")
    ax.axhline(1.0, color="#303030", linestyle="--", linewidth=1.0)
    ax.set_xlabel("relation pressure $R/H$")
    ax.set_ylabel(f"Graphormer error ratio at L={max_layer}")
    ax.set_yscale("log")
    ax.set_title("C. Residual gap after depth")
    ax.grid(alpha=0.22)
    add_legend_if_any(ax, frameon=False)

    fig.suptitle("Depth as an indirect rescue for routing-only Graphormer", y=1.03, fontsize=11)
    save_figure(fig, paper_dir / "depth_core")
    plt.close(fig)


def pivot_depth_gain(
    rows: Sequence[Mapping[str, object]],
    student: str,
    first_layer: int,
    last_layer: int,
) -> tuple[list[int], list[int], np.ndarray]:
    relations = unique_ints(rows, "relation_types")
    heads = unique_ints(rows, "heads")
    mat = np.full((len(relations), len(heads)), np.nan)
    for i, relation in enumerate(relations):
        for j, head in enumerate(heads):
            y1 = metric_mean(rows, student, relation, head, first_layer, "relative_mse")
            yk = metric_mean(rows, student, relation, head, last_layer, "relative_mse")
            if math.isfinite(y1) and math.isfinite(yk) and yk > 0:
                mat[i, j] = y1 / yk
    return relations, heads, mat


def plot_depth_gain_heatmaps(rows: Sequence[Mapping[str, object]], root: Path) -> None:
    plt = import_plotting()
    appendix_dir = root / "depth_appendix_figures"
    layers = unique_ints(rows, "layers")
    if len(layers) < 2:
        return
    first_layer = min(layers)
    last_layer = max(layers)
    students = [s for s in ["graphormer_struct_support", "graphormer_support", "csa_bias_support"] if any(r.get("student") == s for r in rows)]
    if not students:
        return
    fig, axes = plt.subplots(1, len(students), figsize=(3.6 * len(students), 3.05), squeeze=False)
    finite = []
    mats = {}
    for student in students:
        rels, heads, mat = pivot_depth_gain(rows, student, first_layer, last_layer)
        mats[student] = (rels, heads, mat)
        finite.extend(np.log10(mat[np.isfinite(mat)]).tolist())
    lim = max(abs(np.percentile(finite, 5)), abs(np.percentile(finite, 95))) if finite else 1.0
    for ax, student in zip(axes[0], students):
        rels, heads, mat = mats[student]
        im = ax.imshow(np.log10(mat), aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
        ax.set_title(STUDENT_LABELS.get(student, student).replace("\n", " "))
        ax.set_xticks(range(len(heads)), [str(h) for h in heads])
        ax.set_yticks(range(len(rels)), [str(r) for r in rels])
        ax.set_xlabel("heads H")
        ax.set_ylabel("relations R")
        for i in range(len(rels)):
            for j in range(len(heads)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.1f}x", ha="center", va="center", fontsize=7, color="#111111")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label(f"log10 error improvement L={first_layer}/L={last_layer}")
    fig.suptitle("Appendix: where depth improves routing-only models", y=1.02)
    save_figure(fig, appendix_dir / "depth_gain_heatmaps")
    plt.close(fig)


def plot_all_depth_curves(rows: Sequence[Mapping[str, object]], root: Path) -> None:
    plt = import_plotting()
    appendix_dir = root / "depth_appendix_figures"
    relations = unique_ints(rows, "relation_types")
    heads = unique_ints(rows, "heads")
    if not relations or not heads:
        return
    cells = [(r, h) for r in relations for h in heads]
    cols = min(3, len(cells))
    rows_n = int(math.ceil(len(cells) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.4 * cols, 2.75 * rows_n), squeeze=False)
    for idx, (relation, head) in enumerate(cells):
        ax = axes[idx // cols][idx % cols]
        filters = {"relation_types": relation, "heads": head}
        for student in available_students(rows):
            xs, ys, _ = group_metric(rows, student, "layers", "relative_mse", filters)
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                marker=STUDENT_MARKERS.get(student, "o"),
                color=STUDENT_COLORS.get(student),
                linewidth=1.3,
                markersize=3.2,
                label=STUDENT_LABELS.get(student, student).replace("\n", " "),
            )
        ax.set_title(f"R={relation}, H={head}, R/H={relation/head:.1f}")
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.set_xlabel("layers")
        ax.set_ylabel("rel. MSE")
    for idx in range(len(cells), rows_n * cols):
        axes[idx // cols][idx % cols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(3, len(handles)), frameon=False)
        fig.subplots_adjust(bottom=0.12)
    fig.suptitle("Appendix: depth curves for all relation/head settings", y=0.995)
    save_figure(fig, appendix_dir / "all_depth_curves")
    plt.close(fig)


def plot_purity_rank_vs_depth(rows: Sequence[Mapping[str, object]], root: Path, relation: int | None, heads: int | None) -> None:
    plt = import_plotting()
    appendix_dir = root / "depth_appendix_figures"
    hard_r, hard_h = hardest_setting(rows)
    main_r = relation if relation is not None else hard_r
    main_h = heads if heads is not None else hard_h
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    filters = {"relation_types": main_r, "heads": main_h}
    for metric, ax, title in [
        ("head_relation_purity", axes[0], "Head purity"),
        ("relation_mass_rank", axes[1], "Relation mass rank"),
    ]:
        for student in ["graphormer_struct_support", "graphormer_support", "csa_bias_support", "rel_value_support"]:
            if not any(r.get("student") == student for r in rows):
                continue
            xs, ys, sems = group_metric(rows, student, "layers", metric, filters)
            if not xs:
                continue
            ax.errorbar(
                xs,
                ys,
                yerr=sems,
                marker=STUDENT_MARKERS.get(student, "o"),
                color=STUDENT_COLORS.get(student),
                linewidth=1.5,
                markersize=4,
                capsize=2,
                label=STUDENT_LABELS.get(student, student).replace("\n", " "),
            )
        ax.set_xlabel("layers")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(title)
        ax.grid(alpha=0.22)
        add_legend_if_any(ax, frameon=False)
    fig.suptitle(f"Appendix: routing diagnostics with depth (R={main_r}, H={main_h})", y=1.02)
    save_figure(fig, appendix_dir / "purity_rank_vs_depth")
    plt.close(fig)


def write_findings(rows: Sequence[Mapping[str, object]], root: Path, relation: int | None, heads: int | None) -> None:
    layers = unique_ints(rows, "layers")
    hard_r, hard_h = hardest_setting(rows)
    main_r = relation if relation is not None else hard_r
    main_h = heads if heads is not None else hard_h
    first_layer = min(layers)
    last_layer = max(layers)
    lines = [
        "# Depth Relation-Rank Findings",
        "",
        "Generated from `summary.csv`; use as a draft quantitative guide.",
        "",
        "## Figure Files",
        "",
        "- `depth_paper_figures/depth_core.pdf`: main depth evidence figure.",
        "- `depth_appendix_figures/depth_gain_heatmaps.pdf`: where depth helps across R,H.",
        "- `depth_appendix_figures/all_depth_curves.pdf`: all depth curves.",
        "- `depth_appendix_figures/purity_rank_vs_depth.pdf`: routing diagnostics with depth.",
        "",
        "## Key Checks",
        "",
        f"- Focus setting: `R={main_r}`, `H={main_h}`, relation pressure `R/H={main_r/main_h:.2f}`.",
    ]
    for student in available_students(rows):
        y1 = metric_mean(rows, student, main_r, main_h, first_layer, "relative_mse")
        yk = metric_mean(rows, student, main_r, main_h, last_layer, "relative_mse")
        if math.isfinite(y1) and math.isfinite(yk) and yk > 0:
            lines.append(
                f"- `{student}`: relative MSE `{y1:.4g}` at L={first_layer}, `{yk:.4g}` at L={last_layer}; improvement `{y1 / yk:.2f}x`."
            )
    graph = metric_mean(rows, "graphormer_struct_support", main_r, main_h, last_layer, "relative_mse")
    rel = metric_mean(rows, "rel_value_support", main_r, main_h, last_layer, "relative_mse")
    edge = metric_mean(rows, "edge_gnn", main_r, main_h, last_layer, "relative_mse")
    if math.isfinite(graph) and math.isfinite(rel) and rel > 0:
        lines.append(f"- At L={last_layer}, Graphormer structural routing / pair-value error ratio: `{graph / rel:.2f}x`.")
    if math.isfinite(graph) and math.isfinite(edge) and edge > 0:
        lines.append(f"- At L={last_layer}, Graphormer structural routing / Edge-GNN error ratio: `{graph / edge:.2f}x`.")
    lines.extend(
        [
            "",
            "## Interpretation Template",
            "",
            "- If Graphormer improves substantially with depth, report this as an indirect latent-state rescue path.",
            "- If explicit transport remains lower-error at equal or smaller depth, the one-layer transport bottleneck claim is still supported.",
            "- If depth improves performance without increasing head purity, that is evidence against a pure head-specialisation explanation and in favour of latent node-state mediation.",
            "",
        ]
    )
    (root / "depth_findings_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    summary = root / "summary.csv"
    if not summary.exists():
        raise FileNotFoundError(f"missing summary.csv at {summary}")
    rows = read_csv(summary)
    plot_depth_core(rows, root, args.relation, args.heads)
    plot_depth_gain_heatmaps(rows, root)
    plot_all_depth_curves(rows, root)
    plot_purity_rank_vs_depth(rows, root, args.relation, args.heads)
    write_findings(rows, root, args.relation, args.heads)
    print(f"[saved] {root / 'depth_paper_figures'}")
    print(f"[saved] {root / 'depth_appendix_figures'}")
    print(f"[saved] {root / 'depth_findings_summary.md'}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/content/drive/MyDrive/graph_operator_distillation/relation_rank_depth"),
        help="Directory containing summary.csv from graphormer_relation_rank_sweep.py.",
    )
    parser.add_argument("--relation", type=int, default=None, help="Optional relation count for the core depth panels.")
    parser.add_argument("--heads", type=int, default=None, help="Optional head count for the core depth panels.")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"[argparse] ignored notebook/kernel args: {' '.join(unknown)}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
