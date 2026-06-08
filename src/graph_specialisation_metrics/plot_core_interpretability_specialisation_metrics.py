#!/usr/bin/env python3
"""Paper-quality plots for core interpretability specialisation metrics.

Input is a metric directory produced by
``core_interpretability_specialisation_metrics.py``. The plotting code avoids
raw activation caches: it reads compact per-head/per-graph CSVs and only reloads
a model checkpoint when optional attention-example figures are requested.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

from graph_specialisation_metrics.core_interpretability_specialisation_metrics import (
    build_screen_config,
    load_runner,
    make_collector,
    resolve_device,
    select_graphs,
)


STYLE = {
    "figure.dpi": 160,
    "savefig.dpi": 320,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def read_metric_dir(metric_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metric_dir = metric_dir.expanduser().resolve()
    summary = pd.read_csv(metric_dir / "per_head_summary.csv")
    per_graph = pd.read_csv(metric_dir / "per_graph_head_scores.csv")
    meta_path = metric_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    for frame in (summary, per_graph):
        if "centered" in frame:
            frame["centered"] = frame["centered"].map(parse_boolish)
    return summary, per_graph, metadata


def parse_boolish(value: Any) -> Any:
    if pd.isna(value) or value == "":
        return np.nan
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return value


def ensure_out_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def metric_frame(
    summary: pd.DataFrame,
    *,
    metric: str,
    intervention: str,
    block: str,
    centered: Optional[bool],
) -> pd.DataFrame:
    frame = summary[
        (summary["metric"] == metric)
        & (summary["intervention"] == intervention)
        & (summary["block"] == block)
    ].copy()
    if centered is None:
        frame = frame[frame["centered"].isna()]
    else:
        frame = frame[frame["centered"] == centered]
    return frame


def heatmap_matrix(
    frame: pd.DataFrame,
    value: str = "mean",
) -> tuple[np.ndarray, list[int], list[int]]:
    if frame.empty:
        return np.zeros((0, 0)), [], []
    layers = sorted(int(v) for v in frame["layer"].dropna().unique())
    heads = sorted(int(v) for v in frame["head"].dropna().unique())
    mat = np.full((len(layers), len(heads)), np.nan)
    layer_idx = {layer: idx for idx, layer in enumerate(layers)}
    head_idx = {head: idx for idx, head in enumerate(heads)}
    for row in frame.itertuples(index=False):
        mat[layer_idx[int(row.layer)], head_idx[int(row.head)]] = float(getattr(row, value))
    return mat, layers, heads


def draw_heatmap(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    title: str,
    value: str = "mean",
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = 1.0,
    cmap: str = "viridis",
) -> Any:
    mat, layers, heads = heatmap_matrix(frame, value=value)
    if mat.size == 0:
        ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.set_axis_off()
        return None
    image = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(len(heads)), heads)
    ax.set_yticks(range(len(layers)), layers)
    return image


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_main_heatmaps(summary: pd.DataFrame, out_dir: Path, *, block: str, centered: bool) -> None:
    specs = [
        ("routing_invariant", "content", "Routing invariant under content"),
        ("routing_follow", "content", "Routing follows content"),
        ("routing_invariant", "structure", "Routing invariant under structure"),
        ("routing_follow", "structure", "Routing follows structure"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.7), constrained_layout=True)
    images = []
    for ax, (metric, intervention, title) in zip(axes.flat, specs):
        frame = metric_frame(
            summary,
            metric=metric,
            intervention=intervention,
            block=block,
            centered=centered,
        )
        images.append(draw_heatmap(ax, frame, title=title))
    if any(image is not None for image in images):
        fig.colorbar(next(image for image in images if image is not None), ax=axes, shrink=0.82)
    save_figure(fig, out_dir, "main_routing_heatmaps")


def plot_transport_heatmaps(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    block: str,
    centered: bool,
) -> None:
    specs = [
        ("transport_invariant", "content", "Transport invariant under content"),
        ("transport_follow", "content", "Transport follows content"),
        ("transport_invariant", "structure", "Transport invariant under structure"),
        ("transport_follow", "structure", "Transport follows structure"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.7), constrained_layout=True)
    images = []
    for ax, (metric, intervention, title) in zip(axes.flat, specs):
        frame = metric_frame(
            summary,
            metric=metric,
            intervention=intervention,
            block=block,
            centered=centered,
        )
        images.append(draw_heatmap(ax, frame, title=title))
    if any(image is not None for image in images):
        fig.colorbar(next(image for image in images if image is not None), ax=axes, shrink=0.82)
    save_figure(fig, out_dir, "appendix_transport_heatmaps")


def plot_centered_appendix(summary: pd.DataFrame, out_dir: Path, *, block: str) -> None:
    specs = [
        ("routing_invariant", "content", "Routing invariant | content"),
        ("routing_follow", "content", "Routing follows | content"),
        ("routing_invariant", "structure", "Routing invariant | structure"),
        ("routing_follow", "structure", "Routing follows | structure"),
        ("transport_invariant", "content", "Transport invariant | content"),
        ("transport_follow", "content", "Transport follows | content"),
        ("transport_invariant", "structure", "Transport invariant | structure"),
        ("transport_follow", "structure", "Transport follows | structure"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(10.6, 4.8), constrained_layout=True)
    images = []
    for ax, (metric, intervention, title) in zip(axes.flat, specs):
        frame = metric_frame(
            summary,
            metric=metric,
            intervention=intervention,
            block=block,
            centered=True,
        )
        images.append(draw_heatmap(ax, frame, title=title, vmin=-1.0, vmax=1.0, cmap="coolwarm"))
    if any(image is not None for image in images):
        fig.colorbar(next(image for image in images if image is not None), ax=axes, shrink=0.82)
    save_figure(fig, out_dir, "appendix_centered_score_heatmaps")


def paired_metric_points(
    summary: pd.DataFrame,
    *,
    field: str,
    intervention: str,
    block: str,
    centered: bool,
) -> pd.DataFrame:
    inv = metric_frame(
        summary,
        metric=f"{field}_invariant",
        intervention=intervention,
        block=block,
        centered=centered,
    )[["layer", "head", "mean"]].rename(columns={"mean": "invariant"})
    fol = metric_frame(
        summary,
        metric=f"{field}_follow",
        intervention=intervention,
        block=block,
        centered=centered,
    )[["layer", "head", "mean"]].rename(columns={"mean": "follow"})
    return inv.merge(fol, on=["layer", "head"], how="inner")


def plot_scatter_panels(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    block: str,
    centered: bool,
    stem: str = "main_specialisation_scatter",
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.2), constrained_layout=True)
    fields = ["routing", "transport"]
    interventions = ["content", "structure"]
    image = None
    for row_idx, intervention in enumerate(interventions):
        for col_idx, field in enumerate(fields):
            ax = axes[row_idx, col_idx]
            points = paired_metric_points(
                summary,
                field=field,
                intervention=intervention,
                block=block,
                centered=centered,
            )
            if points.empty:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                ax.set_axis_off()
                continue
            image = ax.scatter(
                points["invariant"],
                points["follow"],
                c=points["layer"],
                cmap="viridis",
                s=28,
                edgecolor="white",
                linewidth=0.35,
            )
            ax.plot([0, 1], [0, 1], color="0.75", linewidth=0.8, zorder=0)
            ax.set_xlim(-0.04, 1.04)
            ax.set_ylim(-0.04, 1.04)
            ax.set_xlabel("Invariant")
            ax.set_ylabel("Follows swap")
            ax.set_title(f"{field.title()} | {intervention}")
        ax = axes[row_idx, 2]
        output = summary[
            (summary["field"] == "output")
            & (summary["intervention"] == intervention)
            & (summary["block"] == block)
        ]
        resp = output.pivot_table(
            index=["layer", "head"],
            columns="metric",
            values="mean",
            aggfunc="mean",
        ).reset_index()
        needed = {"output_routing_responsibility", "output_transport_responsibility"}
        if resp.empty or not needed.issubset(resp.columns):
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.set_axis_off()
            continue
        size = 30.0
        if "output_sensitivity" in resp:
            size = 20.0 + 80.0 * resp["output_sensitivity"].clip(lower=0, upper=1)
        image = ax.scatter(
            resp["output_routing_responsibility"],
            resp["output_transport_responsibility"],
            c=resp["layer"],
            s=size,
            cmap="viridis",
            edgecolor="white",
            linewidth=0.35,
        )
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("Routing responsibility")
        ax.set_ylabel("Transport responsibility")
        ax.set_title(f"Realised output | {intervention}")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.82, label="Layer")
    save_figure(fig, out_dir, stem)


def plot_layer_trends(summary: pd.DataFrame, out_dir: Path, *, block: str, centered: bool) -> None:
    specs = [
        ("routing_follow", "content", "Routing follows content"),
        ("routing_follow", "structure", "Routing follows structure"),
        ("transport_follow", "content", "Transport follows content"),
        ("transport_follow", "structure", "Transport follows structure"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for metric, intervention, label in specs:
        frame = metric_frame(
            summary,
            metric=metric,
            intervention=intervention,
            block=block,
            centered=centered,
        )
        if frame.empty:
            continue
        layer = frame.groupby("layer")["mean"].agg(["mean", "sem"]).reset_index()
        ax.errorbar(
            layer["layer"],
            layer["mean"],
            yerr=layer["sem"].fillna(0.0),
            marker="o",
            linewidth=1.3,
            capsize=2.0,
            label=label,
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean over heads")
    ax.set_ylim(-0.04, 1.04)
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Layer-wise specialisation trends")
    save_figure(fig, out_dir, "main_layer_trends")


def plot_variation(summary: pd.DataFrame, out_dir: Path, *, block: str, centered: bool) -> None:
    specs = [
        ("routing_follow", "content", "Routing content-following"),
        ("routing_follow", "structure", "Routing structure-following"),
        ("transport_follow", "content", "Transport content-following"),
        ("transport_follow", "structure", "Transport structure-following"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.7), constrained_layout=True)
    images = []
    for ax, (metric, intervention, title) in zip(axes.flat, specs):
        frame = metric_frame(
            summary,
            metric=metric,
            intervention=intervention,
            block=block,
            centered=centered,
        )
        images.append(
            draw_heatmap(
                ax,
                frame,
                title=title,
                value="between_graph_variance",
                vmin=0.0,
                vmax=None,
                cmap="magma",
            )
        )
    if any(image is not None for image in images):
        fig.colorbar(next(image for image in images if image is not None), ax=axes, shrink=0.82)
    save_figure(fig, out_dir, "appendix_between_graph_variation")


def plot_variation_scatter(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    block: str,
    centered: bool,
) -> None:
    specs = [
        ("routing_follow", "content", "Routing follows content"),
        ("routing_follow", "structure", "Routing follows structure"),
        ("transport_follow", "content", "Transport follows content"),
        ("transport_follow", "structure", "Transport follows structure"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.7), constrained_layout=True)
    image = None
    for ax, (metric, intervention, title) in zip(axes.flat, specs):
        frame = metric_frame(
            summary,
            metric=metric,
            intervention=intervention,
            block=block,
            centered=centered,
        )
        if frame.empty:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.set_axis_off()
            continue
        image = ax.scatter(
            frame["mean"],
            frame["std_between_graphs"],
            c=frame["layer"],
            cmap="viridis",
            s=28,
            edgecolor="white",
            linewidth=0.35,
        )
        ax.set_xlabel("Mean score")
        ax.set_ylabel("Across-graph SD")
        ax.set_title(title)
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(bottom=-0.01)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.82, label="Layer")
    save_figure(fig, out_dir, "appendix_variation_scatter")


def plot_output_heatmaps(summary: pd.DataFrame, out_dir: Path, *, block: str) -> None:
    specs = [
        ("output_routing_responsibility", "content", "Routing resp. | content"),
        ("output_transport_responsibility", "content", "Transport resp. | content"),
        ("output_sensitivity", "content", "Sensitivity | content"),
        ("output_routing_responsibility", "structure", "Routing resp. | structure"),
        ("output_transport_responsibility", "structure", "Transport resp. | structure"),
        ("output_sensitivity", "structure", "Sensitivity | structure"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 4.8), constrained_layout=True)
    for ax, (metric, intervention, title) in zip(axes.flat, specs):
        frame = metric_frame(
            summary,
            metric=metric,
            intervention=intervention,
            block=block,
            centered=None,
        )
        vmax = 1.0 if metric != "output_sensitivity" else None
        image = draw_heatmap(ax, frame, title=title, vmin=0.0, vmax=vmax)
        if image is not None:
            fig.colorbar(image, ax=ax, fraction=0.046)
    save_figure(fig, out_dir, "appendix_output_response_heatmaps")


def plot_entropy_heatmap(summary: pd.DataFrame, out_dir: Path) -> None:
    frame = metric_frame(
        summary,
        metric="attention_entropy",
        intervention="none",
        block="all",
        centered=None,
    )
    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    image = draw_heatmap(
        ax,
        frame,
        title="Attention entropy",
        vmin=0.0,
        vmax=1.0,
        cmap="magma",
    )
    if image is not None:
        fig.colorbar(image, ax=ax, shrink=0.82, label="Normalised entropy")
    save_figure(fig, out_dir, "appendix_attention_entropy_heatmap")


def plot_locality(summary: pd.DataFrame, out_dir: Path, *, centered: bool) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.7), constrained_layout=True)
    specs = [
        ("global_routing_gate", "none", "global", None, "Global routing mass"),
        ("global_transport_gate", "none", "global", None, "Global transported norm"),
        ("routing_follow", "content", "local", centered, "Local content-following routing"),
        ("routing_follow", "content", "global", centered, "Global content-following routing"),
    ]
    images = []
    for ax, (metric, intervention, block, cflag, title) in zip(axes.flat, specs):
        frame = metric_frame(
            summary,
            metric=metric,
            intervention=intervention,
            block=block,
            centered=cflag,
        )
        images.append(draw_heatmap(ax, frame, title=title))
    if any(image is not None for image in images):
        fig.colorbar(next(image for image in images if image is not None), ax=axes, shrink=0.82)
    save_figure(fig, out_dir, "appendix_locality_globality")


def plot_support_partition_trends(summary: pd.DataFrame, out_dir: Path, *, centered: bool) -> None:
    specs = [
        ("routing_follow", "content", "Routing follows content"),
        ("routing_follow", "structure", "Routing follows structure"),
        ("transport_follow", "content", "Transport follows content"),
        ("transport_follow", "structure", "Transport follows structure"),
    ]
    colors = {"all": "0.25", "local": "#2a9d8f", "global": "#e76f51"}
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.7), constrained_layout=True)
    for ax, (metric, intervention, title) in zip(axes.flat, specs):
        for block, color in colors.items():
            frame = metric_frame(
                summary,
                metric=metric,
                intervention=intervention,
                block=block,
                centered=centered,
            )
            if frame.empty:
                continue
            layer = frame.groupby("layer")["mean"].agg(["mean", "sem"]).reset_index()
            ax.errorbar(
                layer["layer"],
                layer["mean"],
                yerr=layer["sem"].fillna(0.0),
                marker="o",
                linewidth=1.2,
                capsize=2.0,
                color=color,
                label=block,
            )
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean over heads")
        ax.set_ylim(-0.04, 1.04)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3)
    save_figure(fig, out_dir, "appendix_support_partition_trends")


def plot_graph_convergence(
    per_graph: pd.DataFrame,
    out_dir: Path,
    *,
    block: str,
    centered: bool,
) -> None:
    if per_graph.empty or "graph_index" not in per_graph:
        return
    specs = [
        ("routing_follow", "content", centered, "Routing follows content"),
        ("routing_follow", "structure", centered, "Routing follows structure"),
        ("transport_follow", "content", centered, "Transport follows content"),
        ("transport_follow", "structure", centered, "Transport follows structure"),
        ("output_sensitivity", "content", None, "Output sensitivity | content"),
        ("output_sensitivity", "structure", None, "Output sensitivity | structure"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 4.8), constrained_layout=True)
    for ax, (metric, intervention, cflag, title) in zip(axes.flat, specs):
        frame = per_graph[
            (per_graph["metric"] == metric)
            & (per_graph["intervention"] == intervention)
            & (per_graph["block"] == block)
        ].copy()
        if cflag is None:
            frame = frame[frame["centered"].isna()]
        else:
            frame = frame[frame["centered"] == cflag]
        if frame.empty:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.set_axis_off()
            continue
        by_graph = frame.groupby("graph_index")["score"].mean().sort_index()
        values = by_graph.to_numpy(dtype=float)
        counts = np.arange(1, len(values) + 1)
        cumulative = np.cumsum(values) / counts
        ax.plot(counts, cumulative, color="#264653", linewidth=1.5)
        final = float(np.nanmean(values))
        sem = float(pd.Series(values).sem()) if len(values) > 1 else 0.0
        ax.axhline(final, color="0.55", linewidth=0.9, linestyle="--")
        ax.fill_between(counts, final - sem, final + sem, color="0.85", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Graphs")
        ax.set_ylabel("Cumulative mean")
    save_figure(fig, out_dir, "appendix_graph_count_convergence")


def parse_convergence_dirs(items: Sequence[str]) -> list[tuple[float, Path]]:
    parsed = []
    for item in items:
        if "=" in item:
            label, path = item.split("=", 1)
            x = float(label)
        else:
            path = item
            x = float(len(parsed))
        parsed.append((x, Path(path)))
    return parsed


def plot_convergence(
    dirs: Sequence[tuple[float, Path]],
    out_dir: Path,
    *,
    metric: str,
    intervention: str,
    block: str,
    centered: bool,
) -> None:
    if not dirs:
        return
    xs = []
    ys = []
    yerr = []
    for x, path in dirs:
        summary, _per_graph, _meta = read_metric_dir(path)
        frame = metric_frame(
            summary,
            metric=metric,
            intervention=intervention,
            block=block,
            centered=centered,
        )
        if frame.empty:
            continue
        xs.append(x)
        ys.append(frame["mean"].mean())
        yerr.append(frame["mean"].sem())
    if not xs:
        return
    order = np.argsort(xs)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.errorbar(np.array(xs)[order], np.array(ys)[order], yerr=np.array(yerr)[order], marker="o")
    ax.set_xlabel("Budget")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Metric convergence")
    save_figure(fig, out_dir, "appendix_metric_convergence")


def parse_label_dirs(items: Sequence[str]) -> list[tuple[str, Path]]:
    parsed = []
    for item in items:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            path = item
            label = Path(path).name
        parsed.append((label, Path(path)))
    return parsed


def plot_comparison(
    compare_dirs: Sequence[tuple[str, Path]],
    out_dir: Path,
    *,
    metric: str,
    intervention: str,
    block: str,
    centered: bool,
) -> None:
    if not compare_dirs:
        return
    loaded = []
    for label, path in compare_dirs:
        summary, _per_graph, _meta = read_metric_dir(path)
        frame = metric_frame(
            summary,
            metric=metric,
            intervention=intervention,
            block=block,
            centered=centered,
        )
        if not frame.empty:
            loaded.append((label, frame))
    if not loaded:
        return

    fig, axes = plt.subplots(
        1,
        len(loaded),
        figsize=(2.6 * len(loaded), 2.6),
        squeeze=False,
        constrained_layout=True,
    )
    images = []
    for ax, (label, frame) in zip(axes.flat, loaded):
        images.append(draw_heatmap(ax, frame, title=label))
    if any(image is not None for image in images):
        fig.colorbar(next(image for image in images if image is not None), ax=axes, shrink=0.8)
    save_figure(fig, out_dir, "comparison_heatmaps")

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for label, frame in loaded:
        layer = frame.groupby("layer")["mean"].agg(["mean", "sem"]).reset_index()
        ax.errorbar(
            layer["layer"],
            layer["mean"],
            yerr=layer["sem"].fillna(0.0),
            marker="o",
            linewidth=1.2,
            capsize=2.0,
            label=label,
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(-0.04, 1.04)
    ax.legend(frameon=False)
    ax.set_title("Model comparison")
    save_figure(fig, out_dir, "comparison_layer_trends")


def graph_to_networkx(graph: Any) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(int(graph.num_nodes)))
    if graph.edge_index.numel():
        edges = graph.edge_index.t().cpu().tolist()
        g.add_edges_from((int(src), int(dst)) for src, dst in edges)
    return g


def parse_int_csv(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def attention_graph_indices(
    args: argparse.Namespace,
    metadata: Mapping[str, Any],
    per_graph: pd.DataFrame,
) -> list[int]:
    if args.attention_graph_indices:
        return parse_int_csv(args.attention_graph_indices)[: args.attention_num_graphs]
    if "graph_index" in per_graph and not per_graph.empty:
        values = sorted(int(value) for value in per_graph["graph_index"].dropna().unique())
        if values:
            return values[: args.attention_num_graphs]
    selected = metadata.get("selected_graph_indices")
    if isinstance(selected, Sequence) and not isinstance(selected, str):
        return [int(value) for value in selected[: args.attention_num_graphs]]
    return []


def load_attention_example_context(
    args: argparse.Namespace,
    metadata: Mapping[str, Any],
    graph_indices: Sequence[int],
):
    runner = load_runner(args.runner_path)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = build_screen_config(runner, args, checkpoint)
    splits = runner.load_official_graphbench_task(
        args.dataset_root,
        args.task or str(metadata.get("task")),
        cfg,
        force_reload=False,
        log=print,
    )
    splits = runner.attach_or_build_pe_cache(
        splits,
        args.pe_cache_root,
        cfg,
        namespace=args.pe_cache_namespace,
        dtype_name=args.pe_cache_dtype,
        pe_workers=1,
        pe_save_every=500,
        force_recompute=False,
        build_missing=False,
        require_present=True,
        log=print,
    )
    if graph_indices:
        graphs = [splits[args.split][int(index)] for index in graph_indices]
    else:
        graphs = select_graphs(splits[args.split], args.attention_num_graphs, args.graph_seed)
        graph_indices = list(range(len(graphs)))
    model = runner.build_model(
        args.model or str(metadata.get("model")),
        cfg,
        backend=args.model_backend,
    )
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    device = resolve_device(args.device)
    model.to(device).eval()
    batch = runner.collate_graphs(graphs).to(device)
    collector = make_collector(
        model,
        args.model or str(metadata.get("model")),
        "auto",
        max_nodes=max(graph.num_nodes for graph in graphs),
    )
    layers = collector.collect(batch)
    return graphs, layers, list(graph_indices)


def select_top_heads(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frame = metric_frame(
        summary,
        metric=args.attention_select_metric,
        intervention=args.attention_select_intervention,
        block=args.attention_select_block,
        centered=args.attention_select_centered,
    )
    if frame.empty:
        return frame
    return frame.sort_values("mean", ascending=False).head(args.attention_heads)


def draw_attention_weighted_graph(
    ax: plt.Axes,
    graph: Any,
    attention: torch.Tensor,
    *,
    title: str,
) -> None:
    g = graph_to_networkx(graph)
    n = int(graph.num_nodes)
    pos = nx.spring_layout(
        g,
        seed=17,
        k=2.6 / np.sqrt(max(n, 1)),
        iterations=200,
        scale=1.0,
    )
    attn = attention[:n, :n].float().clamp_min(0.0)
    key_mass = attn.sum(dim=0).numpy()
    query_mass = attn.sum(dim=1).numpy()
    node_weight = 0.5 * (key_mass + query_mass)
    denom = max(float(node_weight.max()), 1.0e-8)
    edges = list(g.edges())
    edge_scores = []
    for src, dst in edges:
        weight = float(0.5 * (attn[src, dst] + attn[dst, src]))
        edge_scores.append(weight)
    nx.draw_networkx_edges(g, pos, width=0.25, alpha=0.08, edge_color="0.35", ax=ax)
    if edges:
        top_k = min(len(edges), 80)
        order = np.argsort(edge_scores)[-top_k:]
        top_edges = [edges[idx] for idx in order]
        top_scores = np.array([edge_scores[idx] for idx in order])
        score_max = max(float(top_scores.max()), 1.0e-8)
        top_widths = 0.35 + 3.2 * top_scores / score_max
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=top_edges,
            width=top_widths,
            alpha=0.62,
            edge_color="#264653",
            ax=ax,
        )
    nx.draw_networkx_nodes(
        g,
        pos,
        node_size=20 + 120 * node_weight / denom,
        node_color=node_weight,
        cmap="viridis",
        linewidths=0.18,
        edgecolors="white",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_axis_off()


def graph_metric_score(
    per_graph: pd.DataFrame,
    *,
    graph_index: int,
    layer: int,
    head: int,
    metric: str,
    block: str,
    centered: bool,
) -> float:
    frame = per_graph[
        (per_graph["graph_index"] == graph_index)
        & (per_graph["layer"] == layer)
        & (per_graph["head"] == head)
        & (per_graph["metric"] == metric)
        & (per_graph["intervention"] == "content")
        & (per_graph["block"] == block)
        & (per_graph["centered"] == centered)
    ]
    if frame.empty:
        return float("nan")
    return float(frame["score"].mean())


def format_content_scores(
    per_graph: pd.DataFrame,
    *,
    graph_index: int,
    layer: int,
    head: int,
    block: str,
    centered: bool,
) -> str:
    scores = {
        "R sym": graph_metric_score(
            per_graph,
            graph_index=graph_index,
            layer=layer,
            head=head,
            metric="routing_follow",
            block=block,
            centered=centered,
        ),
        "R inv": graph_metric_score(
            per_graph,
            graph_index=graph_index,
            layer=layer,
            head=head,
            metric="routing_invariant",
            block=block,
            centered=centered,
        ),
        "T sym": graph_metric_score(
            per_graph,
            graph_index=graph_index,
            layer=layer,
            head=head,
            metric="transport_follow",
            block=block,
            centered=centered,
        ),
        "T inv": graph_metric_score(
            per_graph,
            graph_index=graph_index,
            layer=layer,
            head=head,
            metric="transport_invariant",
            block=block,
            centered=centered,
        ),
    }
    return " | ".join(
        f"{name} {value:.2f}" if np.isfinite(value) else f"{name} n/a"
        for name, value in scores.items()
    )


def plot_attention_examples(
    summary: pd.DataFrame,
    per_graph: pd.DataFrame,
    metadata: Mapping[str, Any],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    if args.checkpoint is None:
        return
    top = select_top_heads(summary, args)
    if top.empty:
        print("[attention] no heads matched selection; skipping examples", flush=True)
        return
    graph_indices = attention_graph_indices(args, metadata, per_graph)
    graphs, layers, graph_indices = load_attention_example_context(args, metadata, graph_indices)
    n_heads = len(top)
    n_graphs = min(len(graphs), args.attention_num_graphs)
    fig, axes = plt.subplots(
        n_heads,
        2 * n_graphs,
        figsize=(4.6 * n_graphs, 3.0 * n_heads),
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.35, 1.0] * n_graphs},
    )
    for row_idx, row in enumerate(top.itertuples(index=False)):
        layer = next(item for item in layers if int(item.layer) == int(row.layer))
        attn = layer.attention.detach().cpu()
        layer_idx = int(row.layer)
        head_idx = int(row.head)
        for graph_idx in range(n_graphs):
            graph_ax = axes[row_idx, 2 * graph_idx]
            mat_ax = axes[row_idx, 2 * graph_idx + 1]
            graph = graphs[graph_idx]
            source_index = int(graph_indices[graph_idx])
            n = int(graph.num_nodes)
            head_attn = attn[graph_idx, head_idx, :n, :n]
            score_text = format_content_scores(
                per_graph,
                graph_index=source_index,
                layer=layer_idx,
                head=head_idx,
                block=args.attention_select_block,
                centered=args.attention_select_centered,
            )
            draw_attention_weighted_graph(
                graph_ax,
                graph,
                head_attn,
                title=f"graph index {source_index}",
            )
            image = mat_ax.imshow(head_attn, cmap="viridis", vmin=0.0)
            mat_ax.set_title(score_text, fontsize=7)
            mat_ax.set_xlabel("key")
            mat_ax.set_ylabel("query")
            mat_ax.tick_params(length=2)
            fig.colorbar(image, ax=mat_ax, fraction=0.046)
        axes[row_idx, 0].set_ylabel(
            f"L{layer_idx} H{head_idx}\nselected mean={float(row.mean):.3f}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=32,
        )
    save_figure(fig, out_dir, "main_attention_examples")


def masked_head_matrices(
    layer: Any,
    *,
    graph_idx: int,
    head_idx: int,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    attn = layer.attention.detach().cpu()[graph_idx, head_idx, :num_nodes, :num_nodes].float()
    msg = layer.message.detach().cpu()[graph_idx, head_idx, :num_nodes, :num_nodes].float()
    mask = layer.mask.detach().cpu()[graph_idx, head_idx, :num_nodes, :num_nodes].bool()
    transport = torch.sqrt((msg * msg).sum(dim=-1).clamp_min(0.0))
    realised = attn.clamp_min(0.0) * transport
    zero = torch.zeros_like(attn)
    return (
        torch.where(mask, attn, zero).numpy(),
        torch.where(mask, transport, zero).numpy(),
        torch.where(mask, realised, zero).numpy(),
    )


def draw_operator_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    title: str,
    cmap: str,
) -> Any:
    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.percentile(finite, 99.0)) if finite.size else 1.0
    vmax = max(vmax, 1.0e-12)
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_title(f"{title}\nmax {float(np.nanmax(matrix)):.2g}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def plot_message_operator_examples(
    summary: pd.DataFrame,
    per_graph: pd.DataFrame,
    metadata: Mapping[str, Any],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    if args.checkpoint is None:
        return
    top = select_top_heads(summary, args)
    if top.empty:
        return
    graph_indices = attention_graph_indices(args, metadata, per_graph)
    graphs, layers, graph_indices = load_attention_example_context(args, metadata, graph_indices)
    n_graphs = min(len(graphs), args.attention_num_graphs)
    examples = []
    for row in top.itertuples(index=False):
        layer_idx = int(row.layer)
        head_idx = int(row.head)
        layer = next(item for item in layers if int(item.layer) == layer_idx)
        for graph_pos in range(n_graphs):
            examples.append((layer, layer_idx, head_idx, graph_pos, int(graph_indices[graph_pos])))
    if not examples:
        return

    fig, axes = plt.subplots(
        len(examples),
        3,
        figsize=(9.8, 2.35 * len(examples)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, (layer, layer_idx, head_idx, graph_pos, source_index) in enumerate(examples):
        n = int(graphs[graph_pos].num_nodes)
        attention, transport, realised = masked_head_matrices(
            layer,
            graph_idx=graph_pos,
            head_idx=head_idx,
            num_nodes=n,
        )
        score_text = format_content_scores(
            per_graph,
            graph_index=source_index,
            layer=layer_idx,
            head=head_idx,
            block=args.attention_select_block,
            centered=args.attention_select_centered,
        )
        panels = [
            (attention, "attention A_ij", "viridis"),
            (transport, "transport ||m_ij||", "plasma"),
            (realised, "realised ||A_ij m_ij||", "magma"),
        ]
        for ax, (matrix, title, cmap) in zip(axes[row_idx], panels):
            draw_operator_matrix(ax, matrix, title=title, cmap=cmap)
        axes[row_idx, 0].set_ylabel(
            f"L{layer_idx} H{head_idx} G{source_index}\n{score_text}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=78,
            fontsize=7,
        )
    save_figure(fig, out_dir, "appendix_message_operator_matrices")


def run(args: argparse.Namespace) -> None:
    plt.rcParams.update(STYLE)
    out_dir = ensure_out_dir(args.output_dir)
    summary, per_graph, metadata = read_metric_dir(args.metric_dir)
    plot_main_heatmaps(summary, out_dir, block=args.block, centered=args.centered)
    plot_scatter_panels(summary, out_dir, block=args.block, centered=args.centered)
    plot_layer_trends(summary, out_dir, block=args.block, centered=args.centered)
    plot_centered_appendix(summary, out_dir, block=args.block)
    plot_scatter_panels(
        summary,
        out_dir,
        block=args.block,
        centered=True,
        stem="appendix_centered_specialisation_scatter",
    )
    plot_transport_heatmaps(summary, out_dir, block=args.block, centered=args.centered)
    plot_variation(summary, out_dir, block=args.block, centered=args.centered)
    plot_variation_scatter(summary, out_dir, block=args.block, centered=args.centered)
    plot_output_heatmaps(summary, out_dir, block=args.block)
    plot_entropy_heatmap(summary, out_dir)
    plot_locality(summary, out_dir, centered=args.centered)
    plot_support_partition_trends(summary, out_dir, centered=args.centered)
    plot_graph_convergence(per_graph, out_dir, block=args.block, centered=args.centered)
    plot_convergence(
        parse_convergence_dirs(args.convergence_dirs),
        out_dir,
        metric=args.convergence_metric,
        intervention=args.convergence_intervention,
        block=args.convergence_block,
        centered=args.centered,
    )
    plot_comparison(
        parse_label_dirs(args.compare_dirs),
        out_dir,
        metric=args.compare_metric,
        intervention=args.compare_intervention,
        block=args.compare_block,
        centered=args.centered,
    )
    plot_attention_examples(summary, per_graph, metadata, out_dir, args)
    plot_message_operator_examples(summary, per_graph, metadata, out_dir, args)
    print(f"[done] wrote figures to {out_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block", default="all")
    parser.add_argument("--centered", action="store_true", default=True)
    parser.add_argument("--uncentered", dest="centered", action="store_false")
    parser.add_argument("--convergence-dirs", nargs="*", default=[])
    parser.add_argument("--convergence-metric", default="routing_follow")
    parser.add_argument("--convergence-intervention", default="content")
    parser.add_argument("--convergence-block", default="all")
    parser.add_argument("--compare-dirs", nargs="*", default=[])
    parser.add_argument("--compare-metric", default="routing_follow")
    parser.add_argument("--compare-intervention", default="content")
    parser.add_argument("--compare-block", default="all")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-backend", default="official")
    parser.add_argument("--runner-path", type=Path, default=None)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            os.environ.get(
                "GRAPHBENCH_DATASET_ROOT",
                "/rds/user/jgg45/hpc-work/graphbench-algoreas/datasets",
            )
        ),
    )
    parser.add_argument(
        "--pe-cache-root",
        type=Path,
        default=Path(
            os.environ.get(
                "GRAPHBENCH_PE_CACHE_ROOT",
                "/rds/user/jgg45/hpc-work/graphbench-algoreas/pe_cache",
            )
        ),
    )
    parser.add_argument("--pe-cache-namespace", default="base_40k4k4k_n64")
    parser.add_argument("--pe-cache-dtype", default="float32")
    parser.add_argument("--split", default="test")
    parser.add_argument("--graph-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-num-graphs", type=int, default=4)
    parser.add_argument(
        "--attention-graph-indices",
        default="",
        help="Optional comma list of dataset graph indices for attention examples.",
    )
    parser.add_argument("--attention-select-metric", default="routing_follow")
    parser.add_argument("--attention-select-intervention", default="content")
    parser.add_argument("--attention-select-block", default="all")
    parser.add_argument("--attention-select-centered", action="store_true", default=True)
    parser.add_argument(
        "--attention-select-uncentered",
        dest="attention_select_centered",
        action="store_false",
    )
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--train-node-size", type=int, default=None)
    parser.add_argument("--val-node-size", type=int, default=None)
    parser.add_argument("--test-node-size", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
