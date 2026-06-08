#!/usr/bin/env python3
"""Paper-quality plots for core interpretability specialisation metrics.

Input is a metric directory produced by
``core_interpretability_specialisation_metrics.py``. The plotting code avoids
raw activation caches: it reads compact per-head/per-graph CSVs and only reloads
a model checkpoint when optional attention-example figures are requested.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

from graph_specialisation_metrics.core_interpretability_specialisation_metrics import (
    OfficialGRITFieldCollector,
    build_screen_config,
    find_default_runner_path,
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
    save_figure(fig, out_dir, "main_specialisation_scatter")


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


def import_module_from_path(path: Path, module_name: str):
    path = path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def graph_to_networkx(graph: Any) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(int(graph.num_nodes)))
    if graph.edge_index.numel():
        edges = graph.edge_index.t().cpu().tolist()
        g.add_edges_from((int(src), int(dst)) for src, dst in edges)
    return g


def load_attention_example_context(args: argparse.Namespace, metadata: Mapping[str, Any]):
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
    graphs = select_graphs(splits[args.split], args.attention_num_graphs, args.graph_seed)
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
    return graphs, layers


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


def plot_attention_examples(
    summary: pd.DataFrame,
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
    graphs, layers = load_attention_example_context(args, metadata)
    n_heads = len(top)
    n_graphs = min(len(graphs), args.attention_num_graphs)
    fig, axes = plt.subplots(
        n_heads,
        n_graphs + 1,
        figsize=(2.2 * (n_graphs + 1), 2.0 * n_heads),
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, row in enumerate(top.itertuples(index=False)):
        layer = next(item for item in layers if int(item.layer) == int(row.layer))
        attn = layer.attention.detach().cpu()
        for graph_idx in range(n_graphs):
            ax = axes[row_idx, graph_idx]
            graph = graphs[graph_idx]
            g = graph_to_networkx(graph)
            pos = nx.spring_layout(g, seed=17)
            n = int(graph.num_nodes)
            head_attn = attn[graph_idx, int(row.head), :n, :n]
            node_weight = head_attn.sum(dim=0).numpy()
            edge_weight = []
            for src, dst in g.edges():
                weight = float(0.5 * (head_attn[src, dst] + head_attn[dst, src]))
                edge_weight.append(0.4 + 4.0 * weight)
            nx.draw_networkx_edges(g, pos, width=edge_weight, alpha=0.45, ax=ax)
            nx.draw_networkx_nodes(
                g,
                pos,
                node_size=50 + 350 * node_weight / max(float(node_weight.max()), 1.0e-8),
                node_color=node_weight,
                cmap="viridis",
                ax=ax,
            )
            ax.set_title(f"graph {graph_idx}")
            ax.set_axis_off()
        ax = axes[row_idx, -1]
        image = ax.imshow(
            attn[0, int(row.head), : graphs[0].num_nodes, : graphs[0].num_nodes],
            cmap="viridis",
            vmin=0.0,
        )
        ax.set_title(f"L{int(row.layer)} H{int(row.head)}")
        ax.set_xlabel("key")
        ax.set_ylabel("query")
        fig.colorbar(image, ax=ax, fraction=0.046)
    save_figure(fig, out_dir, "main_attention_examples")


def run(args: argparse.Namespace) -> None:
    plt.rcParams.update(STYLE)
    out_dir = ensure_out_dir(args.output_dir)
    summary, _per_graph, metadata = read_metric_dir(args.metric_dir)
    plot_main_heatmaps(summary, out_dir, block=args.block, centered=args.centered)
    plot_scatter_panels(summary, out_dir, block=args.block, centered=args.centered)
    plot_layer_trends(summary, out_dir, block=args.block, centered=args.centered)
    plot_transport_heatmaps(summary, out_dir, block=args.block, centered=args.centered)
    plot_variation(summary, out_dir, block=args.block, centered=args.centered)
    plot_locality(summary, out_dir, centered=args.centered)
    plot_convergence(
        parse_convergence_dirs(args.convergence_dirs),
        out_dir,
        metric=args.convergence_metric,
        intervention=args.convergence_intervention,
        block=args.convergence_block,
        centered=args.centered,
    )
    plot_attention_examples(summary, metadata, out_dir, args)
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
    parser.add_argument("--attention-select-metric", default="routing_follow")
    parser.add_argument("--attention-select-intervention", default="content")
    parser.add_argument("--attention-select-block", default="all")
    parser.add_argument("--attention-select-centered", action="store_true", default=True)
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
