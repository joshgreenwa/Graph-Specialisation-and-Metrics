"""CPU-only qualitative attention figures for the focused GraphBench GRIT run."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .cache import atomic_json, checkpoint_sha256, load_cache_artifact_file
from .figures import FigureBuilder, FigureTheme, publication_style
from .protocol import PROTOCOL_VERSION, MethodologyConfig, stable_hash
from .tasks import get_task


ATTENTION_FIGURE_VERSION = "graphbench-grit-attention-v1"
ATTENTION_TASK = "graphbench_bipartite_matching_hard"


def _value(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record[name]
    return getattr(record, name)


def _head_tuple(value: Sequence[int]) -> tuple[int, int]:
    return int(value[0]), int(value[1])


def select_attention_heads(scores: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Select two semantic, one structural, and one high-J generalist head."""

    coordinates = scores["coordinates"]
    joint = np.asarray(_value(coordinates, "joint_sensitivity"), dtype=np.float64)
    selectivity = np.asarray(_value(coordinates, "selectivity"), dtype=np.float64)
    active = np.asarray(
        _value(coordinates, "active")
        if (isinstance(coordinates, Mapping) and "active" in coordinates)
        or hasattr(coordinates, "active")
        else np.isfinite(joint),
        dtype=bool,
    )
    finite = active & np.isfinite(joint) & np.isfinite(selectivity)
    all_heads = tuple(
        (int(layer), int(head))
        for layer, head in np.argwhere(finite).tolist()
    )
    if len(all_heads) < 4:
        all_heads = tuple(
            (int(layer), int(head))
            for layer, head in np.argwhere(
                np.isfinite(joint) & np.isfinite(selectivity)
            ).tolist()
        )
    if len(all_heads) < 4:
        raise ValueError("attention visualisation requires at least four finite heads")

    classification = scores.get("specialist_classification") or {}
    ranking = classification.get("strength_ranking", {})
    classified_heads = classification.get("heads", {})
    threshold = float(classification.get("preference_threshold", 0.10))

    semantic_ranked = [
        _head_tuple(row["head"])
        for row in ranking.get("semantic_candidates", ())
    ]
    structural_ranked = [
        _head_tuple(row["head"])
        for row in ranking.get("structural_candidates", ())
    ]
    semantic_fallback = sorted(
        (head for head in all_heads if selectivity[head] > 0),
        key=lambda head: (-float(selectivity[head]), -float(joint[head]), head),
    )
    structural_fallback = sorted(
        (head for head in all_heads if selectivity[head] < 0),
        key=lambda head: (float(selectivity[head]), -float(joint[head]), head),
    )

    used: set[tuple[int, int]] = set()

    def take_unique(
        preferred: Sequence[tuple[int, int]],
        fallback: Sequence[tuple[int, int]],
        count: int,
    ) -> list[tuple[int, int]]:
        output = []
        for head in tuple(preferred) + tuple(fallback) + all_heads:
            head = _head_tuple(head)
            if head in used or head not in all_heads:
                continue
            output.append(head)
            used.add(head)
            if len(output) == count:
                break
        return output

    semantic = take_unique(semantic_ranked, semantic_fallback, 2)
    structural = take_unique(structural_ranked, structural_fallback, 1)

    generalist_pool = [
        _head_tuple(head)
        for head in classified_heads.get("generalist", ())
        if _head_tuple(head) in all_heads and _head_tuple(head) not in used
    ]
    generalist_rule = (
        f"highest J among active heads with |D_rel| <= {threshold:.2f}"
    )
    if not generalist_pool:
        generalist_pool = [
            head
            for head in all_heads
            if head not in used and abs(float(selectivity[head])) <= threshold
        ]
    if generalist_pool:
        generalist = sorted(
            generalist_pool,
            key=lambda head: (-float(joint[head]), abs(float(selectivity[head])), head),
        )[0]
    else:
        generalist = sorted(
            (head for head in all_heads if head not in used),
            key=lambda head: (abs(float(selectivity[head])), -float(joint[head]), head),
        )[0]
        generalist_rule = "least-selective active fallback; no threshold generalist available"
    used.add(generalist)

    selected = (
        ("semantic_1", "Most semantic head", semantic[0], "descending D_rel"),
        ("semantic_2", "Semantic head 2", semantic[1], "descending D_rel"),
        ("structural", "Most structural head", structural[0], "ascending D_rel"),
        ("generalist", "High-J generalist", generalist, generalist_rule),
    )
    return tuple(
        {
            "role": role,
            "label": label,
            "layer": int(head[0]),
            "head": int(head[1]),
            "J": float(joint[head]),
            "D_rel": float(selectivity[head]),
            "selection_rule": rule,
        }
        for role, label, head, rule in selected
    )


def _theme(config: MethodologyConfig) -> FigureTheme:
    values: Mapping[str, Any] = config.figure_overrides
    if ATTENTION_TASK in values and isinstance(values[ATTENTION_TASK], Mapping):
        values = values[ATTENTION_TASK]
    return FigureTheme(
        width=9.2,
        height=12.4,
        dpi=600,
        font_size=9.5,
        label_size=10.0,
        title_size=10.5,
        tick_size=7.5,
        marker_size=32.0,
        line_width=1.2,
        grid_alpha=0.12,
    ).with_overrides(values)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=path.name,
        suffix=".partial",
        dir=path.parent,
    )
    try:
        with open(handle, "wb", closefd=True) as stream:
            np.savez_compressed(stream, **arrays)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _cache_paths(
    config: MethodologyConfig,
    task_name: str,
    seed: int,
    graph_id: int,
) -> tuple[Path, Path]:
    root = config.root / task_name / "attention_cache"
    stem = f"seed_{int(seed)}_graph_{int(graph_id):04d}"
    return root / f"{stem}.npz", root / f"{stem}.json"


def _capture_fingerprint(
    *,
    task_name: str,
    seed: int,
    graph_id: int,
    heads: Sequence[Mapping[str, Any]],
    score_cache_sha256: str,
    checkpoint_sha256_value: str,
) -> str:
    return stable_hash(
        {
            "version": ATTENTION_FIGURE_VERSION,
            "task": task_name,
            "seed": int(seed),
            "graph_id": int(graph_id),
            "heads": [dict(head) for head in heads],
            "score_cache_sha256": score_cache_sha256,
            "checkpoint_sha256": checkpoint_sha256_value,
            "attention_site": "official GRIT post-softmax batch.attn",
        }
    )


def _load_cached_capture(
    array_path: Path,
    metadata_path: Path,
    fingerprint: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    if not array_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != fingerprint:
            return None
        with np.load(array_path, allow_pickle=False) as payload:
            arrays = {name: payload[name].copy() for name in payload.files}
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    required = {
        "attention",
        "edge_index",
        "edge_value",
        "edge_target",
        "node_type",
    }
    return (arrays, metadata) if required.issubset(arrays) else None


def _model_record(config: MethodologyConfig, task_name: str, seed: int) -> dict[str, Any]:
    path = config.root / task_name / f"seed_{int(seed)}" / "model.json"
    if not path.is_file():
        raise FileNotFoundError(f"attention extraction requires {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_attention_capture(
    config: MethodologyConfig,
    task_name: str,
    seed: int,
    graph_id: int,
    selected_heads: Sequence[Mapping[str, Any]],
    model_record: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    from .graphbench import GraphBenchGritBackend, build_graphbench_runtime

    checkpoint = Path(str(model_record["checkpoint"])).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"attention extraction requires checkpoint {checkpoint}")
    expected_sha = str(model_record["checkpoint_sha256"])
    observed_sha = checkpoint_sha256(checkpoint)
    if observed_sha != expected_sha:
        raise RuntimeError(
            "attention checkpoint digest does not match the immutable analysis model record"
        )
    task = get_task(task_name)
    overrides = dict(config.task_overrides.get(task_name, {}))
    overrides["skip_checkpoint_metric_reproduction"] = True
    torch.set_num_threads(int(config.num_threads))
    runtime, _ = build_graphbench_runtime(
        task,
        checkpoint_path=checkpoint,
        train_seed=int(seed),
        accelerator="cpu",
        overrides=overrides,
        jacobian_output_chunk=int(config.execution.jacobian_output_chunk),
    )
    if not 0 <= int(graph_id) < len(runtime.eval_ds):
        raise IndexError(
            f"validation graph {int(graph_id)} is outside [0, {len(runtime.eval_ds)})"
        )
    sigma = np.asarray(model_record.get("sigma", [1.0]), dtype=np.float64)
    backend = GraphBenchGritBackend(
        runtime,
        task,
        sigma,
        jacobian_output_chunk=int(config.execution.jacobian_output_chunk),
    )
    graph = runtime.eval_ds[int(graph_id)]
    all_attention = backend.clean_attention_matrices(graph)
    attention = np.stack(
        [
            all_attention[int(record["layer"]), int(record["head"])]
            for record in selected_heads
        ],
        axis=0,
    )
    receiver_mass = attention.sum(axis=-1)
    normalization_error = float(np.max(np.abs(receiver_mass - 1.0)))
    arrays = {
        "attention": attention.astype(np.float32, copy=False),
        "edge_index": graph.edge_index.detach().cpu().numpy().astype(np.int64),
        "edge_value": graph.edge_value.detach().cpu().numpy().astype(np.float32),
        "edge_target": graph.target.detach().cpu().numpy().astype(np.float32),
        "node_type": graph.node_type.detach().cpu().numpy().astype(np.int64),
    }
    metadata = {
        "num_nodes": int(graph.num_nodes),
        "attention_normalization_max_error": normalization_error,
        "attention_shape": list(attention.shape),
        "attention_axis_order": ["selected_head", "receiver", "sender"],
        "graph_split": "validation",
        "official_grit_commit": runtime.checks.get("official_grit_commit"),
        "checkpoint_metric_reproduction_skipped": True,
    }
    return arrays, metadata


def _undirected_input_edges(
    edge_index: np.ndarray,
    edge_value: np.ndarray,
) -> tuple[list[tuple[int, int]], list[float]]:
    values: dict[tuple[int, int], list[float]] = {}
    for position, (left, right) in enumerate(np.asarray(edge_index).T):
        left, right = int(left), int(right)
        if left == right:
            continue
        key = tuple(sorted((left, right)))
        values.setdefault(key, []).append(float(edge_value[position]))
    edges = sorted(values)
    return edges, [float(np.mean(values[edge])) for edge in edges]


def _top_attention_edges(
    matrix: np.ndarray,
    *,
    top_k_per_receiver: int,
) -> list[tuple[int, int, float]]:
    output = []
    for receiver in range(matrix.shape[0]):
        order = np.argsort(-matrix[receiver], kind="stable")
        for sender in order[: int(top_k_per_receiver)]:
            value = float(matrix[receiver, sender])
            if np.isfinite(value) and value > 0:
                output.append((int(sender), int(receiver), value))
    return output


def plot_attention_head_grid(
    payload: Mapping[str, Any],
    theme: FigureTheme,
    *,
    top_k_per_receiver: int,
):
    """Plot complete attention matrices beside sparse overlays on one input graph."""

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    attention = np.asarray(payload["attention"], dtype=np.float64)
    edge_index = np.asarray(payload["edge_index"], dtype=np.int64)
    edge_value = np.asarray(payload["edge_value"], dtype=np.float64)
    selected_heads = tuple(payload["selected_heads"])
    num_nodes = int(payload["num_nodes"])
    input_edges, input_values = _undirected_input_edges(edge_index, edge_value)
    input_edge_set = set(input_edges)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(input_edges)
    positions = nx.spring_layout(graph, seed=31_415, weight=None)
    directed = nx.DiGraph()
    directed.add_nodes_from(range(num_nodes))

    finite_positive = attention[np.isfinite(attention) & (attention > 0)]
    vmax = float(np.max(finite_positive)) if finite_positive.size else 1.0
    cmap = mpl.colormaps["magma"]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(vmax, 1.0e-12))
    raw_scale = np.asarray(input_values, dtype=np.float64)
    if raw_scale.size and float(np.ptp(raw_scale)) > 0:
        raw_scale = (raw_scale - np.min(raw_scale)) / np.ptp(raw_scale)
    else:
        raw_scale = np.zeros_like(raw_scale)

    with publication_style(theme):
        fig, axes = plt.subplots(
            len(selected_heads),
            2,
            figsize=(theme.width, theme.height),
            gridspec_kw={"width_ratios": (1.0, 1.12)},
        )
        for row, record in enumerate(selected_heads):
            matrix_axis, graph_axis = axes[row]
            matrix = attention[row]
            matrix_axis.imshow(
                matrix,
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                aspect="equal",
                rasterized=True,
            )
            matrix_axis.set_xticks(np.arange(num_nodes))
            matrix_axis.set_yticks(np.arange(num_nodes))
            matrix_axis.tick_params(labelsize=6.2, length=2.0, pad=1.5)
            matrix_axis.set_ylabel(
                f"{record['label']}\n"
                f"L{int(record['layer'])}H{int(record['head'])}\n"
                rf"$D_{{\rm rel}}={float(record['D_rel']):+.2f}$; "
                rf"$J={float(record['J']):.2f}$",
                rotation=0,
                ha="right",
                va="center",
                labelpad=24,
            )
            if row == 0:
                matrix_axis.set_title(
                    "Complete attention matrix\n"
                    "(receiving node × sending node)"
                )

            nx.draw_networkx_edges(
                graph,
                positions,
                ax=graph_axis,
                edgelist=input_edges,
                edge_color="#C7C7C7",
                width=(0.55 + 1.15 * raw_scale).tolist(),
                alpha=0.78,
            )
            selected_edges = _top_attention_edges(
                matrix,
                top_k_per_receiver=top_k_per_receiver,
            )
            for is_input, style in ((True, "solid"), (False, "dashed")):
                rows = [
                    item
                    for item in selected_edges
                    if (tuple(sorted(item[:2])) in input_edge_set) == is_input
                ]
                if not rows:
                    continue
                nx.draw_networkx_edges(
                    directed,
                    positions,
                    ax=graph_axis,
                    edgelist=[(sender, receiver) for sender, receiver, _ in rows],
                    edge_color=[value for _, _, value in rows],
                    edge_cmap=cmap,
                    edge_vmin=0.0,
                    edge_vmax=vmax,
                    width=[
                        0.65 + 2.9 * np.sqrt(value / max(vmax, 1.0e-12))
                        for _, _, value in rows
                    ],
                    style=style,
                    alpha=0.88,
                    arrows=True,
                    arrowstyle="-|>",
                    arrowsize=8,
                    connectionstyle="arc3,rad=0.08",
                    min_source_margin=7,
                    min_target_margin=7,
                )
            nx.draw_networkx_nodes(
                graph,
                positions,
                ax=graph_axis,
                node_size=210,
                node_color="white",
                edgecolors="#333333",
                linewidths=0.8,
            )
            nx.draw_networkx_labels(
                graph,
                positions,
                ax=graph_axis,
                font_size=6.8,
            )
            graph_axis.set_axis_off()
            if row == 0:
                graph_axis.set_title(
                    f"Attention on the input graph\n"
                    f"(top {int(top_k_per_receiver)} senders per receiving node)"
                )

        fig.suptitle(
            f"Clean GRIT attention - seed {int(payload['seed'])}, "
            f"validation graph {int(payload['graph_id'])}",
            y=0.992,
        )
        fig.subplots_adjust(
            left=0.24,
            right=0.97,
            top=0.905,
            bottom=0.16,
            hspace=0.34,
            wspace=0.18,
        )
        colorbar_axis = fig.add_axes((0.27, 0.068, 0.55, 0.014))
        colorbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=colorbar_axis,
            orientation="horizontal",
        )
        colorbar.set_label("Post-softmax attention probability")
        colorbar.ax.xaxis.set_label_position("top")
        colorbar.ax.xaxis.labelpad = 3
        legend_handles = (
            Line2D([0], [0], color="#C7C7C7", linewidth=1.4, label="Input edge"),
            Line2D(
                [0],
                [0],
                color=cmap(0.72),
                linewidth=1.8,
                label="Attention along an input edge",
            ),
            Line2D(
                [0],
                [0],
                color=cmap(0.72),
                linewidth=1.8,
                linestyle="--",
                label="Global attention to a non-edge",
            ),
        )
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.008),
            frameon=False,
            ncol=3,
            fontsize=8.2,
        )
    return fig, axes


def render_graphbench_attention_visualisation(
    config: MethodologyConfig,
    task_name: str,
    *,
    seed: int = 0,
    graph_id: int = 0,
    top_k_per_receiver: int = 2,
) -> dict[str, list[str]]:
    """Extract once on CPU, cache, then render the selected-head attention figure."""

    if task_name != ATTENTION_TASK:
        raise ValueError(f"attention visualisation is registered only for {ATTENTION_TASK}")
    if int(top_k_per_receiver) < 1:
        raise ValueError("top_k_per_receiver must be positive")
    score_path = (
        config.root
        / task_name
        / f"seed_{int(seed)}"
        / "cache"
        / "scores"
        / "raw.pt"
    )
    score_artifact = load_cache_artifact_file(score_path)
    selected_heads = select_attention_heads(score_artifact.value)
    model_record = _model_record(config, task_name, seed)
    fingerprint = _capture_fingerprint(
        task_name=task_name,
        seed=seed,
        graph_id=graph_id,
        heads=selected_heads,
        score_cache_sha256=score_artifact.file_sha256,
        checkpoint_sha256_value=str(model_record["checkpoint_sha256"]),
    )
    array_path, metadata_path = _cache_paths(config, task_name, seed, graph_id)
    cached = None if config.force else _load_cached_capture(
        array_path,
        metadata_path,
        fingerprint,
    )
    if cached is None:
        print(
            f"[attention] extracting seed={int(seed)} graph={int(graph_id)} on CPU",
            flush=True,
        )
        arrays, extraction_metadata = _extract_attention_capture(
            config,
            task_name,
            seed,
            graph_id,
            selected_heads,
            model_record,
        )
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "version": ATTENTION_FIGURE_VERSION,
            "fingerprint": fingerprint,
            "task": task_name,
            "seed": int(seed),
            "graph_id": int(graph_id),
            "selected_heads": selected_heads,
            "score_cache": str(score_path),
            "score_cache_sha256": score_artifact.file_sha256,
            "checkpoint": str(model_record["checkpoint"]),
            "checkpoint_sha256": str(model_record["checkpoint_sha256"]),
            **extraction_metadata,
        }
        _atomic_npz(array_path, **arrays)
        atomic_json(metadata_path, metadata)
        print(f"[attention] cached {array_path}", flush=True)
    else:
        arrays, metadata = cached
        print(f"[attention] cache hit {array_path}", flush=True)

    payload = {
        **arrays,
        **metadata,
        "selected_heads": tuple(metadata["selected_heads"]),
    }
    theme = _theme(config)
    figure, axes = plot_attention_head_grid(
        payload,
        theme,
        top_k_per_receiver=top_k_per_receiver,
    )
    output_dir = config.root / task_name / "population_figures"
    builder = FigureBuilder(
        output_dir,
        theme,
        common_metadata={
            "protocol_version": PROTOCOL_VERSION,
            "task": task_name,
            "train_seed": int(seed),
            "validation_graph": int(graph_id),
        },
        preserve_canvas=True,
    )
    paths = builder.save(
        f"07_seed{int(seed)}_selected_head_attention",
        figure,
        axes,
        metadata={
            "attention_site": "official GRIT post-softmax batch.attn",
            "matrix_axis_order": ["receiver", "sender"],
            "graph_overlay": (
                f"top {int(top_k_per_receiver)} senders per receiving node; "
                "solid attention arrows are input edges and dashed arrows are non-edges"
            ),
            "selected_heads": selected_heads,
            "capture_cache": str(array_path),
            "capture_fingerprint": fingerprint,
            "attention_normalization_max_error": metadata.get(
                "attention_normalization_max_error"
            ),
            "head_alignment": "none assumed across trained seeds",
        },
    )
    return {"selected_head_attention": [str(path) for path in paths]}


__all__ = (
    "ATTENTION_FIGURE_VERSION",
    "plot_attention_head_grid",
    "render_graphbench_attention_visualisation",
    "select_attention_heads",
)
