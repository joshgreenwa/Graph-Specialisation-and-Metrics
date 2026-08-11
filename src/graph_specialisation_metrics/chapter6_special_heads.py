"""Controlled multi-seed molecular diagnostics for selected Chapter 6 heads.

For each architecture, the analysis selects one semantic specialist, one
structural specialist, and one distinct highest-J head from the pooled set of
independently trained seeds.  Selection is deterministic and is joined to the
independent clean-head ablation endpoint before any model is reconstructed.

Every expensive forward-pass result is stored in a contract-keyed supplemental
cache.  Figures can therefore be restyled, or a paper subset of the ten cached
molecules can be chosen, without loading a checkpoint again.
"""

from __future__ import annotations

import csv
import ctypes
import gc
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .chapter6_clean_ablation import load_rows as load_ablation_rows
from .chapter6_multiseed import SEEDS, _scale_figure_text, dataset_spec
from .chapter6_spatial_explorer import head_metrics, load_models
from .methodology.cache import checkpoint_sha256
from .methodology.grit_figure_data import (
    CHEMISTRY_FOCUS_VERSION,
    SupplementalCache,
    build_verified_grit_figure_runtime,
    collect_attention_examples,
    compute_av_pca_inputs_many,
    figure_identity,
    load_canonical_model_record,
    load_canonical_score_artifact,
    methodology_config_from_record,
    retarget_verified_grit_figure_runtime,
)
from .methodology.grit_figure_plots import (
    ORANGE,
    apply_publication_style,
    plot_attention_grid_publication,
    plot_av_pca_publication,
    save_figure_bundle,
)
from .zinc_cached_rrwp_comparison import group_distance

ANALYSIS_VERSION = "chapter6-special-heads-v2-all-heads"
FIGURE_STYLE_VERSION = "chapter6-special-heads-paper-attention-v3"
ROLE_ORDER = ("semantic", "structural", "highest_joint")
ROLE_LABELS = {
    "semantic": "Most semantic head",
    "structural": "Most structural head",
    "highest_joint": r"Highest-$J$ head",
}
DEFAULT_GRAPH_INDICES = {
    "zinc": (0, 80, 100, 120, 160, 200, 220, 320, 560, 750),
    "qm9": (0, 24, 72, 80, 112, 208, 320, 560, 608, 750),
}
SPECIAL_HEAD_PREVIEW_DPI = 150
SPECIAL_HEAD_PDF_RASTER_DPI = 1200


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _normalise_profile(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array = np.where(np.isfinite(array), np.maximum(array, 0.0), 0.0)
    total = float(np.sum(array))
    return array / total if total > 1.0e-12 else np.zeros_like(array)


def head_distance_profiles(scores: Mapping[str, Any], head: tuple[int, int]) -> dict[str, Any]:
    """Return separate semantic, structural, and attention profile masses."""

    layer, index = int(head[0]), int(head[1])
    axis = tuple(scores["axis"])
    profiles: dict[str, Any] = {}
    labels: tuple[str, ...] | None = None
    for channel in ("semantic", "structural"):
        exact = scores["channels"][channel]["heatmap_exact_head"]
        if hasattr(exact, "detach"):
            exact = exact.detach().cpu().numpy()
        grouped_labels, grouped = group_distance(
            np.asarray(exact, dtype=np.float64)[layer, index], axis
        )
        labels = grouped_labels if labels is None else labels
        if grouped_labels != labels:
            raise ValueError("semantic and structural distance axes do not agree")
        profiles[channel] = _normalise_profile(grouped)
    attention = scores.get("clean_attention_distance")
    if attention is not None:
        if hasattr(attention, "detach"):
            attention = attention.detach().cpu().numpy()
        grouped_labels, grouped = group_distance(
            np.asarray(attention, dtype=np.float64)[layer, index], axis
        )
        if grouped_labels != labels:
            raise ValueError("attention and score distance axes do not agree")
        profiles["attention"] = _normalise_profile(grouped)
    if labels is None:
        raise ValueError("score cache has no distance profiles")
    return {"labels": labels, **profiles}


def select_special_heads_across_seeds(
    head_rows: Sequence[Mapping[str, Any]],
    ablation_rows: Sequence[Mapping[str, Any]],
    *,
    model_labels: Mapping[str, str],
    generalist_max_abs_drel: float = 0.10,
) -> list[dict[str, Any]]:
    """Select three distinct heads per architecture across all finite heads."""

    ablations = {
        (
            str(row["task"]),
            int(row["seed"]),
            int(row["layer"]),
            int(row["head"]),
        ): row
        for row in ablation_rows
    }
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in head_rows:
        if (
            not np.isfinite(float(row.get("selectivity", np.nan)))
            or not np.isfinite(float(row.get("joint_sensitivity", np.nan)))
        ):
            continue
        by_task.setdefault(str(row["task"]), []).append(row)

    output: list[dict[str, Any]] = []
    for task in model_labels:
        candidates = by_task.get(task, [])
        if len(candidates) < 3:
            raise ValueError(f"{task}: fewer than three finite heads are available")

        def identity(row: Mapping[str, Any]) -> tuple[int, int, int]:
            return int(row["seed"]), int(row["layer"]), int(row["head"])

        semantic_order = sorted(
            candidates,
            key=lambda row: (
                -float(row["selectivity"]),
                -float(row["joint_sensitivity"]),
                identity(row),
            ),
        )
        structural_order = sorted(
            candidates,
            key=lambda row: (
                float(row["selectivity"]),
                -float(row["joint_sensitivity"]),
                identity(row),
            ),
        )
        joint_order = sorted(
            candidates,
            key=lambda row: (
                -float(row["joint_sensitivity"]),
                abs(float(row["selectivity"])),
                identity(row),
            ),
        )
        chosen: list[tuple[str, Mapping[str, Any]]] = []
        used: set[tuple[int, int, int]] = set()
        for role, ordering in (
            ("semantic", semantic_order),
            ("structural", structural_order),
            ("highest_joint", joint_order),
        ):
            row = next(item for item in ordering if identity(item) not in used)
            used.add(identity(row))
            chosen.append((role, row))

        for role, row in chosen:
            key = (task, *identity(row))
            if key not in ablations:
                raise KeyError(f"missing clean-head ablation result for {key}")
            ablation = ablations[key]
            if not np.isclose(
                float(ablation["joint_sensitivity"]),
                float(row["joint_sensitivity"]),
                rtol=1.0e-7,
                atol=1.0e-10,
            ):
                raise ValueError(
                    f"clean-head ablation and score cache disagree on J for {key}"
                )
            d_rel = float(row["selectivity"])
            if role == "semantic":
                display_family = "semantic"
            elif role == "structural":
                display_family = "structural"
            elif abs(d_rel) <= float(generalist_max_abs_drel):
                display_family = "generalist"
            else:
                display_family = "semantic" if d_rel > 0 else "structural"
            output.append(
                {
                    **dict(row),
                    "task": task,
                    "model_label": str(model_labels[task]),
                    "role": role,
                    "role_label": ROLE_LABELS[role],
                    "display_family": display_family,
                    "head_ablation_impact": float(ablation["prediction_movement"]),
                    "head_ablation_loss_change": float(ablation["loss_change"]),
                    "head_ablation_graphs": int(ablation["clean_ablation_graphs"]),
                }
            )
    task_order = {task: index for index, task in enumerate(model_labels)}
    role_order = {role: index for index, role in enumerate(ROLE_ORDER)}
    output.sort(key=lambda row: (task_order[str(row["task"])], role_order[str(row["role"])]))
    return output


def _index_multiseed_models_streaming(
    canonical_root: Path,
    *,
    dataset: str,
    seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]], list[str]]:
    """Index head metrics while retaining no full score or carriage cache."""

    spec = dataset_spec(dataset)
    rows: list[dict[str, Any]] = []
    references: dict[tuple[str, int], dict[str, Any]] = {}
    warnings: list[str] = []
    for seed in seeds:
        for task in spec.tasks:
            values, messages = load_models(
                (Path(canonical_root),),
                (task,),
                seed=int(seed),
                load_carriage=False,
            )
            warnings.extend(messages)
            for model in values:
                rows.extend(
                    {"seed": int(model.seed), **row}
                    for row in head_metrics((model,))
                )
                references[(str(model.task), int(model.seed))] = {
                    "task": str(model.task),
                    "artifact_task": str(model.artifact_task),
                    "seed": int(model.seed),
                    "score_path": str(model.score_path),
                    "model_path": (
                        None if model.model_path is None else str(model.model_path)
                    ),
                    "score_contract": dict(model.score_metadata.get("contract", {})),
                }
            if values:
                del model
            values.clear()
            gc.collect()
    expected = {(task, int(seed)) for task in spec.tasks for seed in seeds}
    observed = set(references)
    missing = sorted(expected - observed)
    if missing:
        raise FileNotFoundError(f"missing score caches for selected-head analysis: {missing}")
    return rows, references, warnings


def _release_figure_runtime(figure_runtime: Any | None) -> None:
    """Drop the transformed dataset/model for one architecture immediately."""

    if figure_runtime is not None:
        prepared = figure_runtime.prepared
        runtime = prepared.runtime
        if runtime is not None:
            runtime.attn_layers = ()
            runtime.model = None
            runtime.loaders = None
            runtime.eval_ds = None
            runtime.donor_ds = None
        prepared.runtime = None
        prepared.backend = None
        prepared.donor_pool = None
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except (ImportError, RuntimeError):
        pass


def _plot_distance_profiles(
    profiles: Mapping[str, Any],
    *,
    display_title: str,
    role_label: str,
    head: tuple[int, int],
    d_rel: float,
    joint_sensitivity: float,
    head_ablation_impact: float,
):
    import matplotlib.pyplot as plt

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    labels = tuple(str(value) for value in profiles["labels"])
    x = np.arange(len(labels))
    styles = (
        ("semantic", "Semantic score", ORANGE, "o"),
        ("structural", "Structural score", "#2878B5", "s"),
        ("attention", "Attention mass", "#009E73", "^"),
    )
    for key, label, colour, marker in styles:
        if key not in profiles:
            continue
        ax.plot(
            x,
            np.asarray(profiles[key], dtype=np.float64),
            color=colour,
            marker=marker,
            linewidth=2.2,
            markersize=7,
            label=label,
        )
    ax.set_xticks(x, [label.replace("_", " ") for label in labels])
    ax.set_xlabel("Graph distance")
    ax.set_ylabel("Profile mass")
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"{display_title} — {role_label} — L{head[0]} H{head[1]}\n"
        "Distance dependence of score and attention\n"
        rf"$D_{{\rm rel}} = {d_rel:+.3f};\quad J = {joint_sensitivity:.3f};\quad "
        rf"\Delta \hat{{y}}_{{\rm ablate}} = {head_ablation_impact:.4g}$",
        fontsize=15,
    )
    ax.grid(axis="y")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
    )
    return fig


def _record_metadata(row: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "analysis_version": ANALYSIS_VERSION,
        "figure_style_version": FIGURE_STYLE_VERSION,
        "task": str(row["task"]),
        "model_label": str(row["model_label"]),
        "role": str(row["role"]),
        "display_family": str(row["display_family"]),
        "seed": int(row["seed"]),
        "layer": int(row["layer"]),
        "head": int(row["head"]),
        "D_rel": float(row["selectivity"]),
        "J": float(row["joint_sensitivity"]),
        "raw_semantic_score": float(row["raw_semantic_score"]),
        "raw_structural_score": float(row["raw_structural_score"]),
        "head_ablation_impact": float(row["head_ablation_impact"]),
        "head_ablation_graphs": int(row["head_ablation_graphs"]),
        **extra,
    }


def _discover_existing_outputs(
    selected: Sequence[Mapping[str, Any]],
    figures_dir: Path,
    *,
    render_graph_count: int,
    examples_per_page: int,
) -> list[dict[str, Any]]:
    """Recover complete deterministic bundles after a pre-manifest crash."""

    outputs: list[dict[str, Any]] = []
    for row in selected:
        task = str(row["task"])
        seed = int(row["seed"])
        role = str(row["role"])
        layer = int(row["layer"])
        head = int(row["head"])
        prefix = f"{task}_seed_{seed}_{role}_L{layer}_H{head}"
        stems: list[tuple[str, str]] = []
        for page_start in range(0, int(render_graph_count), int(examples_per_page)):
            page_end = min(
                page_start + int(examples_per_page), int(render_graph_count)
            )
            stems.append(
                (
                    "attention",
                    f"{prefix}_attention_{page_start + 1:02d}_{page_end:02d}",
                )
            )
        stems.extend(
            (
                ("pca", f"{prefix}_routed_output_pca"),
                ("distance", f"{prefix}_distance_profiles"),
            )
        )
        for figure_type, stem in stems:
            paths = {
                "png": figures_dir / f"{stem}.png",
                "pdf": figures_dir / f"{stem}.pdf",
                "metadata": figures_dir / f"{stem}.json",
            }
            metadata_matches = False
            if paths["metadata"].is_file():
                try:
                    saved_metadata = json.loads(
                        paths["metadata"].read_text(encoding="utf-8")
                    )
                    metadata_matches = (
                        saved_metadata.get("figure_style_version")
                        == FIGURE_STYLE_VERSION
                    )
                except (OSError, UnicodeError, json.JSONDecodeError):
                    metadata_matches = False
            if metadata_matches and all(path.is_file() for path in paths.values()):
                outputs.append(
                    {
                        "task": task,
                        "seed": seed,
                        "role": role,
                        "figure": figure_type,
                        **{key: str(value) for key, value in paths.items()},
                    }
                )
    return outputs


def _release_rendered_figure(figure: Any) -> None:
    """Destroy large Agg buffers and return their pages to the host OS."""

    import matplotlib.pyplot as plt

    figure.clear()
    plt.close(figure)
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def validate_special_head_outputs(
    selected: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    render_graph_count: int,
    examples_per_page: int,
) -> list[dict[str, Any]]:
    """Require a complete, readable figure set for every model and role."""

    expected_pages = (
        int(render_graph_count) + int(examples_per_page) - 1
    ) // int(examples_per_page)
    selected_lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in selected:
        key = (str(row["task"]), str(row["role"]))
        if key in selected_lookup:
            raise RuntimeError(f"duplicate selected special head: {key}")
        selected_lookup[key] = row

    expected = {(str(task), role) for task in tasks for role in ROLE_ORDER}
    if set(selected_lookup) != expected:
        missing = sorted(expected - set(selected_lookup))
        extra = sorted(set(selected_lookup) - expected)
        raise RuntimeError(
            f"special-head selection is incomplete; missing={missing}, extra={extra}"
        )

    completion: list[dict[str, Any]] = []
    for task in tasks:
        for role in ROLE_ORDER:
            key = (str(task), role)
            bundles = [
                row
                for row in outputs
                if (str(row["task"]), str(row["role"])) == key
            ]
            by_figure = {
                figure: [row for row in bundles if str(row["figure"]) == figure]
                for figure in ("attention", "pca", "distance")
            }
            observed = {
                "attention": len(by_figure["attention"]),
                "pca": len(by_figure["pca"]),
                "distance": len(by_figure["distance"]),
            }
            required = {
                "attention": expected_pages,
                "pca": 1,
                "distance": 1,
            }
            if observed != required:
                raise RuntimeError(
                    f"incomplete special-head figures for {task}/{role}: "
                    f"observed={observed}, required={required}"
                )
            missing_files = [
                str(row[path_type])
                for row in bundles
                for path_type in ("png", "pdf", "metadata")
                if not Path(row[path_type]).is_file()
            ]
            if missing_files:
                raise FileNotFoundError(
                    f"missing figure bundles for {task}/{role}: {missing_files}"
                )
            selection = selected_lookup[key]
            completion.append(
                {
                    "task": str(task),
                    "model_label": str(selection["model_label"]),
                    "role": role,
                    "seed": int(selection["seed"]),
                    "layer": int(selection["layer"]),
                    "head": int(selection["head"]),
                    "attention_pages": observed["attention"],
                    "pca_figures": observed["pca"],
                    "distance_figures": observed["distance"],
                    "attention_pngs": ";".join(
                        str(row["png"]) for row in by_figure["attention"]
                    ),
                    "pca_png": str(by_figure["pca"][0]["png"]),
                    "distance_png": str(by_figure["distance"][0]["png"]),
                    "complete": True,
                }
            )
    return completion


def generate_special_head_analysis(
    canonical_root: str | Path,
    ablation_root: str | Path,
    output_dir: str | Path,
    *,
    dataset: str,
    seeds: Sequence[int] = SEEDS,
    graph_indices: Sequence[int] | None = None,
    render_graph_indices: Sequence[int] | None = None,
    n_pca_graphs: int = 500,
    examples_per_page: int = 5,
    accelerator: str = "cuda:0",
    compute_missing: bool = True,
    force: bool = False,
    generalist_max_abs_drel: float = 0.10,
    verbose: bool = True,
) -> dict[str, Any]:
    """Select, cache, and render controlled diagnostics for fifteen heads."""

    canonical_root = Path(canonical_root)
    output_dir = Path(output_dir)
    analysis_dir = output_dir / "special_head_analysis"
    figures_dir = analysis_dir / "figures"
    cache_dir = analysis_dir / "cache"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    spec = dataset_spec(dataset)
    graph_indices = tuple(
        int(value)
        for value in (DEFAULT_GRAPH_INDICES[spec.name] if graph_indices is None else graph_indices)
    )
    if len(graph_indices) != 10:
        raise ValueError("controlled special-head analysis requires exactly 10 molecules")
    render_graph_indices = tuple(
        graph_indices
        if render_graph_indices is None
        else (int(value) for value in render_graph_indices)
    )
    if not render_graph_indices:
        raise ValueError("at least one cached molecule must be rendered")
    unavailable = sorted(set(render_graph_indices) - set(graph_indices))
    if unavailable:
        raise ValueError(
            "render_graph_indices must be selected from the ten cached molecules; "
            f"unavailable indices: {unavailable}"
        )
    if int(examples_per_page) < 1:
        raise ValueError("examples_per_page must be positive")

    manifest_path = analysis_dir / "manifest.json"
    existing_manifest: dict[str, Any] | None = None
    if manifest_path.is_file() and not force:
        try:
            candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
            request_matches = (
                candidate.get("analysis_version") == ANALYSIS_VERSION
                and candidate.get("figure_style_version") == FIGURE_STYLE_VERSION
                and candidate.get("dataset") == spec.name
                and candidate.get("seeds") == [int(seed) for seed in seeds]
                and candidate.get("models") == list(spec.tasks)
                and candidate.get("graph_indices") == list(graph_indices)
                and candidate.get("render_graph_indices") == list(render_graph_indices)
                and int(candidate.get("n_pca_graphs", -1)) == int(n_pca_graphs)
                and int(candidate.get("examples_per_page", -1))
                == int(examples_per_page)
            )
            if request_matches:
                existing_manifest = candidate
                try:
                    completion = validate_special_head_outputs(
                        candidate["selected_heads"],
                        candidate["outputs"],
                        tasks=spec.tasks,
                        render_graph_count=len(render_graph_indices),
                        examples_per_page=int(examples_per_page),
                    )
                except (KeyError, OSError, RuntimeError, FileNotFoundError):
                    completion = []
                if completion:
                    if verbose:
                        print(
                            "[special-heads-resume] all requested figures already "
                            "exist; no score cache or checkpoint loaded",
                            flush=True,
                        )
                    return {
                        **candidate,
                        "completion": completion,
                        "manifest_path": str(manifest_path),
                    }
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            existing_manifest = None

    persisted_selection: list[dict[str, Any]] | None = None
    persisted_warnings: list[str] = []
    if existing_manifest is not None:
        persisted_selection = [
            dict(row) for row in existing_manifest["selected_heads"]
        ]
        persisted_warnings = list(existing_manifest.get("warnings", ()))
    elif (analysis_dir / "selected_heads.json").is_file() and not force:
        try:
            candidate_selection = json.loads(
                (analysis_dir / "selected_heads.json").read_text(encoding="utf-8")
            )
            expected_selection = {
                (task, role) for task in spec.tasks for role in ROLE_ORDER
            }
            observed_selection = {
                (str(row["task"]), str(row["role"]))
                for row in candidate_selection
            }
            required_fields = {
                "task",
                "seed",
                "layer",
                "head",
                "role",
                "score_path",
                "selectivity",
                "joint_sensitivity",
            }
            if (
                len(candidate_selection) == len(expected_selection)
                and observed_selection == expected_selection
                and all(required_fields <= set(row) for row in candidate_selection)
                and all(
                    int(row["seed"]) in {int(seed) for seed in seeds}
                    and Path(str(row["score_path"])).is_file()
                    for row in candidate_selection
                )
            ):
                persisted_selection = [dict(row) for row in candidate_selection]
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            persisted_selection = None

    if persisted_selection is not None:
        selected = persisted_selection
        warnings = persisted_warnings
        for row in selected:
            if str(row["role"]) == "highest_joint":
                d_rel = float(row["selectivity"])
                row["display_family"] = (
                    "generalist"
                    if abs(d_rel) <= float(generalist_max_abs_drel)
                    else ("semantic" if d_rel > 0 else "structural")
                )
        model_references: dict[tuple[str, int], dict[str, Any]] = {}
        for row in selected:
            score_path = Path(str(row["score_path"]))
            key = (str(row["task"]), int(row["seed"]))
            model_references[key] = {
                "task": key[0],
                "artifact_task": score_path.parents[3].name,
                "seed": key[1],
                "score_path": str(score_path),
                "model_path": str(score_path.parents[2] / "model.json"),
                "score_contract": {
                    "checkpoint_sha256": str(row.get("checkpoint_sha256", ""))
                },
            }
        if verbose:
            print(
                "[special-heads-resume] reusing persisted all-head selection; "
                "no population score scan",
                flush=True,
            )
    else:
        rows, model_references, warnings = _index_multiseed_models_streaming(
            canonical_root, dataset=spec.name, seeds=seeds
        )
        ablations = load_ablation_rows(
            Path(ablation_root), dataset=spec.name, seeds=seeds, strict=True
        )
        selected = select_special_heads_across_seeds(
            rows,
            ablations,
            model_labels=spec.labels,
            generalist_max_abs_drel=generalist_max_abs_drel,
        )
        for row in selected:
            reference = model_references[(str(row["task"]), int(row["seed"]))]
            contract = reference["score_contract"]
            row["score_path"] = str(reference["score_path"])
            row["checkpoint_sha256"] = str(contract.get("checkpoint_sha256", ""))
    _write_csv(analysis_dir / "selected_heads.csv", selected)
    (analysis_dir / "selected_heads.json").write_text(
        json.dumps(selected, indent=2, default=_json_default), encoding="utf-8"
    )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault((str(row["task"]), int(row["seed"])), []).append(row)

    outputs: list[dict[str, Any]] = []
    cache_records: list[dict[str, Any]] = []
    existing_outputs = (
        list(existing_manifest.get("outputs", ()))
        if existing_manifest is not None
        else _discover_existing_outputs(
            selected,
            figures_dir,
            render_graph_count=len(render_graph_indices),
            examples_per_page=int(examples_per_page),
        )
    )
    existing_caches = (
        list(existing_manifest.get("caches", ()))
        if existing_manifest is not None
        else []
    )
    # Keep exactly one live transformed dataset/model for each incomplete
    # task/seed group, reuse it across that group's diagnostics, then destroy it.
    # This is load-bearing for standard-RAM QM9 Colab.
    active_runtime: dict[str, Any] = {}
    for (task, seed), group in grouped.items():
        group_roles = {str(row["role"]) for row in group}
        resumable_outputs = [
            dict(row)
            for row in existing_outputs
            if str(row.get("task")) == task
            and int(row.get("seed", -1)) == int(seed)
            and str(row.get("role")) in group_roles
        ]
        expected_pages = (
            len(render_graph_indices) + int(examples_per_page) - 1
        ) // int(examples_per_page)
        resumable = True
        for role in group_roles:
            role_outputs = [
                row for row in resumable_outputs if str(row.get("role")) == role
            ]
            counts = {
                figure: sum(
                    str(row.get("figure")) == figure for row in role_outputs
                )
                for figure in ("attention", "pca", "distance")
            }
            if counts != {
                "attention": expected_pages,
                "pca": 1,
                "distance": 1,
            } or any(
                not Path(str(row.get(field, ""))).is_file()
                for row in role_outputs
                for field in ("png", "pdf", "metadata")
            ):
                resumable = False
                break
        if resumable and resumable_outputs:
            outputs.extend(resumable_outputs)
            cache_records.extend(
                dict(row)
                for row in existing_caches
                if str(row.get("task")) == task
                and int(row.get("seed", -1)) == int(seed)
            )
            if verbose:
                print(
                    f"[special-heads-resume:{task}/seed_{seed}] complete; skipped",
                    flush=True,
                )
            continue
        if active_runtime.get("task") not in {None, task}:
            _release_figure_runtime(active_runtime.get("value"))
            active_runtime.clear()
        active_runtime["task"] = task
        reference = model_references[(task, seed)]
        score_path = Path(str(reference["score_path"]))
        artifact_task = str(reference["artifact_task"])
        artifact = load_canonical_score_artifact(score_path, expected_task=artifact_task)
        model_path = (
            Path(str(reference["model_path"]))
            if reference["model_path"] is not None
            else score_path.parents[2] / "model.json"
        )
        model_record = load_canonical_model_record(model_path, artifact)
        # Isolated multi-seed workers write their exact protocol beside
        # model.json inside <task>/seed_<n>/, not at the shared corpus root.
        protocol_path = score_path.parents[2] / "protocol.json"
        protocol = methodology_config_from_record(protocol_path, accelerator=accelerator)
        heads = {str(row["role"]): (int(row["layer"]), int(row["head"])) for row in group}
        cache = SupplementalCache(cache_dir / task / f"seed_{seed}")
        base_contract = {
            "analysis_version": ANALYSIS_VERSION,
            "task": artifact_task,
            "seed": int(seed),
            "canonical_score_sha256": artifact.file_sha256,
            "canonical_contract_fingerprint": artifact.metadata["contract_fingerprint"],
            "checkpoint_sha256": artifact.metadata["contract"]["checkpoint_sha256"],
            "heads": {role: list(head) for role, head in heads.items()},
            "chemistry_focus_version": CHEMISTRY_FOCUS_VERSION,
        }
        def runtime(
            runtime_state=active_runtime,
            current_task=task,
            current_seed=seed,
            current_artifact=artifact,
            current_model_record=model_record,
            current_protocol=protocol,
        ):
            target_checkpoint = str(
                current_artifact.metadata["contract"]["checkpoint_sha256"]
            )
            if "value" not in runtime_state:
                if verbose:
                    print(
                        f"[special-heads:{current_task}/seed_{current_seed}] "
                        "building one low-memory architecture runtime",
                        flush=True,
                    )
                overrides = current_protocol.task_overrides.get(
                    str(current_artifact.metadata["contract"]["task"]), {}
                )
                eval_split = str(overrides.get("eval_split", "test"))
                donor_split = str(overrides.get("donor_split", "train"))
                retained_eval_graphs = max(
                    int(n_pca_graphs), max(graph_indices) + 1
                )
                split_limits = {"train": 1, "val": 1, "test": 1}
                split_limits[eval_split] = retained_eval_graphs
                split_limits[donor_split] = max(
                    split_limits.get(donor_split, 1), 1
                )
                runtime_state["value"] = build_verified_grit_figure_runtime(
                    current_artifact,
                    current_model_record,
                    current_protocol,
                    runtime_output_dir=(
                        cache_dir / current_task / f"seed_{current_seed}" / "runtime"
                    ),
                    analysis_split_limits=split_limits,
                    require_protocol_match=False,
                    require_adapter_match=False,
                )
                runtime_state["checkpoint_sha256"] = target_checkpoint
            elif runtime_state.get("checkpoint_sha256") != target_checkpoint:
                if verbose:
                    print(
                        f"[special-heads:{current_task}/seed_{current_seed}] "
                        "reusing architecture loaders and swapping checkpoint",
                        flush=True,
                    )
                runtime_state["value"] = retarget_verified_grit_figure_runtime(
                    runtime_state["value"],
                    current_artifact,
                    current_model_record,
                    current_protocol,
                    require_protocol_match=False,
                    require_adapter_match=False,
                )
                runtime_state["checkpoint_sha256"] = target_checkpoint
            return runtime_state["value"]

        attention_contract = {
            **base_contract,
            "diagnostic": "controlled_attention_examples_v1",
            "graph_indices": list(graph_indices),
            "include_virtual_attention": True,
        }
        pca_contract = {
            **base_contract,
            "diagnostic": "chemistry_labelled_routed_output_pca_v1",
            "n_graphs": int(n_pca_graphs),
            "focus_mass": 0.75,
            "diffuse_threshold": 0.35,
            "molecular_receivers_only": True,
        }

        def attention_compute(current_heads=heads, current_runtime=runtime):
            return collect_attention_examples(
                current_runtime(),
                graph_indices=graph_indices,
                heads=current_heads,
                include_virtual_attention=True,
            )

        def pca_compute(current_heads=heads, current_runtime=runtime):
            return compute_av_pca_inputs_many(
                current_runtime(),
                heads=tuple(current_heads.values()),
                n_graphs=int(n_pca_graphs),
                focus_mass=0.75,
                diffuse_threshold=0.35,
                verbose=verbose,
            )

        if compute_missing:
            attention_payload, attention_path, attention_hit = cache.load_or_compute(
                "controlled-head-attention", attention_contract, attention_compute, force=force
            )
            pca_payload, pca_path, pca_hit = cache.load_or_compute(
                "controlled-head-pca", pca_contract, pca_compute, force=force
            )
        else:
            attention_cached = cache.load("controlled-head-attention", attention_contract)
            pca_cached = cache.load("controlled-head-pca", pca_contract)
            if attention_cached is None or pca_cached is None:
                raise FileNotFoundError(f"missing supplemental cache for {task}/seed_{seed}")
            attention_payload, attention_path = attention_cached
            pca_payload, pca_path = pca_cached
            attention_hit = pca_hit = True
        if verbose:
            print(
                f"[special-heads-cache:{task}/seed_{seed}] "
                f"attention={'hit' if attention_hit else 'computed'}; "
                f"pca={'hit' if pca_hit else 'computed'}",
                flush=True,
            )
        cache_records.append(
            {
                "task": task,
                "seed": seed,
                "attention_cache": str(attention_path),
                "attention_cache_hit": bool(attention_hit),
                "pca_cache": str(pca_path),
                "pca_cache_hit": bool(pca_hit),
                "checkpoint": str(model_record["checkpoint"]),
                "checkpoint_sha256": checkpoint_sha256(model_record["checkpoint"]),
            }
        )
        profiles_by_role = {
            str(row["role"]): head_distance_profiles(
                artifact.value, (int(row["layer"]), int(row["head"]))
            )
            for row in group
        }
        # Everything required from the canonical score and live model is now
        # represented by small CPU payloads.  Do not overlap either large
        # object with high-resolution PDF rendering.
        del attention_compute, pca_compute, runtime, artifact
        _release_figure_runtime(active_runtime.get("value"))
        active_runtime.clear()
        identity = figure_identity(task)
        display_attention = {**dict(attention_payload), **identity}
        for row in group:
            role = str(row["role"])
            head = (int(row["layer"]), int(row["head"]))
            stem_prefix = f"{task}_seed_{seed}_{role}_L{head[0]}_H{head[1]}"
            metadata = _record_metadata(
                row,
                score_path=str(score_path),
                attention_cache=str(attention_path),
                pca_cache=str(pca_path),
                graph_indices=list(graph_indices),
                pca_graphs=int(n_pca_graphs),
            )
            examples_by_index = {
                int(example["dataset_index"]): example
                for example in display_attention["examples"]
            }
            examples = [
                examples_by_index[index] for index in render_graph_indices
            ]
            for page_start in range(0, len(examples), int(examples_per_page)):
                page_end = min(page_start + int(examples_per_page), len(examples))
                page_payload = {
                    **display_attention,
                    "examples": examples[page_start:page_end],
                }
                figure = plot_attention_grid_publication(
                    page_payload,
                    attention_key=role,
                    attention_family=str(row["display_family"]),
                    head=head,
                    title_label=str(row["role_label"]),
                    net_d_rel=float(row["selectivity"]),
                    net_joint_sensitivity=float(row["joint_sensitivity"]),
                    head_ablation_impact=float(row["head_ablation_impact"]),
                )
                paths = save_figure_bundle(
                    figure,
                    figures_dir,
                    f"{stem_prefix}_attention_{page_start + 1:02d}_{page_end:02d}",
                    metadata={
                        **metadata,
                        "figure": "attention_examples",
                        "page_graph_indices": list(
                            render_graph_indices[page_start:page_end]
                        ),
                    },
                    dpi=SPECIAL_HEAD_PREVIEW_DPI,
                    pdf_dpi=SPECIAL_HEAD_PDF_RASTER_DPI,
                    supersede_stem_globs=(
                        (
                            f"{task}_seed_*_{role}_L*_H*_attention_"
                            f"{page_start + 1:02d}_{page_end:02d}.*"
                        ),
                    ),
                )
                _release_rendered_figure(figure)
                outputs.append(
                    {
                        "task": task,
                        "seed": seed,
                        "role": role,
                        "figure": "attention",
                        **{key: str(value) for key, value in paths.items()},
                    }
                )

            pca_head_payload = dict(pca_payload[f"{head[0]}:{head[1]}"])
            pca_head_payload.update(identity)
            figure = plot_av_pca_publication(
                pca_head_payload,
                title_label=str(row["role_label"]),
                d_rel=float(row["selectivity"]),
                joint_sensitivity=float(row["joint_sensitivity"]),
                head_ablation_impact=float(row["head_ablation_impact"]),
            )
            _scale_figure_text(figure, factor=1.25)
            paths = save_figure_bundle(
                figure,
                figures_dir,
                f"{stem_prefix}_routed_output_pca",
                metadata={**metadata, "figure": "routed_output_pca"},
                dpi=SPECIAL_HEAD_PREVIEW_DPI,
                pdf_dpi=SPECIAL_HEAD_PDF_RASTER_DPI,
                supersede_stem_globs=(
                    f"{task}_seed_*_{role}_L*_H*_routed_output_pca.*",
                ),
            )
            _release_rendered_figure(figure)
            outputs.append(
                {
                    "task": task,
                    "seed": seed,
                    "role": role,
                    "figure": "pca",
                    **{key: str(value) for key, value in paths.items()},
                }
            )

            profiles = profiles_by_role[role]
            figure = _plot_distance_profiles(
                profiles,
                display_title=identity["display_title"],
                role_label=str(row["role_label"]),
                head=head,
                d_rel=float(row["selectivity"]),
                joint_sensitivity=float(row["joint_sensitivity"]),
                head_ablation_impact=float(row["head_ablation_impact"]),
            )
            _scale_figure_text(figure, factor=1.25)
            paths = save_figure_bundle(
                figure,
                figures_dir,
                f"{stem_prefix}_distance_profiles",
                metadata={
                    **metadata,
                    "figure": "distance_profiles",
                    "distance_labels": list(profiles["labels"]),
                },
                dpi=SPECIAL_HEAD_PREVIEW_DPI,
                pdf_dpi=SPECIAL_HEAD_PDF_RASTER_DPI,
                supersede_stem_globs=(
                    f"{task}_seed_*_{role}_L*_H*_distance_profiles.*",
                ),
            )
            _release_rendered_figure(figure)
            outputs.append(
                {
                    "task": task,
                    "seed": seed,
                    "role": role,
                    "figure": "distance",
                    **{key: str(value) for key, value in paths.items()},
                }
            )
            for example in display_attention["examples"]:
                example["attention"].pop(role, None)
            pca_payload.pop(f"{head[0]}:{head[1]}", None)
            profiles_by_role.pop(role, None)
            del pca_head_payload, examples_by_index, examples, profiles
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except (OSError, AttributeError):
                pass
        # Cached arrays and rendered figures can otherwise survive until the
        # next checkpoint group in notebook runtimes.  Release the complete
        # group, including its model and loaders, before proceeding.
        del (
            attention_payload,
            pca_payload,
            display_attention,
            profiles_by_role,
        )
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    _release_figure_runtime(active_runtime.get("value"))
    active_runtime.clear()

    completion = validate_special_head_outputs(
        selected,
        outputs,
        tasks=spec.tasks,
        render_graph_count=len(render_graph_indices),
        examples_per_page=int(examples_per_page),
    )
    _write_csv(analysis_dir / "completion.csv", completion)

    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "figure_style_version": FIGURE_STYLE_VERSION,
        "dataset": spec.name,
        "seeds": [int(seed) for seed in seeds],
        "models": list(spec.tasks),
        "graph_indices": list(graph_indices),
        "render_graph_indices": list(render_graph_indices),
        "n_pca_graphs": int(n_pca_graphs),
        "examples_per_page": int(examples_per_page),
        "selection_rule": {
            "semantic": "maximum finite D_rel across all heads and seeds; ties prefer larger J",
            "structural": "minimum finite D_rel across all heads and seeds; ties prefer larger J",
            "highest_joint": "largest J not already selected",
            "generalist_colour_bound": float(generalist_max_abs_drel),
        },
        "selected_heads": selected,
        "caches": cache_records,
        "outputs": outputs,
        "completion": completion,
        "warnings": warnings,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
    )
    return {**manifest, "manifest_path": str(manifest_path)}


__all__ = [
    "DEFAULT_GRAPH_INDICES",
    "generate_special_head_analysis",
    "head_distance_profiles",
    "select_special_heads_across_seeds",
    "validate_special_head_outputs",
]
