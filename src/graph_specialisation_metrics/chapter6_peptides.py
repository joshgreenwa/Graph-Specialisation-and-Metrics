"""Cache-only Chapter 6 figures for Peptides-func and Peptides-struct.

The Peptides graphs are substantially larger than ZINC and QM9.  This module
therefore streams one consolidated score artifact and one carriage artifact at
a time, never reconstructs a model, and emits only the compact figure set used
in the dissertation.
"""

from __future__ import annotations

import csv
import ctypes
import gc
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .chapter6_multiseed import (
    CHANNELS,
    SEEDS,
    DatasetSpec,
    _joint_normalised_response_rows,
    _plot_distance_alignment,
    _plot_expected_graph_distance,
    _plot_overlaid_final_state_response,
    _plot_seed_mean_reach_gap_heatmap,
    _plot_specialisation_landscapes,
    _plot_specialisation_selectivity_landscapes,
    _plot_vnode_allocation,
    _save_figure,
    _scale_figure_text,
    _write_csv,
    layer_organisation_rows,
    vnode_allocation_rows,
)
from .chapter6_spatial_explorer import _load_payload, head_metrics, inventory, load_models
from .methodology.bootstrap import trimmed_mean

ANALYSIS_VERSION = "chapter6-peptides-multiseed-v1"


@dataclass(frozen=True)
class DistanceBin:
    label: str
    lower: int
    upper: int | None

    def contains(self, distance: int) -> bool:
        return distance >= self.lower and (self.upper is None or distance <= self.upper)


# Nine molecular groups retain single-hop resolution locally and compress only
# the long tail.  The virtual node is added as a separate non-distance category.
PEPTIDE_DISTANCE_BINS = (
    DistanceBin("0", 0, 0),
    DistanceBin("1", 1, 1),
    DistanceBin("2", 2, 2),
    DistanceBin("3", 3, 3),
    DistanceBin("4", 4, 4),
    DistanceBin("5-6", 5, 6),
    DistanceBin("7-9", 7, 9),
    DistanceBin("10-14", 10, 14),
    DistanceBin("15+", 15, None),
)

FIGURE_FILENAMES = (
    "01b_expected_graph_distance.pdf",
    "02_specialisation_landscapes.pdf",
    "02b_specialisation_landscapes_selectivity.pdf",
    "06_virtual_node_allocation.pdf",
    "05_semantic_structural_distance_alignment.pdf",
    "03b_semantic_attention_score_reach_gap_seed_mean.pdf",
    "04b_structural_attention_score_reach_gap_seed_mean.pdf",
    "09c_final_state_response_by_model.pdf",
    "09d_normalised_final_state_response_by_model.pdf",
    "10_expected_final_state_response_distance.pdf",
)


def dataset_spec(dataset: str) -> DatasetSpec:
    """Return the five trained architectures for one Peptides benchmark."""

    key = str(dataset).strip().lower().replace("-", "_")
    aliases = {
        "func": "peptides_func",
        "peptides_func": "peptides_func",
        "struct": "peptides_struct",
        "peptides_struct": "peptides_struct",
    }
    try:
        prefix = aliases[key]
    except KeyError as error:
        raise ValueError(
            "dataset must be 'peptides_func' or 'peptides_struct'"
        ) from error
    tasks = (
        f"{prefix}_1hop",
        f"{prefix}_1hop_vnode",
        f"{prefix}_2hop",
        f"{prefix}_2hop_vnode",
        f"{prefix}_dense",
    )
    labels = {
        tasks[0]: "1-hop",
        tasks[1]: "1-hop + VNode",
        tasks[2]: "2-hop",
        tasks[3]: "2-hop + VNode",
        tasks[4]: "Dense GRIT",
    }
    return DatasetSpec(name=prefix.replace("_", "-"), tasks=tasks, labels=labels)


def cache_inventory(
    canonical_root: Path,
    *,
    dataset: str,
    seeds: Sequence[int] = SEEDS,
) -> list[dict[str, Any]]:
    spec = dataset_spec(dataset)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rows.extend(inventory((Path(canonical_root),), spec.tasks, seed=int(seed)))
    return rows


def _release_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _mean_range(values: Sequence[float]) -> tuple[float, float, float, int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan"), float("nan"), float("nan"), 0
    return (
        float(np.mean(finite)),
        float(np.min(finite)),
        float(np.max(finite)),
        int(finite.size),
    )


def _distance_group(distance: float, carrier_kind: Any) -> tuple[str | None, float | None]:
    if np.isfinite(distance) and distance >= 0 and float(distance).is_integer():
        integer = int(distance)
        for group in PEPTIDE_DISTANCE_BINS:
            if group.contains(integer):
                return group.label, float(integer)
        return None, None
    kind = str(carrier_kind).strip().lower().replace("_", " ")
    if "virtual" in kind:
        return "virtual", None
    return None, None


def peptide_carriage_profile(
    carriage: Mapping[str, Any],
    channel: str,
    *,
    minimum_graphs: int = 10,
    minimum_pairs: int = 50,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Return binned raw response and response-weighted distance per carrier.

    The estimator matches the graph-balanced donor/source aggregation used by
    the molecular analysis, while accumulating all nine bins in one pass.
    """

    rows = carriage["channels"][channel]["pairs"]
    events: dict[tuple[str, int, int, int], list[float]] = {}
    graphs: dict[str, set[int]] = {}
    pairs: dict[str, set[tuple[int, int, int]]] = {}
    observed_specials: set[str] = set()
    for row in rows:
        response = float(row["F_sens"])
        if not np.isfinite(response):
            continue
        label, numeric_distance = _distance_group(
            float(row["distance"]), row.get("carrier_kind", "molecular_node")
        )
        if label is None:
            continue
        if numeric_distance is None:
            observed_specials.add(label)
        graph = int(row["graph_id"])
        source = int(row["source"])
        donor = int(row["donor"])
        carrier = int(row["carrier"])
        graph_set = graphs.setdefault(label, set())
        if len(graph_set) < int(minimum_graphs):
            graph_set.add(graph)
        pair_set = pairs.setdefault(label, set())
        if len(pair_set) < int(minimum_pairs):
            pair_set.add((graph, carrier, source))
        accumulator = events.setdefault((label, graph, source, donor), [0.0, 0.0, 0.0])
        accumulator[0] += response
        accumulator[1] += 1.0
        if numeric_distance is not None:
            accumulator[2] += response * numeric_distance

    by_source: dict[tuple[str, int, int], list[tuple[float, float, float]]] = {}
    for (label, graph, source, _donor), (response, count, weighted) in events.items():
        by_source.setdefault((label, graph, source), []).append(
            (response, count, weighted)
        )
    del events

    by_graph: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
    for (label, graph, _source), donor_values in by_source.items():
        by_graph.setdefault((label, graph), []).append(
            (
                float(np.mean([value[0] for value in donor_values])),
                float(np.mean([value[1] for value in donor_values])),
                float(np.mean([value[2] for value in donor_values])),
            )
        )
    del by_source

    labels = tuple(group.label for group in PEPTIDE_DISTANCE_BINS) + tuple(
        sorted(observed_specials)
    )
    estimates = np.full(len(labels), np.nan)
    weighted_estimates = np.full(len(labels), np.nan)
    for position, label in enumerate(labels):
        if len(graphs.get(label, ())) < int(minimum_graphs):
            continue
        if len(pairs.get(label, ())) < int(minimum_pairs):
            continue
        graph_response: list[float] = []
        graph_weighted: list[float] = []
        for (candidate_label, _graph), source_values in by_graph.items():
            if candidate_label != label:
                continue
            count = float(np.sum([value[1] for value in source_values]))
            if count <= 0:
                continue
            graph_response.append(
                float(np.sum([value[0] for value in source_values])) / count
            )
            graph_weighted.append(
                float(np.sum([value[2] for value in source_values])) / count
            )
        if graph_response:
            estimates[position] = float(trimmed_mean(graph_response, 0.20, axis=0))
        if label != "virtual" and graph_weighted:
            weighted_estimates[position] = float(
                trimmed_mean(graph_weighted, 0.20, axis=0)
            )
    return labels, estimates, weighted_estimates


def _peptide_peak(label: Any) -> tuple[str, int | None]:
    text = str(label).strip().lower().replace("_", " ")
    if "virtual" in text:
        return "virtual", None
    try:
        distance = int(float(text))
    except ValueError:
        return text, None
    for index, group in enumerate(PEPTIDE_DISTANCE_BINS):
        if group.contains(distance):
            return group.label, index
    return text, None


def peptide_alignment_summary_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarise head-distance agreement using the Peptides display bins."""

    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if np.isfinite(float(row.get("semantic_expected_distance", np.nan))) and np.isfinite(
            float(row.get("structural_expected_distance", np.nan))
        ):
            by_task.setdefault(str(row["task"]), []).append(row)
    output: list[dict[str, Any]] = []
    for task, group in sorted(by_task.items()):
        semantic = np.asarray([float(row["semantic_expected_distance"]) for row in group])
        structural = np.asarray([float(row["structural_expected_distance"]) for row in group])
        rho = float("nan")
        if (
            len(group) >= 3
            and float(np.std(semantic)) > 1.0e-12
            and float(np.std(structural)) > 1.0e-12
        ):
            from scipy.stats import spearmanr

            rho = float(spearmanr(semantic, structural).statistic)
        same: list[bool] = []
        adjacent: list[bool] = []
        for row in group:
            left, left_order = _peptide_peak(row.get("semantic_peak_distance", ""))
            right, right_order = _peptide_peak(row.get("structural_peak_distance", ""))
            same.append(left == right)
            adjacent.append(
                left == right
                if left_order is None or right_order is None
                else abs(left_order - right_order) <= 1
            )
        output.append(
            {
                "task": task,
                "heads": len(group),
                "eligibility": "finite semantic and structural distance profiles",
                "spearman_rho": rho,
                "same_peak_fraction": float(np.mean(same)),
                "same_or_adjacent_peak_fraction": float(np.mean(adjacent)),
            }
        )
    return output


def _aggregate_response_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["task"]), str(row["channel"]), str(row["distance"])), []
        ).append(float(row["value"]))
    output: list[dict[str, Any]] = []
    for (task, channel, distance), values in sorted(groups.items()):
        mean, low, high, seeds = _mean_range(values)
        output.append(
            {
                "task": task,
                "channel": channel,
                "variant": "raw",
                "distance": distance,
                "seeds": seeds,
                "response_mean": mean,
                "response_min": low,
                "response_max": high,
            }
        )
    return output + _joint_normalised_response_rows(output)


def _aggregate_expected_distance_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), str(row["channel"])), []).append(
            float(row["expected_distance"])
        )
    output: list[dict[str, Any]] = []
    for (task, channel), values in sorted(groups.items()):
        mean, low, high, seeds = _mean_range(values)
        output.append(
            {
                "task": task,
                "channel": channel,
                "seeds": seeds,
                "expected_distance_mean": mean,
                "expected_distance_min": low,
                "expected_distance_max": high,
            }
        )
    return output


def _aggregate_vnode_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[float]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["task"]), int(row["layer"]), str(row["source"])), []
        ).append(float(row["virtual_share_mean"]))
    output: list[dict[str, Any]] = []
    for (task, layer, source), values in sorted(groups.items()):
        mean, low, high, seeds = _mean_range(values)
        output.append(
            {
                "task": task,
                "layer": layer,
                "source": source,
                "seeds": seeds,
                "virtual_share_mean": mean,
                "virtual_share_min": low,
                "virtual_share_max": high,
            }
        )
    return output


def _plot_expected_final_state_response_distance(
    rows: Sequence[Mapping[str, Any]],
    spec: DatasetSpec,
    figures_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    x = np.arange(len(spec.tasks), dtype=np.float64)
    width = 0.36
    figure, axis = plt.subplots(figsize=(11.5, 4.6), constrained_layout=True)
    styles = {
        "semantic": ("#0072B2", -width / 2, "semantic"),
        "structural": ("#D55E00", width / 2, "structural"),
    }
    for channel, (colour, offset, label) in styles.items():
        lookup = {
            str(row["task"]): row for row in rows if str(row["channel"]) == channel
        }
        means = np.asarray(
            [float(lookup[task]["expected_distance_mean"]) for task in spec.tasks]
        )
        lows = np.asarray(
            [float(lookup[task]["expected_distance_min"]) for task in spec.tasks]
        )
        highs = np.asarray(
            [float(lookup[task]["expected_distance_max"]) for task in spec.tasks]
        )
        errors = np.maximum(np.vstack((means - lows, highs - means)), 0.0)
        axis.bar(
            x + offset,
            means,
            width,
            yerr=errors,
            color=colour,
            alpha=0.88,
            capsize=3,
            linewidth=0,
            label=label,
        )
    axis.set_xticks(x, [spec.labels[task] for task in spec.tasks])
    axis.set_ylabel("expected graph distance")
    axis.set_xlabel("architecture")
    axis.grid(axis="y", alpha=0.16, linewidth=0.6)
    axis.legend(frameon=False)
    figure.suptitle(
        f"{spec.name.upper()}: expected distance of final-state response"
    )
    _scale_figure_text(figure)
    return _save_figure(
        figure,
        figures_dir,
        "10_expected_final_state_response_distance",
    )


def _load_rows_streaming(
    canonical_root: Path,
    spec: DatasetSpec,
    seeds: Sequence[int],
    *,
    minimum_graphs: int,
    minimum_pairs: int,
    verbose: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    int,
]:
    head_rows: list[dict[str, Any]] = []
    vnode_seed_rows: list[dict[str, Any]] = []
    response_seed_rows: list[dict[str, Any]] = []
    expected_seed_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    runs_loaded = 0
    for seed in seeds:
        for task in spec.tasks:
            models, model_warnings = load_models(
                (canonical_root,), (task,), seed=int(seed), load_carriage=False
            )
            warnings.extend(model_warnings)
            if len(models) != 1:
                raise FileNotFoundError(
                    f"expected one usable score cache for {task}/seed_{seed}; found {len(models)}"
                )
            model = models[0]
            head_rows.extend(
                {"seed": int(seed), **row} for row in head_metrics((model,))
            )
            vnode_seed_rows.extend(vnode_allocation_rows((model,)))
            models.clear()
            del model
            _release_memory()

            carriage_path = (
                canonical_root
                / task
                / f"seed_{int(seed)}"
                / "cache"
                / "carriage"
                / "fields.pt"
            )
            carriage, _metadata = _load_payload(carriage_path)
            for channel in CHANNELS:
                labels, response, distance_weighted_response = peptide_carriage_profile(
                    carriage,
                    channel,
                    minimum_graphs=minimum_graphs,
                    minimum_pairs=minimum_pairs,
                )
                for label, value in zip(labels, response):
                    response_seed_rows.append(
                        {
                            "task": task,
                            "seed": int(seed),
                            "channel": channel,
                            "distance": label,
                            "value": float(value),
                        }
                    )
                finite = (
                    np.isfinite(response)
                    & np.isfinite(distance_weighted_response)
                    & (response >= 0.0)
                )
                denominator = float(np.sum(response[finite])) if np.any(finite) else 0.0
                expected = (
                    float(np.sum(distance_weighted_response[finite]) / denominator)
                    if denominator > 1.0e-12
                    else float("nan")
                )
                expected_seed_rows.append(
                    {
                        "task": task,
                        "seed": int(seed),
                        "channel": channel,
                        "expected_distance": expected,
                    }
                )
            del carriage
            _release_memory()
            runs_loaded += 1
            if verbose:
                print(
                    f"[chapter6-peptides] loaded {task}/seed_{seed}; "
                    f"completed {runs_loaded}/{len(spec.tasks) * len(seeds)}",
                    flush=True,
                )
    return (
        head_rows,
        _aggregate_vnode_rows(vnode_seed_rows),
        _aggregate_response_rows(response_seed_rows),
        _aggregate_expected_distance_rows(expected_seed_rows),
        warnings,
        runs_loaded,
    )


def _prune_figure_directory(figures_dir: Path) -> None:
    allowed = set(FIGURE_FILENAMES)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for path in figures_dir.iterdir():
        if path.is_file() and path.name not in allowed:
            path.unlink()


def run(
    canonical_root: Path,
    output_dir: Path,
    *,
    dataset: str,
    seeds: Sequence[int] = SEEDS,
    minimum_graphs: int = 10,
    minimum_pairs: int = 50,
    strict_inventory: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate exactly the dissertation Peptides figure subset from caches."""

    canonical_root = Path(canonical_root)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    spec = dataset_spec(dataset)
    inventory_rows = cache_inventory(canonical_root, dataset=dataset, seeds=seeds)
    missing = [
        row
        for row in inventory_rows
        if not bool(row.get("score_exists")) or not bool(row.get("carriage_exists"))
    ]
    if strict_inventory and missing:
        detail = ", ".join(
            f"{row['task']}/seed_{row['seed']} "
            f"(score={row.get('score_exists')}, carriage={row.get('carriage_exists')})"
            for row in missing
        )
        raise FileNotFoundError(f"incomplete Peptides cache inventory: {detail}")

    (
        head_rows,
        vnode_rows,
        response_rows,
        expected_response_rows,
        warnings,
        runs_loaded,
    ) = _load_rows_streaming(
        canonical_root,
        spec,
        seeds,
        minimum_graphs=minimum_graphs,
        minimum_pairs=minimum_pairs,
        verbose=verbose,
    )
    expected_runs = len(spec.tasks) * len(seeds)
    if runs_loaded != expected_runs:
        raise RuntimeError(f"loaded {runs_loaded}/{expected_runs} expected Peptides runs")

    organisation_rows = layer_organisation_rows(head_rows)
    alignment_rows = peptide_alignment_summary_rows(head_rows)
    tables = {
        "cache_inventory.csv": inventory_rows,
        "head_metrics.csv": head_rows,
        "layer_spatial_organisation.csv": organisation_rows,
        "semantic_structural_alignment.csv": alignment_rows,
        "vnode_allocation.csv": vnode_rows,
        "final_state_response_variants.csv": response_rows,
        "final_state_response_expected_distance.csv": expected_response_rows,
    }
    for filename, rows in tables.items():
        _write_csv(output_dir / filename, rows)

    _prune_figure_directory(figures_dir)
    previous_pdf_only = os.environ.get("CHAPTER6_PDF_ONLY")
    os.environ["CHAPTER6_PDF_ONLY"] = "1"
    try:
        figures: list[Path] = []
        figures.extend(_plot_expected_graph_distance(organisation_rows, spec, figures_dir))
        figures.extend(_plot_specialisation_landscapes(head_rows, spec, figures_dir))
        figures.extend(
            _plot_specialisation_selectivity_landscapes(head_rows, spec, figures_dir)
        )
        figures.extend(_plot_vnode_allocation(vnode_rows, spec, figures_dir))
        figures.extend(_plot_distance_alignment(head_rows, alignment_rows, spec, figures_dir))
        figures.extend(
            _plot_seed_mean_reach_gap_heatmap(
                head_rows,
                spec,
                seeds,
                figures_dir,
                channel="semantic",
                stem="03b_semantic_attention_score_reach_gap_seed_mean",
            )
        )
        figures.extend(
            _plot_seed_mean_reach_gap_heatmap(
                head_rows,
                spec,
                seeds,
                figures_dir,
                channel="structural",
                stem="04b_structural_attention_score_reach_gap_seed_mean",
            )
        )
        figures.extend(
            _plot_overlaid_final_state_response(
                response_rows,
                spec,
                figures_dir,
                variant="raw",
                stem="09c_final_state_response_by_model",
            )
        )
        figures.extend(
            _plot_overlaid_final_state_response(
                response_rows,
                spec,
                figures_dir,
                variant="normalised",
                stem="09d_normalised_final_state_response_by_model",
            )
        )
        figures.extend(
            _plot_expected_final_state_response_distance(
                expected_response_rows, spec, figures_dir
            )
        )
    finally:
        if previous_pdf_only is None:
            os.environ.pop("CHAPTER6_PDF_ONLY", None)
        else:
            os.environ["CHAPTER6_PDF_ONLY"] = previous_pdf_only

    _prune_figure_directory(figures_dir)
    actual = {path.name for path in figures_dir.iterdir() if path.is_file()}
    expected = set(FIGURE_FILENAMES)
    if actual != expected:
        raise RuntimeError(
            f"Peptides figure contract mismatch; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "dataset": spec.name,
        "canonical_root": str(canonical_root),
        "output_dir": str(output_dir),
        "tasks": list(spec.tasks),
        "seeds": [int(seed) for seed in seeds],
        "runs_loaded": runs_loaded,
        "distance_bins": [group.label for group in PEPTIDE_DISTANCE_BINS],
        "virtual_node_is_separate": True,
        "minimum_graphs": int(minimum_graphs),
        "minimum_pairs": int(minimum_pairs),
        "warnings": warnings,
        "figures": [str(figures_dir / filename) for filename in FIGURE_FILENAMES],
        "tables": [str(output_dir / filename) for filename in tables],
        "interpretation": {
            "score_distance": "exact molecular graph distance; VNode excluded",
            "carriage_distance_bins": (
                "nine fixed molecular groups with exact response-weighted distance "
                "retained for the expected-distance summary"
            ),
            "normalised_final_state_response": (
                "one denominator per architecture shared across semantic and structural "
                "channels and all molecular/VNode groups"
            ),
            "expected_final_state_response_distance": (
                "response-weighted molecular graph distance; VNode excluded"
            ),
            "head_identity": (
                "seed-mean reach-gap heatmaps average layer/head coordinates as a "
                "model-level summary, not cross-seed head matching"
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def read_manifest(output_dir: Path) -> dict[str, Any] | None:
    path = Path(output_dir) / "manifest.json"
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if manifest.get("analysis_version") != ANALYSIS_VERSION:
        return None
    figures = [Path(path) for path in manifest.get("figures", ())]
    tables = [Path(path) for path in manifest.get("tables", ())]
    if not figures or not tables or not all(path.is_file() for path in (*figures, *tables)):
        return None
    if {path.name for path in figures} != set(FIGURE_FILENAMES):
        return None
    return manifest


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
