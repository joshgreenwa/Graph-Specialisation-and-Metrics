"""Graph-derived RRWP performance gap with score-distance measurements.

Each graph contains a cycle candidate and a long-chain candidate carrying
independent semantic values.  The target is the cycle value.  A leaf attached
next to one candidate creates a one-step local-pair RRWP clue that is correct
with probability 0.75.  Higher-order diagonal return probabilities provide a
substantially more reliable clue to which candidate lies on the cycle.

Local and global models are parameter-matched linear structural gates: the
local model receives the one-step pair summary and zeroed higher-order channels,
whereas the global model receives the same full-width vector through P^16.
Both are fitted end-to-end for semantic prediction.  A diagnostic
confidence-matched comparison preserves each model's selected candidate while
equalising absolute gate logits; this separates clue correctness from response
amplitude and softmax saturation.

Canonical projected-transport masses are measured for semantic payload and
structural-footprint donor events.  A fixed two-step lazy random-walk carrier
map supplies a non-trivial source-to-carrier distance decomposition while
holding communication geometry identical across RRWP horizons.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

from .local_messages_nonlocal_structure import (
    _transition_matrix,
    make_cycle_chain_graph,
    rrwp_powers,
)

plt.switch_backend("Agg")

BLUE = "#4477AA"
ORANGE = "#EE7733"
GREEN = "#228833"
PURPLE = "#AA3377"
RED = "#CC6677"
GREY = "#777777"
LIGHT_GREY = "#D7D7D7"
TEXT = "#222222"

SCOPE_ORDER = ("local", "global")
CALIBRATION_ORDER = ("free", "matched")
SCOPE_COLOURS = {"local": ORANGE, "global": BLUE}


@dataclass(frozen=True)
class Config:
    output_dir: Path
    cycle_sizes: tuple[int, ...] = (6, 8, 10)
    max_horizon: int = 16
    local_clue_reliability: float = 0.75
    train_examples: int = 8_192
    test_examples: int = 8_192
    seeds: tuple[int, ...] = tuple(range(8))
    learning_rate: float = 5.0e-2
    training_steps: int = 5_000
    ridge: float = 1.0e-2
    data_seed: int = 71_903

    def validate(self) -> None:
        if not self.cycle_sizes or tuple(sorted(set(self.cycle_sizes))) != self.cycle_sizes:
            raise ValueError("cycle_sizes must be unique and increasing")
        if max(self.cycle_sizes) > self.max_horizon:
            raise ValueError("max_horizon must cover every cycle size")
        if not 0.5 < self.local_clue_reliability < 1.0:
            raise ValueError("local_clue_reliability must lie in (0.5,1)")
        if min(self.train_examples, self.test_examples, self.training_steps) < 1:
            raise ValueError("sample sizes and training_steps must be positive")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be non-empty and unique")
        if self.learning_rate <= 0 or self.ridge < 0:
            raise ValueError("learning_rate must be positive and ridge non-negative")


@dataclass(frozen=True)
class GraphType:
    cycle_size: int
    local_clue_correct: bool
    candidate_features: np.ndarray
    carrier_profiles: np.ndarray


@dataclass(frozen=True)
class Dataset:
    type_index: np.ndarray
    values: np.ndarray

    @property
    def target(self) -> np.ndarray:
        return self.values[:, 0]


def _candidate_carrier_profile(
    graph: nx.Graph,
    candidate: int,
    *,
    maximum_distance: int = 2,
) -> np.ndarray:
    transition, nodes = _transition_matrix(graph)
    position = {node: index for index, node in enumerate(nodes)}
    identity = np.eye(len(nodes), dtype=np.float64)
    lazy = 0.50 * identity + 0.30 * transition + 0.20 * (transition @ transition)
    distances = nx.single_source_shortest_path_length(
        graph,
        int(candidate),
        cutoff=int(maximum_distance),
    )
    profile = np.zeros(int(maximum_distance) + 1, dtype=np.float64)
    row = lazy[position[int(candidate)]]
    for node, distance in distances.items():
        if int(distance) <= int(maximum_distance):
            profile[int(distance)] += float(row[position[int(node)]])
    if not np.isclose(profile.sum(), 1.0, atol=1.0e-12):
        raise RuntimeError("registered two-step carrier profile does not sum to one")
    return profile


def build_graph_types(config: Config) -> tuple[tuple[GraphType, ...], np.ndarray, np.ndarray]:
    raw_types: list[dict[str, Any]] = []
    for cycle_size in config.cycle_sizes:
        for local_clue_correct in (False, True):
            graph, cycle_candidate, chain_candidate = make_cycle_chain_graph(
                cycle_size,
                max_horizon=config.max_horizon,
            )
            marked = cycle_candidate if local_clue_correct else chain_candidate
            marked_neighbour = min(int(node) for node in graph.neighbors(marked))
            graph.add_edge(marked_neighbour, max(int(node) for node in graph.nodes) + 1)
            powers, nodes = rrwp_powers(graph, config.max_horizon)
            transition, transition_nodes = _transition_matrix(graph)
            if nodes != transition_nodes:
                raise RuntimeError("RRWP and transition node orders differ")
            position = {node: index for index, node in enumerate(nodes)}
            features: list[list[float]] = []
            profiles: list[np.ndarray] = []
            for candidate in (cycle_candidate, chain_candidate):
                incoming = [
                    transition[position[int(neighbour)], position[int(candidate)]]
                    for neighbour in graph.neighbors(candidate)
                ]
                local_pair_summary = 0.5 - min(incoming)
                diagonal_rrwp = powers[:, position[candidate], position[candidate]]
                features.append([float(local_pair_summary), *diagonal_rrwp.tolist()])
                profiles.append(_candidate_carrier_profile(graph, candidate))
            raw_types.append(
                {
                    "cycle_size": int(cycle_size),
                    "local_clue_correct": bool(local_clue_correct),
                    "candidate_features": np.asarray(features, dtype=np.float64),
                    "carrier_profiles": np.stack(profiles),
                }
            )

    all_features = np.concatenate(
        [item["candidate_features"] for item in raw_types],
        axis=0,
    )
    feature_mean = np.mean(all_features, axis=0)
    feature_scale = np.std(all_features, axis=0)
    feature_scale[feature_scale < 1.0e-12] = 1.0
    registered = tuple(
        GraphType(
            cycle_size=item["cycle_size"],
            local_clue_correct=item["local_clue_correct"],
            candidate_features=(item["candidate_features"] - feature_mean) / feature_scale,
            carrier_profiles=item["carrier_profiles"],
        )
        for item in raw_types
    )
    return registered, feature_mean, feature_scale


def generate_dataset(config: Config, *, count: int, seed: int) -> Dataset:
    rng = np.random.default_rng(int(seed))
    size_index = rng.integers(0, len(config.cycle_sizes), size=int(count))
    clue_correct = rng.random(int(count)) < float(config.local_clue_reliability)
    type_index = 2 * size_index + clue_correct.astype(np.int64)
    return Dataset(
        type_index=type_index,
        values=rng.standard_normal((int(count), 2)),
    )


def _feature_differences(
    graph_types: Sequence[GraphType],
    type_index: np.ndarray,
    *,
    scope: str,
) -> np.ndarray:
    features = np.stack([item.candidate_features for item in graph_types])
    difference = features[type_index, 0] - features[type_index, 1]
    if scope == "local":
        difference = np.array(difference, copy=True)
        # Full width is retained. Only local-pair, I, and P channels are visible.
        difference[:, 3:] = 0.0
    elif scope != "global":
        raise ValueError(f"unknown scope {scope!r}")
    return difference


def fit_gate(
    config: Config,
    graph_types: Sequence[GraphType],
    train: Dataset,
    *,
    scope: str,
) -> np.ndarray:
    contrast_weight = np.square(train.values[:, 0] - train.values[:, 1])
    type_difference = np.stack(
        [
            _feature_differences(
                graph_types,
                np.asarray([type_index], dtype=np.int64),
                scope=scope,
            )[0]
            for type_index in range(len(graph_types))
        ]
    )
    weight_by_type = np.asarray(
        [
            np.sum(contrast_weight[train.type_index == type_index])
            for type_index in range(len(graph_types))
        ],
        dtype=np.float64,
    )
    denominator = max(float(np.sum(weight_by_type)), 1.0e-15)
    delta = torch.tensor(type_difference, dtype=torch.float64)
    type_weight = torch.tensor(weight_by_type / denominator, dtype=torch.float64)
    weight = torch.zeros(type_difference.shape[1], dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.Adam([weight], lr=float(config.learning_rate))
    for _ in range(int(config.training_steps)):
        cycle_mass = torch.sigmoid(delta @ weight)
        loss = torch.sum(type_weight * (1.0 - cycle_mass).square())
        loss = loss + float(config.ridge) * torch.sum(weight.square())
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    return weight.detach().cpu().numpy()


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponent = np.exp(value[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def confidence_match_logits(local_logits: np.ndarray, global_logits: np.ndarray) -> np.ndarray:
    """Retain the global selected candidate with the local absolute confidence."""

    reference = float(np.mean(np.abs(local_logits)))
    sign = np.where(global_logits >= 0.0, 1.0, -1.0)
    return sign * reference


def evaluate_model(
    graph_types: Sequence[GraphType],
    test: Dataset,
    *,
    scope: str,
    calibration: str,
    logits: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cycle_mass = _sigmoid(logits)
    prediction = cycle_mass * test.values[:, 0] + (1.0 - cycle_mass) * test.values[:, 1]
    graph_profiles = np.stack([item.carrier_profiles for item in graph_types])
    carrier_profiles = graph_profiles[test.type_index]

    channel_distance_events: dict[str, list[np.ndarray]] = {
        "semantic": [],
        "structural": [],
    }
    semantic_events: list[np.ndarray] = []
    structural_events: list[np.ndarray] = []
    for source in range(2):
        source_mass = cycle_mass if source == 0 else 1.0 - cycle_mass
        semantic_delta = test.values[:, source] - np.roll(test.values[:, source], 1 + seed)
        semantic_event = np.abs(semantic_delta * source_mass)
        semantic_events.append(semantic_event)
        channel_distance_events["semantic"].append(
            semantic_event[:, None] * carrier_profiles[:, source]
        )

        source_gradient = source_mass * (test.values[:, source] - prediction)
        source_logit_delta = logits if source == 0 else -logits
        structural_event = np.abs(source_logit_delta * source_gradient)
        structural_events.append(structural_event)
        channel_distance_events["structural"].append(
            structural_event[:, None] * carrier_profiles[:, source]
        )

    semantic_score = float(np.mean(np.stack(semantic_events)))
    structural_score = float(np.mean(np.stack(structural_events)))
    result = {
        "seed": int(seed),
        "scope": scope,
        "calibration": calibration,
        "model": f"{scope}_{calibration}",
        "test_mse": float(np.mean(np.square(prediction - test.target))),
        "test_mae": float(np.mean(np.abs(prediction - test.target))),
        "mean_cycle_mass": float(np.mean(cycle_mass)),
        "mean_confidence": float(np.mean(np.maximum(cycle_mass, 1.0 - cycle_mass))),
        "structural_classification_accuracy": float(np.mean(logits > 0.0)),
        "semantic_score_raw": semantic_score,
        "structural_score_raw": structural_score,
    }
    distance_rows: list[dict[str, Any]] = []
    for channel, score in (
        ("semantic", semantic_score),
        ("structural", structural_score),
    ):
        raw_profile = np.mean(np.stack(channel_distance_events[channel]), axis=(0, 1))
        if not np.isclose(raw_profile.sum(), score, atol=1.0e-12):
            raise RuntimeError("distance profile does not reconstruct its raw score")
        normalized = raw_profile / max(float(raw_profile.sum()), 1.0e-15)
        for distance, (raw_mass, normalized_mass) in enumerate(
            zip(raw_profile, normalized, strict=True)
        ):
            distance_rows.append(
                {
                    "seed": int(seed),
                    "scope": scope,
                    "calibration": calibration,
                    "model": f"{scope}_{calibration}",
                    "channel": channel,
                    "distance": int(distance),
                    "raw_mass": float(raw_mass),
                    "normalized_mass": float(normalized_mass),
                }
            )
    return result, distance_rows


def run_experiment(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config.validate()
    torch.set_num_threads(1)
    graph_types, feature_mean, feature_scale = build_graph_types(config)
    rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    health: dict[str, Any] = {"seeds": {}}
    for seed in config.seeds:
        train = generate_dataset(
            config,
            count=config.train_examples,
            seed=config.data_seed + int(seed),
        )
        test = generate_dataset(
            config,
            count=config.test_examples,
            seed=config.data_seed + 10_000 + int(seed),
        )
        weights = {
            scope: fit_gate(config, graph_types, train, scope=scope)
            for scope in SCOPE_ORDER
        }
        free_logits = {
            scope: _feature_differences(
                graph_types,
                test.type_index,
                scope=scope,
            )
            @ weights[scope]
            for scope in SCOPE_ORDER
        }
        matched_logits = {
            "local": free_logits["local"],
            "global": confidence_match_logits(
                free_logits["local"],
                free_logits["global"],
            ),
        }
        for calibration, logits_by_scope in (
            ("free", free_logits),
            ("matched", matched_logits),
        ):
            for scope in SCOPE_ORDER:
                result, distances = evaluate_model(
                    graph_types,
                    test,
                    scope=scope,
                    calibration=calibration,
                    logits=logits_by_scope[scope],
                    seed=int(seed),
                )
                rows.append(result)
                distance_rows.extend(distances)
        health["seeds"][str(seed)] = {
            "local_weight_norm": float(np.linalg.norm(weights["local"])),
            "global_weight_norm": float(np.linalg.norm(weights["global"])),
            "global_type_margins": [
                float(value)
                for value in (
                    np.stack(
                        [
                            item.candidate_features[0] - item.candidate_features[1]
                            for item in graph_types
                        ]
                    )
                    @ weights["global"]
                )
            ],
        }
    health["feature_mean"] = feature_mean.tolist()
    health["feature_scale"] = feature_scale.tolist()
    return pd.DataFrame(rows), pd.DataFrame(distance_rows), health


def _mean_ci(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=np.float64)
    mean = float(np.mean(array))
    if array.size < 2:
        return mean, 0.0
    return mean, float(1.96 * np.std(array, ddof=1) / np.sqrt(array.size))


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _clean_axis(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(width=0.8)


def _panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.14,
        1.10,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        color=TEXT,
    )


def plot_results(results: pd.DataFrame, distances: pd.DataFrame, config: Config) -> tuple[Path, Path]:
    _configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.10, top=0.84, wspace=0.28, hspace=0.40)
    figure.suptitle(
        "A graph-derived RRWP performance gap with nearly unchanged score geometry",
        fontsize=14.5,
        y=0.97,
        color=TEXT,
    )
    figure.text(
        0.5,
        0.91,
        "Local and global gates share the same width and two-step carrier map; higher RRWP channels are zeroed locally",
        ha="center",
        fontsize=9.2,
        color=GREY,
    )

    axis = axes[0, 0]
    _panel_label(axis, "a")
    width = 0.34
    x = np.arange(2)
    for offset, scope in ((-width / 2, "local"), (width / 2, "global")):
        means, cis = [], []
        for calibration in CALIBRATION_ORDER:
            subset = results[
                (results["scope"] == scope) & (results["calibration"] == calibration)
            ]
            mean, ci = _mean_ci(subset["test_mse"])
            means.append(mean)
            cis.append(ci)
        bars = axis.bar(
            x + offset,
            means,
            width,
            yerr=cis,
            capsize=3,
            color=SCOPE_COLOURS[scope],
            alpha=0.86,
            edgecolor="none",
            label=f"{scope} RRWP",
        )
        for bar, mean in zip(bars, means, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                mean + 0.012,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                color=TEXT,
            )
    axis.set_title("Higher-order RRWP improves prediction")
    axis.set_ylabel("held-out MSE")
    axis.set_xticks(x, ("freely fitted", "confidence matched"))
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, loc="upper right")
    _clean_axis(axis)

    axis = axes[0, 1]
    _panel_label(axis, "b")
    positions = np.asarray((0.0, 1.0, 2.5, 3.5))
    labels = (
        "semantic\nfree",
        "structural\nfree",
        "semantic\nmatched",
        "structural\nmatched",
    )
    axis.axvline(1.75, color=LIGHT_GREY, linewidth=0.9)
    for scope, marker in (("local", "o"), ("global", "s")):
        means, cis = [], []
        for calibration, channel in (
            ("free", "semantic"),
            ("free", "structural"),
            ("matched", "semantic"),
            ("matched", "structural"),
        ):
            subset = results[
                (results["scope"] == scope) & (results["calibration"] == calibration)
            ]
            mean, ci = _mean_ci(subset[f"{channel}_score_raw"])
            means.append(mean)
            cis.append(ci)
        axis.errorbar(
            positions,
            means,
            yerr=cis,
            marker=marker,
            markersize=6,
            linewidth=1.5,
            capsize=2.5,
            color=SCOPE_COLOURS[scope],
            label=f"{scope} RRWP",
        )
    axis.set_title("Raw score scale mostly reflects confidence")
    axis.set_ylabel("raw projected-transport score")
    axis.set_xticks(positions, labels)
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, loc="center right")
    _clean_axis(axis)

    line_specs = (
        ("local", "free", "-", "o", "local, free"),
        ("global", "free", "-", "s", "global, free"),
        ("local", "matched", "--", "o", "local, matched"),
        ("global", "matched", "--", "s", "global, matched"),
    )
    for panel, channel in enumerate(("semantic", "structural")):
        axis = axes[1, panel]
        _panel_label(axis, "c" if panel == 0 else "d")
        for scope, calibration, linestyle, marker, label in line_specs:
            subset = distances[
                (distances["scope"] == scope)
                & (distances["calibration"] == calibration)
                & (distances["channel"] == channel)
            ]
            means, cis = [], []
            distance_values = sorted(int(value) for value in subset["distance"].unique())
            for distance in distance_values:
                mean, ci = _mean_ci(
                    subset.loc[subset["distance"] == distance, "normalized_mass"]
                )
                means.append(mean)
                cis.append(ci)
            distance_array = np.asarray(distance_values)
            mean_array = np.asarray(means)
            ci_array = np.asarray(cis)
            axis.plot(
                distance_array,
                mean_array,
                color=SCOPE_COLOURS[scope],
                linestyle=linestyle,
                marker=marker,
                markersize=4.2,
                linewidth=1.7,
                label=label,
            )
            axis.fill_between(
                distance_array,
                mean_array - ci_array,
                mean_array + ci_array,
                color=SCOPE_COLOURS[scope],
                alpha=0.10,
                linewidth=0,
            )
        axis.set_title(f"{channel.capitalize()} score-distance profile")
        axis.set_xlabel("source-to-carrier distance")
        axis.set_ylabel("fraction of raw score mass")
        axis.set_xticks((0, 1, 2))
        axis.set_ylim(0.0, 0.64)
        comparison = _profile_comparison(distances)
        free = comparison[f"free_{channel}"]
        matched = comparison[f"matched_{channel}"]
        maximum_tv = max(
            float(free["profile_total_variation"]),
            float(matched["profile_total_variation"]),
        )
        maximum_ci = max(
            float(free["local_max_95ci_half_width"]),
            float(free["global_max_95ci_half_width"]),
            float(matched["local_max_95ci_half_width"]),
            float(matched["global_max_95ci_half_width"]),
        )
        if maximum_tv < 1.0e-12:
            note = "local-global TV < 10^-12\n95% CI half-width < 10^-12"
        else:
            note = (
                f"local-global TV <= {maximum_tv:.4f}\n"
                f"max 95% CI half-width <= {maximum_ci:.5f}"
            )
        axis.text(
            0.98,
            0.96 if panel == 0 else 0.68,
            note,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color=GREY,
        )
        _clean_axis(axis)
        if panel == 1:
            axis.legend(frameon=False, loc="upper right", ncol=2)

    figures = config.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / "05_rrwp_performance_scores_and_distance.png"
    pdf = figures / "05_rrwp_performance_scores_and_distance.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _condition_summary(results: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for scope in SCOPE_ORDER:
        for calibration in CALIBRATION_ORDER:
            subset = results[
                (results["scope"] == scope) & (results["calibration"] == calibration)
            ]
            item: dict[str, float] = {}
            for column in (
                "test_mse",
                "mean_confidence",
                "structural_classification_accuracy",
                "semantic_score_raw",
                "structural_score_raw",
            ):
                mean, ci = _mean_ci(subset[column])
                item[f"{column}_mean"] = mean
                item[f"{column}_95ci_half_width"] = ci
            output[f"{scope}_{calibration}"] = item
    return output


def _paired_contrasts(results: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for calibration in CALIBRATION_ORDER:
        pivot = results[results["calibration"] == calibration].pivot(
            index="seed",
            columns="scope",
            values=[
                "test_mse",
                "mean_confidence",
                "semantic_score_raw",
                "structural_score_raw",
            ],
        )
        item: dict[str, float] = {}
        for column in (
            "test_mse",
            "mean_confidence",
            "semantic_score_raw",
            "structural_score_raw",
        ):
            difference = pivot[(column, "local")] - pivot[(column, "global")]
            mean, ci = _mean_ci(difference)
            item[f"local_minus_global_{column}_mean"] = mean
            item[f"local_minus_global_{column}_95ci_half_width"] = ci
        output[calibration] = item
    return output


def _profile_comparison(distances: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for calibration in CALIBRATION_ORDER:
        for channel in ("semantic", "structural"):
            rows: dict[str, list[float]] = {}
            ci_widths: dict[str, list[float]] = {}
            for scope in SCOPE_ORDER:
                subset = distances[
                    (distances["scope"] == scope)
                    & (distances["calibration"] == calibration)
                    & (distances["channel"] == channel)
                ]
                means, cis = [], []
                for distance in sorted(subset["distance"].unique()):
                    mean, ci = _mean_ci(
                        subset.loc[
                            subset["distance"] == distance,
                            "normalized_mass",
                        ]
                    )
                    means.append(mean)
                    cis.append(ci)
                rows[scope] = means
                ci_widths[scope] = cis
            local = np.asarray(rows["local"])
            global_values = np.asarray(rows["global"])
            output[f"{calibration}_{channel}"] = {
                "profile_total_variation": float(0.5 * np.abs(local - global_values).sum()),
                "local_max_95ci_half_width": float(max(ci_widths["local"])),
                "global_max_95ci_half_width": float(max(ci_widths["global"])),
                "local_profile": rows["local"],
                "global_profile": rows["global"],
            }
    return output


def run(config: Config) -> dict[str, Any]:
    results, distances, health = run_experiment(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.output_dir / "model_scores.csv", index=False)
    distances.to_csv(config.output_dir / "distance_profiles.csv", index=False)
    figures = plot_results(results, distances, config)
    summary = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "conditions": _condition_summary(results),
        "paired_contrasts": _paired_contrasts(results),
        "profile_comparisons": _profile_comparison(distances),
        "health": health,
        "figures": [str(path) for path in figures],
    }
    with (config.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Compare local/global RRWP performance and score-distance profiles."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/chapter6_rrwp_score_distance_v1"),
    )
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--train-examples", type=int, default=8_192)
    parser.add_argument("--test-examples", type=int, default=8_192)
    parser.add_argument("--training-steps", type=int, default=5_000)
    parser.add_argument("--ridge", type=float, default=1.0e-2)
    parser.add_argument("--local-clue-reliability", type=float, default=0.75)
    args = parser.parse_args(argv)
    config = Config(
        output_dir=args.output_dir,
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        train_examples=int(args.train_examples),
        test_examples=int(args.test_examples),
        training_steps=int(args.training_steps),
        ridge=float(args.ridge),
        local_clue_reliability=float(args.local_clue_reliability),
    )
    result = run(config)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
