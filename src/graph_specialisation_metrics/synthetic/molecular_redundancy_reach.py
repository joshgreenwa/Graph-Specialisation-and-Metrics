"""Apparent long-range use versus task necessity on molecular graph supports.

Each example uses a real ZINC molecular topology and one anchored binary target.
In the redundant task, the target bit is presented at the anchor and copied to
nodes at increasing shortest-path distances.  A minimum-norm graph filter learns
to share weight across the interchangeable copies.  Jacobian range and finite
Functional carriage therefore report long-range use even though the local copy
alone is sufficient for exact prediction.

An essential-distance control presents the target bit only at the far node.
The frozen far-input deletion measures fitted-model reliance; a separately
fitted local-only model measures achievable performance without the far input.
Thus the benchmark separates three objects:

* apparent reach: Jacobian or finite input-to-output sensitivity;
* reliance: loss incurred by deleting far inputs from the fitted model; and
* necessity: loss retained after fitting the same model class without far input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .molecular_nonlinear_reach import shortest_path_matrix


PROTOCOL_VERSION = "molecular-redundancy-reach-v1"
METHOD_LABELS = {
    "bamberger": "Bamberger Jacobian",
    "finite": "Functional carriage",
}
METHOD_COLOURS = {"bamberger": "#0072B2", "finite": "#D55E00"}


@dataclass(frozen=True)
class Config:
    output_dir: Path
    data_root: Path
    train_graphs: int = 512
    val_graphs: int = 96
    test_graphs: int = 96
    copy_distances: tuple[int, ...] = (2, 4, 6)
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    train_steps: int = 500
    learning_rate: float = 0.05
    weight_decay: float = 1.0e-5
    bootstrap_replicates: int = 2_000
    analysis_seed: int = 72_031

    @property
    def max_distance(self) -> int:
        return max(self.copy_distances)

    @property
    def fingerprint(self) -> str:
        payload = asdict(self)
        payload.pop("output_dir")
        payload.pop("data_root")
        payload["protocol_version"] = PROTOCOL_VERSION
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]

    def validate(self) -> None:
        if min(self.train_graphs, self.val_graphs, self.test_graphs) < 1:
            raise ValueError("each split requires at least one graph")
        if not self.copy_distances:
            raise ValueError("copy_distances cannot be empty")
        if tuple(sorted(set(self.copy_distances))) != self.copy_distances:
            raise ValueError("copy_distances must be unique and increasing")
        if min(self.copy_distances) < 1:
            raise ValueError("copy distances must be positive")
        if self.train_steps < 1 or self.learning_rate <= 0:
            raise ValueError("invalid optimisation settings")


@dataclass
class AnchoredMolecule:
    graph_id: int
    spd: torch.Tensor
    anchor: int
    copy_nodes: dict[int, int]
    target: float

    @property
    def num_nodes(self) -> int:
        return int(self.spd.shape[0])


class AnchoredRadialRegressor(nn.Module):
    """A scalar-output graph filter with a learned coefficient per SPD shell."""

    def __init__(self, max_distance: int):
        super().__init__()
        self.max_distance = int(max_distance)
        self.radial_weight = nn.Parameter(torch.empty(max_distance + 1))
        self.bias = nn.Parameter(torch.zeros(()))
        # Gradient descent from zero selects the symmetric minimum-norm solution
        # when several shells carry exactly interchangeable target copies.
        nn.init.zeros_(self.radial_weight)

    def forward(
        self,
        cue: torch.Tensor,
        role: torch.Tensor,
        spd: torch.Tensor,
        anchor: int,
    ) -> torch.Tensor:
        feature = anchored_radial_features(
            cue, role, spd, anchor, max_distance=self.max_distance
        )
        return torch.dot(feature, self.radial_weight) + self.bias


def anchored_radial_features(
    cue: torch.Tensor,
    role: torch.Tensor,
    spd: torch.Tensor,
    anchor: int,
    *,
    max_distance: int,
) -> torch.Tensor:
    """Sum marked cue values in every shell around the registered anchor."""

    marked = cue * role.to(cue.dtype)
    distance = spd[int(anchor)]
    return torch.stack(
        [marked[distance == shell].sum() for shell in range(int(max_distance) + 1)]
    )


def _select_indices(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.permutation(int(length))


def _target_bit(graph_id: int, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{graph_id}".encode("utf-8")).digest()
    return 1.0 if digest[0] % 2 else -1.0


def load_anchored_molecules(
    config: Config,
    *,
    split: str,
    count: int,
    split_seed: int,
) -> list[AnchoredMolecule]:
    try:
        from torch_geometric.datasets import ZINC
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise ImportError("This benchmark requires torch_geometric") from error

    dataset = ZINC(root=str(config.data_root), subset=True, split=split)
    offset = {"train": 0, "val": 20_000, "test": 40_000}[split]
    rng = np.random.default_rng(int(split_seed))
    samples: list[AnchoredMolecule] = []
    for dataset_index in _select_indices(len(dataset), split_seed).tolist():
        graph = dataset[int(dataset_index)]
        spd = shortest_path_matrix(int(graph.num_nodes), graph.edge_index)
        eligible = [
            node
            for node in range(int(graph.num_nodes))
            if all(bool((spd[node] == distance).any()) for distance in config.copy_distances)
        ]
        if not eligible:
            continue
        anchor = int(rng.choice(eligible))
        copy_nodes = {
            distance: int(rng.choice(torch.where(spd[anchor] == distance)[0].numpy()))
            for distance in config.copy_distances
        }
        graph_id = offset + int(dataset_index)
        samples.append(
            AnchoredMolecule(
                graph_id=graph_id,
                spd=spd,
                anchor=anchor,
                copy_nodes=copy_nodes,
                target=_target_bit(graph_id, config.analysis_seed),
            )
        )
        if len(samples) >= int(count):
            break
    if len(samples) < int(count):
        print(
            f"[data:warning] {split} supplied {len(samples)}/{count} molecules "
            f"with every requested shell {config.copy_distances}"
        )
    return samples


def task_inputs(
    sample: AnchoredMolecule,
    *,
    task: str,
    active_distances: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build legal input cues for redundant or far-essential tasks."""

    cue = torch.zeros(sample.num_nodes, dtype=torch.float64)
    role = torch.zeros(sample.num_nodes, dtype=torch.bool)
    distances = tuple(int(value) for value in active_distances)
    if 0 in distances:
        role[sample.anchor] = True
        if task == "redundant":
            cue[sample.anchor] = float(sample.target)
    if task == "redundant":
        for distance in distances:
            if distance == 0:
                continue
            node = sample.copy_nodes[int(distance)]
            role[node] = True
            cue[node] = float(sample.target)
    elif task == "essential":
        far_distance = max(sample.copy_nodes)
        if far_distance in distances:
            node = sample.copy_nodes[far_distance]
            role[node] = True
            cue[node] = float(sample.target)
    else:
        raise ValueError(f"unknown task {task!r}")
    return cue, role


def redundancy_variants(config: Config) -> list[tuple[str, tuple[int, ...]]]:
    variants = [("local", (0,))]
    active = [0]
    for distance in config.copy_distances:
        active.append(int(distance))
        variants.append((f"through_d{distance}", tuple(active)))
    return variants


def _design(
    samples: Sequence[AnchoredMolecule],
    *,
    task: str,
    active_distances: Sequence[int],
    max_distance: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    targets: list[float] = []
    for sample in samples:
        cue, role = task_inputs(sample, task=task, active_distances=active_distances)
        features.append(
            anchored_radial_features(
                cue,
                role,
                sample.spd,
                sample.anchor,
                max_distance=max_distance,
            )
        )
        targets.append(float(sample.target))
    return torch.stack(features), torch.tensor(targets, dtype=torch.float64)


def train_model(
    config: Config,
    train_samples: Sequence[AnchoredMolecule],
    val_samples: Sequence[AnchoredMolecule],
    *,
    task: str,
    active_distances: Sequence[int],
    seed: int,
) -> tuple[AnchoredRadialRegressor, list[dict[str, Any]]]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    model = AnchoredRadialRegressor(config.max_distance).to(dtype=torch.float64)
    train_x, train_y = _design(
        train_samples,
        task=task,
        active_distances=active_distances,
        max_distance=config.max_distance,
    )
    val_x, val_y = _design(
        val_samples,
        task=task,
        active_distances=active_distances,
        max_distance=config.max_distance,
    )
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    log: list[dict[str, Any]] = []
    report_every = max(1, int(config.train_steps) // 20)
    for step in range(1, int(config.train_steps) + 1):
        optimiser.zero_grad(set_to_none=True)
        prediction = train_x @ model.radial_weight + model.bias
        loss = torch.mean((prediction - train_y) ** 2)
        loss.backward()
        optimiser.step()
        if step == 1 or step % report_every == 0 or step == config.train_steps:
            with torch.no_grad():
                train_mae = torch.mean(torch.abs(prediction - train_y)).item()
                val_prediction = val_x @ model.radial_weight + model.bias
                val_mae = torch.mean(torch.abs(val_prediction - val_y)).item()
            log.append({"step": step, "train_mae": train_mae, "val_mae": val_mae})
            if val_mae < best_val:
                best_val = val_mae
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, log


def _normalised_profile(mass: np.ndarray, distance: np.ndarray, max_distance: int) -> np.ndarray:
    profile = np.zeros(int(max_distance) + 1, dtype=np.float64)
    for shell in range(int(max_distance) + 1):
        profile[shell] = float(mass[distance == shell].sum())
    total = float(profile.sum())
    return profile / total if total > 1.0e-15 else profile


def measure_reach(
    config: Config,
    model: AnchoredRadialRegressor,
    samples: Sequence[AnchoredMolecule],
    *,
    task: str,
    variant: str,
    active_distances: Sequence[int],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        cue, role = task_inputs(sample, task=task, active_distances=active_distances)
        cue = cue.requires_grad_(True)
        clean = model(cue, role, sample.spd, sample.anchor)
        gradient = torch.autograd.grad(clean, cue)[0].detach()
        active_sources = torch.where(role & (cue.detach().abs() > 0))[0]
        bamberger_mass = gradient[active_sources].abs().numpy()
        finite_mass: list[float] = []
        for source in active_sources.tolist():
            donor = cue.detach().clone()
            donor[int(source)] = -donor[int(source)]
            with torch.no_grad():
                finite_mass.append(float(torch.abs(clean.detach() - model(
                    donor, role, sample.spd, sample.anchor
                ))))
        distances = sample.spd[sample.anchor, active_sources].numpy()
        for method, mass in (
            ("bamberger", bamberger_mass),
            ("finite", np.asarray(finite_mass, dtype=np.float64)),
        ):
            profile = _normalised_profile(mass, distances, config.max_distance)
            expected = float(np.dot(np.arange(config.max_distance + 1), profile))
            for distance, value in enumerate(profile.tolist()):
                rows.append(
                    {
                        "seed": int(seed),
                        "graph_id": sample.graph_id,
                        "task": task,
                        "variant": variant,
                        "copies": max(0, len(active_distances) - 1),
                        "method": method,
                        "distance": distance,
                        "normalised_mass": value,
                        "expected_distance": expected,
                    }
                )
    return rows


def evaluate_performance(
    config: Config,
    model: AnchoredRadialRegressor,
    local_model: AnchoredRadialRegressor,
    samples: Sequence[AnchoredMolecule],
    *,
    task: str,
    variant: str,
    active_distances: Sequence[int],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        cue, role = task_inputs(sample, task=task, active_distances=active_distances)
        local_cue, local_role = task_inputs(sample, task=task, active_distances=(0,))
        with torch.no_grad():
            full = model(cue, role, sample.spd, sample.anchor)
            frozen = model(local_cue, local_role, sample.spd, sample.anchor)
            refit = local_model(local_cue, local_role, sample.spd, sample.anchor)
        target = float(sample.target)
        rows.append(
            {
                "seed": int(seed),
                "graph_id": sample.graph_id,
                "task": task,
                "variant": variant,
                "copies": max(0, len(active_distances) - 1),
                "full_mae": abs(float(full) - target),
                "frozen_far_removed_mae": abs(float(frozen) - target),
                "local_refit_mae": abs(float(refit) - target),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _bootstrap(
    values: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size < 2 or replicates < 2:
        return mean, mean, mean
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        boot[index] = float(np.mean(values[rng.integers(0, values.size, values.size)]))
    return mean, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def summarise(
    config: Config,
    reach_rows: Sequence[dict[str, str]],
    performance_rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    profile_summary: list[dict[str, Any]] = []
    expected_summary: list[dict[str, Any]] = []
    combinations = sorted(
        {
            (row["task"], row["variant"], int(row["copies"]), row["method"])
            for row in reach_rows
        }
    )
    for task, variant, copies, method in combinations:
        selected = [
            row
            for row in reach_rows
            if row["task"] == task
            and row["variant"] == variant
            and int(row["copies"]) == copies
            and row["method"] == method
        ]
        expected_by_graph: dict[int, list[float]] = {}
        profile_by_graph: dict[tuple[int, int], list[float]] = {}
        for row in selected:
            graph = int(row["graph_id"])
            expected_by_graph.setdefault(graph, []).append(float(row["expected_distance"]))
            profile_by_graph.setdefault((graph, int(row["distance"])), []).append(
                float(row["normalised_mass"])
            )
        expected_values = np.asarray(
            [np.mean(values) for values in expected_by_graph.values()], dtype=np.float64
        )
        mean, low, high = _bootstrap(
            expected_values,
            config.bootstrap_replicates,
            config.analysis_seed + 19 * copies + len(method),
        )
        expected_summary.append(
            {
                "task": task,
                "variant": variant,
                "copies": copies,
                "method": method,
                "mean": mean,
                "low": low,
                "high": high,
            }
        )
        for distance in range(config.max_distance + 1):
            values = np.asarray(
                [
                    np.mean(rows)
                    for (graph, shell), rows in profile_by_graph.items()
                    if shell == distance
                ],
                dtype=np.float64,
            )
            mean, low, high = _bootstrap(
                values,
                config.bootstrap_replicates,
                config.analysis_seed + 101 * distance + copies,
            )
            profile_summary.append(
                {
                    "task": task,
                    "variant": variant,
                    "copies": copies,
                    "method": method,
                    "distance": distance,
                    "mean": mean,
                    "low": low,
                    "high": high,
                }
            )

    performance_summary: list[dict[str, Any]] = []
    performance_combinations = sorted(
        {
            (row["task"], row["variant"], int(row["copies"]))
            for row in performance_rows
        }
    )
    for task, variant, copies in performance_combinations:
        selected = [
            row
            for row in performance_rows
            if row["task"] == task
            and row["variant"] == variant
            and int(row["copies"]) == copies
        ]
        for metric in ("full_mae", "frozen_far_removed_mae", "local_refit_mae"):
            graph_values: dict[int, list[float]] = {}
            for row in selected:
                graph_values.setdefault(int(row["graph_id"]), []).append(float(row[metric]))
            values = np.asarray(
                [np.mean(items) for items in graph_values.values()], dtype=np.float64
            )
            mean, low, high = _bootstrap(
                values,
                config.bootstrap_replicates,
                config.analysis_seed + 13 * copies + len(metric),
            )
            performance_summary.append(
                {
                    "task": task,
                    "variant": variant,
                    "copies": copies,
                    "metric": metric,
                    "mean": mean,
                    "low": low,
                    "high": high,
                }
            )
    return profile_summary, expected_summary, performance_summary


def _row(
    rows: Sequence[dict[str, Any]],
    **selection: Any,
) -> dict[str, Any]:
    matched = [
        row
        for row in rows
        if all(row[key] == value for key, value in selection.items())
    ]
    if len(matched) != 1:
        raise ValueError(f"selection {selection} returned {len(matched)} rows")
    return matched[0]


def plot_headline(
    config: Config,
    profile_summary: Sequence[dict[str, Any]],
    expected_summary: Sequence[dict[str, Any]],
    performance_summary: Sequence[dict[str, Any]],
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.09, top=0.80, wspace=0.27, hspace=0.42)
    maximal_variant = redundancy_variants(config)[-1][0]
    for method, marker, linestyle in (
        ("bamberger", "^", "--"),
        ("finite", "s", ":"),
    ):
        selected = [
            row
            for row in profile_summary
            if row["task"] == "redundant"
            and row["variant"] == maximal_variant
            and row["method"] == method
        ]
        selected.sort(key=lambda row: int(row["distance"]))
        x = np.asarray([int(row["distance"]) for row in selected])
        y = np.asarray([float(row["mean"]) for row in selected])
        low = np.asarray([float(row["low"]) for row in selected])
        high = np.asarray([float(row["high"]) for row in selected])
        axes[0, 0].plot(
            x,
            y,
            color=METHOD_COLOURS[method],
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            label=METHOD_LABELS[method],
        )
        axes[0, 0].fill_between(x, low, high, color=METHOD_COLOURS[method], alpha=0.12)
    axes[0, 0].axvline(0, color="#333333", linewidth=1.3, alpha=0.7)
    axes[0, 0].text(0.2, 0.96, "local cue alone is sufficient", transform=axes[0, 0].transAxes,
                    ha="left", va="top", fontsize=8, color="#666666")
    axes[0, 0].set_title("Apparent use of redundant copies")
    axes[0, 0].set_xlabel("Shortest-path distance")
    axes[0, 0].set_ylabel("Normalised usage mass")
    axes[0, 0].set_xlim(-0.15, config.max_distance + 0.15)
    axes[0, 0].set_ylim(bottom=0)

    for method, marker, linestyle in (
        ("bamberger", "^", "--"),
        ("finite", "s", ":"),
    ):
        selected = [
            row
            for row in expected_summary
            if row["task"] == "redundant" and row["method"] == method
        ]
        selected.sort(key=lambda row: int(row["copies"]))
        x = np.asarray([int(row["copies"]) for row in selected])
        y = np.asarray([float(row["mean"]) for row in selected])
        low = np.asarray([float(row["low"]) for row in selected])
        high = np.asarray([float(row["high"]) for row in selected])
        axes[0, 1].plot(
            x, y, color=METHOD_COLOURS[method], marker=marker, linestyle=linestyle,
            linewidth=2, label=METHOD_LABELS[method]
        )
        axes[0, 1].fill_between(x, low, high, color=METHOD_COLOURS[method], alpha=0.12)
    axes[0, 1].axhline(0, color="#333333", linewidth=1.4, label="Minimum required distance")
    axes[0, 1].set_title("Apparent reach grows with redundancy")
    axes[0, 1].set_xlabel("Number of distant redundant copies")
    axes[0, 1].set_ylabel("Expected distance")
    axes[0, 1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    essential_variant = f"essential_d{config.max_distance}"
    task_variants = (("redundant", maximal_variant), ("essential", essential_variant))
    metrics = (
        ("full_mae", "Full model", "#555555"),
        ("frozen_far_removed_mae", "Far removed\n(frozen)", "#CC79A7"),
        ("local_refit_mae", "Local-only\n(refit)", "#009E73"),
    )
    x = np.arange(len(task_variants), dtype=np.float64)
    width = 0.24
    for metric_index, (metric, label, colour) in enumerate(metrics):
        values, low, high = [], [], []
        for task, variant in task_variants:
            row = _row(
                performance_summary,
                task=task,
                variant=variant,
                metric=metric,
            )
            values.append(float(row["mean"]))
            low.append(float(row["low"]))
            high.append(float(row["high"]))
        values = np.asarray(values)
        axes[1, 0].bar(
            x + (metric_index - 1) * width,
            values,
            width,
            label=label,
            color=colour,
            alpha=0.85,
            yerr=np.asarray([values - np.asarray(low), np.asarray(high) - values]),
            capsize=3,
        )
    axes[1, 0].set_xticks(x, ("Redundant far cues", "Essential far cue"))
    axes[1, 0].set_ylabel("Held-out MAE")
    axes[1, 0].set_title("Frozen reliance versus local-only sufficiency")
    axes[1, 0].legend(frameon=False, fontsize=8, ncol=3)

    reliance, necessity = [], []
    labels = []
    for task, variant in task_variants:
        full = _row(performance_summary, task=task, variant=variant, metric="full_mae")
        frozen = _row(
            performance_summary,
            task=task,
            variant=variant,
            metric="frozen_far_removed_mae",
        )
        refit = _row(
            performance_summary,
            task=task,
            variant=variant,
            metric="local_refit_mae",
        )
        reliance.append(float(frozen["mean"]) - float(full["mean"]))
        necessity.append(float(refit["mean"]) - float(full["mean"]))
        labels.append("Redundant" if task == "redundant" else "Essential")
    axes[1, 1].bar(x - 0.18, reliance, 0.36, label="Fitted-model reliance", color="#CC79A7")
    axes[1, 1].bar(x + 0.18, necessity, 0.36, label="Task necessity", color="#009E73")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylabel("MAE increase")
    axes[1, 1].set_title("Long-range reliance need not imply necessity")
    axes[1, 1].legend(frameon=False, fontsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.88))
    fig.suptitle("Apparent long-range use under redundant molecular cues", fontsize=15, y=0.985)
    fig.text(
        0.5,
        0.935,
        "ZINC topology; linear task where Jacobian and finite carriage agree exactly",
        ha="center",
        va="top",
        color="#666666",
        fontsize=9,
    )
    figure_dir = config.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "molecular_redundancy_reach.png"
    pdf = figure_dir / "molecular_redundancy_reach.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def measure(config: Config) -> dict[str, Any]:
    train_samples = load_anchored_molecules(
        config, split="train", count=config.train_graphs, split_seed=config.analysis_seed + 1
    )
    val_samples = load_anchored_molecules(
        config, split="val", count=config.val_graphs, split_seed=config.analysis_seed + 2
    )
    test_samples = load_anchored_molecules(
        config, split="test", count=config.test_graphs, split_seed=config.analysis_seed + 3
    )
    variants = redundancy_variants(config)
    essential_variant = f"essential_d{config.max_distance}"
    reach_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    health_rows: list[dict[str, Any]] = []
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for seed in config.seeds:
        models: dict[tuple[str, str], AnchoredRadialRegressor] = {}
        for task, task_variants in (
            ("redundant", variants),
            (
                "essential",
                [("essential_local", (0,)), (essential_variant, (0, config.max_distance))],
            ),
        ):
            for variant, active_distances in task_variants:
                model, log = train_model(
                    config,
                    train_samples,
                    val_samples,
                    task=task,
                    active_distances=active_distances,
                    seed=int(seed),
                )
                models[(task, variant)] = model
                _write_csv(
                    config.output_dir / "training" / f"{task}_{variant}_seed_{seed}.csv",
                    log,
                )
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "task": task,
                        "variant": variant,
                        "seed": int(seed),
                        "fingerprint": config.fingerprint,
                    },
                    checkpoint_dir / f"{task}_{variant}_seed_{seed}.pt",
                )
                health_rows.append(
                    {
                        "seed": int(seed),
                        "task": task,
                        "variant": variant,
                        "train_mae": float(log[-1]["train_mae"]),
                        "val_mae": float(log[-1]["val_mae"]),
                        "weights": json.dumps(model.radial_weight.detach().tolist()),
                    }
                )
        redundant_local = models[("redundant", "local")]
        for variant, active_distances in variants:
            model = models[("redundant", variant)]
            reach_rows.extend(
                measure_reach(
                    config,
                    model,
                    test_samples,
                    task="redundant",
                    variant=variant,
                    active_distances=active_distances,
                    seed=int(seed),
                )
            )
            performance_rows.extend(
                evaluate_performance(
                    config,
                    model,
                    redundant_local,
                    test_samples,
                    task="redundant",
                    variant=variant,
                    active_distances=active_distances,
                    seed=int(seed),
                )
            )
        essential_model = models[("essential", essential_variant)]
        essential_local = models[("essential", "essential_local")]
        reach_rows.extend(
            measure_reach(
                config,
                essential_model,
                test_samples,
                task="essential",
                variant=essential_variant,
                active_distances=(0, config.max_distance),
                seed=int(seed),
            )
        )
        performance_rows.extend(
            evaluate_performance(
                config,
                essential_model,
                essential_local,
                test_samples,
                task="essential",
                variant=essential_variant,
                active_distances=(0, config.max_distance),
                seed=int(seed),
            )
        )
        print(f"[seed {seed}] trained {len(models)} radial models")
    _write_csv(config.output_dir / "results" / "reach_profiles.csv", reach_rows)
    _write_csv(config.output_dir / "results" / "performance.csv", performance_rows)
    _write_csv(config.output_dir / "results" / "model_health.csv", health_rows)
    return {
        "reach_rows": len(reach_rows),
        "performance_rows": len(performance_rows),
        "models": len(health_rows),
    }


def figures(config: Config) -> dict[str, Any]:
    reach_rows = _read_csv(config.output_dir / "results" / "reach_profiles.csv")
    performance_rows = _read_csv(config.output_dir / "results" / "performance.csv")
    profile, expected, performance = summarise(config, reach_rows, performance_rows)
    _write_csv(config.output_dir / "results" / "profile_summary.csv", profile)
    _write_csv(config.output_dir / "results" / "expected_distance_summary.csv", expected)
    _write_csv(config.output_dir / "results" / "performance_summary.csv", performance)
    paths = plot_headline(config, profile, expected, performance)
    _write_json(
        config.output_dir / "results" / "summary.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "fingerprint": config.fingerprint,
            "figures": paths,
            "estimands": {
                "apparent_reach": "input-to-output Jacobian or finite donor response",
                "reliance": "frozen-model MAE increase after deleting every far cue",
                "necessity": "local-only refit MAE increase over the full model",
            },
        },
    )
    return {"figures": paths, "profile": profile, "expected": expected, "performance": performance}


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "measure", "figures"), default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/molecular_redundancy_reach_v1"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / ".cache" / "graph_specialisation_metrics" / "zinc_subset",
    )
    parser.add_argument("--train-graphs", type=int, default=512)
    parser.add_argument("--val-graphs", type=int, default=96)
    parser.add_argument("--test-graphs", type=int, default=96)
    parser.add_argument("--copy-distances", default="2,4,6")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--analysis-seed", type=int, default=72_031)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    config = Config(
        output_dir=args.output_dir,
        data_root=args.data_root,
        train_graphs=int(args.train_graphs),
        val_graphs=int(args.val_graphs),
        test_graphs=int(args.test_graphs),
        copy_distances=_parse_ints(args.copy_distances),
        seeds=_parse_ints(args.seeds),
        train_steps=int(args.train_steps),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        bootstrap_replicates=int(args.bootstrap_replicates),
        analysis_seed=int(args.analysis_seed),
    )
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        config.output_dir / "config.json",
        {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "data_root": str(config.data_root),
            "fingerprint": config.fingerprint,
            "protocol_version": PROTOCOL_VERSION,
        },
    )
    result: dict[str, Any] = {"config": config, "output_dir": str(config.output_dir)}
    if args.phase in {"all", "measure"}:
        result["measurement"] = measure(config)
    if args.phase in {"all", "figures"}:
        result.update(figures(config))
    if "figures" in result:
        print(f"[figure] {result['figures']['png']}")
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
