"""Molecular-inspired multi-site RRWP mediation synthetic.

Three reporter sites independently carry a bounded scalar ``x_i`` and either
contain a remotely closed carbon-like ring or two open arms.  A configurable
target mixes local bond-order and remote ring-state contributions. A
fixed-support single-versus-triple bond motif is also an imperfect local proxy
for closure, while higher-order RRWP at the reporter exposes the remote
endpoint bond. All learned messages remain one-hop.

The experiment trains parameter-matched local, global, and shuffled-global
RRWP arms using an ordinary linear all-node mean readout.  It measures native
learned-head semantic and structural score distance, Functional carriage,
event dose, site-level selection fidelity, uncertainty decomposition, and
matched-head mediation alignment against a same-layer other-head null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

from .rrwp_semantic_relay import RRWPSemanticRelay

plt.switch_backend("Agg")

DTYPE = torch.float32
SITE_COUNT = 3
NODE_COUNT = 37
ARMS = ("local", "global", "global_shuffled")
MEASURED_ARMS = ("local", "global")
CHANNELS = (
    "semantic",
    "structural_full",
    "structural_proxy",
    "structural_remote",
)
STRUCTURAL_CHANNELS = CHANNELS[1:]
SCALES = ("raw", "per_unit")
PROTOCOL_VERSION = "molecular_sites_v4_sign_flip_bond_order"

ORANGE = "#EE7733"
BLUE = "#4477AA"
GREY = "#777777"
GREEN = "#228833"
PURPLE = "#AA3377"
LIGHT_GREY = "#D7D7D7"
ARM_COLOURS = {
    "local": ORANGE,
    "global": BLUE,
    "global_shuffled": GREY,
    "global_test_shuffled": PURPLE,
}
CHANNEL_COLOURS = {
    "semantic": BLUE,
    "structural_full": ORANGE,
    "structural_proxy": GREEN,
    "structural_remote": PURPLE,
}


@dataclass(frozen=True)
class Config:
    output_dir: Path
    protocol_version: str = PROTOCOL_VERSION
    arm_length: int = 4
    rrwp_horizon: int = 12
    hidden_dim: int = 24
    heads: int = 4
    layers: int = 2
    train_examples: int = 2_048
    validation_examples: int = 512
    test_examples: int = 1_024
    measurement_examples: int = 192
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    batch_size: int = 64
    training_steps: int = 600
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-4
    local_clue_reliability: float = 0.75
    remote_target_weight: float = 1.0
    bootstrap_replicates: int = 500
    data_seed: int = 91_103
    value_minimum_magnitude: float = 0.25

    def validate(self) -> None:
        if self.arm_length != 4:
            raise ValueError("the registered 37-node design requires arm_length=4")
        if self.arm_length <= self.layers:
            raise ValueError("the closure must lie beyond learned message depth")
        if 5 <= 2 * self.layers:
            raise ValueError(
                "reporter-to-closure distance must exceed twice message depth"
            )
        if self.rrwp_horizon < 2 * self.arm_length + 3:
            raise ValueError("RRWP horizon must include the reporter closure return")
        if self.hidden_dim < self.heads or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be positive and divisible by heads")
        if min(
            self.train_examples,
            self.validation_examples,
            self.test_examples,
            self.measurement_examples,
            self.batch_size,
            self.training_steps,
        ) < 1:
            raise ValueError("sample and optimization sizes must be positive")
        if self.measurement_examples > self.test_examples:
            raise ValueError("measurement_examples cannot exceed test_examples")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be non-empty and unique")
        if not 0.5 < self.local_clue_reliability < 1.0:
            raise ValueError("local_clue_reliability must lie in (0.5, 1)")
        if not 0.0 <= self.remote_target_weight <= 1.0:
            raise ValueError("remote_target_weight must lie in [0, 1]")
        if not 0.0 < self.value_minimum_magnitude < 1.0:
            raise ValueError("value_minimum_magnitude must lie in (0, 1)")
        if self.bootstrap_replicates < 100:
            raise ValueError("bootstrap_replicates must be at least 100")


@dataclass(frozen=True)
class SiteNodes:
    anchor: int
    reporter: int
    arm_left: tuple[int, ...]
    arm_right: tuple[int, ...]
    cue: tuple[int, int]


@dataclass(frozen=True)
class GraphType:
    cycle_state: tuple[bool, ...]
    cue_state: tuple[bool, ...]
    roles: np.ndarray
    adjacency: np.ndarray
    distances: np.ndarray
    rrwp: np.ndarray
    sites: tuple[SiteNodes, ...]


@dataclass(frozen=True)
class Dataset:
    type_index: np.ndarray
    shuffled_type_index: np.ndarray
    values: np.ndarray
    cycle_state: np.ndarray
    cue_state: np.ndarray
    site_contribution: np.ndarray


def target_weight_second_moment(config: Config) -> float:
    """Return E[((1-w)q + w c)^2] under the registered cue process."""
    remote = float(config.remote_target_weight)
    local = 1.0 - remote
    reliability = float(config.local_clue_reliability)
    return 0.5 * (local**2 + remote**2) + local * remote * reliability


def target_scale(config: Config) -> float:
    """Scale the three-site target to unit population variance."""
    return 1.0 / math.sqrt(SITE_COUNT * target_weight_second_moment(config))


def cue_only_bayes_mse(config: Config) -> float:
    """Analytic Bayes MSE for a predictor that observes values and local cues."""
    remote = float(config.remote_target_weight)
    reliability = float(config.local_clue_reliability)
    return (
        remote**2
        * reliability
        * (1.0 - reliability)
        / target_weight_second_moment(config)
    )


def _transition(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    if np.any(degree <= 0):
        raise ValueError("synthetic graphs must have no isolates")
    return adjacency / degree[:, None]


def _rrwp(adjacency: np.ndarray, horizon: int) -> np.ndarray:
    transition = _transition(adjacency)
    powers = [np.eye(adjacency.shape[0], dtype=np.float64)]
    for _ in range(int(horizon)):
        powers.append(powers[-1] @ transition)
    # Attention tensors use [receiver, sender, feature].
    return np.stack(powers, axis=-1).transpose(1, 0, 2)


def _distances(adjacency: np.ndarray) -> np.ndarray:
    graph = nx.from_numpy_array(adjacency)
    result = np.full(adjacency.shape, np.inf, dtype=np.float64)
    for source, lengths in nx.all_pairs_shortest_path_length(graph):
        for target, distance in lengths.items():
            result[int(source), int(target)] = int(distance)
    return result


def _site_nodes(site: int) -> SiteNodes:
    start = 1 + 12 * int(site)
    return SiteNodes(
        anchor=start,
        reporter=start + 1,
        arm_left=tuple(range(start + 2, start + 6)),
        arm_right=tuple(range(start + 6, start + 10)),
        cue=(start + 10, start + 11),
    )


def _graph_arrays(
    cycle_state: Sequence[bool],
    cue_state: Sequence[bool],
) -> tuple[np.ndarray, np.ndarray, tuple[SiteNodes, ...]]:
    if len(cycle_state) != SITE_COUNT or len(cue_state) != SITE_COUNT:
        raise ValueError("every graph must specify all three sites")
    adjacency = np.zeros((NODE_COUNT, NODE_COUNT), dtype=np.float64)
    roles = np.full(NODE_COUNT, 3, dtype=np.int64)  # carbon-like arm context
    roles[0] = 0  # shared molecular hub
    sites = tuple(_site_nodes(site) for site in range(SITE_COUNT))

    def add_edge(left: int, right: int, weight: float = 1.0) -> None:
        adjacency[left, right] = float(weight)
        adjacency[right, left] = float(weight)

    for site, nodes in enumerate(sites):
        roles[nodes.anchor] = 1
        roles[nodes.reporter] = 2
        roles[list(nodes.cue)] = 4
        add_edge(0, nodes.anchor)
        add_edge(nodes.anchor, nodes.reporter)
        for arm in (nodes.arm_left, nodes.arm_right):
            previous = nodes.anchor
            for node in arm:
                add_edge(previous, node)
                previous = node
        if bool(cycle_state[site]):
            add_edge(nodes.arm_left[-1], nodes.arm_right[-1])
        # Both potential bonds are always present, so message support is fixed.
        # The cue is a local bond-order state carried by one reporter bond.
        add_edge(
            nodes.reporter,
            nodes.cue[0],
            weight=3.0 if bool(cue_state[site]) else 1.0,
        )
        add_edge(
            nodes.reporter,
            nodes.cue[1],
            weight=3.0 if bool(cue_state[site]) else 1.0,
        )
        add_edge(nodes.anchor, nodes.cue[0])
        add_edge(nodes.anchor, nodes.cue[1])
    if not nx.is_connected(nx.from_numpy_array(adjacency)):
        raise RuntimeError("registered molecular-sites graph is disconnected")
    return adjacency, roles, sites


def _bits(values: Sequence[bool]) -> int:
    return int(sum(int(bool(value)) << index for index, value in enumerate(values)))


def _type_index(cycle_state: Sequence[bool], cue_state: Sequence[bool]) -> int:
    return 8 * _bits(cycle_state) + _bits(cue_state)


def _states_from_index(index: int) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    cycle_bits, cue_bits = divmod(int(index), 8)
    cycle = tuple(bool(cycle_bits & (1 << site)) for site in range(SITE_COUNT))
    cue = tuple(bool(cue_bits & (1 << site)) for site in range(SITE_COUNT))
    return cycle, cue


def build_graph_types(config: Config) -> tuple[GraphType, ...]:
    """Register every independent cycle/cue configuration (8 x 8)."""
    result: list[GraphType] = []
    for index in range(64):
        cycle, cue = _states_from_index(index)
        adjacency, roles, sites = _graph_arrays(cycle, cue)
        result.append(
            GraphType(
                cycle_state=cycle,
                cue_state=cue,
                roles=roles,
                adjacency=adjacency,
                distances=_distances(adjacency),
                rrwp=_rrwp(adjacency, config.rrwp_horizon),
                sites=sites,
            )
        )
    for index, graph_type in enumerate(result):
        if _type_index(graph_type.cycle_state, graph_type.cue_state) != index:
            raise RuntimeError("graph-type registration order changed")
        for nodes in graph_type.sites:
            for endpoint in (nodes.arm_left[-1], nodes.arm_right[-1]):
                if graph_type.distances[nodes.reporter, endpoint] != 5:
                    raise RuntimeError("closure endpoint is not five bonds from reporter")
    return tuple(result)


def _bounded_values(
    rng: np.random.Generator,
    count: int,
    minimum_magnitude: float,
) -> np.ndarray:
    minimum = float(minimum_magnitude)
    magnitude = rng.uniform(minimum, 1.0, size=(int(count), SITE_COUNT))
    sign = rng.choice(np.asarray([-1.0, 1.0]), size=magnitude.shape)
    # Uniform magnitudes have E[m^2] = (1 + a + a^2) / 3.  Standardising
    # makes E[x^2]=1 while retaining a finite lower bound on event dose.
    rms = math.sqrt((1.0 + minimum + minimum**2) / 3.0)
    return (magnitude * sign / rms).astype(np.float32)


def make_dataset(config: Config, *, count: int, seed: int) -> Dataset:
    rng = np.random.default_rng(int(seed))
    cycle = rng.random((int(count), SITE_COUNT)) < 0.5
    correct = rng.random((int(count), SITE_COUNT)) < float(
        config.local_clue_reliability
    )
    cue = np.where(correct, cycle, ~cycle)
    cycle_bits = np.sum(
        cycle.astype(np.int64) << np.arange(SITE_COUNT, dtype=np.int64), axis=1
    )
    cue_bits = np.sum(
        cue.astype(np.int64) << np.arange(SITE_COUNT, dtype=np.int64), axis=1
    )
    shuffled_cycle = rng.integers(0, 8, size=int(count), dtype=np.int64)
    values = _bounded_values(
        rng, int(count), float(config.value_minimum_magnitude)
    )
    site_weight = (
        (1.0 - float(config.remote_target_weight)) * cue.astype(np.float32)
        + float(config.remote_target_weight) * cycle.astype(np.float32)
    )
    return Dataset(
        type_index=(8 * cycle_bits + cue_bits).astype(np.int64),
        shuffled_type_index=(8 * shuffled_cycle + cue_bits).astype(np.int64),
        values=values,
        cycle_state=cycle.astype(bool),
        cue_state=cue.astype(bool),
        site_contribution=(target_scale(config) * site_weight * values).astype(
            np.float32
        ),
    )


def _stack_graph_types(
    graph_types: Sequence[GraphType],
) -> dict[str, torch.Tensor]:
    adjacency = torch.tensor(
        np.stack([item.adjacency for item in graph_types]), dtype=torch.bool
    )
    identity = torch.eye(NODE_COUNT, dtype=torch.bool)[None]
    pair_mask = adjacency | identity
    raw_rrwp = np.stack([item.rrwp for item in graph_types])
    registered = raw_rrwp[pair_mask.numpy()]
    scale = np.std(registered, axis=0)
    scale[scale < 1.0e-6] = 1.0
    reporters = np.stack(
        [[site.reporter for site in graph_type.sites] for graph_type in graph_types]
    )
    return {
        "roles": torch.tensor(
            np.stack([item.roles for item in graph_types]), dtype=torch.long
        ),
        "pair_mask": pair_mask,
        "rrwp": torch.tensor(raw_rrwp / scale, dtype=DTYPE),
        "reporters": torch.tensor(reporters, dtype=torch.long),
        "route": torch.arange(NODE_COUNT, dtype=torch.long)[None].repeat(64, 1),
        "scale": torch.tensor(scale, dtype=DTYPE),
    }


def _batch_inputs(
    dataset: Dataset,
    indices: np.ndarray,
    tensors: dict[str, torch.Tensor],
    *,
    arm: str,
) -> dict[str, torch.Tensor]:
    type_index = torch.tensor(dataset.type_index[indices], dtype=torch.long)
    pe_type_index = type_index
    pair_pe = tensors["rrwp"][type_index].clone()
    if arm == "local":
        pair_pe[..., 2:] = 0.0
    elif arm == "global_shuffled":
        pe_type_index = torch.tensor(
            dataset.shuffled_type_index[indices], dtype=torch.long
        )
        shuffled = tensors["rrwp"][pe_type_index]
        # The control preserves the exact true I,P channels and replaces only
        # higher-order context.  In particular, closure-edge P entries are not
        # corrupted by the shuffled graph type.
        pair_pe[..., 2:] = shuffled[..., 2:]
    elif arm != "global":
        raise ValueError(f"unknown arm {arm!r}")
    roles = tensors["roles"][type_index]
    reporters = tensors["reporters"][type_index]
    values = torch.zeros(roles.shape, dtype=DTYPE)
    semantic = torch.tensor(dataset.values[indices], dtype=DTYPE)
    batch = torch.arange(len(indices))[:, None]
    values[batch, reporters] = semantic
    target = torch.tensor(
        dataset.site_contribution[indices].sum(axis=1), dtype=DTYPE
    )
    return {
        "roles": roles,
        "values": values,
        "pair_pe": pair_pe,
        "pair_mask": tensors["pair_mask"][type_index],
        "route_nodes": tensors["route"][type_index],
        "reporters": reporters,
        "target": target,
        "type_index": type_index,
        "pe_type_index": pe_type_index,
        "cycle_state": torch.tensor(dataset.cycle_state[indices], dtype=torch.bool),
        "cue_state": torch.tensor(dataset.cue_state[indices], dtype=torch.bool),
        "semantic": semantic,
    }


def _forward(
    model: RRWPSemanticRelay,
    inputs: dict[str, torch.Tensor],
    **kwargs: Any,
) -> Any:
    return model(
        inputs["roles"],
        inputs["values"],
        inputs["pair_pe"],
        inputs["pair_mask"],
        inputs["route_nodes"],
        **kwargs,
    )


def _model(config: Config, seed: int) -> RRWPSemanticRelay:
    return RRWPSemanticRelay(config, role_count=5, seed=int(seed)).to(dtype=DTYPE)


def train_model(
    config: Config,
    tensors: dict[str, torch.Tensor],
    train: Dataset,
    validation: Dataset,
    *,
    arm: str,
    seed: int,
) -> tuple[RRWPSemanticRelay, dict[str, float | int]]:
    model = _model(config, int(seed))
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    rng = np.random.default_rng(config.data_seed + 200_000 + 131 * int(seed))
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for step in range(1, int(config.training_steps) + 1):
        index = rng.integers(
            0, len(train.type_index), size=int(config.batch_size), dtype=np.int64
        )
        inputs = _batch_inputs(train, index, tensors, arm=arm)
        model.train()
        prediction = _forward(model, inputs)
        loss = torch.mean((prediction - inputs["target"]) ** 2)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if step % 20 and step != int(config.training_steps):
            continue
        model.eval()
        with torch.no_grad():
            validation_index = np.arange(len(validation.type_index), dtype=np.int64)
            validation_inputs = _batch_inputs(
                validation, validation_index, tensors, arm=arm
            )
            validation_prediction = _forward(model, validation_inputs)
            validation_loss = float(
                torch.mean(
                    (validation_prediction - validation_inputs["target"]) ** 2
                )
            )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= 12:
            break
    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "steps": int(step),
        "best_validation_mse": float(best_loss),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def evaluate(
    model: RRWPSemanticRelay,
    dataset: Dataset,
    tensors: dict[str, torch.Tensor],
    *,
    input_arm: str,
    label: str,
    training_arm: str,
    seed: int,
) -> dict[str, Any]:
    index = np.arange(len(dataset.type_index), dtype=np.int64)
    inputs = _batch_inputs(dataset, index, tensors, arm=input_arm)
    with torch.no_grad():
        prediction = _forward(model, inputs)
    error = prediction - inputs["target"]
    return {
        "seed": int(seed),
        "arm": str(label),
        "training_arm": str(training_arm),
        "input_arm": str(input_arm),
        "test_mse": float(torch.mean(error.square())),
        "test_mae": float(torch.mean(torch.abs(error))),
    }


def measure_selection_fidelity(
    model: RRWPSemanticRelay,
    dataset: Dataset,
    tensors: dict[str, torch.Tensor],
    *,
    input_arm: str,
    label: str,
    training_arm: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Measure antisymmetric site contribution using an in-support sign flip."""
    index = np.arange(len(dataset.type_index), dtype=np.int64)
    clean = _batch_inputs(dataset, index, tensors, arm=input_arm)
    with torch.no_grad():
        clean_prediction = _forward(model, clean)
    result: list[dict[str, Any]] = []
    for site in range(SITE_COUNT):
        event = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in clean.items()
        }
        batch = torch.arange(len(index))
        reporter = clean["reporters"][:, site]
        event["values"][batch, reporter] *= -1.0
        with torch.no_grad():
            event_prediction = _forward(model, event)
        delta = (0.5 * (clean_prediction - event_prediction)).cpu().numpy()
        truth = dataset.site_contribution[:, site]
        for graph in range(len(index)):
            result.append(
                {
                    "seed": int(seed),
                    "graph": int(graph),
                    "arm": str(label),
                    "training_arm": str(training_arm),
                    "input_arm": str(input_arm),
                    "site": int(site),
                    "cycle": bool(dataset.cycle_state[graph, site]),
                    "cue": bool(dataset.cue_state[graph, site]),
                    "cue_correct": bool(
                        dataset.cycle_state[graph, site]
                        == dataset.cue_state[graph, site]
                    ),
                    "x": float(dataset.values[graph, site]),
                    "fidelity_intervention": "symmetric_sign_flip",
                    "delta_i": float(delta[graph]),
                    "true_contribution": float(truth[graph]),
                    "error": float(delta[graph] - truth[graph]),
                }
            )
    return result


def _toggle_type_indices(
    type_index: torch.Tensor,
    *,
    site: int,
    toggle_cycle: bool,
    toggle_cue: bool,
) -> torch.Tensor:
    cycle = torch.div(type_index, 8, rounding_mode="floor")
    cue = torch.remainder(type_index, 8)
    bit = 1 << int(site)
    if toggle_cycle:
        cycle = torch.bitwise_xor(cycle, bit)
    if toggle_cue:
        cue = torch.bitwise_xor(cue, bit)
    return 8 * cycle + cue


def _event_inputs(
    clean: dict[str, torch.Tensor],
    tensors: dict[str, torch.Tensor],
    *,
    arm: str,
    channel: str,
    site: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    event = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in clean.items()
    }
    batch = torch.arange(clean["roles"].shape[0])
    reporter = clean["reporters"][:, int(site)]
    if channel == "semantic":
        event["values"][batch, reporter] *= -1.0
        dose = 2.0 * torch.abs(clean["values"][batch, reporter])
        return event, dose

    toggles = {
        "structural_full": (True, True),
        "structural_proxy": (False, True),
        "structural_remote": (True, False),
    }
    if channel not in toggles:
        raise ValueError(f"unknown event channel {channel!r}")
    toggle_cycle, toggle_cue = toggles[channel]
    donor_index = _toggle_type_indices(
        clean["type_index"],
        site=int(site),
        toggle_cycle=toggle_cycle,
        toggle_cue=toggle_cue,
    )
    donor = tensors["rrwp"][donor_index].clone()
    if arm == "local":
        donor[..., 2:] = 0.0
    nodes = torch.arange(clean["roles"].shape[1])[None, :]
    footprint = (nodes[:, :, None] == reporter[:, None, None]) | (
        nodes[:, None, :] == reporter[:, None, None]
    )
    footprint = footprint & clean["pair_mask"]
    event["pair_pe"][footprint] = donor[footprint]
    visible_difference = (
        (clean["pair_pe"] - event["pair_pe"])
        * clean["pair_mask"][..., None]
    )
    dose = torch.linalg.vector_norm(visible_difference.flatten(start_dim=1), dim=1)
    return event, dose


def _scaled_rows(
    *,
    base: dict[str, Any],
    value: float,
    dose: float,
) -> list[dict[str, Any]]:
    zero = bool(dose <= 1.0e-10)
    return [
        {
            **base,
            "scale": "raw",
            "value": float(value),
            "event_dose": float(dose),
            "zero_dose": zero,
        },
        {
            **base,
            "scale": "per_unit",
            "value": float("nan") if zero else float(value / dose),
            "event_dose": float(dose),
            "zero_dose": zero,
        },
    ]


def measure_scores_and_carriage(
    model: RRWPSemanticRelay,
    dataset: Dataset,
    graph_types: Sequence[GraphType],
    tensors: dict[str, torch.Tensor],
    config: Config,
    *,
    arm: str,
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Measure exact native-head scores, carriage, dose, and site fidelity."""
    index = np.arange(int(config.measurement_examples), dtype=np.int64)
    clean = _batch_inputs(dataset, index, tensors, arm=arm)
    model.zero_grad(set_to_none=True)
    prediction, _, final_hidden, routed, _ = _forward(
        model, clean, return_details=True
    )
    gradients = torch.autograd.grad(
        prediction.sum(), (*routed, final_hidden), allow_unused=False
    )
    routed_gradient = tuple(value.detach() for value in gradients[:-1])
    final_gradient = gradients[-1].detach()
    clean_routed = tuple(value.detach() for value in routed)
    clean_final = final_hidden.detach()
    distance_matrices = np.stack(
        [graph_types[int(value)].distances for value in dataset.type_index[index]]
    )
    maximum_distance = int(config.layers)

    score_rows: list[dict[str, Any]] = []
    carriage_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    dose_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []

    for channel in CHANNELS:
        for site in range(SITE_COUNT):
            event, dose_tensor = _event_inputs(
                clean, tensors, arm=arm, channel=channel, site=site
            )
            with torch.no_grad():
                event_prediction, _, event_final, event_routed, _ = _forward(
                    model, event, return_details=True
                )
            doses = dose_tensor.detach().cpu().numpy()
            source_nodes = clean["reporters"][:, site].cpu().numpy()
            for graph_position, dose in enumerate(doses):
                dose_rows.append(
                    {
                        "seed": int(seed),
                        "graph": int(graph_position),
                        "arm": str(arm),
                        "channel": str(channel),
                        "site": int(site),
                        "cycle": bool(
                            dataset.cycle_state[index, site][graph_position]
                        ),
                        "cue": bool(
                            dataset.cue_state[index, site][graph_position]
                        ),
                        "event_dose": float(dose),
                        "zero_dose": bool(dose <= 1.0e-10),
                    }
                )
            if channel == "semantic":
                delta = (
                    0.5 * (prediction.detach() - event_prediction)
                ).cpu().numpy()
                truth = dataset.site_contribution[index, site]
                for graph_position in range(len(index)):
                    fidelity_rows.append(
                        {
                            "seed": int(seed),
                            "graph": int(graph_position),
                            "arm": str(arm),
                            "site": int(site),
                            "cycle": bool(
                                dataset.cycle_state[index, site][graph_position]
                            ),
                            "cue": bool(
                                dataset.cue_state[index, site][graph_position]
                            ),
                            "cue_correct": bool(
                                dataset.cycle_state[index, site][graph_position]
                                == dataset.cue_state[index, site][graph_position]
                            ),
                            "x": float(dataset.values[index, site][graph_position]),
                            "fidelity_intervention": "symmetric_sign_flip",
                            "delta_i": float(delta[graph_position]),
                            "true_contribution": float(truth[graph_position]),
                            "error": float(delta[graph_position] - truth[graph_position]),
                        }
                    )

            for layer, (clean_value, event_value, gradient) in enumerate(
                zip(clean_routed, event_routed, routed_gradient, strict=True)
            ):
                projected = torch.abs(
                    torch.sum((clean_value - event_value) * gradient, dim=-1)
                ).cpu().numpy()
                for graph_position in range(len(index)):
                    source = int(source_nodes[graph_position])
                    carrier_distance = distance_matrices[graph_position, source]
                    for head in range(config.heads):
                        total = float(projected[graph_position, :, head].sum())
                        binned_total = float(
                            projected[
                                graph_position,
                                carrier_distance <= maximum_distance,
                                head,
                            ].sum()
                        )
                        if not np.isclose(
                            binned_total, total, atol=1.0e-6, rtol=1.0e-5
                        ):
                            raise RuntimeError(
                                "score-distance bins do not reconstruct head score"
                            )
                        head_rows.extend(
                            _scaled_rows(
                                base={
                                    "seed": int(seed),
                                    "graph": int(graph_position),
                                    "arm": str(arm),
                                    "channel": str(channel),
                                    "site": int(site),
                                    "layer": int(layer),
                                    "head": int(head),
                                },
                                value=total,
                                dose=float(doses[graph_position]),
                            )
                        )
                    for distance in range(maximum_distance + 1):
                        mask = carrier_distance == distance
                        contribution = projected[graph_position, mask].sum(axis=0)
                        for head in range(config.heads):
                            score_rows.extend(
                                _scaled_rows(
                                    base={
                                        "seed": int(seed),
                                        "graph": int(graph_position),
                                        "arm": str(arm),
                                        "channel": str(channel),
                                        "site": int(site),
                                        "layer": int(layer),
                                        "head": int(head),
                                        "distance": int(distance),
                                    },
                                    value=float(contribution[head]),
                                    dose=float(doses[graph_position]),
                                )
                            )

            final_projected = torch.abs(
                torch.sum((clean_final - event_final) * final_gradient, dim=-1)
            ).cpu().numpy()
            signed_complete = torch.sum(
                (clean_final - event_final) * final_gradient, dim=(-1, -2)
            )
            if not torch.allclose(
                signed_complete,
                prediction.detach() - event_prediction,
                atol=3.0e-5,
                rtol=3.0e-5,
            ):
                raise RuntimeError("Functional carriage does not reconstruct output")
            for graph_position in range(len(index)):
                source = int(source_nodes[graph_position])
                carrier_distance = distance_matrices[graph_position, source]
                total_carriage = float(final_projected[graph_position].sum())
                binned_carriage = float(
                    final_projected[
                        graph_position,
                        carrier_distance <= maximum_distance,
                    ].sum()
                )
                if not np.isclose(
                    binned_carriage,
                    total_carriage,
                    atol=1.0e-6,
                    rtol=1.0e-5,
                ):
                    raise RuntimeError(
                        "carriage-distance bins do not reconstruct total mass"
                    )
                for distance in range(maximum_distance + 1):
                    contribution = float(
                        final_projected[
                            graph_position, carrier_distance == distance
                        ].sum()
                    )
                    carriage_rows.extend(
                        _scaled_rows(
                            base={
                                "seed": int(seed),
                                "graph": int(graph_position),
                                "arm": str(arm),
                                "channel": str(channel),
                                "site": int(site),
                                "distance": int(distance),
                            },
                            value=contribution,
                            dose=float(doses[graph_position]),
                        )
                    )
    return score_rows, carriage_rows, head_rows, dose_rows, fidelity_rows


def _stable_seed(key: tuple[Any, ...], offset: int, data_seed: int) -> int:
    digest = hashlib.sha256(repr(tuple(key)).encode("utf-8")).hexdigest()
    return int(data_seed + offset + int(digest[:8], 16) % 100_000)


def _nested_interval(
    rows: pd.DataFrame,
    *,
    value_column: str,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap training seeds, then graphs (with sites clustered) per seed."""
    grouped: dict[int, np.ndarray] = {}
    for training_seed, seed_rows in rows.groupby("seed"):
        values = (
            seed_rows.groupby("graph")[value_column]
            .mean()
            .dropna()
            .to_numpy(dtype=np.float64)
        )
        if len(values):
            grouped[int(training_seed)] = values
    if not grouped:
        return float("nan"), float("nan"), float("nan")
    seeds = np.asarray(sorted(grouped), dtype=np.int64)
    point = float(np.mean([grouped[int(item)].mean() for item in seeds]))
    rng = np.random.default_rng(int(seed))
    within = np.empty((len(seeds), int(replicates)), dtype=np.float64)
    for position, training_seed in enumerate(seeds):
        values = grouped[int(training_seed)]
        sampled = rng.integers(0, len(values), size=(int(replicates), len(values)))
        within[position] = values[sampled].mean(axis=1)
    outer = rng.integers(
        0, len(seeds), size=(int(replicates), len(seeds)), dtype=np.int64
    )
    draws = within[outer, np.arange(int(replicates))[:, None]].mean(axis=1)
    low, high = np.percentile(draws, (2.5, 97.5))
    return point, float(low), float(high)


def summarise_scores(
    score_distance: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Summarise architecture profiles after aggregating exchangeable heads."""
    keys = [
        "seed",
        "graph",
        "site",
        "arm",
        "channel",
        "scale",
        "distance",
    ]
    # Heads are permutation-symmetric between independent fits. Aggregate
    # layer/head responses inside each fitted model before seed-level inference.
    graph_rows = score_distance.groupby(keys, as_index=False)["value"].mean()
    profile_keys = keys[:-1]
    denominator = graph_rows.groupby(profile_keys)["value"].transform("sum")
    graph_rows["profile_value"] = np.where(
        denominator > 1.0e-12,
        graph_rows["value"] / denominator,
        np.nan,
    )
    output: list[dict[str, Any]] = []
    summary_keys = ["arm", "channel", "scale", "distance"]
    for key, rows in graph_rows.groupby(summary_keys, sort=True, dropna=False):
        mean, low, high = _nested_interval(
            rows,
            value_column="value",
            replicates=config.bootstrap_replicates,
            seed=_stable_seed(tuple(key), 0, config.data_seed),
        )
        profile_mean, profile_low, profile_high = _nested_interval(
            rows,
            value_column="profile_value",
            replicates=config.bootstrap_replicates,
            seed=_stable_seed(tuple(key), 100_000, config.data_seed),
        )
        within_values = []
        seed_profile_means = []
        for _, seed_rows in rows.groupby("seed"):
            values = (
                seed_rows.groupby("graph")["profile_value"]
                .mean()
                .dropna()
                .to_numpy()
            )
            if len(values) > 1:
                within_values.append(float(np.std(values, ddof=1)))
            if len(values):
                seed_profile_means.append(float(np.mean(values)))
        between = (
            float(np.std(seed_profile_means, ddof=1))
            if len(seed_profile_means) > 1
            else 0.0
        )
        output.append(
            {
                "arm": str(key[0]),
                "channel": str(key[1]),
                "scale": str(key[2]),
                "distance": int(key[3]),
                "mean": mean,
                "low": low,
                "high": high,
                "ci_width": high - low,
                "profile_mean": profile_mean,
                "profile_low": profile_low,
                "profile_high": profile_high,
                "profile_ci_width": profile_high - profile_low,
                "within_seed_profile_sd": float(np.mean(within_values))
                if within_values
                else float("nan"),
                "between_seed_profile_sd": between,
                "valid_graph_sites": int(rows["profile_value"].notna().sum()),
            }
        )
    return pd.DataFrame(output)


def summarise_carriage(
    carriage: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    keys = ["seed", "graph", "site", "arm", "channel", "scale", "distance"]
    graph_rows = carriage.groupby(keys, as_index=False)["value"].mean()
    denominator = graph_rows.groupby(keys[:-1])["value"].transform("sum")
    graph_rows["profile_value"] = np.where(
        denominator > 1.0e-12,
        graph_rows["value"] / denominator,
        np.nan,
    )
    output: list[dict[str, Any]] = []
    for key, rows in graph_rows.groupby(
        ["arm", "channel", "scale", "distance"], sort=True, dropna=False
    ):
        mean, low, high = _nested_interval(
            rows,
            value_column="value",
            replicates=config.bootstrap_replicates,
            seed=_stable_seed(tuple(key), 200_000, config.data_seed),
        )
        profile_mean, profile_low, profile_high = _nested_interval(
            rows,
            value_column="profile_value",
            replicates=config.bootstrap_replicates,
            seed=_stable_seed(tuple(key), 300_000, config.data_seed),
        )
        output.append(
            {
                "arm": str(key[0]),
                "channel": str(key[1]),
                "scale": str(key[2]),
                "distance": int(key[3]),
                "mean": mean,
                "low": low,
                "high": high,
                "ci_width": high - low,
                "profile_mean": profile_mean,
                "profile_low": profile_low,
                "profile_high": profile_high,
                "profile_ci_width": profile_high - profile_low,
            }
        )
    return pd.DataFrame(output)


def summarise_head_scores(
    head_scores: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Summarise mean head response after within-fit head aggregation."""
    graph_rows = (
        head_scores.groupby(
            ["seed", "graph", "site", "arm", "channel", "scale"],
            as_index=False,
        )["value"]
        .mean()
    )
    output: list[dict[str, Any]] = []
    keys = ["arm", "channel", "scale"]
    for key, rows in graph_rows.groupby(keys, sort=True, dropna=False):
        mean, low, high = _nested_interval(
            rows,
            value_column="value",
            replicates=config.bootstrap_replicates,
            seed=_stable_seed(tuple(key), 400_000, config.data_seed),
        )
        output.append(
            {
                "arm": str(key[0]),
                "channel": str(key[1]),
                "scale": str(key[2]),
                "mean": mean,
                "low": low,
                "high": high,
                "ci_width": high - low,
            }
        )
    return pd.DataFrame(output)


def compute_profile_alignment(score_distance: pd.DataFrame) -> pd.DataFrame:
    """Compare matched heads with every other head in the same layer."""
    rows = score_distance[
        (score_distance["scale"] == "raw")
        & score_distance["channel"].isin(CHANNELS)
    ]
    grouped = (
        rows.groupby(
            [
                "seed",
                "graph",
                "site",
                "arm",
                "channel",
                "layer",
                "head",
                "distance",
            ],
            as_index=False,
        )["value"]
        .mean()
    )
    result: list[dict[str, Any]] = []
    unit_keys = ["seed", "graph", "site", "arm", "layer"]
    for unit_key, unit in grouped.groupby(unit_keys, sort=True):
        vectors: dict[tuple[str, int], np.ndarray] = {}
        for (channel, head), vector_rows in unit.groupby(["channel", "head"]):
            vectors[(str(channel), int(head))] = (
                vector_rows.sort_values("distance")["value"].to_numpy(dtype=np.float64)
            )
        for structural_channel in STRUCTURAL_CHANNELS:
            for semantic_head in range(int(grouped["head"].max()) + 1):
                semantic = vectors.get(("semantic", semantic_head))
                if semantic is None:
                    continue
                for structural_head in range(int(grouped["head"].max()) + 1):
                    structural = vectors.get((structural_channel, structural_head))
                    if structural is None or len(structural) != len(semantic):
                        continue
                    pairing = (
                        "matched" if semantic_head == structural_head else "other_head"
                    )
                    denominator = float(
                        np.linalg.norm(semantic) * np.linalg.norm(structural)
                    )
                    valid = bool(denominator > 1.0e-12)
                    result.append(
                        {
                            "seed": int(unit_key[0]),
                            "graph": int(unit_key[1]),
                            "site": int(unit_key[2]),
                            "arm": str(unit_key[3]),
                            "layer": int(unit_key[4]),
                            "semantic_head": int(semantic_head),
                            "structural_head": int(structural_head),
                            "structural_channel": str(structural_channel),
                            "pairing": pairing,
                            "cosine": float(np.dot(semantic, structural) / denominator)
                            if valid
                            else float("nan"),
                            "peak_match": float(
                                np.argmax(semantic) == np.argmax(structural)
                            )
                            if valid
                            else float("nan"),
                            "semantic_mass": float(semantic.sum()),
                            "structural_mass": float(structural.sum()),
                            "valid": valid,
                        }
                    )
    return pd.DataFrame(result)


def summarise_alignment_by_seed(alignment: pd.DataFrame) -> pd.DataFrame:
    if alignment.empty:
        return pd.DataFrame()
    return (
        alignment.groupby(
            ["seed", "arm", "structural_channel", "pairing"], as_index=False
        )
        .agg(
            cosine_mean=("cosine", "mean"),
            peak_match_fraction=("peak_match", "mean"),
            valid_comparisons=("cosine", "count"),
            total_comparisons=("valid", "size"),
        )
    )


def summarise_alignment(alignment_by_seed: pd.DataFrame) -> pd.DataFrame:
    if alignment_by_seed.empty:
        return pd.DataFrame()
    output: list[dict[str, Any]] = []
    for key, rows in alignment_by_seed.groupby(
        ["arm", "structural_channel", "pairing"], sort=True
    ):
        cosine_mean, cosine_interval = _mean_seed_interval(
            rows["cosine_mean"].dropna()
        )
        peak_mean, peak_interval = _mean_seed_interval(
            rows["peak_match_fraction"].dropna()
        )
        output.append(
            {
                "arm": str(key[0]),
                "structural_channel": str(key[1]),
                "pairing": str(key[2]),
                "cosine_mean": cosine_mean,
                "cosine_seed_sd": float(rows["cosine_mean"].std(ddof=1)),
                "cosine_ci_half_width": cosine_interval,
                "peak_match_fraction": peak_mean,
                "peak_match_seed_sd": float(
                    rows["peak_match_fraction"].std(ddof=1)
                ),
                "peak_match_ci_half_width": peak_interval,
                "valid_comparisons": int(rows["valid_comparisons"].sum()),
                "total_comparisons": int(rows["total_comparisons"].sum()),
            }
        )
    return pd.DataFrame(output)


def alignment_excess_by_seed(alignment_by_seed: pd.DataFrame) -> pd.DataFrame:
    if alignment_by_seed.empty:
        return pd.DataFrame()
    pivot = alignment_by_seed.pivot(
        index=["seed", "arm", "structural_channel"],
        columns="pairing",
        values=["cosine_mean", "peak_match_fraction"],
    )
    output = pivot.index.to_frame(index=False)
    output["cosine_excess"] = (
        pivot[("cosine_mean", "matched")]
        - pivot[("cosine_mean", "other_head")]
    ).to_numpy()
    output["peak_match_excess"] = (
        pivot[("peak_match_fraction", "matched")]
        - pivot[("peak_match_fraction", "other_head")]
    ).to_numpy()
    return output


def summarise_alignment_excess(excess_by_seed: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for key, rows in excess_by_seed.groupby(
        ["arm", "structural_channel"], sort=True
    ):
        cosine_mean, cosine_interval = _mean_seed_interval(
            rows["cosine_excess"].dropna()
        )
        peak_mean, peak_interval = _mean_seed_interval(
            rows["peak_match_excess"].dropna()
        )
        output.append(
            {
                "arm": str(key[0]),
                "structural_channel": str(key[1]),
                "cosine_excess": cosine_mean,
                "cosine_excess_ci_half_width": cosine_interval,
                "peak_match_excess": peak_mean,
                "peak_match_excess_ci_half_width": peak_interval,
                "seeds": int(rows["seed"].nunique()),
            }
        )
    return pd.DataFrame(output)


def summarise_fidelity_by_seed(fidelity: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for (seed, arm, stratum), rows in fidelity.assign(
        stratum=np.where(fidelity["cue_correct"], "cue_correct", "cue_wrong")
    ).groupby(["seed", "arm", "stratum"], sort=True):
        truth = rows["true_contribution"].to_numpy(dtype=np.float64)
        fitted = rows["delta_i"].to_numpy(dtype=np.float64)
        denominator = float(np.dot(truth, truth))
        output.append(
            {
                "seed": int(seed),
                "arm": str(arm),
                "stratum": str(stratum),
                "count": len(rows),
                "fidelity_mse": float(np.mean((fitted - truth) ** 2)),
                "fidelity_mae": float(np.mean(np.abs(fitted - truth))),
                "correlation": float(np.corrcoef(truth, fitted)[0, 1])
                if np.std(truth) > 1.0e-12 and np.std(fitted) > 1.0e-12
                else float("nan"),
                "origin_slope": float(np.dot(truth, fitted) / denominator)
                if denominator > 1.0e-12
                else float("nan"),
            }
        )
    return pd.DataFrame(output)


def summarise_fidelity(fidelity_by_seed: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    metrics = ("fidelity_mse", "fidelity_mae", "correlation", "origin_slope")
    for (arm, stratum), rows in fidelity_by_seed.groupby(
        ["arm", "stratum"], sort=True
    ):
        result: dict[str, Any] = {
            "arm": str(arm),
            "stratum": str(stratum),
            "count": int(rows["count"].sum()),
            "seeds": int(rows["seed"].nunique()),
        }
        for metric in metrics:
            mean, interval = _mean_seed_interval(rows[metric].dropna())
            result[metric] = mean
            result[f"{metric}_seed_sd"] = float(rows[metric].std(ddof=1))
            result[f"{metric}_ci_half_width"] = interval
        output.append(result)
    return pd.DataFrame(output)


def summarise_specialisation(head_scores: pd.DataFrame) -> pd.DataFrame:
    """Compute relative semantic/full-structural balance for each learned head."""
    selected = head_scores[
        head_scores["channel"].isin(("semantic", "structural_full"))
    ]
    pivot = (
        selected.groupby(["seed", "arm", "scale", "layer", "head", "channel"])[
            "value"
        ]
        .mean()
        .unstack("channel")
        .reset_index()
    )
    for channel in ("semantic", "structural_full"):
        channel_mean = pivot.groupby(["seed", "arm", "scale"])[channel].transform(
            "mean"
        )
        pivot[f"{channel}_normalized"] = pivot[channel] / channel_mean.clip(
            lower=1.0e-12
        )
    total = pivot["semantic_normalized"] + pivot["structural_full_normalized"]
    pivot["D_rel"] = (
        pivot["semantic_normalized"] - pivot["structural_full_normalized"]
    ) / (total + 1.0e-12)
    pivot["regime"] = np.where(
        pivot["D_rel"] > 0.5,
        "semantic-leaning",
        np.where(pivot["D_rel"] < -0.5, "structural-leaning", "generalist"),
    )
    return pivot


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
        }
    )


def _save_figure(
    figure: plt.Figure,
    stem: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / f"{stem}.png"
    pdf = figures / f"{stem}.pdf"
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _mean_seed_interval(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=np.float64)
    if len(array) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, 0.0
    critical = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }.get(len(array) - 1, 1.96)
    return mean, float(
        critical * np.std(array, ddof=1) / math.sqrt(len(array))
    )


def plot_core_questions(
    performance: pd.DataFrame,
    fidelity_by_seed: pd.DataFrame,
    head_scores: pd.DataFrame,
    carriage_summary: pd.DataFrame,
    config: Config,
) -> tuple[Path, Path]:
    _configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    figure.suptitle(
        "Molecular-sites synthetic: performance, selection, scores, and carriage",
        fontsize=14,
        fontweight="bold",
    )

    order = ["local", "global", "global_shuffled", "global_test_shuffled"]
    labels = ["Local", "Global", "Shuffled\ntrained", "Global @\nshuffled"]
    means, errors = [], []
    for arm in order:
        mean, error = _mean_seed_interval(
            performance.loc[performance["arm"] == arm, "test_mse"]
        )
        means.append(mean)
        errors.append(error)
    axis = axes[0, 0]
    axis.bar(
        np.arange(len(order)),
        means,
        yerr=errors,
        color=[ARM_COLOURS[item] for item in order],
        capsize=3,
    )
    axis.patches[-1].set_hatch("///")
    axis.patches[-1].set_facecolor("white")
    axis.patches[-1].set_edgecolor(PURPLE)
    axis.axvline(2.5, color=LIGHT_GREY, linewidth=1)
    axis.text(
        2.75,
        0.98,
        "post-training\ncausal test",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.5,
        color=PURPLE,
    )
    axis.axhline(
        cue_only_bayes_mse(config),
        color=GREY,
        linestyle="--",
        linewidth=1,
        label="analytic cue-only Bayes MSE",
    )
    axis.set_xticks(np.arange(len(order)), labels)
    axis.set_ylabel("held-out MSE")
    axis.set_title("a  Does useful higher-order RRWP improve performance?")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[0, 1]
    strata = ("cue_correct", "cue_wrong")
    x = np.arange(len(strata), dtype=float)
    offsets = {"local": -0.12, "global": 0.12}
    for arm, colour in (("local", ORANGE), ("global", BLUE)):
        rows = fidelity_by_seed[fidelity_by_seed["arm"] == arm]
        values_by_stratum = []
        for stratum in strata:
            values = rows.loc[
                rows["stratum"] == stratum, "fidelity_mse"
            ].to_numpy(dtype=float)
            values_by_stratum.append(values)
            axis.scatter(
                np.full(len(values), x[len(values_by_stratum) - 1] + offsets[arm]),
                values,
                s=24,
                alpha=0.65,
                color=colour,
                edgecolors="none",
            )
        means = [float(np.mean(values)) for values in values_by_stratum]
        axis.scatter(
            x + offsets[arm],
            means,
            marker="D",
            s=38,
            color=colour,
            label=arm.capitalize(),
            zorder=4,
        )
    axis.set_xticks(x, ("local cue correct", "local cue wrong"))
    axis.set_ylabel("in-support site-selection MSE")
    axis.set_title("b  Does structure guide the right semantic contribution?")
    axis.legend(frameon=False)

    axis = axes[1, 0]
    raw = head_scores[
        (head_scores["scale"] == "raw")
        & head_scores["channel"].isin(("semantic", "structural_full"))
    ].groupby(
        ["seed", "arm", "channel", "layer", "head"], as_index=False
    )["value"].mean()
    positions = {
        ("local", "semantic"): 0.0,
        ("local", "structural_full"): 0.8,
        ("global", "semantic"): 2.0,
        ("global", "structural_full"): 2.8,
    }
    box_values, box_positions, colours = [], [], []
    for key, position in positions.items():
        values = raw.loc[
            (raw["arm"] == key[0]) & (raw["channel"] == key[1]), "value"
        ].to_numpy(dtype=float)
        box_values.append(values)
        box_positions.append(position)
        colours.append(CHANNEL_COLOURS[key[1]])
    boxes = axis.boxplot(
        box_values,
        positions=box_positions,
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#222222"},
    )
    for patch, colour in zip(boxes["boxes"], colours, strict=True):
        patch.set_facecolor(colour)
        patch.set_alpha(0.72)
    axis.set_xticks((0.4, 2.4), ("Local", "Global"))
    axis.set_ylabel("raw native-head score")
    axis.set_title("c  How do raw score magnitudes compare?")
    axis.legend(
        handles=[
            plt.Line2D([], [], color=BLUE, linewidth=7, label="semantic"),
            plt.Line2D([], [], color=ORANGE, linewidth=7, label="full structural"),
        ],
        frameon=False,
        fontsize=8,
    )

    axis = axes[1, 1]
    for arm in MEASURED_ARMS:
        rows = carriage_summary[
            (carriage_summary["arm"] == arm)
            & (carriage_summary["channel"] == "semantic")
            & (carriage_summary["scale"] == "raw")
        ].sort_values("distance")
        raw_total = float(rows["mean"].sum())
        values = rows["profile_mean"].to_numpy(dtype=float)
        values = values / max(float(values.sum()), 1.0e-12)
        axis.plot(
            rows["distance"],
            values,
            marker="o",
            color=ARM_COLOURS[arm],
            label=f"{arm.capitalize()} (total={raw_total:.3f})",
        )
    axis.set_xlabel("reporter-to-carrier distance")
    axis.set_ylabel("normalized semantic carriage")
    axis.set_title("d  Does semantic carriage geometry match?")
    axis.set_xticks(range(config.layers + 1))
    axis.legend(frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_figure(figure, "01_core_scientific_questions", config.output_dir)


def plot_distance_and_confidence(
    score_summary: pd.DataFrame,
    dose: pd.DataFrame,
    config: Config,
) -> tuple[Path, Path]:
    _configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    figure.suptitle(
        "Where do learned scores differ, and how variable are their profiles?",
        fontsize=14,
        fontweight="bold",
    )
    for axis, channel, letter in (
        (axes[0, 0], "semantic", "a"),
        (axes[0, 1], "structural_full", "b"),
    ):
        for arm in MEASURED_ARMS:
            rows = score_summary[
                (score_summary["arm"] == arm)
                & (score_summary["channel"] == channel)
                & (score_summary["scale"] == "raw")
            ]
            rows = rows.groupby("distance", as_index=False)["mean"].mean()
            axis.plot(
                rows["distance"],
                rows["mean"],
                marker="o",
                color=ARM_COLOURS[arm],
                label=arm.capitalize(),
            )
        axis.set_xlabel("reporter-to-carrier distance")
        axis.set_ylabel("mean raw score mass")
        axis.set_xticks(range(config.layers + 1))
        axis.set_title(
            f"{letter}  {channel.replace('_', ' ').capitalize()} score distance"
        )
        axis.legend(frameon=False)

    axis = axes[1, 0]
    variability = (
        score_summary[score_summary["scale"] == "raw"]
        .groupby(["arm", "channel"], as_index=False)
        .agg(
            profile_ci_width=("profile_ci_width", "mean"),
            within_seed_sd=("within_seed_profile_sd", "mean"),
            between_seed_sd=("between_seed_profile_sd", "mean"),
        )
    )
    x = np.arange(len(CHANNELS), dtype=float)
    width = 0.34
    for offset, arm in ((-width / 2, "local"), (width / 2, "global")):
        rows = variability.set_index(["arm", "channel"])
        values = [rows.loc[(arm, channel), "profile_ci_width"] for channel in CHANNELS]
        axis.bar(
            x + offset,
            values,
            width=width,
            color=ARM_COLOURS[arm],
            label=arm.capitalize(),
        )
    axis.set_xticks(
        x,
        ("semantic", "full", "proxy", "remote"),
        rotation=18,
        ha="right",
    )
    axis.set_ylabel("mean 95% profile-CI width")
    axis.set_title("c  Is either architecture less stable across fits/graphs?")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    dose_summary = (
        dose.groupby(["arm", "channel"], as_index=False)
        .agg(mean_dose=("event_dose", "mean"), zero_fraction=("zero_dose", "mean"))
    )
    x = np.arange(len(CHANNELS), dtype=float)
    for offset, arm in ((-width / 2, "local"), (width / 2, "global")):
        rows = dose_summary.set_index(["arm", "channel"])
        values = [rows.loc[(arm, channel), "mean_dose"] for channel in CHANNELS]
        bars = axis.bar(
            x + offset,
            values,
            width=width,
            color=ARM_COLOURS[arm],
            label=arm.capitalize(),
        )
        for bar, channel in zip(bars, CHANNELS, strict=True):
            zero_fraction = rows.loc[(arm, channel), "zero_fraction"]
            if zero_fraction > 0:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{zero_fraction:.0%} zero",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
    axis.set_xticks(
        x,
        ("semantic", "full", "proxy", "remote"),
        rotation=18,
        ha="right",
    )
    axis.set_ylabel("natural input-event L2 dose")
    axis.set_title("d  Are raw effects confounded by event dose?")
    axis.legend(frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_figure(
        figure, "02_distance_confidence_and_dose", config.output_dir
    )


def _bar_score_means(
    axis: plt.Axes,
    head_summary: pd.DataFrame,
    *,
    scale: str,
    channels: tuple[str, ...],
    title: str,
) -> None:
    subset = head_summary[
        (head_summary["scale"] == scale)
        & head_summary["channel"].isin(channels)
    ]
    x = np.arange(len(channels), dtype=float)
    width = 0.34
    for offset, arm in ((-width / 2, "local"), (width / 2, "global")):
        indexed = subset[subset["arm"] == arm].set_index("channel")
        values = [
            float(indexed.loc[channel, "mean"])
            if channel in indexed.index
            else np.nan
            for channel in channels
        ]
        lower = [
            float(indexed.loc[channel, "mean"] - indexed.loc[channel, "low"])
            if channel in indexed.index
            else np.nan
            for channel in channels
        ]
        upper = [
            float(indexed.loc[channel, "high"] - indexed.loc[channel, "mean"])
            if channel in indexed.index
            else np.nan
            for channel in channels
        ]
        bars = axis.bar(
            x + offset,
            values,
            yerr=np.asarray([lower, upper]),
            width=width,
            color=ARM_COLOURS[arm],
            label=arm.capitalize(),
            capsize=3,
        )
        for bar, value in zip(bars, values, strict=True):
            if math.isfinite(value):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    labels = {
        "semantic": "semantic",
        "structural_full": "full structural",
        "structural_proxy": "local proxy",
        "structural_remote": "remote closure",
    }
    axis.set_xticks(x, [labels[channel] for channel in channels], rotation=15)
    axis.set_title(title)
    axis.legend(frameon=False, fontsize=8)


def plot_score_scale_and_specialisation(
    head_summary: pd.DataFrame,
    specialisation: pd.DataFrame,
    config: Config,
) -> tuple[Path, Path]:
    """Separate native event effects from dose-normalised head responses."""
    _configure_style()
    figure, axes = plt.subplots(1, 4, figsize=(15.5, 3.9))
    figure.suptitle(
        "What do native head scores say after accounting for intervention dose?",
        fontsize=13.5,
        fontweight="bold",
    )
    _bar_score_means(
        axes[0],
        head_summary,
        scale="raw",
        channels=("semantic", "structural_full"),
        title="a  Natural-event effects",
    )
    axes[0].set_ylabel("mean raw head score")
    _bar_score_means(
        axes[1],
        head_summary,
        scale="per_unit",
        channels=("semantic", "structural_full"),
        title="b  Effects per unit input change",
    )
    axes[1].set_ylabel("mean score / event L2 dose")

    _bar_score_means(
        axes[2],
        head_summary,
        scale="per_unit",
        channels=("structural_proxy", "structural_remote"),
        title="c  Which structural cue responds?",
    )
    axes[2].set_ylabel("mean score / event L2 dose")
    axes[2].text(
        1 - 0.17,
        0.02,
        "local: no\nvisible event",
        transform=axes[2].get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7,
        color=ORANGE,
    )

    axis = axes[3]
    regime_order = ("generalist", "semantic-leaning", "structural-leaning")
    regime_colours = (LIGHT_GREY, BLUE, ORANGE)
    raw = specialisation[specialisation["scale"] == "raw"]
    bottom = np.zeros(len(MEASURED_ARMS), dtype=float)
    x = np.arange(len(MEASURED_ARMS), dtype=float)
    for regime, colour in zip(regime_order, regime_colours, strict=True):
        values = np.asarray(
            [
                float((raw.loc[raw["arm"] == arm, "regime"] == regime).mean())
                for arm in MEASURED_ARMS
            ]
        )
        axis.bar(x, values, bottom=bottom, color=colour, label=regime)
        bottom += values
    axis.set_xticks(x, ("Local", "Global"))
    axis.set_ylim(0, 1)
    axis.set_ylabel("fraction of learned heads")
    axis.set_title("d  Are strong relative leanings common?")
    axis.legend(frameon=False, fontsize=8, loc="lower center")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return _save_figure(
        figure, "04_score_scale_and_specialisation", config.output_dir
    )


def plot_exact_head_distance_profiles(
    score_distance: pd.DataFrame,
    config: Config,
) -> tuple[Path, Path]:
    """Show every seed-specific learned head's normalized distance profile."""
    selected = score_distance[
        (score_distance["scale"] == "raw")
        & score_distance["channel"].isin(("semantic", "structural_full"))
    ]
    grouped = (
        selected.groupby(
            ["seed", "arm", "channel", "layer", "head", "distance"],
            as_index=False,
        )["value"]
        .mean()
        .sort_values(["seed", "layer", "head", "distance"])
    )
    _configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 11.0), sharex=True)
    figure.suptitle(
        "Exact learned-head score-distance profiles",
        fontsize=14,
        fontweight="bold",
    )
    image_artist = None
    for row_index, arm in enumerate(MEASURED_ARMS):
        for column_index, channel in enumerate(("semantic", "structural_full")):
            axis = axes[row_index, column_index]
            subset = grouped[
                (grouped["arm"] == arm) & (grouped["channel"] == channel)
            ]
            keys = list(
                subset[["seed", "layer", "head"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            matrix = np.full((len(keys), config.layers + 1), np.nan)
            for position, key in enumerate(keys):
                values = (
                    subset[
                        (subset["seed"] == key[0])
                        & (subset["layer"] == key[1])
                        & (subset["head"] == key[2])
                    ]
                    .set_index("distance")["value"]
                    .reindex(range(config.layers + 1), fill_value=0.0)
                    .to_numpy(dtype=float)
                )
                total = float(values.sum())
                if total > 1.0e-12:
                    matrix[position] = values / total
            image_artist = axis.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                vmin=0,
                vmax=1,
                cmap="viridis",
            )
            labels = [f"S{s} L{layer + 1}H{head + 1}" for s, layer, head in keys]
            axis.set_yticks(np.arange(len(keys)), labels, fontsize=5.5)
            axis.set_xticks(range(config.layers + 1))
            axis.tick_params(axis="x", labelbottom=True)
            axis.set_xlabel("reporter-to-carrier distance")
            axis.set_title(
                f"{arm.capitalize()} — {channel.replace('_', ' ')}"
            )
    if image_artist is not None:
        colourbar = figure.colorbar(
            image_artist,
            ax=axes,
            fraction=0.025,
            pad=0.02,
        )
        colourbar.set_label("within-head fraction of score mass")
    figure.text(
        0.5,
        0.015,
        "Each row is one fitted seed/layer/head and sums to one across distance.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(top=0.93, bottom=0.06, left=0.12, right=0.91, hspace=0.25)
    return _save_figure(
        figure, "05_exact_head_distance_profiles", config.output_dir
    )


def plot_alignment(
    excess_by_seed: pd.DataFrame,
    config: Config,
) -> tuple[Path, Path]:
    _configure_style()
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    figure.suptitle(
        "Is semantic–structural alignment specific to the matched head?",
        fontsize=13.5,
        fontweight="bold",
    )
    for axis, channel in zip(axes, STRUCTURAL_CHANNELS, strict=True):
        for arm_index, arm in enumerate(MEASURED_ARMS):
            values = excess_by_seed.loc[
                (excess_by_seed["arm"] == arm)
                & (excess_by_seed["structural_channel"] == channel),
                "cosine_excess",
            ].dropna()
            if values.empty:
                axis.text(
                    arm_index + 0.04,
                    0.01,
                    "no visible\nevent",
                    ha="left",
                    va="bottom",
                    color=ARM_COLOURS[arm],
                    fontsize=8,
                )
                continue
            x = np.full(len(values), float(arm_index))
            axis.scatter(
                x,
                values,
                color=ARM_COLOURS[arm],
                s=28,
                alpha=0.7,
                zorder=3,
            )
            mean, error = _mean_seed_interval(values)
            axis.errorbar(
                arm_index,
                mean,
                yerr=error,
                color="#222222",
                marker="D",
                markersize=5,
                capsize=4,
                linewidth=1.2,
                zorder=4,
            )
        axis.axhline(0, color=GREY, linewidth=1, linestyle="--")
        axis.set_xticks((0, 1), ("Local", "Global"))
        axis.set_title(channel.replace("structural_", "").capitalize())
    axes[0].set_ylabel("matched minus other-head profile cosine")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return _save_figure(figure, "03_head_alignment_vs_null", config.output_dir)


def _event_dose_summary(dose: pd.DataFrame) -> pd.DataFrame:
    return (
        dose.groupby(["arm", "channel"], as_index=False)
        .agg(
            mean_dose=("event_dose", "mean"),
            median_dose=("event_dose", "median"),
            minimum_dose=("event_dose", "min"),
            maximum_dose=("event_dose", "max"),
            zero_dose_fraction=("zero_dose", "mean"),
        )
    )


def _event_dose_state_summary(dose: pd.DataFrame) -> pd.DataFrame:
    return (
        dose.groupby(["arm", "channel", "cycle", "cue"], as_index=False)
        .agg(
            mean_dose=("event_dose", "mean"),
            median_dose=("event_dose", "median"),
            zero_dose_fraction=("zero_dose", "mean"),
            events=("event_dose", "size"),
        )
    )


def _finite_or_none(value: Any) -> float | int | str | None:
    if isinstance(value, (str, int)):
        return value
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _summary_payload(
    performance: pd.DataFrame,
    fidelity_summary: pd.DataFrame,
    head_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
    carriage_summary: pd.DataFrame,
    dose_summary: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    alignment_excess_summary: pd.DataFrame,
    specialisation: pd.DataFrame,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "performance": {},
        "selection_fidelity": {},
        "head_scores": {},
        "profile_variability": {},
        "score_profile_tv": {},
        "within_arm_score_profile_tv": {},
        "carriage": {},
        "carriage_profile_tv": {},
        "within_arm_carriage_profile_tv": {},
        "event_dose": {},
        "alignment": {},
        "specialisation": {},
    }
    for arm, rows in performance.groupby("arm"):
        result["performance"][str(arm)] = {
            "test_mse_mean": float(rows["test_mse"].mean()),
            "test_mse_seed_sd": float(rows["test_mse"].std(ddof=1))
            if len(rows) > 1
            else 0.0,
            "test_mae_mean": float(rows["test_mae"].mean()),
        }
    for arm, rows in fidelity_summary.groupby("arm"):
        result["selection_fidelity"][str(arm)] = {
            str(row["stratum"]): {
                key: _finite_or_none(row[key])
                for key in (
                    "fidelity_mse",
                    "fidelity_mse_ci_half_width",
                    "fidelity_mae",
                    "fidelity_mae_ci_half_width",
                    "correlation",
                    "correlation_ci_half_width",
                    "origin_slope",
                    "origin_slope_ci_half_width",
                )
            }
            for _, row in rows.iterrows()
        }
    for (arm, scale, channel), rows in head_summary.groupby(
        ["arm", "scale", "channel"]
    ):
        result["head_scores"].setdefault(str(arm), {}).setdefault(
            str(scale), {}
        )[str(channel)] = {
            "mean": _finite_or_none(rows["mean"].mean()),
            "mean_ci_width": _finite_or_none(rows["ci_width"].mean()),
        }
    for (arm, scale, channel), rows in score_summary.groupby(
        ["arm", "scale", "channel"]
    ):
        result["profile_variability"].setdefault(str(arm), {}).setdefault(
            str(scale), {}
        )[str(channel)] = {
            "profile_ci_width_mean": _finite_or_none(
                rows["profile_ci_width"].mean()
            ),
            "within_seed_profile_sd_mean": _finite_or_none(
                rows["within_seed_profile_sd"].mean()
            ),
            "between_seed_profile_sd_mean": _finite_or_none(
                rows["between_seed_profile_sd"].mean()
            ),
        }
    for scale in SCALES:
        scale_rows = score_summary[score_summary["scale"] == scale]
        score_profiles: dict[tuple[str, str], np.ndarray] = {}
        for arm in MEASURED_ARMS:
            for channel in CHANNELS:
                rows = scale_rows[
                    (scale_rows["arm"] == arm)
                    & (scale_rows["channel"] == channel)
                ].sort_values("distance")
                values = rows["profile_mean"].to_numpy(dtype=float)
                total = float(np.nansum(values))
                if len(values) and total > 1.0e-12:
                    score_profiles[(arm, channel)] = values / total
        for channel in CHANNELS:
            if all((arm, channel) in score_profiles for arm in MEASURED_ARMS):
                result["score_profile_tv"].setdefault(str(scale), {})[
                    str(channel)
                ] = float(
                    0.5
                    * np.abs(
                        score_profiles[("local", channel)]
                        - score_profiles[("global", channel)]
                    ).sum()
                )
        for arm in MEASURED_ARMS:
            semantic_key = (arm, "semantic")
            if semantic_key not in score_profiles:
                continue
            for channel in STRUCTURAL_CHANNELS:
                structural_key = (arm, channel)
                if structural_key in score_profiles:
                    result["within_arm_score_profile_tv"].setdefault(
                        str(arm), {}
                    ).setdefault(str(scale), {})[str(channel)] = float(
                        0.5
                        * np.abs(
                            score_profiles[semantic_key]
                            - score_profiles[structural_key]
                        ).sum()
                    )
    for (arm, scale, channel), rows in carriage_summary.groupby(
        ["arm", "scale", "channel"]
    ):
        ordered = rows.sort_values("distance")
        raw_values = ordered["mean"].to_numpy(dtype=float)
        profile_values = ordered["profile_mean"].to_numpy(dtype=float)
        finite = np.isfinite(raw_values)
        total = (
            float(raw_values[finite].sum()) if finite.any() else float("nan")
        )
        profile_total = float(np.nansum(profile_values))
        result["carriage"].setdefault(str(arm), {}).setdefault(
            str(scale), {}
        )[str(channel)] = {
            "total": _finite_or_none(total),
            "normalized_profile": [
                _finite_or_none(value / profile_total)
                if profile_total > 1.0e-12
                else None
                for value in profile_values
            ],
        }
    for scale in SCALES:
        scale_rows = carriage_summary[carriage_summary["scale"] == scale]
        carriage_profiles: dict[tuple[str, str], np.ndarray] = {}
        for channel in CHANNELS:
            profiles = {}
            for arm in MEASURED_ARMS:
                rows = scale_rows[
                    (scale_rows["arm"] == arm)
                    & (scale_rows["channel"] == channel)
                ].sort_values("distance")
                values = rows["profile_mean"].to_numpy(dtype=float)
                total = float(np.nansum(values))
                if len(values) and total > 1.0e-12:
                    profiles[arm] = values / total
                    carriage_profiles[(arm, channel)] = profiles[arm]
            if set(profiles) == set(MEASURED_ARMS):
                result["carriage_profile_tv"].setdefault(str(scale), {})[
                    str(channel)
                ] = float(0.5 * np.abs(profiles["local"] - profiles["global"]).sum())
        for arm in MEASURED_ARMS:
            semantic_key = (arm, "semantic")
            if semantic_key not in carriage_profiles:
                continue
            for channel in STRUCTURAL_CHANNELS:
                structural_key = (arm, channel)
                if structural_key in carriage_profiles:
                    result["within_arm_carriage_profile_tv"].setdefault(
                        str(arm), {}
                    ).setdefault(str(scale), {})[str(channel)] = float(
                        0.5
                        * np.abs(
                            carriage_profiles[semantic_key]
                            - carriage_profiles[structural_key]
                        ).sum()
                    )
    for _, row in dose_summary.iterrows():
        result["event_dose"].setdefault(str(row["arm"]), {})[
            str(row["channel"])
        ] = {
            key: _finite_or_none(row[key])
            for key in (
                "mean_dose",
                "median_dose",
                "minimum_dose",
                "maximum_dose",
                "zero_dose_fraction",
            )
        }
    for _, row in alignment_summary.iterrows():
        key = f"{row['structural_channel']}:{row['pairing']}"
        result["alignment"].setdefault(str(row["arm"]), {})[key] = {
            "cosine_mean": _finite_or_none(row["cosine_mean"]),
            "peak_match_fraction": _finite_or_none(row["peak_match_fraction"]),
            "valid_comparisons": int(row["valid_comparisons"]),
            "total_comparisons": int(row["total_comparisons"]),
        }
    for _, row in alignment_excess_summary.iterrows():
        target = result["alignment"][str(row["arm"])]
        target[f"{row['structural_channel']}:excess_over_other_head"] = {
            "cosine": _finite_or_none(row["cosine_excess"]),
            "cosine_ci_half_width": _finite_or_none(
                row["cosine_excess_ci_half_width"]
            ),
            "peak_match_fraction": _finite_or_none(row["peak_match_excess"]),
            "peak_match_ci_half_width": _finite_or_none(
                row["peak_match_excess_ci_half_width"]
            ),
        }
    for (arm, scale), rows in specialisation.groupby(["arm", "scale"]):
        result["specialisation"].setdefault(str(arm), {})[str(scale)] = {
            "D_rel_mean": float(rows["D_rel"].mean()),
            "D_rel_mean_absolute": float(rows["D_rel"].abs().mean()),
            "regime_fraction": {
                regime: float((rows["regime"] == regime).mean())
                for regime in (
                    "generalist",
                    "semantic-leaning",
                    "structural-leaning",
                )
            },
        }
    return result


def _analyse_and_write(
    *,
    config: Config,
    performance: pd.DataFrame,
    fidelity: pd.DataFrame,
    score_distance: pd.DataFrame,
    carriage: pd.DataFrame,
    head_scores: pd.DataFrame,
    event_dose: pd.DataFrame,
    health: dict[str, Any],
) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    score_summary = summarise_scores(score_distance, config)
    carriage_summary = summarise_carriage(carriage, config)
    head_summary = summarise_head_scores(head_scores, config)
    fidelity_by_seed = summarise_fidelity_by_seed(fidelity)
    fidelity_summary = summarise_fidelity(fidelity_by_seed)
    dose_summary = _event_dose_summary(event_dose)
    dose_state_summary = _event_dose_state_summary(event_dose)
    alignment = compute_profile_alignment(score_distance)
    alignment_by_seed = summarise_alignment_by_seed(alignment)
    alignment_summary = summarise_alignment(alignment_by_seed)
    alignment_excess = alignment_excess_by_seed(alignment_by_seed)
    alignment_excess_summary = summarise_alignment_excess(alignment_excess)
    specialisation = summarise_specialisation(head_scores)
    tables = {
        "performance.csv": performance,
        "selection_fidelity_per_graph.csv": fidelity,
        "selection_fidelity_by_seed.csv": fidelity_by_seed,
        "selection_fidelity_summary.csv": fidelity_summary,
        "score_distance_per_graph.csv": score_distance,
        "score_distance_summary.csv": score_summary,
        "carriage_distance_per_graph.csv": carriage,
        "carriage_distance_summary.csv": carriage_summary,
        "head_scores_per_graph.csv": head_scores,
        "head_scores_summary.csv": head_summary,
        "event_dose_per_graph.csv": event_dose,
        "event_dose_summary.csv": dose_summary,
        "event_dose_by_state.csv": dose_state_summary,
        "profile_alignment_per_graph.csv": alignment,
        "profile_alignment_by_seed.csv": alignment_by_seed,
        "profile_alignment_summary.csv": alignment_summary,
        "profile_alignment_excess_by_seed.csv": alignment_excess,
        "profile_alignment_excess_summary.csv": alignment_excess_summary,
        "specialisation_by_head.csv": specialisation,
    }
    for name, table in tables.items():
        table.to_csv(config.output_dir / name, index=False)
    figures = [
        *plot_core_questions(
            performance,
            fidelity_by_seed,
            head_scores,
            carriage_summary,
            config,
        ),
        *plot_distance_and_confidence(score_summary, event_dose, config),
        *plot_alignment(alignment_excess, config),
        *plot_score_scale_and_specialisation(
            head_summary, specialisation, config
        ),
        *plot_exact_head_distance_profiles(score_distance, config),
    ]
    summary = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "task_contract": {
            "target_scale": target_scale(config),
            "target_weight_second_moment": target_weight_second_moment(config),
            "cue_only_bayes_mse": cue_only_bayes_mse(config),
            "reporter_to_closure_distance": 5,
            "twice_message_depth": 2 * int(config.layers),
        },
        "results": _summary_payload(
            performance,
            fidelity_summary,
            head_summary,
            score_summary,
            carriage_summary,
            dose_summary,
            alignment_summary,
            alignment_excess_summary,
            specialisation,
        ),
        "health": health,
        "assumptions": [
            "Values have bounded magnitude and are rescaled so E[x_i^2]=1.",
            "The target is variance-normalized after mixing local cue and remote cycle weights.",
            "The analytic cue-only Bayes MSE accounts for the configured remote target weight.",
            "Semantic fidelity and semantic score events use an in-support symmetric sign flip.",
            "The local proxy changes bond order but not message support.",
            "Structural events change RRWP only on attended pairs incident to the reporter; topology and message support remain fixed.",
            "Per-unit effects are undefined (NaN), not zero, when the visible event dose is zero.",
            "The remote-only local-RRWP event is expected to have exactly zero visible dose.",
            "Architecture-level profile intervals aggregate exchangeable heads within each fitted seed before seed-level inference.",
        ],
        "figures": [str(path) for path in figures],
    }
    with (config.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return summary


def run(config: Config) -> dict[str, Any]:
    config.validate()
    torch.set_num_threads(1)
    graph_types = build_graph_types(config)
    tensors = _stack_graph_types(graph_types)
    train = make_dataset(config, count=config.train_examples, seed=config.data_seed + 1)
    validation = make_dataset(
        config, count=config.validation_examples, seed=config.data_seed + 2
    )
    test = make_dataset(config, count=config.test_examples, seed=config.data_seed + 3)
    performance_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    carriage_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    dose_rows: list[dict[str, Any]] = []
    health: dict[str, Any] = {}
    checkpoint_directory = config.output_dir / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    for arm in ARMS:
        for seed in config.seeds:
            model, training = train_model(
                config,
                tensors,
                train,
                validation,
                arm=arm,
                seed=int(seed),
            )
            health[f"{arm}_{seed}"] = training
            torch.save(
                {
                    "arm": str(arm),
                    "seed": int(seed),
                    "model_state": model.state_dict(),
                    "config": {**asdict(config), "output_dir": str(config.output_dir)},
                },
                checkpoint_directory / f"{arm}_seed_{seed}.pt",
            )
            performance_rows.append(
                {
                    **evaluate(
                        model,
                        test,
                        tensors,
                        input_arm=arm,
                        label=arm,
                        training_arm=arm,
                        seed=int(seed),
                    ),
                    **training,
                }
            )
            fidelity_rows.extend(
                measure_selection_fidelity(
                    model,
                    test,
                    tensors,
                    input_arm=arm,
                    label=arm,
                    training_arm=arm,
                    seed=int(seed),
                )
            )
            if arm == "global":
                performance_rows.append(
                    {
                        **evaluate(
                            model,
                            test,
                            tensors,
                            input_arm="global_shuffled",
                            label="global_test_shuffled",
                            training_arm="global",
                            seed=int(seed),
                        ),
                        **training,
                    }
                )
                fidelity_rows.extend(
                    measure_selection_fidelity(
                        model,
                        test,
                        tensors,
                        input_arm="global_shuffled",
                        label="global_test_shuffled",
                        training_arm="global",
                        seed=int(seed),
                    )
                )
            if arm not in MEASURED_ARMS:
                continue
            scores, carriage, heads, doses, _ = measure_scores_and_carriage(
                model,
                test,
                graph_types,
                tensors,
                config,
                arm=arm,
                seed=int(seed),
            )
            score_rows.extend(scores)
            carriage_rows.extend(carriage)
            head_rows.extend(heads)
            dose_rows.extend(doses)

    return _analyse_and_write(
        config=config,
        performance=pd.DataFrame(performance_rows),
        fidelity=pd.DataFrame(fidelity_rows),
        score_distance=pd.DataFrame(score_rows),
        carriage=pd.DataFrame(carriage_rows),
        head_scores=pd.DataFrame(head_rows),
        event_dose=pd.DataFrame(dose_rows),
        health=health,
    )


def reanalyze_output(output_directory: Path) -> dict[str, Any]:
    """Rebuild summaries and figures from raw CSV files without training."""
    with (output_directory / "summary.json").open(encoding="utf-8") as handle:
        previous = json.load(handle)
    payload = dict(previous["config"])
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(
            "cached measurements use a different intervention protocol; "
            "rerun training/measurement instead of relabelling stale outputs"
        )
    payload["output_dir"] = output_directory
    payload["seeds"] = tuple(int(seed) for seed in payload["seeds"])
    config = Config(**payload)
    return _analyse_and_write(
        config=config,
        performance=pd.read_csv(output_directory / "performance.csv"),
        fidelity=pd.read_csv(
            output_directory / "selection_fidelity_per_graph.csv"
        ),
        score_distance=pd.read_csv(
            output_directory / "score_distance_per_graph.csv"
        ),
        carriage=pd.read_csv(
            output_directory / "carriage_distance_per_graph.csv"
        ),
        head_scores=pd.read_csv(output_directory / "head_scores_per_graph.csv"),
        event_dose=pd.read_csv(output_directory / "event_dose_per_graph.csv"),
        health=previous.get("health", {}),
    )


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/chapter6_rrwp_molecular_sites_v1"),
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--training-steps", type=int, default=600)
    parser.add_argument("--train-examples", type=int, default=2_048)
    parser.add_argument("--validation-examples", type=int, default=512)
    parser.add_argument("--test-examples", type=int, default=1_024)
    parser.add_argument("--measurement-examples", type=int, default=192)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--local-clue-reliability", type=float, default=0.75)
    parser.add_argument("--remote-target-weight", type=float, default=1.0)
    parser.add_argument("--reanalyze-only", action="store_true")
    args = parser.parse_args(argv)
    if args.reanalyze_only:
        result = reanalyze_output(args.output_dir)
        print(json.dumps(result, indent=2))
        return result
    config = Config(
        output_dir=args.output_dir,
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        training_steps=int(args.training_steps),
        train_examples=int(args.train_examples),
        validation_examples=int(args.validation_examples),
        test_examples=int(args.test_examples),
        measurement_examples=int(args.measurement_examples),
        bootstrap_replicates=int(args.bootstrap_replicates),
        local_clue_reliability=float(args.local_clue_reliability),
        remote_target_weight=float(args.remote_target_weight),
    )
    result = run(config)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
