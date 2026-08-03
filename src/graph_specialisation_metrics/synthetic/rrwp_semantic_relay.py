"""Learned-head synthetic for RRWP-guided semantic selection.

Two scalar-valued candidate nodes feed a three-node readout route. One candidate
belongs to a remotely closed cycle and the other to an open chain. Candidate
degree is an imperfect local structural cue, while higher-order RRWP exposes the
remote closure. A small parameter-matched 1-hop transformer must select the
cycle value and relay it to the output.

The experiment captures the native routed value of every learned attention
head. Semantic and structural donor events are projected through the clean
output Jacobian and binned by exact source-to-carrier shortest-path distance.
Final-state Functional carriage is measured separately through the linear
route-node readout.
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
from torch import nn

plt.switch_backend("Agg")

DTYPE = torch.float32
ARMS = ("local", "global", "global_shuffled")
MEASURED_ARMS = ("local", "global")
CHANNELS = ("semantic", "structural")

ORANGE = "#EE7733"
BLUE = "#4477AA"
GREY = "#777777"
LIGHT_GREY = "#D7D7D7"
TEXT = "#222222"
ARM_COLOURS = {"local": ORANGE, "global": BLUE, "global_shuffled": GREY}
CHANNEL_MARKERS = {"semantic": "o", "structural": "s"}


@dataclass(frozen=True)
class Config:
    output_dir: Path
    arm_length: int = 4
    rrwp_horizon: int = 12
    hidden_dim: int = 24
    heads: int = 4
    layers: int = 3
    train_examples: int = 2_048
    validation_examples: int = 512
    test_examples: int = 1_024
    measurement_examples: int = 192
    seeds: tuple[int, ...] = (0, 1, 2)
    batch_size: int = 64
    training_steps: int = 600
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-4
    route_auxiliary_weight: float = 0.25
    local_clue_reliability: float = 0.75
    bootstrap_replicates: int = 500
    data_seed: int = 53_011

    def validate(self) -> None:
        if self.arm_length <= self.layers:
            raise ValueError("remote closure must lie beyond learned message depth")
        if self.rrwp_horizon < 2 * self.arm_length + 1:
            raise ValueError("RRWP horizon must cover the registered cycle closure")
        if self.hidden_dim < self.heads or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be positive and divisible by heads")
        if min(
            self.train_examples,
            self.validation_examples,
            self.test_examples,
            self.measurement_examples,
            self.training_steps,
            self.batch_size,
        ) < 1:
            raise ValueError("sample and optimization sizes must be positive")
        if self.measurement_examples > self.test_examples:
            raise ValueError("measurement examples cannot exceed the test set")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be non-empty and unique")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer settings are invalid")
        if not 0.5 < self.local_clue_reliability < 1.0:
            raise ValueError("local_clue_reliability must lie in (0.5,1)")
        if self.bootstrap_replicates < 100:
            raise ValueError("bootstrap_replicates must be at least 100")


@dataclass(frozen=True)
class GraphType:
    cycle_side: int
    tag_category: str
    tag_correct: bool
    tag_strength: int
    candidate_nodes: tuple[int, int]
    route_nodes: tuple[int, ...]
    roles: np.ndarray
    adjacency: np.ndarray
    distances: np.ndarray
    rrwp: np.ndarray
    cycle_swapped_rrwp: np.ndarray
    role_swapped_rrwp: np.ndarray


@dataclass(frozen=True)
class Dataset:
    type_index: np.ndarray
    values: np.ndarray
    high_order_swap: np.ndarray


def _transition(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    if np.any(degree <= 0):
        raise ValueError("synthetic graph must be connected and have no isolates")
    return adjacency / degree[:, None]


def _rrwp(adjacency: np.ndarray, horizon: int) -> np.ndarray:
    transition = _transition(adjacency)
    powers = [np.eye(adjacency.shape[0], dtype=np.float64)]
    for _ in range(int(horizon)):
        powers.append(powers[-1] @ transition)
    # Attention tensors are indexed [receiver, sender, feature].
    return np.stack(powers, axis=-1).transpose(1, 0, 2)


def _all_pairs_distances(adjacency: np.ndarray) -> np.ndarray:
    graph = nx.from_numpy_array(adjacency)
    distance = np.full(adjacency.shape, np.inf, dtype=np.float64)
    for source, lengths in nx.all_pairs_shortest_path_length(graph):
        for target, value in lengths.items():
            distance[int(source), int(target)] = int(value)
    return distance


def _graph_arrays(
    *,
    arm_length: int,
    cycle_side: int,
    level_by_side: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, ...]]:
    route_nodes = (0, 1, 2)  # root, relay, selector
    candidate_nodes = (3, 4 + 2 * int(arm_length))
    leaf_start = 5 + 4 * int(arm_length)
    total_nodes = leaf_start + sum(level_by_side)
    roles = np.full(total_nodes, 4, dtype=np.int64)  # context
    roles[0] = 0  # root
    roles[1] = 1  # relay
    roles[2] = 2  # selector
    roles[list(candidate_nodes)] = 3  # candidates
    roles[leaf_start:] = 5  # local tag leaves
    adjacency = np.zeros((total_nodes, total_nodes), dtype=np.float64)

    def add_edge(left: int, right: int) -> None:
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0

    add_edge(0, 1)
    add_edge(1, 2)
    leaf = leaf_start
    for side, candidate in enumerate(candidate_nodes):
        add_edge(2, candidate)
        arm_offset = 4 if side == 0 else 5 + 2 * int(arm_length)
        endpoints: list[int] = []
        for branch in range(2):
            previous = candidate
            start = arm_offset + branch * int(arm_length)
            for node in range(start, start + int(arm_length)):
                add_edge(previous, node)
                previous = node
            endpoints.append(previous)
        if side == int(cycle_side):
            add_edge(endpoints[0], endpoints[1])
        for _ in range(int(level_by_side[side])):
            add_edge(candidate, leaf)
            leaf += 1
    if leaf != total_nodes:
        raise RuntimeError("tag leaf accounting changed")
    if not nx.is_connected(nx.from_numpy_array(adjacency)):
        raise RuntimeError("registered relay graph is disconnected")
    return adjacency, roles, candidate_nodes, route_nodes


def build_graph_types(config: Config) -> tuple[GraphType, ...]:
    # Correct with probability .75; strong and weak margins occur on both sides
    # of the truth so local confidence is heterogeneous rather than binary.
    tag_categories = (
        ("correct_strong", 4, 1, True, 3),
        ("correct_weak", 3, 2, True, 1),
        ("wrong_weak", 2, 3, False, 1),
        ("wrong_strong", 1, 4, False, 3),
    )
    result: list[GraphType] = []
    for cycle_side in (0, 1):
        for category, cycle_level, chain_level, correct, strength in tag_categories:
            levels = [0, 0]
            levels[cycle_side] = int(cycle_level)
            levels[1 - cycle_side] = int(chain_level)
            adjacency, roles, candidates, route = _graph_arrays(
                arm_length=config.arm_length,
                cycle_side=cycle_side,
                level_by_side=(levels[0], levels[1]),
            )
            cycle_swap, _, _, _ = _graph_arrays(
                arm_length=config.arm_length,
                cycle_side=1 - cycle_side,
                level_by_side=(levels[0], levels[1]),
            )
            role_swap, _, _, _ = _graph_arrays(
                arm_length=config.arm_length,
                cycle_side=1 - cycle_side,
                level_by_side=(levels[1], levels[0]),
            )
            result.append(
                GraphType(
                    cycle_side=int(cycle_side),
                    tag_category=str(category),
                    tag_correct=bool(correct),
                    tag_strength=int(strength),
                    candidate_nodes=tuple(int(value) for value in candidates),
                    route_nodes=tuple(int(value) for value in route),
                    roles=roles,
                    adjacency=adjacency,
                    distances=_all_pairs_distances(adjacency),
                    rrwp=_rrwp(adjacency, config.rrwp_horizon),
                    cycle_swapped_rrwp=_rrwp(cycle_swap, config.rrwp_horizon),
                    role_swapped_rrwp=_rrwp(role_swap, config.rrwp_horizon),
                )
            )
    node_counts = {len(item.roles) for item in result}
    if len(node_counts) != 1:
        raise RuntimeError("registered graph types must have fixed node count")
    return tuple(result)


def make_dataset(config: Config, *, count: int, seed: int) -> Dataset:
    rng = np.random.default_rng(int(seed))
    cycle_side = rng.integers(0, 2, size=int(count))
    # Correct and incorrect examples are each evenly split between weak and
    # strong local margins.
    reliability = float(config.local_clue_reliability)
    category = rng.choice(
        4,
        size=int(count),
        p=(reliability / 2, reliability / 2, (1 - reliability) / 2, (1 - reliability) / 2),
    )
    return Dataset(
        type_index=4 * cycle_side + category,
        values=rng.standard_normal((int(count), 2)).astype(np.float32),
        high_order_swap=(rng.random(int(count)) < 0.5),
    )


class SparseRRWPAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, pe_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = int(hidden_dim) // int(heads)
        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.pe_bias = nn.Linear(pe_dim, heads, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        pair_pe: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalised = self.norm(hidden)
        batch, nodes, _ = normalised.shape
        qkv = self.qkv(normalised).view(
            batch, nodes, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        # [batch, head, receiver, sender]
        logits = torch.einsum("bnhe,bmhe->bhnm", query, key) / math.sqrt(
            self.head_dim
        )
        logits = logits + self.pe_bias(pair_pe).permute(0, 3, 1, 2)
        mask = pair_mask[:, None, :, :]
        logits = logits.masked_fill(~mask, -1.0e9)
        attention = torch.softmax(logits, dim=-1).masked_fill(~mask, 0.0)
        routed = torch.einsum("bhnm,bmhe->bnhe", attention, value)
        update = self.out(routed.reshape(batch, nodes, self.hidden_dim))
        hidden = hidden + update
        hidden = hidden + self.ffn(self.norm_ffn(hidden))
        return hidden, routed, attention


class RRWPSemanticRelay(nn.Module):
    def __init__(self, config: Config, *, role_count: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        self.role_embedding = nn.Embedding(int(role_count), config.hidden_dim)
        self.value_projection = nn.Linear(1, config.hidden_dim, bias=False)
        self.layers = nn.ModuleList(
            [
                SparseRRWPAttention(
                    config.hidden_dim,
                    config.heads,
                    config.rrwp_horizon + 1,
                )
                for _ in range(config.layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.node_readout = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        roles: torch.Tensor,
        values: torch.Tensor,
        pair_pe: torch.Tensor,
        pair_mask: torch.Tensor,
        route_nodes: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> Any:
        hidden = self.role_embedding(roles) + self.value_projection(values[..., None])
        routed: list[torch.Tensor] = []
        attention: list[torch.Tensor] = []
        for layer in self.layers:
            hidden, head_output, weights = layer(hidden, pair_pe, pair_mask)
            routed.append(head_output)
            attention.append(weights)
        final_hidden = self.final_norm(hidden)
        batch = torch.arange(final_hidden.shape[0], device=final_hidden.device)[:, None]
        route_hidden = final_hidden[batch, route_nodes]
        route_prediction = self.node_readout(route_hidden).squeeze(-1)
        prediction = route_prediction.mean(dim=-1)
        if return_details:
            return prediction, route_prediction, final_hidden, tuple(routed), tuple(attention)
        return prediction


def _stack_graph_types(graph_types: Sequence[GraphType]) -> dict[str, torch.Tensor]:
    adjacency = torch.tensor(np.stack([item.adjacency for item in graph_types]), dtype=torch.bool)
    node_count = adjacency.shape[-1]
    identity = torch.eye(node_count, dtype=torch.bool)[None]
    pair_mask = adjacency | identity
    raw_rrwp = np.stack([item.rrwp for item in graph_types])
    registered = raw_rrwp[pair_mask.numpy()]
    scale = np.std(registered, axis=0)
    scale[scale < 1.0e-6] = 1.0
    return {
        "roles": torch.tensor(np.stack([item.roles for item in graph_types]), dtype=torch.long),
        "pair_mask": pair_mask,
        "rrwp": torch.tensor(raw_rrwp / scale, dtype=DTYPE),
        "cycle_swapped_rrwp": torch.tensor(
            np.stack([item.cycle_swapped_rrwp for item in graph_types]) / scale,
            dtype=DTYPE,
        ),
        "role_swapped_rrwp": torch.tensor(
            np.stack([item.role_swapped_rrwp for item in graph_types]) / scale,
            dtype=DTYPE,
        ),
        "candidates": torch.tensor(
            np.stack([item.candidate_nodes for item in graph_types]), dtype=torch.long
        ),
        "route": torch.tensor(
            np.stack([item.route_nodes for item in graph_types]), dtype=torch.long
        ),
        "cycle_side": torch.tensor(
            [item.cycle_side for item in graph_types], dtype=torch.long
        ),
    }


def _batch_inputs(
    dataset: Dataset,
    indices: np.ndarray,
    tensors: dict[str, torch.Tensor],
    *,
    arm: str,
) -> dict[str, torch.Tensor]:
    type_index = torch.tensor(dataset.type_index[indices], dtype=torch.long)
    roles = tensors["roles"][type_index]
    pair_mask = tensors["pair_mask"][type_index]
    clean_pe = tensors["rrwp"][type_index]
    if arm == "local":
        pair_pe = clean_pe.clone()
        pair_pe[..., 2:] = 0.0
    elif arm == "global":
        pair_pe = clean_pe
    elif arm == "global_shuffled":
        swapped = tensors["cycle_swapped_rrwp"][type_index]
        selector = torch.tensor(dataset.high_order_swap[indices], dtype=torch.bool)
        pair_pe = clean_pe.clone()
        pair_pe[selector, ..., 2:] = swapped[selector, ..., 2:]
    else:
        raise ValueError(f"unknown arm {arm!r}")
    candidate_nodes = tensors["candidates"][type_index]
    values = torch.zeros(roles.shape, dtype=DTYPE)
    semantic = torch.tensor(dataset.values[indices], dtype=DTYPE)
    batch = torch.arange(len(indices))[:, None]
    values[batch, candidate_nodes] = semantic
    cycle_side = tensors["cycle_side"][type_index]
    target = semantic[torch.arange(len(indices)), cycle_side]
    return {
        "roles": roles,
        "values": values,
        "pair_pe": pair_pe,
        "pair_mask": pair_mask,
        "route_nodes": tensors["route"][type_index],
        "candidate_nodes": candidate_nodes,
        "target": target,
        "type_index": type_index,
    }


def _forward(model: RRWPSemanticRelay, inputs: dict[str, torch.Tensor], **kwargs: Any) -> Any:
    return model(
        inputs["roles"],
        inputs["values"],
        inputs["pair_pe"],
        inputs["pair_mask"],
        inputs["route_nodes"],
        **kwargs,
    )


def train_model(
    config: Config,
    graph_types: Sequence[GraphType],
    tensors: dict[str, torch.Tensor],
    train: Dataset,
    validation: Dataset,
    *,
    arm: str,
    seed: int,
) -> tuple[RRWPSemanticRelay, dict[str, float | int]]:
    model = RRWPSemanticRelay(config, role_count=6, seed=int(seed)).to(dtype=DTYPE)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    rng = np.random.default_rng(config.data_seed + 100_000 + 97 * int(seed) + len(arm))
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    evaluation_interval = 20
    patience_checks = 12
    for step in range(1, int(config.training_steps) + 1):
        indices = rng.integers(0, len(train.type_index), size=int(config.batch_size))
        batch = _batch_inputs(train, indices, tensors, arm=arm)
        model.train()
        prediction, route_prediction, _, _, _ = _forward(
            model, batch, return_details=True
        )
        loss = torch.mean((prediction - batch["target"]) ** 2)
        auxiliary = torch.mean((route_prediction - batch["target"][:, None]) ** 2)
        loss = loss + float(config.route_auxiliary_weight) * auxiliary
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if step % evaluation_interval:
            continue
        model.eval()
        with torch.no_grad():
            val_index = np.arange(len(validation.type_index), dtype=np.int64)
            val = _batch_inputs(validation, val_index, tensors, arm=arm)
            val_prediction = _forward(model, val)
            val_loss = float(torch.mean((val_prediction - val["target"]) ** 2))
        if val_loss < best_loss - 1.0e-6:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience_checks:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
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
    arm: str,
    seed: int,
) -> dict[str, float | int | str]:
    index = np.arange(len(dataset.type_index), dtype=np.int64)
    inputs = _batch_inputs(dataset, index, tensors, arm=arm)
    with torch.no_grad():
        prediction, _, _, _, attention = _forward(model, inputs, return_details=True)
    error = prediction - inputs["target"]
    selector = inputs["route_nodes"][:, -1]
    candidates = inputs["candidate_nodes"]
    final_attention = attention[0]
    batch = torch.arange(len(index))[:, None]
    heads = torch.arange(final_attention.shape[1])[None, :]
    receiver = selector[:, None]
    cycle_side = tensors["cycle_side"][inputs["type_index"]]
    cycle_candidate = candidates[torch.arange(len(index)), cycle_side]
    chosen_mass = final_attention[batch, heads, receiver, cycle_candidate[:, None]]
    return {
        "seed": int(seed),
        "arm": str(arm),
        "test_mse": float(torch.mean(error.square())),
        "test_mae": float(torch.mean(torch.abs(error))),
        "selector_cycle_attention": float(chosen_mass.mean()),
    }


def _event_inputs(
    clean: dict[str, torch.Tensor],
    tensors: dict[str, torch.Tensor],
    *,
    arm: str,
    channel: str,
    source_side: int,
) -> dict[str, torch.Tensor]:
    event = {key: value.clone() if torch.is_tensor(value) else value for key, value in clean.items()}
    batch = torch.arange(clean["roles"].shape[0])
    source = clean["candidate_nodes"][:, int(source_side)]
    other = clean["candidate_nodes"][:, 1 - int(source_side)]
    if channel == "semantic":
        event["values"][batch, source] = clean["values"][batch, other]
    elif channel == "structural":
        donor = tensors["role_swapped_rrwp"][clean["type_index"]]
        if arm == "local":
            donor = donor.clone()
            donor[..., 2:] = 0.0
        nodes = torch.arange(clean["roles"].shape[1])[None, :]
        incident = (nodes[:, :, None] == source[:, None, None]) | (
            nodes[:, None, :] == source[:, None, None]
        )
        incident = incident & clean["pair_mask"]
        event["pair_pe"][incident] = donor[incident]
    else:
        raise ValueError(f"unknown channel {channel!r}")
    return event


def measure_scores_and_carriage(
    model: RRWPSemanticRelay,
    dataset: Dataset,
    graph_types: Sequence[GraphType],
    tensors: dict[str, torch.Tensor],
    config: Config,
    *,
    arm: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    index = np.arange(int(config.measurement_examples), dtype=np.int64)
    clean = _batch_inputs(dataset, index, tensors, arm=arm)
    model.zero_grad(set_to_none=True)
    prediction, _, final_hidden, routed, _ = _forward(model, clean, return_details=True)
    gradients = torch.autograd.grad(
        prediction.sum(),
        (*routed, final_hidden),
        retain_graph=False,
        allow_unused=False,
    )
    routed_gradient = tuple(value.detach() for value in gradients[:-1])
    final_gradient = gradients[-1].detach()
    clean_routed = tuple(value.detach() for value in routed)
    clean_final = final_hidden.detach()
    distances = np.stack(
        [graph_types[int(value)].distances for value in dataset.type_index[index]]
    )

    score_rows: list[dict[str, Any]] = []
    carriage_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    for channel in CHANNELS:
        graph_head_total = np.zeros(
            (len(index), config.layers, config.heads), dtype=np.float64
        )
        for source_side in (0, 1):
            event = _event_inputs(
                clean,
                tensors,
                arm=arm,
                channel=channel,
                source_side=source_side,
            )
            with torch.no_grad():
                _, _, event_final, event_routed, _ = _forward(
                    model, event, return_details=True
                )
            source_nodes = clean["candidate_nodes"][:, source_side].cpu().numpy()
            for layer, (clean_value, event_value, gradient) in enumerate(
                zip(clean_routed, event_routed, routed_gradient, strict=True)
            ):
                projected = torch.abs(
                    torch.sum((clean_value - event_value) * gradient, dim=-1)
                ).cpu().numpy()  # [graph, carrier, head]
                graph_head_total[:, layer] += projected.sum(axis=1) / 2.0
                for graph_position in range(len(index)):
                    source = int(source_nodes[graph_position])
                    carrier_distance = distances[graph_position, source]
                    for distance in range(config.layers + 1):
                        mask = carrier_distance == distance
                        contribution = projected[graph_position, mask].sum(axis=0)
                        for head in range(config.heads):
                            score_rows.append(
                                {
                                    "seed": int(seed),
                                    "graph": int(graph_position),
                                    "arm": str(arm),
                                    "channel": str(channel),
                                    "source": int(source_side),
                                    "layer": int(layer),
                                    "head": int(head),
                                    "distance": int(distance),
                                    "value": float(contribution[head]),
                                }
                            )

            # The final route-node readout is linear, so this finite carrier
            # response is the exact straight-line Functional carriage mass.
            final_projected = torch.abs(
                torch.sum((clean_final - event_final) * final_gradient, dim=-1)
            ).cpu().numpy()
            signed_complete = torch.sum(
                (clean_final - event_final) * final_gradient,
                dim=(-1, -2),
            )
            with torch.no_grad():
                event_prediction = _forward(model, event)
            if not torch.allclose(
                signed_complete,
                prediction.detach() - event_prediction,
                atol=2.0e-5,
                rtol=2.0e-5,
            ):
                raise RuntimeError("final-state carriage does not reconstruct output movement")
            for graph_position in range(len(index)):
                source = int(source_nodes[graph_position])
                carrier_distance = distances[graph_position, source]
                for distance in range(config.layers + 1):
                    carriage_rows.append(
                        {
                            "seed": int(seed),
                            "graph": int(graph_position),
                            "arm": str(arm),
                            "channel": str(channel),
                            "source": int(source_side),
                            "distance": int(distance),
                            "value": float(
                                final_projected[
                                    graph_position, carrier_distance == distance
                                ].sum()
                            ),
                        }
                    )
        for graph_position in range(len(index)):
            for layer in range(config.layers):
                for head in range(config.heads):
                    head_rows.append(
                        {
                            "seed": int(seed),
                            "graph": int(graph_position),
                            "arm": str(arm),
                            "channel": str(channel),
                            "layer": int(layer),
                            "head": int(head),
                            "value": float(graph_head_total[graph_position, layer, head]),
                        }
                    )
    return score_rows, carriage_rows, head_rows


def _bootstrap_interval(
    rows: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    value_column: str = "value",
) -> tuple[float, float, float]:
    if rows.empty:
        return float("nan"), float("nan"), float("nan")
    grouped = {
        int(training_seed): group.groupby("graph")[value_column].mean().dropna().to_numpy()
        for training_seed, group in rows.groupby("seed")
    }
    grouped = {key: values for key, values in grouped.items() if len(values)}
    if not grouped:
        return float("nan"), float("nan"), float("nan")
    seeds = np.asarray(sorted(grouped), dtype=np.int64)
    point = float(np.mean([values.mean() for values in grouped.values()]))
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(replicates), dtype=np.float64)
    for draw in range(int(replicates)):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for training_seed in sampled_seeds:
            values = grouped[int(training_seed)]
            seed_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        draws[draw] = float(np.mean(seed_means))
    low, high = np.percentile(draws, (2.5, 97.5))
    return point, float(low), float(high)


def summarise_intervals(
    score_distance: pd.DataFrame,
    carriage: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def key_seed(key: tuple[Any, ...], offset: int) -> int:
        digest = hashlib.sha256(repr(tuple(key)).encode("utf-8")).hexdigest()
        return int(config.data_seed + offset + int(digest[:8], 16) % 10_000)

    score_graph = (
        score_distance.groupby(
            ["seed", "graph", "arm", "channel", "layer", "head", "distance"],
            as_index=False,
        )["value"]
        .mean()
    )
    denominator = score_graph.groupby(
        ["seed", "graph", "arm", "channel", "layer", "head"]
    )["value"].transform("sum")
    score_graph["profile_value"] = np.where(
        denominator > 1.0e-10,
        score_graph["value"] / denominator,
        np.nan,
    )
    score_summary: list[dict[str, Any]] = []
    for key, rows in score_graph.groupby(
        ["arm", "channel", "layer", "head", "distance"], sort=True
    ):
        point, low, high = _bootstrap_interval(
            rows,
            replicates=config.bootstrap_replicates,
            seed=key_seed(key, 0),
        )
        profile_point, profile_low, profile_high = _bootstrap_interval(
            rows,
            replicates=config.bootstrap_replicates,
            seed=key_seed(key, 10_000),
            value_column="profile_value",
        )
        score_summary.append(
            {
                "arm": key[0],
                "channel": key[1],
                "layer": int(key[2]),
                "head": int(key[3]),
                "distance": int(key[4]),
                "mean": point,
                "low": low,
                "high": high,
                "ci_width": high - low,
                "profile_mean": profile_point,
                "profile_low": profile_low,
                "profile_high": profile_high,
                "profile_ci_width": profile_high - profile_low,
            }
        )
    carriage_graph = (
        carriage.groupby(
            ["seed", "graph", "arm", "channel", "distance"], as_index=False
        )["value"]
        .mean()
    )
    carriage_summary: list[dict[str, Any]] = []
    for key, rows in carriage_graph.groupby(
        ["arm", "channel", "distance"], sort=True
    ):
        point, low, high = _bootstrap_interval(
            rows,
            replicates=config.bootstrap_replicates,
            seed=key_seed(key, 20_000),
        )
        carriage_summary.append(
            {
                "arm": key[0],
                "channel": key[1],
                "distance": int(key[2]),
                "mean": point,
                "low": low,
                "high": high,
                "ci_width": high - low,
            }
        )
    return pd.DataFrame(score_summary), pd.DataFrame(carriage_summary)


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
        -0.15,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        color=TEXT,
    )


def plot_core_questions(
    performance: pd.DataFrame,
    head_scores: pd.DataFrame,
    carriage_summary: pd.DataFrame,
    config: Config,
) -> tuple[Path, Path]:
    _configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.3))
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.10, top=0.87, wspace=0.28, hspace=0.38)
    figure.suptitle(
        "Learned RRWP relay: performance, head scores, and carriage",
        fontsize=14.5,
        y=0.97,
        color=TEXT,
    )
    figure.text(
        0.5,
        0.92,
        "Parameter-matched 1-hop attention; points are training seeds and bands use seed/graph bootstrap",
        ha="center",
        fontsize=9.0,
        color=GREY,
    )

    axis = axes[0, 0]
    _panel_label(axis, "a")
    x = np.arange(len(ARMS))
    for position, arm in enumerate(ARMS):
        values = performance.loc[performance["arm"] == arm, "test_mse"].to_numpy()
        mean = float(values.mean())
        ci = 0.0 if len(values) < 2 else float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
        axis.bar(position, mean, color=ARM_COLOURS[arm], alpha=0.86, width=0.68)
        axis.errorbar(position, mean, yerr=ci, color=TEXT, capsize=3, linewidth=1)
        axis.scatter(
            np.full(len(values), position) + np.linspace(-0.10, 0.10, len(values)),
            values,
            color=TEXT,
            s=14,
            zorder=3,
        )
        axis.text(position, mean + ci + 0.015, f"{mean:.3f}", ha="center", fontsize=8.2)
    axis.set_xticks(x, ("local", "global", "shuffled\nglobal"))
    axis.set_ylabel("held-out MSE")
    axis.set_title("Higher-order RRWP must carry the useful information")
    axis.set_ylim(bottom=0.0)
    _clean_axis(axis)

    axis = axes[0, 1]
    _panel_label(axis, "b")
    aggregated = (
        head_scores.groupby(["seed", "arm", "channel", "layer", "head"])["value"]
        .mean()
        .reset_index()
    )
    positions = {("local", "semantic"): 0, ("local", "structural"): 1, ("global", "semantic"): 2.5, ("global", "structural"): 3.5}
    rng = np.random.default_rng(config.data_seed)
    for (arm, channel), position in positions.items():
        values = aggregated.loc[
            (aggregated["arm"] == arm) & (aggregated["channel"] == channel), "value"
        ].to_numpy()
        jitter = rng.uniform(-0.16, 0.16, len(values))
        axis.scatter(
            position + jitter,
            values,
            s=16,
            alpha=0.62,
            marker=CHANNEL_MARKERS[channel],
            color=ARM_COLOURS[arm],
            edgecolor="none",
        )
        axis.hlines(np.median(values), position - 0.22, position + 0.22, color=TEXT, linewidth=1.4)
    axis.axvline(1.75, color=LIGHT_GREY, linewidth=0.9)
    axis.set_xticks((0.5, 3.0), ("local RRWP", "global RRWP"))
    axis.set_ylabel("raw projected head score")
    axis.set_title("Raw head-score distributions")
    axis.legend(
        handles=[
            plt.Line2D([], [], marker="o", linestyle="none", color=GREY, label="semantic"),
            plt.Line2D([], [], marker="s", linestyle="none", color=GREY, label="structural"),
        ],
        frameon=False,
        loc="upper right",
    )
    _clean_axis(axis)

    for panel, channel in enumerate(CHANNELS):
        axis = axes[1, panel]
        _panel_label(axis, "c" if panel == 0 else "d")
        for arm in MEASURED_ARMS:
            rows = carriage_summary[
                (carriage_summary["arm"] == arm)
                & (carriage_summary["channel"] == channel)
            ].sort_values("distance")
            x_value = rows["distance"].to_numpy()
            mean = rows["mean"].to_numpy()
            axis.plot(
                x_value,
                mean,
                marker="o" if arm == "local" else "s",
                color=ARM_COLOURS[arm],
                linewidth=1.8,
                markersize=4.5,
                label=f"{arm} RRWP",
            )
            axis.fill_between(
                x_value,
                rows["low"].to_numpy(),
                rows["high"].to_numpy(),
                color=ARM_COLOURS[arm],
                alpha=0.14,
                linewidth=0,
            )
        axis.set_xlabel("source-to-carrier distance")
        axis.set_ylabel("raw Functional carriage")
        axis.set_title(f"{channel.capitalize()} carriage")
        axis.set_xticks(range(config.layers + 1))
        axis.set_ylim(bottom=0.0)
        _clean_axis(axis)
        if panel == 1:
            axis.legend(frameon=False, loc="upper right")

    figures = config.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / "01_core_scientific_questions.png"
    pdf = figures / "01_core_scientific_questions.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def plot_score_distance(
    score_summary: pd.DataFrame,
    config: Config,
) -> tuple[Path, Path]:
    _configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.10, top=0.86, wspace=0.28, hspace=0.38)
    figure.suptitle(
        "Exact learned-head semantic and structural score distance",
        fontsize=14.5,
        y=0.97,
        color=TEXT,
    )
    figure.text(
        0.5,
        0.915,
        "Means aggregate native routed-head transport; uncertainty is computed for each layer-head-distance cell",
        ha="center",
        fontsize=9.0,
        color=GREY,
    )
    for row, channel in enumerate(CHANNELS):
        for column, metric in enumerate(("mean", "profile_ci_width")):
            axis = axes[row, column]
            _panel_label(axis, chr(ord("a") + 2 * row + column))
            for arm in MEASURED_ARMS:
                selected = score_summary[
                    (score_summary["arm"] == arm)
                    & (score_summary["channel"] == channel)
                ]
                reduced = selected.groupby("distance")[metric].mean()
                axis.plot(
                    reduced.index,
                    reduced.to_numpy(),
                    marker="o" if arm == "local" else "s",
                    color=ARM_COLOURS[arm],
                    linewidth=1.8,
                    markersize=4.5,
                    label=f"{arm} RRWP",
                )
            axis.set_title(
                f"{channel.capitalize()} score "
                + (
                    "mass"
                    if metric == "mean"
                    else "profile 95% interval width"
                )
            )
            axis.set_ylabel("mean across learned heads")
            axis.set_xticks(range(config.layers + 1))
            axis.set_ylim(bottom=0.0)
            _clean_axis(axis)
            if row == 1:
                axis.set_xlabel("source-to-carrier distance")
            if row == 0 and column == 0:
                axis.legend(frameon=False, loc="upper right")
    figures = config.output_dir / "figures"
    png = figures / "02_score_distance_and_confidence.png"
    pdf = figures / "02_score_distance_and_confidence.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def plot_exact_head_heatmaps(
    score_distance: pd.DataFrame,
    config: Config,
) -> tuple[Path, Path]:
    _configure_style()
    head_profiles = (
        score_distance.groupby(
            ["seed", "arm", "channel", "layer", "head", "distance"]
        )["value"]
        .mean()
        .reset_index()
    )
    seeds = sorted(int(seed) for seed in head_profiles["seed"].unique())
    heads_per_seed = config.layers * config.heads
    figure_height = max(8.1, 2.0 + 0.16 * heads_per_seed * len(seeds))
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.3, figure_height),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(
        left=0.12,
        right=0.92,
        bottom=0.10,
        top=0.87,
        wspace=0.20,
        hspace=0.30,
    )
    figure.suptitle(
        "Every learned head: exact score mass by source-to-carrier distance",
        fontsize=14.5,
        y=0.97,
        color=TEXT,
    )
    maximum = float(head_profiles["value"].max())
    image_handle = None
    for row, arm in enumerate(MEASURED_ARMS):
        for column, channel in enumerate(CHANNELS):
            axis = axes[row, column]
            _panel_label(axis, chr(ord("a") + 2 * row + column))
            selected = head_profiles[
                (head_profiles["arm"] == arm)
                & (head_profiles["channel"] == channel)
            ].copy()
            selected["head_index"] = (
                selected["seed"].map({seed: index for index, seed in enumerate(seeds)})
                * heads_per_seed
                + selected["layer"].astype(int) * config.heads
                + selected["head"].astype(int)
            )
            matrix = selected.pivot(
                index="head_index", columns="distance", values="value"
            ).reindex(
                index=range(heads_per_seed * len(seeds)),
                columns=range(config.layers + 1),
                fill_value=0.0,
            )
            image_handle = axis.imshow(
                matrix.to_numpy(),
                aspect="auto",
                interpolation="nearest",
                cmap="magma",
                vmin=0.0,
                vmax=max(maximum, 1.0e-12),
            )
            axis.set_title(f"{arm.capitalize()} RRWP — {channel}")
            axis.set_xticks(range(config.layers + 1))
            labels = [
                f"S{seed}·L{layer + 1}H{head + 1}"
                for seed in seeds
                for layer in range(config.layers)
                for head in range(config.heads)
            ]
            axis.set_yticks(range(len(labels)), labels)
            axis.tick_params(axis="y", labelsize=5.5)
            for seed_boundary in range(1, len(seeds)):
                axis.axhline(
                    seed_boundary * heads_per_seed - 0.5,
                    color="white",
                    linewidth=0.45,
                    alpha=0.65,
                )
            if row == 1:
                axis.set_xlabel("source-to-carrier distance")
            if column == 0:
                axis.set_ylabel("learned attention head")
    if image_handle is not None:
        colourbar = figure.colorbar(image_handle, ax=axes, fraction=0.025, pad=0.025)
        colourbar.set_label("raw projected score mass")
    figures = config.output_dir / "figures"
    png = figures / "03_exact_per_head_score_distance.png"
    pdf = figures / "03_exact_per_head_score_distance.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def plot_reliability_sensitivity(
    run_directories: Sequence[Path],
    output_directory: Path,
) -> tuple[Path, Path]:
    """Compare completed runs that differ only in local-cue reliability."""
    records: list[dict[str, Any]] = []
    for directory in run_directories:
        with (directory / "summary.json").open(encoding="utf-8") as handle:
            payload = json.load(handle)
        reliability = float(payload["config"].get("local_clue_reliability", 0.75))
        records.append(
            {
                "reliability": reliability,
                "results": payload["results"],
            }
        )
    records.sort(key=lambda record: record["reliability"])
    if len(records) < 2:
        raise ValueError("sensitivity comparison needs at least two completed runs")

    reliability = np.asarray(
        [record["reliability"] for record in records], dtype=float
    )
    _configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.3))
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.10,
        top=0.87,
        wspace=0.30,
        hspace=0.40,
    )
    figure.suptitle(
        "Robustness to the quality of the local structural cue",
        fontsize=14.5,
        y=0.97,
        color=TEXT,
    )
    figure.text(
        0.5,
        0.92,
        "Only cue reliability changes; architecture, parameter count, data size, and optimization are fixed",
        ha="center",
        fontsize=9.0,
        color=GREY,
    )

    axis = axes[0, 0]
    _panel_label(axis, "a")
    for arm in ARMS:
        values = [
            record["results"]["performance"][arm]["test_mse"]
            for record in records
        ]
        axis.plot(
            reliability,
            values,
            marker="o" if arm == "local" else "s",
            color=ARM_COLOURS[arm],
            linewidth=1.8,
            label=arm.replace("_", " "),
        )
    axis.set_title("Global advantage persists as the local cue improves")
    axis.set_ylabel("held-out MSE")
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, loc="best")
    _clean_axis(axis)

    axis = axes[0, 1]
    _panel_label(axis, "b")
    for arm in MEASURED_ARMS:
        for channel in CHANNELS:
            values = [
                record["results"]["head_scores"][arm][f"{channel}_raw_mean"]
                for record in records
            ]
            axis.plot(
                reliability,
                values,
                marker=CHANNEL_MARKERS[channel],
                linestyle="-" if arm == "local" else "--",
                color=ARM_COLOURS[arm],
                linewidth=1.8,
                label=f"{arm} {channel}",
            )
    axis.set_title("Raw score separation is stable")
    axis.set_ylabel("mean raw head score")
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, loc="best", ncols=2, fontsize=7.3)
    _clean_axis(axis)

    axis = axes[1, 0]
    _panel_label(axis, "c")
    for channel in CHANNELS:
        values = [
            record["results"]["distance"][channel][
                "carriage_profile_total_variation"
            ]
            for record in records
        ]
        axis.plot(
            reliability,
            values,
            marker=CHANNEL_MARKERS[channel],
            color=BLUE if channel == "semantic" else ORANGE,
            linewidth=1.8,
            label=channel,
        )
    axis.set_title("Carriage geometry can converge (0 = exact match)")
    axis.set_ylabel("local/global profile TV")
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, loc="best")
    _clean_axis(axis)

    axis = axes[1, 1]
    _panel_label(axis, "d")
    axis.axhline(1.0, color=GREY, linewidth=1.0, linestyle=":")
    for channel in CHANNELS:
        values = [
            record["results"]["distance"][channel][
                "score_profile_ci_width_ratio_local_over_global"
            ]
            for record in records
        ]
        axis.plot(
            reliability,
            values,
            marker=CHANNEL_MARKERS[channel],
            color=BLUE if channel == "semantic" else ORANGE,
            linewidth=1.8,
            label=channel,
        )
    axis.set_title("Local score intervals are not wider")
    axis.set_ylabel("profile interval width: local / global")
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, loc="best")
    _clean_axis(axis)

    for axis in axes[1, :]:
        axis.set_xlabel("probability local cue is correct")
    for axis in axes.flat:
        axis.set_xticks(reliability)
        axis.set_xticklabels([f"{value:.2f}" for value in reliability])

    figures = output_directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / "04_local_cue_reliability_sensitivity.png"
    pdf = figures / "04_local_cue_reliability_sensitivity.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def plot_mediation_alignment(
    run_directories: Sequence[Path],
    output_directory: Path,
) -> tuple[Path, Path]:
    """Show whether semantic and structural distance profiles align per seed."""
    records: list[dict[str, Any]] = []
    for directory in run_directories:
        with (directory / "summary.json").open(encoding="utf-8") as handle:
            payload = json.load(handle)
        reliability = float(payload["config"].get("local_clue_reliability", 0.75))
        scores = pd.read_csv(directory / "score_distance_per_graph.csv")
        profiles = (
            scores.groupby(["seed", "arm", "layer", "head", "channel", "distance"])[
                "value"
            ]
            .mean()
            .reset_index()
        )
        for (seed, arm), seed_rows in profiles.groupby(["seed", "arm"]):
            cosines: list[float] = []
            peak_matches: list[float] = []
            for (_, _), head_rows in seed_rows.groupby(["layer", "head"]):
                channel_profiles = {
                    channel: head_rows[head_rows["channel"] == channel]
                    .sort_values("distance")["value"]
                    .to_numpy()
                    for channel in CHANNELS
                }
                semantic = channel_profiles["semantic"]
                structural = channel_profiles["structural"]
                denominator = float(
                    np.linalg.norm(semantic) * np.linalg.norm(structural)
                )
                if denominator > 1.0e-12:
                    cosines.append(float(np.dot(semantic, structural) / denominator))
                peak_matches.append(float(np.argmax(semantic) == np.argmax(structural)))
            records.append(
                {
                    "reliability": reliability,
                    "seed": int(seed),
                    "arm": str(arm),
                    "cosine": float(np.mean(cosines)),
                    "peak_match": float(np.mean(peak_matches)),
                }
            )
    table = pd.DataFrame(records).sort_values(["reliability", "seed", "arm"])
    reliabilities = sorted(float(value) for value in table["reliability"].unique())
    _configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    figure.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.20,
        top=0.79,
        wspace=0.30,
    )
    figure.suptitle(
        "Does global RRWP improve structural–semantic mediation alignment?",
        fontsize=14.5,
        y=0.97,
        color=TEXT,
    )
    figure.text(
        0.5,
        0.88,
        "Each line pairs local and global models for one training seed; metrics use every learned head",
        ha="center",
        fontsize=9.0,
        color=GREY,
    )
    metrics = (
        ("cosine", "Mean profile cosine", "Distance profiles align more under global RRWP"),
        ("peak_match", "Fraction with same peak distance", "Peak distance agrees more often under global RRWP"),
    )
    positions: dict[tuple[float, str], float] = {}
    labels: list[str] = []
    tick_positions: list[float] = []
    for reliability_index, reliability in enumerate(reliabilities):
        base = reliability_index * 2.7
        for arm_index, arm in enumerate(MEASURED_ARMS):
            position = base + arm_index
            positions[(reliability, arm)] = position
            tick_positions.append(position)
            labels.append(f"{arm}\nr={reliability:.2f}")
    for panel, (metric, ylabel, title) in enumerate(metrics):
        axis = axes[panel]
        _panel_label(axis, "a" if panel == 0 else "b")
        for reliability in reliabilities:
            selected = table[table["reliability"] == reliability]
            pivot = selected.pivot(index="seed", columns="arm", values=metric)
            local_x = positions[(reliability, "local")]
            global_x = positions[(reliability, "global")]
            for _, row in pivot.iterrows():
                axis.plot(
                    [local_x, global_x],
                    [row["local"], row["global"]],
                    color=LIGHT_GREY,
                    linewidth=1.0,
                    zorder=1,
                )
            for arm, position in (("local", local_x), ("global", global_x)):
                values = pivot[arm].to_numpy()
                axis.scatter(
                    np.full(len(values), position),
                    values,
                    color=ARM_COLOURS[arm],
                    s=26,
                    zorder=2,
                )
                axis.hlines(
                    float(values.mean()),
                    position - 0.20,
                    position + 0.20,
                    color=TEXT,
                    linewidth=1.5,
                    zorder=3,
                )
        axis.set_xticks(tick_positions, labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_ylim(0.0, 1.0)
        _clean_axis(axis)

    figures = output_directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / "05_semantic_structural_alignment.png"
    pdf = figures / "05_semantic_structural_alignment.pdf"
    table.to_csv(output_directory / "mediation_alignment_by_seed.csv", index=False)
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _head_summary(head_scores: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        head_scores.groupby(["seed", "arm", "layer", "head", "channel"])["value"]
        .mean()
        .unstack("channel")
        .reset_index()
    )
    semantic_mean = pivot.groupby(["seed", "arm"])["semantic"].transform("mean")
    structural_mean = pivot.groupby(["seed", "arm"])["structural"].transform("mean")
    pivot["semantic_normalized"] = pivot["semantic"] / semantic_mean.clip(lower=1.0e-12)
    pivot["structural_normalized"] = pivot["structural"] / structural_mean.clip(lower=1.0e-12)
    total = pivot["semantic_normalized"] + pivot["structural_normalized"]
    pivot["J"] = 0.5 * total
    pivot["D_rel"] = (
        pivot["semantic_normalized"] - pivot["structural_normalized"]
    ) / (total + 1.0e-12)
    pivot["regime"] = np.where(
        pivot["D_rel"] > 0.5,
        "semantic-leaning",
        np.where(pivot["D_rel"] < -0.5, "structural-leaning", "generalist"),
    )
    return pivot


def _summary_payload(
    performance: pd.DataFrame,
    head_summary: pd.DataFrame,
    score_distance: pd.DataFrame,
    score_summary: pd.DataFrame,
    carriage_summary: pd.DataFrame,
) -> dict[str, Any]:
    result: dict[str, Any] = {"performance": {}, "head_scores": {}, "distance": {}}
    for arm, rows in performance.groupby("arm"):
        result["performance"][str(arm)] = {
            column: float(rows[column].mean())
            for column in ("test_mse", "test_mae", "selector_cycle_attention")
        }
    for arm, rows in head_summary.groupby("arm"):
        counts = rows["regime"].value_counts(normalize=True)
        result["head_scores"][str(arm)] = {
            "semantic_raw_mean": float(rows["semantic"].mean()),
            "structural_raw_mean": float(rows["structural"].mean()),
            "semantic_structural_correlation": float(
                rows[["semantic", "structural"]].corr().iloc[0, 1]
            ),
            "generalist_fraction": float(counts.get("generalist", 0.0)),
            "semantic_leaning_fraction": float(counts.get("semantic-leaning", 0.0)),
            "structural_leaning_fraction": float(counts.get("structural-leaning", 0.0)),
        }
        distance_rows = (
            score_distance[score_distance["arm"] == arm]
            .groupby(["seed", "layer", "head", "channel", "distance"])["value"]
            .mean()
            .reset_index()
        )
        peak_matches: list[float] = []
        profile_cosines: list[float] = []
        for (_, _, _), head_distance in distance_rows.groupby(
            ["seed", "layer", "head"]
        ):
            profiles = {
                channel: head_distance[head_distance["channel"] == channel]
                .sort_values("distance")["value"]
                .to_numpy()
                for channel in CHANNELS
            }
            semantic = profiles["semantic"]
            structural = profiles["structural"]
            if len(semantic) != len(structural) or not len(semantic):
                continue
            peak_matches.append(float(np.argmax(semantic) == np.argmax(structural)))
            denominator = float(np.linalg.norm(semantic) * np.linalg.norm(structural))
            if denominator > 1.0e-12:
                profile_cosines.append(float(np.dot(semantic, structural) / denominator))
        result["head_scores"][str(arm)]["distance_peak_match_fraction"] = float(
            np.mean(peak_matches)
        )
        result["head_scores"][str(arm)]["distance_profile_cosine_mean"] = float(
            np.mean(profile_cosines)
        )
    for channel in CHANNELS:
        local_score = score_summary[
            (score_summary["arm"] == "local") & (score_summary["channel"] == channel)
        ]
        global_score = score_summary[
            (score_summary["arm"] == "global") & (score_summary["channel"] == channel)
        ]
        local_carriage = carriage_summary[
            (carriage_summary["arm"] == "local")
            & (carriage_summary["channel"] == channel)
        ].sort_values("distance")
        global_carriage = carriage_summary[
            (carriage_summary["arm"] == "global")
            & (carriage_summary["channel"] == channel)
        ].sort_values("distance")
        local_profile = local_carriage["mean"].to_numpy(copy=True)
        global_profile = global_carriage["mean"].to_numpy(copy=True)
        local_profile /= max(float(local_profile.sum()), 1.0e-12)
        global_profile /= max(float(global_profile.sum()), 1.0e-12)
        result["distance"][channel] = {
            "score_ci_width_local_mean": float(local_score["ci_width"].mean()),
            "score_ci_width_global_mean": float(global_score["ci_width"].mean()),
            "score_ci_width_ratio_local_over_global": float(
                local_score["ci_width"].mean()
                / max(float(global_score["ci_width"].mean()), 1.0e-12)
            ),
            "score_profile_ci_width_local_mean": float(
                local_score["profile_ci_width"].mean()
            ),
            "score_profile_ci_width_global_mean": float(
                global_score["profile_ci_width"].mean()
            ),
            "score_profile_ci_width_ratio_local_over_global": float(
                local_score["profile_ci_width"].mean()
                / max(float(global_score["profile_ci_width"].mean()), 1.0e-12)
            ),
            "carriage_profile_total_variation": float(
                0.5 * np.abs(local_profile - global_profile).sum()
            ),
            "carriage_total_local": float(local_carriage["mean"].sum()),
            "carriage_total_global": float(global_carriage["mean"].sum()),
        }
    return result


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
    score_rows: list[dict[str, Any]] = []
    carriage_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    health: dict[str, Any] = {}
    for arm in ARMS:
        for seed in config.seeds:
            model, training = train_model(
                config,
                graph_types,
                tensors,
                train,
                validation,
                arm=arm,
                seed=int(seed),
            )
            performance_rows.append(
                {
                    **evaluate(model, test, tensors, arm=arm, seed=int(seed)),
                    **training,
                }
            )
            health[f"{arm}_{seed}"] = training
            if arm in MEASURED_ARMS:
                score, carriage, heads = measure_scores_and_carriage(
                    model,
                    test,
                    graph_types,
                    tensors,
                    config,
                    arm=arm,
                    seed=int(seed),
                )
                score_rows.extend(score)
                carriage_rows.extend(carriage)
                head_rows.extend(heads)

    performance = pd.DataFrame(performance_rows)
    score_distance = pd.DataFrame(score_rows)
    carriage = pd.DataFrame(carriage_rows)
    head_scores = pd.DataFrame(head_rows)
    score_summary, carriage_summary = summarise_intervals(
        score_distance, carriage, config
    )
    head_summary = _head_summary(head_scores)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "performance.csv": performance,
        "score_distance_per_graph.csv": score_distance,
        "score_distance_summary.csv": score_summary,
        "carriage_distance_per_graph.csv": carriage,
        "carriage_distance_summary.csv": carriage_summary,
        "head_scores_per_graph.csv": head_scores,
        "head_scores_summary.csv": head_summary,
    }
    for name, table in tables.items():
        table.to_csv(config.output_dir / name, index=False)
    figures = [
        *plot_core_questions(performance, head_scores, carriage_summary, config),
        *plot_score_distance(score_summary, config),
        *plot_exact_head_heatmaps(score_distance, config),
    ]
    summary = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "results": _summary_payload(
            performance,
            head_summary,
            score_distance,
            score_summary,
            carriage_summary,
        ),
        "health": health,
        "figures": [str(path) for path in figures],
    }
    with (config.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def reanalyze_output(output_directory: Path) -> dict[str, Any]:
    """Regenerate summaries and figures from saved measurements, without training."""
    with (output_directory / "summary.json").open(encoding="utf-8") as handle:
        previous = json.load(handle)
    config_payload = dict(previous["config"])
    config_payload["output_dir"] = output_directory
    config_payload["seeds"] = tuple(int(seed) for seed in config_payload["seeds"])
    config = Config(**config_payload)
    performance = pd.read_csv(output_directory / "performance.csv")
    score_distance = pd.read_csv(output_directory / "score_distance_per_graph.csv")
    carriage = pd.read_csv(output_directory / "carriage_distance_per_graph.csv")
    head_scores = pd.read_csv(output_directory / "head_scores_per_graph.csv")
    score_summary, carriage_summary = summarise_intervals(
        score_distance, carriage, config
    )
    head_summary = _head_summary(head_scores)
    score_summary.to_csv(output_directory / "score_distance_summary.csv", index=False)
    carriage_summary.to_csv(
        output_directory / "carriage_distance_summary.csv", index=False
    )
    head_summary.to_csv(output_directory / "head_scores_summary.csv", index=False)
    figures = [
        *plot_core_questions(performance, head_scores, carriage_summary, config),
        *plot_score_distance(score_summary, config),
        *plot_exact_head_heatmaps(score_distance, config),
    ]
    summary = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "results": _summary_payload(
            performance,
            head_summary,
            score_distance,
            score_summary,
            carriage_summary,
        ),
        "health": previous["health"],
        "figures": [str(path) for path in figures],
    }
    with (output_directory / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/chapter6_rrwp_semantic_relay_v1"),
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--training-steps", type=int, default=600)
    parser.add_argument("--train-examples", type=int, default=2_048)
    parser.add_argument("--test-examples", type=int, default=1_024)
    parser.add_argument("--measurement-examples", type=int, default=192)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--local-clue-reliability", type=float, default=0.75)
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
        test_examples=int(args.test_examples),
        measurement_examples=int(args.measurement_examples),
        bootstrap_replicates=int(args.bootstrap_replicates),
        local_clue_reliability=float(args.local_clue_reliability),
    )
    result = run(config)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
