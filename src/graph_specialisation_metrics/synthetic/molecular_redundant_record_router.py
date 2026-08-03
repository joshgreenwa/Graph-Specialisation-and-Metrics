"""Redundant local and dense routes on real molecular graph supports.

Every example is placed on a ZINC molecular topology.  Four distant record
nodes form a two-bank, two-key memory: semantic context selects the key and
structural context selects the bank.  The selected record value is also copied
to the anchor, so the task is exactly solvable without any distant route.

A learned softmax router retrieves the distant record and a fixed residual mix
combines that prediction with the local copy.  Sweeping the mix constructs
equally accurate models with different implemented reach while task necessity
remains zero.  The semantic-by-structural finite contrast therefore detects a
real pathway used by the model, but deliberately cannot establish that the
pathway is required by the task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .molecular_nonlinear_reach import shortest_path_matrix

DTYPE = torch.float64
BANKS = torch.tensor((0, 0, 1, 1), dtype=torch.long)
KEYS = torch.tensor((0, 1, 0, 1), dtype=torch.long)
# (bank 0, key 0), (bank 0, key 1), (bank 1, key 0), (bank 1, key 1).
# The repeated distance four makes the canonical semantic and structural
# marginal profiles symmetric while retaining distinct physical carrier nodes.
RECORD_DISTANCES = torch.tensor((1, 4, 4, 6), dtype=torch.long)
FIELD_DISTANCES = np.asarray((0, 1, 4, 4, 6), dtype=np.int64)
UNIQUE_DISTANCES = np.asarray((0, 1, 4, 6), dtype=np.int64)


@dataclass(frozen=True)
class Config:
    output_dir: Path
    data_root: Path
    train_graphs: int = 64
    val_graphs: int = 16
    test_graphs: int = 16
    examples_per_graph: int = 8
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    local_mixes: tuple[float, ...] = (0.25, 0.5, 0.75)
    max_steps: int = 4_000
    learning_rate: float = 2.0e-2
    weight_decay: float = 1.0e-5
    early_stop_mse: float = 1.0e-10
    data_seed: int = 80_203

    def validate(self) -> None:
        if min(self.train_graphs, self.val_graphs, self.test_graphs) < 1:
            raise ValueError("each split requires at least one graph")
        if self.examples_per_graph < 4:
            raise ValueError("examples_per_graph must be at least four")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be non-empty and unique")
        if (
            not self.local_mixes
            or tuple(sorted(set(self.local_mixes))) != self.local_mixes
            or any(not 0 < value < 1 for value in self.local_mixes)
        ):
            raise ValueError("local_mixes must be unique, increasing, and lie in (0,1)")
        if self.max_steps < 1 or self.learning_rate <= 0 or self.early_stop_mse <= 0:
            raise ValueError("invalid optimisation settings")


@dataclass(frozen=True)
class MolecularSupport:
    graph_id: int
    anchor: int
    record_nodes: tuple[int, int, int, int]
    record_distances: tuple[int, int, int, int]


@dataclass(frozen=True)
class RouterDataset:
    graph_ids: torch.Tensor
    query: torch.Tensor
    structure: torch.Tensor
    record_values: torch.Tensor
    local_value: torch.Tensor
    target: torch.Tensor

    def __len__(self) -> int:
        return int(self.target.numel())


def _split_offset(split: str) -> int:
    return {"train": 0, "val": 20_000, "test": 40_000}[split]


def load_molecular_supports(
    data_root: Path,
    *,
    split: str,
    count: int,
    seed: int,
) -> list[MolecularSupport]:
    """Select anchors with records at distances 1, 4, 4, and 6."""

    try:
        from torch_geometric.datasets import ZINC
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("This experiment requires torch_geometric") from error

    dataset = ZINC(root=str(data_root), subset=True, split=split)
    rng = np.random.default_rng(int(seed))
    supports: list[MolecularSupport] = []
    for dataset_index in rng.permutation(len(dataset)).tolist():
        graph = dataset[int(dataset_index)]
        spd = shortest_path_matrix(int(graph.num_nodes), graph.edge_index)
        eligible = [
            node
            for node in range(int(graph.num_nodes))
            if int((spd[node] == 1).sum()) >= 1
            and int((spd[node] == 4).sum()) >= 2
            and int((spd[node] == 6).sum()) >= 1
        ]
        if not eligible:
            continue
        anchor = int(rng.choice(eligible))
        at_one = torch.where(spd[anchor] == 1)[0].numpy()
        at_four = torch.where(spd[anchor] == 4)[0].numpy()
        at_six = torch.where(spd[anchor] == 6)[0].numpy()
        record_nodes = (
            int(rng.choice(at_one)),
            *tuple(int(value) for value in rng.choice(at_four, size=2, replace=False)),
            int(rng.choice(at_six)),
        )
        supports.append(
            MolecularSupport(
                graph_id=_split_offset(split) + int(dataset_index),
                anchor=anchor,
                record_nodes=record_nodes,
                record_distances=(1, 4, 4, 6),
            )
        )
        if len(supports) >= int(count):
            break
    if len(supports) != int(count):
        raise RuntimeError(f"found {len(supports)}/{count} eligible {split} molecules")
    return supports


def make_router_dataset(
    supports: Sequence[MolecularSupport],
    *,
    examples_per_graph: int,
    seed: int,
) -> RouterDataset:
    rng = np.random.default_rng(int(seed))
    graph_ids: list[int] = []
    queries: list[int] = []
    structures: list[int] = []
    values: list[np.ndarray] = []
    targets: list[float] = []
    states = np.asarray(((0, 0), (1, 0), (0, 1), (1, 1)), dtype=np.int64)
    for support in supports:
        order = np.arange(int(examples_per_graph), dtype=np.int64) % 4
        rng.shuffle(order)
        for state_index in order.tolist():
            query, structure = states[int(state_index)]
            signs = rng.choice(np.asarray((-1.0, 1.0)), size=4)
            magnitudes = rng.uniform(0.5, 1.5, size=4)
            record_values = signs * magnitudes
            selected = int(2 * structure + query)
            target = float(record_values[selected])
            graph_ids.append(int(support.graph_id))
            queries.append(int(query))
            structures.append(int(structure))
            values.append(record_values)
            targets.append(target)
    target_tensor = torch.tensor(targets, dtype=DTYPE)
    return RouterDataset(
        graph_ids=torch.tensor(graph_ids, dtype=torch.long),
        query=torch.tensor(queries, dtype=torch.long),
        structure=torch.tensor(structures, dtype=torch.long),
        record_values=torch.tensor(np.stack(values), dtype=DTYPE),
        local_value=target_tensor.clone(),
        target=target_tensor,
    )


class RedundantRecordRouter(nn.Module):
    """Categorical softmax record router plus an exact local residual route."""

    def __init__(self, local_mix: float, *, seed: int) -> None:
        super().__init__()
        if not 0 < float(local_mix) < 1:
            raise ValueError("local_mix must lie in (0,1)")
        torch.manual_seed(int(seed))
        self.local_mix = float(local_mix)
        self.semantic_logits = nn.Parameter(torch.empty(2, 2, dtype=DTYPE))
        self.structural_logits = nn.Parameter(torch.empty(2, 2, dtype=DTYPE))
        self.distance_bias = nn.Parameter(torch.zeros(7, dtype=DTYPE))
        nn.init.normal_(self.semantic_logits, std=0.02)
        nn.init.normal_(self.structural_logits, std=0.02)

    def attention(self, query: torch.Tensor, structure: torch.Tensor) -> torch.Tensor:
        semantic = self.semantic_logits[query][:, KEYS]
        structural = self.structural_logits[structure][:, BANKS]
        logits = semantic + structural + self.distance_bias[RECORD_DISTANCES]
        return torch.softmax(logits, dim=-1)

    def global_route(
        self,
        query: torch.Tensor,
        structure: torch.Tensor,
        record_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention = self.attention(query, structure)
        return torch.sum(attention * record_values, dim=-1), attention

    def mix_tensor(self) -> torch.Tensor:
        return self.semantic_logits.new_tensor(self.local_mix)

    def effective_local_mix(self) -> float:
        return float(self.local_mix)

    def forward(self, dataset: RouterDataset) -> torch.Tensor:
        global_value, _ = self.global_route(
            dataset.query, dataset.structure, dataset.record_values
        )
        local_mix = self.mix_tensor()
        return local_mix * dataset.local_value + (1 - local_mix) * global_value

    def contribution_field(
        self,
        dataset: RouterDataset,
        *,
        query: torch.Tensor,
        structure: torch.Tensor,
    ) -> torch.Tensor:
        attention = self.attention(query, structure)
        local_mix = self.mix_tensor()
        local = local_mix * dataset.local_value[:, None]
        distant = (1 - local_mix) * attention * dataset.record_values
        return torch.cat((local, distant), dim=-1)


class LearnedMixRecordRouter(RedundantRecordRouter):
    """Same router with a learned local-versus-global residual mixture."""

    def __init__(self, *, seed: int) -> None:
        super().__init__(0.5, seed=int(seed))
        self.local_mix_logit = nn.Parameter(torch.zeros((), dtype=DTYPE))

    def mix_tensor(self) -> torch.Tensor:
        return torch.sigmoid(self.local_mix_logit)

    def effective_local_mix(self) -> float:
        return float(self.mix_tensor().detach())


def train_model(
    config: Config,
    train: RouterDataset,
    validation: RouterDataset,
    *,
    local_mix: float,
    seed: int,
) -> tuple[RedundantRecordRouter, dict[str, float | int]]:
    model = RedundantRecordRouter(local_mix, seed=int(seed))
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    step = 0
    for step in range(1, int(config.max_steps) + 1):
        prediction = model(train)
        loss = torch.mean((prediction - train.target) ** 2)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            validation_loss = float(torch.mean((model(validation) - validation.target) ** 2))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
        if validation_loss <= float(config.early_stop_mse):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(validation)
        global_prediction, attention = model.global_route(
            validation.query, validation.structure, validation.record_values
        )
        selected = 2 * validation.structure + validation.query
        selected_attention = attention.gather(1, selected[:, None]).squeeze(1)
    return model, {
        "steps": int(step),
        "validation_mae": float(torch.mean(torch.abs(prediction - validation.target))),
        "global_route_mae": float(
            torch.mean(torch.abs(global_prediction - validation.target))
        ),
        "mean_selected_attention": float(selected_attention.mean()),
        "routing_accuracy": float((torch.argmax(attention, dim=-1) == selected).to(DTYPE).mean()),
    }


def train_learned_mix_model(
    config: Config,
    train: RouterDataset,
    validation: RouterDataset,
    *,
    seed: int,
) -> tuple[LearnedMixRecordRouter, dict[str, float | int]]:
    """Train without forcing use of either route."""

    model = LearnedMixRecordRouter(seed=int(seed))
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    step = 0
    for step in range(1, int(config.max_steps) + 1):
        loss = torch.mean((model(train) - train.target) ** 2)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            validation_loss = float(torch.mean((model(validation) - validation.target) ** 2))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
        if validation_loss <= float(config.early_stop_mse):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(validation)
        global_prediction, attention = model.global_route(
            validation.query, validation.structure, validation.record_values
        )
        selected = 2 * validation.structure + validation.query
    return model, {
        "steps": int(step),
        "validation_mae": float(torch.mean(torch.abs(prediction - validation.target))),
        "global_route_mae": float(
            torch.mean(torch.abs(global_prediction - validation.target))
        ),
        "effective_local_mix": model.effective_local_mix(),
        "mean_selected_attention": float(
            attention.gather(1, selected[:, None]).mean()
        ),
        "routing_accuracy": float(
            (torch.argmax(attention, dim=-1) == selected).to(DTYPE).mean()
        ),
    }


def _profile(response: torch.Tensor) -> tuple[np.ndarray, float, float]:
    mass = torch.abs(response).detach().cpu().numpy()
    grouped = np.stack(
        [mass[:, FIELD_DISTANCES == distance].sum(axis=1) for distance in UNIQUE_DISTANCES],
        axis=1,
    )
    total = grouped.sum(axis=1)
    valid = total > 1.0e-12
    normalised = np.zeros_like(grouped)
    normalised[valid] = grouped[valid] / total[valid, None]
    expected = normalised @ UNIQUE_DISTANCES
    far = normalised[:, UNIQUE_DISTANCES >= 4].sum(axis=1)
    return normalised, float(np.mean(expected[valid])), float(np.mean(far[valid]))


def evaluate(model: RedundantRecordRouter, dataset: RouterDataset) -> dict[str, Any]:
    with torch.no_grad():
        prediction = model(dataset)
        global_prediction, attention = model.global_route(
            dataset.query, dataset.structure, dataset.record_values
        )
        selected = 2 * dataset.structure + dataset.query
        selected_attention = attention.gather(1, selected[:, None]).squeeze(1)
        attention_expected_distance = torch.sum(
            attention * RECORD_DISTANCES.to(DTYPE), dim=-1
        )

        query_donor = 1 - dataset.query
        structure_donor = 1 - dataset.structure
        clean = model.contribution_field(
            dataset, query=dataset.query, structure=dataset.structure
        )
        semantic = model.contribution_field(
            dataset, query=query_donor, structure=dataset.structure
        )
        structural = model.contribution_field(
            dataset, query=dataset.query, structure=structure_donor
        )
        joint = model.contribution_field(
            dataset, query=query_donor, structure=structure_donor
        )
        semantic_response = clean - semantic
        structural_response = clean - structural
        interaction = clean - semantic - structural + joint

    semantic_profile, semantic_distance, semantic_far = _profile(semantic_response)
    structural_profile, structural_distance, structural_far = _profile(structural_response)
    _interaction_profile, interaction_distance, interaction_far = _profile(interaction)
    profile_tv = 0.5 * np.abs(
        semantic_profile.mean(axis=0) - structural_profile.mean(axis=0)
    ).sum()
    local_mix = model.effective_local_mix()
    no_global = local_mix * dataset.local_value
    return {
        "effective_local_mix": local_mix,
        "effective_global_mix": float(1 - local_mix),
        "full_mae": float(torch.mean(torch.abs(prediction - dataset.target))),
        "global_route_mae": float(
            torch.mean(torch.abs(global_prediction - dataset.target))
        ),
        "frozen_global_removed_mae": float(
            torch.mean(torch.abs(no_global - dataset.target))
        ),
        "local_refit_mae": 0.0,
        "mean_selected_attention": float(selected_attention.mean()),
        "routing_accuracy": float((torch.argmax(attention, dim=-1) == selected).to(DTYPE).mean()),
        "attention_expected_distance": float(attention_expected_distance.mean()),
        "semantic_effect_mass": float(torch.abs(semantic_response).sum(dim=-1).mean()),
        "structural_effect_mass": float(torch.abs(structural_response).sum(dim=-1).mean()),
        "interaction_effect_mass": float(torch.abs(interaction).sum(dim=-1).mean()),
        "mean_abs_output_interaction": float(torch.abs(interaction.sum(dim=-1)).mean()),
        "semantic_expected_distance": semantic_distance,
        "structural_expected_distance": structural_distance,
        "interaction_expected_distance": interaction_distance,
        "semantic_far_share": semantic_far,
        "structural_far_share": structural_far,
        "interaction_far_share": interaction_far,
        "semantic_structural_profile_tv": float(profile_tv),
    }


def _mean_records(records: Sequence[Mapping[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([float(record[key]) for record in records]))
        for key in records[0]
    }


def _serialise(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _serialise(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialise(item) for item in value]
    return value


def run(config: Config) -> dict[str, Any]:
    config.validate()
    torch.set_num_threads(1)
    split_specs = (
        ("train", config.train_graphs, config.data_seed + 1),
        ("val", config.val_graphs, config.data_seed + 2),
        ("test", config.test_graphs, config.data_seed + 3),
    )
    supports = {
        split: load_molecular_supports(
            config.data_root, split=split, count=count, seed=seed
        )
        for split, count, seed in split_specs
    }
    datasets = {
        split: make_router_dataset(
            supports[split],
            examples_per_graph=config.examples_per_graph,
            seed=config.data_seed + 100 + index,
        )
        for index, (split, _, _) in enumerate(split_specs)
    }

    result: dict[str, Any] = {
        "config": asdict(config),
        "support": {
            split: {
                "graphs": len(items),
                "examples": len(datasets[split]),
                "record_distances": list(items[0].record_distances),
            }
            for split, items in supports.items()
        },
        "local_oracle": {
            "full_mae": 0.0,
            "task_necessity_mae": 0.0,
            "implemented_max_distance": 0,
        },
        "mixtures": {},
    }
    for local_mix in config.local_mixes:
        seed_records: list[dict[str, Any]] = []
        for seed in config.seeds:
            model, training = train_model(
                config,
                datasets["train"],
                datasets["val"],
                local_mix=float(local_mix),
                seed=int(seed),
            )
            seed_records.append(
                {
                    "seed": int(seed),
                    "training": training,
                    "test": evaluate(model, datasets["test"]),
                }
            )
        result["mixtures"][str(local_mix)] = {
            "local_mix": float(local_mix),
            "global_mix": float(1 - local_mix),
            "seeds": seed_records,
            "test_mean": _mean_records([record["test"] for record in seed_records]),
            "test_max_full_mae": float(
                max(record["test"]["full_mae"] for record in seed_records)
            ),
            "test_min_routing_accuracy": float(
                min(record["test"]["routing_accuracy"] for record in seed_records)
            ),
        }

    learned_seed_records: list[dict[str, Any]] = []
    for seed in config.seeds:
        model, training = train_learned_mix_model(
            config,
            datasets["train"],
            datasets["val"],
            seed=int(seed),
        )
        learned_seed_records.append(
            {
                "seed": int(seed),
                "training": training,
                "test": evaluate(model, datasets["test"]),
            }
        )
    result["learned_mix"] = {
        "seeds": learned_seed_records,
        "test_mean": _mean_records(
            [record["test"] for record in learned_seed_records]
        ),
        "test_min_local_mix": float(
            min(record["test"]["effective_local_mix"] for record in learned_seed_records)
        ),
        "test_max_local_mix": float(
            max(record["test"]["effective_local_mix"] for record in learned_seed_records)
        ),
    }

    payload = _serialise(result)
    fingerprint_payload = {
        **payload["config"],
        "output_dir": None,
        "protocol": "molecular-redundant-record-router-v1",
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/molecular_redundant_record_router")
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-graphs", type=int, default=64)
    parser.add_argument("--val-graphs", type=int, default=16)
    parser.add_argument("--test-graphs", type=int, default=16)
    parser.add_argument("--examples-per-graph", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--local-mixes", default="0.25,0.5,0.75")
    parser.add_argument("--max-steps", type=int, default=4_000)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    config = Config(
        output_dir=args.output_dir,
        data_root=args.data_root,
        train_graphs=int(args.train_graphs),
        val_graphs=int(args.val_graphs),
        test_graphs=int(args.test_graphs),
        examples_per_graph=int(args.examples_per_graph),
        seeds=_parse_ints(args.seeds),
        local_mixes=_parse_floats(args.local_mixes),
        max_steps=int(args.max_steps),
    )
    result = run(config)
    summary = {key: value["test_mean"] for key, value in result["mixtures"].items()}
    summary["learned"] = result["learned_mix"]["test_mean"]
    print(json.dumps(summary, indent=2))
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
