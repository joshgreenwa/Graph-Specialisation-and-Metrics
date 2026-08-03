"""A redundant-route stress test with two overlapping noisy clues.

The target is a hidden scalar.  A noisy local clue is stored at an anchor and a
second noisy clue is stored in one of four distant records.  A semantic key and
a structural bank jointly select the distant record.  The two clues have equal
quality, but an adjustable fraction of their noise is shared.  Consequently,
averaging both clues becomes less useful as their mistakes become more alike,
even though a trained model may continue to use the distant route.

Candidate records occupy distances 1, 4, 4, and 6 on supports sampled from
real ZINC graph topologies.  A small dense transformer is trained without an
explicit local/distant mixture parameter.  Clean factorial carriage is compared
with route-specific failures and with simple local-only and two-clue reference
fits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .molecular_multihead_redundancy import (
    Config as TransformerConfig,
)
from .molecular_multihead_redundancy import (
    TinyMultiHeadGraphTransformer,
    measure_internal,
    train_model,
)
from .molecular_redundant_record_router import (
    MolecularSupport,
    RouterDataset,
    load_molecular_supports,
)
from .molecular_redundant_route_robustness import (
    corrupt_local_copy,
    corrupt_selected_distant_record,
)

DTYPE = torch.float32


@dataclass(frozen=True)
class Config:
    output_dir: Path
    data_root: Path
    train_graphs: int = 256
    val_graphs: int = 64
    test_graphs: int = 64
    examples_per_graph: int = 8
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    shared_noise: tuple[float, ...] = (0.0, 0.5, 0.9, 1.0)
    clue_noise: float = 0.50
    train_clue_dropout: float = 0.0
    hidden_dim: int = 32
    layers: int = 2
    heads: int = 4
    max_steps: int = 1_500
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-4
    early_stop_patience: int = 250
    data_seed: int = 91_207

    def validate(self) -> None:
        if min(self.train_graphs, self.val_graphs, self.test_graphs) < 1:
            raise ValueError("each split requires at least one graph")
        if self.examples_per_graph < 4 or self.examples_per_graph % 4:
            raise ValueError("examples_per_graph must be a multiple of four")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be non-empty and unique")
        if (
            not self.shared_noise
            or tuple(sorted(set(self.shared_noise))) != self.shared_noise
            or any(not 0.0 <= value <= 1.0 for value in self.shared_noise)
        ):
            raise ValueError("shared_noise must be unique, increasing, and in [0,1]")
        if self.clue_noise <= 0:
            raise ValueError("clue_noise must be positive")
        if not 0.0 <= self.train_clue_dropout < 0.5:
            raise ValueError("train_clue_dropout must lie in [0,0.5)")
        if self.hidden_dim < 4 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if self.layers < 1 or self.max_steps < 1:
            raise ValueError("layers and max_steps must be positive")


def make_correlated_clue_dataset(
    supports: Sequence[MolecularSupport],
    *,
    examples_per_graph: int,
    shared_noise: float,
    clue_noise: float,
    seed: int,
) -> RouterDataset:
    """Generate matched local and selected-record clues with shared mistakes."""

    if not 0.0 <= float(shared_noise) <= 1.0:
        raise ValueError("shared_noise must lie in [0,1]")
    if examples_per_graph < 4 or examples_per_graph % 4:
        raise ValueError("examples_per_graph must be a multiple of four")
    rng = np.random.default_rng(int(seed))
    states = np.asarray(((0, 0), (1, 0), (0, 1), (1, 1)), dtype=np.int64)
    graph_ids: list[int] = []
    queries: list[int] = []
    structures: list[int] = []
    record_values: list[np.ndarray] = []
    local_values: list[float] = []
    targets: list[float] = []
    shared_scale = math.sqrt(float(shared_noise))
    private_scale = math.sqrt(max(1.0 - float(shared_noise), 0.0))
    distractor_scale = math.sqrt(1.0 + float(clue_noise) ** 2)
    for support in supports:
        ordering = np.arange(int(examples_per_graph), dtype=np.int64) % 4
        rng.shuffle(ordering)
        for state_index in ordering.tolist():
            query, structure = states[int(state_index)]
            target = float(rng.normal())
            common_error = float(rng.normal())
            local_error = float(rng.normal())
            distant_error = float(rng.normal())
            local_clue = target + float(clue_noise) * (
                shared_scale * common_error + private_scale * local_error
            )
            distant_clue = target + float(clue_noise) * (
                shared_scale * common_error + private_scale * distant_error
            )
            values = rng.normal(scale=distractor_scale, size=4).astype(np.float32)
            selected = int(2 * structure + query)
            values[selected] = distant_clue
            graph_ids.append(int(support.graph_id))
            queries.append(int(query))
            structures.append(int(structure))
            record_values.append(values)
            local_values.append(local_clue)
            targets.append(target)
    return RouterDataset(
        graph_ids=torch.tensor(graph_ids, dtype=torch.long),
        query=torch.tensor(queries, dtype=torch.long),
        structure=torch.tensor(structures, dtype=torch.long),
        record_values=torch.tensor(np.stack(record_values), dtype=DTYPE),
        local_value=torch.tensor(local_values, dtype=DTYPE),
        target=torch.tensor(targets, dtype=DTYPE),
    )


def _selected_values(dataset: RouterDataset) -> torch.Tensor:
    rows = torch.arange(len(dataset), dtype=torch.long)
    selected = 2 * dataset.structure + dataset.query
    return dataset.record_values[rows, selected]


def drop_training_clues(
    dataset: RouterDataset,
    *,
    probability: float,
    seed: int,
) -> RouterDataset:
    """Independently hide local and selected distant clues during training."""

    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    generator = torch.Generator().manual_seed(int(seed))
    local_mask = torch.rand(len(dataset), generator=generator) < float(probability)
    distant_mask = torch.rand(len(dataset), generator=generator) < float(probability)
    local = dataset.local_value.clone()
    local[local_mask] = 0.0
    records = dataset.record_values.clone()
    rows = torch.arange(len(dataset), dtype=torch.long)
    selected = 2 * dataset.structure + dataset.query
    records[rows[distant_mask], selected[distant_mask]] = 0.0
    return RouterDataset(
        graph_ids=dataset.graph_ids,
        query=dataset.query,
        structure=dataset.structure,
        record_values=records,
        local_value=local,
        target=dataset.target,
    )


def _linear_fit(
    train_columns: Sequence[torch.Tensor],
    train_target: torch.Tensor,
    test_columns: Sequence[torch.Tensor],
    test_target: torch.Tensor,
) -> dict[str, Any]:
    train_x = np.column_stack(
        (
            np.ones(len(train_target), dtype=np.float64),
            *(column.detach().cpu().numpy().astype(np.float64) for column in train_columns),
        )
    )
    test_x = np.column_stack(
        (
            np.ones(len(test_target), dtype=np.float64),
            *(column.detach().cpu().numpy().astype(np.float64) for column in test_columns),
        )
    )
    coefficients = np.linalg.lstsq(
        train_x,
        train_target.detach().cpu().numpy().astype(np.float64),
        rcond=None,
    )[0]
    prediction = test_x @ coefficients
    truth = test_target.detach().cpu().numpy().astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(prediction - truth))),
        "coefficients": coefficients.tolist(),
    }


def reference_fits(
    train: RouterDataset,
    test: RouterDataset,
) -> dict[str, Any]:
    """Fit simple references that know which distant record is selected."""

    train_distant = _selected_values(train)
    test_distant = _selected_values(test)
    local = _linear_fit(
        (train.local_value,),
        train.target,
        (test.local_value,),
        test.target,
    )
    distant = _linear_fit(
        (train_distant,),
        train.target,
        (test_distant,),
        test.target,
    )
    both = _linear_fit(
        (train.local_value, train_distant),
        train.target,
        (test.local_value, test_distant),
        test.target,
    )
    return {
        "local_only": local,
        "distant_only": distant,
        "two_clue": both,
        "available_two_clue_gain": float(local["mae"] - both["mae"]),
    }


def measure_route_failures(
    model: TinyMultiHeadGraphTransformer,
    clean: RouterDataset,
) -> dict[str, float]:
    local_failed = corrupt_local_copy(clean, probability=1.0, seed=0)
    distant_failed = corrupt_selected_distant_record(clean)
    with torch.no_grad():
        clean_mae = float(torch.mean(torch.abs(model(clean) - clean.target)))
        local_failure_mae = float(
            torch.mean(torch.abs(model(local_failed) - clean.target))
        )
        distant_failure_mae = float(
            torch.mean(torch.abs(model(distant_failed) - clean.target))
        )
    return {
        "clean_mae": clean_mae,
        "local_failure_mae": local_failure_mae,
        "distant_failure_mae": distant_failure_mae,
        "local_failure_damage": local_failure_mae - clean_mae,
        "distant_failure_damage": distant_failure_mae - clean_mae,
    }


def _mean(records: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = [key for key, value in records[0].items() if isinstance(value, (int, float))]
    return {
        key: float(np.mean([float(record[key]) for record in records])) for key in keys
    }


def _correlation(records: Sequence[Mapping[str, float]], left: str, right: str) -> float:
    x = np.asarray([float(record[left]) for record in records], dtype=np.float64)
    y = np.asarray([float(record[right]) for record in records], dtype=np.float64)
    if x.size < 2 or float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _serialise(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _serialise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(item) for item in value]
    return value


def _load_supports(config: Config) -> dict[str, list[MolecularSupport]]:
    specs = (
        ("train", config.train_graphs, config.data_seed + 1),
        ("val", config.val_graphs, config.data_seed + 2),
        ("test", config.test_graphs, config.data_seed + 3),
    )
    return {
        split: load_molecular_supports(
            config.data_root,
            split=split,
            count=count,
            seed=seed,
        )
        for split, count, seed in specs
    }


def _transformer_config(config: Config) -> TransformerConfig:
    return TransformerConfig(
        output_dir=config.output_dir,
        data_root=config.data_root,
        train_graphs=config.train_graphs,
        val_graphs=config.val_graphs,
        test_graphs=config.test_graphs,
        examples_per_graph=config.examples_per_graph,
        seeds=config.seeds,
        train_local_corruption=(0.0,),
        train_distant_corruption=0.0,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        heads=config.heads,
        readout="sum",
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        early_stop_patience=config.early_stop_patience,
        data_seed=config.data_seed,
    )


def run(config: Config) -> dict[str, Any]:
    config.validate()
    torch.set_num_threads(1)
    supports = _load_supports(config)
    model_config = _transformer_config(config)
    result: dict[str, Any] = {"config": asdict(config), "conditions": {}}
    flat: list[dict[str, float]] = []
    for overlap in config.shared_noise:
        datasets = {
            split: make_correlated_clue_dataset(
                supports[split],
                examples_per_graph=config.examples_per_graph,
                shared_noise=float(overlap),
                clue_noise=float(config.clue_noise),
                seed=config.data_seed + 100 + index,
            )
            for index, split in enumerate(("train", "val", "test"))
        }
        references = reference_fits(datasets["train"], datasets["test"])
        seed_records: list[dict[str, Any]] = []
        for seed in config.seeds:
            train_dataset = drop_training_clues(
                datasets["train"],
                probability=float(config.train_clue_dropout),
                seed=int(config.data_seed + 10_000 * float(overlap) + seed),
            )
            validation_dataset = drop_training_clues(
                datasets["val"],
                probability=float(config.train_clue_dropout),
                seed=int(config.data_seed + 20_000 * float(overlap) + seed),
            )
            model, training = train_model(
                model_config,
                train_dataset,
                validation_dataset,
                seed=int(seed),
            )
            failures = measure_route_failures(model, datasets["test"])
            internal = measure_internal(model, datasets["test"])
            measurement = {
                **failures,
                **internal,
                "model_gain_over_local": float(
                    references["local_only"]["mae"] - failures["clean_mae"]
                ),
                "available_two_clue_gain": float(
                    references["available_two_clue_gain"]
                ),
            }
            seed_records.append(
                {
                    "seed": int(seed),
                    "training": training,
                    "measurement": measurement,
                }
            )
            flat.append(
                {
                    "shared_noise": float(overlap),
                    "seed": float(seed),
                    **{
                        key: float(value)
                        for key, value in measurement.items()
                        if isinstance(value, (int, float))
                    },
                }
            )
        means = _mean(
            [
                {
                    key: value
                    for key, value in record["measurement"].items()
                    if key != "heads"
                }
                for record in seed_records
            ]
        )
        result["conditions"][str(overlap)] = {
            "shared_noise": float(overlap),
            "references": references,
            "seeds": seed_records,
            "mean": means,
        }
    result["correlations"] = {
        "interaction_mass_vs_distant_damage": _correlation(
            flat, "interaction_effect_mass", "distant_failure_damage"
        ),
        "interaction_mass_vs_local_damage": _correlation(
            flat, "interaction_effect_mass", "local_failure_damage"
        ),
        "interaction_mass_vs_available_gain": _correlation(
            flat, "interaction_effect_mass", "available_two_clue_gain"
        ),
        "shared_noise_vs_interaction_mass": _correlation(
            flat, "shared_noise", "interaction_effect_mass"
        ),
        "shared_noise_vs_available_gain": _correlation(
            flat, "shared_noise", "available_two_clue_gain"
        ),
    }
    payload = _serialise(result)
    fingerprint_record = {**payload["config"], "output_dir": None}
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_record, sort_keys=True).encode("utf-8")
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
        "--output-dir",
        type=Path,
        default=Path("outputs/molecular_correlated_clues"),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-graphs", type=int, default=256)
    parser.add_argument("--val-graphs", type=int, default=64)
    parser.add_argument("--test-graphs", type=int, default=64)
    parser.add_argument("--examples-per-graph", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--shared-noise", default="0,0.5,0.9,1")
    parser.add_argument("--clue-noise", type=float, default=0.5)
    parser.add_argument("--train-clue-dropout", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1_500)
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
        shared_noise=_parse_floats(args.shared_noise),
        clue_noise=float(args.clue_noise),
        train_clue_dropout=float(args.train_clue_dropout),
        hidden_dim=int(args.hidden_dim),
        layers=int(args.layers),
        heads=int(args.heads),
        max_steps=int(args.max_steps),
    )
    result = run(config)
    summary = {
        "conditions": {
            key: {
                "references": value["references"],
                "mean": value["mean"],
            }
            for key, value in result["conditions"].items()
        },
        "correlations": result["correlations"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
