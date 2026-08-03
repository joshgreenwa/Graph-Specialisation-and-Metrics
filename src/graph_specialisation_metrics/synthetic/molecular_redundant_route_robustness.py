"""Does redundant semantic--structural routing predict route-specific robustness?

This experiment reuses the ZINC-supported redundant record router.  The target
is available both at the anchor and through a distant semantic-key/structural-
bank lookup.  Models are evaluated after either the local copy or the selected
distant record is sign-flipped.  The same distant pathway should rescue the
first failure and amplify the second, providing a route-specific check that a
clean-data internal measurement has predictive value.

A second sweep introduces rare local sign flips during training and lets the
model learn its local/global residual mixture.  This tests whether the measured
distant route is acquired in response to reliability pressure rather than only
being imposed by the experimental architecture.
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

from .molecular_redundant_record_router import (
    Config as RouterConfig,
)
from .molecular_redundant_record_router import (
    LearnedMixRecordRouter,
    RouterDataset,
    evaluate,
    load_molecular_supports,
    make_router_dataset,
    train_learned_mix_model,
    train_model,
)


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
    train_local_corruption: tuple[float, ...] = (0.0, 0.01, 0.05, 0.20)
    train_distant_corruption: float = 0.05
    max_steps: int = 4_000
    learning_rate: float = 2.0e-2
    weight_decay: float = 1.0e-5
    data_seed: int = 80_203

    def validate(self) -> None:
        if min(self.train_graphs, self.val_graphs, self.test_graphs) < 1:
            raise ValueError("each split requires at least one graph")
        if self.examples_per_graph < 4 or self.max_steps < 1:
            raise ValueError("invalid data or training size")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be non-empty and unique")
        if any(not 0 < value < 1 for value in self.local_mixes):
            raise ValueError("local mixtures must lie in (0,1)")
        corruption = self.train_local_corruption
        if (
            not corruption
            or tuple(sorted(set(corruption))) != corruption
            or any(not 0 <= value < 0.5 for value in corruption)
        ):
            raise ValueError("training corruption must be unique, increasing, and in [0,.5)")
        if not 0 <= self.train_distant_corruption < 0.5:
            raise ValueError("distant training corruption must lie in [0,.5)")


def corrupt_local_copy(
    dataset: RouterDataset,
    *,
    probability: float,
    seed: int,
) -> RouterDataset:
    """Sign-flip a registered fraction of local target copies."""

    if not 0 <= float(probability) <= 1:
        raise ValueError("probability must lie in [0,1]")
    generator = torch.Generator().manual_seed(int(seed))
    mask = torch.rand(len(dataset), generator=generator) < float(probability)
    local = dataset.local_value.clone()
    local[mask] = -local[mask]
    return RouterDataset(
        graph_ids=dataset.graph_ids,
        query=dataset.query,
        structure=dataset.structure,
        record_values=dataset.record_values,
        local_value=local,
        target=dataset.target,
    )


def corrupt_selected_distant_record(
    dataset: RouterDataset,
    *,
    probability: float = 1.0,
    seed: int = 0,
) -> RouterDataset:
    """Flip selected records with the registered probability."""

    if not 0 <= float(probability) <= 1:
        raise ValueError("probability must lie in [0,1]")
    values = dataset.record_values.clone()
    selected = 2 * dataset.structure + dataset.query
    rows = torch.arange(len(dataset), dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    mask = torch.rand(len(dataset), generator=generator) < float(probability)
    values[rows[mask], selected[mask]] = -values[rows[mask], selected[mask]]
    return RouterDataset(
        graph_ids=dataset.graph_ids,
        query=dataset.query,
        structure=dataset.structure,
        record_values=values,
        local_value=dataset.local_value,
        target=dataset.target,
    )


def route_failure_metrics(model: Any, clean: RouterDataset) -> dict[str, float]:
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
        local_only_failure_mae = float(
            torch.mean(torch.abs(local_failed.local_value - clean.target))
        )
    return {
        "clean_mae": clean_mae,
        "local_failure_mae": local_failure_mae,
        "distant_failure_mae": distant_failure_mae,
        "local_only_failure_mae": local_only_failure_mae,
        "local_failure_rescue": local_only_failure_mae - local_failure_mae,
        "distant_failure_damage": distant_failure_mae - clean_mae,
    }


def train_gate_only(
    config: RouterConfig,
    pretrained: Any,
    train: RouterDataset,
    validation: RouterDataset,
    *,
    seed: int,
) -> tuple[LearnedMixRecordRouter, dict[str, float | int]]:
    """Freeze a clean-trained distant router and learn only route allocation."""

    model = LearnedMixRecordRouter(seed=int(seed))
    pretrained_state = pretrained.state_dict()
    with torch.no_grad():
        model.semantic_logits.copy_(pretrained_state["semantic_logits"])
        model.structural_logits.copy_(pretrained_state["structural_logits"])
        model.distance_bias.copy_(pretrained_state["distance_bias"])
    model.semantic_logits.requires_grad_(False)
    model.structural_logits.requires_grad_(False)
    model.distance_bias.requires_grad_(False)
    optimiser = torch.optim.Adam(
        (model.local_mix_logit,),
        lr=float(config.learning_rate),
        weight_decay=0.0,
    )
    best_loss = float("inf")
    best_logit = model.local_mix_logit.detach().clone()
    step = 0
    for step in range(1, min(int(config.max_steps), 2_000) + 1):
        loss = torch.mean((model(train) - train.target) ** 2)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            validation_loss = float(torch.mean((model(validation) - validation.target) ** 2))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_logit = model.local_mix_logit.detach().clone()
    with torch.no_grad():
        model.local_mix_logit.copy_(best_logit)
    model.eval()
    return model, {
        "steps": int(step),
        "validation_mse": best_loss,
        "effective_local_mix": model.effective_local_mix(),
    }


def _router_config(config: Config) -> RouterConfig:
    return RouterConfig(
        output_dir=config.output_dir,
        data_root=config.data_root,
        train_graphs=int(config.train_graphs),
        val_graphs=int(config.val_graphs),
        test_graphs=int(config.test_graphs),
        examples_per_graph=int(config.examples_per_graph),
        seeds=tuple(config.seeds),
        local_mixes=tuple(config.local_mixes),
        max_steps=int(config.max_steps),
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        data_seed=int(config.data_seed),
    )


def _load_datasets(config: Config) -> dict[str, RouterDataset]:
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
    return {
        split: make_router_dataset(
            supports[split],
            examples_per_graph=int(config.examples_per_graph),
            seed=int(config.data_seed) + 100 + index,
        )
        for index, (split, _, _) in enumerate(split_specs)
    }


def _record(model: Any, test: RouterDataset) -> dict[str, float]:
    internal = evaluate(model, test)
    failures = route_failure_metrics(model, test)
    return {**internal, **failures}


def _mean(records: Sequence[Mapping[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([float(record[key]) for record in records]))
        for key in records[0]
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


def run(config: Config) -> dict[str, Any]:
    config.validate()
    torch.set_num_threads(1)
    router_config = _router_config(config)
    datasets = _load_datasets(config)
    result: dict[str, Any] = {
        "config": asdict(config),
        "fixed_routes": {},
        "learned_clean_route": {},
        "route_acquisition_joint": {},
        "route_acquisition_gate_only": {},
    }

    fixed_records: list[dict[str, Any]] = []
    for local_mix in config.local_mixes:
        mixture_records: list[dict[str, Any]] = []
        for seed in config.seeds:
            model, training = train_model(
                router_config,
                datasets["train"],
                datasets["val"],
                local_mix=float(local_mix),
                seed=int(seed),
            )
            measurement = _record(model, datasets["test"])
            record = {"seed": int(seed), "training": training, "test": measurement}
            mixture_records.append(record)
            fixed_records.append(
                {"local_mix": float(local_mix), "seed": int(seed), **measurement}
            )
        result["fixed_routes"][str(local_mix)] = {
            "local_mix": float(local_mix),
            "global_mix": float(1 - local_mix),
            "seeds": mixture_records,
            "test_mean": _mean([record["test"] for record in mixture_records]),
        }

    learned_clean: list[dict[str, Any]] = []
    for seed in config.seeds:
        model, training = train_learned_mix_model(
            router_config,
            datasets["train"],
            datasets["val"],
            seed=int(seed),
        )
        learned_clean.append(
            {
                "seed": int(seed),
                "training": training,
                "test": _record(model, datasets["test"]),
            }
        )
    result["learned_clean_route"] = {
        "seeds": learned_clean,
        "test_mean": _mean([record["test"] for record in learned_clean]),
    }

    acquisition_flat: list[dict[str, Any]] = []
    gate_only_flat: list[dict[str, Any]] = []
    pretrained_by_seed: dict[int, Any] = {}
    for seed in config.seeds:
        pretrained_by_seed[int(seed)], _ = train_model(
            router_config,
            datasets["train"],
            datasets["val"],
            local_mix=0.5,
            seed=int(seed),
        )
    for probability in config.train_local_corruption:
        probability_records: list[dict[str, Any]] = []
        gate_only_records: list[dict[str, Any]] = []
        for seed in config.seeds:
            train = corrupt_local_copy(
                datasets["train"],
                probability=float(probability),
                seed=int(config.data_seed + 10_000 * probability + seed),
            )
            validation = corrupt_local_copy(
                datasets["val"],
                probability=float(probability),
                seed=int(config.data_seed + 20_000 * probability + seed),
            )
            train = corrupt_selected_distant_record(
                train,
                probability=float(config.train_distant_corruption),
                seed=int(config.data_seed + 30_000 * probability + seed),
            )
            validation = corrupt_selected_distant_record(
                validation,
                probability=float(config.train_distant_corruption),
                seed=int(config.data_seed + 40_000 * probability + seed),
            )
            model, training = train_learned_mix_model(
                router_config,
                train,
                validation,
                seed=int(seed),
            )
            measurement = _record(model, datasets["test"])
            probability_records.append(
                {"seed": int(seed), "training": training, "test": measurement}
            )
            acquisition_flat.append(
                {
                    "train_local_corruption": float(probability),
                    "seed": int(seed),
                    **measurement,
                }
            )
            gate_model, gate_training = train_gate_only(
                router_config,
                pretrained_by_seed[int(seed)],
                train,
                validation,
                seed=int(seed),
            )
            gate_measurement = _record(gate_model, datasets["test"])
            gate_only_records.append(
                {
                    "seed": int(seed),
                    "training": gate_training,
                    "test": gate_measurement,
                }
            )
            gate_only_flat.append(
                {
                    "train_local_corruption": float(probability),
                    "seed": int(seed),
                    **gate_measurement,
                }
            )
        result["route_acquisition_joint"][str(probability)] = {
            "train_local_corruption": float(probability),
            "train_distant_corruption": float(config.train_distant_corruption),
            "seeds": probability_records,
            "test_mean": _mean([record["test"] for record in probability_records]),
        }
        result["route_acquisition_gate_only"][str(probability)] = {
            "train_local_corruption": float(probability),
            "train_distant_corruption": float(config.train_distant_corruption),
            "seeds": gate_only_records,
            "test_mean": _mean([record["test"] for record in gate_only_records]),
        }

    result["fixed_route_correlations"] = {
        "interaction_mass_vs_local_rescue": _correlation(
            fixed_records, "interaction_effect_mass", "local_failure_rescue"
        ),
        "interaction_mass_vs_distant_damage": _correlation(
            fixed_records, "interaction_effect_mass", "distant_failure_damage"
        ),
        "interaction_distance_range": float(
            max(record["interaction_expected_distance"] for record in fixed_records)
            - min(record["interaction_expected_distance"] for record in fixed_records)
        ),
    }
    result["acquisition_correlations"] = {
        "joint": {
            "train_corruption_vs_global_mix": _correlation(
                acquisition_flat, "train_local_corruption", "effective_global_mix"
            ),
            "clean_interaction_mass_vs_local_rescue": _correlation(
                acquisition_flat, "interaction_effect_mass", "local_failure_rescue"
            ),
            "clean_interaction_mass_vs_distant_damage": _correlation(
                acquisition_flat, "interaction_effect_mass", "distant_failure_damage"
            ),
        },
        "gate_only": {
            "train_corruption_vs_global_mix": _correlation(
                gate_only_flat, "train_local_corruption", "effective_global_mix"
            ),
            "clean_interaction_mass_vs_local_rescue": _correlation(
                gate_only_flat, "interaction_effect_mass", "local_failure_rescue"
            ),
            "clean_interaction_mass_vs_distant_damage": _correlation(
                gate_only_flat, "interaction_effect_mass", "distant_failure_damage"
            ),
        },
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
        default=Path("outputs/molecular_redundant_route_robustness"),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-graphs", type=int, default=64)
    parser.add_argument("--val-graphs", type=int, default=16)
    parser.add_argument("--test-graphs", type=int, default=16)
    parser.add_argument("--examples-per-graph", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--local-mixes", default="0.25,0.5,0.75")
    parser.add_argument("--train-local-corruption", default="0,0.01,0.05,0.2")
    parser.add_argument("--train-distant-corruption", type=float, default=0.05)
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
        train_local_corruption=_parse_floats(args.train_local_corruption),
        train_distant_corruption=float(args.train_distant_corruption),
        max_steps=int(args.max_steps),
    )
    result = run(config)
    summary = {
        "fixed_routes": {
            key: value["test_mean"] for key, value in result["fixed_routes"].items()
        },
        "learned_clean_route": result["learned_clean_route"]["test_mean"],
        "route_acquisition_joint": {
            key: value["test_mean"]
            for key, value in result["route_acquisition_joint"].items()
        },
        "route_acquisition_gate_only": {
            key: value["test_mean"]
            for key, value in result["route_acquisition_gate_only"].items()
        },
        "correlations": {
            "fixed": result["fixed_route_correlations"],
            "acquisition": result["acquisition_correlations"],
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
