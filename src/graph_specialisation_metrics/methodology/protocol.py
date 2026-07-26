"""Versioned configuration for the repository's canonical methodology.

This module is deliberately free of GRIT, PyG, and plotting imports.  A protocol record can be
created and audited on a laptop, then replayed unchanged in Colab.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_VERSION = "donor-swap-specialisation-carriage-v3"
CHANNELS = ("semantic", "structural")
PHASES = ("scores", "causal", "carriage", "figures")
BOOTSTRAP_REPLICATES = 2_000


def stable_hash(value: Any, *, length: int = 24) -> str:
    """Return a deterministic digest for JSON-compatible protocol data."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[: int(length)]


@dataclass(frozen=True)
class RunSizes:
    """Sampling sizes for one trained model seed."""

    discovery_graphs: int = 48
    causal_graphs: int = 24
    clean_ablation_graphs: int = 64
    semantic_donor_graphs: int = 2_000
    sources_per_graph: int = 6
    donors_per_source: int = 8
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES

    def validate(self) -> None:
        for item in dataclasses.fields(self):
            if int(getattr(self, item.name)) < 1:
                raise ValueError(f"{item.name} must be positive")
        if int(self.bootstrap_replicates) != BOOTSTRAP_REPLICATES:
            raise ValueError(
                f"the canonical protocol fixes bootstrap_replicates={BOOTSTRAP_REPLICATES}"
            )

    @classmethod
    def smoke(cls) -> "RunSizes":
        """Small end-to-end engineering check; never a paper analysis."""

        return cls(
            discovery_graphs=3,
            causal_graphs=2,
            clean_ablation_graphs=2,
            semantic_donor_graphs=8,
            sources_per_graph=2,
            donors_per_source=2,
        )


@dataclass(frozen=True)
class NumericalPolicy:
    duplicate_tolerance: float = 1e-7
    reconstruction_tolerance: float = 1e-6
    batch_invariance_tolerance: float = 5e-5
    attention_tolerance: float = 1e-4
    no_op_tolerance: float = 1e-6
    score_floor: float = 1e-12
    selectivity_epsilon: float = 1e-12
    effect_floor: float = 1e-8
    integrated_atol: float = 1e-5
    integrated_rtol: float = 1e-4
    integrated_max_intervals: int = 64
    integrated_unconverged_fraction: float = 0.01

    def validate(self) -> None:
        for item in dataclasses.fields(self):
            value = float(getattr(self, item.name))
            if not (value >= 0.0):
                raise ValueError(f"{item.name} must be non-negative")
        if self.duplicate_tolerance <= 0 or self.reconstruction_tolerance <= 0:
            raise ValueError("structural duplicate and reconstruction tolerances must be positive")
        if self.score_floor <= 0 or self.effect_floor <= 0:
            raise ValueError("score/effect floors must be positive")
        if int(self.integrated_max_intervals) < 1:
            raise ValueError("integrated_max_intervals must be positive")


@dataclass(frozen=True)
class BootstrapPolicy:
    replicates: int = BOOTSTRAP_REPLICATES
    rng_seed: int = 17_071
    confidence: float = 0.95
    trim_fraction: float = 0.20
    minimum_graphs: int = 10
    minimum_pairs: int = 50
    resample_seed: bool = True
    resample_graph: bool = True
    resample_source: bool = True
    resample_donor: bool = True

    def validate(self) -> None:
        if int(self.replicates) != BOOTSTRAP_REPLICATES:
            raise ValueError(f"production bootstrap requires exactly {BOOTSTRAP_REPLICATES} draws")
        if float(self.confidence) != 0.95:
            raise ValueError("production intervals are fixed at 95%")
        if float(self.trim_fraction) != 0.20:
            raise ValueError("the registered graph estimator is a 20% trimmed mean")
        if int(self.minimum_graphs) != 10 or int(self.minimum_pairs) != 50:
            raise ValueError("reporting floors are fixed at 10 graphs and 50 eligible pairs")


@dataclass(frozen=True)
class FamilyPolicy:
    activity_floor: float = 0.20
    tail_fraction: float = 0.20
    central_fraction: float = 0.20
    equivalence_half_width: float = 0.10

    def validate(self) -> None:
        if self.activity_floor < 0:
            raise ValueError("activity_floor must be non-negative")
        for name in ("tail_fraction", "central_fraction"):
            value = float(getattr(self, name))
            if not 0 < value <= 0.5:
                raise ValueError(f"{name} must lie in (0, .5]")


@dataclass(frozen=True)
class ExecutionPolicy:
    """Performance controls that must leave cached scientific quantities unchanged."""

    graphs_per_batch: int = 4
    oom_backoff: bool = True

    def validate(self) -> None:
        if int(self.graphs_per_batch) < 1:
            raise ValueError("graphs_per_batch must be positive")


@dataclass(frozen=True)
class MethodologyConfig:
    """Complete public configuration for canonical multi-backend analyses."""

    output_dir: str = (
        "/content/drive/MyDrive/graph_specialisation_metrics/canonical_methodology"
    )
    tasks: tuple[str, ...] = ("zinc",)
    train_seeds: tuple[int, ...] = (42,)
    task_train_seeds: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    phases: tuple[str, ...] = PHASES
    sizes: RunSizes = field(default_factory=RunSizes)
    numerical: NumericalPolicy = field(default_factory=NumericalPolicy)
    bootstrap: BootstrapPolicy = field(default_factory=BootstrapPolicy)
    families: FamilyPolicy = field(default_factory=FamilyPolicy)
    execution: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    analysis_seed: int = 31_415
    accelerator: str = "cuda:0"
    num_threads: int = 4
    checkpoints: Mapping[str, str] = field(default_factory=dict)
    task_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    figure_overrides: Mapping[str, Any] = field(default_factory=dict)
    skip_install: bool = False
    resume: bool = True
    force: bool = False
    # Numerical/estimability audits report and continue by default; True restores fail-closed runs.
    strict_audits: bool = False
    # Functional carriage is always computed in the carriage phase. Some task-facing analyses
    # deliberately omit Beneficial carriage when it is outside the requested scientific scope.
    compute_beneficial_carriage: bool = True

    def validate(self) -> None:
        self.sizes.validate()
        self.numerical.validate()
        self.bootstrap.validate()
        self.families.validate()
        self.execution.validate()
        if not self.tasks:
            raise ValueError("at least one task is required")
        if not self.train_seeds:
            raise ValueError("at least one training seed is required")
        unknown_seed_tasks = sorted(set(self.task_train_seeds) - set(self.tasks))
        if unknown_seed_tasks:
            raise ValueError(
                f"task_train_seeds contains tasks not selected for this run: "
                f"{unknown_seed_tasks}"
            )
        empty_seed_tasks = sorted(
            task for task, seeds in self.task_train_seeds.items() if not tuple(seeds)
        )
        if empty_seed_tasks:
            raise ValueError(f"task_train_seeds entries cannot be empty: {empty_seed_tasks}")
        unknown = sorted(set(self.phases) - set(PHASES))
        if unknown:
            raise ValueError(f"unknown phases {unknown}; expected a subset of {PHASES}")
        if int(self.num_threads) < 1:
            raise ValueError("num_threads must be positive")
        if int(self.bootstrap.replicates) != int(self.sizes.bootstrap_replicates):
            raise ValueError("RunSizes and BootstrapPolicy disagree on bootstrap replicates")

    @property
    def root(self) -> Path:
        return Path(self.output_dir)

    def seeds_for(self, task: str) -> tuple[int, ...]:
        return tuple(
            int(value) for value in self.task_train_seeds.get(task, self.train_seeds)
        )

    @property
    def scientific_record(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "channels": list(CHANNELS),
            "tasks": list(self.tasks),
            "train_seeds": list(self.train_seeds),
            "task_train_seeds": {
                str(task): [int(seed) for seed in seeds]
                for task, seeds in self.task_train_seeds.items()
            },
            "sizes": dataclasses.asdict(self.sizes),
            "numerical": dataclasses.asdict(self.numerical),
            "bootstrap": dataclasses.asdict(self.bootstrap),
            "families": dataclasses.asdict(self.families),
            "analysis_seed": int(self.analysis_seed),
            "task_overrides": {
                str(key): dict(value) for key, value in self.task_overrides.items()
            },
            "donor_law": {
                "semantic": (
                    "different payload; global minimum absolute degree gap; "
                    "uniform eligible graph then uniform eligible node; iid replacement"
                ),
                "structural": (
                    "different footprint; minimum absolute degree gap; "
                    "uniform eligible node within base graph; iid replacement"
                ),
            },
            "raw_score_aggregation": "event -> source -> graph",
            "functional_estimand": "F_sens",
            "beneficial_carriage": bool(self.compute_beneficial_carriage),
            "beneficial_sign": (
                "positive-is-beneficial"
                if self.compute_beneficial_carriage
                else "not_computed"
            ),
            "causal_mismatch_control": (
                "same graph/channel/degree tier; distinct donor; prefer same source; "
                "minimum absolute intervention-dose gap"
            ),
        }

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.scientific_record)

    def record(self) -> dict[str, Any]:
        return {
            **self.scientific_record,
            "fingerprint": self.fingerprint,
            "phases": list(self.phases),
            "output_dir": self.output_dir,
            "accelerator": self.accelerator,
            "num_threads": self.num_threads,
            # Execution policy, not a scientific boundary: excluded from the cache fingerprint.
            "execution": dataclasses.asdict(self.execution),
            "strict_audits": bool(self.strict_audits),
            "checkpoints": dict(self.checkpoints),
            "figure_overrides": dict(self.figure_overrides),
        }


@dataclass(frozen=True)
class SplitManifest:
    """Disjoint stage base IDs plus a reusable semantic donor pool."""

    discovery: tuple[int, ...]
    causal: tuple[int, ...]
    clean_ablation: tuple[int, ...]
    semantic_donor_pool: tuple[int, ...]
    same_index_space: bool
    seed: int

    def validate(self) -> None:
        bases = [set(self.discovery), set(self.causal), set(self.clean_ablation)]
        if bases[0] & bases[1] or bases[0] & bases[2] or bases[1] & bases[2]:
            raise ValueError("discovery, causal, and clean-ablation graph IDs must be disjoint")
        if self.same_index_space and set(self.semantic_donor_pool) & set.union(*bases):
            raise ValueError("semantic donor graph IDs must be disjoint from every base split")

    @property
    def fingerprint(self) -> str:
        return stable_hash(dataclasses.asdict(self))


def deterministic_splits(
    eval_length: int,
    donor_length: int,
    sizes: RunSizes,
    seed: int,
    *,
    same_index_space: bool,
) -> SplitManifest:
    """Construct the exact split-disjointness contract from stable integer graph IDs."""

    import numpy as np

    sizes.validate()
    base_needed = (
        int(sizes.discovery_graphs)
        + int(sizes.causal_graphs)
        + int(sizes.clean_ablation_graphs)
    )
    donor_needed = min(int(sizes.semantic_donor_graphs), int(donor_length))
    total_needed = base_needed + (donor_needed if same_index_space else 0)
    if total_needed > int(eval_length):
        raise ValueError(
            f"need {total_needed} disjoint IDs in the shared index space, "
            f"but eval split contains {eval_length}"
        )
    if not same_index_space and donor_needed > int(donor_length):
        raise ValueError("semantic donor pool exceeds donor split")

    rng = np.random.default_rng(int(seed))
    base_order = rng.permutation(int(eval_length))
    cursor = 0

    def take(count: int) -> tuple[int, ...]:
        nonlocal cursor
        values = tuple(sorted(int(v) for v in base_order[cursor : cursor + int(count)]))
        cursor += int(count)
        return values

    discovery = take(sizes.discovery_graphs)
    causal = take(sizes.causal_graphs)
    ablation = take(sizes.clean_ablation_graphs)
    if same_index_space:
        donors = take(donor_needed)
    else:
        donors = tuple(
            sorted(
                int(v)
                for v in np.random.default_rng(int(seed) + 1).choice(
                    int(donor_length), size=donor_needed, replace=False
                )
            )
        )
    manifest = SplitManifest(
        discovery=discovery,
        causal=causal,
        clean_ablation=ablation,
        semantic_donor_pool=donors,
        same_index_space=bool(same_index_space),
        seed=int(seed),
    )
    manifest.validate()
    return manifest


def parse_csv(value: str | Sequence[Any]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())
