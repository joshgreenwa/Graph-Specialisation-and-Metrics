"""Configuration and deterministic split contracts for scoring refinement."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_VERSION = "scoring-refinement-v1"
METHODS = (
    "M1_DD",
    "M1_DT",
    "M1_TD",
    "M1_TT",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
)
PHASES = ("scores", "validation", "figures", "all")
DEFAULT_OUTPUT_DIR = (
    "/content/drive/MyDrive/graph_specialisation_metrics/"
    "scoring_metric_refinement_dense"
)


def stable_hash(payload: Mapping[str, Any]) -> str:
    """Return a stable short digest for a JSON-compatible mapping."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class RunSizes:
    """Sampling sizes. Defaults are the predeclared full experiment."""

    score_graphs: int = 48
    score_sources: int = 6
    events_per_source: int = 6
    topology_events: int = 3
    topology_pool: int = 4_000
    semantic_pool: int = 2_000
    causal_graphs: int = 24
    causal_sources: int = 1
    causal_events_per_variant: int = 1
    ablation_graphs: int = 64
    bootstrap_samples: int = 1_000

    @classmethod
    def fast(cls) -> "RunSizes":
        return cls(
            score_graphs=6,
            score_sources=2,
            events_per_source=2,
            topology_events=1,
            topology_pool=128,
            semantic_pool=128,
            causal_graphs=4,
            causal_sources=1,
            causal_events_per_variant=1,
            ablation_graphs=8,
            bootstrap_samples=40,
        )

    def validate(self) -> None:
        for item in dataclasses.fields(self):
            if int(getattr(self, item.name)) < 1:
                raise ValueError(f"{item.name} must be positive")


@dataclass(frozen=True)
class RefinementConfig:
    """Complete task-independent experiment configuration."""

    output_dir: str = DEFAULT_OUTPUT_DIR
    tasks: tuple[str, ...] = ("zinc", "qm9_gap_dense")
    phase: str = "all"
    methods: tuple[str, ...] = METHODS
    sizes: RunSizes = field(default_factory=RunSizes)
    analysis_seed: int = 1771
    device: str = "cuda:0"
    train_seed: int = 42
    num_threads: int = 4
    checkpoint_every: int = 1
    top_k: tuple[int, ...] = (3, 5, 10)
    reference_method: str = "mean"
    pe_partner_match: str = "degree"
    topology_allow_relaxed: bool = True
    skip_install: bool = False
    pyg_version: str = "2.2.0"
    force_fresh_grit: bool = False
    force: bool = False
    checkpoints: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        self.sizes.validate()
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {self.phase!r}")
        unknown = sorted(set(self.methods) - set(METHODS))
        if unknown:
            raise ValueError(f"unknown scoring methods: {unknown}")
        if not self.tasks:
            raise ValueError("at least one task is required")
        if self.pe_partner_match not in {"degree", "any"}:
            raise ValueError("pe_partner_match must be 'degree' or 'any'")
        if self.reference_method != "mean":
            raise ValueError("only the predeclared mean reference is supported")
        if any(int(value) < 1 for value in self.top_k):
            raise ValueError("top_k values must be positive")
        if int(self.checkpoint_every) < 1:
            raise ValueError("checkpoint_every must be positive")

    @property
    def root(self) -> Path:
        return Path(self.output_dir)

    @property
    def fingerprint_payload(self) -> dict[str, Any]:
        """Scientific settings only; runtime paths/devices do not invalidate scores."""

        return {
            "protocol_version": PROTOCOL_VERSION,
            "sizes": dataclasses.asdict(self.sizes),
            "analysis_seed": self.analysis_seed,
            "top_k": list(self.top_k),
            "reference_method": self.reference_method,
            "pe_partner_match": self.pe_partner_match,
            "topology_allow_relaxed": self.topology_allow_relaxed,
        }

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.fingerprint_payload)

    def protocol_record(self) -> dict[str, Any]:
        return {
            **self.fingerprint_payload,
            "fingerprint": self.fingerprint,
            "tasks": list(self.tasks),
            "methods": list(self.methods),
            "output_dir": self.output_dir,
            "phase": self.phase,
            "device": self.device,
            "checkpoints": dict(self.checkpoints),
        }

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        *,
        phase: str = "all",
        output_dir: str | None = None,
        checkpoints: Mapping[str, str] | None = None,
        force: bool = False,
        skip_install: bool = False,
        force_fresh_grit: bool = False,
    ) -> "RefinementConfig":
        """Recreate scientific settings from ``protocol.json`` for a resumed run."""

        sizes = RunSizes(**dict(record["sizes"]))
        return cls(
            output_dir=str(output_dir or record.get("output_dir", DEFAULT_OUTPUT_DIR)),
            tasks=tuple(record.get("tasks", ("zinc", "qm9_gap_dense"))),
            phase=phase,
            methods=tuple(record.get("methods", METHODS)),
            sizes=sizes,
            analysis_seed=int(record.get("analysis_seed", 1771)),
            top_k=tuple(int(value) for value in record.get("top_k", (3, 5, 10))),
            reference_method=str(record.get("reference_method", "mean")),
            pe_partner_match=str(record.get("pe_partner_match", "degree")),
            topology_allow_relaxed=bool(record.get("topology_allow_relaxed", True)),
            checkpoints=dict(checkpoints or record.get("checkpoints", {})),
            force=force,
            skip_install=skip_install,
            force_fresh_grit=force_fresh_grit,
        )


def deterministic_splits(
    length: int,
    sizes: RunSizes,
    seed: int,
) -> dict[str, list[int]]:
    """Disjoint deterministic score/causal/ablation graph indices."""

    import numpy as np

    needed = sizes.score_graphs + sizes.causal_graphs + sizes.ablation_graphs
    if needed > int(length):
        raise ValueError(
            f"requested {needed} disjoint graphs from an evaluation split of length {length}"
        )
    order = np.random.default_rng(int(seed)).permutation(int(length))[:needed]
    score_end = sizes.score_graphs
    causal_end = score_end + sizes.causal_graphs
    return {
        "score": sorted(int(value) for value in order[:score_end]),
        "causal": sorted(int(value) for value in order[score_end:causal_end]),
        "ablation": sorted(int(value) for value in order[causal_end:]),
    }


def parse_methods(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        if value.strip().lower() == "all":
            return METHODS
        values = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    else:
        values = tuple(str(part).strip().upper() for part in value if str(part).strip())
    unknown = sorted(set(values) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods {unknown}; allowed={list(METHODS)}")
    return tuple(dict.fromkeys(values))
