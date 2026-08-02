"""Bamberger-versus-finite reach analysis for trained ZINC GRIT models.

This extension keeps two comparisons separate:

* The literal Bamberger et al. graph-level proxy is the mean node-level
  pre-pooling Jacobian range.  For categorical ZINC atoms, the input is a
  differentiable one-hot vector followed by the checkpoint's learned embedding.
* Functional carriage is evaluated at final pre-pooling node states for exact
  semantic or structural donor events.
* A semantic interpolation sweep varies the donor-swap fraction while holding
  donor events fixed, testing whether the finite profile departs progressively
  from the Bamberger Jacobian profile.
* A cache-only estimand decomposition separates shell-summed radial allocation
  from shell-size-adjusted per-carrier sensitivity for both methods.

The Bamberger proxy has no canonical structural-donor counterpart.  It is
therefore shown only for semantic usage; structural usage reports Functional
carriage without inventing a prior-work quantity.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .carriage import env
from .carriage.core import integrated_loss_carriage
from .carriage.tasks import get_task as get_grit_task
from .methodology.distance import shortest_path_distances
from .methodology.events import build_channel_events
from .methodology.protocol import (
    BootstrapPolicy,
    ExecutionPolicy,
    MethodologyConfig,
    RunSizes,
    stable_hash,
)
from .methodology.runner import prepare_task
from .methodology.sampling import sample_sources
from .reach_redundancy import (
    SemanticCoalition,
    build_shell_permutation,
    build_shell_replacement,
    combine_semantic_coalitions,
    survival_components,
)


ANALYSIS_VERSION = "zinc-bamberger-functional-reach-v8"
OUTPUT_CARRIAGE_VERSION = "signed-output-path-carriage-v2-soft-audit"
BENEFICIAL_CARRIAGE_VERSION = "positive-beneficial-path-carriage-v1"
SURVIVAL_VERSION = "shell-coalition-survival-v2-reused-singletons"
OUTPUT_PATH_ATOL = 1.0e-6
OUTPUT_PATH_RTOL = 1.0e-5
OUTPUT_PATH_MAX_INTERVALS = 128
TASKS = (
    "zinc_1hop",
    "zinc_1hop_localrrwp",
    "zinc_2hop",
    "zinc_1hop_vnode",
    "zinc",
)
QM9_TASKS = ("qm9_gap_1hop", "qm9_gap_1hop_vnode", "qm9_gap_dense")
PEPTIDES_STRUCT_TASKS = ("peptides_struct_1hop", "peptides_struct")
DEFAULT_INTERPOLATION_DOSES = (0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00)
TASK_LABELS = {
    "zinc_1hop": "1-hop GRIT",
    "zinc_1hop_localrrwp": "1-hop GRIT + local RRWP",
    "zinc_2hop": "2-hop GRIT",
    "zinc_1hop_vnode": "1-hop GRIT + VN",
    "zinc": "Dense GRIT",
    "qm9_gap_1hop": "1-hop GRIT",
    "qm9_gap_1hop_vnode": "1-hop GRIT + VN",
    "qm9_gap_dense": "Dense GRIT",
    "peptides_struct_1hop": "1-hop GRIT",
    "peptides_struct": "Dense GRIT",
}
CHANNELS = ("semantic", "structural")
DONOR_METHODS = ("functional_carriage",)
DONOR_PROFILE_METHODS = DONOR_METHODS
METHOD_LABELS = {
    "bamberger": "Bamberger (coordinatewise Jacobian)",
    "functional_carriage": "Functional carriage (task-projected)",
}
METHOD_COLOURS = {
    "bamberger": "#202020",
    "functional_carriage": "#D55E00",
}
METHOD_MARKERS = {
    "bamberger": "^",
    "functional_carriage": "s",
}
METHOD_LINESTYLES = {
    "bamberger": "-",
    "functional_carriage": ":",
}
USAGE_ESTIMANDS = (
    "radial_allocation",
    "normalised_per_carrier_sensitivity",
)
USAGE_ESTIMAND_LABELS = {
    "radial_allocation": "Normalised radial allocation\n(shell-summed)",
    "normalised_per_carrier_sensitivity": (
        "Normalised per-carrier sensitivity\n(shell-size adjusted)"
    ),
    "shell_opportunity": "Uniform-carrier opportunity",
}
MODEL_COLOURS = {
    "zinc_1hop": "#0072B2",
    "zinc_1hop_localrrwp": "#E69F00",
    "zinc_2hop": "#009E73",
    "zinc_1hop_vnode": "#CC79A7",
    "zinc": "#D55E00",
    "qm9_gap_1hop": "#0072B2",
    "qm9_gap_1hop_vnode": "#CC79A7",
    "qm9_gap_dense": "#D55E00",
    "peptides_struct_1hop": "#0072B2",
    "peptides_struct": "#D55E00",
}
MODEL_MARKERS = {
    "zinc_1hop": "o",
    "zinc_1hop_localrrwp": "P",
    "zinc_2hop": "^",
    "zinc_1hop_vnode": "D",
    "zinc": "s",
    "qm9_gap_1hop": "o",
    "qm9_gap_1hop_vnode": "D",
    "qm9_gap_dense": "s",
    "peptides_struct_1hop": "o",
    "peptides_struct": "s",
}
MODEL_LINESTYLES = {
    "zinc_1hop": "-",
    "zinc_1hop_localrrwp": (0, (3, 1, 1, 1)),
    "zinc_2hop": "--",
    "zinc_1hop_vnode": "-.",
    "zinc": ":",
    "qm9_gap_1hop": "-",
    "qm9_gap_1hop_vnode": "-.",
    "qm9_gap_dense": ":",
    "peptides_struct_1hop": "-",
    "peptides_struct": ":",
}


@dataclass(frozen=True)
class ReachProfile:
    """Dataset-specific controls for the shared reach experiment."""

    name: str
    analysis_version: str
    tasks: tuple[str, ...]
    reference_task: str
    atom_vocab_size: int | None
    node_feature_fields: int
    event_stage: str
    figure_prefix: str
    default_output_dir: str


ZINC_PROFILE = ReachProfile(
    name="ZINC",
    analysis_version=ANALYSIS_VERSION,
    tasks=TASKS,
    reference_task="zinc",
    atom_vocab_size=21,
    node_feature_fields=1,
    event_stage="zinc_reach",
    figure_prefix="zinc",
    default_output_dir=(
        "/content/drive/MyDrive/graph_specialisation_metrics/"
        "zinc_bamberger_functional_reach_v8"
    ),
)
QM9_PROFILE = ReachProfile(
    name="QM9",
    analysis_version="qm9-bamberger-functional-reach-v3",
    tasks=QM9_TASKS,
    reference_task="qm9_gap_dense",
    atom_vocab_size=10,
    node_feature_fields=1,
    event_stage="qm9_reach",
    figure_prefix="qm9",
    default_output_dir=(
        "/content/drive/MyDrive/graph_specialisation_metrics/"
        "qm9_bamberger_functional_reach_v3"
    ),
)
PEPTIDES_STRUCT_PROFILE = ReachProfile(
    name="Peptides-struct",
    analysis_version="peptides-struct-bamberger-functional-reach-v1",
    tasks=PEPTIDES_STRUCT_TASKS,
    reference_task="peptides_struct",
    atom_vocab_size=None,
    node_feature_fields=9,
    event_stage="peptides_struct_reach",
    figure_prefix="peptides_struct",
    default_output_dir=(
        "/content/drive/MyDrive/graph_specialisation_metrics/"
        "peptides_struct_bamberger_functional_reach_v1"
    ),
)


@dataclass(frozen=True)
class ZincReachConfig:
    """Scientific and runtime controls for the standalone analysis."""

    profile: ReachProfile = ZINC_PROFILE
    tasks: tuple[str, ...] = TASKS
    channels: tuple[str, ...] = CHANNELS
    seed: int = 0
    graphs: int = 64
    sources_per_graph: int = 6
    donors_per_source: int = 4
    semantic_donor_graphs: int = 256
    bamberger_output_nodes: int = 6
    bamberger_output_channels: int = 8
    interpolation_doses: tuple[float, ...] = DEFAULT_INTERPOLATION_DOSES
    interpolation_batch_size: int = 64
    survival_carriers_per_graph: int = 1
    survival_draws: int = 1
    survival_tail_radii: tuple[int, ...] = (2, 3, 4, 5)
    survival_replacement_candidates: int = 32
    survival_exact_limit: int = 12
    survival_random_attempts: int = 512
    survival_replica_batch_size: int = 2_048
    beneficial_donors_per_source: int = 1
    beneficial_atol: float = 1.0e-5
    beneficial_rtol: float = 1.0e-4
    beneficial_max_intervals: int = 64
    effect_floor: float = 1.0e-12
    bootstrap_replicates: int = 2_000
    analysis_seed: int = 91_021
    accelerator: str = "cuda:0"
    num_threads: int = 4
    compute_output_carriage: bool = True
    compute_beneficial: bool = True
    compute_survival: bool = True
    compute_scale_analysis: bool = True

    def validate(self) -> None:
        if not self.tasks or any(task not in self.profile.tasks for task in self.tasks):
            raise ValueError(f"tasks must be drawn from {self.profile.tasks}")
        if not self.channels or any(channel not in CHANNELS for channel in self.channels):
            raise ValueError(f"channels must be drawn from {CHANNELS}")
        for name in (
            "graphs",
            "sources_per_graph",
            "donors_per_source",
            "semantic_donor_graphs",
            "bamberger_output_nodes",
            "bamberger_output_channels",
            "interpolation_batch_size",
            "survival_carriers_per_graph",
            "survival_draws",
            "survival_replacement_candidates",
            "survival_exact_limit",
            "survival_random_attempts",
            "survival_replica_batch_size",
            "beneficial_donors_per_source",
            "beneficial_max_intervals",
            "bootstrap_replicates",
            "num_threads",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if float(self.effect_floor) <= 0:
            raise ValueError("effect_floor must be positive")
        if float(self.beneficial_atol) < 0 or float(self.beneficial_rtol) < 0:
            raise ValueError("beneficial integration tolerances must be non-negative")
        radii = tuple(int(value) for value in self.survival_tail_radii)
        if not radii or tuple(sorted(set(radii))) != radii or radii[0] < 1:
            raise ValueError("survival_tail_radii must be unique increasing positive integers")
        doses = tuple(float(value) for value in self.interpolation_doses)
        if (
            not doses
            or any(not 0 < value <= 1 for value in doses)
            or tuple(sorted(set(doses))) != doses
            or not np.isclose(doses[-1], 1.0)
        ):
            raise ValueError(
                "interpolation_doses must be unique, increasing, in (0, 1], "
                "and end at 1"
            )

    @property
    def scientific_record(self) -> dict[str, Any]:
        return {
            "analysis_version": self.profile.analysis_version,
            "dataset": self.profile.name,
            "tasks": list(self.tasks),
            "channels": list(self.channels),
            "seed": int(self.seed),
            "graphs": int(self.graphs),
            "sources_per_graph": int(self.sources_per_graph),
            "donors_per_source": int(self.donors_per_source),
            "semantic_donor_graphs": int(self.semantic_donor_graphs),
            "bamberger_output_nodes": int(self.bamberger_output_nodes),
            "bamberger_output_channels": int(self.bamberger_output_channels),
            "interpolation_doses": [
                float(value) for value in self.interpolation_doses
            ],
            "survival_carriers_per_graph": int(self.survival_carriers_per_graph),
            "survival_draws": int(self.survival_draws),
            "survival_tail_radii": [
                int(value) for value in self.survival_tail_radii
            ],
            "survival_replacement_candidates": int(
                self.survival_replacement_candidates
            ),
            "survival_exact_limit": int(self.survival_exact_limit),
            "survival_random_attempts": int(self.survival_random_attempts),
            "beneficial_donors_per_source": int(self.beneficial_donors_per_source),
            "beneficial_path": {
                "version": BENEFICIAL_CARRIAGE_VERSION,
                "atol": float(self.beneficial_atol),
                "rtol": float(self.beneficial_rtol),
                "max_intervals": int(self.beneficial_max_intervals),
                "sign": "positive means the donor intervention increases task loss",
            },
            "matched_reference_dose": float(min(self.interpolation_doses)),
            "effect_floor": float(self.effect_floor),
            "bootstrap_replicates": int(self.bootstrap_replicates),
            "analysis_seed": int(self.analysis_seed),
            "extensions": {
                "output_carriage": bool(self.compute_output_carriage),
                "beneficial": bool(self.compute_beneficial),
                "survival": bool(self.compute_survival),
                "scale_analysis": bool(self.compute_scale_analysis),
            },
            "finite_estimand": (
                "clean-minus-donor final pre-pooling node-state change, projected "
                "through the clean graph-output Jacobian"
            ),
            "bamberger_estimand": (
                "mean node-level pre-pooling range from entrywise-absolute Jacobians "
                "with respect to differentiable one-hot atom inputs"
            ),
            "interpolation_estimand": (
                "Functional carriage under convex interpolation from the clean atom "
                "embedding to the realised donor atom embedding, retaining the clean "
                "graph-output projection"
            ),
            "matched_reference_estimand": (
                "each finite-dose profile compared with the smallest-dose profile on "
                "the identical graph, source, donor direction, carrier set, task "
                "projection, and aggregation"
            ),
            "aggregation": (
                "donor-normalise; donor -> source -> graph; 95% graph bootstrap"
            ),
            "survival_estimand": (
                "signed task-output-projected singleton vectors versus the actual joint "
                "semantic intervention: R=J/A, additive survival=C/A, and nonlinear "
                "residual=(J-C)/A; one mapping per exact shell, with the same singleton "
                "responses and mappings composed into cumulative far tails"
            ),
            "shell_replacement_control": (
                "the identical source coalition receives external graph-balanced, "
                "minimum-degree-gap semantic donors selected by pre-outcome dose matching"
            ),
        }

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.scientific_record)

    @property
    def core_cache_fingerprint(self) -> str:
        """Scientific identity of the original reach shard only."""

        return stable_hash(
            {
                "component": "reach-core-v8",
                "analysis_version": self.profile.analysis_version,
                "dataset": self.profile.name,
                "seed": int(self.seed),
                "analysis_seed": int(self.analysis_seed),
                "sources_per_graph": int(self.sources_per_graph),
                "donors_per_source": int(self.donors_per_source),
                "semantic_donor_graphs": int(self.semantic_donor_graphs),
                "bamberger_output_nodes": int(self.bamberger_output_nodes),
                "bamberger_output_channels": int(self.bamberger_output_channels),
                "interpolation_doses": list(self.interpolation_doses),
                "atom_vocab_size": (
                    None
                    if self.profile.atom_vocab_size is None
                    else int(self.profile.atom_vocab_size)
                ),
                "node_feature_fields": int(self.profile.node_feature_fields),
                "channels": list(self.channels),
                "event_stage": self.profile.event_stage,
            }
        )

    @property
    def output_cache_fingerprint(self) -> str:
        return stable_hash(
            {
                "component": OUTPUT_CARRIAGE_VERSION,
                "core_events": self.core_cache_fingerprint,
                "atol": OUTPUT_PATH_ATOL,
                "rtol": OUTPUT_PATH_RTOL,
                "max_intervals": OUTPUT_PATH_MAX_INTERVALS,
            }
        )

    @property
    def beneficial_cache_fingerprint(self) -> str:
        return stable_hash(
            {
                "component": BENEFICIAL_CARRIAGE_VERSION,
                "analysis_version": self.profile.analysis_version,
                "dataset": self.profile.name,
                "seed": int(self.seed),
                "analysis_seed": int(self.analysis_seed),
                "sources_per_graph": int(self.sources_per_graph),
                "donors_per_source": int(self.beneficial_donors_per_source),
                "semantic_donor_graphs": int(self.semantic_donor_graphs),
                "atol": float(self.beneficial_atol),
                "rtol": float(self.beneficial_rtol),
                "max_intervals": int(self.beneficial_max_intervals),
                "channels": list(self.channels),
            }
        )

    @property
    def survival_cache_fingerprint(self) -> str:
        return stable_hash(
            {
                "component": SURVIVAL_VERSION,
                "analysis_version": self.profile.analysis_version,
                "dataset": self.profile.name,
                "seed": int(self.seed),
                "analysis_seed": int(self.analysis_seed),
                "semantic_donor_graphs": int(self.semantic_donor_graphs),
                "carriers_per_graph": int(self.survival_carriers_per_graph),
                "draws": int(self.survival_draws),
                "tail_radii": list(self.survival_tail_radii),
                "replacement_candidates": int(
                    self.survival_replacement_candidates
                ),
                "exact_limit": int(self.survival_exact_limit),
                "random_attempts": int(self.survival_random_attempts),
                "effect_floor": float(self.effect_floor),
            }
        )


def _legacy_slow_run_fingerprint(config: ZincReachConfig) -> str:
    """Fingerprint emitted by the initial, prohibitively expensive v8/v3 run."""

    record = copy.deepcopy(config.scientific_record)
    record["graphs"] = 128
    record["survival_carriers_per_graph"] = 6
    record["survival_draws"] = 4
    record["survival_replica_batch_size"] = 128
    record.pop("beneficial_donors_per_source", None)
    record["beneficial_path"] = {
        "version": BENEFICIAL_CARRIAGE_VERSION,
        "atol": 1.0e-6,
        "rtol": 1.0e-5,
        "max_intervals": 128,
        "sign": "positive means the donor intervention increases task loss",
    }
    record["survival_estimand"] = (
        "signed task-output-projected singleton vectors versus the actual joint "
        "semantic intervention: R=J/A, additive survival=C/A, and nonlinear "
        "residual=(J-C)/A; exact shells and independently permuted far tails"
    )
    return stable_hash(record)

def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        temporary.write_text("", encoding="utf-8")
        os.replace(temporary, path)
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _repository_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _seed(config: ZincReachConfig, *parts: Any) -> int:
    digest = stable_hash(
        {"analysis_seed": int(config.analysis_seed), "parts": list(parts)},
        length=16,
    )
    return int(digest, 16) % (2**32)


def discover_seed_checkpoint(results_root: str | Path, seed: int = 0) -> Path:
    """Resolve one seed's best checkpoint without relying on modification time."""

    root = Path(results_root)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint results root not found: {root}")

    standard = [
        path
        for path in root.glob("**/ckpt/*.ckpt")
        if path.parent.parent.name == str(int(seed))
    ]
    if standard:
        numeric = [path for path in standard if path.stem.isdigit()]
        return (
            max(numeric, key=lambda path: int(path.stem))
            if numeric
            else sorted(standard)[-1]
        )

    recovery_roots = sorted(
        path
        for path in (root / "_recovery_checkpoints").glob(f"seed{int(seed)}_*")
        if path.is_dir()
    )
    for filename in ("best.ckpt", "latest.ckpt", "first_after_resume.ckpt"):
        matches = [path / filename for path in recovery_roots if (path / filename).is_file()]
        if matches:
            if len(matches) > 1:
                raise RuntimeError(
                    f"ambiguous seed-{seed} recovery checkpoints under {root}: {matches}"
                )
            return matches[0]
    fallbacks = [
        checkpoint
        for directory in recovery_roots
        for checkpoint in sorted(directory.glob("*.ckpt"))
    ]
    if len(fallbacks) == 1:
        return fallbacks[0]
    if fallbacks:
        raise RuntimeError(
            f"ambiguous seed-{seed} recovery checkpoints under {root}: {fallbacks}"
        )
    raise FileNotFoundError(f"no checkpoint for seed {seed} under {root}")


def checkpoint_registry(
    tasks: Sequence[str],
    *,
    seed: int,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return explicit ``task:seed -> path`` entries for the canonical loader."""

    supplied = dict(overrides or {})
    resolved: dict[str, str] = {}
    for task in tasks:
        key = f"{task}:{int(seed)}"
        if key in supplied:
            checkpoint = Path(supplied[key])
            if not checkpoint.is_file():
                raise FileNotFoundError(f"explicit checkpoint does not exist: {checkpoint}")
        else:
            spec = get_grit_task(task)
            checkpoint = discover_seed_checkpoint(
                Path(spec.drive_dir) / "results",
                seed=int(seed),
            )
        resolved[key] = str(checkpoint)
    return resolved


def _methodology_config(
    config: ZincReachConfig,
    *,
    output_dir: Path,
    checkpoints: Mapping[str, str],
) -> MethodologyConfig:
    sizes = RunSizes(
        discovery_graphs=int(config.graphs),
        causal_graphs=1,
        clean_ablation_graphs=1,
        semantic_donor_graphs=int(config.semantic_donor_graphs),
        sources_per_graph=int(config.sources_per_graph),
        donors_per_source=int(config.donors_per_source),
    )
    return MethodologyConfig(
        output_dir=str(output_dir / "prepared"),
        tasks=tuple(config.tasks),
        train_seeds=(int(config.seed),),
        sizes=sizes,
        bootstrap=BootstrapPolicy(),
        execution=ExecutionPolicy(graphs_per_batch=1),
        analysis_seed=int(config.analysis_seed),
        accelerator=str(config.accelerator),
        num_threads=int(config.num_threads),
        checkpoints=dict(checkpoints),
        skip_install=True,
        resume=True,
        strict_audits=False,
        compute_beneficial_carriage=False,
    )


def _atom_embedding(net: Any, *, vocab_size: int) -> Any:
    """Find the checkpoint's unique atomic-number/type embedding."""

    import torch

    node_encoder = getattr(getattr(net, "encoder", None), "node_encoder", None)
    candidates = [
        module
        for module in node_encoder.modules()
        if isinstance(module, torch.nn.Embedding)
        and int(module.num_embeddings) == int(vocab_size)
    ] if node_encoder is not None else []
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one {int(vocab_size)}-type atom embedding, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _atom_embeddings(net: Any, *, profile: ReachProfile) -> tuple[Any, ...]:
    """Return the ordered categorical embeddings used by the node encoder.

    ZINC and QM9 have one categorical atom field.  OGB Peptides has nine and
    its AtomEncoder sums one embedding per field.  Restrict discovery to the
    node encoder so bond embeddings cannot be selected accidentally.
    """

    import torch

    if int(profile.node_feature_fields) == 1:
        if profile.atom_vocab_size is None:
            raise RuntimeError("single-field input requires an atom vocabulary size")
        return (
            _atom_embedding(net, vocab_size=int(profile.atom_vocab_size)),
        )

    node_encoder = getattr(getattr(net, "encoder", None), "node_encoder", None)
    candidates = (
        tuple(
            module
            for module in node_encoder.modules()
            if isinstance(module, torch.nn.Embedding)
        )
        if node_encoder is not None
        else ()
    )
    expected = int(profile.node_feature_fields)
    if len(candidates) != expected:
        raise RuntimeError(
            f"expected {expected} categorical node-feature embeddings, "
            f"found {len(candidates)}"
        )
    return candidates


def _after_feature_encoder(
    prepared: Any,
    graphs: Sequence[Any],
    *,
    profile: ReachProfile,
) -> tuple[Any, Any, tuple[Any, ...]]:
    """Run only the categorical node/edge encoder."""

    from torch_geometric.data import Batch

    if not graphs:
        raise ValueError("at least one graph is required")
    net = prepared.runtime.model.model
    batch = Batch.from_data_list([graph.clone() for graph in graphs]).to(
        prepared.runtime.device
    )
    raw_features = batch.x.long().detach().clone()
    if raw_features.ndim == 1:
        raw_features = raw_features[:, None]
    if int(raw_features.shape[1]) != int(profile.node_feature_fields):
        raise RuntimeError(
            f"expected {int(profile.node_feature_fields)} node-feature fields, "
            f"found {int(raw_features.shape[1])}"
        )
    encoded = net.encoder(batch)
    embeddings = _atom_embeddings(net, profile=profile)
    expected = sum(
        embedding(raw_features[:, field])
        for field, embedding in enumerate(embeddings)
    )
    if encoded.x.shape != expected.shape:
        raise RuntimeError("atom encoder geometry differs from its categorical embeddings")
    if not bool((encoded.x.detach() - expected.detach()).abs().max() <= 1.0e-6):
        raise RuntimeError(
            "node encoder is not equivalent to the registered categorical embeddings"
        )
    return encoded, raw_features, embeddings


def _finish_encoding(net: Any, after_encoder: Any) -> Any:
    """Run RRWP, optional pre-MP, and optional VNode up to the layer input."""

    data = after_encoder
    if hasattr(net, "rrwp_abs_encoder"):
        data = net.rrwp_abs_encoder(data)
        data = net.rrwp_rel_encoder(data)
    if hasattr(net, "pre_mp"):
        data = net.pre_mp(data)
    global_vnode = getattr(net, "global_vnode", None)
    if global_vnode is not None:
        data = global_vnode(data)
    return data


def _real_mask(data: Any) -> Any:
    import torch

    mask = getattr(data, "real_node_mask", None)
    if mask is None:
        return torch.ones(int(data.x.shape[0]), dtype=torch.bool, device=data.x.device)
    return mask


def _forward_final(net: Any, template: Any, x: Any, edge_attr: Any, real_mask: Any) -> Any:
    data = copy.copy(template)
    data.x = x
    data.edge_attr = edge_attr
    output = net.layers(data)
    return output.x[real_mask]


def _bamberger_rows(
    config: ZincReachConfig,
    prepared: Any,
    *,
    task: str,
    graph_id: int,
    base: Any,
) -> list[dict[str, Any]]:
    """Literal pre-pooling node-level Jacobian proxy for one molecule."""

    import torch
    import torch.nn.functional as functional

    net = prepared.runtime.model.model
    after_encoder, raw_features, embeddings = _after_feature_encoder(
        prepared,
        [base],
        profile=config.profile,
    )
    nodes = int(base.num_nodes)
    relaxed_inputs = []
    encoded_input = None
    for field, embedding in enumerate(embeddings):
        one_hot = functional.one_hot(
            raw_features[:, field],
            num_classes=int(embedding.num_embeddings),
        ).to(dtype=embedding.weight.dtype)
        one_hot.requires_grad_(True)
        relaxed_inputs.append(one_hot)
        contribution = one_hot @ embedding.weight
        encoded_input = contribution if encoded_input is None else encoded_input + contribution

    data = after_encoder.clone()
    data.x = encoded_input
    layer_input = _finish_encoding(net, data)
    mask = _real_mask(layer_input)
    final = _forward_final(
        net,
        layer_input,
        layer_input.x,
        layer_input.edge_attr,
        mask,
    )
    if int(final.shape[0]) != nodes:
        raise RuntimeError("Bamberger proxy did not return one final state per real node")

    rng = np.random.default_rng(_seed(config, "bamberger", graph_id))
    output_nodes = np.sort(
        rng.choice(
            nodes,
            size=min(nodes, int(config.bamberger_output_nodes)),
            replace=False,
        )
    )
    width = int(final.shape[-1])
    output_channels = np.sort(
        rng.choice(
            width,
            size=min(width, int(config.bamberger_output_channels)),
            replace=False,
        )
    )
    distances = shortest_path_distances(base.edge_index, nodes)
    rows: list[dict[str, Any]] = []
    calls = len(output_nodes) * len(output_channels)
    completed = 0
    for output_node in output_nodes:
        influence = torch.zeros(nodes, device=final.device, dtype=final.dtype)
        for output_channel in output_channels:
            completed += 1
            gradients = torch.autograd.grad(
                final[int(output_node), int(output_channel)],
                tuple(relaxed_inputs),
                retain_graph=completed < calls,
                allow_unused=False,
            )
            influence += sum(
                gradient.abs().sum(dim=-1) for gradient in gradients
            )
        for input_node in range(nodes):
            distance = distances[int(output_node), input_node]
            if not np.isfinite(distance):
                continue
            rows.append(
                {
                    "analysis_version": config.profile.analysis_version,
                    "fingerprint": config.fingerprint,
                    "task": task,
                    "model_label": TASK_LABELS[task],
                    "seed": int(config.seed),
                    "graph": int(graph_id),
                    "output_node": int(output_node),
                    "input_node": int(input_node),
                    "distance": int(distance),
                    "influence": float(influence[input_node].detach().cpu()),
                    "sampled_output_channels": int(len(output_channels)),
                    "input_space": (
                        "differentiable one-hot atom type"
                        if len(embeddings) == 1
                        else "differentiable concatenated one-hot OGB atom features"
                    ),
                    "output_space": "final pre-pooling node embedding",
                }
            )
    return rows


def _project_final_change(change: Any, clean_gradient: Any) -> Any:
    """Project ``[E,N,W]`` changes through ``[T,N,W]`` and return ``[E,N]``."""

    import torch

    if change.ndim != 3 or clean_gradient.ndim != 3:
        raise ValueError("change and gradient must be [E,N,W] and [T,N,W]")
    if tuple(change.shape[1:]) != tuple(clean_gradient.shape[1:]):
        raise ValueError("carrier change and output-gradient geometry differ")
    contribution = torch.einsum("enw,tnw->ent", change, clean_gradient)
    return torch.linalg.vector_norm(contribution, dim=-1)


def _project_final_vector(change: Any, clean_gradient: Any) -> Any:
    """Project ``[E,N,W]`` changes through ``[T,N,W]`` into ``[E,N,T]``."""

    import torch

    if change.ndim != 3 or clean_gradient.ndim != 3:
        raise ValueError("change and gradient must be [E,N,W] and [T,N,W]")
    if tuple(change.shape[1:]) != tuple(clean_gradient.shape[1:]):
        raise ValueError("carrier change and output-gradient geometry differ")
    return torch.einsum("enw,tnw->ent", change, clean_gradient)


def _semantic_interpolation_mass(
    config: ZincReachConfig,
    prepared: Any,
    *,
    base: Any,
    variants: Sequence[Any],
    events: Sequence[Any],
    clean_final: Any,
    clean_gradient: Any,
    full_mass: Any,
) -> Any:
    """Return task-projected mass as ``[dose, event, carrier]``."""

    import torch

    if len(variants) != len(events):
        raise ValueError("semantic variants and events are misaligned")
    nodes = int(base.num_nodes)
    event_count = len(events)
    doses = tuple(float(value) for value in config.interpolation_doses)
    output = torch.empty(
        (len(doses), event_count, nodes),
        dtype=full_mass.dtype,
        device=full_mass.device,
    )
    full_index = next(
        index for index, dose in enumerate(doses) if np.isclose(dose, 1.0)
    )
    output[full_index] = full_mass

    net = prepared.runtime.model.model
    embeddings = _atom_embeddings(net, profile=config.profile)

    conditions = [
        (dose_index, event_index)
        for dose_index, dose in enumerate(doses)
        if dose_index != full_index
        for event_index in range(event_count)
    ]
    batch_size = int(config.interpolation_batch_size)
    for start in range(0, len(conditions), batch_size):
        chunk = conditions[start : start + batch_size]
        hook_calls = [0 for _ in embeddings]

        def make_interpolation_hook(field: int, embedding: Any):
            def interpolate_embedding(
                _module: Any,
                inputs: Any,
                result: Any,
            ) -> Any:
                hook_calls[field] += 1
                expected_nodes = (len(chunk) + 1) * nodes
                if int(result.shape[0]) != expected_nodes:
                    raise RuntimeError("interpolation embedding batch lost node alignment")
                feature_values = inputs[0].reshape(-1)
                modified = result.clone()
                for condition_index, (dose_index, event_index) in enumerate(chunk):
                    source = int(events[event_index].source)
                    global_source = (condition_index + 1) * nodes + source
                    clean_value = int(feature_values[global_source])
                    donor_row = variants[event_index].x[source].reshape(-1)
                    donor_value = int(donor_row[field])
                    dose = doses[dose_index]
                    modified[global_source] = (
                        (1.0 - dose) * embedding.weight[clean_value]
                        + dose * embedding.weight[donor_value]
                    )
                return modified

            return interpolate_embedding

        handles = [
            embedding.register_forward_hook(
                make_interpolation_hook(field, embedding)
            )
            for field, embedding in enumerate(embeddings)
        ]
        try:
            with torch.no_grad():
                captured = prepared.backend.capture(
                    [base] * (len(chunk) + 1),
                    require_grad=False,
                )
        finally:
            for handle in handles:
                handle.remove()
        if any(calls != 1 for calls in hook_calls):
            raise RuntimeError(
                f"atom embeddings fired {hook_calls} times during interpolation"
            )
        batch_clean = captured.final_state[0]
        if tuple(batch_clean.shape) != tuple(clean_final.shape):
            raise RuntimeError("interpolation clean-state geometry differs from capture")
        error = torch.linalg.vector_norm((batch_clean - clean_final).reshape(-1))
        scale = torch.linalg.vector_norm(clean_final.reshape(-1)).clamp_min(1.0e-12)
        if float((error / scale).detach().cpu()) > 1.0e-5:
            raise RuntimeError("interpolation clean-state audit failed")
        dosed_final = captured.final_state[1:]
        with torch.no_grad():
            masses = _project_final_change(
                batch_clean.unsqueeze(0) - dosed_final,
                clean_gradient,
            )
            if not bool(torch.isfinite(masses).all()):
                raise RuntimeError("non-finite interpolation carriage mass")
            for batch_index, (dose_index, event_index) in enumerate(chunk):
                output[dose_index, event_index] = masses[batch_index]
    return output


def _donor_rows(
    config: ZincReachConfig,
    prepared: Any,
    *,
    task: str,
    graph_id: int,
    channel: str,
    base: Any,
    sources: Sequence[int],
    variants: Sequence[Any],
    events: Sequence[Any],
    clean_jacobians: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate finite Functional-carriage mass for one graph/channel."""

    import torch

    capture = prepared.backend.capture_groups([[base, *variants]])[0]
    finite_change = clean_jacobians.capture.final_state.unsqueeze(0) - capture.final_state[1:]
    gradient = clean_jacobians.final_state
    finite_mass = _project_final_change(finite_change, gradient)
    if not bool(torch.isfinite(finite_mass).all()):
        raise RuntimeError("non-finite final-state reach mass")
    if len(events) != int(finite_mass.shape[0]):
        raise RuntimeError("event manifest and carrier tensor are misaligned")
    interpolation_mass = (
        _semantic_interpolation_mass(
            config,
            prepared,
            base=base,
            variants=variants,
            events=events,
            clean_final=clean_jacobians.capture.final_state,
            clean_gradient=gradient,
            full_mass=finite_mass,
        )
        if channel == "semantic"
        else None
    )

    distances = shortest_path_distances(base.edge_index, int(base.num_nodes))
    rows: list[dict[str, Any]] = []
    interpolation_rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        source = int(event.source)
        if source not in sources:
            raise RuntimeError("event source is absent from the frozen source set")
        for carrier in range(int(base.num_nodes)):
            distance = distances[source, carrier]
            if not np.isfinite(distance):
                continue
            row = {
                "analysis_version": config.profile.analysis_version,
                "fingerprint": config.fingerprint,
                "task": task,
                "model_label": TASK_LABELS[task],
                "seed": int(config.seed),
                "graph": int(graph_id),
                "channel": channel,
                "source": source,
                "donor_graph": int(event.donor_graph_id),
                "donor_node": int(event.donor_node),
                "draw": int(event.draw),
                "dose": float(event.dose),
                "carrier": int(carrier),
                "distance": int(distance),
                "functional_carriage": float(
                    finite_mass[event_index, carrier].detach().cpu()
                ),
            }
            rows.append(row)
            if interpolation_mass is not None:
                for dose_index, dose in enumerate(config.interpolation_doses):
                    interpolation_rows.append(
                        {
                            "analysis_version": config.profile.analysis_version,
                            "fingerprint": config.fingerprint,
                            "task": task,
                            "model_label": TASK_LABELS[task],
                            "seed": int(config.seed),
                            "graph": int(graph_id),
                            "source": source,
                            "donor_graph": int(event.donor_graph_id),
                            "donor_node": int(event.donor_node),
                            "draw": int(event.draw),
                            "interpolation_dose": float(dose),
                            "carrier": int(carrier),
                            "distance": int(distance),
                            "functional_carriage": float(
                                interpolation_mass[
                                    dose_index,
                                    event_index,
                                    carrier,
                                ].detach().cpu()
                            ),
                        }
                    )
    return rows, interpolation_rows


def _signed_output_carriage_rows(
    config: ZincReachConfig,
    prepared: Any,
    *,
    task: str,
    graph_id: int,
    base: Any,
    variants: Sequence[Any],
    events: Sequence[Any],
) -> list[dict[str, Any]]:
    """Integrate signed scalar-output contribution for each semantic donor path."""

    import torch

    if len(variants) != len(events) or not events:
        raise ValueError("signed output carriage requires aligned donor events")
    capture = prepared.backend.capture_groups([[base, *variants]])[0]
    event_count = len(events)
    clean = capture.final_state[0].unsqueeze(0).expand(event_count, -1, -1)
    intervened = capture.final_state[1:]
    if tuple(intervened.shape) != tuple(clean.shape):
        raise RuntimeError("output-carriage endpoint states lost event alignment")
    output_from_pooled = prepared.backend.output_from_pooled(capture.target[0:1])
    path = integrated_loss_carriage(
        clean,
        intervened,
        output_from_pooled,
        carrier_weights=prepared.backend.carriage_weights(base, clean),
        atol=OUTPUT_PATH_ATOL,
        rtol=OUTPUT_PATH_RTOL,
        max_intervals=OUTPUT_PATH_MAX_INTERVALS,
    )
    signed = path["carriage"]
    if tuple(signed.shape) != (event_count, int(base.num_nodes)):
        raise RuntimeError("signed output carriage has invalid carrier geometry")
    expected_delta = (
        capture.z[0].reshape(-1)[0] - capture.z[1:].reshape(event_count, -1)[:, 0]
    )
    endpoint_error = (path["loss_delta"] - expected_delta).abs()
    endpoint_tolerance = OUTPUT_PATH_ATOL + OUTPUT_PATH_RTOL * expected_delta.abs()
    completeness = path["completeness_residual"].abs()
    completeness_tolerance = OUTPUT_PATH_ATOL + OUTPUT_PATH_RTOL * path[
        "loss_delta"
    ].abs()
    quadrature_tolerance = OUTPUT_PATH_ATOL + OUTPUT_PATH_RTOL * signed.abs().sum(
        dim=-1
    )
    finite_path = torch.isfinite(signed).all(dim=-1)
    accepted = (
        finite_path
        & (endpoint_error <= 5.0 * endpoint_tolerance)
        & (completeness <= 5.0 * completeness_tolerance)
        & (path["quadrature_error"] <= 5.0 * quadrature_tolerance)
    )
    if not bool(accepted.all()):
        failed = int((~accepted).sum().detach().cpu())
        finite_endpoint = endpoint_error[torch.isfinite(endpoint_error)]
        finite_completeness = completeness[torch.isfinite(completeness)]
        finite_quadrature = path["quadrature_error"][
            torch.isfinite(path["quadrature_error"])
        ]
        endpoint_max = (
            float(finite_endpoint.max().detach().cpu())
            if finite_endpoint.numel()
            else np.nan
        )
        completeness_max = (
            float(finite_completeness.max().detach().cpu())
            if finite_completeness.numel()
            else np.nan
        )
        quadrature_max = (
            float(finite_quadrature.max().detach().cpu())
            if finite_quadrature.numel()
            else np.nan
        )
        print(
            f"[output-carriage:warning] {task} graph={int(graph_id)}: "
            f"retaining best estimates for {failed}/{event_count} paths outside "
            "the soft numerical audit; "
            f"max endpoint error={endpoint_max:.3e}, "
            f"completeness={completeness_max:.3e}, "
            f"quadrature error={quadrature_max:.3e}, "
            f"intervals={int(path['intervals'].max().detach().cpu())}. "
            "The task-level audit table reports all retained errors.",
            flush=True,
        )

    distances = shortest_path_distances(base.edge_index, int(base.num_nodes))
    output: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        source = int(event.source)
        for carrier in range(int(base.num_nodes)):
            distance = distances[source, carrier]
            if not np.isfinite(distance):
                continue
            value = float(signed[event_index, carrier].detach().cpu())
            output.append(
                {
                    "analysis_version": config.profile.analysis_version,
                    "fingerprint": config.fingerprint,
                    "output_carriage_version": OUTPUT_CARRIAGE_VERSION,
                    "task": task,
                    "model_label": TASK_LABELS[task],
                    "seed": int(config.seed),
                    "graph": int(graph_id),
                    "channel": "semantic",
                    "source": source,
                    "donor_graph": int(event.donor_graph_id),
                    "donor_node": int(event.donor_node),
                    "draw": int(event.draw),
                    "carrier": int(carrier),
                    "distance": int(distance),
                    "signed_output_carriage": value,
                    "absolute_output_carriage": abs(value),
                    "event_output_delta": float(
                        path["loss_delta"][event_index].detach().cpu()
                    ),
                    "completeness_residual": float(
                        path["completeness_residual"][event_index].detach().cpu()
                    ),
                    "quadrature_error": float(
                        path["quadrature_error"][event_index].detach().cpu()
                    ),
                    "intervals": int(path["intervals"][event_index].detach().cpu()),
                    "converged": bool(path["converged"][event_index].detach().cpu()),
                    "finite_path": bool(finite_path[event_index].detach().cpu()),
                    "audit_accepted": bool(accepted[event_index].detach().cpu()),
                    "endpoint_replay_error": float(
                        endpoint_error[event_index].detach().cpu()
                    ),
                    "endpoint_replay_tolerance": float(
                        endpoint_tolerance[event_index].detach().cpu()
                    ),
                    "completeness_tolerance": float(
                        completeness_tolerance[event_index].detach().cpu()
                    ),
                    "quadrature_tolerance": float(
                        quadrature_tolerance[event_index].detach().cpu()
                    ),
                }
            )
    return output


def _measure_output_carriage_graph(
    config: ZincReachConfig,
    prepared: Any,
    *,
    task: str,
    graph_id: int,
) -> dict[str, Any]:
    """Measure semantic signed-output carriage without recomputing base reach caches."""

    base = prepared.runtime.eval_ds[int(graph_id)]
    sources = tuple(
        int(value)
        for value in sample_sources(
            int(base.num_nodes),
            int(config.sources_per_graph),
            np.random.default_rng(_seed(config, "sources", graph_id)),
        )
    )
    variants: list[Any] = []
    events: list[Any] = []
    for source in sources:
        source_variants, source_events = build_channel_events(
            base,
            graph_id=int(graph_id),
            source=int(source),
            channel="semantic",
            stage=config.profile.event_stage,
            donors=int(config.donors_per_source),
            rng=np.random.default_rng(
                _seed(config, "events", graph_id, "semantic", int(source))
            ),
            task=prepared.task,
            semantic_pool=prepared.donor_pool,
            duplicate_tolerance=1.0e-7,
        )
        variants.extend(source_variants)
        events.extend(source_events)
    rows = (
        _signed_output_carriage_rows(
            config,
            prepared,
            task=task,
            graph_id=int(graph_id),
            base=base,
            variants=variants,
            events=events,
        )
        if variants
        else []
    )
    return {
        "output_carriage_version": OUTPUT_CARRIAGE_VERSION,
        "analysis_version": config.profile.analysis_version,
        "fingerprint": config.fingerprint,
        "cache_fingerprint": config.output_cache_fingerprint,
        "checkpoint_sha256": str(prepared.checkpoint_sha),
        "task": task,
        "graph": int(graph_id),
        "output_carriage_rows": rows,
    }


def _beneficial_rows_for_channel(
    config: ZincReachConfig,
    prepared: Any,
    *,
    task: str,
    graph_id: int,
    channel: str,
    base: Any,
    variants: Sequence[Any],
    events: Sequence[Any],
    capture: Any | None = None,
) -> list[dict[str, Any]]:
    """Integrate positive-is-beneficial task-loss carriage for donor events."""

    import torch

    if len(variants) != len(events) or not variants:
        return []
    if capture is None:
        capture = prepared.backend.capture_groups([[base, *variants]])[0]
    event_count = len(events)
    clean = capture.final_state[0].unsqueeze(0).expand(event_count, -1, -1)
    intervened = capture.final_state[1:]
    path = integrated_loss_carriage(
        clean,
        intervened,
        prepared.backend.loss_from_pooled(capture.target[0:1]),
        carrier_weights=prepared.backend.carriage_weights(base, clean),
        atol=float(config.beneficial_atol),
        rtol=float(config.beneficial_rtol),
        max_intervals=int(config.beneficial_max_intervals),
    )
    # The numerical engine integrates intervention -> clean.  Canonical B is
    # positive when the clean learned function reduces the intervention loss.
    event_b = -path["carriage"]
    event_loss_increase = -path["loss_delta"]
    direct_losses = prepared.backend.loss_per_graph(
        capture.prediction,
        capture.target,
    ).reshape(-1)
    direct_increase = direct_losses[1:] - direct_losses[0]
    endpoint_error = (event_loss_increase - direct_increase).abs()
    completeness = event_b.sum(dim=-1) - event_loss_increase
    finite = (
        torch.isfinite(event_b).all(dim=-1)
        & torch.isfinite(event_loss_increase)
        & torch.isfinite(direct_increase)
    )
    tolerance = float(config.beneficial_atol) + float(config.beneficial_rtol) * (
        event_loss_increase.abs() + event_b.abs().sum(dim=-1)
    )
    accepted = (
        finite
        & path["converged"]
        & (endpoint_error <= 5.0 * tolerance)
        & (completeness.abs() <= 5.0 * tolerance)
        & (path["quadrature_error"] <= 5.0 * tolerance)
    )
    if not bool(accepted.all()):
        print(
            f"[beneficial:warning] {task} {channel} graph={int(graph_id)}: "
            f"retaining {int((~accepted).sum().detach().cpu())}/{event_count} "
            "paths outside the soft numerical audit",
            flush=True,
        )
    distances = shortest_path_distances(base.edge_index, int(base.num_nodes))
    rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        source = int(event.source)
        for carrier in range(int(base.num_nodes)):
            distance = distances[source, carrier]
            if not np.isfinite(distance):
                continue
            rows.append(
                {
                    "analysis_version": config.profile.analysis_version,
                    "fingerprint": config.fingerprint,
                    "beneficial_version": BENEFICIAL_CARRIAGE_VERSION,
                    "task": task,
                    "model_label": TASK_LABELS[task],
                    "seed": int(config.seed),
                    "graph": int(graph_id),
                    "channel": channel,
                    "source": source,
                    "donor_graph": int(event.donor_graph_id),
                    "donor_node": int(event.donor_node),
                    "draw": int(event.draw),
                    "source_degree": int(event.source_degree),
                    "donor_degree": int(event.donor_degree),
                    "degree_gap": int(event.degree_gap),
                    "dose": float(event.dose),
                    "payload_fingerprint": str(event.payload_fingerprint),
                    "carrier": int(carrier),
                    "distance": int(distance),
                    "beneficial_carriage": float(
                        event_b[event_index, carrier].detach().cpu()
                    ),
                    "event_loss_increase": float(
                        event_loss_increase[event_index].detach().cpu()
                    ),
                    "direct_loss_increase": float(
                        direct_increase[event_index].detach().cpu()
                    ),
                    "completeness_residual": float(
                        completeness[event_index].detach().cpu()
                    ),
                    "endpoint_replay_error": float(
                        endpoint_error[event_index].detach().cpu()
                    ),
                    "quadrature_error": float(
                        path["quadrature_error"][event_index].detach().cpu()
                    ),
                    "intervals": int(path["intervals"][event_index].detach().cpu()),
                    "converged": bool(path["converged"][event_index].detach().cpu()),
                    "audit_accepted": bool(accepted[event_index].detach().cpu()),
                }
            )
    return rows


def _measure_beneficial_graph(
    config: ZincReachConfig,
    prepared: Any,
    *,
    task: str,
    graph_id: int,
) -> dict[str, Any]:
    """Measure both semantic and structural Beneficial carriage in one cache shard."""

    base = prepared.runtime.eval_ds[int(graph_id)]
    sources = tuple(
        int(value)
        for value in sample_sources(
            int(base.num_nodes),
            int(config.sources_per_graph),
            np.random.default_rng(_seed(config, "sources", graph_id)),
        )
    )
    channel_batches: list[tuple[str, list[Any], list[Any]]] = []
    for channel in config.channels:
        variants: list[Any] = []
        events: list[Any] = []
        for source in sources:
            source_variants, source_events = build_channel_events(
                base,
                graph_id=int(graph_id),
                source=source,
                channel=channel,
                stage=f"{config.profile.event_stage}_beneficial",
                donors=int(config.beneficial_donors_per_source),
                rng=np.random.default_rng(
                    _seed(config, "events", graph_id, channel, source)
                ),
                task=prepared.task,
                semantic_pool=prepared.donor_pool,
                duplicate_tolerance=1.0e-7,
            )
            variants.extend(source_variants)
            events.extend(source_events)
        if variants:
            channel_batches.append((channel, variants, events))
    captures = prepared.backend.capture_groups(
        [[base, *variants] for _channel, variants, _events in channel_batches]
    ) if channel_batches else []
    rows: list[dict[str, Any]] = []
    for (channel, variants, events), capture in zip(channel_batches, captures):
        rows.extend(
            _beneficial_rows_for_channel(
                config,
                prepared,
                task=task,
                graph_id=int(graph_id),
                channel=channel,
                base=base,
                variants=variants,
                events=events,
                capture=capture,
            )
        )
    return {
        "beneficial_version": BENEFICIAL_CARRIAGE_VERSION,
        "analysis_version": config.profile.analysis_version,
        "fingerprint": config.fingerprint,
        "cache_fingerprint": config.beneficial_cache_fingerprint,
        "checkpoint_sha256": str(prepared.checkpoint_sha),
        "task": task,
        "graph": int(graph_id),
        "beneficial_rows": rows,
    }


def _empty_survival_row(
    config: ZincReachConfig,
    *,
    task: str,
    graph_id: int,
    carrier: int,
    draw: int,
    condition: str,
    radius: int,
    shell_sizes: Sequence[int],
    reason: str,
    intervention: str,
) -> dict[str, Any]:
    return {
        "analysis_version": config.profile.analysis_version,
        "fingerprint": config.fingerprint,
        "survival_version": SURVIVAL_VERSION,
        "task": task,
        "model_label": TASK_LABELS[task],
        "seed": int(config.seed),
        "graph": int(graph_id),
        "carrier": int(carrier),
        "draw": int(draw),
        "condition": condition,
        "radius": int(radius),
        "intervention": intervention,
        "coalition_size": 0,
        "shell_sizes": json.dumps([int(value) for value in shell_sizes]),
        "skipped_shell_sizes": json.dumps([int(value) for value in shell_sizes]),
        "carrier_estimable": False,
        "output_estimable": False,
        "exclusion_reason": reason,
    }


@dataclass(frozen=True)
class _SurvivalCondition:
    condition: str
    radius: int
    coalition: SemanticCoalition
    singleton_positions: tuple[int, ...]
    joint_position: int


@dataclass(frozen=True)
class _SurvivalCaptureGroup:
    carrier: int
    draw: int
    intervention: str
    variants: tuple[Any, ...]
    conditions: tuple[_SurvivalCondition, ...]


def _coalition_survival_row_from_vectors(
    config: ZincReachConfig,
    *,
    task: str,
    graph_id: int,
    carrier: int,
    draw: int,
    condition: str,
    radius: int,
    coalition: SemanticCoalition,
    singleton_vectors: Any,
    joint_vector: Any,
    output_singletons: Any,
    output_joint: Any,
) -> dict[str, Any]:
    """Evaluate one coalition at both carrier and graph-output levels."""

    event_count = len(coalition.assignments)
    carrier_result = survival_components(
        singleton_vectors.detach().cpu().numpy(),
        joint_vector.detach().cpu().numpy(),
        effect_floor=float(config.effect_floor),
    )
    output_result = survival_components(
        output_singletons.detach().cpu().numpy(),
        output_joint.detach().cpu().numpy(),
        effect_floor=float(config.effect_floor),
    )
    assignments = [
        {
            "source": item.source,
            "donor_graph": item.donor_graph,
            "donor_node": item.donor_node,
            "source_degree": item.source_degree,
            "donor_degree": item.donor_degree,
            "dose": item.dose,
        }
        for item in coalition.assignments
    ]
    row: dict[str, Any] = {
        "analysis_version": config.profile.analysis_version,
        "fingerprint": config.fingerprint,
        "survival_version": SURVIVAL_VERSION,
        "task": task,
        "model_label": TASK_LABELS[task],
        "seed": int(config.seed),
        "graph": int(graph_id),
        "carrier": int(carrier),
        "draw": int(draw),
        "condition": condition,
        "radius": int(radius),
        "intervention": coalition.intervention,
        "coalition_size": int(event_count),
        "shell_sizes": json.dumps(list(coalition.shell_sizes)),
        "skipped_shell_sizes": json.dumps(list(coalition.skipped_shell_sizes)),
        "exact_derangement": bool(coalition.exact_derangement),
        "matching_error": float(coalition.matching_error),
        "assignments": json.dumps(assignments, sort_keys=True),
        "mean_dose": float(np.mean(coalition.doses)),
        "mean_degree_gap": float(
            np.mean(
                [
                    abs(item.source_degree - item.donor_degree)
                    for item in coalition.assignments
                ]
            )
        ),
        "exclusion_reason": "",
    }
    row.update({f"carrier_{key}": value for key, value in carrier_result.items()})
    row.update({f"output_{key}": value for key, value in output_result.items()})
    return row


def _build_survival_capture_group(
    base: Any,
    *,
    carrier: int,
    draw: int,
    intervention: str,
    shell_coalitions: Mapping[int, SemanticCoalition],
    tail_radii: Sequence[int],
    task: Any,
) -> _SurvivalCaptureGroup:
    """Create one forward group reusing each singleton across every condition."""

    variants: list[Any] = []
    source_position: dict[int, int] = {}
    for distance in sorted(shell_coalitions):
        coalition = shell_coalitions[distance]
        if len(coalition.assignments) != len(coalition.singleton_variants):
            raise RuntimeError("shell assignments and singleton variants are misaligned")
        for assignment, variant in zip(
            coalition.assignments, coalition.singleton_variants
        ):
            if assignment.source in source_position:
                raise RuntimeError("a source appears in more than one exact shell")
            source_position[assignment.source] = len(variants)
            variants.append(variant)

    conditions: list[_SurvivalCondition] = []
    for distance, coalition in sorted(shell_coalitions.items()):
        joint_position = len(variants)
        variants.append(coalition.joint_variant)
        conditions.append(
            _SurvivalCondition(
                condition="exact_shell",
                radius=int(distance),
                coalition=coalition,
                singleton_positions=tuple(
                    source_position[item.source] for item in coalition.assignments
                ),
                joint_position=joint_position,
            )
        )
    for radius in tail_radii:
        tail = combine_semantic_coalitions(
            base,
            [
                coalition
                for distance, coalition in sorted(shell_coalitions.items())
                if int(distance) >= int(radius)
            ],
            task=task,
        )
        if tail is None:
            continue
        joint_position = len(variants)
        variants.append(tail.joint_variant)
        conditions.append(
            _SurvivalCondition(
                condition="far_tail",
                radius=int(radius),
                coalition=tail,
                singleton_positions=tuple(
                    source_position[item.source] for item in tail.assignments
                ),
                joint_position=joint_position,
            )
        )
    return _SurvivalCaptureGroup(
        carrier=int(carrier),
        draw=int(draw),
        intervention=intervention,
        variants=tuple(variants),
        conditions=tuple(conditions),
    )


def _evaluate_survival_groups(
    config: ZincReachConfig,
    prepared: Any,
    *,
    task: str,
    graph_id: int,
    base: Any,
    clean_gradient: Any,
    pending: Sequence[_SurvivalCaptureGroup],
) -> list[dict[str, Any]]:
    """Evaluate nested intervention groups in one strong PyG batch."""

    if not pending:
        return []
    groups = [[base, *group.variants] for group in pending]
    captures = prepared.backend.capture_groups(groups)
    if len(captures) != len(pending):
        raise RuntimeError("batched nested survival captures lost group alignment")
    rows: list[dict[str, Any]] = []
    for group, capture in zip(pending, captures):
        clean_final = capture.final_state[0]
        delta = clean_final.unsqueeze(0) - capture.final_state[1:]
        if int(delta.shape[0]) != len(group.variants):
            raise RuntimeError("nested survival capture lost variant alignment")
        projected = _project_final_vector(delta, clean_gradient)
        output_delta = capture.z[0].reshape(1, -1) - capture.z[1:].reshape(
            len(group.variants), -1
        )
        for condition in group.conditions:
            positions = list(condition.singleton_positions)
            if len(positions) != len(condition.coalition.assignments):
                raise RuntimeError("nested singleton positions lost coalition alignment")
            rows.append(
                _coalition_survival_row_from_vectors(
                    config,
                    task=task,
                    graph_id=int(graph_id),
                    carrier=group.carrier,
                    draw=group.draw,
                    condition=condition.condition,
                    radius=condition.radius,
                    coalition=condition.coalition,
                    singleton_vectors=projected[
                        positions, int(group.carrier), :
                    ],
                    joint_vector=projected[
                        condition.joint_position, int(group.carrier), :
                    ],
                    output_singletons=output_delta[positions],
                    output_joint=output_delta[condition.joint_position],
                )
            )
    return rows


def _measure_survival_graph(
    config: ZincReachConfig,
    prepared: Any,
    *,
    task: str,
    graph_id: int,
) -> dict[str, Any]:
    """Measure exact-shell and far-tail assignment survival for one graph."""

    base = prepared.runtime.eval_ds[int(graph_id)]
    nodes = int(base.num_nodes)
    distances = shortest_path_distances(base.edge_index, nodes)
    carriers = tuple(
        int(value)
        for value in sample_sources(
            nodes,
            int(config.survival_carriers_per_graph),
            np.random.default_rng(_seed(config, "survival_carriers", graph_id)),
        )
    )
    clean = prepared.backend.clean_jacobians(base)
    rows: list[dict[str, Any]] = []
    pending: list[_SurvivalCaptureGroup] = []
    pending_replicas = 0
    total_forward_replicas = 0

    def flush_pending() -> None:
        nonlocal pending_replicas
        rows.extend(
            _evaluate_survival_groups(
                config,
                prepared,
                task=task,
                graph_id=int(graph_id),
                base=base,
                clean_gradient=clean.final_state,
                pending=pending,
            )
        )
        pending.clear()
        pending_replicas = 0

    for carrier in carriers:
        finite = distances[carrier][np.isfinite(distances[carrier])]
        maximum = int(finite.max(initial=0))
        shell_nodes = {
            distance: np.flatnonzero(distances[carrier] == distance)
            .astype(np.int64)
            .tolist()
            for distance in range(1, maximum + 1)
        }
        for draw in range(int(config.survival_draws)):
            permutation_shells: dict[int, SemanticCoalition] = {}
            replacement_shells: dict[int, SemanticCoalition] = {}
            for distance, nodes_at_distance in shell_nodes.items():
                permutation = build_shell_permutation(
                    base,
                    [nodes_at_distance],
                    graph_id=int(graph_id),
                    task=prepared.task,
                    rng=np.random.default_rng(
                        _seed(
                            config,
                            "survival",
                            graph_id,
                            carrier,
                            "exact_shell",
                            distance,
                            draw,
                        )
                    ),
                    exact_limit=int(config.survival_exact_limit),
                    random_attempts=int(config.survival_random_attempts),
                )
                if permutation is None:
                    for intervention in (
                        "shell_permutation",
                        "shell_replacement",
                    ):
                        rows.append(
                            _empty_survival_row(
                                config,
                                task=task,
                                graph_id=int(graph_id),
                                carrier=carrier,
                                draw=draw,
                                condition="exact_shell",
                                radius=distance,
                                shell_sizes=[len(nodes_at_distance)],
                                reason=(
                                    "no nonzero multiset-preserving shell permutation"
                                    if intervention == "shell_permutation"
                                    else "paired shell permutation unavailable"
                                ),
                                intervention=intervention,
                            )
                        )
                    continue
                permutation_shells[int(distance)] = permutation
                replacement = build_shell_replacement(
                    base,
                    permutation,
                    task=prepared.task,
                    semantic_pool=prepared.donor_pool,
                    rng=np.random.default_rng(
                        _seed(
                            config,
                            "survival_replacement",
                            graph_id,
                            carrier,
                            "exact_shell",
                            distance,
                            draw,
                        )
                    ),
                    candidates=int(config.survival_replacement_candidates),
                )
                replacement_shells[int(distance)] = replacement

            for radius in config.survival_tail_radii:
                if any(distance >= int(radius) for distance in permutation_shells):
                    continue
                shell_sizes = [
                    len(nodes_at_distance)
                    for distance, nodes_at_distance in shell_nodes.items()
                    if distance >= int(radius)
                ]
                for intervention in ("shell_permutation", "shell_replacement"):
                    rows.append(
                        _empty_survival_row(
                            config,
                            task=task,
                            graph_id=int(graph_id),
                            carrier=carrier,
                            draw=draw,
                            condition="far_tail",
                            radius=int(radius),
                            shell_sizes=shell_sizes,
                            reason="no estimable exact shell in this far tail",
                            intervention=intervention,
                        )
                    )

            pair = (
                _build_survival_capture_group(
                    base,
                    carrier=carrier,
                    draw=draw,
                    intervention="shell_permutation",
                    shell_coalitions=permutation_shells,
                    tail_radii=config.survival_tail_radii,
                    task=prepared.task,
                ),
                _build_survival_capture_group(
                    base,
                    carrier=carrier,
                    draw=draw,
                    intervention="shell_replacement",
                    shell_coalitions=replacement_shells,
                    tail_radii=config.survival_tail_radii,
                    task=prepared.task,
                ),
            )
            pair = tuple(group for group in pair if group.variants)
            pair_replicas = sum(len(group.variants) + 1 for group in pair)
            if pair:
                total_forward_replicas += pair_replicas
                if (
                    pending
                    and pending_replicas + pair_replicas
                    > int(config.survival_replica_batch_size)
                ):
                    flush_pending()
                for group in pair:
                    pending.append(group)
                    pending_replicas += len(group.variants) + 1
                if pending_replicas >= int(config.survival_replica_batch_size):
                    flush_pending()
    flush_pending()
    return {
        "survival_version": SURVIVAL_VERSION,
        "analysis_version": config.profile.analysis_version,
        "fingerprint": config.fingerprint,
        "cache_fingerprint": config.survival_cache_fingerprint,
        "checkpoint_sha256": str(prepared.checkpoint_sha),
        "task": task,
        "graph": int(graph_id),
        "survival_rows": rows,
        "forward_replicas": int(total_forward_replicas),
        "sampled_carriers": int(len(carriers)),
    }


def _measure_graph(
    config: ZincReachConfig,
    prepared: Any,
    *,
    task: str,
    graph_id: int,
) -> dict[str, Any]:
    import torch

    base = prepared.runtime.eval_ds[int(graph_id)]
    sources = tuple(
        int(value)
        for value in sample_sources(
            int(base.num_nodes),
            int(config.sources_per_graph),
            np.random.default_rng(_seed(config, "sources", graph_id)),
        )
    )
    clean_jacobians = prepared.backend.clean_jacobians(base)
    graph_distances = shortest_path_distances(base.edge_index, int(base.num_nodes))
    finite_distances = graph_distances[np.isfinite(graph_distances)]
    if not finite_distances.size:
        raise RuntimeError("molecule has no finite shortest-path distances")
    graph_loss = prepared.backend.loss_per_graph(
        clean_jacobians.capture.prediction,
        clean_jacobians.capture.target,
    ).reshape(-1)
    if int(graph_loss.numel()) != 1 or not bool(torch.isfinite(graph_loss).all()):
        raise RuntimeError("expected one finite clean per-graph loss")
    graph_record = {
        "analysis_version": config.profile.analysis_version,
        "fingerprint": config.fingerprint,
        "task": task,
        "model_label": TASK_LABELS[task],
        "seed": int(config.seed),
        "graph": int(graph_id),
        "num_nodes": int(base.num_nodes),
        "diameter": int(finite_distances.max()),
        "graph_mae": float(graph_loss[0].detach().cpu()),
    }
    bamberger = _bamberger_rows(
        config,
        prepared,
        task=task,
        graph_id=int(graph_id),
        base=base,
    )
    donor_rows: list[dict[str, Any]] = []
    interpolation_rows: list[dict[str, Any]] = []
    for channel in config.channels:
        variants: list[Any] = []
        events: list[Any] = []
        for source in sources:
            source_variants, source_events = build_channel_events(
                base,
                graph_id=int(graph_id),
                source=int(source),
                channel=channel,
                stage=config.profile.event_stage,
                donors=int(config.donors_per_source),
                rng=np.random.default_rng(
                    _seed(config, "events", graph_id, channel, int(source))
                ),
                task=prepared.task,
                semantic_pool=prepared.donor_pool,
                duplicate_tolerance=1.0e-7,
            )
            variants.extend(source_variants)
            events.extend(source_events)
        if not variants:
            print(
                f"[reach:warning] no estimable {channel} events "
                f"({task}, graph={graph_id})",
                flush=True,
            )
            continue
        rows, channel_interpolation_rows = _donor_rows(
            config,
            prepared,
            task=task,
            graph_id=int(graph_id),
            channel=channel,
            base=base,
            sources=sources,
            variants=variants,
            events=events,
            clean_jacobians=clean_jacobians,
        )
        donor_rows.extend(rows)
        interpolation_rows.extend(channel_interpolation_rows)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "analysis_version": config.profile.analysis_version,
        "fingerprint": config.fingerprint,
        "cache_fingerprint": config.core_cache_fingerprint,
        "checkpoint_sha256": str(prepared.checkpoint_sha),
        "task": task,
        "graph": int(graph_id),
        "graph_record": graph_record,
        "donor_rows": donor_rows,
        "interpolation_rows": interpolation_rows,
        "bamberger_rows": bamberger,
    }


def _shard_path(output_dir: Path, task: str, graph_id: int) -> Path:
    return output_dir / "cache" / task / f"graph_{int(graph_id):06d}.pt"


def _output_carriage_shard_path(
    output_dir: Path,
    task: str,
    graph_id: int,
) -> Path:
    return (
        output_dir
        / "output_carriage_cache"
        / task
        / f"graph_{int(graph_id):06d}.pt"
    )


def _beneficial_shard_path(output_dir: Path, task: str, graph_id: int) -> Path:
    return output_dir / "beneficial_cache" / task / f"graph_{int(graph_id):06d}.pt"


def _survival_shard_path(output_dir: Path, task: str, graph_id: int) -> Path:
    return output_dir / "survival_cache" / task / f"graph_{int(graph_id):06d}.pt"


def _load_shard(
    path: Path,
    *,
    analysis_version: str,
    fingerprint: str,
    cache_fingerprint: str,
    checkpoint_sha256: str,
    compatible_legacy_fingerprints: Sequence[str] = (),
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    observed_cache = payload.get("cache_fingerprint")
    identity_matches = (
        observed_cache == cache_fingerprint
        if observed_cache is not None
        else payload.get("fingerprint")
        in {fingerprint, *compatible_legacy_fingerprints}
    )
    if (
        payload.get("analysis_version") != analysis_version
        or not identity_matches
        or payload.get("checkpoint_sha256") != checkpoint_sha256
        or not {
            "graph_record",
            "donor_rows",
            "interpolation_rows",
            "bamberger_rows",
        }.issubset(payload)
    ):
        return None
    return payload


def _save_shard(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _load_output_carriage_shard(
    path: Path,
    *,
    analysis_version: str,
    fingerprint: str,
    cache_fingerprint: str,
    checkpoint_sha256: str,
    compatible_legacy_fingerprints: Sequence[str] = (),
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    observed_cache = payload.get("cache_fingerprint")
    identity_matches = (
        observed_cache == cache_fingerprint
        if observed_cache is not None
        else payload.get("fingerprint")
        in {fingerprint, *compatible_legacy_fingerprints}
    )
    if (
        payload.get("output_carriage_version") != OUTPUT_CARRIAGE_VERSION
        or payload.get("analysis_version") != analysis_version
        or not identity_matches
        or payload.get("checkpoint_sha256") != checkpoint_sha256
        or "output_carriage_rows" not in payload
    ):
        return None
    return payload


def _load_extension_shard(
    path: Path,
    *,
    version_key: str,
    version: str,
    rows_key: str,
    analysis_version: str,
    fingerprint: str,
    cache_fingerprint: str,
    checkpoint_sha256: str,
    compatible_legacy_fingerprints: Sequence[str] = (),
) -> dict[str, Any] | None:
    """Load one independently versioned extension shard."""

    if not path.is_file():
        return None
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    observed_cache = payload.get("cache_fingerprint")
    identity_matches = (
        observed_cache == cache_fingerprint
        if observed_cache is not None
        else payload.get("fingerprint")
        in {fingerprint, *compatible_legacy_fingerprints}
    )
    if (
        payload.get(version_key) != version
        or payload.get("analysis_version") != analysis_version
        or not identity_matches
        or payload.get("checkpoint_sha256") != checkpoint_sha256
        or rows_key not in payload
    ):
        return None
    return payload


def _full_test_rows(
    config: ZincReachConfig,
    prepared: Any,
    *,
    task: str,
) -> list[dict[str, Any]]:
    """Return checkpoint-recomputed per-molecule test errors and scale metadata."""

    predictions = getattr(prepared.runtime, "test_predictions", None)
    targets = getattr(prepared.runtime, "test_targets", None)
    metadata = getattr(prepared.runtime, "test_graph_metadata", None)
    if predictions is None or targets is None or metadata is None:
        raise RuntimeError("full-test predictions were not retained by the GRIT loader")
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape or len(metadata) != int(predictions.shape[0]):
        raise RuntimeError("full-test prediction, target and metadata rows are misaligned")
    errors = np.abs(predictions - targets).mean(axis=1)
    output: list[dict[str, Any]] = []
    for index, meta in enumerate(metadata):
        output.append(
            {
                "analysis_version": config.profile.analysis_version,
                "fingerprint": config.fingerprint,
                "task": task,
                "model_label": TASK_LABELS[task],
                "seed": int(config.seed),
                "test_graph": int(meta["graph"]),
                "num_nodes": int(meta["num_nodes"]),
                "diameter": int(meta["diameter"]),
                "prediction": float(predictions[index].mean()),
                "target": float(targets[index].mean()),
                "graph_mae": float(errors[index]),
            }
        )
    return output


def measure(
    config: ZincReachConfig,
    *,
    output_dir: Path,
    checkpoints: Mapping[str, str] | None = None,
    install_dependencies: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Measure all requested models and cache each graph independently on Drive."""

    config.validate()
    if install_dependencies:
        env.install_dependencies(pyg_version="2.2.0")
    env.apply_compat_patches()

    import torch

    torch.set_num_threads(int(config.num_threads))
    resolved = checkpoint_registry(
        config.tasks,
        seed=int(config.seed),
        overrides=checkpoints,
    )
    print("[checkpoints] Explicit checkpoint registry:", flush=True)
    for key, path in resolved.items():
        print(f"[checkpoints] {key}: {path}", flush=True)
    methodology = _methodology_config(
        config,
        output_dir=output_dir,
        checkpoints=resolved,
    )
    compatible_slow_fingerprints = (_legacy_slow_run_fingerprint(config),)

    donor_rows: list[dict[str, Any]] = []
    interpolation_rows: list[dict[str, Any]] = []
    bamberger_rows: list[dict[str, Any]] = []
    core_failures: list[dict[str, Any]] = []
    output_carriage_rows: list[dict[str, Any]] = []
    output_carriage_failures: list[dict[str, Any]] = []
    beneficial_rows: list[dict[str, Any]] = []
    beneficial_failures: list[dict[str, Any]] = []
    survival_rows: list[dict[str, Any]] = []
    survival_failures: list[dict[str, Any]] = []
    survival_execution: list[dict[str, Any]] = []
    graph_records: list[dict[str, Any]] = []
    full_test_records: list[dict[str, Any]] = []
    health: list[dict[str, Any]] = []
    completed = 0
    total = len(config.tasks) * int(config.graphs)
    for task in config.tasks:
        prepared = prepare_task(
            methodology,
            task,
            int(config.seed),
            force_fresh_grit=False,
        )
        health.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "seed": int(config.seed),
                "test_mae": prepared.runtime.test_metric,
                "validation_mae": prepared.runtime.val_metric,
                "checkpoint": str(prepared.checkpoint),
                "checkpoint_sha256": str(prepared.checkpoint_sha),
                "parameters": prepared.runtime.checks.get("num_parameters"),
            }
        )
        if config.compute_scale_analysis:
            full_test_records.extend(
                _full_test_rows(config, prepared, task=task)
            )
        results_dir = output_dir / "results"
        _write_csv(results_dir / "full_test_metrics.csv", full_test_records)
        graph_ids = tuple(prepared.splits.discovery)[: int(config.graphs)]
        for graph_id in graph_ids:
            path = _shard_path(output_dir, task, int(graph_id))
            shard = _load_shard(
                path,
                analysis_version=config.profile.analysis_version,
                fingerprint=config.fingerprint,
                cache_fingerprint=config.core_cache_fingerprint,
                checkpoint_sha256=str(prepared.checkpoint_sha),
                compatible_legacy_fingerprints=compatible_slow_fingerprints,
            )
            if shard is None:
                try:
                    if progress:
                        print(
                            f"[reach:core] {TASK_LABELS[task]} graph={int(graph_id)}",
                            flush=True,
                        )
                    shard = _measure_graph(
                        config,
                        prepared,
                        task=task,
                        graph_id=int(graph_id),
                    )
                except Exception as error:
                    core_failures.append(
                        {
                            "task": task,
                            "model_label": TASK_LABELS[task],
                            "graph": int(graph_id),
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "will_retry": True,
                        }
                    )
                    print(
                        f"[reach:warning] {task} graph={int(graph_id)} core reach "
                        f"could not be estimated ({type(error).__name__}: {error}); "
                        "the analysis continues and the graph will be retried next run.",
                        flush=True,
                    )
                    continue
                else:
                    _save_shard(path, shard)
            elif "cache_fingerprint" not in shard:
                if progress:
                    print(
                        f"[reach:cache] reusing completed core shard for "
                        f"{TASK_LABELS[task]} graph={int(graph_id)}",
                        flush=True,
                    )
                shard = {**shard, "cache_fingerprint": config.core_cache_fingerprint}
                _save_shard(path, shard)
            donor_rows.extend(shard["donor_rows"])
            interpolation_rows.extend(shard["interpolation_rows"])
            bamberger_rows.extend(shard["bamberger_rows"])
            graph_records.append(dict(shard["graph_record"]))
            output_path = _output_carriage_shard_path(
                output_dir,
                task,
                int(graph_id),
            )
            output_shard = (
                _load_output_carriage_shard(
                    output_path,
                    analysis_version=config.profile.analysis_version,
                    fingerprint=config.fingerprint,
                    cache_fingerprint=config.output_cache_fingerprint,
                    checkpoint_sha256=str(prepared.checkpoint_sha),
                    compatible_legacy_fingerprints=compatible_slow_fingerprints,
                )
                if config.compute_output_carriage
                else {"output_carriage_rows": []}
            )
            if output_shard is None:
                try:
                    if progress:
                        print(
                            f"[reach:output-path] {TASK_LABELS[task]} "
                            f"graph={int(graph_id)}",
                            flush=True,
                        )
                    output_shard = _measure_output_carriage_graph(
                        config,
                        prepared,
                        task=task,
                        graph_id=int(graph_id),
                    )
                except Exception as error:
                    failure = {
                        "task": task,
                        "model_label": TASK_LABELS[task],
                        "graph": int(graph_id),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "estimate_retained": False,
                        "will_retry": True,
                    }
                    output_carriage_failures.append(failure)
                    print(
                        f"[output-carriage:warning] {task} graph={int(graph_id)} "
                        f"could not be estimated ({type(error).__name__}: {error}); "
                        "the core analysis continues and the graph will be retried next run.",
                        flush=True,
                    )
                    output_shard = {"output_carriage_rows": []}
                else:
                    _save_shard(output_path, output_shard)
            elif config.compute_output_carriage and "cache_fingerprint" not in output_shard:
                if progress:
                    print(
                        f"[reach:cache] reusing completed output-path shard for "
                        f"{TASK_LABELS[task]} graph={int(graph_id)}",
                        flush=True,
                    )
                output_shard = {
                    **output_shard,
                    "cache_fingerprint": config.output_cache_fingerprint,
                }
                _save_shard(output_path, output_shard)
            output_carriage_rows.extend(output_shard["output_carriage_rows"])
            beneficial_path = _beneficial_shard_path(
                output_dir,
                task,
                int(graph_id),
            )
            beneficial_shard = (
                _load_extension_shard(
                    beneficial_path,
                    version_key="beneficial_version",
                    version=BENEFICIAL_CARRIAGE_VERSION,
                    rows_key="beneficial_rows",
                    analysis_version=config.profile.analysis_version,
                    fingerprint=config.fingerprint,
                    cache_fingerprint=config.beneficial_cache_fingerprint,
                    checkpoint_sha256=str(prepared.checkpoint_sha),
                )
                if config.compute_beneficial
                else {"beneficial_rows": []}
            )
            if beneficial_shard is None:
                try:
                    if progress:
                        print(
                            f"[reach:beneficial] {TASK_LABELS[task]} "
                            f"graph={int(graph_id)} | "
                            f"{int(config.beneficial_donors_per_source)} donor/source",
                            flush=True,
                        )
                    beneficial_shard = _measure_beneficial_graph(
                        config,
                        prepared,
                        task=task,
                        graph_id=int(graph_id),
                    )
                except Exception as error:
                    beneficial_failures.append(
                        {
                            "task": task,
                            "model_label": TASK_LABELS[task],
                            "graph": int(graph_id),
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "will_retry": True,
                        }
                    )
                    print(
                        f"[beneficial:warning] {task} graph={int(graph_id)} "
                        f"could not be estimated ({type(error).__name__}: {error}); "
                        "the graph will be retried next run.",
                        flush=True,
                    )
                    beneficial_shard = {"beneficial_rows": []}
                else:
                    _save_shard(beneficial_path, beneficial_shard)
            beneficial_rows.extend(beneficial_shard["beneficial_rows"])

            survival_path = _survival_shard_path(output_dir, task, int(graph_id))
            survival_shard = (
                _load_extension_shard(
                    survival_path,
                    version_key="survival_version",
                    version=SURVIVAL_VERSION,
                    rows_key="survival_rows",
                    analysis_version=config.profile.analysis_version,
                    fingerprint=config.fingerprint,
                    cache_fingerprint=config.survival_cache_fingerprint,
                    checkpoint_sha256=str(prepared.checkpoint_sha),
                )
                if config.compute_survival
                else {"survival_rows": []}
            )
            if survival_shard is None:
                try:
                    if progress:
                        print(
                            f"[reach:survival] {TASK_LABELS[task]} "
                            f"graph={int(graph_id)} | "
                            f"{int(config.survival_carriers_per_graph)} carrier, "
                            f"{int(config.survival_draws)} draw; shared singletons",
                            flush=True,
                        )
                    survival_shard = _measure_survival_graph(
                        config,
                        prepared,
                        task=task,
                        graph_id=int(graph_id),
                    )
                except Exception as error:
                    survival_failures.append(
                        {
                            "task": task,
                            "model_label": TASK_LABELS[task],
                            "graph": int(graph_id),
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "will_retry": True,
                        }
                    )
                    print(
                        f"[survival:warning] {task} graph={int(graph_id)} "
                        f"could not be estimated ({type(error).__name__}: {error}); "
                        "the graph will be retried next run.",
                        flush=True,
                    )
                    survival_shard = {"survival_rows": []}
                else:
                    _save_shard(survival_path, survival_shard)
            survival_rows.extend(survival_shard["survival_rows"])
            if "forward_replicas" in survival_shard:
                survival_execution.append(
                    {
                        "task": task,
                        "model_label": TASK_LABELS[task],
                        "graph": int(graph_id),
                        "forward_replicas": int(survival_shard["forward_replicas"]),
                        "sampled_carriers": int(
                            survival_shard.get(
                                "sampled_carriers",
                                config.survival_carriers_per_graph,
                            )
                        ),
                    }
                )
            completed += 1
            if progress:
                print(
                    f"[reach] {completed}/{total} | {TASK_LABELS[task]} "
                    f"graph={int(graph_id)}",
                    flush=True,
                )

    results_dir = output_dir / "results"
    output_carriage_audit = summarise_output_carriage_audit(
        output_carriage_rows,
        failures=output_carriage_failures,
    )
    _print_output_carriage_audit(output_carriage_audit)
    _write_csv(results_dir / "donor_carrier_mass.csv", donor_rows)
    _write_csv(results_dir / "semantic_interpolation_mass.csv", interpolation_rows)
    _write_csv(results_dir / "bamberger_input_output_influence.csv", bamberger_rows)
    _write_csv(results_dir / "core_failures.csv", core_failures)
    _write_csv(results_dir / "semantic_output_carriage.csv", output_carriage_rows)
    _write_csv(results_dir / "beneficial_carriage.csv", beneficial_rows)
    _write_csv(results_dir / "shell_survival.csv", survival_rows)
    _write_csv(results_dir / "output_carriage_audit.csv", output_carriage_audit)
    _write_csv(
        results_dir / "output_carriage_failures.csv",
        output_carriage_failures,
    )
    _write_csv(results_dir / "beneficial_failures.csv", beneficial_failures)
    _write_csv(results_dir / "survival_failures.csv", survival_failures)
    _write_csv(results_dir / "survival_execution.csv", survival_execution)
    _write_csv(results_dir / "graph_metrics.csv", graph_records)
    _write_csv(results_dir / "model_health.csv", health)
    _write_json(
        results_dir / "measurement_manifest.json",
        {
            **config.scientific_record,
            "fingerprint": config.fingerprint,
            "repository_commit": _repository_commit(),
            "checkpoints": resolved,
            "health": health,
            "completed_graph_shards": completed,
            "donor_rows": len(donor_rows),
            "interpolation_rows": len(interpolation_rows),
            "bamberger_rows": len(bamberger_rows),
            "core_failures": core_failures,
            "output_carriage_rows": len(output_carriage_rows),
            "output_carriage_audit": output_carriage_audit,
            "output_carriage_failures": output_carriage_failures,
            "beneficial_rows": len(beneficial_rows),
            "beneficial_failures": beneficial_failures,
            "survival_rows": len(survival_rows),
            "survival_failures": survival_failures,
            "survival_forward_replicas": int(
                sum(row["forward_replicas"] for row in survival_execution)
            ),
            "component_cache_fingerprints": {
                "core": config.core_cache_fingerprint,
                "output": config.output_cache_fingerprint,
                "beneficial": config.beneficial_cache_fingerprint,
                "survival": config.survival_cache_fingerprint,
            },
            "graph_metric_rows": len(graph_records),
            "full_test_metric_rows": len(full_test_records),
            "comparison_scope": {
                "semantic": (
                    "literal Bamberger pre-pooling Jacobian proxy and task-projected "
                    "Functional carriage"
                ),
                "structural": (
                    "finite Functional carriage only; no canonical Bamberger "
                    "structural quantity is claimed"
                ),
                "structural_interpretation": (
                    "source-conditioned structural usage, not pure hop-by-hop "
                    "transport: globally encoded RRWP relations may be accessed "
                    "directly by a carrier"
                ),
                "ground_truth": (
                    f"none for learned {config.profile.name} range; architecture "
                    "constrains accessibility but does not specify the learned usage "
                    "distribution"
                ),
            },
            "comparison_fairness": {
                "shared": (
                    "checkpoint, held-out graphs, original-graph SPD, final pre-pooling "
                    "carrier site, and graph-level bootstrap unit"
                ),
                "interpolation_sweep": (
                    "Functional carriage at every nonzero donor fraction shares sources, "
                    "donor draws, source-to-carrier distances, clean output projection, "
                    "and event-wise normalisation"
                ),
                "literal_bamberger_difference": (
                    "Bamberger remains output-centric, uses all input nodes and sampled "
                    "output nodes/channels, sums absolute coordinatewise derivatives, and "
                    "normalises per output node"
                ),
                "matched_small_dose_reference": (
                    "the smallest-dose comparison fixes graph, source, donor direction, "
                    "carrier, clean task-output projection, event normalisation, and "
                    "graph aggregation; its zero-dose-end TV is zero by construction"
                ),
                "output_coherence": (
                    "signed scalar z-output carriage is integrated along each finite "
                    "semantic donor path. Carrier contributions sum to the exact "
                    "clean-minus-intervened output change; apparent shell mass sums "
                    "magnitudes before carrier aggregation and coherent shell mass "
                    "takes magnitude after signed carrier aggregation"
                ),
                "beneficial_carriage": (
                    "exact signed task-loss allocation along each donor path; positive "
                    "means the clean learned function avoids loss caused by the semantic "
                    "or structural donor intervention"
                ),
                "shell_survival": (
                    "within-shell permutations preserve each shell multiset and test "
                    "assignment redundancy; external shell replacements use the same "
                    "source coalition with degree-law and dose matching to test aggregate "
                    "content sensitivity"
                ),
                "interpretation": (
                    "literal Bamberger versus Functional carriage remains an operational "
                    "estimand comparison. Departure from the matched smallest-dose profile "
                    "more cleanly isolates finite nonlinear change; the residual smallest-dose "
                    "TV from Bamberger measures estimand and sampling mismatch"
                ),
            },
        },
    )
    return {
        "donor_rows": donor_rows,
        "interpolation_rows": interpolation_rows,
        "bamberger_rows": bamberger_rows,
        "core_failures": core_failures,
        "output_carriage_rows": output_carriage_rows,
        "output_carriage_audit": output_carriage_audit,
        "output_carriage_failures": output_carriage_failures,
        "beneficial_rows": beneficial_rows,
        "beneficial_failures": beneficial_failures,
        "survival_rows": survival_rows,
        "survival_failures": survival_failures,
        "survival_execution": survival_execution,
        "graph_records": graph_records,
        "full_test_records": full_test_records,
        "health": health,
    }


def _float(row: Mapping[str, Any], key: str) -> float:
    return float(row[key])


def _integer(row: Mapping[str, Any], key: str) -> int:
    return int(float(row[key]))


def _boolean(row: Mapping[str, Any], key: str, *, default: bool) -> bool:
    if key not in row or row[key] == "":
        return bool(default)
    value = row[key]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def summarise_output_carriage_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    failures: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Collapse duplicated carrier diagnostics into one soft audit per task."""

    event_fields = (
        "task",
        "graph",
        "source",
        "donor_graph",
        "donor_node",
        "draw",
    )
    events: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        events.setdefault(tuple(row[field] for field in event_fields), row)
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for key, row in events.items():
        by_task[str(key[0])].append(row)
    failures_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in failures:
        failures_by_task[str(row["task"])].append(row)

    def values(task_rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
        return np.asarray(
            [float(row.get(key, 0.0) or 0.0) for row in task_rows],
            dtype=np.float64,
        )

    output: list[dict[str, Any]] = []
    for task in sorted(set(by_task) | set(failures_by_task)):
        task_rows = by_task.get(task, [])
        paths = len(task_rows)
        finite = np.asarray(
            [
                _boolean(
                    row,
                    "finite_path",
                    default=np.isfinite(_float(row, "signed_output_carriage")),
                )
                for row in task_rows
            ],
            dtype=bool,
        )
        accepted = np.asarray(
            [_boolean(row, "audit_accepted", default=True) for row in task_rows],
            dtype=bool,
        )
        converged = np.asarray(
            [_boolean(row, "converged", default=True) for row in task_rows],
            dtype=bool,
        )
        endpoint = values(task_rows, "endpoint_replay_error")
        completeness = np.abs(values(task_rows, "completeness_residual"))
        quadrature = values(task_rows, "quadrature_error")
        intervals = values(task_rows, "intervals")
        def quantile(value: np.ndarray, probability: float) -> float:
            return float(np.quantile(value, probability)) if value.size else np.nan

        def maximum(value: np.ndarray) -> float:
            return float(value.max()) if value.size else np.nan

        output.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "paths": int(paths),
                "soft_warning_paths": int((~accepted).sum()),
                "soft_warning_fraction": (
                    float((~accepted).mean()) if accepted.size else 0.0
                ),
                "nonfinite_paths": int((~finite).sum()),
                "unconverged_paths": int((~converged).sum()),
                "unconverged_fraction": (
                    float((~converged).mean()) if converged.size else 0.0
                ),
                "failed_graphs": int(len(failures_by_task.get(task, []))),
                "endpoint_error_p95": quantile(endpoint, 0.95),
                "endpoint_error_max": maximum(endpoint),
                "completeness_error_p95": quantile(completeness, 0.95),
                "completeness_error_max": maximum(completeness),
                "quadrature_error_p95": quantile(quadrature, 0.95),
                "quadrature_error_max": maximum(quadrature),
                "intervals_p95": quantile(intervals, 0.95),
                "intervals_max": int(maximum(intervals)) if intervals.size else 0,
                "estimates_retained": bool(paths),
            }
        )
    return output


def _print_output_carriage_audit(rows: Sequence[Mapping[str, Any]]) -> None:
    print("[output-carriage:audit] retained path-integration estimates", flush=True)
    for row in rows:
        print(
            f"  {row['model_label']}: paths={row['paths']}, "
            f"soft warnings={row['soft_warning_paths']} "
            f"({100.0 * float(row['soft_warning_fraction']):.2f}%), "
            f"non-finite={row['nonfinite_paths']}, "
            f"unconverged={row['unconverged_paths']}; "
            f"failed graphs={row['failed_graphs']}; "
            f"max endpoint/completeness/quadrature="
            f"{float(row['endpoint_error_max']):.3e}/"
            f"{float(row['completeness_error_max']):.3e}/"
            f"{float(row['quadrature_error_max']):.3e}; "
            f"max intervals={row['intervals_max']}",
            flush=True,
        )


def graph_donor_profiles(
    rows: Sequence[Mapping[str, Any]],
    *,
    effect_floor: float,
) -> list[dict[str, Any]]:
    """Normalise each event, then aggregate donor -> source -> graph."""

    event_fields = (
        "task",
        "graph",
        "channel",
        "source",
        "donor_graph",
        "donor_node",
        "draw",
    )
    events: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        events[tuple(row[field] for field in event_fields)].append(row)

    source_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    task_max_distance: dict[str, int] = defaultdict(int)
    for key, event_rows in events.items():
        common = dict(zip(event_fields, key))
        distances = np.asarray(
            [_integer(row, "distance") for row in event_rows],
            dtype=np.int64,
        )
        task = str(common["task"])
        task_max_distance[task] = max(
            int(task_max_distance[task]),
            int(distances.max(initial=0)),
        )
        for method in DONOR_PROFILE_METHODS:
            availability = [method in row and row[method] != "" for row in event_rows]
            if not any(availability):
                continue
            if not all(availability):
                raise RuntimeError(
                    f"method {method!r} is present for only part of one donor event"
                )
            masses = np.asarray(
                [_float(row, method) for row in event_rows],
                dtype=np.float64,
            )
            total = float(masses.sum())
            if not np.isfinite(total) or total <= float(effect_floor):
                continue
            for distance in np.unique(distances):
                source_key = (
                    task,
                    _integer(common, "graph"),
                    str(common["channel"]),
                    _integer(common, "source"),
                    method,
                    int(distance),
                )
                source_values[source_key].append(
                    float(masses[distances == int(distance)].sum() / total)
                )

    graph_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    present_sources: set[tuple[Any, ...]] = {
        key[:5] for key in source_values
    }
    for task, graph, channel, source, method in present_sources:
        maximum = int(task_max_distance[task])
        for distance in range(maximum + 1):
            values = source_values.get(
                (task, graph, channel, source, method, distance),
                [],
            )
            graph_values[(task, graph, channel, method, distance)].append(
                float(np.mean(values)) if values else 0.0
            )

    output: list[dict[str, Any]] = []
    for (task, graph, channel, method, distance), values in graph_values.items():
        output.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "graph": int(graph),
                "channel": channel,
                "method": method,
                "distance": int(distance),
                "mass": float(np.mean(values)),
            }
        )
    return output


def graph_output_coherence_profiles(
    rows: Sequence[Mapping[str, Any]],
    *,
    effect_floor: float,
) -> list[dict[str, Any]]:
    """Build graph-balanced apparent/coherent profiles from signed output paths."""

    event_fields = (
        "task",
        "graph",
        "source",
        "donor_graph",
        "donor_node",
        "draw",
    )
    events: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        events[tuple(row[field] for field in event_fields)].append(row)

    source_profiles: dict[tuple[str, int, int, str, int], list[float]] = defaultdict(
        list
    )
    active_sources: set[tuple[str, int, int, str]] = set()
    ratio_numerator: dict[tuple[str, int, int], float] = defaultdict(float)
    ratio_denominator: dict[tuple[str, int, int], float] = defaultdict(float)
    task_max_distance: dict[str, int] = defaultdict(int)
    for key, event_rows in events.items():
        task, graph, source, _donor_graph, _donor_node, _draw = key
        task = str(task)
        graph = int(graph)
        source = int(source)
        distances = np.asarray(
            [_integer(row, "distance") for row in event_rows],
            dtype=np.int64,
        )
        signed = np.asarray(
            [_float(row, "signed_output_carriage") for row in event_rows],
            dtype=np.float64,
        )
        if not np.isfinite(signed).all():
            print(
                f"[output-carriage:warning] skipping non-finite retained path "
                f"for {task} graph={graph} source={source}; see the audit table",
                flush=True,
            )
            continue
        task_max_distance[task] = max(
            int(task_max_distance[task]),
            int(distances.max(initial=0)),
        )
        apparent_by_distance: dict[int, float] = {}
        coherent_by_distance: dict[int, float] = {}
        for distance in np.unique(distances):
            selected = signed[distances == int(distance)]
            apparent = float(np.abs(selected).sum())
            coherent = float(abs(selected.sum()))
            apparent_by_distance[int(distance)] = apparent
            coherent_by_distance[int(distance)] = coherent
            ratio_numerator[(task, graph, int(distance))] += coherent
            ratio_denominator[(task, graph, int(distance))] += apparent
        apparent_total = float(sum(apparent_by_distance.values()))
        coherent_total = float(sum(coherent_by_distance.values()))
        if apparent_total > float(effect_floor):
            active_sources.add((task, graph, source, "apparent_mass"))
            for distance, value in apparent_by_distance.items():
                source_profiles[
                    (task, graph, source, "apparent_mass", distance)
                ].append(value / apparent_total)
        if coherent_total > float(effect_floor):
            active_sources.add((task, graph, source, "coherent_mass"))
            for distance, value in coherent_by_distance.items():
                source_profiles[
                    (task, graph, source, "coherent_mass", distance)
                ].append(value / coherent_total)

    graph_profiles: dict[tuple[str, int, str, int], list[float]] = defaultdict(list)
    for task, graph, source, metric in active_sources:
        for distance in range(int(task_max_distance[task]) + 1):
            donor_values = source_profiles.get(
                (task, graph, source, metric, distance),
                [],
            )
            graph_profiles[(task, graph, metric, distance)].append(
                float(np.mean(donor_values)) if donor_values else 0.0
            )

    graph_keys = sorted({(key[0], key[1]) for key in graph_profiles})
    output: list[dict[str, Any]] = []
    for task, graph in graph_keys:
        for distance in range(int(task_max_distance[task]) + 1):
            apparent_values = graph_profiles.get(
                (task, graph, "apparent_mass", distance),
                [],
            )
            coherent_values = graph_profiles.get(
                (task, graph, "coherent_mass", distance),
                [],
            )
            denominator = ratio_denominator.get((task, graph, distance), 0.0)
            output.append(
                {
                    "task": task,
                    "model_label": TASK_LABELS[task],
                    "graph": int(graph),
                    "channel": "semantic",
                    "distance": int(distance),
                    "apparent_mass": (
                        float(np.mean(apparent_values)) if apparent_values else 0.0
                    ),
                    "coherent_mass": (
                        float(np.mean(coherent_values)) if coherent_values else 0.0
                    ),
                    "coherence_ratio": (
                        ratio_numerator[(task, graph, distance)] / denominator
                        if denominator > float(effect_floor)
                        else np.nan
                    ),
                }
            )
    return output


def summarise_output_coherence(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarise distance profiles and paired expected-distance contraction."""

    profile_groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        for metric in ("apparent_mass", "coherent_mass", "coherence_ratio"):
            value = _float(row, metric)
            if np.isfinite(value):
                profile_groups[
                    (str(row["task"]), metric, _integer(row, "distance"))
                ].append(value)
    profiles: list[dict[str, Any]] = []
    for key, values in sorted(profile_groups.items()):
        task, metric, distance = key
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed)
            + int(stable_hash({"output_coherence": key}, length=8), 16),
        )
        profiles.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "metric": metric,
                "distance": int(distance),
                "mean": mean,
                "low": low,
                "high": high,
                "graphs": int(len(values)),
            }
        )

    by_graph: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_graph[(str(row["task"]), _integer(row, "graph"))].append(row)
    expected_by_task: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (task, _graph), graph_rows in by_graph.items():
        expected: dict[str, float] = {}
        for metric in ("apparent_mass", "coherent_mass"):
            mass = np.asarray(
                [_float(row, metric) for row in graph_rows],
                dtype=np.float64,
            )
            distances = np.asarray(
                [_integer(row, "distance") for row in graph_rows],
                dtype=np.float64,
            )
            total = float(mass.sum())
            if total > 0:
                expected[metric] = float(np.dot(mass, distances) / total)
                expected_by_task[(task, metric)].append(expected[metric])
        if {"apparent_mass", "coherent_mass"}.issubset(expected):
            expected_by_task[(task, "expected_distance_change")].append(
                expected["coherent_mass"] - expected["apparent_mass"]
            )

    expected_rows: list[dict[str, Any]] = []
    for key, values in sorted(expected_by_task.items()):
        task, metric = key
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed)
            + int(stable_hash({"output_expected": key}, length=8), 16),
        )
        expected_rows.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "metric": metric,
                "mean": mean,
                "low": low,
                "high": high,
                "graphs": int(len(values)),
            }
        )
    return profiles, expected_rows


def graph_bamberger_profiles(
    rows: Sequence[Mapping[str, Any]],
    *,
    effect_floor: float,
) -> list[dict[str, Any]]:
    """Normalise each output node's influence, then average output nodes per graph."""

    output_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    task_max_distance: dict[str, int] = defaultdict(int)
    for row in rows:
        task = str(row["task"])
        task_max_distance[task] = max(
            int(task_max_distance[task]),
            _integer(row, "distance"),
        )
        output_groups[
            (task, _integer(row, "graph"), _integer(row, "output_node"))
        ].append(row)

    graph_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for (task, graph, _output_node), values in output_groups.items():
        influence = np.asarray(
            [_float(row, "influence") for row in values],
            dtype=np.float64,
        )
        distances = np.asarray(
            [_integer(row, "distance") for row in values],
            dtype=np.int64,
        )
        total = float(influence.sum())
        if not np.isfinite(total) or total <= float(effect_floor):
            continue
        for distance in range(int(task_max_distance[task]) + 1):
            graph_values[(task, graph, distance)].append(
                float(influence[distances == distance].sum() / total)
            )

    output: list[dict[str, Any]] = []
    for (task, graph, distance), values in graph_values.items():
        output.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "graph": int(graph),
                "channel": "semantic",
                "method": "bamberger",
                "distance": int(distance),
                "mass": float(np.mean(values)),
            }
        )
    return output


def _usage_profiles_for_event(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
    maximum_distance: int,
    effect_floor: float,
) -> dict[str, np.ndarray] | None:
    """Return matched shell-summed and per-carrier profiles for one event."""

    distances = np.asarray(
        [_integer(row, "distance") for row in rows],
        dtype=np.int64,
    )
    values = np.asarray(
        [_float(row, value_field) for row in rows],
        dtype=np.float64,
    )
    if (
        not len(values)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or int(distances.min(initial=0)) < 0
    ):
        return None
    shell_mass = np.bincount(
        distances,
        weights=values,
        minlength=int(maximum_distance) + 1,
    ).astype(np.float64, copy=False)[: int(maximum_distance) + 1]
    shell_count = np.bincount(
        distances,
        minlength=int(maximum_distance) + 1,
    ).astype(np.float64, copy=False)[: int(maximum_distance) + 1]
    total_mass = float(shell_mass.sum())
    if not np.isfinite(total_mass) or total_mass <= float(effect_floor):
        return None

    mean_per_carrier = np.zeros_like(shell_mass)
    available = shell_count > 0
    mean_per_carrier[available] = shell_mass[available] / shell_count[available]
    per_carrier_total = float(mean_per_carrier.sum())
    carrier_total = float(shell_count.sum())
    if (
        not np.isfinite(per_carrier_total)
        or per_carrier_total <= float(effect_floor)
        or carrier_total <= 0
    ):
        return None

    radial = shell_mass / total_mass
    per_carrier = mean_per_carrier / per_carrier_total
    opportunity = shell_count / carrier_total
    reconstructed = shell_count * mean_per_carrier
    reconstructed /= float(reconstructed.sum())
    error = float(np.max(np.abs(radial - reconstructed), initial=0.0))
    if error > 1.0e-10:
        print(
            "[reach:warning] shell decomposition reconstruction error "
            f"{error:.3e}; retaining estimate",
            flush=True,
        )
    return {
        "radial_allocation": radial,
        "normalised_per_carrier_sensitivity": per_carrier,
        "shell_opportunity": opportunity,
        "absolute_per_carrier_sensitivity": mean_per_carrier,
    }


def graph_semantic_usage_estimands(
    donor_rows: Sequence[Mapping[str, Any]],
    bamberger_rows: Sequence[Mapping[str, Any]],
    *,
    effect_floor: float,
) -> list[dict[str, Any]]:
    """Build matched semantic range estimands entirely from cached raw rows.

    Radial allocation sums carrier mass within a distance shell before event
    normalisation. Per-carrier sensitivity first averages within each shell and
    then normalises over distance, removing shell cardinality while retaining
    the shape of absolute per-carrier response. Functional events aggregate
    donor -> source -> graph; Bamberger events aggregate output node -> graph.
    """

    semantic_donor_rows = [
        row for row in donor_rows if str(row.get("channel", "")) == "semantic"
    ]
    maximum_by_task: dict[str, int] = defaultdict(int)
    for row in semantic_donor_rows:
        task = str(row["task"])
        maximum_by_task[task] = max(
            maximum_by_task[task],
            _integer(row, "distance"),
        )
    for row in bamberger_rows:
        task = str(row["task"])
        maximum_by_task[task] = max(
            maximum_by_task[task],
            _integer(row, "distance"),
        )

    finite_fields = (
        "task",
        "graph",
        "source",
        "donor_graph",
        "donor_node",
        "draw",
    )
    finite_events: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in semantic_donor_rows:
        finite_events[tuple(row[field] for field in finite_fields)].append(row)

    source_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    active_sources: set[tuple[str, int, int]] = set()
    for key, event_rows in finite_events.items():
        task, graph, source, *_ = key
        task = str(task)
        graph = int(graph)
        source = int(source)
        profiles = _usage_profiles_for_event(
            event_rows,
            value_field="functional_carriage",
            maximum_distance=int(maximum_by_task[task]),
            effect_floor=float(effect_floor),
        )
        if profiles is None:
            continue
        active_sources.add((task, graph, source))
        for estimand, values in profiles.items():
            for distance, value in enumerate(values):
                if (
                    estimand == "absolute_per_carrier_sensitivity"
                    and profiles["shell_opportunity"][distance] <= 0
                ):
                    continue
                source_values[
                    (task, graph, source, estimand, int(distance))
                ].append(float(value))

    graph_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for task, graph, source in active_sources:
        for estimand in (
            *USAGE_ESTIMANDS,
            "shell_opportunity",
            "absolute_per_carrier_sensitivity",
        ):
            for distance in range(int(maximum_by_task[task]) + 1):
                values = source_values.get(
                    (task, graph, source, estimand, distance),
                    [],
                )
                if values:
                    graph_values[
                        (
                            task,
                            graph,
                            "functional_carriage",
                            estimand,
                            distance,
                        )
                    ].append(float(np.mean(values)))

    bamberger_events: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in bamberger_rows:
        bamberger_events[
            (
                str(row["task"]),
                _integer(row, "graph"),
                _integer(row, "output_node"),
            )
        ].append(row)
    for (task, graph, _output_node), event_rows in bamberger_events.items():
        profiles = _usage_profiles_for_event(
            event_rows,
            value_field="influence",
            maximum_distance=int(maximum_by_task[task]),
            effect_floor=float(effect_floor),
        )
        if profiles is None:
            continue
        for estimand, values in profiles.items():
            for distance, value in enumerate(values):
                if (
                    estimand == "absolute_per_carrier_sensitivity"
                    and profiles["shell_opportunity"][distance] <= 0
                ):
                    continue
                graph_values[
                    (task, graph, "bamberger", estimand, int(distance))
                ].append(float(value))

    output: list[dict[str, Any]] = []
    for (task, graph, method, estimand, distance), values in sorted(
        graph_values.items()
    ):
        output.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "graph": int(graph),
                "channel": "semantic",
                "method": method,
                "estimand": estimand,
                "distance": int(distance),
                "value": float(np.mean(values)),
                "centres": int(len(values)),
            }
        )
    return output


def summarise_semantic_usage_estimands(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bootstrap estimand profiles and paired finite-versus-Jacobian gaps."""

    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = _float(row, "value")
        if np.isfinite(value):
            groups[
                (
                    str(row["task"]),
                    str(row["method"]),
                    str(row["estimand"]),
                    _integer(row, "distance"),
                )
            ].append(value)

    profiles: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        task, method, estimand, distance = key
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed)
            + int(stable_hash({"semantic_estimand_profile": key}, length=8), 16),
        )
        profiles.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "method": method,
                "estimand": estimand,
                "distance": int(distance),
                "mean": mean,
                "low": low,
                "high": high,
                "graphs": int(len(values)),
            }
        )

    graph_profiles: dict[tuple[Any, ...], dict[int, float]] = defaultdict(dict)
    for row in rows:
        estimand = str(row["estimand"])
        if estimand not in USAGE_ESTIMANDS:
            continue
        graph_profiles[
            (
                str(row["task"]),
                _integer(row, "graph"),
                str(row["method"]),
                estimand,
            )
        ][_integer(row, "distance")] = _float(row, "value")

    paired_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    tasks = sorted({key[0] for key in graph_profiles})
    for task in tasks:
        for estimand in USAGE_ESTIMANDS:
            finite_graphs = {
                key[1]
                for key in graph_profiles
                if key[0] == task
                and key[2] == "functional_carriage"
                and key[3] == estimand
            }
            bamberger_graphs = {
                key[1]
                for key in graph_profiles
                if key[0] == task
                and key[2] == "bamberger"
                and key[3] == estimand
            }
            for graph in sorted(finite_graphs & bamberger_graphs):
                finite = graph_profiles[
                    (task, graph, "functional_carriage", estimand)
                ]
                bamberger = graph_profiles[(task, graph, "bamberger", estimand)]
                distances = sorted(set(finite) | set(bamberger))
                finite_values = np.asarray(
                    [finite.get(distance, 0.0) for distance in distances],
                    dtype=np.float64,
                )
                bamberger_values = np.asarray(
                    [bamberger.get(distance, 0.0) for distance in distances],
                    dtype=np.float64,
                )
                finite_total = float(finite_values.sum())
                bamberger_total = float(bamberger_values.sum())
                if finite_total <= 0 or bamberger_total <= 0:
                    continue
                finite_values /= finite_total
                bamberger_values /= bamberger_total
                distance_values = np.asarray(distances, dtype=np.float64)
                paired_values[(task, estimand, "profile_tv")].append(
                    float(0.5 * np.abs(finite_values - bamberger_values).sum())
                )
                paired_values[
                    (task, estimand, "expected_distance_difference")
                ].append(
                    float(
                        np.dot(finite_values, distance_values)
                        - np.dot(bamberger_values, distance_values)
                    )
                )

    contrasts: list[dict[str, Any]] = []
    for key, values in sorted(paired_values.items()):
        task, estimand, metric = key
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed)
            + int(stable_hash({"semantic_estimand_contrast": key}, length=8), 16),
        )
        contrasts.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "estimand": estimand,
                "metric": metric,
                "mean": mean,
                "low": low,
                "high": high,
                "paired_graphs": int(len(values)),
                "contrast": "functional_carriage - bamberger",
            }
        )
    return profiles, contrasts


def graph_interpolation_profiles(
    rows: Sequence[Mapping[str, Any]],
    *,
    effect_floor: float,
) -> list[dict[str, Any]]:
    """Normalise each semantic event and aggregate profiles per graph and dose."""

    events: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["task"]),
            _integer(row, "graph"),
            _integer(row, "source"),
            _integer(row, "donor_graph"),
            _integer(row, "donor_node"),
            _integer(row, "draw"),
            _float(row, "interpolation_dose"),
        )
        events[key].append(row)

    source_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    task_max_distance: dict[str, int] = defaultdict(int)
    for key, event_rows in events.items():
        task, graph, source, _donor_graph, _donor_node, _draw, dose = key
        distances = np.asarray(
            [_integer(row, "distance") for row in event_rows],
            dtype=np.int64,
        )
        task_max_distance[task] = max(
            int(task_max_distance[task]),
            int(distances.max(initial=0)),
        )
        masses = np.asarray(
            [_float(row, "functional_carriage") for row in event_rows],
            dtype=np.float64,
        )
        total = float(masses.sum())
        if not np.isfinite(total) or total <= float(effect_floor):
            continue
        for distance in np.unique(distances):
            source_values[(task, graph, source, dose, int(distance))].append(
                float(masses[distances == int(distance)].sum() / total)
            )

    graph_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    present_sources = {key[:4] for key in source_values}
    for task, graph, source, dose in present_sources:
        for distance in range(int(task_max_distance[task]) + 1):
            values = source_values.get(
                (task, graph, source, dose, distance),
                [],
            )
            graph_values[(task, graph, dose, distance)].append(
                float(np.mean(values)) if values else 0.0
            )

    output: list[dict[str, Any]] = []
    for (task, graph, dose, distance), values in graph_values.items():
        output.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "graph": int(graph),
                "interpolation_dose": float(dose),
                "distance": int(distance),
                "mass": float(np.mean(values)),
            }
        )
    return output


def _bootstrap_interval(
    values: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return np.nan, np.nan, np.nan
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(replicates), len(array)))
    draws = array[indices].mean(axis=1)
    return mean, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarise_interpolation_contrasts(
    interpolation_rows: Sequence[Mapping[str, Any]],
    bamberger_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    matched_reference_dose: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare finite profiles with literal and fully matched local references."""

    interpolation_profiles: dict[
        tuple[str, int, float],
        dict[int, float],
    ] = defaultdict(dict)
    for row in interpolation_rows:
        interpolation_profiles[
            (
                str(row["task"]),
                _integer(row, "graph"),
                _float(row, "interpolation_dose"),
            )
        ][_integer(row, "distance")] = _float(row, "mass")

    available_doses = sorted({key[2] for key in interpolation_profiles})
    if not available_doses:
        raise RuntimeError("interpolation profiles are empty")
    reference_dose = (
        float(available_doses[0])
        if matched_reference_dose is None
        else float(matched_reference_dose)
    )
    if not any(np.isclose(reference_dose, dose) for dose in available_doses):
        raise ValueError(
            f"matched reference dose {reference_dose:g} is absent from "
            f"{available_doses}"
        )

    bamberger_profiles: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    for row in bamberger_rows:
        bamberger_profiles[
            (str(row["task"]), _integer(row, "graph"))
        ][_integer(row, "distance")] = _float(row, "mass")

    graph_contrasts: list[dict[str, Any]] = []
    for (task, graph, dose), finite_profile in sorted(interpolation_profiles.items()):
        bamberger_profile = bamberger_profiles.get((task, graph))
        matched_profile = interpolation_profiles.get((task, graph, reference_dose))
        references = (
            ("bamberger", bamberger_profile),
            ("matched_small_dose", matched_profile),
        )
        for baseline, reference_profile in references:
            if not reference_profile:
                continue
            distances = sorted(set(finite_profile) | set(reference_profile))
            finite = np.asarray(
                [finite_profile.get(distance, 0.0) for distance in distances],
                dtype=np.float64,
            )
            reference = np.asarray(
                [reference_profile.get(distance, 0.0) for distance in distances],
                dtype=np.float64,
            )
            graph_contrasts.append(
                {
                    "task": task,
                    "model_label": TASK_LABELS[task],
                    "graph": int(graph),
                    "interpolation_dose": float(dose),
                    "baseline": baseline,
                    "matched_reference_dose": float(reference_dose),
                    "profile_tv": float(0.5 * np.abs(finite - reference).sum()),
                    "expected_distance_difference": float(
                        sum(
                            distance * value
                            for distance, value in zip(distances, finite)
                        )
                        - sum(
                            distance * value
                            for distance, value in zip(distances, reference)
                        )
                    ),
                }
            )

    grouped: dict[tuple[str, float, str, str], list[float]] = defaultdict(list)
    for row in graph_contrasts:
        for metric in ("profile_tv", "expected_distance_difference"):
            grouped[
                (
                    str(row["task"]),
                    _float(row, "interpolation_dose"),
                    str(row["baseline"]),
                    metric,
                )
            ].append(_float(row, metric))

    summary: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        task, dose, baseline, metric = key
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed)
            + int(stable_hash({"interpolation": key}, length=8), 16),
        )
        summary.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "interpolation_dose": float(dose),
                "baseline": baseline,
                "matched_reference_dose": float(reference_dose),
                "metric": metric,
                "mean": mean,
                "low": low,
                "high": high,
                "graphs": len(values),
            }
        )
    return graph_contrasts, summary


def graph_profile_trajectory_decomposition(
    interpolation_rows: Sequence[Mapping[str, Any]],
    bamberger_rows: Sequence[Mapping[str, Any]],
    *,
    effect_floor: float,
    small_dose: float | None = None,
    full_dose: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Decompose the finite endpoint gap into local mismatch and finite drift.

    For every paired graph, ``p_full - p_J`` is written exactly as
    ``(p_small - p_J) + (p_full - p_small)``.  This distinguishes disagreement
    already present near the clean input from subsequent finite-dose movement.
    """

    interpolation_profiles: dict[
        tuple[str, int, float],
        dict[int, float],
    ] = defaultdict(dict)
    for row in interpolation_rows:
        interpolation_profiles[
            (
                str(row["task"]),
                _integer(row, "graph"),
                _float(row, "interpolation_dose"),
            )
        ][_integer(row, "distance")] = _float(row, "mass")

    bamberger_profiles: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    for row in bamberger_rows:
        bamberger_profiles[
            (str(row["task"]), _integer(row, "graph"))
        ][_integer(row, "distance")] = _float(row, "mass")

    doses = sorted({key[2] for key in interpolation_profiles})
    if not doses:
        print(
            "[reach:warning] profile trajectory decomposition has no finite "
            "interpolation profiles",
            flush=True,
        )
        return [], []
    small = float(doses[0] if small_dose is None else small_dose)
    full = float(doses[-1] if full_dose is None else full_dose)
    if not any(np.isclose(small, dose) for dose in doses):
        print(
            f"[reach:warning] requested trajectory small dose {small:g} is "
            f"absent from {doses}; no decomposition produced",
            flush=True,
        )
        return [], []
    if not any(np.isclose(full, dose) for dose in doses):
        print(
            f"[reach:warning] requested trajectory full dose {full:g} is "
            f"absent from {doses}; no decomposition produced",
            flush=True,
        )
        return [], []
    small = min(doses, key=lambda dose: abs(dose - small))
    full = min(doses, key=lambda dose: abs(dose - full))

    graph_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    candidate_graphs = sorted({key[:2] for key in interpolation_profiles})
    missing = 0
    maximum_identity_error = 0.0
    for task, graph in candidate_graphs:
        jacobian_profile = bamberger_profiles.get((task, graph))
        small_profile = interpolation_profiles.get((task, graph, small))
        full_profile = interpolation_profiles.get((task, graph, full))
        if not jacobian_profile or not small_profile or not full_profile:
            missing += 1
            continue
        distances = sorted(
            set(jacobian_profile) | set(small_profile) | set(full_profile)
        )
        jacobian = np.asarray(
            [jacobian_profile.get(distance, 0.0) for distance in distances],
            dtype=np.float64,
        )
        finite_small = np.asarray(
            [small_profile.get(distance, 0.0) for distance in distances],
            dtype=np.float64,
        )
        finite_full = np.asarray(
            [full_profile.get(distance, 0.0) for distance in distances],
            dtype=np.float64,
        )
        totals = tuple(float(values.sum()) for values in (jacobian, finite_small, finite_full))
        if any(
            not np.isfinite(total) or total <= float(effect_floor)
            for total in totals
        ):
            missing += 1
            continue
        jacobian /= totals[0]
        finite_small /= totals[1]
        finite_full /= totals[2]

        baseline = finite_small - jacobian
        drift = finite_full - finite_small
        endpoint = finite_full - jacobian
        identity_error = float(np.max(np.abs(endpoint - baseline - drift)))
        maximum_identity_error = max(maximum_identity_error, identity_error)
        baseline_tv = float(0.5 * np.abs(baseline).sum())
        drift_tv = float(0.5 * np.abs(drift).sum())
        endpoint_tv = float(0.5 * np.abs(endpoint).sum())
        tv_path_length = baseline_tv + drift_tv
        cancellation = (
            float(np.clip(1.0 - endpoint_tv / tv_path_length, 0.0, 1.0))
            if tv_path_length > float(effect_floor)
            else np.nan
        )
        baseline_norm = float(np.linalg.vector_norm(baseline))
        drift_norm = float(np.linalg.vector_norm(drift))
        cosine = (
            float(np.dot(baseline, drift) / (baseline_norm * drift_norm))
            if baseline_norm > float(effect_floor)
            and drift_norm > float(effect_floor)
            else np.nan
        )
        distance_values = np.asarray(distances, dtype=np.float64)
        graph_rows.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "graph": int(graph),
                "small_dose": float(small),
                "full_dose": float(full),
                "baseline_tv": baseline_tv,
                "finite_drift_tv": drift_tv,
                "endpoint_tv": endpoint_tv,
                "discrepancy_cancellation": cancellation,
                "directional_cosine": cosine,
                "baseline_expected_distance": float(
                    np.dot(distance_values, baseline)
                ),
                "finite_drift_expected_distance": float(
                    np.dot(distance_values, drift)
                ),
                "endpoint_expected_distance": float(
                    np.dot(distance_values, endpoint)
                ),
                "identity_error": identity_error,
            }
        )
        for index, distance in enumerate(distances):
            for component, values in (
                ("baseline_mismatch", baseline),
                ("finite_drift", drift),
                ("endpoint_gap", endpoint),
            ):
                distance_rows.append(
                    {
                        "task": task,
                        "model_label": TASK_LABELS[task],
                        "graph": int(graph),
                        "small_dose": float(small),
                        "full_dose": float(full),
                        "component": component,
                        "distance": int(distance),
                        "value": float(values[index]),
                    }
                )

    if missing:
        print(
            f"[reach:warning] profile trajectory decomposition skipped {missing}/"
            f"{len(candidate_graphs)} graph-model pairs with incomplete profiles",
            flush=True,
        )
    if maximum_identity_error > 1.0e-10:
        print(
            "[reach:warning] profile trajectory identity residual reached "
            f"{maximum_identity_error:.3e}; retaining estimates",
            flush=True,
        )
    return graph_rows, distance_rows


def summarise_profile_trajectory_decomposition(
    graph_rows: Sequence[Mapping[str, Any]],
    distance_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bootstrap scalar and distance-wise trajectory components by graph."""

    scalar_metrics = (
        "baseline_tv",
        "finite_drift_tv",
        "endpoint_tv",
        "discrepancy_cancellation",
        "directional_cosine",
        "baseline_expected_distance",
        "finite_drift_expected_distance",
        "endpoint_expected_distance",
    )
    scalar_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in graph_rows:
        for metric in scalar_metrics:
            value = _float(row, metric)
            if np.isfinite(value):
                scalar_groups[(str(row["task"]), metric)].append(value)

    scalar_summary: list[dict[str, Any]] = []
    for key, values in sorted(scalar_groups.items()):
        task, metric = key
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed)
            + int(stable_hash({"trajectory_scalar": key}, length=8), 16),
        )
        example = next(row for row in graph_rows if str(row["task"]) == task)
        scalar_summary.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "small_dose": _float(example, "small_dose"),
                "full_dose": _float(example, "full_dose"),
                "metric": metric,
                "mean": mean,
                "low": low,
                "high": high,
                "graphs": int(len(values)),
            }
        )

    distance_groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in distance_rows:
        value = _float(row, "value")
        if np.isfinite(value):
            distance_groups[
                (
                    str(row["task"]),
                    str(row["component"]),
                    _integer(row, "distance"),
                )
            ].append(value)

    distance_summary: list[dict[str, Any]] = []
    for key, values in sorted(distance_groups.items()):
        task, component, distance = key
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed)
            + int(stable_hash({"trajectory_distance": key}, length=8), 16),
        )
        distance_summary.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "component": component,
                "distance": int(distance),
                "mean": mean,
                "low": low,
                "high": high,
                "graphs": int(len(values)),
            }
        )
    return scalar_summary, distance_summary


def summarise_graph_profiles(
    graph_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarise profiles and their expected distances across held-out graphs."""

    profile_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    by_graph: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in graph_rows:
        profile_groups[
            (
                str(row["task"]),
                str(row["channel"]),
                str(row["method"]),
                _integer(row, "distance"),
            )
        ].append(_float(row, "mass"))
        by_graph[
            (
                str(row["task"]),
                _integer(row, "graph"),
                str(row["channel"]),
                str(row["method"]),
            )
        ].append(row)

    profiles: list[dict[str, Any]] = []
    for key, values in profile_groups.items():
        task, channel, method, distance = key
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed)
            + int(stable_hash({"profile": key}, length=8), 16),
        )
        profiles.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "channel": channel,
                "method": method,
                "distance": int(distance),
                "mean": mean,
                "low": low,
                "high": high,
                "graphs": int(len(values)),
            }
        )

    expected_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for (task, _graph, channel, method), values in by_graph.items():
        total = sum(_float(row, "mass") for row in values)
        if total <= 0:
            continue
        expected_groups[(task, channel, method)].append(
            sum(
                _float(row, "mass") * _integer(row, "distance")
                for row in values
            )
            / total
        )
    expected: list[dict[str, Any]] = []
    for key, values in expected_groups.items():
        task, channel, method = key
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed)
            + int(stable_hash({"expected": key}, length=8), 16),
        )
        expected.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "channel": channel,
                "method": method,
                "mean": mean,
                "low": low,
                "high": high,
                "graphs": int(len(values)),
            }
        )
    return profiles, expected


def _adaptive_scale_bins(
    values_by_graph: Mapping[int, int],
) -> tuple[dict[int, int], dict[int, str], dict[int, float], dict[int, int]]:
    """Merge adjacent integer values toward equal-count, density-adaptive bins."""

    frequencies: dict[int, int] = defaultdict(int)
    for value in values_by_graph.values():
        frequencies[int(value)] += 1
    if not frequencies:
        raise ValueError("cannot bin an empty scale descriptor")
    observations = len(values_by_graph)
    maximum_bins = min(len(frequencies), min(12, max(4, round(np.sqrt(observations)))))
    provisional: dict[int, list[int]] = defaultdict(list)
    cumulative = 0
    for value, count in sorted(frequencies.items()):
        midpoint = cumulative + 0.5 * int(count)
        quantile_bin = min(
            maximum_bins - 1,
            int(midpoint * maximum_bins / observations),
        )
        provisional[quantile_bin].append(int(value))
        cumulative += int(count)
    grouped_values = list(provisional.values())

    value_to_bin = {
        value: bin_index
        for bin_index, values in enumerate(grouped_values)
        for value in values
    }
    graph_bins = {
        int(graph): int(value_to_bin[int(value)])
        for graph, value in values_by_graph.items()
    }
    labels: dict[int, str] = {}
    means: dict[int, float] = {}
    counts: dict[int, int] = {}
    for bin_index, bin_values in enumerate(grouped_values):
        value_set = set(bin_values)
        members = [
            int(value)
            for value in values_by_graph.values()
            if int(value) in value_set
        ]
        labels[bin_index] = (
            str(min(bin_values))
            if min(bin_values) == max(bin_values)
            else f"{min(bin_values)}–{max(bin_values)}"
        )
        means[bin_index] = float(np.mean(members))
        counts[bin_index] = int(len(members))
    return graph_bins, labels, means, counts


def _bootstrap_slope(
    x: Sequence[float],
    y: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    """OLS slope with a paired graph bootstrap, evaluated in bounded chunks."""

    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if len(x_array) != len(y_array) or len(x_array) < 2:
        return np.nan, np.nan, np.nan

    def slope(x_value: np.ndarray, y_value: np.ndarray) -> np.ndarray:
        centered_x = x_value - x_value.mean(axis=-1, keepdims=True)
        centered_y = y_value - y_value.mean(axis=-1, keepdims=True)
        denominator = np.square(centered_x).sum(axis=-1)
        numerator = (centered_x * centered_y).sum(axis=-1)
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=np.float64),
            where=denominator > 0,
        )

    estimate = float(slope(x_array[None, :], y_array[None, :])[0])
    rng = np.random.default_rng(int(seed))
    draws: list[np.ndarray] = []
    chunk_size = 128
    for start in range(0, int(replicates), chunk_size):
        count = min(chunk_size, int(replicates) - start)
        indices = rng.integers(0, len(x_array), size=(count, len(x_array)))
        draws.append(slope(x_array[indices], y_array[indices]))
    samples = np.concatenate(draws)
    samples = samples[np.isfinite(samples)]
    if not samples.size:
        return estimate, np.nan, np.nan
    return estimate, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarise_scale_dependence(
    graph_rows: Sequence[Mapping[str, Any]],
    graph_records: Sequence[Mapping[str, Any]],
    full_test_records: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    reference_task: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarise full-test performance and sampled carriage on separate supports."""

    sampled_metadata = {
        (str(row["task"]), _integer(row, "graph")): row
        for row in graph_records
    }
    sampled_reference = {
        _integer(row, "graph"): row
        for row in graph_records
        if str(row["task"]) == reference_task
    }
    test_metadata = {
        (str(row["task"]), _integer(row, "test_graph")): row
        for row in full_test_records
    }
    test_reference = {
        _integer(row, "test_graph"): row
        for row in full_test_records
        if str(row["task"]) == reference_task
    }
    if not sampled_reference or not test_reference:
        raise RuntimeError("scale analysis is missing sampled or full-test reference rows")

    reach_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in graph_rows:
        if (
            str(row["channel"]) == "semantic"
            and str(row["method"]) == "functional_carriage"
        ):
            reach_groups[(str(row["task"]), _integer(row, "graph"))].append(row)
    expected_reach: dict[tuple[str, int], float] = {}
    for key, values in reach_groups.items():
        total = sum(_float(row, "mass") for row in values)
        if total > 0:
            expected_reach[key] = float(
                sum(
                    _float(row, "mass") * _integer(row, "distance")
                    for row in values
                )
                / total
            )

    joined: list[dict[str, Any]] = []
    for descriptor in ("num_nodes", "diameter"):
        for analysis, reference_rows in (
            ("performance", test_reference),
            ("reach", sampled_reference),
        ):
            descriptor_values = {
                graph: _integer(row, descriptor)
                for graph, row in reference_rows.items()
            }
            graph_bins, labels, means, counts = _adaptive_scale_bins(
                descriptor_values
            )
            for task in tasks:
                for graph, reference_row in reference_rows.items():
                    row = (
                        test_metadata.get((task, graph))
                        if analysis == "performance"
                        else sampled_metadata.get((task, graph))
                    )
                    if row is None:
                        continue
                    if (
                        _integer(row, "num_nodes")
                        != _integer(reference_row, "num_nodes")
                        or _integer(row, "diameter")
                        != _integer(reference_row, "diameter")
                    ):
                        raise RuntimeError(
                            "paired model records disagree on molecular graph geometry"
                        )
                    if analysis == "performance" and not np.isclose(
                        _float(row, "target"),
                        _float(reference_row, "target"),
                        rtol=0,
                        atol=1.0e-7,
                    ):
                        raise RuntimeError("paired full-test targets differ across models")
                    reach = expected_reach.get((task, graph))
                    if analysis == "reach" and reach is None:
                        continue
                    bin_index = graph_bins[graph]
                    output = {
                        "analysis": analysis,
                        "task": task,
                        "model_label": TASK_LABELS[task],
                        "reference_task": reference_task,
                        "graph": int(graph),
                        "descriptor": descriptor,
                        "descriptor_value": int(descriptor_values[graph]),
                        "scale_bin": int(bin_index),
                        "scale_label": labels[bin_index],
                        "bin_descriptor_mean": means[bin_index],
                        "bin_graphs": counts[bin_index],
                    }
                    if analysis == "performance":
                        output.update(
                            {
                                "graph_mae": _float(row, "graph_mae"),
                                "reference_graph_mae": _float(
                                    reference_row, "graph_mae"
                                ),
                                "mae_difference_from_reference": (
                                    _float(row, "graph_mae")
                                    - _float(reference_row, "graph_mae")
                                ),
                            }
                        )
                    else:
                        output["functional_expected_distance"] = float(reach)
                    joined.append(output)

    grouped: dict[tuple[str, str, int, str, str], list[float]] = defaultdict(list)
    group_meta: dict[tuple[str, str, int, str, str], Mapping[str, Any]] = {}
    for row in joined:
        metrics = (
            ("graph_mae", "mae_difference_from_reference")
            if str(row["analysis"]) == "performance"
            else ("functional_expected_distance",)
        )
        for metric in metrics:
            key = (
                str(row["analysis"]),
                str(row["descriptor"]),
                _integer(row, "scale_bin"),
                str(row["task"]),
                metric,
            )
            grouped[key].append(_float(row, metric))
            group_meta[key] = row

    summary: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        analysis, descriptor, bin_index, task, metric = key
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed)
            + int(stable_hash({"scale": key}, length=8), 16),
        )
        exemplar = group_meta[key]
        summary.append(
            {
                "analysis": analysis,
                "task": task,
                "model_label": TASK_LABELS[task],
                "reference_task": reference_task,
                "descriptor": descriptor,
                "scale_bin": int(bin_index),
                "scale_label": str(exemplar["scale_label"]),
                "bin_descriptor_mean": _float(exemplar, "bin_descriptor_mean"),
                "metric": metric,
                "mean": mean,
                "low": low,
                "high": high,
                "graphs": int(len(values)),
            }
        )

    trend_rows: list[dict[str, Any]] = []
    trend_metrics = {
        "performance": "mae_difference_from_reference",
        "reach": "functional_expected_distance",
    }
    for analysis, metric in trend_metrics.items():
        for descriptor in ("num_nodes", "diameter"):
            for task in tasks:
                values = [
                    row
                    for row in joined
                    if str(row["analysis"]) == analysis
                    and str(row["descriptor"]) == descriptor
                    and str(row["task"]) == task
                ]
                slope, low, high = _bootstrap_slope(
                    [_float(row, "descriptor_value") for row in values],
                    [_float(row, metric) for row in values],
                    replicates=int(bootstrap_replicates),
                    seed=int(bootstrap_seed)
                    + int(
                        stable_hash(
                            {"scale_slope": (analysis, descriptor, task, metric)},
                            length=8,
                        ),
                        16,
                    ),
                )
                trend_rows.append(
                    {
                        "analysis": analysis,
                        "task": task,
                        "model_label": TASK_LABELS[task],
                        "reference_task": reference_task,
                        "descriptor": descriptor,
                        "metric": metric,
                        "slope": slope,
                        "low": low,
                        "high": high,
                        "graphs": int(len(values)),
                    }
                )
    return joined, summary, trend_rows


def summarise_dense_profile_contrasts(
    graph_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    reference_task: str = "zinc",
) -> list[dict[str, Any]]:
    """Compute paired graph-level profile differences from a reference model."""

    profiles: dict[tuple[str, int, str, str], dict[int, float]] = defaultdict(dict)
    for row in graph_rows:
        profiles[
            (
                str(row["task"]),
                _integer(row, "graph"),
                str(row["channel"]),
                str(row["method"]),
            )
        ][_integer(row, "distance")] = _float(row, "mass")

    combinations = sorted(
        {
            (str(row["task"]), str(row["channel"]), str(row["method"]))
            for row in graph_rows
            if str(row["task"]) != reference_task
        }
    )
    contrasts: list[dict[str, Any]] = []
    for task, channel, method in combinations:
        model_graphs = {
            graph
            for candidate_task, graph, candidate_channel, candidate_method in profiles
            if candidate_task == task
            and candidate_channel == channel
            and candidate_method == method
        }
        reference_graphs = {
            graph
            for candidate_task, graph, candidate_channel, candidate_method in profiles
            if candidate_task == reference_task
            and candidate_channel == channel
            and candidate_method == method
        }
        paired_graphs = sorted(model_graphs & reference_graphs)
        if not paired_graphs:
            continue
        distances = sorted(
            {
                distance
                for graph in paired_graphs
                for profile in (
                    profiles[(task, graph, channel, method)],
                    profiles[(reference_task, graph, channel, method)],
                )
                for distance in profile
            }
        )
        for distance in distances:
            values = [
                profiles[(task, graph, channel, method)].get(distance, 0.0)
                - profiles[
                    (reference_task, graph, channel, method)
                ].get(distance, 0.0)
                for graph in paired_graphs
            ]
            mean, low, high = _bootstrap_interval(
                values,
                replicates=int(bootstrap_replicates),
                seed=int(bootstrap_seed)
                + int(
                    stable_hash(
                        {
                            "dense_profile_contrast": (
                                task,
                                channel,
                                method,
                                distance,
                            )
                        },
                        length=8,
                    ),
                    16,
                ),
            )
            contrasts.append(
                {
                    "task": task,
                    "model_label": TASK_LABELS[task],
                    "reference_task": reference_task,
                    "reference_model_label": TASK_LABELS[reference_task],
                    "channel": channel,
                    "method": method,
                    "distance": int(distance),
                    "mean": mean,
                    "low": low,
                    "high": high,
                    "paired_graphs": int(len(values)),
                }
            )
    return contrasts


def summarise_beneficial_carriage(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate signed B donor -> source -> graph, retaining task-loss units."""

    events: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    task_max_distance: dict[str, int] = defaultdict(int)
    for row in rows:
        key = (
            str(row["task"]),
            _integer(row, "graph"),
            str(row["channel"]),
            _integer(row, "source"),
            _integer(row, "donor_graph"),
            _integer(row, "donor_node"),
            _integer(row, "draw"),
        )
        events[key].append(row)
        task_max_distance[key[0]] = max(
            task_max_distance[key[0]], _integer(row, "distance")
        )

    source_profiles: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    source_far: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for key, event_rows in events.items():
        task, graph, channel, source, *_ = key
        distance = np.asarray(
            [_integer(row, "distance") for row in event_rows], dtype=np.int64
        )
        values = np.asarray(
            [_float(row, "beneficial_carriage") for row in event_rows],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            continue
        for current in range(task_max_distance[task] + 1):
            source_profiles[(task, graph, channel, source, current)].append(
                float(values[distance == current].sum())
            )
        for threshold in range(task_max_distance[task]):
            source_far[(task, graph, channel, source, threshold)].append(
                float(values[distance > threshold].sum())
            )

    graph_rows: list[dict[str, Any]] = []
    for kind, container in (("distance", source_profiles), ("far", source_far)):
        graph_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        for (task, graph, channel, _source, index), donor_values in container.items():
            graph_values[(task, graph, channel, index)].append(
                float(np.mean(donor_values))
            )
        for (task, graph, channel, index), source_values in sorted(graph_values.items()):
            graph_rows.append(
                {
                    "task": task,
                    "model_label": TASK_LABELS[task],
                    "graph": int(graph),
                    "channel": channel,
                    "estimand": kind,
                    "index": int(index),
                    "beneficial_carriage": float(np.mean(source_values)),
                    "sampled_sources": int(len(source_values)),
                }
            )

    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in graph_rows:
        grouped[
            (
                str(row["task"]),
                str(row["channel"]),
                str(row["estimand"]),
                _integer(row, "index"),
            )
        ].append(_float(row, "beneficial_carriage"))
    summary: list[dict[str, Any]] = []
    for (task, channel, estimand, index), values in sorted(grouped.items()):
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(
                stable_hash(
                    {
                        "seed": int(bootstrap_seed),
                        "analysis": "beneficial",
                        "task": task,
                        "channel": channel,
                        "estimand": estimand,
                        "index": index,
                    },
                    length=8,
                ),
                16,
            ),
        )
        summary.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "channel": channel,
                "estimand": estimand,
                "index": int(index),
                "mean": mean,
                "low": low,
                "high": high,
                "graphs": int(len(values)),
            }
        )
    return graph_rows, summary


def summarise_shell_survival(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Average draws/carriers inside graph, then bootstrap held-out graphs."""

    metrics = (
        "carrier_survival",
        "carrier_additive_survival",
        "carrier_nonlinear_residual",
        "output_survival",
        "output_additive_survival",
        "output_nonlinear_residual",
    )
    graph_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    graph_sizes: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    total: dict[tuple[Any, ...], int] = defaultdict(int)
    for row in rows:
        base_key = (
            str(row["task"]),
            _integer(row, "graph"),
            str(row["intervention"]),
            str(row["condition"]),
            _integer(row, "radius"),
        )
        total[base_key] += 1
        if _integer(row, "coalition_size") < 1:
            continue
        graph_sizes[base_key].append(_integer(row, "coalition_size"))
        for metric in metrics:
            if metric not in row or row[metric] == "":
                continue
            value = _float(row, metric)
            if np.isfinite(value):
                graph_values[(*base_key, metric)].append(value)

    graph_rows: list[dict[str, Any]] = []
    for key, values in sorted(graph_values.items()):
        task, graph, intervention, condition, radius, metric = key
        base_key = key[:-1]
        sizes = graph_sizes[base_key]
        graph_rows.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "graph": int(graph),
                "intervention": intervention,
                "condition": condition,
                "radius": int(radius),
                "metric": metric,
                "value": float(np.mean(values)),
                "coalitions": int(len(values)),
                "mean_coalition_size": float(np.mean(sizes)),
                "estimable_fraction": float(
                    len(values) / max(1, total[base_key])
                ),
            }
        )

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in graph_rows:
        grouped[
            (
                str(row["task"]),
                str(row["intervention"]),
                str(row["condition"]),
                _integer(row, "radius"),
                str(row["metric"]),
            )
        ].append(row)
    summary: list[dict[str, Any]] = []
    for (task, intervention, condition, radius, metric), group in sorted(grouped.items()):
        values = [_float(row, "value") for row in group]
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(
                stable_hash(
                    {
                        "seed": int(bootstrap_seed),
                        "analysis": "survival",
                        "task": task,
                        "intervention": intervention,
                        "condition": condition,
                        "radius": radius,
                        "metric": metric,
                    },
                    length=8,
                ),
                16,
            ),
        )
        summary.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "intervention": intervention,
                "condition": condition,
                "radius": int(radius),
                "metric": metric,
                "mean": mean,
                "low": low,
                "high": high,
                "graphs": int(len(values)),
                "mean_coalition_size": float(
                    np.mean([_float(row, "mean_coalition_size") for row in group])
                ),
                "mean_estimable_fraction": float(
                    np.mean([_float(row, "estimable_fraction") for row in group])
                ),
            }
        )

    # Paired control differences use the exact same graph/carrier/draw/condition.
    paired: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if _integer(row, "coalition_size") < 1:
            continue
        key = (
            str(row["task"]),
            _integer(row, "graph"),
            _integer(row, "carrier"),
            _integer(row, "draw"),
            str(row["condition"]),
            _integer(row, "radius"),
        )
        paired[key][str(row["intervention"])] = row
    graph_differences: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for key, pair in paired.items():
        if not {"shell_permutation", "shell_replacement"}.issubset(pair):
            continue
        task, graph, _carrier, _draw, condition, radius = key
        for metric in metrics:
            left = pair["shell_permutation"].get(metric, "")
            right = pair["shell_replacement"].get(metric, "")
            if left == "" or right == "":
                continue
            difference = float(right) - float(left)
            if np.isfinite(difference):
                graph_differences[(task, graph, condition, radius, metric)].append(
                    difference
                )
    contrast_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for (task, _graph, condition, radius, metric), values in graph_differences.items():
        contrast_groups[(task, condition, radius, metric)].append(float(np.mean(values)))
    contrasts: list[dict[str, Any]] = []
    for (task, condition, radius, metric), values in sorted(contrast_groups.items()):
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(
                stable_hash(
                    {
                        "seed": int(bootstrap_seed),
                        "analysis": "survival_control_contrast",
                        "task": task,
                        "condition": condition,
                        "radius": radius,
                        "metric": metric,
                    },
                    length=8,
                ),
                16,
            ),
        )
        contrasts.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "condition": condition,
                "radius": int(radius),
                "metric": metric,
                "contrast": "shell_replacement - shell_permutation",
                "mean": mean,
                "low": low,
                "high": high,
                "paired_graphs": int(len(values)),
            }
        )
    return graph_rows, summary, contrasts


def summarise_shell_survival_by_size(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Sensitivity table stratified by realised nonzero coalition size."""

    metrics = (
        "carrier_survival",
        "carrier_additive_survival",
        "carrier_nonlinear_residual",
        "output_survival",
    )
    within_graph: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        size = _integer(row, "coalition_size")
        if size < 1:
            continue
        for metric in metrics:
            if metric not in row or row[metric] == "":
                continue
            value = _float(row, metric)
            if not np.isfinite(value):
                continue
            within_graph[
                (
                    str(row["task"]),
                    _integer(row, "graph"),
                    str(row["intervention"]),
                    str(row["condition"]),
                    _integer(row, "radius"),
                    int(size),
                    metric,
                )
            ].append(value)
    across_graphs: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for key, values in within_graph.items():
        task, _graph, intervention, condition, radius, size, metric = key
        across_graphs[(task, intervention, condition, radius, size, metric)].append(
            float(np.mean(values))
        )
    output: list[dict[str, Any]] = []
    for (task, intervention, condition, radius, size, metric), values in sorted(
        across_graphs.items()
    ):
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(
                stable_hash(
                    {
                        "seed": int(bootstrap_seed),
                        "analysis": "survival_by_coalition_size",
                        "task": task,
                        "intervention": intervention,
                        "condition": condition,
                        "radius": radius,
                        "coalition_size": size,
                        "metric": metric,
                    },
                    length=8,
                ),
                16,
            ),
        )
        output.append(
            {
                "task": task,
                "model_label": TASK_LABELS[task],
                "intervention": intervention,
                "condition": condition,
                "radius": int(radius),
                "coalition_size": int(size),
                "metric": metric,
                "mean": mean,
                "low": low,
                "high": high,
                "graphs": int(len(values)),
            }
        )
    return output


def _figure_theme() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.7,
            "savefig.dpi": 300,
        }
    )


def _save_figure(fig: Any, figures_dir: Path, name: str) -> dict[str, str]:
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": str(figures_dir / f"{name}.png"),
        "pdf": str(figures_dir / f"{name}.pdf"),
    }
    fig.savefig(paths["png"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def plot_beneficial_carriage(
    rows: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
    tasks: Sequence[str],
    dataset_label: str,
    figure_prefix: str,
) -> dict[str, str]:
    """Signed B(d) and additive B_far(k), in native task-loss units."""

    import matplotlib.pyplot as plt

    _figure_theme()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex="col")
    for row_index, channel in enumerate(CHANNELS):
        for column_index, (estimand, xlabel, ylabel) in enumerate(
            (
                ("distance", "Source–carrier distance d", "Mean B(d) per source"),
                ("far", "Threshold k", r"Additive $B_{far}(k)$ per source"),
            )
        ):
            axis = axes[row_index, column_index]
            plotted = False
            magnitude: list[float] = []
            for task in tasks:
                values = sorted(
                    (
                        row
                        for row in rows
                        if str(row["task"]) == task
                        and str(row["channel"]) == channel
                        and str(row["estimand"]) == estimand
                    ),
                    key=lambda row: _integer(row, "index"),
                )
                if not values:
                    continue
                x = np.asarray([_integer(row, "index") for row in values])
                mean = np.asarray([_float(row, "mean") for row in values])
                low = np.asarray([_float(row, "low") for row in values])
                high = np.asarray([_float(row, "high") for row in values])
                finite = np.isfinite(mean) & np.isfinite(low) & np.isfinite(high)
                if not finite.any():
                    continue
                plotted = True
                magnitude.extend(np.abs(np.concatenate((mean[finite], low[finite], high[finite]))))
                axis.plot(
                    x[finite],
                    mean[finite],
                    color=MODEL_COLOURS[task],
                    marker=MODEL_MARKERS[task],
                    linestyle=MODEL_LINESTYLES[task],
                    linewidth=1.7,
                    markersize=4.3,
                    label=TASK_LABELS[task],
                )
                axis.fill_between(
                    x[finite], low[finite], high[finite],
                    color=MODEL_COLOURS[task], alpha=0.13, linewidth=0,
                )
            axis.axhline(0.0, color="#555555", linewidth=0.9, zorder=0)
            finite_magnitude = [value for value in magnitude if np.isfinite(value) and value > 0]
            if finite_magnitude:
                axis.set_yscale(
                    "symlog",
                    linthresh=max(1.0e-9, float(np.quantile(finite_magnitude, 0.25)) * 0.2),
                )
            if not plotted:
                axis.text(0.5, 0.5, "No estimable paths", transform=axis.transAxes,
                          ha="center", va="center", color="#666666")
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            if row_index == 0:
                axis.set_title("Distance profile" if estimand == "distance" else "Far-distance total")
            if column_index == 0:
                axis.text(
                    -0.22,
                    0.5,
                    channel.capitalize(),
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    fontweight="bold",
                )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
                   ncol=min(len(handles), 5), frameon=False)
    fig.suptitle(
        f"Beneficial carriage on {dataset_label}: where information improves task loss",
        fontsize=13,
        y=0.99,
    )
    fig.text(
        0.5,
        0.885,
        "Positive = intervention increases loss; donor → source → graph; 95% graph bootstrap",
        ha="center",
        color="#666666",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.09, top=0.81,
                        wspace=0.28, hspace=0.32)
    return _save_figure(fig, figures_dir, f"{figure_prefix}_beneficial_carriage")


def plot_shell_survival(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    figures_dir: Path,
    tasks: Sequence[str],
    dataset_label: str,
    figure_prefix: str,
) -> dict[str, str]:
    """Plot the complete R=C/A+(J-C)/A carrier-level decomposition."""

    import matplotlib.pyplot as plt

    if condition not in {"exact_shell", "far_tail"}:
        raise ValueError(f"unknown survival condition {condition!r}")
    _figure_theme()
    interventions = ("shell_permutation", "shell_replacement")
    metric_specs = (
        ("carrier_survival", r"Joint survival $R=J/A$", 1.0),
        ("carrier_additive_survival", r"Additive survival $C/A$", 1.0),
        ("carrier_nonlinear_residual", r"Nonlinear residual $(J-C)/A$", 0.0),
    )
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 8.3), sharex=True)
    for row_index, (metric, ylabel, reference) in enumerate(metric_specs):
        for column_index, intervention in enumerate(interventions):
            axis = axes[row_index, column_index]
            plotted = False
            for task in tasks:
                values = sorted(
                    (
                        row
                        for row in rows
                        if str(row["task"]) == task
                        and str(row["condition"]) == condition
                        and str(row["intervention"]) == intervention
                        and str(row["metric"]) == metric
                    ),
                    key=lambda row: _integer(row, "radius"),
                )
                if not values:
                    continue
                x = np.asarray([_integer(row, "radius") for row in values])
                mean = np.asarray([_float(row, "mean") for row in values])
                low = np.asarray([_float(row, "low") for row in values])
                high = np.asarray([_float(row, "high") for row in values])
                finite = np.isfinite(mean) & np.isfinite(low) & np.isfinite(high)
                if not finite.any():
                    continue
                plotted = True
                axis.plot(
                    x[finite], mean[finite],
                    color=MODEL_COLOURS[task],
                    marker=MODEL_MARKERS[task],
                    linestyle=MODEL_LINESTYLES[task],
                    linewidth=1.65,
                    markersize=4.1,
                    label=TASK_LABELS[task],
                )
                axis.fill_between(
                    x[finite], low[finite], high[finite],
                    color=MODEL_COLOURS[task], alpha=0.13, linewidth=0,
                )
            axis.axhline(reference, color="#666666", linewidth=0.9, zorder=0)
            axis.set_ylabel(ylabel)
            if row_index == 0:
                axis.set_title(
                    "Within-shell permutation\n(multiset preserved)"
                    if intervention == "shell_permutation"
                    else "External shell replacement\n(multiset changed)"
                )
            if row_index == 2:
                axis.set_xlabel(
                    "Exact shell distance d"
                    if condition == "exact_shell"
                    else "Far-tail radius r (all shells d ≥ r)"
                )
            if not plotted:
                axis.text(0.5, 0.5, "No estimable coalitions", transform=axis.transAxes,
                          ha="center", va="center", color="#666666")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.947),
                   ncol=min(len(handles), 5), frameon=False)
    scope = "exact distance shells" if condition == "exact_shell" else "cumulative far tails"
    fig.suptitle(
        f"Functional-carriage survival under joint semantic interventions: {scope}",
        fontsize=13,
        y=0.992,
    )
    fig.text(
        0.5,
        0.89,
        f"{dataset_label}; signed task-output vectors; no ratio clipping; draws/carriers → graph; 95% graph bootstrap",
        ha="center",
        color="#666666",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.075, top=0.80,
                        wspace=0.25, hspace=0.28)
    suffix = "shell_redundancy" if condition == "exact_shell" else "tail_redundancy"
    return _save_figure(fig, figures_dir, f"{figure_prefix}_{suffix}")


def plot_model_profiles(
    rows: Sequence[Mapping[str, Any]],
    contrast_rows: Sequence[Mapping[str, Any]],
    *,
    channel: str,
    method: str,
    title: str,
    filename: str,
    figures_dir: Path,
    tasks: Sequence[str] = TASKS,
    reference_task: str = "zinc",
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    _figure_theme()
    fig, (profile_axis, contrast_axis) = plt.subplots(
        2,
        1,
        figsize=(8.8, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": (2.8, 1.25), "hspace": 0.08},
    )
    maximum_distance = 0
    for draw_order, task in enumerate(tasks):
        values = sorted(
            (
                row
                for row in rows
                if str(row["task"]) == task
                and str(row["channel"]) == channel
                and str(row["method"]) == method
            ),
            key=lambda row: _integer(row, "distance"),
        )
        if not values:
            continue
        x = np.asarray([_integer(value, "distance") for value in values])
        y = np.asarray([_float(value, "mean") for value in values])
        low = np.asarray([_float(value, "low") for value in values])
        high = np.asarray([_float(value, "high") for value in values])
        maximum_distance = max(maximum_distance, int(x.max(initial=0)))
        profile_axis.fill_between(
            x,
            low,
            high,
            color=MODEL_COLOURS[task],
            alpha=0.08,
            linewidth=0,
            zorder=1 + draw_order,
        )
        profile_axis.plot(
            x,
            y,
            color=MODEL_COLOURS[task],
            marker=MODEL_MARKERS[task],
            linestyle=MODEL_LINESTYLES[task],
            markerfacecolor="white",
            markeredgewidth=1.15,
            markersize=5.4,
            linewidth=2.0,
            label=TASK_LABELS[task],
            zorder=5 + draw_order,
        )

    contrast_bound = 0.0
    comparison_tasks = [task for task in tasks if task != reference_task]
    for draw_order, task in enumerate(comparison_tasks):
        values = sorted(
            (
                row
                for row in contrast_rows
                if str(row["task"]) == task
                and str(row["channel"]) == channel
                and str(row["method"]) == method
            ),
            key=lambda row: _integer(row, "distance"),
        )
        if not values:
            continue
        x = np.asarray([_integer(value, "distance") for value in values])
        y = np.asarray([_float(value, "mean") for value in values])
        low = np.asarray([_float(value, "low") for value in values])
        high = np.asarray([_float(value, "high") for value in values])
        maximum_distance = max(maximum_distance, int(x.max(initial=0)))
        contrast_bound = max(
            contrast_bound,
            float(np.max(np.abs(np.concatenate((low, high))))),
        )
        contrast_axis.fill_between(
            x,
            low,
            high,
            color=MODEL_COLOURS[task],
            alpha=0.10,
            linewidth=0,
            zorder=1 + draw_order,
        )
        contrast_axis.plot(
            x,
            y,
            color=MODEL_COLOURS[task],
            marker=MODEL_MARKERS[task],
            linestyle=MODEL_LINESTYLES[task],
            markerfacecolor="white",
            markeredgewidth=1.05,
            markersize=4.7,
            linewidth=1.7,
            zorder=5 + draw_order,
        )

    profile_axis.set_ylabel("Normalised usage mass")
    profile_axis.set_ylim(bottom=0)
    contrast_axis.axhline(0, color="#666666", linewidth=0.9, zorder=0)
    contrast_axis.set_ylabel(
        f"Difference from\n{TASK_LABELS[reference_task]}"
    )
    contrast_axis.set_xlabel("Shortest-path distance")
    if contrast_bound > 0:
        contrast_bound *= 1.12
        contrast_axis.set_ylim(-contrast_bound, contrast_bound)
    else:
        contrast_axis.set_ylim(-0.01, 0.01)
    contrast_axis.set_xlim(-0.15, maximum_distance + 0.15)
    contrast_axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    contrast_axis.text(
        0.995,
        0.04,
        "Paired by held-out graph",
        transform=contrast_axis.transAxes,
        ha="right",
        va="bottom",
        color="#666666",
        fontsize=8,
    )
    handles, labels = profile_axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(tasks),
        frameon=False,
        handlelength=2.7,
        columnspacing=1.4,
    )
    fig.suptitle(title, fontsize=13, y=0.988)
    fig.text(
        0.5,
        0.865,
        "Mean with 95% graph-bootstrap confidence interval",
        ha="center",
        color="#666666",
        fontsize=8.5,
    )
    fig.subplots_adjust(
        left=0.115,
        right=0.985,
        bottom=0.105,
        top=0.82,
    )
    return _save_figure(fig, figures_dir, filename)


def plot_semantic_usage_estimands(
    profile_rows: Sequence[Mapping[str, Any]],
    contrast_rows: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
    tasks: Sequence[str],
    dataset_label: str,
    figure_prefix: str,
) -> dict[str, str]:
    """Compare shell-summed and shell-size-adjusted semantic usage."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _figure_theme()
    available_tasks = [
        task
        for task in tasks
        if any(str(row["task"]) == task for row in profile_rows)
    ]
    if not available_tasks:
        print(
            "[reach:warning] semantic usage estimand plot has no available "
            "tasks; writing an empty audit figure",
            flush=True,
        )
        fig, axis = plt.subplots(figsize=(8.0, 3.0))
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "No estimable semantic usage profiles",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#666666",
        )
        fig.suptitle(
            f"Finite and Jacobian semantic usage on {dataset_label}",
            fontsize=13,
        )
        return _save_figure(
            fig,
            figures_dir,
            f"{figure_prefix}_semantic_usage_estimand_comparison",
        )
    figure_height = 2.05 * len(available_tasks) + 2.0
    fig, axes = plt.subplots(
        len(available_tasks),
        len(USAGE_ESTIMANDS),
        figsize=(10.2, figure_height),
        sharex=True,
        sharey="col",
        squeeze=False,
    )
    maximum_distance = 0
    for row_index, task in enumerate(available_tasks):
        for column_index, estimand in enumerate(USAGE_ESTIMANDS):
            axis = axes[row_index, column_index]
            for draw_order, method in enumerate(
                ("bamberger", "functional_carriage")
            ):
                values = sorted(
                    (
                        row
                        for row in profile_rows
                        if str(row["task"]) == task
                        and str(row["method"]) == method
                        and str(row["estimand"]) == estimand
                    ),
                    key=lambda row: _integer(row, "distance"),
                )
                if not values:
                    continue
                x = np.asarray(
                    [_integer(value, "distance") for value in values],
                    dtype=np.int64,
                )
                y = np.asarray(
                    [_float(value, "mean") for value in values],
                    dtype=np.float64,
                )
                low = np.asarray(
                    [_float(value, "low") for value in values],
                    dtype=np.float64,
                )
                high = np.asarray(
                    [_float(value, "high") for value in values],
                    dtype=np.float64,
                )
                maximum_distance = max(
                    maximum_distance,
                    int(x.max(initial=0)),
                )
                axis.fill_between(
                    x,
                    low,
                    high,
                    color=METHOD_COLOURS[method],
                    alpha=0.09,
                    linewidth=0,
                    zorder=1 + draw_order,
                )
                axis.plot(
                    x,
                    y,
                    color=METHOD_COLOURS[method],
                    marker=METHOD_MARKERS[method],
                    linestyle=METHOD_LINESTYLES[method],
                    markerfacecolor="white",
                    markeredgewidth=1.1,
                    markersize=4.6,
                    linewidth=1.8,
                    zorder=5 + draw_order,
                )

            if estimand == "radial_allocation":
                opportunity = sorted(
                    (
                        row
                        for row in profile_rows
                        if str(row["task"]) == task
                        and str(row["method"]) == "functional_carriage"
                        and str(row["estimand"]) == "shell_opportunity"
                    ),
                    key=lambda row: _integer(row, "distance"),
                )
                if opportunity:
                    x = np.asarray(
                        [_integer(value, "distance") for value in opportunity],
                        dtype=np.int64,
                    )
                    y = np.asarray(
                        [_float(value, "mean") for value in opportunity],
                        dtype=np.float64,
                    )
                    axis.plot(
                        x,
                        y,
                        color="#8A8A8A",
                        linestyle=(0, (2, 2)),
                        linewidth=1.15,
                        alpha=0.8,
                        zorder=2,
                    )

            tv = next(
                (
                    row
                    for row in contrast_rows
                    if str(row["task"]) == task
                    and str(row["estimand"]) == estimand
                    and str(row["metric"]) == "profile_tv"
                ),
                None,
            )
            if tv is not None:
                axis.text(
                    0.98,
                    0.94,
                    (
                        f"TV={_float(tv, 'mean'):.2f} "
                        f"[{_float(tv, 'low'):.2f}, {_float(tv, 'high'):.2f}]"
                    ),
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    color="#555555",
                    fontsize=7.7,
                )
            axis.set_ylim(bottom=0)
            if row_index == 0:
                axis.set_title(USAGE_ESTIMAND_LABELS[estimand], pad=8)
            if column_index == 0:
                axis.set_ylabel(
                    f"{TASK_LABELS[task]}\nNormalised mass",
                    fontsize=8.7,
                )
            if row_index == len(available_tasks) - 1:
                axis.set_xlabel("Shortest-path distance")

    for axis in axes[-1]:
        axis.set_xlim(-0.15, maximum_distance + 0.15)
        axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLOURS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            markerfacecolor="white",
            linewidth=1.8,
            label=METHOD_LABELS[method],
        )
        for method in ("bamberger", "functional_carriage")
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#8A8A8A",
            linestyle=(0, (2, 2)),
            linewidth=1.15,
            label="Finite-source shell opportunity",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        frameon=False,
        handlelength=2.7,
        columnspacing=1.35,
    )
    fig.suptitle(
        f"Finite and Jacobian semantic usage on {dataset_label}",
        fontsize=13,
        y=0.992,
    )
    fig.text(
        0.5,
        0.865,
        (
            "Radial allocation ∝ mean per-carrier sensitivity × carriers in "
            "shell; 95% held-out-graph bootstrap intervals"
        ),
        ha="center",
        color="#666666",
        fontsize=8.4,
    )
    fig.subplots_adjust(
        left=0.13,
        right=0.985,
        bottom=0.075,
        top=0.82,
        hspace=0.35,
        wspace=0.18,
    )
    return _save_figure(
        fig,
        figures_dir,
        f"{figure_prefix}_semantic_usage_estimand_comparison",
    )


def plot_profile_trajectory_decomposition(
    rows: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
    tasks: Sequence[str],
    dataset_label: str,
    figure_prefix: str,
) -> dict[str, str]:
    """Plot local mismatch, finite drift, and their resulting endpoint gap."""

    import matplotlib.pyplot as plt

    _figure_theme()
    available_tasks = [
        task for task in tasks if any(str(row["task"]) == task for row in rows)
    ]
    if not available_tasks:
        print(
            "[reach:warning] profile trajectory summary is empty; writing an "
            "empty audit figure",
            flush=True,
        )
        fig, axis = plt.subplots(figsize=(8.0, 3.0))
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "No estimable profile trajectories",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#666666",
        )
        return _save_figure(
            fig,
            figures_dir,
            f"{figure_prefix}_semantic_trajectory_decomposition",
        )

    dose_row = next(
        row
        for row in rows
        if str(row["task"]) == available_tasks[0]
    )
    small_dose = _float(dose_row, "small_dose")
    full_dose = _float(dose_row, "full_dose")
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 4.8),
        gridspec_kw={"width_ratios": (1.05, 1.05, 0.9), "wspace": 0.36},
    )
    component_specs = (
        ("baseline", rf"Local mismatch\n$p_{{{small_dose:g}}}-p_J$"),
        (
            "finite_drift",
            rf"Finite drift\n$p_{{{full_dose:g}}}-p_{{{small_dose:g}}}$",
        ),
        ("endpoint", rf"Endpoint gap\n$p_{{{full_dose:g}}}-p_J$"),
    )
    positions = np.arange(len(component_specs), dtype=np.float64)
    for task in available_tasks:
        task_rows = {
            str(row["metric"]): row
            for row in rows
            if str(row["task"]) == task
        }
        tv_values = [task_rows[f"{name}_tv"] for name, _label in component_specs]
        expected_values = [
            task_rows[f"{name}_expected_distance"]
            for name, _label in component_specs
        ]
        for axis, values in zip(axes[:2], (tv_values, expected_values)):
            means = np.asarray([_float(value, "mean") for value in values])
            lows = np.asarray([_float(value, "low") for value in values])
            highs = np.asarray([_float(value, "high") for value in values])
            axis.fill_between(
                positions,
                lows,
                highs,
                color=MODEL_COLOURS[task],
                alpha=0.09,
                linewidth=0,
            )
            axis.plot(
                positions,
                means,
                color=MODEL_COLOURS[task],
                marker=MODEL_MARKERS[task],
                linestyle=MODEL_LINESTYLES[task],
                markerfacecolor="white",
                markeredgewidth=1.1,
                markersize=5.0,
                linewidth=1.8,
                label=TASK_LABELS[task],
            )

    labels = [label.replace("\\n", "\n") for _name, label in component_specs]
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Profile distance (TV)")
    axes[0].set_ylim(bottom=0)
    axes[0].set_title("Magnitude of each profile change", fontsize=10)
    axes[1].set_xticks(positions, labels)
    axes[1].axhline(0, color="#666666", linewidth=0.9, zorder=0)
    axes[1].set_ylabel("Expected-distance component")
    axes[1].set_title("Signed movement in expected distance", fontsize=10)

    cancellation_by_task = {
        task: next(
            row
            for row in rows
            if str(row["task"]) == task
            and str(row["metric"]) == "discrepancy_cancellation"
        )
        for task in available_tasks
    }
    y_positions = np.arange(len(available_tasks), dtype=np.float64)
    for position, task in zip(y_positions, available_tasks):
        value = cancellation_by_task[task]
        mean = _float(value, "mean")
        low = _float(value, "low")
        high = _float(value, "high")
        axes[2].errorbar(
            mean,
            position,
            xerr=np.asarray([[mean - low], [high - mean]]),
            color=MODEL_COLOURS[task],
            marker=MODEL_MARKERS[task],
            markerfacecolor="white",
            markeredgewidth=1.1,
            markersize=5.2,
            linewidth=1.4,
            capsize=2.5,
        )
    axes[2].set_yticks(
        y_positions,
        [TASK_LABELS[task] for task in available_tasks],
    )
    axes[2].tick_params(axis="y", labelsize=8.2)
    axes[2].invert_yaxis()
    axes[2].set_xlim(-0.03, 1.03)
    axes[2].set_xlabel("Discrepancy cancellation")
    axes[2].set_title("Direction of finite drift", fontsize=10)
    axes[2].text(
        0.02,
        0.03,
        "0 = reinforcing\n1 = fully opposing",
        transform=axes[2].transAxes,
        ha="left",
        va="bottom",
        color="#666666",
        fontsize=7.5,
    )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.84),
        ncol=len(available_tasks),
        frameon=False,
        handlelength=2.5,
        columnspacing=1.2,
    )
    fig.suptitle(
        f"Bamberger mismatch and finite semantic drift on {dataset_label}",
        fontsize=13,
        y=0.992,
    )
    fig.text(
        0.5,
        0.895,
        (
            rf"$p_{{{full_dose:g}}}-p_J="
            rf"(p_{{{small_dose:g}}}-p_J)+"
            rf"(p_{{{full_dose:g}}}-p_{{{small_dose:g}}})$; "
            "mean with 95% paired-graph bootstrap interval"
        ),
        ha="center",
        color="#666666",
        fontsize=8.4,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.69)
    return _save_figure(
        fig,
        figures_dir,
        f"{figure_prefix}_semantic_trajectory_decomposition",
    )


def plot_profile_trajectory_distance_components(
    rows: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
    tasks: Sequence[str],
    dataset_label: str,
    figure_prefix: str,
) -> dict[str, str]:
    """Show where finite drift reinforces or offsets local-reference mismatch."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _figure_theme()
    available_tasks = [
        task for task in tasks if any(str(row["task"]) == task for row in rows)
    ]
    if not available_tasks:
        print(
            "[reach:warning] distance-wise profile trajectory is empty; writing "
            "an empty audit figure",
            flush=True,
        )
        fig, axis = plt.subplots(figsize=(8.0, 3.0))
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "No estimable distance-wise trajectory",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#666666",
        )
        return _save_figure(
            fig,
            figures_dir,
            f"{figure_prefix}_semantic_trajectory_distance_components",
        )

    component_specs = (
        ("baseline_mismatch", "Local mismatch", "#222222", "--", "^"),
        ("finite_drift", "Finite drift", "#D55E00", ":", "s"),
        ("endpoint_gap", "Endpoint gap", "#0072B2", "-", "o"),
    )
    fig, axes = plt.subplots(
        len(available_tasks),
        1,
        figsize=(8.9, 1.85 * len(available_tasks) + 1.8),
        sharex=True,
        squeeze=False,
    )
    maximum_distance = 0
    absolute_bound = 0.0
    for row_index, task in enumerate(available_tasks):
        axis = axes[row_index, 0]
        for component, _label, colour, linestyle, marker in component_specs:
            values = sorted(
                (
                    row
                    for row in rows
                    if str(row["task"]) == task
                    and str(row["component"]) == component
                ),
                key=lambda row: _integer(row, "distance"),
            )
            if not values:
                continue
            x = np.asarray([_integer(value, "distance") for value in values])
            mean = np.asarray([_float(value, "mean") for value in values])
            low = np.asarray([_float(value, "low") for value in values])
            high = np.asarray([_float(value, "high") for value in values])
            maximum_distance = max(maximum_distance, int(x.max(initial=0)))
            absolute_bound = max(
                absolute_bound,
                float(np.max(np.abs(np.concatenate((low, high))))),
            )
            axis.fill_between(
                x,
                low,
                high,
                color=colour,
                alpha=0.08,
                linewidth=0,
            )
            axis.plot(
                x,
                mean,
                color=colour,
                linestyle=linestyle,
                marker=marker,
                markerfacecolor="white",
                markeredgewidth=1.0,
                markersize=4.2,
                linewidth=1.6,
            )
        axis.axhline(0, color="#777777", linewidth=0.8, zorder=0)
        axis.set_ylabel(
            f"{TASK_LABELS[task]}\n$\\Delta$ mass",
            fontsize=8.5,
        )
    bound = max(absolute_bound * 1.12, 0.01)
    for axis in axes[:, 0]:
        axis.set_ylim(-bound, bound)
        axis.set_xlim(-0.15, maximum_distance + 0.15)
        axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[-1, 0].set_xlabel("Shortest-path distance")

    handles = [
        Line2D(
            [0],
            [0],
            color=colour,
            linestyle=linestyle,
            marker=marker,
            markerfacecolor="white",
            linewidth=1.6,
            label=label,
        )
        for _component, label, colour, linestyle, marker in component_specs
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=3,
        frameon=False,
        handlelength=2.5,
    )
    fig.suptitle(
        f"Distance-wise finite–Jacobian profile decomposition on {dataset_label}",
        fontsize=13,
        y=0.992,
    )
    fig.text(
        0.5,
        0.855,
        (
            "Endpoint gap = local-reference mismatch + finite drift; "
            "mean with 95% paired-graph bootstrap interval"
        ),
        ha="center",
        color="#666666",
        fontsize=8.4,
    )
    fig.subplots_adjust(
        left=0.14,
        right=0.985,
        bottom=0.09,
        top=0.80,
        hspace=0.30,
    )
    return _save_figure(
        fig,
        figures_dir,
        f"{figure_prefix}_semantic_trajectory_distance_components",
    )


def plot_interpolation_sweep(
    rows: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
    tasks: Sequence[str] = TASKS,
    figure_prefix: str = "zinc",
) -> dict[str, str]:
    """Plot literal-method and matched finite-dose departures side by side."""

    import matplotlib.pyplot as plt

    _figure_theme()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), sharex=True)
    metric_specs = (
        ("profile_tv", "Profile distance (TV)"),
        (
            "expected_distance_difference",
            "Expected-distance difference",
        ),
    )
    baseline_specs = (
        ("bamberger", "Reference: literal Bamberger Jacobian range"),
        (
            "matched_small_dose",
            "Reference: matched smallest finite dose",
        ),
    )
    all_doses = sorted({_float(row, "interpolation_dose") for row in rows})
    if not all_doses:
        raise RuntimeError("interpolation sweep is empty; rerun PHASE='all'")
    reference_doses = {
        _float(row, "matched_reference_dose")
        for row in rows
        if str(row["baseline"]) == "matched_small_dose"
    }
    if len(reference_doses) != 1:
        raise RuntimeError("matched interpolation reference dose is ambiguous")
    reference_dose = next(iter(reference_doses))
    for row_index, (baseline, row_title) in enumerate(baseline_specs):
        for column_index, (metric, ylabel) in enumerate(metric_specs):
            axis = axes[row_index, column_index]
            for draw_order, task in enumerate(tasks):
                values = sorted(
                    (
                        row
                        for row in rows
                        if str(row["task"]) == task
                        and str(row["metric"]) == metric
                        and str(row["baseline"]) == baseline
                    ),
                    key=lambda row: _float(row, "interpolation_dose"),
                )
                if not values:
                    raise RuntimeError(
                        f"interpolation sweep is missing {baseline!r}/{metric!r} "
                        f"for {task!r}; rerun PHASE='all' with the current version"
                    )
                x = np.asarray(
                    [_float(value, "interpolation_dose") for value in values]
                )
                y = np.asarray([_float(value, "mean") for value in values])
                low = np.asarray([_float(value, "low") for value in values])
                high = np.asarray([_float(value, "high") for value in values])
                axis.fill_between(
                    x,
                    low,
                    high,
                    color=MODEL_COLOURS[task],
                    alpha=0.09,
                    linewidth=0,
                    zorder=1 + draw_order,
                )
                axis.plot(
                    x,
                    y,
                    color=MODEL_COLOURS[task],
                    marker=MODEL_MARKERS[task],
                    linestyle=MODEL_LINESTYLES[task],
                    markerfacecolor="white",
                    markeredgewidth=1.1,
                    markersize=4.7,
                    linewidth=1.8,
                    label=TASK_LABELS[task],
                    zorder=5 + draw_order,
                )
            axis.set_xscale("log")
            axis.set_xticks(all_doses)
            axis.set_xticklabels([f"{dose:g}" for dose in all_doses])
            axis.set_ylabel(ylabel)
            if metric == "profile_tv":
                axis.set_ylim(bottom=0)
                zero_text = "0 = identical profile"
            else:
                axis.axhline(0, color="#666666", linewidth=0.9, zorder=0)
                zero_text = "0 = identical expected distance"
            axis.text(
                0.03,
                0.94,
                zero_text,
                transform=axis.transAxes,
                va="top",
                color="#666666",
                fontsize=7.5,
            )
            if column_index == 0:
                axis.set_title(row_title, loc="left", fontsize=9.5)
    for axis in axes[-1]:
        axis.set_xlabel(r"Donor-swap fraction $\alpha$")
    axes[1, 0].text(
        0.97,
        0.94,
        rf"matched reference: $\alpha={reference_dose:g}$",
        transform=axes[1, 0].transAxes,
        ha="right",
        va="top",
        color="#666666",
        fontsize=7.5,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(tasks),
        frameon=False,
        handlelength=2.7,
        columnspacing=1.4,
    )
    fig.suptitle(
        "Semantic distance-profile change across donor-swap magnitude",
        fontsize=13,
        y=0.992,
    )
    fig.text(
        0.5,
        0.872,
        "Paired held-out graphs; mean with 95% graph-bootstrap confidence interval",
        ha="center",
        color="#666666",
        fontsize=8.5,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.09,
        top=0.82,
        wspace=0.22,
        hspace=0.30,
    )
    return _save_figure(
        fig,
        figures_dir,
        f"{figure_prefix}_semantic_interpolation_sweep",
    )


def plot_expected_distance(
    rows: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
    tasks: Sequence[str] = TASKS,
    channels: Sequence[str] = CHANNELS,
    dataset_label: str = "ZINC",
    figure_prefix: str = "zinc",
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    _figure_theme()
    fig, axes_grid = plt.subplots(
        1,
        len(channels),
        figsize=(5.4 * len(channels), 4.1),
        sharey=True,
        squeeze=False,
    )
    axes = axes_grid[0]
    positions = np.arange(len(tasks), dtype=np.float64)
    for axis, channel in zip(axes, channels):
        methods = (
            ("bamberger", *DONOR_METHODS)
            if channel == "semantic"
            else DONOR_METHODS
        )
        offsets = np.linspace(-0.18, 0.18, len(methods))
        for offset, method in zip(offsets, methods):
            values_by_task = {
                str(row["task"]): row
                for row in rows
                if str(row["channel"]) == channel
                and str(row["method"]) == method
            }
            x_values: list[float] = []
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for index, task in enumerate(tasks):
                if task not in values_by_task:
                    continue
                value = values_by_task[task]
                x_values.append(float(positions[index] + offset))
                means.append(_float(value, "mean"))
                lows.append(_float(value, "low"))
                highs.append(_float(value, "high"))
            mean_array = np.asarray(means)
            axis.errorbar(
                x_values,
                mean_array,
                yerr=np.vstack(
                    (
                        mean_array - np.asarray(lows),
                        np.asarray(highs) - mean_array,
                    )
                ),
                fmt=METHOD_MARKERS[method],
                markersize=5,
                capsize=2.5,
                linewidth=1.3,
                color=METHOD_COLOURS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_title(f"{channel.capitalize()} perturbations")
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [TASK_LABELS[task] for task in tasks],
            rotation=22,
            ha="right",
        )
        axis.set_ylabel("Expected shortest-path distance")
        axis.set_ylim(bottom=0)
    if "structural" in channels:
        structural_axis = axes[list(channels).index("structural")]
        structural_axis.text(
            0.03,
            0.95,
            "Bamberger structural analogue not defined",
            transform=structural_axis.transAxes,
            va="top",
            color="#666666",
            fontsize=8,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        f"Expected distance of model usage on {dataset_label}",
        fontsize=13,
        y=0.985,
    )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.27, top=0.72, wspace=0.22)
    return _save_figure(fig, figures_dir, f"{figure_prefix}_expected_reach")


def plot_output_coherence(
    profile_rows: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
    tasks: Sequence[str],
    dataset_label: str,
    figure_prefix: str,
) -> dict[str, str]:
    """Show how much finite semantic output carriage survives carrier cancellation."""

    import matplotlib.pyplot as plt

    _figure_theme()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0))
    profile_specs = (
        (
            axes[0, 0],
            "apparent_mass",
            "Apparent output carriage",
            "Normalised output-carriage mass",
        ),
        (
            axes[0, 1],
            "coherent_mass",
            "Coherent output carriage",
            "Normalised output-carriage mass",
        ),
        (
            axes[1, 0],
            "coherence_ratio",
            "Carriage surviving within-shell cancellation",
            "Coherent / apparent mass",
        ),
    )
    for axis, metric, title, ylabel in profile_specs:
        for draw_order, task in enumerate(tasks):
            values = sorted(
                (
                    row
                    for row in profile_rows
                    if str(row["task"]) == task and str(row["metric"]) == metric
                ),
                key=lambda row: _integer(row, "distance"),
            )
            if not values:
                continue
            x = np.asarray([_integer(value, "distance") for value in values])
            mean = np.asarray([_float(value, "mean") for value in values])
            low = np.asarray([_float(value, "low") for value in values])
            high = np.asarray([_float(value, "high") for value in values])
            axis.fill_between(
                x,
                low,
                high,
                color=MODEL_COLOURS[task],
                alpha=0.09,
                linewidth=0,
                zorder=1 + draw_order,
            )
            axis.plot(
                x,
                mean,
                color=MODEL_COLOURS[task],
                marker=MODEL_MARKERS[task],
                linestyle=MODEL_LINESTYLES[task],
                markerfacecolor="white",
                markeredgewidth=1.0,
                markersize=4.2,
                linewidth=1.8,
                label=TASK_LABELS[task],
                zorder=5 + draw_order,
            )
        axis.set_title(title)
        axis.set_xlabel("Shortest-path distance")
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0)
        axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    top_limit = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
    axes[0, 0].set_ylim(0, top_limit)
    axes[0, 1].set_ylim(0, top_limit)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].axhline(1, color="#777777", linewidth=0.8, zorder=0)
    axes[1, 0].text(
        0.97,
        0.94,
        "1 = no within-shell cancellation",
        transform=axes[1, 0].transAxes,
        ha="right",
        va="top",
        color="#666666",
        fontsize=7.5,
    )

    contraction_axis = axes[1, 1]
    positions = np.arange(len(tasks), dtype=np.float64)
    values_by_task = {
        str(row["task"]): row
        for row in expected_rows
        if str(row["metric"]) == "expected_distance_change"
    }
    for index, task in enumerate(tasks):
        value = values_by_task.get(task)
        if value is None:
            continue
        mean = _float(value, "mean")
        low = _float(value, "low")
        high = _float(value, "high")
        contraction_axis.errorbar(
            positions[index],
            mean,
            yerr=np.asarray([[mean - low], [high - mean]]),
            fmt=MODEL_MARKERS[task],
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5.5,
            capsize=2.5,
            color=MODEL_COLOURS[task],
            linewidth=1.4,
        )
    contraction_axis.axhline(0, color="#666666", linewidth=0.9, zorder=0)
    contraction_axis.set_title("Expected-distance change after cancellation")
    contraction_axis.set_ylabel("Coherent minus apparent distance")
    contraction_axis.set_xticks(positions)
    contraction_axis.set_xticklabels(
        [TASK_LABELS[task] for task in tasks],
        rotation=22,
        ha="right",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(tasks),
        frameon=False,
        handlelength=2.7,
        columnspacing=1.4,
    )
    fig.suptitle(
        f"Apparent and coherent finite semantic carriage on {dataset_label}",
        fontsize=13,
        y=0.992,
    )
    fig.text(
        0.5,
        0.872,
        "Signed scalar-output path integration; mean with 95% graph-bootstrap interval",
        ha="center",
        color="#666666",
        fontsize=8.5,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.13,
        top=0.82,
        wspace=0.25,
        hspace=0.38,
    )
    return _save_figure(
        fig,
        figures_dir,
        f"{figure_prefix}_semantic_output_coherence",
    )


def plot_scale_dependence(
    rows: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
    tasks: Sequence[str],
    reference_task: str,
    dataset_label: str,
    figure_prefix: str,
) -> dict[str, str]:
    """Plot paired performance and semantic reach against molecular scale."""

    import matplotlib.pyplot as plt

    _figure_theme()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0))
    descriptor_specs = (
        ("num_nodes", "Molecule size", "Atoms per molecule"),
        ("diameter", "Graph diameter", "Diameter (shortest-path hops)"),
    )
    metric_specs = (
        (
            "performance",
            "mae_difference_from_reference",
            f"Per-molecule MAE difference\nfrom {TASK_LABELS[reference_task]}",
        ),
        (
            "reach",
            "functional_expected_distance",
            "Semantic Functional-carriage\nexpected distance",
        ),
    )
    for column_index, (descriptor, title, xlabel) in enumerate(descriptor_specs):
        for row_index, (analysis, metric, ylabel) in enumerate(metric_specs):
            axis = axes[row_index, column_index]
            for draw_order, task in enumerate(tasks):
                values = sorted(
                    (
                        row
                        for row in rows
                        if str(row["descriptor"]) == descriptor
                        and str(row["analysis"]) == analysis
                        and str(row["metric"]) == metric
                        and str(row["task"]) == task
                    ),
                    key=lambda row: _float(row, "bin_descriptor_mean"),
                )
                if not values:
                    continue
                x = np.asarray(
                    [_float(value, "bin_descriptor_mean") for value in values],
                    dtype=np.float64,
                )
                y = np.asarray([_float(value, "mean") for value in values])
                low = np.asarray([_float(value, "low") for value in values])
                high = np.asarray([_float(value, "high") for value in values])
                axis.fill_between(
                    x,
                    low,
                    high,
                    color=MODEL_COLOURS[task],
                    alpha=0.09,
                    linewidth=0,
                    zorder=1 + draw_order,
                )
                axis.plot(
                    x,
                    y,
                    color=MODEL_COLOURS[task],
                    marker=MODEL_MARKERS[task],
                    linestyle=MODEL_LINESTYLES[task],
                    markerfacecolor="white",
                    markeredgewidth=1.1,
                    markersize=5.0,
                    linewidth=1.9,
                    label=TASK_LABELS[task],
                    zorder=5 + draw_order,
                )
            axis.set_ylabel(ylabel)
            axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            counts = sorted(
                {
                    _integer(row, "graphs")
                    for row in rows
                    if str(row["descriptor"]) == descriptor
                    and str(row["analysis"]) == analysis
                    and str(row["metric"]) == metric
                }
            )
            count_text = (
                f"n={counts[0]} per bin"
                if len(counts) == 1
                else f"n={counts[0]}–{counts[-1]} per bin"
            ) if counts else ""
            axis.text(
                0.98,
                0.05 if row_index == 0 else 0.95,
                count_text,
                transform=axis.transAxes,
                ha="right",
                va="bottom" if row_index == 0 else "top",
                color="#666666",
                fontsize=7.5,
            )
            if row_index == 0:
                axis.set_title(title)
                axis.axhline(0, color="#666666", linewidth=0.9, zorder=0)
            else:
                axis.set_ylim(bottom=0)
            axis.set_xlabel(xlabel)
    axes[0, 0].text(
        0.03,
        0.95,
        "positive = higher error than dense",
        transform=axes[0, 0].transAxes,
        va="top",
        color="#666666",
        fontsize=7.5,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(tasks),
        frameon=False,
        handlelength=2.7,
        columnspacing=1.4,
    )
    fig.suptitle(
        f"Full-test performance and semantic functional reach across {dataset_label} scale",
        fontsize=13,
        y=0.992,
    )
    fig.text(
        0.5,
        0.872,
        "Adjacent-value adaptive bins; paired molecules; 95% graph-bootstrap interval",
        ha="center",
        color="#666666",
        fontsize=8.5,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.10,
        top=0.82,
        wspace=0.24,
        hspace=0.22,
    )
    return _save_figure(
        fig,
        figures_dir,
        f"{figure_prefix}_scale_dependence",
    )


def plot_scale_slopes(
    rows: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
    tasks: Sequence[str],
    reference_task: str,
    dataset_label: str,
    figure_prefix: str,
) -> dict[str, str]:
    """Forest plot of bin-independent continuous scale trends."""

    import matplotlib.pyplot as plt

    _figure_theme()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.7))
    descriptor_specs = (
        ("num_nodes", "Molecule size", "per additional atom"),
        ("diameter", "Graph diameter", "per additional hop"),
    )
    analysis_specs = (
        (
            "performance",
            f"MAE difference slope from\n{TASK_LABELS[reference_task]}",
        ),
        ("reach", "Functional expected-distance slope"),
    )
    positions = np.arange(len(tasks), dtype=np.float64)
    for row_index, (analysis, xlabel) in enumerate(analysis_specs):
        for column_index, (descriptor, title, unit) in enumerate(descriptor_specs):
            axis = axes[row_index, column_index]
            values_by_task = {
                str(row["task"]): row
                for row in rows
                if str(row["analysis"]) == analysis
                and str(row["descriptor"]) == descriptor
            }
            for index, task in enumerate(tasks):
                value = values_by_task.get(task)
                if value is None:
                    continue
                mean = _float(value, "slope")
                low = _float(value, "low")
                high = _float(value, "high")
                if not np.isfinite([mean, low, high]).all():
                    continue
                axis.errorbar(
                    mean,
                    positions[index],
                    xerr=np.asarray([[mean - low], [high - mean]]),
                    fmt=MODEL_MARKERS[task],
                    markerfacecolor="white",
                    markeredgewidth=1.2,
                    markersize=5.5,
                    capsize=2.5,
                    color=MODEL_COLOURS[task],
                    linewidth=1.4,
                )
            axis.axvline(0, color="#666666", linewidth=0.9, zorder=0)
            axis.set_yticks(positions)
            axis.set_yticklabels([TASK_LABELS[task] for task in tasks])
            axis.invert_yaxis()
            axis.set_xlabel(f"{xlabel} ({unit})")
            if row_index == 0:
                axis.set_title(title)
            finite_intervals = [
                abs(_float(value, key))
                for value in values_by_task.values()
                for key in ("low", "high")
                if np.isfinite(_float(value, key))
            ]
            if not finite_intervals:
                axis.text(
                    0.5,
                    0.5,
                    "Not estimable:\nno descriptor variation",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="#666666",
                    fontsize=8,
                )
            elif max(finite_intervals) < 1.0e-12:
                axis.set_xlim(-1.0e-6, 1.0e-6)
    fig.suptitle(
        f"Continuous molecular-scale trends on {dataset_label}",
        fontsize=13,
        y=0.985,
    )
    fig.text(
        0.5,
        0.915,
        "OLS slope with 95% paired-molecule bootstrap confidence interval",
        ha="center",
        color="#666666",
        fontsize=8.5,
    )
    fig.subplots_adjust(
        left=0.16,
        right=0.985,
        bottom=0.10,
        top=0.84,
        wspace=0.30,
        hspace=0.40,
    )
    return _save_figure(
        fig,
        figures_dir,
        f"{figure_prefix}_scale_slopes",
    )


def figures(
    config: ZincReachConfig,
    *,
    output_dir: Path,
    print_audit: bool = True,
) -> dict[str, Any]:
    """Build every table and paper figure from cached CSV files only."""

    results_dir = output_dir / "results"
    donor_path = results_dir / "donor_carrier_mass.csv"
    interpolation_path = results_dir / "semantic_interpolation_mass.csv"
    bamberger_path = results_dir / "bamberger_input_output_influence.csv"
    output_carriage_path = results_dir / "semantic_output_carriage.csv"
    beneficial_path = results_dir / "beneficial_carriage.csv"
    survival_path = results_dir / "shell_survival.csv"
    output_carriage_failures_path = results_dir / "output_carriage_failures.csv"
    graph_metrics_path = results_dir / "graph_metrics.csv"
    full_test_metrics_path = results_dir / "full_test_metrics.csv"
    required_paths = [
        donor_path,
        interpolation_path,
        bamberger_path,
        graph_metrics_path,
    ]
    output_carriage_failures: list[dict[str, Any]] = []
    if config.compute_output_carriage:
        required_paths.append(output_carriage_path)
    if config.compute_beneficial:
        required_paths.append(beneficial_path)
    if config.compute_survival:
        required_paths.append(survival_path)
    if config.compute_scale_analysis:
        required_paths.append(full_test_metrics_path)
    if any(not path.is_file() for path in required_paths):
        raise FileNotFoundError(
            "cached measurement CSVs are missing; run PHASE='measure' or 'all' first"
        )
    donor_raw = _read_csv(donor_path)
    bamberger_raw = _read_csv(bamberger_path)
    donor_graph = graph_donor_profiles(
        donor_raw,
        effect_floor=float(config.effect_floor),
    )
    bamberger_graph = graph_bamberger_profiles(
        bamberger_raw,
        effect_floor=float(config.effect_floor),
    )
    semantic_usage_graph = graph_semantic_usage_estimands(
        donor_raw,
        bamberger_raw,
        effect_floor=float(config.effect_floor),
    )
    semantic_usage_profiles, semantic_usage_contrasts = (
        summarise_semantic_usage_estimands(
            semantic_usage_graph,
            bootstrap_replicates=int(config.bootstrap_replicates),
            bootstrap_seed=int(config.analysis_seed) + 900,
        )
    )
    interpolation_graph = graph_interpolation_profiles(
        _read_csv(interpolation_path),
        effect_floor=float(config.effect_floor),
    )
    output_carriage_audit: list[dict[str, Any]] = []
    output_coherence_graph: list[dict[str, Any]] = []
    output_coherence_profiles: list[dict[str, Any]] = []
    output_coherence_expected: list[dict[str, Any]] = []
    if config.compute_output_carriage:
        output_carriage_raw = _read_csv(output_carriage_path)
        output_carriage_failures = (
            _read_csv(output_carriage_failures_path)
            if output_carriage_failures_path.is_file()
            else []
        )
        output_carriage_audit = summarise_output_carriage_audit(
            output_carriage_raw,
            failures=output_carriage_failures,
        )
        if print_audit:
            _print_output_carriage_audit(output_carriage_audit)
        output_coherence_graph = graph_output_coherence_profiles(
            output_carriage_raw,
            effect_floor=float(config.effect_floor),
        )
        output_coherence_profiles, output_coherence_expected = summarise_output_coherence(
            output_coherence_graph,
            bootstrap_replicates=int(config.bootstrap_replicates),
            bootstrap_seed=int(config.analysis_seed) + 500,
        )
    beneficial_graph: list[dict[str, Any]] = []
    beneficial_summary: list[dict[str, Any]] = []
    if config.compute_beneficial:
        beneficial_graph, beneficial_summary = summarise_beneficial_carriage(
            _read_csv(beneficial_path),
            bootstrap_replicates=int(config.bootstrap_replicates),
            bootstrap_seed=int(config.analysis_seed) + 600,
        )
    survival_graph: list[dict[str, Any]] = []
    survival_summary: list[dict[str, Any]] = []
    survival_contrasts: list[dict[str, Any]] = []
    survival_by_size: list[dict[str, Any]] = []
    if config.compute_survival:
        survival_raw = _read_csv(survival_path)
        survival_graph, survival_summary, survival_contrasts = summarise_shell_survival(
            survival_raw,
            bootstrap_replicates=int(config.bootstrap_replicates),
            bootstrap_seed=int(config.analysis_seed) + 700,
        )
        survival_by_size = summarise_shell_survival_by_size(
            survival_raw,
            bootstrap_replicates=int(config.bootstrap_replicates),
            bootstrap_seed=int(config.analysis_seed) + 800,
        )
    interpolation_contrasts, interpolation_summary = (
        summarise_interpolation_contrasts(
            interpolation_graph,
            bamberger_graph,
            bootstrap_replicates=int(config.bootstrap_replicates),
            bootstrap_seed=int(config.analysis_seed) + 300,
            matched_reference_dose=float(min(config.interpolation_doses)),
        )
    )
    trajectory_graph, trajectory_distance_graph = (
        graph_profile_trajectory_decomposition(
            interpolation_graph,
            bamberger_graph,
            effect_floor=float(config.effect_floor),
            small_dose=float(min(config.interpolation_doses)),
            full_dose=float(max(config.interpolation_doses)),
        )
    )
    trajectory_summary, trajectory_distance_summary = (
        summarise_profile_trajectory_decomposition(
            trajectory_graph,
            trajectory_distance_graph,
            bootstrap_replicates=int(config.bootstrap_replicates),
            bootstrap_seed=int(config.analysis_seed) + 1_000,
        )
    )
    graph_rows = [*donor_graph, *bamberger_graph]
    profile_rows, expected_rows = summarise_graph_profiles(
        graph_rows,
        bootstrap_replicates=int(config.bootstrap_replicates),
        bootstrap_seed=int(config.analysis_seed) + 100,
    )
    contrast_rows = summarise_dense_profile_contrasts(
        graph_rows,
        bootstrap_replicates=int(config.bootstrap_replicates),
        bootstrap_seed=int(config.analysis_seed) + 200,
        reference_task=config.profile.reference_task,
    )
    scale_rows: list[dict[str, Any]] = []
    scale_summary: list[dict[str, Any]] = []
    scale_trends: list[dict[str, Any]] = []
    if config.compute_scale_analysis:
        scale_rows, scale_summary, scale_trends = summarise_scale_dependence(
            graph_rows,
            _read_csv(graph_metrics_path),
            _read_csv(full_test_metrics_path),
            tasks=config.tasks,
            reference_task=config.profile.reference_task,
            bootstrap_replicates=int(config.bootstrap_replicates),
            bootstrap_seed=int(config.analysis_seed) + 400,
        )
    _write_csv(results_dir / "graph_distance_profiles.csv", graph_rows)
    _write_csv(results_dir / "distance_profile_summary.csv", profile_rows)
    _write_csv(results_dir / "dense_profile_contrasts.csv", contrast_rows)
    _write_csv(results_dir / "expected_distance_summary.csv", expected_rows)
    _write_csv(
        results_dir / "semantic_usage_estimand_graph_profiles.csv",
        semantic_usage_graph,
    )
    _write_csv(
        results_dir / "semantic_usage_estimand_summary.csv",
        semantic_usage_profiles,
    )
    _write_csv(
        results_dir / "semantic_usage_estimand_contrasts.csv",
        semantic_usage_contrasts,
    )
    _write_csv(
        results_dir / "interpolation_graph_profiles.csv",
        interpolation_graph,
    )
    _write_csv(
        results_dir / "interpolation_graph_contrasts.csv",
        interpolation_contrasts,
    )
    _write_csv(
        results_dir / "interpolation_sweep_summary.csv",
        interpolation_summary,
    )
    _write_csv(
        results_dir / "profile_trajectory_graph_decomposition.csv",
        trajectory_graph,
    )
    _write_csv(
        results_dir / "profile_trajectory_summary.csv",
        trajectory_summary,
    )
    _write_csv(
        results_dir / "profile_trajectory_distance_graph_rows.csv",
        trajectory_distance_graph,
    )
    _write_csv(
        results_dir / "profile_trajectory_distance_summary.csv",
        trajectory_distance_summary,
    )
    _write_csv(results_dir / "scale_dependence_graph_rows.csv", scale_rows)
    _write_csv(results_dir / "scale_dependence_summary.csv", scale_summary)
    _write_csv(results_dir / "scale_trend_slopes.csv", scale_trends)
    _write_csv(
        results_dir / "output_coherence_graph_profiles.csv",
        output_coherence_graph,
    )
    _write_csv(
        results_dir / "output_coherence_profile_summary.csv",
        output_coherence_profiles,
    )
    _write_csv(
        results_dir / "output_coherence_expected_distance.csv",
        output_coherence_expected,
    )
    _write_csv(results_dir / "output_carriage_audit.csv", output_carriage_audit)
    _write_csv(results_dir / "beneficial_graph_profiles.csv", beneficial_graph)
    _write_csv(results_dir / "beneficial_carriage_summary.csv", beneficial_summary)
    _write_csv(results_dir / "shell_survival_graph_rows.csv", survival_graph)
    _write_csv(results_dir / "shell_survival_summary.csv", survival_summary)
    _write_csv(results_dir / "shell_survival_control_contrasts.csv", survival_contrasts)
    _write_csv(results_dir / "shell_survival_by_coalition_size.csv", survival_by_size)

    figures_dir = output_dir / "figures"
    paths = {
        "trajectory_decomposition": plot_profile_trajectory_decomposition(
            trajectory_summary,
            figures_dir=figures_dir,
            tasks=config.tasks,
            dataset_label=config.profile.name,
            figure_prefix=config.profile.figure_prefix,
        ),
        "trajectory_distance_components": (
            plot_profile_trajectory_distance_components(
                trajectory_distance_summary,
                figures_dir=figures_dir,
                tasks=config.tasks,
                dataset_label=config.profile.name,
                figure_prefix=config.profile.figure_prefix,
            )
        ),
        "semantic_estimand_comparison": plot_semantic_usage_estimands(
            semantic_usage_profiles,
            semantic_usage_contrasts,
            figures_dir=figures_dir,
            tasks=config.tasks,
            dataset_label=config.profile.name,
            figure_prefix=config.profile.figure_prefix,
        ),
        "interpolation_sweep": plot_interpolation_sweep(
            interpolation_summary,
            figures_dir=figures_dir,
            tasks=config.tasks,
            figure_prefix=config.profile.figure_prefix,
        ),
        "semantic_functional": plot_model_profiles(
            profile_rows,
            contrast_rows,
            channel="semantic",
            method="functional_carriage",
            title="Semantic usage by Functional carriage",
            filename=f"{config.profile.figure_prefix}_semantic_functional_profiles",
            figures_dir=figures_dir,
            tasks=config.tasks,
            reference_task=config.profile.reference_task,
        ),
        "semantic_bamberger": plot_model_profiles(
            profile_rows,
            contrast_rows,
            channel="semantic",
            method="bamberger",
            title="Semantic usage by Bamberger Jacobian range",
            filename=f"{config.profile.figure_prefix}_semantic_bamberger_profiles",
            figures_dir=figures_dir,
            tasks=config.tasks,
            reference_task=config.profile.reference_task,
        ),
        "expected_distance": plot_expected_distance(
            expected_rows,
            figures_dir=figures_dir,
            tasks=config.tasks,
            channels=config.channels,
            dataset_label=config.profile.name,
            figure_prefix=config.profile.figure_prefix,
        ),
    }
    if "structural" in config.channels:
        paths["structural_functional"] = plot_model_profiles(
            profile_rows,
            contrast_rows,
            channel="structural",
            method="functional_carriage",
            title="Structural usage by Functional carriage",
            filename=f"{config.profile.figure_prefix}_structural_functional_profiles",
            figures_dir=figures_dir,
            tasks=config.tasks,
            reference_task=config.profile.reference_task,
        )
    if config.compute_output_carriage:
        paths["output_coherence"] = plot_output_coherence(
            output_coherence_profiles,
            output_coherence_expected,
            figures_dir=figures_dir,
            tasks=config.tasks,
            dataset_label=config.profile.name,
            figure_prefix=config.profile.figure_prefix,
        )
    if config.compute_beneficial:
        paths["beneficial_carriage"] = plot_beneficial_carriage(
            beneficial_summary,
            figures_dir=figures_dir,
            tasks=config.tasks,
            dataset_label=config.profile.name,
            figure_prefix=config.profile.figure_prefix,
        )
    if config.compute_survival:
        paths["shell_redundancy"] = plot_shell_survival(
            survival_summary,
            condition="exact_shell",
            figures_dir=figures_dir,
            tasks=config.tasks,
            dataset_label=config.profile.name,
            figure_prefix=config.profile.figure_prefix,
        )
        paths["tail_redundancy"] = plot_shell_survival(
            survival_summary,
            condition="far_tail",
            figures_dir=figures_dir,
            tasks=config.tasks,
            dataset_label=config.profile.name,
            figure_prefix=config.profile.figure_prefix,
        )
    if config.compute_scale_analysis:
        paths["scale_dependence"] = plot_scale_dependence(
            scale_summary,
            figures_dir=figures_dir,
            tasks=config.tasks,
            reference_task=config.profile.reference_task,
            dataset_label=config.profile.name,
            figure_prefix=config.profile.figure_prefix,
        )
        paths["scale_slopes"] = plot_scale_slopes(
            scale_trends,
            figures_dir=figures_dir,
            tasks=config.tasks,
            reference_task=config.profile.reference_task,
            dataset_label=config.profile.name,
            figure_prefix=config.profile.figure_prefix,
        )
    _write_json(
        results_dir / "figure_manifest.json",
        {
            "analysis_version": config.profile.analysis_version,
            "fingerprint": config.fingerprint,
            "figures": paths,
            "uncertainty": (
                "95% percentile bootstrap over held-out graphs; one trained "
                "checkpoint per architecture, so intervals do not include training-seed "
                "variance. Difference panels use paired graph-level bootstraps against "
                "Dense GRIT; the interpolation sweep pairs each graph with its "
                "Bamberger and matched smallest-dose profiles. "
                "The profile-trajectory decomposition writes the finite endpoint "
                "gap exactly as local-reference mismatch plus finite-dose drift; "
                "all component intervals remain paired by graph. "
                "Performance-scale analysis uses every test molecule; "
                "carriage-scale analysis uses the "
                "registered graph sample. Adjacent integer values are merged into "
                "density-adaptive bins, and continuous slopes use paired bootstraps."
                " The semantic estimand comparison is rebuilt entirely from cached "
                "carrier-level rows: radial allocation sums within shells, while "
                "normalised per-carrier sensitivity averages within shells before "
                "normalising over distance. Its TV intervals pair methods by graph."
                " Output-coherence profiles integrate the scalar z-output along each "
                "finite semantic donor path and compare carrier magnitudes before and "
                "after signed within-distance aggregation."
                " Beneficial carriage is an exact positive-is-beneficial task-loss "
                "allocation. Shell-survival figures average repeated coalitions and "
                "sampled carriers within graph before bootstrapping graphs; external "
                "replacement-minus-permutation contrasts remain paired through draw."
            ),
        },
    )
    return {
        "figures": paths,
        "profile_rows": profile_rows,
        "contrast_rows": contrast_rows,
        "expected_rows": expected_rows,
        "semantic_usage_graph": semantic_usage_graph,
        "semantic_usage_profiles": semantic_usage_profiles,
        "semantic_usage_contrasts": semantic_usage_contrasts,
        "interpolation_rows": interpolation_summary,
        "trajectory_graph": trajectory_graph,
        "trajectory_summary": trajectory_summary,
        "trajectory_distance_graph": trajectory_distance_graph,
        "trajectory_distance_summary": trajectory_distance_summary,
        "scale_rows": scale_summary,
        "scale_trends": scale_trends,
        "output_coherence_profiles": output_coherence_profiles,
        "output_coherence_expected": output_coherence_expected,
        "output_carriage_audit": output_carriage_audit,
        "output_carriage_failures": output_carriage_failures,
        "beneficial_graph": beneficial_graph,
        "beneficial_summary": beneficial_summary,
        "survival_graph": survival_graph,
        "survival_summary": survival_summary,
        "survival_contrasts": survival_contrasts,
        "survival_by_size": survival_by_size,
    }


def build_parser(
    profile: ReachProfile = ZINC_PROFILE,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "measure", "figures"), default="all")
    parser.add_argument(
        "--output-dir",
        default=profile.default_output_dir,
    )
    parser.add_argument("--tasks", default=",".join(profile.tasks))
    parser.add_argument("--channels", default=",".join(CHANNELS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graphs", type=int, default=64)
    parser.add_argument("--sources-per-graph", type=int, default=6)
    parser.add_argument("--donors-per-source", type=int, default=4)
    parser.add_argument("--semantic-donor-graphs", type=int, default=256)
    parser.add_argument("--bamberger-output-nodes", type=int, default=6)
    parser.add_argument("--bamberger-output-channels", type=int, default=8)
    parser.add_argument(
        "--interpolation-doses",
        default=",".join(f"{value:g}" for value in DEFAULT_INTERPOLATION_DOSES),
    )
    parser.add_argument("--interpolation-batch-size", type=int, default=64)
    parser.add_argument("--survival-carriers-per-graph", type=int, default=1)
    parser.add_argument("--survival-draws", type=int, default=1)
    parser.add_argument("--survival-tail-radii", default="2,3,4,5")
    parser.add_argument("--survival-replacement-candidates", type=int, default=32)
    parser.add_argument("--survival-exact-limit", type=int, default=12)
    parser.add_argument("--survival-random-attempts", type=int, default=512)
    parser.add_argument("--survival-replica-batch-size", type=int, default=2_048)
    parser.add_argument("--beneficial-donors-per-source", type=int, default=1)
    parser.add_argument("--beneficial-atol", type=float, default=1.0e-5)
    parser.add_argument("--beneficial-rtol", type=float, default=1.0e-4)
    parser.add_argument("--beneficial-max-intervals", type=int, default=64)
    parser.add_argument("--effect-floor", type=float, default=1.0e-12)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--analysis-seed", type=int, default=91_021)
    parser.add_argument("--accelerator", default="cuda:0")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--output-carriage",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--beneficial",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--survival",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--scale-analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--skip-dependency-install", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    profile: ReachProfile = ZINC_PROFILE,
) -> dict[str, Any]:
    args = build_parser(profile).parse_args(argv)
    config = ZincReachConfig(
        profile=profile,
        tasks=tuple(value.strip() for value in args.tasks.split(",") if value.strip()),
        channels=tuple(
            value.strip() for value in args.channels.split(",") if value.strip()
        ),
        seed=int(args.seed),
        graphs=int(args.graphs),
        sources_per_graph=int(args.sources_per_graph),
        donors_per_source=int(args.donors_per_source),
        semantic_donor_graphs=int(args.semantic_donor_graphs),
        bamberger_output_nodes=int(args.bamberger_output_nodes),
        bamberger_output_channels=int(args.bamberger_output_channels),
        interpolation_doses=tuple(
            float(value.strip())
            for value in args.interpolation_doses.split(",")
            if value.strip()
        ),
        interpolation_batch_size=int(args.interpolation_batch_size),
        survival_carriers_per_graph=int(args.survival_carriers_per_graph),
        survival_draws=int(args.survival_draws),
        survival_tail_radii=tuple(
            int(value.strip())
            for value in args.survival_tail_radii.split(",")
            if value.strip()
        ),
        survival_replacement_candidates=int(args.survival_replacement_candidates),
        survival_exact_limit=int(args.survival_exact_limit),
        survival_random_attempts=int(args.survival_random_attempts),
        survival_replica_batch_size=int(args.survival_replica_batch_size),
        beneficial_donors_per_source=int(args.beneficial_donors_per_source),
        beneficial_atol=float(args.beneficial_atol),
        beneficial_rtol=float(args.beneficial_rtol),
        beneficial_max_intervals=int(args.beneficial_max_intervals),
        effect_floor=float(args.effect_floor),
        bootstrap_replicates=int(args.bootstrap_replicates),
        analysis_seed=int(args.analysis_seed),
        accelerator=str(args.accelerator),
        num_threads=int(args.num_threads),
        compute_output_carriage=bool(args.output_carriage),
        compute_beneficial=bool(args.beneficial),
        compute_survival=bool(args.survival),
        compute_scale_analysis=bool(args.scale_analysis),
    )
    config.validate()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "analysis_config.json",
        {
            **asdict(config),
            "fingerprint": config.fingerprint,
            "repository_commit": _repository_commit(),
        },
    )
    result: dict[str, Any] = {"config": config, "output_dir": str(output_dir)}
    if args.phase in {"all", "measure"}:
        result["measurement"] = measure(
            config,
            output_dir=output_dir,
            install_dependencies=not bool(args.skip_dependency_install),
            progress=not bool(args.quiet),
        )
    if args.phase in {"all", "figures"}:
        try:
            result.update(
                figures(
                    config,
                    output_dir=output_dir,
                    print_audit=args.phase == "figures",
                )
            )
        except Exception as error:
            failure = {
                "error_type": type(error).__name__,
                "error": str(error),
                "phase": str(args.phase),
            }
            result["figure_failures"] = [failure]
            print(
                f"[figures:warning] figures could not be generated "
                f"({type(error).__name__}: {error}); cached measurements are retained.",
                flush=True,
            )
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
