"""ZINC semantic, structural, and interaction-carriage analysis.

The lightweight output-modulation phase evaluates the same four endpoints with
ordinary batched predictions.  Its statistic ``M`` measures how much the signed
semantic output effect changes under a matched structural swap; it installs no
Jacobians or layer-attribution hooks and caches all endpoints for reanalysis.

The analysis forms paired semantic and structural donor interventions at the same
source node and evaluates the four endpoints ``clean``, ``semantic``,
``structural``, and ``joint``.  The final-state contrast

    clean - semantic - structural + joint

is projected through the clean graph-output Jacobian carrier by carrier.  Raw
effect mass is primary; distance profiles are emitted only above an absolute
and marginal-relative estimability floor.

The same signed contrasts are also formed at every layer's native per-head
transport site and projected through the corresponding clean output Jacobian.
This makes the layer-by-distance analysis a direct extension of the canonical
specialisation-score estimand rather than a second attribution method.

This measures conditional pathway use.  It is not a necessity test.  Dense,
1-hop, 2-hop, and 1-hop-with-virtual-node checkpoints are compared as trained
architectural solutions; similar task error establishes sufficiency only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .methodology.distance import shortest_path_distances
from .methodology.events import build_channel_events
from .methodology.interventions import (
    semantic_donor_swap,
    structural_donor_swap,
    structural_intervention_dose,
)
from .methodology.protocol import stable_hash
from .methodology.runner import prepare_task
from .methodology.sampling import payload_array
from .methodology.scores import project_transport
from .zinc_reach_analysis import (
    TASK_LABELS,
    ZINC_PROFILE,
    ZincReachConfig,
    _methodology_config,
    _project_final_vector,
    _repository_commit,
    _seed,
    checkpoint_registry,
)

PILOT_VERSION = "zinc-semantic-structural-interaction-carriage-v2"
OUTPUT_MODULATION_VERSION = "zinc-output-semantic-structural-modulation-v1"
TERMS = ("semantic", "structural", "interaction")
DEFAULT_TASKS = ("zinc_1hop", "zinc_2hop", "zinc_1hop_vnode", "zinc")
OUTPUT_MODULATION_TASKS = (
    "zinc_1hop_localrrwp",
    "zinc_1hop",
    "zinc_1hop_vnode",
    "zinc_2hop",
    "zinc_2hop_vnode",
    "zinc",
)
OUTPUT_MODULATION_LABELS = {
    "zinc_1hop_localrrwp": "1-hop + local RRWP",
    "zinc_1hop": "1-hop + global RRWP",
    "zinc_1hop_vnode": "1-hop + VN",
    "zinc_2hop": "2-hop",
    "zinc_2hop_vnode": "2-hop + VN",
    "zinc": "Dense + global RRWP",
}


def _output_model_label(task: str) -> str:
    return OUTPUT_MODULATION_LABELS.get(task, TASK_LABELS.get(task, task))


@dataclass(frozen=True)
class PilotConfig:
    output_dir: Path
    tasks: tuple[str, ...] = DEFAULT_TASKS
    seed: int = 0
    graphs: int = 64
    sources_per_graph: int = 6
    donor_pairs_per_source: int = 4
    semantic_donor_graphs: int = 256
    absolute_effect_floor: float = 1.0e-6
    relative_effect_floor: float = 1.0e-3
    far_distance: int = 4
    bootstrap_replicates: int = 2_000
    analysis_seed: int = 260_803
    accelerator: str = "cuda:0"
    num_threads: int = 4

    def validate(self) -> None:
        if not self.tasks or any(task not in ZINC_PROFILE.tasks for task in self.tasks):
            raise ValueError(f"tasks must be drawn from {ZINC_PROFILE.tasks}")
        for name in (
            "graphs",
            "sources_per_graph",
            "donor_pairs_per_source",
            "semantic_donor_graphs",
            "far_distance",
            "bootstrap_replicates",
            "num_threads",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.absolute_effect_floor <= 0 or self.relative_effect_floor <= 0:
            raise ValueError("effect floors must be positive")

    @property
    def scientific_record(self) -> dict[str, Any]:
        return {
            "pilot_version": PILOT_VERSION,
            "tasks": list(self.tasks),
            "seed": int(self.seed),
            "graphs": int(self.graphs),
            "sources_per_graph": int(self.sources_per_graph),
            "donor_pairs_per_source": int(self.donor_pairs_per_source),
            "semantic_donor_graphs": int(self.semantic_donor_graphs),
            "absolute_effect_floor": float(self.absolute_effect_floor),
            "relative_effect_floor": float(self.relative_effect_floor),
            "far_distance": int(self.far_distance),
            "bootstrap_replicates": int(self.bootstrap_replicates),
            "analysis_seed": int(self.analysis_seed),
            "estimand": (
                "clean-output-Jacobian projection of finite semantic, structural, "
                "and semantic-by-structural 2x2 contrasts at final-state carriers "
                "and native layer/head transport sites"
            ),
            "interpretation": "conditional model reliance, not task necessity",
            "aggregation": "donor pair -> source -> graph; retain raw mass and eligibility",
        }

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.scientific_record)


@dataclass(frozen=True)
class OutputModulationConfig:
    """Controls for the cheap exact output-level semantic-context test."""

    output_dir: Path
    tasks: tuple[str, ...] = OUTPUT_MODULATION_TASKS
    seed: int = 0
    graphs: int = 128
    sources_per_graph: int = 6
    donor_pairs_per_source: int = 2
    semantic_donor_graphs: int = 64
    effect_floor: float = 1.0e-6
    graphs_per_batch: int = 8
    bootstrap_replicates: int = 2_000
    analysis_seed: int = 260_803
    accelerator: str = "cuda:0"
    num_threads: int = 4

    def validate(self) -> None:
        if tuple(self.tasks) != OUTPUT_MODULATION_TASKS:
            raise ValueError(
                "output modulation currently requires the ordered tasks "
                f"{OUTPUT_MODULATION_TASKS} so interventions remain exactly paired"
            )
        for name in (
            "graphs",
            "sources_per_graph",
            "donor_pairs_per_source",
            "semantic_donor_graphs",
            "graphs_per_batch",
            "bootstrap_replicates",
            "num_threads",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if float(self.effect_floor) <= 0:
            raise ValueError("effect_floor must be positive")

    @property
    def scientific_record(self) -> dict[str, Any]:
        return {
            "analysis_version": OUTPUT_MODULATION_VERSION,
            "seed": int(self.seed),
            "sources_per_graph": int(self.sources_per_graph),
            "donor_pairs_per_source": int(self.donor_pairs_per_source),
            "semantic_donor_graphs": int(self.semantic_donor_graphs),
            "analysis_seed": int(self.analysis_seed),
            "estimand": (
                "absolute exact output 2x2 contrast divided by the mean absolute "
                "semantic effect under original and swapped structural context"
            ),
            "aggregation": "donor pair -> source -> graph; bootstrap graphs",
            "interpretation": "structural modulation of semantic output effect, not necessity",
        }

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.scientific_record)

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "output_modulation" / self.fingerprint[:12]


def four_state_contrast(
    clean: Any,
    semantic: Any,
    structural: Any,
    joint: Any,
) -> Any:
    return clean - semantic - structural + joint


def output_modulation_metrics(
    clean: float,
    semantic: float,
    structural: float,
    joint: float,
    *,
    effect_floor: float,
) -> dict[str, float | bool]:
    """Return the exact scalar-output semantic modulation statistic ``M``.

    ``semantic_effect_original`` is the semantic donor effect under the clean
    structural context; ``semantic_effect_swapped_structure`` is the same
    semantic donor effect after the matched structural swap.  Their difference
    is exactly the four-state interaction contrast.
    """

    values = np.asarray([clean, semantic, structural, joint], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("output modulation endpoints must be finite")
    semantic_original = float(clean - semantic)
    semantic_swapped = float(structural - joint)
    structural_original = float(clean - structural)
    structural_swapped = float(semantic - joint)
    interaction = float(semantic_original - semantic_swapped)
    structural_identity = float(structural_original - structural_swapped)
    if not np.isclose(interaction, structural_identity, atol=1.0e-10, rtol=1.0e-8):
        raise RuntimeError("the two exact 2x2 interaction identities disagree")
    semantic_reference = 0.5 * (abs(semantic_original) + abs(semantic_swapped))
    estimable = bool(semantic_reference > float(effect_floor))
    modulation = abs(interaction) / semantic_reference if estimable else float("nan")
    return {
        "semantic_effect_original": semantic_original,
        "semantic_effect_swapped_structure": semantic_swapped,
        "structural_effect_original": structural_original,
        "structural_effect_swapped_semantic": structural_swapped,
        "interaction_signed": interaction,
        "interaction_abs": abs(interaction),
        "semantic_reference_abs": semantic_reference,
        "modulation_m": modulation,
        "modulation_estimable": estimable,
    }


def event_distance_summary(
    semantic_mass: np.ndarray,
    structural_mass: np.ndarray,
    interaction_mass: np.ndarray,
    distances: np.ndarray,
    *,
    absolute_floor: float,
    relative_floor: float,
    far_distance: int,
) -> dict[str, float | bool]:
    """Summarise one event without normalising a null interaction."""

    semantic_mass = np.asarray(semantic_mass, dtype=np.float64)
    structural_mass = np.asarray(structural_mass, dtype=np.float64)
    interaction_mass = np.asarray(interaction_mass, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    if not (
        semantic_mass.shape == structural_mass.shape == interaction_mass.shape == distances.shape
    ):
        raise ValueError("carrier fields and distances must align")
    finite = np.isfinite(distances)
    sem_total = float(semantic_mass[finite].sum())
    str_total = float(structural_mass[finite].sum())
    int_total = float(interaction_mass[finite].sum())
    marginal_reference = 0.5 * (sem_total + str_total)
    floor = max(float(absolute_floor), float(relative_floor) * marginal_reference)
    eligible = bool(np.isfinite(int_total) and int_total > floor)

    def profile_stats(mass: np.ndarray) -> tuple[float, float]:
        total = float(mass[finite].sum())
        if total <= 0:
            return float("nan"), float("nan")
        expected = float(np.dot(mass[finite], distances[finite]) / total)
        far = float(mass[finite & (distances >= int(far_distance))].sum() / total)
        return expected, far

    semantic_expected, semantic_far = profile_stats(semantic_mass)
    structural_expected, structural_far = profile_stats(structural_mass)
    interaction_expected, interaction_far = (
        profile_stats(interaction_mass) if eligible else (float("nan"), float("nan"))
    )
    return {
        "semantic_mass": sem_total,
        "structural_mass": str_total,
        "interaction_mass": int_total,
        "marginal_reference_mass": marginal_reference,
        "interaction_relative_mass": (
            int_total / marginal_reference if marginal_reference > 0 else float("nan")
        ),
        "estimability_floor": floor,
        "interaction_estimable": eligible,
        "semantic_expected_distance": semantic_expected,
        "structural_expected_distance": structural_expected,
        "interaction_expected_distance": interaction_expected,
        "semantic_far_share": semantic_far,
        "structural_far_share": structural_far,
        "interaction_far_share": interaction_far,
    }


def layer_event_summary(
    semantic_mass: np.ndarray,
    structural_mass: np.ndarray,
    interaction_mass: np.ndarray,
    distances: np.ndarray,
    *,
    absolute_floor: float,
    relative_floor: float,
    far_distance: int,
    virtual_index: int | None = None,
) -> dict[str, float | bool]:
    """Summarise one layer while retaining a virtual carrier separately.

    Expected graph distance is conditional on real-node carriage.  Far shares
    and virtual shares use all carriers in the denominator, so the virtual-node
    model cannot appear artificially local after its virtual route is removed.
    """

    fields = {
        "semantic": np.asarray(semantic_mass, dtype=np.float64),
        "structural": np.asarray(structural_mass, dtype=np.float64),
        "interaction": np.asarray(interaction_mass, dtype=np.float64),
    }
    shape = fields["semantic"].shape
    if any(value.shape != shape for value in fields.values()):
        raise ValueError("layer carrier fields must align")
    if len(shape) != 1:
        raise ValueError("layer carrier fields must be one-dimensional")
    real_mask = np.ones(shape[0], dtype=bool)
    if virtual_index is not None:
        if not 0 <= int(virtual_index) < shape[0]:
            raise ValueError("virtual carrier index is out of range")
        real_mask[int(virtual_index)] = False
    real_distances = np.asarray(distances, dtype=np.float64)
    if real_distances.shape != (int(real_mask.sum()),):
        raise ValueError("real-node distances do not match layer carriers")
    finite = np.isfinite(real_distances)

    totals = {term: float(value.sum()) for term, value in fields.items()}
    reference = 0.5 * (totals["semantic"] + totals["structural"])
    floor = max(float(absolute_floor), float(relative_floor) * reference)
    interaction_estimable = bool(totals["interaction"] > floor)
    interaction_node_mass = float(fields["interaction"][real_mask][finite].sum())
    interaction_distance_estimable = bool(interaction_estimable and interaction_node_mass > floor)

    output: dict[str, float | bool] = {
        "semantic_mass": totals["semantic"],
        "structural_mass": totals["structural"],
        "interaction_mass": totals["interaction"],
        "marginal_reference_mass": reference,
        "interaction_relative_mass": (
            totals["interaction"] / reference if reference > 0 else float("nan")
        ),
        "estimability_floor": floor,
        "interaction_estimable": interaction_estimable,
        "interaction_distance_estimable": interaction_distance_estimable,
    }
    for term, mass in fields.items():
        real_mass = mass[real_mask][finite]
        real_total = float(real_mass.sum())
        total = totals[term]
        distance_ok = real_total > 0 and (term != "interaction" or interaction_distance_estimable)
        output[f"{term}_expected_distance"] = (
            float(np.dot(real_mass, real_distances[finite]) / real_total)
            if distance_ok
            else float("nan")
        )
        output[f"{term}_far_share"] = (
            float(real_mass[real_distances[finite] >= int(far_distance)].sum() / total)
            if distance_ok and total > 0
            else float("nan")
        )
        virtual_mass = float(mass[int(virtual_index)]) if virtual_index is not None else 0.0
        output[f"{term}_virtual_share"] = (
            virtual_mass / total
            if total > 0 and (term != "interaction" or interaction_estimable)
            else float("nan")
        )
    return output


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _joint_variant(
    prepared: Any,
    semantic_variant: Any,
    structural_variant: Any,
    *,
    source: int,
    structural_donor: int,
) -> Any:
    adapter = prepared.task.content_adapter
    donor_payload = payload_array(semantic_variant, adapter)[int(source)]
    joint = semantic_donor_swap(
        structural_variant,
        int(source),
        donor_payload,
        adapter=adapter,
    )
    # The semantic and structural fields are disjoint by task declaration, so
    # applying the operations in the reverse order must produce the same graph.
    reverse = structural_donor_swap(
        semantic_variant,
        int(source),
        int(structural_donor),
        task=prepared.task,
        duplicate_tolerance=1.0e-7,
    )
    for name in prepared.task.semantic_fields:
        left = getattr(joint, name, None)
        right = getattr(reverse, name, None)
        if left is not None and not bool((left == right).all()):
            raise RuntimeError(f"joint intervention does not commute for {name!r}")
    return joint


def _measure_graph(
    config: PilotConfig,
    prepared: Any,
    *,
    task: str,
    graph_id: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    import torch

    base = prepared.runtime.eval_ds[int(graph_id)]
    source_rng = np.random.default_rng(
        _seed(
            ZincReachConfig(analysis_seed=config.analysis_seed),
            PILOT_VERSION,
            task,
            graph_id,
            "sources",
        )
    )
    sources = tuple(
        int(value)
        for value in source_rng.choice(
            int(base.num_nodes),
            size=min(int(base.num_nodes), int(config.sources_per_graph)),
            replace=False,
        )
    )
    endpoint_graphs: list[Any] = []
    manifests: list[dict[str, Any]] = []
    seed_config = ZincReachConfig(analysis_seed=config.analysis_seed)
    for source in sources:
        semantic_variants, semantic_events = build_channel_events(
            base,
            graph_id=int(graph_id),
            source=int(source),
            channel="semantic",
            stage=PILOT_VERSION,
            donors=int(config.donor_pairs_per_source),
            rng=np.random.default_rng(
                _seed(seed_config, PILOT_VERSION, task, graph_id, source, "semantic")
            ),
            task=prepared.task,
            semantic_pool=prepared.donor_pool,
            duplicate_tolerance=1.0e-7,
        )
        structural_variants, structural_events = build_channel_events(
            base,
            graph_id=int(graph_id),
            source=int(source),
            channel="structural",
            stage=PILOT_VERSION,
            donors=int(config.donor_pairs_per_source),
            rng=np.random.default_rng(
                _seed(seed_config, PILOT_VERSION, task, graph_id, source, "structural")
            ),
            task=prepared.task,
            semantic_pool=prepared.donor_pool,
            duplicate_tolerance=1.0e-7,
        )
        if not (
            len(semantic_variants) == len(structural_variants) == int(config.donor_pairs_per_source)
        ):
            raise RuntimeError("paired donor construction returned the wrong count")
        for pair, (sem_graph, sem_event, str_graph, str_event) in enumerate(
            zip(
                semantic_variants,
                semantic_events,
                structural_variants,
                structural_events,
                strict=True,
            )
        ):
            joint = _joint_variant(
                prepared,
                sem_graph,
                str_graph,
                source=int(source),
                structural_donor=int(str_event.donor_node),
            )
            endpoint_graphs.extend((sem_graph, str_graph, joint))
            manifests.append(
                {
                    "source": int(source),
                    "pair": int(pair),
                    "semantic_donor_graph": int(sem_event.donor_graph_id),
                    "semantic_donor_node": int(sem_event.donor_node),
                    "semantic_dose": float(sem_event.dose),
                    "structural_donor_node": int(str_event.donor_node),
                    "structural_dose": float(str_event.dose),
                }
            )
    if not endpoint_graphs:
        return [], [], [], []

    capture = prepared.backend.capture_groups([[base, *endpoint_graphs]])[0]
    clean_jacobian = prepared.backend.clean_jacobians(base)
    clean_state = capture.final_state[0]
    endpoint_states = capture.final_state[1:].reshape(len(manifests), 3, int(base.num_nodes), -1)
    semantic_state = endpoint_states[:, 0]
    structural_state = endpoint_states[:, 1]
    joint_state = endpoint_states[:, 2]
    semantic_delta = clean_state.unsqueeze(0) - semantic_state
    structural_delta = clean_state.unsqueeze(0) - structural_state
    interaction_delta = four_state_contrast(
        clean_state.unsqueeze(0), semantic_state, structural_state, joint_state
    )
    gradient = clean_jacobian.final_state
    semantic_vector = _project_final_vector(semantic_delta, gradient)
    structural_vector = _project_final_vector(structural_delta, gradient)
    interaction_vector = _project_final_vector(interaction_delta, gradient)
    semantic_mass = torch.linalg.vector_norm(semantic_vector, dim=-1)
    structural_mass = torch.linalg.vector_norm(structural_vector, dim=-1)
    interaction_mass = torch.linalg.vector_norm(interaction_vector, dim=-1)

    transport = torch.stack(capture.transport, dim=1)
    clean_transport = transport[0]
    endpoint_transport = transport[1:].reshape(len(manifests), 3, *transport.shape[1:])
    semantic_transport_delta = clean_transport.unsqueeze(0) - endpoint_transport[:, 0]
    structural_transport_delta = clean_transport.unsqueeze(0) - endpoint_transport[:, 1]
    interaction_transport_delta = four_state_contrast(
        clean_transport.unsqueeze(0),
        endpoint_transport[:, 0],
        endpoint_transport[:, 1],
        endpoint_transport[:, 2],
    )
    semantic_q = project_transport(semantic_transport_delta, clean_jacobian.transport)
    structural_q = project_transport(structural_transport_delta, clean_jacobian.transport)
    interaction_q = project_transport(interaction_transport_delta, clean_jacobian.transport)
    # q is [event, layer, head, carrier, output].  Output norms are taken
    # before summing heads, matching the canonical raw-mass score system.
    semantic_layer_mass = torch.linalg.vector_norm(semantic_q, dim=-1).sum(dim=2)
    structural_layer_mass = torch.linalg.vector_norm(structural_q, dim=-1).sum(dim=2)
    interaction_layer_mass = torch.linalg.vector_norm(interaction_q, dim=-1).sum(dim=2)

    predictions = capture.z.reshape(1 + 3 * len(manifests), -1)
    clean_prediction = predictions[0]
    endpoint_predictions = predictions[1:].reshape(len(manifests), 3, -1)
    output_interaction = four_state_contrast(
        clean_prediction.unsqueeze(0),
        endpoint_predictions[:, 0],
        endpoint_predictions[:, 1],
        endpoint_predictions[:, 2],
    )
    distances = shortest_path_distances(base.edge_index, int(base.num_nodes))
    carrier_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    layer_event_rows: list[dict[str, Any]] = []
    layer_distance_rows: list[dict[str, Any]] = []
    for event_index, manifest in enumerate(manifests):
        source = int(manifest["source"])
        event_distances = distances[source]
        summary = event_distance_summary(
            semantic_mass[event_index].detach().cpu().numpy(),
            structural_mass[event_index].detach().cpu().numpy(),
            interaction_mass[event_index].detach().cpu().numpy(),
            event_distances,
            absolute_floor=config.absolute_effect_floor,
            relative_floor=config.relative_effect_floor,
            far_distance=config.far_distance,
        )
        event_rows.append(
            {
                "pilot_version": PILOT_VERSION,
                "fingerprint": config.fingerprint,
                "task": task,
                "model_label": TASK_LABELS[task],
                "seed": int(config.seed),
                "graph": int(graph_id),
                **manifest,
                **summary,
                "output_interaction_l2": float(
                    torch.linalg.vector_norm(output_interaction[event_index]).detach().cpu()
                ),
                "projected_interaction_l2": float(
                    torch.linalg.vector_norm(interaction_vector[event_index].sum(dim=0))
                    .detach()
                    .cpu()
                ),
            }
        )
        for carrier, distance in enumerate(event_distances):
            if not np.isfinite(distance):
                continue
            carrier_rows.append(
                {
                    "pilot_version": PILOT_VERSION,
                    "fingerprint": config.fingerprint,
                    "task": task,
                    "seed": int(config.seed),
                    "graph": int(graph_id),
                    "source": source,
                    "pair": int(manifest["pair"]),
                    "carrier": int(carrier),
                    "distance": int(distance),
                    "semantic_mass": float(semantic_mass[event_index, carrier].detach().cpu()),
                    "structural_mass": float(structural_mass[event_index, carrier].detach().cpu()),
                    "interaction_mass": float(
                        interaction_mass[event_index, carrier].detach().cpu()
                    ),
                    "interaction_estimable": bool(summary["interaction_estimable"]),
                }
            )

        transport_distances = prepared.backend.transport_distances(
            base,
            source,
            distances,
            channel="semantic",
        )
        carrier_count = int(semantic_layer_mass.shape[-1])
        if len(transport_distances) != carrier_count:
            raise RuntimeError(
                "transport-distance labels do not align with captured carriers: "
                f"{len(transport_distances)} versus {carrier_count}"
            )
        virtual_positions = [
            index for index, value in enumerate(transport_distances) if isinstance(value, str)
        ]
        if len(virtual_positions) > 1:
            raise RuntimeError("the ZINC analysis supports at most one virtual carrier")
        virtual_index = virtual_positions[0] if virtual_positions else None
        real_positions = [index for index in range(carrier_count) if index != virtual_index]
        real_distances = np.asarray(
            [float(transport_distances[index]) for index in real_positions],
            dtype=np.float64,
        )
        for layer in range(int(semantic_layer_mass.shape[1])):
            sem = semantic_layer_mass[event_index, layer].detach().cpu().numpy()
            struct = structural_layer_mass[event_index, layer].detach().cpu().numpy()
            interact = interaction_layer_mass[event_index, layer].detach().cpu().numpy()
            layer_summary = layer_event_summary(
                sem,
                struct,
                interact,
                real_distances,
                absolute_floor=config.absolute_effect_floor,
                relative_floor=config.relative_effect_floor,
                far_distance=config.far_distance,
                virtual_index=virtual_index,
            )
            layer_event_rows.append(
                {
                    "pilot_version": PILOT_VERSION,
                    "fingerprint": config.fingerprint,
                    "task": task,
                    "model_label": TASK_LABELS[task],
                    "seed": int(config.seed),
                    "graph": int(graph_id),
                    **manifest,
                    "layer": int(layer),
                    **layer_summary,
                }
            )
            by_label: dict[str, dict[str, Any]] = {}
            for carrier, raw_label in enumerate(transport_distances):
                is_virtual = isinstance(raw_label, str)
                label = "VN" if is_virtual else str(int(float(raw_label)))
                row = by_label.setdefault(
                    label,
                    {
                        "distance_label": label,
                        "distance": float("nan") if is_virtual else int(float(raw_label)),
                        "distance_order": 10_000 if is_virtual else int(float(raw_label)),
                        "carrier_kind": "virtual" if is_virtual else "molecular_node",
                        "semantic_mass": 0.0,
                        "structural_mass": 0.0,
                        "interaction_mass": 0.0,
                    },
                )
                row["semantic_mass"] += float(sem[carrier])
                row["structural_mass"] += float(struct[carrier])
                row["interaction_mass"] += float(interact[carrier])
            for row in sorted(by_label.values(), key=lambda value: value["distance_order"]):
                layer_distance_rows.append(
                    {
                        "pilot_version": PILOT_VERSION,
                        "fingerprint": config.fingerprint,
                        "task": task,
                        "model_label": TASK_LABELS[task],
                        "seed": int(config.seed),
                        "graph": int(graph_id),
                        "source": source,
                        "pair": int(manifest["pair"]),
                        "layer": int(layer),
                        **row,
                        "interaction_estimable": bool(layer_summary["interaction_estimable"]),
                        "interaction_distance_estimable": bool(
                            layer_summary["interaction_distance_estimable"]
                        ),
                    }
                )
    return event_rows, carrier_rows, layer_event_rows, layer_distance_rows


def measure(
    config: PilotConfig,
    *,
    checkpoints: Mapping[str, str] | None = None,
    install_dependencies: bool = True,
) -> dict[str, Any]:
    config.validate()
    if install_dependencies:
        from .carriage import env

        env.install_dependencies(pyg_version="2.2.0")
        env.apply_compat_patches()
    import torch

    torch.set_num_threads(int(config.num_threads))
    resolved = checkpoint_registry(
        config.tasks,
        seed=int(config.seed),
        overrides=checkpoints,
    )
    reach_config = ZincReachConfig(
        tasks=config.tasks,
        channels=("semantic", "structural"),
        seed=int(config.seed),
        graphs=int(config.graphs),
        sources_per_graph=int(config.sources_per_graph),
        donors_per_source=int(config.donor_pairs_per_source),
        semantic_donor_graphs=int(config.semantic_donor_graphs),
        analysis_seed=int(config.analysis_seed),
        accelerator=str(config.accelerator),
        num_threads=int(config.num_threads),
        compute_output_carriage=False,
        compute_beneficial=False,
        compute_survival=False,
        compute_scale_analysis=False,
    )
    methodology = _methodology_config(
        reach_config,
        output_dir=config.output_dir,
        checkpoints=resolved,
    )
    events: list[dict[str, Any]] = []
    carriers: list[dict[str, Any]] = []
    layer_events: list[dict[str, Any]] = []
    layer_distances: list[dict[str, Any]] = []
    health: list[dict[str, Any]] = []
    for task in config.tasks:
        prepared = prepare_task(methodology, task, int(config.seed), force_fresh_grit=False)
        health.append(
            {
                "task": task,
                "seed": int(config.seed),
                "checkpoint": str(prepared.checkpoint),
                "checkpoint_sha256": str(prepared.checkpoint_sha),
                "test_mae": prepared.runtime.test_metric,
                "validation_mae": prepared.runtime.val_metric,
            }
        )
        graph_ids = tuple(prepared.splits.discovery)[: int(config.graphs)]
        for position, graph_id in enumerate(graph_ids, start=1):
            print(
                f"[interaction] {TASK_LABELS[task]} graph {position}/{len(graph_ids)}",
                flush=True,
            )
            (
                graph_events,
                graph_carriers,
                graph_layer_events,
                graph_layer_distances,
            ) = _measure_graph(
                config,
                prepared,
                task=task,
                graph_id=int(graph_id),
            )
            events.extend(graph_events)
            carriers.extend(graph_carriers)
            layer_events.extend(graph_layer_events)
            layer_distances.extend(graph_layer_distances)
            _write_csv(config.output_dir / "results" / "events.csv", events)
            _write_csv(config.output_dir / "results" / "carriers.csv", carriers)
            _write_csv(config.output_dir / "results" / "layer_events.csv", layer_events)
            _write_csv(
                config.output_dir / "results" / "layer_distance_mass.csv",
                layer_distances,
            )
    _write_csv(config.output_dir / "results" / "model_health.csv", health)
    return {
        "events": len(events),
        "carrier_rows": len(carriers),
        "layer_events": len(layer_events),
        "layer_distance_rows": len(layer_distances),
        "models": len(health),
        "estimable_events": sum(bool(row["interaction_estimable"]) for row in events),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _output_modulation_group(
    config: OutputModulationConfig,
    prepared: Any,
    *,
    graph_id: int,
    reference_manifests: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Build one clean/semantic/structural/joint prediction group.

    Donors are sampled once with the local-RRWP model.  The resulting graph and
    node identities are then replayed for the global-RRWP and dense models, so
    architecture contrasts never conflate model response with a different draw.
    """

    base = prepared.runtime.eval_ds[int(graph_id)]
    if reference_manifests is not None:
        rows = payload_array(base, prepared.task.content_adapter)
        endpoints: list[Any] = []
        manifests: list[dict[str, Any]] = []
        ordered = sorted(
            reference_manifests,
            key=lambda row: (int(row["source"]), int(row["pair"])),
        )
        for reference in ordered:
            source = int(reference["source"])
            donor_graph_id = int(reference["semantic_donor_graph"])
            donor_node_id = int(reference["semantic_donor_node"])
            structural_donor = int(reference["structural_donor_node"])
            donor_nodes = prepared.donor_pool.by_graph.get(donor_graph_id)
            if donor_nodes is None or donor_node_id >= len(donor_nodes):
                raise RuntimeError(
                    "the paired semantic donor is absent from this model's donor pool"
                )
            semantic_donor = donor_nodes[donor_node_id]
            if int(semantic_donor.node) != donor_node_id:
                raise RuntimeError("semantic donor-pool node ordering changed across models")
            semantic_variant = semantic_donor_swap(
                base,
                source,
                semantic_donor.payload,
                adapter=prepared.task.content_adapter,
            )
            structural_variant = structural_donor_swap(
                base,
                source,
                structural_donor,
                task=prepared.task,
                duplicate_tolerance=1.0e-7,
            )
            joint = _joint_variant(
                prepared,
                semantic_variant,
                structural_variant,
                source=source,
                structural_donor=structural_donor,
            )
            endpoints.extend((semantic_variant, structural_variant, joint))
            manifests.append(
                {
                    "source": source,
                    "pair": int(reference["pair"]),
                    "semantic_donor_graph": donor_graph_id,
                    "semantic_donor_node": donor_node_id,
                    "semantic_dose": float(
                        np.linalg.norm(
                            np.asarray(semantic_donor.payload, dtype=np.float64)
                            - np.asarray(rows[source], dtype=np.float64).reshape(-1)
                        )
                    ),
                    "structural_donor_node": structural_donor,
                    "structural_dose": float(
                        structural_intervention_dose(
                            base,
                            structural_variant,
                            prepared.task,
                            tolerance=1.0e-7,
                        )
                    ),
                }
            )
        return [base, *endpoints], manifests

    seed_config = ZincReachConfig(analysis_seed=int(config.analysis_seed))
    source_rng = np.random.default_rng(
        _seed(seed_config, OUTPUT_MODULATION_VERSION, int(graph_id), "sources")
    )
    sources = tuple(
        int(value)
        for value in source_rng.choice(
            int(base.num_nodes),
            size=min(int(base.num_nodes), int(config.sources_per_graph)),
            replace=False,
        )
    )
    endpoints: list[Any] = []
    manifests: list[dict[str, Any]] = []
    for source in sources:
        semantic_variants, semantic_events = build_channel_events(
            base,
            graph_id=int(graph_id),
            source=int(source),
            channel="semantic",
            stage=OUTPUT_MODULATION_VERSION,
            donors=int(config.donor_pairs_per_source),
            rng=np.random.default_rng(
                _seed(
                    seed_config,
                    OUTPUT_MODULATION_VERSION,
                    int(graph_id),
                    int(source),
                    "semantic",
                )
            ),
            task=prepared.task,
            semantic_pool=prepared.donor_pool,
            duplicate_tolerance=1.0e-7,
        )
        structural_variants, structural_events = build_channel_events(
            base,
            graph_id=int(graph_id),
            source=int(source),
            channel="structural",
            stage=OUTPUT_MODULATION_VERSION,
            donors=int(config.donor_pairs_per_source),
            rng=np.random.default_rng(
                _seed(
                    seed_config,
                    OUTPUT_MODULATION_VERSION,
                    int(graph_id),
                    int(source),
                    "structural",
                )
            ),
            task=prepared.task,
            semantic_pool=prepared.donor_pool,
            duplicate_tolerance=1.0e-7,
        )
        if not (
            len(semantic_variants) == len(structural_variants) == int(config.donor_pairs_per_source)
        ):
            raise RuntimeError("paired output-modulation donors have the wrong count")
        for pair, (sem_graph, sem_event, str_graph, str_event) in enumerate(
            zip(
                semantic_variants,
                semantic_events,
                structural_variants,
                structural_events,
                strict=True,
            )
        ):
            joint = _joint_variant(
                prepared,
                sem_graph,
                str_graph,
                source=int(source),
                structural_donor=int(str_event.donor_node),
            )
            endpoints.extend((sem_graph, str_graph, joint))
            manifests.append(
                {
                    "source": int(source),
                    "pair": int(pair),
                    "semantic_donor_graph": int(sem_event.donor_graph_id),
                    "semantic_donor_node": int(sem_event.donor_node),
                    "semantic_dose": float(sem_event.dose),
                    "structural_donor_node": int(str_event.donor_node),
                    "structural_dose": float(str_event.dose),
                }
            )
    return [base, *endpoints], manifests


def _cached_output_manifests(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["task"]) != task:
            continue
        output[int(row["graph"])].append(
            {
                "source": int(row["source"]),
                "pair": int(row["pair"]),
                "semantic_donor_graph": int(row["semantic_donor_graph"]),
                "semantic_donor_node": int(row["semantic_donor_node"]),
                "structural_donor_node": int(row["structural_donor_node"]),
            }
        )
    for graph_rows in output.values():
        graph_rows.sort(key=lambda row: (int(row["source"]), int(row["pair"])))
    return dict(output)


def _prediction_only_groups(prepared: Any, groups: Sequence[Sequence[Any]]) -> list[Any]:
    predict_groups = getattr(prepared.backend, "predict_groups", None)
    if callable(predict_groups):
        return list(predict_groups(groups))
    captures = prepared.backend.capture_groups(groups)
    return [capture.z for capture in captures]


def _output_modulation_rows(
    config: OutputModulationConfig,
    *,
    task: str,
    graph_id: int,
    manifests: Sequence[Mapping[str, Any]],
    predictions: Any,
) -> list[dict[str, Any]]:
    import torch

    values = predictions.reshape(int(predictions.shape[0]), -1)
    if int(values.shape[1]) != 1:
        raise ValueError("ZINC output modulation requires a scalar transformed output")
    expected = 1 + 3 * len(manifests)
    if int(values.shape[0]) != expected:
        raise RuntimeError(
            f"output modulation received {int(values.shape[0])} endpoints; expected {expected}"
        )
    clean = float(values[0, 0].detach().cpu())
    endpoints = values[1:, 0].reshape(len(manifests), 3)
    output: list[dict[str, Any]] = []
    for index, manifest in enumerate(manifests):
        semantic = float(endpoints[index, 0].detach().cpu())
        structural = float(endpoints[index, 1].detach().cpu())
        joint = float(endpoints[index, 2].detach().cpu())
        metrics = output_modulation_metrics(
            clean,
            semantic,
            structural,
            joint,
            effect_floor=float(config.effect_floor),
        )
        output.append(
            {
                "analysis_version": OUTPUT_MODULATION_VERSION,
                "fingerprint": config.fingerprint,
                "task": task,
                "model_label": _output_model_label(task),
                "seed": int(config.seed),
                "graph": int(graph_id),
                **manifest,
                "output_clean": clean,
                "output_semantic": semantic,
                "output_structural": structural,
                "output_joint": joint,
                **metrics,
            }
        )
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("non-finite output modulation predictions")
    return output


def _derive_output_modulation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    effect_floor: float,
) -> list[dict[str, Any]]:
    """Recompute every derived field from cached signed endpoints."""

    output: list[dict[str, Any]] = []
    for row in rows:
        derived = output_modulation_metrics(
            _as_float(row.get("output_clean")),
            _as_float(row.get("output_semantic")),
            _as_float(row.get("output_structural")),
            _as_float(row.get("output_joint")),
            effect_floor=float(effect_floor),
        )
        output.append({**dict(row), **derived})
    return output


def _compatible_output_cache(
    payload: Mapping[str, Any],
    config: OutputModulationConfig,
) -> bool:
    """Whether an older cache used the identical intervention sampling law."""

    integer_fields = (
        "seed",
        "sources_per_graph",
        "donor_pairs_per_source",
        "semantic_donor_graphs",
        "analysis_seed",
    )
    try:
        return all(int(payload[field]) == int(getattr(config, field)) for field in integer_fields)
    except (KeyError, TypeError, ValueError):
        return False


def _merge_compatible_output_caches(
    config: OutputModulationConfig,
    *,
    events: Sequence[Mapping[str, Any]],
    completed: Sequence[Mapping[str, Any]],
    health: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Import compatible task/graph prefixes from earlier cache directories."""

    event_by_key: dict[tuple[str, int, int, int], dict[str, Any]] = {
        (
            str(row["task"]),
            int(row["graph"]),
            int(row["source"]),
            int(row["pair"]),
        ): dict(row)
        for row in events
    }
    completed_keys = {(str(row["task"]), int(row["graph"])) for row in completed}
    health_by_task = {str(row["task"]): dict(row) for row in health}
    imported: list[str] = []
    cache_root = config.output_dir / "output_modulation"
    if not cache_root.is_dir():
        return list(event_by_key.values()), list(completed), list(health), imported
    for directory in sorted(path for path in cache_root.iterdir() if path.is_dir()):
        if directory == config.cache_dir:
            continue
        config_path = directory / "output_modulation_config.json"
        events_path = directory / "output_modulation_events.csv"
        if not config_path.is_file() or not events_path.is_file():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not _compatible_output_cache(payload, config):
            continue
        directory_imported = False
        for row in _read_csv(events_path):
            task = str(row.get("task"))
            if task not in config.tasks or row.get("analysis_version") != OUTPUT_MODULATION_VERSION:
                continue
            key = (
                task,
                int(row["graph"]),
                int(row["source"]),
                int(row["pair"]),
            )
            candidate = {**dict(row), "fingerprint": config.fingerprint}
            existing = event_by_key.get(key)
            if existing is not None:
                identity_fields = (
                    "semantic_donor_graph",
                    "semantic_donor_node",
                    "structural_donor_node",
                )
                endpoint_fields = (
                    "output_clean",
                    "output_semantic",
                    "output_structural",
                    "output_joint",
                )
                if any(
                    int(existing[field]) != int(candidate[field]) for field in identity_fields
                ) or any(
                    not np.isclose(
                        float(existing[field]),
                        float(candidate[field]),
                        atol=1.0e-8,
                        rtol=1.0e-7,
                    )
                    for field in endpoint_fields
                ):
                    raise RuntimeError(f"compatible output caches disagree for event {key}")
            else:
                event_by_key[key] = candidate
                directory_imported = True
        completed_path = directory / "completed_graphs.csv"
        if completed_path.is_file():
            event_graphs = {event_key[:2] for event_key in event_by_key}
            for row in _read_csv(completed_path):
                task = str(row.get("task"))
                key = (task, int(row["graph"]))
                if task in config.tasks and key in event_graphs:
                    completed_keys.add(key)
        health_path = directory / "model_health.csv"
        if health_path.is_file():
            for row in _read_csv(health_path):
                task = str(row.get("task"))
                if task in config.tasks and task not in health_by_task:
                    health_by_task[task] = {
                        **dict(row),
                        "fingerprint": config.fingerprint,
                    }
        if directory_imported:
            imported.append(str(directory))
    completed_rows = [
        {"fingerprint": config.fingerprint, "task": task, "graph": graph}
        for task, graph in sorted(completed_keys)
    ]
    return (
        list(event_by_key.values()),
        completed_rows,
        list(health_by_task.values()),
        imported,
    )


def measure_output_modulation(
    config: OutputModulationConfig,
    *,
    checkpoints: Mapping[str, str] | None = None,
    install_dependencies: bool = True,
) -> dict[str, Any]:
    """Measure exact output modulation with resumable prediction-only batches."""

    config.validate()
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    if install_dependencies:
        from .carriage import env

        env.install_dependencies(pyg_version="2.2.0")
        env.apply_compat_patches()
    import torch

    torch.set_num_threads(int(config.num_threads))
    resolved = checkpoint_registry(
        config.tasks,
        seed=int(config.seed),
        overrides=checkpoints,
    )
    reach_config = ZincReachConfig(
        tasks=config.tasks,
        channels=("semantic", "structural"),
        seed=int(config.seed),
        graphs=int(config.graphs),
        sources_per_graph=int(config.sources_per_graph),
        donors_per_source=int(config.donor_pairs_per_source),
        semantic_donor_graphs=int(config.semantic_donor_graphs),
        analysis_seed=int(config.analysis_seed),
        accelerator=str(config.accelerator),
        num_threads=int(config.num_threads),
        compute_output_carriage=False,
        compute_beneficial=False,
        compute_survival=False,
        compute_scale_analysis=False,
    )
    methodology = _methodology_config(
        reach_config,
        output_dir=config.cache_dir,
        checkpoints=resolved,
    )
    events_path = config.cache_dir / "output_modulation_events.csv"
    completed_path = config.cache_dir / "completed_graphs.csv"
    health_path = config.cache_dir / "model_health.csv"
    cached_events = _read_csv(events_path) if events_path.is_file() else []
    cached_completed = _read_csv(completed_path) if completed_path.is_file() else []
    cached_health = _read_csv(health_path) if health_path.is_file() else []
    for row in [*cached_events, *cached_completed, *cached_health]:
        if row.get("fingerprint") != config.fingerprint:
            raise RuntimeError("cached output-modulation fingerprint does not match config")
    cached_events, cached_completed, cached_health, imported = _merge_compatible_output_caches(
        config,
        events=cached_events,
        completed=cached_completed,
        health=cached_health,
    )
    if imported:
        _write_csv(events_path, cached_events)
        _write_csv(completed_path, cached_completed)
        _write_csv(health_path, cached_health)
        print(
            f"[output-modulation] imported compatible caches from {len(imported)} run(s)",
            flush=True,
        )
    event_by_key: dict[tuple[str, int, int, int], dict[str, Any]] = {
        (
            str(row["task"]),
            int(row["graph"]),
            int(row["source"]),
            int(row["pair"]),
        ): dict(row)
        for row in cached_events
    }
    completed: set[tuple[str, int]] = {
        (str(row["task"]), int(row["graph"])) for row in cached_completed
    }
    health_by_task: dict[str, dict[str, Any]] = {
        str(row["task"]): dict(row) for row in cached_health
    }
    reference_task = config.tasks[0]
    reference_manifests = _cached_output_manifests(
        list(event_by_key.values()),
        task=reference_task,
    )
    reference_graph_ids: tuple[int, ...] | None = None
    for task in config.tasks:
        if sum(task_name == task for task_name, _graph in completed) >= int(config.graphs):
            print(f"[output-modulation] {_output_model_label(task)} already cached", flush=True)
            continue
        prepared = prepare_task(methodology, task, int(config.seed), force_fresh_grit=False)
        graph_ids = tuple(int(value) for value in prepared.splits.discovery)[: int(config.graphs)]
        if reference_graph_ids is None:
            reference_graph_ids = graph_ids
        elif graph_ids != reference_graph_ids:
            raise RuntimeError("output-modulation tasks do not share discovery graph IDs")
        health_by_task[task] = {
            "fingerprint": config.fingerprint,
            "task": task,
            "model_label": _output_model_label(task),
            "seed": int(config.seed),
            "checkpoint": str(prepared.checkpoint),
            "checkpoint_sha256": str(prepared.checkpoint_sha),
            "test_mae": prepared.runtime.test_metric,
            "validation_mae": prepared.runtime.val_metric,
        }
        pending = [graph for graph in graph_ids if (task, int(graph)) not in completed]
        for start in range(0, len(pending), int(config.graphs_per_batch)):
            graph_batch = pending[start : start + int(config.graphs_per_batch)]
            groups: list[list[Any]] = []
            manifests_by_graph: list[list[dict[str, Any]]] = []
            for graph_id in graph_batch:
                if task != reference_task and int(graph_id) not in reference_manifests:
                    raise RuntimeError(
                        f"paired local-RRWP donor manifest is missing for graph {int(graph_id)}"
                    )
                group, manifests = _output_modulation_group(
                    config,
                    prepared,
                    graph_id=int(graph_id),
                    reference_manifests=(
                        None if task == reference_task else reference_manifests.get(int(graph_id))
                    ),
                )
                groups.append(group)
                manifests_by_graph.append(manifests)
            predictions = _prediction_only_groups(prepared, groups)
            if len(predictions) != len(graph_batch):
                raise RuntimeError("prediction-only batch changed the number of graphs")
            for graph_id, manifests, graph_predictions in zip(
                graph_batch,
                manifests_by_graph,
                predictions,
                strict=True,
            ):
                for row in _output_modulation_rows(
                    config,
                    task=task,
                    graph_id=int(graph_id),
                    manifests=manifests,
                    predictions=graph_predictions,
                ):
                    key = (
                        str(row["task"]),
                        int(row["graph"]),
                        int(row["source"]),
                        int(row["pair"]),
                    )
                    event_by_key[key] = row
                completed.add((task, int(graph_id)))
            if task == reference_task:
                reference_manifests = _cached_output_manifests(
                    list(event_by_key.values()),
                    task=reference_task,
                )
            _write_csv(events_path, list(event_by_key.values()))
            _write_csv(
                completed_path,
                [
                    {"fingerprint": config.fingerprint, "task": task_name, "graph": graph}
                    for task_name, graph in sorted(completed)
                ],
            )
            _write_csv(health_path, list(health_by_task.values()))
            print(
                f"[output-modulation] {_output_model_label(task)} "
                f"graphs {min(start + len(graph_batch), len(pending))}/{len(pending)}",
                flush=True,
            )
    return {
        "cache_dir": str(config.cache_dir),
        "events": len(event_by_key),
        "completed_graphs": len(completed),
        "models": len(health_by_task),
    }


def audit_output_modulation_pairing(
    rows: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str],
) -> dict[str, int]:
    """Require matched graphs, sources, and donor identities across models."""

    grouped: dict[tuple[int, int, int], dict[str, tuple[int, int, int]]] = defaultdict(dict)
    for row in rows:
        key = (int(row["graph"]), int(row["source"]), int(row["pair"]))
        grouped[key][str(row["task"])] = (
            int(row["semantic_donor_graph"]),
            int(row["semantic_donor_node"]),
            int(row["structural_donor_node"]),
        )
    complete = 0
    for key, by_task in grouped.items():
        missing = [task for task in tasks if task not in by_task]
        if missing:
            raise RuntimeError(f"output-modulation event {key} is missing tasks {missing}")
        identities = {by_task[task] for task in tasks}
        if len(identities) != 1:
            raise RuntimeError(
                f"output-modulation donor identities differ across tasks for event {key}"
            )
        complete += 1
    return {"paired_events": complete, "models": len(tasks)}


def output_modulation_graph_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate exact output metrics pair -> source -> graph."""

    metrics = (
        "modulation_m",
        "modulation_estimable",
        "interaction_abs",
        "semantic_reference_abs",
        "semantic_effect_original_abs",
        "semantic_effect_swapped_structure_abs",
    )
    source_values: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    for row in rows:
        values = {
            "modulation_m": _as_float(row.get("modulation_m")),
            "modulation_estimable": float(_as_bool(row.get("modulation_estimable"))),
            "interaction_abs": _as_float(row.get("interaction_abs")),
            "semantic_reference_abs": _as_float(row.get("semantic_reference_abs")),
            "semantic_effect_original_abs": abs(_as_float(row.get("semantic_effect_original"))),
            "semantic_effect_swapped_structure_abs": abs(
                _as_float(row.get("semantic_effect_swapped_structure"))
            ),
        }
        for metric in metrics:
            value = values[metric]
            if np.isfinite(value):
                source_values[
                    (
                        str(row["task"]),
                        int(row["graph"]),
                        int(row["source"]),
                        metric,
                    )
                ].append(float(value))
    graph_values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for (task, graph, _source, metric), values in source_values.items():
        graph_values[(task, graph, metric)].append(float(np.mean(values)))
    return [
        {
            "task": task,
            "model_label": _output_model_label(task),
            "graph": int(graph),
            "metric": metric,
            "value": float(np.mean(values)),
            "eligible_sources": len(values),
        }
        for (task, graph, metric), values in sorted(graph_values.items())
    ]


def summarise_output_modulation(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["metric"]))].append(float(row["value"]))
    output: list[dict[str, Any]] = []
    for index, ((task, metric), values) in enumerate(sorted(grouped.items())):
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed) + index,
        )
        output.append(
            {
                "task": task,
                "model_label": _output_model_label(task),
                "metric": metric,
                "mean": mean,
                "low": low,
                "high": high,
                "graphs": len(values),
            }
        )
    return output


def paired_output_modulation_contrasts(
    rows: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["task"]), int(row["graph"]), str(row["metric"])): float(row["value"])
        for row in rows
    }
    output: list[dict[str, Any]] = []
    comparisons = [
        (tasks[left], tasks[right])
        for left in range(len(tasks))
        for right in range(left + 1, len(tasks))
    ]
    metrics = sorted({str(row["metric"]) for row in rows})
    for index, (reference, target) in enumerate(comparisons):
        for metric_index, metric in enumerate(metrics):
            graphs = sorted(
                {
                    int(row["graph"])
                    for row in rows
                    if row["task"] == reference
                    and (target, int(row["graph"]), metric) in lookup
                    and (reference, int(row["graph"]), metric) in lookup
                }
            )
            differences = [
                lookup[(target, graph, metric)] - lookup[(reference, graph, metric)]
                for graph in graphs
            ]
            if not differences:
                continue
            mean, low, high = _bootstrap_interval(
                differences,
                replicates=int(bootstrap_replicates),
                seed=int(bootstrap_seed) + 100 * index + metric_index,
            )
            output.append(
                {
                    "reference_task": reference,
                    "reference_label": _output_model_label(reference),
                    "target_task": target,
                    "target_label": _output_model_label(target),
                    "metric": metric,
                    "mean_difference": mean,
                    "low": low,
                    "high": high,
                    "graphs": len(differences),
                }
            )
    return output


def plot_output_modulation(
    graph_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    figures_dir: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    _figure_theme()
    colours = {
        "zinc_1hop_localrrwp": "#E69F00",
        "zinc_1hop": "#0072B2",
        "zinc_1hop_vnode": "#56B4E9",
        "zinc_2hop": "#009E73",
        "zinc_2hop_vnode": "#CC79A7",
        "zinc": "#D55E00",
    }
    positions = {task: index for index, task in enumerate(tasks)}
    summary_lookup = {(str(row["task"]), str(row["metric"])): row for row in summary_rows}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 3.8))

    modulation = {
        (str(row["task"]), int(row["graph"])): float(row["value"])
        for row in graph_rows
        if row["metric"] == "modulation_m"
    }
    common_graphs = sorted(
        set.intersection(
            *({graph for task_name, graph in modulation if task_name == task} for task in tasks)
        )
    )
    for graph in common_graphs:
        axes[0].plot(
            range(len(tasks)),
            [modulation[(task, graph)] for task in tasks],
            color="#A8A8A8",
            alpha=0.28,
            linewidth=0.7,
            zorder=1,
        )
    plotted_modulation = False
    for task in tasks:
        x = positions[task]
        values = [
            float(row["value"])
            for row in graph_rows
            if row["task"] == task and row["metric"] == "modulation_m"
        ]
        if values:
            jitter = np.linspace(-0.06, 0.06, len(values))
            axes[0].scatter(
                x + jitter,
                values,
                s=12,
                color=colours.get(task, "#555555"),
                alpha=0.35,
                linewidth=0,
                zorder=2,
            )
        summary = summary_lookup.get((task, "modulation_m"))
        if summary is None:
            continue
        plotted_modulation = True
        axes[0].errorbar(
            x,
            float(summary["mean"]),
            yerr=[
                [float(summary["mean"]) - float(summary["low"])],
                [float(summary["high"]) - float(summary["mean"])],
            ],
            fmt="o",
            ms=6,
            capsize=3,
            color=colours.get(task, "#555555"),
            zorder=3,
        )
    if not plotted_modulation:
        axes[0].text(
            0.5,
            0.5,
            "No semantic effects above the floor",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
    axes[0].set_title("Structural modulation of semantic effect")
    axes[0].set_ylabel(r"$M=|s_0-s_1|/[0.5(|s_0|+|s_1|)]$")
    axes[0].set_ylim(0, 2.05)

    component_metrics = (
        ("semantic_reference_abs", "Mean semantic effect"),
        ("interaction_abs", "Interaction difference"),
    )
    offsets = (-0.10, 0.10)
    for task in tasks:
        for offset, (metric, label) in zip(offsets, component_metrics, strict=True):
            summary = summary_lookup[(task, metric)]
            axes[1].errorbar(
                positions[task] + offset,
                float(summary["mean"]),
                yerr=[
                    [float(summary["mean"]) - float(summary["low"])],
                    [float(summary["high"]) - float(summary["mean"])],
                ],
                fmt="o" if metric == "semantic_reference_abs" else "s",
                ms=5.5,
                capsize=3,
                color=colours.get(task, "#555555"),
                fillstyle="full" if metric == "semantic_reference_abs" else "none",
                label=label if task == tasks[0] else None,
            )
    axes[1].set_title("Absolute transformed-output effects")
    axes[1].set_ylabel("Mean absolute effect")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False, loc="best")

    for task in tasks:
        summary = summary_lookup[(task, "modulation_estimable")]
        axes[2].errorbar(
            positions[task],
            float(summary["mean"]),
            yerr=[
                [float(summary["mean"]) - float(summary["low"])],
                [float(summary["high"]) - float(summary["mean"])],
            ],
            fmt="o",
            ms=6,
            capsize=3,
            color=colours.get(task, "#555555"),
        )
    axes[2].set_title("Estimable paired interventions")
    axes[2].set_ylabel("Fraction")
    axes[2].set_ylim(0, 1.02)

    labels = [_output_model_label(task) for task in tasks]
    for axis in axes:
        axis.set_xticks(range(len(tasks)), labels, rotation=24, ha="right")
    fig.suptitle("Does structural context change the semantic output effect?", y=1.02, fontsize=12)
    fig.tight_layout()
    return _save_figure(fig, figures_dir, "zinc_output_semantic_modulation")


def figures_output_modulation(config: OutputModulationConfig) -> dict[str, Any]:
    """Build the exact-output M analysis entirely from its cached endpoints."""

    events_path = config.cache_dir / "output_modulation_events.csv"
    if not events_path.is_file():
        raise FileNotFoundError(
            "cached output-modulation endpoints are missing; run phase='output-measure' first"
        )
    events = _read_csv(events_path)
    if any(row.get("fingerprint") != config.fingerprint for row in events):
        raise RuntimeError("cached output-modulation fingerprint does not match config")
    if not events:
        raise RuntimeError("cached output-modulation endpoints are empty")
    events = _derive_output_modulation_rows(events, effect_floor=float(config.effect_floor))
    pairing = audit_output_modulation_pairing(events, tasks=config.tasks)
    graph_rows = output_modulation_graph_metrics(events)
    summary_rows = summarise_output_modulation(
        graph_rows,
        bootstrap_replicates=int(config.bootstrap_replicates),
        bootstrap_seed=int(config.analysis_seed) + 1_000,
    )
    contrasts = paired_output_modulation_contrasts(
        graph_rows,
        tasks=config.tasks,
        bootstrap_replicates=int(config.bootstrap_replicates),
        bootstrap_seed=int(config.analysis_seed) + 2_000,
    )
    _write_csv(config.cache_dir / "output_modulation_graph_metrics.csv", graph_rows)
    _write_csv(config.cache_dir / "output_modulation_derived_events.csv", events)
    _write_csv(config.cache_dir / "output_modulation_summary.csv", summary_rows)
    _write_csv(config.cache_dir / "output_modulation_paired_contrasts.csv", contrasts)
    paths = {
        "output_modulation": plot_output_modulation(
            graph_rows,
            summary_rows,
            tasks=config.tasks,
            figures_dir=config.cache_dir / "figures",
        )
    }
    _write_json(
        config.cache_dir / "figure_manifest.json",
        {
            "analysis_version": OUTPUT_MODULATION_VERSION,
            "fingerprint": config.fingerprint,
            "pairing_audit": pairing,
            "figures": paths,
            "interpretation": (
                "M is the exact transformed-output semantic effect difference between "
                "original and swapped structural contexts. Endpoints are paired across "
                "models; donor pair -> source -> graph aggregation precedes graph bootstrap. "
                "M measures model response, not task necessity."
            ),
        },
    )
    return {
        "output_modulation_figures": paths,
        "output_modulation_cache_dir": str(config.cache_dir),
        "output_modulation_pairing": pairing,
        "output_modulation_summary": summary_rows,
        "output_modulation_contrasts": contrasts,
    }


def graph_distance_profiles(
    rows: Sequence[Mapping[str, Any]],
    *,
    layerwise: bool,
) -> list[dict[str, Any]]:
    """Normalise event profiles, then average pair -> source -> graph."""

    if not rows:
        return []
    event_fields: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    label_metadata: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        task = str(row["task"])
        layer = int(row["layer"]) if layerwise else -1
        event_key = (
            task,
            int(row["graph"]),
            int(row["source"]),
            int(row["pair"]),
            layer,
        )
        event_fields[event_key].append(row)
        label = str(row.get("distance_label", row.get("distance")))
        distance = _as_float(row.get("distance", label))
        is_virtual = str(row.get("carrier_kind", "")) == "virtual" or label == "VN"
        label_metadata[(task, layer)][label] = {
            "distance_label": label,
            "distance": float("nan") if is_virtual else distance,
            "distance_order": 10_000 if is_virtual else int(distance),
            "carrier_kind": "virtual" if is_virtual else "molecular_node",
        }

    event_profiles: dict[tuple[Any, ...], dict[str, float]] = {}
    for event_key, event_rows in event_fields.items():
        for term in TERMS:
            eligible = True
            if term == "interaction":
                eligible = _as_bool(event_rows[0].get("interaction_estimable", False))
            masses: dict[str, float] = defaultdict(float)
            for row in event_rows:
                label = str(row.get("distance_label", row.get("distance")))
                masses[label] += _as_float(row[f"{term}_mass"])
            total = float(sum(masses.values()))
            if not eligible or not np.isfinite(total) or total <= 0:
                continue
            event_profiles[(*event_key, term)] = {
                label: value / total for label, value in masses.items()
            }

    source_buckets: dict[tuple[Any, ...], list[dict[str, float]]] = defaultdict(list)
    for key, profile in event_profiles.items():
        task, graph, source, _pair, layer, term = key
        source_buckets[(task, graph, source, layer, term)].append(profile)
    source_profiles: dict[tuple[Any, ...], dict[str, float]] = {}
    for key, profiles in source_buckets.items():
        task, _graph, _source, layer, _term = key
        labels = label_metadata[(task, layer)]
        source_profiles[key] = {
            label: float(np.mean([profile.get(label, 0.0) for profile in profiles]))
            for label in labels
        }

    graph_buckets: dict[tuple[Any, ...], list[dict[str, float]]] = defaultdict(list)
    for key, profile in source_profiles.items():
        task, graph, _source, layer, term = key
        graph_buckets[(task, graph, layer, term)].append(profile)
    output: list[dict[str, Any]] = []
    for (task, graph, layer, term), profiles in graph_buckets.items():
        labels = label_metadata[(task, layer)]
        for label, metadata in labels.items():
            output.append(
                {
                    "task": task,
                    "model_label": TASK_LABELS[task],
                    "graph": int(graph),
                    "layer": int(layer),
                    "term": term,
                    **metadata,
                    "share": float(np.mean([profile.get(label, 0.0) for profile in profiles])),
                    "eligible_sources": len(profiles),
                }
            )
    return sorted(
        output,
        key=lambda row: (
            DEFAULT_TASKS.index(row["task"]) if row["task"] in DEFAULT_TASKS else 99,
            row["graph"],
            row["layer"],
            TERMS.index(row["term"]),
            row["distance_order"],
        ),
    )


def graph_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    layerwise: bool,
) -> list[dict[str, Any]]:
    """Average scalar event metrics with the registered hierarchy."""

    if layerwise:
        metrics = (
            *(f"{term}_mass" for term in TERMS),
            "interaction_relative_mass",
            "interaction_estimable",
            "interaction_distance_estimable",
            *(f"{term}_expected_distance" for term in TERMS),
            *(f"{term}_far_share" for term in TERMS),
            *(f"{term}_virtual_share" for term in TERMS),
        )
    else:
        metrics = (
            *(f"{term}_mass" for term in TERMS),
            "interaction_relative_mass",
            "interaction_estimable",
            *(f"{term}_expected_distance" for term in TERMS),
            *(f"{term}_far_share" for term in TERMS),
            "output_interaction_l2",
            "projected_interaction_l2",
        )
    source_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        task = str(row["task"])
        graph = int(row["graph"])
        source = int(row["source"])
        layer = int(row["layer"]) if layerwise else -1
        for metric in metrics:
            if metric not in row:
                continue
            value = (
                float(_as_bool(row[metric]))
                if metric in {"interaction_estimable", "interaction_distance_estimable"}
                else _as_float(row[metric])
            )
            if np.isfinite(value):
                source_values[(task, graph, source, layer, metric)].append(value)
    graph_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for (task, graph, _source, layer, metric), values in source_values.items():
        graph_values[(task, graph, layer, metric)].append(float(np.mean(values)))
    return [
        {
            "task": task,
            "model_label": TASK_LABELS[task],
            "graph": int(graph),
            "layer": int(layer),
            "metric": metric,
            "value": float(np.mean(values)),
            "eligible_sources": len(values),
        }
        for (task, graph, layer, metric), values in sorted(graph_values.items())
    ]


def _bootstrap_interval(
    values: Sequence[float], *, replicates: int, seed: int
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan"), float("nan")
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(int(seed))
    sampled = rng.choice(array, size=(int(replicates), len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return mean, float(low), float(high)


def summarise_graph_profiles(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    metadata: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            row["task"],
            int(row["layer"]),
            row["term"],
            row["distance_label"],
        )
        grouped[key].append(float(row["share"]))
        metadata[key] = row
    output: list[dict[str, Any]] = []
    for index, (key, values) in enumerate(sorted(grouped.items())):
        task, layer, term, label = key
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed) + index,
        )
        row = metadata[key]
        output.append(
            {
                "task": task,
                "model_label": TASK_LABELS[str(task)],
                "layer": int(layer),
                "term": term,
                "distance_label": label,
                "distance": row["distance"],
                "distance_order": int(row["distance_order"]),
                "carrier_kind": row["carrier_kind"],
                "mean": mean,
                "low": low,
                "high": high,
                "graphs": len(values),
            }
        )
    return output


def summarise_graph_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], int(row["layer"]), row["metric"])].append(float(row["value"]))
    output: list[dict[str, Any]] = []
    for index, ((task, layer, metric), values) in enumerate(sorted(grouped.items())):
        mean, low, high = _bootstrap_interval(
            values,
            replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed) + index,
        )
        output.append(
            {
                "task": task,
                "model_label": TASK_LABELS[str(task)],
                "layer": int(layer),
                "metric": metric,
                "mean": mean,
                "low": low,
                "high": high,
                "graphs": len(values),
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


def _task_colours(tasks: Sequence[str]) -> dict[str, str]:
    palette = ("#3767A6", "#2A9D8F", "#E09F3E", "#B4475A")
    return {task: palette[index % len(palette)] for index, task in enumerate(tasks)}


def plot_final_profiles(
    rows: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    figures_dir: Path,
    display_max_distance: int | None,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    _figure_theme()
    colours = _task_colours(tasks)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), sharey=True)
    for axis, term in zip(axes, TERMS, strict=True):
        for task in tasks:
            selected = [
                row
                for row in rows
                if row["task"] == task
                and row["term"] == term
                and row["carrier_kind"] == "molecular_node"
                and (
                    display_max_distance is None
                    or int(row["distance_order"]) <= int(display_max_distance)
                )
            ]
            selected.sort(key=lambda row: int(row["distance_order"]))
            if not selected:
                continue
            x = np.asarray([int(row["distance_order"]) for row in selected])
            y = np.asarray([float(row["mean"]) for row in selected])
            low = np.asarray([float(row["low"]) for row in selected])
            high = np.asarray([float(row["high"]) for row in selected])
            axis.plot(x, y, marker="o", ms=3, lw=1.6, color=colours[task], label=TASK_LABELS[task])
            axis.fill_between(x, low, high, color=colours[task], alpha=0.14, linewidth=0)
        axis.set_title(f"{term.capitalize()} carriage")
        axis.set_xlabel("Graph distance from intervention")
        axis.set_ylim(bottom=0)
    axes[0].set_ylabel("Mean allocation share")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(tasks),
        frameon=False,
    )
    fig.suptitle("Final-state carriage allocation", y=1.13, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save_figure(fig, figures_dir, "zinc_interaction_final_profiles")


def plot_layer_heatmaps(
    rows: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    figures_dir: Path,
    display_max_distance: int | None,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    _figure_theme()
    numeric = sorted(
        {
            int(row["distance_order"])
            for row in rows
            if row["carrier_kind"] == "molecular_node"
            and (
                display_max_distance is None
                or int(row["distance_order"]) <= int(display_max_distance)
            )
        }
    )
    include_virtual = any(row["carrier_kind"] == "virtual" for row in rows)
    labels = [str(value) for value in numeric] + (["VN"] if include_virtual else [])
    label_index = {label: index for index, label in enumerate(labels)}
    layers = sorted({int(row["layer"]) for row in rows})
    matrices: dict[tuple[str, str], np.ndarray] = {}
    maximum = 0.0
    for task in tasks:
        for term in TERMS:
            matrix = np.full((len(layers), len(labels)), np.nan, dtype=np.float64)
            for row in rows:
                if row["task"] != task or row["term"] != term:
                    continue
                label = str(row["distance_label"])
                if label not in label_index:
                    continue
                matrix[layers.index(int(row["layer"])), label_index[label]] = float(row["mean"])
            matrices[(task, term)] = matrix
            if np.isfinite(matrix).any():
                maximum = max(maximum, float(np.nanmax(matrix)))
    fig, axes = plt.subplots(len(tasks), 3, figsize=(10.8, 2.0 * len(tasks) + 1.0), squeeze=False)
    image = None
    for row_index, task in enumerate(tasks):
        for column, term in enumerate(TERMS):
            axis = axes[row_index, column]
            image = axis.imshow(
                matrices[(task, term)],
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap="magma",
                vmin=0,
                vmax=maximum if maximum > 0 else 1,
            )
            axis.set_xticks(range(len(labels)), labels)
            axis.set_yticks(range(len(layers)), [value + 1 for value in layers])
            if row_index == 0:
                axis.set_title(term.capitalize())
            if column == 0:
                axis.set_ylabel(f"{TASK_LABELS[task]}\nLayer")
            if row_index == len(tasks) - 1:
                axis.set_xlabel("Distance (VN separate)")
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.018, pad=0.02, label="Mean allocation share")
    fig.suptitle(
        "Where finite semantic and structural changes enter layer transports", y=1.01, fontsize=12
    )
    return _save_figure(fig, figures_dir, "zinc_interaction_layer_heatmaps")


def plot_layer_reach(
    rows: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    figures_dir: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    _figure_theme()
    colours = _task_colours(tasks)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), sharey=True)
    for axis, term in zip(axes, TERMS, strict=True):
        metric = f"{term}_expected_distance"
        for task in tasks:
            selected = [row for row in rows if row["task"] == task and row["metric"] == metric]
            selected.sort(key=lambda row: int(row["layer"]))
            if not selected:
                continue
            x = np.asarray([int(row["layer"]) + 1 for row in selected])
            y = np.asarray([float(row["mean"]) for row in selected])
            low = np.asarray([float(row["low"]) for row in selected])
            high = np.asarray([float(row["high"]) for row in selected])
            axis.plot(x, y, marker="o", ms=3, lw=1.6, color=colours[task], label=TASK_LABELS[task])
            axis.fill_between(x, low, high, color=colours[task], alpha=0.14, linewidth=0)
        axis.set_title(term.capitalize())
        axis.set_xlabel("Layer")
        axis.set_ylim(bottom=0)
    axes[0].set_ylabel("Expected physical distance")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(tasks),
        frameon=False,
    )
    fig.suptitle("Layer-by-distance reach (real-node carriage only)", y=1.13, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save_figure(fig, figures_dir, "zinc_interaction_layer_reach")


def plot_interaction_strength(
    rows: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    figures_dir: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    _figure_theme()
    colours = _task_colours(tasks)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    for axis, metric, title, ylabel in (
        (
            axes[0],
            "interaction_relative_mass",
            "Interaction relative to marginal carriage",
            r"$C_{int}/[(C_{sem}+C_{str})/2]$",
        ),
        (
            axes[1],
            "interaction_estimable",
            "Fraction of paired interventions estimable",
            "Estimable fraction",
        ),
    ):
        for task in tasks:
            selected = [row for row in rows if row["task"] == task and row["metric"] == metric]
            selected.sort(key=lambda row: int(row["layer"]))
            if not selected:
                continue
            x = np.asarray([int(row["layer"]) + 1 for row in selected])
            y = np.asarray([float(row["mean"]) for row in selected])
            low = np.asarray([float(row["low"]) for row in selected])
            high = np.asarray([float(row["high"]) for row in selected])
            axis.plot(x, y, marker="o", ms=3, lw=1.6, color=colours[task], label=TASK_LABELS[task])
            axis.fill_between(x, low, high, color=colours[task], alpha=0.14, linewidth=0)
        axis.set_title(title)
        axis.set_xlabel("Layer")
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(tasks),
        frameon=False,
    )
    fig.suptitle(
        "Strength and estimability of semantic–structural interaction", y=1.13, fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save_figure(fig, figures_dir, "zinc_interaction_strength")


def figures(
    config: PilotConfig,
    *,
    display_max_distance: int | None = None,
) -> dict[str, Any]:
    """Rebuild all summaries and figures from cached measurements."""

    results_dir = config.output_dir / "results"
    required = {
        "events": results_dir / "events.csv",
        "carriers": results_dir / "carriers.csv",
        "layer_events": results_dir / "layer_events.csv",
        "layer_distances": results_dir / "layer_distance_mass.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "cached measurement CSVs are missing; run PHASE='measure' or 'all' first: "
            + ", ".join(missing)
        )
    final_graph = graph_distance_profiles(_read_csv(required["carriers"]), layerwise=False)
    layer_graph = graph_distance_profiles(_read_csv(required["layer_distances"]), layerwise=True)
    final_metrics_graph = graph_metric_rows(_read_csv(required["events"]), layerwise=False)
    layer_metrics_graph = graph_metric_rows(_read_csv(required["layer_events"]), layerwise=True)
    final_summary = summarise_graph_profiles(
        final_graph,
        bootstrap_replicates=config.bootstrap_replicates,
        bootstrap_seed=config.analysis_seed + 100,
    )
    layer_summary = summarise_graph_profiles(
        layer_graph,
        bootstrap_replicates=config.bootstrap_replicates,
        bootstrap_seed=config.analysis_seed + 200,
    )
    final_metric_summary = summarise_graph_metrics(
        final_metrics_graph,
        bootstrap_replicates=config.bootstrap_replicates,
        bootstrap_seed=config.analysis_seed + 300,
    )
    layer_metric_summary = summarise_graph_metrics(
        layer_metrics_graph,
        bootstrap_replicates=config.bootstrap_replicates,
        bootstrap_seed=config.analysis_seed + 400,
    )
    for name, values in (
        ("final_graph_profiles.csv", final_graph),
        ("final_profile_summary.csv", final_summary),
        ("layer_graph_profiles.csv", layer_graph),
        ("layer_profile_summary.csv", layer_summary),
        ("final_graph_metrics.csv", final_metrics_graph),
        ("final_metric_summary.csv", final_metric_summary),
        ("layer_graph_metrics.csv", layer_metrics_graph),
        ("layer_metric_summary.csv", layer_metric_summary),
    ):
        _write_csv(results_dir / name, values)
    figures_dir = config.output_dir / "figures"
    paths = {
        "final_profiles": plot_final_profiles(
            final_summary,
            tasks=config.tasks,
            figures_dir=figures_dir,
            display_max_distance=display_max_distance,
        ),
        "layer_heatmaps": plot_layer_heatmaps(
            layer_summary,
            tasks=config.tasks,
            figures_dir=figures_dir,
            display_max_distance=display_max_distance,
        ),
        "layer_reach": plot_layer_reach(
            layer_metric_summary,
            tasks=config.tasks,
            figures_dir=figures_dir,
        ),
        "interaction_strength": plot_interaction_strength(
            layer_metric_summary,
            tasks=config.tasks,
            figures_dir=figures_dir,
        ),
    }
    _write_json(
        results_dir / "figure_manifest.json",
        {
            "pilot_version": PILOT_VERSION,
            "fingerprint": config.fingerprint,
            "figures": paths,
            "interpretation": (
                "All intervals bootstrap held-out graphs from one seed-0 checkpoint per "
                "architecture. Profiles normalise each finite donor pair before averaging "
                "pair -> source -> graph. Interaction profiles condition on the registered "
                "estimability floor. Layer transport mass takes output norms before summing "
                "heads; layers are diagnostic sites and are not additive causal stages. "
                "Virtual-node allocation is retained as a separate VN column."
            ),
        },
    )
    return {
        "figures": paths,
        "final_profile_summary": final_summary,
        "layer_profile_summary": layer_summary,
        "final_metric_summary": final_metric_summary,
        "layer_metric_summary": layer_metric_summary,
    }


def _parse_mapping(values: Sequence[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or not key or not path:
            raise ValueError("checkpoint overrides must be TASK:SEED=/path/to/checkpoint")
        output[key] = path
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "all",
            "measure",
            "figures",
            "output-all",
            "output-measure",
            "output-figures",
        ),
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/zinc_semantic_structural_interaction_carriage_v2"),
    )
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graphs", type=int, default=64)
    parser.add_argument("--sources-per-graph", type=int, default=6)
    parser.add_argument("--donor-pairs-per-source", type=int, default=4)
    parser.add_argument("--semantic-donor-graphs", type=int, default=256)
    parser.add_argument("--absolute-effect-floor", type=float, default=1.0e-6)
    parser.add_argument("--relative-effect-floor", type=float, default=1.0e-3)
    parser.add_argument("--far-distance", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--analysis-seed", type=int, default=260_803)
    parser.add_argument("--accelerator", default="cuda:0")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--display-max-distance",
        type=int,
        default=0,
        help="Plot numeric distances through this value; 0 retains all distances.",
    )
    parser.add_argument("--output-tasks", default=",".join(OUTPUT_MODULATION_TASKS))
    parser.add_argument("--output-graphs", type=int, default=128)
    parser.add_argument("--output-sources-per-graph", type=int, default=6)
    parser.add_argument("--output-donor-pairs-per-source", type=int, default=2)
    parser.add_argument("--output-semantic-donor-graphs", type=int, default=64)
    parser.add_argument("--output-effect-floor", type=float, default=1.0e-6)
    parser.add_argument("--output-graphs-per-batch", type=int, default=8)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--skip-dependency-install", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    config = PilotConfig(
        output_dir=args.output_dir,
        tasks=tuple(value.strip() for value in args.tasks.split(",") if value.strip()),
        seed=int(args.seed),
        graphs=int(args.graphs),
        sources_per_graph=int(args.sources_per_graph),
        donor_pairs_per_source=int(args.donor_pairs_per_source),
        semantic_donor_graphs=int(args.semantic_donor_graphs),
        absolute_effect_floor=float(args.absolute_effect_floor),
        relative_effect_floor=float(args.relative_effect_floor),
        far_distance=int(args.far_distance),
        bootstrap_replicates=int(args.bootstrap_replicates),
        analysis_seed=int(args.analysis_seed),
        accelerator=str(args.accelerator),
        num_threads=int(args.num_threads),
    )
    config.validate()
    output_config = OutputModulationConfig(
        output_dir=args.output_dir,
        tasks=tuple(value.strip() for value in args.output_tasks.split(",") if value.strip()),
        seed=int(args.seed),
        graphs=int(args.output_graphs),
        sources_per_graph=int(args.output_sources_per_graph),
        donor_pairs_per_source=int(args.output_donor_pairs_per_source),
        semantic_donor_graphs=int(args.output_semantic_donor_graphs),
        effect_floor=float(args.output_effect_floor),
        graphs_per_batch=int(args.output_graphs_per_batch),
        bootstrap_replicates=int(args.bootstrap_replicates),
        analysis_seed=int(args.analysis_seed),
        accelerator=str(args.accelerator),
        num_threads=int(args.num_threads),
    )
    output_config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        config.output_dir / "analysis_config.json",
        {
            **asdict(config),
            "fingerprint": config.fingerprint,
            "repository_commit": _repository_commit(),
            "pre_registered_decision": {
                "primary": (
                    "report final-state semantic, structural, and interaction distance "
                    "profiles, plus layer/head-transport profiles using the same contrasts"
                ),
                "distance": "report only for events above both estimability floors",
                "necessity": "make no necessity claim; compare checkpoint test MAE only",
                "aggregation": "donor pair -> source -> graph; bootstrap graphs",
                "layer_caveat": "layer sites are comparable diagnostics, not additive stages",
            },
        },
    )
    output_config.cache_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_config.cache_dir / "output_modulation_config.json",
        {
            **asdict(output_config),
            "fingerprint": output_config.fingerprint,
            "repository_commit": _repository_commit(),
            "cached_endpoints": (
                "clean, semantic-only, structural-only, and joint transformed predictions"
            ),
            "pre_registered_decision": {
                "primary": (
                    "M = absolute semantic-effect change between original and swapped "
                    "structural contexts, divided by their mean absolute magnitude"
                ),
                "pairing": (
                    "sample graph/source/donor identities once under local RRWP and replay "
                    "the exact identities under global-RRWP 1-hop and dense models"
                ),
                "aggregation": "donor pair -> source -> graph; bootstrap graphs",
                "interpretation": "model response, not task necessity",
            },
        },
    )
    result: dict[str, Any] = {
        "config": config,
        "output_dir": str(config.output_dir),
        "output_modulation_cache_dir": str(output_config.cache_dir),
    }
    if args.phase in {"all", "measure"}:
        result["measurement"] = measure(
            config,
            checkpoints=_parse_mapping(args.checkpoint),
            install_dependencies=not bool(args.skip_dependency_install),
        )
    if args.phase in {"all", "figures"}:
        result.update(
            figures(
                config,
                display_max_distance=(
                    int(args.display_max_distance) if int(args.display_max_distance) > 0 else None
                ),
            )
        )
    if args.phase in {"output-all", "output-measure"}:
        result["output_modulation_measurement"] = measure_output_modulation(
            output_config,
            checkpoints=_parse_mapping(args.checkpoint),
            install_dependencies=not bool(args.skip_dependency_install),
        )
    if args.phase in {"output-all", "output-figures"}:
        result.update(figures_output_modulation(output_config))
    print(
        json.dumps(
            {
                "output_dir": str(config.output_dir),
                "measurement": result.get("measurement"),
                "figures": sorted(result.get("figures", {})),
                "output_modulation_measurement": result.get("output_modulation_measurement"),
                "output_modulation_cache_dir": result.get("output_modulation_cache_dir"),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
