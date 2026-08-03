"""ZINC semantic, structural, and interaction-carriage analysis.

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
from .methodology.interventions import semantic_donor_swap, structural_donor_swap
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
TERMS = ("semantic", "structural", "interaction")
DEFAULT_TASKS = ("zinc_1hop", "zinc_2hop", "zinc_1hop_vnode", "zinc")


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


def four_state_contrast(
    clean: Any,
    semantic: Any,
    structural: Any,
    joint: Any,
) -> Any:
    return clean - semantic - structural + joint


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
    parser.add_argument("--phase", choices=("all", "measure", "figures"), default="all")
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
    result: dict[str, Any] = {"config": config, "output_dir": str(config.output_dir)}
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
    print(
        json.dumps(
            {
                "output_dir": str(config.output_dir),
                "measurement": result.get("measurement"),
                "figures": sorted(result.get("figures", {})),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
