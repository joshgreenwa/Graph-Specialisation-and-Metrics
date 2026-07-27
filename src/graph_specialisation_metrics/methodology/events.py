"""Build stage-independent semantic and structural donor events."""

from __future__ import annotations

from typing import Any

import numpy as np

from .interventions import (
    semantic_donor_swap,
    structural_donor_swap,
    structural_footprints,
    structural_intervention_dose,
    verify_semantic_swap,
)
from .protocol import stable_hash
from .sampling import (
    DonorEvent,
    SemanticDonorPool,
    draw_structural_donors,
    node_degrees,
    payload_array,
)


def build_channel_events(
    base: Any,
    *,
    graph_id: int,
    source: int,
    channel: str,
    stage: str,
    donors: int,
    rng: np.random.Generator,
    task: Any,
    semantic_pool: SemanticDonorPool,
    duplicate_tolerance: float,
) -> tuple[list[Any], list[DonorEvent]]:
    source = int(source)
    degrees = node_degrees(base)
    variants: list[Any] = []
    records: list[DonorEvent] = []
    if channel == "semantic":
        rows = payload_array(base, task.content_adapter)
        selected = semantic_pool.draw(
            rows[source],
            int(degrees[source]),
            donors,
            rng,
            # Donor and base ID spaces can overlap numerically even when their datasets differ.
            base_graph_id=None,
        )
        for draw, donor in enumerate(selected):
            event = semantic_donor_swap(
                base,
                source,
                donor.payload,
                adapter=task.content_adapter,
            )
            verify_semantic_swap(base, event, source, donor.payload, task=task)
            variants.append(event)
            records.append(
                DonorEvent(
                    channel=channel,
                    stage=stage,
                    graph_id=int(graph_id),
                    source=source,
                    donor_graph_id=donor.graph_id,
                    donor_node=donor.node,
                    source_degree=int(degrees[source]),
                    donor_degree=donor.degree,
                    degree_gap=abs(int(degrees[source]) - donor.degree),
                    dose=float(
                        np.linalg.norm(
                            np.asarray(donor.payload, dtype=np.float64)
                            - np.asarray(rows[source], dtype=np.float64).reshape(-1)
                        )
                    ),
                    payload_fingerprint=stable_hash({"payload": donor.payload}),
                    draw=draw,
                )
            )
    elif channel == "structural":
        footprints = structural_footprints(
            base, task, tolerance=float(duplicate_tolerance)
        )
        selected = draw_structural_donors(
            footprints,
            source,
            donors,
            rng,
            equal=lambda left, right: left == right,
        )
        for draw, donor in enumerate(selected):
            donor = int(donor)
            event = structural_donor_swap(
                base,
                source,
                donor,
                task=task,
                duplicate_tolerance=float(duplicate_tolerance),
            )
            variants.append(event)
            records.append(
                DonorEvent(
                    channel=channel,
                    stage=stage,
                    graph_id=int(graph_id),
                    source=source,
                    donor_graph_id=int(graph_id),
                    donor_node=donor,
                    source_degree=int(degrees[source]),
                    donor_degree=int(degrees[donor]),
                    degree_gap=abs(int(degrees[source]) - int(degrees[donor])),
                    dose=structural_intervention_dose(
                        base,
                        event,
                        task,
                        tolerance=float(duplicate_tolerance),
                    ),
                    payload_fingerprint=stable_hash(
                        {"structural_footprint": footprints[donor].hex()}
                    ),
                    draw=draw,
                )
            )
    else:
        raise ValueError(f"unknown channel {channel!r}")
    return variants, records
