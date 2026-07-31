"""Coalitional controls for apparent long-range Functional carriage.

The estimand in this module is deliberately model-agnostic.  For a carrier and a
coalition of sources it compares the sum of singleton, task-output-projected
responses with the response to changing the whole coalition at once.  Two
interventions share the same sources:

* ``shell_permutation`` reassigns semantic payloads within each distance shell,
  preserving the shell multiset exactly; and
* ``shell_replacement`` uses external semantic donors, changing that multiset.

Topology and all structural/RRWP fields remain fixed in both cases.  This makes
the first intervention an assignment-redundancy test and the second a matched
content-replacement control, rather than treating either as a necessity test.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from .methodology.interventions import semantic_donor_swap, verify_semantic_swap
from .methodology.sampling import node_degrees, payload_array


@dataclass(frozen=True)
class SemanticAssignment:
    """One aligned source-to-donor semantic assignment."""

    source: int
    donor_graph: int
    donor_node: int
    source_degree: int
    donor_degree: int
    dose: float
    payload: tuple[Any, ...]


@dataclass(frozen=True)
class SemanticCoalition:
    """Singleton and joint variants for one auditable source coalition."""

    intervention: str
    assignments: tuple[SemanticAssignment, ...]
    singleton_variants: tuple[Any, ...]
    joint_variant: Any
    shell_sizes: tuple[int, ...]
    skipped_shell_sizes: tuple[int, ...]
    exact_derangement: bool
    matching_error: float

    @property
    def sources(self) -> tuple[int, ...]:
        return tuple(item.source for item in self.assignments)

    @property
    def doses(self) -> tuple[float, ...]:
        return tuple(item.dose for item in self.assignments)


def _different(left: np.ndarray, right: np.ndarray) -> bool:
    return not np.array_equal(np.asarray(left), np.asarray(right))


def minimum_gap_derangement(
    nodes: Sequence[int],
    payloads: np.ndarray,
    degrees: np.ndarray,
    rng: np.random.Generator,
    *,
    exact_limit: int = 12,
    random_attempts: int = 512,
) -> tuple[tuple[int, ...], bool] | None:
    """Return a maximum-change, minimum-degree-gap within-set derangement.

    For ordinary molecular shells (up to ``exact_limit`` nodes) a bitmask
    dynamic programme finds the global minimum.  Randomised tie-breaking avoids
    making chemically arbitrary node indices part of the intervention law.  A
    bounded random search is used only for unusually large shells.
    """

    source_nodes = tuple(int(node) for node in nodes)
    size = len(source_nodes)
    if size < 2:
        return None
    rows = np.asarray(payloads)
    degree = np.asarray(degrees, dtype=np.int64)
    valid = np.zeros((size, size), dtype=bool)
    cost = np.zeros((size, size), dtype=np.float64)
    maximum_gap = int(np.max(degree[list(source_nodes)])) - int(
        np.min(degree[list(source_nodes)])
    )
    # Lexicographic objective encoded as a scalar: first minimise zero-dose
    # semantic assignments, then minimise total degree mismatch.
    identical_penalty = float(size * max(1, maximum_gap) + 1)
    for left, source in enumerate(source_nodes):
        for right, donor in enumerate(source_nodes):
            valid[left, right] = source != donor
            cost[left, right] = (
                abs(int(degree[source]) - int(degree[donor]))
                + (identical_penalty if not _different(rows[source], rows[donor]) else 0.0)
            )

    if size <= int(exact_limit):
        infinity = float("inf")

        @lru_cache(maxsize=None)
        def optimum(position: int, used: int) -> float:
            if position == size:
                return 0.0
            return min(
                (
                    cost[position, donor] + optimum(position + 1, used | (1 << donor))
                    for donor in range(size)
                    if valid[position, donor] and not (used & (1 << donor))
                ),
                default=infinity,
            )

        if not np.isfinite(optimum(0, 0)):
            return None
        used = 0
        assignment: list[int] = []
        for position in range(size):
            target = optimum(position, used)
            eligible = [
                donor
                for donor in range(size)
                if valid[position, donor]
                and not (used & (1 << donor))
                and np.isclose(
                    cost[position, donor]
                    + optimum(position + 1, used | (1 << donor)),
                    target,
                    atol=1.0e-12,
                    rtol=0.0,
                )
            ]
            if not eligible:
                raise RuntimeError("derangement reconstruction lost an optimal branch")
            donor = int(rng.choice(np.asarray(eligible, dtype=np.int64)))
            assignment.append(source_nodes[donor])
            used |= 1 << donor
        return tuple(assignment), True

    best: tuple[int, ...] | None = None
    best_cost = float("inf")
    for _ in range(int(random_attempts)):
        order = tuple(int(value) for value in rng.permutation(size))
        if not all(valid[position, donor] for position, donor in enumerate(order)):
            continue
        total = float(sum(cost[position, donor] for position, donor in enumerate(order)))
        if total < best_cost:
            best = order
            best_cost = total
    if best is None:
        return None
    return tuple(source_nodes[index] for index in best), False


def _materialise(
    base: Any,
    assignments: Sequence[SemanticAssignment],
    *,
    task: Any,
) -> tuple[tuple[Any, ...], Any]:
    singletons: list[Any] = []
    joint = base.clone()
    for assignment in assignments:
        singleton = semantic_donor_swap(
            base,
            assignment.source,
            assignment.payload,
            adapter=task.content_adapter,
        )
        verify_semantic_swap(
            base,
            singleton,
            assignment.source,
            assignment.payload,
            task=task,
        )
        singletons.append(singleton)
        joint = semantic_donor_swap(
            joint,
            assignment.source,
            assignment.payload,
            adapter=task.content_adapter,
        )
    return tuple(singletons), joint


def build_shell_permutation(
    base: Any,
    shell_groups: Sequence[Sequence[int]],
    *,
    graph_id: int,
    task: Any,
    rng: np.random.Generator,
    exact_limit: int = 12,
    random_attempts: int = 512,
) -> SemanticCoalition | None:
    """Independently derange semantic payloads within each supplied shell."""

    rows = payload_array(base, task.content_adapter)
    degrees = node_degrees(base)
    assignments: list[SemanticAssignment] = []
    retained_sizes: list[int] = []
    skipped_sizes: list[int] = []
    exact = True
    for raw_group in shell_groups:
        group = tuple(sorted({int(node) for node in raw_group}))
        result = minimum_gap_derangement(
            group,
            rows,
            degrees,
            rng,
            exact_limit=int(exact_limit),
            random_attempts=int(random_attempts),
        )
        if result is None:
            skipped_sizes.append(len(group))
            continue
        donors, group_exact = result
        exact = exact and group_exact
        retained_sizes.append(len(group))
        for source, donor in zip(group, donors):
            if not _different(rows[source], rows[donor]):
                # This assignment is part of the node derangement but has no
                # semantic effect.  Omitting it leaves the realised joint graph
                # exactly unchanged relative to applying it explicitly.
                continue
            payload = tuple(np.asarray(rows[donor]).reshape(-1).tolist())
            assignments.append(
                SemanticAssignment(
                    source=source,
                    donor_graph=int(graph_id),
                    donor_node=donor,
                    source_degree=int(degrees[source]),
                    donor_degree=int(degrees[donor]),
                    dose=float(
                        np.linalg.norm(
                            np.asarray(payload, dtype=np.float64)
                            - np.asarray(rows[source], dtype=np.float64).reshape(-1)
                        )
                    ),
                    payload=payload,
                )
            )
    if not assignments:
        return None
    singletons, joint = _materialise(base, assignments, task=task)
    return SemanticCoalition(
        intervention="shell_permutation",
        assignments=tuple(assignments),
        singleton_variants=singletons,
        joint_variant=joint,
        shell_sizes=tuple(retained_sizes),
        skipped_shell_sizes=tuple(skipped_sizes),
        exact_derangement=exact,
        matching_error=0.0,
    )


def build_shell_replacement(
    base: Any,
    reference: SemanticCoalition,
    *,
    task: Any,
    semantic_pool: Any,
    rng: np.random.Generator,
    candidates: int = 32,
) -> SemanticCoalition:
    """Build a degree-law-conforming external replacement matched to swap dose.

    Complete candidate coalitions are drawn from the canonical graph-balanced
    semantic donor law.  Selection minimises a pre-outcome, scale-normalised
    discrepancy from the permutation's per-source intervention doses.
    """

    rows = payload_array(base, task.content_adapter)
    degrees = node_degrees(base)
    sources = reference.sources
    reference_dose = np.asarray(reference.doses, dtype=np.float64)
    scale = np.maximum(reference_dose, 1.0)
    best: tuple[SemanticAssignment, ...] | None = None
    best_error = float("inf")
    for _ in range(int(candidates)):
        proposal: list[SemanticAssignment] = []
        for source in sources:
            donor = semantic_pool.draw(
                rows[source],
                int(degrees[source]),
                1,
                rng,
                base_graph_id=None,
            )[0]
            payload = tuple(donor.payload)
            proposal.append(
                SemanticAssignment(
                    source=int(source),
                    donor_graph=int(donor.graph_id),
                    donor_node=int(donor.node),
                    source_degree=int(degrees[source]),
                    donor_degree=int(donor.degree),
                    dose=float(
                        np.linalg.norm(
                            np.asarray(payload, dtype=np.float64)
                            - np.asarray(rows[source], dtype=np.float64).reshape(-1)
                        )
                    ),
                    payload=payload,
                )
            )
        proposal_dose = np.asarray([item.dose for item in proposal], dtype=np.float64)
        error = float(np.mean(np.abs(proposal_dose - reference_dose) / scale))
        if error < best_error:
            best = tuple(proposal)
            best_error = error
    if best is None:
        raise RuntimeError("external shell replacement produced no candidate coalition")
    singletons, joint = _materialise(base, best, task=task)
    return SemanticCoalition(
        intervention="shell_replacement",
        assignments=best,
        singleton_variants=singletons,
        joint_variant=joint,
        shell_sizes=reference.shell_sizes,
        skipped_shell_sizes=reference.skipped_shell_sizes,
        exact_derangement=True,
        matching_error=best_error,
    )


def survival_components(
    singleton_vectors: Any,
    joint_vector: Any,
    *,
    effect_floor: float,
) -> dict[str, float | bool]:
    """Return the complete additive/nonlinear survival decomposition.

    ``A`` is apparent singleton mass, ``C`` is its signed/coherent resultant,
    and ``J`` is the actual joint response.  Hence ``J/A`` is the primary
    survival ratio, ``C/A`` isolates additive cancellation, and
    ``(J-C)/A`` is the nonlinear residual.  Ratios are intentionally not
    clipped: values above one are evidence of synergy.
    """

    singleton = np.asarray(singleton_vectors, dtype=np.float64)
    joint = np.asarray(joint_vector, dtype=np.float64).reshape(-1)
    if singleton.ndim != 2 or singleton.shape[1:] != joint.shape:
        raise ValueError("singleton vectors must be [source,output] and align with joint")
    apparent = float(np.linalg.norm(singleton, axis=-1).sum())
    additive = float(np.linalg.norm(singleton.sum(axis=0)))
    joint_mass = float(np.linalg.norm(joint))
    estimable = bool(
        np.isfinite(singleton).all()
        and np.isfinite(joint).all()
        and apparent > float(effect_floor)
    )
    denominator = apparent if estimable else np.nan
    return {
        "apparent_mass": apparent,
        "additive_resultant": additive,
        "joint_mass": joint_mass,
        "survival": float(joint_mass / denominator),
        "additive_survival": float(additive / denominator),
        "nonlinear_residual": float((joint_mass - additive) / denominator),
        "estimable": estimable,
    }
