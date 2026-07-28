"""Shared machinery for the range-measure comparison.

Two measurements of the same object are put side by side:

* **Jacobian range** (Bamberger et al., ICML 2025, *On Measuring Long-Range Interactions in
  Graph Neural Networks*).  For a differentiable map ``F`` the normalised node range is
  ``rho_hat_u = sum_v I_u(v) d(u,v)`` with influence distribution
  ``I_u(v) ∝ sum_{a,b} |dF^a_u / dx^b_v|``.

* **Functional carriage** (this repository's canonical methodology,
  ``src/graph_specialisation_metrics/README.md`` §2, §3, §5, §7).  A semantic donor-swap replaces
  the whole payload of one source node ``s`` with a real donor node's payload; the response is read
  at every carrier ``i`` as ``F_sens[i,s] = mean_k ||q_{s,k,i}||_2``.  Its expected distance under
  the carrier-conditional source distribution is the carriage analogue of ``rho_hat_u``.

The donor law, the swap, the carriage reduction, the shortest-path accounting, the 20% trimmed
graph estimator and the bootstrap policy are imported from the production package rather than
re-implemented.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from graph_specialisation_metrics.methodology.bootstrap import (  # noqa: E402
    trimmed_mean,
)
from graph_specialisation_metrics.methodology.carriage import (  # noqa: E402
    functional_carriage_events,
)
from graph_specialisation_metrics.methodology.distance import (  # noqa: E402
    shortest_path_distances,
)
from graph_specialisation_metrics.methodology.events import (  # noqa: E402
    build_channel_events,
)
from graph_specialisation_metrics.methodology.protocol import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BootstrapPolicy,
)
from graph_specialisation_metrics.methodology.sampling import (  # noqa: E402
    SemanticDonorPool,
)

DTYPE = torch.float64


# --------------------------------------------------------------------------------------
# Minimal PyG-shaped graph container
# --------------------------------------------------------------------------------------


class TinyData:
    """The smallest object satisfying the production intervention/donor code paths."""

    def __init__(self, x: torch.Tensor, edge_index: torch.Tensor) -> None:
        self.x = x
        self.edge_index = edge_index
        self.num_nodes = int(x.shape[0])

    @property
    def keys(self) -> tuple[str, ...]:
        return ("x", "edge_index")

    def clone(self) -> "TinyData":
        return TinyData(self.x.clone(), self.edge_index.clone())


@dataclass(frozen=True)
class StubTask:
    """Declares the semantic boundary: only ``x`` is replaceable, ``edge_index`` is support."""

    content_adapter: Any = None
    backend_kind: Any = None
    node_structural_fields: tuple[str, ...] = ()
    pair_structural_fields: tuple[tuple[str, str], ...] = ()
    dense_pair_structural_fields: tuple[str, ...] = ()
    fixed_support_fields: tuple[str, ...] = ("edge_index",)
    immutable_control_fields: tuple[str, ...] = ()


TASK = StubTask()


# --------------------------------------------------------------------------------------
# Topologies
# --------------------------------------------------------------------------------------


def path_edge_index(n: int) -> torch.Tensor:
    left = torch.arange(n - 1)
    right = left + 1
    return torch.stack(
        [torch.cat([left, right]), torch.cat([right, left])]
    ).to(torch.long)


def grid_edge_index(rows: int, cols: int) -> torch.Tensor:
    src: list[int] = []
    dst: list[int] = []
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            if c + 1 < cols:
                src += [node, node + 1]
                dst += [node + 1, node]
            if r + 1 < rows:
                src += [node, node + cols]
                dst += [node + cols, node]
    return torch.tensor([src, dst], dtype=torch.long)


def make_graphs(
    count: int,
    *,
    nodes: int,
    channels: int,
    rng: np.random.Generator,
    topology: str = "path",
    grid_shape: tuple[int, int] | None = None,
    scale_profile: str | None = None,
) -> list[TinyData]:
    """Build a graph list.

    ``scale_profile='linear'`` gives node ``i`` payload standard deviation ramping from 0.25 to
    2.0 across the node index.  Real payloads are rarely homoscedastic, and the per-source donor
    scale ``c_s = mean_k ||delta_{s,k}||`` inherits that variation -- which is exactly what the
    event normalisation divides out.
    """

    if topology == "path":
        edge_index = path_edge_index(nodes)
        n = nodes
    elif topology == "grid":
        if grid_shape is None:
            side = int(round(nodes ** 0.5))
            grid_shape = (side, side)
        edge_index = grid_edge_index(*grid_shape)
        n = grid_shape[0] * grid_shape[1]
    else:
        raise ValueError(f"unknown topology {topology!r}")
    if scale_profile is None:
        scale = np.ones((n, 1))
    elif scale_profile == "linear":
        scale = np.linspace(0.25, 2.0, n)[:, None]
    else:
        raise ValueError(f"unknown scale profile {scale_profile!r}")
    return [
        TinyData(
            torch.tensor(scale * rng.standard_normal((n, channels)), dtype=DTYPE),
            edge_index.clone(),
        )
        for _ in range(count)
    ]


# --------------------------------------------------------------------------------------
# The three ground-truth operators of Bamberger et al. Figure 3
# --------------------------------------------------------------------------------------


def normalised_adjacency(edge_index: torch.Tensor, n: int, *, self_loops: bool = False) -> np.ndarray:
    """Symmetric normalised adjacency ``D^-1/2 A D^-1/2``.

    Section 5.2 of the paper gives both definitions -- "with self loops" for the topology
    experiment and "without self-loops" fifty lines later in the k-Power task list.  The Figure 3
    code uses self-loops, so ``power_loops`` is the published variant; both are measured because
    they differ substantially on a bipartite graph.
    """

    adjacency = np.zeros((n, n), dtype=np.float64)
    edge = edge_index.numpy()
    adjacency[edge[0], edge[1]] = 1.0
    if self_loops:
        adjacency += np.eye(n)
    degree = adjacency.sum(axis=1)
    inverse = np.where(degree > 0, 1.0 / np.sqrt(np.maximum(degree, 1e-12)), 0.0)
    return inverse[:, None] * adjacency * inverse[None, :]


def operator_matrix(name: str, k: int, distances: np.ndarray, edge_index: torch.Tensor) -> np.ndarray:
    """Return the ``n x n`` matrix ``L`` of the named task.

    ``dirac``       ``F(X)_u = mean over nodes at exactly k hops``
    ``rectangle``   ``F(X)_u = mean over the ball of radius k (u included)``
    ``power``       ``F(X)   = A_tilde^k X``  (no self-loops)
    ``power_loops`` the same with self-loops added before normalisation
    """

    n = distances.shape[0]
    if name == "dirac":
        mask = (distances == k).astype(np.float64)
    elif name == "rectangle":
        mask = (distances <= k).astype(np.float64)
    elif name in ("power", "power_loops"):
        base = normalised_adjacency(edge_index, n, self_loops=name == "power_loops")
        return np.linalg.matrix_power(base, int(k))
    else:
        raise ValueError(f"unknown task {name!r}")
    counts = mask.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.where(counts > 0, mask / np.maximum(counts, 1.0), 0.0)
    return matrix


def figure3_operator(name: str, k: int, distances: np.ndarray, edge_index: torch.Tensor) -> np.ndarray:
    """The operator actually plotted in the paper's Figure 3.

    Their task builds ``dist_fn(data) * interaction_fn(x)`` and reduces over the first axis, so the
    realised map is ``M^T X``.  It changes the reported range only for k-Rectangle: k-Power is
    symmetric, and k-Dirac puts all its mass at distance exactly ``k`` under either orientation, so
    its normalised range is ``k`` either way.  With the transpose the published
    ``grid_task_range_*.csv`` values are reproduced exactly.
    """

    return operator_matrix(name, k, distances, edge_index).T


def linear_operator_fn(matrix: np.ndarray) -> Callable[[torch.Tensor], torch.Tensor]:
    tensor = torch.tensor(matrix, dtype=DTYPE)

    def apply(x: torch.Tensor) -> torch.Tensor:
        return tensor @ x

    return apply


# --------------------------------------------------------------------------------------
# Method A: Jacobian range (Bamberger et al.)
# --------------------------------------------------------------------------------------


def influence_matrix_autograd(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> np.ndarray:
    """``M[u,v] = sum_{a,b} |dF^a_u / dx^b_v|`` by exact autograd Jacobian."""

    x = x.detach().clone().requires_grad_(True)
    jac = torch.autograd.functional.jacobian(fn, x, vectorize=True)
    # jac: [n_out, c, n_in, d]
    return jac.abs().sum(dim=(1, 3)).detach().numpy()


def influence_matrix_analytic(matrix: np.ndarray, channels: int) -> np.ndarray:
    """Channel-shared linear operator: ``sum_{a,b}|J| = channels * |L_uv|``."""

    return float(channels) * np.abs(matrix)


def normalised_range_from_influence(influence: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """``rho_hat_u`` per node; NaN where a row carries no influence mass."""

    finite = np.where(np.isfinite(distances), distances, 0.0)
    reachable = np.isfinite(distances)
    mass = influence * reachable
    total = mass.sum(axis=1)
    weighted = (mass * finite).sum(axis=1)
    safe = np.where(total > 0, total, 1.0)
    return np.where(total > 0, weighted / safe, np.nan)


# --------------------------------------------------------------------------------------
# Method B: donor-swap Functional carriage
# --------------------------------------------------------------------------------------


def identity_readout_gradient(channels: int, nodes: int) -> torch.Tensor:
    """``g_out[t,i,m] = d z_t / d h^L_{i,m}`` for the identity node-level readout.

    The methodology contracts ``q_{i,t} = <g_out_{t,i}, delta h_i>`` at each carrier, so an
    identity readout on node outputs needs only the ``channels x channels`` identity replicated
    across carriers; ``||q_i||_2 = ||delta h_i||_2``.  ``full_identity_readout_gradient`` builds the
    literal ``T = nodes * channels`` object and the two agree exactly (asserted in the tests).
    """

    eye = torch.eye(channels, dtype=DTYPE)
    return eye[:, None, :].expand(channels, nodes, channels).contiguous()


def full_identity_readout_gradient(channels: int, nodes: int) -> torch.Tensor:
    grad = torch.zeros(nodes * channels, nodes, channels, dtype=DTYPE)
    for node in range(nodes):
        for channel in range(channels):
            grad[node * channels + channel, node, channel] = 1.0
    return grad


@dataclass(frozen=True)
class CarriageEvents:
    """Donor-resolved carriage for one graph.

    ``magnitude`` is ``[source, donor, carrier]`` eventwise ``||q||_2``; ``dose`` is
    ``[source, donor]`` and holds the registered ``||x_s - x_donor||_2`` from the event manifest.
    """

    sources: np.ndarray
    magnitude: np.ndarray
    dose: np.ndarray
    degree_gap: np.ndarray

    @property
    def field(self) -> np.ndarray:
        """``F_sens[carrier, source]`` -- the production functional carriage field."""

        return self.magnitude.mean(axis=1).T

    @property
    def normalised_field(self) -> np.ndarray:
        """Event-normalised carriage ``mean_k ||q_{s,k,i}||_2 / ||delta_{s,k}||_2``."""

        return (self.magnitude / self.dose[:, :, None]).mean(axis=1).T


def carriage_events(
    graph: TinyData,
    fn: Callable[[torch.Tensor], torch.Tensor],
    pool: SemanticDonorPool,
    *,
    graph_id: int,
    donors: int,
    rng: np.random.Generator,
    sources: Sequence[int] | None = None,
    readout_gradient: torch.Tensor | None = None,
) -> CarriageEvents:
    """Run the production semantic donor-swap protocol and read carriage at node outputs.

    ``fn`` maps ``[n, d] -> [n, width]`` and must broadcast over a leading batch dimension.
    ``readout_gradient`` is ``g_out[t, i, m] = d z_t / d h_{i,m}``; the default is the identity
    node-level readout.
    """

    if int(graph_id) in pool.by_graph:
        raise ValueError(
            f"donor pool contains the evaluated base graph {graph_id} "
            "(methodology section 2.1, eligibility rule 1)"
        )
    n = graph.num_nodes
    channels = int(graph.x.shape[1])
    source_ids = np.arange(n) if sources is None else np.asarray(sources, dtype=np.int64)
    with torch.no_grad():
        # Section 2.3 asks for the clean member to sit in the event batch.  For a deterministic
        # map an unbatched clean forward is bit-identical, and it is asserted against the batched
        # path in test_batched_and_looped_application_agree.
        clean = fn(graph.x)
    width = int(clean.shape[-1])

    magnitudes: list[np.ndarray] = []
    doses: list[np.ndarray] = []
    gaps: list[np.ndarray] = []
    gradient = (
        identity_readout_gradient(width, n) if readout_gradient is None else readout_gradient
    )
    for source in source_ids:
        variants, records = build_channel_events(
            graph,
            graph_id=int(graph_id),
            source=int(source),
            channel="semantic",
            stage="range-comparison",
            donors=int(donors),
            rng=rng,
            task=TASK,
            semantic_pool=pool,
            duplicate_tolerance=0.0,
        )
        with torch.no_grad():
            stacked = torch.stack([variant.x for variant in variants])
            outputs = fn(stacked)
            if outputs.shape[0] != stacked.shape[0] or tuple(outputs.shape[1:]) != tuple(clean.shape):
                raise ValueError(
                    f"fn did not respect the batch dimension: got {tuple(outputs.shape)}, "
                    f"expected {(stacked.shape[0],) + tuple(clean.shape)}"
                )
            delta = (clean[None] - outputs)[None]  # [1, K, N, M]
            events = functional_carriage_events(delta, gradient)[0]  # [K, N]
        magnitudes.append(events.numpy())
        doses.append(np.asarray([record.dose for record in records], dtype=np.float64))
        gaps.append(np.asarray([record.degree_gap for record in records], dtype=np.float64))
    return CarriageEvents(
        sources=source_ids,
        magnitude=np.stack(magnitudes),
        dose=np.stack(doses),
        degree_gap=np.stack(gaps),
    )


def carriage_range_per_carrier(field: np.ndarray, distances: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Expected donor-swap distance at each carrier: ``sum_s F[i,s] d(i,s) / sum_s F[i,s]``."""

    d = distances[:, sources]
    reachable = np.isfinite(d)
    finite = np.where(reachable, d, 0.0)
    mass = field * reachable
    total = mass.sum(axis=1)
    weighted = (mass * finite).sum(axis=1)
    safe = np.where(total > 0, total, 1.0)
    return np.where(total > 0, weighted / safe, np.nan)


def safe_nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


# --------------------------------------------------------------------------------------
# Registered bootstrap, vectorised
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    replicates: int
    resampled_levels: tuple[str, ...] = ()
    rng_seed: int = 0


# Every node of every graph is used as a source, so the source level is exhaustively enumerated
# and is held fixed -- the methodology's section 7 exception, which is sufficient on its own: the
# estimand is conditional on those sources.  Donors are genuinely drawn from a pool, so that level
# is resampled.  A secondary, task-dependent reason to prefer the exception here: the range is a
# ratio over sources, and on operators whose per-carrier mass is concentrated on few sources (the
# k-Power family) resampling them shifts the interval clear of the point estimate.  On k-Dirac and
# k-Rectangle it does not, so this is a diagnostic, not a general law.
ENUMERATED_SOURCES = BootstrapPolicy(resample_source=False)


def _graph_range(
    magnitude: np.ndarray,
    dose: np.ndarray,
    distances: np.ndarray,
    sources: np.ndarray,
    *,
    normalise: bool,
    source_index: np.ndarray | None = None,
    donor_index: np.ndarray | None = None,
) -> float:
    weights = magnitude / dose[:, :, None] if normalise else magnitude
    if donor_index is not None:
        weights = np.take_along_axis(weights, donor_index[:, :, None], axis=1)
    field = weights.mean(axis=1).T
    picked = sources
    if source_index is not None:
        field = field[:, source_index]
        picked = sources[source_index]
    return safe_nanmean(carriage_range_per_carrier(field, distances, picked))


def _ratio_terms(entry: dict[str, Any], *, normalise: bool) -> tuple[np.ndarray, np.ndarray]:
    """Distance-weighted and unweighted event terms, flattened to ``[source*donor, carrier]``.

    Both the numerator and the denominator of the carrier-conditional range are *linear* in the
    event weights, so a donor resample is a matrix-vector product against per-event multiplicities.
    The source axis is kept inside the flattened index so donors can be resampled independently
    within each source, exactly as the production ``_nested_estimate`` does.  Exact, not an
    approximation -- valid whenever the source level itself is held fixed.
    """

    weights = entry["magnitude"] / entry["dose"][:, :, None] if normalise else entry["magnitude"]
    sources, donors, carriers = weights.shape
    d = entry["distances"][:, entry["sources"]]
    reachable = np.isfinite(d).astype(np.float64)
    finite = np.where(reachable > 0, d, 0.0) * reachable
    # [source, donor, carrier] * [carrier, source] -> keep every axis; transpose to source-major.
    numerator = weights * finite.T[:, None, :]
    denominator = weights * reachable.T[:, None, :]
    shape = (sources * donors, carriers)
    return numerator.reshape(shape), denominator.reshape(shape)


def _range_from_terms(numerator: np.ndarray, denominator: np.ndarray, counts: np.ndarray) -> float:
    """``counts`` holds one multiplicity per ``(source, donor)`` event, already divided by ``K``."""

    top = counts @ numerator
    bottom = counts @ denominator
    safe = np.where(bottom > 0, bottom, 1.0)
    return safe_nanmean(np.where(bottom > 0, top / safe, np.nan))


def _donor_multiplicities(
    sources: int, donors: int, rng: np.random.Generator, *, resample: bool
) -> np.ndarray:
    """Per-source independent draws with replacement, flattened to match ``_ratio_terms``."""

    if not resample:
        return np.full(sources * donors, 1.0 / donors)
    picks = rng.integers(0, donors, size=(sources, donors))
    # Offset each source's picks into its own block so one bincount covers every source at once.
    flat = picks + (np.arange(sources) * donors)[:, None]
    counts = np.bincount(flat.reshape(-1), minlength=sources * donors).astype(np.float64)
    return counts / donors


def bootstrap_carriage_range(
    per_graph: Sequence[dict[str, Any]],
    *,
    normalise: bool,
    policy: BootstrapPolicy | None = None,
) -> Interval:
    """Registered percentile bootstrap of the dataset range.

    Graph values are combined with the registered 20% trimmed mean; every replicate reruns the
    complete aggregation, so the interval is of the reported estimator and not of a surrogate.
    The default policy resamples graphs and donors and holds the exhaustively enumerated source
    level fixed (see ``ENUMERATED_SOURCES``).
    """

    policy = policy or ENUMERATED_SOURCES
    policy.validate()
    graphs = len(per_graph)
    rng = np.random.default_rng(int(policy.rng_seed))
    draws = np.empty(policy.replicates, dtype=np.float64)

    if policy.resample_source:
        values = np.asarray(
            [
                _graph_range(
                    e["magnitude"], e["dose"], e["distances"], e["sources"], normalise=normalise
                )
                for e in per_graph
            ]
        )
        estimate = float(trimmed_mean(values, proportion=policy.trim_fraction))
        for replicate in range(policy.replicates):
            chosen = rng.integers(0, graphs, size=graphs) if policy.resample_graph else np.arange(graphs)
            replicate_values = np.empty(graphs, dtype=np.float64)
            for slot, index in enumerate(chosen):
                entry = per_graph[int(index)]
                n_sources, n_donors = entry["magnitude"].shape[:2]
                replicate_values[slot] = _graph_range(
                    entry["magnitude"],
                    entry["dose"],
                    entry["distances"],
                    entry["sources"],
                    normalise=normalise,
                    source_index=rng.integers(0, n_sources, size=n_sources),
                    donor_index=(
                        rng.integers(0, n_donors, size=(n_sources, n_donors))
                        if policy.resample_donor
                        else None
                    ),
                )
            draws[replicate] = trimmed_mean(replicate_values, proportion=policy.trim_fraction)
        alpha = (1.0 - policy.confidence) / 2.0
        low, high = np.quantile(draws, [alpha, 1.0 - alpha])
        return Interval(
        estimate, float(low), float(high), int(policy.replicates), _levels(policy),
        int(policy.rng_seed),
    )

    terms = [_ratio_terms(entry, normalise=normalise) for entry in per_graph]
    shapes = [entry["magnitude"].shape[:2] for entry in per_graph]
    uniform = [
        _donor_multiplicities(s, k, rng, resample=False) for s, k in shapes
    ]
    values = np.asarray(
        [_range_from_terms(num, den, weight) for (num, den), weight in zip(terms, uniform)]
    )
    estimate = float(trimmed_mean(values, proportion=policy.trim_fraction))
    for replicate in range(policy.replicates):
        chosen = rng.integers(0, graphs, size=graphs) if policy.resample_graph else np.arange(graphs)
        replicate_values = np.empty(graphs, dtype=np.float64)
        for slot, index in enumerate(chosen):
            index = int(index)
            num, den = terms[index]
            counts = (
                _donor_multiplicities(*shapes[index], rng, resample=True)
                if policy.resample_donor
                else uniform[index]
            )
            replicate_values[slot] = _range_from_terms(num, den, counts)
        draws[replicate] = trimmed_mean(replicate_values, proportion=policy.trim_fraction)
    alpha = (1.0 - policy.confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return Interval(
        estimate, float(low), float(high), int(policy.replicates), _levels(policy),
        int(policy.rng_seed),
    )


def _levels(policy: BootstrapPolicy) -> tuple[str, ...]:
    return tuple(
        name
        for name, enabled in (
            ("graph", policy.resample_graph),
            ("source", policy.resample_source),
            ("donor", policy.resample_donor),
        )
        if enabled
    )


def bootstrap_graph_values(values: Sequence[float], *, policy: BootstrapPolicy | None = None) -> Interval:
    """Graph-level bootstrap for a statistic already reduced to one value per graph.

    The Jacobian and Hessian ranges have no donor level, so this interval resamples graphs only.
    It is therefore *not* width-comparable with the carriage interval, which also resamples donors.
    """

    policy = policy or ENUMERATED_SOURCES
    policy.validate()
    array = np.asarray(values, dtype=np.float64)
    estimate = float(trimmed_mean(array, proportion=policy.trim_fraction))
    rng = np.random.default_rng(int(policy.rng_seed))
    draws = np.asarray(
        [
            trimmed_mean(
                array[rng.integers(0, array.size, size=array.size)]
                if policy.resample_graph
                else array,
                proportion=policy.trim_fraction,
            )
            for _ in range(policy.replicates)
        ]
    )
    alpha = (1.0 - policy.confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return Interval(
        estimate,
        float(low),
        float(high),
        int(policy.replicates),
        ("graph",) if policy.resample_graph else (),
        int(policy.rng_seed),
    )


__all__ = [
    "ENUMERATED_SOURCES",
    "BOOTSTRAP_REPLICATES",
    "BootstrapPolicy",
    "CarriageEvents",
    "DTYPE",
    "Interval",
    "SemanticDonorPool",
    "StubTask",
    "TinyData",
    "bootstrap_carriage_range",
    "bootstrap_graph_values",
    "carriage_events",
    "carriage_range_per_carrier",
    "full_identity_readout_gradient",
    "grid_edge_index",
    "identity_readout_gradient",
    "influence_matrix_analytic",
    "influence_matrix_autograd",
    "linear_operator_fn",
    "make_graphs",
    "normalised_adjacency",
    "normalised_range_from_influence",
    "operator_matrix",
    "path_edge_index",
    "safe_nanmean",
    "shortest_path_distances",
    "trimmed_mean",
]
