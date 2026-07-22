"""Standalone Colab beta for the ZINC specialisation-method redesign.

Paste this file into one Colab cell, or run it from the repository.  It loads the
trained dense-GRIT and 1-hop-GRIT ZINC checkpoints read-only, evaluates the successor
specialisation protocol on deterministic disjoint graph splits, and writes resumable
artifacts beneath ``MyDrive/graph_specialisation_metrics/zinc_redesign_beta_v1``.

This is deliberately isolated from the production ``specialisation`` and ``carriage``
packages.  Nothing here changes the current methodology.  The beta tests:

* coherent/eventwise x gross/net score aggregation (CG, EG, CN, EN);
* graph-balanced donor -> source -> graph estimation and donor-count convergence;
* D selectivity, evoked strength J, and joint-channel strength G;
* whole-transport restore/inject patching, zero-ablation necessity, sham and
  same-node-count cross-graph mismatch controls on an independent graph split;
* individual and family ablation on a third split;
* fixed-support routing/message decomposition for semantic and PE interventions;
* sensitivity/coherence carriage with intervention-specific distance;
* sample-split conditional-specialisation screening;
* a matched-real, non-isomorphic molecular-topology donor intervention with donor
  topology/RRWP copied in aligned base-node coordinates while atom content and target
  stay fixed.

The topology intervention is an exploratory third structural probe.  It is not silently
substituted for PE transposition in D: a whole-graph topology donor has a different
intervention unit and dose.  Its coverage, matching tier, graph-edit dose, agreement with
PE scores, and causal mediation are reported explicitly before any adoption decision.

The expensive phases are resumable: ``scores``, ``causal``, ``mechanism``, ``ablations``,
and ``figures``.  ``all`` runs them in that order.  ``--fast-dev-run`` is plumbing only.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

import numpy as np


BETA_VERSION = "zinc-specialisation-redesign-beta-v1"
BETA_SCHEMA = 1
REPOSITORY_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
REPOSITORY_BRANCH = "codex/cfim-grit-experiments"
COLAB_REPOSITORY = Path("/content/Graph-Specialisation-and-Metrics")
DEFAULT_OUTPUT = Path(
    "/content/drive/MyDrive/graph_specialisation_metrics/zinc_redesign_beta_v1"
)
SECRET_NAME = "dissertation_key"
ARCHITECTURES = ("zinc", "zinc_1hop")
DISPLAY = {"zinc": "Dense GRIT", "zinc_1hop": "1-hop GRIT"}
PRIMARY_CHANNELS = ("semantic", "pe")
CHANNELS = ("semantic", "pe", "topology")
AGGREGATIONS = ("CG", "EG", "CN", "EN")
EPS = 1.0e-12
CUDA_EQUIVALENCE_ATOL = 1.0e-4
CUDA_EQUIVALENCE_RTOL = 2.0e-5
MECHANISM_ATOL = 3.0e-4


# =====================================================================================
# Bootstrap, serialisation, and configuration
# =====================================================================================


def _command(*parts: str, check: bool = True) -> None:
    shown = [
        "https://***@github.com/" + part.split("@github.com/", 1)[1]
        if "@github.com/" in part else part
        for part in parts
    ]
    print(f"[cmd] {' '.join(shown)}", flush=True)
    subprocess.run(list(parts), check=check)


def bootstrap_repository(*, branch: str, skip_checkout: bool) -> Path:
    """Mount Drive, check out the requested branch, and install the local package."""

    in_colab = importlib.util.find_spec("google.colab") is not None
    if in_colab:
        from google.colab import drive, userdata  # type: ignore

        drive.mount("/content/drive", force_remount=False)
        try:
            token = userdata.get(SECRET_NAME) or os.environ.get(SECRET_NAME)
        except Exception as exc:  # pragma: no cover - Colab UI dependent
            print(f"[bootstrap] secret unavailable ({exc}); trying public clone", flush=True)
            token = os.environ.get(SECRET_NAME)
    else:
        token = os.environ.get(SECRET_NAME)
        if not skip_checkout:
            print("[bootstrap] local run: using current repository", flush=True)
            return Path(__file__).resolve().parents[3]

    if skip_checkout:
        root = Path(__file__).resolve().parents[3] if "__file__" in globals() else Path.cwd()
    else:
        suffix = REPOSITORY_URL.removeprefix("https://github.com/")
        authenticated = REPOSITORY_URL
        if token:
            authenticated = (
                f"https://x-access-token:{quote(str(token).strip(), safe='')}@github.com/{suffix}"
            )
        if (COLAB_REPOSITORY / ".git").exists():
            _command("git", "-C", str(COLAB_REPOSITORY), "remote", "set-url", "origin", authenticated)
            _command("git", "-C", str(COLAB_REPOSITORY), "fetch", "origin", branch)
            _command("git", "-C", str(COLAB_REPOSITORY), "checkout", branch)
            _command("git", "-C", str(COLAB_REPOSITORY), "reset", "--hard", f"origin/{branch}")
        else:
            if COLAB_REPOSITORY.exists():
                shutil.rmtree(COLAB_REPOSITORY)
            _command(
                "git", "clone", "--branch", branch, "--single-branch", authenticated,
                str(COLAB_REPOSITORY),
            )
        _command(
            "git", "-C", str(COLAB_REPOSITORY), "remote", "set-url", "origin", REPOSITORY_URL
        )
        root = COLAB_REPOSITORY
    _command(sys.executable, "-m", "pip", "install", "-q", "-e", str(root))
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    return root


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except ImportError:
        pass
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def atomic_torch_save(payload: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class BetaConfig:
    output_dir: str = str(DEFAULT_OUTPUT)
    score_graphs: int = 96
    score_sources: int = 8
    semantic_donors: int = 8
    pe_partners: int = 8
    topology_donors: int = 2
    topology_pool: int = 10_000
    causal_graphs: int = 48
    causal_sources: int = 2
    causal_batch_graphs: int = 8
    mechanism_graphs: int = 16
    ablation_graphs: int = 128
    family_size: int = 3
    bootstrap_samples: int = 1000
    conditional_bootstrap_samples: int = 1000
    analysis_seed: int = 1771
    device: str = "cuda:0"
    dense_checkpoint: str | None = None
    onehop_checkpoint: str | None = None
    allow_relaxed_topology: bool = True
    activity_floor_relative: float = 0.10
    causal_effect_floor_relative: float = 0.05
    selectivity_threshold: float = 0.20
    min_condition_graphs: int = 12
    min_condition_fraction: float = 0.20

    def validate(self) -> None:
        for name in (
            "score_graphs", "score_sources", "semantic_donors", "pe_partners",
            "topology_donors", "topology_pool", "causal_graphs",
            "causal_sources", "causal_batch_graphs", "mechanism_graphs", "ablation_graphs",
            "family_size", "bootstrap_samples", "conditional_bootstrap_samples",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.score_graphs + self.causal_graphs + self.ablation_graphs > 1000:
            raise ValueError("disjoint ZINC test subsets exceed the 1000-graph test split")
        if not 0.0 < self.activity_floor_relative < 1.0:
            raise ValueError("activity_floor_relative must be in (0,1)")
        if not 0.0 <= self.causal_effect_floor_relative < 1.0:
            raise ValueError("causal_effect_floor_relative must be in [0,1)")

    @property
    def root(self) -> Path:
        return Path(self.output_dir)

    @property
    def fingerprint(self) -> str:
        values = asdict(self)
        for key in ("output_dir", "device", "dense_checkpoint", "onehop_checkpoint"):
            values.pop(key, None)
        return stable_hash({"version": BETA_VERSION, **values})


def cache_path(cfg: BetaConfig, task: str, phase: str, checkpoint_sha: str) -> Path:
    return cfg.root / "cache" / task / (
        f"{phase}__{cfg.fingerprint}__{checkpoint_sha[:12]}.pt"
    )


def valid_cache(path: Path, cfg: BetaConfig, checkpoint_sha: str) -> dict[str, Any] | None:
    import torch

    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        return None
    if (
        payload.get("version") != BETA_VERSION
        or int(payload.get("schema", -1)) != BETA_SCHEMA
        or payload.get("fingerprint") != cfg.fingerprint
        or payload.get("checkpoint_sha256") != checkpoint_sha
    ):
        return None
    return dict(payload)


# =====================================================================================
# Pure numerical definitions
# =====================================================================================


def masked_mean(values: Any, valid: Any, dim: int) -> Any:
    weights = valid.to(device=values.device, dtype=values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def aggregate_projected_events(q: Any, valid: Any) -> dict[str, Any]:
    """Four scores for q=[G,S,K,L,H,N,T], with donors averaged within source.

    CG: carrier-wise magnitude after donor averaging; EG: donorwise carrier gross;
    CN: net-carrier magnitude after donor averaging; EN: donorwise net-carrier magnitude.
    The source dimension is retained here so callers can average sources within each graph
    before averaging graphs; padded sources/events never enter an estimand.
    """

    import torch

    if q.dim() != 7 or tuple(valid.shape) != tuple(q.shape[:3]):
        raise ValueError(
            f"expected q=[G,S,K,L,H,N,T], valid=[G,S,K], got {q.shape}, {valid.shape}"
        )
    donor_mean = masked_mean(q, valid, dim=2)  # [G,S,L,H,N,T]
    carrier_coherent = torch.linalg.vector_norm(donor_mean, dim=-1)
    carrier_event = torch.linalg.vector_norm(q, dim=-1)
    event_gross = carrier_event.sum(dim=-1)
    event_net = torch.linalg.vector_norm(q.sum(dim=-2), dim=-1)
    cg = carrier_coherent.sum(dim=-1)
    eg = masked_mean(event_gross, valid, dim=2)
    cn = torch.linalg.vector_norm(donor_mean.sum(dim=-2), dim=-1)
    en = masked_mean(event_net, valid, dim=2)
    source_valid = valid.any(dim=2)
    return {
        "per_source": {"CG": cg, "EG": eg, "CN": cn, "EN": en},
        "per_graph": {
            name: masked_mean(value, source_valid, dim=1)
            for name, value in {"CG": cg, "EG": eg, "CN": cn, "EN": en}.items()
        },
        "F_coh_source": carrier_coherent,
        "F_sens_source": masked_mean(carrier_event, valid, dim=2),
        "source_valid": source_valid,
    }


def aggregate_topology_events(q: Any, valid: Any) -> dict[str, Any]:
    """Topology counterpart for q=[G,K,L,H,N,T] (one whole-graph source per graph)."""

    if q.dim() != 6 or tuple(valid.shape) != tuple(q.shape[:2]):
        raise ValueError(f"expected q=[G,K,L,H,N,T], got {q.shape}, {valid.shape}")
    lifted = q[:, None]
    result = aggregate_projected_events(lifted, valid[:, None])
    result["per_source"] = {key: value[:, 0] for key, value in result["per_source"].items()}
    return result


def score_coordinates(semantic: Any, structural: Any) -> dict[str, np.ndarray]:
    sem = np.asarray(semantic, dtype=float)
    st = np.asarray(structural, dtype=float)
    if sem.shape != st.shape:
        raise ValueError("semantic and structural scores must have identical shape")
    scale = float(np.nanmean(np.concatenate([sem.reshape(-1), st.reshape(-1)])))
    scale = max(scale, EPS)
    sem_tilde = sem / scale
    st_tilde = st / scale
    total = sem_tilde + st_tilde
    return {
        "semantic": sem_tilde,
        "structural": st_tilde,
        "D": (sem_tilde - st_tilde) / (total + EPS),
        "J": 0.5 * total,
        "G": np.sqrt(np.maximum(sem_tilde * st_tilde, 0.0)),
        "shared_scale": np.asarray(scale),
    }


def spearman(left: Iterable[float], right: Iterable[float]) -> float:
    from scipy.stats import spearmanr

    a = np.asarray(list(left), dtype=float)
    b = np.asarray(list(right), dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.unique(a[keep]).size < 2 or np.unique(b[keep]).size < 2:
        return float("nan")
    return float(spearmanr(a[keep], b[keep]).statistic)


def bootstrap_spearman(
    left: np.ndarray, right: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float, float]:
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    rng = np.random.default_rng(seed)
    point = spearman(left, right)
    values = []
    for _ in range(samples):
        idx = rng.integers(0, len(left), len(left))
        values.append(spearman(left[idx], right[idx]))
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if not finite.size:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def tensor_equivalent(reference: Any, candidate: Any) -> dict[str, Any]:
    import torch

    if tuple(reference.shape) != tuple(candidate.shape):
        return {"passed": False, "reason": "shape", "max_abs_error": float("inf")}
    difference = (reference.detach() - candidate.detach()).abs()
    signal = torch.maximum(reference.detach().abs(), candidate.detach().abs())
    allowed = CUDA_EQUIVALENCE_ATOL + CUDA_EQUIVALENCE_RTOL * signal
    return {
        "passed": bool(torch.isfinite(difference).all() and torch.all(difference <= allowed)),
        "max_abs_error": float(difference.max()) if difference.numel() else 0.0,
        "max_tolerance_ratio": float((difference / allowed.clamp_min(EPS)).max())
        if difference.numel() else 0.0,
    }


# =====================================================================================
# Graph descriptors, matching, and real-topology donor intervention
# =====================================================================================


STRUCTURAL_NODE_FIELDS = (
    "rrwp", "deg", "log_deg", "abs_pe", "pestat_RRWP", "EigVals", "EigVecs",
)
STRUCTURAL_PAIR_FIELDS = (
    ("edge_index", "edge_attr"),
    ("rrwp_index", "rrwp_val"),
    ("rrwp_local_edge_index", None),
)


def _rows(array: Any) -> np.ndarray:
    value = array.detach().cpu().numpy() if hasattr(array, "detach") else np.asarray(array)
    return value.reshape(value.shape[0], -1)


def node_labels(data: Any) -> np.ndarray:
    return _rows(data.x).astype(np.int64, copy=False)


def undirected_edges(data: Any, n: int | None = None) -> np.ndarray:
    """Canonical non-self support edges; intended for 1-hop ZINC molecular Data."""

    edge_index = data.edge_index.detach().cpu().numpy().astype(np.int64)
    if n is None:
        n = int(data.num_nodes)
    keep = (
        (edge_index[0] >= 0) & (edge_index[1] >= 0)
        & (edge_index[0] < n) & (edge_index[1] < n)
        & (edge_index[0] != edge_index[1])
    )
    edges = np.sort(edge_index[:, keep].T, axis=1)
    if not len(edges):
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(edges, axis=0)


def molecular_edges(data: Any, n: int | None = None) -> np.ndarray:
    """Recover molecular bonds from the one-step RRWP channel when available.

    This avoids treating dense-GRIT's all-pairs attention support as molecular topology.
    GRIT RRWP stores ``[I, P, P^2, ...]``; positive off-diagonal entries in channel 1
    are precisely edges of the random-walk transition graph.  The 1-hop support is a
    checked fallback only.
    """

    n = int(data.num_nodes) if n is None else int(n)
    index = getattr(data, "rrwp_index", None)
    value = getattr(data, "rrwp_val", None)
    if index is not None and value is not None:
        rows = value.detach().cpu().numpy().reshape(value.shape[0], -1)
        if rows.shape[1] >= 2:
            pair = index.detach().cpu().numpy().astype(np.int64)
            keep = (
                (rows[:, 1] > 1.0e-12) & (pair[0] != pair[1])
                & (pair[0] >= 0) & (pair[1] >= 0) & (pair[0] < n) & (pair[1] < n)
            )
            edges = np.sort(pair[:, keep].T, axis=1)
            if len(edges):
                return np.unique(edges, axis=0)
    return undirected_edges(data, n)


def degrees_from_edges(n: int, edges: np.ndarray) -> np.ndarray:
    degree = np.zeros(n, dtype=np.int64)
    for left, right in np.asarray(edges, dtype=np.int64):
        degree[left] += 1
        degree[right] += 1
    return degree


def edge_label_multiset(data: Any, edges: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Undirected bond-label multiset where a matching edge_attr row is available."""

    attr = getattr(data, "edge_attr", None)
    if attr is None:
        return tuple()
    index = data.edge_index.detach().cpu().numpy().astype(np.int64)
    values = _rows(attr).astype(np.int64, copy=False)
    lookup: dict[tuple[int, int], tuple[int, ...]] = {}
    for position in range(index.shape[1]):
        u, v = int(index[0, position]), int(index[1, position])
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        lookup.setdefault(key, tuple(int(x) for x in values[position]))
    return tuple(sorted(lookup.get(tuple(edge), tuple()) for edge in edges))


def graph_descriptor(data: Any, *, graph_id: int, edges: np.ndarray | None = None) -> dict[str, Any]:
    n = int(data.num_nodes)
    edges = molecular_edges(data, n) if edges is None else np.asarray(edges, dtype=np.int64)
    labels = node_labels(data)
    degree = degrees_from_edges(n, edges)
    return {
        "graph_id": int(graph_id),
        "n": n,
        "node_multiset": tuple(sorted(tuple(int(x) for x in row) for row in labels)),
        "degree_multiset": tuple(sorted(int(x) for x in degree)),
        "edge_count": int(len(edges)),
        "edge_labels": edge_label_multiset(data, edges),
        "degree": degree,
        "edges": edges,
        "labels": labels,
        "cycle_rank": int(len(edges) - n + 1),
        "target": float(np.asarray(data.y.detach().cpu()).reshape(-1)[0])
        if getattr(data, "y", None) is not None else float("nan"),
    }


def _nx_graph(descriptor: Mapping[str, Any]) -> Any:
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(int(descriptor["n"])))
    graph.add_edges_from(np.asarray(descriptor["edges"], dtype=np.int64).tolist())
    return graph


def nonisomorphic(base: Mapping[str, Any], donor: Mapping[str, Any]) -> bool:
    import networkx as nx

    if int(base["n"]) != int(donor["n"]):
        return True
    return not nx.is_isomorphic(_nx_graph(base), _nx_graph(donor))


def topology_match_tier(base: Mapping[str, Any], donor: Mapping[str, Any]) -> int | None:
    """0=strict chemistry/degree support, 1=content+edge-count, 2=size-only nearest."""

    if int(base["n"]) != int(donor["n"]):
        return None
    same_content = base["node_multiset"] == donor["node_multiset"]
    same_degree = base["degree_multiset"] == donor["degree_multiset"]
    same_edges = int(base["edge_count"]) == int(donor["edge_count"])
    same_bonds = base["edge_labels"] == donor["edge_labels"]
    if same_content and same_degree and same_edges and same_bonds:
        return 0
    if same_content and same_edges:
        return 1
    return 2


def topology_match_cost(base: Mapping[str, Any], donor: Mapping[str, Any]) -> float:
    from collections import Counter

    if int(base["n"]) != int(donor["n"]):
        return float("inf")
    base_labels = Counter(base["node_multiset"])
    donor_labels = Counter(donor["node_multiset"])
    content_mismatch = sum((base_labels - donor_labels).values()) + sum(
        (donor_labels - base_labels).values()
    )
    degree_left = np.asarray(base["degree_multiset"], dtype=float)
    degree_right = np.asarray(donor["degree_multiset"], dtype=float)
    degree_cost = float(np.abs(degree_left - degree_right).sum())
    edge_cost = abs(int(base["edge_count"]) - int(donor["edge_count"]))
    bond_cost = 0.0 if base["edge_labels"] == donor["edge_labels"] else 1.0
    target_cost = 2.0 * abs(float(base.get("target", 0.0)) - float(donor.get("target", 0.0)))
    return 1000.0 * content_mismatch + 20.0 * edge_cost + degree_cost + bond_cost + target_cost


def align_donor_nodes(base: Mapping[str, Any], donor: Mapping[str, Any]) -> np.ndarray:
    """Return donor index for each base node via label/degree/neighbour-label matching."""

    from scipy.optimize import linear_sum_assignment

    n = int(base["n"])
    if int(donor["n"]) != n:
        raise ValueError("topology donor must have the same node count")
    base_labels = np.asarray(base["labels"])
    donor_labels = np.asarray(donor["labels"])
    base_degree = np.asarray(base["degree"])
    donor_degree = np.asarray(donor["degree"])

    def neighbourhood_signature(desc: Mapping[str, Any]) -> list[tuple[tuple[int, ...], ...]]:
        neighbours: list[list[int]] = [[] for _ in range(n)]
        for left, right in np.asarray(desc["edges"], dtype=np.int64):
            neighbours[left].append(right)
            neighbours[right].append(left)
        labels = np.asarray(desc["labels"])
        return [
            tuple(sorted(tuple(int(x) for x in labels[j]) for j in neighbours[i]))
            for i in range(n)
        ]

    base_neighbour = neighbourhood_signature(base)
    donor_neighbour = neighbourhood_signature(donor)
    cost = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            label_penalty = 10_000.0 * float(not np.array_equal(base_labels[i], donor_labels[j]))
            degree_penalty = 100.0 * abs(int(base_degree[i]) - int(donor_degree[j]))
            neighbourhood_penalty = float(base_neighbour[i] != donor_neighbour[j])
            cost[i, j] = label_penalty + degree_penalty + neighbourhood_penalty + 1e-6 * j
    rows, cols = linear_sum_assignment(cost)
    if not np.array_equal(rows, np.arange(n)):
        raise RuntimeError("Hungarian alignment did not cover base nodes")
    return cols.astype(np.int64)


def relabel_pair_index(index: Any, donor_for_base: np.ndarray) -> Any:
    """Relabel a donor-coordinate pair index into base coordinates."""

    import torch

    donor_for_base = np.asarray(donor_for_base, dtype=np.int64)
    base_for_donor = np.empty_like(donor_for_base)
    base_for_donor[donor_for_base] = np.arange(len(donor_for_base), dtype=np.int64)
    mapping = torch.as_tensor(base_for_donor, dtype=index.dtype, device=index.device)
    return mapping[index.long()]


def topology_donor_variant(base: Any, donor: Any, donor_for_base: np.ndarray) -> Any:
    """Copy donor topology/RRWP into base coordinates, preserving base content and y.

    The donor Data has already passed the same GRIT transform, so its RRWP is the
    recomputation for the donor molecular topology.  Pair indices are conjugated into
    base coordinates; node-structural rows follow the Hungarian alignment.  Unknown
    attributes remain from base and are audited by ``structural_field_audit``.
    """

    import torch

    n = int(base.num_nodes)
    if int(donor.num_nodes) != n:
        raise ValueError("base and topology donor must have equal node count")
    permutation = torch.as_tensor(donor_for_base, dtype=torch.long)
    out = base.clone()
    original_x = base.x.clone()
    original_y = base.y.clone() if getattr(base, "y", None) is not None else None

    for field in STRUCTURAL_NODE_FIELDS:
        value = getattr(donor, field, None)
        if value is None:
            continue
        if value.dim() >= 1 and int(value.size(0)) == n:
            setattr(out, field, value[permutation.to(value.device)].clone())
        else:
            # Graph-level eigenvalue/statistic tensors have no node axis but still belong
            # to the donor topology and must not remain from the base graph.
            setattr(out, field, value.clone())

    for index_name, value_name in STRUCTURAL_PAIR_FIELDS:
        index = getattr(donor, index_name, None)
        if index is None:
            continue
        setattr(out, index_name, relabel_pair_index(index, donor_for_base).clone())
        if value_name is not None:
            value = getattr(donor, value_name, None)
            if value is not None:
                setattr(out, value_name, value.clone())

    out.x = original_x
    if original_y is not None:
        out.y = original_y
    if not torch.equal(out.x, base.x):
        raise RuntimeError("topology donor altered semantic x")
    if original_y is not None and not torch.equal(out.y, base.y):
        raise RuntimeError("topology donor altered target y")
    return out


def structural_field_audit(data: Any) -> dict[str, list[str]]:
    """Classify transformed PyG fields so unhandled structural channels fail visibly."""

    keys = list(data.keys()) if callable(getattr(data, "keys", None)) else list(data.keys)
    handled = {
        "x", "y", "num_nodes", "edge_index", "edge_attr", "rrwp", "rrwp_index", "rrwp_val",
        "rrwp_local_edge_index", "deg", "log_deg", "abs_pe", "pestat_RRWP", "EigVals", "EigVecs",
    }
    suspicious_tokens = ("rrwp", "rw", "lap", "eig", "deg", "edge", "adj", "pe", "spd")
    suspicious = [
        str(key) for key in keys
        if str(key) not in handled and any(token in str(key).lower() for token in suspicious_tokens)
    ]
    return {"keys": sorted(map(str, keys)), "unhandled_structural": sorted(suspicious)}


def topology_edit_dose(
    base_edges: np.ndarray, donor_edges_base_coordinates: np.ndarray
) -> dict[str, Any]:
    base_set = {tuple(edge) for edge in np.asarray(base_edges, dtype=np.int64)}
    donor_set = {tuple(edge) for edge in np.asarray(donor_edges_base_coordinates, dtype=np.int64)}
    changed = base_set.symmetric_difference(donor_set)
    changed_nodes = sorted({node for edge in changed for node in edge})
    denominator = max(len(base_set) + len(donor_set), 1)
    return {
        "edge_symmetric_difference": int(len(changed)),
        "edge_jaccard_distance": float(len(changed) / denominator),
        "changed_nodes": changed_nodes,
    }


# =====================================================================================
# Task loading, deterministic splits, and intervention manifests
# =====================================================================================


def prepare_task(task_name: str, cfg: BetaConfig, *, force_fresh_grit: bool = False) -> dict[str, Any]:
    """Reconstruct the exact trained GRIT variant and load its checkpoint read-only."""

    import platform

    from graph_specialisation_metrics.carriage import env
    from graph_specialisation_metrics.carriage.tasks import get_task, resolve_dataset_dir
    from graph_specialisation_metrics.specialisation.model import GritHeadModel, SpecConfig

    spec = get_task(task_name)
    task_out = cfg.root / task_name
    task_out.mkdir(parents=True, exist_ok=True)
    default_repo = f"/content/GRIT_{spec.name}" if spec.env_hooks else "/content/GRIT"
    repo_dir = Path(spec.grit_repo_dir or default_repo)
    env.clone_grit(repo_dir, spec.grit_repo, spec.grit_commit, force_fresh=force_fresh_grit)
    for hook in spec.env_hooks:
        hook(repo_dir)
    env.prepare_inprocess_grit(repo_dir)
    config_file = env.resolve_config(spec, repo_dir, task_out)
    explicit = cfg.dense_checkpoint if task_name == "zinc" else cfg.onehop_checkpoint
    checkpoint, epoch = env.find_checkpoint(Path(spec.drive_dir) / "results", explicit)
    digest = sha256_file(checkpoint)
    sc = SpecConfig(
        ckpt=str(checkpoint),
        out_dir=str(task_out),
        dataset_dir=resolve_dataset_dir(spec),
        config_file=config_file,
        accelerator=cfg.device,
        seed=42,
        num_threads=4,
        eval_split="test",
        donor_split="train",
        num_graphs=cfg.score_graphs,
        donors=max(cfg.semantic_donors, cfg.pe_partners),
        ablation_graphs=cfg.ablation_graphs,
        analysis_seed=cfg.analysis_seed,
        resume=False,
    )
    gm = GritHeadModel(spec, sc).load()
    record = {
        "task": task_name,
        "title": spec.title,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "checkpoint_epoch": int(epoch),
        "grit_repository": str(repo_dir),
        "grit_commit": spec.grit_commit,
        "config_file": str(config_file),
        "parameters": int(gm.checks["num_parameters"]),
        "test_metric": gm.test_metric,
        "val_metric": gm.val_metric,
        "python": sys.version,
        "platform": platform.platform(),
    }
    write_json(cfg.root / "provenance" / f"{task_name}.json", record)
    return {"gm": gm, "spec": spec, "sc": sc, "checkpoint": checkpoint, "sha": digest}


def deterministic_splits(length: int, cfg: BetaConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.analysis_seed)
    order = rng.permutation(length)
    cursor = 0
    output = {}
    for name, count in (
        ("score", cfg.score_graphs),
        ("causal", cfg.causal_graphs),
        ("ablation", cfg.ablation_graphs),
    ):
        output[name] = np.sort(order[cursor:cursor + count]).astype(np.int64)
        cursor += count
    mechanism_count = min(cfg.mechanism_graphs, len(output["causal"]))
    output["mechanism"] = output["causal"][:mechanism_count].copy()
    if set(output["score"]).intersection(output["causal"]):
        raise RuntimeError("score and causal graph splits overlap")
    if set(output["score"]).intersection(output["ablation"]):
        raise RuntimeError("score and ablation graph splits overlap")
    if set(output["causal"]).intersection(output["ablation"]):
        raise RuntimeError("causal and ablation graph splits overlap")
    return output


def dataset_alignment_signature(data: Any) -> tuple[Any, ...]:
    y = tuple(float(x) for x in data.y.detach().cpu().reshape(-1))
    return int(data.num_nodes), tuple(map(tuple, node_labels(data))), y


def audit_cross_architecture_datasets(
    dense_ds: Any, onehop_ds: Any, graph_ids: Sequence[int]
) -> dict[str, Any]:
    failures = []
    topology_failures = []
    for graph_id in graph_ids:
        dense = dense_ds[int(graph_id)]
        onehop = onehop_ds[int(graph_id)]
        if dataset_alignment_signature(dense) != dataset_alignment_signature(onehop):
            failures.append(int(graph_id))
        dense_edges = molecular_edges(dense)
        onehop_edges = molecular_edges(onehop)
        if not np.array_equal(dense_edges, onehop_edges):
            topology_failures.append(int(graph_id))
    result = {
        "graphs_checked": int(len(graph_ids)),
        "content_target_failures": failures,
        "molecular_topology_failures": topology_failures,
        "passed": not failures and not topology_failures,
    }
    if not result["passed"]:
        raise RuntimeError(f"dense/1-hop dataset alignment failed: {result}")
    return result


def donor_row_pool(gm: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, graph_ids, node_ids = [], [], []
    for graph_id in range(len(gm.donor_ds)):
        value = gm.adapter.rows(gm.donor_ds[graph_id])
        rows.append(value)
        graph_ids.append(np.full(len(value), graph_id, dtype=np.int64))
        node_ids.append(np.arange(len(value), dtype=np.int64))
    return np.concatenate(rows), np.concatenate(graph_ids), np.concatenate(node_ids)


def semantic_variant(base: Any, source: int, donor_row: np.ndarray) -> Any:
    import torch

    out = base.clone()
    donor = torch.as_tensor(donor_row, device=out.x.device, dtype=out.x.dtype)
    if out.x.dim() == 1:
        out.x[source] = donor.reshape(-1)[0]
    else:
        out.x[source] = donor.reshape_as(out.x[source])
    return out


def pe_variant(base: Any, source: int, partner: int) -> Any:
    from graph_specialisation_metrics.specialisation.scores import _perturb_mask_frozen

    return _perturb_mask_frozen(base, int(source), int(partner))


def pe_payload_dose(base: Any, variant: Any) -> float:
    """RMS change across node- and pair-RRWP payloads, excluding fixed mask/bonds."""

    components = []
    for field in STRUCTURAL_NODE_FIELDS:
        left, right = getattr(base, field, None), getattr(variant, field, None)
        if left is not None and right is not None and tuple(left.shape) == tuple(right.shape):
            difference = (left.detach().float() - right.detach().float()).reshape(-1)
            components.append(difference.cpu().numpy())
    left_index, left_value = getattr(base, "rrwp_index", None), getattr(base, "rrwp_val", None)
    right_index, right_value = getattr(variant, "rrwp_index", None), getattr(variant, "rrwp_val", None)
    if all(value is not None for value in (left_index, left_value, right_index, right_value)):
        n = int(base.num_nodes)
        width = int(left_value.reshape(left_value.shape[0], -1).shape[1])
        left_dense = np.zeros((n, n, width), dtype=np.float32)
        right_dense = np.zeros_like(left_dense)
        li = left_index.detach().cpu().numpy().astype(np.int64)
        ri = right_index.detach().cpu().numpy().astype(np.int64)
        left_dense[li[0], li[1]] = left_value.detach().cpu().numpy().reshape(-1, width)
        right_dense[ri[0], ri[1]] = right_value.detach().cpu().numpy().reshape(-1, width)
        components.append((left_dense - right_dense).reshape(-1))
    if not components:
        return 0.0
    joined = np.concatenate(components).astype(float)
    return float(np.sqrt(np.mean(np.square(joined))))


def shortest_paths(n: int, edges: np.ndarray) -> np.ndarray:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    if len(edges):
        row = np.concatenate([edges[:, 0], edges[:, 1]])
        col = np.concatenate([edges[:, 1], edges[:, 0]])
        matrix = csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    else:
        matrix = csr_matrix((n, n))
    return np.asarray(shortest_path(matrix, directed=False, unweighted=True), dtype=float)


def build_topology_index(
    donor_ds: Any, *, maximum: int, seed: int
) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
    """Index real train molecules by node count; descriptors retain no model tensors."""

    rng = np.random.default_rng(seed)
    count = min(int(maximum), len(donor_ds))
    graph_ids = np.sort(rng.choice(len(donor_ds), size=count, replace=False))
    descriptors = [
        graph_descriptor(donor_ds[int(graph_id)], graph_id=int(graph_id))
        for graph_id in graph_ids
    ]
    by_n: dict[int, list[int]] = {}
    for position, descriptor in enumerate(descriptors):
        by_n.setdefault(int(descriptor["n"]), []).append(position)
    return descriptors, by_n


def load_or_build_topology_index(
    gm: Any, cfg: BetaConfig, task_name: str, checkpoint_sha: str
) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
    path = cache_path(cfg, task_name, "topology_index", checkpoint_sha)
    cached = valid_cache(path, cfg, checkpoint_sha)
    if cached is not None:
        return cached["descriptors"], cached["by_n"]
    descriptors, by_n = build_topology_index(
        gm.donor_ds, maximum=cfg.topology_pool, seed=cfg.analysis_seed + 19
    )
    atomic_torch_save({
        "version": BETA_VERSION,
        "schema": BETA_SCHEMA,
        "fingerprint": cfg.fingerprint,
        "checkpoint_sha256": checkpoint_sha,
        "task": task_name,
        "descriptors": descriptors,
        "by_n": by_n,
    }, path)
    return descriptors, by_n


def select_topology_donors(
    base: Mapping[str, Any],
    descriptors: Sequence[Mapping[str, Any]],
    by_n: Mapping[int, Sequence[int]],
    *,
    count: int,
    allow_relaxed: bool,
) -> list[dict[str, Any]]:
    candidates = []
    for position in by_n.get(int(base["n"]), []):
        donor = descriptors[int(position)]
        if not nonisomorphic(base, donor):
            continue
        tier = topology_match_tier(base, donor)
        if tier is None or (tier > 1 and not allow_relaxed):
            continue
        candidates.append((int(tier), topology_match_cost(base, donor), int(donor["graph_id"]), donor))
    candidates.sort(key=lambda item: item[:3])
    selected = []
    for tier, cost, graph_id, donor in candidates[:count]:
        alignment = align_donor_nodes(base, donor)
        content_alignment_exact = bool(np.array_equal(
            np.asarray(base["labels"]), np.asarray(donor["labels"])[alignment]
        ))
        if tier <= 1 and not content_alignment_exact:
            raise RuntimeError("exact-content topology donor could not be atom-aligned exactly")
        donor_edges_base = relabel_pair_index_numpy(np.asarray(donor["edges"]), alignment)
        dose = topology_edit_dose(np.asarray(base["edges"]), donor_edges_base)
        selected.append({
            "donor_graph_id": graph_id,
            "tier": tier,
            "cost": float(cost),
            "alignment": alignment,
            "dose": dose,
            "target_gap": abs(float(base.get("target", 0.0)) - float(donor.get("target", 0.0))),
            "content_alignment_exact": content_alignment_exact,
        })
    return selected


def relabel_pair_index_numpy(edges: np.ndarray, donor_for_base: np.ndarray) -> np.ndarray:
    donor_for_base = np.asarray(donor_for_base, dtype=np.int64)
    base_for_donor = np.empty_like(donor_for_base)
    base_for_donor[donor_for_base] = np.arange(len(donor_for_base), dtype=np.int64)
    relabelled = base_for_donor[np.asarray(edges, dtype=np.int64)]
    return np.sort(relabelled, axis=1)


def sample_sources(n: int, maximum: int, rng: np.random.Generator) -> np.ndarray:
    count = min(int(maximum), int(n))
    return np.sort(rng.choice(n, size=count, replace=False)).astype(np.int64)


def plan_graph_events(
    gm: Any,
    graph_id: int,
    cfg: BetaConfig,
    *,
    donor_rows: np.ndarray,
    donor_graph_ids: np.ndarray,
    topology_descriptors: Sequence[Mapping[str, Any]],
    topology_by_n: Mapping[int, Sequence[int]],
    sources_limit: int | None = None,
) -> dict[str, Any]:
    base = gm.eval_ds[int(graph_id)]
    descriptor = graph_descriptor(base, graph_id=int(graph_id))
    seed = cfg.analysis_seed + 1_000_003 * int(graph_id)
    rng = np.random.default_rng(seed)
    sources = sample_sources(
        int(base.num_nodes), cfg.score_sources if sources_limit is None else sources_limit, rng
    )
    labels = node_labels(base)
    semantic = []
    pe = []
    for source in sources:
        different = np.any(donor_rows != labels[source], axis=1)
        candidates = np.flatnonzero(different)
        if not candidates.size:
            candidates = np.arange(len(donor_rows))
        chosen = rng.choice(candidates, size=cfg.semantic_donors, replace=len(candidates) < cfg.semantic_donors)
        semantic.append({
            "source": int(source),
            "donor_rows": donor_rows[chosen].copy(),
            "donor_graph_ids": donor_graph_ids[chosen].copy(),
            "dose": np.linalg.norm(donor_rows[chosen].astype(float) - labels[source], axis=1),
        })
        degree = np.asarray(descriptor["degree"])
        candidates = np.flatnonzero((degree == degree[source]) & (np.arange(len(degree)) != source))
        if not candidates.size:
            others = np.flatnonzero(np.arange(len(degree)) != source)
            gap = np.abs(degree[others] - degree[source])
            candidates = others[gap == gap.min()]
        if not candidates.size:
            candidates = np.asarray([source])
        partners = rng.choice(
            candidates, size=cfg.pe_partners, replace=len(candidates) < cfg.pe_partners
        ).astype(np.int64)
        pe.append({
            "source": int(source),
            "partners": partners,
            "degree_gap": np.abs(degree[partners] - degree[source]),
            "dose": np.asarray([
                pe_payload_dose(base, pe_variant(base, int(source), int(partner)))
                for partner in partners
            ], dtype=np.float32),
        })
    topology = select_topology_donors(
        descriptor,
        topology_descriptors,
        topology_by_n,
        count=cfg.topology_donors,
        allow_relaxed=cfg.allow_relaxed_topology,
    )
    return {
        "graph_id": int(graph_id),
        "n": int(base.num_nodes),
        "sources": sources,
        "semantic": semantic,
        "pe": pe,
        "topology": topology,
        "descriptor": descriptor,
    }


def graph_event_variants(gm: Any, plan: Mapping[str, Any], channel: str) -> list[list[Any]]:
    """Return source groups of Data variants; topology has one graph-level group."""

    base = gm.eval_ds[int(plan["graph_id"])]
    if channel == "semantic":
        return [
            [semantic_variant(base, int(item["source"]), row) for row in item["donor_rows"]]
            for item in plan["semantic"]
        ]
    if channel == "pe":
        return [
            [pe_variant(base, int(item["source"]), int(partner)) for partner in item["partners"]]
            for item in plan["pe"]
        ]
    if channel == "topology":
        return [[
            topology_donor_variant(
                base,
                gm.donor_ds[int(item["donor_graph_id"])],
                np.asarray(item["alignment"], dtype=np.int64),
            )
            for item in plan["topology"]
        ]]
    raise ValueError(channel)


# =====================================================================================
# Event-level projected scoring and carriage
# =====================================================================================


def clean_gradients(gm: Any, base: Any) -> tuple[Any, list[Any], Any]:
    import torch
    from torch_geometric.data import Batch

    batch = Batch.from_data_list([base.clone()]).to(gm.device)
    capture = gm.capture(batch, want_grad=True, want_attn=False)
    pred = capture["pred"]
    outputs = pred.reshape(pred.shape[0], -1)
    gradients = []
    for target in range(outputs.shape[1]):
        gradients.append(torch.autograd.grad(
            outputs[0, target], capture["wV"], retain_graph=target + 1 < outputs.shape[1]
        ))
    phi = [torch.stack([gradient[layer] for gradient in gradients], dim=0) for layer in range(gm.L)]
    return pred.detach(), phi, capture


def projected_event_group(
    gm: Any,
    base: Any,
    variants: Sequence[Any],
    phi: Sequence[Any],
) -> tuple[Any, np.ndarray, np.ndarray]:
    """q=[K,L,H,N,T] using within-batch clean baselines to cancel scatter jitter."""

    import torch
    from torch_geometric.data import Batch

    if not variants:
        return torch.empty(0, gm.L, gm.H, int(base.num_nodes), len(phi[0])), np.empty(0), np.empty(0)
    replicas = [base.clone(), *[variant.clone() for variant in variants]]
    batch = Batch.from_data_list(replicas).to(gm.device)
    capture = gm.capture(batch, want_grad=False, want_attn=False, include_virtual_transport=True)
    per_replica = int(capture["wV"][0].shape[0] // len(replicas))
    if per_replica != int(base.num_nodes):
        raise RuntimeError("unexpected virtual-node transport in ZINC beta")
    q_layers = []
    for layer in range(gm.L):
        wv = capture["wV"][layer].reshape(len(replicas), per_replica, gm.H, gm.dh)
        delta = wv[0:1] - wv[1:]
        # phi[layer]=[T,N,H,D], delta=[K,N,H,D] -> [K,H,N,T]
        projected = torch.einsum("tnhd,knhd->khnt", phi[layer], delta)
        q_layers.append(projected)
    q = torch.stack(q_layers, dim=1)  # [K,L,H,N,T]
    predictions = capture["pred"].detach().cpu().numpy().reshape(len(replicas), -1)
    within_clean = predictions[0]
    prediction_deltas = within_clean[None] - predictions[1:]
    no_op = np.max(np.abs(prediction_deltas), axis=1) == 0.0
    return q.detach().cpu(), predictions, no_op


def distance_profile_one_graph(q: Any, distances: np.ndarray) -> list[dict[str, Any]]:
    """F_sens/F_coh with event-specific changed sets and equal carrier weighting."""

    import torch

    distance = torch.as_tensor(distances, dtype=torch.long)
    if tuple(distance.shape) != (int(q.shape[0]), int(q.shape[-2])):
        raise ValueError(f"distance {distance.shape} is not [events,carriers] for q {q.shape}")
    event_norm = torch.linalg.vector_norm(q, dim=-1)  # [K,L,H,N]
    rows = []
    finite = distance[distance >= 0]
    if not finite.numel():
        return rows
    for hop in range(int(finite.max()) + 1):
        pair_mask = distance == hop  # [K,N]
        carrier_count = pair_mask.sum(dim=0)
        carrier_valid = carrier_count > 0
        if not carrier_valid.any():
            continue
        weight = pair_mask[:, None, None, :].to(q.dtype)
        per_carrier_sens = (event_norm * weight).sum(dim=0) / carrier_count[None, None, :].clamp_min(1)
        expanded = pair_mask[:, None, None, :, None].to(q.dtype)
        per_carrier_q = (q * expanded).sum(dim=0) / carrier_count[None, None, :, None].clamp_min(1)
        per_carrier_coh = torch.linalg.vector_norm(per_carrier_q, dim=-1)
        rows.append({
            "distance": hop,
            "F_sens": per_carrier_sens[..., carrier_valid].mean(dim=-1).numpy(),
            "F_coh": per_carrier_coh[..., carrier_valid].mean(dim=-1).numpy(),
            "carriers": int(carrier_valid.sum()),
            "event_carrier_pairs": int(pair_mask.sum()),
        })
    return rows


def score_graph_channel(
    gm: Any,
    plan: Mapping[str, Any],
    channel: str,
    phi: Sequence[Any],
) -> dict[str, Any]:
    import torch

    base = gm.eval_ds[int(plan["graph_id"])]
    groups = graph_event_variants(gm, plan, channel)
    if not groups or not any(groups):
        return {"available": False, "channel": channel, "graph_id": int(plan["graph_id"])}
    q_groups = []
    prediction_groups = []
    for variants in groups:
        q, predictions, _no_op = projected_event_group(gm, base, variants, phi)
        if q.numel():
            q_groups.append(q)
            prediction_groups.append(predictions)
    if not q_groups:
        return {"available": False, "channel": channel, "graph_id": int(plan["graph_id"])}
    max_events = max(int(q.shape[0]) for q in q_groups)
    n = int(base.num_nodes)
    q_all = torch.zeros(len(q_groups), max_events, gm.L, gm.H, n, len(phi[0]))
    valid = torch.zeros(len(q_groups), max_events, dtype=torch.bool)
    for source, q in enumerate(q_groups):
        q_all[source, :len(q)] = q
        valid[source, :len(q)] = True
    if channel == "topology":
        aggregation = aggregate_topology_events(q_all[0:1], valid[0:1])
        per_graph = {key: value[0].numpy() for key, value in aggregation["per_graph"].items()}
        per_source = {key: value[0].numpy() for key, value in aggregation["per_source"].items()}
    else:
        aggregation = aggregate_projected_events(q_all[None], valid[None])
        per_graph = {key: value[0].numpy() for key, value in aggregation["per_graph"].items()}
        per_source = {key: value[0].numpy() for key, value in aggregation["per_source"].items()}

    edges = np.asarray(plan["descriptor"]["edges"], dtype=np.int64)
    distance = shortest_paths(n, edges)
    carriage_rows = []
    for source_index, q in enumerate(q_groups):
        if channel == "semantic":
            source = int(plan["semantic"][source_index]["source"])
            event_distances = np.repeat(distance[:, source][None, :], len(q), axis=0)
        elif channel == "pe":
            item = plan["pe"][source_index]
            source = int(item["source"])
            event_distances = np.stack([
                np.minimum(distance[:, source], distance[:, int(partner)])
                for partner in item["partners"]
            ])
        else:
            event_distances = []
            for item in plan["topology"]:
                changed = list(map(int, item["dose"]["changed_nodes"]))
                event_distances.append(
                    np.min(distance[:, changed], axis=1) if changed else np.full(n, np.inf)
                )
            event_distances = np.stack(event_distances)
        # Encode disconnected/undefined carriers as -1 so they are omitted, not a fake far bin.
        event_distances = np.where(np.isfinite(event_distances), event_distances, -1).astype(int)
        for row in distance_profile_one_graph(q, event_distances):
            carriage_rows.append({"source_index": int(source_index), **row})

    y = base.y.detach().cpu().numpy().reshape(1, -1)
    event_outcomes = []
    for predictions in prediction_groups:
        clean_loss = np.abs(predictions[0:1] - y).mean(axis=1)[0]
        corrupt_loss = np.abs(predictions[1:] - y).mean(axis=1)
        event_outcomes.extend((clean_loss - corrupt_loss).tolist())
    return {
        "available": True,
        "channel": channel,
        "graph_id": int(plan["graph_id"]),
        "per_graph": per_graph,
        "per_source": per_source,
        "q_groups": q_groups,
        "carriage": carriage_rows,
        "event_outcomes": np.asarray(event_outcomes, dtype=np.float32),
    }


def donor_prefix_scores(q_groups: Sequence[Any], prefixes: Sequence[int]) -> dict[str, Any]:
    import torch

    if not q_groups:
        return {}
    n = int(q_groups[0].shape[-2])
    output = {}
    for prefix in prefixes:
        count = min(int(prefix), min(int(q.shape[0]) for q in q_groups))
        q_all = torch.stack([q[:count] for q in q_groups], dim=0)
        valid = torch.ones(q_all.shape[:2], dtype=torch.bool)
        aggregate = aggregate_projected_events(q_all[None], valid[None])
        output[str(prefix)] = {
            key: value[0].numpy() for key, value in aggregate["per_graph"].items()
        }
    return output


def verify_interventions(
    gm: Any,
    plan: Mapping[str, Any],
    phi: Sequence[Any],
) -> dict[str, Any]:
    """Fail-closed content/structure/no-op/isomorphism checks on one real molecule."""

    import torch

    from graph_specialisation_metrics.carriage import structural

    base = gm.eval_ds[int(plan["graph_id"])]
    source = int(plan["sources"][0])
    current_row = node_labels(base)[source]
    semantic_noop = semantic_variant(base, source, current_row)
    pe_noop = pe_variant(base, source, source)
    semantic_q, semantic_predictions, _ = projected_event_group(
        gm, base, [semantic_noop], phi
    )
    pe_q, pe_predictions, _ = projected_event_group(gm, base, [pe_noop], phi)
    checks = {
        "semantic_noop_prediction": tensor_equivalent(
            torch.as_tensor(semantic_predictions[0]), torch.as_tensor(semantic_predictions[1])
        ),
        "pe_noop_prediction": tensor_equivalent(
            torch.as_tensor(pe_predictions[0]), torch.as_tensor(pe_predictions[1])
        ),
        "semantic_noop_q_max": float(semantic_q.abs().max()),
        "pe_noop_q_max": float(pe_q.abs().max()),
    }
    actual_semantic = semantic_variant(
        base, source, np.asarray(plan["semantic"][0]["donor_rows"][0])
    )
    for field in ("edge_index", "edge_attr", "rrwp", "rrwp_index", "rrwp_val", "deg", "log_deg"):
        left, right = getattr(base, field, None), getattr(actual_semantic, field, None)
        if left is not None:
            if right is None or not torch.equal(left, right):
                raise RuntimeError(f"semantic intervention changed structural field {field}")
    actual_pe = pe_variant(
        base, source, int(plan["pe"][0]["partners"][0])
    )
    for field in ("x", "y", "edge_index", "edge_attr", "rrwp_local_edge_index"):
        left, right = getattr(base, field, None), getattr(actual_pe, field, None)
        if left is not None and (right is None or not torch.equal(left, right)):
            raise RuntimeError(f"mask-frozen PE intervention changed fixed field {field}")
    changed_pe = any(
        getattr(base, field, None) is not None
        and not torch.equal(getattr(base, field), getattr(actual_pe, field))
        for field in STRUCTURAL_NODE_FIELDS + ("rrwp_index",)
    )
    if not changed_pe:
        raise RuntimeError("selected PE transposition changed no structural payload")
    partner = int(plan["pe"][0]["partners"][0])
    relabelled = structural.full_relabel(base, source, partner)
    relabel_predictions = forward_data_list(gm, [base, relabelled])
    checks["full_relabel_prediction"] = tensor_equivalent(
        torch.as_tensor(relabel_predictions[0]), torch.as_tensor(relabel_predictions[1])
    )
    checks["semantic_fixed_structure"] = True
    checks["pe_fixed_content_and_mask"] = True
    checks["pe_payload_changed"] = True
    if plan["topology"]:
        item = plan["topology"][0]
        donor = gm.donor_ds[int(item["donor_graph_id"])]
        variant = topology_donor_variant(
            base, donor, np.asarray(item["alignment"], dtype=np.int64)
        )
        if not torch.equal(variant.x, base.x) or not torch.equal(variant.y, base.y):
            raise RuntimeError("topology donor changed base content/target")
        expected_edges = relabel_pair_index_numpy(
            molecular_edges(donor), np.asarray(item["alignment"], dtype=np.int64)
        )
        observed_edges = molecular_edges(variant)
        if not np.array_equal(np.unique(expected_edges, axis=0), observed_edges):
            raise RuntimeError("topology donor molecular support was not transplanted exactly")
        checks["topology_content_target_fixed"] = True
        checks["topology_content_alignment_exact"] = bool(item["content_alignment_exact"])
        checks["topology_support_exact"] = True
        checks["topology_nonisomorphic"] = True
    required = (
        checks["semantic_noop_prediction"]["passed"],
        checks["pe_noop_prediction"]["passed"],
        checks["full_relabel_prediction"]["passed"],
        checks["semantic_noop_q_max"] <= CUDA_EQUIVALENCE_ATOL,
        checks["pe_noop_q_max"] <= CUDA_EQUIVALENCE_ATOL,
    )
    if not all(required):
        raise RuntimeError(f"intervention verification failed: {checks}")
    return checks


def run_scores_task(
    loaded: Mapping[str, Any], cfg: BetaConfig, *, force: bool = False
) -> dict[str, Any]:
    import torch

    gm, checkpoint_sha = loaded["gm"], loaded["sha"]
    task_name = loaded["spec"].name
    path = cache_path(cfg, task_name, "scores", checkpoint_sha)
    cached = None if force else valid_cache(path, cfg, checkpoint_sha)
    if cached is not None:
        print(f"[scores:{task_name}] cache hit {path}", flush=True)
        return cached
    splits = deterministic_splits(len(gm.eval_ds), cfg)
    donor_rows, donor_graph_ids, _donor_node_ids = donor_row_pool(gm)
    descriptors, by_n = load_or_build_topology_index(
        gm, cfg, task_name, checkpoint_sha
    )
    field_audit = structural_field_audit(gm.eval_ds[int(splits["score"][0])])
    if field_audit["unhandled_structural"]:
        raise RuntimeError(
            "unhandled topology-derived fields: " + ", ".join(field_audit["unhandled_structural"])
        )
    records = []
    verification = None
    for position, graph_id in enumerate(splits["score"]):
        graph_id = int(graph_id)
        chunk_path = cfg.root / "cache" / task_name / "score_graphs" / (
            f"graph_{graph_id}__{cfg.fingerprint}__{checkpoint_sha[:12]}.pt"
        )
        chunk = None if force else valid_cache(chunk_path, cfg, checkpoint_sha)
        if chunk is not None:
            records.append(chunk["record"])
            if verification is None and chunk.get("intervention_verification") is not None:
                verification = chunk["intervention_verification"]
            print(
                f"[scores:{task_name}] graph {position + 1}/{len(splits['score'])} "
                f"id={graph_id} cache hit", flush=True,
            )
            continue
        print(
            f"[scores:{task_name}] graph {position + 1}/{len(splits['score'])} id={graph_id}",
            flush=True,
        )
        base = gm.eval_ds[graph_id]
        clean_prediction, phi, clean_capture = clean_gradients(gm, base)
        plan = plan_graph_events(
            gm,
            graph_id,
            cfg,
            donor_rows=donor_rows,
            donor_graph_ids=donor_graph_ids,
            topology_descriptors=descriptors,
            topology_by_n=by_n,
        )
        verified_this_graph = False
        if verification is None:
            verification = verify_interventions(gm, plan, phi)
            verified_this_graph = True
        channel_records = {
            channel: score_graph_channel(gm, plan, channel, phi) for channel in CHANNELS
        }
        prefixes = {}
        for channel in PRIMARY_CHANNELS:
            if channel_records[channel]["available"]:
                prefixes[channel] = donor_prefix_scores(
                    channel_records[channel]["q_groups"], (1, 2, 4, 8)
                )
        record = {
            "graph_id": graph_id,
            "n": int(base.num_nodes),
            "clean_prediction": clean_prediction.cpu().numpy(),
            "target": base.y.detach().cpu().numpy(),
            "plan": plan,
            "channels": channel_records,
            "prefixes": prefixes,
            "throughput": np.stack([
                value.detach().norm(dim=-1).mean(dim=0).cpu().numpy()
                for value in clean_capture["wV"]
            ]),
        }
        records.append(record)
        atomic_torch_save({
            "version": BETA_VERSION,
            "schema": BETA_SCHEMA,
            "fingerprint": cfg.fingerprint,
            "checkpoint_sha256": checkpoint_sha,
            "task": task_name,
            "graph_id": graph_id,
            "record": record,
            "intervention_verification": verification if verified_this_graph else None,
        }, chunk_path)
        del phi, clean_capture
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "version": BETA_VERSION,
        "schema": BETA_SCHEMA,
        "fingerprint": cfg.fingerprint,
        "checkpoint_sha256": checkpoint_sha,
        "task": task_name,
        "L": gm.L,
        "H": gm.H,
        "splits": splits,
        "records": sorted(records, key=lambda item: int(item["graph_id"])),
        "field_audit": field_audit,
        "intervention_verification": verification,
    }
    atomic_torch_save(payload, path)
    return payload


def stack_graph_scores(
    score_payload: Mapping[str, Any],
    channel: str,
    aggregation: str,
    *,
    topology_max_tier: int | None = 1,
) -> np.ndarray:
    values = []
    for record in score_payload["records"]:
        result = record["channels"][channel]
        if not result["available"]:
            continue
        if channel == "topology" and topology_max_tier is not None:
            import torch

            tiers = np.asarray([item["tier"] for item in record["plan"]["topology"]])
            keep = np.flatnonzero(tiers <= int(topology_max_tier))
            if not keep.size:
                continue
            q = result["q_groups"][0][torch.as_tensor(keep, dtype=torch.long)]
            valid = torch.ones(1, len(q), dtype=torch.bool)
            subset = aggregate_topology_events(q[None], valid)
            values.append(subset["per_graph"][aggregation][0].numpy())
        else:
            values.append(result["per_graph"][aggregation])
    if not values:
        return np.empty((0, int(score_payload["L"]), int(score_payload["H"])))
    return np.stack(values)


# =====================================================================================
# Independent whole-transport patching and causal controls
# =====================================================================================


@contextlib.contextmanager
def patched_head(gm: Any, layer: int, head: int, source: Any | None):
    """Patch one full node-aligned wV head, or zero it when source is None."""

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        h_out, e_out = output
        changed = h_out.clone()
        if source is None:
            changed[:, int(head), :] = 0.0
        else:
            if tuple(source.shape) != tuple(h_out.shape):
                raise RuntimeError(
                    f"patch tensor {tuple(source.shape)} != transport {tuple(h_out.shape)}"
                )
            changed[:, int(head), :] = source[:, int(head), :]
        return changed, e_out

    handle = gm.attn_layers[int(layer)].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def annotate_graph_num_nodes(batch: Any, data_list: Sequence[Any]) -> Any:
    """Attach the per-graph node counts required by mechanistic collectors.

    Ordinary ``torch_geometric.data.Batch`` objects expose ``ptr``/``batch`` but
    not the repository-specific ``graph_num_nodes`` field.  The shared GRIT
    collector deliberately consumes the latter so that sparse edge indices can
    be converted to graph-local coordinates.  Keep the compatibility adapter in
    this beta analysis rather than changing the production collector.
    """

    import torch

    counts = [int(data.num_nodes) for data in data_list]
    if not counts or any(count <= 0 for count in counts):
        raise RuntimeError(f"invalid graph node counts for GRIT batch: {counts}")
    observed = int(batch.num_nodes)
    if sum(counts) != observed:
        raise RuntimeError(
            f"graph node counts sum to {sum(counts)}, but PyG batch has {observed} nodes"
        )
    batch.graph_num_nodes = torch.as_tensor(counts, dtype=torch.long)
    return batch


def make_grit_batch(data_list: Sequence[Any], device: Any) -> Any:
    """Clone, collate, annotate and move a list of graphs to ``device``."""

    from torch_geometric.data import Batch

    batch = Batch.from_data_list([data.clone() for data in data_list])
    return annotate_graph_num_nodes(batch, data_list).to(device)


def forward_data_list(
    gm: Any,
    data_list: Sequence[Any],
    *,
    patch: tuple[int, int, Any | None] | None = None,
) -> np.ndarray:
    import torch

    batch = make_grit_batch(data_list, gm.device)
    manager = contextlib.nullcontext()
    if patch is not None:
        manager = patched_head(gm, patch[0], patch[1], patch[2])
    with manager, torch.no_grad():
        prediction, _target = gm.model(batch)
    return prediction.detach().cpu().numpy().reshape(len(data_list), -1)


def capture_transport_list(gm: Any, data_list: Sequence[Any]) -> tuple[np.ndarray, list[np.ndarray]]:
    batch = make_grit_batch(data_list, gm.device)
    capture = gm.capture(batch, want_grad=False, want_attn=False, include_virtual_transport=True)
    counts = [int(data.num_nodes) for data in data_list]
    if sum(counts) != int(capture["wV"][0].shape[0]):
        raise RuntimeError("unexpected transport row count in ZINC patch capture")
    layers = []
    for value in capture["wV"]:
        cpu = value.detach().cpu().numpy()
        offsets = np.cumsum([0, *counts])
        layers.append([cpu[offsets[i]:offsets[i + 1]] for i in range(len(counts))])
    per_event = [
        np.stack([layers[layer][event] for layer in range(gm.L)])
        for event in range(len(data_list))
    ]
    prediction = capture["pred"].detach().cpu().numpy().reshape(len(data_list), -1)
    return prediction, per_event


def causal_event_catalog(
    gm: Any,
    cfg: BetaConfig,
    graph_ids: Sequence[int],
    *,
    donor_rows: np.ndarray,
    donor_graph_ids: np.ndarray,
    topology_descriptors: Sequence[Mapping[str, Any]],
    topology_by_n: Mapping[int, Sequence[int]],
) -> list[dict[str, Any]]:
    events = []
    for graph_id in graph_ids:
        plan = plan_graph_events(
            gm,
            int(graph_id),
            cfg,
            donor_rows=donor_rows,
            donor_graph_ids=donor_graph_ids,
            topology_descriptors=topology_descriptors,
            topology_by_n=topology_by_n,
            sources_limit=cfg.causal_sources,
        )
        base = gm.eval_ds[int(graph_id)]
        for source_index, item in enumerate(plan["semantic"]):
            events.append({
                "graph_id": int(graph_id),
                "channel": "semantic",
                "source": int(item["source"]),
                "event_index": int(source_index),
                "clean": base,
                "corrupt": semantic_variant(base, int(item["source"]), item["donor_rows"][0]),
                "descriptor": plan["descriptor"],
                "donor_id": int(item["donor_graph_ids"][0]),
            })
        for source_index, item in enumerate(plan["pe"]):
            events.append({
                "graph_id": int(graph_id),
                "channel": "pe",
                "source": int(item["source"]),
                "event_index": int(source_index),
                "clean": base,
                "corrupt": pe_variant(base, int(item["source"]), int(item["partners"][0])),
                "descriptor": plan["descriptor"],
                "partner": int(item["partners"][0]),
            })
        if plan["topology"]:
            item = plan["topology"][0]
            events.append({
                "graph_id": int(graph_id),
                "channel": "topology",
                "source": -1,
                "event_index": 0,
                "clean": base,
                "corrupt": topology_donor_variant(
                    base,
                    gm.donor_ds[int(item["donor_graph_id"])],
                    np.asarray(item["alignment"], dtype=np.int64),
                ),
                "descriptor": plan["descriptor"],
                "donor_id": int(item["donor_graph_id"]),
                "topology_tier": int(item["tier"]),
                "topology_dose": item["dose"],
            })
    return events


def add_cross_graph_mismatch(events: list[dict[str, Any]]) -> None:
    """Attach a deterministic different-graph, same-n clean transport control."""

    for index, event in enumerate(events):
        candidates = [
            candidate for candidate, other in enumerate(events)
            if other["channel"] == event["channel"]
            and other["graph_id"] != event["graph_id"]
            and int(other["descriptor"]["n"]) == int(event["descriptor"]["n"])
        ]
        if not candidates:
            event["mismatch_event"] = None
            event["mismatch_alignment"] = None
            continue
        candidate = candidates[index % len(candidates)]
        event["mismatch_event"] = int(candidate)
        event["mismatch_alignment"] = align_donor_nodes(
            event["descriptor"], events[candidate]["descriptor"]
        )


def concatenate_layer_transport(
    event_records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    layer: int,
    kind: str,
    device: Any,
) -> Any:
    import torch

    arrays = [np.asarray(event_records[index][kind][layer]) for index in indices]
    return torch.as_tensor(np.concatenate(arrays, axis=0), device=device)


def concatenate_mismatch_transport(
    event_records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    layer: int,
    device: Any,
) -> tuple[Any, np.ndarray]:
    import torch

    arrays = []
    valid = []
    for index in indices:
        event = event_records[index]
        mismatch = event.get("mismatch_event")
        alignment = event.get("mismatch_alignment")
        if mismatch is None or alignment is None:
            arrays.append(np.asarray(event["corrupt_wv"][layer]))
            valid.append(False)
        else:
            source = np.asarray(event_records[int(mismatch)]["clean_wv"][layer])
            arrays.append(source[np.asarray(alignment, dtype=np.int64)])
            valid.append(True)
    return torch.as_tensor(np.concatenate(arrays, axis=0), device=device), np.asarray(valid, bool)


def run_causal_task(
    loaded: Mapping[str, Any], cfg: BetaConfig, *, force: bool = False
) -> dict[str, Any]:
    import torch

    gm, checkpoint_sha = loaded["gm"], loaded["sha"]
    task_name = loaded["spec"].name
    path = cache_path(cfg, task_name, "causal", checkpoint_sha)
    cached = None if force else valid_cache(path, cfg, checkpoint_sha)
    if cached is not None:
        print(f"[causal:{task_name}] cache hit {path}", flush=True)
        return cached
    splits = deterministic_splits(len(gm.eval_ds), cfg)
    donor_rows, donor_graph_ids, _ = donor_row_pool(gm)
    descriptors, by_n = load_or_build_topology_index(
        gm, cfg, task_name, checkpoint_sha
    )
    events = causal_event_catalog(
        gm,
        cfg,
        splits["causal"],
        donor_rows=donor_rows,
        donor_graph_ids=donor_graph_ids,
        topology_descriptors=descriptors,
        topology_by_n=by_n,
    )
    if not events:
        raise RuntimeError("causal catalog is empty")
    batch_size = max(1, int(cfg.causal_batch_graphs))
    for start in range(0, len(events), batch_size):
        chunk = events[start:start + batch_size]
        clean_prediction, clean_wv = capture_transport_list(gm, [item["clean"] for item in chunk])
        corrupt_prediction, corrupt_wv = capture_transport_list(gm, [item["corrupt"] for item in chunk])
        for local, item in enumerate(chunk):
            item["clean_prediction"] = clean_prediction[local]
            item["corrupt_prediction"] = corrupt_prediction[local]
            item["clean_wv"] = clean_wv[local]
            item["corrupt_wv"] = corrupt_wv[local]
            item["target"] = item["clean"].y.detach().cpu().numpy().reshape(-1)
    add_cross_graph_mismatch(events)

    event_count = len(events)
    shape = (event_count, gm.L, gm.H)
    metrics = {
        name: np.full(shape, np.nan, dtype=np.float32)
        for name in (
            "restore", "inject", "necessity", "mismatch", "sham",
            "restore_fraction", "inject_fraction", "necessity_fraction",
            "restore_loss", "inject_loss", "necessity_loss",
        )
    }
    progress_path = cache_path(cfg, task_name, "causal_progress", checkpoint_sha)
    progress = None if force else valid_cache(progress_path, cfg, checkpoint_sha)
    completed_chunks: set[int] = set()
    if progress is not None:
        if int(progress.get("event_count", -1)) != event_count:
            raise RuntimeError("causal progress event catalog size changed")
        metrics = {key: np.asarray(value) for key, value in progress["metrics"].items()}
        completed_chunks = {int(value) for value in progress.get("completed_chunks", [])}
    baseline_check = tensor_equivalent(
        torch.as_tensor(events[0]["clean_prediction"]),
        torch.as_tensor(forward_data_list(gm, [events[0]["clean"]])[0]),
    )
    if not baseline_check["passed"]:
        raise RuntimeError(f"batch-invariance check failed: {baseline_check}")

    for start in range(0, event_count, batch_size):
        if start in completed_chunks:
            print(f"[causal:{task_name}] chunk starting {start} cache hit", flush=True)
            continue
        indices = list(range(start, min(start + batch_size, event_count)))
        clean_data = [events[index]["clean"] for index in indices]
        corrupt_data = [events[index]["corrupt"] for index in indices]
        clean = np.stack([events[index]["clean_prediction"] for index in indices])
        corrupt = np.stack([events[index]["corrupt_prediction"] for index in indices])
        target = np.stack([events[index]["target"] for index in indices])
        delta = clean - corrupt
        direction = np.sign(delta)
        denominator = np.square(delta) + 1.0e-8 * np.maximum(np.square(clean), 1.0)
        for layer in range(gm.L):
            clean_source = concatenate_layer_transport(
                events, indices, layer=layer, kind="clean_wv", device=gm.device
            )
            corrupt_source = concatenate_layer_transport(
                events, indices, layer=layer, kind="corrupt_wv", device=gm.device
            )
            mismatch_source, mismatch_valid = concatenate_mismatch_transport(
                events, indices, layer=layer, device=gm.device
            )
            for head in range(gm.H):
                restore_inject = forward_data_list(
                    gm,
                    [*corrupt_data, *clean_data],
                    patch=(layer, head, torch.cat([clean_source, corrupt_source], dim=0)),
                )
                restored = restore_inject[:len(indices)]
                injected = restore_inject[len(indices):]
                ablated = forward_data_list(
                    gm, [*clean_data, *corrupt_data], patch=(layer, head, None)
                )
                ablated_clean = ablated[:len(indices)]
                ablated_corrupt = ablated[len(indices):]
                mismatch_sham = forward_data_list(
                    gm,
                    [*corrupt_data, *corrupt_data],
                    patch=(layer, head, torch.cat([mismatch_source, corrupt_source], dim=0)),
                )
                mismatched = mismatch_sham[:len(indices)]
                sham = mismatch_sham[len(indices):]
                movements = {
                    "restore": restored - corrupt,
                    "inject": clean - injected,
                    "necessity": delta - (ablated_clean - ablated_corrupt),
                    "mismatch": mismatched - corrupt,
                    "sham": sham - corrupt,
                }
                for name, movement in movements.items():
                    desired = np.mean(movement * direction, axis=1)
                    if name == "mismatch":
                        desired = np.where(mismatch_valid, desired, np.nan)
                    metrics[name][indices, layer, head] = desired
                metrics["restore_fraction"][indices, layer, head] = np.mean(
                    (restored - corrupt) * delta / denominator, axis=1
                )
                metrics["inject_fraction"][indices, layer, head] = np.mean(
                    (clean - injected) * delta / denominator, axis=1
                )
                metrics["necessity_fraction"][indices, layer, head] = np.mean(
                    (delta - (ablated_clean - ablated_corrupt)) * delta / denominator, axis=1
                )
                clean_loss = np.abs(clean - target).mean(axis=1)
                corrupt_loss = np.abs(corrupt - target).mean(axis=1)
                metrics["restore_loss"][indices, layer, head] = (
                    corrupt_loss - np.abs(restored - target).mean(axis=1)
                )
                metrics["inject_loss"][indices, layer, head] = (
                    np.abs(injected - target).mean(axis=1) - clean_loss
                )
                ablated_clean_loss = np.abs(ablated_clean - target).mean(axis=1)
                ablated_corrupt_loss = np.abs(ablated_corrupt - target).mean(axis=1)
                metrics["necessity_loss"][indices, layer, head] = (
                    (corrupt_loss - clean_loss)
                    - (ablated_corrupt_loss - ablated_clean_loss)
                )
            print(
                f"[causal:{task_name}] events {indices[0] + 1}-{indices[-1] + 1}/{event_count} "
                f"layer {layer + 1}/{gm.L}", flush=True,
            )
        completed_chunks.add(start)
        atomic_torch_save({
            "version": BETA_VERSION,
            "schema": BETA_SCHEMA,
            "fingerprint": cfg.fingerprint,
            "checkpoint_sha256": checkpoint_sha,
            "task": task_name,
            "event_count": event_count,
            "metrics": metrics,
            "completed_chunks": sorted(completed_chunks),
        }, progress_path)

    metadata = []
    for event in events:
        metadata.append({
            key: value for key, value in event.items()
            if key not in {"clean", "corrupt", "clean_wv", "corrupt_wv", "descriptor"}
        })
    payload = {
        "version": BETA_VERSION,
        "schema": BETA_SCHEMA,
        "fingerprint": cfg.fingerprint,
        "checkpoint_sha256": checkpoint_sha,
        "task": task_name,
        "L": gm.L,
        "H": gm.H,
        "events": metadata,
        "metrics": metrics,
        "batch_invariance": baseline_check,
        "causal_effect_floor_relative": cfg.causal_effect_floor_relative,
    }
    atomic_torch_save(payload, path)
    return payload


def channel_mean_metric(
    causal_payload: Mapping[str, Any],
    channel: str,
    metric: str,
    *,
    effect_gated: bool = True,
    topology_max_tier: int | None = 1,
) -> np.ndarray:
    events = causal_payload["events"]
    mask = np.asarray([event["channel"] == channel for event in events])
    if channel == "topology" and topology_max_tier is not None:
        mask &= np.asarray([
            int(event.get("topology_tier", 99)) <= int(topology_max_tier)
            for event in events
        ])
    if effect_gated and mask.any():
        effects = np.asarray([
            float(np.mean(np.abs(
                np.asarray(event["clean_prediction"], float)
                - np.asarray(event["corrupt_prediction"], float)
            )))
            for event in events
        ])
        channel_effects = effects[mask]
        floor = float(causal_payload.get("causal_effect_floor_relative", 0.05)) * float(
            np.nanmax(channel_effects)
        )
        mask &= effects >= floor
    values = np.asarray(causal_payload["metrics"][metric], dtype=float)
    return np.nanmean(values[mask], axis=0) if mask.any() else np.full(values.shape[1:], np.nan)


# =====================================================================================
# Aggregation comparison, gated family selection, and independent ablation
# =====================================================================================


def mean_score(score_payload: Mapping[str, Any], channel: str, aggregation: str) -> np.ndarray:
    values = stack_graph_scores(score_payload, channel, aggregation)
    if not len(values):
        return np.full((int(score_payload["L"]), int(score_payload["H"])), np.nan)
    return np.nanmean(values, axis=0)


def layer_centered(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values - np.nanmean(values, axis=1, keepdims=True)


def partial_spearman(left: np.ndarray, right: np.ndarray, controls: np.ndarray) -> float:
    from scipy.stats import rankdata

    left = np.asarray(left, float).reshape(-1)
    right = np.asarray(right, float).reshape(-1)
    controls = np.asarray(controls, float).reshape(len(left), -1)
    keep = np.isfinite(left) & np.isfinite(right) & np.isfinite(controls).all(axis=1)
    if keep.sum() < controls.shape[1] + 4:
        return float("nan")
    design = np.column_stack([np.ones(keep.sum()), controls[keep]])
    left_rank = rankdata(left[keep])
    right_rank = rankdata(right[keep])
    left_residual = left_rank - design @ np.linalg.lstsq(design, left_rank, rcond=None)[0]
    right_residual = right_rank - design @ np.linalg.lstsq(design, right_rank, rcond=None)[0]
    if np.std(left_residual) <= EPS or np.std(right_residual) <= EPS:
        return float("nan")
    return float(np.corrcoef(left_residual, right_residual)[0, 1])


def causal_coordinates(causal_payload: Mapping[str, Any], metric: str = "restore") -> dict[str, np.ndarray]:
    semantic = channel_mean_metric(causal_payload, "semantic", metric)
    structural = channel_mean_metric(causal_payload, "pe", metric)
    total_magnitude = np.abs(semantic) + np.abs(structural)
    return {
        "semantic": semantic,
        "structural": structural,
        "D": (semantic - structural) / (total_magnitude + EPS),
        "J": 0.5 * total_magnitude,
        "G": np.sqrt(np.maximum(semantic, 0.0) * np.maximum(structural, 0.0)),
        "semantic_anti_aligned": semantic < 0.0,
        "structural_anti_aligned": structural < 0.0,
    }


def aggregation_diagnostics(
    score_payload: Mapping[str, Any], causal_payload: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    causal = causal_coordinates(causal_payload, "restore")
    causal_necessity = causal_coordinates(causal_payload, "necessity")
    L, H = int(score_payload["L"]), int(score_payload["H"])
    layers = _head_layers(L, H)
    layer_one_hot = np.eye(L)[layers]
    throughput = np.mean(np.stack([
        np.asarray(record["throughput"], float) for record in score_payload["records"]
    ]), axis=0).reshape(-1)
    for aggregation in AGGREGATIONS:
        semantic = mean_score(score_payload, "semantic", aggregation)
        structural = mean_score(score_payload, "pe", aggregation)
        coordinates = score_coordinates(semantic, structural)
        j_gate = coordinates["J"] >= 0.10 * np.nanmax(coordinates["J"])
        for scope, transform in (("pooled", lambda x: x), ("within_layer", layer_centered)):
            sem_score = transform(semantic).reshape(-1)
            str_score = transform(structural).reshape(-1)
            d_score = transform(coordinates["D"]).reshape(-1)
            j_score = transform(coordinates["J"]).reshape(-1)
            g_score = transform(coordinates["G"]).reshape(-1)
            gate = j_gate.reshape(-1)
            rows.append({
                "aggregation": aggregation,
                "scope": scope,
                "rho_sem_restore": spearman(sem_score, transform(causal["semantic"]).reshape(-1)),
                "rho_pe_restore": spearman(str_score, transform(causal["structural"]).reshape(-1)),
                "rho_J_restore": spearman(j_score, transform(causal["J"]).reshape(-1)),
                "rho_G_restore": spearman(g_score, transform(causal["G"]).reshape(-1)),
                "rho_D_restore_ungated": spearman(d_score, transform(causal["D"]).reshape(-1)),
                "rho_D_restore_gated": spearman(d_score[gate], transform(causal["D"]).reshape(-1)[gate]),
                "rho_D_necessity_gated": spearman(
                    d_score[gate], transform(causal_necessity["D"]).reshape(-1)[gate]
                ),
                "rho_G_beyond_J_and_layer": partial_spearman(
                    coordinates["G"].reshape(-1),
                    causal["G"].reshape(-1),
                    np.column_stack([coordinates["J"].reshape(-1), layer_one_hot]),
                ),
                "rho_J_beyond_layer_and_throughput": partial_spearman(
                    coordinates["J"].reshape(-1),
                    causal["J"].reshape(-1),
                    np.column_stack([throughput, layer_one_hot]),
                ),
            })
    return rows


def prefix_reliability(score_payload: Mapping[str, Any], aggregation: str) -> dict[str, float]:
    values = {channel: {} for channel in PRIMARY_CHANNELS}
    for channel in PRIMARY_CHANNELS:
        for record in score_payload["records"]:
            for prefix, payload in record.get("prefixes", {}).get(channel, {}).items():
                values[channel].setdefault(prefix, []).append(payload[aggregation])
    correlations = []
    topk = []
    for channel in PRIMARY_CHANNELS:
        if "all" in values[channel]:
            reference_key = "all"
        else:
            numeric = sorted(values[channel], key=lambda key: int(key))
            if not numeric:
                continue
            reference_key = numeric[-1]
        reference = np.mean(np.stack(values[channel][reference_key]), axis=0).reshape(-1)
        for prefix, items in values[channel].items():
            if prefix == reference_key:
                continue
            candidate = np.mean(np.stack(items), axis=0).reshape(-1)
            correlations.append(spearman(candidate, reference))
            count = min(3, len(reference))
            reference_top = set(np.argsort(reference, kind="stable")[-count:])
            candidate_top = set(np.argsort(candidate, kind="stable")[-count:])
            topk.append(len(reference_top & candidate_top) / max(len(reference_top | candidate_top), 1))
    return {
        "rank_rho": float(np.nanmean(correlations)) if correlations else float("nan"),
        "top3_jaccard": float(np.nanmean(topk)) if topk else float("nan"),
    }


def choose_aggregation(
    score_payload: Mapping[str, Any], causal_payload: Mapping[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    rows = aggregation_diagnostics(score_payload, causal_payload)
    scored = []
    for aggregation in AGGREGATIONS:
        relevant = [row for row in rows if row["aggregation"] == aggregation]
        validity = np.nanmean([
            row[key]
            for row in relevant
            for key in (
                "rho_sem_restore", "rho_pe_restore", "rho_J_restore", "rho_G_restore",
                "rho_D_restore_gated", "rho_D_necessity_gated",
            )
        ])
        reliability = prefix_reliability(score_payload, aggregation)
        reliability_mean = finite_mean([reliability["rank_rho"], reliability["top3_jaccard"]])
        objective = float(np.nanmean([validity, reliability_mean]))
        scored.append((objective, aggregation, validity, reliability))
    scored.sort(key=lambda item: (-np.nan_to_num(item[0], nan=-np.inf), AGGREGATIONS.index(item[1])))
    winner = scored[0][1]
    for objective, aggregation, validity, reliability in scored:
        rows.append({
            "aggregation": aggregation,
            "scope": "decision",
            "validity_mean": validity,
            "prefix_reliability": reliability["rank_rho"],
            "prefix_top3_jaccard": reliability["top3_jaccard"],
            "objective": objective,
            "selected": aggregation == winner,
        })
    return winner, rows


def bootstrap_d_intervals(
    semantic_graphs: np.ndarray,
    structural_graphs: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    count = min(len(semantic_graphs), len(structural_graphs))
    values = []
    for _ in range(samples):
        index = rng.integers(0, count, count)
        coordinates = score_coordinates(
            semantic_graphs[index].mean(axis=0), structural_graphs[index].mean(axis=0)
        )
        values.append(coordinates["D"])
    stacked = np.stack(values)
    return np.quantile(stacked, 0.025, axis=0), np.quantile(stacked, 0.975, axis=0)


def head_order(values: np.ndarray, *, reverse: bool = True) -> list[tuple[int, int]]:
    flat = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(np.nan_to_num(flat, nan=-np.inf if reverse else np.inf), kind="stable")
    if reverse:
        order = order[::-1]
    return [tuple(map(int, np.unravel_index(index, values.shape))) for index in order]


def select_families(
    score_payload: Mapping[str, Any], cfg: BetaConfig, aggregation: str
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, Any]]:
    semantic_graphs = stack_graph_scores(score_payload, "semantic", aggregation)
    structural_graphs = stack_graph_scores(score_payload, "pe", aggregation)
    semantic = semantic_graphs.mean(axis=0)
    structural = structural_graphs.mean(axis=0)
    coordinates = score_coordinates(semantic, structural)
    lower, upper = bootstrap_d_intervals(
        semantic_graphs,
        structural_graphs,
        samples=cfg.bootstrap_samples,
        seed=cfg.analysis_seed + AGGREGATIONS.index(aggregation) * 101,
    )
    activity_floor = cfg.activity_floor_relative * float(np.nanmax(coordinates["J"]))
    active = coordinates["J"] >= activity_floor
    semantic_mask = (
        active & (coordinates["D"] >= cfg.selectivity_threshold) & (lower > 0.0)
    )
    structural_mask = (
        active & (coordinates["D"] <= -cfg.selectivity_threshold) & (upper < 0.0)
    )
    balanced = active & (lower <= 0.0) & (upper >= 0.0)

    def take(order: Sequence[tuple[int, int]], mask: np.ndarray | None = None) -> list[tuple[int, int]]:
        selected = []
        for head in order:
            if mask is not None and not bool(mask[head]):
                continue
            selected.append(head)
            if len(selected) == cfg.family_size:
                break
        return selected

    semantic_specialists = take(head_order(coordinates["D"]), semantic_mask)
    structural_specialists = take(head_order(-coordinates["D"]), structural_mask)

    def same_layer_controls(
        targets: Sequence[tuple[int, int]], candidate_mask: np.ndarray, *, match_j: bool
    ) -> list[tuple[int, int]]:
        chosen: list[tuple[int, int]] = []
        for target in targets:
            candidates = [
                (layer, head)
                for layer in range(candidate_mask.shape[0])
                for head in range(candidate_mask.shape[1])
                if layer == target[0]
                and bool(candidate_mask[layer, head])
                and (layer, head) not in chosen
                and (layer, head) not in targets
            ]
            if not candidates:
                continue
            if match_j:
                candidates.sort(key=lambda item: (abs(
                    coordinates["J"][item] - coordinates["J"][target]
                ), item))
            else:
                candidates.sort(key=lambda item: (coordinates["J"][item], item))
            chosen.append(candidates[0])
        return chosen

    topology = mean_score(score_payload, "topology", aggregation)
    families = {
        "raw_semantic": take(head_order(semantic)),
        "raw_pe": take(head_order(structural)),
        "semantic_specialist": semantic_specialists,
        "pe_specialist": structural_specialists,
        "semantic_same_layer_J_matched": same_layer_controls(
            semantic_specialists, balanced, match_j=True
        ),
        "pe_same_layer_J_matched": same_layer_controls(
            structural_specialists, balanced, match_j=True
        ),
        "semantic_same_layer_inert": same_layer_controls(
            semantic_specialists, ~active, match_j=False
        ),
        "pe_same_layer_inert": same_layer_controls(
            structural_specialists, ~active, match_j=False
        ),
        "high_J": take(head_order(coordinates["J"])),
        "high_G_balanced": take(head_order(coordinates["G"]), balanced),
        "low_J_inert": take(head_order(coordinates["J"], reverse=False)),
        "topology_responsive": take(head_order(topology)) if np.isfinite(topology).any() else [],
    }
    return families, {
        "semantic": semantic,
        "pe": structural,
        "topology": topology,
        "coordinates": coordinates,
        "D_ci_lower": lower,
        "D_ci_upper": upper,
        "activity_floor": activity_floor,
        "active": active,
        "families": families,
    }


def batch_groups(data: Sequence[Any], size: int = 64) -> list[list[Any]]:
    return [list(data[start:start + size]) for start in range(0, len(data), size)]


def ablation_summary(prediction: np.ndarray, clean: np.ndarray, target: np.ndarray) -> dict[str, float]:
    functional = np.abs(prediction - clean).mean()
    clean_loss = np.abs(clean - target).mean()
    ablated_loss = np.abs(prediction - target).mean()
    return {
        "functional": float(functional),
        "loss_increase": float(ablated_loss - clean_loss),
        "mae": float(ablated_loss),
    }


def run_ablation_task(
    loaded: Mapping[str, Any],
    cfg: BetaConfig,
    score_payload: Mapping[str, Any],
    causal_payload: Mapping[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    gm, checkpoint_sha = loaded["gm"], loaded["sha"]
    task_name = loaded["spec"].name
    path = cache_path(cfg, task_name, "ablations", checkpoint_sha)
    cached = None if force else valid_cache(path, cfg, checkpoint_sha)
    if cached is not None:
        print(f"[ablations:{task_name}] cache hit {path}", flush=True)
        return cached
    splits = deterministic_splits(len(gm.eval_ds), cfg)
    data = [gm.eval_ds[int(graph_id)] for graph_id in splits["ablation"]]
    groups = batch_groups(data, size=64)
    clean = gm.collect_preds_ablated(groups)
    target = np.stack([item.y.detach().cpu().numpy().reshape(-1) for item in data])
    per_head = {
        "functional": np.zeros((gm.L, gm.H), dtype=np.float32),
        "loss_increase": np.zeros((gm.L, gm.H), dtype=np.float32),
        "mae": np.zeros((gm.L, gm.H), dtype=np.float32),
    }
    for layer in range(gm.L):
        for head in range(gm.H):
            prediction = gm.collect_preds_ablated(groups, [(layer, head)])
            summary = ablation_summary(prediction, clean, target)
            for key in per_head:
                per_head[key][layer, head] = summary[key]
        print(f"[ablations:{task_name}] layer {layer + 1}/{gm.L}", flush=True)

    winner, method_rows = choose_aggregation(score_payload, causal_payload)
    for aggregation in AGGREGATIONS:
        semantic = mean_score(score_payload, "semantic", aggregation)
        structural = mean_score(score_payload, "pe", aggregation)
        coordinates = score_coordinates(semantic, structural)
        for scope, transform in (("ablation_pooled", lambda x: x), ("ablation_within_layer", layer_centered)):
            functional = transform(per_head["functional"]).reshape(-1)
            loss = transform(per_head["loss_increase"]).reshape(-1)
            method_rows.append({
                "aggregation": aggregation,
                "scope": scope,
                "rho_sem_functional_ablation": spearman(transform(semantic).reshape(-1), functional),
                "rho_pe_functional_ablation": spearman(transform(structural).reshape(-1), functional),
                "rho_J_functional_ablation": spearman(transform(coordinates["J"]).reshape(-1), functional),
                "rho_G_functional_ablation": spearman(transform(coordinates["G"]).reshape(-1), functional),
                "rho_J_loss_ablation": spearman(transform(coordinates["J"]).reshape(-1), loss),
            })
    selections = {}
    family_curves = {}
    for aggregation in AGGREGATIONS:
        families, diagnostics = select_families(score_payload, cfg, aggregation)
        selections[aggregation] = diagnostics
        family_curves[aggregation] = {}
        for family, heads in families.items():
            rows = []
            for count in range(1, len(heads) + 1):
                prediction = gm.collect_preds_ablated(groups, heads[:count])
                rows.append({"count": count, **ablation_summary(prediction, clean, target)})
            family_curves[aggregation][family] = rows
    payload = {
        "version": BETA_VERSION,
        "schema": BETA_SCHEMA,
        "fingerprint": cfg.fingerprint,
        "checkpoint_sha256": checkpoint_sha,
        "task": task_name,
        "winner": winner,
        "method_rows": method_rows,
        "per_head": per_head,
        "selections": selections,
        "family_curves": family_curves,
        "clean_mae": float(np.abs(clean - target).mean()),
        "graph_ids": splits["ablation"],
    }
    atomic_torch_save(payload, path)
    return payload


# =====================================================================================
# Exact fixed-support routing/message decomposition and component patching
# =====================================================================================


def capture_mechanism(gm: Any, data: Any) -> dict[str, Any]:
    import torch

    from graph_specialisation_metrics.mechanistic_operator_analysis import (
        OfficialGRITMechanisticCollector,
        view_head_grad,
    )

    gm.model.zero_grad(set_to_none=True)
    batch = make_grit_batch([data], gm.device)
    with OfficialGRITMechanisticCollector(gm.model.model) as collector:
        prediction, target = gm.model(batch)
        prediction.reshape(-1).sum().backward()
    records = sorted(collector.records, key=lambda item: item.layer)
    if len(records) != gm.L:
        raise RuntimeError(f"mechanism collector captured {len(records)} != {gm.L} layers")
    output = []
    for record in records:
        head_output = record.head_output.detach()
        if head_output.dim() == 2:
            head_output = head_output.reshape(head_output.shape[0], record.heads, -1)
        gradient = view_head_grad(record).detach()
        edge_count = int(record.local_src.numel())
        expected_head_prefix = (int(data.num_nodes), int(gm.H))
        contracts = {
            "attention": tuple(record.attention.shape) == (edge_count, int(gm.H)),
            "message": tuple(record.message.shape[:2]) == (edge_count, int(gm.H)),
            "head_output": tuple(head_output.shape[:2]) == expected_head_prefix,
            "gradient": tuple(gradient.shape) == tuple(head_output.shape),
            "pair_coordinates": tuple(record.local_dst.shape) == tuple(record.local_src.shape),
        }
        failed = [name for name, passed in contracts.items() if not passed]
        if failed:
            shapes = {
                "attention": tuple(record.attention.shape),
                "message": tuple(record.message.shape),
                "head_output": tuple(head_output.shape),
                "gradient": tuple(gradient.shape),
                "src": tuple(record.local_src.shape),
                "dst": tuple(record.local_dst.shape),
            }
            raise RuntimeError(
                f"mechanism capture contract failed at layer {record.layer}: "
                f"{failed}; shapes={shapes}"
            )
        finite = {
            "attention": bool(torch.isfinite(record.attention).all()),
            "message": bool(torch.isfinite(record.message).all()),
            "head_output": bool(torch.isfinite(head_output).all()),
            "gradient": bool(torch.isfinite(gradient).all()),
        }
        nonfinite = [name for name, passed in finite.items() if not passed]
        if nonfinite:
            raise RuntimeError(
                f"non-finite mechanism capture at layer {record.layer}: {nonfinite}"
            )
        output.append({
            "src": record.local_src.detach(),
            "dst": record.local_dst.detach(),
            "attention": record.attention.detach(),
            "message": record.message.detach(),
            "head_output": head_output,
            "gradient": gradient,
        })
    return {
        "prediction": prediction.detach().cpu().numpy().reshape(-1),
        "target": target.detach().cpu().numpy().reshape(-1),
        "layers": output,
    }


def aggregate_pairs_to_nodes(pair_value: Any, destination: Any, n: int) -> Any:
    import torch

    out = torch.zeros(
        n, pair_value.shape[1], pair_value.shape[2],
        dtype=pair_value.dtype, device=pair_value.device,
    )
    out.index_add_(0, destination.long(), pair_value)
    return out


def decompose_fixed_support_layer(clean: Mapping[str, Any], corrupt: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    if not torch.equal(clean["src"], corrupt["src"]) or not torch.equal(clean["dst"], corrupt["dst"]):
        raise RuntimeError("routing/message decomposition requires identical ordered support")
    n = int(clean["head_output"].shape[0])
    clean_attention = clean["attention"].unsqueeze(-1)
    corrupt_attention = corrupt["attention"].unsqueeze(-1)
    clean_message = clean["message"]
    corrupt_message = corrupt["message"]
    route_pairs = (clean_attention - corrupt_attention) * 0.5 * (
        clean_message + corrupt_message
    )
    message_pairs = 0.5 * (clean_attention + corrupt_attention) * (
        clean_message - corrupt_message
    )
    route_nodes = aggregate_pairs_to_nodes(route_pairs, clean["dst"], n)
    message_nodes = aggregate_pairs_to_nodes(message_pairs, clean["dst"], n)
    direct_nodes = clean["head_output"] - corrupt["head_output"]
    reconstruction = direct_nodes - route_nodes - message_nodes
    gradient = clean["gradient"]
    direct_q = torch.einsum("nhd,nhd->h", gradient, direct_nodes)
    route_q = torch.einsum("nhd,nhd->h", gradient, route_nodes)
    message_q = torch.einsum("nhd,nhd->h", gradient, message_nodes)

    routing_hybrid = aggregate_pairs_to_nodes(
        clean_attention * corrupt_message, clean["dst"], n
    )
    message_hybrid = aggregate_pairs_to_nodes(
        corrupt_attention * clean_message, clean["dst"], n
    )
    return {
        "direct_q": direct_q,
        "routing_q": route_q,
        "message_q": message_q,
        "reconstruction_max": float(reconstruction.abs().max()),
        "routing_hybrid": routing_hybrid,
        "message_hybrid": message_hybrid,
        "full_clean": clean["head_output"],
    }


def run_mechanism_task(
    loaded: Mapping[str, Any], cfg: BetaConfig, *, force: bool = False
) -> dict[str, Any]:
    import torch

    gm, checkpoint_sha = loaded["gm"], loaded["sha"]
    task_name = loaded["spec"].name
    path = cache_path(cfg, task_name, "mechanism", checkpoint_sha)
    cached = None if force else valid_cache(path, cfg, checkpoint_sha)
    if cached is not None:
        print(f"[mechanism:{task_name}] cache hit {path}", flush=True)
        return cached
    splits = deterministic_splits(len(gm.eval_ds), cfg)
    donor_rows, donor_graph_ids, _ = donor_row_pool(gm)
    descriptors, by_n = load_or_build_topology_index(
        gm, cfg, task_name, checkpoint_sha
    )
    catalog = causal_event_catalog(
        gm,
        cfg,
        splits["mechanism"],
        donor_rows=donor_rows,
        donor_graph_ids=donor_graph_ids,
        topology_descriptors=descriptors,
        topology_by_n=by_n,
    )
    events = [
        event for event in catalog
        if event["channel"] in PRIMARY_CHANNELS and int(event["event_index"]) == 0
    ]
    shape = (len(events), gm.L, gm.H)
    arrays = {
        key: np.full(shape, np.nan, dtype=np.float32)
        for key in (
            "direct_q", "routing_q", "message_q", "routing_rescue", "message_rescue",
            "full_rescue", "finite_interaction",
        )
    }
    reconstruction = np.zeros((len(events), gm.L), dtype=np.float32)
    progress_path = cache_path(cfg, task_name, "mechanism_progress", checkpoint_sha)
    progress = None if force else valid_cache(progress_path, cfg, checkpoint_sha)
    completed_events: set[int] = set()
    if progress is not None:
        if int(progress.get("event_count", -1)) != len(events):
            raise RuntimeError("mechanism progress event catalog size changed")
        arrays = {key: np.asarray(value) for key, value in progress["metrics"].items()}
        reconstruction = np.asarray(progress["reconstruction_max"])
        completed_events = {int(value) for value in progress.get("completed_events", [])}
    for event_index, event in enumerate(events):
        if event_index in completed_events:
            print(f"[mechanism:{task_name}] event {event_index + 1} cache hit", flush=True)
            continue
        clean = capture_mechanism(gm, event["clean"])
        corrupt = capture_mechanism(gm, event["corrupt"])
        delta = clean["prediction"] - corrupt["prediction"]
        direction = np.sign(delta)
        for layer in range(gm.L):
            decomposition = decompose_fixed_support_layer(
                clean["layers"][layer], corrupt["layers"][layer]
            )
            reconstruction[event_index, layer] = decomposition["reconstruction_max"]
            if decomposition["reconstruction_max"] > MECHANISM_ATOL:
                raise RuntimeError(
                    f"routing/message reconstruction failed: {decomposition['reconstruction_max']:.3e}"
                )
            for key in ("direct_q", "routing_q", "message_q"):
                arrays[key][event_index, layer] = decomposition[key].detach().cpu().numpy()
            for head in range(gm.H):
                names = ("routing_rescue", "message_rescue", "full_rescue")
                sources = (
                    decomposition["routing_hybrid"],
                    decomposition["message_hybrid"],
                    decomposition["full_clean"],
                )
                patched = forward_data_list(
                    gm,
                    [event["corrupt"], event["corrupt"], event["corrupt"]],
                    patch=(layer, head, torch.cat([source.to(gm.device) for source in sources], dim=0)),
                )
                predictions = {}
                for name, prediction in zip(names, patched):
                    desired = float(np.mean((prediction - corrupt["prediction"]) * direction))
                    arrays[name][event_index, layer, head] = desired
                    predictions[name] = desired
                arrays["finite_interaction"][event_index, layer, head] = (
                    predictions["full_rescue"]
                    - predictions["routing_rescue"]
                    - predictions["message_rescue"]
                )
        print(
            f"[mechanism:{task_name}] event {event_index + 1}/{len(events)} "
            f"channel={event['channel']}", flush=True,
        )
        completed_events.add(event_index)
        atomic_torch_save({
            "version": BETA_VERSION,
            "schema": BETA_SCHEMA,
            "fingerprint": cfg.fingerprint,
            "checkpoint_sha256": checkpoint_sha,
            "task": task_name,
            "event_count": len(events),
            "metrics": arrays,
            "reconstruction_max": reconstruction,
            "completed_events": sorted(completed_events),
        }, progress_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "version": BETA_VERSION,
        "schema": BETA_SCHEMA,
        "fingerprint": cfg.fingerprint,
        "checkpoint_sha256": checkpoint_sha,
        "task": task_name,
        "events": [
            {key: value for key, value in event.items() if key not in {"clean", "corrupt", "descriptor"}}
            for event in events
        ],
        "metrics": arrays,
        "reconstruction_max": reconstruction,
    }
    atomic_torch_save(payload, path)
    return payload


# =====================================================================================
# General, sample-split conditional-specialisation screen
# =====================================================================================


def _condition_observations(
    score_payload: Mapping[str, Any], aggregation: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = score_payload["records"]
    graph_features = []
    discovery_graphs = {
        int(record["graph_id"]) for index, record in enumerate(records) if index % 2 == 0
    }
    atom_counts: dict[tuple[int, ...], int] = {}
    for record in records:
        descriptor = record["plan"]["descriptor"]
        labels = [tuple(row) for row in np.asarray(descriptor["labels"], dtype=int)]
        if int(record["graph_id"]) in discovery_graphs:
            for label in set(labels):
                atom_counts[label] = atom_counts.get(label, 0) + 1
        graph_features.append({
            "graph_id": int(record["graph_id"]),
            "n": int(record["n"]),
            "cycle_rank": int(descriptor["cycle_rank"]),
            "atom_diversity": len(set(labels)),
            "labels": labels,
        })
    discovery_features = [row for row in graph_features if row["graph_id"] in discovery_graphs]
    thresholds = {
        name: float(np.median([row[name] for row in discovery_features]))
        for name in ("n", "cycle_rank", "atom_diversity")
    }
    common_atoms = [
        label for label, _count in sorted(atom_counts.items(), key=lambda item: (-item[1], item[0]))[:4]
    ]
    by_id = {row["graph_id"]: row for row in graph_features}
    observations = []

    def concentration(q: Any) -> np.ndarray:
        event_magnitude = np.linalg.norm(q.numpy(), axis=-1).mean(axis=0)  # [L,H,N]
        probability = event_magnitude / np.maximum(event_magnitude.sum(axis=-1, keepdims=True), EPS)
        entropy = -(probability * np.log(np.maximum(probability, EPS))).sum(axis=-1)
        return 1.0 - entropy / max(math.log(event_magnitude.shape[-1]), EPS)

    for record in records:
        semantic = np.asarray(record["channels"]["semantic"]["per_source"][aggregation], float)
        structural = np.asarray(record["channels"]["pe"]["per_source"][aggregation], float)
        total = semantic + structural
        d_value = (semantic - structural) / (total + EPS)
        j_value = 0.5 * total
        sources = np.asarray(record["plan"]["sources"], dtype=np.int64)
        feature = by_id[int(record["graph_id"])]
        graph_rules = {
            "graph_size_high": feature["n"] > thresholds["n"],
            "cycle_rank_high": feature["cycle_rank"] > thresholds["cycle_rank"],
            "atom_diversity_high": feature["atom_diversity"] > thresholds["atom_diversity"],
        }
        for atom in common_atoms:
            graph_rules[f"graph_contains_atom_{'_'.join(map(str, atom))}"] = atom in feature["labels"]
        for source_index, source in enumerate(sources):
            source_label = tuple(feature["labels"][int(source)])
            rules = dict(graph_rules)
            for atom in common_atoms:
                rules[f"source_atom_{'_'.join(map(str, atom))}"] = source_label == atom
            observations.append({
                "graph_id": int(record["graph_id"]),
                "split": "discovery" if int(record["graph_id"]) in discovery_graphs else "confirmation",
                "source": int(source),
                "D": d_value[source_index],
                "J": j_value[source_index],
                "semantic_dose": float(np.mean(record["plan"]["semantic"][source_index]["dose"])),
                "pe_dose": float(np.mean(record["plan"]["pe"][source_index]["dose"])),
                "semantic_concentration": concentration(
                    record["channels"]["semantic"]["q_groups"][source_index]
                ),
                "pe_concentration": concentration(
                    record["channels"]["pe"]["q_groups"][source_index]
                ),
                "rules": rules,
            })
    return observations, {
        "thresholds_from_discovery": thresholds,
        "common_atom_codes": common_atoms,
        "discovery_graph_ids": sorted(discovery_graphs),
    }


def _conditional_effect(
    observations: Sequence[Mapping[str, Any]], rule: str, head: tuple[int, int]
) -> tuple[float, int, int]:
    return _conditional_feature_effect(observations, rule, head, "D")


def _conditional_feature_effect(
    observations: Sequence[Mapping[str, Any]],
    rule: str,
    head: tuple[int, int],
    feature: str,
) -> tuple[float, int, int]:
    by_state: dict[bool, dict[int, list[float]]] = {False: {}, True: {}}
    for row in observations:
        state = bool(row["rules"][rule])
        by_state[state].setdefault(int(row["graph_id"]), []).append(float(row[feature][head]))
    means = {
        state: np.asarray([np.mean(values) for values in groups.values()], dtype=float)
        for state, groups in by_state.items()
    }
    if not len(means[True]) or not len(means[False]):
        return float("nan"), len(means[True]), len(means[False])
    return float(means[True].mean() - means[False].mean()), len(means[True]), len(means[False])


def _conditional_p_value(
    observations: Sequence[Mapping[str, Any]], rule: str, head: tuple[int, int]
) -> float:
    from scipy.stats import ttest_1samp, ttest_ind

    by_state: dict[bool, dict[int, list[float]]] = {False: {}, True: {}}
    for row in observations:
        state = bool(row["rules"][rule])
        by_state[state].setdefault(int(row["graph_id"]), []).append(float(row["D"][head]))
    means = {
        state: {graph_id: float(np.mean(values)) for graph_id, values in groups.items()}
        for state, groups in by_state.items()
    }
    paired = sorted(set(means[True]) & set(means[False]))
    if len(paired) >= 3:
        differences = np.asarray([means[True][graph] - means[False][graph] for graph in paired])
        return float(ttest_1samp(differences, 0.0).pvalue)
    left = np.asarray(list(means[True].values()))
    right = np.asarray(list(means[False].values()))
    if min(len(left), len(right)) < 2:
        return float("nan")
    return float(ttest_ind(left, right, equal_var=False).pvalue)


def bh_q_values(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    output = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return output
    order = finite[np.argsort(values[finite])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output[order] = np.minimum(ranked, 1.0)
    return output


def conditional_dose_diagnostics(
    observations: Sequence[Mapping[str, Any]], rule: str
) -> dict[str, float]:
    output = {}
    for key in ("semantic_dose", "pe_dose"):
        values = {
            state: np.asarray([float(row[key]) for row in observations if bool(row["rules"][rule]) == state])
            for state in (False, True)
        }
        left, right = values[False], values[True]
        if not len(left) or not len(right):
            output[f"{key}_standardized_difference"] = float("nan")
            output[f"{key}_common_support_fraction"] = 0.0
            continue
        pooled = math.sqrt(0.5 * (float(np.var(left)) + float(np.var(right)))) if len(left) and len(right) else 0.0
        common_low = max(float(np.quantile(left, 0.05)), float(np.quantile(right, 0.05)))
        common_high = min(float(np.quantile(left, 0.95)), float(np.quantile(right, 0.95)))
        in_common = np.concatenate([
            (left >= common_low) & (left <= common_high),
            (right >= common_low) & (right <= common_high),
        ])
        output[f"{key}_standardized_difference"] = (
            float((right.mean() - left.mean()) / pooled) if pooled > EPS else 0.0
        )
        output[f"{key}_common_support_fraction"] = (
            float(in_common.mean()) if common_high >= common_low else 0.0
        )
    return output


def _bootstrap_conditional_effect(
    observations: Sequence[Mapping[str, Any]],
    rule: str,
    head: tuple[int, int],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    graph_ids = sorted({int(row["graph_id"]) for row in observations})
    by_graph = {graph_id: [row for row in observations if int(row["graph_id"]) == graph_id] for graph_id in graph_ids}
    values = []
    for _ in range(samples):
        selected = rng.choice(graph_ids, size=len(graph_ids), replace=True)
        boot = []
        for new_id, graph_id in enumerate(selected):
            for row in by_graph[int(graph_id)]:
                copied = dict(row)
                copied["graph_id"] = new_id
                boot.append(copied)
        effect, _, _ = _conditional_effect(boot, rule, head)
        if np.isfinite(effect):
            values.append(effect)
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def conditional_analysis(
    score_payload: Mapping[str, Any], aggregation: str, cfg: BetaConfig
) -> dict[str, Any]:
    observations, metadata = _condition_observations(score_payload, aggregation)
    discovery = [row for row in observations if row["split"] == "discovery"]
    confirmation = [row for row in observations if row["split"] == "confirmation"]
    rules = sorted(observations[0]["rules"]) if observations else []
    L, H = int(score_payload["L"]), int(score_payload["H"])
    candidates = []
    for rule in rules:
        true_fraction = np.mean([bool(row["rules"][rule]) for row in discovery])
        if not cfg.min_condition_fraction <= true_fraction <= 1.0 - cfg.min_condition_fraction:
            continue
        for layer in range(L):
            for head in range(H):
                effect, true_graphs, false_graphs = _conditional_effect(discovery, rule, (layer, head))
                if min(true_graphs, false_graphs) < cfg.min_condition_graphs:
                    continue
                activity = np.mean([float(row["J"][layer, head]) for row in discovery])
                candidates.append({
                    "rule": rule,
                    "layer": layer,
                    "head": head,
                    "discovery_effect": effect,
                    "discovery_activity": activity,
                    "true_graphs": true_graphs,
                    "false_graphs": false_graphs,
                })
    if not candidates:
        return {"metadata": metadata, "candidates": [], "confirmed": []}
    activity_floor = cfg.activity_floor_relative * max(row["discovery_activity"] for row in candidates)
    eligible = [row for row in candidates if row["discovery_activity"] >= activity_floor]
    eligible.sort(key=lambda row: (-abs(row["discovery_effect"]), row["rule"], row["layer"], row["head"]))
    tested = eligible[: min(12, len(eligible))]
    confirmed = []
    for index, row in enumerate(tested):
        head = (int(row["layer"]), int(row["head"]))
        effect, true_graphs, false_graphs = _conditional_effect(confirmation, row["rule"], head)
        lower, upper = _bootstrap_conditional_effect(
            confirmation,
            row["rule"],
            head,
            samples=cfg.conditional_bootstrap_samples,
            seed=cfg.analysis_seed + 70_001 + index,
        )
        confirmed.append({
            **row,
            "confirmation_effect": effect,
            "confirmation_ci_low": lower,
            "confirmation_ci_high": upper,
            "confirmation_true_graphs": true_graphs,
            "confirmation_false_graphs": false_graphs,
            "same_sign": bool(np.sign(effect) == np.sign(row["discovery_effect"])),
            "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
            "confirmation_p": _conditional_p_value(confirmation, row["rule"], head),
            **conditional_dose_diagnostics(confirmation, row["rule"]),
            "semantic_concentration_interaction": _conditional_feature_effect(
                confirmation, row["rule"], head, "semantic_concentration"
            )[0],
            "pe_concentration_interaction": _conditional_feature_effect(
                confirmation, row["rule"], head, "pe_concentration"
            )[0],
        })
    q_values = bh_q_values([row["confirmation_p"] for row in confirmed])
    for row, q_value in zip(confirmed, q_values):
        row["confirmation_q_global"] = float(q_value)
        row["confirmed_at_fdr_0.05"] = bool(
            row["same_sign"] and row["ci_excludes_zero"] and np.isfinite(q_value) and q_value <= 0.05
        )
    return {
        "metadata": metadata,
        "activity_floor": activity_floor,
        "candidates": tested,
        "confirmed": confirmed,
        "interpretation": (
            "Finite predeclared condition basis with sample-split rule/head selection and "
            "graph-bootstrap confirmation; a screen for conditional specialists, not a proof "
            "that no untested condition exists."
        ),
    }


# =====================================================================================
# Tables, paper figures, and predeclared beta decisions
# =====================================================================================


def configure_matplotlib() -> Any:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


def save_figure(fig: Any, path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".png", ".pdf"):
        target = path.with_suffix(suffix)
        fig.savefig(target, bbox_inches="tight")
        outputs.append(str(target))
    return outputs


def layer_colors(L: int) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=0, vmax=max(L - 1, 1))
    return plt.get_cmap("viridis"), norm


def _head_layers(L: int, H: int) -> np.ndarray:
    return np.repeat(np.arange(L), H)


def _identity_limits(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(left).reshape(-1), np.asarray(right).reshape(-1)])
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 1.0
    low, high = float(values.min()), float(values.max())
    pad = 0.05 * max(high - low, EPS)
    return low - pad, high + pad


def figure_aggregation_planes(runs: Mapping[str, Mapping[str, Any]], path: Path) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.0), constrained_layout=True)
    for row, task in enumerate(("zinc", "zinc_1hop")):
        score = runs[task]["scores"]
        cmap, norm = layer_colors(int(score["L"]))
        layers = _head_layers(int(score["L"]), int(score["H"]))
        for column, aggregation in enumerate(AGGREGATIONS):
            semantic = mean_score(score, "semantic", aggregation)
            structural = mean_score(score, "pe", aggregation)
            axis = axes[row, column]
            axis.scatter(
                structural.reshape(-1), semantic.reshape(-1), c=layers, cmap=cmap, norm=norm,
                s=24, edgecolor="black", linewidth=0.25, alpha=0.9,
            )
            low, high = _identity_limits(structural, semantic)
            axis.plot([low, high], [low, high], ":", color="black", linewidth=1)
            axis.set_xlim(low, high); axis.set_ylim(low, high)
            axis.set_title(aggregation)
            if column == 0:
                axis.set_ylabel(f"{DISPLAY[task]}\nsemantic score")
            if row == 1:
                axis.set_xlabel("PE-transposition score")
    scalar = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis"), norm=norm)
    colorbar = fig.colorbar(scalar, ax=axes, fraction=0.018, pad=0.01)
    colorbar.set_label("layer")
    fig.suptitle("ZINC beta: all four graph-balanced score aggregations", fontsize=12)
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def figure_method_validation(
    runs: Mapping[str, Mapping[str, Any]], path: Path
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), constrained_layout=True)
    x = np.arange(len(AGGREGATIONS))
    width = 0.36
    for offset, task in zip((-width / 2, width / 2), ("zinc", "zinc_1hop")):
        rows = [
            row for row in runs[task]["ablations"]["method_rows"]
            if row.get("scope") == "decision"
        ]
        by_aggregation = {row["aggregation"]: row for row in rows}
        axes[0].bar(
            x + offset,
            [by_aggregation[name]["validity_mean"] for name in AGGREGATIONS],
            width,
            label=DISPLAY[task],
        )
        axes[1].bar(
            x + offset,
            [by_aggregation[name]["prefix_reliability"] for name in AGGREGATIONS],
            width,
            label=DISPLAY[task],
        )
    axes[0].set_title("Independent causal validity")
    axes[0].set_ylabel("mean Spearman ρ")
    axes[1].set_title("Donor-prefix rank reliability")
    axes[1].set_ylabel("mean Spearman ρ")
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xticks(x, AGGREGATIONS)
    axes[0].legend(frameon=False)
    fig.suptitle("Aggregation choice uses causal validity and donor convergence", fontsize=11)
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def figure_selectivity_strength(
    runs: Mapping[str, Mapping[str, Any]], aggregation: str, path: Path
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), constrained_layout=True, sharey=True)
    for axis, task in zip(axes, ("zinc", "zinc_1hop")):
        score = runs[task]["scores"]
        selection = runs[task]["ablations"]["selections"][aggregation]
        coordinates = selection["coordinates"]
        cmap, norm = layer_colors(int(score["L"]))
        layers = _head_layers(int(score["L"]), int(score["H"]))
        active = np.asarray(selection["active"], bool).reshape(-1)
        axis.scatter(
            coordinates["D"].reshape(-1)[~active], coordinates["J"].reshape(-1)[~active],
            color="0.78", s=20, label="below activity gate",
        )
        scatter = axis.scatter(
            coordinates["D"].reshape(-1)[active], coordinates["J"].reshape(-1)[active],
            c=layers[active], cmap=cmap, norm=norm, s=30, edgecolor="black", linewidth=0.3,
            label="eligible",
        )
        axis.axvline(0, color="black", linestyle=":", linewidth=0.8)
        axis.axvline(-0.2, color="0.5", linestyle="--", linewidth=0.7)
        axis.axvline(0.2, color="0.5", linestyle="--", linewidth=0.7)
        axis.axhline(selection["activity_floor"], color="0.5", linestyle="--", linewidth=0.7)
        axis.set_title(DISPLAY[task])
        axis.set_xlabel("relative selectivity D  (− PE, + semantic)")
    axes[0].set_ylabel("evoked strength J")
    axes[0].legend(frameon=False, loc="upper left")
    fig.colorbar(scatter, ax=axes, fraction=0.025, pad=0.02, label="layer")
    fig.suptitle(f"Activity-gated specialisation ({aggregation})", fontsize=11)
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def figure_causal_coordinates(
    runs: Mapping[str, Mapping[str, Any]], aggregation: str, path: Path
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), constrained_layout=True)
    for row, task in enumerate(("zinc", "zinc_1hop")):
        selection = runs[task]["ablations"]["selections"][aggregation]
        score = selection["coordinates"]
        causal = causal_coordinates(runs[task]["causal"], "restore")
        active = np.asarray(selection["active"], bool)
        for column, key in enumerate(("D", "J")):
            axis = axes[row, column]
            axis.scatter(
                score[key][~active], causal[key][~active], color="0.78", s=18,
            )
            axis.scatter(
                score[key][active], causal[key][active], color="#3b7ddd", s=27,
                edgecolor="black", linewidth=0.25,
            )
            rho = spearman(score[key][active].reshape(-1), causal[key][active].reshape(-1))
            axis.set_title(f"{DISPLAY[task]} · {key} · gated ρ={rho:.2f}")
            axis.set_xlabel(f"score {key}")
            axis.set_ylabel(f"restore causal {key}")
            axis.axhline(0, color="0.6", linewidth=0.6)
            axis.axvline(0, color="0.6", linewidth=0.6)
    fig.suptitle("Score coordinates predict independent whole-transport mediation", fontsize=11)
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def family_channel_matrix(
    causal: Mapping[str, Any],
    families: Mapping[str, Sequence[tuple[int, int]]],
    family_names: Sequence[str],
    metric: str,
) -> np.ndarray:
    matrix = np.full((len(family_names), len(CHANNELS)), np.nan)
    for row, family in enumerate(family_names):
        heads = families.get(family, [])
        for column, channel in enumerate(CHANNELS):
            value = channel_mean_metric(causal, channel, metric)
            if heads and np.isfinite(value).any():
                matrix[row, column] = np.nanmean([value[head] for head in heads])
    return matrix


def figure_family_patching(
    runs: Mapping[str, Mapping[str, Any]], aggregation: str, path: Path
) -> list[str]:
    plt = configure_matplotlib()
    family_names = (
        "semantic_specialist", "semantic_same_layer_J_matched", "semantic_same_layer_inert",
        "pe_specialist", "pe_same_layer_J_matched", "pe_same_layer_inert",
        "high_J", "high_G_balanced", "topology_responsive", "low_J_inert",
    )
    metrics = ("restore", "inject", "necessity")
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 7.0), constrained_layout=True)
    images = []
    matrices = []
    for task in ("zinc", "zinc_1hop"):
        families = runs[task]["ablations"]["selections"][aggregation]["families"]
        for metric in metrics:
            matrices.append(family_channel_matrix(runs[task]["causal"], families, family_names, metric))
    bound = max(float(np.nanmax(np.abs(matrix))) for matrix in matrices if np.isfinite(matrix).any())
    bound = max(bound, EPS)
    cursor = 0
    for row, task in enumerate(("zinc", "zinc_1hop")):
        for column, metric in enumerate(metrics):
            matrix = matrices[cursor]; cursor += 1
            axis = axes[row, column]
            image = axis.imshow(matrix, cmap="coolwarm", vmin=-bound, vmax=bound, aspect="auto")
            images.append(image)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if np.isfinite(matrix[i, j]):
                        axis.text(j, i, f"{matrix[i,j]:.2g}", ha="center", va="center", fontsize=7)
            axis.set_xticks(range(len(CHANNELS)), ("semantic", "PE", "topology"))
            axis.set_title(f"{DISPLAY[task]} · {metric}")
            if column == 0:
                axis.set_yticks(range(len(family_names)), [name.replace("_", " ") for name in family_names])
            else:
                axis.set_yticks([])
    fig.colorbar(images[-1], ax=axes, fraction=0.018, pad=0.01, label="desired output movement")
    fig.suptitle("Independent family × intervention causal confirmation", fontsize=11)
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def topology_summary(
    score: Mapping[str, Any], causal: Mapping[str, Any], aggregation: str
) -> dict[str, Any]:
    total = len(score["records"])
    tiers = [
        int(item["tier"])
        for record in score["records"] for item in record["plan"]["topology"]
    ]
    graphs_near = sum(
        any(int(item["tier"]) <= 1 for item in record["plan"]["topology"])
        for record in score["records"]
    )
    pe = mean_score(score, "pe", aggregation)
    topology = mean_score(score, "topology", aggregation)
    topology_restore = channel_mean_metric(causal, "topology", "restore")
    return {
        "graphs": total,
        "graphs_with_tier_le_1": graphs_near,
        "coverage_tier_le_1": graphs_near / max(total, 1),
        "tier_counts": {str(tier): tiers.count(tier) for tier in sorted(set(tiers))},
        "rho_pe_topology_pooled": spearman(pe.reshape(-1), topology.reshape(-1)),
        "rho_pe_topology_within_layer": spearman(
            layer_centered(pe).reshape(-1), layer_centered(topology).reshape(-1)
        ),
        "rho_topology_score_restore": spearman(topology.reshape(-1), topology_restore.reshape(-1)),
        "pe": pe,
        "topology": topology,
        "topology_restore": topology_restore,
    }


def figure_topology_validation(
    runs: Mapping[str, Mapping[str, Any]], aggregation: str, path: Path
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.5), constrained_layout=True)
    for row, task in enumerate(("zinc", "zinc_1hop")):
        summary = topology_summary(runs[task]["scores"], runs[task]["causal"], aggregation)
        counts = summary["tier_counts"]
        axes[row, 0].bar(list(counts), list(counts.values()), color="#5a8f6b")
        axes[row, 0].set_title(f"{DISPLAY[task]} · donor tiers")
        axes[row, 0].set_xlabel("tier (0 strict, 1 near, 2 relaxed)")
        axes[row, 0].set_ylabel("matched donor events")
        axes[row, 1].scatter(
            summary["pe"].reshape(-1), summary["topology"].reshape(-1),
            s=24, color="#7765b5", edgecolor="black", linewidth=0.25,
        )
        axes[row, 1].set_title(f"PE vs topology · ρ={summary['rho_pe_topology_pooled']:.2f}")
        axes[row, 1].set_xlabel("PE-transposition score")
        axes[row, 1].set_ylabel("matched-topology score")
        axes[row, 2].scatter(
            summary["topology"].reshape(-1), summary["topology_restore"].reshape(-1),
            s=24, color="#c06b53", edgecolor="black", linewidth=0.25,
        )
        axes[row, 2].set_title(f"topology causal validity · ρ={summary['rho_topology_score_restore']:.2f}")
        axes[row, 2].set_xlabel("matched-topology score")
        axes[row, 2].set_ylabel("topology restore")
    fig.suptitle("Matched-real non-isomorphic topology probe: feasibility and validity", fontsize=11)
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def figure_ablation(
    runs: Mapping[str, Mapping[str, Any]], aggregation: str, path: Path
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), constrained_layout=True)
    colors = {"semantic_specialist": "#c75b5b", "pe_specialist": "#4c78a8", "high_J": "#55a868", "low_J_inert": "0.5"}
    for row, task in enumerate(("zinc", "zinc_1hop")):
        score = runs[task]["scores"]
        ablation = runs[task]["ablations"]
        coordinates = ablation["selections"][aggregation]["coordinates"]
        impact = np.asarray(ablation["per_head"]["functional"])
        axes[row, 0].scatter(coordinates["J"].reshape(-1), impact.reshape(-1), s=24, color="#5a8f6b")
        axes[row, 0].set_title(f"{DISPLAY[task]} · ρ={spearman(coordinates['J'].reshape(-1), impact.reshape(-1)):.2f}")
        axes[row, 0].set_xlabel("evoked strength J")
        axes[row, 0].set_ylabel("single-head functional ablation")
        curves = ablation["family_curves"][aggregation]
        for family, color in colors.items():
            rows = curves.get(family, [])
            if rows:
                axes[row, 1].plot(
                    [item["count"] for item in rows], [item["functional"] for item in rows],
                    marker="o", color=color, label=family.replace("_", " "),
                )
        axes[row, 1].set_title(f"{DISPLAY[task]} · cumulative family ablation")
        axes[row, 1].set_xlabel("heads ablated")
        axes[row, 1].set_ylabel("mean |Δ prediction|")
    axes[0, 1].legend(frameon=False, fontsize=7)
    fig.suptitle("Independent ordinary and score-ranked family ablations", fontsize=11)
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def figure_patch_controls(
    runs: Mapping[str, Mapping[str, Any]], path: Path
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), constrained_layout=True)
    metrics = ("restore", "mismatch", "sham")
    width = 0.22
    x = np.arange(len(CHANNELS))
    for axis, task in zip(axes, ("zinc", "zinc_1hop")):
        causal = runs[task]["causal"]
        for offset_index, metric in enumerate(metrics):
            values = [np.nanmean(channel_mean_metric(causal, channel, metric)) for channel in CHANNELS]
            axis.bar(x + (offset_index - 1) * width, values, width, label=metric)
        axis.set_xticks(x, ("semantic", "PE", "topology"))
        axis.set_title(DISPLAY[task])
        axis.set_ylabel("mean desired output movement")
        axis.axhline(0, color="black", linewidth=0.7)
    axes[0].legend(frameon=False)
    fig.suptitle("Patching calibration: matched restore vs cross-graph and sham", fontsize=11)
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def figure_routing_message(
    runs: Mapping[str, Mapping[str, Any]], aggregation: str, path: Path
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.8), constrained_layout=True)
    for row, task in enumerate(("zinc", "zinc_1hop")):
        mechanism = runs[task]["mechanism"]
        events = mechanism["events"]
        families = runs[task]["ablations"]["selections"][aggregation]["families"]
        for column, channel in enumerate(PRIMARY_CHANNELS):
            axis = axes[row, column]
            mask = np.asarray([event["channel"] == channel for event in events])
            routing = np.nanmean(np.abs(mechanism["metrics"]["routing_q"][mask]), axis=0)
            message = np.nanmean(np.abs(mechanism["metrics"]["message_q"][mask]), axis=0)
            for family, color in (("semantic_specialist", "#c75b5b"), ("pe_specialist", "#4c78a8"), ("high_J", "#55a868")):
                heads = families.get(family, [])
                if heads:
                    axis.scatter(
                        [routing[head] for head in heads], [message[head] for head in heads],
                        s=45, color=color, label=family.replace("_", " "), edgecolor="black", linewidth=0.3,
                    )
            low, high = _identity_limits(routing, message)
            axis.plot([low, high], [low, high], ":", color="black", linewidth=0.8)
            axis.set_xlabel("routing contribution |q|")
            axis.set_ylabel("message contribution |q|")
            axis.set_title(f"{DISPLAY[task]} · {channel}")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Why specialisation appears: fixed-support routing vs message", fontsize=11)
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def carriage_profile(
    score: Mapping[str, Any],
    channel: str,
    heads: Sequence[tuple[int, int]],
    *,
    bootstrap_samples: int = 500,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rows: dict[int, dict[int, dict[str, list[float]]]] = {}
    for record in score["records"]:
        result = record["channels"][channel]
        if not result["available"]:
            continue
        for item in result["carriage"]:
            distance = int(item["distance"])
            rows.setdefault(distance, {}).setdefault(
                int(record["graph_id"]), {"F_sens": [], "F_coh": []}
            )
            for key in ("F_sens", "F_coh"):
                matrix = np.asarray(item[key], dtype=float)
                value = np.nanmean([matrix[head] for head in heads]) if heads else np.nanmean(matrix)
                rows[distance][int(record["graph_id"])][key].append(float(value))
    distances = sorted(rows)
    graph_ids = sorted({graph for distance in distances for graph in rows[distance]})
    matrices = {}
    for key in ("F_sens", "F_coh"):
        matrix = np.full((len(graph_ids), len(distances)), np.nan)
        for i, graph_id in enumerate(graph_ids):
            for j, distance in enumerate(distances):
                if graph_id in rows[distance]:
                    matrix[i, j] = np.nanmean(rows[distance][graph_id][key])
        matrices[key] = matrix
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {"distance": np.asarray(distances, dtype=int)}
    for key, matrix in matrices.items():
        result[key] = np.nanmean(matrix, axis=0)
        bootstrap = []
        for _ in range(int(bootstrap_samples)):
            index = rng.integers(0, len(graph_ids), len(graph_ids))
            bootstrap.append(np.nanmean(matrix[index], axis=0))
        result[f"{key}_ci_low"] = np.nanquantile(bootstrap, 0.025, axis=0)
        result[f"{key}_ci_high"] = np.nanquantile(bootstrap, 0.975, axis=0)
    result["graph_support"] = np.sum(np.isfinite(matrices["F_sens"]), axis=0)
    return result


def figure_carriage(
    runs: Mapping[str, Mapping[str, Any]], aggregation: str, path: Path
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.8), constrained_layout=True, sharey=False)
    family_for = {"semantic": "semantic_specialist", "pe": "pe_specialist"}
    for row, task in enumerate(("zinc", "zinc_1hop")):
        families = runs[task]["ablations"]["selections"][aggregation]["families"]
        for column, channel in enumerate(PRIMARY_CHANNELS):
            axis = axes[row, column]
            profile = carriage_profile(
                runs[task]["scores"], channel, families.get(family_for[channel], []),
                seed=31_001 + row * 101 + column,
            )
            for key, style in (("F_sens", "-"), ("F_coh", "--")):
                value = profile[key]
                denominator = max(float(np.nansum(value)), EPS)
                normalised = value / denominator
                axis.plot(profile["distance"], normalised, style, marker="o", label=key)
                axis.fill_between(
                    profile["distance"],
                    profile[f"{key}_ci_low"] / denominator,
                    profile[f"{key}_ci_high"] / denominator,
                    alpha=0.12,
                )
            axis.set_xlabel("molecular hop distance from changed set")
            axis.set_ylabel("normalised functional carriage")
            axis.set_title(f"{DISPLAY[task]} · {channel}")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Sensitivity versus coherent functional carriage", fontsize=11)
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def figure_conditional(
    conditional: Mapping[str, Mapping[str, Any]], path: Path
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    for axis, task in zip(axes, ("zinc", "zinc_1hop")):
        rows = conditional[task].get("confirmed", [])[:8]
        if not rows:
            axis.text(0.5, 0.5, "No eligible condition", ha="center", va="center")
            axis.set_axis_off()
            continue
        labels = [f"{row['rule']}\nL{row['layer']}H{row['head']}" for row in rows]
        y = np.arange(len(rows))
        discovery = np.asarray([row["discovery_effect"] for row in rows])
        confirmation = np.asarray([row["confirmation_effect"] for row in rows])
        low = np.maximum(
            confirmation - np.asarray([row["confirmation_ci_low"] for row in rows]), 0.0
        )
        high = np.maximum(
            np.asarray([row["confirmation_ci_high"] for row in rows]) - confirmation, 0.0
        )
        axis.scatter(discovery, y - 0.12, marker="x", color="0.3", label="discovery")
        axis.errorbar(confirmation, y + 0.12, xerr=[low, high], fmt="o", color="#3b7ddd", label="confirmation")
        axis.axvline(0, color="black", linewidth=0.7)
        axis.set_yticks(y, labels)
        axis.set_xlabel("conditional ΔD")
        axis.set_title(DISPLAY[task])
        axis.invert_yaxis()
    axes[0].legend(frameon=False)
    fig.suptitle("Sample-split conditional-specialisation screen", fontsize=11)
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def jaccard(left: Sequence[Any], right: Sequence[Any]) -> float:
    a, b = set(map(tuple, left)), set(map(tuple, right))
    return len(a & b) / max(len(a | b), 1)


def finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def intervention_event_rows(
    task: str, score: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for record in score["records"]:
        for channel in CHANNELS:
            result = record["channels"][channel]
            if not result["available"]:
                continue
            outcome_cursor = 0
            for source_index, q in enumerate(result["q_groups"]):
                q_numpy = q.numpy()
                event_gross = np.linalg.norm(q_numpy, axis=-1).sum(axis=-1).mean(axis=(1, 2))
                for event_index in range(len(q_numpy)):
                    if channel == "semantic":
                        item = record["plan"]["semantic"][source_index]
                        dose = float(item["dose"][event_index])
                        metadata = {
                            "source": int(item["source"]),
                            "donor_graph_id": int(item["donor_graph_ids"][event_index]),
                        }
                    elif channel == "pe":
                        item = record["plan"]["pe"][source_index]
                        dose = float(item["dose"][event_index])
                        metadata = {
                            "source": int(item["source"]),
                            "partner": int(item["partners"][event_index]),
                            "degree_gap": int(item["degree_gap"][event_index]),
                        }
                    else:
                        item = record["plan"]["topology"][event_index]
                        dose = float(item["dose"]["edge_jaccard_distance"])
                        metadata = {
                            "source": -1,
                            "donor_graph_id": int(item["donor_graph_id"]),
                            "match_tier": int(item["tier"]),
                            "match_cost": float(item["cost"]),
                            "target_gap": float(item["target_gap"]),
                            "content_alignment_exact": bool(item["content_alignment_exact"]),
                            "changed_node_fraction": len(item["dose"]["changed_nodes"])
                            / max(int(record["n"]), 1),
                        }
                    loss_delta = float(result["event_outcomes"][outcome_cursor])
                    outcome_cursor += 1
                    rows.append({
                        "architecture": task,
                        "graph_id": int(record["graph_id"]),
                        "channel": channel,
                        "source_index": int(source_index),
                        "event_index": int(event_index),
                        "input_dose": dose,
                        "mean_head_event_gross": float(event_gross[event_index]),
                        "loss_clean_minus_intervention": loss_delta,
                        "intervention_improves_loss": loss_delta > 0.0,
                        **metadata,
                    })
            if outcome_cursor != len(result["event_outcomes"]):
                raise RuntimeError("event outcome cursor does not align with score events")
    return rows


def donor_outcome_summary_rows(event_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    keys = sorted({(row["architecture"], row["channel"]) for row in event_rows})
    for architecture, channel in keys:
        values = np.asarray([
            row["loss_clean_minus_intervention"] for row in event_rows
            if row["architecture"] == architecture and row["channel"] == channel
        ], dtype=float)
        helpful = values > 0.0
        harmful = values < 0.0
        rows.append({
            "architecture": architecture,
            "channel": channel,
            "events": len(values),
            "probability_intervention_improves_loss": float(helpful.mean()),
            "probability_intervention_harms_loss": float(harmful.mean()),
            "conditional_improvement_magnitude": float(values[helpful].mean()) if helpful.any() else 0.0,
            "conditional_harm_magnitude": float((-values[harmful]).mean()) if harmful.any() else 0.0,
            "mean_clean_minus_intervention_loss": float(values.mean()),
        })
    return rows


def create_decisions(
    runs: Mapping[str, Mapping[str, Any]], aggregation: str
) -> dict[str, Any]:
    topology = {
        task: topology_summary(runs[task]["scores"], runs[task]["causal"], aggregation)
        for task in ARCHITECTURES
    }
    method = {}
    validation_rows = {}
    for task in ARCHITECTURES:
        rows = [
            row for row in runs[task]["ablations"]["method_rows"]
            if row.get("scope") == "decision" and row["aggregation"] == aggregation
        ][0]
        selections = runs[task]["ablations"]["selections"][aggregation]["families"]
        method[task] = {
            "aggregation": rows,
            "D_gate_required": True,
            "topology_pe_family_jaccard": jaccard(
                selections.get("topology_responsive", []), selections.get("pe_specialist", [])
            ),
        }
        validation_rows[task] = next(
            row for row in runs[task]["ablations"]["method_rows"]
            if row.get("scope") == "within_layer" and row["aggregation"] == aggregation
        )
    topology_coverage = finite_mean([topology[task]["coverage_tier_le_1"] for task in ARCHITECTURES])
    topology_validity = finite_mean([
        topology[task]["rho_topology_score_restore"] for task in ARCHITECTURES
    ])
    topology_alignment = finite_mean([
        topology[task]["rho_pe_topology_within_layer"] for task in ARCHITECTURES
    ])
    if topology_coverage >= 0.60 and topology_validity >= 0.40:
        topology_verdict = "retain_as_complementary_structural_validation"
    elif topology_coverage >= 0.35 and topology_validity >= 0.20:
        topology_verdict = "retain_exploratory_only"
    else:
        topology_verdict = "do_not_integrate"
    mechanism_max = max(
        float(np.nanmax(runs[task]["mechanism"]["reconstruction_max"]))
        for task in ARCHITECTURES
    )
    mechanism_full = []
    mechanism_interaction_ratio = []
    for task in ARCHITECTURES:
        mechanism = runs[task]["mechanism"]
        full = np.asarray(mechanism["metrics"]["full_rescue"], float)
        interaction = np.asarray(mechanism["metrics"]["finite_interaction"], float)
        mechanism_full.append(float(np.nanmean(full)))
        mechanism_interaction_ratio.append(float(
            np.nanmean(np.abs(interaction)) / max(np.nanmean(np.abs(full)), EPS)
        ))
    mechanism_valid = (
        mechanism_max <= MECHANISM_ATOL
        and all(value > 0.0 for value in mechanism_full)
        and all(value <= 0.50 for value in mechanism_interaction_ratio)
    )
    d_valid = all(
        validation_rows[task]["rho_D_restore_gated"] >= 0.30
        and validation_rows[task]["rho_D_necessity_gated"] >= 0.20
        for task in ARCHITECTURES
    )
    j_partial = finite_mean([
        validation_rows[task]["rho_J_beyond_layer_and_throughput"] for task in ARCHITECTURES
    ])
    g_partial = finite_mean([
        validation_rows[task]["rho_G_beyond_J_and_layer"] for task in ARCHITECTURES
    ])

    patch_advantages = []
    for task in ARCHITECTURES:
        causal = runs[task]["causal"]
        families = runs[task]["ablations"]["selections"][aggregation]["families"]
        for family, matched_channel, other_channel in (
            ("semantic_specialist", "semantic", "pe"),
            ("pe_specialist", "pe", "semantic"),
        ):
            heads = families.get(family, [])
            if not heads:
                patch_advantages.append(float("nan"))
                continue
            for metric in ("restore", "inject", "necessity"):
                matched = np.nanmean([
                    channel_mean_metric(causal, matched_channel, metric)[head] for head in heads
                ])
                off_channel = np.nanmean([
                    channel_mean_metric(causal, other_channel, metric)[head] for head in heads
                ])
                mismatch = (
                    np.nanmean([
                        channel_mean_metric(causal, matched_channel, "mismatch")[head]
                        for head in heads
                    ])
                    if metric == "restore" else 0.0
                )
                patch_advantages.append(float(matched - max(off_channel, mismatch)))
    patch_valid = all(np.isfinite(value) and value > 0.0 for value in patch_advantages)
    return {
        "overall_aggregation": aggregation,
        "aggregation_rule": (
            "replace CG only if one candidate is not worse in either architecture and improves "
            "the cross-architecture mean of causal-validity plus donor-prefix reliability by >=0.03"
        ),
        "D": {
            "verdict": "retain_gated" if d_valid else "descriptive_only",
            "rule": (
                "requires J/activity floor, bootstrap sign, and within-layer restore rho>=0.30 "
                "plus necessity rho>=0.20 in both architectures; ungated D is a failure-control"
            ),
        },
        "J": {
            "verdict": "retain_as_evoked_strength" if j_partial >= 0.30 else "retain_descriptive_only",
            "partial_rho_beyond_layer_and_throughput": j_partial,
        },
        "G": {
            "verdict": "retain_as_distinct_joint_strength" if g_partial >= 0.20 else "do_not_claim_incremental_value",
            "partial_rho_beyond_J_and_layer": g_partial,
            "constraint": "joint-channel strength, not synergy or selectivity",
        },
        "patching": {
            "verdict": "retain_required_confirmation" if patch_valid else "retain_diagnostic_not_label_gate",
            "matched_minus_max_offchannel_mismatch": patch_advantages,
            "protocol": "restore + inject + zero-ablation necessity, with sham and cross-graph calibration",
        },
        "routing_message": {
            "verdict": "retain_fixed_support_decomposition" if mechanism_valid else "descriptive_only",
            "maximum_exact_reconstruction_error": mechanism_max,
            "mean_full_rescue_by_architecture": mechanism_full,
            "finite_interaction_to_full_ratio": mechanism_interaction_ratio,
            "topology_limitation": (
                "not decomposed into only routing/message because changed support is a third wiring term"
            ),
        },
        "carriage": (
            "retain F_sens beside F_coh; use the changed-node set for PE distance and suppress a "
            "local topology profile when the topology edit is effectively global"
        ),
        "beneficial_carriage": (
            "retain the existing signed path-integrated loss-complete estimator unchanged; this "
            "beta adds exact donor-level helpful/harmful probabilities and conditional magnitudes, "
            "but does not replace carrier-wise beneficial carriage with an absolute score"
        ),
        "topology_donor": {
            "verdict": topology_verdict,
            "mean_tier_le_1_coverage": topology_coverage,
            "mean_score_restore_rho": topology_validity,
            "mean_within_layer_pe_alignment_rho": topology_alignment,
            "integration_constraint": (
                "keep as a separate topology axis; do not put whole-graph topology scores into "
                "the PE-based D denominator without a matched intervention unit/dose"
            ),
        },
        "conditional_specialisation": (
            "retain as a sample-split finite-basis screen; expand the condition basis only with "
            "predeclared hypotheses and independent confirmation"
        ),
        "replication_limit": (
            "This beta has one trained checkpoint per architecture. Graph bootstraps quantify "
            "evaluation uncertainty, not training-seed uncertainty; production adoption should "
            "be revisited if additional ZINC seeds disagree."
        ),
        "per_architecture": method,
    }


def create_outputs(runs: Mapping[str, Mapping[str, Any]], cfg: BetaConfig) -> dict[str, Any]:
    tables = cfg.root / "tables"
    figures = cfg.root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    aggregation_objectives: dict[str, list[float]] = {name: [] for name in AGGREGATIONS}
    method_rows = []
    for task in ARCHITECTURES:
        for row in runs[task]["ablations"]["method_rows"]:
            method_rows.append({"architecture": task, **row})
            if row.get("scope") == "decision":
                aggregation_objectives[row["aggregation"]].append(float(row["objective"]))
    overall = sorted(
        AGGREGATIONS,
        key=lambda name: (-np.nanmean(aggregation_objectives[name]), AGGREGATIONS.index(name)),
    )[0]
    objective_mean = {
        name: finite_mean(aggregation_objectives[name]) for name in AGGREGATIONS
    }
    candidate = overall
    if candidate != "CG":
        per_architecture_not_worse = all(
            next(
                row["objective"] for row in runs[task]["ablations"]["method_rows"]
                if row.get("scope") == "decision" and row["aggregation"] == candidate
            )
            >= next(
                row["objective"] for row in runs[task]["ablations"]["method_rows"]
                if row.get("scope") == "decision" and row["aggregation"] == "CG"
            )
            for task in ARCHITECTURES
        )
        if not per_architecture_not_worse or objective_mean[candidate] - objective_mean["CG"] < 0.03:
            overall = "CG"
    write_csv(tables / "aggregation_validation.csv", method_rows)

    topology_rows = []
    ablation_rows = []
    causal_rows = []
    mechanism_rows = []
    family_patch_rows = []
    family_ablation_rows = []
    carriage_rows = []
    conditional = {}
    event_rows = []
    for task in ARCHITECTURES:
        event_rows.extend(intervention_event_rows(task, runs[task]["scores"]))
        summary = topology_summary(runs[task]["scores"], runs[task]["causal"], overall)
        topology_rows.append({
            "architecture": task,
            **{key: value for key, value in summary.items() if np.isscalar(value) or isinstance(value, dict)},
        })
        selection = runs[task]["ablations"]["selections"][overall]
        families = selection["families"]
        coordinates = selection["coordinates"]
        impact = runs[task]["ablations"]["per_head"]
        for layer in range(int(runs[task]["scores"]["L"])):
            for head in range(int(runs[task]["scores"]["H"])):
                ablation_rows.append({
                    "architecture": task, "layer": layer, "head": head,
                    "S_sem": selection["semantic"][layer, head],
                    "S_pe": selection["pe"][layer, head],
                    "S_topology": selection["topology"][layer, head],
                    "D": coordinates["D"][layer, head],
                    "J": coordinates["J"][layer, head],
                    "G": coordinates["G"][layer, head],
                    "active": bool(selection["active"][layer, head]),
                    "functional_ablation": impact["functional"][layer, head],
                    "loss_ablation": impact["loss_increase"][layer, head],
                })
        causal = runs[task]["causal"]
        for channel in CHANNELS:
            for metric in ("restore", "inject", "necessity", "mismatch", "sham"):
                matrix = channel_mean_metric(causal, channel, metric)
                for layer in range(matrix.shape[0]):
                    for head in range(matrix.shape[1]):
                        causal_rows.append({
                            "architecture": task, "channel": channel, "metric": metric,
                            "layer": layer, "head": head, "value": matrix[layer, head],
                        })
        for family, heads in families.items():
            for channel in CHANNELS:
                for metric in ("restore", "inject", "necessity", "mismatch", "sham"):
                    matrix = channel_mean_metric(causal, channel, metric)
                    family_patch_rows.append({
                        "architecture": task,
                        "family": family,
                        "heads": str(list(heads)),
                        "channel": channel,
                        "metric": metric,
                        "mean": float(np.nanmean([matrix[head] for head in heads])) if heads else float("nan"),
                    })
            for row in runs[task]["ablations"]["family_curves"][overall].get(family, []):
                family_ablation_rows.append({"architecture": task, "family": family, **row})
        mechanism = runs[task]["mechanism"]
        for channel in PRIMARY_CHANNELS:
            mask = np.asarray([event["channel"] == channel for event in mechanism["events"]])
            for layer in range(int(runs[task]["scores"]["L"])):
                for head in range(int(runs[task]["scores"]["H"])):
                    mechanism_rows.append({
                        "architecture": task,
                        "channel": channel,
                        "layer": layer,
                        "head": head,
                        **{
                            key: float(np.nanmean(np.asarray(value)[mask, layer, head]))
                            for key, value in mechanism["metrics"].items()
                        },
                    })
        for channel, family in (("semantic", "semantic_specialist"), ("pe", "pe_specialist")):
            profile = carriage_profile(runs[task]["scores"], channel, families.get(family, []))
            for index, distance in enumerate(profile["distance"]):
                carriage_rows.append({
                    "architecture": task,
                    "channel": channel,
                    "family": family,
                    "distance": int(distance),
                    "F_sens": float(profile["F_sens"][index]),
                    "F_coh": float(profile["F_coh"][index]),
                    "F_sens_ci_low": float(profile["F_sens_ci_low"][index]),
                    "F_sens_ci_high": float(profile["F_sens_ci_high"][index]),
                    "F_coh_ci_low": float(profile["F_coh_ci_low"][index]),
                    "F_coh_ci_high": float(profile["F_coh_ci_high"][index]),
                    "graph_support": int(profile["graph_support"][index]),
                })
        conditional[task] = conditional_analysis(runs[task]["scores"], overall, cfg)
        write_json(tables / f"conditional_{task}.json", conditional[task])
    write_csv(tables / "topology_validation.csv", topology_rows)
    write_csv(tables / "head_scores_and_ablation.csv", ablation_rows)
    write_csv(tables / "causal_head_metrics.csv", causal_rows)
    write_csv(tables / "family_patch_metrics.csv", family_patch_rows)
    write_csv(tables / "family_ablation_curves.csv", family_ablation_rows)
    write_csv(tables / "routing_message_metrics.csv", mechanism_rows)
    write_csv(tables / "carriage_profiles.csv", carriage_rows)
    write_csv(tables / "intervention_events_and_dose.csv", event_rows)
    write_csv(tables / "donor_outcome_summary.csv", donor_outcome_summary_rows(event_rows))

    figure_paths = []
    figure_paths += figure_aggregation_planes(runs, figures / "zinc_beta_fig01_aggregation_planes")
    figure_paths += figure_method_validation(runs, figures / "zinc_beta_fig02_method_validation")
    figure_paths += figure_selectivity_strength(runs, overall, figures / "zinc_beta_fig03_selectivity_strength")
    figure_paths += figure_causal_coordinates(runs, overall, figures / "zinc_beta_fig04_causal_coordinates")
    figure_paths += figure_family_patching(runs, overall, figures / "zinc_beta_fig05_family_patching")
    figure_paths += figure_patch_controls(runs, figures / "zinc_beta_fig06_patch_controls")
    figure_paths += figure_topology_validation(runs, overall, figures / "zinc_beta_fig07_topology_validation")
    figure_paths += figure_ablation(runs, overall, figures / "zinc_beta_fig08_ablation")
    figure_paths += figure_routing_message(runs, overall, figures / "zinc_beta_fig09_routing_message")
    figure_paths += figure_carriage(runs, overall, figures / "zinc_beta_fig10_carriage")
    figure_paths += figure_conditional(conditional, figures / "zinc_beta_fig11_conditional")

    decisions = create_decisions(runs, overall)
    confirmed_rules = {
        task: {
            row["rule"] for row in conditional[task].get("confirmed", [])
            if row.get("confirmed_at_fdr_0.05", False)
        }
        for task in ARCHITECTURES
    }
    shared_confirmed = sorted(set.intersection(*confirmed_rules.values())) if confirmed_rules else []
    decisions["conditional_specialisation"] = {
        "pipeline_verdict": "retain_sample_split_screen",
        "confirmed_rule_counts": {task: len(values) for task, values in confirmed_rules.items()},
        "shared_confirmed_rules": shared_confirmed,
        "label_verdict": "approve_shared_conditions" if shared_confirmed else "exploratory_only",
        "constraint": "finite predeclared basis with held-out global-FDR, dose and localisation diagnostics",
    }
    write_json(tables / "methodology_decisions.json", decisions)
    scope = {
        "primary_comparison": "semantic donor swaps versus mask-frozen PE transpositions",
        "topology_probe": (
            "matched-real non-isomorphic molecular donors, same n, preferentially exact atom/degree/"
            "bond multisets, Hungarian node alignment, donor-recomputed RRWP, base x/y fixed"
        ),
        "data_independence": (
            "score discovery, causal patching, and ordinary/family ablation use disjoint deterministic "
            "ZINC test graph subsets; conditional rules and heads use a further discovery/confirmation split"
        ),
        "models": [DISPLAY[task] for task in ARCHITECTURES],
        "training_seed_limit": "one available checkpoint per architecture",
    }
    write_json(tables / "scope_and_limitations.json", scope)
    summary = {
        "version": BETA_VERSION,
        "schema": BETA_SCHEMA,
        "fingerprint": cfg.fingerprint,
        "selected_aggregation": overall,
        "aggregation_candidate": candidate,
        "aggregation_objective_mean": objective_mean,
        "successor_minimum_margin_over_CG": 0.03,
        "decisions": decisions,
        "figures": figure_paths,
        "tables": [str(item) for item in sorted(tables.iterdir())],
        "scope": scope,
    }
    write_json(cfg.root / "summary.json", summary)
    return summary


# =====================================================================================
# Orchestration
# =====================================================================================


def find_cached_phase(cfg: BetaConfig, task: str, phase: str) -> dict[str, Any]:
    import torch

    pattern = f"{phase}__{cfg.fingerprint}__*.pt"
    paths = sorted((cfg.root / "cache" / task).glob(pattern))
    if len(paths) > 1:
        provenance_path = cfg.root / "provenance" / f"{task}.json"
        if provenance_path.exists():
            expected = json.loads(provenance_path.read_text(encoding="utf-8")).get(
                "checkpoint_sha256"
            )
            if expected:
                matching = []
                for candidate in paths:
                    payload = torch.load(candidate, map_location="cpu", weights_only=False)
                    if payload.get("checkpoint_sha256") == expected:
                        matching.append(candidate)
                if len(matching) == 1:
                    paths = matching
    if len(paths) != 1:
        listing = "\n".join(f"  - {path}" for path in paths) or "  (none)"
        raise RuntimeError(
            f"expected exactly one {phase!r} cache for {task!r}, found {len(paths)}:\n{listing}"
        )
    payload = torch.load(paths[0], map_location="cpu", weights_only=False)
    if (
        payload.get("version") != BETA_VERSION
        or int(payload.get("schema", -1)) != BETA_SCHEMA
        or payload.get("fingerprint") != cfg.fingerprint
    ):
        raise RuntimeError(f"stale/incompatible cache: {paths[0]}")
    return dict(payload)


def cached_runs(cfg: BetaConfig) -> dict[str, dict[str, Any]]:
    return {
        task: {
            "scores": find_cached_phase(cfg, task, "scores"),
            "causal": find_cached_phase(cfg, task, "causal"),
            "mechanism": find_cached_phase(cfg, task, "mechanism"),
            "ablations": find_cached_phase(cfg, task, "ablations"),
        }
        for task in ARCHITECTURES
    }


def audit_cached_architecture_alignment(runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    dense = {int(row["graph_id"]): row for row in runs["zinc"]["scores"]["records"]}
    onehop = {int(row["graph_id"]): row for row in runs["zinc_1hop"]["scores"]["records"]}
    common = sorted(set(dense) & set(onehop))
    failures = []
    for graph_id in common:
        left = dense[graph_id]
        right = onehop[graph_id]
        left_descriptor = left["plan"]["descriptor"]
        right_descriptor = right["plan"]["descriptor"]
        same = (
            int(left["n"]) == int(right["n"])
            and np.array_equal(left_descriptor["labels"], right_descriptor["labels"])
            and np.array_equal(left_descriptor["edges"], right_descriptor["edges"])
            and np.allclose(left["target"], right["target"])
        )
        left_plan, right_plan = left["plan"], right["plan"]
        same = same and np.array_equal(left_plan["sources"], right_plan["sources"])
        if same:
            same = all(
                np.array_equal(a["donor_graph_ids"], b["donor_graph_ids"])
                for a, b in zip(left_plan["semantic"], right_plan["semantic"])
            ) and all(
                np.array_equal(a["partners"], b["partners"])
                for a, b in zip(left_plan["pe"], right_plan["pe"])
            ) and (
                [int(item["donor_graph_id"]) for item in left_plan["topology"]]
                == [int(item["donor_graph_id"]) for item in right_plan["topology"]]
            )
        if not same:
            failures.append(graph_id)
    result = {
        "common_score_graphs": len(common),
        "failures": failures,
        "passed": len(common) == len(dense) == len(onehop) and not failures,
    }
    if not result["passed"]:
        raise RuntimeError(f"cached dense/1-hop graph alignment failed: {result}")
    return result


def environment_record(cfg: BetaConfig) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "torch_geometric", "numpy", "scipy", "networkx", "matplotlib"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            packages[name] = f"unavailable: {exc}"
    try:
        import torch

        cuda = torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        cuda, gpu = None, None
    return {
        "version": BETA_VERSION,
        "schema": BETA_SCHEMA,
        "fingerprint": cfg.fingerprint,
        "python": sys.version,
        "packages": packages,
        "cuda": cuda,
        "gpu": gpu,
    }


def make_config(args: argparse.Namespace) -> BetaConfig:
    values = {
        "output_dir": args.output_dir,
        "score_graphs": args.score_graphs,
        "score_sources": args.score_sources,
        "semantic_donors": args.semantic_donors,
        "pe_partners": args.pe_partners,
        "topology_donors": args.topology_donors,
        "topology_pool": args.topology_pool,
        "causal_graphs": args.causal_graphs,
        "causal_sources": args.causal_sources,
        "causal_batch_graphs": args.causal_batch_graphs,
        "mechanism_graphs": args.mechanism_graphs,
        "ablation_graphs": args.ablation_graphs,
        "family_size": args.family_size,
        "bootstrap_samples": args.bootstrap_samples,
        "conditional_bootstrap_samples": args.conditional_bootstrap_samples,
        "analysis_seed": args.analysis_seed,
        "device": args.device,
        "dense_checkpoint": args.dense_checkpoint,
        "onehop_checkpoint": args.onehop_checkpoint,
        "allow_relaxed_topology": not args.strict_topology_only,
        "causal_effect_floor_relative": args.causal_effect_floor_relative,
    }
    if args.fast_dev_run:
        values.update({
            "output_dir": args.output_dir + "_fast_dev",
            "score_graphs": 6,
            "score_sources": 2,
            "semantic_donors": 2,
            "pe_partners": 2,
            "topology_donors": 1,
            "topology_pool": 300,
            "causal_graphs": 4,
            "causal_sources": 1,
            "causal_batch_graphs": 4,
            "mechanism_graphs": 2,
            "ablation_graphs": 8,
            "family_size": 1,
            "bootstrap_samples": 40,
            "conditional_bootstrap_samples": 40,
        })
    cfg = BetaConfig(**values)
    cfg.validate()
    return cfg


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone ZINC beta validation of successor graph-head specialisation"
    )
    parser.add_argument(
        "--phase",
        choices=("all", "scores", "causal", "mechanism", "ablations", "figures"),
        default="all",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--score-graphs", type=int, default=96)
    parser.add_argument("--score-sources", type=int, default=8)
    parser.add_argument("--semantic-donors", type=int, default=8)
    parser.add_argument("--pe-partners", type=int, default=8)
    parser.add_argument("--topology-donors", type=int, default=2)
    parser.add_argument("--topology-pool", type=int, default=10_000)
    parser.add_argument("--causal-graphs", type=int, default=48)
    parser.add_argument("--causal-sources", type=int, default=2)
    parser.add_argument("--causal-batch-graphs", type=int, default=8)
    parser.add_argument("--mechanism-graphs", type=int, default=16)
    parser.add_argument("--ablation-graphs", type=int, default=128)
    parser.add_argument("--family-size", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--conditional-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--analysis-seed", type=int, default=1771)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dense-checkpoint", default=None)
    parser.add_argument("--onehop-checkpoint", default=None)
    parser.add_argument("--strict-topology-only", action="store_true")
    parser.add_argument("--causal-effect-floor-relative", type=float, default=0.05)
    parser.add_argument("--repository-branch", default=REPOSITORY_BRANCH)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--pyg-version", default="2.2.0")
    parser.add_argument("--force-fresh-grit", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    repository = bootstrap_repository(
        branch=args.repository_branch, skip_checkout=args.skip_bootstrap
    )
    source = str(repository / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    cfg = make_config(args)
    cfg.root.mkdir(parents=True, exist_ok=True)
    write_json(
        cfg.root / "beta_config.json",
        {"version": BETA_VERSION, "schema": BETA_SCHEMA, "fingerprint": cfg.fingerprint, "config": asdict(cfg)},
    )

    if args.phase != "figures":
        from graph_specialisation_metrics.carriage import env

        if not args.skip_install:
            env.install_dependencies(pyg_version=args.pyg_version)
        else:
            env.apply_compat_patches()
        write_json(cfg.root / "environment.json", environment_record(cfg))
        # 1-hop first makes molecular-support failures obvious before dense all-pairs analysis.
        for task in ("zinc_1hop", "zinc"):
            print("\n" + "#" * 90 + f"\n# {DISPLAY[task]}\n" + "#" * 90, flush=True)
            loaded = prepare_task(task, cfg, force_fresh_grit=args.force_fresh_grit)
            try:
                score_payload = None
                causal_payload = None
                if args.phase in {"all", "scores"}:
                    score_payload = run_scores_task(loaded, cfg, force=args.force)
                if args.phase in {"all", "causal"}:
                    causal_payload = run_causal_task(loaded, cfg, force=args.force)
                if args.phase in {"all", "mechanism"}:
                    run_mechanism_task(loaded, cfg, force=args.force)
                if args.phase in {"all", "ablations"}:
                    if score_payload is None:
                        score_payload = find_cached_phase(cfg, task, "scores")
                    if causal_payload is None:
                        causal_payload = find_cached_phase(cfg, task, "causal")
                    run_ablation_task(
                        loaded, cfg, score_payload, causal_payload, force=args.force
                    )
            finally:
                try:
                    import torch

                    del loaded["gm"]
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    if args.phase not in {"all", "figures"}:
        result = {"version": BETA_VERSION, "phase": args.phase, "output_dir": str(cfg.root)}
        print(f"[done] {result}", flush=True)
        return result

    runs = cached_runs(cfg)
    alignment = audit_cached_architecture_alignment(runs)
    write_json(cfg.root / "tables" / "cross_architecture_dataset_alignment.json", alignment)
    summary = create_outputs(runs, cfg)
    print("\n[done] ZINC specialisation redesign beta", flush=True)
    print(f"  output: {cfg.root}", flush=True)
    print(f"  selected aggregation: {summary['selected_aggregation']}", flush=True)
    print(f"  topology decision: {summary['decisions']['topology_donor']['verdict']}", flush=True)
    return summary


CELL_ARGS = [
    "--phase", "all",
    "--output-dir", str(DEFAULT_OUTPUT),
]


if __name__ == "__main__":
    try:
        colab_installed = importlib.util.find_spec("google.colab") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        colab_installed = False
    in_colab = (
        bool(os.environ.get("COLAB_RELEASE_TAG"))
        or "google.colab" in sys.modules
        or colab_installed
    )
    main(CELL_ARGS if in_colab else None)
