"""Standalone Colab rerun for the headline ZINC specialisation methodology.

Paste this file into one Colab cell, or run it from the repository.  It loads the
trained dense-GRIT and 1-hop-GRIT ZINC checkpoints read-only, evaluates the successor
specialisation protocol on deterministic disjoint graph splits, and writes resumable
artifacts beneath ``MyDrive/graph_specialisation_metrics/zinc_redesign_headline_v2``.

This remains deliberately isolated from the production analysis packages, but its two headline
estimands are fixed rather than reselected from the data: graph-balanced eventwise-gross ``EG``
specialisation and eventwise functional sensitivity ``F_sens`` carriage.  It tests:

* graph-balanced donor -> source -> graph EG estimation and donor-count convergence;
* D selectivity, evoked strength J, and joint-channel strength G;
* whole-transport restore/inject patching, zero-ablation necessity, sham and
  same-node-count cross-graph mismatch controls on an independent graph split;
* individual and family ablation on a third split;
* support-aware routing/message/wiring decomposition for all three interventions;
* final-state ``F_sens`` and path-integrated beneficial carriage for semantic, PE and topology;
* an exact distance decomposition of each head's EG score, with a separate hub bucket,
  a support-normalised per-opportunity companion, clean attention-distance locality,
  and frozen-family reach profiles aligned to functional and beneficial carriage;
* an expanded sample-split conditional screen over raw scores and all pairwise D/J/G coordinates;
* a matched-real, non-isomorphic molecular-topology donor intervention with donor
  topology/RRWP copied in aligned base-node coordinates while atom content and target
  stay fixed.

The topology intervention remains a separate third channel.  The rerun explicitly adds semantic
versus topology score planes, activity-gated PE versus topology contrasts, pooled/within-layer
correlations, top-k graph-bootstrap stability, frozen PE-specific/topology-specific/shared families,
held-out simultaneous three-channel family patching, and tier/dose-stratified results.

The expensive phases are resumable: ``scores``, ``attention``, ``causal``, ``mechanism``,
``ablations``, and ``figures``. Score estimation checkpoints every completed source group,
channel and graph; clean attention locality checkpoints every graph;
rare quadrature cap hits are retained only under explicit per-path error and global-rate gates.
``all`` runs the phases in order. ``--fast-dev-run`` is plumbing only.
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


BETA_VERSION = "zinc-specialisation-headline-v2"
BETA_SCHEMA = 2
REPOSITORY_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
REPOSITORY_BRANCH = "codex/cfim-grit-experiments"
COLAB_REPOSITORY = Path("/content/Graph-Specialisation-and-Metrics")
DEFAULT_OUTPUT = Path(
    "/content/drive/MyDrive/graph_specialisation_metrics/zinc_redesign_headline_v2"
)
SECRET_NAME = "dissertation_key"
ARCHITECTURES = ("zinc", "zinc_1hop")
DISPLAY = {"zinc": "Dense GRIT", "zinc_1hop": "1-hop GRIT"}
PRIMARY_CHANNELS = ("semantic", "pe")
CHANNELS = ("semantic", "pe", "topology")
AGGREGATIONS = ("CG", "EG", "CN", "EN")
HEADLINE_AGGREGATION = "EG"
PATCH_FAMILY_NAMES = (
    "semantic_specialist",
    "pe_specialist",
    "structural_pe_specific",
    "structural_topology_specific",
    "structural_shared",
    "high_J",
    "high_G_balanced",
    "low_J_inert",
)
DISTANCE_FAMILIES = {
    "semantic": ("semantic_specialist", "high_G_balanced"),
    "pe": ("structural_pe_specific", "structural_shared"),
    "topology": ("structural_topology_specific", "structural_shared"),
}
CONDITIONAL_SCORE_FEATURES = (
    "S_semantic", "S_pe", "S_topology",
    "D_semantic_pe", "J_semantic_pe", "G_semantic_pe",
    "D_semantic_topology", "J_semantic_topology", "G_semantic_topology",
    "D_pe_topology", "J_pe_topology", "G_pe_topology",
    "J_three_channel", "G_three_channel",
)
EPS = 1.0e-12
CUDA_EQUIVALENCE_ATOL = 1.0e-4
CUDA_EQUIVALENCE_RTOL = 2.0e-5
MECHANISM_ATOL = 3.0e-4
DISTANCE_UNREACHABLE = -1
DISTANCE_HUB = -2
DISTANCE_IDENTITY_ATOL = 2.0e-5
DISTANCE_IDENTITY_RTOL = 2.0e-5


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
    topology_donors: int = 6
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
    conditional_per_feature: int = 2
    conditional_max_tests: int = 48
    topk_values: tuple[int, ...] = (3, 5, 10)
    # ZINC's L1/readout paths can contain isolated kinks.  This established carriage
    # policy converges quickly for almost all paths and retains only rare, numerically
    # bounded cap hits rather than aborting an otherwise valid multi-hour run.
    integrated_atol: float = 5.0e-4
    integrated_rtol: float = 1.0e-4
    integrated_max_intervals: int = 256
    integrated_unconverged_error_cap: float = 5.0e-3
    integrated_max_unconverged_fraction: float = 1.0e-2

    def validate(self) -> None:
        for name in (
            "score_graphs", "score_sources", "semantic_donors", "pe_partners",
            "topology_donors", "topology_pool", "causal_graphs",
            "causal_sources", "causal_batch_graphs", "mechanism_graphs", "ablation_graphs",
            "family_size", "bootstrap_samples", "conditional_bootstrap_samples",
            "conditional_per_feature", "conditional_max_tests", "integrated_max_intervals",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.score_graphs + self.causal_graphs + self.ablation_graphs > 1000:
            raise ValueError("disjoint ZINC test subsets exceed the 1000-graph test split")
        if not 0.0 < self.activity_floor_relative < 1.0:
            raise ValueError("activity_floor_relative must be in (0,1)")
        if not 0.0 <= self.causal_effect_floor_relative < 1.0:
            raise ValueError("causal_effect_floor_relative must be in [0,1)")
        if not self.topk_values or any(int(value) < 1 for value in self.topk_values):
            raise ValueError("topk_values must contain positive integers")
        if self.conditional_max_tests < self.conditional_per_feature * len(
            CONDITIONAL_SCORE_FEATURES
        ):
            raise ValueError(
                "conditional_max_tests must cover conditional_per_feature for every score feature"
            )
        if self.integrated_atol < 0 or self.integrated_rtol < 0:
            raise ValueError("integrated carriage tolerances must be non-negative")
        if self.integrated_unconverged_error_cap < 0:
            raise ValueError("integrated_unconverged_error_cap must be non-negative")
        if not 0.0 <= self.integrated_max_unconverged_fraction <= 1.0:
            raise ValueError("integrated_max_unconverged_fraction must lie in [0,1]")

    @property
    def root(self) -> Path:
        return Path(self.output_dir)

    @property
    def fingerprint(self) -> str:
        values = asdict(self)
        for key in ("output_dir", "device", "dense_checkpoint", "onehop_checkpoint"):
            values.pop(key, None)
        return stable_hash({"version": BETA_VERSION, **values})

    @property
    def legacy_strict_score_fingerprint(self) -> str:
        """Fingerprint of the immediately preceding fail-on-any-cap score policy.

        A completed graph under that policy necessarily converged every path at tighter
        tolerances, so it is safe to migrate into the new bounded policy. Partial/failed
        graphs had no complete graph cache and therefore cannot be migrated silently.
        """

        values = asdict(self)
        for key in ("output_dir", "device", "dense_checkpoint", "onehop_checkpoint"):
            values.pop(key, None)
        values.pop("integrated_unconverged_error_cap", None)
        values.pop("integrated_max_unconverged_fraction", None)
        values["integrated_atol"] = 1.0e-5
        values["integrated_rtol"] = 1.0e-4
        values["integrated_max_intervals"] = 128
        return stable_hash({"version": BETA_VERSION, **values})


def cache_path(cfg: BetaConfig, task: str, phase: str, checkpoint_sha: str) -> Path:
    return cfg.root / "cache" / task / (
        f"{phase}__{cfg.fingerprint}__{checkpoint_sha[:12]}.pt"
    )


def score_component_cache_path(
    cfg: BetaConfig,
    task: str,
    graph_id: int,
    component: str,
    checkpoint_sha: str,
) -> Path:
    """Fine-grained score cache: source groups -> channels -> complete graph."""

    safe_component = component.replace("/", "_")
    return cfg.root / "cache" / task / "score_components" / (
        f"graph_{int(graph_id)}__{safe_component}__{cfg.fingerprint}__"
        f"{checkpoint_sha[:12]}.pt"
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


def valid_legacy_strict_graph_cache(
    path: Path, cfg: BetaConfig, checkpoint_sha: str
) -> dict[str, Any] | None:
    """Accept only complete, fully converged graph caches from the prior strict policy."""

    import torch

    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or (
        payload.get("version") != BETA_VERSION
        or int(payload.get("schema", -1)) != BETA_SCHEMA
        or payload.get("fingerprint") != cfg.legacy_strict_score_fingerprint
        or payload.get("checkpoint_sha256") != checkpoint_sha
        or "record" not in payload
    ):
        return None
    audit = integrated_carriage_audit([payload["record"]])
    if int(audit["unconverged"]) != 0:
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
    by_tier: dict[int, list[tuple[int, float, int, Mapping[str, Any]]]] = {}
    for candidate in candidates:
        by_tier.setdefault(int(candidate[0]), []).append(candidate)
    # Interleave tiers so the sensitivity analysis actually contains strict, near and
    # relaxed donors when available. Headline scores still discard tier 2 downstream.
    stratified_candidates = []
    tier_offsets = {tier: 0 for tier in sorted(by_tier)}
    while len(stratified_candidates) < int(count):
        added = False
        for tier in sorted(by_tier):
            offset = tier_offsets[tier]
            if offset < len(by_tier[tier]):
                stratified_candidates.append(by_tier[tier][offset])
                tier_offsets[tier] += 1
                added = True
                if len(stratified_candidates) == int(count):
                    break
        if not added:
            break
    selected = []
    for tier, cost, graph_id, donor in stratified_candidates:
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


def clean_gradients(gm: Any, base: Any) -> tuple[Any, list[Any], Any, Any]:
    import torch
    from torch_geometric.data import Batch

    batch = Batch.from_data_list([base.clone()]).to(gm.device)
    final: dict[str, Any] = {}

    def final_hook(_module: Any, _inputs: Any, output: Any) -> None:
        final["h"] = output.x

    handle = gm.model.model.layers.register_forward_hook(final_hook)
    try:
        capture = gm.capture(batch, want_grad=True, want_attn=False)
    finally:
        handle.remove()
    pred = capture["pred"]
    outputs = pred.reshape(pred.shape[0], -1)
    gradients, final_gradients = [], []
    for target in range(outputs.shape[1]):
        values = torch.autograd.grad(
            outputs[0, target], [*capture["wV"], final["h"]],
            retain_graph=target + 1 < outputs.shape[1],
        )
        gradients.append(values[:-1])
        final_gradients.append(values[-1])
    phi = [torch.stack([gradient[layer] for gradient in gradients], dim=0) for layer in range(gm.L)]
    final_phi = torch.stack(final_gradients, dim=0).detach()  # [T,N,D]
    if tuple(final_phi.shape[1:]) != (int(base.num_nodes), int(gm.dim_h)):
        raise RuntimeError(f"unexpected final-state gradient shape {tuple(final_phi.shape)}")
    return pred.detach(), phi, capture, final_phi


def projected_event_group(
    gm: Any,
    base: Any,
    variants: Sequence[Any],
    phi: Sequence[Any],
) -> tuple[Any, np.ndarray, np.ndarray, Any]:
    """q=[K,L,H,N,T] using within-batch clean baselines to cancel scatter jitter."""

    import torch
    from torch_geometric.data import Batch

    if not variants:
        return (
            torch.empty(0, gm.L, gm.H, int(base.num_nodes), len(phi[0])),
            np.empty(0), np.empty(0), torch.empty(0, device=gm.device),
        )
    replicas = [base.clone(), *[variant.clone() for variant in variants]]
    batch = Batch.from_data_list(replicas).to(gm.device)
    final: dict[str, Any] = {}

    def final_hook(_module: Any, _inputs: Any, output: Any) -> None:
        final["h"] = output.x.detach()

    handle = gm.model.model.layers.register_forward_hook(final_hook)
    try:
        capture = gm.capture(batch, want_grad=False, want_attn=False, include_virtual_transport=True)
    finally:
        handle.remove()
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
    final_states = final["h"].reshape(len(replicas), per_replica, gm.dim_h)
    return q.detach().cpu(), predictions, no_op, final_states


def final_state_carriage_group(
    gm: Any,
    final_states: Any,
    final_output_gradient: Any,
    target: Any,
    cfg: BetaConfig,
    *,
    full_predictions: np.ndarray | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Return eventwise output projections and complete signed loss carriage.

    ``q_final`` is ``[K,N,T]`` and supplies production F_sens. ``B`` is ``[K,N]``:
    every donor/partner path is integrated before later source/graph aggregation.
    """

    import torch

    from graph_specialisation_metrics.carriage import core, metrics
    from graph_specialisation_metrics.carriage.grit_runner import (
        _pooled_head_predictions,
        _retain_or_reject_unconverged_paths,
    )

    if final_states.ndim != 3 or int(final_states.shape[0]) < 2:
        raise ValueError(f"final_states must be [clean+K,N,D], got {tuple(final_states.shape)}")
    clean = final_states[0:1].expand(int(final_states.shape[0]) - 1, -1, -1)
    swap = final_states[1:]
    delta = clean - swap
    q_final = torch.einsum("tnd,knd->knt", final_output_gradient, delta)
    target_row = target.detach().reshape(1, -1).float().to(gm.device)
    pooling = str(gm.cfg.model.graph_pooling)

    def loss_from_pooled(pooled: Any) -> Any:
        pred = _pooled_head_predictions(gm.model, pooled, target_row)
        return metrics.per_graph_loss(
            pred, target_row.expand(int(pooled.shape[0]), -1), gm.loss_fun
        )

    path = core.integrated_loss_carriage(
        clean,
        swap,
        loss_from_pooled,
        pooling=pooling,
        atol=cfg.integrated_atol,
        rtol=cfg.integrated_rtol,
        max_intervals=cfg.integrated_max_intervals,
    )
    residual = path["completeness_residual"].abs()
    converged = _retain_or_reject_unconverged_paths(
        path, cfg, "ZINC beta donor"
    )
    failed = ~np.asarray(converged, dtype=bool)
    residual_numpy = residual.detach().cpu().numpy().astype(float)
    error_numpy = path["quadrature_error"].detach().cpu().numpy().astype(float)
    endpoint_replay = None
    if full_predictions is not None:
        with torch.no_grad():
            pooled = core.pool_final_states(final_states, pooling)
            replay = _pooled_head_predictions(
                gm.model, pooled, target_row
            ).detach().cpu()
        endpoint_replay = tensor_equivalent(
            torch.as_tensor(np.asarray(full_predictions)), replay
        )
        if not endpoint_replay["passed"]:
            raise RuntimeError(f"final-state readout endpoint replay failed: {endpoint_replay}")
    diagnostics = {
        "completeness_max": float(residual.max()),
        "quadrature_error_max": float(path["quadrature_error"].max()),
        "unconverged": int(failed.sum()),
        "paths": int(path["converged"].numel()),
        "unconverged_fraction": float(failed.mean()),
        "unconverged_indices": np.flatnonzero(failed).astype(np.int64),
        "unconverged_completeness_max": (
            float(residual_numpy[failed].max()) if failed.any() else 0.0
        ),
        "unconverged_quadrature_error_max": (
            float(error_numpy[failed].max()) if failed.any() else 0.0
        ),
        "intervals_max": int(path["intervals"].max()),
        "integrated_atol": float(cfg.integrated_atol),
        "integrated_rtol": float(cfg.integrated_rtol),
        "integrated_max_intervals": int(cfg.integrated_max_intervals),
        "integrated_unconverged_error_cap": float(
            cfg.integrated_unconverged_error_cap
        ),
        "endpoint_replay": endpoint_replay,
    }
    return q_final.detach().cpu(), path["carriage"].detach().cpu(), diagnostics


def integrated_carriage_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate the predeclared rare-cap policy over every cached score path."""

    diagnostics = [
        diagnostic
        for record in records
        for channel in CHANNELS
        for diagnostic in record["channels"][channel].get("beneficial_diagnostics", [])
    ]
    paths = int(sum(int(item.get("paths", 0)) for item in diagnostics))
    unconverged = int(sum(int(item.get("unconverged", 0)) for item in diagnostics))
    return {
        "groups": len(diagnostics),
        "paths": paths,
        "unconverged": unconverged,
        "unconverged_fraction": float(unconverged / paths) if paths else 0.0,
        "completeness_max": float(max(
            (float(item.get("completeness_max", 0.0)) for item in diagnostics),
            default=0.0,
        )),
        "quadrature_error_max": float(max(
            (float(item.get("quadrature_error_max", 0.0)) for item in diagnostics),
            default=0.0,
        )),
        "unconverged_completeness_max": float(max(
            (float(item.get("unconverged_completeness_max", 0.0)) for item in diagnostics),
            default=0.0,
        )),
        "unconverged_quadrature_error_max": float(max(
            (float(item.get("unconverged_quadrature_error_max", 0.0)) for item in diagnostics),
            default=0.0,
        )),
        "intervals_max": int(max(
            (int(item.get("intervals_max", 0)) for item in diagnostics), default=0
        )),
    }


def distance_profile_one_graph(
    q: Any, distances: np.ndarray, beneficial: Any | None = None
) -> list[dict[str, Any]]:
    """F_sens/F_coh with event-specific changed sets and equal carrier weighting."""

    import torch

    distance = torch.as_tensor(distances, dtype=torch.long)
    if tuple(distance.shape) != (int(q.shape[0]), int(q.shape[-2])):
        raise ValueError(f"distance {distance.shape} is not [events,carriers] for q {q.shape}")
    if beneficial is not None and tuple(beneficial.shape) != tuple(distance.shape):
        raise ValueError(
            f"beneficial {tuple(beneficial.shape)} is not [events,carriers] for {tuple(distance.shape)}"
        )
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
        row = {
            "distance": hop,
            "F_sens": per_carrier_sens[..., carrier_valid].mean(dim=-1).numpy(),
            "F_coh": per_carrier_coh[..., carrier_valid].mean(dim=-1).numpy(),
            "carriers": int(carrier_valid.sum()),
            "event_carrier_pairs": int(pair_mask.sum()),
        }
        if beneficial is not None:
            benefit = torch.as_tensor(beneficial, dtype=q.dtype)
            per_carrier_b = (benefit * pair_mask.to(q.dtype)).sum(dim=0) / carrier_count.clamp_min(1)
            row["B"] = float(per_carrier_b[carrier_valid].mean())
            row["B_sum"] = float(
                (benefit * pair_mask.to(q.dtype)).sum(dim=-1).mean()
            )
        rows.append(row)
    return rows


def event_distance_matrix(
    record: Mapping[str, Any],
    channel: str,
    source_index: int,
    *,
    event_count: int,
    carrier_count: int,
) -> np.ndarray:
    """Distance from every event's changed set to every transport carrier.

    Real nodes receive molecular shortest-path distance. Disconnected real nodes and
    virtual-node carriers receive distinct codes so neither can silently disappear from
    the exact score decomposition.
    """

    n = int(record["n"])
    if carrier_count < n:
        raise ValueError(f"transport has {carrier_count} carriers for a {n}-node graph")
    plan = record["plan"]
    edges = np.asarray(plan["descriptor"]["edges"], dtype=np.int64)
    distance = shortest_paths(n, edges)
    if channel == "semantic":
        source = int(plan["semantic"][source_index]["source"])
        real = np.repeat(distance[:, source][None, :], event_count, axis=0)
    elif channel == "pe":
        item = plan["pe"][source_index]
        partners = np.asarray(item["partners"], dtype=np.int64)
        if len(partners) != event_count:
            raise ValueError(
                f"PE plan has {len(partners)} events but q has {event_count}"
            )
        source = int(item["source"])
        real = np.stack([
            np.minimum(distance[:, source], distance[:, int(partner)])
            for partner in partners
        ])
    elif channel == "topology":
        if source_index != 0:
            raise ValueError("topology is a single whole-graph source")
        items = plan["topology"]
        if len(items) != event_count:
            raise ValueError(
                f"topology plan has {len(items)} events but q has {event_count}"
            )
        real_rows = []
        for item in items:
            changed = np.asarray(item["dose"]["changed_nodes"], dtype=np.int64)
            real_rows.append(
                np.min(distance[:, changed], axis=1)
                if len(changed) else np.full(n, np.inf)
            )
        real = np.stack(real_rows)
    else:
        raise ValueError(f"unknown intervention channel {channel!r}")

    encoded = np.full(
        (event_count, carrier_count), DISTANCE_UNREACHABLE, dtype=np.int64
    )
    encoded[:, :n] = np.where(np.isfinite(real), real, DISTANCE_UNREACHABLE).astype(
        np.int64
    )
    if carrier_count > n:
        encoded[:, n:] = DISTANCE_HUB
    return encoded


def _distance_label(code: int) -> str:
    if int(code) == DISTANCE_HUB:
        return "hub"
    if int(code) == DISTANCE_UNREACHABLE:
        return "unreachable"
    return str(int(code))


def _distance_sort_key(code: int) -> tuple[int, int]:
    if int(code) >= 0:
        return 0, int(code)
    if int(code) == DISTANCE_UNREACHABLE:
        return 1, 0
    if int(code) == DISTANCE_HUB:
        return 2, 0
    return 3, int(code)


def distance_resolved_eg_one_graph(
    q_groups: Sequence[Any], distance_groups: Sequence[np.ndarray]
) -> dict[str, Any]:
    """Exact source-balanced distance decomposition of graph-level EG.

    For each source, carrier magnitudes are summed inside each event-specific distance
    bucket, donors/events are averaged, then sources are averaged. Consequently the sum
    over returned buckets is exactly the established graph EG score.  The matching
    event-carrier opportunity is retained for every bucket so a second, explicitly
    support-normalised response-density view can be formed without changing EG.
    """

    import torch

    if not q_groups or len(q_groups) != len(distance_groups):
        raise ValueError("q_groups and distance_groups must be non-empty and aligned")
    source_rows: list[dict[int, Any]] = []
    source_opportunities: list[dict[int, float]] = []
    source_support: list[set[int]] = []
    for q, distances in zip(q_groups, distance_groups):
        if q.dim() != 5:
            raise ValueError(f"expected q=[K,L,H,N,T], got {tuple(q.shape)}")
        encoded = torch.as_tensor(distances, dtype=torch.long, device=q.device)
        if tuple(encoded.shape) != (int(q.shape[0]), int(q.shape[-2])):
            raise ValueError(
                f"distance {tuple(encoded.shape)} is not [events,carriers] for q {tuple(q.shape)}"
            )
        magnitude = torch.linalg.vector_norm(q, dim=-1)  # [K,L,H,N]
        row: dict[int, Any] = {}
        opportunity_row: dict[int, float] = {}
        codes = sorted(set(map(int, encoded.unique().tolist())), key=_distance_sort_key)
        for code in codes:
            mask = (encoded == code)[:, None, None, :].to(magnitude.dtype)
            row[code] = (magnitude * mask).sum(dim=-1).mean(dim=0)
            # Mean number of available carriers per intervention event.  Dividing the
            # exact bucket contribution by this scalar removes shell cardinality while
            # preserving the established donor/source weighting of the EG numerator.
            opportunity_row[code] = float((encoded == code).sum(dim=-1).float().mean())
        source_rows.append(row)
        source_opportunities.append(opportunity_row)
        source_support.append(set(codes))

    all_codes = sorted(
        set().union(*(set(row) for row in source_rows)), key=_distance_sort_key
    )
    template = torch.zeros_like(next(iter(source_rows[0].values())))
    contribution = {
        code: torch.stack([row.get(code, template) for row in source_rows]).mean(dim=0)
        for code in all_codes
    }
    opportunity = {
        code: float(np.mean([row.get(code, 0.0) for row in source_opportunities]))
        for code in all_codes
    }
    density = {
        code: (
            contribution[code] / opportunity[code]
            if opportunity[code] > 0.0 else torch.zeros_like(template)
        )
        for code in all_codes
    }
    total = sum(contribution.values(), torch.zeros_like(template))
    direct = torch.stack([
        torch.linalg.vector_norm(q, dim=-1).sum(dim=-1).mean(dim=0)
        for q in q_groups
    ]).mean(dim=0)
    error = (total - direct).abs()
    return {
        "codes": all_codes,
        "contribution": {code: value.detach().cpu().numpy() for code, value in contribution.items()},
        "opportunity": opportunity,
        "density": {code: value.detach().cpu().numpy() for code, value in density.items()},
        "support": {code: any(code in support for support in source_support) for code in all_codes},
        "total": total.detach().cpu().numpy(),
        "direct": direct.detach().cpu().numpy(),
        "identity_max_abs_error": float(error.max()) if error.numel() else 0.0,
        "sources": len(q_groups),
        "events": int(sum(int(q.shape[0]) for q in q_groups)),
    }


def record_distance_resolved_eg(
    record: Mapping[str, Any], channel: str, *, donor_scope: str = "headline"
) -> dict[str, Any] | None:
    """Recover the exact distance-resolved EG tensor from one cached score graph."""

    import torch

    result = record["channels"][channel]
    if not result["available"]:
        return None
    q_groups = list(result["q_groups"])
    distance_groups = [
        event_distance_matrix(
            record,
            channel,
            source_index,
            event_count=int(q.shape[0]),
            carrier_count=int(q.shape[-2]),
        )
        for source_index, q in enumerate(q_groups)
    ]
    if channel == "topology":
        tiers = np.asarray(
            [item["tier"] for item in record["plan"]["topology"]], dtype=np.int64
        )
        if donor_scope == "headline":
            keep = np.flatnonzero(tiers <= 1)
        elif donor_scope.startswith("tier_"):
            tier = int(donor_scope.rsplit("_", 1)[1])
            keep = np.flatnonzero(tiers == tier)
        elif donor_scope == "all":
            keep = np.arange(len(tiers), dtype=np.int64)
        else:
            raise ValueError(f"unknown topology donor scope {donor_scope!r}")
        if not len(keep):
            return None
        index = torch.as_tensor(keep, dtype=torch.long)
        q_groups = [q_groups[0][index]]
        distance_groups = [distance_groups[0][keep]]
    elif donor_scope != "headline":
        raise ValueError(f"{channel} only defines the headline donor scope")

    decomposed = distance_resolved_eg_one_graph(q_groups, distance_groups)
    if channel == "topology" and donor_scope == "headline":
        reference = record_channel_score(
            record, channel, HEADLINE_AGGREGATION, topology_max_tier=1
        )
        if reference is None:
            return None
    elif channel == "topology" and donor_scope.startswith("tier_"):
        # Tier-specific scores have no pre-existing scalar cache field. Their direct EG
        # definition is retained as the reference for the per-distance accounting audit.
        reference = decomposed["direct"]
    else:
        reference = np.asarray(result["per_graph"][HEADLINE_AGGREGATION], dtype=float)
    difference = np.abs(np.asarray(decomposed["total"], float) - reference)
    allowed = DISTANCE_IDENTITY_ATOL + DISTANCE_IDENTITY_RTOL * np.abs(reference)
    decomposed.update({
        "graph_id": int(record["graph_id"]),
        "channel": channel,
        "donor_scope": donor_scope,
        "reference": reference,
        "reference_identity_max_abs_error": float(np.max(difference)),
        "reference_identity_passed": bool(np.all(difference <= allowed)),
    })
    return decomposed


def distance_specialisation_profile(
    score: Mapping[str, Any],
    channel: str,
    *,
    donor_scope: str = "headline",
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Graph-balanced EG-by-distance tensors, uncertainty and reach summaries."""

    decomposed = [
        item for item in (
            record_distance_resolved_eg(record, channel, donor_scope=donor_scope)
            for record in score["records"]
        ) if item is not None
    ]
    if not decomposed:
        raise RuntimeError(f"no distance-resolved score support for {channel}/{donor_scope}")
    failed = [item for item in decomposed if not item["reference_identity_passed"]]
    if failed:
        worst = max(failed, key=lambda item: item["reference_identity_max_abs_error"])
        raise RuntimeError(
            "distance-resolved EG does not reconstruct the cached graph score: "
            f"channel={channel} graph={worst['graph_id']} "
            f"error={worst['reference_identity_max_abs_error']:.3e}"
        )
    codes = sorted(
        set().union(*(set(item["codes"]) for item in decomposed)), key=_distance_sort_key
    )
    L, H = int(score["L"]), int(score["H"])
    graph_cube = np.zeros((len(decomposed), len(codes), L, H), dtype=np.float64)
    graph_opportunity = np.zeros((len(decomposed), len(codes)), dtype=np.float64)
    support = np.zeros((len(decomposed), len(codes)), dtype=bool)
    references = np.zeros((len(decomposed), L, H), dtype=np.float64)
    for graph_index, item in enumerate(decomposed):
        references[graph_index] = np.asarray(item["reference"], dtype=float)
        for bucket_index, code in enumerate(codes):
            if code in item["contribution"]:
                graph_cube[graph_index, bucket_index] = item["contribution"][code]
                graph_opportunity[graph_index, bucket_index] = float(
                    item["opportunity"].get(code, 0.0)
                )
                support[graph_index, bucket_index] = bool(item["support"].get(code, False))
    contribution = graph_cube.mean(axis=0)
    total = contribution.sum(axis=0)
    fraction = np.divide(
        contribution,
        total[None],
        out=np.zeros_like(contribution),
        where=total[None] > EPS,
    )
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(
        0, len(decomposed), size=(int(bootstrap_samples), len(decomposed))
    )
    draw_means = graph_cube[draw_indices].mean(axis=1)
    draw_totals = draw_means.sum(axis=1)
    draw_fractions = np.divide(
        draw_means,
        draw_totals[:, None],
        out=np.zeros_like(draw_means),
        where=draw_totals[:, None] > EPS,
    )
    graph_density = np.divide(
        graph_cube,
        graph_opportunity[:, :, None, None],
        out=np.full_like(graph_cube, np.nan),
        where=graph_opportunity[:, :, None, None] > 0,
    )

    def finite_mean(value: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
        count = np.isfinite(value).sum(axis=axis)
        total_value = np.nansum(value, axis=axis)
        return np.divide(
            total_value,
            count,
            out=np.full_like(total_value, np.nan, dtype=float),
            where=count > 0,
        )

    density = finite_mean(graph_density, axis=0)
    density_total = np.nansum(density, axis=0)
    density_fraction = np.divide(
        density,
        density_total[None],
        out=np.zeros_like(density),
        where=density_total[None] > EPS,
    )
    draw_density = finite_mean(graph_density[draw_indices], axis=1)
    draw_density_total = np.nansum(draw_density, axis=1)
    draw_density_fraction = np.divide(
        draw_density,
        draw_density_total[:, None],
        out=np.zeros_like(draw_density),
        where=draw_density_total[:, None] > EPS,
    )

    aggregate_graph_mass = graph_cube.sum(axis=(2, 3))
    aggregate_mass_mean = aggregate_graph_mass.mean(axis=0)
    aggregate_mass_fraction = aggregate_mass_mean / max(
        float(aggregate_mass_mean.sum()), EPS
    )
    aggregate_mass_draw = aggregate_graph_mass[draw_indices].mean(axis=1)
    aggregate_mass_draw_fraction = np.divide(
        aggregate_mass_draw,
        aggregate_mass_draw.sum(axis=1, keepdims=True),
        out=np.zeros_like(aggregate_mass_draw),
        where=aggregate_mass_draw.sum(axis=1, keepdims=True) > EPS,
    )
    aggregate_graph_density = np.nansum(graph_density, axis=(2, 3))
    aggregate_graph_density[graph_opportunity <= 0] = np.nan
    aggregate_density_mean = finite_mean(aggregate_graph_density, axis=0)
    aggregate_density_fraction = aggregate_density_mean / max(
        float(np.nansum(aggregate_density_mean)), EPS
    )
    aggregate_density_draw = finite_mean(
        aggregate_graph_density[draw_indices], axis=1
    )
    aggregate_density_draw_fraction = np.divide(
        aggregate_density_draw,
        np.nansum(aggregate_density_draw, axis=1, keepdims=True),
        out=np.zeros_like(aggregate_density_draw),
        where=np.nansum(aggregate_density_draw, axis=1, keepdims=True) > EPS,
    )
    numeric = np.asarray([code >= 0 for code in codes], dtype=bool)
    hops = np.asarray([max(code, 0) for code in codes], dtype=float)
    numeric_mass = contribution[numeric].sum(axis=0)
    expected_distance = np.divide(
        (contribution * hops[:, None, None] * numeric[:, None, None]).sum(axis=0),
        numeric_mass,
        out=np.full_like(total, np.nan),
        where=numeric_mass > EPS,
    )
    near_mass = contribution[
        np.asarray([(code >= 0 and code <= 1) for code in codes], dtype=bool)
    ].sum(axis=0)
    far_mass = contribution[
        np.asarray([code >= 2 for code in codes], dtype=bool)
    ].sum(axis=0)
    code_to_index = {code: index for index, code in enumerate(codes)}

    def share_for(code: int) -> np.ndarray:
        value = (
            contribution[code_to_index[code]]
            if code in code_to_index else np.zeros_like(total)
        )
        return np.divide(value, total, out=np.zeros_like(total), where=total > EPS)

    return {
        "channel": channel,
        "donor_scope": donor_scope,
        "codes": np.asarray(codes, dtype=np.int64),
        "labels": [_distance_label(code) for code in codes],
        "graph_ids": np.asarray([item["graph_id"] for item in decomposed], dtype=np.int64),
        "graph_contribution": graph_cube,
        "graph_opportunity": graph_opportunity,
        "mean_opportunity": graph_opportunity.mean(axis=0),
        "graph_density": graph_density,
        "graph_support": support,
        "contribution": contribution,
        "fraction": fraction,
        "fraction_ci_low": np.quantile(draw_fractions, 0.025, axis=0),
        "fraction_ci_high": np.quantile(draw_fractions, 0.975, axis=0),
        "support_normalised_response": density,
        "support_normalised_fraction": density_fraction,
        "support_normalised_fraction_ci_low": np.nanquantile(
            draw_density_fraction, 0.025, axis=0
        ),
        "support_normalised_fraction_ci_high": np.nanquantile(
            draw_density_fraction, 0.975, axis=0
        ),
        "aggregate_mass_fraction": aggregate_mass_fraction,
        "aggregate_mass_fraction_ci_low": np.quantile(
            aggregate_mass_draw_fraction, 0.025, axis=0
        ),
        "aggregate_mass_fraction_ci_high": np.quantile(
            aggregate_mass_draw_fraction, 0.975, axis=0
        ),
        "aggregate_support_normalised_fraction": aggregate_density_fraction,
        "aggregate_support_normalised_fraction_ci_low": np.nanquantile(
            aggregate_density_draw_fraction, 0.025, axis=0
        ),
        "aggregate_support_normalised_fraction_ci_high": np.nanquantile(
            aggregate_density_draw_fraction, 0.975, axis=0
        ),
        "total": total,
        "reference_mean": references.mean(axis=0),
        "expected_distance": expected_distance,
        "near_share": np.divide(near_mass, total, out=np.zeros_like(total), where=total > EPS),
        "far_share": np.divide(far_mass, total, out=np.zeros_like(total), where=total > EPS),
        "unreachable_share": share_for(DISTANCE_UNREACHABLE),
        "hub_share": share_for(DISTANCE_HUB),
        "identity_max_abs_error": float(max(
            item["reference_identity_max_abs_error"] for item in decomposed
        )),
        "identity_passed": True,
    }


def clean_attention_distance_record(
    gm: Any, data: Any, graph_id: int
) -> dict[str, Any]:
    """Clean post-softmax attention mass by pristine molecular query-key distance."""

    import torch
    from torch_geometric.data import Batch

    batch = Batch.from_data_list([data.clone()]).to(gm.device)
    capture = gm.capture(batch, want_grad=False, want_attn=True)
    edge_index = capture["edge_index"].detach().cpu().numpy().astype(np.int64)
    n = int(data.num_nodes)
    molecular_distance = shortest_paths(n, molecular_edges(data))
    source, destination = edge_index
    code = np.full(len(source), DISTANCE_HUB, dtype=np.int64)
    real = (source >= 0) & (source < n) & (destination >= 0) & (destination < n)
    real_distance = molecular_distance[destination[real], source[real]]
    code[real] = np.where(
        np.isfinite(real_distance), real_distance, DISTANCE_UNREACHABLE
    ).astype(np.int64)
    codes = sorted(set(map(int, code.tolist())), key=_distance_sort_key)
    mass = {value: np.zeros((gm.L, gm.H), dtype=np.float64) for value in codes}
    pair_count = {value: int(np.sum(code == value)) for value in codes}
    normalisation_error = np.zeros((gm.L,), dtype=np.float64)
    minimum_weight = np.zeros((gm.L,), dtype=np.float64)
    for layer, tensor in enumerate(capture["attn"]):
        attention = tensor.detach().cpu().numpy().astype(np.float64)
        if tuple(attention.shape) != (len(source), int(gm.H)):
            raise RuntimeError(
                f"attention shape {attention.shape} != {(len(source), int(gm.H))}"
            )
        minimum_weight[layer] = float(np.min(attention)) if attention.size else 0.0
        if minimum_weight[layer] < -1.0e-7:
            raise RuntimeError(
                f"post-softmax attention has negative weight {minimum_weight[layer]:.3e}"
            )
        attention = np.maximum(attention, 0.0)
        for value in codes:
            mass[value][layer] = attention[code == value].sum(axis=0)

        # GRIT normalises incoming sender weights for each receiver and head.
        incoming = np.zeros((n, gm.H), dtype=np.float64)
        np.add.at(incoming, destination[real], attention[real])
        receiver_support = np.bincount(destination[real], minlength=n) > 0
        normalisation_error[layer] = (
            float(np.max(np.abs(incoming[receiver_support] - 1.0)))
            if receiver_support.any() else 0.0
        )
    if float(np.max(normalisation_error)) > 2.0e-4:
        raise RuntimeError(
            "clean attention is not receiver-normalised: "
            f"max error={float(np.max(normalisation_error)):.3e}"
        )
    return {
        "graph_id": int(graph_id),
        "n": n,
        "codes": codes,
        "mass": mass,
        "pair_count": pair_count,
        "normalisation_error": normalisation_error,
        "minimum_weight": minimum_weight,
    }


def attention_distance_profile(
    payload: Mapping[str, Any],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Graph-balanced clean-attention locality with per-head bootstrap uncertainty."""

    records = list(payload["records"])
    if not records:
        raise RuntimeError("attention-distance cache contains no records")
    codes = sorted(
        set().union(*(set(map(int, row["codes"])) for row in records)),
        key=_distance_sort_key,
    )
    first_mass = next(iter(records[0]["mass"].values()))
    L, H = map(int, np.asarray(first_mass).shape)
    graph_mass = np.zeros((len(records), len(codes), L, H), dtype=np.float64)
    graph_pairs = np.zeros((len(records), len(codes)), dtype=np.int64)
    for graph_index, record in enumerate(records):
        for bucket_index, code in enumerate(codes):
            if code in record["mass"]:
                graph_mass[graph_index, bucket_index] = np.asarray(
                    record["mass"][code], dtype=float
                )
                graph_pairs[graph_index, bucket_index] = int(
                    record["pair_count"].get(code, 0)
                )
    graph_total = graph_mass.sum(axis=1)
    graph_fraction = np.divide(
        graph_mass,
        graph_total[:, None],
        out=np.zeros_like(graph_mass),
        where=graph_total[:, None] > EPS,
    )
    fraction = graph_fraction.mean(axis=0)
    aggregate_graph_fraction = graph_fraction.mean(axis=(2, 3))
    aggregate_fraction = aggregate_graph_fraction.mean(axis=0)
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(
        0, len(records), size=(int(bootstrap_samples), len(records))
    )
    draw_fraction = graph_fraction[draw_indices].mean(axis=1)
    aggregate_draw = aggregate_graph_fraction[draw_indices].mean(axis=1)
    return {
        "codes": np.asarray(codes, dtype=np.int64),
        "labels": [_distance_label(code) for code in codes],
        "graph_ids": np.asarray([row["graph_id"] for row in records], dtype=np.int64),
        "graph_mass": graph_mass,
        "graph_pair_count": graph_pairs,
        "fraction": fraction,
        "fraction_ci_low": np.quantile(draw_fraction, 0.025, axis=0),
        "fraction_ci_high": np.quantile(draw_fraction, 0.975, axis=0),
        "aggregate_fraction": aggregate_fraction,
        "aggregate_fraction_ci_low": np.quantile(aggregate_draw, 0.025, axis=0),
        "aggregate_fraction_ci_high": np.quantile(aggregate_draw, 0.975, axis=0),
        "mean_pair_count": graph_pairs.mean(axis=0),
        "graph_support": (graph_pairs > 0).sum(axis=0),
        "normalisation_error_max": float(max(
            np.max(np.asarray(row["normalisation_error"], dtype=float))
            for row in records
        )),
        "minimum_weight": float(min(
            np.min(np.asarray(row["minimum_weight"], dtype=float))
            for row in records
        )),
    }


def score_graph_channel(
    gm: Any,
    plan: Mapping[str, Any],
    channel: str,
    phi: Sequence[Any],
    final_output_gradient: Any,
    cfg: BetaConfig,
    *,
    task_name: str | None = None,
    checkpoint_sha: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    import torch

    base = gm.eval_ds[int(plan["graph_id"])]
    groups = graph_event_variants(gm, plan, channel)
    if not groups or not any(groups):
        return {"available": False, "channel": channel, "graph_id": int(plan["graph_id"])}
    q_groups = []
    prediction_groups = []
    final_q_groups = []
    benefit_groups = []
    benefit_diagnostics = []
    source_indices = []
    for source_index, variants in enumerate(groups):
        if not variants:
            continue
        group_path = None
        group_chunk = None
        if task_name is not None and checkpoint_sha is not None:
            group_path = score_component_cache_path(
                cfg,
                task_name,
                int(plan["graph_id"]),
                f"{channel}_source_{source_index}",
                checkpoint_sha,
            )
            group_chunk = None if force else valid_cache(
                group_path, cfg, checkpoint_sha
            )
        if group_chunk is not None:
            group = group_chunk["group"]
            q = group["q"]
            predictions = group["predictions"]
            q_final = group["q_final"]
            benefit = group["benefit"]
            diagnostic = group["diagnostic"]
            print(
                f"[scores:{task_name}] graph {int(plan['graph_id'])} {channel} "
                f"source {source_index + 1}/{len(groups)} cache hit",
                flush=True,
            )
        else:
            q, predictions, _no_op, final_states = projected_event_group(
                gm, base, variants, phi
            )
            if not q.numel():
                continue
            q_final, benefit, diagnostic = final_state_carriage_group(
                gm,
                final_states,
                final_output_gradient,
                base.y,
                cfg,
                full_predictions=predictions,
            )
            if group_path is not None and checkpoint_sha is not None:
                atomic_torch_save({
                    "version": BETA_VERSION,
                    "schema": BETA_SCHEMA,
                    "fingerprint": cfg.fingerprint,
                    "checkpoint_sha256": checkpoint_sha,
                    "task": task_name,
                    "graph_id": int(plan["graph_id"]),
                    "channel": channel,
                    "source_index": source_index,
                    "group": {
                        "q": q,
                        "predictions": predictions,
                        "q_final": q_final,
                        "benefit": benefit,
                        "diagnostic": diagnostic,
                    },
                }, group_path)
        source_indices.append(source_index)
        q_groups.append(q)
        prediction_groups.append(predictions)
        final_q_groups.append(q_final)
        benefit_groups.append(benefit)
        benefit_diagnostics.append(diagnostic)
    if not q_groups:
        return {"available": False, "channel": channel, "graph_id": int(plan["graph_id"])}
    max_events = max(int(q.shape[0]) for q in q_groups)
    n = int(base.num_nodes)
    n_carriers = int(q_groups[0].shape[-2])
    q_all = torch.zeros(
        len(q_groups), max_events, gm.L, gm.H, n_carriers, len(phi[0])
    )
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
    model_carriage_rows = []
    for group_index, (source_index, q) in enumerate(zip(source_indices, q_groups)):
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
        subsets: list[tuple[str, np.ndarray]] = [
            ("headline", np.arange(len(q), dtype=np.int64))
        ]
        if channel == "topology":
            tiers = np.asarray([item["tier"] for item in plan["topology"]], dtype=np.int64)
            subsets = [("headline", np.flatnonzero(tiers <= 1))]
            subsets.extend(
                (f"tier_{int(tier)}", np.flatnonzero(tiers == int(tier)))
                for tier in sorted(set(tiers.tolist()))
            )
        for donor_scope, event_indices in subsets:
            if not len(event_indices):
                continue
            q_subset = q[event_indices]
            distance_subset = event_distances[event_indices]
            for row in distance_profile_one_graph(q_subset, distance_subset):
                carriage_rows.append({
                    "source_index": int(source_index),
                    "donor_scope": donor_scope,
                    **row,
                })
            q_final = final_q_groups[group_index][event_indices, None, None, :, :]
            benefit_subset = benefit_groups[group_index][event_indices]
            for row in distance_profile_one_graph(
                q_final, distance_subset, beneficial=benefit_subset
            ):
                model_carriage_rows.append({
                    "source_index": int(source_index),
                    "donor_scope": donor_scope,
                    **row,
                    "F_sens": float(np.asarray(row["F_sens"]).reshape(-1)[0]),
                    "F_coh": float(np.asarray(row["F_coh"]).reshape(-1)[0]),
                })

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
        "model_carriage": model_carriage_rows,
        "beneficial_diagnostics": benefit_diagnostics,
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
    semantic_q, semantic_predictions, _, _ = projected_event_group(
        gm, base, [semantic_noop], phi
    )
    pe_q, pe_predictions, _, _ = projected_event_group(gm, base, [pe_noop], phi)
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
        donor_descriptor = graph_descriptor(
            donor, graph_id=int(item["donor_graph_id"])
        )
        if not nonisomorphic(plan["descriptor"], donor_descriptor):
            raise RuntimeError("topology intervention donor is isomorphic to the base graph")
        permutation = torch.as_tensor(item["alignment"], dtype=torch.long)
        for field in STRUCTURAL_NODE_FIELDS:
            donor_value, observed = getattr(donor, field, None), getattr(variant, field, None)
            if donor_value is None:
                continue
            expected = (
                donor_value[permutation.to(donor_value.device)]
                if donor_value.dim() >= 1 and int(donor_value.shape[0]) == int(base.num_nodes)
                else donor_value
            )
            if observed is None or not torch.equal(observed, expected):
                raise RuntimeError(f"topology donor field {field} was not transplanted exactly")
        for index_name, value_name in STRUCTURAL_PAIR_FIELDS:
            donor_index = getattr(donor, index_name, None)
            if donor_index is None:
                continue
            expected_index = relabel_pair_index(
                donor_index, np.asarray(item["alignment"], dtype=np.int64)
            )
            if not torch.equal(getattr(variant, index_name), expected_index):
                raise RuntimeError(f"topology donor index {index_name} was not relabelled exactly")
            if value_name is not None and getattr(donor, value_name, None) is not None:
                if not torch.equal(getattr(variant, value_name), getattr(donor, value_name)):
                    raise RuntimeError(f"topology donor values {value_name} were not copied exactly")
        if float(item["dose"]["edge_jaccard_distance"]) <= 0.0:
            raise RuntimeError("non-isomorphic topology donor has zero molecular edit dose")
        checks["topology_content_target_fixed"] = True
        checks["topology_content_alignment_exact"] = bool(item["content_alignment_exact"])
        checks["topology_support_exact"] = True
        checks["topology_nonisomorphic"] = True
        checks["topology_structural_payload_exact"] = True
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
        if chunk is None and not force:
            legacy_path = cfg.root / "cache" / task_name / "score_graphs" / (
                f"graph_{graph_id}__{cfg.legacy_strict_score_fingerprint}__"
                f"{checkpoint_sha[:12]}.pt"
            )
            legacy = valid_legacy_strict_graph_cache(
                legacy_path, cfg, checkpoint_sha
            )
            if legacy is not None:
                chunk = {
                    **legacy,
                    "fingerprint": cfg.fingerprint,
                    "migrated_from_fingerprint": cfg.legacy_strict_score_fingerprint,
                }
                atomic_torch_save(chunk, chunk_path)
                print(
                    f"[scores:{task_name}] graph {position + 1}/{len(splits['score'])} "
                    f"id={graph_id} migrated tighter legacy cache",
                    flush=True,
                )
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
        clean_prediction, phi, clean_capture, final_output_gradient = clean_gradients(gm, base)
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
        channel_records = {}
        for channel in CHANNELS:
            channel_path = score_component_cache_path(
                cfg, task_name, graph_id, f"{channel}_complete", checkpoint_sha
            )
            channel_chunk = None if force else valid_cache(
                channel_path, cfg, checkpoint_sha
            )
            if channel_chunk is not None:
                channel_records[channel] = channel_chunk["result"]
                print(
                    f"[scores:{task_name}] graph {graph_id} {channel} channel cache hit",
                    flush=True,
                )
                continue
            channel_records[channel] = score_graph_channel(
                gm,
                plan,
                channel,
                phi,
                final_output_gradient,
                cfg,
                task_name=task_name,
                checkpoint_sha=checkpoint_sha,
                force=force,
            )
            atomic_torch_save({
                "version": BETA_VERSION,
                "schema": BETA_SCHEMA,
                "fingerprint": cfg.fingerprint,
                "checkpoint_sha256": checkpoint_sha,
                "task": task_name,
                "graph_id": graph_id,
                "channel": channel,
                "result": channel_records[channel],
            }, channel_path)
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
        progress_audit = integrated_carriage_audit(records)
        write_json(
            cfg.root / "cache" / task_name / "scores_progress.json",
            {
                "version": BETA_VERSION,
                "schema": BETA_SCHEMA,
                "fingerprint": cfg.fingerprint,
                "checkpoint_sha256": checkpoint_sha,
                "completed_graphs": len(records),
                "total_graphs": len(splits["score"]),
                "last_graph_id": graph_id,
                "integrated_carriage": progress_audit,
            },
        )
        del phi, clean_capture, final_output_gradient
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    integrated_audit = integrated_carriage_audit(records)
    write_json(
        cfg.root / "cache" / task_name / "scores_progress.json",
        {
            "version": BETA_VERSION,
            "schema": BETA_SCHEMA,
            "fingerprint": cfg.fingerprint,
            "checkpoint_sha256": checkpoint_sha,
            "completed_graphs": len(records),
            "total_graphs": len(splits["score"]),
            "integrated_carriage": integrated_audit,
        },
    )
    if integrated_audit["unconverged_fraction"] > cfg.integrated_max_unconverged_fraction:
        raise RuntimeError(
            "Integrated beneficial carriage retained "
            f"{integrated_audit['unconverged']}/{integrated_audit['paths']} capped paths "
            f"({100 * integrated_audit['unconverged_fraction']:.4f}%), exceeding "
            "integrated_max_unconverged_fraction="
            f"{100 * cfg.integrated_max_unconverged_fraction:.4f}%. All completed graph, "
            "channel and source caches were preserved for inspection."
        )
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
        "integrated_carriage_audit": integrated_audit,
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


def record_channel_score(
    record: Mapping[str, Any], channel: str, aggregation: str, *, topology_max_tier: int | None = 1
) -> np.ndarray | None:
    result = record["channels"][channel]
    if not result["available"]:
        return None
    if channel != "topology" or topology_max_tier is None:
        return np.asarray(result["per_graph"][aggregation], dtype=float)
    import torch

    tiers = np.asarray([item["tier"] for item in record["plan"]["topology"]])
    keep = np.flatnonzero(tiers <= int(topology_max_tier))
    if not keep.size:
        return None
    q = result["q_groups"][0][torch.as_tensor(keep, dtype=torch.long)]
    valid = torch.ones(1, len(q), dtype=torch.bool)
    subset = aggregate_topology_events(q[None], valid)
    return subset["per_graph"][aggregation][0].numpy()


def paired_graph_scores(
    score_payload: Mapping[str, Any],
    left: str,
    right: str,
    aggregation: str,
    *,
    topology_max_tier: int | None = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Graph-aligned channel scores; required for paired graph bootstraps."""

    left_values, right_values, graph_ids = [], [], []
    for record in score_payload["records"]:
        lval = record_channel_score(
            record, left, aggregation, topology_max_tier=topology_max_tier
        )
        rval = record_channel_score(
            record, right, aggregation, topology_max_tier=topology_max_tier
        )
        if lval is None or rval is None:
            continue
        left_values.append(lval); right_values.append(rval)
        graph_ids.append(int(record["graph_id"]))
    shape = (0, int(score_payload["L"]), int(score_payload["H"]))
    return (
        np.stack(left_values) if left_values else np.empty(shape),
        np.stack(right_values) if right_values else np.empty(shape),
        np.asarray(graph_ids, dtype=np.int64),
    )


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


@contextlib.contextmanager
def patched_family(
    gm: Any,
    heads: Sequence[tuple[int, int]],
    sources: Mapping[int, Any] | None,
):
    """Patch or zero a frozen multi-layer head family simultaneously."""

    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(int(layer), []).append(int(head))
    handles = []
    for layer, layer_heads in by_layer.items():
        source = None if sources is None else sources[layer]

        def hook(
            _module: Any,
            _inputs: Any,
            output: Any,
            *,
            selected: tuple[int, ...] = tuple(layer_heads),
            replacement: Any | None = source,
        ) -> Any:
            h_out, e_out = output
            changed = h_out.clone()
            if replacement is None:
                changed[:, selected, :] = 0.0
            else:
                if tuple(replacement.shape) != tuple(h_out.shape):
                    raise RuntimeError(
                        f"family patch tensor {tuple(replacement.shape)} != {tuple(h_out.shape)}"
                    )
                changed[:, selected, :] = replacement[:, selected, :]
            return changed, e_out

        handles.append(gm.attn_layers[layer].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
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


def forward_data_list_family(
    gm: Any,
    data_list: Sequence[Any],
    heads: Sequence[tuple[int, int]],
    sources: Mapping[int, Any] | None,
) -> np.ndarray:
    import torch

    batch = make_grit_batch(data_list, gm.device)
    with patched_family(gm, heads, sources), torch.no_grad():
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
        # Causal validation uses the best available donor from each tier. Score estimation
        # retains all planned donors; this keeps tier/dose coverage without duplicating the
        # very expensive all-head patch sweep within a molecule/tier.
        causal_topology_items = []
        seen_topology_tiers: set[int] = set()
        for topology_index, item in enumerate(plan["topology"]):
            tier = int(item["tier"])
            if tier in seen_topology_tiers:
                continue
            seen_topology_tiers.add(tier)
            causal_topology_items.append((topology_index, item))
        for topology_index, item in causal_topology_items:
            events.append({
                "graph_id": int(graph_id),
                "channel": "topology",
                "source": -1,
                "event_index": int(topology_index),
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
    loaded: Mapping[str, Any],
    cfg: BetaConfig,
    score_payload: Mapping[str, Any],
    *,
    force: bool = False,
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
    frozen_families, _family_diagnostics = select_families(
        score_payload, cfg, HEADLINE_AGGREGATION
    )
    family_names = [
        name for name in PATCH_FAMILY_NAMES if frozen_families.get(name)
    ]
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
    family_metrics = {
        name: np.full((event_count, len(family_names)), np.nan, dtype=np.float32)
        for name in (
            "restore", "inject", "necessity", "mismatch", "sham",
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
        family_metrics = {
            key: np.asarray(value) for key, value in progress["family_metrics"].items()
        }
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

        # Frozen families are patched as units, simultaneously across all selected layers.
        clean_sources = {
            layer: concatenate_layer_transport(
                events, indices, layer=layer, kind="clean_wv", device=gm.device
            )
            for layer in range(gm.L)
        }
        corrupt_sources = {
            layer: concatenate_layer_transport(
                events, indices, layer=layer, kind="corrupt_wv", device=gm.device
            )
            for layer in range(gm.L)
        }
        mismatch_sources, mismatch_valid_by_layer = {}, []
        for layer in range(gm.L):
            source, valid = concatenate_mismatch_transport(
                events, indices, layer=layer, device=gm.device
            )
            mismatch_sources[layer] = source
            mismatch_valid_by_layer.append(valid)
        mismatch_valid = np.logical_and.reduce(mismatch_valid_by_layer)
        for family_index, family in enumerate(family_names):
            heads = frozen_families[family]
            restore_sources = {
                layer: torch.cat([clean_sources[layer], corrupt_sources[layer]], dim=0)
                for layer in range(gm.L)
            }
            restore_inject = forward_data_list_family(
                gm, [*corrupt_data, *clean_data], heads, restore_sources
            )
            restored = restore_inject[:len(indices)]
            injected = restore_inject[len(indices):]
            ablated = forward_data_list_family(
                gm, [*clean_data, *corrupt_data], heads, None
            )
            ablated_clean = ablated[:len(indices)]
            ablated_corrupt = ablated[len(indices):]
            mismatch_sources_both = {
                layer: torch.cat([mismatch_sources[layer], corrupt_sources[layer]], dim=0)
                for layer in range(gm.L)
            }
            mismatch_sham = forward_data_list_family(
                gm, [*corrupt_data, *corrupt_data], heads, mismatch_sources_both
            )
            movements = {
                "restore": restored - corrupt,
                "inject": clean - injected,
                "necessity": delta - (ablated_clean - ablated_corrupt),
                "mismatch": mismatch_sham[:len(indices)] - corrupt,
                "sham": mismatch_sham[len(indices):] - corrupt,
            }
            for name, movement in movements.items():
                desired = np.mean(movement * direction, axis=1)
                if name == "mismatch":
                    desired = np.where(mismatch_valid, desired, np.nan)
                family_metrics[name][indices, family_index] = desired
            clean_loss = np.abs(clean - target).mean(axis=1)
            corrupt_loss = np.abs(corrupt - target).mean(axis=1)
            family_metrics["restore_loss"][indices, family_index] = (
                corrupt_loss - np.abs(restored - target).mean(axis=1)
            )
            family_metrics["inject_loss"][indices, family_index] = (
                np.abs(injected - target).mean(axis=1) - clean_loss
            )
            family_metrics["necessity_loss"][indices, family_index] = (
                (corrupt_loss - clean_loss)
                - (
                    np.abs(ablated_corrupt - target).mean(axis=1)
                    - np.abs(ablated_clean - target).mean(axis=1)
                )
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
            "family_metrics": family_metrics,
            "family_names": family_names,
            "family_definitions": frozen_families,
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
        "family_metrics": family_metrics,
        "family_names": family_names,
        "family_definitions": frozen_families,
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


def causal_pair_coordinates(
    causal_payload: Mapping[str, Any],
    left_channel: str,
    right_channel: str,
    metric: str = "restore",
) -> dict[str, np.ndarray]:
    left = channel_mean_metric(causal_payload, left_channel, metric)
    right = channel_mean_metric(causal_payload, right_channel, metric)
    total_magnitude = np.abs(left) + np.abs(right)
    return {
        "semantic": left,
        "structural": right,
        "left": left,
        "right": right,
        "D": (left - right) / (total_magnitude + EPS),
        "J": 0.5 * total_magnitude,
        "G": np.sqrt(np.maximum(left, 0.0) * np.maximum(right, 0.0)),
        "semantic_anti_aligned": left < 0.0,
        "structural_anti_aligned": right < 0.0,
        "left_anti_aligned": left < 0.0,
        "right_anti_aligned": right < 0.0,
    }


def causal_coordinates(causal_payload: Mapping[str, Any], metric: str = "restore") -> dict[str, np.ndarray]:
    return causal_pair_coordinates(causal_payload, "semantic", "pe", metric)


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

    pe_graphs_paired, topology_graphs_paired, _ = paired_graph_scores(
        score_payload, "pe", "topology", aggregation
    )
    if len(pe_graphs_paired):
        pe_topology_pe = pe_graphs_paired.mean(axis=0)
        pe_topology_topology = topology_graphs_paired.mean(axis=0)
        pe_topology_coordinates = score_coordinates(
            pe_topology_pe, pe_topology_topology
        )
        pe_topology_lower, pe_topology_upper = bootstrap_d_intervals(
            pe_graphs_paired,
            topology_graphs_paired,
            samples=cfg.bootstrap_samples,
            seed=cfg.analysis_seed + 8_911,
        )
    else:
        pe_topology_pe = np.full_like(structural, np.nan)
        pe_topology_topology = np.full_like(structural, np.nan)
        pe_topology_coordinates = score_coordinates(
            pe_topology_pe, pe_topology_topology
        )
        pe_topology_lower = np.full_like(structural, np.nan)
        pe_topology_upper = np.full_like(structural, np.nan)
    topology = mean_score(score_payload, "topology", aggregation)
    semantic_topology_semantic_graphs, semantic_topology_graphs, _ = paired_graph_scores(
        score_payload, "semantic", "topology", aggregation
    )
    if len(semantic_topology_semantic_graphs):
        semantic_topology_semantic = semantic_topology_semantic_graphs.mean(axis=0)
        semantic_topology_topology = semantic_topology_graphs.mean(axis=0)
    else:
        semantic_topology_semantic = np.full_like(semantic, np.nan)
        semantic_topology_topology = np.full_like(semantic, np.nan)
    if np.isfinite(pe_topology_coordinates["J"]).any():
        pe_topology_floor = max(
            cfg.activity_floor_relative * float(np.nanmax(pe_topology_coordinates["J"])),
            EPS,
        )
        pe_topology_active = pe_topology_coordinates["J"] >= pe_topology_floor
    else:
        pe_topology_floor = float("nan")
        pe_topology_active = np.zeros_like(pe_topology_coordinates["J"], dtype=bool)
    pe_specific_mask = (
        pe_topology_active
        & (pe_topology_coordinates["D"] >= cfg.selectivity_threshold)
        & (pe_topology_lower > 0.0)
    )
    topology_specific_mask = (
        pe_topology_active
        & (pe_topology_coordinates["D"] <= -cfg.selectivity_threshold)
        & (pe_topology_upper < 0.0)
    )
    pe_topology_shared_mask = (
        pe_topology_active & (pe_topology_lower <= 0.0) & (pe_topology_upper >= 0.0)
    )
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
        "structural_pe_specific": take(
            head_order(pe_topology_coordinates["D"]), pe_specific_mask
        ),
        "structural_topology_specific": take(
            head_order(-pe_topology_coordinates["D"]), topology_specific_mask
        ),
        "structural_shared": take(
            head_order(pe_topology_coordinates["G"]), pe_topology_shared_mask
        ),
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
        "pe_topology_coordinates": pe_topology_coordinates,
        "pe_topology_pe": pe_topology_pe,
        "pe_topology_topology": pe_topology_topology,
        "semantic_topology_semantic": semantic_topology_semantic,
        "semantic_topology_topology": semantic_topology_topology,
        "pe_topology_D_ci_lower": pe_topology_lower,
        "pe_topology_D_ci_upper": pe_topology_upper,
        "pe_topology_activity_floor": pe_topology_floor,
        "pe_topology_active": pe_topology_active,
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
    progress_path = cache_path(cfg, task_name, "ablations_progress", checkpoint_sha)
    progress = None if force else valid_cache(progress_path, cfg, checkpoint_sha)
    per_head = {
        "functional": np.zeros((gm.L, gm.H), dtype=np.float32),
        "loss_increase": np.zeros((gm.L, gm.H), dtype=np.float32),
        "mae": np.zeros((gm.L, gm.H), dtype=np.float32),
    }
    completed_heads: set[tuple[int, int]] = set()
    family_curves: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if progress is not None:
        if tuple(progress.get("head_shape", ())) != (gm.L, gm.H):
            raise RuntimeError("ablation progress head shape changed")
        cached_per_head = progress.get("per_head", {})
        if set(cached_per_head) != set(per_head):
            raise RuntimeError("ablation progress metric schema changed")
        per_head = {key: np.asarray(value) for key, value in cached_per_head.items()}
        completed_heads = {
            (int(value[0]), int(value[1]))
            for value in progress.get("completed_heads", [])
        }
        family_curves = {
            str(aggregation): {
                str(family): list(rows) for family, rows in curves.items()
            }
            for aggregation, curves in progress.get("family_curves", {}).items()
        }

    def save_ablation_progress() -> None:
        atomic_torch_save({
            "version": BETA_VERSION,
            "schema": BETA_SCHEMA,
            "fingerprint": cfg.fingerprint,
            "checkpoint_sha256": checkpoint_sha,
            "task": task_name,
            "head_shape": (gm.L, gm.H),
            "per_head": per_head,
            "completed_heads": sorted(completed_heads),
            "family_curves": family_curves,
        }, progress_path)

    for layer in range(gm.L):
        for head in range(gm.H):
            if (layer, head) in completed_heads:
                continue
            prediction = gm.collect_preds_ablated(groups, [(layer, head)])
            summary = ablation_summary(prediction, clean, target)
            for key in per_head:
                per_head[key][layer, head] = summary[key]
            completed_heads.add((layer, head))
        # One layer is the natural resumable unit and avoids excessive Drive writes.
        save_ablation_progress()
        print(f"[ablations:{task_name}] layer {layer + 1}/{gm.L}", flush=True)

    # EG is predeclared from the synthetic/ZINC redesign decision; diagnostics for the other
    # aggregations remain cheap audit rows but cannot change the headline analysis.
    winner = HEADLINE_AGGREGATION
    method_rows = aggregation_diagnostics(score_payload, causal_payload)
    for aggregation in AGGREGATIONS:
        reliability = prefix_reliability(score_payload, aggregation)
        method_rows.append({
            "aggregation": aggregation,
            "scope": "donor_convergence_audit",
            "prefix_rank_rho": reliability["rank_rho"],
            "prefix_top3_jaccard": reliability["top3_jaccard"],
            "selected": aggregation == HEADLINE_AGGREGATION,
        })
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
    for aggregation in (HEADLINE_AGGREGATION,):
        families, diagnostics = select_families(score_payload, cfg, aggregation)
        selections[aggregation] = diagnostics
        family_curves.setdefault(aggregation, {})
        for family, heads in families.items():
            rows = list(family_curves[aggregation].get(family, []))
            rows_by_count = {int(row["count"]): row for row in rows}
            for count in range(1, len(heads) + 1):
                if count in rows_by_count:
                    continue
                prediction = gm.collect_preds_ablated(groups, heads[:count])
                row = {"count": count, **ablation_summary(prediction, clean, target)}
                rows_by_count[count] = row
                family_curves[aggregation][family] = [
                    rows_by_count[index] for index in sorted(rows_by_count)
                ]
            family_curves[aggregation][family] = [
                rows_by_count[index] for index in sorted(rows_by_count)
            ]
            # Preserve a completed ranked family without writing after every prefix.
            save_ablation_progress()
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


def run_attention_task(
    loaded: Mapping[str, Any],
    cfg: BetaConfig,
    score_payload: Mapping[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Cache clean attention-distance mass on exactly the score graphs."""

    import torch

    gm, checkpoint_sha = loaded["gm"], loaded["sha"]
    task_name = loaded["spec"].name
    path = cache_path(cfg, task_name, "attention", checkpoint_sha)
    cached = None if force else valid_cache(path, cfg, checkpoint_sha)
    if cached is not None:
        print(f"[attention:{task_name}] cache hit {path}", flush=True)
        return cached
    records = []
    chunk_root = cfg.root / "cache" / task_name / "attention_graphs"
    chunk_root.mkdir(parents=True, exist_ok=True)
    score_records = list(score_payload["records"])
    for position, score_record in enumerate(score_records):
        graph_id = int(score_record["graph_id"])
        chunk_path = (
            chunk_root
            / f"graph_{graph_id}__{cfg.fingerprint}__{checkpoint_sha[:12]}.pt"
        )
        chunk = None if force else valid_cache(chunk_path, cfg, checkpoint_sha)
        if chunk is None:
            data = gm.eval_ds[graph_id]
            descriptor = score_record["plan"]["descriptor"]
            if int(data.num_nodes) != int(score_record["n"]) or not np.array_equal(
                molecular_edges(data), np.asarray(descriptor["edges"], dtype=np.int64)
            ):
                raise RuntimeError(
                    f"attention graph {graph_id} no longer matches score graph topology"
                )
            record = clean_attention_distance_record(gm, data, graph_id)
            chunk = {
                "version": BETA_VERSION,
                "schema": BETA_SCHEMA,
                "fingerprint": cfg.fingerprint,
                "checkpoint_sha256": checkpoint_sha,
                "task": task_name,
                "graph_id": graph_id,
                "record": record,
            }
            atomic_torch_save(chunk, chunk_path)
            print(
                f"[attention:{task_name}] graph {position + 1}/{len(score_records)} "
                f"id={graph_id}",
                flush=True,
            )
        else:
            print(
                f"[attention:{task_name}] graph {position + 1}/{len(score_records)} "
                f"id={graph_id} cache hit",
                flush=True,
            )
        records.append(chunk["record"])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "version": BETA_VERSION,
        "schema": BETA_SCHEMA,
        "fingerprint": cfg.fingerprint,
        "checkpoint_sha256": checkpoint_sha,
        "task": task_name,
        "graph_ids": np.asarray([row["graph_id"] for row in records], dtype=np.int64),
        "records": records,
        "normalisation_error_max": float(max(
            np.max(np.asarray(row["normalisation_error"], dtype=float))
            for row in records
        )),
        "minimum_weight": float(min(
            np.min(np.asarray(row["minimum_weight"], dtype=float))
            for row in records
        )),
    }
    atomic_torch_save(payload, path)
    return payload


def aggregate_pairs_to_nodes(pair_value: Any, destination: Any, n: int) -> Any:
    import torch

    out = torch.zeros(
        n, pair_value.shape[1], pair_value.shape[2],
        dtype=pair_value.dtype, device=pair_value.device,
    )
    out.index_add_(0, destination.long(), pair_value)
    return out


def decompose_support_aware_layer(clean: Mapping[str, Any], corrupt: Mapping[str, Any]) -> dict[str, Any]:
    """Exact routing/message/wiring split on the common and exclusive supports.

    Routing and message use the symmetric product decomposition on common directed pairs.
    Clean-only minus corrupt-only transported messages form an explicit wiring component.
    This reduces exactly to the two-way decomposition when support is fixed.
    """

    import torch

    n = int(clean["head_output"].shape[0])
    clean_keys = list(zip(clean["src"].cpu().tolist(), clean["dst"].cpu().tolist()))
    corrupt_keys = list(zip(corrupt["src"].cpu().tolist(), corrupt["dst"].cpu().tolist()))
    if len(set(clean_keys)) != len(clean_keys) or len(set(corrupt_keys)) != len(corrupt_keys):
        raise RuntimeError("mechanism support contains duplicate directed pairs")
    clean_pos = {key: index for index, key in enumerate(clean_keys)}
    corrupt_pos = {key: index for index, key in enumerate(corrupt_keys)}
    common = sorted(set(clean_pos) & set(corrupt_pos))
    clean_only = sorted(set(clean_pos) - set(corrupt_pos))
    corrupt_only = sorted(set(corrupt_pos) - set(clean_pos))

    def indices(mapping: Mapping[tuple[int, int], int], keys: Sequence[tuple[int, int]]) -> Any:
        return torch.as_tensor(
            [mapping[key] for key in keys], device=clean["attention"].device, dtype=torch.long
        )

    ci, xi = indices(clean_pos, common), indices(corrupt_pos, common)
    clean_attention = clean["attention"][ci].unsqueeze(-1)
    corrupt_attention = corrupt["attention"][xi].unsqueeze(-1)
    clean_message = clean["message"][ci]
    corrupt_message = corrupt["message"][xi]
    route_pairs = (clean_attention - corrupt_attention) * 0.5 * (clean_message + corrupt_message)
    message_pairs = 0.5 * (clean_attention + corrupt_attention) * (clean_message - corrupt_message)
    common_dst = clean["dst"][ci]
    route_nodes = aggregate_pairs_to_nodes(route_pairs, common_dst, n)
    message_nodes = aggregate_pairs_to_nodes(message_pairs, common_dst, n)

    wiring_nodes = torch.zeros_like(route_nodes)
    if clean_only:
        oi = indices(clean_pos, clean_only)
        wiring_nodes += aggregate_pairs_to_nodes(
            clean["attention"][oi].unsqueeze(-1) * clean["message"][oi], clean["dst"][oi], n
        )
    if corrupt_only:
        oi = indices(corrupt_pos, corrupt_only)
        wiring_nodes -= aggregate_pairs_to_nodes(
            corrupt["attention"][oi].unsqueeze(-1) * corrupt["message"][oi],
            corrupt["dst"][oi], n,
        )
    direct_nodes = clean["head_output"] - corrupt["head_output"]
    reconstruction = direct_nodes - route_nodes - message_nodes - wiring_nodes
    gradient = clean["gradient"]
    direct_q = torch.einsum("nhd,nhd->h", gradient, direct_nodes)
    route_q = torch.einsum("nhd,nhd->h", gradient, route_nodes)
    message_q = torch.einsum("nhd,nhd->h", gradient, message_nodes)
    wiring_q = torch.einsum("nhd,nhd->h", gradient, wiring_nodes)
    return {
        "direct_q": direct_q,
        "routing_q": route_q,
        "message_q": message_q,
        "wiring_q": wiring_q,
        "reconstruction_max": float(reconstruction.abs().max()),
        "routing_hybrid": corrupt["head_output"] + route_nodes,
        "message_hybrid": corrupt["head_output"] + message_nodes,
        "wiring_hybrid": corrupt["head_output"] + wiring_nodes,
        "full_clean": clean["head_output"],
        "common_pairs": len(common),
        "clean_only_pairs": len(clean_only),
        "corrupt_only_pairs": len(corrupt_only),
    }


def decompose_fixed_support_layer(clean: Mapping[str, Any], corrupt: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility alias; now returns the support-aware decomposition."""

    return decompose_support_aware_layer(clean, corrupt)


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
    events = [event for event in catalog if int(event["event_index"]) == 0]
    shape = (len(events), gm.L, gm.H)
    arrays = {
        key: np.full(shape, np.nan, dtype=np.float32)
        for key in (
            "direct_q", "routing_q", "message_q", "wiring_q", "routing_rescue",
            "message_rescue", "wiring_rescue", "full_rescue", "finite_interaction",
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

    def save_mechanism_progress() -> None:
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

    for event_index, event in enumerate(events):
        if event_index in completed_events:
            print(f"[mechanism:{task_name}] event {event_index + 1} cache hit", flush=True)
            continue
        clean = capture_mechanism(gm, event["clean"])
        corrupt = capture_mechanism(gm, event["corrupt"])
        delta = clean["prediction"] - corrupt["prediction"]
        direction = np.sign(delta)
        for layer in range(gm.L):
            decomposition = decompose_support_aware_layer(
                clean["layers"][layer], corrupt["layers"][layer]
            )
            reconstruction[event_index, layer] = decomposition["reconstruction_max"]
            if decomposition["reconstruction_max"] > MECHANISM_ATOL:
                raise RuntimeError(
                    f"routing/message reconstruction failed: {decomposition['reconstruction_max']:.3e}"
                )
            for key in ("direct_q", "routing_q", "message_q", "wiring_q"):
                arrays[key][event_index, layer] = decomposition[key].detach().cpu().numpy()
            for head in range(gm.H):
                names = ("routing_rescue", "message_rescue", "wiring_rescue", "full_rescue")
                sources = (
                    decomposition["routing_hybrid"],
                    decomposition["message_hybrid"],
                    decomposition["wiring_hybrid"],
                    decomposition["full_clean"],
                )
                patched = forward_data_list(
                    gm,
                    [event["corrupt"] for _ in names],
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
                    - predictions["wiring_rescue"]
                )
        print(
            f"[mechanism:{task_name}] event {event_index + 1}/{len(events)} "
            f"channel={event['channel']}", flush=True,
        )
        completed_events.add(event_index)
        save_mechanism_progress()
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
    import networkx as nx

    records = score_payload["records"]
    graph_features = []
    discovery_graphs = {
        int(record["graph_id"]) for index, record in enumerate(records) if index % 2 == 0
    }
    atom_counts: dict[tuple[int, ...], int] = {}
    for record in records:
        descriptor = record["plan"]["descriptor"]
        labels = [tuple(row) for row in np.asarray(descriptor["labels"], dtype=int)]
        degree = np.asarray(descriptor["degree"], dtype=float)
        graph = nx.Graph()
        graph.add_nodes_from(range(int(record["n"])))
        graph.add_edges_from(np.asarray(descriptor["edges"], dtype=int).tolist())
        cycle_nodes = {node for cycle in nx.cycle_basis(graph) for node in cycle}
        articulation = set(nx.articulation_points(graph))
        if int(record["graph_id"]) in discovery_graphs:
            for label in set(labels):
                atom_counts[label] = atom_counts.get(label, 0) + 1
        graph_features.append({
            "graph_id": int(record["graph_id"]),
            "n": int(record["n"]),
            "cycle_rank": int(descriptor["cycle_rank"]),
            "atom_diversity": len(set(labels)),
            "mean_degree": float(np.mean(degree)),
            "edge_density": float(
                len(descriptor["edges"]) / max(int(record["n"]) * (int(record["n"]) - 1) / 2, 1)
            ),
            "degree": degree,
            "cycle_nodes": cycle_nodes,
            "articulation": articulation,
            "edges": np.asarray(descriptor["edges"], dtype=int),
            "labels": labels,
            "distances": shortest_paths(
                int(record["n"]), np.asarray(descriptor["edges"], dtype=int)
            ),
        })
    discovery_features = [row for row in graph_features if row["graph_id"] in discovery_graphs]
    thresholds = {
        name: float(np.median([row[name] for row in discovery_features]))
        for name in ("n", "cycle_rank", "atom_diversity", "mean_degree", "edge_density")
    }
    thresholds["source_degree"] = float(np.median(np.concatenate([
        row["degree"] for row in discovery_features
    ])))
    common_atoms = [
        label for label, _count in sorted(atom_counts.items(), key=lambda item: (-item[1], item[0]))[:6]
    ]
    by_id = {row["graph_id"]: row for row in graph_features}
    observations = []

    def concentration(q: Any) -> np.ndarray:
        event_magnitude = np.linalg.norm(q.numpy(), axis=-1).mean(axis=0)  # [L,H,N]
        probability = event_magnitude / np.maximum(event_magnitude.sum(axis=-1, keepdims=True), EPS)
        entropy = -(probability * np.log(np.maximum(probability, EPS))).sum(axis=-1)
        return 1.0 - entropy / max(math.log(event_magnitude.shape[-1]), EPS)

    def pair_features(left: np.ndarray, right: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
        total = left + right
        return {
            f"D_{prefix}": (left - right) / (total + EPS),
            f"J_{prefix}": 0.5 * total,
            f"G_{prefix}": np.sqrt(np.maximum(left * right, 0.0)),
        }

    for record in records:
        if not all(record["channels"][channel]["available"] for channel in CHANNELS):
            continue
        semantic = np.asarray(record["channels"]["semantic"]["per_source"][aggregation], float)
        pe = np.asarray(record["channels"]["pe"]["per_source"][aggregation], float)
        topology_graph = record_channel_score(
            record, "topology", aggregation, topology_max_tier=1
        )
        if topology_graph is None:
            continue
        topology_graph = np.asarray(topology_graph, float)
        topology = np.broadcast_to(topology_graph, semantic.shape)
        score_features = {
            "S_semantic": semantic,
            "S_pe": pe,
            "S_topology": topology,
            **pair_features(semantic, pe, "semantic_pe"),
            **pair_features(semantic, topology, "semantic_topology"),
            **pair_features(pe, topology, "pe_topology"),
            "J_three_channel": (semantic + pe + topology) / 3.0,
            "G_three_channel": np.cbrt(np.maximum(semantic * pe * topology, 0.0)),
        }
        sources = np.asarray(record["plan"]["sources"], dtype=np.int64)
        feature = by_id[int(record["graph_id"])]
        graph_rules = {
            "graph_size_high": feature["n"] > thresholds["n"],
            "cycle_rank_high": feature["cycle_rank"] > thresholds["cycle_rank"],
            "atom_diversity_high": feature["atom_diversity"] > thresholds["atom_diversity"],
            "mean_degree_high": feature["mean_degree"] > thresholds["mean_degree"],
            "edge_density_high": feature["edge_density"] > thresholds["edge_density"],
        }
        for atom in common_atoms:
            graph_rules[f"graph_contains_atom_{'_'.join(map(str, atom))}"] = atom in feature["labels"]
        for source_index, source in enumerate(sources):
            source_label = tuple(feature["labels"][int(source)])
            rules = dict(graph_rules)
            for atom in common_atoms:
                rules[f"source_atom_{'_'.join(map(str, atom))}"] = source_label == atom
                atom_nodes = [index for index, label in enumerate(feature["labels"]) if label == atom]
                rules[f"source_near_atom_{'_'.join(map(str, atom))}"] = (
                    bool(
                        np.min(feature["distances"][int(source), atom_nodes]) <= 1
                    )
                    if atom_nodes else False
                )
            neighbors = [
                int(v) if int(u) == int(source) else int(u)
                for u, v in feature["edges"] if int(source) in (int(u), int(v))
            ]
            rules["source_degree_high"] = feature["degree"][int(source)] > thresholds["source_degree"]
            rules["source_in_cycle"] = int(source) in feature["cycle_nodes"]
            rules["source_is_articulation"] = int(source) in feature["articulation"]
            rules["source_neighbor_atom_diverse"] = len({feature["labels"][node] for node in neighbors}) >= 2
            topology_items = [
                item for item in record["plan"]["topology"] if int(item["tier"]) <= 1
            ]
            topology_dose = float(np.mean([
                item["dose"]["edge_jaccard_distance"] for item in topology_items
            ])) if topology_items else float("nan")
            topology_tier = float(np.mean([item["tier"] for item in topology_items])) if topology_items else float("nan")
            topology_indices = [
                index for index, item in enumerate(record["plan"]["topology"])
                if int(item["tier"]) <= 1
            ]
            topology_q = record["channels"]["topology"]["q_groups"][0][topology_indices]
            topology_concentration = concentration(topology_q)
            observations.append({
                "graph_id": int(record["graph_id"]),
                "split": "discovery" if int(record["graph_id"]) in discovery_graphs else "confirmation",
                "source": int(source),
                **{key: value[source_index] for key, value in score_features.items()},
                "D": score_features["D_semantic_pe"][source_index],
                "J": score_features["J_semantic_pe"][source_index],
                "semantic_dose": float(np.mean(record["plan"]["semantic"][source_index]["dose"])),
                "pe_dose": float(np.mean(record["plan"]["pe"][source_index]["dose"])),
                "topology_dose": topology_dose,
                "topology_tier": topology_tier,
                "semantic_concentration": concentration(
                    record["channels"]["semantic"]["q_groups"][source_index]
                ),
                "pe_concentration": concentration(
                    record["channels"]["pe"]["q_groups"][source_index]
                ),
                "topology_concentration": topology_concentration,
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
    observations: Sequence[Mapping[str, Any]],
    rule: str,
    head: tuple[int, int],
    feature: str = "D",
) -> float:
    from scipy.stats import ttest_1samp, ttest_ind

    by_state: dict[bool, dict[int, list[float]]] = {False: {}, True: {}}
    for row in observations:
        state = bool(row["rules"][rule])
        by_state[state].setdefault(int(row["graph_id"]), []).append(float(row[feature][head]))
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
    for key in ("semantic_dose", "pe_dose", "topology_dose"):
        values = {}
        for state in (False, True):
            grouped: dict[int, list[float]] = {}
            for row in observations:
                if bool(row["rules"][rule]) == state:
                    grouped.setdefault(int(row["graph_id"]), []).append(float(row[key]))
            values[state] = np.asarray(
                [np.nanmean(grouped[graph_id]) for graph_id in sorted(grouped)], dtype=float
            )
            values[state] = values[state][np.isfinite(values[state])]
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
    feature: str = "D",
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
        effect, _, _ = _conditional_feature_effect(boot, rule, head, feature)
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
    rules_passing_prevalence: set[str] = set()
    for feature_name in CONDITIONAL_SCORE_FEATURES:
        for rule in rules:
            true_fraction = np.mean([bool(row["rules"][rule]) for row in discovery])
            if not cfg.min_condition_fraction <= true_fraction <= 1.0 - cfg.min_condition_fraction:
                continue
            rules_passing_prevalence.add(rule)
            for layer in range(L):
                for head in range(H):
                    effect, true_graphs, false_graphs = _conditional_feature_effect(
                        discovery, rule, (layer, head), feature_name
                    )
                    if min(true_graphs, false_graphs) < cfg.min_condition_graphs:
                        continue
                    activity = np.mean([
                        float(row["J_three_channel"][layer, head]) for row in discovery
                    ])
                    scale = float(np.std([
                        float(item[feature_name][layer, head]) for item in discovery
                    ]))
                    candidates.append({
                        "feature": feature_name,
                        "rule": rule,
                        "layer": layer,
                        "head": head,
                        "discovery_effect": effect,
                        "discovery_scale": scale,
                        "discovery_effect_standardized": effect / max(scale, EPS),
                        "discovery_activity": activity,
                        "true_graphs": true_graphs,
                        "false_graphs": false_graphs,
                    })
    eligibility_audit = {
        "requires_all_three_channels": True,
        "observations": int(len(observations)),
        "observation_graphs": int(len({
            int(row["graph_id"]) for row in observations
        })),
        "discovery_graphs": int(len({
            int(row["graph_id"]) for row in discovery
        })),
        "confirmation_graphs": int(len({
            int(row["graph_id"]) for row in confirmation
        })),
        "rules_considered": int(len(rules)),
        "rules_passing_prevalence": int(len(rules_passing_prevalence)),
        "minimum_condition_fraction": float(cfg.min_condition_fraction),
        "minimum_graphs_per_state": int(cfg.min_condition_graphs),
        "support_eligible_feature_rule_heads": int(len(candidates)),
        "activity_eligible_feature_rule_heads": 0,
        "tested_feature_rule_heads": 0,
        "fdr_confirmed_feature_rule_heads": 0,
    }
    if not candidates:
        return {
            "metadata": metadata,
            "eligibility_audit": eligibility_audit,
            "candidates": [],
            "confirmed": [],
            "interpretation": (
                "No feature/rule/head combination met the predeclared prevalence and "
                "per-state graph-support gates. This is an ineligible screen, not evidence "
                "that conditional specialisation is absent."
            ),
        }
    activity_floor = cfg.activity_floor_relative * max(row["discovery_activity"] for row in candidates)
    eligible = [row for row in candidates if row["discovery_activity"] >= activity_floor]
    eligibility_audit["activity_eligible_feature_rule_heads"] = int(len(eligible))
    eligible.sort(key=lambda row: (
        -abs(row["discovery_effect_standardized"]), row["feature"], row["rule"],
        row["layer"], row["head"],
    ))
    tested = []
    selected_keys: set[tuple[Any, ...]] = set()

    def candidate_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (row["feature"], row["rule"], int(row["layer"]), int(row["head"]))

    # Guarantee coverage of every score coordinate before filling remaining slots globally.
    for feature_name in CONDITIONAL_SCORE_FEATURES:
        feature_rows = [row for row in eligible if row["feature"] == feature_name]
        for row in feature_rows[: cfg.conditional_per_feature]:
            key = candidate_key(row)
            if key not in selected_keys and len(tested) < cfg.conditional_max_tests:
                tested.append(row); selected_keys.add(key)
    for row in eligible:
        key = candidate_key(row)
        if key not in selected_keys and len(tested) < cfg.conditional_max_tests:
            tested.append(row); selected_keys.add(key)
    eligibility_audit["tested_feature_rule_heads"] = int(len(tested))
    confirmed = []
    for index, row in enumerate(tested):
        head = (int(row["layer"]), int(row["head"]))
        effect, true_graphs, false_graphs = _conditional_feature_effect(
            confirmation, row["rule"], head, row["feature"]
        )
        lower, upper = _bootstrap_conditional_effect(
            confirmation,
            row["rule"],
            head,
            feature=row["feature"],
            samples=cfg.conditional_bootstrap_samples,
            seed=cfg.analysis_seed + 70_001 + index,
        )
        confirmed.append({
            **row,
            "confirmation_effect": effect,
            "confirmation_effect_standardized": effect / max(row["discovery_scale"], EPS),
            "confirmation_ci_low": lower,
            "confirmation_ci_high": upper,
            "confirmation_ci_low_standardized": lower / max(row["discovery_scale"], EPS),
            "confirmation_ci_high_standardized": upper / max(row["discovery_scale"], EPS),
            "confirmation_true_graphs": true_graphs,
            "confirmation_false_graphs": false_graphs,
            "same_sign": bool(np.sign(effect) == np.sign(row["discovery_effect"])),
            "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
            "confirmation_p": _conditional_p_value(
                confirmation, row["rule"], head, row["feature"]
            ),
            **conditional_dose_diagnostics(confirmation, row["rule"]),
            "semantic_concentration_interaction": _conditional_feature_effect(
                confirmation, row["rule"], head, "semantic_concentration"
            )[0],
            "pe_concentration_interaction": _conditional_feature_effect(
                confirmation, row["rule"], head, "pe_concentration"
            )[0],
            "topology_concentration_interaction": _conditional_feature_effect(
                confirmation, row["rule"], head, "topology_concentration"
            )[0],
        })
    q_values = bh_q_values([row["confirmation_p"] for row in confirmed])
    for row, q_value in zip(confirmed, q_values):
        row["confirmation_q_global"] = float(q_value)
        row["confirmed_at_fdr_0.05"] = bool(
            row["same_sign"] and row["ci_excludes_zero"] and np.isfinite(q_value) and q_value <= 0.05
        )
    eligibility_audit["fdr_confirmed_feature_rule_heads"] = int(sum(
        bool(row["confirmed_at_fdr_0.05"]) for row in confirmed
    ))
    return {
        "metadata": metadata,
        "eligibility_audit": eligibility_audit,
        "activity_floor": activity_floor,
        "candidates": tested,
        "discovery_summary": {
            feature_name: {
                "eligible": sum(row["feature"] == feature_name for row in eligible),
                "tested": sum(row["feature"] == feature_name for row in tested),
                "largest_abs_standardized_effect": max(
                    (
                        abs(row["discovery_effect_standardized"])
                        for row in eligible if row["feature"] == feature_name
                    ),
                    default=float("nan"),
                ),
            }
            for feature_name in CONDITIONAL_SCORE_FEATURES
        },
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
    fig, axes = plt.subplots(2, 6, figsize=(16.0, 6.7), constrained_layout=True)
    for row, task in enumerate(("zinc", "zinc_1hop")):
        selection = runs[task]["ablations"]["selections"][aggregation]
        score_pairs = (
            (
                "semantic–PE",
                selection["coordinates"],
                np.asarray(selection["active"], bool),
                ("semantic", "pe"),
            ),
            (
                "semantic–topology",
                score_coordinates(
                    selection["semantic_topology_semantic"],
                    selection["semantic_topology_topology"],
                ),
                None,
                ("semantic", "topology"),
            ),
            (
                "PE–topology",
                selection["pe_topology_coordinates"],
                np.asarray(selection["pe_topology_active"], bool),
                ("pe", "topology"),
            ),
        )
        for pair_index, (pair_label, score, active, channels) in enumerate(score_pairs):
            if active is None:
                floor = max(0.10 * float(np.nanmax(score["J"])), EPS)
                active = np.asarray(score["J"] >= floor, bool)
            causal = causal_pair_coordinates(
                runs[task]["causal"], channels[0], channels[1], "restore"
            )
            for key_index, key in enumerate(("D", "J")):
                column = 2 * pair_index + key_index
                axis = axes[row, column]
                axis.scatter(
                    score[key][~active], causal[key][~active], color="0.78", s=16,
                )
                axis.scatter(
                    score[key][active], causal[key][active], color="#3b7ddd", s=25,
                    edgecolor="black", linewidth=0.25,
                )
                rho = spearman(
                    score[key][active].reshape(-1), causal[key][active].reshape(-1)
                )
                axis.set_title(f"{pair_label} · {key} · ρ={rho:.2f}")
                axis.set_xlabel(f"score {key}")
                axis.set_ylabel(
                    f"{DISPLAY[task]}\nrestore causal {key}"
                    if column == 0 else f"restore causal {key}"
                )
                axis.axhline(0, color="0.6", linewidth=0.6)
                axis.axvline(0, color="0.6", linewidth=0.6)
    fig.suptitle(
        "Activity-gated score coordinates versus held-out whole-transport mediation",
        fontsize=11,
    )
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
    frozen_names = list(causal.get("family_names", []))
    for row, family in enumerate(family_names):
        heads = families.get(family, [])
        for column, channel in enumerate(CHANNELS):
            event_mask = np.asarray([event["channel"] == channel for event in causal["events"]])
            if channel == "topology":
                event_mask &= np.asarray([
                    int(event.get("topology_tier", 99)) <= 1 for event in causal["events"]
                ])
            if family in frozen_names and event_mask.any():
                family_index = frozen_names.index(family)
                matrix[row, column] = np.nanmean(
                    np.asarray(causal["family_metrics"][metric])[event_mask, family_index]
                )
            else:
                value = channel_mean_metric(causal, channel, metric)
                if heads and np.isfinite(value).any():
                    matrix[row, column] = np.nanmean([value[head] for head in heads])
    return matrix


def figure_family_patching(
    runs: Mapping[str, Mapping[str, Any]], aggregation: str, path: Path
) -> list[str]:
    plt = configure_matplotlib()
    family_names = (
        "semantic_specialist", "pe_specialist", "structural_pe_specific",
        "structural_topology_specific", "structural_shared", "high_J",
        "high_G_balanced", "low_J_inert",
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
    pe_graphs, topology_graphs, paired_graph_ids = paired_graph_scores(
        score, "pe", "topology", aggregation
    )
    if len(pe_graphs):
        pe = pe_graphs.mean(axis=0)
        topology = topology_graphs.mean(axis=0)
    else:
        shape = (int(score["L"]), int(score["H"]))
        pe = np.full(shape, np.nan)
        topology = np.full(shape, np.nan)
    topology_restore = channel_mean_metric(causal, "topology", "restore")
    return {
        "graphs": total,
        "paired_graphs": int(len(paired_graph_ids)),
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


def _top_indices(values: np.ndarray, active: np.ndarray, count: int) -> set[int]:
    flat = np.asarray(values, float).reshape(-1)
    eligible = np.flatnonzero(np.asarray(active, bool).reshape(-1) & np.isfinite(flat))
    if not len(eligible):
        return set()
    count = min(int(count), len(eligible))
    order = eligible[np.argsort(flat[eligible], kind="stable")[-count:]]
    return set(map(int, order))


def topology_agreement_analysis(
    score: Mapping[str, Any], cfg: BetaConfig, aggregation: str = HEADLINE_AGGREGATION
) -> dict[str, Any]:
    pe_graphs, topology_graphs, graph_ids = paired_graph_scores(
        score, "pe", "topology", aggregation
    )
    if not len(pe_graphs):
        return {"graph_ids": graph_ids, "available": False}
    pe, topology = pe_graphs.mean(axis=0), topology_graphs.mean(axis=0)
    coordinates = score_coordinates(pe, topology)
    activity_floor = max(
        cfg.activity_floor_relative * float(np.nanmax(coordinates["J"])), EPS
    )
    active = coordinates["J"] >= activity_floor
    flat_active = active.reshape(-1)
    pooled = spearman(pe.reshape(-1)[flat_active], topology.reshape(-1)[flat_active])
    within = spearman(
        layer_centered(pe).reshape(-1)[flat_active],
        layer_centered(topology).reshape(-1)[flat_active],
    )
    full_sets = {
        (channel, int(k)): _top_indices(values, active, int(k))
        for channel, values in (("pe", pe), ("topology", topology))
        for k in cfg.topk_values
    }
    rng = np.random.default_rng(cfg.analysis_seed + 92_101)
    draws: dict[str, list[float]] = {
        "rho_pooled": [], "rho_within_layer": [],
        **{f"rho_layer_{layer}": [] for layer in range(pe.shape[0])},
        **{
            f"top{k}_{name}": []
            for k in cfg.topk_values
            for name in ("overlap", "pe_stability", "topology_stability")
        },
    }
    for _ in range(cfg.bootstrap_samples):
        chosen = rng.integers(0, len(pe_graphs), len(pe_graphs))
        pe_boot = pe_graphs[chosen].mean(axis=0)
        topology_boot = topology_graphs[chosen].mean(axis=0)
        draws["rho_pooled"].append(
            spearman(pe_boot.reshape(-1)[flat_active], topology_boot.reshape(-1)[flat_active])
        )
        draws["rho_within_layer"].append(spearman(
            layer_centered(pe_boot).reshape(-1)[flat_active],
            layer_centered(topology_boot).reshape(-1)[flat_active],
        ))
        for layer in range(pe.shape[0]):
            layer_active = active[layer]
            draws[f"rho_layer_{layer}"].append(spearman(
                pe_boot[layer][layer_active], topology_boot[layer][layer_active]
            ))
        for k in cfg.topk_values:
            pe_set = _top_indices(pe_boot, active, int(k))
            topology_set = _top_indices(topology_boot, active, int(k))
            draws[f"top{k}_overlap"].append(jaccard(pe_set, topology_set))
            draws[f"top{k}_pe_stability"].append(jaccard(pe_set, full_sets[("pe", int(k))]))
            draws[f"top{k}_topology_stability"].append(
                jaccard(topology_set, full_sets[("topology", int(k))])
            )

    def interval(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, float)
        array = array[np.isfinite(array)]
        return {
            "mean": float(np.mean(array)) if len(array) else float("nan"),
            "ci_low": float(np.quantile(array, 0.025)) if len(array) else float("nan"),
            "ci_high": float(np.quantile(array, 0.975)) if len(array) else float("nan"),
        }

    topk = {}
    for k in cfg.topk_values:
        pe_set, topology_set = full_sets[("pe", int(k))], full_sets[("topology", int(k))]
        topk[str(k)] = {
            "point_overlap": jaccard(pe_set, topology_set),
            "overlap": interval(draws[f"top{k}_overlap"]),
            "pe_stability": interval(draws[f"top{k}_pe_stability"]),
            "topology_stability": interval(draws[f"top{k}_topology_stability"]),
        }
    return {
        "available": True,
        "graph_ids": graph_ids,
        "pe": pe,
        "topology": topology,
        "coordinates": coordinates,
        "active": active,
        "activity_floor": activity_floor,
        "rho_pooled": {"point": pooled, **interval(draws["rho_pooled"])},
        "rho_within_layer": {"point": within, **interval(draws["rho_within_layer"])},
        "rho_by_layer": {
            str(layer): {
                "point": spearman(pe[layer][active[layer]], topology[layer][active[layer]]),
                **interval(draws[f"rho_layer_{layer}"]),
            }
            for layer in range(pe.shape[0])
        },
        "topk": topk,
    }


def topology_stratified_analysis(
    score: Mapping[str, Any], causal: Mapping[str, Any], aggregation: str = HEADLINE_AGGREGATION
) -> list[dict[str, Any]]:
    """Tier- and dose-stratified topology/PE agreement and causal validity."""

    event_rows = []
    for record in score["records"]:
        result = record["channels"]["topology"]
        if not result["available"]:
            continue
        for index, item in enumerate(record["plan"]["topology"]):
            q = result["q_groups"][0][index]
            event_score = np.linalg.norm(q.numpy(), axis=-1).sum(axis=-1)  # [L,H]
            event_rows.append({
                "graph_id": int(record["graph_id"]),
                "tier": int(item["tier"]),
                "dose": float(item["dose"]["edge_jaccard_distance"]),
                "score": event_score,
            })
    if not event_rows:
        return []
    doses = np.asarray([row["dose"] for row in event_rows])
    q1, q2 = np.quantile(doses, [1 / 3, 2 / 3])
    strata: list[tuple[str, Any]] = [
        (f"tier_{tier}", lambda row, tier=tier: row["tier"] == tier)
        for tier in sorted({row["tier"] for row in event_rows})
    ] + [
        ("dose_low", lambda row: row["dose"] <= q1),
        ("dose_mid", lambda row: q1 < row["dose"] <= q2),
        ("dose_high", lambda row: row["dose"] > q2),
    ]
    pe_by_graph = {
        int(record["graph_id"]): record_channel_score(
            record, "pe", aggregation, topology_max_tier=None
        )
        for record in score["records"]
    }
    causal_events = causal["events"]
    topology_causal = np.asarray(causal["metrics"]["restore"], float)
    frozen_names = list(causal.get("family_names", []))
    frozen_restore = np.asarray(
        causal.get("family_metrics", {}).get(
            "restore", np.empty((len(causal_events), 0))
        ),
        float,
    )

    def graph_balanced_mean(values: Sequence[Any], graph_ids: Sequence[int]) -> np.ndarray:
        grouped: dict[int, list[np.ndarray]] = {}
        for value, graph_id in zip(values, graph_ids):
            grouped.setdefault(int(graph_id), []).append(np.asarray(value, float))
        if not grouped:
            return np.asarray(float("nan"))
        graph_means = [np.nanmean(np.stack(grouped[key]), axis=0) for key in sorted(grouped)]
        return np.nanmean(np.stack(graph_means), axis=0)

    output = []
    for name, predicate in strata:
        selected = [row for row in event_rows if predicate(row)]
        if not selected:
            continue
        selected_graph_ids = sorted({int(row["graph_id"]) for row in selected})
        topology_score = graph_balanced_mean(
            [row["score"] for row in selected], [row["graph_id"] for row in selected]
        )
        pe = np.nanmean(np.stack([pe_by_graph[graph_id] for graph_id in selected_graph_ids]), axis=0)
        if name.startswith("tier_"):
            tier = int(name.rsplit("_", 1)[1])
            causal_mask = np.asarray([
                event["channel"] == "topology" and int(event.get("topology_tier", -1)) == tier
                for event in causal_events
            ])
        else:
            causal_doses = np.asarray([
                float(event.get("topology_dose", {}).get("edge_jaccard_distance", np.nan))
                for event in causal_events
            ])
            if name == "dose_low":
                dose_mask = causal_doses <= q1
            elif name == "dose_mid":
                dose_mask = (causal_doses > q1) & (causal_doses <= q2)
            else:
                dose_mask = causal_doses > q2
            causal_mask = np.asarray([event["channel"] == "topology" for event in causal_events]) & dose_mask
        causal_graph_ids = np.asarray(
            [int(event["graph_id"]) for event in causal_events], dtype=np.int64
        )
        restore = (
            graph_balanced_mean(topology_causal[causal_mask], causal_graph_ids[causal_mask])
            if causal_mask.any() else np.full_like(pe, np.nan)
        )
        coordinates = score_coordinates(pe, topology_score)
        activity_floor = max(0.10 * float(np.nanmax(coordinates["J"])), EPS)
        active = coordinates["J"] >= activity_floor
        family_restore = {}
        for family in (
            "structural_pe_specific", "structural_topology_specific", "structural_shared"
        ):
            if family in frozen_names and causal_mask.any():
                family_restore[family] = float(graph_balanced_mean(
                    frozen_restore[causal_mask, frozen_names.index(family)],
                    causal_graph_ids[causal_mask],
                ))
            else:
                family_restore[family] = float("nan")
        output.append({
            "stratum": name,
            "events": len(selected),
            "graphs": len({row["graph_id"] for row in selected}),
            "causal_graphs": len(set(causal_graph_ids[causal_mask].tolist())),
            "dose_mean": float(graph_balanced_mean(
                [row["dose"] for row in selected], [row["graph_id"] for row in selected]
            )),
            "active_heads": int(active.sum()),
            "rho_pe_topology_pooled": spearman(
                pe[active].reshape(-1), topology_score[active].reshape(-1)
            ),
            "rho_pe_topology_within_layer": spearman(
                layer_centered(pe)[active].reshape(-1),
                layer_centered(topology_score)[active].reshape(-1),
            ),
            "rho_topology_restore": spearman(
                topology_score[active].reshape(-1), restore[active].reshape(-1)
            ),
            **{
                f"family_restore_{family}": value
                for family, value in family_restore.items()
            },
        })
    return output


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


def figure_three_channel_planes(
    runs: Mapping[str, Mapping[str, Any]], path: Path
) -> list[str]:
    """Complete raw-score and D/J planes for all three pairwise channel contrasts."""

    plt = configure_matplotlib()
    fig, axes = plt.subplots(4, 3, figsize=(12.5, 11.0), constrained_layout=True)
    scatter = None
    for architecture_index, task in enumerate(ARCHITECTURES):
        score = runs[task]["scores"]
        selection = runs[task]["ablations"]["selections"][HEADLINE_AGGREGATION]
        semantic, pe, topology = (
            np.asarray(selection[key], float) for key in ("semantic", "pe", "topology")
        )
        semantic_topology_semantic = np.asarray(
            selection["semantic_topology_semantic"], float
        )
        semantic_topology_topology = np.asarray(
            selection["semantic_topology_topology"], float
        )
        pe_topology_pe = np.asarray(selection["pe_topology_pe"], float)
        pe_topology_topology = np.asarray(
            selection["pe_topology_topology"], float
        )
        cmap, norm = layer_colors(int(score["L"]))
        layers = _head_layers(int(score["L"]), int(score["H"]))
        sem_pe = selection["coordinates"]
        sem_top = score_coordinates(
            semantic_topology_semantic, semantic_topology_topology
        )
        pe_top = selection["pe_topology_coordinates"]
        sem_pe_active = np.asarray(selection["active"], bool)
        sem_top_active = sem_top["J"] >= max(
            0.10 * float(np.nanmax(sem_top["J"])), EPS
        )
        pe_top_active = np.asarray(selection["pe_topology_active"], bool)
        comparisons = (
            (
                (pe, semantic, "PE EG score", "semantic EG score"),
                (sem_pe["D"], sem_pe["J"], "D (− PE, + semantic)", "J"),
                sem_pe_active,
                "semantic–PE",
            ),
            (
                (
                    semantic_topology_topology,
                    semantic_topology_semantic,
                    "topology EG score",
                    "semantic EG score",
                ),
                (
                    sem_top["D"],
                    sem_top["J"],
                    "D (− topology, + semantic)",
                    "J",
                ),
                sem_top_active,
                "semantic–topology",
            ),
            (
                (
                    pe_topology_topology,
                    pe_topology_pe,
                    "topology EG score",
                    "PE EG score",
                ),
                (pe_top["D"], pe_top["J"], "D (− topology, + PE)", "J"),
                pe_top_active,
                "PE–topology",
            ),
        )
        raw_row = 2 * architecture_index
        coordinate_row = raw_row + 1
        for column, (raw, coordinate, active, pair_label) in enumerate(comparisons):
            for axis, values, title, raw_plane in (
                (axes[raw_row, column], raw, f"{pair_label} scores", True),
                (
                    axes[coordinate_row, column],
                    coordinate,
                    f"{pair_label} D/J",
                    False,
                ),
            ):
                xval, yval, xlabel, ylabel = values
                axis.scatter(
                    np.asarray(xval).reshape(-1)[~active.reshape(-1)],
                    np.asarray(yval).reshape(-1)[~active.reshape(-1)],
                    color="0.82", s=18, label="below gate",
                )
                scatter = axis.scatter(
                    np.asarray(xval).reshape(-1)[active.reshape(-1)],
                    np.asarray(yval).reshape(-1)[active.reshape(-1)],
                    c=layers[active.reshape(-1)], cmap=cmap, norm=norm, s=27,
                    edgecolor="black", linewidth=0.25,
                )
                if raw_plane:
                    low, high = _identity_limits(xval, yval)
                    axis.plot(
                        [low, high], [low, high], ":", color="black", linewidth=0.8
                    )
                else:
                    axis.axvline(0.0, color="black", linewidth=0.7)
                axis.set_xlabel(xlabel)
                axis.set_ylabel(ylabel)
                axis.set_title(title)
        axes[raw_row, 0].text(
            0.02, 0.98, DISPLAY[task],
            transform=axes[raw_row, 0].transAxes, va="top",
        )
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, fraction=0.016, pad=0.01, label="layer")
    fig.suptitle("Headline EG: three-channel score and activity-gated selectivity planes", fontsize=11)
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def figure_topology_agreement(
    runs: Mapping[str, Mapping[str, Any]], cfg: BetaConfig, path: Path
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.6), constrained_layout=True)
    for row, task in enumerate(ARCHITECTURES):
        analysis = topology_agreement_analysis(runs[task]["scores"], cfg)
        pe, topology = analysis["pe"], analysis["topology"]
        active = np.asarray(analysis["active"], bool).reshape(-1)
        axis = axes[row, 0]
        axis.scatter(pe.reshape(-1)[~active], topology.reshape(-1)[~active], color="0.82", s=17)
        axis.scatter(pe.reshape(-1)[active], topology.reshape(-1)[active], color="#6f63a8", s=27,
                     edgecolor="black", linewidth=0.25)
        low, high = _identity_limits(pe, topology)
        axis.plot([low, high], [low, high], ":", color="black", linewidth=0.8)
        axis.set_xlabel("PE EG score"); axis.set_ylabel("topology EG score")
        axis.set_title(f"{DISPLAY[task]} · activity gated")

        axis = axes[row, 1]
        labels = ("pooled", "within layer")
        entries = (analysis["rho_pooled"], analysis["rho_within_layer"])
        points = [entry["point"] for entry in entries]
        correlation_x = np.arange(len(labels))
        axis.bar(correlation_x, points, color=("#6f63a8", "#4f8a78"))
        axis.set_xticks(correlation_x, labels)
        axis.errorbar(
            correlation_x, points,
            yerr=[
                [max(point - entry["ci_low"], 0.0) for point, entry in zip(points, entries)],
                [max(entry["ci_high"] - point, 0.0) for point, entry in zip(points, entries)],
            ],
            fmt="none", color="black", capsize=3,
        )
        layer_points = np.asarray([
            entry["point"] for entry in analysis["rho_by_layer"].values()
        ], float)
        finite_layers = layer_points[np.isfinite(layer_points)]
        if len(finite_layers):
            jitter = np.linspace(-0.10, 0.10, len(finite_layers))
            axis.scatter(1.0 + jitter, finite_layers, marker="_", color="black", s=18,
                         label="individual layers" if row == 0 else None)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_ylabel("Spearman ρ (graph bootstrap 95% CI)")
        axis.set_title("PE–topology agreement")

        axis = axes[row, 2]
        ks = [int(value) for value in cfg.topk_values]
        for key, label, color in (
            ("overlap", "PE ∩ topology", "#6f63a8"),
            ("pe_stability", "PE stability", "#4c78a8"),
            ("topology_stability", "topology stability", "#c06b53"),
        ):
            values = [analysis["topk"][str(k)][key]["mean"] for k in ks]
            low = [analysis["topk"][str(k)][key]["ci_low"] for k in ks]
            high = [analysis["topk"][str(k)][key]["ci_high"] for k in ks]
            axis.plot(ks, values, marker="o", color=color, label=label)
            axis.fill_between(ks, low, high, color=color, alpha=0.15)
        axis.set_ylim(-0.03, 1.03); axis.set_xticks(ks)
        axis.set_xlabel("top k"); axis.set_ylabel("Jaccard")
        axis.set_title("Top-k overlap and bootstrap stability")
    axes[0, 2].legend(frameon=False, fontsize=7)
    fig.suptitle("PE versus matched-topology agreement under the fixed EG/activity protocol", fontsize=11)
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def figure_topology_strata(
    runs: Mapping[str, Mapping[str, Any]], path: Path
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.5), constrained_layout=True)
    for row, task in enumerate(ARCHITECTURES):
        rows = topology_stratified_analysis(runs[task]["scores"], runs[task]["causal"])
        labels = [item["stratum"].replace("_", " ") for item in rows]
        x = np.arange(len(rows))
        axes[row, 0].bar(x, [item["rho_pe_topology_within_layer"] for item in rows], color="#6f63a8")
        axes[row, 0].axhline(0, color="black", linewidth=0.7)
        axes[row, 0].set_ylabel("within-layer PE–topology ρ")
        axes[row, 1].bar(x, [item["rho_topology_restore"] for item in rows], color="#c06b53")
        axes[row, 1].axhline(0, color="black", linewidth=0.7)
        axes[row, 1].set_ylabel("topology score–restore ρ")
        width = 0.24
        for offset, (family, label, color) in enumerate((
            ("structural_pe_specific", "PE-specific", "#4c78a8"),
            ("structural_topology_specific", "topology-specific", "#6f63a8"),
            ("structural_shared", "shared", "#55a868"),
        )):
            axes[row, 2].bar(
                x + (offset - 1) * width,
                [item[f"family_restore_{family}"] for item in rows],
                width,
                color=color,
                label=label,
            )
        axes[row, 2].axhline(0, color="black", linewidth=0.7)
        axes[row, 2].set_ylabel("topology-family restore")
        for axis in axes[row]:
            axis.set_xticks(x, labels, rotation=30, ha="right")
            axis.set_title(DISPLAY[task])
    axes[0, 2].legend(frameon=False, fontsize=7)
    fig.suptitle("Topology conclusions separated by donor tier and graph-edit dose", fontsize=11)
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def figure_ablation(
    runs: Mapping[str, Mapping[str, Any]], aggregation: str, path: Path
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.8), constrained_layout=True)
    colors = {
        "semantic_specialist": "#c75b5b", "pe_specialist": "#4c78a8",
        "structural_pe_specific": "#3c78a8", "structural_topology_specific": "#6f63a8",
        "structural_shared": "#55a868", "low_J_inert": "0.5",
    }
    for row, task in enumerate(("zinc", "zinc_1hop")):
        score = runs[task]["scores"]
        ablation = runs[task]["ablations"]
        selection = ablation["selections"][aggregation]
        impact = np.asarray(ablation["per_head"]["functional"])
        for column, (channel, label, color) in enumerate((
            ("semantic", "semantic EG", "#c75b5b"),
            ("pe", "PE EG", "#4c78a8"),
            ("topology", "topology EG", "#6f63a8"),
        )):
            values = np.asarray(selection[channel], float)
            axes[row, column].scatter(values.reshape(-1), impact.reshape(-1), s=22, color=color)
            axes[row, column].set_title(
                f"{DISPLAY[task]} · ρ={spearman(values.reshape(-1), impact.reshape(-1)):.2f}"
            )
            axes[row, column].set_xlabel(label)
            if column == 0:
                axes[row, column].set_ylabel("single-head functional ablation")
        curves = ablation["family_curves"][aggregation]
        for family, color in colors.items():
            rows = curves.get(family, [])
            if rows:
                axes[row, 3].plot(
                    [item["count"] for item in rows], [item["functional"] for item in rows],
                    marker="o", color=color, label=family.replace("_", " "),
                )
        axes[row, 3].set_title(f"{DISPLAY[task]} · frozen-family ablation")
        axes[row, 3].set_xlabel("heads ablated")
        axes[row, 3].set_ylabel("mean |Δ prediction|")
    axes[0, 3].legend(frameon=False, fontsize=6)
    fig.suptitle("Held-out ablation validation for all three headline EG channels", fontsize=11)
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
    fig, axes = plt.subplots(4, 3, figsize=(11.5, 11.0), constrained_layout=True)
    family_styles = (
        ("semantic_specialist", "#c75b5b"),
        ("structural_pe_specific", "#4c78a8"),
        ("structural_topology_specific", "#6f63a8"),
        ("structural_shared", "#55a868"),
    )
    for architecture_index, task in enumerate(("zinc", "zinc_1hop")):
        mechanism = runs[task]["mechanism"]
        events = mechanism["events"]
        families = runs[task]["ablations"]["selections"][aggregation]["families"]
        for column, channel in enumerate(CHANNELS):
            mask = np.asarray([event["channel"] == channel for event in events])
            for mode_index, (prefix, absolute, label) in enumerate((
                ("", True, "projected transport |q|"),
                ("_rescue", False, "finite component patch"),
            )):
                row = 2 * architecture_index + mode_index
                axis = axes[row, column]
                if prefix:
                    routing = np.nanmean(
                        np.asarray(mechanism["metrics"]["routing_rescue"])[mask], axis=0
                    )
                    message = np.nanmean(
                        np.asarray(mechanism["metrics"]["message_rescue"])[mask], axis=0
                    )
                    wiring = np.nanmean(
                        np.asarray(mechanism["metrics"]["wiring_rescue"])[mask], axis=0
                    )
                else:
                    routing = np.nanmean(
                        np.abs(np.asarray(mechanism["metrics"]["routing_q"])[mask]), axis=0
                    )
                    message = np.nanmean(
                        np.abs(np.asarray(mechanism["metrics"]["message_q"])[mask]), axis=0
                    )
                    wiring = np.nanmean(
                        np.abs(np.asarray(mechanism["metrics"]["wiring_q"])[mask]), axis=0
                    )
                wiring_scale = float(np.nanmax(np.abs(wiring))) if np.isfinite(wiring).any() else 0.0
                wiring_scale = max(wiring_scale, EPS)
                for family, color in family_styles:
                    heads = families.get(family, [])
                    if heads:
                        axis.scatter(
                            [routing[head] for head in heads],
                            [message[head] for head in heads],
                            s=[
                                35 + 90 * abs(float(wiring[head])) / wiring_scale
                                for head in heads
                            ],
                            color=color,
                            label=family.replace("_", " "),
                            edgecolor="black",
                            linewidth=0.3,
                        )
                low, high = _identity_limits(routing, message)
                axis.plot([low, high], [low, high], ":", color="black", linewidth=0.8)
                axis.axhline(0, color="0.7", linewidth=0.5)
                axis.axvline(0, color="0.7", linewidth=0.5)
                axis.set_xlabel("routing" + (" |q|" if absolute else " restore"))
                message_label = "message" + (" |q|" if absolute else " restore")
                axis.set_ylabel(
                    f"{DISPLAY[task]}\n{label}\n{message_label}"
                    if column == 0 else message_label
                )
                axis.set_title(
                    ("semantic", "PE", "topology")[column] if row == 0 else ""
                )
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Why specialisation appears: exact transport split and finite component patches "
        "(point area = |wiring|)",
        fontsize=11,
    )
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


def model_carriage_profile(
    score: Mapping[str, Any],
    channel: str,
    *,
    donor_scope: str = "headline",
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Graph-balanced final-state F_sens/F_coh/B distance profile."""

    keys = ("F_sens", "F_coh", "B", "B_sum")
    by_distance: dict[int, dict[int, dict[str, list[float]]]] = {}
    for record in score["records"]:
        result = record["channels"][channel]
        if not result["available"]:
            continue
        for item in result.get("model_carriage", []):
            if str(item.get("donor_scope", "headline")) != donor_scope:
                continue
            distance, graph_id = int(item["distance"]), int(record["graph_id"])
            bucket = by_distance.setdefault(distance, {}).setdefault(
                graph_id, {key: [] for key in keys}
            )
            for key in keys:
                bucket[key].append(float(item[key]))
    distances = sorted(by_distance)
    graph_ids = sorted({gid for distance in distances for gid in by_distance[distance]})
    result: dict[str, Any] = {"distance": np.asarray(distances, dtype=int)}
    if not graph_ids:
        for key in keys:
            result[key] = np.asarray([], dtype=float)
            result[f"{key}_ci_low"] = np.asarray([], dtype=float)
            result[f"{key}_ci_high"] = np.asarray([], dtype=float)
        result["graph_support"] = np.asarray([], dtype=int)
        return result
    rng = np.random.default_rng(seed)
    for key in keys:
        matrix = np.full((len(graph_ids), len(distances)), np.nan)
        for i, graph_id in enumerate(graph_ids):
            for j, distance in enumerate(distances):
                values = by_distance[distance].get(graph_id, {}).get(key, [])
                if values:
                    matrix[i, j] = np.mean(values)
        result[key] = np.nanmean(matrix, axis=0)
        draws = []
        for _ in range(bootstrap_samples):
            chosen = rng.integers(0, len(graph_ids), len(graph_ids))
            draws.append(np.nanmean(matrix[chosen], axis=0))
        stacked = np.stack(draws)
        result[f"{key}_ci_low"] = np.nanquantile(stacked, 0.025, axis=0)
        result[f"{key}_ci_high"] = np.nanquantile(stacked, 0.975, axis=0)
    result["graph_support"] = np.asarray([
        len(by_distance[distance]) for distance in distances
    ], dtype=int)
    return result


def distance_specialisation_rows(
    architecture: str, profile: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Long-form exact EG-by-distance contributions for every layer and head."""

    rows = []
    contribution = np.asarray(profile["contribution"], dtype=float)
    fraction = np.asarray(profile["fraction"], dtype=float)
    low = np.asarray(profile["fraction_ci_low"], dtype=float)
    high = np.asarray(profile["fraction_ci_high"], dtype=float)
    density = np.asarray(profile["support_normalised_response"], dtype=float)
    density_fraction = np.asarray(
        profile["support_normalised_fraction"], dtype=float
    )
    density_low = np.asarray(
        profile["support_normalised_fraction_ci_low"], dtype=float
    )
    density_high = np.asarray(
        profile["support_normalised_fraction_ci_high"], dtype=float
    )
    opportunity = np.asarray(profile["mean_opportunity"], dtype=float)
    support = np.asarray(profile["graph_support"], dtype=bool)
    for bucket, (code, label) in enumerate(zip(profile["codes"], profile["labels"])):
        for layer in range(contribution.shape[1]):
            for head in range(contribution.shape[2]):
                rows.append({
                    "architecture": architecture,
                    "channel": profile["channel"],
                    "donor_scope": profile["donor_scope"],
                    "distance_code": int(code),
                    "distance_bucket": label,
                    "layer": layer,
                    "head": head,
                    "EG_contribution": float(contribution[bucket, layer, head]),
                    "fraction_of_head_EG": float(fraction[bucket, layer, head]),
                    "fraction_ci_low": float(low[bucket, layer, head]),
                    "fraction_ci_high": float(high[bucket, layer, head]),
                    "mean_event_carrier_opportunity": float(opportunity[bucket]),
                    "support_normalised_EG": float(density[bucket, layer, head]),
                    "fraction_of_support_normalised_head_EG": float(
                        density_fraction[bucket, layer, head]
                    ),
                    "support_normalised_fraction_ci_low": float(
                        density_low[bucket, layer, head]
                    ),
                    "support_normalised_fraction_ci_high": float(
                        density_high[bucket, layer, head]
                    ),
                    "graph_support": int(support[:, bucket].sum()),
                    "graphs": int(support.shape[0]),
                })
    return rows


def head_reach_summary_rows(
    architecture: str, profile: Mapping[str, Any]
) -> list[dict[str, Any]]:
    total = np.asarray(profile["total"], dtype=float)
    codes = np.asarray(profile["codes"], dtype=int)
    support_fraction = np.asarray(
        profile["support_normalised_fraction"], dtype=float
    )
    numeric = codes >= 0
    numeric_hops = np.maximum(codes, 0).astype(float)
    support_expected = (
        support_fraction * numeric_hops[:, None, None] * numeric[:, None, None]
    ).sum(axis=0)
    support_near = support_fraction[
        np.asarray([(code >= 0 and code <= 1) for code in codes], dtype=bool)
    ].sum(axis=0)
    support_far = support_fraction[
        np.asarray([code >= 2 for code in codes], dtype=bool)
    ].sum(axis=0)
    rows = []
    for layer in range(total.shape[0]):
        for head in range(total.shape[1]):
            rows.append({
                "architecture": architecture,
                "channel": profile["channel"],
                "donor_scope": profile["donor_scope"],
                "layer": layer,
                "head": head,
                "S_EG_reconstructed": float(total[layer, head]),
                "expected_numeric_distance": float(profile["expected_distance"][layer, head]),
                "near_share_d_le_1": float(profile["near_share"][layer, head]),
                "far_share_d_ge_2": float(profile["far_share"][layer, head]),
                "support_normalised_expected_distance": float(
                    support_expected[layer, head]
                ),
                "support_normalised_near_share_d_le_1": float(
                    support_near[layer, head]
                ),
                "support_normalised_far_share_d_ge_2": float(
                    support_far[layer, head]
                ),
                "unreachable_share": float(profile["unreachable_share"][layer, head]),
                "hub_share": float(profile["hub_share"][layer, head]),
            })
    return rows


def attention_distance_rows(
    architecture: str, profile: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Long-form clean post-softmax attention locality for every head."""

    fraction = np.asarray(profile["fraction"], dtype=float)
    low = np.asarray(profile["fraction_ci_low"], dtype=float)
    high = np.asarray(profile["fraction_ci_high"], dtype=float)
    rows = []
    for bucket, (code, label) in enumerate(zip(profile["codes"], profile["labels"])):
        for layer in range(fraction.shape[1]):
            for head in range(fraction.shape[2]):
                rows.append({
                    "architecture": architecture,
                    "distance_code": int(code),
                    "distance_bucket": label,
                    "layer": layer,
                    "head": head,
                    "fraction_of_clean_attention_mass": float(
                        fraction[bucket, layer, head]
                    ),
                    "fraction_ci_low": float(low[bucket, layer, head]),
                    "fraction_ci_high": float(high[bucket, layer, head]),
                    "mean_supported_attention_pairs": float(
                        profile["mean_pair_count"][bucket]
                    ),
                    "graph_support": int(profile["graph_support"][bucket]),
                    "graphs": int(len(profile["graph_ids"])),
                })
    return rows


def locality_curve_rows(
    architecture: str,
    channel: str,
    profile: Mapping[str, Any],
    attention: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate curve values and graph-bootstrap intervals used by the locality figure."""

    rows = []
    numeric = np.asarray(profile["codes"], dtype=int) >= 0
    for name, value_key, low_key, high_key in (
        (
            "exact_EG_mass",
            "aggregate_mass_fraction",
            "aggregate_mass_fraction_ci_low",
            "aggregate_mass_fraction_ci_high",
        ),
        (
            "support_normalised_EG",
            "aggregate_support_normalised_fraction",
            "aggregate_support_normalised_fraction_ci_low",
            "aggregate_support_normalised_fraction_ci_high",
        ),
    ):
        for index in np.flatnonzero(numeric):
            rows.append({
                "architecture": architecture,
                "channel": channel,
                "profile": name,
                "distance": int(profile["codes"][index]),
                "value": float(profile[value_key][index]),
                "ci_low": float(profile[low_key][index]),
                "ci_high": float(profile[high_key][index]),
                "graph_support": int(
                    np.asarray(profile["graph_support"], dtype=bool)[:, index].sum()
                ),
            })
    attention_numeric = np.asarray(attention["codes"], dtype=int) >= 0
    for index in np.flatnonzero(attention_numeric):
        rows.append({
            "architecture": architecture,
            "channel": channel,
            "profile": "clean_attention_mass",
            "distance": int(attention["codes"][index]),
            "value": float(attention["aggregate_fraction"][index]),
            "ci_low": float(attention["aggregate_fraction_ci_low"][index]),
            "ci_high": float(attention["aggregate_fraction_ci_high"][index]),
            "graph_support": int(attention["graph_support"][index]),
        })
    return rows


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    keep = np.isfinite(left) & np.isfinite(right)
    if not keep.any():
        return float("nan")
    denominator = float(np.linalg.norm(left[keep]) * np.linalg.norm(right[keep]))
    return float(np.dot(left[keep], right[keep]) / denominator) if denominator > EPS else float("nan")


def family_distance_reach_rows(
    architecture: str,
    profile: Mapping[str, Any],
    families: Mapping[str, Sequence[tuple[int, int]]],
    carriage: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Frozen-family score reach, aligned descriptively with final-state F and B."""

    codes = np.asarray(profile["codes"], dtype=int)
    numeric = codes >= 0
    numeric_codes = codes[numeric]
    contribution = np.asarray(profile["contribution"], dtype=float)[numeric]
    all_head = contribution.sum(axis=(1, 2))
    carriage_distance = np.asarray(carriage["distance"], dtype=int)
    carriage_lookup = {int(value): index for index, value in enumerate(carriage_distance)}
    common = np.asarray([code in carriage_lookup for code in numeric_codes], dtype=bool)
    family_map: dict[str, Sequence[tuple[int, int]]] = {"all_heads": [
        (layer, head)
        for layer in range(contribution.shape[1])
        for head in range(contribution.shape[2])
    ]}
    family_map.update({
        name: families.get(name, []) for name in DISTANCE_FAMILIES[profile["channel"]]
    })
    rows = []
    for family, heads in family_map.items():
        valid_heads = [
            (int(layer), int(head)) for layer, head in heads
            if 0 <= int(layer) < contribution.shape[1]
            and 0 <= int(head) < contribution.shape[2]
        ]
        if not valid_heads:
            continue
        family_contribution = np.asarray([
            sum(float(contribution[index, layer, head]) for layer, head in valid_heads)
            for index in range(len(numeric_codes))
        ])
        family_total = float(family_contribution.sum())
        distribution = family_contribution / max(family_total, EPS)
        bin_share = np.divide(
            family_contribution,
            all_head,
            out=np.zeros_like(family_contribution),
            where=all_head > EPS,
        )
        common_family = distribution[common]
        common_f = np.asarray([
            carriage["F_sens"][carriage_lookup[int(code)]]
            for code in numeric_codes[common]
        ], dtype=float)
        common_b = np.asarray([
            carriage["B"][carriage_lookup[int(code)]]
            for code in numeric_codes[common]
        ], dtype=float)
        f_cosine = _cosine_similarity(common_family, common_f)
        b_cosine = _cosine_similarity(common_family, common_b)
        expected = float(np.dot(numeric_codes, distribution)) if family_total > EPS else float("nan")
        for index, code in enumerate(numeric_codes):
            rows.append({
                "architecture": architecture,
                "channel": profile["channel"],
                "donor_scope": profile["donor_scope"],
                "family": family,
                "heads": str(valid_heads),
                "distance": int(code),
                "EG_contribution": float(family_contribution[index]),
                "distance_fraction_within_family": float(distribution[index]),
                "family_share_of_all_head_EG_at_distance": float(bin_share[index]),
                "expected_numeric_distance": expected,
                "cosine_to_F_sens_profile": f_cosine,
                "signed_cosine_to_B_profile": b_cosine,
            })
    return rows


def figure_distance_specialisation_atlas(
    profiles: Mapping[tuple[str, str], Mapping[str, Any]], path: Path
) -> list[str]:
    """Head-by-distance atlas of the fraction of each head's exact EG score."""

    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.0), constrained_layout=True)
    vmax = max(
        float(np.nanmax(np.asarray(profile["fraction"], float)))
        for profile in profiles.values()
    )
    image = None
    for row, task in enumerate(ARCHITECTURES):
        for column, channel in enumerate(CHANNELS):
            profile = profiles[(task, channel)]
            fraction = np.asarray(profile["fraction"], float)
            matrix = fraction.transpose(1, 2, 0).reshape(-1, fraction.shape[0])
            axis = axes[row, column]
            image = axis.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap="magma",
                vmin=0.0,
                vmax=max(vmax, EPS),
            )
            H = fraction.shape[2]
            L = fraction.shape[1]
            for layer in range(1, L):
                axis.axhline(layer * H - 0.5, color="white", linewidth=0.35, alpha=0.7)
            axis.set_yticks(
                [layer * H + (H - 1) / 2 for layer in range(L)],
                [f"L{layer}" for layer in range(L)],
            )
            axis.set_xticks(np.arange(len(profile["labels"])), profile["labels"])
            axis.set_xlabel("carrier distance from changed set [hops]")
            if column == 0:
                axis.set_ylabel(f"{DISPLAY[task]}\nhead rows (layer blocks)")
            axis.set_title(("semantic donor", "PE transposition", "topology donor")[column])
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.01)
        colorbar.set_label("fraction of that head's EG score")
    fig.suptitle(
        "Who implements intervention reach: exact distance decomposition of head EG", fontsize=11
    )
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def figure_support_normalised_distance_atlas(
    profiles: Mapping[tuple[str, str], Mapping[str, Any]], path: Path
) -> list[str]:
    """Per-head response-density atlas after removing distance-shell opportunity."""

    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.0), constrained_layout=True)
    vmax = max(
        float(np.nanmax(np.asarray(
            profile["support_normalised_fraction"], float
        )))
        for profile in profiles.values()
    )
    image = None
    for row, task in enumerate(ARCHITECTURES):
        for column, channel in enumerate(CHANNELS):
            profile = profiles[(task, channel)]
            fraction = np.asarray(
                profile["support_normalised_fraction"], float
            )
            matrix = fraction.transpose(1, 2, 0).reshape(-1, fraction.shape[0])
            axis = axes[row, column]
            image = axis.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap="magma",
                vmin=0.0,
                vmax=max(vmax, EPS),
            )
            H = fraction.shape[2]
            L = fraction.shape[1]
            for layer in range(1, L):
                axis.axhline(layer * H - 0.5, color="white", linewidth=0.35, alpha=0.7)
            axis.set_yticks(
                [layer * H + (H - 1) / 2 for layer in range(L)],
                [f"L{layer}" for layer in range(L)],
            )
            axis.set_xticks(np.arange(len(profile["labels"])), profile["labels"])
            axis.set_xlabel("carrier distance from changed set [hops]")
            if column == 0:
                axis.set_ylabel(f"{DISPLAY[task]}\nhead rows (layer blocks)")
            axis.set_title(("semantic donor", "PE transposition", "topology donor")[column])
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.01)
        colorbar.set_label("fraction of support-normalised head response")
    fig.suptitle(
        "Per-opportunity functional locality: EG divided by event-carrier support",
        fontsize=11,
    )
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def figure_attention_distance_atlas(
    profiles: Mapping[str, Mapping[str, Any]], path: Path
) -> list[str]:
    """Clean attention-mass locality for every layer and head."""

    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    vmax = max(
        float(np.nanmax(np.asarray(profile["fraction"], float)))
        for profile in profiles.values()
    )
    image = None
    for column, task in enumerate(ARCHITECTURES):
        profile = profiles[task]
        fraction = np.asarray(profile["fraction"], float)
        matrix = fraction.transpose(1, 2, 0).reshape(-1, fraction.shape[0])
        axis = axes[column]
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=max(vmax, EPS),
        )
        H = fraction.shape[2]
        L = fraction.shape[1]
        for layer in range(1, L):
            axis.axhline(layer * H - 0.5, color="white", linewidth=0.35, alpha=0.7)
        axis.set_yticks(
            [layer * H + (H - 1) / 2 for layer in range(L)],
            [f"L{layer}" for layer in range(L)],
        )
        axis.set_xticks(np.arange(len(profile["labels"])), profile["labels"])
        axis.set_xlabel("clean attention query-key distance [molecular hops]")
        axis.set_ylabel(f"{DISPLAY[task]}\nhead rows (layer blocks)")
        axis.set_title(DISPLAY[task])
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, fraction=0.022, pad=0.01)
        colorbar.set_label("fraction of clean post-softmax attention mass")
    fig.suptitle(
        "How local are the raw attention matrices?", fontsize=11
    )
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def figure_distance_locality_curves(
    profiles: Mapping[tuple[str, str], Mapping[str, Any]],
    attention_profiles: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> list[str]:
    """Exact score mass, per-opportunity response and raw attention with uncertainty."""

    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), constrained_layout=True)
    colors = {
        "exact": "#6f6f6f",
        "normalised": "#c75b5b",
        "attention": "#4c78a8",
    }
    positive = []
    for profile in profiles.values():
        for key in (
            "aggregate_mass_fraction",
            "aggregate_support_normalised_fraction",
        ):
            values = np.asarray(profile[key], float)
            positive.extend(values[np.isfinite(values) & (values > 0)].tolist())
    for profile in attention_profiles.values():
        values = np.asarray(profile["aggregate_fraction"], float)
        positive.extend(values[np.isfinite(values) & (values > 0)].tolist())
    lower = max(min(positive) * 0.5 if positive else 1.0e-6, 1.0e-8)
    for row, task in enumerate(ARCHITECTURES):
        attention = attention_profiles[task]
        attention_numeric = np.asarray(attention["codes"], int) >= 0
        attention_x = np.asarray(attention["codes"], int)[attention_numeric]
        for column, channel in enumerate(CHANNELS):
            axis = axes[row, column]
            profile = profiles[(task, channel)]
            numeric = np.asarray(profile["codes"], int) >= 0
            x = np.asarray(profile["codes"], int)[numeric]
            series = (
                (
                    "exact EG score mass",
                    "aggregate_mass_fraction",
                    "aggregate_mass_fraction_ci_low",
                    "aggregate_mass_fraction_ci_high",
                    colors["exact"],
                    "o-",
                ),
                (
                    "support-normalised EG",
                    "aggregate_support_normalised_fraction",
                    "aggregate_support_normalised_fraction_ci_low",
                    "aggregate_support_normalised_fraction_ci_high",
                    colors["normalised"],
                    "s-",
                ),
            )
            for label, key, low_key, high_key, color, style in series:
                value = np.asarray(profile[key], float)[numeric]
                low = np.asarray(profile[low_key], float)[numeric]
                high = np.asarray(profile[high_key], float)[numeric]
                axis.fill_between(
                    x, np.maximum(low, lower), np.maximum(high, lower),
                    color=color, alpha=0.13, linewidth=0,
                )
                axis.plot(x, np.maximum(value, lower), style, color=color,
                          linewidth=1.3, markersize=3.5, label=label)
            attention_value = np.asarray(
                attention["aggregate_fraction"], float
            )[attention_numeric]
            attention_low = np.asarray(
                attention["aggregate_fraction_ci_low"], float
            )[attention_numeric]
            attention_high = np.asarray(
                attention["aggregate_fraction_ci_high"], float
            )[attention_numeric]
            axis.fill_between(
                attention_x,
                np.maximum(attention_low, lower),
                np.maximum(attention_high, lower),
                color=colors["attention"],
                alpha=0.10,
                linewidth=0,
            )
            axis.plot(
                attention_x,
                np.maximum(attention_value, lower),
                "^--",
                color=colors["attention"],
                linewidth=1.2,
                markersize=3.5,
                label="clean attention mass",
            )
            axis.set_yscale("log")
            axis.set_ylim(lower, 1.2)
            axis.grid(True, which="major", alpha=0.22)
            axis.set_xlabel("molecular hop distance")
            if column == 0:
                axis.set_ylabel(f"{DISPLAY[task]}\nnormalised profile")
            axis.set_title(("semantic donor", "PE transposition", "topology donor")[column])
            if row == 0 and column == 0:
                axis.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Score mass versus per-opportunity sensitivity and clean attention locality\n"
        "(bands: graph-bootstrap 95% intervals)",
        fontsize=11,
    )
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def figure_distance_usage_coupling(
    runs: Mapping[str, Mapping[str, Any]],
    profiles: Mapping[tuple[str, str], Mapping[str, Any]],
    cfg: BetaConfig,
    path: Path,
) -> list[str]:
    """Align frozen-family head reach with final-state F_sens and signed B."""

    plt = configure_matplotlib()
    fig, axes = plt.subplots(4, 3, figsize=(12.0, 11.0), constrained_layout=True)
    carriage_profiles = {
        (task, channel): model_carriage_profile(
            runs[task]["scores"],
            channel,
            bootstrap_samples=cfg.bootstrap_samples,
            seed=cfg.analysis_seed + 12_701 + 101 * row + column,
        )
        for row, task in enumerate(ARCHITECTURES)
        for column, channel in enumerate(CHANNELS)
    }
    b_values = np.concatenate([
        np.asarray(profile[key], float).reshape(-1)
        for profile in carriage_profiles.values()
        for key in ("B_ci_low", "B_ci_high")
    ])
    finite_b = np.abs(b_values[np.isfinite(b_values)])
    nonzero_b = finite_b[finite_b > 0]
    b_bound = max(float(np.max(finite_b)) * 1.25 if len(finite_b) else 1.0, 1e-12)
    b_linthresh = max(
        float(np.quantile(nonzero_b, 0.15)) if len(nonzero_b) else b_bound * 1e-3,
        1e-12,
    )
    family_styles = {
        "semantic_specialist": ("#c75b5b", "-"),
        "high_G_balanced": ("#55a868", ":"),
        "structural_pe_specific": ("#4c78a8", "-"),
        "structural_topology_specific": ("#6f63a8", "-"),
        "structural_shared": ("#55a868", ":"),
    }
    for architecture_index, task in enumerate(ARCHITECTURES):
        families = runs[task]["ablations"]["selections"][HEADLINE_AGGREGATION]["families"]
        for column, channel in enumerate(CHANNELS):
            profile = profiles[(task, channel)]
            numeric = np.asarray(profile["codes"], int) >= 0
            x = np.asarray(profile["codes"], int)[numeric]
            contribution = np.asarray(profile["contribution"], float)[numeric]
            top = axes[2 * architecture_index, column]
            all_head = contribution.sum(axis=(1, 2))
            top.plot(
                x,
                all_head / max(float(all_head.sum()), EPS),
                color="0.45",
                marker="o",
                linewidth=1.4,
                label="all-head EG",
            )
            for family in DISTANCE_FAMILIES[channel]:
                heads = [tuple(map(int, head)) for head in families.get(family, [])]
                if not heads:
                    continue
                values = np.asarray([
                    sum(float(contribution[index, head[0], head[1]]) for head in heads)
                    for index in range(len(x))
                ])
                color, style = family_styles[family]
                top.plot(
                    x,
                    values / max(float(values.sum()), EPS),
                    linestyle=style,
                    marker="s",
                    color=color,
                    linewidth=1.2,
                    label=family.replace("structural_", "").replace("_", " "),
                )
            carriage = carriage_profiles[(task, channel)]
            f = np.asarray(carriage["F_sens"], float)
            top.plot(
                carriage["distance"],
                f / max(float(np.nansum(f)), EPS),
                "--^",
                color="black",
                linewidth=1.4,
                label="final-state F_sens",
            )
            top.set_ylim(bottom=0.0)
            top.set_ylabel(
                f"{DISPLAY[task]}\nnormalised distance mass" if column == 0
                else "normalised distance mass"
            )
            top.set_title(("semantic donor", "PE transposition", "topology donor")[column])
            top.legend(frameon=False, fontsize=6.5)

            bottom = axes[2 * architecture_index + 1, column]
            bottom.axhline(0, color="black", linewidth=0.7)
            bottom.fill_between(
                carriage["distance"], carriage["B_ci_low"], carriage["B_ci_high"],
                color="0.45", alpha=0.18,
            )
            bottom.plot(carriage["distance"], carriage["B"], "o-", color="0.25")
            bottom.set_yscale("symlog", linthresh=b_linthresh)
            bottom.set_ylim(-b_bound, b_bound)
            bottom.set_xlabel("distance to changed set [hops]")
            bottom.set_ylabel(
                "signed beneficial B\n(B<0 beneficial)" if column == 0
                else "signed beneficial B"
            )
    fig.suptitle(
        "Head-score reach above; functional reach and task-level usage on the same distance axis",
        fontsize=11,
    )
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def figure_model_functional_carriage(
    runs: Mapping[str, Mapping[str, Any]], cfg: BetaConfig, path: Path
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True, sharey=True)
    profiles = {}
    positive = []
    for row, task in enumerate(ARCHITECTURES):
        for column, channel in enumerate(CHANNELS):
            profile = model_carriage_profile(
                runs[task]["scores"], channel,
                bootstrap_samples=cfg.bootstrap_samples,
                seed=cfg.analysis_seed + 101 * row + column,
            )
            profiles[(task, channel)] = profile
            for key in ("F_sens", "F_sens_ci_low", "F_sens_ci_high", "F_coh"):
                values = np.asarray(profile[key], float)
                positive.extend(values[np.isfinite(values) & (values > 0)])
    low = max(float(np.min(positive)) * 0.7, 1e-12) if positive else 1e-12
    high = float(np.max(positive)) * 1.4 if positive else 1.0
    colors = {"semantic": "#c75b5b", "pe": "#4c78a8", "topology": "#6f63a8"}
    for row, task in enumerate(ARCHITECTURES):
        for column, channel in enumerate(CHANNELS):
            axis = axes[row, column]
            profile = profiles[(task, channel)]
            x = profile["distance"]
            axis.fill_between(x, profile["F_sens_ci_low"], profile["F_sens_ci_high"],
                              color=colors[channel], alpha=0.18)
            axis.plot(x, profile["F_sens"], marker="o", color=colors[channel], label="F_sens")
            axis.plot(x, profile["F_coh"], linestyle="--", color="0.35", label="F_coh diagnostic")
            axis.set_yscale("log"); axis.set_ylim(low, high)
            axis.set_xlabel("distance to changed set [hops]")
            if column == 0:
                axis.set_ylabel(f"{DISPLAY[task]}\nfunctional carriage")
            axis.set_title(("semantic donor", "PE transposition", "topology donor")[column])
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Final-state functional carriage: production F_sens on a shared log scale", fontsize=11)
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def figure_model_beneficial_carriage(
    runs: Mapping[str, Mapping[str, Any]], cfg: BetaConfig, path: Path
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True, sharey=True)
    profiles = {
        (task, channel): model_carriage_profile(
            runs[task]["scores"], channel,
            bootstrap_samples=cfg.bootstrap_samples,
            seed=cfg.analysis_seed + 401 + 101 * row + column,
        )
        for row, task in enumerate(ARCHITECTURES)
        for column, channel in enumerate(CHANNELS)
    }
    values = np.concatenate([
        np.asarray(profile[key], float).reshape(-1)
        for profile in profiles.values()
        for key in ("B_ci_low", "B_ci_high")
    ])
    finite = np.abs(values[np.isfinite(values)])
    nonzero = finite[finite > 0]
    bound = max(float(np.max(finite)) * 1.25 if len(finite) else 1.0, 1e-12)
    linthresh = max(float(np.quantile(nonzero, 0.15)) if len(nonzero) else bound * 1e-3, 1e-12)
    colors = {"semantic": "#c75b5b", "pe": "#4c78a8", "topology": "#6f63a8"}
    for row, task in enumerate(ARCHITECTURES):
        for column, channel in enumerate(CHANNELS):
            axis = axes[row, column]
            profile = profiles[(task, channel)]
            x = profile["distance"]
            axis.axhline(0, color="black", linewidth=0.7)
            axis.fill_between(x, profile["B_ci_low"], profile["B_ci_high"],
                              color=colors[channel], alpha=0.18)
            axis.plot(x, profile["B"], marker="o", color=colors[channel])
            axis.set_yscale("symlog", linthresh=linthresh)
            axis.set_ylim(-bound, bound)
            axis.set_xlabel("distance to changed set [hops]")
            if column == 0:
                axis.set_ylabel(f"{DISPLAY[task]}\nbeneficial carriage B")
            axis.set_title(("semantic donor", "PE transposition", "topology donor")[column])
    fig.suptitle(
        "Path-integrated beneficial carriage on shared signed-log limits (B<0 beneficial)", fontsize=11
    )
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


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
        rows = sorted(
            conditional[task].get("confirmed", []),
            key=lambda row: (
                not bool(row.get("confirmed_at_fdr_0.05", False)),
                float(row.get("confirmation_q_global", float("inf"))),
                -abs(float(row.get("confirmation_effect_standardized", 0.0))),
            ),
        )[:8]
        if not rows:
            audit = conditional[task].get("eligibility_audit", {})
            if int(audit.get("observations", 0)) == 0:
                message = "Screen unavailable:\nno aligned three-channel observations"
            elif int(audit.get("rules_passing_prevalence", 0)) == 0:
                message = "No condition met\nthe prevalence gate"
            elif int(audit.get("support_eligible_feature_rule_heads", 0)) == 0:
                message = "No condition met\nthe per-state graph-support gate"
            elif int(audit.get("activity_eligible_feature_rule_heads", 0)) == 0:
                message = "No condition/head met\nthe activity gate"
            else:
                message = "No condition entered\nheld-out confirmation"
            axis.text(0.5, 0.5, message, ha="center", va="center")
            axis.set_axis_off()
            continue
        labels = [
            f"{row['feature']} · {row['rule']}\nL{row['layer']}H{row['head']}" for row in rows
        ]
        y = np.arange(len(rows))
        discovery = np.asarray([row["discovery_effect_standardized"] for row in rows])
        confirmation = np.asarray([row["confirmation_effect_standardized"] for row in rows])
        low = np.maximum(
            confirmation - np.asarray([row["confirmation_ci_low_standardized"] for row in rows]), 0.0
        )
        high = np.maximum(
            np.asarray([row["confirmation_ci_high_standardized"] for row in rows]) - confirmation, 0.0
        )
        axis.scatter(discovery, y - 0.12, marker="x", color="0.3", label="discovery")
        seen_confirmation_labels: set[str] = set()
        for index, row in enumerate(rows):
            confirmed = bool(row.get("confirmed_at_fdr_0.05", False))
            state_label = "FDR-confirmed" if confirmed else "not confirmed"
            axis.errorbar(
                confirmation[index],
                y[index] + 0.12,
                xerr=[[low[index]], [high[index]]],
                fmt="o",
                color="#3b7ddd" if confirmed else "0.55",
                markerfacecolor="#3b7ddd" if confirmed else "white",
                label=state_label if state_label not in seen_confirmation_labels else None,
            )
            seen_confirmation_labels.add(state_label)
        axis.axvline(0, color="black", linewidth=0.7)
        axis.set_yticks(y, labels)
        axis.set_xlabel("standardised conditional score delta")
        axis.set_title(DISPLAY[task])
        axis.invert_yaxis()
    axes[0].legend(frameon=False)
    fig.suptitle("Sample-split conditional-specialisation screen", fontsize=11)
    outputs = save_figure(fig, path)
    plt.close(fig)
    return outputs


def jaccard(left: Sequence[Any], right: Sequence[Any]) -> float:
    a, b = set(left), set(right)
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
            if row.get("scope") in {"pooled", "within_layer"}
            and row["aggregation"] == aggregation
        ]
        selections = runs[task]["ablations"]["selections"][aggregation]["families"]
        method[task] = {
            "aggregation_diagnostics": rows,
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
        for family, matched_channel, other_channels in (
            ("semantic_specialist", "semantic", ("pe", "topology")),
            ("structural_pe_specific", "pe", ("semantic", "topology")),
            ("structural_topology_specific", "topology", ("semantic", "pe")),
        ):
            heads = families.get(family, [])
            if not heads:
                patch_advantages.append(float("nan"))
                continue
            for metric in ("restore", "inject", "necessity"):
                matrix = family_channel_matrix(causal, families, [family], metric)
                matched = float(matrix[0, CHANNELS.index(matched_channel)])
                off_channel = np.nanmax([
                    matrix[0, CHANNELS.index(channel)] for channel in other_channels
                ])
                mismatch = (
                    float(family_channel_matrix(causal, families, [family], "mismatch")[
                        0, CHANNELS.index(matched_channel)
                    ])
                    if metric == "restore" else 0.0
                )
                patch_advantages.append(float(matched - max(off_channel, mismatch)))
    patch_valid = all(np.isfinite(value) and value > 0.0 for value in patch_advantages)
    return {
        "overall_aggregation": aggregation,
        "aggregation_rule": "EG is predeclared; this rerun performs no aggregation reselection",
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
        "routing_message_wiring": {
            "verdict": "retain_support_aware_decomposition" if mechanism_valid else "descriptive_only",
            "maximum_exact_reconstruction_error": mechanism_max,
            "mean_full_rescue_by_architecture": mechanism_full,
            "finite_interaction_to_full_ratio": mechanism_interaction_ratio,
            "topology_convention": "common-support routing/message plus explicit exclusive-support wiring",
        },
        "carriage": (
            "F_sens is the fixed headline; F_coh is diagnostic; all channels use event-specific "
            "changed-set distance and shared log-scale figures"
        ),
        "beneficial_carriage": (
            "signed path-integrated carrier attribution is computed for semantic, PE and topology "
            "events and shown on shared symlog limits; no absolute-value replacement"
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
    method_rows = []
    for task in ARCHITECTURES:
        for row in runs[task]["ablations"]["method_rows"]:
            method_rows.append({"architecture": task, **row})
    overall = HEADLINE_AGGREGATION
    write_csv(tables / "aggregation_validation.csv", method_rows)

    topology_rows = []
    ablation_rows = []
    causal_rows = []
    mechanism_rows = []
    family_patch_rows = []
    family_ablation_rows = []
    carriage_rows = []
    beneficial_diagnostic_rows = []
    conditional = {}
    event_rows = []
    topology_strata_rows = []
    topology_agreement = {}
    distance_profiles: dict[tuple[str, str], dict[str, Any]] = {}
    attention_profiles: dict[str, dict[str, Any]] = {}
    distance_score_rows = []
    attention_distance_table_rows = []
    locality_summary_rows = []
    head_reach_rows = []
    family_reach_rows = []
    distance_identity_rows = []
    integrated_audit_rows = []
    for task in ARCHITECTURES:
        attention_profile = attention_distance_profile(
            runs[task]["attention"],
            bootstrap_samples=cfg.bootstrap_samples,
            seed=cfg.analysis_seed + 10_301 + 997 * ARCHITECTURES.index(task),
        )
        score_graph_ids = np.asarray(
            [row["graph_id"] for row in runs[task]["scores"]["records"]],
            dtype=np.int64,
        )
        if not np.array_equal(
            np.sort(score_graph_ids), np.sort(attention_profile["graph_ids"])
        ):
            raise RuntimeError(
                f"clean-attention graphs do not match score graphs for {task}"
            )
        attention_profiles[task] = attention_profile
        attention_distance_table_rows.extend(
            attention_distance_rows(task, attention_profile)
        )
        event_rows.extend(intervention_event_rows(task, runs[task]["scores"]))
        task_integrated_audit = runs[task]["scores"].get(
            "integrated_carriage_audit",
            integrated_carriage_audit(runs[task]["scores"]["records"]),
        )
        integrated_audit_rows.append({"architecture": task, **task_integrated_audit})
        for record in runs[task]["scores"]["records"]:
            for channel in CHANNELS:
                result = record["channels"][channel]
                for source_index, diagnostic in enumerate(
                    result.get("beneficial_diagnostics", [])
                ):
                    replay = diagnostic.get("endpoint_replay") or {}
                    beneficial_diagnostic_rows.append({
                        "architecture": task,
                        "graph_id": int(record["graph_id"]),
                        "channel": channel,
                        "source_index": int(source_index),
                        "paths": int(diagnostic["paths"]),
                        "unconverged": int(diagnostic["unconverged"]),
                        "unconverged_fraction": float(
                            diagnostic.get("unconverged_fraction", 0.0)
                        ),
                        "completeness_max": float(diagnostic["completeness_max"]),
                        "quadrature_error_max": float(diagnostic["quadrature_error_max"]),
                        "unconverged_completeness_max": float(
                            diagnostic.get("unconverged_completeness_max", 0.0)
                        ),
                        "unconverged_quadrature_error_max": float(
                            diagnostic.get("unconverged_quadrature_error_max", 0.0)
                        ),
                        "intervals_max": int(diagnostic.get("intervals_max", 0)),
                        "endpoint_replay_passed": bool(replay.get("passed", False)),
                        "endpoint_replay_max_abs_error": float(
                            replay.get("max_abs_error", float("nan"))
                        ),
                    })
        summary = topology_summary(runs[task]["scores"], runs[task]["causal"], overall)
        topology_rows.append({
            "architecture": task,
            **{key: value for key, value in summary.items() if np.isscalar(value) or isinstance(value, dict)},
        })
        topology_agreement[task] = topology_agreement_analysis(
            runs[task]["scores"], cfg, overall
        )
        write_json(tables / f"topology_agreement_{task}.json", topology_agreement[task])
        topology_strata_rows.extend({"architecture": task, **row} for row in
                                    topology_stratified_analysis(
                                        runs[task]["scores"], runs[task]["causal"], overall
                                    ))
        selection = runs[task]["ablations"]["selections"][overall]
        families = selection["families"]
        for channel_index, channel in enumerate(CHANNELS):
            donor_scopes = ["headline"]
            if channel == "topology":
                donor_scopes.extend(sorted({
                    f"tier_{int(item['tier'])}"
                    for record in runs[task]["scores"]["records"]
                    for item in record["plan"]["topology"]
                }))
            for scope_index, donor_scope in enumerate(donor_scopes):
                profile = distance_specialisation_profile(
                    runs[task]["scores"],
                    channel,
                    donor_scope=donor_scope,
                    bootstrap_samples=cfg.bootstrap_samples,
                    seed=(
                        cfg.analysis_seed + 10_901 + 997 * ARCHITECTURES.index(task)
                        + 31 * channel_index + scope_index
                    ),
                )
                distance_score_rows.extend(distance_specialisation_rows(task, profile))
                head_reach_rows.extend(head_reach_summary_rows(task, profile))
                distance_identity_rows.append({
                    "architecture": task,
                    "channel": channel,
                    "donor_scope": donor_scope,
                    "graphs": len(profile["graph_ids"]),
                    "identity_passed": bool(profile["identity_passed"]),
                    "maximum_absolute_reconstruction_error": float(
                        profile["identity_max_abs_error"]
                    ),
                })
                if donor_scope == "headline":
                    distance_profiles[(task, channel)] = profile
                    locality_summary_rows.extend(
                        locality_curve_rows(
                            task, channel, profile, attention_profile
                        )
                    )
                    final_carriage = model_carriage_profile(
                        runs[task]["scores"],
                        channel,
                        bootstrap_samples=cfg.bootstrap_samples,
                        seed=cfg.analysis_seed + 11_701 + 101 * channel_index,
                    )
                    family_reach_rows.extend(family_distance_reach_rows(
                        task, profile, families, final_carriage
                    ))
        causal = runs[task]["causal"]
        for family in causal.get("family_names", []):
            score_heads = [tuple(map(int, head)) for head in families.get(family, [])]
            causal_heads = [
                tuple(map(int, head))
                for head in causal.get("family_definitions", {}).get(family, [])
            ]
            if score_heads != causal_heads:
                raise RuntimeError(
                    f"frozen family drift for {task}/{family}: "
                    f"score={score_heads}, causal={causal_heads}"
                )
        coordinates = selection["coordinates"]
        semantic_topology_coordinates = score_coordinates(
            selection["semantic_topology_semantic"],
            selection["semantic_topology_topology"],
        )
        pe_topology_coordinates = selection["pe_topology_coordinates"]
        impact = runs[task]["ablations"]["per_head"]
        for layer in range(int(runs[task]["scores"]["L"])):
            for head in range(int(runs[task]["scores"]["H"])):
                ablation_rows.append({
                    "architecture": task, "layer": layer, "head": head,
                    "S_sem": selection["semantic"][layer, head],
                    "S_pe": selection["pe"][layer, head],
                    "S_topology": selection["topology"][layer, head],
                    "S_semantic_on_topology_paired_graphs": selection[
                        "semantic_topology_semantic"
                    ][layer, head],
                    "S_pe_on_topology_paired_graphs": selection["pe_topology_pe"][layer, head],
                    "S_topology_on_semantic_paired_graphs": selection[
                        "semantic_topology_topology"
                    ][layer, head],
                    "S_topology_on_pe_paired_graphs": selection[
                        "pe_topology_topology"
                    ][layer, head],
                    "D": coordinates["D"][layer, head],
                    "J": coordinates["J"][layer, head],
                    "G": coordinates["G"][layer, head],
                    "D_semantic_topology": semantic_topology_coordinates["D"][layer, head],
                    "J_semantic_topology": semantic_topology_coordinates["J"][layer, head],
                    "G_semantic_topology": semantic_topology_coordinates["G"][layer, head],
                    "D_pe_topology": pe_topology_coordinates["D"][layer, head],
                    "J_pe_topology": pe_topology_coordinates["J"][layer, head],
                    "G_pe_topology": pe_topology_coordinates["G"][layer, head],
                    "D_pe_topology_ci_low": selection["pe_topology_D_ci_lower"][layer, head],
                    "D_pe_topology_ci_high": selection["pe_topology_D_ci_upper"][layer, head],
                    "active": bool(selection["active"][layer, head]),
                    "active_pe_topology": bool(selection["pe_topology_active"][layer, head]),
                    "families": ";".join(
                        family for family, heads in families.items()
                        if (layer, head) in [tuple(map(int, item)) for item in heads]
                    ),
                    "functional_ablation": impact["functional"][layer, head],
                    "loss_ablation": impact["loss_increase"][layer, head],
                })
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
                    matrix = family_channel_matrix(
                        causal, families, [family], metric
                    )
                    family_patch_rows.append({
                        "architecture": task,
                        "family": family,
                        "heads": str(list(heads)),
                        "channel": channel,
                        "metric": metric,
                        "mean": float(matrix[0, CHANNELS.index(channel)]) if heads else float("nan"),
                        "simultaneous_family_patch": family in causal.get("family_names", []),
                    })
            for row in runs[task]["ablations"]["family_curves"][overall].get(family, []):
                family_ablation_rows.append({"architecture": task, "family": family, **row})
        mechanism = runs[task]["mechanism"]
        for channel in CHANNELS:
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
        for channel in CHANNELS:
            donor_scopes = ["headline"]
            if channel == "topology":
                donor_scopes.extend(sorted({
                    str(item.get("donor_scope"))
                    for record in runs[task]["scores"]["records"]
                    for item in record["channels"][channel].get("model_carriage", [])
                    if str(item.get("donor_scope", "")).startswith("tier_")
                }))
            for scope_index, donor_scope in enumerate(donor_scopes):
                profile = model_carriage_profile(
                    runs[task]["scores"],
                    channel,
                    donor_scope=donor_scope,
                    bootstrap_samples=cfg.bootstrap_samples,
                    seed=(
                        cfg.analysis_seed + 717 + 31 * scope_index
                        + CHANNELS.index(channel)
                    ),
                )
                for index, distance in enumerate(profile["distance"]):
                    carriage_rows.append({
                        "architecture": task,
                        "channel": channel,
                        "donor_scope": donor_scope,
                        "level": "final_state_model",
                        "distance": int(distance),
                        "F_sens": float(profile["F_sens"][index]),
                        "F_coh": float(profile["F_coh"][index]),
                        "F_sens_ci_low": float(profile["F_sens_ci_low"][index]),
                        "F_sens_ci_high": float(profile["F_sens_ci_high"][index]),
                        "F_coh_ci_low": float(profile["F_coh_ci_low"][index]),
                        "F_coh_ci_high": float(profile["F_coh_ci_high"][index]),
                        "B": float(profile["B"][index]),
                        "B_ci_low": float(profile["B_ci_low"][index]),
                        "B_ci_high": float(profile["B_ci_high"][index]),
                        "B_sum": float(profile["B_sum"][index]),
                        "graph_support": int(profile["graph_support"][index]),
                    })
        conditional[task] = conditional_analysis(runs[task]["scores"], overall, cfg)
        write_json(tables / f"conditional_{task}.json", conditional[task])
    write_csv(tables / "topology_validation.csv", topology_rows)
    write_csv(tables / "topology_tier_dose_validation.csv", topology_strata_rows)
    write_csv(tables / "head_scores_and_ablation.csv", ablation_rows)
    write_csv(tables / "causal_head_metrics.csv", causal_rows)
    write_csv(tables / "family_patch_metrics.csv", family_patch_rows)
    write_csv(tables / "family_ablation_curves.csv", family_ablation_rows)
    write_csv(tables / "routing_message_metrics.csv", mechanism_rows)
    write_csv(tables / "carriage_profiles.csv", carriage_rows)
    write_csv(tables / "beneficial_carriage_diagnostics.csv", beneficial_diagnostic_rows)
    write_csv(tables / "integrated_carriage_audit.csv", integrated_audit_rows)
    write_csv(tables / "intervention_events_and_dose.csv", event_rows)
    write_csv(tables / "donor_outcome_summary.csv", donor_outcome_summary_rows(event_rows))
    write_csv(tables / "distance_resolved_specialisation.csv", distance_score_rows)
    write_csv(tables / "clean_attention_distance_profiles.csv", attention_distance_table_rows)
    write_csv(tables / "distance_locality_summary_curves.csv", locality_summary_rows)
    write_csv(tables / "head_reach_summary.csv", head_reach_rows)
    write_csv(tables / "family_reach_alignment.csv", family_reach_rows)
    write_csv(tables / "distance_score_identity.csv", distance_identity_rows)

    figure_paths = []
    figure_paths += figure_three_channel_planes(runs, figures / "zinc_headline_fig01_three_channel_planes")
    figure_paths += figure_topology_agreement(runs, cfg, figures / "zinc_headline_fig02_topology_agreement")
    figure_paths += figure_topology_strata(runs, figures / "zinc_headline_fig03_topology_tier_dose")
    figure_paths += figure_family_patching(runs, overall, figures / "zinc_headline_fig04_family_patching")
    figure_paths += figure_patch_controls(runs, figures / "zinc_headline_fig05_patch_controls")
    figure_paths += figure_ablation(runs, overall, figures / "zinc_headline_fig06_ablation")
    figure_paths += figure_routing_message(runs, overall, figures / "zinc_headline_fig07_mechanism")
    figure_paths += figure_model_functional_carriage(
        runs, cfg, figures / "zinc_headline_fig08_functional_carriage"
    )
    figure_paths += figure_model_beneficial_carriage(
        runs, cfg, figures / "zinc_headline_fig09_beneficial_carriage"
    )
    figure_paths += figure_conditional(conditional, figures / "zinc_headline_fig10_conditional")
    figure_paths += figure_causal_coordinates(runs, overall, figures / "zinc_headline_fig11_causal_coordinates")
    figure_paths += figure_distance_specialisation_atlas(
        distance_profiles, figures / "zinc_headline_fig12_distance_specialisation"
    )
    figure_paths += figure_distance_usage_coupling(
        runs, distance_profiles, cfg, figures / "zinc_headline_fig13_reach_usage_coupling"
    )
    figure_paths += figure_support_normalised_distance_atlas(
        distance_profiles,
        figures / "zinc_headline_fig14_support_normalised_distance_specialisation",
    )
    figure_paths += figure_attention_distance_atlas(
        attention_profiles,
        figures / "zinc_headline_fig15_attention_distance_locality",
    )
    figure_paths += figure_distance_locality_curves(
        distance_profiles,
        attention_profiles,
        figures / "zinc_headline_fig16_distance_locality_curves",
    )

    decisions = create_decisions(runs, overall)
    confirmed_rules = {
        task: {
            f"{row['feature']}::{row['rule']}" for row in conditional[task].get("confirmed", [])
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
    decisions["distance_resolved_specialisation"] = {
        "verdict": "retain_exact_decomposition",
        "estimand": (
            "S_EG(layer,head,distance): event magnitude is summed over carriers in the "
            "event-specific distance bucket, then events, sources and graphs are averaged"
        ),
        "methodological_status": "exact decomposition of EG, not a new score",
        "identity_max_abs_error": float(max(
            row["maximum_absolute_reconstruction_error"] for row in distance_identity_rows
        )),
        "hub_rule": "virtual-node transport is assigned to a separate hub bucket",
        "interpretation": (
            "identifies which heads and frozen families implement intervention reach; final-state "
            "F_sens and signed B remain the functional and task-usage endpoints"
        ),
        "support_normalised_diagnostic": (
            "divide each graph's exact distance-bucket EG contribution by its matched mean "
            "event-carrier opportunity, then graph-average; this diagnoses per-opportunity "
            "functional locality without replacing or renormalising the headline EG score"
        ),
        "attention_locality_control": (
            "clean post-softmax attention mass is binned by pristine molecular query-key "
            "distance, normalised within graph/layer/head and graph-bootstrap aggregated"
        ),
        "attention_normalisation_error_max": float(max(
            profile["normalisation_error_max"] for profile in attention_profiles.values()
        )),
    }
    write_json(tables / "methodology_decisions.json", decisions)
    scope = {
        "primary_comparison": "fixed EG analysis of semantic, mask-frozen PE and matched topology channels",
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
        "aggregation_choice": "predeclared headline EG; no data-dependent reselection",
        "functional_carriage_choice": "predeclared F_sens; F_coh retained diagnostically",
        "distance_specialisation_choice": (
            "exact EG-by-distance decomposition retained; support-normalised EG and raw "
            "attention locality are explicitly diagnostic companion views"
        ),
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
            "attention": find_cached_phase(cfg, task, "attention"),
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
        "conditional_per_feature": args.conditional_per_feature,
        "conditional_max_tests": args.conditional_max_tests,
        "analysis_seed": args.analysis_seed,
        "device": args.device,
        "dense_checkpoint": args.dense_checkpoint,
        "onehop_checkpoint": args.onehop_checkpoint,
        "allow_relaxed_topology": not args.strict_topology_only,
        "causal_effect_floor_relative": args.causal_effect_floor_relative,
        "topk_values": tuple(
            int(value.strip()) for value in args.top_k_values.split(",") if value.strip()
        ),
        "integrated_atol": args.integrated_atol,
        "integrated_rtol": args.integrated_rtol,
        "integrated_max_intervals": args.integrated_max_intervals,
        "integrated_unconverged_error_cap": args.integrated_unconverged_error_cap,
        "integrated_max_unconverged_fraction": args.integrated_max_unconverged_fraction,
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
            "conditional_per_feature": 1,
            "conditional_max_tests": 16,
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
        choices=(
            "all", "scores", "attention", "causal", "mechanism", "ablations", "figures"
        ),
        default="all",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--score-graphs", type=int, default=96)
    parser.add_argument("--score-sources", type=int, default=8)
    parser.add_argument("--semantic-donors", type=int, default=8)
    parser.add_argument("--pe-partners", type=int, default=8)
    parser.add_argument("--topology-donors", type=int, default=6)
    parser.add_argument("--topology-pool", type=int, default=10_000)
    parser.add_argument("--causal-graphs", type=int, default=48)
    parser.add_argument("--causal-sources", type=int, default=2)
    parser.add_argument("--causal-batch-graphs", type=int, default=8)
    parser.add_argument("--mechanism-graphs", type=int, default=16)
    parser.add_argument("--ablation-graphs", type=int, default=128)
    parser.add_argument("--family-size", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--conditional-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--conditional-per-feature", type=int, default=2)
    parser.add_argument("--conditional-max-tests", type=int, default=48)
    parser.add_argument("--analysis-seed", type=int, default=1771)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dense-checkpoint", default=None)
    parser.add_argument("--onehop-checkpoint", default=None)
    parser.add_argument("--strict-topology-only", action="store_true")
    parser.add_argument("--causal-effect-floor-relative", type=float, default=0.05)
    parser.add_argument("--top-k-values", default="3,5,10")
    parser.add_argument("--integrated-atol", type=float, default=5.0e-4)
    parser.add_argument("--integrated-rtol", type=float, default=1.0e-4)
    parser.add_argument("--integrated-max-intervals", type=int, default=256)
    parser.add_argument("--integrated-unconverged-error-cap", type=float, default=5.0e-3)
    parser.add_argument("--integrated-max-unconverged-fraction", type=float, default=1.0e-2)
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
                if args.phase in {"all", "attention"}:
                    if score_payload is None:
                        score_payload = find_cached_phase(cfg, task, "scores")
                    run_attention_task(
                        loaded, cfg, score_payload, force=args.force
                    )
                if args.phase in {"all", "causal"}:
                    if score_payload is None:
                        score_payload = find_cached_phase(cfg, task, "scores")
                    causal_payload = run_causal_task(
                        loaded, cfg, score_payload, force=args.force
                    )
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
