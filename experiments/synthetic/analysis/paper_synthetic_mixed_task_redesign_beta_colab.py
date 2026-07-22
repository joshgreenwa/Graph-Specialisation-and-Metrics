"""Standalone Colab beta: validate successor graph-head specialisation metrics.

Paste this complete file into one Google Colab cell.  It mounts Drive, checks out this
repository, loads (never retrains or overwrites) the three trained ``cycle_dual_v2``
checkpoints, and writes a versioned beta analysis beside the legacy experiment.

The experiment compares the full coherent/eventwise x gross/net aggregation family,
tests raw scores and D/J/G against independent ablation and bidirectional whole-wV
patching, audits reliability and intervention dose, runs sample-split conditional-rule
discovery, decomposes fixed-support transport into exact routing/message terms, and
compares score-selected family ablations.  All expensive products are resumable.

Scientific scope
----------------
This trained task is a necessary synthetic screen for score aggregation and causal/mechanistic
validation.  It has one planted source per graph, a query-only readout, identical cycle
topologies, and no planted context gate.  Consequently it can exercise, but cannot by
itself approve, multi-source hierarchical weighting, production integrated beneficial
carriage, non-isomorphic structural donors, or the sensitivity of general conditional
specialist discovery.  The generated decision report states these limitations.
The score stage evaluates the full task-mode × intervention-factor 2×2 design; expensive patching
uses the matched diagonal and is therefore labelled matched task-factor mediation.

No beta setting enters the legacy checkpoint fingerprint.  Inputs are read from

    .../causal_specialisation_double_dissociation/cycle_dual_v2

and every beta write is confined to

    .../cycle_dual_v2/redesign_beta_v1

Use ``--phase scores``, ``causal``, ``mechanism``, ``families``, or ``figures`` for
resumable execution; ``all`` runs them in that order.  Paper claims require all three
seeds and the default graph counts.  ``--fast-dev-run`` is plumbing only.
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
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

import numpy as np


BETA_VERSION = "mixed-task-specialisation-redesign-beta-v3"
BETA_SCHEMA = 3
LEGACY_VERSION = "causal-specialisation-double-dissociation-v2-shared-source-marker"
REPOSITORY_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
REPOSITORY_BRANCH = "codex/cfim-grit-experiments"
COLAB_REPOSITORY = Path("/content/Graph-Specialisation-and-Metrics")
LEGACY_RELATIVE = Path(
    "experiments/synthetic/training/causal_specialisation_double_dissociation_colab.py"
)
DEFAULT_DRIVE_ROOT = Path(
    "/content/drive/MyDrive/graph_specialisation_metrics/"
    "causal_specialisation_double_dissociation"
)
DEFAULT_LEGACY_RUN = "cycle_dual_v2"
DEFAULT_BETA_RUN = "redesign_beta_v1"
DEFAULT_GRIT_DIR = Path("/content/GRIT")
SECRET_NAME = "dissertation_key"
AGGREGATIONS = ("CG", "EG", "CN", "EN")
CHANNELS = ("semantic", "structural")
EPS = 1.0e-12
# Equivalence checks compare independent CUDA forwards.  PyG/GRIT scatter kernels can
# differ by a few ulps between otherwise identical batched executions, especially on
# newer Torch/CUDA stacks, so validity gates must use an absolute+relative tolerance
# rather than a brittle, scale-free max-error cutoff.
CUDA_EQUIVALENCE_RTOL = 2.0e-5
CG_REGRESSION_ATOL = 1.0e-4
FORWARD_EQUIVALENCE_ATOL = 1.0e-4
MECHANISM_RECONSTRUCTION_ATOL = 2.0e-4


# ======================================================================================
# Bootstrap and serialisation
# ======================================================================================


def _command(*parts: str, check: bool = True) -> None:
    shown = [
        "https://***@github.com/" + part.split("@github.com/", 1)[1]
        if "@github.com/" in part
        else part
        for part in parts
    ]
    print(f"[cmd] {' '.join(shown)}", flush=True)
    subprocess.run(list(parts), check=check)


def bootstrap_repository(*, branch: str, skip_checkout: bool) -> Path:
    """Mount Drive and make the repository importable in a pasted Colab cell."""

    try:
        from google.colab import drive, userdata  # type: ignore

        drive.mount("/content/drive", force_remount=False)
        try:
            token = userdata.get(SECRET_NAME) or os.environ.get(SECRET_NAME)
        except Exception as exc:
            print(f"[bootstrap] Colab secret unavailable ({exc}); trying public clone", flush=True)
            token = os.environ.get(SECRET_NAME)
    except ImportError:
        token = os.environ.get(SECRET_NAME)
        if not skip_checkout:
            print("[bootstrap] not in Colab; using the current repository", flush=True)
            if "__file__" in globals():
                return Path(__file__).resolve().parents[3]
            return Path.cwd()

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
            _command(
                "git", "-C", str(COLAB_REPOSITORY), "remote", "set-url", "origin",
                authenticated,
            )
            _command("git", "-C", str(COLAB_REPOSITORY), "fetch", "origin", branch)
            _command("git", "-C", str(COLAB_REPOSITORY), "checkout", branch)
            _command(
                "git", "-C", str(COLAB_REPOSITORY), "reset", "--hard", f"origin/{branch}"
            )
        else:
            if COLAB_REPOSITORY.exists():
                shutil.rmtree(COLAB_REPOSITORY)
            _command(
                "git", "clone", "--branch", branch, "--single-branch", authenticated,
                str(COLAB_REPOSITORY),
            )
        _command(
            "git", "-C", str(COLAB_REPOSITORY), "remote", "set-url", "origin",
            REPOSITORY_URL,
        )
        root = COLAB_REPOSITORY

    _command(sys.executable, "-m", "pip", "install", "-q", "-e", str(root))
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    return root


def load_legacy_module(repository: Path) -> Any:
    path = repository / LEGACY_RELATIVE
    if not path.exists():
        raise FileNotFoundError(f"legacy mixed-task source is missing: {path}")
    name = "_mixed_task_legacy_readonly"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import legacy mixed-task source from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if module.EXPERIMENT_VERSION != LEGACY_VERSION:
        raise RuntimeError(
            f"legacy version changed: expected {LEGACY_VERSION}, got {module.EXPERIMENT_VERSION}"
        )
    return module


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except ImportError:
        pass
    raise TypeError(f"cannot JSON-encode {type(value)!r}")


def _tensor_equivalence_diagnostic(
    reference: Any,
    candidate: Any,
    *,
    atol: float,
    rtol: float = CUDA_EQUIVALENCE_RTOL,
) -> dict[str, Any]:
    """Return a JSON-safe, elementwise absolute+relative equivalence check."""

    import torch

    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "passed": False,
            "reason": f"shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}",
            "max_abs_error": float("inf"),
            "signal_at_max_error": float("nan"),
            "allowed_at_max_error": float(atol),
            "max_tolerance_ratio": float("inf"),
            "atol": float(atol),
            "rtol": float(rtol),
            "finite": False,
        }
    reference = reference.detach()
    candidate = candidate.detach()
    difference = (reference - candidate).abs()
    signal = torch.maximum(reference.abs(), candidate.abs())
    allowed = float(atol) + float(rtol) * signal
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    if not difference.numel():
        return {
            "passed": finite,
            "max_abs_error": 0.0,
            "signal_at_max_error": 0.0,
            "allowed_at_max_error": float(atol),
            "max_tolerance_ratio": 0.0,
            "atol": float(atol),
            "rtol": float(rtol),
            "finite": finite,
        }
    max_index = int(difference.reshape(-1).argmax())
    tolerance_ratio = difference / allowed.clamp_min(EPS)
    passed = finite and bool(torch.all(tolerance_ratio <= 1.0))
    return {
        "passed": passed,
        "max_abs_error": float(difference.max()),
        "signal_at_max_error": float(signal.reshape(-1)[max_index]),
        "allowed_at_max_error": float(allowed.reshape(-1)[max_index]),
        "max_tolerance_ratio": float(tolerance_ratio.max()),
        "atol": float(atol),
        "rtol": float(rtol),
        "finite": finite,
    }


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


def stable_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _beta_cache_is_current(
    payload: Mapping[str, Any], beta_cfg: "BetaConfig", checkpoint_sha256: str
) -> bool:
    return (
        payload.get("version") == BETA_VERSION
        and int(payload.get("schema", -1)) == BETA_SCHEMA
        and payload.get("fingerprint") == beta_cfg.fingerprint
        and payload.get("checkpoint_sha256") == checkpoint_sha256
    )


def _load_validated_beta_cache(
    path: Path, beta_cfg: "BetaConfig", checkpoint_sha256: str
) -> Any:
    import torch

    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not _beta_cache_is_current(
        payload, beta_cfg, checkpoint_sha256
    ):
        raise RuntimeError(
            f"stale/incompatible beta cache {path}; rerun its phase with --force"
        )
    return payload


# ======================================================================================
# Configuration and event manifests
# ======================================================================================


@dataclass(frozen=True)
class BetaConfig:
    legacy_root: str = str(DEFAULT_DRIVE_ROOT)
    legacy_run: str = DEFAULT_LEGACY_RUN
    beta_run: str = DEFAULT_BETA_RUN
    seeds: tuple[int, ...] = (0, 1, 2)
    score_graphs: int = 96
    score_batch_size: int = 8
    score_event_mode: str = "enumerate"
    score_sampled_events: int = 6
    causal_graphs: int = 96
    causal_events: int = 2
    causal_batch_size: int = 48
    mechanism_graphs: int = 32
    mechanism_events: int = 2
    family_graphs: int = 192
    family_size: int = 3
    bootstrap_samples: int = 1000
    conditional_bootstrap_samples: int = 1000
    min_condition_fraction: float = 0.25
    min_condition_graphs: int = 16
    activity_floor_relative: float = 0.10
    causal_effect_floor_relative: float = 0.05
    selectivity_threshold: float = 0.20
    j_floor: float = 0.50
    device: str = "cuda"

    def validate(self) -> None:
        if len(self.seeds) != 3:
            raise ValueError("paper protocol requires exactly three training seeds")
        if self.score_event_mode not in {"enumerate", "sample"}:
            raise ValueError("score_event_mode must be enumerate or sample")
        for name in (
            "score_graphs", "score_batch_size", "causal_graphs", "causal_events",
            "causal_batch_size", "mechanism_graphs", "mechanism_events", "family_graphs",
            "family_size", "bootstrap_samples", "conditional_bootstrap_samples",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.causal_batch_size % self.causal_events != 0:
            raise ValueError("causal_batch_size must contain whole graph event groups")
        graphs_per_causal_batch = self.causal_batch_size // self.causal_events
        if graphs_per_causal_batch < 2:
            raise ValueError("causal batches need at least two graphs for the mismatch control")
        if self.causal_graphs % graphs_per_causal_batch == 1:
            raise ValueError(
                "causal_graphs would leave a one-graph final batch; adjust causal_batch_size"
            )

    @property
    def legacy_run_dir(self) -> Path:
        return Path(self.legacy_root) / self.legacy_run

    @property
    def beta_dir(self) -> Path:
        return self.legacy_run_dir / self.beta_run

    @property
    def fingerprint(self) -> str:
        values = asdict(self)
        for key in ("legacy_root", "beta_run", "device"):
            values.pop(key, None)
        return stable_hash({"version": BETA_VERSION, **values})


@dataclass
class EventBundle:
    replicas: Any
    valid: Any
    source: Any
    partner: Any
    donor_value: Any
    dose: Any
    carrier_distance: Any
    clean_metadata: dict[str, Any]
    event_mode: str

    @property
    def graphs(self) -> int:
        return int(self.valid.size(0))

    @property
    def events(self) -> int:
        return int(self.valid.size(1))


def _node_value(legacy: Any, cfg: Any, batch: Any, graph: int, node: int) -> int:
    import torch

    row = batch.x[int(graph), int(node), cfg.key_vocab : cfg.key_vocab + cfg.classes]
    return int(torch.argmax(row).item())


def _node_key(cfg: Any, batch: Any, graph: int, node: int) -> int:
    import torch

    return int(torch.argmax(batch.x[int(graph), int(node), : cfg.key_vocab]).item())


def clean_metadata(legacy: Any, cfg: Any, clean: Any) -> dict[str, Any]:
    import torch

    rows: dict[str, list[int]] = {
        "source_query_distance": [],
        "source_value": [],
        "query_value": [],
        "source_key": [],
        "query_key": [],
        "query_index": [],
        "source_index": [],
        "query_parity": [],
        "source_value_upper_half": [],
        "query_value_upper_half": [],
    }
    for graph in range(len(clean)):
        query = int(clean.q_idx[graph])
        source = int(clean.target_idx[graph])
        source_value = _node_value(legacy, cfg, clean, graph, source)
        query_value = _node_value(legacy, cfg, clean, graph, query)
        rows["source_query_distance"].append(legacy.cycle_distance(cfg.n, query, source))
        rows["source_value"].append(source_value)
        rows["query_value"].append(query_value)
        rows["source_key"].append(_node_key(cfg, clean, graph, source))
        rows["query_key"].append(_node_key(cfg, clean, graph, query))
        rows["query_index"].append(query)
        rows["source_index"].append(source)
        rows["query_parity"].append(query % 2)
        rows["source_value_upper_half"].append(int(source_value >= cfg.classes // 2))
        rows["query_value_upper_half"].append(int(query_value >= cfg.classes // 2))
    return {key: torch.tensor(value, dtype=torch.long) for key, value in rows.items()}


def _event_choices(
    legacy: Any,
    cfg: Any,
    clean: Any,
    *,
    graph: int,
    factor: str,
    mode: str,
    sampled_events: int,
    rng: np.random.Generator,
) -> list[int]:
    source = int(clean.target_idx[graph])
    query = int(clean.q_idx[graph])
    if factor == "semantic":
        current = _node_value(legacy, cfg, clean, graph, source)
        population = [value for value in range(cfg.classes) if value != current]
    elif factor == "structural":
        population = legacy.structural_partners(cfg, query, source)
    else:
        raise ValueError(f"unknown factor {factor!r}")
    population = [int(value) for value in population]
    if mode == "enumerate":
        rng.shuffle(population)
        return population
    if not population:
        raise RuntimeError(f"empty {factor} event population")
    if factor == "semantic":
        return [
            legacy.draw_other_class(rng, cfg.classes, current)
            for _ in range(int(sampled_events))
        ]
    return [int(rng.choice(population)) for _ in range(int(sampled_events))]


def make_event_bundle(
    legacy: Any,
    cfg: Any,
    clean: Any,
    *,
    factor: str,
    seed: int,
    mode: str,
    sampled_events: int,
    no_op: bool = False,
) -> EventBundle:
    """Create graph-major clean+event replicas and retain every event's metadata."""

    import torch

    rng = np.random.default_rng(int(seed))
    choices: list[list[int]] = []
    for graph in range(len(clean)):
        if no_op:
            source = int(clean.target_idx[graph])
            current = _node_value(legacy, cfg, clean, graph, source)
            choices.append([current if factor == "semantic" else source])
        else:
            choices.append(
                _event_choices(
                    legacy,
                    cfg,
                    clean,
                    graph=graph,
                    factor=factor,
                    mode=mode,
                    sampled_events=sampled_events,
                    rng=rng,
                )
            )
    if no_op:
        max_events = 1
    elif mode == "enumerate":
        max_events = cfg.classes - 1 if factor == "semantic" else cfg.n - 2
    else:
        max_events = int(sampled_events)
    valid = torch.zeros(len(clean), max_events, dtype=torch.bool)
    source_out = torch.full((len(clean), max_events), -1, dtype=torch.long)
    partner_out = torch.full_like(source_out, -1)
    donor_value = torch.full_like(source_out, -1)
    dose = torch.zeros(len(clean), max_events, dtype=torch.float32)
    carrier_distance = torch.full(
        (len(clean), max_events, cfg.n), -1, dtype=torch.long
    )
    replicas: list[Any] = []

    for graph in range(len(clean)):
        one = clean.slice(graph, graph + 1)
        replicas.append(one)
        source = int(one.target_idx[0])
        query = int(one.q_idx[0])
        for event in range(max_events):
            x = one.x.clone()
            rrwp = one.rrwp.clone()
            if event < len(choices[graph]):
                valid[graph, event] = True
                choice = int(choices[graph][event])
                if factor == "semantic":
                    donor_value[graph, event] = choice
                    x[0, source] = legacy.replace_value(cfg, x[0, source], choice)
                    dose[graph, event] = torch.linalg.vector_norm(
                        x[0, source] - one.x[0, source]
                    )
                    for carrier in range(cfg.n):
                        carrier_distance[graph, event, carrier] = legacy.cycle_distance(
                            cfg.n, carrier, source
                        )
                else:
                    partner_out[graph, event] = choice
                    rrwp[0] = legacy.transpose_rrwp(rrwp[0], source, choice)
                    dose[graph, event] = torch.linalg.vector_norm(rrwp[0] - one.rrwp[0]) / math.sqrt(
                        float(cfg.n * cfg.n * cfg.rrwp_steps)
                    )
                    for carrier in range(cfg.n):
                        carrier_distance[graph, event, carrier] = min(
                            legacy.cycle_distance(cfg.n, carrier, source),
                            legacy.cycle_distance(cfg.n, carrier, choice),
                        )
                source_out[graph, event] = source
            # Padding uses an exact no-op but is removed by ``valid`` from every estimand.
            replicas.append(
                legacy.SynthBatch(
                    x=x,
                    rrwp=rrwp,
                    q_idx=one.q_idx.clone(),
                    target_idx=one.target_idx.clone(),
                    anchor_idx=one.anchor_idx.clone(),
                    y=one.y.clone(),
                    mode=one.mode.clone(),
                )
            )
    return EventBundle(
        replicas=legacy.concat_batches(replicas),
        valid=valid,
        source=source_out,
        partner=partner_out,
        donor_value=donor_value,
        dose=dose,
        carrier_distance=carrier_distance,
        clean_metadata=clean_metadata(legacy, cfg, clean),
        event_mode=mode,
    )


def masked_event_mean(values: Any, valid: Any, dim: int = 1) -> Any:
    weights = valid.to(device=values.device, dtype=values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def aggregate_projected_events(q: Any, valid: Any) -> dict[str, Any]:
    """Compute all four per-graph scores from q=[G,K,L,H,N,T]."""

    import torch

    if q.dim() != 6 or valid.shape != q.shape[:2]:
        raise ValueError(f"expected q=[G,K,L,H,N,T], valid=[G,K], got {q.shape}, {valid.shape}")
    q_mean = masked_event_mean(q, valid, dim=1)
    carrier_coherent = torch.linalg.vector_norm(q_mean, dim=-1)
    carrier_event = torch.linalg.vector_norm(q, dim=-1)
    event_gross = carrier_event.sum(dim=-1)
    event_net_vector = q.sum(dim=-2)
    event_net = torch.linalg.vector_norm(event_net_vector, dim=-1)
    cg = carrier_coherent.sum(dim=-1)
    eg = masked_event_mean(event_gross, valid, dim=1)
    cn = torch.linalg.vector_norm(q_mean.sum(dim=-2), dim=-1)
    en = masked_event_mean(event_net, valid, dim=1)
    f_sens = masked_event_mean(carrier_event, valid, dim=1)
    return {
        "per_graph": {"CG": cg, "EG": eg, "CN": cn, "EN": en},
        "q_mean_carrier": q_mean,
        "event_gross": event_gross,
        "carrier_event": carrier_event,
        "event_net_vector": event_net_vector,
        "event_net": event_net,
        "F_coh": carrier_coherent,
        "F_sens": f_sens,
    }


def _prefix_aggregations(q: Any, valid: Any) -> dict[str, dict[str, Any]]:
    maximum = int(valid.size(1))
    requested = sorted({value for value in (1, 2, 4, 6) if value <= maximum})
    output: dict[str, dict[str, Any]] = {}
    for count in requested:
        result = aggregate_projected_events(q[:, :count], valid[:, :count])
        output[str(count)] = result["per_graph"]
    output["all"] = aggregate_projected_events(q, valid)["per_graph"]
    return output


def _repeated_prefix_aggregations(
    q: Any, valid: Any, *, repeats: int = 5
) -> dict[str, dict[str, dict[str, Any]]]:
    """Deterministic donor-order rotations; padding is always kept after valid events."""

    import torch

    output = {}
    for repeat in range(int(repeats)):
        ordered_q = torch.zeros_like(q)
        ordered_valid = torch.zeros_like(valid)
        for graph in range(q.size(0)):
            indices = torch.where(valid[graph])[0]
            if not len(indices):
                continue
            shift = int(math.floor(repeat * len(indices) / repeats))
            indices = torch.roll(indices, shifts=-shift)
            count = len(indices)
            ordered_q[graph, :count] = q[graph, indices]
            ordered_valid[graph, :count] = True
        output[str(repeat)] = _prefix_aggregations(ordered_q, ordered_valid)
    return output


def _merge_tensor_dicts(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    if not items:
        return {}
    output: dict[str, Any] = {}
    for key in items[0]:
        value = items[0][key]
        if isinstance(value, Mapping):
            output[key] = _merge_tensor_dicts([item[key] for item in items])
        elif key.endswith("_max") and isinstance(value, (float, int)):
            output[key] = max(float(item[key]) for item in items)
        elif key == "distance" and torch.is_tensor(value):
            if not all(torch.equal(value, item[key]) for item in items[1:]):
                raise RuntimeError("distance axes differ across score chunks")
            output[key] = value
        elif torch.is_tensor(value) and value.dim() >= 1:
            output[key] = torch.cat([item[key] for item in items], dim=0)
        else:
            output[key] = value
    return output


def distance_profiles(q: Any, valid: Any, distances: Any, max_distance: int) -> dict[str, Any]:
    """Graph-balanced profiles with identical carrier weighting for sensitivity/coherence."""

    import torch

    if distances.shape != (q.size(0), q.size(1), q.size(4)):
        raise ValueError("carrier distances are not aligned to q events and carriers")
    event_norm = torch.linalg.vector_norm(q, dim=-1)  # [G,K,L,H,N]
    sensitivity, coherent, pair_support, carrier_support = [], [], [], []
    for distance in range(int(max_distance) + 1):
        pair_mask = valid[:, :, None] & (distances == int(distance))  # [G,K,N]
        pair_weight = pair_mask[:, :, None, None, :].to(q.dtype)
        per_carrier_count = pair_mask.sum(dim=1)  # [G,N]
        per_carrier_valid = per_carrier_count > 0
        per_carrier_denominator = per_carrier_count[:, None, None, :].clamp_min(1).to(q.dtype)
        per_carrier_sens = (event_norm * pair_weight).sum(dim=1) / per_carrier_denominator

        expanded = pair_mask[:, :, None, None, :, None].to(q.dtype)
        per_carrier_q = (
            (q * expanded).sum(dim=1)
            / per_carrier_count[:, None, None, :, None].clamp_min(1).to(q.dtype)
        )
        per_carrier_f = torch.linalg.vector_norm(per_carrier_q, dim=-1)
        carrier_valid = per_carrier_valid[:, None, None, :]
        carriers = carrier_valid.to(q.dtype).sum(dim=-1)
        sens_value = (
            (per_carrier_sens * carrier_valid.to(q.dtype)).sum(dim=-1)
            / carriers.clamp_min(1.0)
        )
        coherent_value = (
            (per_carrier_f * carrier_valid.to(q.dtype)).sum(dim=-1)
            / carriers.clamp_min(1.0)
        )
        supported = carriers > 0
        sensitivity.append(torch.where(supported, sens_value, torch.nan))
        coherent.append(torch.where(supported, coherent_value, torch.nan))
        pair_support.append(pair_mask.sum(dim=(1, 2)))
        carrier_support.append(pair_mask.any(dim=1).sum(dim=1))
    return {
        "distance": torch.arange(int(max_distance) + 1),
        "F_sens": torch.stack(sensitivity, dim=-1),
        "F_coh": torch.stack(coherent, dim=-1),
        "event_carrier_support": torch.stack(pair_support, dim=-1),
        "carrier_support": torch.stack(carrier_support, dim=-1),
    }


# ======================================================================================
# Read-only checkpoint loading and score-event capture
# ======================================================================================


def discover_checkpoint(legacy: Any, run_dir: Path, seed: int) -> tuple[Path, dict[str, Any]]:
    import torch

    candidates = sorted((run_dir / "checkpoints").glob(f"seed_{int(seed)}__*.pt"))
    accepted: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("seed", -1)) != int(seed):
            continue
        if payload.get("version") != LEGACY_VERSION:
            continue
        config = dict(payload.get("config", {}))
        if config.get("run_name") not in {None, run_dir.name}:
            continue
        try:
            values = dict(config)
            values["seeds"] = tuple(values.get("seeds", (0, 1, 2)))
            model_cfg = legacy.Config(**values)
            model_cfg.validate()
        except Exception:
            continue
        if payload.get("fingerprint") != legacy.config_fingerprint(model_cfg):
            continue
        accepted.append((path, payload))
    if len(accepted) > 1:
        # ``cycle_dual_v2`` has occasionally accumulated checkpoints from exploratory
        # configurations under the same run directory.  The paper task is the exact
        # configuration declared by the canonical source file, whose fingerprint is
        # stable (currently 364da99053dbe173).  Prefer that match without relying on
        # file ordering or modification time.
        canonical_cfg = legacy.Config()
        canonical_fingerprint = legacy.config_fingerprint(canonical_cfg)
        canonical = [
            item for item in accepted
            if item[1].get("fingerprint") == canonical_fingerprint
        ]
        if len(canonical) == 1:
            rejected = ", ".join(
                str(item[1].get("fingerprint"))
                for item in accepted
                if item[0] != canonical[0][0]
            )
            print(
                f"[checkpoint seed={seed}] {len(accepted)} valid configurations found; "
                f"selected canonical paper fingerprint {canonical_fingerprint} "
                f"(ignored: {rejected})",
                flush=True,
            )
            return canonical[0]
    if len(accepted) != 1:
        listing = "\n".join(
            f"  - {path} | fingerprint={payload.get('fingerprint')}"
            for path, payload in accepted
        ) or "  (none accepted)"
        raise RuntimeError(
            f"expected exactly one valid legacy checkpoint for seed {seed}, found "
            f"{len(accepted)} under {run_dir}. No unique canonical-paper match was "
            f"available:\n{listing}"
        )
    return accepted[0]


def load_seed_model(
    legacy: Any,
    run_dir: Path,
    *,
    seed: int,
    device: Any,
) -> tuple[Any, Any, dict[str, Any], Path]:
    import torch

    path, payload = discover_checkpoint(legacy, run_dir, seed)
    config_values = dict(payload["config"])
    config_values["seeds"] = tuple(config_values.get("seeds", (0, 1, 2)))
    cfg = legacy.Config(**config_values)
    heldout = payload.get("heldout_validation", {})
    minimum = min(
        float(heldout.get("semantic_accuracy", 0.0)),
        float(heldout.get("structural_accuracy", 0.0)),
    )
    if minimum < float(cfg.accuracy_gate):
        raise RuntimeError(f"seed {seed} fails the legacy accuracy gate: {heldout}")
    model_class = legacy.build_model_class()
    legacy.set_seed(seed)
    model = model_class(cfg).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    recomputed = legacy.evaluate_model(model, cfg, seed=80_000 + seed, device=device)
    for task in CHANNELS:
        recorded = float(heldout.get(f"{task}_accuracy", float("nan")))
        current = float(recomputed.get(f"{task}_accuracy", float("nan")))
        if not math.isfinite(recorded) or abs(recorded - current) > 1.0e-6:
            raise RuntimeError(
                f"seed {seed} checkpoint replay mismatch for {task}: {recorded} vs {current}"
            )
    print(f"[checkpoint seed={seed}] read-only load {path}", flush=True)
    return model, cfg, payload, path


def discover_legacy_analysis(
    legacy: Any,
    run_dir: Path,
    cfg: Any,
    seed: int,
) -> Path | None:
    exact = run_dir / "analysis" / f"seed_{seed}__{legacy.config_fingerprint(cfg)}.pt"
    return exact if exact.exists() else None


def _score_event_batch(
    legacy: Any,
    model: Any,
    model_cfg: Any,
    clean: Any,
    *,
    factor: str,
    seed: int,
    device: Any,
    event_mode: str,
    sampled_events: int,
    no_op: bool = False,
) -> dict[str, Any]:
    """Capture projected event tensors once, then reduce every candidate estimator."""

    import torch
    import torch.nn.functional as F

    bundle = make_event_bundle(
        legacy,
        model_cfg,
        clean.cpu(),
        factor=factor,
        seed=seed,
        mode=event_mode,
        sampled_events=sampled_events,
        no_op=no_op,
    )
    moved = bundle.replicas.to(device)
    captured = legacy.capture_forward(model, moved, want_grad=True)
    logits = captured["logits"]
    routed = captured["wV"]
    graphs, events = bundle.graphs, bundle.events
    replicas = events + 1
    clean_rows = torch.arange(graphs, device=device) * replicas
    per_layer: list[list[Any]] = [[] for _ in range(model_cfg.layers)]
    for output in range(model_cfg.classes):
        gradients = torch.autograd.grad(
            logits[clean_rows, output].sum(),
            routed,
            retain_graph=output + 1 < model_cfg.classes,
            allow_unused=False,
        )
        for layer in range(model_cfg.layers):
            states = routed[layer].reshape(
                graphs, replicas, model_cfg.n, model_cfg.heads, -1
            )
            grad = gradients[layer].reshape(
                graphs, replicas, model_cfg.n, model_cfg.heads, -1
            )[:, 0]
            delta = states[:, 0, None] - states[:, 1:]
            per_layer[layer].append((grad[:, None] * delta).sum(dim=-1))
    # Each layer: [G,K,N,H,T] -> [G,K,H,N,T]; stack -> [G,K,L,H,N,T].
    q = torch.stack(
        [torch.stack(values, dim=-1).permute(0, 1, 3, 2, 4) for values in per_layer],
        dim=2,
    )
    valid = bundle.valid.to(device)
    reductions = aggregate_projected_events(q, valid)
    prefixes = _prefix_aggregations(q, valid)
    prefix_repeats = _repeated_prefix_aggregations(q, valid)
    profiles = distance_profiles(
        q,
        valid,
        bundle.carrier_distance.to(device),
        model_cfg.max_cycle_distance,
    )

    logits_gr = logits.reshape(graphs, replicas, model_cfg.classes)
    clean_loss = F.cross_entropy(
        logits_gr[:, 0], moved.y.reshape(graphs, replicas)[:, 0].long(), reduction="none"
    )
    labels = moved.y.reshape(graphs, replicas)[:, 0].long()
    variant_loss = F.cross_entropy(
        logits_gr[:, 1:].reshape(graphs * events, -1),
        labels[:, None].expand(graphs, events).reshape(-1),
        reduction="none",
    ).reshape(graphs, events)
    beneficial_outcome = variant_loss - clean_loss[:, None]  # positive: clean factor helped

    cpu = lambda value: value.detach().cpu() if torch.is_tensor(value) else value
    return {
        "per_graph": {key: cpu(value) for key, value in reductions["per_graph"].items()},
        "q_mean_carrier": cpu(reductions["q_mean_carrier"]),
        "event_gross": cpu(reductions["event_gross"]),
        "event_net_vector": cpu(reductions["event_net_vector"]),
        "F_coh": cpu(reductions["F_coh"]),
        "F_sens": cpu(reductions["F_sens"]),
        "prefix": {
            count: {key: cpu(value) for key, value in values.items()}
            for count, values in prefixes.items()
        },
        "prefix_repeats": {
            repeat: {
                count: {key: cpu(value) for key, value in values.items()}
                for count, values in repeated.items()
            }
            for repeat, repeated in prefix_repeats.items()
        },
        "distance_profiles": {key: cpu(value) for key, value in profiles.items()},
        "valid": bundle.valid,
        "source": bundle.source,
        "partner": bundle.partner,
        "donor_value": bundle.donor_value,
        "dose": bundle.dose,
        "carrier_distance": bundle.carrier_distance,
        "clean_metadata": bundle.clean_metadata,
        "beneficial_outcome": cpu(beneficial_outcome),
        "clean_logits": cpu(logits_gr[:, 0]),
        "variant_logits": cpu(logits_gr[:, 1:]),
        "labels": clean.y.detach().cpu(),
        "event_mode": event_mode,
    }


def _legacy_formula_regression(
    legacy: Any,
    model: Any,
    model_cfg: Any,
    *,
    mode: int,
    factor: str,
    seed: int,
    device: Any,
) -> dict[str, Any]:
    """Verify beta CG against legacy CG on identical sampled events.

    The two formulas are evaluated in separate forwards.  On CUDA, nondeterministic
    accumulation order in graph scatter kernels can therefore produce harmless
    float32 differences even though the formulas agree.  Record a scale-aware
    diagnostic and fail only when ``torch.allclose`` semantics are violated.
    """

    clean = legacy.make_batch(model_cfg, 3, seed, mode=mode)
    old = legacy.score_channel_batch(
        model, model_cfg, clean, factor=factor, seed=seed + 1, device=device
    )
    new = _score_event_batch(
        legacy,
        model,
        model_cfg,
        clean,
        factor=factor,
        seed=seed + 1,
        device=device,
        event_mode="sample",
        sampled_events=model_cfg.score_donors,
    )["per_graph"]["CG"]
    diagnostic = _tensor_equivalence_diagnostic(
        old,
        new,
        atol=CG_REGRESSION_ATOL,
    )
    if not bool(diagnostic["passed"]):
        raise RuntimeError(
            "beta CG does not reproduce legacy CG: "
            f"max error={diagnostic['max_abs_error']:.3e}, "
            f"signal={diagnostic['signal_at_max_error']:.3e}, "
            f"allowed={diagnostic['allowed_at_max_error']:.3e}"
        )
    print(
        f"[verification {factor}] legacy CG agreement: "
        f"max error={diagnostic['max_abs_error']:.3e}, "
        f"max tolerance ratio={diagnostic['max_tolerance_ratio']:.3f}",
        flush=True,
    )
    return diagnostic


def _cg_formula_regression_passed(value: Any) -> bool:
    """Read current structured checks and older scalar manifests defensively."""

    if isinstance(value, Mapping):
        return bool(value.get("passed", False))
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(scalar) and scalar <= CG_REGRESSION_ATOL


def score_cache_path(beta_dir: Path, seed: int, channel: str, fingerprint: str) -> Path:
    return beta_dir / "scores" / f"seed_{seed}__{channel}__{fingerprint}.pt"


def run_score_channel(
    legacy: Any,
    model: Any,
    model_cfg: Any,
    beta_cfg: BetaConfig,
    *,
    seed: int,
    channel: str,
    device: Any,
    force: bool,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    import torch

    path = score_cache_path(beta_cfg.beta_dir, seed, channel, beta_cfg.fingerprint)
    if path.exists() and not force:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if _beta_cache_is_current(cached, beta_cfg, checkpoint_sha256):
            print(f"[scores seed={seed} {channel}] loaded {path}", flush=True)
            return cached
        print(f"[scores seed={seed} {channel}] stale cache; recomputing", flush=True)
    mode = legacy.MODE_SEMANTIC if channel == "semantic" else legacy.MODE_STRUCTURAL
    clean = legacy.make_batch(
        model_cfg,
        beta_cfg.score_graphs,
        2_000_000 + 10_000 * seed + mode,
        mode=mode,
    )
    chunks = []
    cross_chunks = []
    cross_factor = "structural" if channel == "semantic" else "semantic"
    for start in range(0, len(clean), beta_cfg.score_batch_size):
        stop = min(start + beta_cfg.score_batch_size, len(clean))
        clean_chunk = clean.slice(start, stop)
        chunks.append(
            _score_event_batch(
                legacy, model, model_cfg, clean_chunk,
                factor=channel,
                seed=2_100_000 + seed * 100_003 + start,
                device=device,
                event_mode=beta_cfg.score_event_mode,
                sampled_events=beta_cfg.score_sampled_events,
            )
        )
        # Negative-control cell of the task-mode × intervention-factor 2×2 design.
        cross_chunks.append(
            _score_event_batch(
                legacy, model, model_cfg, clean_chunk,
                factor=cross_factor,
                seed=2_150_000 + seed * 100_003 + start,
                device=device,
                event_mode=beta_cfg.score_event_mode,
                sampled_events=beta_cfg.score_sampled_events,
            )
        )
        print(f"[scores seed={seed} {channel}] {stop}/{len(clean)}", flush=True)
    merged = _merge_tensor_dicts(chunks)
    no_op = _score_event_batch(
        legacy,
        model,
        model_cfg,
        clean.slice(0, min(4, len(clean))),
        factor=channel,
        seed=2_200_000 + seed,
        device=device,
        event_mode="sample",
        sampled_events=1,
        no_op=True,
    )
    no_op_output_check = _tensor_equivalence_diagnostic(
        no_op["clean_logits"][:, None, :],
        no_op["variant_logits"],
        atol=FORWARD_EQUIVALENCE_ATOL,
    )
    no_op_score_checks = {
        aggregation: _tensor_equivalence_diagnostic(
            torch.zeros_like(no_op["per_graph"][aggregation]),
            no_op["per_graph"][aggregation],
            atol=FORWARD_EQUIVALENCE_ATOL,
        )
        for aggregation in AGGREGATIONS
    }
    if not bool(no_op_output_check["passed"]) or not all(
        bool(check["passed"]) for check in no_op_score_checks.values()
    ):
        worst_score_ratio = max(
            float(check["max_tolerance_ratio"])
            for check in no_op_score_checks.values()
        )
        raise RuntimeError(
            f"score no-op control failed for seed {seed} {channel}: "
            f"output ratio={no_op_output_check['max_tolerance_ratio']:.3f}, "
            f"score ratio={worst_score_ratio:.3f}"
        )
    mode_name = "enumerated population" if beta_cfg.score_event_mode == "enumerate" else "sample"
    payload = {
        "version": BETA_VERSION,
        "schema": BETA_SCHEMA,
        "fingerprint": beta_cfg.fingerprint,
        "checkpoint_sha256": checkpoint_sha256,
        "seed": int(seed),
        "channel": channel,
        "task_mode": channel,
        "intervention_factor": channel,
        "event_distribution": mode_name,
        "score": merged,
        "cross_factor_control": {
            "task_mode": channel,
            "intervention_factor": cross_factor,
            "score": _merge_tensor_dicts(cross_chunks),
        },
        "no_op": no_op,
        "no_op_numerical_check": {
            "output": no_op_output_check,
            "scores": no_op_score_checks,
        },
    }
    atomic_torch_save(payload, path)
    print(f"[scores seed={seed} {channel}] cached {path}", flush=True)
    return payload


# ======================================================================================
# Bidirectional whole-transport mediation and donor-wise necessity
# ======================================================================================


def flatten_valid_event_pairs(bundle: EventBundle) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    graph_index, event_index = torch.where(bundle.valid)
    replicas = bundle.events + 1
    clean_index = graph_index * replicas
    variant_index = graph_index * replicas + event_index + 1
    clean = bundle.replicas.select(clean_index)
    variant = bundle.replicas.select(variant_index)
    metadata = {
        "graph": graph_index,
        "event": event_index,
        "source": bundle.source[graph_index, event_index],
        "partner": bundle.partner[graph_index, event_index],
        "donor_value": bundle.donor_value[graph_index, event_index],
        "dose": bundle.dose[graph_index, event_index],
    }
    return clean, variant, metadata


def _alignment_metrics(reference: Any, movement: Any) -> dict[str, Any]:
    import torch

    denominator = reference.square().sum(dim=-1)
    reference_norm = denominator.sqrt()
    movement_norm = torch.linalg.vector_norm(movement, dim=-1)
    dot = (movement * reference).sum(dim=-1)
    coefficient = dot / (denominator + EPS)
    cosine = dot / (reference_norm * movement_norm + EPS)
    orthogonal = movement - coefficient[:, None] * reference
    return {
        "projection_fraction": coefficient,
        "cosine": cosine,
        "absolute_movement": movement_norm,
        "desired_projection": dot / (reference_norm + EPS),
        "orthogonal_norm": torch.linalg.vector_norm(orthogonal, dim=-1),
        "orthogonal_ratio": torch.linalg.vector_norm(orthogonal, dim=-1)
        / (reference_norm + EPS),
    }


def _cross_graph_mismatch_transport(value: Any, graph_ids: Any, nodes: int) -> Any:
    """Choose clean transport from a different base graph, preserving event alignment."""

    import torch

    ids = np.asarray(graph_ids.detach().cpu(), dtype=int)
    unique = list(dict.fromkeys(ids.tolist()))
    if len(unique) < 2:
        raise RuntimeError("cross-graph mismatch control needs at least two base graphs per batch")
    donor = {graph: unique[(index - 1) % len(unique)] for index, graph in enumerate(unique)}
    first = {graph: int(np.flatnonzero(ids == graph)[0]) for graph in unique}
    source_rows = torch.tensor(
        [first[donor[int(graph)]] for graph in ids],
        device=value.device,
        dtype=torch.long,
    )
    shaped = value.reshape(len(ids), nodes, value.size(-2), value.size(-1))
    return shaped.index_select(0, source_rows).reshape_as(value)


def _clean_projection_vectors(
    logits: Any,
    routed: Sequence[Any],
    corrupt_routed: Sequence[Any],
    *,
    classes: int,
    graphs: int,
    nodes: int,
    heads: int,
) -> Any:
    """First-order clean-gradient prediction [E,L,H,T] for finite head-site deltas."""

    import torch

    by_layer: list[list[Any]] = [[] for _ in routed]
    for output in range(int(classes)):
        gradients = torch.autograd.grad(
            logits[:, output].sum(),
            routed,
            retain_graph=output + 1 < int(classes),
            allow_unused=False,
        )
        for layer, (gradient, clean_value, corrupt_value) in enumerate(
            zip(gradients, routed, corrupt_routed)
        ):
            gradient = gradient.reshape(graphs, nodes, heads, -1)
            delta = clean_value.reshape(graphs, nodes, heads, -1) - corrupt_value.reshape(
                graphs, nodes, heads, -1
            )
            by_layer[layer].append((gradient * delta).sum(dim=-1).sum(dim=1))
    return torch.stack([torch.stack(value, dim=-1) for value in by_layer], dim=1)


def _empty_patch_metrics(layers: int, heads: int, events: int) -> dict[str, Any]:
    import torch

    names = (
        "projection_fraction", "cosine", "absolute_movement", "desired_projection",
        "orthogonal_norm", "orthogonal_ratio",
    )
    return {
        direction: {
            name: torch.full((layers, heads, events), float("nan")) for name in names
        }
        for direction in ("restore", "inject", "necessity", "cross_graph_mismatch")
    }


def bidirectional_patch_sweep(
    legacy: Any,
    model: Any,
    model_cfg: Any,
    beta_cfg: BetaConfig,
    *,
    seed: int,
    channel: str,
    device: Any,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    mode = legacy.MODE_SEMANTIC if channel == "semantic" else legacy.MODE_STRUCTURAL
    base = legacy.make_batch(
        model_cfg,
        beta_cfg.causal_graphs,
        3_000_000 + seed * 10_000 + mode,
        mode=mode,
    )
    bundle = make_event_bundle(
        legacy,
        model_cfg,
        base,
        factor=channel,
        seed=3_100_000 + seed,
        mode="sample",
        sampled_events=beta_cfg.causal_events,
    )
    clean_all, corrupt_all, event_metadata = flatten_valid_event_pairs(bundle)
    total_events = len(clean_all)
    metrics = _empty_patch_metrics(model_cfg.layers, model_cfg.heads, total_events)
    loss_metrics = {
        name: torch.full((model_cfg.layers, model_cfg.heads, total_events), float("nan"))
        for name in ("restore_reduction", "inject_increase", "necessity_reduction")
    }
    predicted = torch.full(
        (total_events, model_cfg.layers, model_cfg.heads, model_cfg.classes), float("nan")
    )
    finite_vectors = {
        name: torch.full(
            (total_events, model_cfg.layers, model_cfg.heads, model_cfg.classes), float("nan")
        )
        for name in ("restore", "inject", "necessity")
    }
    reference_logits = torch.full((total_events, model_cfg.classes), float("nan"))
    effect_norm = torch.full((total_events,), float("nan"))
    clean_loss_all = torch.full((total_events,), float("nan"))
    corrupt_loss_all = torch.full((total_events,), float("nan"))
    sham_max = 0.0
    sham_tolerance_ratio_max = 0.0

    for start in range(0, total_events, beta_cfg.causal_batch_size):
        stop = min(start + beta_cfg.causal_batch_size, total_events)
        clean = clean_all.slice(start, stop).to(device)
        corrupt = corrupt_all.slice(start, stop).to(device)
        events = len(clean)
        clean_capture = legacy.capture_forward(model, clean, want_grad=True)
        corrupt_capture = legacy.capture_forward(model, corrupt, want_grad=False)
        clean_logits = clean_capture["logits"]
        corrupt_logits = corrupt_capture["logits"]
        clean_wv_graph = clean_capture["wV"]
        corrupt_wv = [value.detach() for value in corrupt_capture["wV"]]
        predicted[start:stop] = _clean_projection_vectors(
            clean_logits,
            clean_wv_graph,
            corrupt_wv,
            classes=model_cfg.classes,
            graphs=events,
            nodes=model_cfg.n,
            heads=model_cfg.heads,
        ).detach().cpu()
        clean_wv = [value.detach() for value in clean_wv_graph]
        clean_logits = clean_logits.detach()
        delta = clean_logits - corrupt_logits
        reference_logits[start:stop] = delta.cpu()
        effect_norm[start:stop] = torch.linalg.vector_norm(delta, dim=-1).cpu()
        clean_loss = F.cross_entropy(clean_logits, clean.y.long(), reduction="none")
        corrupt_loss = F.cross_entropy(corrupt_logits, clean.y.long(), reduction="none")
        clean_loss_all[start:stop] = clean_loss.cpu()
        corrupt_loss_all[start:stop] = corrupt_loss.cpu()

        # Explicit sham: corrupt endpoint patched back into itself must be a numerical no-op.
        with legacy.patch_head_output(model, 0, 0, corrupt_wv[0]), torch.no_grad():
            sham = model(corrupt)
        sham_check = _tensor_equivalence_diagnostic(
            corrupt_logits,
            sham,
            atol=FORWARD_EQUIVALENCE_ATOL,
        )
        sham_max = max(sham_max, float(sham_check["max_abs_error"]))
        sham_tolerance_ratio_max = max(
            sham_tolerance_ratio_max,
            float(sham_check["max_tolerance_ratio"]),
        )
        if not bool(sham_check["passed"]):
            raise RuntimeError(
                f"patch sham failed for seed {seed} {channel}: "
                f"max error={sham_check['max_abs_error']:.3e}, "
                f"tolerance ratio={sham_check['max_tolerance_ratio']:.3f}"
            )

        for layer in range(model_cfg.layers):
            wrong_source = _cross_graph_mismatch_transport(
                clean_wv[layer], event_metadata["graph"][start:stop], model_cfg.n
            )
            for head in range(model_cfg.heads):
                with legacy.patch_head_output(model, layer, head, clean_wv[layer]), torch.no_grad():
                    restored = model(corrupt).detach()
                with legacy.patch_head_output(model, layer, head, corrupt_wv[layer]), torch.no_grad():
                    injected = model(clean).detach()
                with legacy.patch_head_output(model, layer, head, wrong_source), torch.no_grad():
                    wrong = model(corrupt).detach()
                with legacy.ablate_head(model, layer, head), torch.no_grad():
                    ablated_clean = model(clean).detach()
                    ablated_corrupt = model(corrupt).detach()

                movements = {
                    "restore": restored - corrupt_logits,
                    # Positive induction means clean moves toward corrupt, represented in
                    # the same clean-minus-corrupt direction as restore.
                    "inject": clean_logits - injected,
                    "necessity": delta - (ablated_clean - ablated_corrupt),
                    "cross_graph_mismatch": wrong - corrupt_logits,
                }
                for direction, movement in movements.items():
                    values = _alignment_metrics(delta, movement)
                    for name, value in values.items():
                        metrics[direction][name][layer, head, start:stop] = value.cpu()
                    if direction in finite_vectors:
                        finite_vectors[direction][start:stop, layer, head] = movement.cpu()

                restored_loss = F.cross_entropy(restored, clean.y.long(), reduction="none")
                injected_loss = F.cross_entropy(injected, clean.y.long(), reduction="none")
                ablated_clean_loss = F.cross_entropy(
                    ablated_clean, clean.y.long(), reduction="none"
                )
                ablated_corrupt_loss = F.cross_entropy(
                    ablated_corrupt, clean.y.long(), reduction="none"
                )
                loss_metrics["restore_reduction"][layer, head, start:stop] = (
                    corrupt_loss - restored_loss
                ).cpu()
                loss_metrics["inject_increase"][layer, head, start:stop] = (
                    injected_loss - clean_loss
                ).cpu()
                loss_metrics["necessity_reduction"][layer, head, start:stop] = (
                    (corrupt_loss - clean_loss)
                    - (ablated_corrupt_loss - ablated_clean_loss)
                ).cpu()
        print(f"[patch seed={seed} {channel}] {stop}/{total_events}", flush=True)

    return {
        "version": BETA_VERSION,
        "seed": int(seed),
        "channel": channel,
        "metrics": metrics,
        "loss_metrics": loss_metrics,
        "predicted_logit_delta": predicted,
        "finite_logit_delta": finite_vectors,
        "actual_logit_delta": reference_logits,
        "effect_norm": effect_norm,
        "clean_loss": clean_loss_all,
        "corrupt_loss": corrupt_loss_all,
        "event_metadata": event_metadata,
        "sham_max": sham_max,
        "sham_tolerance_ratio_max": sham_tolerance_ratio_max,
        "site_claim": "carrier-aligned routed node wV; edge-update output is not patched",
    }


def patch_cache_path(beta_dir: Path, seed: int, channel: str, fingerprint: str) -> Path:
    return beta_dir / "causal" / f"seed_{seed}__{channel}__{fingerprint}.pt"


def run_patch_channel(
    legacy: Any,
    model: Any,
    model_cfg: Any,
    beta_cfg: BetaConfig,
    *,
    seed: int,
    channel: str,
    device: Any,
    force: bool,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    import torch

    path = patch_cache_path(beta_cfg.beta_dir, seed, channel, beta_cfg.fingerprint)
    if path.exists() and not force:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if _beta_cache_is_current(cached, beta_cfg, checkpoint_sha256):
            print(f"[patch seed={seed} {channel}] loaded {path}", flush=True)
            return cached
        print(f"[patch seed={seed} {channel}] stale cache; recomputing", flush=True)
    payload = bidirectional_patch_sweep(
        legacy,
        model,
        model_cfg,
        beta_cfg,
        seed=seed,
        channel=channel,
        device=device,
    )
    payload.update(
        {
            "schema": BETA_SCHEMA,
            "fingerprint": beta_cfg.fingerprint,
            "checkpoint_sha256": checkpoint_sha256,
        }
    )
    atomic_torch_save(payload, path)
    print(f"[patch seed={seed} {channel}] cached {path}", flush=True)
    return payload


# ======================================================================================
# Exact routing/message decomposition and finite component patching
# ======================================================================================


def capture_mechanistic_forward(model: Any, batch: Any, *, device: Any) -> dict[str, Any]:
    import torch
    from graph_specialisation_metrics.mechanistic_operator_analysis import (
        OfficialGRITMechanisticCollector,
    )

    moved = batch.to(device)
    with OfficialGRITMechanisticCollector(model) as collector:
        logits = model(moved)
    records = sorted(collector.records, key=lambda value: value.layer)
    if len(records) != int(model.L):
        raise RuntimeError(f"expected {model.L} mechanistic records, captured {len(records)}")
    layers = []
    graphs, nodes = len(batch), int(model.cfg.n)
    for record in records:
        heads, dim = int(record.heads), int(record.message_dim)
        attention = torch.zeros(
            graphs, heads, nodes, nodes, device=device, dtype=record.attention.dtype
        )
        message = torch.zeros(
            graphs, heads, nodes, nodes, dim, device=device, dtype=record.message.dtype
        )
        attention[
            record.graph.long(), :, record.local_dst.long(), record.local_src.long()
        ] = record.attention.to(device)
        message[
            record.graph.long(), :, record.local_dst.long(), record.local_src.long()
        ] = record.message.to(device)
        routed = record.head_output.reshape(graphs, nodes, heads, dim)
        reconstructed = (
            attention.unsqueeze(-1) * message
        ).sum(dim=3).permute(0, 2, 1, 3)
        reconstruction_check = _tensor_equivalence_diagnostic(
            routed,
            reconstructed,
            atol=MECHANISM_RECONSTRUCTION_ATOL,
        )
        error = float(reconstruction_check["max_abs_error"])
        if not bool(reconstruction_check["passed"]):
            raise RuntimeError(
                f"A/message reconstruction failed at layer {record.layer}: "
                f"max error={error:.3e}, "
                f"tolerance ratio={reconstruction_check['max_tolerance_ratio']:.3f}"
            )
        layers.append(
            {
                "attention": attention,
                "message": message,
                "routed": record.head_output,
                "reconstruction_max": error,
                "reconstruction_tolerance_ratio_max": float(
                    reconstruction_check["max_tolerance_ratio"]
                ),
            }
        )
    return {"logits": logits, "layers": layers, "batch": moved}


def symmetric_routing_message_split(
    attention_clean: Any,
    message_clean: Any,
    attention_variant: Any,
    message_variant: Any,
) -> tuple[Any, Any, Any]:
    a_mean = 0.5 * (attention_clean + attention_variant)
    m_mean = 0.5 * (message_clean + message_variant)
    route = (
        (attention_clean - attention_variant).unsqueeze(-1) * m_mean
    ).sum(dim=-2)
    message = (
        a_mean.unsqueeze(-1) * (message_clean - message_variant)
    ).sum(dim=-2)
    total = (
        attention_clean.unsqueeze(-1) * message_clean
    ).sum(dim=-2) - (
        attention_variant.unsqueeze(-1) * message_variant
    ).sum(dim=-2)
    return route, message, total


def mechanism_score_batch(
    legacy: Any,
    model: Any,
    model_cfg: Any,
    clean: Any,
    *,
    channel: str,
    seed: int,
    events: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    bundle = make_event_bundle(
        legacy,
        model_cfg,
        clean,
        factor=channel,
        seed=seed,
        mode="sample",
        sampled_events=events,
    )
    captured = capture_mechanistic_forward(model, bundle.replicas, device=device)
    graphs, alternatives = bundle.graphs, bundle.events
    replicas = alternatives + 1
    clean_rows = torch.arange(graphs, device=device) * replicas
    logits = captured["logits"]
    routed = [layer["routed"] for layer in captured["layers"]]
    phi_by_output: list[list[Any]] = [[] for _ in routed]
    for output in range(model_cfg.classes):
        gradients = torch.autograd.grad(
            logits[clean_rows, output].sum(),
            routed,
            retain_graph=output + 1 < model_cfg.classes,
            allow_unused=False,
        )
        for layer, gradient in enumerate(gradients):
            phi_by_output[layer].append(
                gradient.reshape(
                    graphs, replicas, model_cfg.n, model_cfg.heads, -1
                )[:, 0]
            )

    route_q, message_q, total_q = [], [], []
    reconstruction = []
    reconstruction_tolerance_ratios = []
    for layer, fields in enumerate(captured["layers"]):
        attention = fields["attention"].reshape(
            graphs, replicas, model_cfg.heads, model_cfg.n, model_cfg.n
        )
        message_field = fields["message"].reshape(
            graphs, replicas, model_cfg.heads, model_cfg.n, model_cfg.n, -1
        )
        route, message, total = symmetric_routing_message_split(
            attention[:, 0, None],
            message_field[:, 0, None],
            attention[:, 1:],
            message_field[:, 1:],
        )
        routed_view = fields["routed"].reshape(
            graphs, replicas, model_cfg.n, model_cfg.heads, -1
        )
        direct = (
            routed_view[:, 0, None] - routed_view[:, 1:]
        ).permute(0, 1, 3, 2, 4)
        reconstruction_check = _tensor_equivalence_diagnostic(
            direct,
            route + message,
            atol=MECHANISM_RECONSTRUCTION_ATOL,
        )
        error = float(reconstruction_check["max_abs_error"])
        reconstruction.append(error)
        reconstruction_tolerance_ratios.append(
            float(reconstruction_check["max_tolerance_ratio"])
        )
        if not bool(reconstruction_check["passed"]):
            raise RuntimeError(
                f"symmetric routing/message split failed at layer {layer}: "
                f"max error={error:.3e}, "
                f"tolerance ratio={reconstruction_check['max_tolerance_ratio']:.3f}"
            )
        phi = torch.stack(phi_by_output[layer], dim=0)
        route_q.append(torch.einsum("tgnhd,gkhnd->gkhnt", phi, route))
        message_q.append(torch.einsum("tgnhd,gkhnd->gkhnt", phi, message))
        total_q.append(torch.einsum("tgnhd,gkhnd->gkhnt", phi, total))
    # [G,K,L,H,N,T]
    route_q_t = torch.stack(route_q, dim=2)
    message_q_t = torch.stack(message_q, dim=2)
    total_q_t = torch.stack(total_q, dim=2)
    projected_check = _tensor_equivalence_diagnostic(
        total_q_t,
        route_q_t + message_q_t,
        atol=MECHANISM_RECONSTRUCTION_ATOL,
    )
    projected_error = float(projected_check["max_abs_error"])
    if not bool(projected_check["passed"]):
        raise RuntimeError(
            "projected mechanism split failed: "
            f"max error={projected_error:.3e}, "
            f"tolerance ratio={projected_check['max_tolerance_ratio']:.3f}"
        )
    valid = bundle.valid.to(device)
    output = {}
    for name, values in (
        ("routing", route_q_t), ("message", message_q_t), ("total", total_q_t)
    ):
        reduced = aggregate_projected_events(values, valid)
        output[name] = {
            "per_graph": {
                key: value.detach().cpu() for key, value in reduced["per_graph"].items()
            },
            "event_net_vector": reduced["event_net_vector"].detach().cpu(),
        }
    return {
        "components": output,
        "valid": bundle.valid,
        "reconstruction_max": max(reconstruction),
        "reconstruction_tolerance_ratio_max": max(reconstruction_tolerance_ratios),
        "projected_reconstruction_max": projected_error,
        "projected_reconstruction_tolerance_ratio_max": float(
            projected_check["max_tolerance_ratio"]
        ),
    }


def component_patch_sweep(
    legacy: Any,
    model: Any,
    model_cfg: Any,
    beta_cfg: BetaConfig,
    *,
    seed: int,
    channel: str,
    device: Any,
) -> dict[str, Any]:
    import torch

    mode = legacy.MODE_SEMANTIC if channel == "semantic" else legacy.MODE_STRUCTURAL
    base = legacy.make_batch(
        model_cfg,
        beta_cfg.mechanism_graphs,
        4_000_000 + seed * 10_000 + mode,
        mode=mode,
    )
    bundle = make_event_bundle(
        legacy,
        model_cfg,
        base,
        factor=channel,
        seed=4_100_000 + seed,
        mode="sample",
        sampled_events=1,
    )
    clean, variant, metadata = flatten_valid_event_pairs(bundle)
    clean_capture = capture_mechanistic_forward(model, clean, device=device)
    variant_capture = capture_mechanistic_forward(model, variant, device=device)
    clean_logits = clean_capture["logits"].detach()
    variant_logits = variant_capture["logits"].detach()
    reference = clean_logits - variant_logits
    events = len(clean)
    names = (
        "projection_fraction", "cosine", "absolute_movement", "desired_projection",
        "orthogonal_norm", "orthogonal_ratio",
    )
    metrics = {
        component: {
            name: torch.full((model_cfg.layers, model_cfg.heads, events), float("nan"))
            for name in names
        }
        for component in ("routing", "message", "full")
    }
    interaction_norm = torch.full((model_cfg.layers, model_cfg.heads, events), float("nan"))
    interaction_projection = torch.full_like(interaction_norm, float("nan"))

    for layer in range(model_cfg.layers):
        clean_fields = clean_capture["layers"][layer]
        variant_fields = variant_capture["layers"][layer]
        clean_a, clean_m = clean_fields["attention"], clean_fields["message"]
        variant_a, variant_m = variant_fields["attention"], variant_fields["message"]
        clean_route = (clean_a.unsqueeze(-1) * variant_m).sum(dim=3).permute(0, 2, 1, 3)
        clean_message = (variant_a.unsqueeze(-1) * clean_m).sum(dim=3).permute(0, 2, 1, 3)
        clean_full = clean_fields["routed"].reshape(
            events, model_cfg.n, model_cfg.heads, -1
        ).detach()
        variant_full = variant_fields["routed"].reshape(
            events, model_cfg.n, model_cfg.heads, -1
        ).detach()
        sources = {
            "routing": clean_route.reshape_as(variant_fields["routed"]),
            "message": clean_message.reshape_as(variant_fields["routed"]),
            "full": clean_full.reshape_as(variant_fields["routed"]),
        }
        for head in range(model_cfg.heads):
            logits_by_component = {}
            for component, source in sources.items():
                with legacy.patch_head_output(model, layer, head, source), torch.no_grad():
                    patched = model(variant.to(device)).detach()
                logits_by_component[component] = patched
                values = _alignment_metrics(reference, patched - variant_logits)
                for name, value in values.items():
                    metrics[component][name][layer, head] = value.cpu()
            factorial = (
                logits_by_component["full"]
                - logits_by_component["routing"]
                - logits_by_component["message"]
                + variant_logits
            )
            interaction_norm[layer, head] = torch.linalg.vector_norm(
                factorial, dim=-1
            ).cpu()
            interaction_projection[layer, head] = (
                (factorial * reference).sum(dim=-1)
                / (torch.linalg.vector_norm(reference, dim=-1) + EPS)
            ).cpu()
        print(
            f"[component patch seed={seed} {channel}] layer {layer + 1}/{model_cfg.layers}",
            flush=True,
        )
    return {
        "metrics": metrics,
        "interaction_norm": interaction_norm,
        "interaction_projection": interaction_projection,
        "effect_norm": torch.linalg.vector_norm(reference, dim=-1).cpu(),
        "event_metadata": metadata,
        "reconstruction_max": max(
            float(layer["reconstruction_max"])
            for layer in clean_capture["layers"] + variant_capture["layers"]
        ),
        "reconstruction_tolerance_ratio_max": max(
            float(layer["reconstruction_tolerance_ratio_max"])
            for layer in clean_capture["layers"] + variant_capture["layers"]
        ),
    }


def mechanism_cache_path(beta_dir: Path, seed: int, channel: str, fingerprint: str) -> Path:
    return beta_dir / "mechanism" / f"seed_{seed}__{channel}__{fingerprint}.pt"


def run_mechanism_channel(
    legacy: Any,
    model: Any,
    model_cfg: Any,
    beta_cfg: BetaConfig,
    *,
    seed: int,
    channel: str,
    device: Any,
    force: bool,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    import torch

    path = mechanism_cache_path(beta_cfg.beta_dir, seed, channel, beta_cfg.fingerprint)
    if path.exists() and not force:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if _beta_cache_is_current(cached, beta_cfg, checkpoint_sha256):
            print(f"[mechanism seed={seed} {channel}] loaded {path}", flush=True)
            return cached
        print(f"[mechanism seed={seed} {channel}] stale cache; recomputing", flush=True)
    mode = legacy.MODE_SEMANTIC if channel == "semantic" else legacy.MODE_STRUCTURAL
    clean = legacy.make_batch(
        model_cfg,
        beta_cfg.mechanism_graphs,
        4_200_000 + seed * 10_000 + mode,
        mode=mode,
    )
    score_parts = []
    for start in range(0, len(clean), beta_cfg.score_batch_size):
        stop = min(start + beta_cfg.score_batch_size, len(clean))
        score_parts.append(
            mechanism_score_batch(
                legacy,
                model,
                model_cfg,
                clean.slice(start, stop),
                channel=channel,
                seed=4_300_000 + seed * 100_003 + start,
                events=beta_cfg.mechanism_events,
                device=device,
            )
        )
        print(f"[mechanism score seed={seed} {channel}] {stop}/{len(clean)}", flush=True)
    payload = {
        "version": BETA_VERSION,
        "schema": BETA_SCHEMA,
        "fingerprint": beta_cfg.fingerprint,
        "checkpoint_sha256": checkpoint_sha256,
        "seed": int(seed),
        "channel": channel,
        "score": _merge_tensor_dicts(score_parts),
        "patch": component_patch_sweep(
            legacy,
            model,
            model_cfg,
            beta_cfg,
            seed=seed,
            channel=channel,
            device=device,
        ),
    }
    atomic_torch_save(payload, path)
    print(f"[mechanism seed={seed} {channel}] cached {path}", flush=True)
    return payload


# ======================================================================================
# Score coordinates, uncertainty-aware families, and independent family ablations
# ======================================================================================


def rankdata(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(left: Iterable[float], right: Iterable[float]) -> float:
    x, y = np.asarray(list(left), dtype=float), np.asarray(list(right), dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 3:
        return float("nan")
    xr, yr = rankdata(x[keep]), rankdata(y[keep])
    if float(np.std(xr)) == 0.0 or float(np.std(yr)) == 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def within_layer_spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Average rank association within layers so depth cannot create the result."""

    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"within-layer correlations need aligned [L,H] arrays, got {x.shape}, {y.shape}")
    values = [spearman(x[layer], y[layer]) for layer in range(x.shape[0])]
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def score_coordinates(semantic: Any, structural: Any) -> dict[str, np.ndarray]:
    sem = np.asarray(semantic, dtype=float)
    st = np.asarray(structural, dtype=float)
    sem_tilde = sem / (float(np.mean(sem)) + EPS)
    str_tilde = st / (float(np.mean(st)) + EPS)
    denominator = sem_tilde + str_tilde
    d_rel = (sem_tilde - str_tilde) / (denominator + EPS)
    j = 0.5 * denominator
    g = np.sqrt(np.maximum(sem_tilde * str_tilde, 0.0))
    return {
        "semantic_tilde": sem_tilde,
        "structural_tilde": str_tilde,
        "D": d_rel,
        "J": j,
        "G": g,
    }


def bootstrap_coordinate_interval(
    semantic_per_graph: Any,
    structural_per_graph: Any,
    *,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    sem = np.asarray(semantic_per_graph, dtype=float)
    st = np.asarray(structural_per_graph, dtype=float)
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(samples)):
        sem_draw = sem[rng.integers(0, len(sem), len(sem))].mean(axis=0)
        st_draw = st[rng.integers(0, len(st), len(st))].mean(axis=0)
        draws.append(score_coordinates(sem_draw, st_draw)["D"])
    stacked = np.stack(draws, axis=0)
    return np.quantile(stacked, 0.025, axis=0), np.quantile(stacked, 0.975, axis=0)


def _head_order(matrix: np.ndarray, *, reverse: bool) -> list[tuple[int, int]]:
    heads = [
        (layer, head)
        for layer in range(matrix.shape[0])
        for head in range(matrix.shape[1])
    ]
    return sorted(heads, key=lambda item: (float(matrix[item]), item), reverse=reverse)


def select_beta_families(
    semantic_per_graph: Any,
    structural_per_graph: Any,
    beta_cfg: BetaConfig,
    *,
    seed: int,
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, Any]]:
    sem_pg = np.asarray(semantic_per_graph, dtype=float)
    str_pg = np.asarray(structural_per_graph, dtype=float)
    sem, st = sem_pg.mean(axis=0), str_pg.mean(axis=0)
    coordinates = score_coordinates(sem, st)
    lower, upper = bootstrap_coordinate_interval(
        sem_pg,
        str_pg,
        samples=min(beta_cfg.bootstrap_samples, 500),
        seed=seed,
    )
    size = int(beta_cfg.family_size)
    raw_activity = 0.5 * (sem + st)
    absolute_activity_floor = float(beta_cfg.activity_floor_relative) * float(
        np.nanmedian(raw_activity)
    )
    reliable = (
        (coordinates["J"] >= float(beta_cfg.j_floor))
        & (raw_activity >= absolute_activity_floor)
    )
    sem_special = reliable & (coordinates["D"] >= beta_cfg.selectivity_threshold) & (lower > 0)
    str_special = reliable & (coordinates["D"] <= -beta_cfg.selectivity_threshold) & (upper < 0)
    generalist = reliable & (np.abs(coordinates["D"]) < beta_cfg.selectivity_threshold)
    inactive = (
        (coordinates["J"] < float(beta_cfg.j_floor))
        & (raw_activity <= np.nanquantile(raw_activity, 0.25))
    )

    def take(order: Sequence[tuple[int, int]], mask: np.ndarray | None = None) -> list[tuple[int, int]]:
        selected = [item for item in order if mask is None or bool(mask[item])]
        return selected[:size]

    semantic_specialists = take(_head_order(coordinates["D"], reverse=True), sem_special)
    structural_specialists = take(_head_order(coordinates["D"], reverse=False), str_special)

    def j_matched_generalists(
        specialists: Sequence[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        available = [item for item in _head_order(coordinates["J"], reverse=True) if generalist[item]]
        chosen: list[tuple[int, int]] = []
        for target in specialists:
            candidates = [item for item in available if item not in chosen and item[0] == target[0]]
            if not candidates:
                candidates = [item for item in available if item not in chosen]
            if candidates:
                chosen.append(
                    min(candidates, key=lambda item: abs(coordinates["J"][item] - coordinates["J"][target]))
                )
        return chosen

    def same_layer_inert(
        specialists: Sequence[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        chosen: list[tuple[int, int]] = []
        for target in specialists:
            candidates = [
                item for item in _head_order(coordinates["J"], reverse=False)
                if item[0] == target[0] and item not in chosen and inactive[item]
            ]
            if candidates:
                chosen.append(candidates[0])
        return chosen

    groups = {
        "raw_semantic": take(_head_order(sem, reverse=True)),
        "raw_structural": take(_head_order(st, reverse=True)),
        "semantic_specialist": semantic_specialists,
        "structural_specialist": structural_specialists,
        "semantic_J_matched_generalist": j_matched_generalists(semantic_specialists),
        "structural_J_matched_generalist": j_matched_generalists(structural_specialists),
        "semantic_layer_inert": same_layer_inert(semantic_specialists),
        "structural_layer_inert": same_layer_inert(structural_specialists),
        "high_J_generalist": take(_head_order(coordinates["J"], reverse=True), generalist),
        "high_G_balanced": take(_head_order(coordinates["G"], reverse=True), reliable),
        "low_J_inert": take(_head_order(coordinates["J"], reverse=False), inactive),
    }
    diagnostics = {
        "coordinates": coordinates,
        "D_ci_lower": lower,
        "D_ci_upper": upper,
        "reliable": reliable,
        "inactive": inactive,
        "raw_activity": raw_activity,
        "absolute_activity_floor": absolute_activity_floor,
        "empty_families": [name for name, values in groups.items() if not values],
        "selection_rule": (
            "raw ranks are always reported; specialist/generalist families require J/activity "
            "and bootstrap selectivity criteria and are allowed to be empty"
        ),
    }
    return groups, diagnostics


def load_ablation_inputs(
    legacy: Any,
    model: Any,
    model_cfg: Any,
    beta_cfg: BetaConfig,
    *,
    seed: int,
    device: Any,
    checkpoint_sha256: str,
    force: bool = False,
) -> dict[str, Any]:
    import torch

    path = beta_cfg.beta_dir / "causal" / (
        f"seed_{seed}__ablation_fallback__{beta_cfg.fingerprint}__{checkpoint_sha256[:12]}.pt"
    )
    if path.exists() and not force:
        return _load_validated_beta_cache(path, beta_cfg, checkpoint_sha256)
    analysis_cfg = replace(model_cfg, ablation_graphs=beta_cfg.family_graphs)
    payload = {
        "version": BETA_VERSION,
        "schema": BETA_SCHEMA,
        "fingerprint": beta_cfg.fingerprint,
        "checkpoint_sha256": checkpoint_sha256,
        "source": "checkpoint-bound beta ablation; legacy cache deliberately not reused",
        "semantic": legacy.ablation_sweep(
            model,
            analysis_cfg,
            mode=legacy.MODE_SEMANTIC,
            seed=5_000_000 + seed,
            device=device,
        ),
        "structural": legacy.ablation_sweep(
            model,
            analysis_cfg,
            mode=legacy.MODE_STRUCTURAL,
            seed=5_100_000 + seed,
            device=device,
        ),
    }
    atomic_torch_save(payload, path)
    return payload


def family_cache_path(beta_dir: Path, seed: int, fingerprint: str) -> Path:
    return beta_dir / "families" / f"seed_{seed}__{fingerprint}.pt"


def run_family_ablation_seed(
    legacy: Any,
    model: Any,
    model_cfg: Any,
    beta_cfg: BetaConfig,
    score_payloads: Mapping[str, dict[str, Any]],
    *,
    seed: int,
    device: Any,
    force: bool,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    import torch

    path = family_cache_path(beta_cfg.beta_dir, seed, beta_cfg.fingerprint)
    if path.exists() and not force:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if _beta_cache_is_current(cached, beta_cfg, checkpoint_sha256):
            print(f"[families seed={seed}] loaded {path}", flush=True)
            return cached
        print(f"[families seed={seed}] stale cache; recomputing", flush=True)
    all_groups: dict[str, list[tuple[int, int]]] = {}
    selection: dict[str, Any] = {}
    for aggregation in AGGREGATIONS:
        sem_pg = score_payloads["semantic"]["score"]["per_graph"][aggregation]
        str_pg = score_payloads["structural"]["score"]["per_graph"][aggregation]
        groups, diagnostics = select_beta_families(
            sem_pg,
            str_pg,
            beta_cfg,
            seed=5_200_000 + seed * 101 + AGGREGATIONS.index(aggregation),
        )
        selection[aggregation] = {"groups": groups, "diagnostics": diagnostics}
        for name, values in groups.items():
            all_groups[f"{aggregation}_{name}"] = values

    analysis_cfg = replace(
        model_cfg,
        ablation_graphs=beta_cfg.family_graphs,
        top_group_size=beta_cfg.family_size,
    )
    ablation = legacy.family_ablation_sweep(
        model,
        analysis_cfg,
        groups=all_groups,
        seed=5_300_000 + seed,
        device=device,
        revision=f"{BETA_VERSION}-all-aggregations",
    )
    payload = {
        "version": BETA_VERSION,
        "schema": BETA_SCHEMA,
        "fingerprint": beta_cfg.fingerprint,
        "checkpoint_sha256": checkpoint_sha256,
        "seed": int(seed),
        "selection": selection,
        "ablation": ablation,
    }
    atomic_torch_save(payload, path)
    print(f"[families seed={seed}] cached {path}", flush=True)
    return payload


# ======================================================================================
# Reliability, conditional discovery, and locked method decision
# ======================================================================================


def _mean_head_metric(
    patch: Mapping[str, Any],
    *,
    direction: str,
    metric: str,
    floor: float,
) -> np.ndarray:
    values = np.asarray(patch["metrics"][direction][metric], dtype=float)
    valid = np.asarray(patch["effect_norm"], dtype=float) > float(floor)
    if not bool(valid.any()):
        return np.full(values.shape[:2], np.nan)
    return np.nanmean(values[..., valid], axis=-1)


def _noop_output_floor(
    score_payload: Mapping[str, Any],
    patch_payload: Mapping[str, Any] | None = None,
    *,
    relative: float = 0.05,
) -> float:
    clean = np.asarray(score_payload["no_op"]["clean_logits"], dtype=float)
    variant = np.asarray(score_payload["no_op"]["variant_logits"], dtype=float)
    movement = np.linalg.norm(clean[:, None, :] - variant, axis=-1)
    numerical = max(float(np.nanmax(movement)) * 5.0, 1.0e-7)
    if patch_payload is None:
        return numerical
    effects = np.asarray(patch_payload["effect_norm"], dtype=float)
    positive = effects[np.isfinite(effects) & (effects > numerical)]
    empirical = float(relative) * float(np.nanmedian(positive)) if len(positive) else 0.0
    return max(numerical, empirical)


def _topk_jaccard_stability(
    per_graph: np.ndarray,
    *,
    top_k: int,
    samples: int,
    seed: int,
) -> float:
    values = np.asarray(per_graph, dtype=float)
    reference = set(np.argsort(values.mean(axis=0).reshape(-1))[-int(top_k) :].tolist())
    rng = np.random.default_rng(int(seed))
    similarities = []
    for _ in range(int(samples)):
        draw = values[rng.integers(0, len(values), len(values))].mean(axis=0).reshape(-1)
        selected = set(np.argsort(draw)[-int(top_k) :].tolist())
        similarities.append(len(reference & selected) / max(len(reference | selected), 1))
    return float(np.mean(similarities))


def _split_half_rank(per_graph: np.ndarray) -> float:
    values = np.asarray(per_graph, dtype=float)
    midpoint = len(values) // 2
    if midpoint < 2:
        return float("nan")
    return spearman(values[:midpoint].mean(axis=0).reshape(-1), values[midpoint:].mean(axis=0).reshape(-1))


def _prefix_rank(score: Mapping[str, Any], aggregation: str) -> float:
    repeats = score.get("prefix_repeats", {"0": score["prefix"]})
    correlations = []
    for prefixes in repeats.values():
        numeric = sorted(int(key) for key in prefixes if key != "all")
        if not numeric:
            continue
        preferred = 6 if 6 in numeric else numeric[-1]
        left = np.asarray(prefixes[str(preferred)][aggregation], dtype=float).mean(axis=0)
        right = np.asarray(prefixes["all"][aggregation], dtype=float).mean(axis=0)
        correlations.append(spearman(left.reshape(-1), right.reshape(-1)))
    return float(np.nanmin(correlations)) if correlations else float("nan")


def _causal_coordinates(
    semantic_patch: Mapping[str, Any],
    structural_patch: Mapping[str, Any],
    semantic_floor: float,
    structural_floor: float,
) -> dict[str, np.ndarray]:
    sem_restore = _mean_head_metric(
        semantic_patch, direction="restore", metric="desired_projection", floor=semantic_floor
    )
    sem_inject = _mean_head_metric(
        semantic_patch, direction="inject", metric="desired_projection", floor=semantic_floor
    )
    str_restore = _mean_head_metric(
        structural_patch, direction="restore", metric="desired_projection", floor=structural_floor
    )
    str_inject = _mean_head_metric(
        structural_patch, direction="inject", metric="desired_projection", floor=structural_floor
    )
    semantic = 0.5 * (sem_restore + sem_inject)
    structural = 0.5 * (str_restore + str_inject)
    coordinates = _signed_causal_coordinates(semantic, structural)
    coordinates.update(
        {
            "semantic_restore": sem_restore,
            "semantic_inject": sem_inject,
            "structural_restore": str_restore,
            "structural_inject": str_inject,
        }
    )
    return coordinates


def _signed_causal_coordinates(
    semantic: np.ndarray, structural: np.ndarray
) -> dict[str, np.ndarray]:
    """Signed role plus nonnegative magnitude coordinates for finite causal effects."""

    semantic_scaled = semantic / (float(np.nanmean(np.abs(semantic))) + EPS)
    structural_scaled = structural / (float(np.nanmean(np.abs(structural))) + EPS)
    magnitude_sum = np.abs(semantic_scaled) + np.abs(structural_scaled)
    return {
        "semantic": semantic,
        "structural": structural,
        "semantic_tilde_signed": semantic_scaled,
        "structural_tilde_signed": structural_scaled,
        "D": (semantic_scaled - structural_scaled) / (magnitude_sum + EPS),
        "J": 0.5 * magnitude_sum,
        "G": np.sqrt(np.abs(semantic_scaled * structural_scaled)),
        "anti_aligned": (semantic < 0) | (structural < 0),
    }


def method_metrics_for_seed(
    score_payloads: Mapping[str, Mapping[str, Any]],
    patch_payloads: Mapping[str, Mapping[str, Any]],
    ablations: Mapping[str, Any],
    beta_cfg: BetaConfig,
    *,
    seed: int,
) -> dict[str, dict[str, float]]:
    semantic_floor = _noop_output_floor(
        score_payloads["semantic"], patch_payloads["semantic"],
        relative=beta_cfg.causal_effect_floor_relative,
    )
    structural_floor = _noop_output_floor(
        score_payloads["structural"], patch_payloads["structural"],
        relative=beta_cfg.causal_effect_floor_relative,
    )
    causal = _causal_coordinates(
        patch_payloads["semantic"],
        patch_payloads["structural"],
        semantic_floor,
        structural_floor,
    )
    sem_ablation = np.asarray(ablations["semantic"]["functional"], dtype=float).mean(axis=-1)
    str_ablation = np.asarray(ablations["structural"]["functional"], dtype=float).mean(axis=-1)
    output = {}
    for aggregation in AGGREGATIONS:
        sem_pg = np.asarray(
            score_payloads["semantic"]["score"]["per_graph"][aggregation], dtype=float
        )
        str_pg = np.asarray(
            score_payloads["structural"]["score"]["per_graph"][aggregation], dtype=float
        )
        sem, st = sem_pg.mean(axis=0), str_pg.mean(axis=0)
        coordinates = score_coordinates(sem, st)
        causal_sem_rho_pooled = spearman(sem.reshape(-1), causal["semantic"].reshape(-1))
        causal_str_rho_pooled = spearman(st.reshape(-1), causal["structural"].reshape(-1))
        causal_sem_rho = within_layer_spearman(sem, causal["semantic"])
        causal_str_rho = within_layer_spearman(st, causal["structural"])
        restore_sem_rho = within_layer_spearman(sem, causal["semantic_restore"])
        restore_str_rho = within_layer_spearman(st, causal["structural_restore"])
        inject_sem_rho = within_layer_spearman(sem, causal["semantic_inject"])
        inject_str_rho = within_layer_spearman(st, causal["structural_inject"])
        output[aggregation] = {
            "causal_semantic_rho": causal_sem_rho,
            "causal_structural_rho": causal_str_rho,
            "causal_same_channel_mean_rho": float(np.nanmean([causal_sem_rho, causal_str_rho])),
            "causal_semantic_rho_pooled": causal_sem_rho_pooled,
            "causal_structural_rho_pooled": causal_str_rho_pooled,
            "restore_same_channel_mean_rho": float(np.nanmean([restore_sem_rho, restore_str_rho])),
            "inject_same_channel_mean_rho": float(np.nanmean([inject_sem_rho, inject_str_rho])),
            "restore_semantic_rho": restore_sem_rho,
            "restore_structural_rho": restore_str_rho,
            "inject_semantic_rho": inject_sem_rho,
            "inject_structural_rho": inject_str_rho,
            "ablation_semantic_rho": within_layer_spearman(sem, sem_ablation),
            "ablation_structural_rho": within_layer_spearman(st, str_ablation),
            "D_causal_role_rho": within_layer_spearman(coordinates["D"], causal["D"]),
            "J_causal_strength_rho": within_layer_spearman(coordinates["J"], causal["J"]),
            "G_balanced_causal_rho": within_layer_spearman(coordinates["G"], causal["G"]),
            "semantic_split_half_rho": _split_half_rank(sem_pg),
            "structural_split_half_rho": _split_half_rank(str_pg),
            "semantic_k_to_all_rho": _prefix_rank(
                score_payloads["semantic"]["score"], aggregation
            ),
            "structural_k_to_all_rho": _prefix_rank(
                score_payloads["structural"]["score"], aggregation
            ),
            "semantic_topk_stability": _topk_jaccard_stability(
                sem_pg,
                top_k=beta_cfg.family_size,
                samples=min(beta_cfg.bootstrap_samples, 500),
                seed=6_000_000 + seed * 101 + AGGREGATIONS.index(aggregation),
            ),
            "structural_topk_stability": _topk_jaccard_stability(
                str_pg,
                top_k=beta_cfg.family_size,
                samples=min(beta_cfg.bootstrap_samples, 500),
                seed=6_100_000 + seed * 101 + AGGREGATIONS.index(aggregation),
            ),
            "semantic_effect_floor": semantic_floor,
            "structural_effect_floor": structural_floor,
            "semantic_excluded_event_fraction": float(
                np.mean(np.asarray(patch_payloads["semantic"]["effect_norm"], dtype=float) <= semantic_floor)
            ),
            "structural_excluded_event_fraction": float(
                np.mean(np.asarray(patch_payloads["structural"]["effect_norm"], dtype=float) <= structural_floor)
            ),
            "causal_anti_aligned_head_fraction": float(np.nanmean(causal["anti_aligned"])),
            "ranking_bootstrap_samples": int(min(beta_cfg.bootstrap_samples, 500)),
        }
    return output


def choose_and_confirm_method(
    per_seed: Mapping[int, Mapping[str, Mapping[str, float]]],
    beta_cfg: BetaConfig,
) -> dict[str, Any]:
    selection_seeds = tuple(beta_cfg.seeds[:2])
    confirmation_seed = int(beta_cfg.seeds[2])
    selection_rows = {}
    for aggregation in AGGREGATIONS:
        causal = np.asarray(
            [per_seed[seed][aggregation]["causal_same_channel_mean_rho"] for seed in selection_seeds]
        )
        role = np.asarray(
            [per_seed[seed][aggregation]["D_causal_role_rho"] for seed in selection_seeds]
        )
        reliability = np.asarray(
            [
                np.nanmean(
                    [
                        per_seed[seed][aggregation]["semantic_split_half_rho"],
                        per_seed[seed][aggregation]["structural_split_half_rho"],
                    ]
                )
                for seed in selection_seeds
            ]
        )
        restore = np.asarray(
            [per_seed[seed][aggregation]["restore_same_channel_mean_rho"] for seed in selection_seeds]
        )
        inject = np.asarray(
            [per_seed[seed][aggregation]["inject_same_channel_mean_rho"] for seed in selection_seeds]
        )
        ablation = np.asarray(
            [
                np.nanmean(
                    [
                        per_seed[seed][aggregation]["ablation_semantic_rho"],
                        per_seed[seed][aggregation]["ablation_structural_rho"],
                    ]
                )
                for seed in selection_seeds
            ]
        )
        selection_rows[aggregation] = {
            "causal_mean": float(np.nanmean(causal)),
            "restore_mean": float(np.nanmean(restore)),
            "inject_mean": float(np.nanmean(inject)),
            "bidirectional_min": float(min(np.nanmean(restore), np.nanmean(inject))),
            "ablation_mean": float(np.nanmean(ablation)),
            "D_role_mean": float(np.nanmean(role)),
            "reliability_mean": float(np.nanmean(reliability)),
        }
    eligible = [
        name
        for name, row in selection_rows.items()
        if row["reliability_mean"] >= 0.80
    ] or list(AGGREGATIONS)
    selected = max(
        eligible,
        key=lambda name: (
            selection_rows[name]["bidirectional_min"],
            selection_rows[name]["ablation_mean"],
            selection_rows[name]["D_role_mean"],
            selection_rows[name]["reliability_mean"],
            -AGGREGATIONS.index(name),
        ),
    )
    candidate = per_seed[confirmation_seed][selected]
    legacy = per_seed[confirmation_seed]["CG"]
    restore_improvement = candidate["restore_same_channel_mean_rho"] - legacy["restore_same_channel_mean_rho"]
    inject_improvement = candidate["inject_same_channel_mean_rho"] - legacy["inject_same_channel_mean_rho"]
    improvement = 0.5 * (restore_improvement + inject_improvement)
    channel_noninferiority = all(
        candidate[f"{direction}_{channel}_rho"]
        >= legacy[f"{direction}_{channel}_rho"] - 0.05
        for direction in ("restore", "inject")
        for channel in CHANNELS
    )
    d_noninferiority = candidate["D_causal_role_rho"] >= legacy["D_causal_role_rho"] - 0.05
    k_gate = min(
        candidate["semantic_k_to_all_rho"], candidate["structural_k_to_all_rho"]
    ) >= 0.90
    stability_gate = min(
        candidate["semantic_topk_stability"], candidate["structural_topk_stability"]
    ) >= 0.67
    candidate_ablation = np.nanmean(
        [candidate["ablation_semantic_rho"], candidate["ablation_structural_rho"]]
    )
    legacy_ablation = np.nanmean(
        [legacy["ablation_semantic_rho"], legacy["ablation_structural_rho"]]
    )
    checks = {
        "seed2_mean_causal_improvement_at_least_0.05": bool(improvement >= 0.05),
        "seed2_restore_improves": bool(restore_improvement > 0),
        "seed2_inject_improves": bool(inject_improvement > 0),
        "seed2_no_channel_worse_by_more_than_0.05": bool(channel_noninferiority),
        "seed2_D_role_noninferior": bool(d_noninferiority),
        "seed2_ablation_noninferior": bool(candidate_ablation >= legacy_ablation - 0.05),
        "seed2_K_to_all_rank_at_least_0.90": bool(k_gate),
        "seed2_top_family_stability_at_least_0.67": bool(stability_gate),
    }
    advances = selected != "CG" and all(checks.values())
    verdict = (
        f"advance {selected} to real-task beta validation; do not integrate into production yet"
        if advances
        else "retain coherent-gross CG as production score pending further evidence"
    )
    return {
        "selection_seeds": list(selection_seeds),
        "confirmation_seed": confirmation_seed,
        "selection_metrics": selection_rows,
        "selected_on_seeds_0_1": selected,
        "seed2_causal_improvement_over_CG": float(improvement),
        "seed2_restore_improvement_over_CG": float(restore_improvement),
        "seed2_inject_improvement_over_CG": float(inject_improvement),
        "confirmation_checks": checks,
        "advances": bool(advances),
        "verdict": verdict,
        "interpretation": (
            "This task is an essential synthetic gate. Passing advances a candidate to real-task "
            "validation; it is not sufficient for production integration."
        ),
    }


def _mean_head_metric_indices(
    patch: Mapping[str, Any],
    *,
    direction: str,
    metric: str,
    floor: float,
    indices: np.ndarray,
) -> np.ndarray:
    values = np.asarray(patch["metrics"][direction][metric], dtype=float)[..., indices]
    valid = np.asarray(patch["effect_norm"], dtype=float)[indices] > float(floor)
    values = np.where(valid[None, None, :], values, np.nan)
    return np.nanmean(values, axis=-1)


def _cluster_event_bootstrap_indices(
    patch: Mapping[str, Any], rng: np.random.Generator
) -> np.ndarray:
    graph = np.asarray(patch["event_metadata"]["graph"], dtype=int)
    unique = np.unique(graph)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(graph == value) for value in sampled])


def paired_graph_bootstrap_confirmation(
    run: Mapping[str, Any],
    candidate: str,
    beta_cfg: BetaConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    """Paired graph bootstrap for candidate-minus-CG on the held-out checkpoint.

    Score graphs and causal graphs are resampled independently.  All sampled donor
    events for a causal graph move together; score donors are exhaustive populations.
    Heads are never resampled.
    """

    if candidate == "CG":
        return {
            "candidate": candidate,
            "samples": 0,
            "gate_passed": False,
            "reason": "CG is the selection winner; no replacement claim is tested",
        }
    rng = np.random.default_rng(int(seed))
    floors = {
        channel: _noop_output_floor(
            run["scores"][channel],
            run["patches"][channel],
            relative=beta_cfg.causal_effect_floor_relative,
        )
        for channel in CHANNELS
    }
    draws: dict[str, list[float]] = {
        "restore_mean": [],
        "inject_mean": [],
        "bidirectional_mean": [],
        "D_role": [],
        "ablation_mean": [],
    }
    for direction in ("restore", "inject"):
        for channel in CHANNELS:
            draws[f"{direction}_{channel}"] = []

    for _ in range(int(beta_cfg.bootstrap_samples)):
        score_mean: dict[str, dict[str, np.ndarray]] = {candidate: {}, "CG": {}}
        causal_by_direction: dict[str, dict[str, np.ndarray]] = {
            direction: {} for direction in ("restore", "inject")
        }
        ablation_target: dict[str, np.ndarray] = {}
        for channel in CHANNELS:
            score = run["scores"][channel]["score"]["per_graph"]
            graphs = len(score[candidate])
            graph_indices = rng.integers(0, graphs, graphs)
            for method in (candidate, "CG"):
                score_mean[method][channel] = np.asarray(
                    score[method], dtype=float
                )[graph_indices].mean(axis=0)

            patch = run["patches"][channel]
            event_indices = _cluster_event_bootstrap_indices(patch, rng)
            for direction in ("restore", "inject"):
                causal_by_direction[direction][channel] = _mean_head_metric_indices(
                    patch,
                    direction=direction,
                    metric="desired_projection",
                    floor=floors[channel],
                    indices=event_indices,
                )

            ablation = np.asarray(
                run["ablations"][channel]["functional"], dtype=float
            )
            ablation_indices = rng.integers(0, ablation.shape[-1], ablation.shape[-1])
            ablation_target[channel] = ablation[..., ablation_indices].mean(axis=-1)

        direction_improvements = []
        for direction in ("restore", "inject"):
            channel_improvements = []
            for channel in CHANNELS:
                target = causal_by_direction[direction][channel]
                candidate_rho = within_layer_spearman(score_mean[candidate][channel], target)
                cg_rho = within_layer_spearman(score_mean["CG"][channel], target)
                difference = candidate_rho - cg_rho
                draws[f"{direction}_{channel}"].append(difference)
                channel_improvements.append(difference)
            direction_value = float(np.nanmean(channel_improvements))
            draws[f"{direction}_mean"].append(direction_value)
            direction_improvements.append(direction_value)
        draws["bidirectional_mean"].append(float(np.nanmean(direction_improvements)))

        causal_semantic = 0.5 * (
            causal_by_direction["restore"]["semantic"]
            + causal_by_direction["inject"]["semantic"]
        )
        causal_structural = 0.5 * (
            causal_by_direction["restore"]["structural"]
            + causal_by_direction["inject"]["structural"]
        )
        causal_coordinates = _signed_causal_coordinates(causal_semantic, causal_structural)
        d_values = {}
        for method in (candidate, "CG"):
            score_coord = score_coordinates(
                score_mean[method]["semantic"], score_mean[method]["structural"]
            )
            d_values[method] = within_layer_spearman(
                score_coord["D"], causal_coordinates["D"]
            )
        draws["D_role"].append(d_values[candidate] - d_values["CG"])

        ablation_values = {}
        for method in (candidate, "CG"):
            ablation_values[method] = float(
                np.nanmean(
                    [
                        within_layer_spearman(
                            score_mean[method][channel], ablation_target[channel]
                        )
                        for channel in CHANNELS
                    ]
                )
            )
        draws["ablation_mean"].append(
            ablation_values[candidate] - ablation_values["CG"]
        )

    summary: dict[str, dict[str, float]] = {}
    for name, values in draws.items():
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        summary[name] = {
            "median": float(np.median(array)) if len(array) else float("nan"),
            "ci_low": float(np.quantile(array, 0.025)) if len(array) else float("nan"),
            "ci_high": float(np.quantile(array, 0.975)) if len(array) else float("nan"),
            "probability_gt_zero": float(np.mean(array > 0)) if len(array) else 0.0,
            "probability_ge_minus_0.05": float(np.mean(array >= -0.05)) if len(array) else 0.0,
        }
    checks = {
        "bidirectional_improvement_ci_above_zero": summary["bidirectional_mean"]["ci_low"] > 0,
        "restore_direction_probability_positive_at_least_0.80": summary["restore_mean"]["probability_gt_zero"] >= 0.80,
        "inject_direction_probability_positive_at_least_0.80": summary["inject_mean"]["probability_gt_zero"] >= 0.80,
        "all_channel_direction_differences_noninferior": all(
            summary[f"{direction}_{channel}"]["probability_ge_minus_0.05"] >= 0.95
            for direction in ("restore", "inject")
            for channel in CHANNELS
        ),
        "D_role_difference_noninferior": summary["D_role"]["probability_ge_minus_0.05"] >= 0.95,
        "ablation_difference_noninferior": summary["ablation_mean"]["probability_ge_minus_0.05"] >= 0.95,
    }
    return {
        "candidate": candidate,
        "samples": int(beta_cfg.bootstrap_samples),
        "resampling_unit": "graphs; all causal events within a graph move together; heads fixed",
        "intervals": summary,
        "checks": checks,
        "gate_passed": bool(all(checks.values())),
    }


def _condition_candidates(metadata: Mapping[str, Any], indices: np.ndarray) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for feature in ("source_query_distance", "source_value", "query_value"):
        values = np.asarray(metadata[feature], dtype=float)[indices]
        quantiles = np.unique(np.quantile(values, [0.25, 0.5, 0.75]))
        for threshold in quantiles:
            candidates.append(
                {"feature": feature, "operator": "le", "threshold": float(threshold)}
            )
    for feature in (
        "source_value_upper_half", "query_value_upper_half", "query_parity"
    ):
        candidates.append({"feature": feature, "operator": "eq", "threshold": 1.0})
    return candidates


def _apply_condition(metadata: Mapping[str, Any], rule: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(metadata[str(rule["feature"])], dtype=float)
    if rule["operator"] == "le":
        return values <= float(rule["threshold"])
    if rule["operator"] == "eq":
        return values == float(rule["threshold"])
    raise ValueError(f"unknown condition operator {rule['operator']!r}")


def _graph_mean_dose(payload: Mapping[str, Any]) -> np.ndarray:
    score = payload["score"]
    dose = np.asarray(score["dose"], dtype=float)
    valid = np.asarray(score["valid"], dtype=bool)
    return (dose * valid).sum(axis=1) / valid.sum(axis=1).clip(min=1)


def _dose_common_support_indices(
    dose: np.ndarray,
    condition: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Trim both strata to their central overlapping dose range."""

    values_1 = dose[indices][condition[indices]]
    values_0 = dose[indices][~condition[indices]]
    if min(len(values_1), len(values_0)) < 2:
        return np.asarray([], dtype=int), float("nan")
    low = max(float(np.quantile(values_1, 0.05)), float(np.quantile(values_0, 0.05)))
    high = min(float(np.quantile(values_1, 0.95)), float(np.quantile(values_0, 0.95)))
    if low > high:
        return np.asarray([], dtype=int), float("inf")
    keep = indices[(dose[indices] >= low) & (dose[indices] <= high)]
    pooled = np.concatenate((values_1, values_0))
    scale = float(np.std(pooled))
    difference = abs(float(np.mean(values_1)) - float(np.mean(values_0))) / (scale + EPS)
    return keep, difference


def _spatial_phenotypes(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Source-centred carrier localisation summaries [G,L,H]."""

    q = np.asarray(payload["score"]["q_mean_carrier"], dtype=float)
    magnitude = np.linalg.norm(q, axis=-1)
    probability = magnitude / (magnitude.sum(axis=-1, keepdims=True) + EPS)
    entropy = -(probability * np.log(probability + EPS)).sum(axis=-1)
    source = np.asarray(
        payload["score"]["clean_metadata"]["source_index"], dtype=int
    )
    nodes = magnitude.shape[-1]
    positions = np.arange(nodes)[None, :]
    displacement = np.abs(positions - source[:, None])
    distance = np.minimum(displacement, nodes - displacement)[:, None, None, :]
    return {
        "carrier_entropy": entropy,
        "effective_carriers": np.exp(entropy),
        "near_source_mass": (probability * (distance <= 1)).sum(axis=-1),
        "response_radius": (probability * distance).sum(axis=-1),
    }


def _additive_channel_condition_interaction(
    semantic: np.ndarray,
    structural: np.ndarray,
    sem_condition: np.ndarray,
    str_condition: np.ndarray,
) -> np.ndarray:
    return (
        semantic[sem_condition].mean(axis=0)
        - semantic[~sem_condition].mean(axis=0)
        - structural[str_condition].mean(axis=0)
        + structural[~str_condition].mean(axis=0)
    )


def _log_interaction(
    semantic: np.ndarray,
    structural: np.ndarray,
    semantic_condition: np.ndarray,
    structural_condition: np.ndarray,
) -> np.ndarray:
    sem_1 = semantic[semantic_condition].mean(axis=0)
    sem_0 = semantic[~semantic_condition].mean(axis=0)
    str_1 = structural[structural_condition].mean(axis=0)
    str_0 = structural[~structural_condition].mean(axis=0)
    scale = max(float(np.nanmean([sem_1, sem_0, str_1, str_0])), EPS)
    floor = scale * 1.0e-8
    return (
        np.log(np.maximum(sem_1, floor))
        - np.log(np.maximum(sem_0, floor))
        - np.log(np.maximum(str_1, floor))
        + np.log(np.maximum(str_0, floor))
    )


def _bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 1.0
    for rank in range(len(p), 0, -1):
        idx = order[rank - 1]
        running = min(running, p[idx] * len(p) / rank)
        adjusted[idx] = running
    return adjusted


def conditional_discovery_confirmation(
    semantic_payload: Mapping[str, Any],
    structural_payload: Mapping[str, Any],
    *,
    aggregation: str,
    beta_cfg: BetaConfig,
    seed: int,
) -> list[dict[str, Any]]:
    sem = np.asarray(semantic_payload["score"]["per_graph"][aggregation], dtype=float)
    st = np.asarray(structural_payload["score"]["per_graph"][aggregation], dtype=float)
    sem_meta = semantic_payload["score"]["clean_metadata"]
    str_meta = structural_payload["score"]["clean_metadata"]
    sem_dose = _graph_mean_dose(semantic_payload)
    str_dose = _graph_mean_dose(structural_payload)
    sem_spatial = _spatial_phenotypes(semantic_payload)
    str_spatial = _spatial_phenotypes(structural_payload)
    sem_split, str_split = len(sem) // 2, len(st) // 2
    sem_discovery, sem_confirm = np.arange(sem_split), np.arange(sem_split, len(sem))
    str_discovery, str_confirm = np.arange(str_split), np.arange(str_split, len(st))
    candidates = _condition_candidates(sem_meta, sem_discovery)
    selected: list[dict[str, Any]] = []
    minimum_sem = max(
        beta_cfg.min_condition_graphs,
        int(math.ceil(beta_cfg.min_condition_fraction * len(sem_discovery))),
    )
    minimum_str = max(
        beta_cfg.min_condition_graphs,
        int(math.ceil(beta_cfg.min_condition_fraction * len(str_discovery))),
    )
    candidate_cache = []
    for rule in candidates:
        sem_condition = _apply_condition(sem_meta, rule)
        str_condition = _apply_condition(str_meta, rule)
        support = (
            int(sem_condition[sem_discovery].sum()),
            int((~sem_condition[sem_discovery]).sum()),
            int(str_condition[str_discovery].sum()),
            int((~str_condition[str_discovery]).sum()),
        )
        if min(support[:2]) < minimum_sem or min(support[2:]) < minimum_str:
            continue
        interaction = _log_interaction(
            sem[sem_discovery],
            st[str_discovery],
            sem_condition[sem_discovery],
            str_condition[str_discovery],
        )
        candidate_cache.append((rule, sem_condition, str_condition, interaction, support))
    if not candidate_cache:
        return []
    layers, heads = sem.shape[1:]
    for layer in range(layers):
        for head in range(heads):
            rule, sem_condition, str_condition, discovery_i, support = max(
                candidate_cache,
                key=lambda item: abs(float(item[3][layer, head])),
            )
            sem_confirm_supported, sem_dose_difference = _dose_common_support_indices(
                sem_dose, sem_condition, sem_confirm
            )
            str_confirm_supported, str_dose_difference = _dose_common_support_indices(
                str_dose, str_condition, str_confirm
            )
            sem_confirm_condition = sem_condition[sem_confirm_supported]
            str_confirm_condition = str_condition[str_confirm_supported]
            confirmation_support = (
                int(sem_confirm_condition.sum()),
                int((~sem_confirm_condition).sum()),
                int(str_confirm_condition.sum()),
                int((~str_confirm_condition).sum()),
            )
            confirmation_support_valid = min(confirmation_support) >= min(
                beta_cfg.min_condition_graphs,
                max(2, min(len(sem_confirm), len(str_confirm)) // 4),
            )
            confirmation_i = (
                _log_interaction(
                    sem[sem_confirm_supported],
                    st[str_confirm_supported],
                    sem_confirm_condition,
                    str_confirm_condition,
                )[layer, head]
                if confirmation_support_valid
                else float("nan")
            )
            sem_discovery_head = sem[sem_discovery, layer, head]
            str_discovery_head = st[str_discovery, layer, head]
            stratum_means = [
                float(sem_discovery_head[sem_condition[sem_discovery]].mean()),
                float(sem_discovery_head[~sem_condition[sem_discovery]].mean()),
                float(str_discovery_head[str_condition[str_discovery]].mean()),
                float(str_discovery_head[~str_condition[str_discovery]].mean()),
            ]
            activity_reference = 0.5 * (
                float(sem_discovery_head.mean()) + float(str_discovery_head.mean())
            )
            activity_valid = min(stratum_means) >= (
                beta_cfg.activity_floor_relative * activity_reference
            )
            confirmation_stratum_means = (
                [
                    float(sem[sem_confirm_supported, layer, head][sem_confirm_condition].mean()),
                    float(sem[sem_confirm_supported, layer, head][~sem_confirm_condition].mean()),
                    float(st[str_confirm_supported, layer, head][str_confirm_condition].mean()),
                    float(st[str_confirm_supported, layer, head][~str_confirm_condition].mean()),
                ]
                if confirmation_support_valid
                else [float("nan")] * 4
            )
            confirmation_reference = (
                float(np.mean(confirmation_stratum_means))
                if confirmation_support_valid else float("nan")
            )
            confirmation_activity_valid = bool(
                confirmation_support_valid
                and np.isfinite(confirmation_stratum_means).all()
                and min(confirmation_stratum_means)
                >= beta_cfg.activity_floor_relative * confirmation_reference
            )
            sign_replicated = bool(
                np.isfinite(confirmation_i)
                and float(discovery_i[layer, head]) * float(confirmation_i) > 0
            )
            spatial_values = {}
            conditional_coordinates: dict[str, float] = {}
            if confirmation_support_valid:
                for name in sem_spatial:
                    spatial_values[f"{name}_interaction"] = float(
                        _additive_channel_condition_interaction(
                            sem_spatial[name][sem_confirm_supported],
                            str_spatial[name][str_confirm_supported],
                            sem_confirm_condition,
                            str_confirm_condition,
                        )[layer, head]
                    )
                sem_reference = float(np.mean(sem[sem_confirm_supported])) + EPS
                str_reference = float(np.mean(st[str_confirm_supported])) + EPS
                for label, sem_mask, str_mask in (
                    ("condition_1", sem_confirm_condition, str_confirm_condition),
                    ("condition_0", ~sem_confirm_condition, ~str_confirm_condition),
                ):
                    sem_value = sem[sem_confirm_supported][sem_mask].mean(axis=0) / sem_reference
                    str_value = st[str_confirm_supported][str_mask].mean(axis=0) / str_reference
                    denominator = sem_value + str_value
                    conditional_coordinates[f"D_{label}"] = float(
                        ((sem_value - str_value) / (denominator + EPS))[layer, head]
                    )
                    conditional_coordinates[f"J_{label}"] = float(
                        (0.5 * denominator)[layer, head]
                    )
                    conditional_coordinates[f"G_{label}"] = float(
                        np.sqrt(np.maximum(sem_value * str_value, 0.0))[layer, head]
                    )
                for coordinate in ("D", "J", "G"):
                    conditional_coordinates[f"delta_{coordinate}"] = (
                        conditional_coordinates[f"{coordinate}_condition_1"]
                        - conditional_coordinates[f"{coordinate}_condition_0"]
                    )
            selected.append(
                {
                    "seed": int(seed),
                    "layer": layer,
                    "head": head,
                    **rule,
                    "discovery_interaction": float(discovery_i[layer, head]),
                    "confirmation_interaction": float(confirmation_i),
                    "discovery_support": list(support),
                    "confirmation_support": list(confirmation_support),
                    "confirmation_support_valid": bool(confirmation_support_valid),
                    "stratum_means": stratum_means,
                    "activity_valid": bool(activity_valid),
                    "confirmation_stratum_means": confirmation_stratum_means,
                    "confirmation_activity_valid": confirmation_activity_valid,
                    "dose_common_support_valid": bool(confirmation_support_valid),
                    "semantic_dose_standardized_difference": sem_dose_difference,
                    "structural_dose_standardized_difference": str_dose_difference,
                    "sign_replicated": sign_replicated,
                    **spatial_values,
                    **conditional_coordinates,
                    "_sem_values": sem[sem_confirm_supported, layer, head],
                    "_str_values": st[str_confirm_supported, layer, head],
                    "_sem_condition": sem_confirm_condition,
                    "_str_condition": str_confirm_condition,
                }
            )

    rng = np.random.default_rng(6_500_000 + seed)
    for row in selected:
        valid_row = (
            bool(row["activity_valid"])
            and bool(row["confirmation_activity_valid"])
            and bool(row["confirmation_support_valid"])
            and bool(row["sign_replicated"])
            and np.isfinite(float(row["confirmation_interaction"]))
        )
        if not valid_row:
            row["bootstrap_p"] = 1.0
            continue
        observed = abs(float(row["confirmation_interaction"]))
        null = []
        for _ in range(beta_cfg.conditional_bootstrap_samples):
            sem_indices = rng.integers(0, len(row["_sem_values"]), len(row["_sem_values"]))
            str_indices = rng.integers(0, len(row["_str_values"]), len(row["_str_values"]))
            sem_condition = row["_sem_condition"][sem_indices]
            str_condition = row["_str_condition"][str_indices]
            if min(
                int(sem_condition.sum()), int((~sem_condition).sum()),
                int(str_condition.sum()), int((~str_condition).sum()),
            ) == 0:
                continue
            value = _log_interaction(
                row["_sem_values"][sem_indices, None, None],
                row["_str_values"][str_indices, None, None],
                sem_condition,
                str_condition,
            )[0, 0]
            if np.isfinite(value):
                null.append(float(value) - float(row["confirmation_interaction"]))
        row["bootstrap_p"] = float(
            (1 + np.sum(np.abs(np.asarray(null)) >= observed)) / (1 + len(null))
        )
    for row in selected:
        # FDR is applied once across every reported seed/aggregation in create_outputs.
        row["fdr_q"] = float("nan")
        row["confirmed_fdr_0.05"] = False
        row["inference"] = "sample-split, dose-overlap-trimmed, centred graph bootstrap"
        row["conditional_bootstrap_samples"] = int(beta_cfg.conditional_bootstrap_samples)
        row.pop("_sem_condition", None)
        row.pop("_str_condition", None)
        row.pop("_sem_values", None)
        row.pop("_str_values", None)
    return selected


# ======================================================================================
# Paper tables and figures
# ======================================================================================


LAYER_COLOURS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")
SEED_MARKERS = ("o", "s", "^")
FAMILY_COLOURS = {
    "raw_semantic": "#D55E00",
    "raw_structural": "#0072B2",
    "semantic_specialist": "#E69F00",
    "structural_specialist": "#56B4E9",
    "high_J_generalist": "#009E73",
    "high_G_balanced": "#CC79A7",
    "low_J_inert": "#777777",
}


def configure_matplotlib() -> Any:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "legend.frameon": False,
        }
    )
    return plt


def save_figure(fig: Any, path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".png", ".pdf"):
        target = path.with_suffix(suffix)
        fig.savefig(target, facecolor="white")
        outputs.append(str(target))
    configure_matplotlib().close(fig)
    return outputs


def build_head_rows(
    runs: Mapping[int, Mapping[str, Any]],
    beta_cfg: BetaConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in beta_cfg.seeds:
        run = runs[int(seed)]
        sem_floor = _noop_output_floor(
            run["scores"]["semantic"], run["patches"]["semantic"],
            relative=beta_cfg.causal_effect_floor_relative,
        )
        str_floor = _noop_output_floor(
            run["scores"]["structural"], run["patches"]["structural"],
            relative=beta_cfg.causal_effect_floor_relative,
        )
        causal = _causal_coordinates(
            run["patches"]["semantic"],
            run["patches"]["structural"],
            sem_floor,
            str_floor,
        )
        sem_ablation = np.asarray(
            run["ablations"]["semantic"]["functional"], dtype=float
        ).mean(axis=-1)
        str_ablation = np.asarray(
            run["ablations"]["structural"]["functional"], dtype=float
        ).mean(axis=-1)
        for aggregation in AGGREGATIONS:
            sem = np.asarray(
                run["scores"]["semantic"]["score"]["per_graph"][aggregation], dtype=float
            ).mean(axis=0)
            st = np.asarray(
                run["scores"]["structural"]["score"]["per_graph"][aggregation], dtype=float
            ).mean(axis=0)
            coordinates = score_coordinates(sem, st)
            sem_route = np.asarray(
                run["mechanisms"]["semantic"]["score"]["components"]["routing"]
                ["per_graph"][aggregation],
                dtype=float,
            ).mean(axis=0)
            sem_message = np.asarray(
                run["mechanisms"]["semantic"]["score"]["components"]["message"]
                ["per_graph"][aggregation],
                dtype=float,
            ).mean(axis=0)
            sem_mechanism_total = np.asarray(
                run["mechanisms"]["semantic"]["score"]["components"]["total"]
                ["per_graph"][aggregation],
                dtype=float,
            ).mean(axis=0)
            str_route = np.asarray(
                run["mechanisms"]["structural"]["score"]["components"]["routing"]
                ["per_graph"][aggregation],
                dtype=float,
            ).mean(axis=0)
            str_message = np.asarray(
                run["mechanisms"]["structural"]["score"]["components"]["message"]
                ["per_graph"][aggregation],
                dtype=float,
            ).mean(axis=0)
            str_mechanism_total = np.asarray(
                run["mechanisms"]["structural"]["score"]["components"]["total"]
                ["per_graph"][aggregation],
                dtype=float,
            ).mean(axis=0)
            coherence_arrays = {}
            for channel in CHANNELS:
                channel_score = run["scores"][channel]["score"]["per_graph"]
                coherence_arrays[channel] = {
                    name: np.asarray(channel_score[name], dtype=float).mean(axis=0)
                    for name in ("CG", "EG", "CN", "EN")
                }
            for layer in range(sem.shape[0]):
                for head in range(sem.shape[1]):
                    rows.append(
                        {
                            "seed": int(seed),
                            "aggregation": aggregation,
                            "layer": layer,
                            "head": head,
                            "semantic_score": float(sem[layer, head]),
                            "structural_score": float(st[layer, head]),
                            "semantic_tilde": float(coordinates["semantic_tilde"][layer, head]),
                            "structural_tilde": float(coordinates["structural_tilde"][layer, head]),
                            "D": float(coordinates["D"][layer, head]),
                            "J_evoked": float(coordinates["J"][layer, head]),
                            "G_balanced": float(coordinates["G"][layer, head]),
                            "semantic_causal": float(causal["semantic"][layer, head]),
                            "structural_causal": float(causal["structural"][layer, head]),
                            "semantic_restore": float(causal["semantic_restore"][layer, head]),
                            "semantic_inject": float(causal["semantic_inject"][layer, head]),
                            "structural_restore": float(causal["structural_restore"][layer, head]),
                            "structural_inject": float(causal["structural_inject"][layer, head]),
                            "causal_D": float(causal["D"][layer, head]),
                            "causal_J": float(causal["J"][layer, head]),
                            "causal_G": float(causal["G"][layer, head]),
                            "causal_anti_aligned": bool(causal["anti_aligned"][layer, head]),
                            "semantic_ablation": float(sem_ablation[layer, head]),
                            "structural_ablation": float(str_ablation[layer, head]),
                            "semantic_routing_score": float(sem_route[layer, head]),
                            "semantic_message_score": float(sem_message[layer, head]),
                            "semantic_routing_share": float(
                                sem_route[layer, head]
                                / (sem_route[layer, head] + sem_message[layer, head] + EPS)
                            ),
                            "semantic_component_alignment": float(
                                sem_mechanism_total[layer, head]
                                / (sem_route[layer, head] + sem_message[layer, head] + EPS)
                            ),
                            "structural_routing_score": float(str_route[layer, head]),
                            "structural_message_score": float(str_message[layer, head]),
                            "structural_routing_share": float(
                                str_route[layer, head]
                                / (str_route[layer, head] + str_message[layer, head] + EPS)
                            ),
                            "structural_component_alignment": float(
                                str_mechanism_total[layer, head]
                                / (str_route[layer, head] + str_message[layer, head] + EPS)
                            ),
                            "donor_coherence_semantic": float(
                                coherence_arrays["semantic"]["CN"][layer, head]
                                / (coherence_arrays["semantic"]["EN"][layer, head] + EPS)
                            )
                            if coherence_arrays["semantic"]["EN"][layer, head]
                            >= beta_cfg.activity_floor_relative
                            * np.nanmedian(coherence_arrays["semantic"]["EN"])
                            else float("nan"),
                            "carrier_coherence_semantic": float(
                                coherence_arrays["semantic"]["EN"][layer, head]
                                / (coherence_arrays["semantic"]["EG"][layer, head] + EPS)
                            )
                            if coherence_arrays["semantic"]["EG"][layer, head]
                            >= beta_cfg.activity_floor_relative
                            * np.nanmedian(coherence_arrays["semantic"]["EG"])
                            else float("nan"),
                            "donor_coherence_structural": float(
                                coherence_arrays["structural"]["CN"][layer, head]
                                / (coherence_arrays["structural"]["EN"][layer, head] + EPS)
                            )
                            if coherence_arrays["structural"]["EN"][layer, head]
                            >= beta_cfg.activity_floor_relative
                            * np.nanmedian(coherence_arrays["structural"]["EN"])
                            else float("nan"),
                            "carrier_coherence_structural": float(
                                coherence_arrays["structural"]["EN"][layer, head]
                                / (coherence_arrays["structural"]["EG"][layer, head] + EPS)
                            )
                            if coherence_arrays["structural"]["EG"][layer, head]
                            >= beta_cfg.activity_floor_relative
                            * np.nanmedian(coherence_arrays["structural"]["EG"])
                            else float("nan"),
                        }
                    )
    return rows


def beneficial_outcome_rows(
    runs: Mapping[int, Mapping[str, Any]], beta_cfg: BetaConfig
) -> list[dict[str, Any]]:
    rows = []
    for seed in beta_cfg.seeds:
        for channel in CHANNELS:
            score = runs[int(seed)]["scores"][channel]["score"]
            outcomes = np.asarray(score["beneficial_outcome"], dtype=float)
            valid = np.asarray(score["valid"], dtype=bool)
            values = outcomes[valid]
            graph_signed, graph_absolute = [], []
            graph_p_helpful, graph_p_harmful = [], []
            graph_helpful_size, graph_harmful_size = [], []
            for graph in range(len(outcomes)):
                donor_values = outcomes[graph, valid[graph]]
                if not len(donor_values):
                    continue
                helpful = donor_values > 0
                harmful = donor_values < 0
                graph_signed.append(float(donor_values.mean()))
                graph_absolute.append(float(np.abs(donor_values).mean()))
                graph_p_helpful.append(float(helpful.mean()))
                graph_p_harmful.append(float(harmful.mean()))
                if helpful.any():
                    graph_helpful_size.append(float(donor_values[helpful].mean()))
                if harmful.any():
                    graph_harmful_size.append(float(donor_values[harmful].mean()))
            rows.append(
                {
                    "seed": int(seed),
                    "channel": channel,
                    "events": int(len(values)),
                    "graphs": int(len(graph_signed)),
                    "probability_clean_helpful": float(np.mean(graph_p_helpful)),
                    "mean_helpful_given_helpful": float(np.mean(graph_helpful_size))
                    if graph_helpful_size else float("nan"),
                    "probability_clean_harmful": float(np.mean(graph_p_harmful)),
                    "mean_harmful_given_harmful": float(np.mean(graph_harmful_size))
                    if graph_harmful_size else float("nan"),
                    "mean_signed_loss_benefit": float(np.mean(graph_signed)),
                    "mean_absolute_loss_effect": float(np.mean(graph_absolute)),
                    "sign_coherence": float(
                        abs(np.mean(graph_signed)) / (np.mean(graph_absolute) + EPS)
                    ),
                    "estimator": "donors within graph, then graphs",
                }
            )
    return rows


def coordinate_uncertainty_rows(
    runs: Mapping[int, Mapping[str, Any]], beta_cfg: BetaConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    samples = min(int(beta_cfg.bootstrap_samples), 500)
    for seed in beta_cfg.seeds:
        for aggregation in AGGREGATIONS:
            sem_pg = np.asarray(
                runs[int(seed)]["scores"]["semantic"]["score"]["per_graph"][aggregation],
                dtype=float,
            )
            str_pg = np.asarray(
                runs[int(seed)]["scores"]["structural"]["score"]["per_graph"][aggregation],
                dtype=float,
            )
            rng = np.random.default_rng(
                7_100_000 + int(seed) * 101 + AGGREGATIONS.index(aggregation)
            )
            draws = {name: [] for name in ("semantic", "structural", "D", "J", "G")}
            for _ in range(samples):
                sem = sem_pg[rng.integers(0, len(sem_pg), len(sem_pg))].mean(axis=0)
                st = str_pg[rng.integers(0, len(str_pg), len(str_pg))].mean(axis=0)
                coordinates = score_coordinates(sem, st)
                draws["semantic"].append(sem)
                draws["structural"].append(st)
                for name in ("D", "J", "G"):
                    draws[name].append(coordinates[name])
            arrays = {name: np.stack(value, axis=0) for name, value in draws.items()}
            point_sem, point_str = sem_pg.mean(axis=0), str_pg.mean(axis=0)
            point_coordinates = score_coordinates(point_sem, point_str)
            for layer in range(point_sem.shape[0]):
                for head in range(point_sem.shape[1]):
                    row = {
                        "seed": int(seed), "aggregation": aggregation,
                        "layer": layer, "head": head, "bootstrap_samples": samples,
                        "semantic": float(point_sem[layer, head]),
                        "structural": float(point_str[layer, head]),
                        "D": float(point_coordinates["D"][layer, head]),
                        "J": float(point_coordinates["J"][layer, head]),
                        "G": float(point_coordinates["G"][layer, head]),
                    }
                    for name, array in arrays.items():
                        row[f"{name}_ci_low"] = float(
                            np.quantile(array[:, layer, head], 0.025)
                        )
                        row[f"{name}_ci_high"] = float(
                            np.quantile(array[:, layer, head], 0.975)
                        )
                    row["probability_D_positive"] = float(
                        np.mean(arrays["D"][:, layer, head] > 0)
                    )
                    rows.append(row)
    return rows


def task_factor_control_rows(
    runs: Mapping[int, Mapping[str, Any]], beta_cfg: BetaConfig
) -> list[dict[str, Any]]:
    """Full synthetic task-mode × intervention-factor negative-control matrix."""

    rows: list[dict[str, Any]] = []
    for seed in beta_cfg.seeds:
        run = runs[int(seed)]
        for aggregation in AGGREGATIONS:
            for task_mode in CHANNELS:
                primary = run["scores"][task_mode]
                cells = {
                    task_mode: primary["score"],
                    str(primary["cross_factor_control"]["intervention_factor"]):
                        primary["cross_factor_control"]["score"],
                }
                for factor, score in cells.items():
                    per_graph = np.asarray(score["per_graph"][aggregation], dtype=float)
                    mean = per_graph.mean(axis=0)
                    for layer in range(mean.shape[0]):
                        for head in range(mean.shape[1]):
                            rows.append(
                                {
                                    "seed": int(seed),
                                    "aggregation": aggregation,
                                    "task_mode": task_mode,
                                    "intervention_factor": factor,
                                    "matched_cell": bool(task_mode == factor),
                                    "layer": layer,
                                    "head": head,
                                    "score": float(mean[layer, head]),
                                }
                            )
    return rows


def task_factor_family_specificity_rows(
    runs: Mapping[int, Mapping[str, Any]],
    beta_cfg: BetaConfig,
    aggregation: str,
) -> list[dict[str, Any]]:
    """Do selected heads respond to their factor beyond the task-context off diagonal?"""

    rows: list[dict[str, Any]] = []
    samples = min(int(beta_cfg.bootstrap_samples), 500)
    for seed in beta_cfg.seeds:
        run = runs[int(seed)]
        groups = run["families"]["selection"][aggregation]["groups"]
        family_pairs = {
            "raw_ranked": (
                [tuple(value) for value in groups["raw_semantic"]],
                [tuple(value) for value in groups["raw_structural"]],
            ),
            "D_specialist": (
                [tuple(value) for value in groups["semantic_specialist"]],
                [tuple(value) for value in groups["structural_specialist"]],
            ),
        }
        cells = {
            "sem_sem": np.asarray(
                run["scores"]["semantic"]["score"]["per_graph"][aggregation], dtype=float
            ),
            "sem_str": np.asarray(
                run["scores"]["semantic"]["cross_factor_control"]["score"]["per_graph"][aggregation],
                dtype=float,
            ),
            "str_str": np.asarray(
                run["scores"]["structural"]["score"]["per_graph"][aggregation], dtype=float
            ),
            "str_sem": np.asarray(
                run["scores"]["structural"]["cross_factor_control"]["score"]["per_graph"][aggregation],
                dtype=float,
            ),
        }

        def enrichments(cell_mean: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
            semantic_scale = 0.5 * (
                float(np.mean(cell_mean["sem_sem"])) + float(np.mean(cell_mean["str_sem"]))
            ) + EPS
            structural_scale = 0.5 * (
                float(np.mean(cell_mean["str_str"])) + float(np.mean(cell_mean["sem_str"]))
            ) + EPS
            return {
                "semantic": (cell_mean["sem_sem"] - cell_mean["str_sem"]) / semantic_scale,
                "structural": (cell_mean["str_str"] - cell_mean["sem_str"]) / structural_scale,
            }

        point_enrichment = enrichments(
            {name: value.mean(axis=0) for name, value in cells.items()}
        )
        for family_type, (sem_heads, str_heads) in family_pairs.items():
            available = bool(sem_heads and str_heads)
            if not available:
                for role_name in (*CHANNELS, "joint"):
                    rows.append(
                        {
                            "seed": int(seed), "aggregation": aggregation,
                            "family_type": family_type, "role": role_name,
                            "available": False, "matched_factor_enrichment": float("nan"),
                            "ci_low": float("nan"), "ci_high": float("nan"),
                            "bootstrap_samples": 0,
                        }
                    )
                continue
            point = {
                "semantic": float(np.mean([point_enrichment["semantic"][item] for item in sem_heads])),
                "structural": float(np.mean([point_enrichment["structural"][item] for item in str_heads])),
            }
            point["joint"] = float(np.mean([point[channel] for channel in CHANNELS]))
            rng = np.random.default_rng(
                7_150_000 + int(seed) * 101
                + (10_000 if family_type == "D_specialist" else 0)
            )
            draws = {name: [] for name in (*CHANNELS, "joint")}
            for _ in range(samples):
                sem_indices = rng.integers(0, len(cells["sem_sem"]), len(cells["sem_sem"]))
                str_indices = rng.integers(0, len(cells["str_str"]), len(cells["str_str"]))
                sampled = enrichments(
                    {
                        "sem_sem": cells["sem_sem"][sem_indices].mean(axis=0),
                        "sem_str": cells["sem_str"][sem_indices].mean(axis=0),
                        "str_str": cells["str_str"][str_indices].mean(axis=0),
                        "str_sem": cells["str_sem"][str_indices].mean(axis=0),
                    }
                )
                values = {
                    "semantic": float(np.mean([sampled["semantic"][item] for item in sem_heads])),
                    "structural": float(np.mean([sampled["structural"][item] for item in str_heads])),
                }
                values["joint"] = float(np.mean([values[channel] for channel in CHANNELS]))
                for role_name, value in values.items():
                    draws[role_name].append(value)
            for role_name in (*CHANNELS, "joint"):
                values = np.asarray(draws[role_name], dtype=float)
                rows.append(
                    {
                        "seed": int(seed), "aggregation": aggregation,
                        "family_type": family_type, "role": role_name,
                        "available": True,
                        "matched_factor_enrichment": point[role_name],
                        "ci_low": float(np.quantile(values, 0.025)),
                        "ci_high": float(np.quantile(values, 0.975)),
                        "positive_probability": float(np.mean(values > 0)),
                        "bootstrap_samples": samples,
                    }
                )
    return rows


def figure_aggregation_planes(
    runs: Mapping[int, Mapping[str, Any]], beta_cfg: BetaConfig, figures: Path
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 9.0))
    descriptors = {
        "CG": "coherent–gross (legacy)",
        "EG": "eventwise–gross",
        "CN": "coherent–net",
        "EN": "eventwise–net candidate",
    }
    for ax, aggregation in zip(axes.ravel(), AGGREGATIONS):
        all_values = []
        for seed_index, seed in enumerate(beta_cfg.seeds):
            sem = np.asarray(
                runs[int(seed)]["scores"]["semantic"]["score"]["per_graph"][aggregation]
            ).mean(axis=0)
            st = np.asarray(
                runs[int(seed)]["scores"]["structural"]["score"]["per_graph"][aggregation]
            ).mean(axis=0)
            sem = sem / (sem.mean() + EPS)
            st = st / (st.mean() + EPS)
            all_values.extend(sem.reshape(-1).tolist() + st.reshape(-1).tolist())
            for layer in range(sem.shape[0]):
                ax.scatter(
                    st[layer],
                    sem[layer],
                    s=35,
                    marker=SEED_MARKERS[seed_index],
                    color=LAYER_COLOURS[layer],
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=0.82,
                )
        low = max(min(all_values) * 0.75, 1.0e-3)
        high = max(all_values) * 1.25
        ax.plot([low, high], [low, high], color="#666666", linestyle="--", linewidth=0.9)
        ax.set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high))
        ax.set_title(f"{aggregation}: {descriptors[aggregation]}", loc="left", fontweight="bold")
        ax.set_xlabel(r"Matched structural task/factor score / head mean")
        ax.set_ylabel(r"Matched semantic task/factor score / head mean")
        ax.grid(True, which="both", alpha=0.18, linewidth=0.5)
    fig.suptitle(
        "The aggregation choice changes what counts as a specialist",
        x=0.08,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.945,
        "Identical intervention events and clean gradients; colour = layer, marker = training seed",
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save_figure(fig, figures / "beta_fig1_aggregation_planes")


def figure_task_factor_controls(
    control_rows: Sequence[Mapping[str, Any]], figures: Path
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.7))
    for ax, aggregation in zip(axes, AGGREGATIONS):
        seed_matrices = []
        for seed in sorted({int(row["seed"]) for row in control_rows}):
            matrix = np.zeros((2, 2), dtype=float)
            for task_index, task in enumerate(CHANNELS):
                for factor_index, factor in enumerate(CHANNELS):
                    values = [
                        float(row["score"])
                        for row in control_rows
                        if row["aggregation"] == aggregation
                        and int(row["seed"]) == seed
                        and row["task_mode"] == task
                        and row["intervention_factor"] == factor
                    ]
                    matrix[task_index, factor_index] = float(np.mean(values))
            matrix = matrix / (matrix.mean(axis=0, keepdims=True) + EPS)
            seed_matrices.append(matrix)
        mean = np.mean(seed_matrices, axis=0)
        image = ax.imshow(mean, cmap="RdBu_r", vmin=0.5, vmax=1.5, aspect="equal")
        for row in range(2):
            for column in range(2):
                seed_values = [matrix[row, column] for matrix in seed_matrices]
                ax.text(
                    column, row,
                    f"{mean[row, column]:.2f}\n[{min(seed_values):.2f}, {max(seed_values):.2f}]",
                    ha="center", va="center", fontsize=8,
                )
        ax.set_xticks((0, 1), ("semantic", "structural"), rotation=20)
        ax.set_yticks((0, 1), ("semantic", "structural"))
        ax.set_xlabel("intervention factor")
        if aggregation == AGGREGATIONS[0]:
            ax.set_ylabel("task mode")
        ax.set_title(aggregation, loc="left", fontweight="bold")
    colour_axis = fig.add_axes((0.92, 0.22, 0.012, 0.57))
    fig.colorbar(image, cax=colour_axis, label="within-factor mean units")
    fig.suptitle(
        "Off-diagonal controls separate task context from intervention specificity",
        x=0.05, ha="left", fontsize=14, fontweight="bold",
    )
    fig.text(
        0.05, 0.88,
        "Cell mean across heads; each intervention-factor column is normalised across task modes; brackets show seed range",
        color="#555555",
    )
    fig.subplots_adjust(left=0.06, right=0.89, bottom=0.20, top=0.80, wspace=0.36)
    return save_figure(fig, figures / "beta_fig1b_task_factor_controls")


def figure_task_factor_family_specificity(
    rows: Sequence[Mapping[str, Any]], figures: Path
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    roles = (*CHANNELS, "joint")
    for ax, family_type, title in zip(
        axes,
        ("raw_ranked", "D_specialist"),
        ("A. Raw-ranked families", "B. D-selected specialists"),
    ):
        available_points = 0
        for seed_index, seed in enumerate(sorted({int(row["seed"]) for row in rows})):
            selected = {
                str(row["role"]): row for row in rows
                if row["family_type"] == family_type and int(row["seed"]) == seed
            }
            for role_index, role in enumerate(roles):
                row = selected.get(role)
                if row is None or not bool(row["available"]):
                    continue
                available_points += 1
                point = float(row["matched_factor_enrichment"])
                x_value = role_index + (seed_index - 1) * 0.08
                colour = "#D55E00" if role == "semantic" else (
                    "#0072B2" if role == "structural" else "#555555"
                )
                ax.vlines(
                    x_value, float(row["ci_low"]), float(row["ci_high"]),
                    color=colour, linewidth=1,
                )
                ax.scatter(
                    x_value, point,
                    marker=SEED_MARKERS[seed_index % len(SEED_MARKERS)],
                    color=colour, s=28, zorder=3,
                )
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.set_xticks(range(3), ("semantic role", "structural role", "joint"))
        if ax is axes[0]:
            ax.set_ylabel("matched-factor enrichment over off-diagonal task")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.16, linewidth=0.5)
        if available_points == 0:
            ax.text(
                0.5, 0.5, "no family cleared\nactivity/uncertainty criteria",
                transform=ax.transAxes, ha="center", va="center", color="#666666",
            )
    fig.suptitle(
        "Selected-family specificity must survive the task-context controls",
        x=0.06, ha="left", fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return save_figure(fig, figures / "beta_fig1c_selected_family_specificity")


def figure_method_comparison(
    method_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    figures: Path,
) -> list[str]:
    plt = configure_matplotlib()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.9), gridspec_kw={"width_ratios": [1.7, 1]})
    metrics = (
        "causal_same_channel_mean_rho",
        "D_causal_role_rho",
        "ablation_mean_rho",
        "split_half_mean_rho",
        "K_to_all_mean_rho",
        "topk_mean_stability",
    )
    labels = (
        "score → patch",
        "D → causal role",
        "score → ablation",
        "split-half rank",
        "K→all rank",
        "top-k stability",
    )
    matrix = np.zeros((len(AGGREGATIONS), len(metrics)))
    selection_seeds = {int(value) for value in decision["selection_seeds"]}
    for row_index, aggregation in enumerate(AGGREGATIONS):
        selected = [
            row
            for row in method_rows
            if row["aggregation"] == aggregation and int(row["seed"]) in selection_seeds
        ]
        for column, metric in enumerate(metrics):
            matrix[row_index, column] = np.nanmean([float(row[metric]) for row in selected])
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(labels)), labels, rotation=28, ha="right")
    ax.set_yticks(range(len(AGGREGATIONS)), AGGREGATIONS)
    ax.set_title("A. Method selection on seeds 0–1", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, label="correlation / stability")

    candidate = str(decision["selected_on_seeds_0_1"])
    point_checks = decision["confirmation_checks"]
    checks = {
        "causal gain ≥ .05": point_checks["seed2_mean_causal_improvement_at_least_0.05"],
        "restore improves": point_checks["seed2_restore_improves"],
        "inject improves": point_checks["seed2_inject_improves"],
        "channels non-inferior": point_checks["seed2_no_channel_worse_by_more_than_0.05"],
        "D role non-inferior": point_checks["seed2_D_role_noninferior"],
        "ablation non-inferior": point_checks["seed2_ablation_noninferior"],
        "K→all / top-k stable": (
            point_checks["seed2_K_to_all_rank_at_least_0.90"]
            and point_checks["seed2_top_family_stability_at_least_0.67"]
        ),
        "paired bootstrap": bool(decision["paired_graph_bootstrap"]["gate_passed"]),
    }
    names = list(checks)
    values = [1 if checks[name] else 0 for name in names]
    colours = ["#009E73" if value else "#D55E00" for value in values]
    ax2.barh(range(len(names)), values, color=colours, height=0.58)
    ax2.scatter(values, range(len(names)), color=colours, s=30, zorder=3, edgecolor="white", linewidth=0.4)
    ax2.set_yticks(
        range(len(names)),
        names,
    )
    ax2.set_xlim(0, 1.08)
    ax2.set_xticks((0, 1), ("fail", "pass"))
    ax2.set_title(f"B. Locked seed-2 check: {candidate}", loc="left", fontweight="bold")
    ax2.invert_yaxis()
    fig.suptitle(
        "Causal validity and reliability decide the successor",
        x=0.06,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(0.06, 0.90, str(decision["verdict"]), color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    return save_figure(fig, figures / "beta_fig2_method_comparison")


def figure_winner_causal_scatter(
    head_rows: Sequence[Mapping[str, Any]], aggregation: str, figures: Path
) -> list[str]:
    plt = configure_matplotlib()
    rows = _rows_for_aggregation(head_rows, aggregation)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, channel in zip(axes, CHANNELS):
        correlations = []
        for seed_index, seed in enumerate(sorted({int(row["seed"]) for row in rows})):
            seed_rows = [row for row in rows if int(row["seed"]) == seed]
            score = np.asarray([float(row[f"{channel}_score"]) for row in seed_rows])
            causal = np.asarray([float(row[f"{channel}_causal"]) for row in seed_rows])
            score = score / (np.mean(score) + EPS)
            causal = causal / (np.nanmean(np.abs(causal)) + EPS)
            correlations.append(spearman(score, causal))
            for layer in sorted({int(row["layer"]) for row in seed_rows}):
                mask = np.asarray([int(row["layer"]) == layer for row in seed_rows])
                ax.scatter(
                    score[mask],
                    causal[mask],
                    color=LAYER_COLOURS[layer],
                    marker=SEED_MARKERS[seed_index],
                    s=36,
                    alpha=0.80,
                    edgecolor="white",
                    linewidth=0.4,
                )
        ax.axhline(0, color="#999999", linewidth=0.7)
        ax.set_xlabel(f"{channel} score (within-seed mean units)")
        ax.set_ylabel("bidirectional desired-direction effect")
        ax.set_title(
            f"{channel.capitalize()}: median seed ρ={np.nanmedian(correlations):.2f}",
            loc="left",
            fontweight="bold",
        )
        ax.grid(True, alpha=0.16, linewidth=0.5)
    fig.suptitle(
        f"Causal validity across three independently trained seeds ({aggregation})",
        x=0.06,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return save_figure(fig, figures / "beta_fig2b_selected_causal_scatter")


def _rows_for_aggregation(
    head_rows: Sequence[Mapping[str, Any]], aggregation: str
) -> list[Mapping[str, Any]]:
    return [row for row in head_rows if row["aggregation"] == aggregation]


def figure_selectivity_strength(
    head_rows: Sequence[Mapping[str, Any]], aggregation: str, figures: Path
) -> list[str]:
    plt = configure_matplotlib()
    rows = _rows_for_aggregation(head_rows, aggregation)
    fig, axes_grid = plt.subplots(2, 2, figsize=(10.8, 8.2))
    axes = axes_grid.ravel()
    for seed_index, seed in enumerate(sorted({int(row["seed"]) for row in rows})):
        for layer in sorted({int(row["layer"]) for row in rows}):
            selected = [
                row
                for row in rows
                if int(row["seed"]) == seed and int(row["layer"]) == layer
            ]
            axes[0].scatter(
                [float(row["D"]) for row in selected],
                [float(row["J_evoked"]) for row in selected],
                color=LAYER_COLOURS[layer],
                marker=SEED_MARKERS[seed_index],
                s=34,
                alpha=0.80,
                edgecolor="white",
                linewidth=0.4,
            )
    axes[0].axvline(0, color="#999999", linewidth=0.7)
    axes[0].set_xlabel(r"relative selectivity $D_{rel}$")
    axes[0].set_ylabel("evoked strength J")
    axes[0].set_title("A. Head allocation plane", loc="left", fontweight="bold")
    axes[0].grid(True, alpha=0.16, linewidth=0.5)
    panels = (
        ("D", "causal_D", r"score selectivity $D_{rel}$", "causal channel role"),
        ("J_evoked", "causal_J", "evoked strength J", "overall causal influence"),
        ("G_balanced", "causal_G", "balanced joint-channel strength G", "balanced causal response"),
    )
    for panel_index, (ax, (x_key, y_key, x_label, y_label)) in enumerate(
        zip(axes[1:], panels), start=2
    ):
        for seed_index, seed in enumerate(sorted({int(row["seed"]) for row in rows})):
            for layer in sorted({int(row["layer"]) for row in rows}):
                selected = [row for row in rows if int(row["seed"]) == seed and int(row["layer"]) == layer]
                ax.scatter(
                    [float(row[x_key]) for row in selected],
                    [float(row[y_key]) for row in selected],
                    color=LAYER_COLOURS[layer],
                    marker=SEED_MARKERS[seed_index],
                    s=34,
                    alpha=0.80,
                    edgecolor="white",
                    linewidth=0.4,
                )
        per_seed = []
        for seed in sorted({int(row["seed"]) for row in rows}):
            selected = [row for row in rows if int(row["seed"]) == seed]
            per_seed.append(
                spearman(
                    [float(row[x_key]) for row in selected],
                    [float(row[y_key]) for row in selected],
                )
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(
            f"{chr(64 + panel_index)}. {x_key.split('_')[0]}: median seed ρ={np.nanmedian(per_seed):.2f}",
            loc="left",
            fontweight="bold",
        )
        ax.axhline(0, color="#999999", linewidth=0.7)
        if x_key == "D":
            ax.axvline(0, color="#999999", linewidth=0.7)
        ax.grid(True, alpha=0.16, linewidth=0.5)
    fig.suptitle(
        f"Do D, J and G predict their intended causal targets? ({aggregation})",
        x=0.06,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save_figure(fig, figures / "beta_fig3_selectivity_strength")


def _family_patch_matrix(
    runs: Mapping[int, Mapping[str, Any]],
    beta_cfg: BetaConfig,
    aggregation: str,
    direction: str,
) -> np.ndarray:
    matrices = []
    for seed in beta_cfg.seeds:
        run = runs[int(seed)]
        groups = run["families"]["selection"][aggregation]["groups"]
        rows = []
        for specialist, fallback in (
            ("semantic_specialist", "raw_semantic"),
            ("structural_specialist", "raw_structural"),
        ):
            family = specialist if groups[specialist] else fallback
            heads = [tuple(value) for value in groups[family]]
            values = []
            for channel in CHANNELS:
                floor = _noop_output_floor(
                    run["scores"][channel], run["patches"][channel],
                    relative=beta_cfg.causal_effect_floor_relative,
                )
                matrix = _mean_head_metric(
                    run["patches"][channel],
                    direction=direction,
                    metric="desired_projection",
                    floor=floor,
                )
                scale = float(np.nanmean(np.abs(matrix))) + EPS
                values.append(float(np.nanmean([matrix[item] for item in heads])) / scale)
            rows.append(values)
        matrices.append(rows)
    return np.nanmean(np.asarray(matrices, dtype=float), axis=0)


def figure_bidirectional_patching(
    runs: Mapping[int, Mapping[str, Any]],
    beta_cfg: BetaConfig,
    aggregation: str,
    figures: Path,
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.9), gridspec_kw={"width_ratios": [1.35, 1, 1, 1]})
    predicted_all, finite_all, cosine_all = [], [], []
    for seed in beta_cfg.seeds:
        for channel in CHANNELS:
            patch = runs[int(seed)]["patches"][channel]
            predicted = np.asarray(patch["predicted_logit_delta"], dtype=float)
            finite = np.asarray(patch["finite_logit_delta"]["inject"], dtype=float)
            floor = _noop_output_floor(
                runs[int(seed)]["scores"][channel], patch,
                relative=beta_cfg.causal_effect_floor_relative,
            )
            valid = np.asarray(patch["effect_norm"], dtype=float) > floor
            predicted = predicted[valid].reshape(-1, predicted.shape[-1])
            finite = finite[valid].reshape(-1, finite.shape[-1])
            predicted_all.append(np.linalg.norm(predicted, axis=-1))
            finite_all.append(np.linalg.norm(finite, axis=-1))
            cosine_all.append(
                np.sum(predicted * finite, axis=-1)
                / (np.linalg.norm(predicted, axis=-1) * np.linalg.norm(finite, axis=-1) + EPS)
            )
    predicted = np.concatenate(predicted_all)
    finite = np.concatenate(finite_all)
    cosine = np.concatenate(cosine_all)
    keep = np.isfinite(predicted) & np.isfinite(finite) & np.isfinite(cosine)
    if bool(keep.any()):
        axes[0].scatter(
            predicted[keep],
            finite[keep],
            c=cosine[keep],
            cmap="viridis",
            vmin=-1,
            vmax=1,
            s=8,
            alpha=0.30,
            rasterized=True,
        )
        high = max(
            float(np.quantile(np.concatenate([predicted[keep], finite[keep]]), 0.99)),
            1.0e-6,
        )
        axes[0].plot([0, high], [0, high], "--", color="#666666", linewidth=0.9)
        axes[0].set(xlim=(0, high), ylim=(0, high))
        calibration_label = f"median cosine={np.median(cosine[keep]):.2f}"
    else:
        axes[0].text(
            0.5,
            0.5,
            "No sites cleared the predeclared effect floor",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
            color="#555555",
        )
        axes[0].set(xlim=(0, 1), ylim=(0, 1))
        calibration_label = "no eligible sites"
    axes[0].set_xlabel("clean-gradient predicted norm")
    axes[0].set_ylabel("finite injected effect norm")
    axes[0].set_title(
        f"A. Local→finite calibration\n{calibration_label}",
        loc="left",
        fontweight="bold",
    )
    for ax, direction, label in zip(
        axes[1:],
        ("restore", "inject", "necessity"),
        ("B. Restore", "C. Inject", "D. Necessity"),
    ):
        matrix = _family_patch_matrix(runs, beta_cfg, aggregation, direction)
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        for row in range(2):
            for column in range(2):
                ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
        ax.set_xticks((0, 1), ("semantic\nevents", "structural\nevents"))
        ax.set_yticks((0, 1), ("semantic family", "structural family"))
        ax.set_title(label, loc="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.91, 0.88))
    colour_axis = fig.add_axes((0.93, 0.20, 0.012, 0.58))
    fig.colorbar(image, cax=colour_axis, label="within-channel mean units")
    fig.suptitle(
        f"Bidirectional whole-wV mediation and donor-wise necessity ({aggregation})",
        x=0.04,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    sem_special_seeds = sum(
        bool(runs[int(seed)]["families"]["selection"][aggregation]["groups"]["semantic_specialist"])
        for seed in beta_cfg.seeds
    )
    str_special_seeds = sum(
        bool(runs[int(seed)]["families"]["selection"][aggregation]["groups"]["structural_specialist"])
        for seed in beta_cfg.seeds
    )
    fig.text(
        0.04, 0.88,
        f"D-specialist family used in {sem_special_seeds}/{len(beta_cfg.seeds)} semantic and {str_special_seeds}/{len(beta_cfg.seeds)} structural seeds; raw-ranked fallback otherwise",
        color="#555555",
    )
    return save_figure(fig, figures / "beta_fig4_bidirectional_patching")


def patch_metric_rows(
    runs: Mapping[int, Mapping[str, Any]], beta_cfg: BetaConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in beta_cfg.seeds:
        for channel in CHANNELS:
            patch = runs[int(seed)]["patches"][channel]
            floor = _noop_output_floor(
                runs[int(seed)]["scores"][channel], patch,
                relative=beta_cfg.causal_effect_floor_relative,
            )
            valid = np.asarray(patch["effect_norm"], dtype=float) > floor
            for direction, metrics in patch["metrics"].items():
                arrays = {name: np.asarray(value, dtype=float) for name, value in metrics.items()}
                loss_name = {
                    "restore": "restore_reduction",
                    "inject": "inject_increase",
                    "necessity": "necessity_reduction",
                }.get(direction)
                loss_array = (
                    np.asarray(patch["loss_metrics"][loss_name], dtype=float)
                    if loss_name is not None else None
                )
                for layer in range(next(iter(arrays.values())).shape[0]):
                    for head in range(next(iter(arrays.values())).shape[1]):
                        rows.append(
                            {
                                "seed": int(seed),
                                "channel": channel,
                                "direction": direction,
                                "layer": layer,
                                "head": head,
                                "valid_events": int(valid.sum()),
                                "excluded_event_fraction": float(1.0 - valid.mean()),
                                "effect_floor": float(floor),
                                "sham_max": float(patch["sham_max"]),
                                "nonlinear_loss_effect_mean": float(
                                    np.nanmean(loss_array[layer, head, valid])
                                )
                                if loss_array is not None and valid.any()
                                else float("nan"),
                                **{
                                    f"{name}_mean": float(np.nanmean(array[layer, head, valid]))
                                    if valid.any() else float("nan")
                                    for name, array in arrays.items()
                                },
                            }
                        )
    return rows


def finite_gradient_calibration_rows(
    runs: Mapping[int, Mapping[str, Any]], beta_cfg: BetaConfig
) -> list[dict[str, Any]]:
    rows = []
    for seed in beta_cfg.seeds:
        for channel in CHANNELS:
            patch = runs[int(seed)]["patches"][channel]
            floor = _noop_output_floor(
                runs[int(seed)]["scores"][channel], patch,
                relative=beta_cfg.causal_effect_floor_relative,
            )
            valid = np.asarray(patch["effect_norm"], dtype=float) > floor
            predicted = np.asarray(patch["predicted_logit_delta"], dtype=float)[valid]
            finite = np.asarray(patch["finite_logit_delta"]["inject"], dtype=float)[valid]
            predicted = predicted.reshape(-1, predicted.shape[-1])
            finite = finite.reshape(-1, finite.shape[-1])
            dot = np.sum(predicted * finite, axis=-1)
            pred_norm = np.linalg.norm(predicted, axis=-1)
            finite_norm = np.linalg.norm(finite, axis=-1)
            cosine = dot / (pred_norm * finite_norm + EPS)
            relative_error = np.linalg.norm(predicted - finite, axis=-1) / (finite_norm + EPS)
            slope = float(np.sum(predicted * finite) / (np.sum(np.square(predicted)) + EPS))
            rows.append(
                {
                    "seed": int(seed),
                    "channel": channel,
                    "sites": int(len(cosine)),
                    "median_cosine": float(np.nanmedian(cosine)),
                    "calibration_slope": slope,
                    "median_relative_error": float(np.nanmedian(relative_error)),
                    "anti_aligned_fraction": float(np.mean(cosine < 0)),
                    "large_error_fraction": float(np.mean(relative_error > 1)),
                    "effect_floor": float(floor),
                }
            )
    return rows


def family_patch_interaction_rows(
    runs: Mapping[int, Mapping[str, Any]],
    beta_cfg: BetaConfig,
    aggregation: str,
) -> list[dict[str, Any]]:
    """Held-out head-family × intervention-channel causal interactions."""

    rows: list[dict[str, Any]] = []
    samples = min(int(beta_cfg.bootstrap_samples), 500)
    for seed in beta_cfg.seeds:
        run = runs[int(seed)]
        groups = run["families"]["selection"][aggregation]["groups"]
        floors = {
            channel: _noop_output_floor(
                run["scores"][channel], run["patches"][channel],
                relative=beta_cfg.causal_effect_floor_relative,
            )
            for channel in CHANNELS
        }
        family_pairs = {
            "raw_ranked": (groups["raw_semantic"], groups["raw_structural"]),
            "D_specialist": (
                groups["semantic_specialist"], groups["structural_specialist"]
            ),
        }
        for family_type, (sem_family, str_family) in family_pairs.items():
            sem_heads = [tuple(value) for value in sem_family]
            str_heads = [tuple(value) for value in str_family]
            for direction in ("restore", "inject", "necessity"):
                if not sem_heads or not str_heads:
                    rows.append(
                        {
                            "seed": int(seed), "aggregation": aggregation,
                            "family_type": family_type, "direction": direction,
                            "available": False, "interaction": float("nan"),
                            "ci_low": float("nan"), "ci_high": float("nan"),
                            "bootstrap_samples": 0,
                        }
                    )
                    continue
                point_matrices = {}
                for channel in CHANNELS:
                    matrix = _mean_head_metric(
                        run["patches"][channel], direction=direction,
                        metric="desired_projection", floor=floors[channel],
                    )
                    point_matrices[channel] = matrix / (np.nanmean(np.abs(matrix)) + EPS)

                def contrast(matrices: Mapping[str, np.ndarray]) -> float:
                    return float(
                        np.nanmean([matrices["semantic"][item] for item in sem_heads])
                        - np.nanmean([matrices["structural"][item] for item in sem_heads])
                        - np.nanmean([matrices["semantic"][item] for item in str_heads])
                        + np.nanmean([matrices["structural"][item] for item in str_heads])
                    )

                point = contrast(point_matrices)
                rng = np.random.default_rng(
                    7_200_000 + int(seed) * 101 + directions_index(direction)
                    + (10_000 if family_type == "D_specialist" else 0)
                )
                draws = []
                for _ in range(samples):
                    matrices = {}
                    for channel in CHANNELS:
                        indices = _cluster_event_bootstrap_indices(
                            run["patches"][channel], rng
                        )
                        matrix = _mean_head_metric_indices(
                            run["patches"][channel], direction=direction,
                            metric="desired_projection", floor=floors[channel],
                            indices=indices,
                        )
                        matrices[channel] = matrix / (np.nanmean(np.abs(matrix)) + EPS)
                    draws.append(contrast(matrices))
                rows.append(
                    {
                        "seed": int(seed), "aggregation": aggregation,
                        "family_type": family_type, "direction": direction,
                        "available": True, "interaction": point,
                        "ci_low": float(np.nanquantile(draws, 0.025)),
                        "ci_high": float(np.nanquantile(draws, 0.975)),
                        "bootstrap_samples": samples,
                        "positive_probability": float(np.mean(np.asarray(draws) > 0)),
                    }
                )
    return rows


def directions_index(direction: str) -> int:
    return ("restore", "inject", "necessity").index(direction)


def matched_control_patch_rows(
    runs: Mapping[int, Mapping[str, Any]],
    beta_cfg: BetaConfig,
    aggregation: str,
) -> list[dict[str, Any]]:
    """Specialist causal role minus frozen J-matched and same-layer inactive controls."""

    rows: list[dict[str, Any]] = []
    samples = min(int(beta_cfg.bootstrap_samples), 500)
    for seed in beta_cfg.seeds:
        run = runs[int(seed)]
        groups = run["families"]["selection"][aggregation]["groups"]
        specialists = {
            "semantic": [tuple(value) for value in groups["semantic_specialist"]],
            "structural": [tuple(value) for value in groups["structural_specialist"]],
        }
        controls = {
            "J_matched_generalist": {
                "semantic": [tuple(value) for value in groups["semantic_J_matched_generalist"]],
                "structural": [tuple(value) for value in groups["structural_J_matched_generalist"]],
            },
            "same_layer_inert": {
                "semantic": [tuple(value) for value in groups["semantic_layer_inert"]],
                "structural": [tuple(value) for value in groups["structural_layer_inert"]],
            },
        }
        floors = {
            channel: _noop_output_floor(
                run["scores"][channel], run["patches"][channel],
                relative=beta_cfg.causal_effect_floor_relative,
            )
            for channel in CHANNELS
        }

        def normalized_matrices(
            direction: str, indices_by_channel: Mapping[str, np.ndarray] | None = None
        ) -> dict[str, np.ndarray]:
            matrices = {}
            for channel in CHANNELS:
                if indices_by_channel is None:
                    matrix = _mean_head_metric(
                        run["patches"][channel], direction=direction,
                        metric="desired_projection", floor=floors[channel],
                    )
                else:
                    matrix = _mean_head_metric_indices(
                        run["patches"][channel], direction=direction,
                        metric="desired_projection", floor=floors[channel],
                        indices=indices_by_channel[channel],
                    )
                matrices[channel] = matrix / (np.nanmean(np.abs(matrix)) + EPS)
            return matrices

        def role(
            matrices: Mapping[str, np.ndarray],
            heads: Sequence[tuple[int, int]],
            expected: str,
        ) -> float:
            other = "structural" if expected == "semantic" else "semantic"
            return float(
                np.nanmean([matrices[expected][item] for item in heads])
                - np.nanmean([matrices[other][item] for item in heads])
            )

        for control_name, control_groups in controls.items():
            for direction in ("restore", "inject", "necessity"):
                available = all(
                    len(specialists[channel]) > 0
                    and len(control_groups[channel]) == len(specialists[channel])
                    for channel in CHANNELS
                )
                if not available:
                    for channel_role in (*CHANNELS, "joint"):
                        rows.append(
                            {
                                "seed": int(seed), "aggregation": aggregation,
                                "control": control_name, "direction": direction,
                                "role": channel_role, "available": False,
                                "specialist_minus_control": float("nan"),
                                "ci_low": float("nan"), "ci_high": float("nan"),
                                "bootstrap_samples": 0,
                            }
                        )
                    continue
                point_matrix = normalized_matrices(direction)
                point = {
                    channel: role(point_matrix, specialists[channel], channel)
                    - role(point_matrix, control_groups[channel], channel)
                    for channel in CHANNELS
                }
                point["joint"] = float(np.mean([point[channel] for channel in CHANNELS]))
                rng = np.random.default_rng(
                    7_300_000 + int(seed) * 101 + directions_index(direction)
                    + (10_000 if control_name == "same_layer_inert" else 0)
                )
                draws = {channel: [] for channel in (*CHANNELS, "joint")}
                for _ in range(samples):
                    indices = {
                        channel: _cluster_event_bootstrap_indices(
                            run["patches"][channel], rng
                        )
                        for channel in CHANNELS
                    }
                    matrices = normalized_matrices(direction, indices)
                    values = {
                        channel: role(matrices, specialists[channel], channel)
                        - role(matrices, control_groups[channel], channel)
                        for channel in CHANNELS
                    }
                    values["joint"] = float(
                        np.mean([values[channel] for channel in CHANNELS])
                    )
                    for channel_role, value in values.items():
                        draws[channel_role].append(value)
                for channel_role in (*CHANNELS, "joint"):
                    values = np.asarray(draws[channel_role], dtype=float)
                    rows.append(
                        {
                            "seed": int(seed), "aggregation": aggregation,
                            "control": control_name, "direction": direction,
                            "role": channel_role, "available": True,
                            "specialist_minus_control": point[channel_role],
                            "ci_low": float(np.nanquantile(values, 0.025)),
                            "ci_high": float(np.nanquantile(values, 0.975)),
                            "positive_probability": float(np.mean(values > 0)),
                            "bootstrap_samples": samples,
                        }
                    )
    return rows


def figure_patch_controls(
    patch_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    figures: Path,
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    directions = ("restore", "inject", "necessity", "cross_graph_mismatch")
    labels = ("restore", "inject", "necessity", "cross-graph\nmismatch")
    colours = ("#009E73", "#56B4E9", "#E69F00", "#777777")
    for index, (direction, colour) in enumerate(zip(directions, colours)):
        selected = [row for row in patch_rows if row["direction"] == direction]
        desired = np.asarray([float(row["desired_projection_mean"]) for row in selected])
        orthogonal = np.asarray([float(row["orthogonal_ratio_mean"]) for row in selected])
        jitter = np.linspace(-0.10, 0.10, len(desired)) if len(desired) else np.asarray([])
        axes[0].scatter(index + jitter, desired, s=9, alpha=0.28, color=colour)
        axes[0].plot(index, np.nanmedian(desired), marker="D", color="#222222", markersize=5)
        axes[1].scatter(index + jitter, orthogonal, s=9, alpha=0.28, color=colour)
        axes[1].plot(index, np.nanmedian(orthogonal), marker="D", color="#222222", markersize=5)
    for ax in axes[:2]:
        ax.set_xticks(range(4), labels)
        ax.axhline(0, color="#999999", linewidth=0.7)
        ax.grid(True, axis="y", alpha=0.16, linewidth=0.5)
    axes[0].set_ylabel("desired-direction logit movement")
    axes[0].set_title("A. Matched effects versus control", loc="left", fontweight="bold")
    axes[1].set_ylabel("orthogonal / clean–corrupt effect")
    axes[1].set_title("B. Off-direction movement", loc="left", fontweight="bold")
    for row in calibration_rows:
        channel = str(row["channel"])
        colour = "#D55E00" if channel == "semantic" else "#0072B2"
        marker = SEED_MARKERS[int(row["seed"]) % len(SEED_MARKERS)]
        axes[2].scatter(
            float(row["calibration_slope"]), float(row["median_cosine"]),
            color=colour, marker=marker, s=55, edgecolor="white", linewidth=0.5,
            label=channel if int(row["seed"]) == 0 else None,
        )
    axes[2].axvline(1, color="#999999", linestyle="--", linewidth=0.8)
    axes[2].axhline(1, color="#999999", linestyle="--", linewidth=0.8)
    axes[2].set_xlabel("finite / local calibration slope")
    axes[2].set_ylabel("median vector cosine")
    axes[2].set_title("C. Per-seed/channel calibration", loc="left", fontweight="bold")
    axes[2].legend()
    axes[2].grid(True, alpha=0.16, linewidth=0.5)
    fig.suptitle(
        "Controls expose non-specific rescue, orthogonal movement and linearisation failure",
        x=0.05, ha="left", fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return save_figure(fig, figures / "beta_fig4b_patch_controls")


def figure_family_ablation(
    runs: Mapping[int, Mapping[str, Any]],
    beta_cfg: BetaConfig,
    aggregation: str,
    figures: Path,
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), sharex=True)
    families = (
        "raw_semantic",
        "raw_structural",
        "semantic_specialist",
        "structural_specialist",
        "high_J_generalist",
        "high_G_balanced",
        "low_J_inert",
    )
    for row_index, task in enumerate(CHANNELS):
        for column, (metric, label) in enumerate(
            (("functional", "logit displacement"), ("loss", "cross-entropy increase"))
        ):
            ax = axes[row_index, column]
            for family in families:
                seed_curves = []
                for seed in beta_cfg.seeds:
                    record = runs[int(seed)]["families"]["ablation"]["tasks"][task][
                        "families"
                    ][f"{aggregation}_{family}"]
                    curve = np.asarray(record[metric], dtype=float).mean(axis=-1)
                    if len(curve) <= 1:
                        continue
                    seed_curves.append(curve)
                    ax.plot(
                        np.arange(len(curve)),
                        curve,
                        color=FAMILY_COLOURS[family],
                        alpha=0.20,
                        linewidth=0.8,
                    )
                if seed_curves:
                    common = min(len(value) for value in seed_curves)
                    mean = np.mean([value[:common] for value in seed_curves], axis=0)
                    ax.plot(
                        np.arange(common),
                        mean,
                        color=FAMILY_COLOURS[family],
                        linewidth=2.0,
                        marker="o",
                        markersize=3.2,
                        label=family.replace("_", " "),
                    )
            ax.axhline(0, color="#999999", linewidth=0.7)
            ax.set_title(f"{task.capitalize()}: {label}", loc="left", fontweight="bold")
            ax.set_xlabel("cumulatively ablated heads")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.16, linewidth=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(
        f"Frozen score-family ablations on independent graphs ({aggregation})",
        x=0.06,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.925,
        "D-selected specialist families are allowed to be empty; functional/loss effects are primary and accuracy remains tabulated",
        color="#555555",
    )
    empties = {
        family: sum(
            not runs[int(seed)]["families"]["selection"][aggregation]["groups"][family]
            for seed in beta_cfg.seeds
        )
        for family in ("semantic_specialist", "structural_specialist", "low_J_inert")
    }
    fig.text(
        0.06, 0.895,
        "Empty seeds: " + ", ".join(
            f"{name.replace('_', ' ')} {count}/{len(beta_cfg.seeds)}"
            for name, count in empties.items()
        ),
        color="#555555", fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.86))
    return save_figure(fig, figures / "beta_fig5_family_ablation")


def figure_routing_message(
    runs: Mapping[int, Mapping[str, Any]],
    head_rows: Sequence[Mapping[str, Any]],
    beta_cfg: BetaConfig,
    aggregation: str,
    figures: Path,
) -> list[str]:
    plt = configure_matplotlib()
    rows = _rows_for_aggregation(head_rows, aggregation)
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.1))
    for seed_index, seed in enumerate(beta_cfg.seeds):
        selected = [row for row in rows if int(row["seed"]) == seed]
        axes[0].scatter(
            [float(row["D"]) for row in selected],
            [float(row["semantic_routing_share"]) for row in selected],
            color="#D55E00",
            marker=SEED_MARKERS[seed_index],
            s=28,
            alpha=0.65,
            label="semantic" if seed_index == 0 else None,
        )
        axes[0].scatter(
            [float(row["D"]) for row in selected],
            [float(row["structural_routing_share"]) for row in selected],
            color="#0072B2",
            marker=SEED_MARKERS[seed_index],
            s=28,
            alpha=0.65,
            label="structural" if seed_index == 0 else None,
        )
    axes[0].axvline(0, color="#999999", linewidth=0.7)
    axes[0].set_xlabel("score selectivity D")
    axes[0].set_ylabel("routing / (routing + message)")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("A. Descriptive mechanism", loc="left", fontweight="bold")
    axes[0].legend()

    for seed_index, seed in enumerate(beta_cfg.seeds):
        selected = [row for row in rows if int(row["seed"]) == seed]
        axes[1].scatter(
            [float(row["D"]) for row in selected],
            [float(row["semantic_component_alignment"]) for row in selected],
            color="#D55E00", marker=SEED_MARKERS[seed_index], s=28, alpha=0.65,
            label="semantic" if seed_index == 0 else None,
        )
        axes[1].scatter(
            [float(row["D"]) for row in selected],
            [float(row["structural_component_alignment"]) for row in selected],
            color="#0072B2", marker=SEED_MARKERS[seed_index], s=28, alpha=0.65,
            label="structural" if seed_index == 0 else None,
        )
    axes[1].axvline(0, color="#999999", linewidth=0.7)
    axes[1].axhline(1, color="#999999", linewidth=0.7, linestyle="--")
    axes[1].set_xlabel("score selectivity D")
    axes[1].set_ylabel("total / (routing + message)")
    axes[1].set_title("B. Component cancellation", loc="left", fontweight="bold")
    axes[1].legend()

    component_values: dict[str, dict[str, list[float]]] = {
        channel: {component: [] for component in ("routing", "message", "full", "interaction")}
        for channel in CHANNELS
    }
    for seed in beta_cfg.seeds:
        groups = runs[int(seed)]["families"]["selection"][aggregation]["groups"]
        for channel, specialist, fallback in (
            ("semantic", "semantic_specialist", "raw_semantic"),
            ("structural", "structural_specialist", "raw_structural"),
        ):
            family = specialist if groups[specialist] else fallback
            heads = [tuple(value) for value in groups[family]]
            patch = runs[int(seed)]["mechanisms"][channel]["patch"]["metrics"]
            for component in ("routing", "message", "full"):
                matrix = np.asarray(patch[component]["desired_projection"], dtype=float).mean(axis=-1)
                component_values[channel][component].append(
                    float(np.mean([matrix[item] for item in heads]))
                )
            interaction = np.asarray(
                runs[int(seed)]["mechanisms"][channel]["patch"]["interaction_projection"],
                dtype=float,
            ).mean(axis=-1)
            component_values[channel]["interaction"].append(
                float(np.mean([interaction[item] for item in heads]))
            )
    for ax, channel, title in zip(
        axes[2:], CHANNELS, ("C. Semantic specialist family", "D. Structural specialist family")
    ):
        components = ("routing", "message", "full", "interaction")
        means = [np.mean(component_values[channel][name]) for name in components]
        ax.bar(
            range(4),
            means,
            color=("#E69F00", "#56B4E9", "#009E73", "#CC79A7"),
            width=0.65,
        )
        for index, component in enumerate(components):
            ax.scatter(
                np.full(len(component_values[channel][component]), index),
                component_values[channel][component],
                color="#222222",
                s=18,
                zorder=3,
            )
        ax.set_xticks(range(4), ("routing", "message", "full wV", "interaction"), rotation=20)
        ax.set_ylabel("desired-direction rescue")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.axhline(0, color="#999999", linewidth=0.7)
    reconstruction = max(
        float(runs[int(seed)]["mechanisms"][channel]["score"]["reconstruction_max"])
        for seed in beta_cfg.seeds
        for channel in CHANNELS
    )
    fig.suptitle(
        f"Routing versus message decomposes the transport mechanism (max residual {reconstruction:.1e})",
        x=0.05,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.05, 0.89,
        "D-specialist families are used when they clear activity/uncertainty criteria; otherwise the frozen raw-ranked family is shown",
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    return save_figure(fig, figures / "beta_fig6_routing_message")


def figure_carriage_coherence(
    runs: Mapping[int, Mapping[str, Any]],
    head_rows: Sequence[Mapping[str, Any]],
    beta_cfg: BetaConfig,
    aggregation: str,
    figures: Path,
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    for ax, channel, family in zip(
        axes[:2], CHANNELS, ("raw_semantic", "raw_structural")
    ):
        sens_curves, coh_curves = [], []
        distance = None
        for seed in beta_cfg.seeds:
            score = runs[int(seed)]["scores"][channel]["score"]
            profiles = score["distance_profiles"]
            distance = np.asarray(profiles["distance"], dtype=int)
            heads = [
                tuple(value)
                for value in runs[int(seed)]["families"]["selection"][aggregation]["groups"][family]
            ]
            sens = np.asarray(profiles["F_sens"], dtype=float)
            coh = np.asarray(profiles["F_coh"], dtype=float)
            sens_curve = np.nanmean(
                [np.nanmean(sens[:, layer, head], axis=0) for layer, head in heads], axis=0
            )
            coh_curve = np.nanmean(
                [np.nanmean(coh[:, layer, head], axis=0) for layer, head in heads], axis=0
            )
            sens_curves.append(sens_curve)
            coh_curves.append(coh_curve)
            ax.plot(distance, sens_curve, color="#D55E00", alpha=0.20, linewidth=0.9)
            ax.plot(distance, coh_curve, color="#0072B2", alpha=0.20, linewidth=0.9)
        ax.plot(
            distance,
            np.nanmean(sens_curves, axis=0),
            color="#D55E00",
            marker="o",
            linewidth=2.0,
            label=r"$F_{sens}$ typical event",
        )
        ax.plot(
            distance,
            np.nanmean(coh_curves, axis=0),
            color="#0072B2",
            marker="s",
            linestyle="--",
            linewidth=2.0,
            label=r"$F_{coh}$ coherent event mean",
        )
        ax.set_yscale("log")
        ax.set_xlabel("distance to changed node set")
        ax.set_ylabel("projected carrier response")
        ax.set_title(f"{channel.capitalize()}-ranked heads", loc="left", fontweight="bold")
        ax.grid(True, which="both", alpha=0.16, linewidth=0.5)
        ax.legend()

    rows = _rows_for_aggregation(head_rows, aggregation)
    for channel, colour, marker in (
        ("semantic", "#D55E00", "o"),
        ("structural", "#0072B2", "s"),
    ):
        axes[2].scatter(
            [float(row[f"donor_coherence_{channel}"]) for row in rows],
            [float(row[f"carrier_coherence_{channel}"]) for row in rows],
            color=colour,
            marker=marker,
            alpha=0.62,
            s=28,
            label=channel,
        )
    axes[2].set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
    axes[2].set_xlabel(r"donor coherence  $S_{CN}/S_{EN}$")
    axes[2].set_ylabel(r"carrier coherence  $S_{EN}/S_{EG}$")
    axes[2].set_title("Event/carrier cancellation", loc="left", fontweight="bold")
    axes[2].grid(True, alpha=0.16, linewidth=0.5)
    axes[2].legend()
    fig.suptitle(
        "Sensitivity, coherent carriage and where cancellation occurs",
        x=0.05,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.05,
        0.91,
        "Structural distance is assigned per partner event as min[d(i,u), d(i,v)] before aggregation",
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return save_figure(fig, figures / "beta_fig7_carriage_coherence")


def figure_conditional_reliability_dose(
    runs: Mapping[int, Mapping[str, Any]],
    conditional_rows: Sequence[Mapping[str, Any]],
    beta_cfg: BetaConfig,
    aggregation: str,
    figures: Path,
) -> list[str]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2))
    selected = [row for row in conditional_rows if row["aggregation"] == aggregation]
    colours = ["#D55E00" if bool(row["confirmed_fdr_0.05"]) else "#888888" for row in selected]
    axes[0].scatter(
        [float(row["discovery_interaction"]) for row in selected],
        [float(row["confirmation_interaction"]) for row in selected],
        color=colours,
        s=28,
        alpha=0.75,
    )
    axes[0].axhline(0, color="#999999", linewidth=0.7)
    axes[0].axvline(0, color="#999999", linewidth=0.7)
    axes[0].set_xlabel("discovery channel × condition interaction")
    axes[0].set_ylabel("held-out interaction")
    confirmed = sum(bool(row["confirmed_fdr_0.05"]) for row in selected)
    axes[0].set_title(
        f"A. Sample-split condition rules\n{confirmed}/{len(selected)} confirmed",
        loc="left",
        fontweight="bold",
    )

    for aggregation_name, colour in zip(
        AGGREGATIONS, ("#777777", "#E69F00", "#56B4E9", "#009E73")
    ):
        by_k: dict[str, list[float]] = {}
        for seed in beta_cfg.seeds:
            for channel in CHANNELS:
                score = runs[int(seed)]["scores"][channel]["score"]
                for repeated in score.get("prefix_repeats", {"0": score["prefix"]}).values():
                    all_score = np.asarray(repeated["all"][aggregation_name]).mean(axis=0)
                    for key, values in repeated.items():
                        estimate = np.asarray(values[aggregation_name]).mean(axis=0)
                        by_k.setdefault(key, []).append(
                            spearman(estimate.reshape(-1), all_score.reshape(-1))
                        )
        keys = sorted((key for key in by_k if key != "all"), key=int) + ["all"]
        axes[1].plot(
            range(len(keys)),
            [np.nanmean(by_k[key]) for key in keys],
            marker="o",
            linewidth=1.8,
            color=colour,
            label=aggregation_name,
        )
    axes[1].set_xticks(range(len(keys)), keys)
    axes[1].set_ylim(0, 1.03)
    axes[1].set_xlabel("events per graph (all = enumerated population)")
    axes[1].set_ylabel("rank correlation with all events")
    axes[1].set_title("B. K convergence", loc="left", fontweight="bold")
    axes[1].legend(ncol=2)
    axes[1].grid(True, alpha=0.16, linewidth=0.5)

    dose_values, score_values = [], []
    for seed in beta_cfg.seeds:
        score = runs[int(seed)]["scores"]["structural"]["score"]
        dose = np.asarray(score["dose"], dtype=float)
        valid = np.asarray(score["valid"], dtype=bool)
        mean_dose = (dose * valid).sum(axis=1) / valid.sum(axis=1).clip(min=1)
        magnitude = np.asarray(score["per_graph"][aggregation], dtype=float).mean(axis=(1, 2))
        dose_values.extend(mean_dose.tolist())
        score_values.extend(magnitude.tolist())
    dose_values_np = np.asarray(dose_values)
    score_values_np = np.asarray(score_values)
    axes[2].scatter(dose_values_np, score_values_np, s=12, alpha=0.35, color="#0072B2")
    rho = spearman(dose_values_np, score_values_np)
    axes[2].set_xlabel("structural RRWP intervention dose")
    axes[2].set_ylabel("mean head response")
    axes[2].set_title(
        f"C. Dose audit ρ={rho:.2f}\nsemantic one-hot dose is constant",
        loc="left",
        fontweight="bold",
    )
    axes[2].grid(True, alpha=0.16, linewidth=0.5)
    fig.suptitle(
        "Conditional discovery is a pipeline/null test here—not a sensitivity validation",
        x=0.05,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return save_figure(fig, figures / "beta_fig8_conditional_reliability_dose")


def family_ablation_rows(
    runs: Mapping[int, Mapping[str, Any]], beta_cfg: BetaConfig
) -> list[dict[str, Any]]:
    rows = []
    for seed in beta_cfg.seeds:
        tasks = runs[int(seed)]["families"]["ablation"]["tasks"]
        for task in CHANNELS:
            for name, record in tasks[task]["families"].items():
                aggregation, family = name.split("_", 1)
                for prefix in range(len(record["prefix_size"])):
                    rows.append(
                        {
                            "seed": int(seed),
                            "task": task,
                            "aggregation": aggregation,
                            "family": family,
                            "prefix": prefix,
                            "functional_mean": float(
                                np.asarray(record["functional"], dtype=float)[prefix].mean()
                            ),
                            "loss_mean": float(
                                np.asarray(record["loss"], dtype=float)[prefix].mean()
                            ),
                            "accuracy_drop_mean": float(
                                np.asarray(record["accuracy_drop"], dtype=float)[prefix].mean()
                            ),
                        }
                    )
    return rows


def mechanism_rows(
    runs: Mapping[int, Mapping[str, Any]], beta_cfg: BetaConfig
) -> list[dict[str, Any]]:
    rows = []
    for seed in beta_cfg.seeds:
        for channel in CHANNELS:
            patch = runs[int(seed)]["mechanisms"][channel]["patch"]
            for component in ("routing", "message", "full"):
                desired = np.asarray(
                    patch["metrics"][component]["desired_projection"], dtype=float
                )
                fraction = np.asarray(
                    patch["metrics"][component]["projection_fraction"], dtype=float
                )
                for layer in range(desired.shape[0]):
                    for head in range(desired.shape[1]):
                        rows.append(
                            {
                                "seed": int(seed),
                                "channel": channel,
                                "layer": layer,
                                "head": head,
                                "component": component,
                                "desired_projection_mean": float(
                                    np.nanmean(desired[layer, head])
                                ),
                                "projection_fraction_mean": float(
                                    np.nanmean(fraction[layer, head])
                                ),
                                "factorial_interaction_norm_mean": float(
                                    np.nanmean(
                                        np.asarray(patch["interaction_norm"], dtype=float)[
                                            layer, head
                                        ]
                                    )
                                ),
                                "factorial_interaction_projection_mean": float(
                                    np.nanmean(
                                        np.asarray(
                                            patch["interaction_projection"], dtype=float
                                        )[layer, head]
                                    )
                                ),
                            }
                        )
    return rows


def carriage_rows(
    runs: Mapping[int, Mapping[str, Any]], beta_cfg: BetaConfig
) -> list[dict[str, Any]]:
    rows = []
    samples = min(int(beta_cfg.bootstrap_samples), 500)

    def interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            return float("nan"), float("nan")
        indices = rng.integers(0, len(finite), size=(samples, len(finite)))
        draws = finite[indices].mean(axis=1)
        return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))

    def effective_support(weights: np.ndarray) -> float:
        values = np.asarray(weights, dtype=float)
        return float(values.sum() ** 2 / (np.square(values).sum() + EPS))

    for seed in beta_cfg.seeds:
        for channel in CHANNELS:
            profile = runs[int(seed)]["scores"][channel]["score"]["distance_profiles"]
            distance = np.asarray(profile["distance"], dtype=int)
            sens = np.asarray(profile["F_sens"], dtype=float)
            coh = np.asarray(profile["F_coh"], dtype=float)
            event_support = np.asarray(profile["event_carrier_support"], dtype=float)
            carrier_support = np.asarray(profile["carrier_support"], dtype=float)
            for layer in range(sens.shape[1]):
                for head in range(sens.shape[2]):
                    for index, value in enumerate(distance):
                        rng = np.random.default_rng(
                            7_400_000 + int(seed) * 10_000 + CHANNELS.index(channel) * 1_000
                            + layer * 100 + head * 10 + index
                        )
                        sens_low, sens_high = interval(sens[:, layer, head, index], rng)
                        coh_low, coh_high = interval(coh[:, layer, head, index], rng)
                        rows.append(
                            {
                                "seed": int(seed),
                                "channel": channel,
                                "layer": layer,
                                "head": head,
                                "distance": int(value),
                                "F_sens_mean": float(np.nanmean(sens[:, layer, head, index])),
                                "F_sens_ci_low": sens_low,
                                "F_sens_ci_high": sens_high,
                                "F_coh_mean": float(np.nanmean(coh[:, layer, head, index])),
                                "F_coh_ci_low": coh_low,
                                "F_coh_ci_high": coh_high,
                                "event_carrier_support": int(event_support[:, index].sum()),
                                "carrier_support": int(carrier_support[:, index].sum()),
                                "supported_graphs": int((carrier_support[:, index] > 0).sum()),
                                "effective_graph_event_support": effective_support(
                                    event_support[:, index]
                                ),
                                "effective_graph_carrier_support": effective_support(
                                    carrier_support[:, index]
                                ),
                                "bootstrap_samples": samples,
                            }
                        )
    return rows


def component_decision_records(
    per_seed_metrics: Mapping[int, Mapping[str, Mapping[str, float]]],
    runs: Mapping[int, Mapping[str, Any]],
    beta_cfg: BetaConfig,
    aggregation: str,
    family_patch_rows: Sequence[Mapping[str, Any]],
    matched_control_rows: Sequence[Mapping[str, Any]],
    task_factor_family_rows: Sequence[Mapping[str, Any]],
    numerical_checks: Mapping[str, bool],
) -> dict[str, Any]:
    def coordinate_decision(metric: str, label: str) -> dict[str, Any]:
        values = np.asarray(
            [per_seed_metrics[int(seed)][aggregation][metric] for seed in beta_cfg.seeds],
            dtype=float,
        )
        passed = bool(
            np.isfinite(values).all()
            and np.median(values) >= 0.70
            and np.min(values) >= 0.50
        )
        return {
            "status": "approve_for_real-task_beta" if passed else "retain_as_diagnostic",
            "criterion": "median seed within-layer rho >= 0.70 and every seed >= 0.50",
            "per_seed_within_layer_rho": values.tolist(),
            "target": label,
        }

    specialist = [
        row for row in family_patch_rows
        if row["family_type"] == "D_specialist" and row["direction"] in {"restore", "inject"}
    ]
    available = [row for row in specialist if bool(row["available"])]
    matched = [
        row for row in matched_control_rows
        if row["control"] == "J_matched_generalist"
        and row["role"] == "joint"
        and row["direction"] in {"restore", "inject"}
        and bool(row["available"])
    ]
    patch_passed = bool(
        len(available) == 2 * len(beta_cfg.seeds)
        and all(float(row["interaction"]) > 0 for row in available)
        and sum(float(row["ci_low"]) > 0 for row in available) >= len(available) - 1
        and len(matched) == 2 * len(beta_cfg.seeds)
        and all(float(row["specialist_minus_control"]) > 0 for row in matched)
        and sum(float(row["ci_low"]) > 0 for row in matched) >= len(matched) - 1
    )
    selected_specificity = [
        row for row in task_factor_family_rows
        if row["family_type"] == "D_specialist"
        and row["role"] == "joint"
        and bool(row["available"])
    ]
    selected_specificity_passed = bool(
        len(selected_specificity) == len(beta_cfg.seeds)
        and all(float(row["matched_factor_enrichment"]) > 0 for row in selected_specificity)
        and sum(float(row["ci_low"]) > 0 for row in selected_specificity)
        >= len(selected_specificity) - 1
    )
    finite_supported = True
    task_factor_interactions = []
    for seed in beta_cfg.seeds:
        run = runs[int(seed)]
        sem_sem = float(np.asarray(
            run["scores"]["semantic"]["score"]["per_graph"][aggregation], dtype=float
        ).mean())
        sem_str = float(np.asarray(
            run["scores"]["semantic"]["cross_factor_control"]["score"]["per_graph"][aggregation],
            dtype=float,
        ).mean())
        str_str = float(np.asarray(
            run["scores"]["structural"]["score"]["per_graph"][aggregation], dtype=float
        ).mean())
        str_sem = float(np.asarray(
            run["scores"]["structural"]["cross_factor_control"]["score"]["per_graph"][aggregation],
            dtype=float,
        ).mean())
        task_factor_interactions.append(
            math.log((sem_sem + EPS) / (str_sem + EPS))
            + math.log((str_str + EPS) / (sem_str + EPS))
        )
        for channel in CHANNELS:
            profile = runs[int(seed)]["scores"][channel]["score"]["distance_profiles"]
            support = np.asarray(profile["carrier_support"], dtype=float)
            sens = np.asarray(profile["F_sens"], dtype=float)
            coh = np.asarray(profile["F_coh"], dtype=float)
            supported = support > 0
            expanded = np.broadcast_to(supported[:, None, None, :], sens.shape)
            finite_supported &= bool(np.isfinite(sens[expanded]).all())
            finite_supported &= bool(np.isfinite(coh[expanded]).all())
    return {
        "aggregation": {
            "status": "primary" if aggregation != "CG" else "retain_legacy_CG",
            "method": aggregation,
        },
        "task_factor_specificity_control": {
            "status": "passes_negative_control"
            if min(task_factor_interactions) > 0 and selected_specificity_passed
            else "withhold_intrinsic_semantic_structural_labels",
            "per_seed_log_diagonal_interaction": task_factor_interactions,
            "selected_D_family_rows": len(selected_specificity),
            "claim": "labels require both an aggregate diagonal interaction and positive D-family matched-factor enrichment across seeds",
        },
        "D_relative_selectivity": coordinate_decision(
            "D_causal_role_rho", "signed finite causal channel role"
        ),
        "J_evoked_strength": coordinate_decision(
            "J_causal_strength_rho", "absolute overall finite causal influence"
        ),
        "G_balanced_strength": coordinate_decision(
            "G_balanced_causal_rho", "nonnegative balanced two-channel causal magnitude"
        ),
        "whole_transport_patching": {
            "status": "approve_for_real-task_beta" if patch_passed else "retain_as_diagnostic",
            "criterion": "D-specialist restore/inject channel interaction and excess over same-layer J-matched generalists are positive in every seed; all but at most one CI in each family exclude zero",
            "available_rows": len(available),
            "J_matched_control_rows": len(matched),
        },
        "routing_message_decomposition": {
            "status": "approve_for_real-task_beta"
            if numerical_checks["routing_message_reconstruction"] else "reject",
            "claim": "exact algebraic decomposition plus finite component patching; not attribution by decomposition alone",
        },
        "F_sens_head_site_carriage": {
            "status": "approve_as_sensitivity_diagnostic" if finite_supported else "reject",
            "claim": "support-aware head-site sensitivity; beneficial carriage is not validated by this query-only readout",
        },
        "conditional_specialisation": {
            "status": "pipeline_only",
            "claim": "sample-split null/plumbing test; no planted conditional specialist exists",
        },
    }


def create_outputs(
    runs: Mapping[int, Mapping[str, Any]],
    beta_cfg: BetaConfig,
) -> dict[str, Any]:
    tables = beta_cfg.beta_dir / "tables"
    figures = beta_cfg.beta_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    per_seed_metrics: dict[int, dict[str, dict[str, float]]] = {}
    method_rows: list[dict[str, Any]] = []
    for seed in beta_cfg.seeds:
        run = runs[int(seed)]
        metrics = method_metrics_for_seed(
            run["scores"],
            run["patches"],
            run["ablations"],
            beta_cfg,
            seed=int(seed),
        )
        per_seed_metrics[int(seed)] = metrics
        for aggregation, values in metrics.items():
            method_rows.append(
                {
                    "seed": int(seed),
                    "aggregation": aggregation,
                    **values,
                    "ablation_mean_rho": float(
                        np.nanmean(
                            [
                                values["ablation_semantic_rho"],
                                values["ablation_structural_rho"],
                            ]
                        )
                    ),
                    "split_half_mean_rho": float(
                        np.nanmean(
                            [
                                values["semantic_split_half_rho"],
                                values["structural_split_half_rho"],
                            ]
                        )
                    ),
                    "K_to_all_mean_rho": float(
                        np.nanmean(
                            [
                                values["semantic_k_to_all_rho"],
                                values["structural_k_to_all_rho"],
                            ]
                        )
                    ),
                    "topk_mean_stability": float(
                        np.nanmean(
                            [
                                values["semantic_topk_stability"],
                                values["structural_topk_stability"],
                            ]
                        )
                    ),
                }
            )
    decision = choose_and_confirm_method(per_seed_metrics, beta_cfg)
    selected_candidate = str(decision["selected_on_seeds_0_1"])
    bootstrap_confirmation = paired_graph_bootstrap_confirmation(
        runs[int(beta_cfg.seeds[2])],
        selected_candidate,
        beta_cfg,
        seed=6_900_000 + int(beta_cfg.seeds[2]),
    )
    decision["paired_graph_bootstrap"] = bootstrap_confirmation
    if selected_candidate != "CG" and not bool(bootstrap_confirmation["gate_passed"]):
        decision["advances"] = False
        decision["verdict"] = (
            "retain coherent-gross CG: the seed-2 paired graph-bootstrap replacement gate failed"
        )

    numerical_checks = {
        "CG_formula_regression": all(
            all(
                _cg_formula_regression_passed(value)
                for value in run["checks"]["cg_formula_error"].values()
            )
            for run in runs.values()
        ),
        "no_op_score": all(
            all(
                bool(
                    run["scores"][channel]["no_op_numerical_check"]["output"][
                        "passed"
                    ]
                )
                and all(
                    bool(
                        run["scores"][channel]["no_op_numerical_check"]["scores"][
                            aggregation
                        ]["passed"]
                    )
                    for aggregation in AGGREGATIONS
                )
                for channel in CHANNELS
            )
            for run in runs.values()
        ),
        "patch_sham": all(
            max(
                float(run["patches"][channel]["sham_tolerance_ratio_max"])
                for channel in CHANNELS
            )
            <= 1.0
            for run in runs.values()
        ),
        "routing_message_reconstruction": all(
            max(
                float(
                    run["mechanisms"][channel]["score"][
                        "reconstruction_tolerance_ratio_max"
                    ]
                )
                for channel in CHANNELS
            )
            <= 1.0
            and max(
                float(
                    run["mechanisms"][channel]["score"][
                        "projected_reconstruction_tolerance_ratio_max"
                    ]
                )
                for channel in CHANNELS
            )
            <= 1.0
            and max(
                float(
                    run["mechanisms"][channel]["patch"][
                        "reconstruction_tolerance_ratio_max"
                    ]
                )
                for channel in CHANNELS
            )
            <= 1.0
            for run in runs.values()
        ),
    }
    decision["numerical_checks"] = numerical_checks
    if not all(numerical_checks.values()):
        decision["advances"] = False
        decision["verdict"] = "retain CG: at least one numerical validity gate failed"
    chosen = selected_candidate if bool(decision["advances"]) else "CG"
    decision["primary_method"] = chosen
    decision["failed_candidate_is_diagnostic_only"] = bool(
        selected_candidate != chosen
    )

    conditional_rows = []
    for aggregation in AGGREGATIONS:
        for seed in beta_cfg.seeds:
            rows = conditional_discovery_confirmation(
                runs[int(seed)]["scores"]["semantic"],
                runs[int(seed)]["scores"]["structural"],
                aggregation=aggregation,
                beta_cfg=beta_cfg,
                seed=int(seed),
            )
            for row in rows:
                row["aggregation"] = aggregation
            conditional_rows.extend(rows)
    if conditional_rows:
        adjusted = _bh_adjust(
            [float(row.get("bootstrap_p", 1.0)) for row in conditional_rows]
        )
        for row, q_value in zip(conditional_rows, adjusted):
            row["fdr_q"] = float(q_value)
            row["confirmed_fdr_0.05"] = bool(
                q_value <= 0.05
                and row.get("sign_replicated", False)
                and row.get("confirmation_activity_valid", False)
                and row.get("dose_common_support_valid", False)
            )

    head_rows = build_head_rows(runs, beta_cfg)
    uncertainty_rows = coordinate_uncertainty_rows(runs, beta_cfg)
    benefit_rows = beneficial_outcome_rows(runs, beta_cfg)
    task_factor_rows = task_factor_control_rows(runs, beta_cfg)
    patch_rows = patch_metric_rows(runs, beta_cfg)
    calibration_rows = finite_gradient_calibration_rows(runs, beta_cfg)
    family_patch_rows = family_patch_interaction_rows(
        runs, beta_cfg, chosen
    )
    matched_patch_rows = matched_control_patch_rows(runs, beta_cfg, chosen)
    task_factor_family_rows = task_factor_family_specificity_rows(
        runs, beta_cfg, chosen
    )
    family_rows = family_ablation_rows(runs, beta_cfg)
    mechanism_table = mechanism_rows(runs, beta_cfg)
    carriage_table = carriage_rows(runs, beta_cfg)
    component_decisions = component_decision_records(
        per_seed_metrics,
        runs,
        beta_cfg,
        chosen,
        family_patch_rows,
        matched_patch_rows,
        task_factor_family_rows,
        numerical_checks,
    )
    write_csv(tables / "method_validation_by_seed.csv", method_rows)
    write_csv(tables / "per_head_beta_metrics.csv", head_rows)
    write_csv(tables / "per_head_score_coordinate_uncertainty.csv", uncertainty_rows)
    write_csv(tables / "conditional_rules_discovery_confirmation.csv", conditional_rows)
    write_csv(tables / "source_level_beneficial_outcomes.csv", benefit_rows)
    write_csv(tables / "task_mode_by_intervention_factor_controls.csv", task_factor_rows)
    write_csv(tables / "whole_transport_patch_metrics.csv", patch_rows)
    write_csv(tables / "finite_gradient_calibration_by_seed.csv", calibration_rows)
    write_csv(tables / "family_by_channel_patch_interactions.csv", family_patch_rows)
    write_csv(tables / "specialist_vs_matched_control_patching.csv", matched_patch_rows)
    write_csv(tables / "selected_family_task_factor_specificity.csv", task_factor_family_rows)
    write_csv(tables / "family_ablation_curves.csv", family_rows)
    write_csv(tables / "routing_message_component_patching.csv", mechanism_table)
    write_csv(tables / "head_site_carriage_distance.csv", carriage_table)
    write_json(tables / "method_decision.json", decision)
    write_json(tables / "component_decisions.json", component_decisions)
    write_json(
        tables / "family_selections.json",
        {
            str(seed): runs[int(seed)]["families"]["selection"]
            for seed in beta_cfg.seeds
        },
    )

    figure_paths = []
    figure_paths += figure_aggregation_planes(runs, beta_cfg, figures)
    figure_paths += figure_task_factor_controls(task_factor_rows, figures)
    figure_paths += figure_task_factor_family_specificity(
        task_factor_family_rows, figures
    )
    figure_paths += figure_method_comparison(method_rows, decision, figures)
    figure_paths += figure_winner_causal_scatter(head_rows, chosen, figures)
    figure_paths += figure_selectivity_strength(head_rows, chosen, figures)
    figure_paths += figure_bidirectional_patching(runs, beta_cfg, chosen, figures)
    figure_paths += figure_patch_controls(patch_rows, calibration_rows, figures)
    figure_paths += figure_family_ablation(runs, beta_cfg, chosen, figures)
    figure_paths += figure_routing_message(runs, head_rows, beta_cfg, chosen, figures)
    figure_paths += figure_carriage_coherence(runs, head_rows, beta_cfg, chosen, figures)
    figure_paths += figure_conditional_reliability_dose(
        runs, conditional_rows, beta_cfg, chosen, figures
    )

    limitations = {
        "not_approvable_with_this_checkpoint": [
            "non-isomorphic structural donors: every connected 2-regular 16-node training graph is a cycle; alternatives are isomorphic or training-OOD",
            "support/wiring decomposition: the trained model uses fixed dense support",
            "multi-source hierarchical weighting: exactly one planted source occurs per graph",
            "production integrated beneficial carriage: the final readout uses only the query node",
            "general conditional-discovery sensitivity: no context-gated specialist was planted",
            "transfer to sparse or molecular architectures",
            "cross-task causal patching: the causal beta validates the matched task-factor diagonal; off-diagonal specificity is scored but not patched",
        ],
        "valid_here": [
            "four score aggregations on identical global interventions",
            "full task-mode by intervention-factor score controls",
            "D/J/G against intended finite causal targets",
            "bidirectional whole-node-transport mediation and zero-ablation necessity",
            "fixed-support routing/message description and component patching",
            "head-site F_sens/F_coh, event-specific structural distance, donor outcomes",
            "sample-split conditional pipeline operation (not false-positive calibration)",
        ],
    }
    write_json(tables / "scope_and_limitations.json", limitations)
    summary = {
        "version": BETA_VERSION,
        "schema": BETA_SCHEMA,
        "fingerprint": beta_cfg.fingerprint,
        "decision": decision,
        "component_decisions": component_decisions,
        "chosen_for_beta_figures": chosen,
        "selection_candidate_diagnostic": selected_candidate,
        "figures": figure_paths,
        "tables": [str(path) for path in sorted(tables.glob("*"))],
        "scope": limitations,
    }
    write_json(beta_cfg.beta_dir / "summary.json", summary)
    return summary


# ======================================================================================
# Orchestration
# ======================================================================================


def environment_record(device: Any) -> dict[str, Any]:
    import torch

    packages = {}
    for name in ("torch", "torch_geometric", "numpy", "scipy", "matplotlib"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            packages[name] = f"unavailable: {exc}"
    return {
        "version": BETA_VERSION,
        "python": sys.version,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "gpu": torch.cuda.get_device_name(device)
        if torch.cuda.is_available() and str(device).startswith("cuda")
        else None,
        "packages": packages,
    }


def load_cached_runs(
    legacy: Any,
    beta_cfg: BetaConfig,
) -> dict[int, dict[str, Any]]:
    import torch

    runs: dict[int, dict[str, Any]] = {}
    for seed in beta_cfg.seeds:
        checkpoint_path, checkpoint = discover_checkpoint(
            legacy, beta_cfg.legacy_run_dir, int(seed)
        )
        checkpoint_sha256 = sha256_file(checkpoint_path)
        config_values = dict(checkpoint["config"])
        config_values["seeds"] = tuple(config_values.get("seeds", beta_cfg.seeds))
        model_cfg = legacy.Config(**config_values)
        score_payloads = {
            channel: _load_validated_beta_cache(
                score_cache_path(
                    beta_cfg.beta_dir, int(seed), channel, beta_cfg.fingerprint
                ),
                beta_cfg,
                checkpoint_sha256,
            )
            for channel in CHANNELS
        }
        patch_payloads = {
            channel: _load_validated_beta_cache(
                patch_cache_path(
                    beta_cfg.beta_dir, int(seed), channel, beta_cfg.fingerprint
                ),
                beta_cfg,
                checkpoint_sha256,
            )
            for channel in CHANNELS
        }
        mechanisms = {
            channel: _load_validated_beta_cache(
                mechanism_cache_path(
                    beta_cfg.beta_dir, int(seed), channel, beta_cfg.fingerprint
                ),
                beta_cfg,
                checkpoint_sha256,
            )
            for channel in CHANNELS
        }
        families = _load_validated_beta_cache(
            family_cache_path(beta_cfg.beta_dir, int(seed), beta_cfg.fingerprint),
            beta_cfg,
            checkpoint_sha256,
        )
        fallbacks = sorted(
            (beta_cfg.beta_dir / "causal").glob(
                f"seed_{seed}__ablation_fallback__{beta_cfg.fingerprint}__{checkpoint_sha256[:12]}.pt"
            )
        )
        if len(fallbacks) != 1:
            raise FileNotFoundError(
                f"no checkpoint-bound beta ablation cache for seed {seed}; run --phase families"
            )
        ablations = _load_validated_beta_cache(
            fallbacks[0], beta_cfg, checkpoint_sha256
        )
        check_path = beta_cfg.beta_dir / "verification" / f"seed_{seed}.json"
        if not check_path.exists():
            raise FileNotFoundError(
                f"missing fail-closed verification manifest for seed {seed}: {check_path}"
            )
        checks = json.loads(check_path.read_text())
        if (
            checks.get("version") != BETA_VERSION
            or int(checks.get("schema", -1)) != BETA_SCHEMA
            or checks.get("fingerprint") != beta_cfg.fingerprint
        ):
            raise RuntimeError(f"stale verification manifest for seed {seed}")
        if checks.get("checkpoint_sha256") != checkpoint_sha256:
            raise RuntimeError(f"verification checkpoint hash mismatch for seed {seed}")
        cg_errors = checks.get("cg_formula_error")
        if not isinstance(cg_errors, Mapping) or set(cg_errors) != set(CHANNELS):
            raise RuntimeError(f"incomplete CG verification for seed {seed}")
        runs[int(seed)] = {
            "model_cfg": model_cfg,
            "scores": score_payloads,
            "patches": patch_payloads,
            "mechanisms": mechanisms,
            "families": families,
            "ablations": ablations,
            "checks": checks,
        }
    return runs


def make_beta_config(args: argparse.Namespace) -> BetaConfig:
    values = {
        "legacy_root": args.legacy_root,
        "legacy_run": args.legacy_run,
        "beta_run": args.beta_run,
        "seeds": tuple(int(value) for value in args.seeds),
        "score_graphs": args.score_graphs,
        "score_batch_size": args.score_batch_size,
        "score_event_mode": args.score_event_mode,
        "score_sampled_events": args.score_sampled_events,
        "causal_graphs": args.causal_graphs,
        "causal_events": args.causal_events,
        "causal_batch_size": args.causal_batch_size,
        "mechanism_graphs": args.mechanism_graphs,
        "mechanism_events": args.mechanism_events,
        "family_graphs": args.family_graphs,
        "family_size": args.family_size,
        "bootstrap_samples": args.bootstrap_samples,
        "conditional_bootstrap_samples": args.conditional_bootstrap_samples,
        "device": args.device,
    }
    if args.fast_dev_run:
        values.update(
            {
                "beta_run": args.beta_run + "_fast_dev",
                "score_graphs": 8,
                "score_batch_size": 4,
                "score_event_mode": "sample",
                "score_sampled_events": 2,
                "causal_graphs": 6,
                "causal_events": 1,
                "causal_batch_size": 6,
                "mechanism_graphs": 4,
                "mechanism_events": 1,
                "family_graphs": 12,
                "family_size": 1,
                "bootstrap_samples": 50,
                "conditional_bootstrap_samples": 50,
            }
        )
    config = BetaConfig(**values)
    config.validate()
    return config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Beta validation of successor semantic/structural head scores"
    )
    parser.add_argument(
        "--phase",
        choices=("all", "scores", "causal", "mechanism", "families", "figures"),
        default="all",
    )
    parser.add_argument("--legacy-root", default=str(DEFAULT_DRIVE_ROOT))
    parser.add_argument("--legacy-run", default=DEFAULT_LEGACY_RUN)
    parser.add_argument("--beta-run", default=DEFAULT_BETA_RUN)
    parser.add_argument("--seeds", nargs=3, type=int, default=[0, 1, 2])
    parser.add_argument("--score-graphs", type=int, default=96)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--score-event-mode", choices=("enumerate", "sample"), default="enumerate")
    parser.add_argument("--score-sampled-events", type=int, default=6)
    parser.add_argument("--causal-graphs", type=int, default=96)
    parser.add_argument("--causal-events", type=int, default=2)
    parser.add_argument("--causal-batch-size", type=int, default=48)
    parser.add_argument("--mechanism-graphs", type=int, default=32)
    parser.add_argument("--mechanism-events", type=int, default=2)
    parser.add_argument("--family-graphs", type=int, default=192)
    parser.add_argument("--family-size", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--conditional-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repository-branch", default=REPOSITORY_BRANCH)
    parser.add_argument("--grit-dir", default=str(DEFAULT_GRIT_DIR))
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-grit-install", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    import torch

    args = parse_args(argv)
    repository = bootstrap_repository(
        branch=args.repository_branch, skip_checkout=args.skip_bootstrap
    )
    legacy = load_legacy_module(repository)
    beta_cfg = make_beta_config(args)
    beta_cfg.beta_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        beta_cfg.beta_dir / "beta_config.json",
        {"version": BETA_VERSION, "fingerprint": beta_cfg.fingerprint, "config": asdict(beta_cfg)},
    )

    needs_model = args.phase != "figures"
    if needs_model:
        legacy.setup_official_grit(
            Path(args.grit_dir), install=not args.skip_grit_install
        )
    device = torch.device(
        beta_cfg.device
        if torch.cuda.is_available() or not beta_cfg.device.startswith("cuda")
        else "cpu"
    )
    environment_path = (
        beta_cfg.beta_dir / "environment.json"
        if needs_model
        else beta_cfg.beta_dir / "render_environment.json"
    )
    write_json(environment_path, environment_record(device))
    if needs_model and device.type != "cuda":
        print("[warn] CUDA unavailable; the full official-GRIT beta will be slow", flush=True)

    if needs_model:
        for seed in beta_cfg.seeds:
            model, model_cfg, _checkpoint, checkpoint_path = load_seed_model(
                legacy,
                beta_cfg.legacy_run_dir,
                seed=int(seed),
                device=device,
            )
            checkpoint_sha256 = sha256_file(checkpoint_path)
            checks = {
                "version": BETA_VERSION,
                "schema": BETA_SCHEMA,
                "fingerprint": beta_cfg.fingerprint,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "cg_formula_error": {
                    channel: _legacy_formula_regression(
                        legacy,
                        model,
                        model_cfg,
                        mode=legacy.MODE_SEMANTIC
                        if channel == "semantic"
                        else legacy.MODE_STRUCTURAL,
                        factor=channel,
                        seed=7_000_000 + int(seed) * 101 + CHANNELS.index(channel),
                        device=device,
                    )
                    for channel in CHANNELS
                },
            }
            write_json(beta_cfg.beta_dir / "verification" / f"seed_{seed}.json", checks)
            score_payloads: dict[str, Any] = {}
            if args.phase in {"all", "scores", "families"}:
                for channel in CHANNELS:
                    score_payloads[channel] = run_score_channel(
                        legacy,
                        model,
                        model_cfg,
                        beta_cfg,
                        seed=int(seed),
                        channel=channel,
                        device=device,
                        force=args.force,
                        checkpoint_sha256=checkpoint_sha256,
                    )
            if args.phase in {"all", "causal"}:
                for channel in CHANNELS:
                    run_patch_channel(
                        legacy,
                        model,
                        model_cfg,
                        beta_cfg,
                        seed=int(seed),
                        channel=channel,
                        device=device,
                        force=args.force,
                        checkpoint_sha256=checkpoint_sha256,
                    )
            if args.phase in {"all", "mechanism"}:
                for channel in CHANNELS:
                    run_mechanism_channel(
                        legacy,
                        model,
                        model_cfg,
                        beta_cfg,
                        seed=int(seed),
                        channel=channel,
                        device=device,
                        force=args.force,
                        checkpoint_sha256=checkpoint_sha256,
                    )
            if args.phase in {"all", "families"}:
                if not score_payloads:
                    score_payloads = {
                        channel: _load_validated_beta_cache(
                            score_cache_path(
                                beta_cfg.beta_dir,
                                int(seed),
                                channel,
                                beta_cfg.fingerprint,
                            ),
                            beta_cfg,
                            checkpoint_sha256,
                        )
                        for channel in CHANNELS
                    }
                # Ensure an ablation cache exists before a later figure-only run.
                load_ablation_inputs(
                    legacy,
                    model,
                    model_cfg,
                    beta_cfg,
                    seed=int(seed),
                    device=device,
                    checkpoint_sha256=checkpoint_sha256,
                    force=args.force,
                )
                run_family_ablation_seed(
                    legacy,
                    model,
                    model_cfg,
                    beta_cfg,
                    score_payloads,
                    seed=int(seed),
                    device=device,
                    force=args.force,
                    checkpoint_sha256=checkpoint_sha256,
                )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.phase not in {"all", "figures"}:
        result = {
            "version": BETA_VERSION,
            "phase": args.phase,
            "beta_dir": str(beta_cfg.beta_dir),
        }
        print(f"[done] {result}", flush=True)
        return result

    runs = load_cached_runs(legacy, beta_cfg)
    summary = create_outputs(runs, beta_cfg)
    print("\n[done]", flush=True)
    print(f"  beta directory: {beta_cfg.beta_dir}", flush=True)
    print(f"  decision: {summary['decision']['verdict']}", flush=True)
    print(f"  figures: {beta_cfg.beta_dir / 'figures'}", flush=True)
    print(f"  tables: {beta_cfg.beta_dir / 'tables'}", flush=True)
    return summary


CELL_ARGS = [
    "--phase", "all",
    "--legacy-root", str(DEFAULT_DRIVE_ROOT),
    "--legacy-run", DEFAULT_LEGACY_RUN,
    "--beta-run", DEFAULT_BETA_RUN,
    "--seeds", "0", "1", "2",
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
