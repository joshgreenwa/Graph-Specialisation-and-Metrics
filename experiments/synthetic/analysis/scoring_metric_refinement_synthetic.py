"""Synthetic-task adapter for the M1/M4/M5/M7 scoring comparison.

The trained ``cycle_dual_v2`` checkpoints and the original causal analysis are
read from their existing Drive location.  New score components and method-specific
family ablations are cached in a separate child directory, so this experiment never
overwrites a checkpoint or an earlier analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from graph_specialisation_metrics.scoring_refinement.synthetic_validation import (
    SYNTHETIC_METHODS,
    create_validation_outputs,
    method_score_axes,
    select_head_groups,
)


SYNTHETIC_REFINEMENT_VERSION = "synthetic-scoring-refinement-m1-m4-m5-m7-v1"
DEFAULT_DRIVE_ROOT = (
    "/content/drive/MyDrive/graph_specialisation_metrics/"
    "causal_specialisation_double_dissociation"
)
DEFAULT_RUN_NAME = "cycle_dual_v2"
DEFAULT_OUTPUT_NAME = "scoring_refinement_m1_m4_m5_m7_v1"
DEFAULT_GRIT_DIR = "/content/GRIT"
VARIANTS = (
    "semantic_single_donor",
    "semantic_transposition",
    "pe_single_donor",
    "pe_transposition",
)
METHOD_FAMILY_REVISION = "synthetic-scoring-refinement-family-ablation-v3"
EPS = 1.0e-12


def load_legacy_module(repository: str | Path) -> Any:
    """Load the frozen training/causal implementation without importing GRIT early."""

    path = (
        Path(repository)
        / "experiments"
        / "synthetic"
        / "training"
        / "causal_specialisation_double_dissociation_colab.py"
    )
    if not path.exists():
        raise FileNotFoundError(f"legacy synthetic implementation is missing: {path}")
    name = "_synthetic_mixed_task_legacy_for_score_refinement"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def copy_rrwp_footprint(rrwp: Any, source: int, partner: int) -> Any:
    """Copy one partner's dense RRWP row/column onto one source node."""

    source, partner = int(source), int(partner)
    if source == partner:
        return rrwp.clone()
    output = rrwp.clone()
    donor_row = rrwp[partner].clone()
    donor_column = rrwp[:, partner].clone()
    output[source] = donor_row
    output[:, source] = donor_column
    output[source, source] = rrwp[partner, partner]
    return output


def _semantic_partners(cfg: Any, query: int, source: int) -> list[int]:
    candidates = [
        node
        for node in range(int(cfg.n))
        if node not in {int(query), int(source)}
    ]
    if not candidates:
        raise RuntimeError("semantic transposition has no non-query partner")
    return candidates


def make_variant_replicas(
    legacy: Any,
    cfg: Any,
    clean: Any,
    *,
    variant: str,
    donors: int,
    seed: int,
) -> tuple[Any, np.ndarray]:
    """Return graph-major ``[clean, event_1, ..., event_K]`` replicas.

    Semantic transposition exchanges complete content rows while RRWP is fixed.
    PE single-donor copies one RRWP footprint; PE transposition conjugates the
    whole RRWP matrix.  Query/control data and labels are held fixed.
    """

    import torch

    if variant not in VARIANTS:
        raise ValueError(f"unknown synthetic score variant {variant!r}")
    rng = np.random.default_rng(int(seed))
    replicas = []
    partners = np.full((len(clean), int(donors)), -1, dtype=np.int64)
    for graph in range(len(clean)):
        one = clean.slice(graph, graph + 1)
        replicas.append(one)
        source = int(one.target_idx[0])
        query = int(one.q_idx[0])
        for event in range(int(donors)):
            x = one.x.clone()
            rrwp = one.rrwp.clone()
            if variant == "semantic_single_donor":
                current = int(
                    torch.argmax(
                        x[
                            0,
                            source,
                            cfg.key_vocab : cfg.key_vocab + cfg.classes,
                        ]
                    ).item()
                )
                donor_value = legacy.draw_other_class(
                    rng, cfg.classes, current
                )
                x[0, source] = legacy.replace_value(
                    cfg, x[0, source], donor_value
                )
            elif variant == "semantic_transposition":
                partner = int(
                    rng.choice(_semantic_partners(cfg, query, source))
                )
                partners[graph, event] = partner
                first = x[0, source].clone()
                x[0, source] = x[0, partner]
                x[0, partner] = first
            elif variant in {"pe_single_donor", "pe_transposition"}:
                partner = int(
                    rng.choice(
                        legacy.structural_partners(cfg, query, source)
                    )
                )
                partners[graph, event] = partner
                if variant == "pe_single_donor":
                    rrwp[0] = copy_rrwp_footprint(
                        rrwp[0], source, partner
                    )
                else:
                    rrwp[0] = legacy.transpose_rrwp(
                        rrwp[0], source, partner
                    )
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
    return legacy.concat_batches(replicas), partners


def _pair_cosine(event: Any, reference: Any) -> tuple[Any, Any]:
    import torch

    numerator = (event * reference).sum(dim=0)
    event_norm = event.square().sum(dim=0)
    reference_norm = reference.square().sum(dim=0)
    valid = (event_norm > EPS) & (reference_norm > EPS)
    cosine = numerator / torch.sqrt(event_norm * reference_norm).clamp_min(EPS)
    cosine = torch.where(valid, cosine, torch.zeros_like(cosine))
    return torch.clamp(cosine, -1.0, 1.0), valid


def appendix_attention_scores(
    attention: Any,
    *,
    clean: Any,
    partners: np.ndarray,
    cfg: Any,
    temperature: float,
) -> tuple[Any, Any]:
    """Appendix-style attention follow/invariance for batched transpositions.

    ``attention`` uses the official synthetic adapter's fixed ordering
    ``[graph, replica, source, receiver, head]``. Swap weights are
    ``softmax(|alpha_u-alpha_v| / temperature)`` over sampled partners, and
    receivers are weighted by their clean attention mass on the swapped pair.
    """

    import torch

    if float(temperature) <= 0.0:
        raise ValueError("cosine temperature must be positive")
    graphs = len(clean)
    events = int(partners.shape[1])
    values = attention.reshape(
        graphs,
        events + 1,
        int(cfg.n),
        int(cfg.n),
        int(cfg.heads),
    ).float()
    following = torch.zeros(graphs, int(cfg.heads), device=values.device)
    invariant = torch.zeros_like(following)
    for graph in range(graphs):
        source = int(clean.target_idx[graph])
        follow_events = []
        invariant_events = []
        valid_events = []
        logits = []
        pair_masses = []
        for event in range(events):
            partner = int(partners[graph, event])
            if partner < 0:
                raise ValueError("attention scoring requires transposition partners")
            indices = torch.as_tensor(
                [source, partner], dtype=torch.long, device=values.device
            )
            clean_pair = values[graph, 0].index_select(0, indices)
            event_pair = values[graph, event + 1].index_select(0, indices)
            invariant_value, invariant_valid = _pair_cosine(
                event_pair, clean_pair
            )
            follow_value, follow_valid = _pair_cosine(
                event_pair, clean_pair.flip(dims=(0,))
            )
            follow_events.append(follow_value)
            invariant_events.append(invariant_value)
            valid_events.append(invariant_valid & follow_valid)
            logits.append(torch.abs(clean_pair[0] - clean_pair[1]))
            pair_masses.append(clean_pair.sum(dim=0))
        valid = torch.stack(valid_events, dim=0)
        logits_tensor = torch.stack(logits, dim=0)
        any_event = valid.any(dim=0, keepdim=True)
        masked_logits = torch.where(
            valid,
            logits_tensor / float(temperature),
            torch.full_like(logits_tensor, -torch.inf),
        )
        masked_logits = torch.where(
            any_event, masked_logits, torch.zeros_like(masked_logits)
        )
        weights = torch.softmax(masked_logits, dim=0)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        pair_mass = (
            weights * torch.stack(pair_masses, dim=0)
        ).sum(dim=0)
        valid_receiver = valid.any(dim=0)
        receiver_weight = torch.where(
            valid_receiver, pair_mass, torch.zeros_like(pair_mass)
        )
        weight_total = receiver_weight.sum(dim=0)
        receiver_weight = torch.where(
            weight_total.unsqueeze(0) > EPS,
            receiver_weight,
            valid_receiver.to(receiver_weight.dtype),
        )
        weight_total = receiver_weight.sum(dim=0).clamp_min(EPS)
        follow_average = (
            weights
            * torch.where(
                valid,
                torch.stack(follow_events, dim=0),
                torch.zeros_like(weights),
            )
        ).sum(dim=0)
        invariant_average = (
            weights
            * torch.where(
                valid,
                torch.stack(invariant_events, dim=0),
                torch.zeros_like(weights),
            )
        ).sum(dim=0)
        following[graph] = (
            receiver_weight * follow_average
        ).sum(dim=0) / weight_total
        invariant[graph] = (
            receiver_weight * invariant_average
        ).sum(dim=0) / weight_total
    return following, invariant


def score_variant_batch(
    legacy: Any,
    model: Any,
    cfg: Any,
    clean: Any,
    *,
    variant: str,
    seed: int,
    device: Any,
    cosine_temperature: float,
) -> dict[str, Any]:
    """Compute output-projected EG and optional attention cosine in one pass."""

    import torch

    replicas, partners = make_variant_replicas(
        legacy,
        cfg,
        clean.cpu(),
        variant=variant,
        donors=cfg.score_donors,
        seed=seed,
    )
    replicas = replicas.to(device)
    transposition = variant.endswith("transposition")
    result = legacy.capture_forward(
        model,
        replicas,
        want_grad=True,
        want_attention=transposition,
    )
    logits = result["logits"]
    routed = result["wV"]
    graphs = len(clean)
    reps = int(cfg.score_donors) + 1
    clean_rows = torch.arange(graphs, device=device) * reps
    accum = [
        torch.zeros(
            graphs, cfg.n, cfg.heads, device=device, dtype=logits.dtype
        )
        for _ in range(cfg.layers)
    ]
    for output_index in range(cfg.classes):
        gradients = torch.autograd.grad(
            logits[clean_rows, output_index].sum(),
            routed,
            retain_graph=output_index < cfg.classes - 1,
            allow_unused=False,
        )
        for layer in range(cfg.layers):
            states = routed[layer].reshape(
                graphs, reps, cfg.n, cfg.heads, -1
            )
            gradient = gradients[layer].reshape(
                graphs, reps, cfg.n, cfg.heads, -1
            )[:, 0]
            delta = states[:, 0] - states[:, 1:].mean(dim=1)
            projected = (gradient * delta).sum(dim=-1)
            accum[layer] += projected.square()
    eg = torch.stack(
        [value.sqrt().sum(dim=1) for value in accum], dim=1
    ).detach().cpu()
    if not bool(torch.isfinite(eg).all()) or float(eg.max()) <= 0.0:
        raise RuntimeError(f"{variant} produced an invalid or identically zero EG score")
    output: dict[str, Any] = {"eg": eg}
    if transposition:
        follow_layers = []
        invariant_layers = []
        for attention in result["attention"]:
            follow, invariant = appendix_attention_scores(
                attention,
                clean=clean,
                partners=partners,
                cfg=cfg,
                temperature=cosine_temperature,
            )
            follow_layers.append(follow)
            invariant_layers.append(invariant)
        output["attention_follow"] = (
            torch.stack(follow_layers, dim=1).detach().cpu()
        )
        output["attention_invariant"] = (
            torch.stack(invariant_layers, dim=1).detach().cpu()
        )
        for key in ("attention_follow", "attention_invariant"):
            value = output[key]
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"{variant} produced non-finite {key}")
    return output


def score_variant(
    legacy: Any,
    model: Any,
    cfg: Any,
    *,
    mode: int,
    variant: str,
    seed: int,
    device: Any,
    cosine_temperature: float,
) -> dict[str, Any]:
    import torch

    clean = legacy.make_batch(cfg, cfg.score_graphs, seed, mode=mode)
    pieces: dict[str, list[Any]] = {}
    for start in range(0, len(clean), cfg.score_batch_size):
        stop = min(start + cfg.score_batch_size, len(clean))
        value = score_variant_batch(
            legacy,
            model,
            cfg,
            clean.slice(start, stop),
            variant=variant,
            seed=seed + start * 97,
            device=device,
            cosine_temperature=cosine_temperature,
        )
        for key, tensor in value.items():
            pieces.setdefault(key, []).append(tensor)
        print(
            f"[score-refinement {variant}] {stop}/{len(clean)}",
            flush=True,
        )
    return {
        key: torch.cat(value, dim=0)
        for key, value in pieces.items()
    }


def _cache_path(
    output_dir: Path,
    cfg: Any,
    legacy: Any,
    seed: int,
    cosine_temperature: float,
) -> Path:
    digest = hashlib.sha1(
        json.dumps(
            {
                "version": SYNTHETIC_REFINEMENT_VERSION,
                "legacy_fingerprint": legacy.config_fingerprint(cfg),
                "methods": list(SYNTHETIC_METHODS),
                "cosine_temperature": float(cosine_temperature),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return output_dir / "analysis" / f"seed_{int(seed)}__{digest}.pt"


def _save_cache(path: Path, value: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _complete(result: Mapping[str, Any]) -> bool:
    return (
        result.get("version") == SYNTHETIC_REFINEMENT_VERSION
        and set(result.get("method_scores", {})) == set(SYNTHETIC_METHODS)
        and set(result.get("method_family_ablation", {}))
        == set(SYNTHETIC_METHODS)
        and all(
            result["method_family_ablation"][method].get("revision")
            == f"{METHOD_FAMILY_REVISION}:{method}"
            for method in SYNTHETIC_METHODS
        )
    )


def _method_scores(
    components: Mapping[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    averaged = {
        key: np.asarray(value, dtype=float).mean(axis=0)
        for key, value in components.items()
    }
    return {
        method: {"semantic": semantic, "structural": structural}
        for method, (semantic, structural) in method_score_axes(averaged).items()
    }


def analyze_seed(
    legacy: Any,
    model: Any,
    cfg: Any,
    *,
    seed: int,
    legacy_run_dir: Path,
    output_dir: Path,
    device: Any,
    cosine_temperature: float,
    force: bool,
) -> dict[str, Any]:
    """Reuse the original causal tensors and add only missing score methods."""

    import torch

    path = _cache_path(
        output_dir, cfg, legacy, seed, cosine_temperature
    )
    if path.exists() and not force:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if _complete(cached):
            print(f"[score-refinement seed={seed}] loaded {path}", flush=True)
            return cached
    else:
        cached = {}

    legacy_result = legacy.analyze_seed(
        model,
        cfg,
        seed=seed,
        run_dir=legacy_run_dir,
        device=device,
        force=False,
    )
    if force or cached.get("version") != SYNTHETIC_REFINEMENT_VERSION:
        cached = {
            "version": SYNTHETIC_REFINEMENT_VERSION,
            "seed": int(seed),
            "legacy_fingerprint": legacy.config_fingerprint(cfg),
            "checks": legacy_result["checks"],
            "ablation_semantic": legacy_result["ablation_semantic"],
            "ablation_structural": legacy_result["ablation_structural"],
            "rescue_semantic": legacy_result["rescue_semantic"],
            "rescue_structural": legacy_result["rescue_structural"],
            "components": {
                "eg_semantic_single": legacy_result[
                    "semantic_score_per_graph"
                ],
                "eg_pe_transposition": legacy_result[
                    "structural_score_per_graph"
                ],
            },
            "method_family_ablation": {},
        }

    components = cached["components"]
    if (
        "eg_semantic_transposition" not in components
        or "semantic_attention_follow" not in components
        or "semantic_attention_invariant" not in components
    ):
        semantic = score_variant(
            legacy,
            model,
            cfg,
            mode=legacy.MODE_SEMANTIC,
            variant="semantic_transposition",
            seed=200_000 + seed,
            device=device,
            cosine_temperature=cosine_temperature,
        )
        components["eg_semantic_transposition"] = semantic["eg"]
        components["semantic_attention_follow"] = semantic[
            "attention_follow"
        ]
        components["semantic_attention_invariant"] = semantic[
            "attention_invariant"
        ]
        _save_cache(path, cached)
    if "eg_pe_single" not in components:
        pe_donor = score_variant(
            legacy,
            model,
            cfg,
            mode=legacy.MODE_STRUCTURAL,
            variant="pe_single_donor",
            seed=300_000 + seed,
            device=device,
            cosine_temperature=cosine_temperature,
        )
        components["eg_pe_single"] = pe_donor["eg"]
        _save_cache(path, cached)
    if (
        "pe_attention_follow" not in components
        or "pe_attention_invariant" not in components
    ):
        pe_transposition = score_variant(
            legacy,
            model,
            cfg,
            mode=legacy.MODE_STRUCTURAL,
            variant="pe_transposition",
            seed=300_000 + seed,
            device=device,
            cosine_temperature=cosine_temperature,
        )
        components["pe_attention_follow"] = pe_transposition[
            "attention_follow"
        ]
        components["pe_attention_invariant"] = pe_transposition[
            "attention_invariant"
        ]
        _save_cache(path, cached)

    cached["method_scores"] = _method_scores(components)
    m1_semantic_reference = float(
        np.mean(cached["method_scores"]["M1_DT"]["semantic"])
    )
    m1_structural_reference = float(
        np.mean(cached["method_scores"]["M1_DT"]["structural"])
    )
    cached["method_selected_groups"] = {}
    for method, score in cached["method_scores"].items():
        shared_m1_reference = method.startswith("M1_")
        cached["method_selected_groups"][method] = select_head_groups(
            score["semantic"],
            score["structural"],
            size=cfg.top_group_size,
            semantic_reference=(
                m1_semantic_reference if shared_m1_reference else None
            ),
            structural_reference=(
                m1_structural_reference if shared_m1_reference else None
            ),
        )
    family_cache = cached.setdefault("method_family_ablation", {})
    for method in SYNTHETIC_METHODS:
        revision = f"{METHOD_FAMILY_REVISION}:{method}"
        if family_cache.get(method, {}).get("revision") == revision:
            continue
        groups = cached["method_selected_groups"][method]
        family_cache[method] = legacy.family_ablation_sweep(
            model,
            cfg,
            groups=groups,
            # Identical held-out graph draw for every method.
            seed=1_200_000 + seed,
            device=device,
            revision=revision,
        )
        _save_cache(path, cached)
    if not _complete(cached):
        raise RuntimeError(f"synthetic refinement cache is incomplete: {path}")
    print(f"[score-refinement seed={seed}] cached {path}", flush=True)
    return cached


def make_config(legacy: Any, args: argparse.Namespace) -> Any:
    values = {
        "run_name": args.run_name,
        "drive_root": args.drive_root,
        "n": args.n,
        "key_vocab": args.key_vocab,
        "classes": args.classes,
        "rrwp_steps": args.rrwp_steps,
        "dim": args.dim,
        "heads": args.heads,
        "layers": args.layers,
        "dropout": args.dropout,
        "attention_dropout": args.attention_dropout,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "eval_every": args.eval_every,
        "validation_graphs": args.validation_graphs,
        "patience_checks": args.patience_checks,
        "accuracy_gate": args.accuracy_gate,
        "score_graphs": args.score_graphs,
        "score_donors": args.score_donors,
        "score_batch_size": args.score_batch_size,
        "ablation_graphs": args.ablation_graphs,
        "rescue_graphs": args.rescue_graphs,
        "analysis_batch_size": args.analysis_batch_size,
        "top_group_size": args.top_group_size,
        "seeds": tuple(int(seed) for seed in args.seeds),
        "device": args.device,
    }
    cfg = legacy.Config(**values)
    cfg.validate()
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument(
        "--phase", choices=("all", "analyze", "figures"), default="all"
    )
    parser.add_argument("--drive-root", default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--grit-dir", default=DEFAULT_GRIT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--key-vocab", type=int, default=32)
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--rrwp-steps", type=int, default=10)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--validation-graphs", type=int, default=512)
    parser.add_argument("--patience-checks", type=int, default=10)
    parser.add_argument("--accuracy-gate", type=float, default=0.90)
    parser.add_argument("--score-graphs", type=int, default=96)
    parser.add_argument("--score-donors", type=int, default=6)
    parser.add_argument("--score-batch-size", type=int, default=12)
    parser.add_argument("--ablation-graphs", type=int, default=256)
    parser.add_argument("--rescue-graphs", type=int, default=128)
    parser.add_argument("--analysis-batch-size", type=int, default=64)
    parser.add_argument("--top-group-size", type=int, default=3)
    parser.add_argument("--cosine-temperature", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-analysis", action="store_true")
    parser.add_argument("--allow-low-accuracy", action="store_true")
    parser.add_argument("--skip-grit-install", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    import torch

    args = build_parser().parse_args(argv)
    legacy = load_legacy_module(args.repository)
    cfg = make_config(legacy, args)
    legacy_run_dir = Path(cfg.drive_root) / cfg.run_name
    output_dir = legacy_run_dir / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "version": SYNTHETIC_REFINEMENT_VERSION,
        "methods": list(SYNTHETIC_METHODS),
        "legacy_run_dir": str(legacy_run_dir),
        "output_dir": str(output_dir),
        "legacy_fingerprint": legacy.config_fingerprint(cfg),
        "cosine_temperature": float(args.cosine_temperature),
        "config": asdict(cfg),
    }
    legacy.write_json(output_dir / "protocol.json", protocol)

    results = []
    if args.phase in {"all", "analyze"}:
        legacy.setup_official_grit(
            Path(args.grit_dir), install=not args.skip_grit_install
        )
        device = torch.device(
            args.device
            if torch.cuda.is_available() or not str(args.device).startswith("cuda")
            else "cpu"
        )
        if device.type != "cuda":
            print("[warn] CUDA unavailable; synthetic GRIT analysis will be slow")
        for seed in cfg.seeds:
            model, checkpoint = legacy.train_seed(
                cfg,
                seed=seed,
                run_dir=legacy_run_dir,
                device=device,
                force=False,
                load_only=True,
            )
            heldout = checkpoint.get("heldout_validation", {})
            minimum = min(
                float(heldout.get("semantic_accuracy", 0.0)),
                float(heldout.get("structural_accuracy", 0.0)),
            )
            if minimum < cfg.accuracy_gate and not args.allow_low_accuracy:
                raise RuntimeError(
                    f"seed {seed} failed the clean accuracy gate: {heldout}"
                )
            results.append(
                analyze_seed(
                    legacy,
                    model,
                    cfg,
                    seed=seed,
                    legacy_run_dir=legacy_run_dir,
                    output_dir=output_dir,
                    device=device,
                    cosine_temperature=args.cosine_temperature,
                    force=args.force_analysis,
                )
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        for seed in cfg.seeds:
            path = _cache_path(
                output_dir,
                cfg,
                legacy,
                seed,
                args.cosine_temperature,
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"figures phase requires cached analysis: {path}"
                )
            result = torch.load(path, map_location="cpu", weights_only=False)
            if not _complete(result):
                raise RuntimeError(
                    f"figures phase found an incomplete analysis cache: {path}"
                )
            results.append(result)

    if args.phase == "analyze":
        return {
            "output_dir": str(output_dir),
            "analysis": [
                str(
                    _cache_path(
                        output_dir,
                        cfg,
                        legacy,
                        seed,
                        args.cosine_temperature,
                    )
                )
                for seed in cfg.seeds
            ],
        }
    summary = create_validation_outputs(results, output_dir=output_dir)
    print("\n[done]", flush=True)
    print(f"  output: {output_dir}", flush=True)
    print(f"  figures: {output_dir / 'figures'}", flush=True)
    print(f"  tables: {output_dir / 'tables'}", flush=True)
    return summary


if __name__ == "__main__":
    main()
