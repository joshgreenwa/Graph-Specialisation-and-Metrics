"""Targeted causal test for unusually large raw semantic or structural score heads.

This is deliberately separate from the D x J factorial programme. It asks whether the heads with
the largest *raw* ``S_sem`` or ``S_str`` are causally important, accepting that this estimand also
contains their high J. Controls prioritise layer and then clean routed-value throughput; when six
targets exhaust a layer, the control falls back to the nearest available layer and records the
offset. Scores are discovery-only (test graphs); impacts use held-out validation graphs.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from ..carriage import env
from ..carriage.env import log
from ..carriage.tasks import get_task
from . import attention_viz
from .channel_ablation import per_graph_loss_np
from .factorial_ablation import _clean_and_throughput, _graph_groups
from .model import GritHeadModel, SpecConfig
from .scores import score_model


CACHE_VERSION = 2
ATTENTION_CACHE_VERSION = 4
SPECIALIST_GALLERY_CACHE_VERSION = 1


def score_fingerprint(scores: dict, score_key: str = "S_sem") -> str:
    a = np.ascontiguousarray(np.asarray(scores[score_key], dtype=np.float64))
    h = hashlib.sha256()
    h.update(str(a.shape).encode("ascii")); h.update(a.tobytes())
    return h.hexdigest()


def specialist_score_fingerprint(scores: dict, gsem: float, gstr: float) -> str:
    """Fingerprint both score channels and the shared reference normalisation defining D_rel."""
    h = hashlib.sha256()
    for key in ("S_sem", "S_str"):
        a = np.ascontiguousarray(np.asarray(scores[key], dtype=np.float64))
        h.update(key.encode("ascii")); h.update(str(a.shape).encode("ascii")); h.update(a.tobytes())
    h.update(np.asarray([gsem, gstr], dtype=np.float64).tobytes())
    return h.hexdigest()


def select_drel_heads(scores: dict, gsem: float, gstr: float,
                      k: int = 3) -> dict[str, list[tuple[int, int]]]:
    """Top-k semantic (high D_rel) and structural (low D_rel) heads."""
    sem = np.asarray(scores["S_sem"], float) / max(float(gsem), 1e-12)
    structural = np.asarray(scores["S_str"], float) / max(float(gstr), 1e-12)
    drel = (sem - structural) / (sem + structural + 1e-9)
    L, H = drel.shape
    k = int(k)
    if not 1 <= k <= (L * H) // 2:
        raise ValueError(f"specialist heads/channel must lie in [1, {(L * H)//2}], got {k}")
    order = np.argsort(drel.reshape(-1), kind="stable")
    low = order[:k]
    high = order[::-1][:k]
    decode = lambda ids: [(int(i // H), int(i % H)) for i in ids]
    return {"semantic": decode(high), "structural": decode(low)}


def select_top_scores(scores: dict, k: int, score_key: str = "S_sem") -> list[tuple[int, int]]:
    values = np.asarray(scores[score_key], float)
    L, H = values.shape
    k = int(k)
    if not 1 <= k < L * H:
        raise ValueError(f"top_k must lie in [1, L*H-1]; got k={k}, L={L}, H={H}")
    order = np.argsort(values.reshape(-1), kind="stable")[::-1][:k]
    return [(int(i // H), int(i % H)) for i in order]


def select_top_semantic(scores: dict, k: int) -> list[tuple[int, int]]:
    """Backward-compatible name for the initial semantic-only beta."""
    return select_top_scores(scores, k, "S_sem")


def throughput_matched_null(top_heads, throughput: np.ndarray) -> list[tuple[int, int]]:
    """Greedy layer-first, nearest-clean-throughput control, excluding every target head."""
    tp = np.asarray(throughput, float)
    L, H = tp.shape
    excluded = set(map(tuple, top_heads))
    available = {(l, h) for l in range(L) for h in range(H)} - excluded
    controls = []
    for target in top_heads:
        l, h = map(int, target)
        candidates = list(available)
        if not candidates:
            raise RuntimeError("no disjoint head remains for the outlier control")
        scale = np.log1p(max(float(tp[l, h]), 0.0))
        chosen = min(
            candidates,
            key=lambda x: (abs(x[0] - l),
                           abs(np.log1p(max(float(tp[x]), 0.0)) - scale), x[0], x[1]),
        )
        controls.append(chosen); available.remove(chosen)
    return controls


def _random_layer_null(rng, reference, *, L: int, H: int, excluded) -> list[tuple[int, int]]:
    available = {(l, h) for l in range(L) for h in range(H)} - set(map(tuple, excluded))
    chosen = []
    for layer, _ in reference:
        candidates = sorted(available, key=lambda x: (abs(x[0] - int(layer)), x[0], x[1]))
        if not candidates:
            raise RuntimeError("cannot construct a disjoint random outlier null")
        nearest = abs(candidates[0][0] - int(layer))
        candidates = [x for x in candidates if abs(x[0] - int(layer)) == nearest]
        head = candidates[int(rng.integers(len(candidates)))]
        chosen.append(head); available.remove(head)
    return chosen


def _budgets(k: int) -> np.ndarray:
    return np.arange(1, int(k) + 1, dtype=np.int64)


def _impact(gm, groups, clean, clean_loss, y, heads):
    pred = gm.collect_preds_ablated(groups, list(heads)).reshape(clean.shape)
    return (np.linalg.norm(pred - clean, axis=1).astype(np.float32),
            (per_graph_loss_np(pred, y, gm.loss_fun) - clean_loss).astype(np.float32))


def run_outlier_ablation(gm, scores: dict, *, score_key: str = "S_sem",
                         channel_name: Optional[str] = None,
                         num_graphs: int = 256, top_k: int = 6,
                         random_sets: int = 24, seed: int = 2718,
                         batch_size: int = 64, reusable: Optional[dict] = None) -> dict:
    """Run nested/individual raw-score ablations and matched controls on validation graphs."""
    channel_name = channel_name or ("semantic" if score_key == "S_sem" else "structural")
    rng = np.random.default_rng(seed)
    if reusable is not None and all(k in reusable for k in
                                    ("graph_ids", "clean_pred", "clean_loss", "y",
                                     "throughput_graph")):
        graph_ids = np.asarray(reusable["graph_ids"], dtype=np.int64)
        clean = np.asarray(reusable["clean_pred"], dtype=np.float64)
        clean_loss = np.asarray(reusable["clean_loss"], dtype=np.float64)
        y = np.asarray(reusable["y"], dtype=np.float64)
        throughput_graph = np.asarray(reusable["throughput_graph"], dtype=np.float64)
        groups, y_check = _graph_groups(gm, graph_ids, batch_size)
        if y_check.shape != y.shape or not np.allclose(y_check, y):
            raise RuntimeError("reused family cache does not align with validation graph labels")
        log(f"[{channel_name}-outlier] reusing clean predictions/throughput for {len(graph_ids)} "
            "validation graphs from the family-ablation cache.")
    else:
        n = min(int(num_graphs), len(gm.eval_ds))
        graph_ids = np.sort(rng.choice(len(gm.eval_ds), size=n, replace=False))
        groups, y = _graph_groups(gm, graph_ids, batch_size)
        clean, throughput_graph = _clean_and_throughput(gm, groups)
        clean_loss = per_graph_loss_np(clean, y, gm.loss_fun)

    throughput = throughput_graph.mean(axis=0)
    top_heads = select_top_scores(scores, top_k, score_key)
    matched_heads = throughput_matched_null(top_heads, throughput)
    matched_layer_delta = np.asarray(
        [int(control[0]) - int(target[0]) for target, control in zip(top_heads, matched_heads)],
        dtype=np.int64)
    if np.any(matched_layer_delta):
        log(f"[{channel_name}-outlier] exact-layer controls exhausted; nearest-layer offsets="
            f"{matched_layer_delta.tolist()}")
    budgets = _budgets(len(top_heads))
    G, B, K = len(graph_ids), len(budgets), len(top_heads)
    target_func = np.zeros((B, G), np.float32); target_loss = np.zeros_like(target_func)
    matched_func = np.zeros_like(target_func); matched_loss = np.zeros_like(target_func)
    reverse_func = np.zeros_like(target_func); reverse_loss = np.zeros_like(target_func)
    individual_func = np.zeros((K, G), np.float32); individual_loss = np.zeros_like(individual_func)
    matched_individual_func = np.zeros_like(individual_func)
    matched_individual_loss = np.zeros_like(individual_func)
    R = max(0, int(random_sets))
    random_func = np.zeros((B, R, G), np.float32); random_loss = np.zeros_like(random_func)
    t0 = time.perf_counter()

    for b, budget in enumerate(budgets):
        target_func[b], target_loss[b] = _impact(
            gm, groups, clean, clean_loss, y, top_heads[:int(budget)])
        matched_func[b], matched_loss[b] = _impact(
            gm, groups, clean, clean_loss, y, matched_heads[:int(budget)])
        reverse_func[b], reverse_loss[b] = _impact(
            gm, groups, clean, clean_loss, y, list(reversed(top_heads))[:int(budget)])
        for r in range(R):
            null = _random_layer_null(
                rng, top_heads[:int(budget)], L=gm.L, H=gm.H, excluded=top_heads)
            random_func[b, r], random_loss[b, r] = _impact(
                gm, groups, clean, clean_loss, y, null)
        log(f"[{channel_name}-outlier] cumulative k={budget}/{K} done "
            f"[{time.perf_counter()-t0:.1f}s]")

    # k=1 for the first head is already available above; evaluate each remaining head alone.
    individual_func[0], individual_loss[0] = target_func[0], target_loss[0]
    matched_individual_func[0], matched_individual_loss[0] = matched_func[0], matched_loss[0]
    for k in range(1, K):
        individual_func[k], individual_loss[k] = _impact(
            gm, groups, clean, clean_loss, y, [top_heads[k]])
        matched_individual_func[k], matched_individual_loss[k] = _impact(
            gm, groups, clean, clean_loss, y, [matched_heads[k]])

    return {
        "budgets": budgets, "top_heads": np.asarray(top_heads, dtype=np.int64),
        "matched_heads": np.asarray(matched_heads, dtype=np.int64),
        "matched_layer_delta": matched_layer_delta,
        "top_scores": np.asarray([scores[score_key][h] for h in top_heads], dtype=np.float64),
        "score_key": np.asarray(score_key), "channel_name": np.asarray(channel_name),
        "target_func": target_func, "target_loss": target_loss,
        "matched_func": matched_func, "matched_loss": matched_loss,
        "reverse_func": reverse_func, "reverse_loss": reverse_loss,
        "individual_func": individual_func, "individual_loss": individual_loss,
        "matched_individual_func": matched_individual_func,
        "matched_individual_loss": matched_individual_loss,
        "random_func": random_func, "random_loss": random_loss,
        "clean_pred": clean, "clean_loss": clean_loss, "y": y,
        "graph_ids": graph_ids, "throughput_graph": throughput_graph,
    }


class _PerGraphChannelCollector:
    """Retain established semantic/structural score estimates per candidate molecule."""

    def __init__(self):
        self.scores: dict[str, dict[int, np.ndarray]] = {"semantic": {}, "structural": {}}

    def __call__(self, *, channel, graph_id, source_nodes, phi_stack,
                 donor_averaged_delta, **_):
        if channel not in self.scores:
            return
        import torch

        layers = []
        for phi, delta in zip(phi_stack, donor_averaged_delta):
            projected = torch.einsum("tnhd,snhd->tsnh", phi, delta)
            functional = projected.square().sum(dim=0).sqrt()  # [source, carrier, head]
            layers.append(functional.sum(dim=(0, 1)).detach().cpu().numpy())
        self.scores[channel][int(graph_id)] = np.stack(layers) / max(len(source_nodes), 1)


# Backward-compatible private alias used by the semantic-only selector/tests.
_PerGraphSemanticCollector = _PerGraphChannelCollector


def _small_attention_candidates(gm, graph_ids, *, max_nodes: int,
                                candidate_graphs: int, seed: int) -> list[int]:
    """Fixed validation pool satisfying the readability constraint, before score ranking."""
    ids = np.asarray(graph_ids, dtype=np.int64)
    eligible = [int(i) for i in ids if int(gm.eval_ds[int(i)].num_nodes) <= int(max_nodes)]
    if len(eligible) < 4:
        raise RuntimeError(
            f"only {len(eligible)} validation molecules have <= {max_nodes} nodes; "
            "increase semantic_outlier_attention_max_nodes")
    if int(candidate_graphs) < 4:
        raise ValueError("semantic_outlier_attention_candidates must be at least four")
    rng = np.random.default_rng(seed)
    if len(eligible) > int(candidate_graphs):
        eligible = sorted(rng.choice(
            eligible, size=int(candidate_graphs), replace=False).astype(int).tolist())
    else:
        eligible = sorted(eligible)
    return eligible


def _ranked_attention(gm, sc: SpecConfig, result: dict, *, attention_graphs: int,
                      attention_heads: int, max_nodes: int, candidate_graphs: int,
                      score_donors: int, analysis_seed: int) -> dict:
    """Estimate Method A on small molecules, then collect each head's highest-scoring examples."""
    candidates = _small_attention_candidates(
        gm, result["graph_ids"], max_nodes=max_nodes,
        candidate_graphs=candidate_graphs, seed=analysis_seed)
    collector = _PerGraphChannelCollector()
    score_sc = replace(sc, num_graphs=len(candidates), donors=int(score_donors), resume=False)
    score_model(
        gm.task, score_sc, with_attn_routing=False, seed=analysis_seed,
        response_observer=collector, graph_ids_override=candidates,
        channels=("semantic",), loaded_gm=gm)
    if set(collector.scores["semantic"]) != set(candidates):
        raise RuntimeError("per-molecule semantic scores are incomplete")

    n_heads = min(int(attention_heads), len(result["top_heads"]))
    heads = [tuple(map(int, x)) for x in np.asarray(result["top_heads"], int)[:n_heads]]
    n_examples = min(int(attention_graphs), len(candidates))
    selected = {}
    selected_union = set()
    for head in heads:
        ranked = sorted(
            candidates,
            key=lambda gid: (-float(collector.scores["semantic"][gid][head]),
                             int(gm.eval_ds[gid].num_nodes), gid))
        gids = ranked[:n_examples]
        selected[head] = {
            "graph_ids": gids,
            "scores": [float(collector.scores["semantic"][gid][head]) for gid in gids],
            "candidate_ranks": list(range(1, n_examples + 1)),
            "candidate_count": len(candidates),
        }
        selected_union.update(gids)
        log(f"[semantic-attention] L{head[0]}H{head[1]} top small-molecule examples: "
            + ", ".join(
                f"id={gid}/n={int(gm.eval_ds[gid].num_nodes)}/"
                f"S={collector.scores['semantic'][gid][head]:.3g}"
                for gid in gids))

    attn = attention_viz.collect_attention(
        gm, sorted(selected_union), heads, seed=analysis_seed)
    attn["selected"] = selected
    attn["selection_config"] = {
        "max_nodes": int(max_nodes), "candidate_graphs": int(candidate_graphs),
        "score_donors": int(score_donors), "examples_per_head": int(attention_graphs),
        "analysis_seed": int(analysis_seed),
    }
    return attn


def prepare_specialist_gallery(task_name: str, scores: dict, *, gsem: float, gstr: float,
                               collate_dir: str, ckpt: Optional[str] = None,
                               heads_per_channel: int = 3, examples_per_head: int = 4,
                               max_nodes: int = 18, candidate_graphs: int = 32,
                               score_donors: int = 32, analysis_seed: int = 2718,
                               eval_split: str = "val", seed: int = 42,
                               accelerator: str = "cuda:0", num_threads: int = 4) -> dict:
    """Build the all-model high/low-D_rel attention cache without rerunning global scores."""
    gm, sc = _load_model(
        task_name, collate_dir=collate_dir, ckpt=ckpt, num_graphs=candidate_graphs,
        analysis_seed=analysis_seed, eval_split=eval_split, seed=seed,
        accelerator=accelerator, num_threads=num_threads)
    try:
        candidates = _small_attention_candidates(
            gm, np.arange(len(gm.eval_ds)), max_nodes=max_nodes,
            candidate_graphs=candidate_graphs, seed=analysis_seed)
        chosen = select_drel_heads(scores, gsem, gstr, heads_per_channel)
        heads = chosen["semantic"] + chosen["structural"]
        head_channels = (["semantic"] * len(chosen["semantic"])
                         + ["structural"] * len(chosen["structural"]))
        collector = _PerGraphChannelCollector()
        score_sc = replace(sc, num_graphs=len(candidates), donors=int(score_donors), resume=False)
        score_model(
            gm.task, score_sc, with_attn_routing=False, seed=analysis_seed,
            response_observer=collector, graph_ids_override=candidates,
            channels=("semantic", "structural"), loaded_gm=gm)
        for channel in ("semantic", "structural"):
            if set(collector.scores[channel]) != set(candidates):
                raise RuntimeError(f"per-molecule {channel} scores are incomplete")

        n_examples = min(int(examples_per_head), len(candidates))
        selected, union = {}, set()
        for head, channel in zip(heads, head_channels):
            ranked = sorted(
                candidates,
                key=lambda gid: (-float(collector.scores[channel][gid][head]),
                                 int(gm.eval_ds[gid].num_nodes), gid))
            gids = ranked[:n_examples]
            selected[head] = {
                "graph_ids": gids,
                "scores": [float(collector.scores[channel][gid][head]) for gid in gids],
                "candidate_ranks": list(range(1, n_examples + 1)),
                "candidate_count": len(candidates),
            }
            union.update(gids)
            log(f"[specialist-gallery] {task_name} {channel} L{head[0]}H{head[1]}: "
                + ", ".join(
                    f"id={gid}/n={int(gm.eval_ds[gid].num_nodes)}/"
                    f"S={collector.scores[channel][gid][head]:.3g}" for gid in gids))

        attn = attention_viz.collect_attention(gm, sorted(union), heads, seed=analysis_seed)
        attn["selected"] = selected
        attn["head_channels"] = head_channels
        attn["selection_config"] = {
            "max_nodes": int(max_nodes), "candidate_graphs": int(candidate_graphs),
            "score_donors": int(score_donors), "examples_per_head": int(examples_per_head),
            "heads_per_channel": int(heads_per_channel), "analysis_seed": int(analysis_seed),
            "eval_split": str(eval_split),
            "heads": [list(h) for h in heads], "head_channels": head_channels,
        }
        return attn
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def save_attention(attn: dict, path, *, score_hash: str = "",
                   cache_version: int = ATTENTION_CACHE_VERSION) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cache_version": np.asarray(cache_version, dtype=np.int64),
               "score_fingerprint": np.asarray(str(score_hash)),
               "heads": np.asarray(attn["heads"], dtype=np.int64),
               "num_molecules": np.asarray(len(attn["molecules"]), dtype=np.int64),
               "has_vnode": np.asarray(bool(attn.get("has_vnode", False))),
               "head_channels": np.asarray(attn.get(
                   "head_channels", ["semantic"] * len(attn["heads"]))),
               "selection_config": np.asarray(json.dumps(
                   attn.get("selection_config", {}), sort_keys=True))}
    for g, mol in enumerate(attn["molecules"]):
        payload[f"graph_id_{g}"] = np.asarray(mol["graph_id"], dtype=np.int64)
        for key in ("atom_types", "bonds", "pos"):
            payload[f"{key}_{g}"] = np.asarray(mol[key])
        payload[f"bond_types_{g}"] = np.asarray(
            mol.get("bond_types", np.ones(len(mol["bonds"]), dtype=np.int64)))
        for h, head in enumerate(attn["heads"]):
            payload[f"map_{g}_{h}"] = np.asarray(mol["maps"][tuple(head)], dtype=np.float32)
    selected = attn.get("selected", {})
    for h, head in enumerate(attn["heads"]):
        item = selected.get(tuple(head), {})
        payload[f"selected_graph_ids_{h}"] = np.asarray(
            item.get("graph_ids", []), dtype=np.int64)
        payload[f"selected_scores_{h}"] = np.asarray(item.get("scores", []), dtype=np.float64)
        payload[f"selected_candidate_ranks_{h}"] = np.asarray(
            item.get("candidate_ranks", []), dtype=np.int64)
        payload[f"selected_candidate_count_{h}"] = np.asarray(
            item.get("candidate_count", 0), dtype=np.int64)
    np.savez_compressed(path, **payload)
    return str(path)


def save_result(result: dict, npz_path, summary_path, *, task: str,
                score_hash: str, config: dict) -> tuple[str, str]:
    npz_path, summary_path = Path(npz_path), Path(summary_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, cache_version=np.asarray(CACHE_VERSION),
                        score_fingerprint=np.asarray(score_hash),
                        **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    summary = {
        "cache_version": CACHE_VERSION, "task": task, "score_fingerprint": score_hash,
        "score_key": str(np.asarray(result["score_key"]).item()),
        "channel_name": str(np.asarray(result["channel_name"]).item()),
        "config": config, "top_heads": result["top_heads"].tolist(),
        "matched_heads": result["matched_heads"].tolist(),
        "matched_layer_delta": result["matched_layer_delta"].tolist(),
        "top_scores": result["top_scores"].tolist(),
        "individual_loss_delta_mean": np.asarray(
            result["individual_loss"], float).mean(axis=1).tolist(),
        "matched_individual_loss_delta_mean": np.asarray(
            result["matched_individual_loss"], float).mean(axis=1).tolist(),
        "clean_loss_mean": float(np.mean(result["clean_loss"])),
        "top1_loss_delta": float(np.mean(result["individual_loss"][0])),
        "topK_loss_delta": float(np.mean(result["target_loss"][-1])),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(npz_path), str(summary_path)


def _load_model(task_name: str, *, collate_dir: str, ckpt: Optional[str], num_graphs: int,
                analysis_seed: int, eval_split: str, seed: int, accelerator: str,
                num_threads: int):
    spec = get_task(task_name)
    task_out = Path(collate_dir) / task_name
    repo_dir = Path(spec.grit_repo_dir or (f"/content/GRIT_{spec.name}" if spec.env_hooks
                                           else "/content/GRIT"))
    env.clone_grit(repo_dir, spec.grit_repo, spec.grit_commit, force_fresh=False)
    for hook in spec.env_hooks:
        hook(repo_dir)
    env.prepare_inprocess_grit(repo_dir)
    config_file = env.resolve_config(spec, repo_dir, task_out)
    chosen_ckpt, _ = env.find_checkpoint(Path(spec.drive_dir) / "results", ckpt)
    log(f"[semantic-outlier] runtime: {platform.platform()} | python {sys.version.split()[0]}")
    sc = SpecConfig(
        ckpt=str(chosen_ckpt), out_dir=str(task_out),
        dataset_dir=str(Path(spec.drive_dir) / "datasets"), config_file=config_file,
        accelerator=accelerator, seed=seed, num_threads=num_threads,
        eval_split=eval_split, donor_split=eval_split, eval_metric=False,
        num_graphs=0, donors=0, ablation_graphs=num_graphs, analysis_seed=analysis_seed)
    return GritHeadModel(spec, sc).load(), sc


def prepare_and_run(task_name: str, scores: dict, *, collate_dir: str,
                    ckpt: Optional[str] = None, score_key: str = "S_sem",
                    channel_name: Optional[str] = None,
                    num_graphs: int = 256, top_k: int = 6,
                    random_sets: int = 24, analysis_seed: int = 2718,
                    eval_split: str = "val", batch_size: int = 64,
                    attention_graphs: int = 0, attention_heads: int = 2,
                    attention_max_nodes: int = 18, attention_candidates: int = 32,
                    attention_score_donors: int = 32,
                    reusable: Optional[dict] = None, seed: int = 42,
                    accelerator: str = "cuda:0", num_threads: int = 4):
    """Load one checkpoint and compute only this new cached-score follow-up."""
    gm, sc = _load_model(
        task_name, collate_dir=collate_dir, ckpt=ckpt, num_graphs=num_graphs,
        analysis_seed=analysis_seed, eval_split=eval_split, seed=seed,
        accelerator=accelerator, num_threads=num_threads)
    try:
        result = run_outlier_ablation(
            gm, scores, score_key=score_key, channel_name=channel_name,
            num_graphs=num_graphs, top_k=top_k, random_sets=random_sets,
            seed=analysis_seed, batch_size=batch_size, reusable=reusable)
        attn = None
        if int(attention_graphs) > 0 and "vnode" not in task_name.lower():
            attn = _ranked_attention(
                gm, sc, result, attention_graphs=attention_graphs,
                attention_heads=attention_heads, max_nodes=attention_max_nodes,
                candidate_graphs=attention_candidates,
                score_donors=attention_score_donors, analysis_seed=analysis_seed)
        return result, attn
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def prepare_attention_only(task_name: str, result: dict, *, collate_dir: str,
                           ckpt: Optional[str] = None, attention_graphs: int = 3,
                           attention_heads: int = 2, analysis_seed: int = 2718,
                           attention_max_nodes: int = 18, attention_candidates: int = 32,
                           attention_score_donors: int = 32,
                           eval_split: str = "val", seed: int = 42,
                           accelerator: str = "cuda:0", num_threads: int = 4):
    """Collect the descriptive panel without repeating any ablation/score/carriage forwards."""
    gm, sc = _load_model(
        task_name, collate_dir=collate_dir, ckpt=ckpt,
        num_graphs=len(np.asarray(result["graph_ids"])), analysis_seed=analysis_seed,
        eval_split=eval_split, seed=seed, accelerator=accelerator, num_threads=num_threads)
    try:
        return _ranked_attention(
            gm, sc, result, attention_graphs=attention_graphs,
            attention_heads=attention_heads, max_nodes=attention_max_nodes,
            candidate_graphs=attention_candidates,
            score_donors=attention_score_donors, analysis_seed=analysis_seed)
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
