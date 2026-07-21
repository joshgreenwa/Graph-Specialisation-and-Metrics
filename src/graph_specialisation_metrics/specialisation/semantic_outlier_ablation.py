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


CACHE_VERSION = 2
ATTENTION_CACHE_VERSION = 3


def score_fingerprint(scores: dict, score_key: str = "S_sem") -> str:
    a = np.ascontiguousarray(np.asarray(scores[score_key], dtype=np.float64))
    h = hashlib.sha256()
    h.update(str(a.shape).encode("ascii")); h.update(a.tobytes())
    return h.hexdigest()


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


def _attention_graph_ids(gm, graph_ids, n_graphs: int) -> list[int]:
    ids = np.asarray(graph_ids, dtype=np.int64)
    sizes = np.asarray([int(gm.eval_ds[int(i)].num_nodes) for i in ids])
    order = np.argsort(sizes, kind="stable")
    picks = np.linspace(0, len(order) - 1, min(int(n_graphs), len(order))).round().astype(int)
    return [int(ids[order[p]]) for p in picks]


def save_attention(attn: dict, path, *, score_hash: str = "") -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cache_version": np.asarray(ATTENTION_CACHE_VERSION, dtype=np.int64),
               "score_fingerprint": np.asarray(str(score_hash)),
               "heads": np.asarray(attn["heads"], dtype=np.int64),
               "num_molecules": np.asarray(len(attn["molecules"]), dtype=np.int64)}
    for g, mol in enumerate(attn["molecules"]):
        payload[f"graph_id_{g}"] = np.asarray(mol["graph_id"], dtype=np.int64)
        for key in ("atom_types", "bonds", "pos"):
            payload[f"{key}_{g}"] = np.asarray(mol[key])
        payload[f"bond_types_{g}"] = np.asarray(
            mol.get("bond_types", np.ones(len(mol["bonds"]), dtype=np.int64)))
        for h, head in enumerate(attn["heads"]):
            payload[f"map_{g}_{h}"] = np.asarray(mol["maps"][tuple(head)], dtype=np.float32)
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
    return GritHeadModel(spec, sc).load()


def prepare_and_run(task_name: str, scores: dict, *, collate_dir: str,
                    ckpt: Optional[str] = None, score_key: str = "S_sem",
                    channel_name: Optional[str] = None,
                    num_graphs: int = 256, top_k: int = 6,
                    random_sets: int = 24, analysis_seed: int = 2718,
                    eval_split: str = "val", batch_size: int = 64,
                    attention_graphs: int = 0, attention_heads: int = 2,
                    reusable: Optional[dict] = None, seed: int = 42,
                    accelerator: str = "cuda:0", num_threads: int = 4):
    """Load one checkpoint and compute only this new cached-score follow-up."""
    gm = _load_model(
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
            ids = _attention_graph_ids(gm, result["graph_ids"], attention_graphs)
            n = min(int(attention_heads), len(result["top_heads"]))
            heads = ([tuple(x) for x in result["top_heads"][:n]]
                     + [tuple(x) for x in result["matched_heads"][:n]])
            attn = attention_viz.collect_attention(gm, ids, heads, seed=analysis_seed)
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
                           eval_split: str = "val", seed: int = 42,
                           accelerator: str = "cuda:0", num_threads: int = 4):
    """Collect the descriptive panel without repeating any ablation/score/carriage forwards."""
    gm = _load_model(
        task_name, collate_dir=collate_dir, ckpt=ckpt,
        num_graphs=len(np.asarray(result["graph_ids"])), analysis_seed=analysis_seed,
        eval_split=eval_split, seed=seed, accelerator=accelerator, num_threads=num_threads)
    try:
        ids = _attention_graph_ids(gm, result["graph_ids"], attention_graphs)
        n = min(int(attention_heads), len(result["top_heads"]))
        heads = ([tuple(x) for x in result["top_heads"][:n]]
                 + [tuple(x) for x in result["matched_heads"][:n]])
        return attention_viz.collect_attention(gm, ids, heads, seed=analysis_seed)
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
