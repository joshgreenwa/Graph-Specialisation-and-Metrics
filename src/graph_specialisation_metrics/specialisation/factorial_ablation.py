"""Matched family ablations on the specialisation selectivity-strength plane.

This is a causal *follow-up* to the cached separate-intervention scores.  It does not estimate
new carriage or new scores.  Instead, cached ``S_sem``/``S_str`` define, within each model, six
disjoint head families on the established coordinates

    D_rel = (S~_sem - S~_str) / (S~_sem + S~_str)    (channel preference)
    J     = (S~_sem + S~_str) / 2                    (transport strength),

namely relatively semantic / relatively structural / generalist crossed with high / low J.
Generalists are closest to D=0; after removing them, the highest-D half is semantic and the
lowest-D half structural. Thus labels are within-model ranks, not claims that D crosses zero.
Families within each J stratum are matched for layer, J, and clean pre-head throughput ||wV||, then ablated
cumulatively at the established routed-value site.  The active scientific null is the matched
generalist family; low-J generalists are the inactive null.  Layer-matched random sets are retained
only as a secondary reference band.

The score cache is discovery-only.  Clean throughput and ablation impacts are evaluated on the
validation split, which is disjoint from the test graphs used by the score estimator.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..carriage import env
from ..carriage.env import log
from ..carriage.tasks import GritTaskSpec, get_task
from .channel_ablation import per_graph_loss_np
from .model import GritHeadModel, SpecConfig


CACHE_VERSION = 2
PREFERENCES = ("semantic", "structural", "generalist")
STRENGTHS = ("highJ", "lowJ")
FAMILY_NAMES = tuple(f"{p}_{s}" for s in STRENGTHS for p in PREFERENCES)


class FactorialNotEstimable(RuntimeError):
    """The observed score geometry cannot instantiate the predeclared six-family design."""


def score_fingerprint(scores: dict) -> str:
    """Stable identity for the two cached score matrices that selected the families."""
    h = hashlib.sha256()
    for key in ("S_sem", "S_str"):
        a = np.ascontiguousarray(np.asarray(scores[key], dtype=np.float64))
        h.update(key.encode("ascii")); h.update(str(a.shape).encode("ascii")); h.update(a.tobytes())
    return h.hexdigest()


def dj_coordinates(scores: dict, gsem: float, gstr: float, eps: float = 1e-12):
    """Return the shared-reference-normalised ``(D_rel, J)`` matrices."""
    if not np.isfinite(gsem) or not np.isfinite(gstr) or gsem <= 0 or gstr <= 0:
        raise ValueError(f"global score norms must be positive, got gsem={gsem}, gstr={gstr}")
    sem = np.asarray(scores["S_sem"], float) / float(gsem)
    stru = np.asarray(scores["S_str"], float) / float(gstr)
    if sem.shape != stru.shape or sem.ndim != 2:
        raise ValueError(f"S_sem/S_str must be matching [L,H] matrices, got {sem.shape}/{stru.shape}")
    total = sem + stru
    return (sem - stru) / (total + eps), 0.5 * total


def _rank01(values: np.ndarray) -> np.ndarray:
    """Deterministic average-rank transform in [0,1], without a scipy dependency."""
    x = np.asarray(values, float).reshape(-1)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks / max(len(x) - 1, 1)


def _candidate_cells(D: np.ndarray, J: np.ndarray, *, generalist_fraction: float,
                     activity_floor_quantile: float) -> tuple[dict[str, np.ndarray], dict]:
    """Partition heads into six disjoint candidate cells using score coordinates only.

    The closest ``generalist_fraction`` of heads to D=0 are generalists. The remaining heads are
    split by within-model D rank: the highest-D half is relatively semantic and the lowest-D half
    relatively structural. This guarantees comparative specialist pools even when every head is
    on the same side of D=0. Each preference pool is split at its own median J; low-J specialists
    below a model-wide activity floor are dropped because relative D there is ratio-noise prone.
    """
    d, j = np.asarray(D, float).reshape(-1), np.asarray(J, float).reshape(-1)
    n = len(d)
    if not (0.1 <= generalist_fraction <= 0.6):
        raise ValueError("generalist_fraction must be between 0.1 and 0.6")
    n_gen = max(2, min(n - 2, int(round(float(generalist_fraction) * n))))
    gen = np.argsort(np.abs(d), kind="stable")[:n_gen]
    is_gen = np.zeros(n, bool); is_gen[gen] = True
    remaining = np.flatnonzero(~is_gen)
    ranked = remaining[np.argsort(d[remaining], kind="stable")]
    split = len(ranked) // 2
    pools = {
        "generalist": gen,
        "structural": ranked[:split],
        "semantic": ranked[split:],
    }
    floor = float(np.quantile(j, activity_floor_quantile))
    cells: dict[str, np.ndarray] = {}
    medians = {}
    for pref, idx in pools.items():
        if len(idx) < 2:
            raise FactorialNotEstimable(
                f"factorial family selection has only {len(idx)} {pref} candidates; the model "
                "does not support the relative semantic/structural/generalist comparison.")
        med = float(np.median(j[idx])); medians[pref] = med
        hi = idx[j[idx] >= med]
        lo = idx[j[idx] < med]
        if pref != "generalist":
            lo = lo[j[lo] >= floor]
        cells[f"{pref}_highJ"] = np.asarray(hi, dtype=np.int64)
        cells[f"{pref}_lowJ"] = np.asarray(lo, dtype=np.int64)
    return cells, {"generalist_count": int(n_gen), "J_activity_floor": floor,
                   "within_preference_J_medians": medians}


def check_factorial_estimable(scores: dict, *, gsem: float, gstr: float,
                              family_size: int = 6, generalist_fraction: float = 0.30,
                              activity_floor_quantile: float = 0.10) -> dict:
    """Cheap score-only preflight; raises before loading a model when the design is absent."""
    D, J = dj_coordinates(scores, gsem, gstr)
    cells, thresholds = _candidate_cells(
        D, J, generalist_fraction=generalist_fraction,
        activity_floor_quantile=activity_floor_quantile)
    sizes = {name: int(len(values)) for name, values in cells.items()}
    available = min(sizes.values())
    if min(int(family_size), available) < 2:
        raise FactorialNotEstimable(
            f"insufficient heads for six factorial families; cell sizes={sizes}")
    return {"cell_sizes": sizes, "available_family_size": int(available),
            "thresholds": thresholds}


def _priority(idx: np.ndarray, pref: str, strength: str, D: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Predeclared score-only priority used to order matched triplets cumulatively."""
    d, j = D.reshape(-1)[idx], J.reshape(-1)[idx]
    if pref == "generalist":
        pref_strength = -np.abs(d)
    elif pref == "semantic":
        pref_strength = d
    else:
        pref_strength = -d
    j_strength = (j if strength == "highJ" else -j)
    return 0.65 * _rank01(pref_strength) + 0.35 * _rank01(j_strength)


def _matched_triplets(cells: dict[str, np.ndarray], D: np.ndarray, J: np.ndarray,
                      throughput: np.ndarray, *, strength: str, family_size: int,
                      L: int, H: int) -> tuple[dict[str, list[tuple[int, int]]], list[dict]]:
    """Greedily choose disjoint semantic/structural/generalist triplets with matched nuisances."""
    flat_t = np.log1p(np.maximum(np.asarray(throughput, float).reshape(-1), 0.0))
    t_rank, j_rank = _rank01(flat_t), _rank01(np.asarray(J, float).reshape(-1))
    pools = {p: list(map(int, cells[f"{p}_{strength}"])) for p in PREFERENCES}
    k = min(int(family_size), *(len(v) for v in pools.values()))
    if k < 2:
        sizes = {p: len(v) for p, v in pools.items()}
        raise RuntimeError(f"fewer than two matched {strength} triplets are available: {sizes}")

    priority = {p: {int(i): float(v) for i, v in zip(
        pools[p], _priority(np.asarray(pools[p]), p, strength, D, J))}
        for p in PREFERENCES}
    selected = []
    # A strong layer penalty makes exact layer matching preferred; J and clean throughput are
    # matched within the high/low stratum.  The weak priority term prevents nuisance matching from
    # selecting only marginal members of a score-defined family.
    for _ in range(k):
        best = None
        for sem, stru, gen in itertools.product(
                pools["semantic"], pools["structural"], pools["generalist"]):
            ids = (sem, stru, gen)
            layers = np.asarray([i // H for i in ids], float)
            layer_cost = (layers.max() - layers.min()) / max(L - 1, 1)
            throughput_cost = np.ptp(t_rank[list(ids)])
            strength_cost = np.ptp(j_rank[list(ids)])
            evidence = np.mean([priority[p][i] for p, i in zip(PREFERENCES, ids)])
            cost = 8.0 * layer_cost + 1.5 * throughput_cost + strength_cost - 0.15 * evidence
            item = (float(cost), ids, float(evidence))
            if best is None or item[0] < best[0]:
                best = item
        _, ids, evidence = best
        selected.append((ids, evidence))
        for p, i in zip(PREFERENCES, ids):
            pools[p].remove(i)

    # Cumulative ablation starts with the strongest score-defined matched triplet, not the triplet
    # that merely happened to be easiest to match.
    selected.sort(key=lambda x: x[1], reverse=True)
    families = {p: [(int(ids[q]) // H, int(ids[q]) % H) for ids, _ in selected]
                for q, p in enumerate(PREFERENCES)}
    matching = []
    for ids, evidence in selected:
        matching.append({
            "heads": {p: [int(i // H), int(i % H)] for p, i in zip(PREFERENCES, ids)},
            "layer_span": int(max(i // H for i in ids) - min(i // H for i in ids)),
            "throughput_rank_span": float(np.ptp(t_rank[list(ids)])),
            "J_rank_span": float(np.ptp(j_rank[list(ids)])),
            "score_priority": float(evidence),
        })
    return families, matching


def select_factorial_families(scores: dict, throughput: np.ndarray, *, gsem: float, gstr: float,
                              family_size: int = 6, generalist_fraction: float = 0.30,
                              activity_floor_quantile: float = 0.10) -> dict:
    """Select six equal-sized, disjoint, approximately nuisance-matched head families."""
    D, J = dj_coordinates(scores, gsem, gstr)
    L, H = D.shape
    tp = np.asarray(throughput, float)
    if tp.shape != (L, H):
        raise ValueError(f"throughput must have shape {(L, H)}, got {tp.shape}")
    cells, thresholds = _candidate_cells(
        D, J, generalist_fraction=generalist_fraction,
        activity_floor_quantile=activity_floor_quantile)
    available = min(len(v) for v in cells.values())
    k = min(int(family_size), int(available))
    if k < 2:
        raise FactorialNotEstimable(
            f"insufficient heads for six factorial families; cell sizes="
            f"{ {name: len(v) for name, v in cells.items()} }")
    if k < int(family_size):
        log(f"[family-selection:WARN] requested K={family_size}, but the smallest relative "
            f"D x J cell supports K={k}; using equal K={k}. "
            f"cell sizes={ {name: len(v) for name, v in cells.items()} }")

    families, matching = {}, {}
    for strength in STRENGTHS:
        fam, match = _matched_triplets(
            cells, D, J, tp, strength=strength, family_size=k, L=L, H=H)
        for pref in PREFERENCES:
            families[f"{pref}_{strength}"] = fam[pref]
        matching[strength] = match

    # Equal size and disjointness are scientific invariants, not plotting conveniences.
    all_heads = [tuple(h) for name in FAMILY_NAMES for h in families[name]]
    if len({h for h in all_heads}) != len(all_heads):
        raise RuntimeError("factorial head families overlap")
    if len({len(families[name]) for name in FAMILY_NAMES}) != 1:
        raise RuntimeError("factorial head families differ in size")

    flat_D, flat_J, flat_T = D.reshape(-1), J.reshape(-1), tp.reshape(-1)
    diagnostics = {}
    for name in FAMILY_NAMES:
        idx = np.asarray([l * H + h for l, h in families[name]], dtype=int)
        diagnostics[name] = {
            "n": int(len(idx)), "heads": [list(map(int, h)) for h in families[name]],
            "mean_D": float(flat_D[idx].mean()), "mean_abs_D": float(np.abs(flat_D[idx]).mean()),
            "mean_J": float(flat_J[idx].mean()),
            "mean_clean_throughput": float(flat_T[idx].mean()),
            "layers": [int(i // H) for i in idx],
        }
    return {"families": families, "matching": matching, "diagnostics": diagnostics,
            "thresholds": thresholds, "cell_sizes": {k: int(len(v)) for k, v in cells.items()},
            "D": D, "J": J, "family_size": int(k)}


def _graph_groups(gm, graph_ids: Sequence[int], batch_size: int):
    groups, ys = [], []
    buf = []
    for gi in graph_ids:
        data = gm.eval_ds[int(gi)]
        buf.append(data)
        ys.append(data.y.reshape(-1).cpu().numpy().astype(np.float64))
        if len(buf) == int(batch_size):
            groups.append(buf); buf = []
    if buf:
        groups.append(buf)
    return groups, np.stack(ys)


def _clean_and_throughput(gm, groups):
    """One clean pass per group, retaining graphwise mean ||wV|| for every head."""
    import torch
    from torch_geometric.data import Batch

    has_vnode = getattr(getattr(gm.model, "model", gm.model), "global_vnode", None) is not None
    preds, throughput = [], []
    with torch.no_grad():
        for group in groups:
            batch = Batch.from_data_list(list(group)).to(gm.device)
            cap = gm.capture(batch, want_grad=False, want_attn=False,
                             include_virtual_transport=has_vnode)
            preds.append(cap["pred"].detach().cpu().numpy().reshape(len(group), -1))
            graph = cap["node_graph"]
            per_graph = []
            for local_g in range(len(group)):
                mask = graph == local_g
                if int(mask.sum()) == 0:
                    raise RuntimeError("clean throughput capture lost a graph's carrier rows")
                per_graph.append(torch.stack([
                    w[mask].norm(dim=-1).mean(dim=0) for w in cap["wV"]
                ]).detach().cpu().numpy())
            throughput.append(np.stack(per_graph))
    return np.concatenate(preds), np.concatenate(throughput)


def _budgets(k: int) -> np.ndarray:
    vals = {1, int(k)}
    vals.update(max(1, int(round(k * q))) for q in (0.25, 0.5, 0.75))
    return np.asarray(sorted(vals), dtype=np.int64)


def _random_layer_matched_heads(rng, reference: Sequence[tuple[int, int]], *, L: int, H: int):
    """Random heads with the exact layer multiset of ``reference`` (secondary null only)."""
    chosen, used = [], set()
    for layer, _ in reference:
        options = [(int(layer), h) for h in range(H) if (int(layer), h) not in used]
        if not options:
            raise RuntimeError("cannot construct a without-replacement layer-matched random set")
        head = options[int(rng.integers(len(options)))]
        chosen.append(head); used.add(head)
    return chosen


def run_family_ablation(gm, scores: dict, *, gsem: float, gstr: float,
                        num_graphs: int = 256, family_size: int = 6,
                        generalist_fraction: float = 0.30,
                        activity_floor_quantile: float = 0.10,
                        random_sets: int = 24, seed: int = 2718,
                        batch_size: int = 64) -> dict:
    """Evaluate cached-score-selected family ablations on ``gm.eval_ds`` (normally validation)."""
    rng = np.random.default_rng(seed)
    n = min(int(num_graphs), len(gm.eval_ds))
    graph_ids = np.sort(rng.choice(len(gm.eval_ds), size=n, replace=False))
    groups, y = _graph_groups(gm, graph_ids, batch_size)
    t0 = time.perf_counter()
    clean, throughput_graph = _clean_and_throughput(gm, groups)
    clean_loss = per_graph_loss_np(clean, y, gm.loss_fun)
    throughput = throughput_graph.mean(axis=0)
    selection = select_factorial_families(
        scores, throughput, gsem=gsem, gstr=gstr, family_size=family_size,
        generalist_fraction=generalist_fraction,
        activity_floor_quantile=activity_floor_quantile)
    log("[family-selection] matched groups (D preference | J strength | clean ||wV||):")
    for name in FAMILY_NAMES:
        d = selection["diagnostics"][name]
        log(f"  {name:<18} K={d['n']} D={d['mean_D']:+.3f} J={d['mean_J']:.3f} "
            f"throughput={d['mean_clean_throughput']:.3e}")
    K = int(selection["family_size"])
    budgets = _budgets(K)
    G, B = len(graph_ids), len(budgets)
    functional = np.zeros((len(FAMILY_NAMES), B, G), dtype=np.float32)
    loss = np.zeros_like(functional)

    def _impact(heads):
        pred = gm.collect_preds_ablated(groups, heads).reshape(G, -1)
        return (np.linalg.norm(pred - clean, axis=1),
                per_graph_loss_np(pred, y, gm.loss_fun) - clean_loss)

    for f, name in enumerate(FAMILY_NAMES):
        heads = selection["families"][name]
        for b, budget in enumerate(budgets):
            functional[f, b], loss[f, b] = _impact(heads[:int(budget)])
        log(f"[family-ablation] {name:<18} K={K} done [{time.perf_counter()-t0:.1f}s]")

    # One random reference per strength stratum, layer-matched to the corresponding matched
    # generalist sequence. It is deliberately secondary to the high/low-J generalist controls.
    R = max(0, int(random_sets))
    random_functional = np.zeros((len(STRENGTHS), B, R, G), dtype=np.float32)
    random_loss = np.zeros_like(random_functional)
    for s, strength in enumerate(STRENGTHS):
        reference = selection["families"][f"generalist_{strength}"]
        for b, budget in enumerate(budgets):
            ref = reference[:int(budget)]
            for r in range(R):
                heads = _random_layer_matched_heads(rng, ref, L=gm.L, H=gm.H)
                random_functional[s, b, r], random_loss[s, b, r] = _impact(heads)
        log(f"[family-ablation] {strength} layer-matched random band R={R} done "
            f"[{time.perf_counter()-t0:.1f}s]")

    log(f"[family-ablation] complete: {G} held-out graphs; K={K}; budgets={budgets.tolist()}; "
        f"clean loss={clean_loss.mean():.4g}")
    return {
        "family_names": np.asarray(FAMILY_NAMES), "budgets": budgets,
        "functional": functional, "loss": loss,
        "random_functional": random_functional, "random_loss": random_loss,
        "clean_pred": clean, "clean_loss": clean_loss, "y": y,
        "graph_ids": graph_ids, "throughput_graph": throughput_graph,
        "D": selection["D"], "J": selection["J"],
        "heads": np.asarray([selection["families"][n] for n in FAMILY_NAMES], dtype=np.int64),
        "selection": selection,
    }


def save_family_ablation(result: dict, npz_path, summary_path, *, task: str,
                         score_hash: str, config: dict) -> tuple[str, str]:
    npz_path, summary_path = Path(npz_path), Path(summary_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path, cache_version=np.asarray(CACHE_VERSION), score_fingerprint=np.asarray(score_hash),
        family_names=result["family_names"], budgets=result["budgets"],
        functional=result["functional"], loss=result["loss"],
        random_functional=result["random_functional"], random_loss=result["random_loss"],
        clean_pred=result["clean_pred"], clean_loss=result["clean_loss"], y=result["y"],
        graph_ids=result["graph_ids"], throughput_graph=result["throughput_graph"],
        D=result["D"], J=result["J"], heads=result["heads"])
    sel = result["selection"]
    summary = {
        "cache_version": CACHE_VERSION, "task": task, "score_fingerprint": score_hash,
        "config": config, "num_graphs": int(len(result["graph_ids"])),
        "family_size": int(sel["family_size"]), "budgets": result["budgets"].tolist(),
        "clean_loss_mean": float(np.mean(result["clean_loss"])),
        "selection": {k: sel[k] for k in
                      ("diagnostics", "matching", "thresholds", "cell_sizes")},
        "endpoint": {},
    }
    for f, name in enumerate(FAMILY_NAMES):
        summary["endpoint"][name] = {
            "functional_mean": float(result["functional"][f, -1].mean()),
            "loss_delta_mean": float(result["loss"][f, -1].mean()),
        }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(npz_path), str(summary_path)


def prepare_and_run(task_name: str, scores: dict, *, gsem: float, gstr: float,
                    collate_dir: str, ckpt: Optional[str] = None,
                    num_graphs: int = 256, family_size: int = 6,
                    generalist_fraction: float = 0.30,
                    activity_floor_quantile: float = 0.10,
                    random_sets: int = 24, analysis_seed: int = 2718,
                    eval_split: str = "val", batch_size: int = 64,
                    seed: int = 42, accelerator: str = "cuda:0", num_threads: int = 4,
                    force_fresh_grit: bool = False) -> dict:
    """Load one checkpoint and run only the cached-score family-ablation stage."""
    spec: GritTaskSpec = get_task(task_name)
    task_out = Path(collate_dir) / spec.name
    task_out.mkdir(parents=True, exist_ok=True)
    default_dir = f"/content/GRIT_{spec.name}" if spec.env_hooks else "/content/GRIT"
    repo_dir = Path(spec.grit_repo_dir or default_dir)
    env.clone_grit(repo_dir, spec.grit_repo, spec.grit_commit, force_fresh=force_fresh_grit)
    for hook in spec.env_hooks:
        hook(repo_dir)
    env.prepare_inprocess_grit(repo_dir)
    config_file = env.resolve_config(spec, repo_dir, task_out)
    chosen_ckpt, _ = env.find_checkpoint(Path(spec.drive_dir) / "results", ckpt)
    log(f"[family-ablation] runtime: {platform.platform()} | python {sys.version.split()[0]}")
    sc = SpecConfig(
        ckpt=str(chosen_ckpt), out_dir=str(task_out),
        dataset_dir=str(Path(spec.drive_dir) / "datasets"), config_file=config_file,
        accelerator=accelerator, seed=seed, num_threads=num_threads,
        eval_split=eval_split, donor_split=eval_split, eval_metric=False,
        num_graphs=0, donors=0, ablation_graphs=num_graphs, analysis_seed=analysis_seed,
        partner_match="degree")
    gm = GritHeadModel(spec, sc).load()
    try:
        return run_family_ablation(
            gm, scores, gsem=gsem, gstr=gstr, num_graphs=num_graphs,
            family_size=family_size, generalist_fraction=generalist_fraction,
            activity_floor_quantile=activity_floor_quantile, random_sets=random_sets,
            seed=analysis_seed, batch_size=batch_size)
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
