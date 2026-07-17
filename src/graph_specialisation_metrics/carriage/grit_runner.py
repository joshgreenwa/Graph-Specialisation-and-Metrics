"""The GRIT carriage analysis loop -- task-agnostic given a GritTaskSpec.

Loads a trained GRIT checkpoint, builds GraphGym loaders, runs the semantic donor-swap
intervention over selected graphs, and returns per-pair (C, B, distance) arrays plus every
verification result. All model/task specifics come from the GritTaskSpec + its content
adapter; the carriage math is in ``core``.

Carriage correctness rests on four preconditions, CHECKED on the loaded cfg (not assumed):
sum/mean pooling (shared g_i), RRWP invariant to a content swap (add_node_attr False),
a pooling->MLP graph head, and a scalar regression target. See check_carriage_preconditions.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import core
from .env import log


@dataclass
class CarriageConfig:
    ckpt: str
    out_dir: str
    dataset_dir: str
    repo_dir: str
    config_file: str                      # absolute path resolved by env.resolve_config
    accelerator: str = "cuda:0"
    seed: int = 42
    num_threads: int = 4
    eval_split: str = "test"
    donor_split: str = "test"
    num_graphs: int = 64
    donors: int = 32
    graph_select: str = "random"          # random | first
    analysis_seed: int = 0
    verify: bool = True
    verify_graphs: int = 2
    eval_metric: bool = True
    allow_param_count_drift: bool = False
    tol: float = 1e-4
    max_replicas: int = 4096
    max_pair_edges: int = 12_000_000


def _plan_chunk(n: int, cfg: CarriageConfig) -> int:
    per = max(n * n, 1)
    return max(1, min(cfg.max_replicas, cfg.max_pair_edges // per))


def _spd(data, n: int) -> np.ndarray:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    ei = data.edge_index.cpu().numpy()
    if ei.shape[1] == 0:
        D = np.full((n, n), np.inf)
        np.fill_diagonal(D, 0.0)
        return D
    A = csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    return shortest_path(A, method="D", unweighted=True, directed=False)


def _verify_structure(base, b, m: int, n: int, rows, adapter) -> None:
    """Only node content changed; edge_index/edge_attr/rrwp* are bit-identical.

    Runs on the collated batch BEFORE the forward pass mutates it (GRIT's encoders rebind
    batch.x/edge_index/edge_attr in place). Content-agnostic: it checks the structural
    tensors and that x differs only at the intended source rows.
    """
    import torch

    def _rep(t, inc=0):
        parts = [t + (r * n if inc else 0) for r in range(m)]
        return torch.cat(parts, dim=(-1 if inc else 0))

    assert torch.equal(b.edge_index, _rep(base.edge_index.to(b.edge_index.device), inc=1)), \
        "edge_index changed under a semantic intervention"
    assert torch.equal(b.edge_attr, _rep(base.edge_attr.to(b.edge_attr.device))), \
        "edge_attr changed under a semantic intervention"
    for name in ("rrwp", "rrwp_val"):
        if hasattr(base, name):
            assert torch.equal(getattr(b, name), _rep(getattr(base, name).to(b.x.device))), \
                f"{name} changed under a semantic intervention"
    if hasattr(base, "rrwp_index"):
        assert torch.equal(b.rrwp_index, _rep(base.rrwp_index.to(b.rrwp_index.device), inc=1)), \
            "rrwp_index changed under a semantic intervention"

    x_ref = _rep(base.x.to(b.x.device))
    diff = (b.x != x_ref).any(dim=1)
    assert int(diff.sum().item()) <= m, "more than one node per replica was modified"
    changed = set(diff.nonzero().view(-1).tolist())
    assert changed.issubset(set(rows.tolist())), \
        "a node other than the intended source j was modified"


def check_carriage_preconditions(cfg, tol: float, checks: dict) -> None:
    """Validate, on the LOADED GraphGym cfg, the conditions carriage depends on.

    Task-agnostic (works for any GRIT config). Fatal violations raise; softer ones warn.
    """
    problems = []
    # (1) Readout gradient shared across nodes requires additive/mean pooling.
    pooling = str(cfg.model.graph_pooling)
    checks["graph_pooling"] = pooling
    if pooling not in ("add", "mean"):
        problems.append(f"model.graph_pooling={pooling!r} (need add/mean so g_i is shared "
                        f"across nodes); carriage's per-node g_i story does not hold.")
    # (2) A content swap must leave the structural encoding fixed.
    if bool(getattr(cfg.posenc_RRWP, "enable", False)) and bool(getattr(cfg.posenc_RRWP, "add_node_attr", False)):
        problems.append("posenc_RRWP.add_node_attr=True makes RRWP depend on node content, "
                        "so a semantic swap would also perturb S (violates Def 3.2.1).")
    # (3) Pooling->MLP graph head.
    head = str(cfg.gnn.head)
    checks["gnn_head"] = head
    if head != "san_graph":
        log(f"[precond-warn] gnn.head={head!r} (expected san_graph pooling->MLP). h^L is still "
            f"the pre-head node state; interpret g_i accordingly.")
    # (4) Scalar regression target (B uses sign of a scalar residual).
    tt = str(cfg.dataset.task_type)
    checks["task_type"] = tt
    if tt != "regression":
        problems.append(f"dataset.task_type={tt!r}; carriage F(d)/B(d) as implemented assume "
                        f"a scalar regression target (ŷ, y scalars).")
    checks["loss_fun"] = str(cfg.model.loss_fun)
    if problems:
        raise RuntimeError("Carriage preconditions not met:\n  - " + "\n  - ".join(problems))
    log("[precond] carriage preconditions OK: pooling={}, RRWP content-invariant, head={}, {} target."
        .format(pooling, head, tt))


def run_grit_carriage(task, cc: CarriageConfig) -> dict:
    """Run the full semantic-intervention carriage analysis for one task/checkpoint.

    Returns a dict with per-pair arrays (graph_id, carrier_i, source_j, distance, C, B, F),
    a `checks` sub-dict (every verification), and `meta` (provenance for captions).
    """
    import torch
    from torch_geometric import seed_everything
    from torch_geometric.data import Batch
    from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
    from torch_geometric.graphgym.loader import create_loader
    from torch_geometric.graphgym.model_builder import create_model
    from torch_geometric.graphgym.utils.comp_budget import params_count

    import grit  # noqa: F401  registers loaders/encoders/layers/heads

    adapter = task.content_adapter
    out_dir = Path(cc.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_graphgym_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    checks: dict = {}

    # ---- config (mirrors main.py, minus the destructive run-dir setup) ----------------
    set_cfg(cfg)
    cfg.set_new_allowed(True)
    cfg.work_dir = os.getcwd()
    opts = [
        "out_dir", str(scratch), "dataset.dir", str(cc.dataset_dir),
        "seed", str(cc.seed), "accelerator", cc.accelerator,
        "wandb.use", "False", "mlflow.use", "False",
        "train.auto_resume", "False", "train.enable_ckpt", "False",
        "num_threads", str(cc.num_threads),
    ]
    load_cfg(cfg, argparse.Namespace(cfg_file=cc.config_file, opts=opts))
    # assert_cfg sets cfg.run_dir = cfg.out_dir (= scratch). Never let GraphGym write near
    # the trained run: custom_set_run_dir (which we never call) would rm-exist it.
    assert Path(cfg.run_dir).resolve() == scratch.resolve(), \
        f"cfg.run_dir unexpectedly {cfg.run_dir}; refusing to run near training outputs."

    device = torch.device(cc.accelerator if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    if device.type != "cuda":
        log("[warn] CUDA unavailable; running on CPU. This will be slow.")
    torch.set_num_threads(cfg.num_threads)
    seed_everything(cfg.seed)

    check_carriage_preconditions(cfg, cc.tol, checks)

    # ---- data + model -----------------------------------------------------------------
    log("\n[data] Building loaders (GRIT recomputes RRWP in-memory; not cached to disk).")
    loaders = create_loader()
    assert len(loaders) == 3, f"Expected [train, val, test] loaders, got {len(loaders)}"
    split_of = {"train": 0, "val": 1, "test": 2}
    eval_ds = loaders[split_of[cc.eval_split]].dataset
    donor_ds = loaders[split_of[cc.donor_split]].dataset
    log(f"[data] eval={cc.eval_split} ({len(eval_ds)}) | donors={cc.donor_split} ({len(donor_ds)})")

    model = create_model()
    n_params = params_count(model)
    checks["num_parameters"] = int(n_params)
    if task.expected_params is not None and n_params != task.expected_params:
        msg = f"[param-check:ERROR] {n_params} != expected {task.expected_params} for task {task.name!r}."
        if not cc.allow_param_count_drift:
            raise RuntimeError(msg)
        log(msg + " Continuing (allow_param_count_drift).")
    else:
        log(f"[param-check] parameters: {n_params}"
            + (f" (matches expected {task.expected_params})" if task.expected_params else ""))

    # ---- checkpoint -------------------------------------------------------------------
    ckpt_path = Path(cc.ckpt)
    log(f"\n[ckpt] Loading: {ckpt_path}")
    blob = torch.load(str(ckpt_path), map_location="cpu")
    ckpt_epoch = int(ckpt_path.stem) if ckpt_path.stem.isdigit() else -1
    if isinstance(blob, dict) and "model_state" in blob:
        state = blob["model_state"]
    elif isinstance(blob, dict) and "state_dict" in blob:
        state = blob["state_dict"]
    else:
        state = blob

    def _try_load(sd):
        try:
            model.load_state_dict(sd, strict=True)
            return True, "strict"
        except RuntimeError as e:
            return False, str(e)

    ok, how = _try_load(state)
    if not ok:
        stripped = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
        prefixed = {f"model.{k}": v for k, v in state.items()}
        for cand, label in ((stripped, "stripped 'model.' prefix"), (prefixed, "added 'model.' prefix")):
            if cand:
                ok, _ = _try_load(cand)
                if ok:
                    how = label
                    break
        if not ok:
            raise RuntimeError(f"Could not load checkpoint state_dict strictly.\n{how}")
    log(f"[ckpt] Loaded strictly (key handling: {how}); epoch {ckpt_epoch}.")
    checks["ckpt_path"] = str(ckpt_path)
    checks["ckpt_epoch"] = int(ckpt_epoch)
    checks["ckpt_key_handling"] = how

    model.to(device)
    model.eval()  # load-bearing: batch_norm + attn_dropout would couple replicas otherwise.
    assert type(model.model).__name__ == "GritTransformer", \
        f"Expected GritTransformer, got {type(model.model).__name__}"

    # ---- recompute the eval metric from the loaded checkpoint (strongest load check) ---
    @torch.no_grad()
    def split_metric(loader) -> float:
        tot, cnt = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            pred, true = model(batch)
            p, t = pred.view(-1), true.view(-1)
            assert p.numel() == t.numel()
            tot += (p - t).abs().sum().item()
            cnt += t.numel()
        return tot / max(cnt, 1)

    if cc.eval_metric:
        t0 = time.perf_counter()
        test_metric = split_metric(loaders[2])
        pm = f" (paper {task.paper_metric[0]} ~{task.paper_metric[1]})" if task.paper_metric else ""
        log(f"[verify] test MAE recomputed from checkpoint: {test_metric:.5f}{pm} "
            f"[{time.perf_counter()-t0:.1f}s]")
        checks["test_metric"] = float(test_metric)
        if test_metric > task.metric_sanity_threshold:
            raise RuntimeError(
                f"Recomputed test metric {test_metric:.5f} exceeds task sanity threshold "
                f"{task.metric_sanity_threshold}. The checkpoint almost certainly did not load "
                f"correctly (or training barely progressed). Refusing to report carriage."
            )
    else:
        checks["test_metric"] = None

    # ---- h^L capture: the tensor entering post_mp (out of the transformer layers) ------
    store: dict = {}

    def _hook(_m, _i, output):
        store["h"] = output.x

    handle = model.model.layers.register_forward_hook(_hook)
    dim_h = int(cfg.gnn.dim_inner)
    num_types = adapter.num_symbols(cfg)

    # ---- donor pool (Def 3.2.2): real content from OTHER graphs of the same dataset ----
    log(f"\n[donors] Building donor pool from '{cc.donor_split}' ({task.node_content_desc}).")
    sym_all, gid_all = [], []
    for gi in range(len(donor_ds)):
        s = adapter.symbols(donor_ds[gi])
        sym_all.append(s)
        gid_all.append(np.full(s.shape[0], gi, dtype=np.int64))
    donor_syms = np.concatenate(sym_all).astype(np.int64)
    donor_gids = np.concatenate(gid_all)
    assert donor_syms.min() >= 0 and donor_syms.max() < num_types, (
        f"Donor symbols out of range for num_types={num_types}: [{donor_syms.min()}, {donor_syms.max()}]"
    )
    log(f"[donors] pool={donor_syms.size} nodes over {len(donor_ds)} graphs; "
        f"{len(np.unique(donor_syms))} distinct symbols (max {donor_syms.max()} < {num_types}).")

    # ---- graph selection --------------------------------------------------------------
    rng = np.random.default_rng(cc.analysis_seed)
    n_graphs = min(cc.num_graphs, len(eval_ds))
    if cc.graph_select == "random":
        graph_ids = np.sort(rng.choice(len(eval_ds), size=n_graphs, replace=False))
    else:
        graph_ids = np.arange(n_graphs)
    K = int(cc.donors)
    log(f"[select] {n_graphs} {cc.eval_split} graphs ({cc.graph_select}, seed={cc.analysis_seed}), K={K}.")

    # ---- accumulators -----------------------------------------------------------------
    all_gid, all_i, all_j, all_d, all_C, all_B = [], [], [], [], [], []
    add_sumC, add_dyhat = [], []
    g_spread_max = noop_max_dh = batchinv_max = bexact_max = 0.0
    noop_total = donor_draws = unreachable_total = struct_checked = peak_mem = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    for gi_pos, gi in enumerate(graph_ids):
        base = eval_ds[int(gi)]
        n = int(base.num_nodes)
        y = float(base.y.view(-1)[0].item())

        # distances from the PRISTINE graph (GRIT's rel encoder overwrites edge_index later).
        D = _spd(base, n)
        unreachable_total += int(np.isinf(D).sum())

        # clean pass: h^L and g_i = d yhat / d h^L_i (Eq 3.1).
        cb = Batch.from_data_list([base]).to(device)
        with torch.enable_grad():
            pred_c, _ = model(cb)
            h_clean_t = store["h"]
            assert h_clean_t.shape == (n, dim_h), f"h^L {tuple(h_clean_t.shape)} != {(n, dim_h)}"
            g_t = torch.autograd.grad(pred_c.sum(), h_clean_t)[0]
        yhat_clean = float(pred_c.view(-1)[0].item())
        h_clean = h_clean_t.detach()
        g = g_t.detach()
        g_spread_max = max(g_spread_max, float((g - g[0:1]).abs().max().item()))

        # donors: K per source j, independent; from other graphs when donors==eval split.
        if cc.donor_split == cc.eval_split:
            pool = np.flatnonzero(donor_gids != int(gi))
        else:
            pool = np.arange(donor_syms.size)
        donor_ids = rng.choice(pool, size=(n, K), replace=True)
        donor_sym = donor_syms[donor_ids]                       # [n, K]
        own_sym = adapter.symbols(base)                         # [n]
        noop_mask = donor_sym == own_sym[:, None]
        noop_total += int(noop_mask.sum())
        donor_draws += donor_sym.size

        S, R = n, n * K
        rows_local = np.repeat(np.arange(S), K)
        flat_donor = donor_sym.reshape(-1)

        h_swap = torch.empty((R, n, dim_h), device=device, dtype=h_clean.dtype)
        yhat_swap = torch.empty((R,), device=device, dtype=torch.float32)

        chunk = _plan_chunk(n, cc)
        r0 = 0
        while r0 < R:
            m = min(chunk, R - r0)
            while True:
                try:
                    with torch.no_grad():
                        b = Batch.from_data_list([base] * m).to(device)
                        rows = (torch.arange(m, device=device) * n
                                + torch.as_tensor(rows_local[r0:r0 + m], device=device))
                        adapter.write_donors(b.x, rows, flat_donor[r0:r0 + m])
                        if cc.verify and struct_checked < cc.verify_graphs and r0 == 0:
                            _verify_structure(base, b, m, n, rows, adapter)
                            struct_checked += 1
                        pred_s, _ = model(b)
                        hs = store["h"]
                        assert hs.shape == (m * n, dim_h)
                        h_swap[r0:r0 + m] = hs.view(m, n, dim_h)
                        yhat_swap[r0:r0 + m] = pred_s.view(-1).float()
                    break
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if m == 1:
                        raise
                    m = max(1, m // 2)
                    chunk = m
                    log(f"[mem] CUDA OOM -> reducing replicas/forward to {m}")
            r0 += m

        if device.type == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())

        if cc.verify and gi_pos < cc.verify_graphs:
            with torch.no_grad():
                mm = min(_plan_chunk(n, cc), 64)
                b = Batch.from_data_list([base] * mm).to(device)
                model(b)
                hb = store["h"].view(mm, n, dim_h)
                batchinv_max = max(batchinv_max, float((hb - h_clean.unsqueeze(0)).abs().max().item()))

        if noop_mask.any():
            nm = torch.as_tensor(noop_mask.reshape(-1), device=device)
            noop_max_dh = max(noop_max_dh, float((h_swap[nm] - h_clean.unsqueeze(0)).abs().max().item()))

        # carriage (Eq 3.4/3.5) and beneficial carriage (exact per-source loss change).
        C = core.carriage_from_states(h_clean, h_swap, g, S, K).cpu().numpy()
        B, dL_j, sumC_j = core.beneficial_from_carriage(C, yhat_clean, yhat_swap, y, S, K)

        add_sumC.append(sumC_j)
        add_dyhat.append(yhat_clean - yhat_swap.view(S, K).mean(dim=1).cpu().numpy())
        moved = np.abs(sumC_j) > 1e-9
        if moved.any():
            bexact_max = max(bexact_max, float(np.abs(B.sum(axis=0)[moved] - dL_j[moved]).max()))

        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        finite = np.isfinite(D)
        all_gid.append(np.full(int(finite.sum()), int(gi), dtype=np.int64))
        all_i.append(ii[finite].astype(np.int64))
        all_j.append(jj[finite].astype(np.int64))
        all_d.append(D[finite].astype(np.int64))
        all_C.append(C[finite].astype(np.float64))
        all_B.append(B[finite].astype(np.float64))

        del h_swap, yhat_swap
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (gi_pos + 1) % max(1, n_graphs // 10) == 0 or gi_pos == 0:
            log(f"[run] graph {gi_pos+1}/{n_graphs} (id={int(gi)}, n={n}, {R} forwards) "
                f"| {time.perf_counter()-t_start:.1f}s")

    handle.remove()

    gid = np.concatenate(all_gid)
    pd_ = np.concatenate(all_d)
    pC = np.concatenate(all_C)
    pB = np.concatenate(all_B)
    F = np.abs(pC)

    # ---- verification summary ---------------------------------------------------------
    tol = cc.tol
    a_sumC = np.concatenate(add_sumC)
    a_dyhat = np.concatenate(add_dyhat)
    if a_sumC.size > 2 and np.std(a_sumC) > 0 and np.std(a_dyhat) > 0:
        r = float(np.corrcoef(a_sumC, a_dyhat)[0, 1])
        slope = float(np.polyfit(a_sumC, a_dyhat, 1)[0])
    else:
        r, slope = float("nan"), float("nan")

    noop_frac = noop_total / max(donor_draws, 1)
    checks.update({
        "g_spread_max": float(g_spread_max),
        "batch_invariance_max_abs_dh": float(batchinv_max),
        "donor_noop_fraction": float(noop_frac),
        "donor_noop_max_abs_dh": float(noop_max_dh),
        "structure_verified_graphs": int(struct_checked),
        "unreachable_pairs_excluded": int(unreachable_total),
        "additivity_pearson_r": r,
        "additivity_slope": slope,
        "beneficial_exactness_max": float(bexact_max),
    })
    if device.type == "cuda":
        checks["peak_cuda_gib"] = float(peak_mem / 1024**3)

    log("\n" + "=" * 84 + "\nVERIFICATION\n" + "=" * 84)
    log(f"  [param]  parameters              : {n_params}"
        + (f" (expected {task.expected_params})" if task.expected_params else ""))
    if checks["test_metric"] is not None:
        log(f"  [4]  test MAE from checkpoint    : {checks['test_metric']:.5f}")
    log(f"  [5]  max_i |g_i - g_1|           : {g_spread_max:.3e}  (pooling => must be ~0)")
    if cc.verify:
        log(f"  [6]  batch-invariance max|dh|    : {batchinv_max:.3e}")
        log(f"  [8]  structure invariance        : verified on {struct_checked} graph(s)")
    log(f"  [7]  no-op donors                : {noop_total}/{donor_draws} "
        f"({100*noop_frac:.1f}%) max|dh|={noop_max_dh:.3e} (must be 0)")
    log(f"  [9]  additivity r/slope          : r={r:.4f}, slope={slope:.4f}")
    log(f"  [10] unreachable pairs (excluded): {unreachable_total}")
    log(f"  [11] beneficial exactness        : max_j|sum_i B - dL_j| = {bexact_max:.3e} (must be ~0)")
    if device.type == "cuda":
        log(f"  [mem] peak CUDA allocated        : {checks['peak_cuda_gib']:.2f} GiB")

    if g_spread_max > tol:
        log(f"  [!] g_i varies by {g_spread_max:.3e} > tol; readout not pooling->MLP as assumed.")
    if noop_max_dh > tol:
        raise RuntimeError(f"No-op donors gave |dh|={noop_max_dh:.3e} > tol {tol:.1e}: a same-symbol "
                           f"swap must be a bit-exact identity (wrong row, or model not in eval()).")
    if cc.verify and batchinv_max > tol:
        raise RuntimeError(f"Batch invariance violated: max|dh|={batchinv_max:.3e} > tol {tol:.1e}.")
    if bexact_max > tol:
        raise RuntimeError(f"Beneficial exactness violated: max_j|sum_i B - dL_j|={bexact_max:.3e} "
                           f"> tol {tol:.1e}; share attribution or loss donor-average miswired.")

    meta = {
        "task": task.name, "title": task.title,
        "grit_repo": task.grit_repo, "grit_commit": task.grit_commit,
        "config": cc.config_file, "checkpoint": str(ckpt_path), "checkpoint_epoch": ckpt_epoch,
        "eval_split": cc.eval_split, "donor_split": cc.donor_split,
        "num_graphs": int(np.unique(gid).size), "donors_K": K,
        "graph_select": cc.graph_select, "analysis_seed": cc.analysis_seed,
        "test_metric": checks["test_metric"], "paper_metric": task.paper_metric,
    }
    return {
        "graph_id": gid,
        "carrier_i": np.concatenate(all_i), "source_j": np.concatenate(all_j),
        "distance": pd_, "C": pC, "B": pB, "F": F,
        "additivity_sumC": a_sumC, "additivity_dyhat": a_dyhat,
        "checks": checks, "meta": meta,
    }
