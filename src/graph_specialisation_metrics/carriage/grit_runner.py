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

from . import core, metrics
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
    beneficial_denom: str = "magnitude"   # "magnitude" (default) | "signed" (legacy)
    tol: float = 1e-4
    # Looser ceiling for the float32-noise checks (no-op / batch-invariance). Real wiring
    # bugs (wrong row, no eval()) give O(1) dh, far above this; large full-attention graphs
    # can push GPU non-determinism to ~1e-3. Kept well below any real-bug scale.
    float_noise_tol: float = 5e-3
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


def _verify_structure(base, b, reps: int, n: int, rows, adapter) -> None:
    """Only node content changed; edge_index/edge_attr/rrwp* are bit-identical.

    Runs on the collated batch BEFORE the forward pass mutates it (GRIT's encoders rebind
    batch.x/edge_index/edge_attr in place). Content-agnostic: it checks the structural
    tensors are the per-graph values replicated ``reps`` times, and that x differs only at
    the intended source rows (``rows`` are the flat global row indices that were written;
    with a within-batch clean baseline, replica 0 is unswapped and contributes no changes).
    """
    import torch

    def _rep(t, inc=0):
        parts = [t + (r * n if inc else 0) for r in range(reps)]
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
    assert int(diff.sum().item()) <= len(rows), "more nodes changed than donors were written"
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
    # (3) Pooling->MLP graph head. GraphGym resolves gnn.head='default' -> dataset.task
    #     ('graph' => GNNGraphHead); san_graph is the ZINC variant. Both pool then MLP.
    head = str(cfg.gnn.head)
    checks["gnn_head"] = head
    if head not in ("san_graph", "graph"):
        log(f"[precond-warn] gnn.head={head!r} (expected graph/san_graph pooling->MLP). h^L is "
            f"still the pre-head node state; interpret g_i accordingly.")
    # (4) Supported readout: scalar/multi-target regression, or multilabel classification.
    #     Beneficial carriage decomposes L_clean - L_swap for the task loss; functional
    #     carriage is ||dŷ from i|| over the T outputs. Both handle T>1.
    tt = str(cfg.dataset.task_type)
    checks["task_type"] = tt
    if tt not in ("regression", "classification_multilabel"):
        problems.append(f"dataset.task_type={tt!r}; carriage supports regression and "
                        f"classification_multilabel (a scalar/vector output with an additive "
                        f"per-target loss). Others need a bespoke readout.")
    checks["loss_fun"] = str(cfg.model.loss_fun)
    if problems:
        raise RuntimeError("Carriage preconditions not met:\n  - " + "\n  - ".join(problems))
    log("[precond] carriage preconditions OK: pooling={}, RRWP content-invariant, head={}, {} ({})."
        .format(pooling, head, tt, cfg.model.loss_fun))


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
    def collect_preds(loader):
        ps, ts = [], []
        for batch in loader:
            batch = batch.to(device)
            pred, true = model(batch)
            ps.append(pred.detach().cpu().numpy().reshape(pred.shape[0], -1))
            ts.append(true.detach().cpu().numpy().reshape(true.shape[0], -1))
        return np.concatenate(ps), np.concatenate(ts)

    metric_name = (task.paper_metric[0] if task.paper_metric else "metric")
    if cc.eval_metric:
        t0 = time.perf_counter()
        preds, trues = collect_preds(loaders[2])
        test_metric = float(task.metric_fn(preds, trues))
        pm = f" (paper {task.paper_metric[0]} ~{task.paper_metric[1]})" if task.paper_metric else ""
        log(f"[verify] test {metric_name} recomputed from checkpoint: {test_metric:.5f}{pm} "
            f"[{time.perf_counter()-t0:.1f}s]")
        checks["test_metric"] = test_metric
        checks["test_metric_name"] = metric_name
        bad = (test_metric < task.metric_abort) if task.metric_higher_better else (test_metric > task.metric_abort)
        if bad:
            arrow = "below" if task.metric_higher_better else "above"
            raise RuntimeError(
                f"Recomputed test {metric_name} {test_metric:.5f} is {arrow} the abort threshold "
                f"{task.metric_abort}. The checkpoint almost certainly did not load correctly "
                f"(or training barely progressed). Refusing to report carriage."
            )
    else:
        checks["test_metric"] = None
        checks["test_metric_name"] = metric_name

    # ---- h^L capture: the tensor entering post_mp (out of the transformer layers) ------
    store: dict = {}

    def _hook(_m, _i, output):
        store["h"] = output.x

    handle = model.model.layers.register_forward_hook(_hook)
    dim_h = int(cfg.gnn.dim_inner)
    loss_fun = str(cfg.model.loss_fun)
    num_types = adapter.num_symbols(cfg)

    # ---- donor pool (Def 3.2.2): real content ROWS from OTHER graphs of the same dataset -
    log(f"\n[donors] Building donor pool from '{cc.donor_split}' ({task.node_content_desc}).")
    row_all, gid_all = [], []
    for gi in range(len(donor_ds)):
        r = adapter.rows(donor_ds[gi])                          # [n, F]
        row_all.append(r)
        gid_all.append(np.full(r.shape[0], gi, dtype=np.int64))
    donor_rows = np.concatenate(row_all).astype(np.int64)       # [Npool, F]
    donor_gids = np.concatenate(gid_all)
    F_feat = donor_rows.shape[1]
    if num_types is not None and F_feat == 1:
        assert donor_rows.min() >= 0 and donor_rows.max() < num_types, (
            f"Donor symbols out of range for num_types={num_types}: "
            f"[{donor_rows.min()}, {donor_rows.max()}]"
        )
    n_distinct = len(np.unique(donor_rows, axis=0))
    log(f"[donors] pool={donor_rows.shape[0]} nodes over {len(donor_ds)} graphs; "
        f"{n_distinct} distinct content rows (F={F_feat} feature col(s)).")

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
    all_gid, all_i, all_j, all_d, all_C, all_B, all_F = [], [], [], [], [], [], []
    add_sumC, add_dyhat = [], []
    g_spread_max = noop_max_dh = batchinv_max = bexact_max = 0.0
    noop_total = donor_draws = unreachable_total = struct_checked = peak_mem = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    for gi_pos, gi in enumerate(graph_ids):
        base = eval_ds[int(gi)]
        n = int(base.num_nodes)

        # distances from the PRISTINE graph (GRIT's rel encoder overwrites edge_index later).
        D = _spd(base, n)
        unreachable_total += int(np.isinf(D).sum())

        # clean pass. Two readout gradients at h^L (Eq 3.1), both at the clean input:
        #   g_out[t] = d yhat_t / d h^L_i  (T of them; functional carriage magnitude)
        #   g_loss   = d L / d h^L_i        (loss carriage; beneficial-carriage basis)
        cb = Batch.from_data_list([base]).to(device)
        with torch.enable_grad():
            pred_c, true_c = model(cb)
            h_clean_t = store["h"]
            assert h_clean_t.shape == (n, dim_h), f"h^L {tuple(h_clean_t.shape)} != {(n, dim_h)}"
            T = int(pred_c.view(pred_c.shape[0], -1).shape[1])
            pred_row = pred_c.view(-1)                                   # [T]
            g_out = torch.stack([
                torch.autograd.grad(pred_row[t], h_clean_t, retain_graph=True)[0]
                for t in range(T)
            ])                                                          # [T, n, m]
            L_c = metrics.per_graph_loss(pred_c, true_c, loss_fun).sum()  # scalar
            g_loss_t = torch.autograd.grad(L_c, h_clean_t)[0]           # [n, m]
        h_clean = h_clean_t.detach()
        g_loss = g_loss_t.detach()
        g_out = g_out.detach()
        true_vec = true_c.detach().view(1, -1).float()                 # [1, T]
        L_clean = float(metrics.per_graph_loss(pred_c.detach(), true_c.detach(), loss_fun)[0].item())
        # mean/add pooling => g_loss shared across nodes; verified, not assumed.
        g_spread_max = max(g_spread_max, float((g_loss - g_loss[0:1]).abs().max().item()))

        # donors: K per source j, independent; from other graphs when donors==eval split.
        if cc.donor_split == cc.eval_split:
            pool = np.flatnonzero(donor_gids != int(gi))
        else:
            pool = np.arange(donor_rows.shape[0])
        donor_ids = rng.choice(pool, size=(n, K), replace=True)        # [n, K]
        donor_content = donor_rows[donor_ids]                          # [n, K, F]
        own_rows = adapter.rows(base)                                  # [n, F]
        noop_mask = (donor_content == own_rows[:, None, :]).all(axis=-1)  # [n, K]
        noop_total += int(noop_mask.sum())
        donor_draws += int(noop_mask.size)

        S, R = n, n * K
        rows_local = np.repeat(np.arange(S), K)
        flat_donor = donor_content.reshape(R, F_feat)                  # [R, F]

        # Transport delta dh_i(j,k) = h^L_i(clean) - h^L_i(swap), computed with a WITHIN-BATCH
        # clean baseline: replica 0 of every chunk is the un-swapped graph, so delta is a
        # same-forward difference. This cancels the batch-context float32 offset that a
        # batch-of-1 clean would carry (~1e-4 in h) -- which on large peptides graphs is
        # comparable to the long-range carriage signal -- and makes no-op donors give ~0.
        delta = torch.empty((R, n, dim_h), device=device, dtype=h_clean.dtype)
        pred_swap = torch.empty((R, T), device=device, dtype=torch.float32)

        chunk = _plan_chunk(n, cc)
        r0 = 0
        while r0 < R:
            m = min(chunk, R - r0)
            while True:
                try:
                    with torch.no_grad():
                        # replica 0 = clean baseline; replicas 1..m = the m swaps.
                        b = Batch.from_data_list([base] * (m + 1)).to(device)
                        rows = (torch.arange(1, m + 1, device=device) * n
                                + torch.as_tensor(rows_local[r0:r0 + m], device=device))
                        adapter.write_donors(b.x, rows, flat_donor[r0:r0 + m])
                        if cc.verify and struct_checked < cc.verify_graphs and r0 == 0:
                            _verify_structure(base, b, m + 1, n, rows, adapter)
                            struct_checked += 1
                        pred_s, _ = model(b)
                        hs = store["h"]
                        assert hs.shape == ((m + 1) * n, dim_h)
                        hs = hs.view(m + 1, n, dim_h)
                        clean_chunk = hs[0]                            # [n, dim] clean, same forward
                        delta[r0:r0 + m] = clean_chunk.unsqueeze(0) - hs[1:]
                        pred_swap[r0:r0 + m] = pred_s.view(m + 1, -1)[1:].float()
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

        # batch invariance: clean-alone (grad forward, batch-1) vs clean inside a big batch.
        # This is now only a model-sanity probe (eval mode); the carriage no longer depends
        # on it since delta uses a within-batch clean baseline.
        if cc.verify and gi_pos < cc.verify_graphs:
            with torch.no_grad():
                mm = min(_plan_chunk(n, cc), 64)
                b = Batch.from_data_list([base] * mm).to(device)
                model(b)
                hb = store["h"].view(mm, n, dim_h)
                batchinv_max = max(batchinv_max, float((hb - h_clean.unsqueeze(0)).abs().max().item()))

        # no-op donors: same content as the source => transport must be ~0. With the
        # within-batch baseline this is now just residual GPU non-determinism (tiny).
        if noop_mask.any():
            nm = torch.as_tensor(noop_mask.reshape(-1), device=device)
            noop_max_dh = max(noop_max_dh, float(delta[nm].abs().max().item()))

        # per-replica swap loss (donor-average the LOSS, not the prediction -- Jensen at the kink).
        L_swap = metrics.per_graph_loss(pred_swap, true_vec.expand(R, -1), loss_fun).cpu().numpy()  # [R]
        dL_j = L_clean - L_swap.reshape(S, K).mean(axis=1)             # [S]  <0 = beneficial

        # functional carriage F[i,j] = ||dŷ from i|| over T outputs (label-free);
        # loss carriage C_loss[i,j] = g_loss_i . dh_i(j) (beneficial-carriage basis).
        F_ij = core.functional_magnitude_from_delta(delta, g_out, S, K)          # [n, n], >=0
        C_loss = core.carriage_from_delta(delta, g_loss, S, K).cpu().numpy()      # [n, n]
        B, denom_used = core.beneficial_attribute(C_loss, dL_j, denom=cc.beneficial_denom)

        # additivity diagnostic uses the SIGNED sum (first-order dL), independent of the
        # denominator choice; exactness/"moved" use the denominator actually applied.
        add_sumC.append(C_loss.sum(axis=0))
        add_dyhat.append(dL_j)
        moved = np.abs(denom_used) > 1e-8
        if moved.any():
            bexact_max = max(bexact_max, float(np.abs(B.sum(axis=0)[moved] - dL_j[moved]).max()))

        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        finite = np.isfinite(D)
        all_gid.append(np.full(int(finite.sum()), int(gi), dtype=np.int64))
        all_i.append(ii[finite].astype(np.int64))
        all_j.append(jj[finite].astype(np.int64))
        all_d.append(D[finite].astype(np.int64))
        all_C.append(C_loss[finite].astype(np.float64))
        all_B.append(B[finite].astype(np.float64))
        all_F.append(F_ij[finite].astype(np.float64))

        del delta, pred_swap
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (gi_pos + 1) % max(1, n_graphs // 10) == 0 or gi_pos == 0:
            log(f"[run] graph {gi_pos+1}/{n_graphs} (id={int(gi)}, n={n}, T={T}, {R} forwards) "
                f"| {time.perf_counter()-t_start:.1f}s")

    handle.remove()

    gid = np.concatenate(all_gid)
    pd_ = np.concatenate(all_d)
    pC = np.concatenate(all_C)      # signed loss-carriage C_loss (beneficial basis)
    pB = np.concatenate(all_B)
    F = np.concatenate(all_F)       # functional carriage magnitude ||dŷ||

    # ---- verification summary ---------------------------------------------------------
    tol = cc.tol
    a_sumC = np.concatenate(add_sumC)     # sum_i C_loss[i,j]  (first-order loss change)
    a_dyhat = np.concatenate(add_dyhat)   # dL_j               (exact loss change)
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
        log(f"  [4]  test {checks['test_metric_name']} from checkpoint : {checks['test_metric']:.5f}")
    log(f"  [5]  max_i |g_i - g_1| (g_loss)  : {g_spread_max:.3e}  (pooling => must be ~0)")
    if cc.verify:
        log(f"  [6]  batch-invariance max|dh|    : {batchinv_max:.3e}")
        log(f"  [8]  structure invariance        : verified on {struct_checked} graph(s)")
    log(f"  [7]  no-op donors                : {noop_total}/{donor_draws} "
        f"({100*noop_frac:.1f}%) max|dh|={noop_max_dh:.3e} (within-batch => ~0; "
        f"float32 floor < {cc.float_noise_tol:.0e})")
    log(f"  [9]  loss additivity r/slope     : r={r:.4f}, slope={slope:.4f}  "
        f"(sum_i C_loss vs dL_j)")
    log(f"  [10] unreachable pairs (excluded): {unreachable_total}")
    log(f"  [11] beneficial exactness        : max_j|sum_i B - dL_j| = {bexact_max:.3e} (must be ~0)")
    if device.type == "cuda":
        log(f"  [mem] peak CUDA allocated        : {checks['peak_cuda_gib']:.2f} GiB")

    # g_spread and beneficial exactness are true-zero quantities -> strict tol. no-op and
    # batch-invariance are float32 GPU-noise probes -> looser float_noise_tol (a real bug
    # gives O(1), far above it).
    noise_tol = max(tol, cc.float_noise_tol)
    if g_spread_max > tol:
        log(f"  [!] g_i varies by {g_spread_max:.3e} > tol; readout not pooling->MLP as assumed.")
    if noop_max_dh > noise_tol:
        raise RuntimeError(f"No-op donors gave |dh|={noop_max_dh:.3e} > {noise_tol:.1e}: a same-content "
                           f"swap shares the within-batch clean baseline, so this must be ~0. A value "
                           f"this large means the wrong row is written or the model is not in eval().")
    if cc.verify and batchinv_max > noise_tol:
        raise RuntimeError(f"Batch invariance violated: max|dh|={batchinv_max:.3e} > {noise_tol:.1e} "
                           f"(model likely not in eval(): BatchNorm using batch stats).")
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
        "test_metric": checks["test_metric"], "test_metric_name": checks["test_metric_name"],
        "paper_metric": task.paper_metric, "loss_units": metrics.loss_units(loss_fun),
        "beneficial_denom": cc.beneficial_denom,
    }
    return {
        "graph_id": gid,
        "carrier_i": np.concatenate(all_i), "source_j": np.concatenate(all_j),
        "distance": pd_, "C": pC, "B": pB, "F": F,
        "additivity_sumC": a_sumC, "additivity_dyhat": a_dyhat,
        "checks": checks, "meta": meta,
    }
