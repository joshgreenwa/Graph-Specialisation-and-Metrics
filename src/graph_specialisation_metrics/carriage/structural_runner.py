"""The GRIT STRUCTURAL carriage loop -- the structural twin of ``grit_runner``.

Same estimator, different intervention. Where ``grit_runner`` overwrites node content and
holds structure fixed, this loop perturbs the topology-derived tensors (``structural.perturb``)
and holds content fixed, then hands the identical ``delta = h_clean - h_swap`` to ``core``.
So F(d)/B(d)/B_far(k), the aggregation, and the figures are all shared, unchanged.

Anchoring (see ``structural.py``): the source is a single node ``u``; the transposition's
partner ``v`` is a degree-matched nuisance marginalised over K draws (the donor-average
recipe), so C[i, u] and the distance d(i, u) are defined exactly as on the semantic side.

Model loading here deliberately MIRRORS ``grit_runner`` rather than sharing its body, so the
validated semantic path is untouched and this whole structural path (this file +
``structural.py`` + the ``intervention='structural'`` branch in ``colab.run``) can be deleted
as a unit. The genuinely shared *methodology* -- the carriage equations -- lives in ``core``.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from . import core, metrics, structural
from .env import log
from .grit_runner import (
    CarriageConfig,
    _integrated_failure_stats,
    _plan_chunk,
    _pooled_head_predictions,
    _retain_or_reject_unconverged_paths,
    _spd,
    check_carriage_preconditions,
)


def run_grit_structural_carriage(task, cc: CarriageConfig) -> dict:
    """Structural-intervention carriage for one task/checkpoint. Returns the same dict shape
    as ``run_grit_carriage`` (per-pair arrays + checks + meta), so figures/colab are shared."""
    import torch
    from torch_geometric import seed_everything
    from torch_geometric.data import Batch
    from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
    from torch_geometric.graphgym.loader import create_loader
    from torch_geometric.graphgym.model_builder import create_model
    from torch_geometric.graphgym.utils.comp_budget import params_count

    import grit  # noqa: F401  registers loaders/encoders/layers/heads

    mode = cc.structural_mode
    if mode not in ("transposition", "single_node"):
        raise ValueError(f"structural_mode must be 'transposition' or 'single_node', got {mode!r}")
    out_dir = Path(cc.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_graphgym_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    checks: dict = {}

    # ---- config (mirrors grit_runner) --------------------------------------------------
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
    assert Path(cfg.run_dir).resolve() == scratch.resolve(), \
        f"cfg.run_dir unexpectedly {cfg.run_dir}; refusing to run near training outputs."

    device = torch.device(cc.accelerator if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    if device.type != "cuda":
        log("[warn] CUDA unavailable; running on CPU. This will be slow.")
    torch.set_num_threads(cfg.num_threads)
    seed_everything(cfg.seed)

    # Structural carriage needs the SAME shared-g_i pooling readout as semantic; it does NOT
    # require add_node_attr=False (it perturbs structure on purpose), but that flag being off
    # only means RRWP is content-independent, which is still fine here.
    check_carriage_preconditions(cfg, cc.tol, checks)

    # ---- data + model (mirrors grit_runner) --------------------------------------------
    log("\n[data] Building loaders (GRIT recomputes RRWP in-memory; not cached to disk).")
    loaders = create_loader()
    assert len(loaders) == 3, f"Expected [train, val, test] loaders, got {len(loaders)}"
    split_of = {"train": 0, "val": 1, "test": 2}
    eval_ds = loaders[split_of[cc.eval_split]].dataset
    log(f"[data] eval={cc.eval_split} ({len(eval_ds)}) | structural mode={mode}, "
        f"partner_match={cc.partner_match}")

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

    # ---- checkpoint (mirrors grit_runner) ----------------------------------------------
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

    # ---- recompute the eval metric (strongest load check; mirrors grit_runner) ----------
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
                f"{task.metric_abort}. The checkpoint almost certainly did not load correctly. "
                f"Refusing to report carriage."
            )
        # Validation metric alongside test (loaders[1]=val), for reporting only -- no abort.
        t0 = time.perf_counter()
        vpreds, vtrues = collect_preds(loaders[1])
        val_metric = float(task.metric_fn(vpreds, vtrues))
        log(f"[verify] val {metric_name} recomputed from checkpoint: {val_metric:.5f} "
            f"[{time.perf_counter()-t0:.1f}s]")
        checks["val_metric"] = val_metric
        checks["val_metric_name"] = metric_name
    else:
        checks["test_metric"] = None
        checks["test_metric_name"] = metric_name
        checks["val_metric"] = None
        checks["val_metric_name"] = metric_name

    # ---- h^L capture (the tensor entering post_mp) -------------------------------------
    # A global-VNode model's layers output carries appended virtual-node row(s) (stripped just
    # before post_mp). Under grad (the clean readout pass) keep the FULL tensor so autograd.grad
    # reaches h; under no_grad (transport passes) strip the virtual rows so h aligns with real-node
    # indexing. The readout gradients are restricted to real nodes below. No-op without a VNode.
    store: dict = {}

    def _hook(_m, _i, output):
        mask = getattr(output, "real_node_mask", None)
        store["real_mask"] = mask
        store["h"] = (output.x[mask] if (mask is not None and not torch.is_grad_enabled())
                      else output.x)

    handle = model.model.layers.register_forward_hook(_hook)
    dim_h = int(cfg.gnn.dim_inner)
    loss_fun = str(cfg.model.loss_fun)
    valid_benefit = {"integrated", "slope", "magnitude", "signed"}
    if cc.beneficial_denom not in valid_benefit:
        raise ValueError(
            f"beneficial_denom must be one of {sorted(valid_benefit)}, "
            f"got {cc.beneficial_denom!r}"
        )
    integrated = cc.beneficial_denom == "integrated"
    if integrated and not 0.0 <= float(cc.integrated_max_unconverged_fraction) <= 1.0:
        raise ValueError("integrated_max_unconverged_fraction must lie in [0, 1]")
    if integrated and float(cc.integrated_unconverged_error_cap) < 0.0:
        raise ValueError("integrated_unconverged_error_cap must be non-negative")
    pooling = str(cfg.model.graph_pooling)

    # ---- graph selection (mirrors grit_runner) -----------------------------------------
    rng = np.random.default_rng(cc.analysis_seed)
    n_graphs = min(cc.num_graphs, len(eval_ds))
    if cc.graph_select == "random":
        graph_ids = np.sort(rng.choice(len(eval_ds), size=n_graphs, replace=False))
    else:
        graph_ids = np.arange(n_graphs)
    K = int(cc.donors)  # reused as the number of matched partners v per anchor u
    log(f"[select] {n_graphs} {cc.eval_split} graphs ({cc.graph_select}, seed={cc.analysis_seed}), "
        f"K={K} partners/anchor.")

    # ---- accumulators ------------------------------------------------------------------
    all_gid, all_i, all_j, all_d, all_C, all_B, all_F = [], [], [], [], [], [], []
    add_sumC, add_dyhat = [], []
    g_spread_max = noop_max_dh = bexact_max = relabel_inv_max = 0.0
    integrated_replay_max = integrated_full_loss_delta_max = 0.0
    clamp_moved = clamp_hit = 0            # slope-clip activation rate over moved sources
    noop_total = partner_draws = unreachable_total = struct_checked = peak_mem = 0
    integrated_residual, integrated_qerr, integrated_converged = [], [], []
    integrated_intervals, integrated_cancellation = [], []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    for gi_pos, gi in enumerate(graph_ids):
        base = eval_ds[int(gi)]
        n = int(base.num_nodes)
        if n < 2:  # a structural swap needs a partner; skip degenerate singletons
            continue

        D = _spd(base, n)                                   # pristine SPD for d(i, u)
        unreachable_total += int(np.isinf(D).sum())

        # clean pass with the two readout gradients at h^L (identical to semantic).
        cb = Batch.from_data_list([base]).to(device)
        with torch.enable_grad():
            pred_c, true_c = model(cb)
            h_clean_t = store["h"]                                       # [N, m]; N=n(+vnode rows)
            rmask = store.get("real_mask")
            T = int(pred_c.view(pred_c.shape[0], -1).shape[1])
            pred_row = pred_c.view(-1)
            g_out = torch.stack([
                torch.autograd.grad(pred_row[t], h_clean_t, retain_graph=True)[0]
                for t in range(T)
            ])
            L_c = metrics.per_graph_loss(pred_c, true_c, loss_fun).sum()
            g_loss_t = torch.autograd.grad(L_c, h_clean_t)[0]
        # Restrict h^L + readout gradients to REAL nodes (a global-VNode row has zero readout
        # gradient and would fail precondition [5] / mis-shape transport). No-op without a vnode.
        if rmask is not None:
            h_clean = h_clean_t.detach()[rmask]
            g_loss = g_loss_t.detach()[rmask]
            g_out = g_out.detach()[:, rmask]
        else:
            h_clean = h_clean_t.detach()
            g_loss = g_loss_t.detach()
            g_out = g_out.detach()
        assert h_clean.shape == (n, dim_h), f"h^L {tuple(h_clean.shape)} != {(n, dim_h)}"
        true_vec = true_c.detach().view(1, -1).float()
        L_clean = float(metrics.per_graph_loss(pred_c.detach(), true_c.detach(), loss_fun)[0].item())
        pred_clean_row = pred_c.detach().view(-1)
        g_spread_max = max(g_spread_max, float((g_loss - g_loss[0:1]).abs().max().item()))

        # completeness check: a FULL node relabelling (structure + content) is an isomorphism,
        # so a permutation-invariant GRIT must give the identical pooled prediction. Drift here
        # means perturb() missed a structural channel.
        if cc.verify and gi_pos < cc.verify_graphs and n >= 2:
            with torch.no_grad():
                rb = Batch.from_data_list([structural.full_relabel(base, 0, 1)]).to(device)
                pred_rel, _ = model(rb)
                relabel_inv_max = max(relabel_inv_max,
                                      float((pred_rel.view(-1) - pred_clean_row).abs().max().item()))

        # partners: K degree-matched v per anchor u (the marginalised nuisance).
        deg = structural.node_degrees(base.edge_index, n)
        partners = np.stack([structural.sample_partners(deg, u, K, rng, cc.partner_match)
                             for u in range(n)])            # [n, K]
        noop_mask = partners == np.arange(n)[:, None]        # v == u  => exact no-op
        noop_total += int(noop_mask.sum())
        partner_draws += int(noop_mask.size)

        S, R = n, n * K                                      # replica r = u*K + k (source-major)
        delta = torch.empty((R, n, dim_h), device=device, dtype=h_clean.dtype)
        pred_swap = torch.empty((R, T), device=device, dtype=torch.float32)
        if integrated:
            path_B = torch.empty((R, n), device="cpu", dtype=torch.float32)
            path_dL = torch.empty(R, device="cpu", dtype=torch.float32)

            def loss_from_pooled(pooled):
                pred = _pooled_head_predictions(model, pooled, true_vec)
                return metrics.per_graph_loss(
                    pred, true_vec.expand(int(pooled.shape[0]), -1), loss_fun
                )

        chunk = _plan_chunk(n, cc)
        r0 = 0
        while r0 < R:
            m = min(chunk, R - r0)
            while True:
                perts = b = pred_s = hs = full_rows = None
                clean_paths = swap_paths = path = None
                full_clean_pred = full_swap_pred = None
                try:
                    with torch.no_grad():
                        # replica 0 = clean within-batch baseline; replicas 1..m = the m swaps.
                        perts = []
                        for r in range(r0, r0 + m):
                            u, k = divmod(r, K)
                            perts.append(structural.perturb(base, int(u), int(partners[u, k]), mode))
                        b = Batch.from_data_list([base] + perts).to(device)
                        if cc.verify and struct_checked < cc.verify_graphs and r0 == 0:
                            u0, k0 = 0, 0
                            structural.verify_perturbation(
                                base, perts[0], u0, int(partners[u0, k0]), mode)
                            struct_checked += 1
                        pred_s, _ = model(b)
                        hs = store["h"]
                        assert hs.shape == ((m + 1) * n, dim_h)
                        hs = hs.view(m + 1, n, dim_h)
                        delta[r0:r0 + m] = hs[0:1] - hs[1:]          # within-batch clean baseline
                        full_rows = pred_s.view(m + 1, -1)
                        pred_swap[r0:r0 + m] = full_rows[1:].float()
                        if integrated:
                            clean_paths = hs[0:1].detach().clone().expand(m, -1, -1)
                            swap_paths = hs[1:].detach().clone()
                            full_clean_pred = full_rows[0:1].detach().clone()
                            full_swap_pred = full_rows[1:].detach().clone()
                    if integrated:
                        perts = b = pred_s = hs = full_rows = None
                        store.pop("h", None)
                        path = core.integrated_loss_carriage(
                            clean_paths,
                            swap_paths,
                            loss_from_pooled,
                            pooling=pooling,
                            atol=cc.integrated_atol,
                            rtol=cc.integrated_rtol,
                            max_intervals=cc.integrated_max_intervals,
                        )
                        path_B[r0:r0 + m] = path["carriage"].detach().float().cpu()
                        path_dL[r0:r0 + m] = path["loss_delta"].detach().float().cpu()
                        if cc.verify and gi_pos < cc.verify_graphs:
                            with torch.no_grad():
                                p_clean_chunk = core.pool_final_states(clean_paths, pooling)
                                p_swap_chunk = core.pool_final_states(swap_paths, pooling)
                                replay_clean = _pooled_head_predictions(
                                    model, p_clean_chunk, true_vec
                                )
                                replay_swap = _pooled_head_predictions(
                                    model, p_swap_chunk, true_vec
                                )
                                replay_error = max(
                                    float((replay_clean - full_clean_pred).abs().max().item()),
                                    float((replay_swap - full_swap_pred).abs().max().item()),
                                )
                                integrated_replay_max = max(
                                    integrated_replay_max, replay_error
                                )
                        integrated_converged.append(
                            _retain_or_reject_unconverged_paths(path, cc, "structural")
                        )
                        integrated_residual.append(
                            path["completeness_residual"].detach().cpu().numpy()
                        )
                        integrated_qerr.append(path["quadrature_error"].detach().cpu().numpy())
                        integrated_intervals.append(path["intervals"].detach().cpu().numpy())
                    break
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    perts = b = pred_s = hs = full_rows = None
                    clean_paths = swap_paths = path = None
                    full_clean_pred = full_swap_pred = None
                    store.pop("h", None)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if m == 1:
                        raise
                    m = max(1, m // 2)
                    chunk = m
                    log(f"[mem] CUDA OOM -> reducing replicas/forward to {m}")
            perts = b = pred_s = hs = full_rows = None
            clean_paths = swap_paths = path = None
            full_clean_pred = full_swap_pred = None
            r0 += m

        if device.type == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())

        # no-op partners (v == u): the within-batch baseline makes transport ~0.
        if noop_mask.any():
            nm = torch.as_tensor(noop_mask.reshape(-1), device=device)
            noop_max_dh = max(noop_max_dh, float(delta[nm].abs().max().item()))

        # per-source loss change (donor-average the LOSS over partners -- Jensen at the kink).
        L_swap = metrics.per_graph_loss(pred_swap, true_vec.expand(R, -1), loss_fun).cpu().numpy()
        dL_full_j = L_clean - L_swap.reshape(S, K).mean(axis=1)     # [S]  <0 = beneficial

        F_ij = core.functional_magnitude_from_delta(delta, g_out, S, K)       # [n, n], >=0
        C_loss = core.carriage_from_delta(delta, g_loss, S, K).cpu().numpy()  # [n, n]
        if integrated:
            B = path_B.view(S, K, n).mean(dim=1).t().double().numpy()
            dL_j = path_dL.view(S, K).mean(dim=1).double().numpy()
            target_mismatch = float(
                np.max(np.abs(dL_j - dL_full_j.astype(np.float64)))
            )
            integrated_full_loss_delta_max = max(
                integrated_full_loss_delta_max, target_mismatch,
            )
            if target_mismatch > cc.tol:
                raise RuntimeError(
                    f"Integrated donor-averaged path loss differs from the existing "
                    f"clean-alone dL target "
                    f"by {target_mismatch:.3e} > tol {cc.tol:.1e}. This is batch-context "
                    f"numerical drift; refusing to compare estimators against different "
                    f"loss changes."
                )
            denom_used = clamped = None
            net = np.abs(dL_j)
            cancel = np.abs(B).sum(axis=0) / np.maximum(net, 1e-12)
            integrated_cancellation.append(cancel[net > 1e-8])
        else:
            dL_j = dL_full_j
            B, denom_used, clamped = core.beneficial_attribute(
                C_loss, dL_j, denom=cc.beneficial_denom
            )

        add_sumC.append(C_loss.sum(axis=0))
        add_dyhat.append(dL_j)
        if integrated:
            source_residual = np.abs(B.sum(axis=0) - dL_j)
            source_tol = cc.integrated_atol + cc.integrated_rtol * np.abs(dL_j)
            if np.any(source_residual > source_tol):
                worst = int(np.argmax(source_residual - source_tol))
                raise RuntimeError(
                    f"Donor-averaged integrated carriage failed completeness for source "
                    f"{worst}: residual={source_residual[worst]:.3e}, allowed="
                    f"{source_tol[worst]:.3e}."
                )
            bexact_max = max(bexact_max, float(source_residual.max()))
        else:
            moved = np.abs(denom_used) > 1e-8
            clamp_moved += int(moved.sum())
            clamp_hit += int((moved & clamped).sum())
            ok = moved & ~clamped                   # slope clipping breaks exactness
            if ok.any():
                bexact_max = max(
                    bexact_max, float(np.abs(B.sum(axis=0)[ok] - dL_j[ok]).max())
                )

        # accumulate finite pairs: source j == anchor u, distance d(i, u) from the pristine SPD.
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
        if integrated:
            del path_B, path_dL
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (gi_pos + 1) % max(1, n_graphs // 10) == 0 or gi_pos == 0:
            log(f"[run] graph {gi_pos+1}/{n_graphs} (id={int(gi)}, n={n}, T={T}, {R} forwards) "
                f"| {time.perf_counter()-t_start:.1f}s")

    handle.remove()

    gid = np.concatenate(all_gid)
    pd_ = np.concatenate(all_d)
    pC = np.concatenate(all_C)
    pB = np.concatenate(all_B)
    F = np.concatenate(all_F)

    # ---- verification summary ----------------------------------------------------------
    tol = cc.tol
    a_sumC = np.concatenate(add_sumC)
    a_dyhat = np.concatenate(add_dyhat)
    if a_sumC.size > 2 and np.std(a_sumC) > 0 and np.std(a_dyhat) > 0:
        r = float(np.corrcoef(a_sumC, a_dyhat)[0, 1])
        slope = float(np.polyfit(a_sumC, a_dyhat, 1)[0])
    else:
        r, slope = float("nan"), float("nan")

    noop_frac = noop_total / max(partner_draws, 1)
    if integrated:
        ig_res = np.abs(np.concatenate(integrated_residual).astype(np.float64))
        ig_qerr = np.concatenate(integrated_qerr).astype(np.float64)
        ig_nint = np.concatenate(integrated_intervals).astype(np.int64)
        ig_converged = np.concatenate(integrated_converged).astype(bool)
        failure_stats = _integrated_failure_stats(ig_converged, ig_res, ig_qerr)
        nonempty_cancel = [x for x in integrated_cancellation if x.size]
        ig_cancel = (
            np.concatenate(nonempty_cancel).astype(np.float64)
            if nonempty_cancel else np.asarray([], dtype=np.float64)
        )
        checks.update({
            "integrated_atol": float(cc.integrated_atol),
            "integrated_rtol": float(cc.integrated_rtol),
            "integrated_max_intervals": int(cc.integrated_max_intervals),
            "integrated_max_unconverged_fraction": float(
                cc.integrated_max_unconverged_fraction
            ),
            "integrated_unconverged_error_cap": float(
                cc.integrated_unconverged_error_cap
            ),
            "integrated_completeness_residual_p95": float(np.quantile(ig_res, 0.95)),
            "integrated_completeness_residual_max": float(ig_res.max()),
            "integrated_quadrature_error_p95": float(np.quantile(ig_qerr, 0.95)),
            "integrated_quadrature_error_max": float(ig_qerr.max()),
            "integrated_intervals_p95": float(np.quantile(ig_nint, 0.95)),
            "integrated_intervals_max": int(ig_nint.max()),
            "integrated_at_max_intervals_fraction": float(
                np.mean(ig_nint >= int(cc.integrated_max_intervals))
            ),
            "integrated_endpoint_replay_max_abs": (
                float(integrated_replay_max) if cc.verify else None
            ),
            "integrated_vs_clean_alone_dL_j_max": float(
                integrated_full_loss_delta_max
            ),
            "integrated_cancellation_ratio_p95": (
                float(np.quantile(ig_cancel, 0.95)) if ig_cancel.size else None
            ),
            "integrated_cancellation_ratio_max": (
                float(ig_cancel.max()) if ig_cancel.size else None
            ),
            **failure_stats,
        })
    checks.update({
        "intervention": "structural",
        "structural_mode": mode,
        "partner_match": cc.partner_match,
        "g_spread_max": float(g_spread_max),
        "relabel_invariance_max_abs": float(relabel_inv_max),
        "partner_noop_fraction": float(noop_frac),
        "partner_noop_max_abs_dh": float(noop_max_dh),
        "structure_verified_graphs": int(struct_checked),
        "unreachable_pairs_excluded": int(unreachable_total),
        "additivity_pearson_r": r,
        "additivity_slope": slope,
        "beneficial_exactness_max": float(bexact_max),
    })
    if device.type == "cuda":
        checks["peak_cuda_gib"] = float(peak_mem / 1024**3)

    log("\n" + "=" * 84 + f"\nVERIFICATION (structural: {mode})\n" + "=" * 84)
    log(f"  [param]  parameters              : {n_params}"
        + (f" (expected {task.expected_params})" if task.expected_params else ""))
    if checks["test_metric"] is not None:
        log(f"  [4]  test {checks['test_metric_name']} from checkpoint : {checks['test_metric']:.5f}")
    log(f"  [5]  max_i |g_i - g_1| (g_loss)  : {g_spread_max:.3e}  (pooling => must be ~0)")
    if cc.verify:
        log(f"  [inv] full-relabel |dpred|       : {relabel_inv_max:.3e}  (isomorphism => must be ~0; "
            f"catches a missed structural channel)")
        log(f"  [8]  content invariance + struct equivariance verified on {struct_checked} graph(s)")
    log(f"  [7]  no-op partners (v==u)       : {noop_total}/{partner_draws} "
        f"({100*noop_frac:.1f}%) max|dh|={noop_max_dh:.3e} (within-batch => ~0)")
    log(f"  [9]  loss additivity r/slope     : r={r:.4f}, slope={slope:.4f}  (sum_i C_loss vs dL_j)")
    log(f"  [10] unreachable pairs (excluded): {unreachable_total}")
    if integrated:
        log(f"  [11] beneficial completeness     : max_j|sum_i B - dL_j| = "
            f"{bexact_max:.3e} (path integral; no clipping)")
        log(f"  [11b] path quadrature residual   : p95={checks['integrated_completeness_residual_p95']:.3e}, "
            f"max={checks['integrated_completeness_residual_max']:.3e}; carrier-error "
            f"p95={checks['integrated_quadrature_error_p95']:.3e}; intervals "
            f"p95/max={checks['integrated_intervals_p95']:.0f}/{checks['integrated_intervals_max']}; "
            f"at-cap={100*checks['integrated_at_max_intervals_fraction']:.1f}%")
        log(f"  [11c] clean-target alignment     : max|dL_path-dL_clean-alone|="
            f"{integrated_full_loss_delta_max:.3e}")
        if cc.verify:
            log(f"  [11d] pooled-head endpoint replay: max|dpred|={integrated_replay_max:.3e}")
        n_failed = checks["integrated_unconverged_count"]
        n_paths = checks["integrated_path_count"]
        fail_rate = checks["integrated_unconverged_fraction"]
        if n_failed:
            log(f"  [11e] capped paths retained      : {n_failed}/{n_paths} "
                f"({100*fail_rate:.4f}%); completeness residual median/p95/max="
                f"{checks['integrated_unconverged_completeness_residual_median']:.3e}/"
                f"{checks['integrated_unconverged_completeness_residual_p95']:.3e}/"
                f"{checks['integrated_unconverged_completeness_residual_max']:.3e}; "
                f"carrier-error median/p95/max="
                f"{checks['integrated_unconverged_carrier_error_median']:.3e}/"
                f"{checks['integrated_unconverged_carrier_error_p95']:.3e}/"
                f"{checks['integrated_unconverged_carrier_error_max']:.3e}")
        else:
            log(f"  [11e] capped paths retained      : 0/{n_paths} (0.0000%)")
    else:
        log(f"  [11] beneficial exactness        : max_j|sum_i B - dL_j| = "
            f"{bexact_max:.3e} (must be ~0, unclamped sources)")
    if cc.beneficial_denom == "slope":
        log(f"  [11b] slope-clip activation      : {clamp_hit}/{clamp_moved} moved sources "
            f"({(100.0*clamp_hit/max(1,clamp_moved)):.1f}%) hit |s_j|>1 "
            f"(finite intervention not represented by the clean tangent)")
    if device.type == "cuda":
        log(f"  [mem] peak CUDA allocated        : {checks['peak_cuda_gib']:.2f} GiB")

    noise_tol = max(tol, cc.float_noise_tol)
    if g_spread_max > tol:
        log(f"  [!] g_i varies by {g_spread_max:.3e} > tol; readout not pooling->MLP as assumed.")
    if noop_max_dh > noise_tol:
        raise RuntimeError(f"No-op partner (v==u) gave |dh|={noop_max_dh:.3e} > {noise_tol:.1e}: a "
                           f"self-transposition shares the within-batch clean baseline, so this must "
                           f"be ~0. A larger value means the perturb/batch wiring is wrong or the "
                           f"model is not in eval().")
    if cc.verify and relabel_inv_max > noise_tol:
        raise RuntimeError(f"Full-relabel prediction moved by {relabel_inv_max:.3e} > {noise_tol:.1e}: "
                           f"a structure+content relabelling is an isomorphism, so the pooled "
                           f"prediction must be invariant. The transposition is missing a "
                           f"structure-derived channel (extend structural.NODE_STRUCT_ATTRS / "
                           f"PAIR_TENSORS).")
    if integrated and cc.verify and integrated_replay_max > tol:
        raise RuntimeError(
            f"Pooled-head endpoint replay moved predictions by {integrated_replay_max:.3e} "
            f"> tol {tol:.1e}; integrated carriage is not reaching the model's exact "
            f"post-transformer readout."
        )
    if (integrated and checks["integrated_unconverged_fraction"]
            > float(cc.integrated_max_unconverged_fraction)):
        raise RuntimeError(
            f"Integrated quadrature retained {checks['integrated_unconverged_count']}/"
            f"{checks['integrated_path_count']} capped paths "
            f"({100*checks['integrated_unconverged_fraction']:.4f}%), exceeding "
            f"integrated_max_unconverged_fraction="
            f"{100*float(cc.integrated_max_unconverged_fraction):.4f}%."
        )
    if not integrated and bexact_max > tol:
        raise RuntimeError(f"Beneficial exactness violated: max_j|sum_i B - dL_j|={bexact_max:.3e} "
                           f"> tol {tol:.1e}; carrier attribution or loss donor-average miswired.")

    label = f"Structural {mode.replace('_', '-')} (anchor u, partner {cc.partner_match}-matched)"
    meta = {
        "task": task.name, "title": task.title,
        "grit_repo": task.grit_repo, "grit_commit": task.grit_commit,
        "config": cc.config_file, "checkpoint": str(ckpt_path), "checkpoint_epoch": ckpt_epoch,
        "eval_split": cc.eval_split, "donor_split": f"within-graph ({cc.partner_match}-matched)",
        "num_graphs": int(np.unique(gid).size), "donors_K": K,
        "graph_select": cc.graph_select, "analysis_seed": cc.analysis_seed,
        "test_metric": checks["test_metric"], "test_metric_name": checks["test_metric_name"],
        "val_metric": checks.get("val_metric"), "val_metric_name": checks.get("val_metric_name"),
        "paper_metric": task.paper_metric, "loss_units": metrics.loss_units(loss_fun),
        "beneficial_denom": cc.beneficial_denom,
        "intervention": "structural", "structural_mode": mode,
        "intervention_label": label,
        "fig_suptitle": f"{label} interventions on GRIT",
        "fig_tag": f"structural_{mode}",
        "swap_word": "partner-swaps",
    }
    if integrated:
        meta.update({
            "integrated_atol": float(cc.integrated_atol),
            "integrated_rtol": float(cc.integrated_rtol),
            "integrated_max_intervals": int(cc.integrated_max_intervals),
            "integrated_max_unconverged_fraction": float(
                cc.integrated_max_unconverged_fraction
            ),
            "integrated_unconverged_error_cap": float(
                cc.integrated_unconverged_error_cap
            ),
        })
    return {
        "graph_id": gid,
        "carrier_i": np.concatenate(all_i), "source_j": np.concatenate(all_j),
        "distance": pd_, "C": pC, "B": pB, "F": F,
        "additivity_sumC": a_sumC, "additivity_dyhat": a_dyhat,
        "checks": checks, "meta": meta,
    }
