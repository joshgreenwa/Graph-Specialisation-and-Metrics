"""Load a pretrained GRIT checkpoint and capture per-head attention + transport.

This is the model half of the per-head specialisation-score methodology
(SPECIALISATION_SCORES.md), lifted from the spec-lite ``Net`` onto the REAL GRIT
transformer. It deliberately MIRRORS ``carriage.grit_runner`` for the load path (config ->
loaders -> model -> checkpoint -> eval-metric sanity check) rather than importing its body,
so the validated carriage path is untouched and this whole package can be deleted as a unit.

The one thing this module adds over ``grit_runner`` is a hook layer that exposes, per GRIT
attention layer ``l`` and head ``h``:

  * the TRANSPORT site  o^{lh}_i = ``batch.wV``  [N, n_heads, out_dim]  (kept IN-GRAPH so
    ``phi^{lh}_i = d yhat / d o^{lh}_i`` is a single ``autograd.grad`` call); and
  * the SELECTION site  a^{lh}_{i<-j} = ``batch.attn``  [E, n_heads, 1]  (post-softmax,
    detached) with ``edge_index`` (src = row 0, dest = row 1).

In ``grit/layer/grit_layer.py`` the attention module's ``forward`` returns ``(h_out, e_out)``
with ``h_out is batch.wV`` (it includes the edge-enhance VeRow term), and writes
``batch.attn`` before returning. A forward hook on each ``layer.attention`` therefore captures
both. ``batch.wV`` is overwritten every layer, so we grab it during that layer's hook.

Ablating head ``(l, h)`` = zero that head's routed value: a forward hook on
``layers[l].attention`` that returns ``(h_out with [:, h, :]=0, e_out)``. The layer consumes
the returned value, so the head delivers no message while every other head is unchanged.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..carriage import metrics
from ..carriage.content import FullNodeContentAdapter
from ..carriage.env import log


def _graph_major_order(graph_index):
    """Stable row order grouping real nodes and an appended VNode by graph id."""
    import torch

    return torch.argsort(graph_index, stable=True)


def _enable_grit_reregistration() -> None:
    """Backward-compatible alias for the shared carriage/specialisation registry guard."""
    from ..carriage.env import enable_grit_reregistration

    enable_grit_reregistration()


@dataclass
class SpecConfig:
    """Everything the per-head analysis needs for one checkpoint (subset of CarriageConfig)."""

    ckpt: str
    out_dir: str
    dataset_dir: str
    config_file: str
    accelerator: str = "cuda:0"
    seed: int = 42
    num_threads: int = 4
    eval_split: str = "test"
    donor_split: str = "test"
    eval_metric: bool = True
    allow_param_count_drift: bool = False
    tol: float = 1e-4
    float_noise_tol: float = 5e-3
    # analysis knobs (used by scores/ablation, carried here for provenance)
    num_graphs: int = 200          # graphs scored per model
    donors: int = 8                # K donors/partners per source
    ablation_graphs: int = 256     # graphs for the ablation impact sweep
    analysis_seed: int = 0
    partner_match: str = "degree"
    content_adapter: object = field(default_factory=FullNodeContentAdapter)
    resume: bool = True
    checkpoint_every: int = 4


class GritHeadModel:
    """A loaded GRIT model wrapped for per-head capture and ablation.

    Attributes set after :meth:`load`:
      model            the GraphGym model wrapper (``model.model`` is GritTransformer).
      eval_ds/donor_ds the eval / donor datasets (PyG in-memory).
      attn_layers      [L] the ``MultiHeadAttentionLayerGritSparse`` modules, in order.
      L, H, dh, dim_h  #layers, #heads, per-head dim, hidden dim.
      loss_fun, T      training loss id and #targets (T=1 for ZINC scalar regression).
      test_metric      recomputed eval metric (checkpoint-load sanity check).
    """

    def __init__(self, task, sc: SpecConfig):
        self.task = task
        self.sc = sc
        self.adapter = sc.content_adapter
        self.model = None
        self.device = None
        self.checks: dict = {}

    # ---- load path (mirrors carriage.grit_runner) ------------------------------------
    def load(self) -> "GritHeadModel":
        import torch
        from torch_geometric import seed_everything
        from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
        from torch_geometric.graphgym.loader import create_loader
        from torch_geometric.graphgym.model_builder import create_model
        from torch_geometric.graphgym.utils.comp_budget import params_count

        # Allow the 2nd task to re-import a DIFFERENT (patched) GRIT clone without the GraphGym
        # registry raising on the duplicate @register_* keys; MUST run before `import grit`.
        _enable_grit_reregistration()
        import grit  # noqa: F401  registers loaders/encoders/layers/heads

        sc = self.sc
        out_dir = Path(sc.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        scratch = out_dir / "_graphgym_scratch"
        scratch.mkdir(parents=True, exist_ok=True)

        set_cfg(cfg)
        cfg.set_new_allowed(True)
        cfg.work_dir = os.getcwd()
        opts = [
            "out_dir", str(scratch), "dataset.dir", str(sc.dataset_dir),
            "seed", str(sc.seed), "accelerator", sc.accelerator,
            "wandb.use", "False", "mlflow.use", "False",
            "train.auto_resume", "False", "train.enable_ckpt", "False",
            "num_threads", str(sc.num_threads),
        ]
        load_cfg(cfg, argparse.Namespace(cfg_file=sc.config_file, opts=opts))
        assert Path(cfg.run_dir).resolve() == scratch.resolve(), \
            f"cfg.run_dir unexpectedly {cfg.run_dir}; refusing to run near training outputs."

        device = torch.device(sc.accelerator if torch.cuda.is_available() else "cpu")
        cfg.device = str(device)
        if device.type != "cuda":
            log("[warn] CUDA unavailable; running on CPU. This will be slow.")
        torch.set_num_threads(cfg.num_threads)
        seed_everything(cfg.seed)
        self.cfg = cfg
        self.device = device

        # carriage preconditions (shared-g pooling, RRWP content-invariant, scalar head).
        from ..carriage.grit_runner import check_carriage_preconditions
        check_carriage_preconditions(cfg, sc.tol, self.checks)

        log("\n[data] Building loaders (GRIT recomputes RRWP in-memory; not cached to disk).")
        loaders = create_loader()
        assert len(loaders) == 3, f"Expected [train, val, test] loaders, got {len(loaders)}"
        split_of = {"train": 0, "val": 1, "test": 2}
        self.loaders = loaders
        self.eval_ds = loaders[split_of[sc.eval_split]].dataset
        self.donor_ds = loaders[split_of[sc.donor_split]].dataset
        log(f"[data] eval={sc.eval_split} ({len(self.eval_ds)}) | "
            f"donors={sc.donor_split} ({len(self.donor_ds)})")

        model = create_model()
        n_params = params_count(model)
        self.checks["num_parameters"] = int(n_params)
        if self.task.expected_params is not None and n_params != self.task.expected_params:
            msg = (f"[param-check:ERROR] {n_params} != expected {self.task.expected_params} "
                   f"for task {self.task.name!r}.")
            if not sc.allow_param_count_drift:
                raise RuntimeError(msg)
            log(msg + " Continuing (allow_param_count_drift).")
        else:
            log(f"[param-check] parameters: {n_params}"
                + (f" (matches expected {self.task.expected_params})" if self.task.expected_params else ""))

        # checkpoint (strict load, with the same prefix-fixups as grit_runner).
        ckpt_path = Path(sc.ckpt)
        log(f"\n[ckpt] Loading: {ckpt_path}")
        blob = torch.load(str(ckpt_path), map_location="cpu")
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
        log(f"[ckpt] Loaded strictly (key handling: {how}).")

        model.to(device)
        model.eval()  # load-bearing: BN uses running stats -> batch-size invariant.
        assert type(model.model).__name__ == "GritTransformer", \
            f"Expected GritTransformer, got {type(model.model).__name__}"
        self.model = model

        # dims from the config (asserted against the model on first capture).
        self.dim_h = int(cfg.gnn.dim_inner)
        self.L = int(cfg.gt.layers)
        self.H = int(cfg.gt.n_heads)
        self.dh = self.dim_h // self.H
        self.loss_fun = str(cfg.model.loss_fun)
        self.attn_layers = [layer.attention for layer in model.model.layers]
        assert len(self.attn_layers) == self.L, \
            f"found {len(self.attn_layers)} attention layers, cfg.gt.layers={self.L}"
        for a in self.attn_layers:
            assert int(a.num_heads) == self.H and int(a.out_dim) == self.dh, \
                f"head geometry mismatch: layer has {a.num_heads}h x {a.out_dim}d, cfg says {self.H}x{self.dh}"

        # recompute the eval metric (strongest load check).
        metric_name = (getattr(self.task, "metric_name", None)
                       or (self.task.paper_metric[0] if self.task.paper_metric else "metric"))
        if sc.eval_metric:
            t0 = time.perf_counter()
            preds, trues = self._collect_preds(loaders[2])
            test_metric = float(self.task.metric_fn(preds, trues))
            pm = f" (paper {self.task.paper_metric[0]} ~{self.task.paper_metric[1]})" if self.task.paper_metric else ""
            log(f"[verify] test {metric_name} recomputed from checkpoint: {test_metric:.5f}{pm} "
                f"[{time.perf_counter()-t0:.1f}s]")
            bad = (test_metric < self.task.metric_abort) if self.task.metric_higher_better \
                else (test_metric > self.task.metric_abort)
            if bad:
                arrow = "below" if self.task.metric_higher_better else "above"
                raise RuntimeError(
                    f"Recomputed test {metric_name} {test_metric:.5f} is {arrow} the abort "
                    f"threshold {self.task.metric_abort}. The checkpoint almost certainly did "
                    f"not load correctly. Refusing to report specialisation scores.")
            self.test_metric = test_metric
            self.checks["test_metric"] = test_metric
            # Validation metric alongside test (loaders[1]=val), for reporting only -- no abort.
            t0 = time.perf_counter()
            vpreds, vtrues = self._collect_preds(loaders[1])
            val_metric = float(self.task.metric_fn(vpreds, vtrues))
            log(f"[verify] val {metric_name} recomputed from checkpoint: {val_metric:.5f} "
                f"[{time.perf_counter()-t0:.1f}s]")
            self.val_metric = val_metric
            self.checks["val_metric"] = val_metric
        else:
            self.test_metric = None
            self.checks["test_metric"] = None
            self.val_metric = None
            self.checks["val_metric"] = None
        self.checks["test_metric_name"] = metric_name
        self.checks["val_metric_name"] = metric_name
        return self

    def _collect_preds(self, loader):
        import torch
        ps, ts = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                pred, true = self.model(batch)
                ps.append(pred.detach().cpu().numpy().reshape(pred.shape[0], -1))
                ts.append(true.detach().cpu().numpy().reshape(true.shape[0], -1))
        return np.concatenate(ps), np.concatenate(ts)

    # ---- per-head capture ------------------------------------------------------------
    def capture(self, batch, *, want_grad: bool, want_attn: bool = False,
                include_virtual_transport: bool = False):
        """Run one forward and capture per-layer wV (transport) and optionally attn.

        Args:
            batch:      a PyG ``Batch`` already on ``self.device``.
            want_grad:  keep wV in the autograd graph (for phi = d yhat / d wV). Uses
                         ``torch.enable_grad``; caller takes the gradients.
            want_attn:  also stash detached ``batch.attn`` + ``edge_index`` per layer.
            include_virtual_transport: for a no-grad VNode replica batch, retain its virtual row
                         and reorder rows graph-major so ``[replica, n+1, H, dh]`` is exact.

        Returns:
            dict(pred=[G,T] tensor, wV=[L] list of [N,H,dh] tensors, attn=[L] list of
                 [E,H] tensors or None, edge_index=[2,E] or None). N is the total node count
                 of the batch; reshape to [reps, n, ...] outside for same-``n`` replica batches.
        """
        import torch

        cap = {"wV": [None] * self.L, "attn": [None] * self.L,
               "edge_index": None, "node_graph": None}
        idx_of = {id(a): l for l, a in enumerate(self.attn_layers)}

        def _hook(module, inputs, output):
            l = idx_of[id(module)]
            h_out = output[0] if isinstance(output, (tuple, list)) else output
            cap["wV"][l] = h_out                       # [N, H, dh], in-graph if want_grad
            if cap["node_graph"] is None:
                cap["node_graph"] = inputs[0].batch.detach().clone()
            if want_attn:
                b = inputs[0]
                cap["attn"][l] = b.attn.detach().squeeze(-1)   # [E, H]
                if cap["edge_index"] is None:
                    cap["edge_index"] = b.edge_index.detach()

        handles = [a.register_forward_hook(_hook) for a in self.attn_layers]
        try:
            ctx = torch.enable_grad() if want_grad else torch.no_grad()
            with ctx:
                pred, true = self.model(batch)
        finally:
            for hd in handles:
                hd.remove()

        # Global-VNode models append all virtual rows after all real rows. For ordinary callers we
        # retain the historical real-only capture. The specialisation score, however, must include
        # the VNode as an internal carrier because its earlier-layer wV has nonzero readout gradient.
        # Replica batches are reordered graph-major so a [replica,n+1,H,dh] view is valid.
        real_mask = getattr(batch, "real_node_mask", None)
        if real_mask is not None and not want_grad:
            if include_virtual_transport:
                # Save this inside the attention hook: GritTransformer strips VNode entries from
                # ``batch.batch`` just before pooling, after the wV rows have already been made.
                order = _graph_major_order(cap["node_graph"])
                cap["wV"] = [w[order] for w in cap["wV"]]
                cap["node_graph"] = cap["node_graph"][order]
            else:
                cap["wV"] = [w[real_mask] for w in cap["wV"]]
        for l in range(self.L):
            assert cap["wV"][l] is not None, f"layer {l} attention hook did not fire"
            assert cap["wV"][l].dim() == 3 and cap["wV"][l].shape[1:] == (self.H, self.dh), \
                f"wV[{l}] shape {tuple(cap['wV'][l].shape)} != [N,{self.H},{self.dh}]"
        return {"pred": pred, "true": true, "wV": cap["wV"], "attn": cap["attn"],
                "edge_index": cap["edge_index"], "real_mask": real_mask,
                "node_graph": cap["node_graph"]}

    # ---- ablation forward ------------------------------------------------------------
    def collect_preds_ablated(self, data_groups, ablations=None):
        """Per-graph predictions over fixed groups of graphs, optionally ablating heads.

        Args:
            data_groups: list of lists of PyG ``Data`` objects. A FRESH ``Batch`` is built from
                         each group on every call -- GRIT's encoders rebind ``batch.x`` /
                         ``batch.edge_index`` in place, so a reused Batch would be corrupted on
                         the second forward (mirrors ``grit_runner`` rebuilding every forward).
            ablations:   iterable of (layer, head) to zero (their routed value wV[:, head, :]).
                         None/empty = clean.

        Returns:
            [num_graphs, T] numpy array of pooled predictions; graph order = concatenation of the
            groups in order (so it aligns with the caller's fixed graph order).
        """
        import torch
        from torch_geometric.data import Batch

        ablations = list(ablations or [])
        by_layer: dict[int, list[int]] = {}
        for (l, h) in ablations:
            by_layer.setdefault(int(l), []).append(int(h))

        handles = []
        for l, heads in by_layer.items():
            heads_t = torch.as_tensor(sorted(set(heads)), device=self.device, dtype=torch.long)

            def _mk(hs):
                def _hook(module, inputs, output):
                    h_out, e_out = output
                    h_out = h_out.clone()
                    h_out[:, hs, :] = 0.0
                    return (h_out, e_out)
                return _hook

            handles.append(self.attn_layers[l].register_forward_hook(_mk(heads_t)))

        preds = []
        try:
            with torch.no_grad():
                for group in data_groups:
                    b = Batch.from_data_list(list(group)).to(self.device)
                    pred, _ = self.model(b)
                    preds.append(pred.detach().cpu().numpy().reshape(pred.shape[0], -1))
        finally:
            for hd in handles:
                hd.remove()
        return np.concatenate(preds, axis=0)
