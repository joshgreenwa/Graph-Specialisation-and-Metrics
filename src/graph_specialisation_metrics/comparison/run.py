"""One-call cross-model comparison: evaluate, carriage, specialise, cache, and plot.

    from graph_specialisation_metrics.comparison import run_all
    run_all()                              # all 5 ZINC models, semantic + structural, full run
    run_all(force=False)                   # re-run only stages whose cache is missing
    from graph_specialisation_metrics.comparison import build_figures
    build_figures(drop_vnode=True)         # re-draw the deliverables from cache, no re-compute

``run_all`` drives the EXISTING carriage (``carriage.colab.run``) and specialisation
(``specialisation.colab.run``) orchestrators once per model, skipping any stage whose cached
artefact already exists on Drive (so a re-run is incremental), then builds the cross-model
deliverables from that cache:

* (i)   the cache itself -- per-model carriage summaries (F/B curves), carriage pair npz, and
        per-head specialisation score matrices -- plus a consolidated ``comparison`` manifest;
* (ii)  ``fig_spec_scatter_grid.png``          -- side-by-side S_str-vs-S_sem scatter per model;
* (iii) ``fig_carriage_smallmult_<intv>.png``  -- functional/beneficial carriage per model,
        standardised (reference-fixed) y-axis;
* (iv)  ``fig_carriage_overlay_<intv>.png``    -- functional/beneficial carriage overlaid across
        models for direct comparison;
plus ``fig_performance.png`` + ``performance.json`` for the val/test evaluation.

Every figure accepts an ``include`` / ``exclude`` list and ``drop_vnode`` so a final figure can
drop or keep specific methods without recomputing anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from ..carriage import env
from ..carriage.env import log
from . import data as _data
from . import plots as _plots

# Torch-free collate roots (mirror carriage.colab / specialisation.colab DEFAULT_COLLATE_DIR).
CARRIAGE_COLLATE = _data.CARRIAGE_COLLATE_DIR
SPEC_COLLATE = _data.SPEC_COLLATE_DIR
DEFAULT_COMPARISON_DIR = _data.COMPARISON_DIR

# The new k-hop / VNode checkpoints the user pointed at (recovery `latest.ckpt`). Dense / 1-hop
# are left to auto-discovery (their standard GraphGym ckpt/ dirs). find_checkpoint also has a
# recovery-checkpoint fallback, so a task omitted here still resolves if its drive_dir is set;
# these explicit entries pin the exact files the user specified.
DEFAULT_CKPTS = {
    "zinc_2hop": "/content/drive/MyDrive/grit_zinc_2hop/results/_recovery_checkpoints/"
                 "seed0_ColabDrive.2hop.GRITwRRWP/latest.ckpt",
    "zinc_1hop_vnode": "/content/drive/MyDrive/grit_zinc_1hop_vnode/results/_recovery_checkpoints/"
                       "seed0_ColabDrive.1hop.GRITwRRWP.VNode/latest.ckpt",
    "zinc_2hop_vnode": "/content/drive/MyDrive/grit_zinc_2hop_vnode/results/_recovery_checkpoints/"
                       "seed0_ColabDrive.2hop.GRITwRRWP.VNode/latest.ckpt",
}


def _mount_drive(mount_point: str = "/content/drive") -> None:
    try:
        from google.colab import drive
        log(f"[drive] Mounting Google Drive at {mount_point} ...")
        drive.mount(mount_point, force_remount=False)
    except Exception:  # noqa: BLE001
        log("[drive] google.colab unavailable; assuming a non-Colab run (paths used as-is).")


# --------------------------------------------------------------------------------------
# evaluation table (deliverable 1), read from the cached carriage/spec metadata
# --------------------------------------------------------------------------------------

def _metric_from_summary(summary: Optional[dict]) -> Optional[dict]:
    if not summary:
        return None
    meta = summary.get("meta", {})
    if meta.get("test_metric") is None and meta.get("val_metric") is None:
        return None
    return {"val": meta.get("val_metric"), "test": meta.get("test_metric"),
            "name": meta.get("test_metric_name") or meta.get("val_metric_name") or "MAE"}


def performance_table(tasks: Sequence[str] = _data.DEFAULT_TASKS, *,
                      carriage_collate: str = CARRIAGE_COLLATE,
                      spec_collate: str = SPEC_COLLATE,
                      structural_mode: str = "transposition") -> dict:
    """{task: {"val","test","name"}} read from the cached carriage meta (falls back to spec stats)."""
    out: dict = {}
    for t in tasks:
        m = _metric_from_summary(_data.load_carriage_summary(
            _data.carriage_summary_path(carriage_collate, t, "semantic")))
        if m is None:
            m = _metric_from_summary(_data.load_carriage_summary(
                _data.carriage_summary_path(carriage_collate, t, "structural", structural_mode)))
        if m is None:
            stats = _data.load_spec_stats(_data.spec_stats_path(spec_collate, t))
            if stats and (stats.get("val_metric") is not None or stats.get("test_metric") is not None):
                m = {"val": stats.get("val_metric"), "test": stats.get("test_metric"),
                     "name": stats.get("test_metric_name") or "MAE"}
        if m is not None:
            out[t] = m
    return out


# --------------------------------------------------------------------------------------
# figure builder (pure cache -> figures; re-run this to restyle/subset without recompute)
# --------------------------------------------------------------------------------------

def build_figures(tasks: Sequence[str] = _data.DEFAULT_TASKS, *,
                  carriage_collate: str = CARRIAGE_COLLATE,
                  spec_collate: str = SPEC_COLLATE,
                  comparison_dir: str = DEFAULT_COMPARISON_DIR,
                  interventions: Sequence[str] = ("semantic", "structural"),
                  structural_mode: str = "transposition",
                  include: Optional[Sequence[str]] = None,
                  exclude: Optional[Sequence[str]] = None,
                  drop_vnode: bool = False,
                  include_self_B: bool = False,
                  display: bool = False) -> dict:
    """Build the cross-model deliverables purely from the cache. No GPU, no re-run.

    ``tasks`` is the reference set (used for the standardised y-axes and global score norms);
    ``include`` / ``exclude`` / ``drop_vnode`` choose which methods are *drawn*. Returns a dict
    of figure keys -> saved paths.
    """
    out_dir = Path(comparison_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shown = _data.select_methods(tasks, include=include, exclude=exclude, drop_vnode=drop_vnode)
    log(f"[figures] reference methods: {list(tasks)}")
    log(f"[figures] shown methods:     {shown}")
    metrics = performance_table(tasks, carriage_collate=carriage_collate,
                                spec_collate=spec_collate, structural_mode=structural_mode)

    figs: dict = {}
    made = []  # (key, fig) to optionally display/close

    # (ii) specialisation scatter grid ---------------------------------------------------
    scores_ref = _data.load_scores_by_task(spec_collate, tasks)
    if scores_ref:
        gsem, gstr = _plots.global_norms(scores_ref, list(scores_ref.keys()))
        shown_scores = [t for t in shown if t in scores_ref]
        if shown_scores:
            fig, p = _plots.plot_spec_scatter_grid(
                scores_ref, shown_scores, out_dir / "fig_spec_scatter_grid.png",
                gsem=gsem, gstr=gstr, metrics_by_task=metrics)
            figs["spec_scatter_grid"] = p
            made.append(fig)
    else:
        log("[figures] no specialisation score caches found; skipping the scatter grid.")

    # (iii) + (iv) carriage figures per intervention -------------------------------------
    for intv in interventions:
        ref_curves = _data.load_carriage_curves_by_task(carriage_collate, tasks, intv, structural_mode)
        if not ref_curves:
            log(f"[figures] no cached {intv} carriage curves; skipping its F/B figures.")
            continue
        shown_curves_tasks = [t for t in shown if t in ref_curves]
        if not shown_curves_tasks:
            continue
        # (iii) standardised small multiples
        fig, p = _plots.plot_carriage_small_multiples(
            ref_curves, shown_curves_tasks, out_dir / f"fig_carriage_smallmult_{intv}.png",
            intervention=intv, ref_curves_by_task=ref_curves, include_self_B=include_self_B,
            metrics_by_task=metrics)
        figs[f"carriage_smallmult_{intv}"] = p
        made.append(fig)
        # (iv) overlay
        fig, p = _plots.plot_carriage_overlay(
            ref_curves, shown_curves_tasks, out_dir / f"fig_carriage_overlay_{intv}.png",
            intervention=intv, ref_curves_by_task=ref_curves, include_self_B=include_self_B,
            standardise=True)
        figs[f"carriage_overlay_{intv}"] = p
        made.append(fig)

    # (1) performance bar ----------------------------------------------------------------
    if metrics:
        shown_metrics = [t for t in shown if t in metrics]
        if shown_metrics:
            name = next(iter(metrics.values())).get("name", "MAE")
            fig, p = _plots.plot_performance(metrics, shown_metrics,
                                             out_dir / "fig_performance.png", metric_name=name)
            figs["performance"] = p
            made.append(fig)

    (out_dir / "performance.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "figures_manifest.json").write_text(json.dumps({
        "reference_tasks": list(tasks), "shown_tasks": shown, "interventions": list(interventions),
        "structural_mode": structural_mode, "figures": figs,
        "performance": str(out_dir / "performance.json"),
    }, indent=2), encoding="utf-8")

    if display:
        try:
            from IPython.display import Image, display as _disp
            for _, p in figs.items():
                _disp(Image(filename=p))
        except Exception:  # noqa: BLE001
            pass
    try:
        import matplotlib.pyplot as plt
        for fig in made:
            plt.close(fig)
    except Exception:  # noqa: BLE001
        pass

    log(f"[figures] wrote {len(figs)} figure(s) to {out_dir}")
    return figs


# --------------------------------------------------------------------------------------
# full orchestrator: compute-with-cache-skip, then build figures
# --------------------------------------------------------------------------------------

def run_all(tasks: Sequence[str] = _data.DEFAULT_TASKS, *,
            ckpts: Optional[dict] = None,
            force: bool = False,
            interventions: Sequence[str] = ("semantic", "structural"),
            # carriage knobs (default to the user's structural example)
            carriage_num_graphs: int = 128,
            carriage_donors: int = 64,
            beneficial_denom: str = "integrated",
            integrated_max_intervals: int = 256,
            integrated_atol: float = 5e-4,
            integrated_unconverged_error_cap: float = 5e-3,
            integrated_max_unconverged_fraction: float = 1e-2,
            structural_mode: str = "transposition",
            partner_match: str = "degree",
            # any other carriage.colab.run option (e.g. {"integrated_rtol": 1e-4, "bin_strategy":
            # "hop", "central": "median", "analysis_seed": 1, "n_boot": 2000}); overrides the above.
            carriage_kwargs: Optional[dict] = None,
            # specialisation knobs
            run_specialisation: bool = True,
            spec_num_graphs: int = 200,
            spec_donors: int = 8,
            with_attn_routing: bool = True,
            spec_kwargs: Optional[dict] = None,   # any other specialisation.colab.run option
            # cache / output
            carriage_collate: str = CARRIAGE_COLLATE,
            spec_collate: str = SPEC_COLLATE,
            comparison_dir: str = DEFAULT_COMPARISON_DIR,
            # figure selection
            include: Optional[Sequence[str]] = None,
            exclude: Optional[Sequence[str]] = None,
            drop_vnode: bool = False,
            display: bool = False,
            # environment
            mount: bool = True,
            skip_install: bool = False,
            pyg_version: str = "2.2.0") -> dict:
    """Evaluate + carriage + specialise every model (cache-skipping), then build the deliverables.

    Compute is delegated to the existing ``carriage.colab.run`` and ``specialisation.colab.run``;
    a stage is skipped when its cached artefact already exists (unless ``force=True``). Figures are
    then built from cache via ``build_figures`` and honour ``include``/``exclude``/``drop_vnode``.
    """
    # lazy import so `import comparison` stays torch/GRIT-free
    from ..carriage.colab import run as carriage_run
    from ..specialisation.colab import run as spec_run

    ckpts = dict(DEFAULT_CKPTS) if ckpts is None else dict(ckpts)
    if mount:
        _mount_drive()
    # install deps once here; every delegated run then uses skip_install=True.
    if not skip_install:
        env.install_dependencies(pyg_version=pyg_version)
    else:
        log("[deps] Skipping dependency installation (skip_install=True).")

    def _resolve_ckpt(task):
        # Honour a pinned checkpoint only if it actually exists; otherwise fall back to
        # auto-discovery (env.find_checkpoint prefers a GraphGym ckpt/ dir, then the recovery
        # best.ckpt, then latest.ckpt), which is robust to a stale/missing pinned path.
        p = ckpts.get(task)
        if p and not Path(p).exists():
            log(f"[ckpt] pinned checkpoint for {task} not found ({p}); auto-discovering under "
                f"the task's drive_dir instead.")
            return None
        return p

    status: dict = {"carriage": {}, "specialisation": {}, "errors": []}

    # ---- carriage: per task, per intervention, cache-skipping --------------------------
    for task in tasks:
        for intv in interventions:
            summary = _data.carriage_summary_path(carriage_collate, task, intv, structural_mode)
            key = f"{task}:{intv}"
            if summary.exists() and not force:
                log(f"[cache] carriage {key}: reuse {summary}")
                status["carriage"][key] = "cached"
                continue
            log(f"[run] carriage {key} ...")
            car_kw = dict(
                task=task, intervention=intv, ckpt=_resolve_ckpt(task),
                collate_dir=carriage_collate, mount=False, skip_install=True,
                num_graphs=carriage_num_graphs, donors=carriage_donors,
                beneficial_denom=beneficial_denom,
                integrated_max_intervals=integrated_max_intervals,
                integrated_atol=integrated_atol,
                integrated_unconverged_error_cap=integrated_unconverged_error_cap,
                integrated_max_unconverged_fraction=integrated_max_unconverged_fraction,
                structural_mode=structural_mode, partner_match=partner_match,
                eval_metric=True,
            )
            if carriage_kwargs:
                car_kw.update(carriage_kwargs)          # caller overrides any carriage option
            try:
                carriage_run(**car_kw)
                status["carriage"][key] = "computed"
            except Exception as exc:  # noqa: BLE001
                log(f"[error] carriage {key}: {exc}")
                status["carriage"][key] = f"error: {exc}"
                status["errors"].append(f"carriage {key}: {exc}")

    # ---- specialisation scores (scores-only; ablation/attention not needed here) --------
    if run_specialisation:
        missing = [t for t in tasks
                   if force or not _data.scores_npz_path(spec_collate, t).exists()]
        for t in tasks:
            status["specialisation"][t] = ("cached"
                                           if _data.scores_npz_path(spec_collate, t).exists()
                                           and t not in missing else "pending")
        if missing:
            log(f"[run] specialisation scores for: {missing}")
            spec_ckpts = {t: _resolve_ckpt(t) for t in missing if _resolve_ckpt(t)}
            spec_kw = dict(
                tasks=missing, ckpt=(spec_ckpts or None), collate_dir=spec_collate,
                num_graphs=spec_num_graphs, donors=spec_donors,
                with_attn_routing=with_attn_routing,
                with_ablation=False, with_attention=False,
                mount=False, skip_install=True,
            )
            if spec_kwargs:
                spec_kw.update(spec_kwargs)              # caller overrides any specialisation option
            try:
                spec_run(**spec_kw)
                for t in missing:
                    status["specialisation"][t] = (
                        "computed" if _data.scores_npz_path(spec_collate, t).exists()
                        else "missing-after-run")
            except Exception as exc:  # noqa: BLE001
                log(f"[error] specialisation: {exc}")
                status["errors"].append(f"specialisation: {exc}")
        else:
            log("[cache] all specialisation score caches present; skipping.")

    # ---- deliverables from cache -------------------------------------------------------
    figs = build_figures(
        tasks, carriage_collate=carriage_collate, spec_collate=spec_collate,
        comparison_dir=comparison_dir, interventions=interventions,
        structural_mode=structural_mode, include=include, exclude=exclude,
        drop_vnode=drop_vnode, display=display)

    Path(comparison_dir).mkdir(parents=True, exist_ok=True)
    (Path(comparison_dir) / "run_status.json").write_text(json.dumps(status, indent=2),
                                                          encoding="utf-8")
    log(f"\n[done] comparison artefacts + figures under: {comparison_dir}")
    return {"status": status, "figures": figs, "comparison_dir": comparison_dir}
