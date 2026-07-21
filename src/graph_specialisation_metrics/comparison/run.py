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
* (ii-b)``fig_spec_DJ_grid.png``               -- selectivity D vs joint-strength J per model;
* (iii) ``fig_carriage_smallmult_<intv>.png``  -- functional/beneficial carriage per model,
        standardised (reference-fixed) y-axis;
* (iv)  ``fig_carriage_overlay_<intv>.png``    -- functional/beneficial carriage overlaid across
        models for direct comparison;
* (v)   ``fig_DJ_ablation_validation.png`` (+ ``fig_DJ_quadrants.png`` /
        ``fig_DJ_influence_strength.png``) -- validates the score selectivity D_rel against the
        channel-split (swap x ablate) ablation contrast, functional & loss; needs
        ``with_channel_ablation=True`` (else the score-only quadrant/influence figures still build);
* (vi)  ``fig_DJ_family_ablation_curves.png`` + ``fig_DJ_family_ablation_contrasts.png`` --
        cumulative pre-head ablation of semantic/structural/generalist families crossed with
        high/low J, matched for layer and clean throughput. This consumes the score cache and has
        its own cache; it never recomputes carriage or scores;
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
        candidates = [
            _metric_from_summary(_data.load_carriage_summary(
                _data.carriage_summary_path(carriage_collate, t, "semantic"))),
            _metric_from_summary(_data.load_carriage_summary(
                _data.carriage_summary_path(carriage_collate, t, "structural", structural_mode))),
        ]
        stats = _data.load_spec_stats(_data.spec_stats_path(spec_collate, t))
        if stats and (stats.get("val_metric") is not None or stats.get("test_metric") is not None):
            candidates.append({"val": stats.get("val_metric"), "test": stats.get("test_metric"),
                               "name": stats.get("test_metric_name") or "MAE"})
        candidates = [candidate for candidate in candidates if candidate is not None]
        m = None
        if candidates:
            # Preserve an existing carriage result exactly; fill any old missing val/test field
            # from structural/spec metadata rather than recomputing expensive carriage solely to
            # backfill a plot label.
            m = dict(candidates[0])
            for candidate in candidates[1:]:
                for key in ("val", "test", "name"):
                    if m.get(key) is None and candidate.get(key) is not None:
                        m[key] = candidate[key]
        if m is not None:
            out[t] = m
    return out


# --------------------------------------------------------------------------------------
# diagnostic: what is on Drive for each model (checkpoint found? which caches exist?)
# --------------------------------------------------------------------------------------

def inventory(tasks: Sequence[str] = _data.DEFAULT_TASKS, *,
              carriage_collate: str = CARRIAGE_COLLATE,
              spec_collate: str = SPEC_COLLATE,
              structural_mode: str = "transposition",
              check_ckpt: bool = True, mount: bool = False) -> dict:
    """Per-model cache/checkpoint status, so a missing figure method is easy to diagnose.

    For each task reports whether a checkpoint is discoverable (honouring a pinned path, else
    auto-discovery under the task's ``drive_dir``) and whether the semantic-carriage,
    structural-carriage, and specialisation-score caches exist (with val/test where present).
    A method absent from a figure is missing its cache -- this shows if that is because the
    checkpoint was not found (so compute failed) or because compute was never run.
    """
    if mount:
        _mount_drive()

    def _valtest(summary):
        m = (summary or {}).get("meta", {})
        return m.get("val_metric"), m.get("test_metric")

    rows: dict = {}
    log(f"{'model':<18} {'ckpt':<7} {'sem(F/B)':<9} {'struct':<7} {'scores':<7} "
        f"{'chanAbl':<8} {'famAbl':<7} {'val/test'}")
    for t in tasks:
        sem = _data.load_carriage_summary(_data.carriage_summary_path(carriage_collate, t, "semantic"))
        strc = _data.load_carriage_summary(
            _data.carriage_summary_path(carriage_collate, t, "structural", structural_mode))
        scores_exists = _data.scores_npz_path(spec_collate, t).exists()
        score_stats = _data.load_spec_stats(_data.spec_stats_path(spec_collate, t)) or {}
        score_version = int(score_stats.get("score_cache_version", 0))
        scores = scores_exists and (not _data.is_vnode(t) or score_version >= 2)
        chan = _data.channel_ablation_npz_path(spec_collate, t).exists()
        fam = _data.family_ablation_npz_path(spec_collate, t).exists()
        sem_val, sem_test = _valtest(sem)

        ckpt_status = "-"
        if check_ckpt:
            pinned = DEFAULT_CKPTS.get(t)
            if pinned and Path(pinned).exists():
                ckpt_status = "pinned"
            else:
                try:
                    from ..carriage.tasks import get_task
                    spec = get_task(t)
                    p, _ = env.find_checkpoint(Path(spec.drive_dir) / "results", None)
                    ckpt_status = "found" if p else "MISSING"
                except Exception:  # noqa: BLE001
                    ckpt_status = "MISSING"

        vt = (f"{sem_val:.3f}/{sem_test:.3f}"
              if (sem_val is not None and sem_test is not None) else
              (f"-/{sem_test:.3f}" if sem_test is not None else "-"))
        rows[t] = {"ckpt": ckpt_status, "carriage_semantic": bool(sem),
                   "carriage_structural": bool(strc), "scores": bool(scores),
                   "score_cache_version": score_version,
                   "channel_ablation": bool(chan),
                   "factorial_family_ablation": bool(fam),
                   "val_metric": sem_val, "test_metric": sem_test}
        log(f"{t:<18} {ckpt_status:<7} {('yes' if sem else 'no'):<9} "
            f"{('yes' if strc else 'no'):<7} {('yes' if scores else 'no'):<7} "
            f"{('yes' if chan else 'no'):<8} {('yes' if fam else 'no'):<7} {vt}")
    return rows


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
            # (ii-b) selectivity D (x) vs joint strength J (y), same norms + reference-fixed plane
            fig, p = _plots.plot_spec_DJ_grid(
                scores_ref, shown_scores, out_dir / "fig_spec_DJ_grid.png",
                gsem=gsem, gstr=gstr, metrics_by_task=metrics,
                ref_tasks=list(scores_ref.keys()))
            figs["spec_DJ_grid"] = p
            made.append(fig)

            # (v) D/J causal validation against the channel-split (swap x ablate) ablation,
            # when that heavier cache is present. Degrades to the score-only figures otherwise.
            chan_ref = _data.load_channel_ablation_by_task(spec_collate, tasks)
            # quadrant taxonomy (scores only; influence rho uses chan where present)
            fig, p = _plots.plot_DJ_quadrants(
                scores_ref, chan_ref, shown_scores, out_dir / "fig_DJ_quadrants.png",
                gsem=gsem, gstr=gstr, ref_tasks=list(scores_ref.keys()), metrics_by_task=metrics)
            figs["DJ_quadrants"] = p
            made.append(fig)
            fig, p = _plots.plot_DJ_influence_strength(
                scores_ref, chan_ref, shown_scores, out_dir / "fig_DJ_influence_strength.png",
                gsem=gsem, gstr=gstr)
            figs["DJ_influence_strength"] = p
            made.append(fig)
            shown_chan = [t for t in shown_scores if t in chan_ref]
            if shown_chan:
                fig, p = _plots.plot_DJ_ablation_validation(
                    scores_ref, chan_ref, shown_chan, out_dir / "fig_DJ_ablation_validation.png",
                    gsem=gsem, gstr=gstr, metrics_by_task=metrics)
                figs["DJ_ablation_validation"] = p
                made.append(fig)
            else:
                log("[figures] no channel-ablation cache; skipping the D/J ablation-validation "
                    "figure (run with with_channel_ablation=True to produce it).")

            # (vi) Family-level necessity/redundancy programme. These plots are pure cache reads;
            # generating/restyling them never loads a model.
            family_ref = _data.load_family_ablation_by_task(spec_collate, tasks)
            if family_ref:
                from ..specialisation.factorial_ablation import CACHE_VERSION, score_fingerprint
                stale = [t for t, item in family_ref.items()
                         if (t not in scores_ref or item.get("cache_version") != CACHE_VERSION
                             or item.get("score_fingerprint") != score_fingerprint(scores_ref[t]))]
                for t in stale:
                    family_ref.pop(t, None)
                    log(f"[figures] ignoring stale factorial family-ablation cache for {t}.")
            shown_family = [t for t in shown_scores if t in family_ref]
            if shown_family:
                fig, p = _plots.plot_factorial_family_ablation_curves(
                    family_ref, shown_family, out_dir / "fig_DJ_family_ablation_curves.png")
                figs["DJ_family_ablation_curves"] = p
                made.append(fig)
                fig, p = _plots.plot_factorial_family_ablation_contrasts(
                    family_ref, shown_family, out_dir / "fig_DJ_family_ablation_contrasts.png")
                figs["DJ_family_ablation_contrasts"] = p
                made.append(fig)
            else:
                log("[figures] no factorial family-ablation cache; skipping its two figures.")
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
            integrated_atol: float = 1e-4,
            integrated_unconverged_error_cap: float = 1e-2,
            integrated_max_unconverged_fraction: float = 1e-2,
            structural_mode: str = "transposition",
            partner_match: str = "degree",
            # any other carriage.colab.run option (e.g. {"integrated_rtol": 1e-4, "bin_strategy":
            # "hop", "central": "median", "analysis_seed": 1, "n_boot": 2000}); overrides the above.
            carriage_kwargs: Optional[dict] = None,
            # specialisation knobs
            run_specialisation: bool = True,
            spec_num_graphs: int = 128,
            spec_donors: int = 64,
            with_attn_routing: bool = True,
            # channel-split causal ablation (I_sem/I_str functional + loss) -> the D/J validation
            # figure. This is the heavy swap x ablate sweep (L*H passes per model), but is part of
            # the default paper-analysis run; set False explicitly for a score/carriage-only run.
            with_channel_ablation: bool = True,
            channel_ablation_graphs: int = 128,
            channel_ablation_sources: int = 32,
            channel_ablation_donors: int = 16,
            # cached-score family ablation (enabled by default). This is a separate, incremental
            # stage: it loads S_sem/S_str from disk and computes only new held-out ablation passes.
            with_factorial_family_ablation: bool = True,
            family_ablation_graphs: int = 256,
            family_size: int = 6,
            family_generalist_fraction: float = 0.30,
            family_activity_floor_quantile: float = 0.10,
            family_random_sets: int = 24,
            family_analysis_seed: int = 2718,
            family_eval_split: str = "val",
            family_batch_size: int = 64,
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
            allow_partial: bool = False,
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

    status: dict = {"carriage": {}, "specialisation": {},
                    "factorial_family_ablation": {}, "errors": []}

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
                resume=not force,
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
    # PER-MODEL, isolated: run spec_run one task at a time so a single bad/missing checkpoint
    # (e.g. a VNode variant) cannot abort scoring for the other models and drop them all from
    # the scatter grid. spec_run re-clones/re-imports its GRIT per task anyway, so this adds no
    # extra setup relative to the batched call.
    if run_specialisation:
        for t in tasks:
            # A model is fully cached only if its scores exist AND, when channel ablation is
            # requested, its channel_ablation npz exists too. Otherwise (re)run this model.
            score_stats = _data.load_spec_stats(_data.spec_stats_path(spec_collate, t)) or {}
            score_version = int(score_stats.get("score_cache_version", 0))
            vnode_score_current = (not _data.is_vnode(t) or score_version >= 2)
            scores_cached = (_data.scores_npz_path(spec_collate, t).exists()
                             and vnode_score_current)
            chan_cached = _data.channel_ablation_npz_path(spec_collate, t).exists()
            fully_cached = scores_cached and (not with_channel_ablation or chan_cached)
            if not force and fully_cached:
                log(f"[cache] specialisation {t}: reuse {_data.scores_npz_path(spec_collate, t)}"
                    + (" (+ channel ablation)" if with_channel_ablation else ""))
                status["specialisation"][t] = "cached"
                continue
            need_channel_ablation = with_channel_ablation and not chan_cached
            if not vnode_score_current and _data.scores_npz_path(spec_collate, t).exists():
                log(f"[cache] specialisation {t}: invalidating pre-v2 VNode score cache "
                    "(virtual carrier was omitted); channel-ablation cache remains reusable.")
            log(f"[run] specialisation scores for {t} ..."
                + (" (+ channel-split ablation)" if need_channel_ablation else ""))
            ck = _resolve_ckpt(t)
            spec_kw = dict(
                tasks=[t], ckpt=({t: ck} if ck else None), collate_dir=spec_collate,
                num_graphs=spec_num_graphs, donors=spec_donors,
                with_attn_routing=with_attn_routing,
                with_ablation=False, with_attention=False,
                with_channel_ablation=need_channel_ablation,
                channel_ablation_graphs=channel_ablation_graphs,
                channel_ablation_sources=channel_ablation_sources,
                channel_ablation_donors=channel_ablation_donors,
                mount=False, skip_install=True, resume=not force,
            )
            if spec_kwargs:
                spec_kw.update(spec_kwargs)              # caller overrides any specialisation option
            try:
                spec_run(**spec_kw)
                status["specialisation"][t] = (
                    "computed" if _data.scores_npz_path(spec_collate, t).exists()
                    else "missing-after-run")
            except Exception as exc:  # noqa: BLE001
                log(f"[error] specialisation {t}: {exc}")
                status["specialisation"][t] = f"error: {exc}"
                status["errors"].append(f"specialisation {t}: {exc}")

    # ---- D x J factorial family ablation: cached scores -> new held-out ablations only --------
    family_current: dict[str, bool] = {}
    if with_factorial_family_ablation:
        from ..specialisation import factorial_ablation as family_mod

        scores_ref = _data.load_scores_by_task(spec_collate, tasks)
        if scores_ref:
            gsem, gstr = _plots.global_norms(scores_ref, list(scores_ref))
        else:
            gsem = gstr = None
        for task in tasks:
            if task not in scores_ref:
                status["factorial_family_ablation"][task] = "missing-score-cache"
                family_current[task] = False
                continue
            npz_path = _data.family_ablation_npz_path(spec_collate, task)
            summary_path = _data.family_ablation_summary_path(spec_collate, task)
            fingerprint = family_mod.score_fingerprint(scores_ref[task])
            request = {
                "eval_split": family_eval_split, "num_graphs": int(family_ablation_graphs),
                "requested_family_size": int(family_size),
                "generalist_fraction": float(family_generalist_fraction),
                "activity_floor_quantile": float(family_activity_floor_quantile),
                "random_sets": int(family_random_sets), "analysis_seed": int(family_analysis_seed),
                "batch_size": int(family_batch_size), "gsem": float(gsem), "gstr": float(gstr),
            }
            cached_summary = None
            if summary_path.exists():
                try:
                    cached_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    cached_summary = None
            summary_current = bool(
                cached_summary
                and int(cached_summary.get("cache_version", 0)) == family_mod.CACHE_VERSION
                and cached_summary.get("score_fingerprint") == fingerprint
                and cached_summary.get("config") == request)
            current = bool(npz_path.exists() and summary_current
                           and cached_summary.get("status", "complete") == "complete")
            if current and not force:
                log(f"[cache] factorial family ablation {task}: reuse {npz_path}")
                status["factorial_family_ablation"][task] = "cached"
                family_current[task] = True
                continue
            if (summary_current and cached_summary.get("status") == "not-estimable"
                    and not force):
                reason = cached_summary.get("reason", "six-family D x J design unavailable")
                log(f"[family-selection] {task}: not estimable from its score geometry; "
                    f"reusing recorded verdict ({reason})")
                status["factorial_family_ablation"][task] = f"not-estimable: {reason}"
                family_current[task] = True
                continue
            try:
                preflight = family_mod.check_factorial_estimable(
                    scores_ref[task], gsem=gsem, gstr=gstr, family_size=family_size,
                    generalist_fraction=family_generalist_fraction,
                    activity_floor_quantile=family_activity_floor_quantile)
            except family_mod.FactorialNotEstimable as exc:
                reason = str(exc)
                verdict = {
                    "cache_version": family_mod.CACHE_VERSION, "status": "not-estimable",
                    "task": task, "score_fingerprint": fingerprint, "config": request,
                    "reason": reason,
                }
                summary_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
                log(f"[family-selection] {task}: NOT ESTIMABLE — {reason}. "
                    "No heads were relabelled; the core all-model comparison will continue.")
                status["factorial_family_ablation"][task] = f"not-estimable: {reason}"
                family_current[task] = True
                continue
            log(f"[run] factorial family ablation {task} from cached scores "
                f"(no carriage/score recomputation; cells={preflight['cell_sizes']}) ...")
            try:
                result = family_mod.prepare_and_run(
                    task, scores_ref[task], gsem=gsem, gstr=gstr,
                    collate_dir=spec_collate, ckpt=_resolve_ckpt(task),
                    num_graphs=family_ablation_graphs, family_size=family_size,
                    generalist_fraction=family_generalist_fraction,
                    activity_floor_quantile=family_activity_floor_quantile,
                    random_sets=family_random_sets, analysis_seed=family_analysis_seed,
                    eval_split=family_eval_split, batch_size=family_batch_size)
                family_mod.save_family_ablation(
                    result, npz_path, summary_path, task=task,
                    score_hash=fingerprint, config=request)
                status["factorial_family_ablation"][task] = "computed"
                family_current[task] = True
            except Exception as exc:  # noqa: BLE001
                log(f"[error] factorial family ablation {task}: {exc}")
                status["factorial_family_ablation"][task] = f"error: {exc}"
                status["errors"].append(f"factorial family ablation {task}: {exc}")
                family_current[task] = False

    # ---- completeness gate: never silently publish a two-line "all-model" overlay --------
    missing_carriage = [
        f"{task}:{intv}" for task in tasks for intv in interventions
        if not _data.carriage_summary_path(
            carriage_collate, task, intv, structural_mode).exists()
    ]
    def _score_cache_current(task):
        if not _data.scores_npz_path(spec_collate, task).exists():
            return False
        version = int((_data.load_spec_stats(
            _data.spec_stats_path(spec_collate, task)) or {}).get("score_cache_version", 0))
        return not _data.is_vnode(task) or version >= 2

    missing_scores = ([task for task in tasks if not _score_cache_current(task)]
                      if run_specialisation else [])
    missing_family = ([task for task in tasks if not family_current.get(task, False)]
                      if with_factorial_family_ablation else [])
    status["missing_carriage"] = missing_carriage
    status["missing_scores"] = missing_scores
    status["missing_factorial_family_ablation"] = missing_family
    Path(comparison_dir).mkdir(parents=True, exist_ok=True)
    status_path = Path(comparison_dir) / "run_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    if missing_family:
        log(f"[warn] unresolved optional factorial family stages: {missing_family}; core "
            "carriage/specialisation figures will still be built.")
    if not allow_partial and (missing_carriage or missing_scores):
        details = "; ".join(status["errors"][-5:]) or "see the per-stage status entries"
        raise RuntimeError(
            "Refusing to draw a partial all-model comparison. Missing carriage="
            f"{missing_carriage}; missing scores={missing_scores}. {details}. "
            f"Full status: {status_path}"
        )

    # ---- deliverables from cache -------------------------------------------------------
    figs = build_figures(
        tasks, carriage_collate=carriage_collate, spec_collate=spec_collate,
        comparison_dir=comparison_dir, interventions=interventions,
        structural_mode=structural_mode, include=include, exclude=exclude,
        drop_vnode=drop_vnode, display=display)

    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    log(f"\n[done] comparison artefacts + figures under: {comparison_dir}")
    return {"status": status, "figures": figs, "comparison_dir": comparison_dir}
