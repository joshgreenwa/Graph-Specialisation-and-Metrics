"""Smoke + invariant tests for the cross-model comparison figures.

Pure numpy/matplotlib (Agg), no torch/GRIT, so they run anywhere. They assert the two things
the deliverables promise: the specialisation scatter/D-J grids render, and the D-J plane is
reference-fixed (does not rescale when methods are dropped) with D signed by channel preference.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np

from graph_specialisation_metrics.comparison import plots as P


def _scores(sem_scale: float, str_scale: float, L: int = 6, H: int = 4, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {"S_sem": np.abs(rng.normal(0, 1, (L, H))) * sem_scale,
            "S_str": np.abs(rng.normal(0, 1, (L, H))) * str_scale}


def _scores_by_task() -> dict:
    # balanced dense, structural-leaning 1-hop, semantic-leaning 2-hop
    return {"zinc": _scores(1.0, 1.0, seed=1),
            "zinc_1hop": _scores(0.6, 1.4, seed=2),
            "zinc_2hop": _scores(1.3, 0.7, seed=3)}


def test_spec_scatter_and_DJ_grids_render(tmp_path):
    scores = _scores_by_task()
    gsem, gstr = P.global_norms(scores, list(scores))
    _, p1 = P.plot_spec_scatter_grid(scores, list(scores), tmp_path / "scatter.png",
                                     gsem=gsem, gstr=gstr)
    _, p2 = P.plot_spec_DJ_grid(scores, list(scores), tmp_path / "dj.png",
                                gsem=gsem, gstr=gstr, ref_tasks=list(scores))
    assert (tmp_path / "scatter.png").stat().st_size > 0
    assert (tmp_path / "dj.png").stat().st_size > 0


def test_DJ_plane_is_reference_fixed_under_drop_include(tmp_path):
    scores = _scores_by_task()
    gsem, gstr = P.global_norms(scores, list(scores))
    fig_all, _ = P.plot_spec_DJ_grid(scores, list(scores), tmp_path / "all.png",
                                     gsem=gsem, gstr=gstr, ref_tasks=list(scores))
    fig_sub, _ = P.plot_spec_DJ_grid(scores, ["zinc", "zinc_1hop"], tmp_path / "sub.png",
                                     gsem=gsem, gstr=gstr, ref_tasks=list(scores))
    # dropping a method must not move the plane (limits come from ref_tasks, not the shown set)
    assert np.allclose(fig_all.axes[0].get_xlim(), fig_sub.axes[0].get_xlim())
    assert np.allclose(fig_all.axes[0].get_ylim(), fig_sub.axes[0].get_ylim())


def test_DJ_selectivity_sign_tracks_channel_preference():
    scores = _scores_by_task()
    gsem, gstr = P.global_norms(scores, list(scores))

    def mean_D(t):
        ssem = scores[t]["S_sem"] / gsem
        sstr = scores[t]["S_str"] / gstr
        return float((0.5 * (ssem - sstr)).mean())

    # structural-leaning model -> D<0; semantic-leaning -> D>0
    assert mean_D("zinc_1hop") < 0.0
    assert mean_D("zinc_2hop") > 0.0
    # J is a nonnegative strength (scores are magnitudes, norms positive)
    ssem = scores["zinc"]["S_sem"] / gsem
    sstr = scores["zinc"]["S_str"] / gstr
    assert float((0.5 * (ssem + sstr)).min()) >= 0.0


# ---- channel-split ablation validation figures --------------------------------------

def _channel_ablation(scores, gsem, gstr, seed=7):
    rng = np.random.default_rng(seed)
    ch = {}
    for t in scores:
        ss, st = scores[t]["S_sem"] / gsem, scores[t]["S_str"] / gstr
        ch[t] = dict(
            I_sem_func=np.abs(ss + rng.normal(0, 0.1, ss.shape)),
            I_str_func=np.abs(st + rng.normal(0, 0.1, st.shape)),
            I_sem_loss=(ss - st) * 0.3 + rng.normal(0, 0.1, ss.shape),
            I_str_loss=(st - ss) * 0.3 + rng.normal(0, 0.1, st.shape),
            overall_func=np.abs(0.5 * (ss + st) + rng.normal(0, 0.1, ss.shape)),
            overall_loss=rng.normal(0, 0.1, ss.shape))
    return ch


def test_signed_contrast_and_drel_bounded():
    rng = np.random.default_rng(0)
    a, b = np.abs(rng.normal(size=200)), np.abs(rng.normal(size=200))
    assert np.all(np.abs(P.signed_contrast(a, b)) <= 1.0)          # a,b>=0
    ssem, sstr = np.abs(rng.normal(size=50)), np.abs(rng.normal(size=50))
    assert np.all(np.abs(P.d_rel(ssem, sstr)) <= 1.0)


def test_DJ_ablation_validation_and_summaries_render(tmp_path):
    scores = _scores_by_task()
    gsem, gstr = P.global_norms(scores, list(scores))
    ch = _channel_ablation(scores, gsem, gstr)
    _, p1 = P.plot_DJ_ablation_validation(scores, ch, list(scores), tmp_path / "val.png",
                                          gsem=gsem, gstr=gstr)
    _, p2 = P.plot_DJ_quadrants(scores, ch, list(scores), tmp_path / "quad.png",
                                gsem=gsem, gstr=gstr, ref_tasks=list(scores))
    _, p3 = P.plot_DJ_influence_strength(scores, ch, list(scores), tmp_path / "infl.png",
                                         gsem=gsem, gstr=gstr)
    for p in (p1, p2, p3):
        from pathlib import Path
        assert Path(p).stat().st_size > 0


def test_DJ_summaries_work_without_channel_cache(tmp_path):
    # quadrants + influence must still render (influence rho just unavailable) with no chan cache
    scores = _scores_by_task()
    gsem, gstr = P.global_norms(scores, list(scores))
    _, pq = P.plot_DJ_quadrants(scores, {}, list(scores), tmp_path / "q.png",
                                gsem=gsem, gstr=gstr, ref_tasks=list(scores))
    _, pi = P.plot_DJ_influence_strength(scores, {}, list(scores), tmp_path / "i.png",
                                         gsem=gsem, gstr=gstr)
    from pathlib import Path
    assert Path(pq).stat().st_size > 0 and Path(pi).stat().st_size > 0
