"""Unit tests for the pure-numpy channel-split ablation aggregation.

The model half (building intervention replicas, ``collect_preds_ablated``) is GPU/GRIT-only and
validated in Colab by the no-op self-check. The aggregation math -- the six per-head impacts from
clean + ablated replica predictions -- is pure numpy and is pinned here against an independent
manual recomputation plus the invariants it must satisfy.
"""

from __future__ import annotations

import numpy as np

from graph_specialisation_metrics.specialisation.channel_ablation import (
    _impacts_for_head, per_graph_loss_np)


def _fixture(seed: int = 0, T: int = 2):
    rng = np.random.default_rng(seed)
    # 3 graphs, K=2. Flat catalogue per graph: clean + 2 sem + 2 str = 5 replicas.
    clean_g = np.array([0, 5, 10])
    unit_g = np.array([0, 1, 2])
    sem_idx = np.array([[1, 2], [6, 7], [11, 12]])
    str_idx = np.array([[3, 4], [8, 9], [13, 14]])
    preds = rng.normal(size=(15, T))
    abl = rng.normal(size=(15, T))
    y_g = rng.normal(size=(3, T))
    return dict(preds=preds, abl=abl, clean_g=clean_g, y_g=y_g, unit_g=unit_g,
                sem_idx=sem_idx, str_idx=str_idx)


def test_impacts_match_manual_recompute():
    f = _fixture()
    out = _impacts_for_head(f["preds"], f["abl"], clean_g=f["clean_g"], y_g=f["y_g"],
                            unit_g=f["unit_g"], sem_idx=f["sem_idx"], str_idx=f["str_idx"],
                            loss_fun="l1")
    p, a, cg, y = f["preds"], f["abl"], f["clean_g"], f["y_g"]
    pc, pca = p[cg], a[cg]
    assert np.isclose(out["overall_func"], np.linalg.norm(pca - pc, axis=1).mean())
    assert np.isclose(out["overall_loss"],
                      (np.abs(pca - y).mean(1) - np.abs(pc - y).mean(1)).mean())

    def chan(idx):
        pc_u, pca_u, y_u = p[cg[f["unit_g"]]], a[cg[f["unit_g"]]], y[f["unit_g"]]
        ps, psa = p[idx].mean(1), a[idx].mean(1)
        func = np.linalg.norm((pc_u - ps) - (pca_u - psa), axis=1).mean()
        ls = np.abs(ps - y_u).mean(1) - np.abs(pc_u - y_u).mean(1)
        lsa = np.abs(psa - y_u).mean(1) - np.abs(pca_u - y_u).mean(1)
        return func, (ls - lsa).mean()

    sf, sl = chan(f["sem_idx"])
    tf, tl = chan(f["str_idx"])
    assert np.isclose(out["I_sem_func"], sf) and np.isclose(out["I_str_func"], tf)
    assert np.isclose(out["I_sem_loss"], sl) and np.isclose(out["I_str_loss"], tl)


def test_noop_ablation_gives_zero_impact():
    f = _fixture()
    out = _impacts_for_head(f["preds"], f["preds"].copy(), clean_g=f["clean_g"], y_g=f["y_g"],
                            unit_g=f["unit_g"], sem_idx=f["sem_idx"], str_idx=f["str_idx"],
                            loss_fun="l1")
    assert all(abs(out[k]) < 1e-12 for k in out)


def test_channel_symmetry_and_functional_nonneg():
    f = _fixture()
    a = _impacts_for_head(f["preds"], f["abl"], clean_g=f["clean_g"], y_g=f["y_g"],
                          unit_g=f["unit_g"], sem_idx=f["sem_idx"], str_idx=f["str_idx"], loss_fun="l1")
    b = _impacts_for_head(f["preds"], f["abl"], clean_g=f["clean_g"], y_g=f["y_g"],
                          unit_g=f["unit_g"], sem_idx=f["str_idx"], str_idx=f["sem_idx"], loss_fun="l1")
    assert np.isclose(b["I_sem_func"], a["I_str_func"])
    assert np.isclose(b["I_str_loss"], a["I_sem_loss"])
    assert a["I_sem_func"] >= 0 and a["I_str_func"] >= 0 and a["overall_func"] >= 0


def test_per_graph_loss_np_matches_definitions():
    pred = np.array([[0.0, 2.0], [1.0, -1.0]])
    y = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert np.allclose(per_graph_loss_np(pred, y, "l1"), [1.0, 1.0])       # mean|.|
    assert np.allclose(per_graph_loss_np(pred, y, "mse"), [2.0, 2.0])      # mean(.^2)
    # bce stable form, mean over T; shape only (value monotonic checked loosely)
    b = per_graph_loss_np(pred, y, "bce")
    assert b.shape == (2,) and np.all(np.isfinite(b))
