"""Pure selection/cache/plot tests for the D x J matched family-ablation programme."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import numpy as np

from graph_specialisation_metrics.comparison import data as D
from graph_specialisation_metrics.comparison import plots as P
from graph_specialisation_metrics.specialisation import factorial_ablation as F


def _score_fixture(L=10, H=8):
    # Twenty-four true generalists, then balanced signed specialists. Every preference class spans
    # low/high J, so the test checks the factorial rather than an unavailable-cell edge case.
    n = L * H
    d = np.concatenate([
        np.linspace(-.05, .05, 24), np.linspace(.35, .9, 28), np.linspace(-.9, -.35, 28)])
    j = np.tile(np.linspace(.2, 2.0, 8), 10)
    order = np.random.default_rng(3).permutation(n)
    d, j = d[order], j[order]
    sem = (j * (1 + d)).reshape(L, H)
    stru = (j * (1 - d)).reshape(L, H)
    throughput = (0.4 + .3 * j + np.random.default_rng(4).uniform(0, .1, n)).reshape(L, H)
    return {"S_sem": sem, "S_str": stru}, throughput


def test_factorial_selection_is_disjoint_signed_and_strength_split():
    scores, throughput = _score_fixture()
    out = F.select_factorial_families(
        scores, throughput, gsem=1.0, gstr=1.0, family_size=5,
        generalist_fraction=.30, activity_floor_quantile=.05)
    assert set(out["families"]) == set(F.FAMILY_NAMES)
    assert {len(v) for v in out["families"].values()} == {5}
    all_heads = [tuple(h) for name in F.FAMILY_NAMES for h in out["families"][name]]
    assert len(all_heads) == len(set(all_heads))
    diag = out["diagnostics"]
    for strength in F.STRENGTHS:
        assert diag[f"semantic_{strength}"]["mean_D"] > 0
        assert diag[f"structural_{strength}"]["mean_D"] < 0
        assert diag[f"generalist_{strength}"]["mean_abs_D"] < max(
            diag[f"semantic_{strength}"]["mean_abs_D"],
            diag[f"structural_{strength}"]["mean_abs_D"])
    for pref in F.PREFERENCES:
        assert diag[f"{pref}_highJ"]["mean_J"] > diag[f"{pref}_lowJ"]["mean_J"]


def test_factorial_selection_refuses_missing_signed_channel():
    # A genuinely all-semantic model must not have its least-semantic heads renamed structural.
    J = np.ones((4, 8))
    Drel = np.linspace(.1, .9, 32).reshape(4, 8)
    scores = {"S_sem": J * (1 + Drel), "S_str": J * (1 - Drel)}
    with np.testing.assert_raises_regex(RuntimeError, "structural candidates"):
        F.select_factorial_families(scores, np.ones_like(J), gsem=1, gstr=1, family_size=2)


def _family_cache_fixture(G=40, K=4, B=4, R=5):
    rng = np.random.default_rng(9)
    names = list(F.FAMILY_NAMES)
    loss = rng.normal(.01, .004, (6, B, G))
    functional = np.abs(rng.normal(.02, .006, (6, B, G)))
    return {
        "family_names": names, "budgets": np.arange(1, B + 1),
        "loss": loss, "functional": functional,
        "random_loss": rng.normal(.005, .003, (2, B, R, G)),
        "random_functional": np.abs(rng.normal(.01, .003, (2, B, R, G))),
        "heads": np.zeros((6, K, 2), dtype=int), "clean_loss": np.full(G, .07),
    }


def test_family_cache_loader_and_figures(tmp_path):
    task = "zinc"
    path = D.family_ablation_npz_path(tmp_path, task)
    path.parent.mkdir(parents=True)
    item = _family_cache_fixture()
    np.savez(path, cache_version=np.asarray(1), score_fingerprint=np.asarray("abc"), **item)
    loaded = D.load_family_ablation(path)
    assert loaded["family_names"] == list(F.FAMILY_NAMES)
    assert loaded["loss"].shape == item["loss"].shape
    _, p1 = P.plot_factorial_family_ablation_curves(
        {task: loaded}, [task], tmp_path / "curves.png", n_boot=50)
    _, p2 = P.plot_factorial_family_ablation_contrasts(
        {task: loaded}, [task], tmp_path / "contrasts.png", n_boot=50)
    assert (tmp_path / "curves.png").stat().st_size > 0
    assert (tmp_path / "contrasts.png").stat().st_size > 0


def test_family_cache_paths_are_separate_from_scores(tmp_path):
    task = "zinc_2hop"
    assert D.family_ablation_npz_path(tmp_path, task).name == \
        "factorial_family_ablation_zinc_2hop.npz"
    assert D.family_ablation_summary_path(tmp_path, task).name == \
        "factorial_family_ablation_zinc_2hop.json"
