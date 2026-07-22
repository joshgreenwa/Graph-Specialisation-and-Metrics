"""Cache-path + method-selection tests for the cross-model comparison layer (torch-free)."""

import json

import numpy as np

from graph_specialisation_metrics.comparison import data as D
from graph_specialisation_metrics.comparison.run import performance_table


def test_carriage_summary_path_prefix():
    sem = D.carriage_summary_path("/c", "zinc", "semantic")
    stru = D.carriage_summary_path("/c", "zinc", "structural", "transposition")
    assert sem.as_posix().endswith("/c/zinc/carriage_summary.json")
    assert stru.as_posix().endswith("/c/zinc/structural_transposition_carriage_summary.json")


def test_select_methods_include_exclude_dropvnode():
    tasks = ["zinc", "zinc_1hop", "zinc_2hop", "zinc_1hop_vnode", "zinc_2hop_vnode"]
    assert D.select_methods(tasks, drop_vnode=True) == ["zinc", "zinc_1hop", "zinc_2hop"]
    assert D.select_methods(tasks, include=["zinc", "zinc_2hop"]) == ["zinc", "zinc_2hop"]
    assert D.select_methods(tasks, exclude=["zinc"]) == [
        "zinc_1hop", "zinc_2hop", "zinc_1hop_vnode", "zinc_2hop_vnode"]
    assert D.is_vnode("zinc_2hop_vnode") and not D.is_vnode("zinc_2hop")


def test_master_bins_orders_union_by_lower_edge():
    def curves(labels, los):
        return {"curves": {"bin_label": labels, "bin_lo": los}}
    by_task = {
        "a": curves(["0", "1", "2", "4-7"], [0, 1, 2, 4]),
        "b": curves(["0", "1", "2", "3", "4-7", "8-15"], [0, 1, 2, 3, 4, 8]),  # longer diameter
    }
    assert D.master_bins(by_task, ["a", "b"]) == ["0", "1", "2", "3", "4-7", "8-15"]


def test_curve_on_master_fills_nan_for_absent_bins():
    curve = {"bin_label": ["0", "1", "2"], "F_mean": [5.0, 1.0, 0.5]}
    out = D.curve_on_master(curve, "F_mean", ["0", "1", "2", "3", "4-7"])
    assert out[:3].tolist() == [5.0, 1.0, 0.5]
    assert np.isnan(out[3]) and np.isnan(out[4])


def test_load_scores_and_summary_roundtrip(tmp_path):
    z = tmp_path / "zinc" / "scores_zinc.npz"
    z.parent.mkdir(parents=True)
    np.savez(z, S_sem=np.ones((10, 8)), S_str=np.full((10, 8), 2.0),
             S_sem_CG=np.full((10, 8), 0.9), S_str_CG=np.full((10, 8), 1.8),
             S_attn_sem=np.zeros((10, 8)), score_aggregation=np.asarray("EG"))
    s = D.load_scores(z)
    assert s["S_sem"].shape == (10, 8) and s["S_str"].mean() == 2.0
    assert s["score_aggregation"] == "EG" and np.isclose(s["S_sem_CG"].mean(), 0.9)

    p = tmp_path / "zinc" / "carriage_summary.json"
    p.write_text(json.dumps({"meta": {"val_metric": 0.11, "test_metric": 0.10,
                                       "test_metric_name": "mae"}, "curves": {}}))
    summ = D.load_carriage_summary(p)
    assert summ["meta"]["val_metric"] == 0.11
    assert D.load_carriage_summary(tmp_path / "missing.json") is None


def test_performance_table_backfills_val_without_recomputing_carriage(tmp_path):
    carriage, spec = tmp_path / "carriage", tmp_path / "spec"
    cpath = D.carriage_summary_path(carriage, "zinc", "semantic")
    cpath.parent.mkdir(parents=True)
    cpath.write_text(json.dumps({"meta": {"test_metric": .058,
                                           "test_metric_name": "mae"}}))
    spath = D.spec_stats_path(spec, "zinc")
    spath.parent.mkdir(parents=True)
    spath.write_text(json.dumps({"val_metric": .070, "test_metric": .058,
                                 "test_metric_name": "mae"}))
    table = performance_table(["zinc"], carriage_collate=str(carriage),
                              spec_collate=str(spec))
    assert table["zinc"]["test"] == .058
    assert table["zinc"]["val"] == .070


def test_pre_v3_cg_score_cache_is_not_loaded(tmp_path):
    for task in ("zinc", "zinc_1hop_vnode"):
        score_path = D.scores_npz_path(tmp_path, task)
        score_path.parent.mkdir(parents=True)
        np.savez(score_path, S_sem=np.ones((2, 2)), S_str=np.ones((2, 2)))
        stats_path = D.spec_stats_path(tmp_path, task)
        stats_path.write_text(json.dumps({"score_cache_version": 2}))
        assert task not in D.load_scores_by_task(tmp_path, [task])
        stats_path.write_text(json.dumps({"score_cache_version": 3}))
        assert task in D.load_scores_by_task(tmp_path, [task])
