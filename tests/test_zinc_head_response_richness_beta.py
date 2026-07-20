import copy
from pathlib import Path

import numpy as np
import torch

from graph_specialisation_metrics.specialisation import richness_beta as beta
from graph_specialisation_metrics.specialisation.scores import _perturb_mask_frozen


class _Data:
    def __init__(self):
        self.num_nodes = 3
        self.x = torch.tensor([[1], [2], [3]])
        self.y = torch.tensor([0.25])
        self.edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]])
        self.edge_attr = torch.tensor([[1], [1], [2]])
        self.rrwp_index = torch.tensor([[0, 1, 2], [0, 2, 1]])
        self.rrwp_val = torch.tensor([[0.1], [0.2], [0.3]])
        self.rrwp_local_edge_index = self.edge_index.clone()

    def clone(self):
        return copy.deepcopy(self)


def test_mask_frozen_intervention_freezes_local_rrwp_support():
    base = _Data()
    pert = _perturb_mask_frozen(base, 0, 2)
    assert torch.equal(pert.edge_index, base.edge_index)
    assert torch.equal(pert.rrwp_local_edge_index, base.rrwp_local_edge_index)
    assert torch.equal(pert.x, base.x)
    assert not torch.equal(pert.rrwp_index, base.rrwp_index)


def test_response_collector_reconstructs_score_and_captures_throughput(monkeypatch):
    base = _Data()
    monkeypatch.setattr(beta, "_spd", lambda _base, _n: np.asarray(
        [[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float))
    collector = beta.ResponseCollector("zinc")
    phi = torch.ones(1, 3, 2, 1)
    delta = torch.tensor([
        [[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]],
        [[[2.0], [1.0]], [[4.0], [3.0]], [[6.0], [5.0]]],
    ])
    collector(channel="semantic", graph_id=7, base=base,
              source_nodes=np.asarray([0, 2]), phi_stack=[phi],
              donor_averaged_delta=[delta], clean_prediction=torch.tensor([0.5]),
              clean_head_output=[torch.ones(3, 2, 1)])
    expected = delta.abs().sum(dim=(0, 1)).numpy() / 2.0
    assert np.allclose(collector.score_reconstruction("semantic"), expected.reshape(-1))
    assert collector.abs_error[7] == 0.25
    assert collector.throughput[7].shape == (1, 2)


def test_partial_spearman_removes_shared_throughput_confound():
    rng = np.random.default_rng(4)
    throughput = rng.normal(0, 3, 200)
    score = throughput + rng.normal(0, .5, 200)
    impact = throughput + rng.normal(0, .5, 200)
    raw = beta._spearman(score, impact)
    partial = beta.partial_spearman(score, impact, throughput)
    assert raw > .95
    assert abs(partial) < .35


def test_score_families_are_distinct_and_select_channel_extremes():
    score = {
        "S_sem": np.asarray([[9., 8., 7., 2.], [6., 5., 1., 1.]]),
        "S_str": np.asarray([[1., 1., 2., 8.], [2., 3., 9., 7.]]),
    }
    groups = beta.select_score_families(score, family_size=2)
    assert len(groups["semantic"]) == len(groups["structural"]) == 2
    assert set(groups["semantic"]).isdisjoint(groups["structural"])
    assert np.mean([score["S_sem"][h] / score["S_str"][h] for h in groups["semantic"]]) > 1
    assert np.mean([score["S_str"][h] / score["S_sem"][h] for h in groups["structural"]]) > 1


def test_double_dissociation_detects_matching_rescue():
    score = {
        "S_sem": np.asarray([[9., 8., 2., 1.], [7., 6., 2., 1.]]),
        "S_str": np.asarray([[1., 2., 8., 9.], [2., 1., 7., 6.]]),
    }
    groups = beta.select_score_families(score, family_size=2)
    G = 40
    sem = np.zeros((2, 4, G)); st = np.zeros_like(sem)
    for h in groups["semantic"]:
        sem[h] = .8; st[h] = .1
    for h in groups["structural"]:
        sem[h] = .1; st[h] = .8
    rescue = {
        "semantic": {"graph_ids": np.arange(G), "mediation": sem,
                     "valid_effect_fraction": 1.0},
        "structural": {"graph_ids": np.arange(G), "mediation": st,
                       "valid_effect_fraction": 1.0},
    }
    out = beta.analyse_double_dissociation(
        score, rescue, np.ones((2, 4)), family_size=2, n_null=30, n_boot=40, seed=5)
    assert out["interaction"] > 1.0
    assert out["interaction_ci"][0] > 1.0


def _record(graph_id, scale, p=8):
    response = np.zeros((2, len(beta.BIN_LABELS), p), dtype=np.float32)
    response[:, :, :] = scale / (2 * len(beta.BIN_LABELS))
    return {"graph_id": graph_id, "source_nodes": np.asarray([0, 1]), "response": response}


def _synthetic_task_results():
    gids = np.arange(20, dtype=np.int64)
    score = {"S_sem": np.asarray([[8., 7., 2., 1.], [6., 5., 2., 1.]]),
             "S_str": np.asarray([[1., 2., 7., 8.], [2., 1., 6., 5.]]),
             "L": 2, "H": 4, "test_metric": .1, "title": "x"}
    groups = beta.select_score_families(score, family_size=2)
    out = {}
    for task_i, task in enumerate(beta.BETA_TASKS):
        collector = beta.ResponseCollector(task)
        for g in gids:
            sem_scale = 2.0 + .1 * g + (1.0 if task == "zinc" else 0)
            str_scale = 2.0 + .1 * g + (1.0 if task == "zinc_1hop" else 0)
            collector.records["semantic"].append(_record(int(g), sem_scale))
            collector.records["structural"].append(_record(int(g), str_scale))
            collector.throughput[int(g)] = np.ones((2, 4)) * (1 + .01 * g)
            collector.abs_error[int(g)] = float(.3 - .004 * g + .03 * task_i)
        functional = np.tile(np.arange(1, 9, dtype=float)[:, None], (1, len(gids))).reshape(2, 4, -1)
        ablation = {
            "graph_ids": gids, "functional": functional, "loss": functional * .1,
            "features": np.column_stack([gids + 10, gids % 3, gids % 4]),
        }
        causality = {
            outcome: {ch: {"rho": .5, "ci": (.2, .7), "partial_rho": .3,
                            "partial_ci": (.1, .5)} for ch in beta.CHANNELS}
            for outcome in ("functional", "loss")
        }
        dd = {"selected_heads": {k: [list(h) for h in v] for k, v in groups.items()},
              "mediation_matrix": np.asarray([[.5, .1], [.1, .5]]),
              "interaction": .8, "interaction_ci": (.5, 1.0), "p_ge": .01}
        out[task] = {"collector": collector, "graph_ids": gids, "score": dict(score),
                     "ablation": ablation, "causality": causality,
                     "double_dissociation": dd}
    return out


def test_new_figures_smoke_and_old_rank_figures_are_absent(tmp_path):
    results = _synthetic_task_results()
    graphwise = beta.analyse_graphwise_gaps(results, n_boot=20, seed=3)
    paths = [
        beta.make_score_scatter(results, tmp_path / "scatter.png"),
        beta.make_ablation_figure(results, tmp_path / "ablation.png"),
        beta.make_rescue_figure(results, tmp_path / "rescue.png"),
        beta.make_graph_gap_figure(graphwise, tmp_path / "gaps.png"),
    ]
    assert all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in paths)
    assert not hasattr(beta, "make_distance_rank_figure")
    assert not hasattr(beta, "make_verdict_figure")


def test_standalone_colab_calls_causal_beta_and_not_rank_analysis():
    path = Path(__file__).resolve().parents[1] / "HeadResponseRichness_ZINC_Beta_Colab.py"
    code = path.read_text(encoding="utf-8")
    compile(code, str(path), "exec")
    assert "specialisation.richness_beta import run" in code
    assert "rescue_graphs=128" in code
    assert "activity_floor_fraction" not in code
    assert "NO CLEAR SEPARATION" not in code
