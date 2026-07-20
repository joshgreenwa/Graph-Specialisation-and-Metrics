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


def test_conditional_coordinate_novelty_detects_distinctions_local_codes_alias():
    local = np.asarray([[1., 0.], [1., 0.], [1., .5], [1., .5]])
    high = np.asarray([[0., 1.], [1., 0.], [2., 0.], [0., 2.]])
    present = beta._conditional_coordinate_novelty(local, high)
    absent = beta._conditional_coordinate_novelty(local, np.zeros_like(high))
    assert present["rms"] > 0
    assert present["stable_rank"] >= 1
    assert absent["rms"] == 0


def test_compensation_path_recovers_positive_associative_pathway():
    rng = np.random.default_rng(12)
    n = 240
    z = rng.normal(size=(n, 2))
    x = rng.normal(size=n)
    mediator = .85 * x + .2 * z[:, 0] + rng.normal(scale=.35, size=n)
    outcome = .75 * mediator + .15 * x + .2 * z[:, 1] + rng.normal(scale=.35, size=n)
    result = beta._path_analysis(x, mediator, outcome, z, n_boot=300, seed=4)
    assert result["coefficients"]["a"]["ci"][0] > 0
    assert result["coefficients"]["b"]["ci"][0] > 0
    assert result["coefficients"]["indirect"]["ci"][0] > 0


def _record(graph_id, scale, p=8):
    response = np.zeros((2, len(beta.BIN_LABELS), p), dtype=np.float32)
    response[:, :, :] = scale / (2 * len(beta.BIN_LABELS))
    return {"graph_id": graph_id, "source_nodes": np.asarray([0, 1]), "response": response}


def _synthetic_task_results():
    rng = np.random.default_rng(9)
    gids = np.arange(40, dtype=np.int64)
    score = {"S_sem": np.asarray([[8., 7., 2., 1.], [6., 5., 2., 1.]]),
             "S_str": np.asarray([[1., 2., 7., 8.], [2., 1., 6., 5.]]),
             "L": 2, "H": 4, "test_metric": .1, "title": "x"}
    out = {}
    for task_i, task in enumerate(beta.BETA_TASKS):
        collector = beta.ResponseCollector(task)
        for g in gids:
            q = .5 * g / max(gids)
            sem_scale = 2.0 + (q if task == "zinc_1hop_local" else 0)
            str_scale = 2.0 - (q if task == "zinc_1hop_local" else 0)
            collector.records["semantic"].append(_record(int(g), sem_scale))
            collector.records["structural"].append(_record(int(g), str_scale))
            collector.throughput[int(g)] = np.ones((2, 4)) * (1 + .01 * g)
            collector.abs_error[int(g)] = float(
                .1 + (.08 * g / max(gids) if task == "zinc_1hop_local" else 0))
            novelty = {"node_rms": .1 + .01 * g, "pair_rms": .2 + .015 * g,
                       "node_stable_rank": 2., "pair_stable_rank": 2.,
                       "node_roles": 20, "pair_roles": 40}
            if task == "zinc_1hop_local":
                novelty = dict(novelty, node_rms=0., pair_rms=0.)
            collector.rrwp_novelty[int(g)] = novelty
        functional = np.tile(np.arange(1, 9, dtype=float)[:, None], (1, len(gids))).reshape(2, 4, -1)
        ablation = {
            "graph_ids": gids, "functional": functional, "loss": functional * .1,
            "features": rng.normal(size=(len(gids), 4)),
            "feature_names": ["n_nodes", "n_rings", "diameter", "atom_entropy"],
        }
        causality = {
            outcome: {ch: {"rho": .5, "ci": (.2, .7), "partial_rho": .3,
                            "partial_ci": (.1, .5)} for ch in beta.CHANNELS}
            for outcome in ("functional", "loss")
        }
        out[task] = {"collector": collector, "graph_ids": gids, "score": dict(score),
                     "ablation": ablation, "causality": causality}
    return out


def test_new_figures_smoke_and_old_rank_figures_are_absent(tmp_path):
    results = _synthetic_task_results()
    compensation = beta.analyse_semantic_compensation(results, n_boot=40, seed=3)
    assert compensation["verdict"] in {"SUPPORTED ASSOCIATION", "WEIGHT-SENSITIVE SIGNAL"}
    paths = [
        beta.make_score_scatter(results, tmp_path / "scatter.png"),
        beta.make_ablation_figure(results, tmp_path / "ablation.png"),
        beta.make_compensation_figure(compensation, tmp_path / "compensation.png"),
    ]
    assert all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in paths)
    assert not hasattr(beta, "make_distance_rank_figure")
    assert not hasattr(beta, "make_verdict_figure")
    assert not hasattr(beta, "run_rescue_sweep")
    assert not hasattr(beta, "analyse_double_dissociation")
    assert not hasattr(beta, "analyse_graphwise_gaps")
    assert not hasattr(beta, "make_graph_gap_figure")


def test_standalone_colab_calls_causal_beta_and_not_rank_analysis():
    path = Path(__file__).resolve().parents[1] / "HeadResponseRichness_ZINC_Beta_Colab.py"
    code = path.read_text(encoding="utf-8")
    compile(code, str(path), "exec")
    assert "specialisation.richness_beta import run" in code
    assert "rescue_graphs=" not in code
    assert "family_size=" not in code
    assert "activity_floor_fraction" not in code
    assert "NO CLEAR SEPARATION" not in code
