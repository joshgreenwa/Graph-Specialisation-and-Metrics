import copy
import json
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


def test_participation_ratio_is_scale_invariant_and_counts_orthogonal_patterns():
    rank_one = np.diag([4.0, 0.0, 0.0])
    rank_two = np.diag([2.0, 2.0, 0.0])

    assert beta.participation_ratio(rank_one) == 1.0
    assert beta.participation_ratio(10.0 * rank_one) == 1.0
    assert beta.participation_ratio(rank_two) == 2.0


def test_response_collector_reconstructs_existing_score(monkeypatch):
    base = _Data()
    D = np.asarray([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    monkeypatch.setattr(beta, "_spd", lambda _base, _n: D)
    collector = beta.ResponseCollector("zinc")

    phi = torch.ones(1, 3, 2, 1)
    delta = torch.tensor([
        [[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]],
        [[[2.0], [1.0]], [[4.0], [3.0]], [[6.0], [5.0]]],
    ])
    collector(
        channel="semantic",
        graph_id=7,
        base=base,
        source_nodes=np.asarray([0, 2]),
        phi_stack=[phi],
        donor_averaged_delta=[delta],
        clean_prediction=torch.tensor([0.5]),
    )

    expected = delta.abs().sum(dim=(0, 1)).numpy() / 2.0
    rebuilt = collector.score_reconstruction("semantic")
    assert np.allclose(rebuilt, expected.reshape(-1))
    assert collector.abs_error[7] == 0.25


def _record(graph_id, patterns):
    response = np.zeros((len(patterns), len(beta.BIN_LABELS), 3), dtype=np.float32)
    response[:, 0, :] = np.asarray(patterns, dtype=np.float32)
    return {
        "graph_id": graph_id,
        "source_nodes": np.arange(len(patterns), dtype=np.int64),
        "response": response,
    }


def _synthetic_task_results():
    gids = np.arange(6, dtype=np.int64)
    same = [[1, 0, 0]] * 6
    varied = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] * 2
    out = {}
    for task in beta.BETA_TASKS:
        collector = beta.ResponseCollector(task)
        collector.records["semantic"] = [_record(int(g), same) for g in gids]
        structural = varied if task == "zinc_1hop" else same
        collector.records["structural"] = [_record(int(g), structural) for g in gids]
        collector.abs_error = {
            int(g): float(0.05 + 0.01 * g + (0.06 if task == "zinc_1hop_local" else 0.0))
            for g in gids
        }
        out[task] = {
            "collector": collector,
            "graph_ids": gids,
            "score": {
                "S_sem": np.asarray([[1.0, 0.5, 0.2]]),
                "S_str": np.asarray([[0.8, 0.4, 0.1]]),
                "L": 1,
                "H": 3,
                "title": task,
                "test_metric": collector.abs_error[0],
                "test_metric_name": "mae",
                "checks": {},
            },
            "score_reconstruction_max_abs": 0.0,
        }
    return out


def test_paired_beta_verdict_detects_structural_specific_rank_gain():
    analysis = beta.analyse_response_rank(
        _synthetic_task_results(), n_boot=100, bootstrap_seed=3
    )

    assert analysis["verdict"] == "SUPPORTED"
    assert analysis["contrasts"]["overall"]["structural"]["ci_low"] > 0
    assert analysis["contrasts"]["overall"]["semantic"]["difference"] == 0


def test_beta_figures_and_outputs_smoke(tmp_path):
    task_results = _synthetic_task_results()
    analysis = beta.analyse_response_rank(task_results, n_boot=30, bootstrap_seed=5)

    figure_paths = [
        beta.make_score_scatter(task_results, tmp_path / "scatter.png"),
        beta.make_distance_rank_figure(task_results, analysis, tmp_path / "distance.png"),
        beta.make_verdict_figure(task_results, analysis, tmp_path / "verdict.png"),
    ]
    raw, summary = beta._save_raw(task_results, analysis, tmp_path)

    assert all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in figure_paths)
    assert Path(raw).is_file()
    payload = json.loads(Path(summary).read_text(encoding="utf-8"))
    assert payload["status"] == "BETA"
    assert payload["analysis"]["verdict"] == "SUPPORTED"


def test_standalone_colab_script_is_valid_and_calls_beta_runner():
    path = Path(__file__).resolve().parents[1] / "HeadResponseRichness_ZINC_Beta_Colab.py"
    code = path.read_text(encoding="utf-8")

    compile(code, str(path), "exec")
    assert "specialisation.richness_beta import run" in code
    assert 'num_graphs=128' in code
    assert "NO CLEAR SEPARATION" in code
