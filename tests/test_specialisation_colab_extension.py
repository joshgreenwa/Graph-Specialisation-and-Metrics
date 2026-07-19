import inspect
import json
from pathlib import Path

import numpy as np

from graph_specialisation_metrics.specialisation.colab import (
    _failed_transport_result,
    _persist_causal_extension,
    run,
)


def _summary(value):
    return {"mean": value, "ci_low": value - 0.1, "ci_high": value + 0.1}


def test_causal_extension_is_default_and_has_explicit_legacy_switch():
    parameters = inspect.signature(run).parameters
    assert parameters["causal_extension"].default is True
    assert parameters["legacy_outputs"].default is False


def test_extension_persistence_writes_reproducibility_bundle(tmp_path):
    patched = np.zeros((2, 2, 1), dtype=float)
    mediation = {
        "calibration": {
            "importance": np.array([[2.0, 1.0]]),
            "preference": np.array([[0.8, -0.7]]),
            "eligible": np.array([[True, True]]),
            "semantic_calibrated": np.array([[1.8, 0.2]]),
            "rrwp_role_calibrated": np.array([[0.2, 1.8]]),
            "semantic_scale": 1.0,
            "rrwp_role_scale": 2.0,
            "importance_floor": 1.0,
            "rankings": {"semantic": [[0, 0]], "rrwp_role": [[0, 1]]},
        },
        "intervention_bank": [
            {"event_id": "e0", "graph_id": 3, "channel": "semantic"},
            {"event_id": "e1", "graph_id": 4, "channel": "rrwp_role"},
        ],
        "predictions": {
            "clean": np.zeros((2, 1)),
            "counterfactual": np.ones((2, 1)),
            "targets": np.zeros((2, 1)),
        },
        "head_order": np.array([[0, 0], [0, 1]]),
        "discovery_graph_ids": np.array([0, 1]),
        "confirmation_graph_ids": np.array([3, 4]),
        "single_head": {
            "summary": [{"layer": 0, "head": 0, "beta": 0.5}],
            "patched_predictions": {"noising": patched, "denoising": patched},
        },
        "groups": {
            "summary": [{"group_id": "semantic_top1", "beta": 0.5}],
            "specs": [{"group_id": "semantic_top1", "heads": [[0, 0]]}],
            "patched_predictions": {"noising": patched, "denoising": patched},
        },
        "contrasts": {
            "topk": [{"k": 1, "double_dissociation": 0.4}],
            "interactions": [{"k": 1, "interaction_beta": 0.1}],
            "continuous": [{"preference_slope": 0.3}],
            "primary": [{"k": 1, "double_dissociation": 0.4}],
        },
        "config": {"primary_k": 1},
        "checks": {"all_finite": True},
    }
    effect = {
        "heads": [[0, 0]],
        "delta_mae": _summary(0.02),
        "abs_delta_pred": _summary(0.03),
        "delta_mae_per_graph": np.array([0.01, 0.03]),
        "abs_delta_pred_per_graph": np.array([0.02, 0.04]),
    }
    transport = {
        "graph_ids": np.array([3, 4]),
        "head_metrics": {"routing_js_static": np.array([[0.2, 0.3]])},
        "head_metrics_per_graph": {
            "routing_js_static": np.array([[[0.2, 0.3]], [[0.1, 0.4]]])
        },
        "causal": {"semantic": {"1": {"broadcast_only": effect}}},
        "config": {"primary_k": 1},
        "checks": {"passed": True},
        "head_selection": {"semantic": {"1": [[0, 0]]}},
        "clean": {"mae": 0.1},
    }

    written = _persist_causal_extension(tmp_path, mediation, transport)

    extension_dir = Path(written["directory"])
    expected = {
        "head_selection.csv",
        "intervention_bank.json",
        "mediation_predictions.npz",
        "mediation_single_head.csv",
        "mediation_groups.csv",
        "mediation_topk_contrasts.csv",
        "mediation_group_interactions.csv",
        "mediation_continuous_selectivity.csv",
        "effective_transport_head_metrics.npz",
        "effective_transport_summary.csv",
        "effective_transport_per_graph.csv",
        "causal_extension_summary.json",
    }
    assert expected <= {path.name for path in extension_dir.iterdir()}
    summary = json.loads((extension_dir / "causal_extension_summary.json").read_text())
    assert summary["mediation"]["checks"]["all_finite"] is True
    assert summary["effective_transport"]["checks"]["passed"] is True


def test_failed_transport_is_persisted_without_mechanism_estimates(tmp_path):
    mediation = {
        "calibration": {
            "importance": np.array([[2.0]]),
            "preference": np.array([[1.0]]),
            "eligible": np.array([[True]]),
            "semantic_calibrated": np.array([[2.0]]),
            "rrwp_role_calibrated": np.array([[0.0]]),
            "semantic_scale": 1.0,
            "rrwp_role_scale": 1.0,
            "importance_floor": 1.0,
            "rankings": {"semantic": [[0, 0]], "rrwp_role": [[0, 0]]},
        },
        "intervention_bank": [],
        "predictions": {
            "clean": np.zeros((1, 1)),
            "counterfactual": np.ones((1, 1)),
            "targets": np.zeros((1, 1)),
        },
        "head_order": np.array([[0, 0]]),
        "discovery_graph_ids": np.array([0]),
        "confirmation_graph_ids": np.array([1]),
        "single_head": {
            "summary": [],
            "patched_predictions": {
                "noising": np.zeros((1, 1, 1)),
                "denoising": np.zeros((1, 1, 1)),
            },
        },
        "groups": {
            "summary": [],
            "specs": [],
            "patched_predictions": {
                "noising": np.zeros((1, 1, 1)),
                "denoising": np.zeros((1, 1, 1)),
            },
        },
        "contrasts": {"topk": [], "interactions": [], "continuous": [], "primary": []},
        "config": {},
        "checks": {},
    }
    failed = _failed_transport_result(
        mediation,
        topk=(1,),
        primary_k=1,
        residual_permutations=1,
        bootstrap_replicates=10,
        seed=0,
        exc=RuntimeError("pair reconstruction mismatch"),
    )

    _persist_causal_extension(tmp_path, mediation, failed)
    summary = json.loads(
        (tmp_path / "causal_extension" / "causal_extension_summary.json").read_text()
    )
    assert summary["effective_transport"]["checks"]["passed"] is False
    assert summary["effective_transport"]["checks"]["mechanistic_estimates_withheld"] is True
    assert summary["effective_transport"]["clean_mae"] is None
