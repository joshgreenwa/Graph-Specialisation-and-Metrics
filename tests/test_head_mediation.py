from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import graph_specialisation_metrics.methodology.head_mediation as head_mediation
from graph_specialisation_metrics.methodology.head_mediation import (
    Config,
    _aggregate_graph_rows,
    _bootstrap_mediation_shares,
    _normalise_mass,
    _spearman,
)


def test_normalise_mass_is_channel_allocation() -> None:
    observed = _normalise_mass(np.asarray([1.0, 2.0, 1.0]))
    np.testing.assert_allclose(observed, [0.25, 0.50, 0.25])


def test_graph_aggregation_nests_donors_within_source() -> None:
    rows = [
        {"graph": 0, "source": 0, "P_gross_matched": 1.0},
        {"graph": 0, "source": 0, "P_gross_matched": 3.0},
        {"graph": 0, "source": 1, "P_gross_matched": 6.0},
        {"graph": 1, "source": 0, "P_gross_matched": 4.0},
    ]
    assert _aggregate_graph_rows(rows, "P_gross_matched") == {0: 4.0, 1: 4.0}


def test_bootstrap_mediation_shares_are_finite_and_normalised() -> None:
    matrix = np.asarray([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    low, high = _bootstrap_mediation_shares(matrix, replicates=100, seed=3)
    assert np.isfinite(low).all()
    assert np.isfinite(high).all()
    assert np.all(low <= high)
    assert np.all((low >= 0) & (high <= 1))


def test_spearman_handles_ties() -> None:
    assert np.isclose(_spearman(np.asarray([1, 2, 2, 4]), np.asarray([2, 3, 3, 8])), 1.0)


def test_analyse_compares_cached_scores_and_symmetric_mediation(monkeypatch, tmp_path) -> None:
    shape = (2, 2)
    semantic = np.asarray([[4.0, 2.0], [1.0, 1.0]])
    structural = np.asarray([[1.0, 1.0], [2.0, 4.0]])
    sem_norm = semantic / semantic.mean()
    str_norm = structural / structural.mean()
    score_value = {
        "coordinates": {
            "raw_semantic": semantic,
            "raw_structural": structural,
            "normalized_semantic": sem_norm,
            "normalized_structural": str_norm,
            "joint_sensitivity": 0.5 * (sem_norm + str_norm),
            "selectivity": (sem_norm - str_norm) / (sem_norm + str_norm + 1.0e-12),
            "active": np.ones(shape, dtype=bool),
            "estimable": True,
        },
        "channels": {
            "semantic": {"raw": semantic},
            "structural": {"raw": structural},
        },
        "intervals": SimpleNamespace(
            low=np.stack((semantic, structural, sem_norm, str_norm, sem_norm, sem_norm)),
            high=np.stack((semantic, structural, sem_norm, str_norm, sem_norm, sem_norm)),
        ),
    }
    targets = {}
    records = {}
    for layer in range(shape[0]):
        for head in range(shape[1]):
            name = f"head_L{layer}_H{head}"
            flat = layer * shape[1] + head
            targets[name] = {}
            records[name] = {}
            for channel, values in (("semantic", semantic), ("structural", structural)):
                value = float(values.reshape(-1)[flat])
                targets[name][channel] = {
                    "P_gross_matched": value,
                    "P_gross_mismatch": 0.1 * value,
                    "G_c": 0.9 * value,
                }
                records[name][channel] = [
                    {"graph": graph, "source": 0, "P_gross_matched": value}
                    for graph in (0, 1)
                ]
    contract = {
        "task": "zinc",
        "train_seed": 42,
        "checkpoint_sha256": "same",
        "task_adapter_version": "same",
        "model_geometry": {"layers": 2, "heads": 2},
        "split_fingerprint": "same",
        "sigma": [1.0],
    }
    score_artifact = SimpleNamespace(value=score_value, metadata={"contract": contract})
    causal_artifact = SimpleNamespace(
        value={"summary": {"targets": targets}, "event_records": records},
        metadata={"contract": contract},
    )
    monkeypatch.setattr(head_mediation, "load_canonical_score_artifact", lambda *a, **k: score_artifact)
    monkeypatch.setattr(head_mediation, "load_cache_artifact_file", lambda *a, **k: causal_artifact)

    result = head_mediation.analyse(
        Config(tmp_path / "canonical", tmp_path / "output", bootstrap_replicates=20)
    )
    assert not result["warnings"]
    assert len(result["rows"]) == 8
    assert result["channel_summary"][0]["spearman_raw"] == 1.0
    assert result["channel_summary"][0]["top_k_overlap"] == 4
