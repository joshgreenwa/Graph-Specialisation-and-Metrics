from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np

from graph_specialisation_metrics.methodology.causal_spatial_support import (
    Config,
    _event_conditions,
    _graph_profile_matrix,
    _load_task_context,
    _measurement_contract,
    _normalise_profile,
    _profile_interval,
    _summary_contract,
    select_role_heads,
)
from graph_specialisation_metrics.methodology.graphormer_figure_data import (
    CanonicalHeadMetrics,
)


def test_event_conditions_cover_each_shell_and_whole_graph() -> None:
    pristine = np.asarray([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    conditions = _event_conditions({"family:semantic": ((0, 1),)}, pristine, 0)
    assert [row["distance"] for row in conditions] == [0, 1, 2, "all"]
    assert conditions[2]["nodes"] == (2,)
    assert conditions[-1]["nodes"] == (0, 1, 2)


def test_event_conditions_add_graph_token_as_an_explicit_carrier() -> None:
    pristine = np.asarray([[0, 1], [1, 0]], dtype=float)
    conditions = _event_conditions(
        {"head:0:1": ((0, 1),)},
        pristine,
        0,
        special_carriers=("graph_token",),
    )
    graph_token = [row for row in conditions if row["distance"] == "graph_token"]
    assert len(graph_token) == 1
    assert graph_token[0]["nodes"] == ()
    assert graph_token[0]["special_carriers"] == ("graph_token",)
    whole = [row for row in conditions if row["distance"] == "all"]
    assert whole[0]["special_carriers"] == ("graph_token",)


def test_graph_profile_aggregates_donors_then_sources() -> None:
    rows = [
        {"graph": 0, "source": 0, "donor": 0, "target": "x", "distance": 0, "M": 1.0},
        {"graph": 0, "source": 0, "donor": 1, "target": "x", "distance": 0, "M": 3.0},
        {"graph": 0, "source": 1, "donor": 0, "target": "x", "distance": 0, "M": 6.0},
    ]
    matrix, graphs = _graph_profile_matrix(rows, key="M", distances=(0, 1), target="x")
    assert graphs == (0,)
    np.testing.assert_allclose(matrix, [[4.0, 0.0]])


def test_role_selection_uses_frozen_families_and_controls() -> None:
    shape = (2, 4)
    metrics = CanonicalHeadMetrics(
        raw_semantic=np.ones(shape),
        raw_structural=np.ones(shape),
        normalized_semantic=np.ones(shape),
        normalized_structural=np.ones(shape),
        joint_sensitivity=np.arange(8, dtype=float).reshape(shape),
        selectivity=np.linspace(-1, 1, 8).reshape(shape),
        active=np.ones(shape, dtype=bool),
        estimable=True,
        distance_axis=(),
        clean_attention_distance=None,
    )
    scores = {
        "families": {
            "semantic_leaning": ((1, 3), (1, 2)),
            "structural_leaning": ((0, 0), (0, 1)),
            "central_responsive": ((1, 1), (1, 0)),
        },
        "matched_controls": {
            "semantic_leaning_central_control": ((0, 3), (0, 2)),
            "structural_leaning_central_control": ((1, 0), (1, 1)),
        },
    }
    roles = select_role_heads(scores, metrics, count=1)
    assert roles["semantic"] == ((1, 3),)
    assert roles["structural"] == ((0, 0),)
    assert roles["generalist"] == ((1, 1),)
    assert roles["semantic_control"] == ((0, 3),)


def test_graphormer_context_uses_score_contract_without_protocol_file(
    tmp_path, monkeypatch
) -> None:
    from graph_specialisation_metrics.methodology import causal_spatial_support as module

    artifact = SimpleNamespace(
        value={},
        file_sha256="score-sha",
        metadata={
            "contract_fingerprint": "contract-fingerprint",
            "contract": {
                "checkpoint_sha256": "checkpoint-sha",
                "split_fingerprint": "split-fingerprint",
                "model_geometry": {"layers": 1, "heads": 1},
            },
        },
    )
    monkeypatch.setattr(module, "load_graphormer_score_artifact", lambda *_: artifact)
    monkeypatch.setattr(module, "load_graphormer_model_record", lambda *_: {"splits": {}})
    monkeypatch.setattr(module.CanonicalHeadMetrics, "from_scores", lambda *_: object())
    monkeypatch.setattr(
        module, "select_role_heads", lambda *_args, **_kwargs: {"semantic": ((0, 0),)}
    )
    context = _load_task_context(
        Config(
            canonical_root=tmp_path,
            output_dir=tmp_path / "output",
            tasks=("graphormer_pcqm4mv2",),
            train_seed=0,
        ),
        "graphormer_pcqm4mv2",
    )
    assert context["runtime_kind"] == "graphormer"
    assert context["protocol_config"] is None


def test_graph_shard_contract_survives_pilot_to_full_summary_change(tmp_path) -> None:
    artifact = SimpleNamespace(
        file_sha256="score-sha",
        metadata={
            "contract_fingerprint": "canonical-fingerprint",
            "contract": {
                "checkpoint_sha256": "checkpoint-sha",
                "split_fingerprint": "split-fingerprint",
                "model_geometry": {"layers": 12, "heads": 32},
            },
        },
    )
    pilot = Config(
        canonical_root=tmp_path,
        output_dir=tmp_path / "out",
        graphs=4,
        bootstrap_replicates=1_000,
    )
    full = dataclasses.replace(pilot, graphs=8, bootstrap_replicates=2_000)
    roles = {"semantic": ((0, 0),)}
    pilot_measurement = _measurement_contract(pilot, "graphormer_pcqm4mv2", artifact, roles)
    full_measurement = _measurement_contract(full, "graphormer_pcqm4mv2", artifact, roles)
    assert pilot_measurement == full_measurement
    assert _summary_contract(pilot, pilot_measurement) != _summary_contract(full, full_measurement)


def test_normalise_profile_is_radial_allocation() -> None:
    np.testing.assert_allclose(_normalise_profile([1.0, 2.0, 1.0]), [0.25, 0.5, 0.25])


def test_empty_profile_interval_preserves_distance_axis() -> None:
    profile = _profile_interval(np.asarray([], dtype=float), width=4, replicates=10, seed=1)
    assert len(profile["mean"]) == 4
    assert np.isnan(profile["mean"]).all()


def test_profile_interval_derives_long_range_share() -> None:
    profile = _profile_interval(
        np.asarray([[1.0, 1.0, 2.0]], dtype=float),
        width=3,
        distances=(0, 1, 3),
        long_range_radius=1,
        replicates=10,
        seed=1,
    )
    assert profile["expected_distance"] == 1.75
    assert profile["long_range_share"] == 0.5


def test_profile_interval_reports_graph_token_without_treating_it_as_spd() -> None:
    profile = _profile_interval(
        np.asarray([[1.0, 1.0, 2.0]], dtype=float),
        width=3,
        distances=(0, 3, "graph_token"),
        long_range_radius=1,
        replicates=10,
        seed=1,
    )
    assert profile["expected_distance"] == 1.5
    assert profile["long_range_share"] == 0.25
    assert profile["special_carrier_share"]["graph_token"] == 0.5
