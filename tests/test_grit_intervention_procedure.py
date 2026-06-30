from types import SimpleNamespace

import torch

from graph_specialisation_metrics.grit_intervention_procedure import (
    assert_attention_support_audit,
    attention_faithfulness_rows,
    attention_support_audit_rows,
    direct_fraction_value,
    r_nc_estimate,
)


def _cache(edge_index: torch.Tensor, n: int = 3):
    dense = torch.zeros(1, n, n)
    dense[:, edge_index[1], edge_index[0]] = 1.0
    return SimpleNamespace(attention=[dense], extras={"attention_edges": [edge_index]})


def test_onehop_attention_support_audit_accepts_self_and_neighbors():
    dist = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )
    edge_index = torch.tensor([[0, 1, 2, 1], [0, 1, 2, 2]])
    rows = attention_support_audit_rows(
        _cache(edge_index),
        dist,
        "grit_1hop",
        "g0",
        expected_max_direct_distance=1,
    )
    assert rows[0]["passes_expected_direct_support"] is True
    assert rows[0]["expected_distance_violating_edges"] == 0
    assert_attention_support_audit(rows)


def test_onehop_attention_support_audit_rejects_far_direct_edges():
    dist = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )
    edge_index = torch.tensor([[0, 0], [0, 2]])
    rows = attention_support_audit_rows(
        _cache(edge_index),
        dist,
        "grit_1hop",
        "g0",
        expected_max_direct_distance=1,
    )
    assert rows[0]["passes_expected_direct_support"] is False
    assert rows[0]["expected_distance_violating_edges"] == 1
    try:
        assert_attention_support_audit(rows)
    except RuntimeError as exc:
        assert "1-hop GRIT locality audit failed" in str(exc)
    else:
        raise AssertionError("expected 1-hop GRIT locality audit failure")


def test_attention_faithfulness_rows_records_estimator_name():
    dist = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    carriage = torch.eye(2)
    attention = {"attention_last": torch.eye(2)}
    rows = attention_faithfulness_rows(
        "dense_grit",
        "g0",
        attention,
        carriage,
        dist,
        tau=0,
        carriage_estimator="swap",
    )
    assert rows
    assert {row["carriage_estimator"] for row in rows} == {"swap"}


def test_direct_fraction_ignores_tiny_unclamped_effects():
    value, nontrivial = direct_fraction_value(0.5, 1.0e-9, min_effect_abs=1.0e-6)
    assert value != value
    assert nontrivial is False
    value, nontrivial = direct_fraction_value(0.5, 2.0, min_effect_abs=1.0e-6)
    assert value == 0.25
    assert nontrivial is True


def test_r_nc_scales_sampled_pairs_and_records_coverage():
    direct = torch.full((3, 3), float("nan"))
    direct[0, 2] = 2.0
    dist = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )
    stats = r_nc_estimate(direct, dist, threshold=1)
    assert stats["r_nc_raw_sample_sum"] == 2.0
    assert stats["r_nc_sampled_pairs"] == 1
    assert stats["r_nc_total_far_pairs"] == 2
    assert stats["r_nc"] == 4.0
    assert stats["r_nc_scaled_from_sample"] is True
