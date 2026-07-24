from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "experiments/zinc/analysis/zinc_specialisation_redesign_beta_colab.py"
)
SPEC = importlib.util.spec_from_file_location("_zinc_redesign_beta", SOURCE)
assert SPEC is not None and SPEC.loader is not None
BETA = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BETA
SPEC.loader.exec_module(BETA)


def test_four_aggregations_preserve_donor_order_and_carrier_cancellation() -> None:
    torch = pytest.importorskip("torch")
    q = torch.zeros(1, 1, 2, 1, 1, 2, 1)
    q[0, 0, 0, 0, 0, :, 0] = torch.tensor([1.0, -1.0])
    q[0, 0, 1, 0, 0, :, 0] = torch.tensor([3.0, 1.0])
    result = BETA.aggregate_projected_events(q, torch.ones(1, 1, 2, dtype=torch.bool))
    observed = {key: float(value[0, 0, 0]) for key, value in result["per_graph"].items()}
    assert observed == pytest.approx({"CG": 2.0, "EG": 3.0, "CN": 2.0, "EN": 2.0})
    assert result["F_sens_source"][0, 0, 0, 0].tolist() == pytest.approx([2.0, 1.0])


def test_graph_balancing_averages_donors_then_sources() -> None:
    torch = pytest.importorskip("torch")
    q = torch.zeros(1, 2, 3, 1, 1, 1, 1)
    valid = torch.tensor([[[True, True, True], [True, False, False]]])
    q[0, 0, :, 0, 0, 0, 0] = torch.tensor([0.0, 0.0, 6.0])
    q[0, 1, 0, 0, 0, 0, 0] = 10.0
    result = BETA.aggregate_projected_events(q, valid)
    # Source means are 2 and 10; the graph mean is 6. Donor-rich source 0 is not upweighted.
    assert float(result["per_graph"]["CG"][0, 0, 0]) == pytest.approx(6.0)


def test_distance_profiles_use_each_events_own_changed_set() -> None:
    torch = pytest.importorskip("torch")
    q = torch.zeros(2, 1, 1, 2, 1)
    q[0, 0, 0, :, 0] = torch.tensor([2.0, 0.0])
    q[1, 0, 0, :, 0] = torch.tensor([0.0, 4.0])
    rows = BETA.distance_profile_one_graph(q, np.array([[0, 1], [1, 0]]))
    assert [row["distance"] for row in rows] == [0, 1]
    # At d=0, each event contributes at its own changed carrier: mean(|2|,|4|)=3.
    assert rows[0]["F_sens"][0, 0] == pytest.approx(3.0)


def test_distance_profiles_keep_signed_beneficial_carriage() -> None:
    torch = pytest.importorskip("torch")
    q = torch.ones(2, 1, 1, 2, 1)
    benefit = torch.tensor([[-2.0, 1.0], [-4.0, 3.0]])
    rows = BETA.distance_profile_one_graph(
        q, np.array([[0, 1], [0, 1]]), beneficial=benefit
    )
    assert rows[0]["B"] == pytest.approx(-3.0)
    assert rows[1]["B"] == pytest.approx(2.0)
    assert sum(row["B_sum"] for row in rows) == pytest.approx(-1.0)


def test_causal_hierarchy_averages_events_then_sources_within_graph() -> None:
    events = [
        {"channel": "semantic", "graph_id": 0, "source": 0,
         "clean_prediction": [1.0], "corrupt_prediction": [0.0]},
        {"channel": "semantic", "graph_id": 0, "source": 0,
         "clean_prediction": [1.0], "corrupt_prediction": [0.0]},
        {"channel": "semantic", "graph_id": 0, "source": 1,
         "clean_prediction": [1.0], "corrupt_prediction": [0.0]},
        {"channel": "semantic", "graph_id": 1, "source": 0,
         "clean_prediction": [1.0], "corrupt_prediction": [0.0]},
    ]
    graph_ids, values = BETA.hierarchical_channel_values(
        {"events": events, "causal_effect_floor_relative": 0.0},
        np.asarray([[1.0], [3.0], [10.0], [4.0]]),
        "semantic",
        reduction="signed",
    )
    np.testing.assert_array_equal(graph_ids, [0, 1])
    # graph 0: mean donors for source 0 = 2, then mean sources (2,10) = 6.
    np.testing.assert_allclose(values[:, 0], [6.0, 4.0])


def test_causal_focus_compares_channel_magnitudes_not_effect_signs() -> None:
    coordinates = BETA.causal_strength_coordinates(
        np.asarray([[[-2.0]]]),
        np.asarray([[[1.0]]]),
        positive_after_graph_mean=False,
    )
    assert coordinates["D"][0, 0] == pytest.approx(1.0 / 3.0)
    assert coordinates["J"][0, 0] == pytest.approx(1.5)


def test_distance_resolved_eg_exactly_reconstructs_score_with_event_specific_bins() -> None:
    torch = pytest.importorskip("torch")
    q_source_0 = torch.zeros(2, 1, 1, 3, 1)
    q_source_1 = torch.zeros(1, 1, 1, 3, 1)
    q_source_0[0, 0, 0, :, 0] = torch.tensor([1.0, 2.0, 3.0])
    q_source_0[1, 0, 0, :, 0] = torch.tensor([4.0, 5.0, 6.0])
    q_source_1[0, 0, 0, :, 0] = torch.tensor([7.0, 8.0, 9.0])
    result = BETA.distance_resolved_eg_one_graph(
        [q_source_0, q_source_1],
        [np.array([[0, 1, 2], [1, 0, 2]]), np.array([[0, 1, 2]])],
    )
    # Donors average within source, then the two source scores average equally.
    expected = ((1 + 2 + 3 + 4 + 5 + 6) / 2 + (7 + 8 + 9)) / 2
    assert float(result["total"][0, 0]) == pytest.approx(expected)
    assert sum(float(value[0, 0]) for value in result["contribution"].values()) == pytest.approx(
        expected
    )
    assert result["identity_max_abs_error"] < 1.0e-6


def test_distance_resolved_eg_keeps_unreachable_and_virtual_hub_buckets() -> None:
    torch = pytest.importorskip("torch")
    q = torch.tensor([[[[[1.0], [2.0], [3.0], [4.0]]]]])  # [K,L,H,N,T]
    result = BETA.distance_resolved_eg_one_graph(
        [q], [np.array([[0, 1, BETA.DISTANCE_UNREACHABLE, BETA.DISTANCE_HUB]])]
    )
    assert result["contribution"][BETA.DISTANCE_UNREACHABLE][0, 0] == pytest.approx(3.0)
    assert result["contribution"][BETA.DISTANCE_HUB][0, 0] == pytest.approx(4.0)
    assert result["total"][0, 0] == pytest.approx(10.0)


def test_distance_resolved_eg_retains_support_normalised_response() -> None:
    torch = pytest.importorskip("torch")
    q = torch.tensor([[[[[1.0], [2.0], [4.0]]]]])  # [K,L,H,N,T]
    result = BETA.distance_resolved_eg_one_graph(
        [q], [np.array([[0, 1, 1]])]
    )
    assert result["opportunity"] == pytest.approx({0: 1.0, 1: 2.0})
    assert result["contribution"][1][0, 0] == pytest.approx(6.0)
    assert result["density"][1][0, 0] == pytest.approx(3.0)
    # Support normalisation is diagnostic and does not alter exact EG reconstruction.
    assert result["total"][0, 0] == pytest.approx(7.0)


def test_attention_distance_profile_is_graph_balanced_and_head_normalised() -> None:
    records = []
    for graph_id, values in enumerate(((0.8, 0.2), (0.2, 0.8))):
        records.append({
            "graph_id": graph_id,
            "codes": [0, 1],
            "mass": {
                0: np.full((2, 2), values[0]),
                1: np.full((2, 2), values[1]),
            },
            "pair_count": {0: 2, 1: 6},
            "normalisation_error": np.zeros(2),
            "minimum_weight": np.zeros(2),
        })
    profile = BETA.attention_distance_profile(
        {"records": records}, bootstrap_samples=20, seed=4
    )
    assert np.allclose(profile["fraction"].sum(axis=0), 1.0)
    assert profile["aggregate_fraction"] == pytest.approx([0.5, 0.5])
    assert profile["graph_support"].tolist() == [2, 2]


def test_score_coordinates_separate_selectivity_activity_and_joint_strength() -> None:
    result = BETA.score_coordinates(np.array([[4.0, 2.0]]), np.array([[0.0, 2.0]]))
    assert np.allclose(result["D"], np.array([[1.0, 0.0]]))
    assert result["J"][0, 0] == pytest.approx(result["J"][0, 1])
    assert result["G"][0, 0] == pytest.approx(0.0)
    assert result["G"][0, 1] > 0.0


def test_causal_coordinates_retain_anti_aligned_signed_effects() -> None:
    payload = {
        "events": [
            {"channel": "semantic", "clean_prediction": [1.0], "corrupt_prediction": [0.0]},
            {"channel": "pe", "clean_prediction": [1.0], "corrupt_prediction": [0.0]},
        ],
        "causal_effect_floor_relative": 0.0,
        "metrics": {"restore": np.array([[[-1.0]], [[1.0]]])},
    }
    result = BETA.causal_coordinates(payload)
    assert result["D"][0, 0] == pytest.approx(-1.0)
    assert result["J"][0, 0] == pytest.approx(1.0)
    assert result["G"][0, 0] == pytest.approx(0.0)
    assert bool(result["semantic_anti_aligned"][0, 0])


def test_deterministic_splits_are_disjoint_and_repeatable() -> None:
    cfg = BETA.BetaConfig(score_graphs=10, causal_graphs=11, ablation_graphs=12)
    first = BETA.deterministic_splits(100, cfg)
    second = BETA.deterministic_splits(100, cfg)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert not set(first["score"]) & set(first["causal"])
    assert not set(first["score"]) & set(first["ablation"])
    assert not set(first["causal"]) & set(first["ablation"])
    assert set(first["mechanism"]).issubset(set(first["causal"]))


def test_integrated_carriage_defaults_use_bounded_rare_cap_policy() -> None:
    cfg = BETA.BetaConfig()
    assert cfg.integrated_atol == pytest.approx(5.0e-4)
    assert cfg.integrated_max_intervals == 256
    assert cfg.integrated_unconverged_error_cap == pytest.approx(5.0e-3)
    assert cfg.integrated_max_unconverged_fraction == pytest.approx(1.0e-2)
    cfg.validate()
    with pytest.raises(ValueError, match="must lie in"):
        BETA.BetaConfig(integrated_max_unconverged_fraction=1.1).validate()


def test_colab_resume_config_preserves_prior_cache_fingerprint(tmp_path) -> None:
    saved = BETA.BetaConfig(
        output_dir=str(tmp_path),
        score_graphs=128,
        score_sources=10,
        causal_graphs=64,
        ablation_graphs=160,
        bootstrap_samples=2000,
        conditional_bootstrap_samples=2000,
    )
    BETA.write_json(
        tmp_path / "beta_config.json",
        {
            "version": BETA.BETA_VERSION,
            "schema": BETA.BETA_SCHEMA,
            "fingerprint": saved.fingerprint,
            "config": BETA.asdict(saved),
        },
    )
    args = BETA.parse_args([
        "--phase", "all",
        "--output-dir", str(tmp_path),
        "--resume-config",
    ])
    resumed = BETA.resume_saved_config(args)
    assert not resumed.force
    assert resumed.score_graphs == 128
    assert resumed.bootstrap_samples == 2000
    assert BETA.make_config(resumed).fingerprint == saved.fingerprint


def test_integrated_carriage_audit_aggregates_cached_source_groups() -> None:
    diagnostics = [
        {
            "paths": 8,
            "unconverged": 1,
            "completeness_max": 2.0e-3,
            "quadrature_error_max": 3.0e-3,
            "unconverged_completeness_max": 2.0e-3,
            "unconverged_quadrature_error_max": 3.0e-3,
            "intervals_max": 256,
        },
        {
            "paths": 8,
            "unconverged": 0,
            "completeness_max": 1.0e-4,
            "quadrature_error_max": 2.0e-4,
            "intervals_max": 3,
        },
    ]
    records = [{
        "channels": {
            channel: {"beneficial_diagnostics": diagnostics if channel == "semantic" else []}
            for channel in BETA.CHANNELS
        }
    }]
    audit = BETA.integrated_carriage_audit(records)
    assert audit["paths"] == 16
    assert audit["unconverged"] == 1
    assert audit["unconverged_fraction"] == pytest.approx(1 / 16)
    assert audit["unconverged_quadrature_error_max"] == pytest.approx(3.0e-3)
    assert audit["intervals_max"] == 256


def test_score_component_cache_separates_source_channel_and_graph(tmp_path) -> None:
    cfg = BETA.BetaConfig(output_dir=str(tmp_path))
    semantic_source = BETA.score_component_cache_path(
        cfg, "zinc", 10, "semantic_source_0", "a" * 64
    )
    semantic_channel = BETA.score_component_cache_path(
        cfg, "zinc", 10, "semantic_complete", "a" * 64
    )
    pe_source = BETA.score_component_cache_path(
        cfg, "zinc", 10, "pe_source_0", "a" * 64
    )
    assert len({semantic_source, semantic_channel, pe_source}) == 3
    assert all("score_components" in str(path) for path in (
        semantic_source, semantic_channel, pe_source
    ))


def test_completed_tighter_legacy_graph_cache_is_migratable(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    cfg = BETA.BetaConfig(output_dir=str(tmp_path))
    checkpoint_sha = "b" * 64
    record = {
        "channels": {
            channel: {
                "beneficial_diagnostics": [{
                    "paths": 8,
                    "unconverged": 0,
                    "completeness_max": 1.0e-5,
                    "quadrature_error_max": 1.0e-5,
                }]
            }
            for channel in BETA.CHANNELS
        }
    }
    path = tmp_path / "legacy.pt"
    torch.save({
        "version": BETA.BETA_VERSION,
        "schema": BETA.BETA_SCHEMA,
        "fingerprint": cfg.legacy_strict_score_fingerprint,
        "checkpoint_sha256": checkpoint_sha,
        "record": record,
    }, path)
    loaded = BETA.valid_legacy_strict_graph_cache(path, cfg, checkpoint_sha)
    assert loaded is not None
    assert cfg.legacy_strict_score_fingerprint != cfg.fingerprint


def test_graph_num_nodes_adapter_supplies_collector_metadata() -> None:
    torch = pytest.importorskip("torch")

    class FakeBatch:
        num_nodes = 7

    class SizedGraph:
        def __init__(self, size: int):
            self.num_nodes = size

    batch = BETA.annotate_graph_num_nodes(
        FakeBatch(), [SizedGraph(3), SizedGraph(4)]
    )
    assert torch.equal(batch.graph_num_nodes, torch.tensor([3, 4]))


def test_graph_num_nodes_adapter_rejects_inconsistent_collation() -> None:
    class FakeBatch:
        num_nodes = 6

    class SizedGraph:
        num_nodes = 7

    with pytest.raises(RuntimeError, match="PyG batch has 6 nodes"):
        BETA.annotate_graph_num_nodes(FakeBatch(), [SizedGraph()])


def test_make_grit_batch_persists_counts_in_pyg_global_storage() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    graphs = [Data(x=torch.zeros(size, 1)) for size in (2, 5)]
    batch = BETA.make_grit_batch(graphs, torch.device("cpu"))
    assert batch.graph_num_nodes.tolist() == [2, 5]
    assert int(batch.graph_num_nodes.sum()) == int(batch.num_nodes)


class FakeData:
    def __init__(self, **values):
        self.__dict__.update(values)

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    def clone(self):
        return copy.deepcopy(self)

    def keys(self):
        return list(self.__dict__)


def _descriptor(edges, labels, graph_id=0):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1, 1)
    edges = np.asarray(edges, dtype=np.int64)
    n = len(labels)
    return {
        "graph_id": graph_id,
        "n": n,
        "node_multiset": tuple(sorted(tuple(row) for row in labels)),
        "degree_multiset": tuple(sorted(BETA.degrees_from_edges(n, edges))),
        "edge_count": len(edges),
        "edge_labels": tuple(),
        "degree": BETA.degrees_from_edges(n, edges),
        "edges": edges,
        "labels": labels,
        "cycle_rank": len(edges) - n + 1,
    }


def test_topology_matching_excludes_isomorphic_graphs_and_ranks_tiers() -> None:
    path = _descriptor([[0, 1], [1, 2], [2, 3]], [0, 1, 1, 0])
    relabelled_path = _descriptor([[3, 2], [2, 1], [1, 0]], [0, 1, 1, 0], 1)
    star = _descriptor([[0, 1], [0, 2], [0, 3]], [0, 1, 1, 0], 2)
    assert not BETA.nonisomorphic(path, relabelled_path)
    assert BETA.nonisomorphic(path, star)
    assert BETA.topology_match_tier(path, relabelled_path) == 0
    assert BETA.topology_match_tier(path, star) == 1


def test_topology_donor_selection_interleaves_available_tiers() -> None:
    base = _descriptor([[0, 1], [1, 2], [2, 3]], [0, 0, 0, 0])
    tier0 = _descriptor([[0, 1], [0, 2], [0, 3]], [0, 0, 0, 0], 10)
    # The test descriptor deliberately controls the matching summary independently of
    # topology so this candidate exercises the strict-tier branch while remaining non-isomorphic.
    tier0["degree_multiset"] = base["degree_multiset"]
    tier1 = _descriptor([[0, 1], [0, 2], [0, 3]], [0, 0, 0, 0], 11)
    tier2 = _descriptor([[0, 1], [1, 2], [2, 3], [3, 0]], [0, 0, 0, 0], 12)
    selected = BETA.select_topology_donors(
        base, [tier0, tier1, tier2], {4: [0, 1, 2]}, count=3, allow_relaxed=True
    )
    assert [item["tier"] for item in selected] == [0, 1, 2]


def test_hungarian_alignment_and_topology_transplant_hold_x_and_y_fixed() -> None:
    torch = pytest.importorskip("torch")
    base_desc = _descriptor([[0, 1], [1, 2]], [0, 1, 2])
    donor_desc = _descriptor([[1, 2], [2, 0]], [2, 0, 1], 7)
    alignment = BETA.align_donor_nodes(base_desc, donor_desc)
    assert alignment.tolist() == [1, 2, 0]

    base = FakeData(
        x=torch.tensor([[0], [1], [2]]),
        y=torch.tensor([1.5]),
        rrwp=torch.tensor([[10.0], [11.0], [12.0]]),
        deg=torch.tensor([1, 2, 1]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.tensor([[1], [1], [2], [2]]),
        rrwp_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        rrwp_val=torch.tensor([[0.5], [0.5], [0.5], [0.5]]),
    )
    donor = FakeData(
        x=torch.tensor([[2], [0], [1]]),
        y=torch.tensor([99.0]),
        rrwp=torch.tensor([[20.0], [21.0], [22.0]]),
        deg=torch.tensor([2, 1, 1]),
        edge_index=torch.tensor([[1, 2, 2, 0], [2, 1, 0, 2]]),
        edge_attr=torch.tensor([[3], [3], [4], [4]]),
        rrwp_index=torch.tensor([[1, 2, 2, 0], [2, 1, 0, 2]]),
        rrwp_val=torch.tensor([[0.25], [0.25], [0.75], [0.75]]),
    )
    variant = BETA.topology_donor_variant(base, donor, alignment)
    assert torch.equal(variant.x, base.x)
    assert torch.equal(variant.y, base.y)
    assert variant.rrwp.reshape(-1).tolist() == pytest.approx([21.0, 22.0, 20.0])
    expected_index = BETA.relabel_pair_index(donor.edge_index, alignment)
    assert torch.equal(variant.edge_index, expected_index)
    assert torch.equal(variant.edge_attr, donor.edge_attr)


def test_structural_field_audit_fails_unknown_pe_channel_visibly() -> None:
    torch = pytest.importorskip("torch")
    data = FakeData(
        x=torch.zeros(2, 1, dtype=torch.long),
        y=torch.zeros(1),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        mysterious_pe=torch.ones(2, 3),
    )
    audit = BETA.structural_field_audit(data)
    assert audit["unhandled_structural"] == ["mysterious_pe"]


def test_topology_edit_dose_uses_undirected_symmetric_difference() -> None:
    result = BETA.topology_edit_dose(
        np.array([[0, 1], [1, 2]]), np.array([[0, 1], [0, 2]])
    )
    assert result["edge_symmetric_difference"] == 2
    assert result["changed_nodes"] == [0, 1, 2]


def _path_data_for_local_topology(n: int = 14):
    torch = pytest.importorskip("torch")
    directed = []
    for node in range(n - 1):
        directed.extend(((node, node + 1), (node + 1, node)))
    edge_index = torch.tensor(directed, dtype=torch.long).T.contiguous()
    fields = BETA.recompute_rrwp_fields(edge_index, num_nodes=n, width=7)
    return FakeData(
        x=torch.zeros(n, 1, dtype=torch.long),
        y=torch.tensor([0.0]),
        edge_index=edge_index,
        edge_attr=torch.zeros(len(directed), 1, dtype=torch.long),
        **fields,
    )


def test_local_topology_switch_is_degree_preserving_and_exposes_long_reach() -> None:
    torch = pytest.importorskip("torch")
    base = _path_data_for_local_topology()
    plan = BETA.plan_local_topology_events(base, graph_id=3, seed=17)
    assert plan
    assert max(group["maximum_pristine_distance"] for group in plan) > 3
    event = plan[0]["events"][0]
    variant = BETA.local_topology_variant(base, event)
    assert torch.equal(base.x, variant.x)
    assert torch.equal(base.y, variant.y)
    assert torch.equal(base.deg, variant.deg)
    assert len(
        {tuple(edge) for edge in BETA.molecular_edges(base)}
        ^ {tuple(edge) for edge in BETA.molecular_edges(variant)}
    ) == 4
    assert not torch.equal(base.rrwp_val, variant.rrwp_val)


def test_local_topology_rrwp_recomputation_matches_pristine_loader_contract() -> None:
    base = _path_data_for_local_topology()
    audit = BETA.rrwp_reconstruction_audit(base)
    assert audit["passed"]
    assert audit["pair"]["max_abs_error"] == pytest.approx(0.0)
    assert audit["node"]["max_abs_error"] == pytest.approx(0.0)


def test_local_topology_event_distance_uses_exact_four_endpoint_changed_set() -> None:
    base = _path_data_for_local_topology()
    local_plan = BETA.plan_local_topology_events(
        base, graph_id=0, seed=1, maximum_sources=1, events_per_source=1
    )
    record = {
        "n": base.num_nodes,
        "plan": {
            "descriptor": {
                "edges": BETA.molecular_edges(base),
            },
            BETA.LOCAL_TOPOLOGY_CHANNEL: local_plan,
        },
    }
    observed = BETA.event_distance_matrix(
        record,
        BETA.LOCAL_TOPOLOGY_CHANNEL,
        0,
        event_count=1,
        carrier_count=base.num_nodes + 1,
    )
    changed = np.asarray(local_plan[0]["events"][0]["changed_nodes"], dtype=int)
    pristine = BETA.shortest_paths(base.num_nodes, BETA.molecular_edges(base))
    expected = np.min(pristine[:, changed], axis=1).astype(int)
    np.testing.assert_array_equal(observed[0, :base.num_nodes], expected)
    assert observed[0, -1] == BETA.DISTANCE_HUB
    assert observed[0, :base.num_nodes].max() > 3


def test_support_aware_decomposition_reconstructs_changed_support() -> None:
    torch = pytest.importorskip("torch")

    def capture(src, dst, attention, message, n=4):
        return {
            "src": torch.tensor(src),
            "dst": torch.tensor(dst),
            "attention": torch.tensor(attention, dtype=torch.float32).reshape(-1, 1),
            "message": torch.tensor(message, dtype=torch.float32).reshape(-1, 1, 1),
            "gradient": torch.ones(n, 1, 1),
            "head_output": BETA.aggregate_pairs_to_nodes(
                torch.tensor(attention, dtype=torch.float32).reshape(-1, 1, 1)
                * torch.tensor(message, dtype=torch.float32).reshape(-1, 1, 1),
                torch.tensor(dst),
                n,
            ),
        }

    clean = capture([0, 1, 2], [1, 2, 3], [0.2, 0.3, 0.5], [2.0, 3.0, 4.0])
    corrupt = capture([0, 3], [1, 2], [0.4, 0.6], [1.0, 5.0])
    result = BETA.decompose_support_aware_layer(clean, corrupt)
    assert result["reconstruction_max"] < 1.0e-6
    assert result["clean_only_pairs"] == 2
    assert result["corrupt_only_pairs"] == 1
    assert torch.allclose(
        result["direct_q"],
        result["routing_q"] + result["message_q"] + result["wiring_q"],
        atol=1.0e-6,
    )


def _fake_score_payload(offset: float = 0.0):
    torch = pytest.importorskip("torch")
    L, H, n, sources, events = 2, 2, 4, 2, 2
    records = []
    for graph_id in range(8):
        semantic = np.array([[1.0, 0.6], [0.4, 0.8]]) + offset + graph_id * 0.01
        pe = np.array([[0.3, 0.7], [0.9, 0.5]]) + offset + graph_id * 0.01
        topology = np.array([[0.2, 0.8], [0.7, 0.4]]) + offset + graph_id * 0.01
        channels = {}
        for channel, matrix in (("semantic", semantic), ("pe", pe), ("topology", topology)):
            per_source = np.stack([matrix, matrix * 1.05])
            # Match the cached EG field exactly: every event has carrier magnitudes
            # summing to the graph-level EG matrix.
            eg_matrix = matrix * (1 + 0.03 * BETA.AGGREGATIONS.index("EG"))
            q_template = torch.as_tensor(eg_matrix / n, dtype=torch.float32)[
                None, :, :, None, None
            ].repeat(events, 1, 1, n, 1)
            q_groups = [q_template.clone() for _ in range(sources)]
            if channel == "topology":
                q_groups = q_groups[:1]
            channels[channel] = {
                "available": True,
                "per_graph": {name: matrix * (1 + 0.03 * index) for index, name in enumerate(BETA.AGGREGATIONS)},
                "per_source": {name: per_source * (1 + 0.03 * index) for index, name in enumerate(BETA.AGGREGATIONS)},
                "q_groups": q_groups,
                "event_outcomes": np.linspace(-0.1, 0.1, sum(len(q) for q in q_groups)),
                "carriage": [{
                    "distance": distance,
                    "F_sens": matrix / (distance + 1),
                    "source_index": 0,
                    "carriers": 2,
                } for distance in (0, 1)],
                "model_carriage": [{
                    "distance": distance,
                    "F_sens": float(matrix.mean() / (distance + 1)),
                    "B": float((-1 if distance == 0 else 1) * 0.01 * matrix.mean()),
                    "B_sum": float((-1 if distance == 0 else 1) * 0.02 * matrix.mean()),
                    "source_index": 0,
                    "carriers": 2,
                } for distance in (0, 1)],
            }
        descriptor = {
            "n": n,
            "labels": np.arange(n).reshape(-1, 1),
            "edges": np.array([[0, 1], [1, 2], [2, 3]]),
            "degree": np.array([1, 2, 2, 1]),
            "cycle_rank": 0,
        }
        records.append({
            "graph_id": graph_id,
            "n": n,
            "target": np.array([0.5]),
            "throughput": np.ones((L, H)) * (1.0 + graph_id * 0.01),
            "plan": {
                "descriptor": descriptor,
                "sources": np.array([0, 1]),
                "semantic": [
                    {"source": source, "donor_graph_ids": np.array([10, 11]), "dose": np.array([1.0, 2.0])}
                    for source in (0, 1)
                ],
                "pe": [
                    {
                        "source": source,
                        "partners": np.array([2, 3]),
                        "degree_gap": np.array([0, 0]),
                        "dose": np.array([0.2, 0.3]),
                    }
                    for source in (0, 1)
                ],
                "topology": [
                    {
                        "tier": tier,
                        "donor_graph_id": 20 + tier,
                        "cost": float(tier),
                        "target_gap": 0.1,
                        "content_alignment_exact": True,
                        "dose": {
                            "edge_jaccard_distance": 0.4,
                            "changed_nodes": [0, 1],
                        },
                    }
                    for tier in (0, 1)
                ],
            },
            "channels": channels,
            "prefixes": {
                channel: {
                    "1": {name: channels[channel]["per_graph"][name] * 0.95 for name in BETA.AGGREGATIONS},
                    "2": {name: channels[channel]["per_graph"][name] for name in BETA.AGGREGATIONS},
                }
                for channel in BETA.PRIMARY_CHANNELS
            },
        })
    return {"L": L, "H": H, "records": records}


def _fake_topology_reach_payload(score):
    records = []
    for source in score["records"]:
        result = copy.deepcopy(source["channels"]["topology"])
        result["channel"] = BETA.LOCAL_TOPOLOGY_CHANNEL
        events = []
        for event_index in range(len(result["q_groups"][0])):
            events.append({
                "removed_edges": np.asarray([[0, 1], [2, 3]], dtype=np.int64),
                "added_edges": np.asarray([[0, 2], [1, 3]], dtype=np.int64),
                "changed_nodes": np.asarray([0, 1, 2, 3], dtype=np.int64),
                "dose": {
                    "edge_jaccard_distance": 0.5,
                    "changed_node_fraction": 1.0,
                    "maximum_pristine_distance": 0,
                    "changed_set_diameter": 3.0,
                },
            })
        records.append({
            "graph_id": source["graph_id"],
            "n": source["n"],
            "target": source["target"],
            "plan": {
                "descriptor": source["plan"]["descriptor"],
                BETA.LOCAL_TOPOLOGY_CHANNEL: [{
                    "source_edge": np.asarray([0, 1], dtype=np.int64),
                    "events": events,
                }],
            },
            "channels": {BETA.LOCAL_TOPOLOGY_CHANNEL: result},
        })
    return {
        "L": score["L"],
        "H": score["H"],
        "records": records,
        "reach_audit": {
            "protocol_version": BETA.TOPOLOGY_REACH_PROTOCOL_VERSION,
            "eligible_graphs": len(records),
            "events": 2 * len(records),
            "maximum_planned_distance": 0,
            "fraction_events_reaching_beyond_3": 0.0,
        },
        "integrated_carriage_audit": BETA.integrated_carriage_audit(
            records, (BETA.LOCAL_TOPOLOGY_CHANNEL,)
        ),
    }


def _fake_causal_payload(offset: float = 0.0):
    L, H = 2, 2
    events = []
    for channel_index, channel in enumerate(BETA.CHANNELS):
        for graph_id in range(4):
            events.append({
                "channel": channel,
                "graph_id": graph_id,
                "source": 0,
                "clean_prediction": np.array([1.0]),
                "corrupt_prediction": np.array([0.7 - 0.03 * channel_index]),
                "topology_tier": 0,
                "topology_dose": {"edge_jaccard_distance": 0.3 + 0.1 * graph_id},
            })
    shape = (len(events), L, H)
    base = np.linspace(0.01, 0.3, np.prod(shape)).reshape(shape) + offset
    family_names = list(BETA.PATCH_FAMILY_NAMES)
    family_base = np.linspace(0.02, 0.2, len(events) * len(family_names)).reshape(
        len(events), len(family_names)
    )
    return {
        "events": events,
        "causal_effect_floor_relative": 0.05,
        "metrics": {
            "restore": base,
            "inject": base * 0.9,
            "necessity": base * 0.7,
            "restore_fraction": base * 2.0,
            "inject_fraction": base * 1.8,
            "necessity_fraction": base * 1.4,
            "mismatch": base * 0.1,
            "sham": base * 0.001,
        },
        "family_names": family_names,
        "family_definitions": {
            "semantic_specialist": [(0, 0)],
            "pe_specialist": [(1, 0)],
            "semantic_D_extreme": [(0, 0), (0, 1)],
            "pe_D_extreme": [(1, 0), (1, 1)],
            "semantic_D_J_matched": [(1, 1)],
            "pe_D_J_matched": [(0, 1)],
            "structural_pe_specific": [(1, 0)],
            "structural_topology_specific": [(0, 1)],
            "structural_shared": [(1, 1)],
            "high_J": [(0, 1)],
            "high_G_balanced": [(1, 1)],
            "low_J_inert": [(1, 1)],
        },
        "family_metrics": {
            "restore": family_base,
            "inject": family_base * 0.9,
            "necessity": family_base * 0.7,
            "mismatch": family_base * 0.1,
            "sham": family_base * 0.001,
            "restore_loss": family_base * 0.5,
            "inject_loss": family_base * 0.4,
            "necessity_loss": family_base * 0.3,
        },
    }


def _fake_ablations(score, causal):
    L, H = score["L"], score["H"]
    method_rows = BETA.aggregation_diagnostics(score, causal)
    selections = {}
    family_curves = {}
    for index, aggregation in enumerate(BETA.AGGREGATIONS):
        semantic = BETA.mean_score(score, "semantic", aggregation)
        pe = BETA.mean_score(score, "pe", aggregation)
        topology = BETA.mean_score(score, "topology", aggregation)
        coordinates = BETA.score_coordinates(semantic, pe)
        families = {
            "semantic_specialist": [(0, 0)],
            "pe_specialist": [(1, 0)],
            "semantic_D_extreme": [(0, 0), (0, 1)],
            "pe_D_extreme": [(1, 0), (1, 1)],
            "semantic_D_J_matched": [(1, 1)],
            "pe_D_J_matched": [(0, 1)],
            "high_J": [(0, 1)],
            "high_G_balanced": [(1, 1)],
            "topology_responsive": [(0, 1)],
            "structural_pe_specific": [(1, 0)],
            "structural_topology_specific": [(0, 1)],
            "structural_shared": [(1, 1)],
            "low_J_inert": [(1, 1)],
        }
        pe_topology_coordinates = BETA.score_coordinates(pe, topology)
        selections[aggregation] = {
            "semantic": semantic,
            "pe": pe,
            "topology": topology,
            "coordinates": coordinates,
            "active": np.ones((L, H), dtype=bool),
            "activity_floor": 0.01,
            "pe_topology_coordinates": pe_topology_coordinates,
            "pe_topology_pe": pe,
            "pe_topology_topology": topology,
            "semantic_topology_semantic": semantic,
            "semantic_topology_topology": topology,
            "pe_topology_active": np.ones((L, H), dtype=bool),
            "pe_topology_activity_floor": 0.01,
            "pe_topology_D_ci_lower": pe_topology_coordinates["D"] - 0.05,
            "pe_topology_D_ci_upper": pe_topology_coordinates["D"] + 0.05,
            "families": families,
        }
        method_rows.append({
            "aggregation": aggregation,
            "scope": "decision",
            "validity_mean": 0.5 + index * 0.01,
            "prefix_reliability": 0.8,
            "objective": 0.65 + index * 0.01,
            "selected": aggregation == BETA.HEADLINE_AGGREGATION,
        })
        family_curves[aggregation] = {
            family: [{"count": 1, "functional": 0.1 + 0.01 * j, "loss_increase": 0.0, "mae": 0.1}]
            for j, family in enumerate(families)
        }
    return {
        "method_rows": method_rows,
        "selections": selections,
        "family_curves": family_curves,
        "per_head": {
            "functional": np.array([[0.2, 0.1], [0.15, 0.05]]),
            "loss_increase": np.zeros((L, H)),
        },
    }


def _fake_mechanism():
    events = [{"channel": channel} for channel in BETA.CHANNELS for _ in range(2)]
    shape = (len(events), 2, 2)
    return {
        "events": events,
        "metrics": {
            "routing_q": np.ones(shape) * 0.2,
            "message_q": np.ones(shape) * 0.3,
            "wiring_q": np.ones(shape) * 0.05,
            "routing_rescue": np.ones(shape) * 0.1,
            "message_rescue": np.ones(shape) * 0.15,
            "wiring_rescue": np.ones(shape) * 0.03,
            "full_rescue": np.ones(shape) * 0.28,
            "finite_interaction": np.zeros(shape),
        },
        "reconstruction_max": np.zeros((len(events), 2)),
    }


def _fake_attention():
    records = []
    for graph_id in range(8):
        scale = 1.0 + 0.01 * graph_id
        records.append({
            "graph_id": graph_id,
            "codes": [0, 1, 2, 3],
            "mass": {
                distance: np.full((2, 2), value * scale)
                for distance, value in enumerate((0.5, 0.3, 0.15, 0.05))
            },
            "pair_count": {0: 4, 1: 8, 2: 6, 3: 2},
            "normalisation_error": np.zeros(2),
            "minimum_weight": np.zeros(2),
        })
    return {"records": records}


def _fake_head_gallery(score, causal):
    selected = BETA.select_causal_head_gallery_heads(
        score, causal, heads_per_family=1
    )
    heads = []
    graph_ids = set()
    for family_rows in selected.values():
        for row in family_rows:
            head = (int(row["layer"]), int(row["head"]))
            row["examples"] = BETA.rank_small_head_gallery_examples(
                score, head, row["channel"], max_nodes=8, examples=2
            )
            heads.append(head)
            graph_ids.update(example["graph_id"] for example in row["examples"])
    molecules = []
    for graph_id in sorted(graph_ids):
        record = next(
            item for item in score["records"]
            if int(item["graph_id"]) == int(graph_id)
        )
        n = int(record["n"])
        positions = np.column_stack([
            np.arange(n, dtype=float),
            np.zeros(n, dtype=float),
        ])
        maps = {}
        for layer, head in heads:
            matrix = np.eye(n, dtype=float) * 0.4
            for node in range(n - 1):
                matrix[node + 1, node] = 0.6
            maps[(layer, head)] = matrix
        molecules.append({
            "graph_id": int(graph_id),
            "n": n,
            "atom_types": np.arange(n, dtype=int),
            "bonds": np.asarray(record["plan"]["descriptor"]["edges"], dtype=int),
            "bond_types": np.ones(n - 1, dtype=int),
            "pos": positions,
            "maps": maps,
        })
    return {
        "head_gallery_protocol_version": BETA.HEAD_GALLERY_PROTOCOL_VERSION,
        "selections": selected,
        "attention": {
            "molecules": molecules,
            "heads": heads,
            "has_vnode": False,
            "atom_encoding": "category",
        },
    }


def test_conditional_screen_aligns_condition_and_intervention_units() -> None:
    assert BETA.conditional_feature_rule_compatible(
        "D_semantic_pe", "source_degree_high"
    )
    assert BETA.conditional_feature_rule_compatible(
        "D_semantic_topology", "cycle_rank_high"
    )
    assert not BETA.conditional_feature_rule_compatible(
        "D_semantic_topology", "source_degree_high"
    )
    assert not BETA.conditional_feature_rule_compatible(
        "G_three_channel", "source_in_cycle"
    )
    assert BETA.conditional_feature_effect_type("D_semantic_pe") == "selectivity_shift"
    assert BETA.conditional_rule_family("source_degree_high") == "routing_load"


def test_differential_causal_validation_generalises_to_semantic_topology() -> None:
    score = _fake_score_payload()
    causal = _fake_causal_payload()
    cfg = BETA.BetaConfig(
        bootstrap_samples=10,
        causal_graphs=4,
    )
    result = BETA.differential_causal_validation(
        score, causal, cfg, "EG", "semantic", "topology"
    )
    assert result["left_channel"] == "semantic"
    assert result["right_channel"] == "topology"
    assert result["score"]["D"].shape == (2, 2)
    assert "gross_restore" in result["specifications"]
    assert result["specifications"]["gross_restore"]["graphs"] == 4


def test_conditional_causal_validation_uses_independent_event_context() -> None:
    score = _fake_score_payload()
    causal = _fake_causal_payload()
    descriptors = {
        int(record["graph_id"]): record["plan"]["descriptor"]
        for record in score["records"]
    }
    for event in causal["events"]:
        event["source"] = int(event["graph_id"]) % 2
        event["descriptor"] = descriptors[int(event["graph_id"])]
    conditional = {
        "metadata": {
            "thresholds_from_discovery": {
                "n": 3.0,
                "cycle_rank": 0.0,
                "atom_diversity": 3.0,
                "mean_degree": 1.0,
                "edge_density": 0.2,
                "source_degree": 1.5,
            },
            "common_atom_codes": [],
        },
        "confirmed": [{
            "feature": "D_semantic_pe",
            "rule": "source_degree_high",
            "layer": 0,
            "head": 0,
            "confirmation_effect": 0.2,
            "confirmed_at_fdr_0.05": True,
            "effect_type": "selectivity_shift",
            "condition_family": "routing_load",
            "comparison_scope": "source_local",
        }],
    }
    cfg = BETA.BetaConfig(
        min_condition_graphs=2,
        conditional_bootstrap_samples=20,
    )
    result = BETA.conditional_causal_validation(conditional, causal, cfg)
    assert result["available"]
    assert result["eligible_tests"] == 1
    assert result["tested"][0]["causal_test_eligible"]
    assert result["tested"][0]["causal_true_graphs"] == 2
    assert result["tested"][0]["causal_false_graphs"] == 2


def test_all_paper_outputs_render_from_schema_complete_smoke_payload(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    runs = {}
    for index, task in enumerate(BETA.ARCHITECTURES):
        score = _fake_score_payload(index * 0.02)
        causal = _fake_causal_payload(index * 0.01)
        runs[task] = {
            "scores": score,
            "topology_reach": _fake_topology_reach_payload(score),
            "attention": _fake_attention(),
            "causal": causal,
            "ablations": _fake_ablations(score, causal),
            "mechanism": _fake_mechanism(),
            "head_gallery": _fake_head_gallery(score, causal),
        }
    cfg = BETA.BetaConfig(
        output_dir=str(tmp_path),
        score_graphs=8,
        causal_graphs=4,
        ablation_graphs=8,
        bootstrap_samples=10,
        conditional_bootstrap_samples=10,
        min_condition_graphs=2,
    )
    summary = BETA.create_outputs(runs, cfg)
    assert summary["selected_aggregation"] == "EG"
    assert len(summary["figures"]) == 54
    assert all(Path(path).exists() for path in summary["figures"])
    assert (tmp_path / "tables/causal_head_gallery_selection.csv").exists()
    assert (tmp_path / "tables/methodology_decisions.json").exists()
    assert (tmp_path / "tables/distance_resolved_specialisation.csv").exists()
    assert (tmp_path / "tables/clean_attention_distance_profiles.csv").exists()
    assert (tmp_path / "tables/distance_locality_summary_curves.csv").exists()
    assert summary["decisions"]["distance_resolved_specialisation"]["verdict"] == (
        "retain_exact_decomposition"
    )
