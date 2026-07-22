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
    assert result["F_coh_source"][0, 0, 0, 0].tolist() == pytest.approx([2.0, 0.0])
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
    assert rows[0]["F_coh"][0, 0] == pytest.approx(3.0)


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
            q_groups = [torch.ones(events, L, H, n, 1) * float(matrix.mean()) for _ in range(sources)]
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
                    "F_coh": 0.8 * matrix / (distance + 1),
                    "source_index": 0,
                    "carriers": 2,
                } for distance in (0, 1)],
            }
        descriptor = {
            "n": n,
            "labels": np.arange(n).reshape(-1, 1),
            "edges": np.array([[0, 1], [1, 2], [2, 3]]),
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


def _fake_causal_payload(offset: float = 0.0):
    L, H = 2, 2
    events = []
    for channel_index, channel in enumerate(BETA.CHANNELS):
        for graph_id in range(4):
            events.append({
                "channel": channel,
                "graph_id": graph_id,
                "clean_prediction": np.array([1.0]),
                "corrupt_prediction": np.array([0.7 - 0.03 * channel_index]),
                "topology_tier": 0,
            })
    shape = (len(events), L, H)
    base = np.linspace(0.01, 0.3, np.prod(shape)).reshape(shape) + offset
    return {
        "events": events,
        "causal_effect_floor_relative": 0.05,
        "metrics": {
            "restore": base,
            "inject": base * 0.9,
            "necessity": base * 0.7,
            "mismatch": base * 0.1,
            "sham": base * 0.001,
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
            "high_J": [(0, 1)],
            "high_G_balanced": [(1, 1)],
            "topology_responsive": [(0, 1)],
            "low_J_inert": [(1, 1)],
        }
        selections[aggregation] = {
            "semantic": semantic,
            "pe": pe,
            "topology": topology,
            "coordinates": coordinates,
            "active": np.ones((L, H), dtype=bool),
            "activity_floor": 0.01,
            "families": families,
        }
        method_rows.append({
            "aggregation": aggregation,
            "scope": "decision",
            "validity_mean": 0.5 + index * 0.01,
            "prefix_reliability": 0.8,
            "objective": 0.65 + index * 0.01,
            "selected": aggregation == "EN",
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
    events = [{"channel": channel} for channel in BETA.PRIMARY_CHANNELS for _ in range(2)]
    shape = (len(events), 2, 2)
    return {
        "events": events,
        "metrics": {
            "routing_q": np.ones(shape) * 0.2,
            "message_q": np.ones(shape) * 0.3,
            "routing_rescue": np.ones(shape) * 0.1,
            "message_rescue": np.ones(shape) * 0.15,
            "full_rescue": np.ones(shape) * 0.25,
            "finite_interaction": np.zeros(shape),
        },
        "reconstruction_max": np.zeros((len(events), 2)),
    }


def test_all_paper_outputs_render_from_schema_complete_smoke_payload(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    runs = {}
    for index, task in enumerate(BETA.ARCHITECTURES):
        score = _fake_score_payload(index * 0.02)
        causal = _fake_causal_payload(index * 0.01)
        runs[task] = {
            "scores": score,
            "causal": causal,
            "ablations": _fake_ablations(score, causal),
            "mechanism": _fake_mechanism(),
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
    assert summary["selected_aggregation"] == "EN"
    assert len(summary["figures"]) == 22
    assert all(Path(path).exists() for path in summary["figures"])
    assert (tmp_path / "tables/methodology_decisions.json").exists()
