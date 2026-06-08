from dataclasses import dataclass

import pytest
import torch

from graph_specialisation_metrics.mechanistic_operator_analysis import (
    OperatorTransportAccumulator,
    batch_tensor_bytes,
    build_parser,
    collate_analysis_batches,
    ensure_receiver_support,
    flow_operator_masks,
    load_task_transport_responsibility,
    narrative_claim_row,
    operator_specificity_summary_rows,
    ranked_ablation_auc_rows,
    rate_matched_random_masks,
    valid_pair_mask,
)


@dataclass
class FakeBatch:
    node_type: torch.Tensor
    node_mask: torch.Tensor
    adj: torch.Tensor
    edge_value_mat: torch.Tensor
    degree: torch.Tensor
    spd: torch.Tensor
    edge_batch: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_value: torch.Tensor
    task_type: str = "graph_regression"


@dataclass
class TinyBatch:
    x: torch.Tensor
    y: torch.Tensor

    def to(self, device: torch.device) -> "TinyBatch":
        return TinyBatch(self.x.to(device), self.y.to(device))


class TinyRunner:
    def collate_graphs(self, graphs):
        del graphs
        return TinyBatch(torch.zeros(2, 3, dtype=torch.float32), torch.zeros(1, dtype=torch.long))


def make_flow_batch() -> FakeBatch:
    node_type = torch.tensor([[1, 0, 0, 2]])
    node_mask = torch.ones(1, 4, dtype=torch.bool)
    edges = [(0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)]
    edge_src = torch.tensor([src for src, _dst, _cap in edges])
    edge_dst = torch.tensor([dst for _src, dst, _cap in edges])
    edge_value = torch.tensor([cap for _src, _dst, cap in edges])
    edge_batch = torch.zeros(len(edges), dtype=torch.long)
    adj = torch.zeros(1, 4, 4)
    edge_value_mat = torch.zeros(1, 4, 4)
    for src, dst, cap in edges:
        adj[0, src, dst] = 1.0
        adj[0, dst, src] = 1.0
        edge_value_mat[0, dst, src] = cap
    return FakeBatch(
        node_type=node_type,
        node_mask=node_mask,
        adj=adj,
        edge_value_mat=edge_value_mat,
        degree=adj.sum(dim=-1),
        spd=torch.ones(1, 4, 4, dtype=torch.long),
        edge_batch=edge_batch,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_value=edge_value,
    )


def test_flow_masks_are_solver_derived_for_simple_maxflow_graph():
    batch = make_flow_batch()
    masks = flow_operator_masks(batch, random_controls=False, seed=0)

    assert masks["source_incidence"][0, 1, 0] == 1
    assert masks["sink_incidence"][0, 3, 1] == 1
    assert masks["min_cut_crossing_edge"][0, 1, 0] == 1
    assert masks["min_cut_crossing_edge"][0, 2, 0] == 1
    assert masks["saturated_edge"][0, 1, 0] == 1
    assert masks["saturated_edge"][0, 3, 2] == 1
    assert masks["shortest_st_path_edge"][0, 1, 0] == 1
    assert masks["shortest_st_path_edge"][0, 3, 1] == 1
    assert masks["capacity_weighted_edge"][0, 1, 0] == 1.0


def test_stratified_random_controls_preserve_positive_weight_mass():
    batch = make_flow_batch()
    valid = valid_pair_mask(batch)
    mask = torch.zeros_like(batch.adj)
    mask[0, 1, 0] = 2.5
    mask[0, 3, 2] = 1.5
    controls = rate_matched_random_masks({"weighted_edge": mask}, batch, valid, seed=7)

    sampled = controls["random__weighted_edge"]
    assert torch.isclose(sampled.sum(), mask.sum())
    assert int((sampled > 0).sum()) == int((mask > 0).sum())


def test_task_transport_responsibility_loader_filters_exact_metric_rows(tmp_path):
    path = tmp_path / "per_head_summary.csv"
    path.write_text(
        "layer,head,metric,intervention,block,centered,mean\n"
        "0,1,output_transport_responsibility,content,all,,0.75\n"
        "0,1,output_transport_responsibility,structure,all,,0.25\n",
        encoding="utf-8",
    )
    loaded = load_task_transport_responsibility(
        path,
        intervention="content",
        block="all",
        centered="",
    )
    assert loaded == {(0, 1): 0.75}


def test_support_fallback_keeps_each_receiver_nonempty():
    dst = torch.tensor([0, 0, 1, 1, 2])
    keep = torch.tensor([False, False, True, False, False])
    fallback = torch.tensor([True, False, False, False, False])

    repaired = ensure_receiver_support(keep, dst, fallback)

    assert repaired.tolist() == [True, False, True, False, True]
    for receiver in torch.unique(dst).tolist():
        assert bool(repaired[dst == receiver].any())


def test_base_rate_rows_document_primary_and_random_masks():
    accumulator = OperatorTransportAccumulator(task_transport_responsibility={})
    valid = torch.ones(1, 2, 2)
    masks = {
        "valid_pair": valid.clone(),
        "operator": torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]),
        "random__operator": torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),
    }

    accumulator.graph_count = 1
    accumulator._add_base_rates(masks, valid)
    rows = {row["operator"]: row for row in accumulator.base_rate_rows()}

    assert rows["operator"]["is_primary_operator"] is True
    assert rows["random__operator"]["is_primary_operator"] is False
    assert rows["random__operator"]["matched_operator"] == "operator"
    assert rows["operator"]["base_rate"] == 0.25


def test_specificity_summary_uses_graph_means_not_layer_rows():
    pd = pytest.importorskip("pandas")
    merged = pd.DataFrame(
        [
            {
                "graph_index": 0,
                "operator_real": "cut",
                "log2_real_over_random_influence_lift": 1.0,
            },
            {
                "graph_index": 0,
                "operator_real": "cut",
                "log2_real_over_random_influence_lift": 3.0,
            },
            {
                "graph_index": 1,
                "operator_real": "cut",
                "log2_real_over_random_influence_lift": -1.0,
            },
        ]
    )

    rows = operator_specificity_summary_rows(merged, ["influence_lift"])

    assert rows[0]["samples"] == 2
    assert rows[0]["graph_layer_rows"] == 3
    assert rows[0]["mean_log2_real_over_random"] == 0.5
    assert rows[0]["row_mean_log2_real_over_random"] == 1.0
    assert rows[0]["positive_graph_fraction"] == 0.5


def test_ranked_ablation_auc_prefers_paired_graph_mean_drop():
    rows = [
        {
            "ranking": "ots",
            "ranking_label": "OTS",
            "ranking_family": "task_weighted_transport",
            "percent_ablated": 0,
            "raw_drop": 100.0,
            "paired_drop_graph_mean": 1.0,
        },
        {
            "ranking": "ots",
            "ranking_label": "OTS",
            "ranking_family": "task_weighted_transport",
            "percent_ablated": 10,
            "raw_drop": 100.0,
            "paired_drop_graph_mean": 3.0,
        },
    ]

    auc = ranked_ablation_auc_rows(rows)

    assert auc[0]["ranking"] == "ots"
    assert auc[0]["drop_metric"] == "paired_drop_graph_mean"
    assert auc[0]["ablation_auc"] == 20.0
    assert auc[0]["normalised_auc"] == 2.0


def test_narrative_claim_row_statuses():
    strong = narrative_claim_row(
        "claim",
        "Claim",
        "source.csv",
        "metric",
        1.0,
        supports=True,
        strong=True,
    )
    directional = narrative_claim_row(
        "claim",
        "Claim",
        "source.csv",
        "metric",
        1.0,
        supports=True,
        strong=False,
    )
    unsupported = narrative_claim_row(
        "claim",
        "Claim",
        "source.csv",
        "metric",
        -1.0,
        supports=False,
        strong=False,
    )

    assert strong["evidence_status"] == "strong_support"
    assert directional["evidence_status"] == "directional_support"
    assert unsupported["evidence_status"] == "not_supported"


def test_parser_defaults_require_cache_and_enable_batch_cache():
    args = build_parser().parse_args(["--checkpoint", "ckpt.pt", "--task", "max_flow"])

    assert args.require_pe_cache is True
    assert args.cache_batches == "auto"
    assert args.allow_tf32 is True
    assert args.autocast_dtype == "none"


def test_batch_cache_none_avoids_collation():
    class FailingRunner:
        def collate_graphs(self, graphs):
            raise AssertionError("collate should not run")

    batches, mode, estimated = collate_analysis_batches(
        FailingRunner(),
        graphs=[object()],
        graph_indices=[3],
        batch_size=1,
        device=torch.device("cpu"),
        cache_mode="none",
        gpu_cache_limit_gb=1.0,
        log=lambda *_args, **_kwargs: None,
    )

    assert batches == []
    assert mode == "none"
    assert estimated == 0


def test_cpu_batch_cache_collates_once_and_estimates_tensor_bytes():
    batches, mode, estimated = collate_analysis_batches(
        TinyRunner(),
        graphs=[object(), object()],
        graph_indices=[5, 7],
        batch_size=2,
        device=torch.device("cpu"),
        cache_mode="cpu",
        gpu_cache_limit_gb=1.0,
        log=lambda *_args, **_kwargs: None,
    )

    assert mode == "cpu"
    assert len(batches) == 1
    assert batches[0][0] == [5, 7]
    assert estimated == batch_tensor_bytes(batches[0][1])
