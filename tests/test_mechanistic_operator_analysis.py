from dataclasses import dataclass

import torch

from graph_specialisation_metrics.mechanistic_operator_analysis import (
    flow_operator_masks,
    load_task_transport_responsibility,
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
