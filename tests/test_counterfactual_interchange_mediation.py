import torch

from graph_specialisation_metrics.core_interpretability_specialisation_metrics import (
    make_content_swapped_batch,
)
from graph_specialisation_metrics.counterfactual_interchange_mediation import (
    collate_records,
    load_config,
    make_graph_record,
    pathway_deltas,
    source_for_intervention,
)


def tiny_cfg(task: str):
    cfg = load_config(None, task=task, fast_dev_run=True)
    cfg["graph_generator"]["n_train_min"] = 8
    cfg["graph_generator"]["n_train_max"] = 8
    cfg["graph_generator"]["chord_edges_per_node"] = 0.5
    cfg["graph_generator"]["planted_rings"]["probability"] = 0.0
    cfg["graph_generator"]["unique_hub"]["probability"] = 0.0
    cfg["teachers"]["voronoi"]["n_anchors"] = 3
    return cfg


def test_ppr_teacher_decomposition_and_payload_pathway():
    cfg = tiny_cfg("ppr_diffusion")
    graph = make_graph_record("ppr_diffusion", 8, 123, cfg)

    k = graph["teacher"]["K"]
    m = graph["teacher"]["M"]
    y = graph["teacher"]["Y"]
    assert torch.allclose(k.sum(dim=1), torch.ones(8), atol=1.0e-6)
    assert torch.allclose(y, k @ m, atol=1.0e-6)

    source = source_for_intervention(graph, "ppr_payload_swap", 0, 1, cfg)
    deltas = pathway_deltas(graph, source)
    assert torch.allclose(source["teacher"]["K"], graph["teacher"]["K"])
    assert torch.linalg.vector_norm(deltas["K"]) == 0
    assert torch.allclose(deltas["M"], deltas["total"], atol=1.0e-6)


def test_voronoi_teacher_kernel_and_marker_pathway():
    cfg = tiny_cfg("nearest_anchor_voronoi")
    graph = make_graph_record("nearest_anchor_voronoi", 10, 456, cfg)
    anchors = torch.nonzero(graph["anchor_indicator"] > 0.5, as_tuple=False).reshape(-1)
    non_anchor = int(torch.nonzero(graph["anchor_indicator"] < 0.5, as_tuple=False)[0])
    anchor = int(anchors[0])

    k = graph["teacher"]["K"]
    assert int(anchors.numel()) == 3
    assert torch.allclose(k.sum(dim=1), torch.ones(graph["n"]))
    assert torch.all((k == 0) | (k == 1))

    source = source_for_intervention(graph, "voronoi_anchor_marker_swap", anchor, non_anchor, cfg)
    deltas = pathway_deltas(graph, source)
    assert torch.allclose(source["teacher"]["M"], graph["teacher"]["M"])
    assert torch.linalg.vector_norm(deltas["M"]) == 0
    assert torch.allclose(deltas["K"], deltas["total"], atol=1.0e-6)


def test_content_swap_moves_continuous_x_payload_fields():
    cfg = tiny_cfg("ppr_diffusion")
    records = [make_graph_record("ppr_diffusion", 8, 100 + idx, cfg) for idx in range(2)]
    batch = collate_records(records)
    perm = torch.arange(batch.max_nodes).repeat(batch.num_graphs, 1)
    perm[:, 0] = 1
    perm[:, 1] = 0

    swapped = make_content_swapped_batch(batch, perm)
    assert torch.allclose(swapped.x[:, 0], batch.x[:, 1])
    assert torch.allclose(swapped.payload[:, 1], batch.payload[:, 0])
    assert torch.allclose(swapped.adj, batch.adj)
