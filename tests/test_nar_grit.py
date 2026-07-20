import torch

from graph_specialisation_metrics.synthetic.nar_grit import (
    Config,
    calibrated_jd,
    khop_support,
    make_batch,
    verify_interventions,
)


def tiny_config() -> Config:
    return Config(
        widths=(32,),
        analysis_width=32,
        heads=4,
        models=("1hop", "dense"),
        train_ns=(4, 8),
        eval_ns=(4, 8),
        mechanistic_ns=(4, 8),
        seeds=(0,),
        family_size=1,
    )


def shortest_distance(adj: torch.Tensor, source: int, target: int) -> int:
    frontier, seen = {int(source)}, {int(source)}
    for distance in range(adj.size(0) + 1):
        if int(target) in frontier:
            return distance
        following = set()
        for node in frontier:
            following.update(torch.where(adj[node] > 0)[0].tolist())
        frontier = following - seen
        seen |= frontier
    raise AssertionError("graph is disconnected")


def test_generator_realises_delayed_query_and_balanced_roles() -> None:
    cfg = tiny_config()
    batch = make_batch(cfg, size=5, records=8, seed=123)
    assert batch.x.shape == (5, cfg.nodes_for_n(8), cfg.feature_dim)
    for graph in range(len(batch)):
        records = torch.where(batch.record_mask[graph])[0]
        assert len(records) == 8
        assert sorted(batch.record_role[graph, records].tolist()) == [0] * 4 + [1] * 4
        assert torch.all(batch.adj[graph].sum(-1)[records] == 3)
        assert shortest_distance(
            batch.adj[graph], int(batch.query_idx[graph]), int(batch.central_idx[graph])
        ) == 2
        assert shortest_distance(
            batch.adj[graph], int(batch.target_idx[graph]), int(batch.central_idx[graph])
        ) == 1


def test_structural_role_has_controlled_rrwp_signature() -> None:
    cfg = tiny_config()
    batch = make_batch(cfg, size=8, records=8, seed=456)
    for graph in range(len(batch)):
        records = torch.where(batch.record_mask[graph])[0]
        returns = batch.rrwp[graph, records, records, 3]
        roles = batch.record_role[graph, records]
        assert torch.all(returns[roles == 0] > 0)
        assert torch.all(returns[roles == 1] == 0)


def test_interventions_isolate_content_and_structure() -> None:
    cfg = tiny_config()
    checks = verify_interventions(make_batch(cfg, 3, 8, 789), cfg)
    assert checks["semantic_changed_values"] == 6
    assert checks["semantic_adj_max"] == 0
    assert checks["structural_x_max"] == 0
    assert checks["structural_frozen_adj_max"] == 0


def test_attention_support_is_trained_radius_not_parameter_change() -> None:
    cfg = tiny_config()
    batch = make_batch(cfg, 2, 8, 901)
    one = khop_support(batch.adj, 1)
    two = khop_support(batch.adj, 2)
    dense = khop_support(batch.adj, None)
    assert torch.all(one <= two)
    assert torch.all(two <= dense)
    assert one.sum() < two.sum() < dense.sum()
    for graph in range(len(batch)):
        query, centre = int(batch.query_idx[graph]), int(batch.central_idx[graph])
        assert not one[graph, query, centre]
        assert two[graph, query, centre]


def test_jd_separates_influence_from_preference() -> None:
    semantic = torch.tensor([2.0, 1.0])
    structural = torch.tensor([1.0, 2.0])
    _, _, joint, selectivity = calibrated_jd(semantic, structural)
    assert torch.allclose(joint, torch.ones_like(joint))
    assert selectivity[0] > 0
    assert selectivity[1] < 0
