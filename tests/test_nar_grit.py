import torch

from graph_specialisation_metrics.synthetic.nar_grit_fixed import (
    Config,
    build_parser,
    config_from_args,
    detect_accuracy_outliers,
    khop_support,
    make_batch,
    semantic_replica,
    supplemental_seed_cells,
    verify_intervention,
)


def tiny_config() -> Config:
    return Config(
        widths=(32,),
        analysis_width=32,
        heads=4,
        models=("1hop", "dense"),
        ns=(4, 8),
        mechanistic_ns=(4, 8),
        seeds=(0,),
        family_size=1,
    )


def test_default_cli_matches_released_nar_training_budget() -> None:
    cfg = config_from_args(build_parser().parse_args([]))
    assert cfg.ns == (4, 8, 16, 32, 64, 80)
    assert cfg.seeds == (0, 1, 2)
    assert cfg.train_graphs_per_epoch == 8000
    assert cfg.batch_size == 64
    assert cfg.eval_every == 125
    assert cfg.epochs == 200
    assert cfg.steps == 25_000
    assert cfg.lr == 1.0e-3
    assert cfg.weight_decay == 0.0
    assert cfg.lr_scheduler == "cosine"
    assert cfg.gradient_clip_norm == 1.0
    assert cfg.validation_graphs == 1000
    assert cfg.heldout_graphs == 1000
    assert cfg.early_stopping_loss_threshold == 0.001
    assert cfg.early_stopping_patience_epochs == 50


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


def test_generator_matches_fixed_n_nar_graph() -> None:
    cfg = tiny_config()
    batch = make_batch(cfg, size=5, records=8, seed=123)
    assert batch.x.shape == (5, 11, 2)
    for graph in range(len(batch)):
        records = torch.where(batch.record_mask[graph])[0]
        assert len(records) == 8
        assert batch.adj[graph].sum(-1)[batch.central_idx[graph]] == 9
        assert batch.adj[graph].sum(-1)[batch.intermediate_idx[graph]] == 2
        assert batch.adj[graph].sum(-1)[batch.query_idx[graph]] == 1
        assert torch.all(batch.adj[graph].sum(-1)[records] == 1)
        assert shortest_distance(
            batch.adj[graph], int(batch.query_idx[graph]), int(batch.central_idx[graph])
        ) == 2
        assert shortest_distance(
            batch.adj[graph], int(batch.target_idx[graph]), int(batch.central_idx[graph])
        ) == 1


def test_every_key_occurs_once_and_query_selects_target() -> None:
    cfg = tiny_config()
    records = 8
    batch = make_batch(cfg, size=7, records=records, seed=456)
    for graph in range(len(batch)):
        memory = batch.x[graph, batch.record_mask[graph]]
        assert sorted(memory[:, 0].tolist()) == list(range(records))
        query_key = batch.x[graph, batch.query_idx[graph], 0]
        target_key = batch.x[graph, batch.target_idx[graph], 0]
        assert query_key == target_key
        assert torch.all(batch.x[graph, batch.central_idx[graph]] == records)
        assert torch.all(batch.x[graph, batch.intermediate_idx[graph]] == records)
        assert batch.x[graph, batch.query_idx[graph], 1] == records


def test_semantic_intervention_changes_only_target_value() -> None:
    cfg = tiny_config()
    batch = make_batch(cfg, 3, 8, 789)
    checks = verify_intervention(batch, 8)
    assert checks["changed_value_entries"] == 3
    assert checks["adj_max"] == 0
    assert checks["rrwp_max"] == 0
    assert checks["key_max"] == 0
    changed = semantic_replica(batch, 8, 0)
    for graph in range(len(batch)):
        difference = (changed.x[graph] - batch.x[graph]).abs().sum(dim=-1)
        assert torch.where(difference > 0)[0].tolist() == [int(batch.target_idx[graph])]


def test_attention_support_changes_reach_not_parameters() -> None:
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


def test_feature_and_label_spaces_are_specific_to_n() -> None:
    cfg = tiny_config()
    four = make_batch(cfg, 2, 4, 11)
    eight = make_batch(cfg, 2, 8, 12)
    assert four.x.shape[-1] == 2
    assert eight.x.shape[-1] == 2
    assert int(four.x.max()) <= 4
    assert int(eight.x.max()) <= 8
    assert int(four.y.max()) < 4
    assert int(eight.y.max()) < 8


def test_outlier_audit_requires_peer_and_validation_agreement_and_retries_once() -> None:
    def payload(seed: int, heldout: float, validation: float, *, retried: bool = False):
        result = {
            "model_name": "dense",
            "width": 128,
            "N": 16,
            "seed": seed,
            "heldout": {"accuracy": heldout},
            "best_validation": {"accuracy": validation},
        }
        if retried:
            result["outlier_retraining"] = {"completed": True}
        return result

    clear = [
        payload(0, 0.91, 0.90),
        payload(1, 0.89, 0.88),
        payload(2, 0.61, 0.60),
    ]
    found = detect_accuracy_outliers(clear, min_gap=0.15, peer_range=0.05)
    assert [(item["seed"], item["direction"]) for item in found] == [(2, "low")]

    # Held-out noise alone is insufficient when the independent validation result disagrees.
    inconsistent = [clear[0], clear[1], payload(2, 0.61, 0.89)]
    assert not detect_accuracy_outliers(inconsistent, min_gap=0.15, peer_range=0.05)

    # A replacement is accepted as final even if it remains separated from its peers.
    already_retried = [clear[0], clear[1], payload(2, 0.61, 0.60, retried=True)]
    assert not detect_accuracy_outliers(already_retried, min_gap=0.15, peer_range=0.05)


def test_additional_seeds_are_new_and_cover_every_performance_cell() -> None:
    cfg = Config()
    cells = supplemental_seed_cells(
        cfg,
        seeds=(3, 4),
    )
    expected = {
        (model, width, records, seed)
        for width in cfg.widths
        for records in cfg.ns
        for model in cfg.models
        for seed in (3, 4)
    }
    assert set(cells) == expected
    assert len(cells) == len(cfg.widths) * len(cfg.ns) * len(cfg.models) * 2

    try:
        supplemental_seed_cells(
            cfg,
            seeds=(2, 3),
        )
    except ValueError as error:
        assert "must be new" in str(error)
    else:
        raise AssertionError("base seed overlap should be rejected")

