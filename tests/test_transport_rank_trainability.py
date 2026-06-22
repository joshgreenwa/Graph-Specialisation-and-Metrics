import torch

from graph_specialisation_metrics.transport_rank_trainability import (
    ExperimentConfig,
    TransportRankLayer,
    make_teacher,
    rank_from_singular_values,
    svdvals_relation_stack,
)


def test_teacher_has_requested_transport_rank():
    cfg = ExperimentConfig(
        n_nodes=16,
        heads=2,
        content_dim=4,
        transport_dim=3,
        relation_types=6,
        rho_g_sweep=(1, 2, 3),
        device="cpu",
    )
    teacher = make_teacher(cfg, rho_g=3, seed=0, device=torch.device("cpu"))
    assert teacher.rank99 == 6


def test_uniform_learned_full_gt_starts_near_transport_dim_rank():
    cfg = ExperimentConfig(
        n_nodes=16,
        heads=4,
        content_dim=4,
        transport_dim=2,
        relation_types=8,
        device="cpu",
    )
    model = TransportRankLayer(
        condition="full_learned",
        heads=cfg.heads,
        relation_types=cfg.relation_types,
        content_dim=cfg.content_dim,
        transport_dim=cfg.transport_dim,
        oracle_group_size=cfg.relation_types // cfg.heads,
    )
    with torch.no_grad():
        model.routing_logits.zero_()
    ops = model.relation_operators().detach()
    rank = rank_from_singular_values(svdvals_relation_stack(ops), energy=0.99)
    assert rank <= cfg.transport_dim
