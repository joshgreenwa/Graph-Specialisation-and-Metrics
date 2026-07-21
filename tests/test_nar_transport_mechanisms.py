from types import SimpleNamespace

import torch
from torch import nn

from graph_specialisation_metrics.synthetic.nar_grit_fixed import Config, make_batch
from graph_specialisation_metrics.synthetic.nar_transport_mechanisms import (
    AnalysisConfig,
    _sample_layer_matched_rankings,
    _seed_fillstyle,
    ablate_head_sets_by_replica,
    _densify_sparse_record,
    address_replica,
    batched_output_jacobian,
    build_checkpoint_manifest,
    build_family_interaction_tables,
    build_replica_bundle,
    cumulative_ablation_ks,
    causal_schedule,
    followup_schedule,
    project_components,
    predict_ablated_head_sets,
    record_permutation_replica,
    retrieval_attention_diagnostics,
    select_causal_families,
    scientific_fingerprint,
    structural_query_record_replica,
    structural_record_record_noop,
    symmetric_output_decomposition,
    verify_replica,
)


def tiny_nar_config() -> Config:
    return Config(
        widths=(32,),
        analysis_width=32,
        heads=4,
        models=("1hop", "dense"),
        ns=(4,),
        mechanistic_ns=(4,),
        seeds=(0,),
        family_size=1,
    )


def test_task_specific_replicas_change_only_declared_fields() -> None:
    batch = make_batch(tiny_nar_config(), 24, 8, seed=103)
    bundle = build_replica_bundle(batch, records=8, donors=4)
    for replica in bundle.replicas:
        verify_replica(batch, replica)
    combined = bundle.combined()
    assert hasattr(combined, "structural_degree_swaps")
    assert combined.structural_degree_swaps.shape == (
        len(batch) * (1 + len(bundle.replicas)), 2
    )

    different = address_replica(batch, 8, 0, same_answer=False)
    same = address_replica(batch, 8, 0, same_answer=True)
    for graph in range(len(batch)):
        if bool(different.valid[graph]):
            assert different.variant_label[graph] != different.clean_label[graph]
            changed = different.batch.x[graph] != batch.x[graph]
            assert int(changed[:, 0].sum()) == 1
            assert int(changed[:, 1].sum()) == 0
        if bool(same.valid[graph]):
            assert same.variant_label[graph] == same.clean_label[graph]
            assert same.variant_target_idx[graph] != same.clean_target_idx[graph]


def test_structural_signal_and_automorphism_controls() -> None:
    batch = make_batch(tiny_nar_config(), 5, 8, seed=211)
    signal = structural_query_record_replica(batch, donor=0)
    noop = structural_record_record_noop(batch, donor=0)
    assert float((signal.batch.rrwp - batch.rrwp).abs().max()) > 0
    assert float((noop.batch.rrwp - batch.rrwp).abs().max()) == 0
    assert torch.equal(signal.batch.x, batch.x)
    assert torch.equal(noop.batch.x, batch.x)


def test_record_permutation_preserves_memory_and_answer() -> None:
    batch = make_batch(tiny_nar_config(), 7, 8, seed=307)
    changed = record_permutation_replica(batch)
    assert torch.equal(changed.batch.y, batch.y)
    for graph in range(len(batch)):
        nodes = torch.where(batch.record_mask[graph])[0]
        clean_rows = sorted(map(tuple, batch.x[graph, nodes].tolist()))
        changed_rows = sorted(map(tuple, changed.batch.x[graph, nodes].tolist()))
        assert clean_rows == changed_rows
        query_key = changed.batch.x[graph, changed.batch.query_idx[graph], 0]
        target_key = changed.batch.x[graph, changed.batch.target_idx[graph], 0]
        assert query_key == target_key


def test_symmetric_decomposition_closes_and_projection_is_consistent() -> None:
    torch.manual_seed(401)
    batch, heads, nodes, dim, targets = 3, 4, 6, 5, 7
    clean_attention = torch.softmax(torch.randn(batch, heads, nodes, nodes), dim=-1)
    variant_attention = torch.softmax(torch.randn(batch, heads, nodes, nodes), dim=-1)
    clean_message = torch.randn(batch, heads, nodes, nodes, dim)
    variant_message = torch.randn(batch, heads, nodes, nodes, dim)
    route, message, total = symmetric_output_decomposition(
        clean_attention, clean_message, variant_attention, variant_message
    )
    assert torch.allclose(route + message, total, atol=1.0e-5)
    phi = torch.randn(targets, batch, nodes, heads, dim)
    projected = project_components(phi, route, message)
    route_bnhd = route.permute(0, 2, 1, 3)
    message_bnhd = message.permute(0, 2, 1, 3)
    expected = torch.linalg.vector_norm(
        torch.einsum("tbnhd,bnhd->tbnh", phi, route_bnhd + message_bnhd),
        dim=0,
    ).sum(dim=1)
    assert torch.allclose(projected["total"], expected, atol=1.0e-5)
    assert torch.all(projected["alignment"] <= 1.0 + 1.0e-5)


def test_record_gate_is_not_mistaken_for_address_selective_routing() -> None:
    clean = torch.zeros(1, 1, 4, 4)
    variant = torch.zeros_like(clean)
    # Receiver 0 allocates a common-scaled mass to record sources 2 and 3.
    # A changed competing source alters both raw record weights but not their
    # relative within-record routing profile.
    clean[0, 0, 0] = torch.tensor([0.5, 0.0, 0.2, 0.3])
    variant[0, 0, 0] = torch.tensor([0.75, 0.0, 0.1, 0.15])
    retrieval_mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    retrieval_mask[0, 0, 2:] = True

    diagnostics = retrieval_attention_diagnostics(
        clean, variant, retrieval_mask
    )
    assert float(diagnostics["raw_moved"]) > 0
    assert float(diagnostics["gate_moved"]) > 0
    assert torch.allclose(
        diagnostics["profile_moved"], torch.zeros(1, 1), atol=1.0e-7
    )

    selective = variant.clone()
    selective[0, 0, 0] = torch.tensor([0.7, 0.0, 0.2, 0.1])
    changed = retrieval_attention_diagnostics(clean, selective, retrieval_mask)
    assert float(changed["profile_moved"]) > 0


def test_sparse_densification_reconstructs_head_output() -> None:
    torch.manual_seed(503)
    graphs, nodes, heads, dim = 2, 3, 2, 4
    graph, source, destination = [], [], []
    for graph_index in range(graphs):
        for dst in range(nodes):
            for src in range(nodes):
                graph.append(graph_index)
                source.append(src)
                destination.append(dst)
    graph_tensor = torch.tensor(graph)
    source_tensor = torch.tensor(source)
    destination_tensor = torch.tensor(destination)
    edge_attention = torch.rand(len(graph), heads)
    edge_message = torch.randn(len(graph), heads, dim)
    output = torch.zeros(graphs, nodes, heads, dim)
    for edge in range(len(graph)):
        output[graph[edge], destination[edge]] += (
            edge_attention[edge, :, None] * edge_message[edge]
        )
    record = SimpleNamespace(
        layer=0,
        graph=graph_tensor,
        local_src=source_tensor,
        local_dst=destination_tensor,
        attention=edge_attention,
        message=edge_message,
        head_output=output.reshape(graphs * nodes, heads, dim),
    )
    dense = _densify_sparse_record(
        record, graph_blocks=1, graphs_per_block=graphs, nodes=nodes
    )
    reconstructed = (
        dense.attention[0].unsqueeze(-1) * dense.message[0]
    ).sum(dim=3).permute(0, 2, 1, 3)
    assert torch.allclose(reconstructed, output)


def test_batched_output_jacobian_matches_linear_readout() -> None:
    torch.manual_seed(601)
    graphs, nodes, heads, dim, targets = 2, 3, 2, 4, 5
    routed = torch.randn(graphs * nodes, heads, dim, requires_grad=True)
    weight = torch.randn(targets, heads, dim)
    logits = torch.einsum(
        "bnhd,thd->bt", routed.reshape(graphs, nodes, heads, dim), weight
    )
    jacobian = batched_output_jacobian(
        logits, routed, clean_graphs=graphs, nodes=nodes, retain_graph=False
    )
    expected = weight[:, None, None].expand(-1, graphs, nodes, -1, -1)
    assert jacobian.shape == expected.shape
    assert torch.allclose(jacobian, expected, atol=1.0e-6)


def test_manifest_selection_uses_validation_not_heldout(tmp_path) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for seed, validation_loss, heldout_accuracy in ((0, 0.2, 1.0), (1, 0.1, 0.2)):
        torch.save(
            {
                "version": "test",
                "fingerprint": f"seed-{seed}",
                "official_grit_commit": "test",
                "model_name": "1hop",
                "width": 64,
                "N": 4,
                "seed": seed,
                "state_dict": {},
                "best_validation": {"loss": validation_loss, "accuracy": 0.5},
                "heldout": {"loss": 1.0, "accuracy": heldout_accuracy},
                "parameters": 123,
            },
            checkpoint_dir / f"1hop_seed_{seed}.pt",
        )
    cfg = AnalysisConfig(
        drive_root=str(tmp_path),
        run_name="run",
        analysis_width=64,
        models=("1hop",),
        ns=(4,),
        seeds=(0, 1),
        anchor_ns=(4,),
    )
    rows = build_checkpoint_manifest(cfg, force=True)
    selected = [row for row in rows if row["selected_for_analysis"]]
    assert len(selected) == 1
    assert selected[0]["seed"] == 1


def test_manifest_skips_missing_requested_seed_cells(tmp_path) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    torch.save(
        {
            "version": "test",
            "fingerprint": "seed-0",
            "official_grit_commit": "test",
            "model_name": "1hop",
            "width": 64,
            "N": 4,
            "seed": 0,
            "state_dict": {},
            "best_validation": {"loss": 0.1, "accuracy": 1.0},
            "heldout": {"loss": 0.1, "accuracy": 1.0},
            "parameters": 123,
        },
        checkpoint_dir / "1hop_seed_0.pt",
    )
    cfg = AnalysisConfig(
        drive_root=str(tmp_path),
        run_name="run",
        analysis_width=64,
        models=("1hop",),
        ns=(4,),
        seeds=(0, 1, 2, 3, 4),
        anchor_ns=(4,),
    )
    rows = build_checkpoint_manifest(cfg, force=True)
    assert [(row["N"], row["seed"]) for row in rows] == [(4, 0)]
    assert rows[0]["selected_for_analysis"]


def test_seed_scope_does_not_invalidate_per_checkpoint_caches() -> None:
    three = AnalysisConfig(seeds=(0, 1, 2))
    five = AnalysisConfig(seeds=(0, 1, 2, 3, 4))
    assert scientific_fingerprint(three) == "d444b9381bd91339"
    assert scientific_fingerprint(five) == scientific_fingerprint(three)
    assert len({_seed_fillstyle(seed) for seed in five.seeds}) == 5


def test_causal_family_controls_match_layer_composition() -> None:
    discovery = {
        "head": {
            "address_different_answer": {
                "route": torch.tensor([
                    [[1.0, 0.1, 0.2, 0.3, 0.2, 0.1], [9.0, 8.0, 0.4, 0.5, 0.3, 0.2]],
                    [[1.1, 0.2, 0.1, 0.4, 0.3, 0.2], [8.5, 7.5, 0.3, 0.6, 0.4, 0.1]],
                ])
            },
            "target_payload": {
                "message": torch.tensor([
                    [[8.0, 7.0, 0.2, 0.1, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2, 0.1, 0.2]],
                    [[7.5, 6.5, 0.1, 0.2, 0.4, 0.3], [0.4, 0.5, 0.2, 0.3, 0.2, 0.1]],
                ])
            },
        },
        "clean_attention": {
            "clean_throughput": torch.ones(2, 2, 6),
        },
    }
    cfg = AnalysisConfig(
        analysis_width=64,
        models=("1hop",),
        ns=(4,),
        seeds=(0,),
        anchor_ns=(4,),
        family_size=2,
        random_families=2,
    )
    families = select_causal_families(
        discovery, cfg, {"model": "1hop", "N": 4, "seed": 0}
    )
    for name in ("routing", "message"):
        target_counts = {}
        for layer, _head in families[name]:
            target_counts[layer] = target_counts.get(layer, 0) + 1
        for control in families[f"{name}_controls"]:
            control_counts = {}
            for layer, _head in control:
                control_counts[layer] = control_counts.get(layer, 0) + 1
            assert control_counts == target_counts
            assert not (set(control) & set(families[name]))


def test_cumulative_controls_are_nested_and_layer_matched() -> None:
    target = [
        (0, 0), (1, 0), (0, 1), (1, 1),
        (0, 2), (1, 2), (0, 3), (1, 3),
    ]
    throughput = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
    controls = _sample_layer_matched_rankings(
        target, throughput, count=3, seed=701
    )
    assert cumulative_ablation_ks(len(target)) == (1, 2, 4, 8)
    for control in controls:
        assert len(control) == len(target)
        assert [layer for layer, _head in control] == [layer for layer, _head in target]
        assert len(set(control)) == len(control)


def test_family_joint_excess_corrects_selected_overlap() -> None:
    rows = []
    values = {"routing": 3.0, "message": 4.0, "overlap": 1.0, "union": 8.0}
    for family, value in values.items():
        rows.append({
            "model": "1hop", "N": 4, "seed": 0,
            "selection": "score", "family": family,
            "loss_delta": value,
        })
    cfg = AnalysisConfig(
        models=("1hop",), ns=(4,), seeds=(0,), anchor_ns=(4,),
        bootstrap_samples=20,
    )
    cells, _summary = build_family_interaction_tables(
        {"family_ablation": rows, "union_patching": []}, cfg
    )
    assert len(cells) == 1
    assert cells[0]["joint_excess"] == 2.0


def test_replica_batched_ablation_applies_distinct_head_sets() -> None:
    class Layer(nn.Module):
        def forward(self, value):
            return value.clone(), None

    model = SimpleNamespace(L=2, attention_layers=nn.ModuleList([Layer(), Layer()]))
    # Two replicas, one graph each, two nodes, three heads and one head channel.
    value = torch.ones(4, 3, 1)
    head_sets = [[(0, 0), (1, 1)], [(0, 2)]]
    with ablate_head_sets_by_replica(
        model, head_sets, graphs=1, nodes=2
    ):
        layer0, _ = model.attention_layers[0](value)
        layer1, _ = model.attention_layers[1](value)
    layer0 = layer0.reshape(2, 1, 2, 3, 1)
    layer1 = layer1.reshape(2, 1, 2, 3, 1)
    assert torch.all(layer0[0, :, :, 0] == 0)
    assert torch.all(layer0[1, :, :, 2] == 0)
    assert torch.all(layer1[0, :, :, 1] == 0)
    assert torch.all(layer1[1] == 1)


def test_followup_schedules_replicate_anchor_seeds() -> None:
    manifest = []
    for records in (4, 8):
        for seed in (0, 1, 2):
            manifest.append({
                "model": "1hop", "N": records, "seed": seed,
                "selected_for_analysis": seed == 0,
            })
    cfg = AnalysisConfig(
        models=("1hop",), ns=(4, 8), seeds=(0, 1, 2), anchor_ns=(4,)
    )
    causal = causal_schedule(manifest, cfg)
    followup = followup_schedule(manifest, cfg)
    assert {(row["N"], row["seed"]) for row in causal} == {(4, 0), (4, 1), (4, 2)}
    assert {(row["N"], row["seed"]) for row in followup} == {
        (4, 0), (4, 1), (4, 2), (8, 0)
    }


def test_batched_ablation_forward_returns_one_block_per_family() -> None:
    class Layer(nn.Module):
        def forward(self, value):
            return value, None

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.L = 2
            self.H = 3
            self.attention_layers = nn.ModuleList([Layer(), Layer()])

        def forward(self, batch):
            graphs, nodes = len(batch), int(batch.x.size(1))
            value = torch.ones(graphs * nodes, self.H, 1, device=batch.x.device)
            for layer in self.attention_layers:
                value, _edge = layer(value)
            score = value.reshape(graphs, nodes, self.H).sum(dim=(1, 2))
            return torch.stack([score, -score], dim=-1)

    batch = make_batch(tiny_nar_config(), 2, 4, seed=809)
    logits = predict_ablated_head_sets(
        Model(),
        batch,
        [[(1, 0)], [(1, 1), (1, 2)]],
        device=torch.device("cpu"),
        max_replica_pairs=10_000,
    )
    assert logits.shape == (2, 2, 2)
    assert torch.all(logits[0, :, 0] > logits[1, :, 0])
