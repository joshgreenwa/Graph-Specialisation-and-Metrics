import csv
import json
import math

import numpy as np
import pytest
import torch

from graph_specialisation_metrics.methodology.carriage import (
    event_normalise_functional,
)
from graph_specialisation_metrics.synthetic.query_routing_carriage import (
    DATA_SHARD_GRAPHS,
    ExperimentConfig,
    GraphGPSQueryRouter,
    _directional_jacobian_fields,
    _donor_pool,
    analytic_linear_contributions,
    build_graph_events,
    collate_graphs,
    ensure_data,
    generate_graph,
    run,
    teacher_contributions,
)


def test_event_normalisation_preserves_eligible_mass_and_marks_null_events():
    field = np.asarray([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0]])

    normalised, eligible, denominator = event_normalise_functional(
        field,
        effect_floor=1.0e-8,
    )

    np.testing.assert_allclose(denominator, [4.0, 0.0])
    np.testing.assert_array_equal(eligible, [True, False])
    np.testing.assert_allclose(normalised[0], [0.25, 0.5, 0.25])
    assert np.isnan(normalised[1]).all()
    with pytest.raises(ValueError, match="non-negative"):
        event_normalise_functional([[1.0, -1.0]], effect_floor=1.0e-8)


@pytest.mark.parametrize("ood", [False, True])
def test_generated_query_graph_has_exact_teacher_and_registered_range(ood):
    config = ExperimentConfig.fast_dev()
    graph = generate_graph(
        config,
        graph_id=7,
        rng=np.random.default_rng(20260730),
        ood=ood,
    )

    assert graph.spd.shape == (graph.num_nodes, graph.num_nodes)
    assert torch.equal(graph.spd, graph.spd.T)
    assert int(graph.spd.max()) <= graph.num_nodes - 1
    assert int((graph.role == 1).sum()) == 1
    assert int((graph.role == 2).sum()) == config.num_keys
    record_keys = torch.argmax(graph.x[graph.record_nodes], dim=-1)
    assert sorted(record_keys.tolist()) == list(range(config.num_keys))
    required = config.ood_required_distance if ood else config.id_required_distance
    record_distances = graph.spd[graph.query_node, graph.record_nodes]
    assert int(record_distances.max()) >= required
    torch.testing.assert_close(graph.y_node, teacher_contributions(graph))
    torch.testing.assert_close(graph.y, graph.y_node.sum().reshape(1))


def test_semantic_donor_changes_only_query_key_with_registered_dose(tmp_path):
    config = ExperimentConfig.fast_dev()
    ensure_data(tmp_path, config, progress=False)
    graph = torch.load(
        tmp_path / "data" / "id.pt",
        map_location="cpu",
        weights_only=False,
    )
    # Exercise the public cached loader indirectly through the canonical donor-pool path.
    from graph_specialisation_metrics.synthetic.query_routing_carriage import load_split

    base = load_split(tmp_path, config, "id")[0]
    variants, events = build_graph_events(
        base,
        split="id",
        config=config,
        donor_pool=_donor_pool(tmp_path, config),
    )

    assert graph["fingerprint"] == config.fingerprint
    assert len(variants) == len(events) == config.donors_per_source
    for variant, event in zip(variants, events):
        changed_rows = torch.nonzero(
            torch.any(variant.x != base.x, dim=-1),
            as_tuple=False,
        ).flatten()
        assert changed_rows.tolist() == [base.query_node]
        assert event.dose == pytest.approx(math.sqrt(2.0), abs=1.0e-6)
        for field in (
            "role",
            "value",
            "pe",
            "edge_index",
            "spd",
            "record_nodes",
            "y_node",
            "y",
        ):
            torch.testing.assert_close(getattr(variant, field), getattr(base, field))


def test_linear_control_is_exact_for_donor_matched_direction():
    config = ExperimentConfig.fast_dev()
    graph = generate_graph(
        config,
        graph_id=11,
        rng=np.random.default_rng(91),
    )
    clean = graph.x[graph.query_node].to(torch.float64).clone().requires_grad_(True)
    donor_key = (int(torch.argmax(clean)) + 1) % config.num_keys
    donor = torch.nn.functional.one_hot(
        torch.tensor(donor_key),
        num_classes=config.num_keys,
    ).to(torch.float64)

    clean_contribution = analytic_linear_contributions(graph, clean)
    donor_contribution = analytic_linear_contributions(graph, donor)
    oracle = torch.abs(clean_contribution - donor_contribution)
    jacobian = torch.autograd.functional.jacobian(
        lambda query: analytic_linear_contributions(graph, query),
        clean,
    )
    directional = torch.abs(jacobian @ (clean.detach() - donor))

    torch.testing.assert_close(directional, oracle, rtol=1.0e-12, atol=1.0e-12)


def test_forward_jvp_carrier_jacobian_agrees_with_reverse_mode():
    config = ExperimentConfig.fast_dev()
    graph = generate_graph(
        config,
        graph_id=12,
        rng=np.random.default_rng(92),
    )
    batch = collate_graphs([graph])
    model = GraphGPSQueryRouter(config, init_seed=3).eval()
    clean = graph.x[graph.query_node]
    donors = torch.eye(config.num_keys, dtype=clean.dtype)[:2]

    _, _, observed = _directional_jacobian_fields(
        model,
        batch,
        clean,
        donors,
        beta_multiplier=1.0,
        nodes=graph.num_nodes,
    )

    def carrier_map(query):
        return model(
            batch,
            beta_multiplier=1.0,
            query_override=query.unsqueeze(0),
        ).node_contributions[0, : graph.num_nodes]

    expected = torch.autograd.functional.jacobian(
        carrier_map,
        clean.clone().requires_grad_(True),
        vectorize=True,
    )
    torch.testing.assert_close(observed, expected, rtol=2.0e-5, atol=2.0e-6)


def test_fast_dev_run_is_drive_style_cache_complete_and_figures_are_model_free(
    tmp_path,
):
    config = ExperimentConfig.fast_dev()
    first = run(
        config,
        output_dir=tmp_path,
        phase="all",
        device="cpu",
        progress=False,
    )

    assert first["fingerprint"] == config.fingerprint
    for split in ("train", "validation", "id", "ood", "donor"):
        assert (tmp_path / "data" / f"{split}.pt").is_file()
    train_shards = list(
        (tmp_path / "data" / "shards" / "train").glob("*.pt")
    )
    assert len(train_shards) == math.ceil(
        config.train_graphs / DATA_SHARD_GRAPHS
    )
    (tmp_path / "data" / "train.pt").unlink()
    run(
        config,
        output_dir=tmp_path,
        phase="data",
        device="cpu",
        progress=False,
    )
    assert (tmp_path / "data" / "train.pt").is_file()
    assert (tmp_path / "cache" / "measurements" / "linear_control.pt").is_file()
    assert (
        tmp_path / "cache" / "measurements" / "seed_000_id.pt"
    ).is_file()
    for table in (
        "event_metrics.csv",
        "linear_event_metrics.csv",
        "model_health.csv",
        "distance_profiles.csv",
        "beneficial_profiles.csv",
        "primary_contrasts.csv",
        "dose_ladder_events.csv",
        "dose_ladder_summary.csv",
        "summary.json",
    ):
        assert (tmp_path / "results" / table).is_file()
    with (tmp_path / "results" / "event_metrics.csv").open(newline="") as handle:
        event_row = next(csv.DictReader(handle))
    assert event_row["experiment_fingerprint"] == config.fingerprint
    assert len(event_row["checkpoint_sha256"]) == 64
    assert float(event_row["alpha"]) == pytest.approx(1.0)
    assert event_row["confidence_stratum"] in {"low", "moderate", "high"}
    assert "counterfactual_mae" in event_row
    for name in (
        "query_routing_carriage",
        "query_routing_carriage_id_ood",
        "query_routing_carriage_raw_profiles",
        "query_routing_carriage_seed_profiles",
        "query_routing_carriage_entrywise_jacobian",
        "query_routing_carriage_dose_ladder",
        "query_routing_carriage_beneficial",
    ):
        assert (tmp_path / "figures" / f"{name}.png").is_file()
        assert (tmp_path / "figures" / f"{name}.pdf").is_file()

    id_shards = list(
        (
            tmp_path
            / "cache"
            / "measurement_shards"
            / "seed_000_id"
        ).glob("*.pt")
    )
    assert len(id_shards) == config.id_graphs
    id_measurement = (
        tmp_path / "cache" / "measurements" / "seed_000_id.pt"
    )
    id_measurement.unlink()
    run(
        config,
        output_dir=tmp_path,
        phase="measure",
        device="cpu",
        progress=False,
    )
    assert id_measurement.is_file()

    progress_path = tmp_path / "checkpoints" / "seed_000.progress.pt"
    assert progress_path.is_file()
    (tmp_path / "checkpoints" / "seed_000.pt").unlink()
    resumed = run(
        config,
        output_dir=tmp_path,
        phase="train",
        device="cpu",
        progress=False,
    )
    assert len(resumed["checkpoints"]) == 1
    assert (tmp_path / "checkpoints" / "seed_000.pt").is_file()

    for checkpoint in (tmp_path / "checkpoints").glob("*.pt"):
        if not checkpoint.name.endswith(".progress.pt"):
            checkpoint.unlink()
    assert not (tmp_path / "checkpoints" / "seed_000.pt").exists()
    second = run(
        config,
        output_dir=tmp_path,
        phase="figures",
        device="cpu",
        progress=False,
    )
    assert set(second["figures"]) == {
        "main",
        "id_ood_profiles",
        "raw_profiles",
        "seed_profiles",
        "entrywise_jacobian",
        "dose_ladder",
        "beneficial",
    }
    metadata = json.loads(
        (
            tmp_path
            / "figures"
            / "query_routing_carriage.metadata.json"
        ).read_text()
    )
    assert metadata["rendered_from_cache_only"] is True
    assert set(metadata["health_gate_by_split"]) == {"id", "ood"}
