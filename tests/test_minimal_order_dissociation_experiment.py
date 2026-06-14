import torch

from graph_specialisation_metrics.minimal_order_dissociation_experiment import (
    ExperimentSpec,
    StructuralTransportReadout,
    build_model,
    encode_values,
    make_split,
    oracle_scalar,
)


def test_minimal_v2_generator_respects_continuous_label_constraints():
    spec = ExperimentSpec(n_nodes=32, receptive_radius=4, train_size=16, val_size=8, test_size=8)
    split = make_split(spec, 128, 123)
    assert torch.all(split["p1"] < split["p2"])
    assert torch.all(split["p1"] > spec.receptive_radius)
    assert torch.all(split["p2"] > spec.receptive_radius)
    assert torch.all((split["x1"] >= 0.0) & (split["x1"] <= 1.0))
    assert torch.all((split["x2"] >= 0.0) & (split["x2"] <= 1.0))
    assert torch.allclose(split["y_add"], split["x1"] + split["x2"])
    assert torch.allclose(split["y_prod"], split["x1"] * split["x2"])
    assert torch.allclose(split["y_gap"], torch.clamp(split["x1"] - split["x2"], min=0.0))
    # Exactly two value-bearing nodes; there is no p1/p2 role feature.
    assert torch.all(split["x"][:, :, 0].sum(dim=-1) == 2)


def test_add_oracle_double_resample_is_exactly_additive():
    x1 = torch.tensor([0.1, 0.7, 0.4])
    x2 = torch.tensor([0.6, 0.2, 0.9])
    x1_new = torch.tensor([0.3, 0.1, 0.8])
    x2_new = torch.tensor([0.9, 0.5, 0.2])
    y_base = oracle_scalar("add", x1, x2)
    y_a = oracle_scalar("add", x1_new, x2)
    y_b = oracle_scalar("add", x1, x2_new)
    y_ab = oracle_scalar("add", x1_new, x2_new)
    interaction = (y_ab - y_base) - ((y_a - y_base) + (y_b - y_base))
    assert torch.allclose(interaction, torch.zeros_like(interaction), atol=1.0e-6)


def test_prod_and_gap_oracles_have_nonzero_second_order_interactions():
    x1 = torch.tensor([0.2])
    x2 = torch.tensor([0.8])
    x1_new = torch.tensor([0.9])
    x2_new = torch.tensor([0.1])
    for task in ["prod", "gap"]:
        y_base = oracle_scalar(task, x1, x2)
        y_a = oracle_scalar(task, x1_new, x2)
        y_b = oracle_scalar(task, x1, x2_new)
        y_ab = oracle_scalar(task, x1_new, x2_new)
        interaction = (y_ab - y_base) - ((y_a - y_base) + (y_b - y_base))
        assert torch.abs(interaction).item() > 1.0e-5


def test_plain_mpnn_receiver_is_invariant_to_beyond_radius_value_change():
    spec = ExperimentSpec(n_nodes=12, receptive_radius=2)
    model = build_model("mpnn", spec, hidden_dim=16, gt_heads=2).eval()
    p1 = torch.tensor([5])
    p2 = torch.tensor([8])
    x = encode_values(p1, p2, torch.tensor([0.1]), torch.tensor([0.8]), spec)
    x_changed = encode_values(p1, p2, torch.tensor([0.9]), torch.tensor([0.8]), spec)
    with torch.no_grad():
        out = model(x)
        out_changed = model(x_changed)
    assert torch.allclose(out, out_changed, atol=1.0e-7)


def test_symmetric_global_baseline_is_invariant_to_value_order():
    spec = ExperimentSpec(n_nodes=16, receptive_radius=2)
    model = build_model("mpnn_vn", spec, hidden_dim=16, gt_heads=2).eval()
    p1 = torch.tensor([5])
    p2 = torch.tensor([10])
    x = encode_values(p1, p2, torch.tensor([0.2]), torch.tensor([0.7]), spec)
    x_swapped_values = encode_values(p1, p2, torch.tensor([0.7]), torch.tensor([0.2]), spec)
    with torch.no_grad():
        out = model(x)
        out_swapped = model(x_swapped_values)
    assert torch.allclose(out, out_swapped, atol=1.0e-6)


def test_gt_full_has_explicit_value_structural_channel_and_clamps_it():
    spec = ExperimentSpec(n_nodes=16, receptive_radius=2)
    model = build_model("gt_full", spec, hidden_dim=16, gt_heads=2).eval()
    assert isinstance(model, StructuralTransportReadout)
    p1 = torch.tensor([5])
    p2 = torch.tensor([10])
    x = encode_values(p1, p2, torch.tensor([0.2]), torch.tensor([0.7]), spec)
    x_source = encode_values(p1, p2, torch.tensor([0.9]), torch.tensor([0.1]), spec)
    vs_clean = model.value_struct_channel(x)
    vs_source = model.value_struct_channel(x_source)
    assert torch.linalg.vector_norm(vs_source - vs_clean).item() > 1.0e-6
    with torch.no_grad():
        out_source = model(x_source)
        out_clamped = model(x_source, clamp_vs_to=vs_clean)
    assert out_source.shape == out_clamped.shape == torch.Size([1])
