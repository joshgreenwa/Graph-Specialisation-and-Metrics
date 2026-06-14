import torch

from graph_specialisation_metrics.minimal_order_dissociation_experiment import (
    ExperimentSpec,
    build_model,
    encode_values,
    make_split,
    oracle_scalar,
)


def test_minimal_generator_respects_position_value_and_label_constraints():
    spec = ExperimentSpec(n_nodes=32, receptive_radius=4, train_size=16, val_size=8, test_size=8)
    split = make_split(spec, 128, 123)
    assert torch.all(split["p1"] < split["p2"])
    assert torch.all(split["p1"] > spec.receptive_radius)
    assert torch.all(split["p2"] > spec.receptive_radius)
    assert torch.all(split["x1"] != split["x2"])
    assert torch.allclose(split["y_add"], (split["x1"] + split["x2"]).float())
    assert torch.equal(split["y_rel"], (split["x1"] > split["x2"]).long())
    # Exactly two non-null value nodes, with no p1/p2 role flags.
    value_mass = split["x"][:, :, 1:].sum(dim=-1)
    assert torch.all(value_mass.sum(dim=-1) == 2)


def test_add_oracle_double_intervention_is_exactly_additive():
    x1 = torch.tensor([0, 1, 3])
    x2 = torch.tensor([1, 3, 0])
    x1_new = torch.tensor([2, 3, 1])
    x2_new = torch.tensor([3, 0, 2])
    y_base = oracle_scalar("add", x1, x2)
    y_a = oracle_scalar("add", x1_new, x2)
    y_b = oracle_scalar("add", x1, x2_new)
    y_ab = oracle_scalar("add", x1_new, x2_new)
    interaction = (y_ab - y_base) - ((y_a - y_base) + (y_b - y_base))
    assert torch.allclose(interaction, torch.zeros_like(interaction))


def test_rel_oracle_double_intervention_can_be_second_order():
    x1 = torch.tensor([1])
    x2 = torch.tensor([2])
    x1_new = torch.tensor([3])
    x2_new = torch.tensor([0])
    y_base = oracle_scalar("rel", x1, x2)
    y_a = oracle_scalar("rel", x1_new, x2)
    y_b = oracle_scalar("rel", x1, x2_new)
    y_ab = oracle_scalar("rel", x1_new, x2_new)
    interaction = (y_ab - y_base) - ((y_a - y_base) + (y_b - y_base))
    assert interaction.item() == -1.0


def test_plain_mpnn_receiver_is_invariant_to_beyond_radius_value_change():
    spec = ExperimentSpec(n_nodes=12, receptive_radius=2)
    model = build_model("mpnn", "add", spec, hidden_dim=16, gt_layers=1, gt_heads=2).eval()
    p1 = torch.tensor([5])
    p2 = torch.tensor([8])
    x = encode_values(p1, p2, torch.tensor([1]), torch.tensor([3]), spec)
    x_changed = encode_values(p1, p2, torch.tensor([2]), torch.tensor([3]), spec)
    with torch.no_grad():
        out = model(x)
        out_changed = model(x_changed)
    assert torch.allclose(out, out_changed, atol=1.0e-7)


def test_virtual_node_baseline_is_symmetric_without_position_injection():
    spec = ExperimentSpec(n_nodes=16, receptive_radius=2)
    model = build_model("mpnn_vn", "rel", spec, hidden_dim=16, gt_layers=1, gt_heads=2).eval()
    p1 = torch.tensor([5])
    p2 = torch.tensor([10])
    x = encode_values(p1, p2, torch.tensor([1]), torch.tensor([3]), spec)
    x_swapped_values = encode_values(p1, p2, torch.tensor([3]), torch.tensor([1]), spec)
    # Same unordered values at two structurally equivalent far interior nodes.
    with torch.no_grad():
        out = model(x)
        out_swapped = model(x_swapped_values)
    assert torch.allclose(out, out_swapped, atol=1.0e-6)
