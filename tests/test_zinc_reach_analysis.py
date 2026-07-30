from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import graph_specialisation_metrics.zinc_reach_analysis as reach
from graph_specialisation_metrics.zinc_reach_analysis import (
    TASKS,
    ZincReachConfig,
    discover_seed_checkpoint,
    figures,
    graph_bamberger_profiles,
    graph_donor_profiles,
    summarise_dense_profile_contrasts,
    summarise_graph_profiles,
)


def test_donor_direction_jacobian_matches_linear_response(monkeypatch):
    class Data:
        def __init__(self, x, edge_attr=None):
            self.x = x
            self.edge_attr = edge_attr

        def clone(self):
            return Data(self.x.clone(), self.edge_attr)

    class Layers(torch.nn.Module):
        def __init__(self, mixing):
            super().__init__()
            self.register_buffer("mixing", mixing)

        def forward(self, data):
            output = data.clone()
            output.x = self.mixing @ data.x
            return output

    embedding = torch.nn.Embedding(21, 3)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(63, dtype=torch.float32).reshape(21, 3) / 10)
    raw_atoms = torch.tensor([1, 4, 7], dtype=torch.long)
    mixing = torch.tensor(
        [[1.0, 0.5, 0.0], [0.25, 1.0, 0.5], [0.0, 0.75, 1.0]]
    )
    net = SimpleNamespace(layers=Layers(mixing))
    prepared = SimpleNamespace(
        runtime=SimpleNamespace(model=SimpleNamespace(model=net))
    )
    after_encoder = Data(embedding(raw_atoms).detach())
    monkeypatch.setattr(
        reach,
        "_after_feature_encoder",
        lambda _prepared, _graphs: (after_encoder, raw_atoms, embedding),
    )
    variants = [
        Data(torch.tensor([[1], [6], [7]])),
        Data(torch.tensor([[1], [6], [7]])),
    ]
    events = [SimpleNamespace(source=1), SimpleNamespace(source=1)]

    actual, diagnostics = reach._semantic_directional_jacobian_mass(
        prepared,
        base=Data(raw_atoms[:, None]),
        variants=variants,
        events=events,
    )
    donor_direction = embedding.weight[6] - embedding.weight[4]
    expected = (
        mixing[:, 1].abs() * torch.linalg.vector_norm(donor_direction)
    ).repeat(2, 1)
    assert torch.allclose(actual, expected)
    assert all(
        row["method"] in {"exact_forward_ad_jvp", "exact_reverse_ad_jvp"}
        for row in diagnostics
    )


def test_nonfinite_exact_jvps_use_audited_centered_fallback(monkeypatch):
    def nonfinite_forward_jvp(function, primals, _tangents):
        output = function(*primals)
        return output, torch.full_like(output, torch.nan)

    def nonfinite_reverse_jvp(function, primal, _tangent, **_kwargs):
        output = function(primal)
        return output, torch.full_like(output, torch.nan)

    monkeypatch.setattr(torch.func, "jvp", nonfinite_forward_jvp)
    monkeypatch.setattr(torch.autograd.functional, "jvp", nonfinite_reverse_jvp)
    clean = torch.tensor([1.5, -0.5])
    direction = torch.tensor([0.25, 2.0])
    tangent, diagnostic = reach._directional_jvp(
        lambda value: value.square(),
        clean,
        direction,
    )

    assert torch.allclose(tangent, 2 * clean * direction, rtol=1.0e-3, atol=1.0e-3)
    assert diagnostic["method"] == "audited_centered_difference"
    assert diagnostic["relative_error"] < 1.0e-3
    assert diagnostic["failures"] == [
        "forward_ad:FloatingPointError",
        "reverse_ad:FloatingPointError",
    ]


def test_discover_seed_checkpoint_prefers_recovery_best(tmp_path: Path):
    results = tmp_path / "results"
    recovery = results / "_recovery_checkpoints" / "seed0_ColabDrive.2hop.GRITwRRWP"
    recovery.mkdir(parents=True)
    (recovery / "latest.ckpt").write_bytes(b"latest")
    best = recovery / "best.ckpt"
    best.write_bytes(b"best")

    assert discover_seed_checkpoint(results, seed=0) == best


def test_discover_seed_checkpoint_selects_highest_standard_epoch(tmp_path: Path):
    checkpoint_dir = tmp_path / "results" / "run" / "0" / "ckpt"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "3.ckpt").write_bytes(b"three")
    expected = checkpoint_dir / "12.ckpt"
    expected.write_bytes(b"twelve")
    other_seed = tmp_path / "results" / "run" / "1" / "ckpt"
    other_seed.mkdir(parents=True)
    (other_seed / "99.ckpt").write_bytes(b"other")

    assert discover_seed_checkpoint(tmp_path / "results", seed=0) == expected


def _raw_rows():
    donor = []
    bamberger = []
    for task_index, task in enumerate(TASKS):
        for graph in (0, 1):
            for channel in ("semantic", "structural"):
                for distance in (0, 1, 2):
                    row = {
                        "task": task,
                        "graph": graph,
                        "channel": channel,
                        "source": 0,
                        "donor_graph": 9,
                        "donor_node": 1,
                        "draw": 0,
                        "distance": distance,
                        "functional_carriage": (
                            (1 + distance) + 0.2 * task_index
                        ),
                    }
                    if channel == "semantic":
                        row["directional_jacobian"] = (
                            (3 - distance) + 0.1 * task_index
                        )
                        row["finite_hidden_response"] = (
                            2 + 0.5 * distance + 0.15 * task_index
                        )
                    donor.append(row)
            for output_node in (0, 1):
                for input_node, distance in enumerate((0, 1, 2)):
                    bamberger.append(
                        {
                            "task": task,
                            "graph": graph,
                            "output_node": output_node,
                            "input_node": input_node,
                            "distance": distance,
                            "influence": 3 - distance,
                        }
                    )
    return donor, bamberger


def test_profile_scope_and_normalisation():
    donor, bamberger = _raw_rows()
    graph_rows = [
        *graph_donor_profiles(donor, effect_floor=1e-12),
        *graph_bamberger_profiles(bamberger, effect_floor=1e-12),
    ]
    grouped = {}
    for row in graph_rows:
        key = (row["task"], row["graph"], row["channel"], row["method"])
        grouped.setdefault(key, 0.0)
        grouped[key] += row["mass"]
    assert grouped
    assert all(value == pytest.approx(1.0) for value in grouped.values())
    assert not any(
        row["channel"] == "structural" and row["method"] == "bamberger"
        for row in graph_rows
    )
    assert {
        row["method"]
        for row in graph_rows
        if row["channel"] == "semantic"
    } == {
        "bamberger",
        "directional_jacobian",
        "finite_hidden_response",
        "functional_carriage",
    }
    assert {
        row["method"]
        for row in graph_rows
        if row["channel"] == "structural"
    } == {"functional_carriage"}

    profiles, expected = summarise_graph_profiles(
        graph_rows,
        bootstrap_replicates=40,
        bootstrap_seed=7,
    )
    assert profiles and expected
    assert not any(row["method"] == "local_jacobian" for row in graph_rows)
    contrasts = summarise_dense_profile_contrasts(
        graph_rows,
        bootstrap_replicates=40,
        bootstrap_seed=9,
    )
    assert contrasts
    assert all(row["task"] != "zinc" for row in contrasts)
    assert all(row["paired_graphs"] == 2 for row in contrasts)
    assert any(
        abs(row["mean"]) > 0
        for row in contrasts
        if row["method"] == "functional_carriage"
    )


def test_figure_only_builds_png_and_pdf(tmp_path: Path):
    donor, bamberger = _raw_rows()
    results = tmp_path / "results"
    results.mkdir()

    import csv

    for path, rows in (
        (results / "donor_carrier_mass.csv", donor),
        (results / "bamberger_input_output_influence.csv", bamberger),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    result = figures(
        ZincReachConfig(bootstrap_replicates=40),
        output_dir=tmp_path,
    )
    assert set(result["figures"]) == {
        "semantic_decomposition",
        "semantic_functional",
        "semantic_bamberger",
        "structural_functional",
        "expected_distance",
    }
    assert (results / "dense_profile_contrasts.csv").is_file()
    for formats in result["figures"].values():
        assert Path(formats["png"]).is_file()
        assert Path(formats["pdf"]).is_file()
