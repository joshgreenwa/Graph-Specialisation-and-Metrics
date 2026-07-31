from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import graph_specialisation_metrics.zinc_reach_analysis as reach
from graph_specialisation_metrics.zinc_reach_analysis import (
    QM9_PROFILE,
    QM9_TASKS,
    TASKS,
    ZincReachConfig,
    discover_seed_checkpoint,
    figures,
    graph_bamberger_profiles,
    graph_donor_profiles,
    graph_interpolation_profiles,
    graph_output_coherence_profiles,
    summarise_dense_profile_contrasts,
    summarise_graph_profiles,
    summarise_interpolation_contrasts,
    summarise_output_coherence,
    summarise_scale_dependence,
)


def test_semantic_interpolation_scales_linear_functional_mass(monkeypatch):
    class Data:
        def __init__(self, x):
            self.x = x
            self.edge_attr = None
            self.num_nodes = 3

    embedding = torch.nn.Embedding(21, 3)
    with torch.no_grad():
        embedding.weight.copy_(
            torch.arange(63, dtype=torch.float32).reshape(21, 3) / 10
        )
    raw_atoms = torch.tensor([1, 4, 7], dtype=torch.long)
    mixing = torch.tensor(
        [[1.0, 0.5, 0.0], [0.25, 1.0, 0.5], [0.0, 0.75, 1.0]]
    )
    net = SimpleNamespace()

    class Backend:
        def capture(self, data_list, *, require_grad):
            assert not require_grad
            embedded = embedding(raw_atoms.repeat(len(data_list)))
            blocks = embedded.reshape(len(data_list), 3, 3)
            final = torch.einsum("ij,bjw->biw", mixing, blocks)
            return SimpleNamespace(final_state=final)

    prepared = SimpleNamespace(
        runtime=SimpleNamespace(
            model=SimpleNamespace(model=net)
        ),
        backend=Backend(),
    )
    monkeypatch.setattr(
        reach,
        "_atom_embedding",
        lambda _net, *, vocab_size: embedding,
    )
    base = Data(raw_atoms[:, None])
    variant = Data(torch.tensor([[1], [6], [7]]))
    event = SimpleNamespace(source=1)
    clean_final = mixing @ embedding(raw_atoms)
    full_final = clean_final + torch.outer(
        mixing[:, 1],
        embedding.weight[6] - embedding.weight[4],
    )
    gradient = torch.ones(1, 3, 3)
    full_mass = reach._project_final_change(
        (clean_final - full_final).unsqueeze(0),
        gradient,
    )

    actual = reach._semantic_interpolation_mass(
        ZincReachConfig(
            interpolation_doses=(0.25, 1.0),
            interpolation_batch_size=4,
        ),
        prepared,
        base=base,
        variants=[variant],
        events=[event],
        clean_final=clean_final,
        clean_gradient=gradient,
        full_mass=full_mass,
    )
    assert torch.allclose(actual[0], 0.25 * full_mass, atol=1.0e-6)
    assert torch.allclose(actual[1], full_mass)


def test_signed_output_path_carriage_is_complete_and_preserves_cancellation():
    clean = torch.tensor([[2.0], [1.0]])
    intervened = torch.tensor([[1.0], [2.0]])
    capture = SimpleNamespace(
        final_state=torch.stack((clean, intervened)),
        z=torch.tensor([[3.0], [3.0]]),
        target=torch.zeros(2, 1),
    )

    class Backend:
        def capture_groups(self, groups):
            assert len(groups) == 1 and len(groups[0]) == 2
            return [capture]

        def output_from_pooled(self, target):
            assert tuple(target.shape) == (1, 1)
            return lambda pooled: pooled[:, 0]

        def carriage_weights(self, _base, states):
            return torch.ones(states.shape[-2])

    prepared = SimpleNamespace(backend=Backend())
    base = SimpleNamespace(
        num_nodes=2,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    event = SimpleNamespace(
        source=0,
        donor_graph_id=9,
        donor_node=1,
        draw=0,
    )
    rows = reach._signed_output_carriage_rows(
        ZincReachConfig(tasks=("zinc",)),
        prepared,
        task="zinc",
        graph_id=0,
        base=base,
        variants=[SimpleNamespace()],
        events=[event],
    )

    signed = [row["signed_output_carriage"] for row in rows]
    assert signed == pytest.approx([1.0, -1.0])
    assert sum(signed) == pytest.approx(0.0)
    assert all(abs(row["completeness_residual"]) < 1.0e-7 for row in rows)


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


def _raw_rows(tasks=TASKS):
    donor = []
    bamberger = []
    interpolation = []
    for task_index, task in enumerate(tasks):
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
                    donor.append(row)
            for dose in (0.01, 0.02, 0.10, 0.50, 1.00):
                for distance in (0, 1, 2):
                    interpolation.append(
                        {
                            "task": task,
                            "graph": graph,
                            "source": 0,
                            "donor_graph": 9,
                            "donor_node": 1,
                            "draw": 0,
                            "interpolation_dose": dose,
                            "distance": distance,
                            "functional_carriage": (
                                (1 - dose) * (3 - distance)
                                + dose * (1 + distance)
                                + 0.2 * task_index
                            ),
                        }
                    )
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
    return donor, bamberger, interpolation


def _raw_graph_records(tasks=TASKS):
    return [
        {
            "task": task,
            "model_label": reach.TASK_LABELS[task],
            "graph": graph,
            "num_nodes": 3 + graph,
            "diameter": 2,
            "graph_mae": 0.10 + 0.01 * task_index + 0.02 * graph,
        }
        for task_index, task in enumerate(tasks)
        for graph in (0, 1)
    ]


def _raw_full_test_records(tasks=TASKS):
    return [
        {
            "task": task,
            "model_label": reach.TASK_LABELS[task],
            "test_graph": graph,
            "num_nodes": 8 + graph // 3,
            "diameter": 3 + graph // 6,
            "target": 0.5 + 0.01 * graph,
            "prediction": 0.5 + 0.01 * graph + 0.005 * (task_index + 1),
            "graph_mae": 0.02 + 0.003 * task_index + 0.001 * graph,
        }
        for task_index, task in enumerate(tasks)
        for graph in range(24)
    ]


def _raw_output_carriage(tasks=TASKS):
    values = ((0, 0, 1.0), (1, 1, 2.0), (2, 1, -1.0), (3, 2, 0.5))
    return [
        {
            "task": task,
            "model_label": reach.TASK_LABELS[task],
            "graph": graph,
            "channel": "semantic",
            "source": 0,
            "donor_graph": 9,
            "donor_node": 1,
            "draw": 0,
            "carrier": carrier,
            "distance": distance,
            "signed_output_carriage": signed,
        }
        for task in tasks
        for graph in (0, 1)
        for carrier, distance, signed in values
    ]


def test_output_coherence_reveals_within_shell_cancellation():
    graph_rows = graph_output_coherence_profiles(
        _raw_output_carriage(("zinc",)),
        effect_floor=1.0e-12,
    )
    distance_one = next(
        row
        for row in graph_rows
        if row["graph"] == 0 and row["distance"] == 1
    )
    assert distance_one["apparent_mass"] == pytest.approx(3.0 / 4.5)
    assert distance_one["coherent_mass"] == pytest.approx(1.0 / 2.5)
    assert distance_one["coherence_ratio"] == pytest.approx(1.0 / 3.0)

    profiles, expected = summarise_output_coherence(
        graph_rows,
        bootstrap_replicates=40,
        bootstrap_seed=12,
    )
    assert profiles
    change = next(
        row for row in expected if row["metric"] == "expected_distance_change"
    )
    assert change["mean"] == pytest.approx(0.8 - 4.0 / 4.5)


def test_adaptive_scale_bins_use_available_integer_resolution():
    values = {graph: 8 + graph % 12 for graph in range(120)}

    graph_bins, labels, _means, counts = reach._adaptive_scale_bins(values)

    assert len(set(graph_bins.values())) == 11
    assert len(labels) == 11
    assert min(counts.values()) == 10
    assert max(counts.values()) == 20
    assert all(
        graph_bins[left] == graph_bins[right]
        for left in values
        for right in values
        if values[left] == values[right]
    )


def test_profile_scope_and_normalisation():
    donor, bamberger, interpolation = _raw_rows()
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
    interpolation_graph = graph_interpolation_profiles(
        interpolation,
        effect_floor=1e-12,
    )
    grouped_interpolation = {}
    for row in interpolation_graph:
        key = (row["task"], row["graph"], row["interpolation_dose"])
        grouped_interpolation.setdefault(key, 0.0)
        grouped_interpolation[key] += row["mass"]
    assert all(
        value == pytest.approx(1.0)
        for value in grouped_interpolation.values()
    )
    graph_contrasts, sweep = summarise_interpolation_contrasts(
        interpolation_graph,
        [
            row
            for row in graph_rows
            if row["method"] == "bamberger"
        ],
        bootstrap_replicates=40,
        bootstrap_seed=11,
    )
    assert graph_contrasts and sweep
    assert {
        row["metric"] for row in sweep
    } == {"profile_tv", "expected_distance_difference"}
    assert {row["baseline"] for row in sweep} == {
        "bamberger",
        "matched_small_dose",
    }
    matched_origin = [
        row
        for row in graph_contrasts
        if row["baseline"] == "matched_small_dose"
        and row["interpolation_dose"] == pytest.approx(0.01)
    ]
    assert matched_origin
    assert all(row["profile_tv"] == pytest.approx(0.0) for row in matched_origin)

    scale_rows, scale_summary, scale_trends = summarise_scale_dependence(
        graph_rows,
        _raw_graph_records(),
        _raw_full_test_records(),
        tasks=TASKS,
        reference_task="zinc",
        bootstrap_replicates=40,
        bootstrap_seed=13,
    )
    assert scale_rows and scale_summary and scale_trends
    assert {row["descriptor"] for row in scale_summary} == {
        "num_nodes",
        "diameter",
    }
    assert all(
        row["mae_difference_from_reference"] == pytest.approx(0.0)
        for row in scale_rows
        if row["analysis"] == "performance"
        if row["task"] == "zinc"
    )
    assert {row["analysis"] for row in scale_trends} == {
        "performance",
        "reach",
    }


def test_figure_only_builds_png_and_pdf(tmp_path: Path):
    donor, bamberger, interpolation = _raw_rows()
    results = tmp_path / "results"
    results.mkdir()

    import csv

    for path, rows in (
        (results / "donor_carrier_mass.csv", donor),
        (results / "bamberger_input_output_influence.csv", bamberger),
        (results / "semantic_interpolation_mass.csv", interpolation),
        (results / "graph_metrics.csv", _raw_graph_records()),
        (results / "full_test_metrics.csv", _raw_full_test_records()),
        (results / "semantic_output_carriage.csv", _raw_output_carriage()),
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
        "interpolation_sweep",
        "semantic_functional",
        "semantic_bamberger",
        "structural_functional",
        "expected_distance",
        "output_coherence",
        "scale_dependence",
        "scale_slopes",
    }
    assert (results / "dense_profile_contrasts.csv").is_file()
    assert (results / "interpolation_sweep_summary.csv").is_file()
    for formats in result["figures"].values():
        assert Path(formats["png"]).is_file()
        assert Path(formats["pdf"]).is_file()


def test_qm9_profile_builds_dataset_specific_figures(tmp_path: Path):
    donor, bamberger, interpolation = _raw_rows(QM9_TASKS)
    results = tmp_path / "results"
    results.mkdir()

    import csv

    for path, rows in (
        (results / "donor_carrier_mass.csv", donor),
        (results / "bamberger_input_output_influence.csv", bamberger),
        (results / "semantic_interpolation_mass.csv", interpolation),
        (results / "graph_metrics.csv", _raw_graph_records(QM9_TASKS)),
        (
            results / "full_test_metrics.csv",
            _raw_full_test_records(QM9_TASKS),
        ),
        (
            results / "semantic_output_carriage.csv",
            _raw_output_carriage(QM9_TASKS),
        ),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    result = figures(
        ZincReachConfig(
            profile=QM9_PROFILE,
            tasks=QM9_TASKS,
            bootstrap_replicates=40,
        ),
        output_dir=tmp_path,
    )
    assert set(result["figures"]) == {
        "interpolation_sweep",
        "semantic_functional",
        "semantic_bamberger",
        "structural_functional",
        "expected_distance",
        "output_coherence",
        "scale_dependence",
        "scale_slopes",
    }
    assert all(
        Path(formats["png"]).name.startswith("qm9_")
        for formats in result["figures"].values()
    )
