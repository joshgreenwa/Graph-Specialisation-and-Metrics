import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
    summarise_beneficial_carriage,
    summarise_graph_profiles,
    summarise_interpolation_contrasts,
    summarise_output_coherence,
    summarise_scale_dependence,
    summarise_shell_survival,
)
from graph_specialisation_metrics.reach_redundancy import (
    SemanticAssignment,
    SemanticCoalition,
    minimum_gap_derangement,
    survival_components,
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


def test_signed_output_path_carriage_is_complete_and_preserves_cancellation(
    monkeypatch,
    capsys,
):
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
        source_degree=1,
        donor_degree=1,
        degree_gap=0,
        dose=1.0,
        payload_fingerprint="fixture",
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

    exact_integrator = reach.integrated_loss_carriage

    def unconverged_integrator(*args, **kwargs):
        result = exact_integrator(*args, **kwargs)
        result["quadrature_error"] = torch.ones_like(result["quadrature_error"])
        result["converged"] = torch.zeros_like(result["converged"])
        return result

    monkeypatch.setattr(reach, "integrated_loss_carriage", unconverged_integrator)
    retained = reach._signed_output_carriage_rows(
        ZincReachConfig(tasks=("zinc",)),
        prepared,
        task="zinc",
        graph_id=0,
        base=base,
        variants=[SimpleNamespace()],
        events=[event],
    )
    assert not retained[0]["audit_accepted"]
    assert retained[0]["signed_output_carriage"] == pytest.approx(1.0)
    assert "retaining best estimates" in capsys.readouterr().out
    audit = reach.summarise_output_carriage_audit(retained)
    assert audit[0]["soft_warning_paths"] == 1
    assert audit[0]["unconverged_paths"] == 1


def test_beneficial_carriage_uses_positive_is_beneficial_loss_sign():
    clean = torch.tensor([[2.0], [1.0]])
    intervened = torch.tensor([[3.0], [1.0]])
    capture = SimpleNamespace(
        final_state=torch.stack((clean, intervened)),
        prediction=torch.tensor([[3.0], [4.0]]),
        target=torch.zeros(2, 1),
    )

    class Backend:
        def capture_groups(self, groups):
            assert len(groups) == 1 and len(groups[0]) == 2
            return [capture]

        def loss_from_pooled(self, target):
            assert tuple(target.shape) == (1, 1)
            return lambda pooled: pooled[:, 0].abs()

        def loss_per_graph(self, prediction, target):
            return (prediction[:, 0] - target[:, 0]).abs()

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
        source_degree=1,
        donor_degree=1,
        degree_gap=0,
        dose=1.0,
        payload_fingerprint="fixture",
    )
    rows = reach._beneficial_rows_for_channel(
        ZincReachConfig(tasks=("zinc",)),
        prepared,
        task="zinc",
        graph_id=0,
        channel="semantic",
        base=base,
        variants=[SimpleNamespace()],
        events=[event],
    )

    assert sum(row["beneficial_carriage"] for row in rows) == pytest.approx(1.0)
    assert rows[0]["event_loss_increase"] == pytest.approx(1.0)
    assert rows[0]["direct_loss_increase"] == pytest.approx(1.0)
    assert all(row["audit_accepted"] for row in rows)


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


def _raw_beneficial_carriage(tasks=TASKS):
    return [
        {
            "task": task,
            "model_label": reach.TASK_LABELS[task],
            "graph": graph,
            "channel": channel,
            "source": 0,
            "donor_graph": 9,
            "donor_node": 1,
            "draw": 0,
            "carrier": carrier,
            "distance": distance,
            "beneficial_carriage": value + 0.002 * task_index,
        }
        for task_index, task in enumerate(tasks)
        for graph in (0, 1)
        for channel in ("semantic", "structural")
        for carrier, (distance, value) in enumerate(
            ((0, 0.04), (1, -0.02), (2, 0.01))
        )
    ]


def _raw_shell_survival(tasks=TASKS):
    rows = []
    for task_index, task in enumerate(tasks):
        for graph in (0, 1):
            for condition in ("exact_shell", "far_tail"):
                for radius in (1, 2):
                    for draw in (0, 1):
                        for intervention, offset in (
                            ("shell_permutation", 0.0),
                            ("shell_replacement", 0.25),
                        ):
                            survival = 0.35 + offset + 0.01 * task_index
                            additive = 0.25 + 0.5 * offset
                            rows.append(
                                {
                                    "task": task,
                                    "model_label": reach.TASK_LABELS[task],
                                    "graph": graph,
                                    "carrier": 0,
                                    "draw": draw,
                                    "condition": condition,
                                    "radius": radius,
                                    "intervention": intervention,
                                    "coalition_size": radius + 1,
                                    "carrier_survival": survival,
                                    "carrier_additive_survival": additive,
                                    "carrier_nonlinear_residual": survival - additive,
                                    "output_survival": survival + 0.05,
                                    "output_additive_survival": additive + 0.02,
                                    "output_nonlinear_residual": survival - additive + 0.03,
                                }
                            )
    return rows


def test_survival_decomposition_is_unclipped_and_exact():
    result = survival_components(
        [[1.0, 0.0], [-1.0, 0.0]],
        [3.0, 0.0],
        effect_floor=1.0e-12,
    )
    assert result["apparent_mass"] == pytest.approx(2.0)
    assert result["additive_survival"] == pytest.approx(0.0)
    assert result["survival"] == pytest.approx(1.5)
    assert result["nonlinear_residual"] == pytest.approx(1.5)


def test_shell_derangement_handles_repeated_atom_types_without_losing_multiset():
    payloads = np.asarray([[6], [6], [8]])
    degrees = np.asarray([2, 2, 1])
    result = minimum_gap_derangement(
        (0, 1, 2),
        payloads,
        degrees,
        np.random.default_rng(4),
    )
    assert result is not None
    donors, exact = result
    assert exact
    assert sorted(donors) == [0, 1, 2]
    assert all(source != donor for source, donor in zip((0, 1, 2), donors))
    assert sum(
        payloads[source].tolist() != payloads[donor].tolist()
        for source, donor in zip((0, 1, 2), donors)
    ) == 2


def test_new_analysis_summaries_bootstrap_at_graph_level():
    beneficial_graph, beneficial = summarise_beneficial_carriage(
        _raw_beneficial_carriage(("zinc",)),
        bootstrap_replicates=40,
        bootstrap_seed=21,
    )
    survival_graph, survival, contrasts = summarise_shell_survival(
        _raw_shell_survival(("zinc",)),
        bootstrap_replicates=40,
        bootstrap_seed=22,
    )
    assert beneficial_graph and beneficial
    assert survival_graph and survival and contrasts
    assert all(row["graphs"] == 2 for row in beneficial)
    assert all(row["graphs"] == 2 for row in survival)
    assert all(row["paired_graphs"] == 2 for row in contrasts)
    primary = next(
        row
        for row in contrasts
        if row["metric"] == "carrier_survival"
        and row["condition"] == "exact_shell"
        and row["radius"] == 1
    )
    assert primary["mean"] == pytest.approx(0.25)


def test_sampled_defaults_and_component_cache_fingerprints():
    config = ZincReachConfig()
    assert config.graphs == 64
    assert config.survival_carriers_per_graph == 1
    assert config.survival_draws == 1
    assert config.beneficial_donors_per_source == 1
    changed_runtime_scope = replace(
        config,
        graphs=32,
        survival_replica_batch_size=512,
    )
    assert changed_runtime_scope.fingerprint != config.fingerprint
    assert changed_runtime_scope.core_cache_fingerprint == config.core_cache_fingerprint
    changed_survival = replace(config, survival_draws=2)
    assert changed_survival.core_cache_fingerprint == config.core_cache_fingerprint
    assert changed_survival.survival_cache_fingerprint != config.survival_cache_fingerprint


def test_nested_survival_group_reuses_singletons_for_every_tail():
    class Data:
        def __init__(self, x):
            self.x = x

        def clone(self):
            return copy.deepcopy(self)

    base = Data(torch.tensor([[6], [7]], dtype=torch.long))
    task = SimpleNamespace(content_adapter=None)

    def coalition(source, payload, intervention):
        variant = base.clone()
        variant.x[source] = payload
        assignment = SemanticAssignment(
            source=source,
            donor_graph=9,
            donor_node=source,
            source_degree=1,
            donor_degree=1,
            dose=1.0,
            payload=(payload,),
        )
        return SemanticCoalition(
            intervention=intervention,
            assignments=(assignment,),
            singleton_variants=(variant,),
            joint_variant=variant.clone(),
            shell_sizes=(1,),
            skipped_shell_sizes=(),
            exact_derangement=True,
            matching_error=0.0,
        )

    first = coalition(0, 8, "shell_replacement")
    second = coalition(1, 9, "shell_replacement")
    group = reach._build_survival_capture_group(
        base,
        carrier=0,
        draw=0,
        intervention="shell_replacement",
        shell_coalitions={1: first, 2: second},
        tail_radii=(1, 2, 3),
        task=task,
    )

    # Two singletons, two exact joints and two tail joints. Without reuse the
    # two tails alone would add three more singleton forwards.
    assert len(group.variants) == 6
    assert len(group.conditions) == 4
    exact, exact_two, tail_one, tail_two = group.conditions
    assert exact.singleton_positions == (0,)
    assert exact_two.singleton_positions == (1,)
    assert tail_one.singleton_positions == (0, 1)
    assert tail_two.singleton_positions == (1,)


def test_completed_slow_run_core_shard_is_reused_and_upgradable(tmp_path: Path):
    config = ZincReachConfig(tasks=("zinc",))
    legacy = reach._legacy_slow_run_fingerprint(config)
    path = tmp_path / "graph.pt"
    torch.save(
        {
            "analysis_version": config.profile.analysis_version,
            "fingerprint": legacy,
            "checkpoint_sha256": "checkpoint",
            "graph_record": {},
            "donor_rows": [],
            "interpolation_rows": [],
            "bamberger_rows": [],
        },
        path,
    )

    loaded = reach._load_shard(
        path,
        analysis_version=config.profile.analysis_version,
        fingerprint=config.fingerprint,
        cache_fingerprint=config.core_cache_fingerprint,
        checkpoint_sha256="checkpoint",
        compatible_legacy_fingerprints=(legacy,),
    )
    assert loaded is not None
    rejected = reach._load_shard(
        path,
        analysis_version=config.profile.analysis_version,
        fingerprint=config.fingerprint,
        cache_fingerprint=config.core_cache_fingerprint,
        checkpoint_sha256="checkpoint",
    )
    assert rejected is None


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
        (results / "beneficial_carriage.csv", _raw_beneficial_carriage()),
        (results / "shell_survival.csv", _raw_shell_survival()),
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
        "beneficial_carriage",
        "shell_redundancy",
        "tail_redundancy",
        "scale_dependence",
        "scale_slopes",
    }
    assert (results / "dense_profile_contrasts.csv").is_file()
    assert (results / "interpolation_sweep_summary.csv").is_file()
    assert (results / "shell_survival_by_coalition_size.csv").is_file()
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
        (
            results / "beneficial_carriage.csv",
            _raw_beneficial_carriage(QM9_TASKS),
        ),
        (
            results / "shell_survival.csv",
            _raw_shell_survival(QM9_TASKS),
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
        "beneficial_carriage",
        "shell_redundancy",
        "tail_redundancy",
        "scale_dependence",
        "scale_slopes",
    }
    assert all(
        Path(formats["png"]).name.startswith("qm9_")
        for formats in result["figures"].values()
    )
