from pathlib import Path

import pytest

from graph_specialisation_metrics.zinc_reach_analysis import (
    TASKS,
    ZincReachConfig,
    discover_seed_checkpoint,
    figures,
    graph_bamberger_profiles,
    graph_donor_profiles,
    summarise_graph_profiles,
)


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
                    donor.append(
                        {
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

    profiles, expected = summarise_graph_profiles(
        graph_rows,
        bootstrap_replicates=40,
        bootstrap_seed=7,
    )
    assert profiles and expected
    assert not any(row["method"] == "local_jacobian" for row in graph_rows)


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
        "profiles",
        "expected_distance",
    }
    for formats in result["figures"].values():
        assert Path(formats["png"]).is_file()
        assert Path(formats["pdf"]).is_file()
