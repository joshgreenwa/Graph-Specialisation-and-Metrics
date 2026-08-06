"""Static contract checks for the paste-ready canonical QM9 Colab launcher."""

import ast
from pathlib import Path


LAUNCHER = (
    Path(__file__).parents[1]
    / "experiments"
    / "methodology"
    / "canonical_methodology_colab.py"
)
EXPECTED_TASKS = (
    "qm9_gap_1hop",
    "qm9_gap_1hop_local",
    "qm9_gap_2hop",
    "qm9_gap_1hop_vnode",
    "qm9_gap_2hop_vnode",
    "qm9_gap_dense",
)


def _literal_assignments():
    tree = ast.parse(LAUNCHER.read_text(encoding="utf-8"))
    assignments = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            assignments[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return assignments


def test_launcher_targets_all_qm9_score_and_carriage_controls():
    values = _literal_assignments()

    assert values["BRANCH"] == "expansion/carriage_experiments"
    assert values["TASKS"] == EXPECTED_TASKS
    assert values["TRAIN_SEEDS"] == (42,)
    assert values["TASK_TRAIN_SEEDS"] == {}
    assert values["PHASES"] == ("scores", "carriage")
    assert values["SIZES"]["discovery_graphs"] == 48
    assert values["EXECUTION"] == {
        "graphs_per_batch": 8,
        "oom_backoff": True,
    }


def test_launcher_performs_fail_closed_consolidated_cache_postflight():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "load_cache_artifact_file" in source
    assert '"scores" / "raw.pt"' in source
    assert '"carriage" / "fields.pt"' in source
    assert 'artifacts["scores"].value["channels"][channel]["graph_scores"]' in source
    assert 'artifacts["carriage"].value["channels"][channel]["graph_fields"]' in source
    assert 'seed_root.glob("cache/**/*.partial")' in source
