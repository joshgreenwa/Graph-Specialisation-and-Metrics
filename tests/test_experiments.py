import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from graph_specialisation_metrics.experiments import (
    graphbench,
    grit,
    save_scores,
)
from graph_specialisation_metrics.plotting import load_scores, plot_scores

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_setup_lock_serialises_array_jobs(tmp_path: Path) -> None:
    script = """
import sys, time
from pathlib import Path
from graph_specialisation_metrics.experiments import dataset_lock
cache, log, label, delay = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], float(sys.argv[4])
with dataset_lock(cache, 'test'):
    with log.open('a') as handle:
        handle.write(f'start {label}\\n')
    time.sleep(delay)
    with log.open('a') as handle:
        handle.write(f'end {label}\\n')
"""
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    cache, log = tmp_path / "data", tmp_path / "workers.txt"
    first = subprocess.Popen(
        [sys.executable, "-c", script, str(cache), str(log), "first", "0.3"],
        env=environment,
    )
    deadline = time.monotonic() + 5
    while (
        not log.is_file() or "start first" not in log.read_text()
    ) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert log.is_file(), "first process did not acquire the dataset lock"
    second = subprocess.Popen(
        [sys.executable, "-c", script, str(cache), str(log), "second", "0"],
        env=environment,
    )
    assert first.wait(timeout=5) == 0
    assert second.wait(timeout=5) == 0
    assert log.read_text().splitlines() == [
        "start first",
        "end first",
        "start second",
        "end second",
    ]


def _config(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_job_tables() -> None:
    assert len(graphbench.jobs(_config("graphbench.yaml"))) == 4
    assert len(grit.jobs(_config("grit_zinc.yaml"))) == 15
    assert len(grit.jobs(_config("grit_qm9.yaml"))) == 15
    assert grit.jobs(_config("grit_zinc.yaml"), fast=True) == [
        {"experiment": "grit", "dataset": "zinc", "variant": "dense", "seed": 0}
    ]
    selected = _config("grit_zinc.yaml")
    selected["job_index"] = 7
    assert grit.jobs(selected, fast=True) == [
        {"experiment": "grit", "dataset": "zinc", "variant": "1hop_vnode", "seed": 1}
    ]


def test_grit_array_jobs_share_one_default_dataset_root(tmp_path: Path) -> None:
    first = grit._dataset_root({}, tmp_path / "job_0")
    second = grit._dataset_root({}, tmp_path / "job_29")

    assert first == second == (tmp_path / "data").resolve()
    assert (
        grit._dataset_root({"data": {"root": tmp_path / "custom"}}, tmp_path / "job_0")
        == (tmp_path / "custom").resolve()
    )


def test_score_archive_format_and_distance_contributions(tmp_path: Path) -> None:
    path = save_scores(
        tmp_path,
        [1.0, 2.0],
        [3.0, 4.0],
        semantic_distance_contributions=[[1.0, 0.0], [0.5, 1.5]],
        structural_distance_contributions=[[3.0, 0.0], [1.0, 3.0]],
        distance_categories=[0, 1],
    )
    expected_fields = {
        "semantic_scores",
        "structural_scores",
        "semantic_distance_contributions",
        "structural_distance_contributions",
        "distance_categories",
    }
    with np.load(path) as archive:
        assert set(archive.files) == expected_fields
        np.testing.assert_array_equal(archive["distance_categories"], ["0", "1"])
    assert set(load_scores(path)) == expected_fields

    invalid_schema = tmp_path / "invalid_schema.npz"
    np.savez(invalid_schema, unknown_semantic=[1.0], unknown_structural=[1.0])
    with pytest.raises(ValueError, match="semantic_scores"):
        load_scores(invalid_schema)
    with pytest.raises(ValueError, match="do not reconstruct"):
        save_scores(
            tmp_path / "bad",
            [1.0],
            [1.0],
            semantic_distance_contributions=[[0.5]],
            structural_distance_contributions=[[1.0]],
            distance_categories=[0],
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        save_scores(
            tmp_path / "negative",
            [1.0],
            [1.0],
            semantic_distance_contributions=[[-1.0, 2.0]],
            structural_distance_contributions=[[1.0, 0.0]],
            distance_categories=[0, 1],
        )
    with pytest.raises(ValueError, match="non-empty and unique"):
        save_scores(
            tmp_path / "duplicate",
            [1.0],
            [1.0],
            semantic_distance_contributions=[[0.5, 0.5]],
            structural_distance_contributions=[[0.5, 0.5]],
            distance_categories=[0, 0],
        )


def test_grit_structural_donor_swap_keeps_attention_support_fixed() -> None:
    torch = pytest.importorskip("torch")
    Data = pytest.importorskip("torch_geometric.data").Data
    nodes, width = 3, 2
    dense = torch.arange(nodes * nodes * width, dtype=torch.float).reshape(nodes, nodes, width)
    row, column = torch.meshgrid(torch.arange(nodes), torch.arange(nodes), indexing="ij")
    data = Data(
        x=torch.tensor([[1], [2], [3]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.arange(4),
        rrwp=torch.arange(nodes * width, dtype=torch.float).reshape(nodes, width),
        rrwp_index=torch.stack((row.reshape(-1), column.reshape(-1))),
        rrwp_val=dense.reshape(-1, width),
        rrwp_attention_edge_index=torch.tensor([[0, 1, 1], [0, 0, 1]]),
        deg=torch.tensor([1, 2, 3]),
        log_deg=torch.log(torch.tensor([2.0, 3.0, 4.0])),
    )

    donor_swap = grit._structural_donor_swap(data, source=0, donor=2)
    donor_swap_dense, _ = grit._dense_rrwp(donor_swap)

    torch.testing.assert_close(donor_swap.rrwp[0], data.rrwp[2])
    torch.testing.assert_close(donor_swap_dense[0, 1:], dense[2, 1:])
    torch.testing.assert_close(donor_swap_dense[1:, 0], dense[1:, 2])
    torch.testing.assert_close(donor_swap_dense[0, 0], dense[2, 2])
    torch.testing.assert_close(donor_swap.edge_index, data.edge_index)
    torch.testing.assert_close(donor_swap.edge_attr, data.edge_attr)
    torch.testing.assert_close(donor_swap.rrwp_attention_edge_index, data.rrwp_attention_edge_index)
    torch.testing.assert_close(donor_swap.x, data.x)


def test_plotting_normalises_each_trained_seed_separately() -> None:
    scores = {
        "semantic_scores": np.array([1.0, 3.0, 10.0, 30.0]),
        "structural_scores": np.array([4.0, 4.0, 40.0, 40.0]),
        "seed": np.array([0, 0, 1, 1]),
        "semantic_distance_contributions": np.array([[1.0], [3.0], [10.0], [30.0]]),
        "structural_distance_contributions": np.array([[4.0], [4.0], [40.0], [40.0]]),
        "distance_categories": np.array([0]),
    }

    figure = plot_scores(scores)

    offsets = figure.axes[1].collections[0].get_offsets()
    np.testing.assert_allclose(offsets[:, 0], [-1 / 3, 0.2, -1 / 3, 0.2])
    np.testing.assert_allclose(offsets[:, 1], [0.75, 1.25, 0.75, 1.25])
    assert figure.axes[0].get_title() == "Semantic and structural specialisation scores"
    assert figure.axes[0].get_ylabel() == r"Specialisation score $S_c(t,h)$"
    assert figure.axes[1].get_title() == "Joint sensitivity and selectivity"
    assert figure.axes[1].get_xlabel() == r"Selectivity $D_{\mathrm{rel}}(t,h)$"
    assert figure.axes[1].get_ylabel() == r"Joint sensitivity $J(t,h)$"
    assert figure.axes[2].get_title() == "Distance-resolved score contributions"
    assert figure.axes[2].get_xlabel() == r"Distance category $d$"
    assert figure.axes[2].get_ylabel() == r"Mean score contribution $C_c(t,h,d)$"


def test_grit_archive_variant_names_match_the_paper() -> None:
    from types import SimpleNamespace

    def config(*, dense: bool, hops: int = 1, virtual_node: bool = False):
        class Attention(dict):
            __getattr__ = dict.__getitem__

        attention = Attention(
            full_attn=dense,
            hops=hops,
            global_vnode=virtual_node,
        )
        return SimpleNamespace(gt=SimpleNamespace(attn=attention))

    assert grit._variant_name(config(dense=True)) == "Dense GRIT"
    assert grit._variant_name(config(dense=False)) == "1-hop"
    assert grit._variant_name(config(dense=False, hops=2, virtual_node=True)) == "2-hop + VNode"
