from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


RUNNER_PATH = (
    Path(__file__).parents[1]
    / "experiments/peptides/training/GRIT_peptides_khop.py"
)
RUNNER_SPEC = importlib.util.spec_from_file_location("grit_peptides_khop", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


@pytest.mark.parametrize("task", ["func", "struct"])
def test_requested_peptides_variants_have_isolated_drive_roots(task: str) -> None:
    cases = [
        (["--attention", "khop", "--hops", "1", "--rrwp-horizon", "1"], "1hop_localrrwp_h1"),
        (["--attention", "khop", "--hops", "1", "--global-vnode"], "1hop_vnode"),
        (["--attention", "khop", "--hops", "2"], "2hop"),
    ]
    for options, slug in cases:
        args = runner.parse_args(["--task", task, *options])
        assert runner.variant_slug(args) == slug
        assert args.drive_dir == Path(
            f"/content/drive/MyDrive/grit_peptides_{task}_{slug}"
        )
        assert args.dataset_dir == Path(
            f"/content/drive/MyDrive/grit_peptides_{task}_shared_data"
        )


def test_local_rrwp_preserves_task_specific_encoder_width() -> None:
    func = runner.parse_args(
        ["--task", "func", "--attention", "khop", "--hops", "1", "--rrwp-horizon", "1"]
    )
    struct = runner.parse_args(
        ["--task", "struct", "--attention", "khop", "--hops", "1", "--rrwp-horizon", "1"]
    )

    assert runner.expected_config_values(func)[("posenc_RRWP", "ksteps")] == 17
    assert runner.expected_config_values(struct)[("posenc_RRWP", "ksteps")] == 24
    assert runner.expected_config_values(func)[("posenc_RRWP", "local_horizon")] == 1
    assert runner.expected_config_values(struct)[("posenc_RRWP", "local_horizon")] == 1


@pytest.mark.parametrize(
    ("task", "hops"),
    [("func", 17), ("struct", 24)],
)
def test_hop_limit_tracks_available_rrwp_channels(task: str, hops: int) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--task", task, "--attention", "khop", "--hops", str(hops)])


def test_training_command_forwards_local_rrwp_horizon(tmp_path: Path) -> None:
    args = runner.parse_args(
        [
            "--task",
            "func",
            "--attention",
            "khop",
            "--hops",
            "1",
            "--rrwp-horizon",
            "1",
            "--drive-dir",
            str(tmp_path),
            "--dataset-dir",
            str(tmp_path / "shared_data"),
        ]
    )
    command = runner.build_train_command(args, Path("config.yaml"))

    assert command[command.index("posenc_RRWP.local_horizon") + 1] == "1"


def test_streaming_rrwp_merge_supports_modern_pyg_data_keys(
    tmp_path: Path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    pyg_data = pytest.importorskip("torch_geometric.data")
    from graph_specialisation_metrics.grit_patches import peptides

    transforms = tmp_path / "grit/transform/transforms.py"
    transforms.parent.mkdir(parents=True)
    transforms.write_text("import torch\nfrom tqdm import tqdm\n", encoding="utf-8")
    peptides.apply_peptides_streaming_rrwp_patch(
        SimpleNamespace(log=lambda _message: None), tmp_path
    )

    spec = importlib.util.spec_from_file_location("patched_peptides_transforms", transforms)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    first, _ = module._grit_colab_collate_chunk(
        [pyg_data.Data(x=torch.tensor([[1.0], [2.0]]))]
    )
    second, _ = module._grit_colab_collate_chunk(
        [pyg_data.Data(x=torch.tensor([[3.0]]))]
    )
    assert callable(first.keys)

    merged = module._grit_colab_merge_data_chunks([first, second])
    assert merged.x.tolist() == [[1.0], [2.0], [3.0]]

    class Dataset:
        def __init__(self, items):
            self.items = items
            self._indices = None
            self._data_list = None

        def __len__(self):
            return len(self.items)

        def len(self):
            return len(self.items)

        def get(self, index):
            return self.items[index].clone()

    items = [
        pyg_data.Data(
            x=torch.tensor([[1.0], [2.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            rrwp=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ),
        pyg_data.Data(
            x=torch.tensor([[3.0]]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            rrwp=torch.tensor([[2.0, 2.0]]),
        ),
        pyg_data.Data(
            x=torch.tensor([[4.0], [5.0], [6.0]]),
            edge_index=torch.tensor([[0, 2], [2, 0]]),
            rrwp=torch.tensor([[3.0, 0.0], [0.0, 3.0], [1.0, 1.0]]),
        ),
    ]
    dataset = Dataset(items)
    monkeypatch.setenv("GRIT_PE_STREAM_CHUNK_SIZE", "2")
    module.pre_transform_in_memory(dataset, lambda data: data, show_progress=False)

    assert [dataset.get(index).x.tolist() for index in range(3)] == [
        [[1.0], [2.0]],
        [[3.0]],
        [[4.0], [5.0], [6.0]],
    ]
    for index, expected in enumerate(items):
        actual = dataset.get(index)
        assert torch.equal(actual.edge_index, expected.edge_index)
        assert torch.equal(actual.rrwp, expected.rrwp)


def test_streaming_rrwp_patch_upgrades_legacy_keys_access(tmp_path: Path) -> None:
    from graph_specialisation_metrics.grit_patches import peptides

    transforms = tmp_path / "grit/transform/transforms.py"
    transforms.parent.mkdir(parents=True)
    transforms.write_text(
        "def _grit_colab_streaming_get_no_cache(self, idx):\n"
        "    pass\n\n"
        "def _grit_colab_merge_data_chunks(data_parts):\n"
        "    for key in sorted(list(data_parts[0].keys), key=_key_bytes, reverse=True):\n"
        "        pass\n",
        encoding="utf-8",
    )
    peptides.apply_peptides_streaming_rrwp_patch(
        SimpleNamespace(log=lambda _message: None), tmp_path
    )

    patched = transforms.read_text(encoding="utf-8")
    assert "chunk_keys = data_parts[0].keys" in patched
    assert "if callable(chunk_keys):" in patched
    assert "list(data_parts[0].keys)" not in patched
