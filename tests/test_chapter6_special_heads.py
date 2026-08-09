from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from graph_specialisation_metrics.chapter6_special_heads import (
    head_distance_profiles,
    select_special_heads_across_seeds,
    validate_special_head_outputs,
)
from graph_specialisation_metrics.methodology.grit_figure_data import (
    _apply_figure_analysis_subset_patch,
)
from graph_specialisation_metrics.methodology.grit_figure_plots import (
    attention_cmap_name,
    plot_attention_grid_publication,
    plot_av_pca_publication,
)
from graph_specialisation_metrics.specialisation.model import (
    load_grit_checkpoint_strict,
)


def _head_row(seed, layer, head, d_rel, joint, *, family="other"):
    return {
        "task": "zinc_1hop",
        "seed": seed,
        "layer": layer,
        "head": head,
        "family": family,
        "selectivity": d_rel,
        "joint_sensitivity": joint,
        "raw_semantic_score": joint * (1 + d_rel),
        "raw_structural_score": joint * (1 - d_rel),
    }


def test_cross_seed_selection_is_distinct_and_joins_ablation():
    rows = [
        _head_row(0, 0, 0, 0.8, 1.2),
        _head_row(1, 1, 1, -0.7, 1.1),
        _head_row(2, 2, 2, 0.03, 2.5),
        _head_row(0, 3, 3, 0.2, 1.4),
        _head_row(1, 4, 4, 0.9, 0.05, family="inactive"),
    ]
    ablations = [
        {
            "task": row["task"],
            "seed": row["seed"],
            "layer": row["layer"],
            "head": row["head"],
            "joint_sensitivity": row["joint_sensitivity"],
            "prediction_movement": 0.1 + index,
            "loss_change": 0.01 + index,
            "clean_ablation_graphs": 64,
        }
        for index, row in enumerate(rows)
    ]
    selected = select_special_heads_across_seeds(
        rows,
        ablations,
        model_labels={"zinc_1hop": "1-hop"},
    )
    assert [row["role"] for row in selected] == [
        "semantic",
        "structural",
        "highest_joint",
    ]
    assert [(row["seed"], row["layer"], row["head"]) for row in selected] == [
        (1, 4, 4),
        (1, 1, 1),
        (2, 2, 2),
    ]
    assert selected[2]["display_family"] == "generalist"
    assert selected[2]["head_ablation_impact"] == pytest.approx(2.1)


def test_head_distance_profiles_preserve_virtual_bin():
    semantic = np.asarray([[[1.0, 2.0, 3.0, 4.0]]])
    structural = np.asarray([[[4.0, 3.0, 2.0, 1.0]]])
    attention = np.asarray([[[1.0, 1.0, 1.0, 7.0]]])
    profiles = head_distance_profiles(
        {
            "axis": ("0", "1", "2", "virtual"),
            "channels": {
                "semantic": {"heatmap_exact_head": semantic},
                "structural": {"heatmap_exact_head": structural},
            },
            "clean_attention_distance": attention,
        },
        (0, 0),
    )
    assert profiles["labels"] == ("0", "1", "2", "3", "4-7", "8+", "virtual")
    assert profiles["semantic"].sum() == pytest.approx(1.0)
    assert profiles["structural"].sum() == pytest.approx(1.0)
    assert profiles["attention"][-1] == pytest.approx(0.7)


def test_publication_attention_uses_family_colour_and_labels_virtual_node():
    Chem = pytest.importorskip("rdkit.Chem")
    molecule = Chem.MolFromSmiles("CCO")
    payload = {
        "task": "zinc_1hop_vnode",
        "dataset_label": "ZINC-subset",
        "display_title": "ZINC-subset — 1-hop GRIT+RRWP + VNode",
        "examples": [
            {
                "dataset_index": 7,
                "n_atoms": 3,
                "attention_nodes": 4,
                "has_virtual_node": True,
                "mol_block": Chem.MolToMolBlock(molecule),
                "rdkit_sanitized": True,
                "formula": "C2H6O",
                "attention": {
                    "semantic": np.asarray(
                        [
                            [0.6, 0.1, 0.1, 0.2],
                            [0.1, 0.6, 0.1, 0.2],
                            [0.1, 0.1, 0.6, 0.2],
                            [0.2, 0.2, 0.2, 0.4],
                        ]
                    )
                },
            }
        ],
    }
    figure = plot_attention_grid_publication(
        payload,
        attention_key="semantic",
        attention_family="semantic",
        head=(3, 4),
        title_label="Most semantic head",
        net_d_rel=0.52,
        net_joint_sensitivity=1.7,
        head_ablation_impact=0.04,
    )
    matrix_axis = next(axis for axis in figure.axes if axis.get_xlabel() == "Key atom")
    assert matrix_axis.images[0].get_cmap().name == "Oranges"
    assert "VN" in [label.get_text() for label in matrix_axis.get_xticklabels()]
    title_text = " ".join(text.get_text() for axis in figure.axes for text in axis.texts)
    assert "Most semantic head" in title_text
    assert "ablate" in title_text
    assert attention_cmap_name("generalist") == "Purples"


def test_figure_subset_patch_runs_before_positional_encoding(tmp_path):
    loader = tmp_path / "grit/loader/master_loader.py"
    loader.parent.mkdir(parents=True)
    loader.write_text(
        "import logging\n"
        "import os.path as osp\n"
        "import torch\n"
        "\n"
        "@register_loader('custom_master_loader')\n"
        "def load_dataset_master(format, name, dataset_dir):\n"
        "    dataset = object()\n"
        "    log_loaded_dataset(dataset, format, name)\n"
        "    return dataset\n",
        encoding="utf-8",
    )
    _apply_figure_analysis_subset_patch(tmp_path)
    patched = loader.read_text(encoding="utf-8")
    assert "def _gsm_figure_analysis_subset(dataset):" in patched
    assert patched.index("dataset = _gsm_figure_analysis_subset(dataset)") < patched.index(
        "log_loaded_dataset(dataset, format, name)"
    )
    # Idempotence matters because Colab reuses patched task clones.
    _apply_figure_analysis_subset_patch(tmp_path)
    assert loader.read_text(encoding="utf-8").count(
        "def _gsm_figure_analysis_subset(dataset):"
    ) == 1


def test_checkpoint_can_be_swapped_without_rebuilding_model(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "seed.ckpt"
    expected = {
        key: torch.full_like(value, 0.25) for key, value in model.state_dict().items()
    }
    torch.save({"model_state": expected}, checkpoint)
    how = load_grit_checkpoint_strict(model, checkpoint)
    assert how == "strict"
    for value in model.state_dict().values():
        assert torch.allclose(value, torch.full_like(value, 0.25))
    assert model.training is False


def test_special_head_completion_requires_every_role_and_bundle(tmp_path):
    selected = [
        {
            "task": "zinc_1hop",
            "model_label": "1-hop",
            "role": role,
            "seed": index,
            "layer": index,
            "head": index,
        }
        for index, role in enumerate(("semantic", "structural", "highest_joint"))
    ]
    outputs = []
    for row in selected:
        for figure, count in (("attention", 2), ("pca", 1), ("distance", 1)):
            for page in range(count):
                stem = tmp_path / f"{row['role']}-{figure}-{page}"
                paths = {}
                for suffix, key in ((".png", "png"), (".pdf", "pdf"), (".json", "metadata")):
                    path = stem.with_suffix(suffix)
                    path.write_text("complete", encoding="utf-8")
                    paths[key] = str(path)
                outputs.append(
                    {
                        "task": row["task"],
                        "seed": row["seed"],
                        "role": row["role"],
                        "figure": figure,
                        **paths,
                    }
                )
    completion = validate_special_head_outputs(
        selected,
        outputs,
        tasks=("zinc_1hop",),
        render_graph_count=10,
        examples_per_page=5,
    )
    assert len(completion) == 3
    assert all(row["complete"] for row in completion)
    with pytest.raises(RuntimeError, match="incomplete special-head figures"):
        validate_special_head_outputs(
            selected,
            outputs[:-1],
            tasks=("zinc_1hop",),
            render_graph_count=10,
            examples_per_page=5,
        )


def test_publication_pca_has_fixed_chemistry_legend_and_metrics():
    rng = np.random.default_rng(7)
    figure = plot_av_pca_publication(
        {
            "task": "qm9_gap_dense",
            "display_title": "QM9 — dense GRIT+RRWP",
            "head": (2, 3),
            "vectors": rng.normal(size=(8, 4)),
            "labels": ["O: carbonyl"] * 4 + ["H: hydrogen"] * 4,
            "n_used": 8,
        },
        title_label="Highest-J head",
        d_rel=0.02,
        joint_sensitivity=2.2,
        head_ablation_impact=0.03,
    )
    assert len(figure.legends) == 1
    assert len(figure.legends[0].get_texts()) == 2
    assert "Highest-J head" in figure.axes[0].get_title()
    assert "ablate" in figure.axes[0].get_title()
