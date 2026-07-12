#!/usr/bin/env python3
"""Shared Colab training logic for GRIT on Peptides-struct.

The public entrypoints are:

* ``grit_peptides_struct_core.py`` for the official dense GRIT run;
* ``grit_peptides_struct_1hop_core.py`` for the 1-hop global-RRWP control.

This module intentionally reuses the tested ZINC Colab runner's dependency,
checkpoint, logging, and recovery machinery. Only the task-specific official
config, command overrides, and 1-hop config text differ.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


OFFICIAL_REPO = "https://github.com/LiamMa/GRIT.git"
OFFICIAL_COMMIT = "6c988ea600a606fbb49a2246c64a2d37396b3ab5"
DENSE_OFFICIAL_CFG = "configs/GRIT/peptides-struct-GRIT-RRWP.yaml"
ONEHOP_OFFICIAL_CFG = "configs/GRIT/peptides-struct-GRIT-RRWP-1hop.yaml"
DEFAULT_DENSE_DRIVE_DIR = "/content/drive/MyDrive/grit_peptides_struct_official"
DEFAULT_ONEHOP_DRIVE_DIR = "/content/drive/MyDrive/grit_peptides_struct_1hop"
DEFAULT_DENSE_REPO_DIR = "/content/GRIT_peptides_struct"
DEFAULT_ONEHOP_REPO_DIR = "/content/GRIT_peptides_struct_1hop"


PEPTIDES_STRUCT_ONEHOP_CFG_TEXT = """\
# Parameter-matched 1-hop sparse-control variant of the official Peptides-struct GRIT RRWP config.
#
# This keeps the official Peptides-struct GRIT architecture, RRWP dimensions,
# optimizer, schedule, loss, and metric setup. The only scientific intervention
# is `gt.attn.full_attn=False` plus `gt.attn.sparsity=one_hop`, selecting the
# masked RRWP relative encoder so attention/message passing is restricted to
# molecular bonds plus self while global RRWP values remain available on those
# local edges and as node RRWP features.
out_dir: results
metric_best: mae
metric_agg: argmin
accelerator: "cuda:0"
mlflow:
  use: False
  project: Exp
  name: peptides-struct-GRIT-RRWP-1hop
wandb:
  use: False
  project: peptides-struct
dataset:
  format: OGB
  name: peptides-structural
  task: graph
  task_type: regression
  transductive: False
  node_encoder: True
  node_encoder_name: Atom
  node_encoder_bn: False
  edge_encoder: True
  edge_encoder_name: Bond
  edge_encoder_bn: False
posenc_RRWP:
  enable: True
  ksteps: 24
  add_identity: True
  add_node_attr: False
  add_inverse: False
train:
  mode: custom
  batch_size: 16
  eval_period: 1
  ckpt_period: 100
model:
  type: GritTransformer
  loss_fun: l1
  graph_pooling: mean
gt:
  layer_type: GritTransformer
  layers: 4
  n_heads: 8
  dim_hidden: 96
  dropout: 0.05
  attn_dropout: 0.2
  layer_norm: False
  batch_norm: True
  attn:
    clamp: 5.
    act: 'relu'
    full_attn: False
    sparsity: one_hop
    edge_enhance: True
    O_e: True
    norm_e: True
    signed_sqrt: True
gnn:
  head: default
  layers_pre_mp: 0
  layers_post_mp: 2
  dim_inner: 96
  batchnorm: True
  act: relu
  dropout: 0.0
optim:
  clip_grad_norm: True
  optimizer: adamW
  weight_decay: 0.0
  base_lr: 0.0003
  max_epoch: 200
  scheduler: cosine_with_warmup
  num_warmup_epochs: 10
"""


BASE_EXPECTED_CFG_VALUES: dict[tuple[str, ...], Any] = {
    ("metric_best",): "mae",
    ("metric_agg",): "argmin",
    ("dataset", "format"): "OGB",
    ("dataset", "name"): "peptides-structural",
    ("dataset", "task"): "graph",
    ("dataset", "task_type"): "regression",
    ("dataset", "transductive"): False,
    ("dataset", "node_encoder"): True,
    ("dataset", "node_encoder_name"): "Atom",
    ("dataset", "node_encoder_bn"): False,
    ("dataset", "edge_encoder"): True,
    ("dataset", "edge_encoder_name"): "Bond",
    ("dataset", "edge_encoder_bn"): False,
    ("posenc_RRWP", "enable"): True,
    ("posenc_RRWP", "ksteps"): 24,
    ("posenc_RRWP", "add_identity"): True,
    ("posenc_RRWP", "add_node_attr"): False,
    ("posenc_RRWP", "add_inverse"): False,
    ("train", "mode"): "custom",
    ("train", "batch_size"): 16,
    ("train", "eval_period"): 1,
    ("train", "ckpt_period"): 100,
    ("model", "type"): "GritTransformer",
    ("model", "loss_fun"): "l1",
    ("model", "graph_pooling"): "mean",
    ("gt", "layer_type"): "GritTransformer",
    ("gt", "layers"): 4,
    ("gt", "n_heads"): 8,
    ("gt", "dim_hidden"): 96,
    ("gt", "dropout"): 0.05,
    ("gt", "attn_dropout"): 0.2,
    ("gt", "layer_norm"): False,
    ("gt", "batch_norm"): True,
    ("gt", "attn", "clamp"): 5.0,
    ("gt", "attn", "act"): "relu",
    ("gt", "attn", "edge_enhance"): True,
    ("gt", "attn", "O_e"): True,
    ("gt", "attn", "norm_e"): True,
    ("gt", "attn", "signed_sqrt"): True,
    ("gnn", "head"): "default",
    ("gnn", "layers_pre_mp"): 0,
    ("gnn", "layers_post_mp"): 2,
    ("gnn", "dim_inner"): 96,
    ("gnn", "batchnorm"): True,
    ("gnn", "act"): "relu",
    ("gnn", "dropout"): 0.0,
    ("optim", "clip_grad_norm"): True,
    ("optim", "optimizer"): "adamW",
    ("optim", "weight_decay"): 0.0,
    ("optim", "base_lr"): 0.0003,
    ("optim", "max_epoch"): 200,
    ("optim", "scheduler"): "cosine_with_warmup",
    ("optim", "num_warmup_epochs"): 10,
}


def expected_cfg_values(*, onehop: bool) -> dict[tuple[str, ...], Any]:
    values = dict(BASE_EXPECTED_CFG_VALUES)
    values[("gt", "attn", "full_attn")] = not onehop
    if onehop:
        values[("gt", "attn", "sparsity")] = "one_hop"
    return values


def get_nested(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for part in path:
        current = current[part]
    return current


def semantic_cfg_equal(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - float(expected)) <= max(1e-12, 1e-9 * abs(float(expected)))
        except (TypeError, ValueError):
            return False
    return False


def make_validate_config(base: Any, *, cfg_path: str, onehop: bool):
    label = "1-hop global-RRWP GRIT Peptides-struct" if onehop else "official dense GRIT Peptides-struct"

    def validate(repo_dir: Path, allow_drift: bool) -> None:
        import yaml

        path = repo_dir / cfg_path
        if not path.exists():
            raise FileNotFoundError(f"GRIT Peptides-struct config not found: {path}")
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

        errors: list[str] = []
        for key_path, expected in expected_cfg_values(onehop=onehop).items():
            try:
                actual = get_nested(cfg, key_path)
            except Exception:
                errors.append(f"missing {'.'.join(key_path)}; expected {expected!r}")
                continue
            if not semantic_cfg_equal(actual, expected):
                errors.append(f"{'.'.join(key_path)} = {actual!r}; expected {expected!r}")

        if errors:
            msg = f"{label} config does not match the expected setup:\n"
            msg += "\n".join(f"  - {e}" for e in errors)
            if allow_drift:
                base.log("[config-warning] " + msg)
            else:
                raise RuntimeError(msg + "\nPass --allow-upstream-config-drift to run anyway.")

        base.log(f"[config] {label} config validated.")
        base.log(
            "[config] Key setup: OGB/peptides-structural, graph regression, RRWP-24, "
            "GritTransformer, 4 layers, hidden dim 96, 8 heads, batch 16, L1/MAE, "
            "200 epochs, cosine warmup, mean graph pooling."
        )
        if onehop:
            base.log("[control] 1-hop intervention: full_attn=False, sparsity=one_hop; global RRWP values are retained.")
        else:
            base.log("[official] Dense GRIT uses the unmodified official LiamMa/GRIT Peptides-struct config.")

    return validate


def make_build_training_command(base: Any, *, cfg_path: str):
    def build(args: argparse.Namespace, drive_dir: Path) -> list[str]:
        results_dir = drive_dir / "results"
        dataset_dir = drive_dir / "datasets"
        results_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        ckpt_best = not args.checkpoint_every_epoch
        ckpt_clean = False if (args.keep_all_checkpoints or args.checkpoint_every_epoch or args.guaranteed_checkpoints) else True
        cmd = [
            sys.executable,
            "-u",
            "main.py",
            "--cfg",
            cfg_path,
            "--repeat",
            str(args.repeat),
            "seed",
            str(args.seed),
            "out_dir",
            str(results_dir),
            "dataset.dir",
            str(dataset_dir),
            "name_tag",
            args.name_tag,
            "wandb.use",
            "True" if args.wandb else "False",
            "optim.max_epoch",
            str(args.max_epoch),
            "train.eval_period",
            "1",
            "train.enable_ckpt",
            "True",
            "train.ckpt_period",
            str(args.ckpt_period),
            "train.ckpt_best",
            "True" if ckpt_best else "False",
            "train.ckpt_clean",
            "True" if ckpt_clean else "False",
            "train.auto_resume",
            "True" if args.auto_resume else "False",
            "num_threads",
            str(args.num_threads),
        ]
        if args.accelerator:
            cmd.extend(["accelerator", args.accelerator])
        return cmd

    return build


def install_peptides_dependencies(base: Any) -> None:
    """Install Peptides-specific runtime dependencies missing from ZINC runs."""
    base.log("\n[deps] Installing Peptides-struct extras: RDKit for OGB SMILES featurization.")
    proc = base.run_cmd([sys.executable, "-m", "pip", "install", "rdkit"], check=False)
    if proc.returncode != 0:
        base.log("[deps-warning] `rdkit` wheel install failed; trying legacy `rdkit-pypi` package.")
        proc = base.run_cmd([sys.executable, "-m", "pip", "install", "rdkit-pypi"], check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "Peptides-struct requires RDKit for OGB SMILES featurization, but neither "
            "`pip install rdkit` nor `pip install rdkit-pypi` succeeded in this runtime."
        )

    base.run_cmd(
        [
            sys.executable,
            "-c",
            (
                "from rdkit import Chem\n"
                "try:\n"
                "    from ogb.utils import smiles2graph\n"
                "except ImportError:\n"
                "    from ogb.utils.mol import smiles2graph\n"
                "assert Chem.MolFromSmiles('CCO') is not None; "
                "assert smiles2graph('CCO')['num_nodes'] == 3; "
                "print('Peptides deps OK: RDKit + OGB smiles2graph')"
            ),
        ]
    )


def apply_peptides_dataset_compat_patch(base: Any, repo_dir: Path) -> None:
    """Patch official Peptides dataset files for modern OGB and Colab automation.

    This is a data-loader compatibility patch only. It does not alter the model,
    PE/RRWP construction, optimizer, loss, metric, or training schedule.
    """
    patched_any = False
    for rel in [
        Path("grit/loader/dataset/peptides_functional.py"),
        Path("grit/loader/dataset/peptides_structural.py"),
    ]:
        path = repo_dir / rel
        if not path.exists():
            raise FileNotFoundError(f"Official Peptides dataset file not found: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        original = text
        text = text.replace(
            "from ogb.utils import smiles2graph\n",
            (
                "try:\n"
                "    from ogb.utils import smiles2graph\n"
                "except ImportError:\n"
                "    from ogb.utils.mol import smiles2graph\n"
            ),
            1,
        )
        text = text.replace(
            "        if decide_download(self.url):\n",
            "        if True:  # Colab runner: non-interactive official dataset download.\n",
            1,
        )
        if text != original:
            path.write_text(text, encoding="utf-8")
            patched_any = True
            base.log(f"[dataset-compat] Patched Peptides loader for modern OGB/non-interactive Colab download: {path}")

    if not patched_any:
        base.log("[dataset-compat] Peptides loader compatibility patch already present.")


def apply_peptides_streaming_rrwp_patch(base: Any, repo_dir: Path) -> None:
    """Patch GRIT's PE preprocessing to avoid Colab system-RAM OOM.

    Official GRIT computes exactly the requested RRWP tensors, but its helper
    first builds a Python list containing every transformed graph, then collates
    the whole list. Peptides-struct with RRWP-24 is large enough that this list
    exceeds Colab high-RAM limits. This patch keeps the official transform and
    config unchanged, but collates transformed graphs in chunks and disables
    PyG's separated-graph cache afterwards so the epoch does not duplicate the
    already-large collated RRWP tensors in RAM.
    """
    path = repo_dir / "grit" / "transform" / "transforms.py"
    if not path.exists():
        raise FileNotFoundError(f"Official GRIT transforms.py not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = "def _grit_colab_streaming_get_no_cache("
    if marker in text:
        base.log("[memory] Streaming RRWP pre-transform patch already present.")
        return

    patch = r'''

# ---- Colab Peptides-struct memory patch ------------------------------------
# This overrides the earlier pre_transform_in_memory definition with a
# memory-stable implementation. It applies the same transform_func to every graph
# and stores the same collated InMemoryDataset tensors; it only avoids keeping a
# full Python list of transformed Data objects and avoids PyG's per-graph cache.

def _grit_colab_streaming_get_no_cache(self, idx):
    import copy as _copy
    from torch_geometric.data.separate import separate as _separate

    if self.len() == 1:
        return _copy.copy(self.data)
    return _separate(
        cls=self.data.__class__,
        batch=self.data,
        idx=idx,
        slice_dict=self.slices,
        decrement=False,
    )


def _grit_colab_collate_chunk(data_list):
    from torch_geometric.data.collate import collate as _pyg_collate

    data, slices, _ = _pyg_collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=False,
        add_batch=False,
    )
    return data, slices


def _grit_colab_merge_slice_dicts(slice_parts):
    merged = {}
    for key in slice_parts[0].keys():
        pieces = []
        offset = None
        for slices in slice_parts:
            current = slices[key]
            if offset is None:
                pieces.append(current)
            else:
                pieces.append(current[1:] + offset)
            offset = pieces[-1][-1]
        merged[key] = torch.cat(pieces, dim=0)
    return merged


def _grit_colab_merge_data_chunks(data_parts):
    out = data_parts[0].__class__()
    out.stores_as(data_parts[0])

    def _key_bytes(key):
        total = 0
        for data in data_parts:
            value = data[key]
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
        return total

    # Merge the largest tensors first, then immediately remove them from the
    # chunk objects. This keeps the peak closer to one full collated dataset
    # plus the largest single attribute, instead of two full datasets.
    for key in sorted(list(data_parts[0].keys), key=_key_bytes, reverse=True):
        values = [data[key] for data in data_parts]
        first = values[0]
        if torch.is_tensor(first):
            cat_dim = data_parts[0].__cat_dim__(key, first, data_parts[0]._store)
            if cat_dim is None or first.dim() == 0:
                cat_dim = 0
            out[key] = torch.cat(values, dim=cat_dim)
        elif isinstance(first, (int, float)):
            out[key] = values
        else:
            out[key] = sum(values, []) if isinstance(first, list) else values
        for data in data_parts:
            try:
                del data[key]
            except Exception:
                pass
    return out


def pre_transform_in_memory(dataset, transform_func, show_progress=False, cfg=dict(), posenc_mode=False):
    """Memory-stable replacement for GRIT's original helper.

    The transform itself is unchanged. For Peptides-struct RRWP-24 this avoids
    the peak RAM cost of holding every transformed graph object at once.
    """
    if transform_func is None:
        return dataset

    import gc as _gc
    import os as _os
    import types as _types

    chunk_size = int(_os.environ.get("GRIT_PE_STREAM_CHUNK_SIZE", "128"))
    chunk_size = max(1, chunk_size)
    data_parts = []
    slice_parts = []
    current = []

    iterator = tqdm(
        range(len(dataset)),
        disable=not show_progress,
        mininterval=10,
        miniters=max(1, len(dataset) // 20),
    )
    for i in iterator:
        transformed = transform_func(dataset.get(i))
        if transformed is not None:
            current.append(transformed)
        if len(current) >= chunk_size:
            data, slices = _grit_colab_collate_chunk(current)
            data_parts.append(data)
            slice_parts.append(slices)
            current.clear()
            _gc.collect()

    if current:
        data, slices = _grit_colab_collate_chunk(current)
        data_parts.append(data)
        slice_parts.append(slices)
        current.clear()

    if not data_parts:
        dataset._indices = None
        dataset._data_list = None
        return dataset

    dataset._indices = None
    dataset._data_list = None
    dataset.data = _grit_colab_merge_data_chunks(data_parts)
    dataset.slices = _grit_colab_merge_slice_dicts(slice_parts)
    data_parts.clear()
    slice_parts.clear()
    _gc.collect()

    # PyG InMemoryDataset caches every separated graph on first access. That is
    # normally helpful, but with Peptides RRWP it duplicates the huge collated
    # tensors over the first epoch. Disable only for this dataset instance.
    dataset.get = _types.MethodType(_grit_colab_streaming_get_no_cache, dataset)
    dataset._data_list = None
    return dataset
'''.rstrip()

    path.write_text(text.rstrip() + "\n" + patch + "\n", encoding="utf-8")
    base.log(
        "[memory] Patched GRIT pre_transform_in_memory for streaming RRWP collation "
        "(same RRWP values/config; lower Colab peak RAM)."
    )


def parse_args(base: Any, argv: Sequence[str] | None, *, onehop: bool) -> argparse.Namespace:
    description = (
        "Train parameter-matched 1-hop global-RRWP GRIT on Peptides-struct in Colab."
        if onehop
        else "Train official dense GRIT+RRWP on Peptides-struct in Colab."
    )
    default_drive = DEFAULT_ONEHOP_DRIVE_DIR if onehop else DEFAULT_DENSE_DRIVE_DIR
    default_repo = DEFAULT_ONEHOP_REPO_DIR if onehop else DEFAULT_DENSE_REPO_DIR
    default_tag = (
        "ColabDrive.1hop.GRITwRRWP.peptides_struct"
        if onehop
        else "ColabDrive.official.GRITwRRWP.peptides_struct"
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=description,
        epilog=textwrap.dedent(
            """
            Examples:
              main([])
              main(["--seed", "42"])
              main(["--skip-install", "--auto-resume"])
              main(["--max-epoch", "1", "--dry-run"])  # command-construction smoke test
            """
        ),
    )
    parser.add_argument("--drive-mount", type=Path, default=Path("/content/drive"))
    parser.add_argument("--drive-dir", type=Path, default=Path(default_drive))
    parser.add_argument("--repo-dir", type=Path, default=Path(default_repo))
    parser.add_argument("--repo-url", type=str, default=OFFICIAL_REPO)
    parser.add_argument("--branch", type=str, default="main")
    parser.add_argument("--commit", type=str, default=OFFICIAL_COMMIT)
    parser.add_argument("--expected-params", type=int, default=None, help="Optional parameter-count guard once known.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--name-tag", type=str, default=default_tag)
    parser.add_argument("--max-epoch", type=int, default=200, help="Default matches the official Peptides-struct config.")
    parser.add_argument("--ckpt-period", type=int, default=100, help="Official GraphGym checkpoint period; stable recovery checkpoints still save first/new-best/latest.")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--pyg-version", type=str, default="2.2.0")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--official-torch112", action="store_true")
    parser.add_argument("--force-fresh-repo", action="store_true")
    parser.add_argument("--allow-upstream-config-drift", action="store_true")
    parser.add_argument("--allow-param-count-drift", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--console-verbosity", choices=["compact", "standard", "full"], default="compact")
    parser.add_argument("--console-epoch-period", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="cuda:0")
    parser.add_argument("--auto-resume", action="store_true", default=True)
    parser.add_argument("--no-auto-resume", action="store_false", dest="auto_resume")
    parser.add_argument("--keep-all-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-every-epoch", action="store_true")
    parser.add_argument("--guaranteed-checkpoints", action="store_true", default=True)
    parser.add_argument("--no-guaranteed-checkpoints", action="store_false", dest="guaranteed_checkpoints")
    parser.add_argument("--recovery-ckpt-period", type=int, default=100)
    parser.add_argument("--rrwp-stream-chunk-size", type=int, default=128, help="Chunk size for Colab-safe Peptides RRWP preprocessing.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare/validate and print the train command without launching.")
    clean_argv = base._strip_colab_kernel_args(list(sys.argv[1:] if argv is None else argv))
    return parser.parse_args(clean_argv)


def configure_base(base: Any, *, onehop: bool) -> str:
    cfg_path = ONEHOP_OFFICIAL_CFG if onehop else DENSE_OFFICIAL_CFG
    base.OFFICIAL_REPO = OFFICIAL_REPO
    base.OFFICIAL_COMMIT = OFFICIAL_COMMIT
    base.OFFICIAL_CFG = cfg_path
    base.EXPECTED_ZINC_GRIT_RRWP_PARAMS = None
    base.EXPECTED_CFG_VALUES = expected_cfg_values(onehop=onehop)
    base.validate_official_config = make_validate_config(base, cfg_path=cfg_path, onehop=onehop)
    base.build_training_command = make_build_training_command(base, cfg_path=cfg_path)
    if onehop:
        base.DENSE_OFFICIAL_CFG = DENSE_OFFICIAL_CFG
        base.ONE_HOP_CFG_TEXT = PEPTIDES_STRUCT_ONEHOP_CFG_TEXT
    return cfg_path


def run_training(variant: str, argv: Sequence[str] | None = None) -> None:
    onehop = variant == "onehop"
    if variant not in {"dense", "onehop"}:
        raise ValueError("variant must be 'dense' or 'onehop'")

    if onehop:
        from experiments.zinc.training import grit_zinc_1hop_core as base
    else:
        from experiments.zinc.training import grit_zinc_core as base

    cfg_path = configure_base(base, onehop=onehop)
    args = parse_args(base, argv, onehop=onehop)

    base.mount_drive(args.drive_mount)
    args.drive_dir.mkdir(parents=True, exist_ok=True)

    compat_shim_dir = None
    if sys.version_info >= (3, 12):
        compat_shim_dir = base.write_py312_compat_shim(args.drive_dir)

    if not args.skip_install:
        base.install_dependencies(args)
        install_peptides_dependencies(base)
    else:
        base.log("[deps] Skipping dependency installation (--skip-install).")

    commit = base.clone_or_update_repo(args.repo_dir, args.repo_url, args.branch, args.commit or None, args.force_fresh_repo)
    # The official GRIT checkout is a generated dependency under /content (or
    # the user-provided repo-dir), while checkpoints/results live in Drive.
    # Reset it before applying our training-loop/control patches so rerunning a
    # Colab cell against an already-patched checkout is deterministic.
    base.run_cmd(["git", "reset", "--hard", commit], cwd=args.repo_dir)
    apply_peptides_dataset_compat_patch(base, args.repo_dir)
    apply_peptides_streaming_rrwp_patch(base, args.repo_dir)
    if onehop:
        base.apply_parameter_matched_onehop_patch(args.repo_dir, args.drive_dir)
        base.log(
            "[patch] Peptides-struct note: the reused patch helper may mention ZINC in a provenance filename/log label; "
            "the config validated below is the Peptides-struct 1-hop global-RRWP config."
        )
        base.verify_recovery_checkpoint_patch(args.repo_dir)
    elif args.guaranteed_checkpoints:
        base.apply_recovery_checkpoint_patch(args.repo_dir)
        base.verify_recovery_checkpoint_patch(args.repo_dir)

    base.install_grit_editable(args.repo_dir)
    base.validate_official_config(args.repo_dir, args.allow_upstream_config_drift)
    base.print_environment_summary(args.drive_dir, args.repo_dir, commit)

    cmd = base.build_training_command(args, args.drive_dir)
    if args.dry_run:
        base.log("[dry-run] Training command:")
        base.log(" ".join(map(str, cmd)))
        base.log(f"[dry-run] Config: {cfg_path}")
        base.log(f"[dry-run] Drive dir: {args.drive_dir}")
        return

    label = "grit_peptides_struct_1hop" if onehop else "grit_peptides_struct"
    wrapper_log = args.drive_dir / "wrapper_logs" / f"{label}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    train_env = base.env_with_py312_compat(compat_shim_dir)
    train_env["GRIT_PE_STREAM_CHUNK_SIZE"] = str(max(1, int(args.rrwp_stream_chunk_size)))
    if args.guaranteed_checkpoints:
        recovery_dir = args.drive_dir / "results" / "_recovery_checkpoints" / f"seed{args.seed}_{base.safe_path_fragment(args.name_tag)}"
        recovery_dir.mkdir(parents=True, exist_ok=True)
        train_env["GRIT_FORCE_RECOVERY_CKPT"] = "1"
        train_env["GRIT_RECOVERY_CKPT_PERIOD"] = str(max(0, int(args.recovery_ckpt_period)))
        train_env["GRIT_RECOVERY_CKPT_DIR"] = str(recovery_dir)
        train_env["GRIT_SAVE_FIRST_RECOVERY_CKPT"] = "1"
        train_env.pop("GRIT_FORCE_EPOCH_CKPT", None)
        base.log(
            "[checkpoint-guarantee] Enabled: official GRIT will save compatible recovery checkpoints "
            f"after every completed epoch. The first completed epoch after resume is also copied to first_after_resume.ckpt; "
            f"new best epochs update best.ckpt; epochs divisible by {max(0, int(args.recovery_ckpt_period))} get numbered snapshots."
        )
        base.log(f"[checkpoint-guarantee] Stable best checkpoint path: {recovery_dir / 'best.ckpt'}")
        base.log(f"[checkpoint-guarantee] Stable latest checkpoint path: {recovery_dir / 'latest.ckpt'}")
        base.log(f"[checkpoint-guarantee] First-resume checkpoint path: {recovery_dir / 'first_after_resume.ckpt'}")
    else:
        train_env.pop("GRIT_FORCE_RECOVERY_CKPT", None)
        train_env.pop("GRIT_RECOVERY_CKPT_PERIOD", None)
        train_env.pop("GRIT_RECOVERY_CKPT_DIR", None)
        train_env.pop("GRIT_SAVE_FIRST_RECOVERY_CKPT", None)
        train_env.pop("GRIT_FORCE_EPOCH_CKPT", None)
        base.log("[checkpoint-guarantee] Disabled: using only official GraphGym checkpoint policy.")

    rc = base.run_streaming_to_console_and_log(
        cmd,
        cwd=args.repo_dir,
        log_file=wrapper_log,
        env=train_env,
        expected_params=args.expected_params,
        allow_param_count_drift=args.allow_param_count_drift,
        console_verbosity=args.console_verbosity,
        console_epoch_period=args.console_epoch_period,
    )
    if rc != 0:
        raise SystemExit(rc)

    base.write_checkpoint_audit(args.drive_dir, wrapper_log, args.seed)
    base.log("\n[done] Training process completed successfully.")
    base.log(f"[done] Results/checkpoints root: {args.drive_dir / 'results'}")
    base.log(f"[done] Wrapper log: {wrapper_log}")
