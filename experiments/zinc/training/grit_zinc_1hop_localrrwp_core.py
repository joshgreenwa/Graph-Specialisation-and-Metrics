#!/usr/bin/env python3
"""Train a stricter 1-hop-local GRIT+RRWP ZINC control.

This runner reuses the official LiamMa/GRIT checkout and the existing
parameter-matched 1-hop patch, then adds one stricter scientific change:
RRWP values are truncated to local information only while keeping the official
21-dimensional RRWP encoder and therefore the same trainable parameter count.

The resulting control is intended to be truly local:

  * attention/message support is molecular bonds plus self-loops;
  * relative RRWP pair values exist only on self/bond pairs;
  * RRWP channels for walks longer than one step are zeroed before encoding;
  * the model width/depth/heads/schedule/loss/checkpoint policy remain matched
    to the official ZINC GRIT config.

Default Drive outputs are separate from the older 1-hop run:

    /content/drive/MyDrive/grit_zinc_1hop_localrrwp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Mapping, Sequence

try:
    from experiments.zinc.training import grit_zinc_1hop_core as base
except Exception:  # pragma: no cover - supports "file next to file" Colab usage.
    import grit_zinc_1hop_core as base  # type: ignore


OFFICIAL_CFG = "configs/GRIT/zinc-GRIT-RRWP-1hop-localrrwp.yaml"
EXPECTED_ZINC_GRIT_RRWP_PARAMS = base.EXPECTED_ZINC_GRIT_RRWP_PARAMS

LOCAL_RRWP_CFG_TEXT = """\
# Parameter-matched strictly-local 1-hop GRIT RRWP ZINC control.
#
# This variant keeps the official ZINC subset architecture, training schedule,
# RRWP dimensionality, and decoder settings. Compared with the dense official
# GRIT config, it makes two scientific restrictions:
#
#   1. attention/message passing is restricted to molecular bonds plus self;
#   2. RRWP values are truncated to identity + one-step random-walk channels.
#
# The 21-dimensional RRWP linear encoders are kept, but channels above the
# one-step horizon are zeroed before encoding. This preserves the official
# trainable parameter count while preventing multi-hop structural PE leakage.
out_dir: results
metric_best: mae
metric_agg: argmin
tensorboard_each_run: True
accelerator: "cuda:0"
mlflow:
  use: False
  project: Exp
  name: zinc-GRIT-RRWP-1hop-localrrwp
wandb:
  use: False
  project: ZINC
dataset:
  format: PyG-ZINC
  name: subset
  task: graph
  task_type: regression
  transductive: False
  node_encoder: True
  node_encoder_name: TypeDictNode
  node_encoder_num_types: 21
  node_encoder_bn: False
  edge_encoder: True
  edge_encoder_name: TypeDictEdge
  edge_encoder_num_types: 4
  edge_encoder_bn: False
posenc_RRWP:
  enable: True
  ksteps: 21
  add_identity: True
  add_node_attr: False
  add_inverse: False
  local_horizon: 1
train:
  mode: custom
  batch_size: 32
  eval_period: 1
  enable_ckpt: True
  ckpt_best: True
  ckpt_clean: True
model:
  type: GritTransformer
  loss_fun: l1
  edge_decoding: dot
  graph_pooling: add
gt:
  layer_type: GritTransformer
  layers: 10
  n_heads: 8
  dim_hidden: 64
  dropout: 0.0
  layer_norm: False
  batch_norm: True
  update_e: True
  attn_dropout: 0.2
  attn:
    clamp: 5.
    act: 'relu'
    full_attn: False
    sparsity: one_hop_local_rrwp
    edge_enhance: True
    O_e: True
    norm_e: True
    fwl: False
gnn:
  head: san_graph
  layers_pre_mp: 0
  layers_post_mp: 3
  dim_inner: 64
  batchnorm: True
  act: relu
  dropout: 0.0
  agg: mean
  normalize_adj: False
optim:
  clip_grad_norm: True
  optimizer: adamW
  weight_decay: 1e-5
  base_lr: 1e-3
  max_epoch: 2000
  num_warmup_epochs: 50
  scheduler: cosine_with_warmup
  min_lr: 1e-6
"""

EXPECTED_CFG_VALUES = dict(base.EXPECTED_CFG_VALUES)
EXPECTED_CFG_VALUES.update({
    ("gt", "attn", "sparsity"): "one_hop_local_rrwp",
    ("posenc_RRWP", "local_horizon"): 1,
})


def _replace_exact(path: Path, old: str, new: str, marker: str, label: str) -> bool:
    return base._replace_exact(path, old=old, new=new, marker=marker, label=label)


def _replace_exact_flexible(path: Path, old: str, new: str, marker: str, label: str) -> bool:
    try:
        return base._replace_exact(path, old=old, new=new, marker=marker, label=label)
    except RuntimeError:
        text = base._read_text_preserve_newlines(path)
        if marker in text:
            base.log(f"[patch] {label}: already present")
            return False
        if old not in text:
            raise
        base._write_text_preserve_newlines(path, text.replace(old, new, 1))
        base.log(f"[patch] {label}: applied")
        return True


def _insert_after(path: Path, anchor: str, insertion: str, marker: str, label: str) -> bool:
    return base._insert_after(path, anchor=anchor, insertion=insertion, marker=marker, label=label)


def apply_parameter_matched_onehop_localrrwp_patch(repo_dir: Path, drive_dir: Path) -> None:
    """Apply the stricter local-RRWP patch to an official GRIT clone."""
    base.apply_parameter_matched_onehop_patch(repo_dir, drive_dir)
    base.log("\n[patch] Applying strictly-local RRWP extension.")

    cfg_path = repo_dir / OFFICIAL_CFG
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    current_cfg = base._read_text_preserve_newlines(cfg_path) if cfg_path.exists() else ""
    if current_cfg != LOCAL_RRWP_CFG_TEXT:
        base._write_text_preserve_newlines(cfg_path, LOCAL_RRWP_CFG_TEXT)
        base.log(f"[patch] wrote exact local-RRWP ZINC config: {cfg_path}")
    else:
        base.log(f"[patch] exact local-RRWP ZINC config already present: {cfg_path}")

    posenc_config = repo_dir / "grit" / "config" / "posenc_config.py"
    _insert_after(
        posenc_config,
        anchor="    cfg.posenc_RRWP.spd = False\n",
        insertion="    cfg.posenc_RRWP.local_horizon = -1\n",
        marker="cfg.posenc_RRWP.local_horizon",
        label="RRWP local_horizon config default",
    )

    rrwp_transform = repo_dir / "grit" / "transform" / "rrwp.py"
    _replace_exact(
        rrwp_transform,
        old=(
            '                  add_identity=True,\n'
            '                  spd=False,\n'
            '                  **kwargs\n'
            '                  ):\n'
        ),
        new=(
            '                  add_identity=True,\n'
            '                  spd=False,\n'
            '                  local_horizon=None,\n'
            '                  local_edge_index_attr="rrwp_local_edge_index",\n'
            '                  **kwargs\n'
            '                  ):\n'
        ),
        marker="local_edge_index_attr=",
        label="RRWP local_horizon transform args",
    )
    _insert_after(
        rrwp_transform,
        anchor="    edge_index, edge_weight = data.edge_index, data.edge_weight\n",
        insertion=(
            "\n"
            "    if local_horizon is None:\n"
            "        try:\n"
            "            local_horizon = int(getattr(cfg.posenc_RRWP, 'local_horizon', -1))\n"
            "        except Exception:\n"
            "            local_horizon = -1\n"
            "    if local_horizon is not None:\n"
            "        local_horizon = int(local_horizon)\n"
            "    if local_horizon is not None and local_horizon >= 0 and local_edge_index_attr:\n"
            "        data[local_edge_index_attr] = edge_index.clone()\n"
        ),
        marker="data[local_edge_index_attr] = edge_index.clone()",
        label="preserve molecular edge support for local RRWP",
    )
    _insert_after(
        rrwp_transform,
        anchor="    pe = torch.stack(pe_list, dim=-1) # n x n x k\n",
        insertion=(
            "\n"
            "    if local_horizon is not None and local_horizon >= 0:\n"
            "        # With add_identity=True, channel 0 is I and channel k is P^k.\n"
            "        # For local_horizon=1, keep only I and one-step random-walk PE.\n"
            "        keep_channels = local_horizon + 1 if add_identity else local_horizon\n"
            "        keep_channels = max(0, min(int(keep_channels), pe.size(-1)))\n"
            "        if keep_channels < pe.size(-1):\n"
            "            pe[..., keep_channels:] = 0\n"
        ),
        marker="keep_channels = local_horizon + 1 if add_identity else local_horizon",
        label="truncate RRWP channels to local horizon",
    )

    posenc_stats = repo_dir / "grit" / "transform" / "posenc_stats.py"
    _insert_after(
        posenc_stats,
        anchor="                            spd=param.spd, # by default False\n",
        insertion="                            local_horizon=param.get('local_horizon', -1),\n",
        marker="local_horizon=param.get('local_horizon'",
        label="pass RRWP local_horizon to transform",
    )

    grit_model = repo_dir / "grit" / "network" / "grit_model.py"
    _replace_exact_flexible(
        grit_model,
        old=(
            '            elif attn_sparsity == "one_hop":\n'
            '                self.rrwp_rel_encoder = register.edge_encoder_dict["masked_rrwp_linear"] \\\n'
            '                    (rel_pe_dim, cfg.gnn.dim_edge,\n'
            '                     add_node_attr_as_self_loop=False,\n'
            '                     fill_value=0.,\n'
            '                     mask_index_name="edge_index",\n'
            '                     )\n'
        ),
        new=(
            '            elif attn_sparsity in ("one_hop", "one_hop_local_rrwp"):\n'
            '                mask_index_name = "rrwp_local_edge_index" if attn_sparsity == "one_hop_local_rrwp" else "edge_index"\n'
            '                self.rrwp_rel_encoder = register.edge_encoder_dict["masked_rrwp_linear"] \\\n'
            '                    (rel_pe_dim, cfg.gnn.dim_edge,\n'
            '                     add_node_attr_as_self_loop=False,\n'
            '                     fill_value=0.,\n'
            '                     mask_index_name=mask_index_name,\n'
            '                     )\n'
        ),
        marker='attn_sparsity in ("one_hop", "one_hop_local_rrwp")',
        label="GritTransformer local-RRWP mask selector",
    )
    _replace_exact_flexible(
        grit_model,
        old="                    \"expected 'full' or 'one_hop'.\"\n",
        new="                    \"expected 'full', 'one_hop', or 'one_hop_local_rrwp'.\"\n",
        marker="one_hop_local_rrwp'.",
        label="GritTransformer local-RRWP error text",
    )

    patch_note = drive_dir / "patches" / "zinc_grit_rrwp_1hop_localrrwp_patch.txt"
    patch_note.parent.mkdir(parents=True, exist_ok=True)
    patch_note.write_text(
        "\n".join([
            "Parameter-matched strictly-local 1-hop GRIT RRWP ZINC control patch",
            f"official_repo: {base.OFFICIAL_REPO}",
            f"official_commit: {base.OFFICIAL_COMMIT}",
            f"dense_reference_config: {base.DENSE_OFFICIAL_CFG}",
            f"local_rrwp_config: {OFFICIAL_CFG}",
            f"parameter_count_guard: {EXPECTED_ZINC_GRIT_RRWP_PARAMS}",
            "scientific_change: gt.attn.sparsity=one_hop_local_rrwp and posenc_RRWP.local_horizon=1",
            "support: molecular bonds plus self; RRWP channels above one step zeroed",
            "parameter_matching: RRWP encoder dimensionality remains ksteps=21",
        ]) + "\n",
        encoding="utf-8",
    )
    base.log(f"[patch] wrote patch provenance note: {patch_note}")


def validate_official_config(repo_dir: Path, allow_drift: bool) -> None:
    import yaml

    cfg_path = repo_dir / OFFICIAL_CFG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Local-RRWP config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    errors: List[str] = []
    for path, expected in EXPECTED_CFG_VALUES.items():
        try:
            actual = base.get_nested(cfg, path)
        except Exception:
            errors.append(f"missing {'.'.join(path)}; expected {expected!r}")
            continue
        if not base._semantic_cfg_equal(actual, expected):
            errors.append(f"{'.'.join(path)} = {actual!r}; expected {expected!r}")

    if errors:
        msg = "Local-RRWP ZINC config does not match the expected setup:\n"
        msg += "\n".join(f"  - {e}" for e in errors)
        if allow_drift:
            base.log("[config-warning] " + msg)
        else:
            raise RuntimeError(msg + "\nPass --allow-upstream-config-drift to run anyway.")

    base.log("[config] Parameter-matched strict 1-hop local-RRWP GRIT ZINC config validated.")
    base.log("[config] Key setup: PyG-ZINC/subset, graph regression, RRWP ksteps=21 retained, RRWP local_horizon=1, molecular-bond attention support, 10 layers, 64 hidden dim, 8 heads, batch 32, L1/MAE, 2000 epochs.")
    base.log("[paper-check] Dense-reference GRIT ZINC settings are preserved; only attention support and RRWP values are restricted.")
    base.log(f"[paper-check] Expected matched parameter count: {EXPECTED_ZINC_GRIT_RRWP_PARAMS}")


def build_training_command(args: argparse.Namespace, drive_dir: Path) -> List[str]:
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
        OFFICIAL_CFG,
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
        "2000",
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train parameter-matched strict 1-hop-local GRIT+RRWP on ZINC, saving checkpoints to Drive.",
        epilog=textwrap.dedent(
            """
            Examples:
              from grit_zinc_1hop_localrrwp_core import main
              main([])
              main(["--skip-install", "--force-fresh-repo"])
            """
        ),
    )
    p.add_argument("--drive-mount", type=Path, default=Path("/content/drive"))
    p.add_argument("--drive-dir", type=Path, default=Path("/content/drive/MyDrive/grit_zinc_1hop_localrrwp"))
    p.add_argument("--repo-dir", type=Path, default=Path("/content/GRIT_1hop_localrrwp"))
    p.add_argument("--repo-url", type=str, default=base.OFFICIAL_REPO)
    p.add_argument("--branch", type=str, default="main")
    p.add_argument("--commit", type=str, default=base.OFFICIAL_COMMIT)
    p.add_argument("--expected-params", type=int, default=EXPECTED_ZINC_GRIT_RRWP_PARAMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--name-tag", type=str, default="ColabDrive.1hopLocalRRWP.GRITwRRWP")
    p.add_argument("--ckpt-period", type=int, default=1)
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--pyg-version", type=str, default="2.2.0")
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--official-torch112", action="store_true")
    p.add_argument("--force-fresh-repo", action="store_true")
    p.add_argument("--allow-upstream-config-drift", action="store_true")
    p.add_argument("--allow-param-count-drift", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--console-verbosity", choices=["compact", "standard", "full"], default="compact")
    p.add_argument("--console-epoch-period", type=int, default=1)
    p.add_argument("--accelerator", type=str, default="cuda:0")
    p.add_argument("--auto-resume", action="store_true", default=True)
    p.add_argument("--no-auto-resume", action="store_false", dest="auto_resume")
    p.add_argument("--keep-all-checkpoints", action="store_true")
    p.add_argument("--checkpoint-every-epoch", action="store_true")
    p.add_argument("--guaranteed-checkpoints", action="store_true", default=True)
    p.add_argument("--no-guaranteed-checkpoints", action="store_false", dest="guaranteed_checkpoints")
    p.add_argument("--recovery-ckpt-period", type=int, default=100)
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = base._strip_colab_kernel_args(argv)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    base.mount_drive(args.drive_mount)
    args.drive_dir.mkdir(parents=True, exist_ok=True)

    compat_shim_dir = None
    if sys.version_info >= (3, 12):
        compat_shim_dir = base.write_py312_compat_shim(args.drive_dir)

    if not args.skip_install:
        base.install_dependencies(args)
    else:
        base.log("[deps] Skipping dependency installation (--skip-install).")

    commit = base.clone_or_update_repo(args.repo_dir, args.repo_url, args.branch, args.commit or None, args.force_fresh_repo)
    apply_parameter_matched_onehop_localrrwp_patch(args.repo_dir, args.drive_dir)
    base.verify_recovery_checkpoint_patch(args.repo_dir)
    base.install_grit_editable(args.repo_dir)
    validate_official_config(args.repo_dir, args.allow_upstream_config_drift)
    base.print_environment_summary(args.drive_dir, args.repo_dir, commit)

    cmd = build_training_command(args, args.drive_dir)
    wrapper_log = args.drive_dir / "wrapper_logs" / f"grit_zinc_1hop_localrrwp_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    train_env = base.env_with_py312_compat(compat_shim_dir)
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
            f"after the first completed epoch, on each new best, and at official epochs divisible by {max(0, int(args.recovery_ckpt_period))}."
        )
        base.log(f"[checkpoint-guarantee] Stable best checkpoint path: {recovery_dir / 'best.ckpt'}")
        base.log(f"[checkpoint-guarantee] Stable latest checkpoint path: {recovery_dir / 'latest.ckpt'}")
    else:
        for key in (
            "GRIT_FORCE_RECOVERY_CKPT",
            "GRIT_RECOVERY_CKPT_PERIOD",
            "GRIT_RECOVERY_CKPT_DIR",
            "GRIT_SAVE_FIRST_RECOVERY_CKPT",
            "GRIT_FORCE_EPOCH_CKPT",
        ):
            train_env.pop(key, None)
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

    base.log("\n[done] Training process completed successfully.")
    base.write_checkpoint_audit(args.drive_dir, wrapper_log, args.seed)
    base.log(f"[done] Results/checkpoints root: {args.drive_dir / 'results'}")
    base.log(f"[done] Wrapper log: {wrapper_log}")


if __name__ == "__main__":
    main()
