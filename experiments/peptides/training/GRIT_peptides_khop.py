#!/usr/bin/env python3
"""Unified dense/k-hop/local-RRWP/VNode GRIT runner for both Peptides tasks.

This file is designed to be copied into a fresh Google Colab runtime and run as
one standalone script. It bootstraps the project repo from GitHub, mounts Drive,
installs the GRIT runtime, clones the pinned official LiamMa/GRIT repo, preserves
each task's official model/training settings, and independently controls:

* dense or exact <=k-hop attention (self included);
* an optional communication-only learned global VNode;
* global RRWP or a local horizon that zeroes later channels while retaining the
  official RRWP encoder width and parameter count.

Peptides-func uses RRWP-17 (k <= 16); Peptides-struct uses RRWP-24 (k <= 23).
For example, ``--attention khop --hops 1 --rrwp-horizon 1`` retains only I and
P^1, while ``--attention khop --hops 1 --global-vnode`` retains global RRWP and
adds one 96-dimensional VNode per graph.

Colab usage after uploading this file:

    from GRIT_peptides_khop import main
    main(["--task", "func", "--attention", "khop", "--hops", "1",
          "--rrwp-horizon", "1"])

Use ``--task struct`` for Peptides-struct. Add ``--global-vnode`` for the VNode
variant. Later runs in the same runtime may add ``--skip-install``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote


PROJECT_REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
PROJECT_BRANCH = "codex/cfim-grit-experiments"
PROJECT_REPO_DIR = Path("/content/Graph-Specialisation-and-Metrics")
SECRET_NAME = "dissertation_key"

OFFICIAL_GRIT_REPO = "https://github.com/LiamMa/GRIT.git"
OFFICIAL_GRIT_COMMIT = "6c988ea600a606fbb49a2246c64a2d37396b3ab5"

TASKS: dict[str, dict[str, Any]] = {
    "func": {
        "label": "peptides_func",
        "dataset_name": "peptides-functional",
        "official_cfg": "configs/GRIT/peptides-func-GRIT-RRWP.yaml",
        "generated_cfg": "configs/GRIT/peptides-func-GRIT-RRWP-custom.yaml",
        "metric_best": "ap",
        # The pinned official Peptides-func config sets metric_best: ap and
        # leaves metric_agg to GraphGym defaults; do not require it here.
        "metric_agg": None,
        "task_type": "classification_multilabel",
        "loss_fun": "cross_entropy",
        "rrwp_ksteps": 17,
        "layers": 4,
        "heads": 4,
        "hidden": 96,
        "dropout": 0.0,
        "attn_dropout": 0.5,
        "layers_post_mp": 1,
        "batch_size": 16,
        "max_epoch": 200,
        "warmup": 10,
        "base_lr": 0.0003,
        "graph_pooling": "mean",
    },
    "struct": {
        "label": "peptides_struct",
        "dataset_name": "peptides-structural",
        "official_cfg": "configs/GRIT/peptides-struct-GRIT-RRWP.yaml",
        "generated_cfg": "configs/GRIT/peptides-struct-GRIT-RRWP-custom.yaml",
        "metric_best": "mae",
        "metric_agg": "argmin",
        "task_type": "regression",
        "loss_fun": "l1",
        "rrwp_ksteps": 24,
        "layers": 4,
        "heads": 8,
        "hidden": 96,
        "dropout": 0.05,
        "attn_dropout": 0.2,
        "layers_post_mp": 2,
        "batch_size": 16,
        "max_epoch": 200,
        "warmup": 10,
        "base_lr": 0.0003,
        "graph_pooling": "mean",
    },
}


class CommandError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, flush=True)


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    safe: str | None = None,
) -> subprocess.CompletedProcess:
    printable = safe or " ".join(map(str, cmd))
    log(f"[cmd] {printable}")
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise CommandError(f"command failed with exit code {proc.returncode}: {printable}")
    return proc


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def get_secret(name: str) -> str | None:
    try:
        from google.colab import userdata  # type: ignore

        value = userdata.get(name)
        if value:
            return str(value).strip()
    except Exception:
        pass
    value = os.environ.get(name)
    return value.strip() if value else None


def auth_url(repo_url: str, token: str | None) -> str:
    if not token or not repo_url.startswith("https://github.com/"):
        return repo_url
    suffix = repo_url.removeprefix("https://github.com/")
    return f"https://x-access-token:{quote(token, safe='')}@github.com/{suffix}"


def premount_drive(mount_point: Path) -> None:
    if not in_colab():
        return
    try:
        from google.colab import drive  # type: ignore

        log(f"[drive] Mounting Google Drive at {mount_point} before repository setup ...")
        drive.mount(str(mount_point), force_remount=False)
    except Exception as exc:
        raise RuntimeError(f"failed to mount Google Drive at {mount_point}") from exc


def strip_colab_kernel_args(argv: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "-f" and i + 1 < len(argv) and "kernel-" in argv[i + 1] and argv[i + 1].endswith(".json"):
            log(f"[args] ignoring notebook launcher arguments: {arg} {argv[i + 1]}")
            i += 2
            continue
        if arg.startswith("-f=") and "kernel-" in arg and arg.endswith(".json"):
            log(f"[args] ignoring notebook launcher argument: {arg}")
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    return cleaned


def bootstrap_project_repo(repo_url: str, branch: str, repo_dir: Path, secret_name: str, *, skip_git: bool) -> Path:
    local_root = Path(__file__).resolve().parents[3] if "__file__" in globals() else None
    if local_root and (local_root / "src" / "graph_specialisation_metrics").exists():
        for path in [str(local_root), str(local_root / "src")]:
            if path not in sys.path:
                sys.path.insert(0, path)
        log(f"[project] using local project repo: {local_root}")
        return local_root

    if skip_git:
        for path in [str(repo_dir), str(repo_dir / "src")]:
            if path not in sys.path:
                sys.path.insert(0, path)
        log(f"[project] using existing project repo without git refresh: {repo_dir}")
        return repo_dir

    if not in_colab():
        raise RuntimeError("project repo bootstrap needs Colab, --skip-project-git, or in-repo execution")

    token = get_secret(secret_name)
    if not token:
        raise RuntimeError(f"missing Colab secret {secret_name!r}")
    log(f"[auth] secret {secret_name!r} loaded.")
    authed = auth_url(repo_url, token)

    if (repo_dir / ".git").exists():
        run_cmd(["git", "-C", str(repo_dir), "remote", "set-url", "origin", authed], safe=f"git -C {repo_dir} remote set-url origin <token-authenticated-url>")
        run_cmd(["git", "-C", str(repo_dir), "fetch", "origin", branch])
        run_cmd(["git", "-C", str(repo_dir), "checkout", branch])
        run_cmd(["git", "-C", str(repo_dir), "reset", "--hard", f"origin/{branch}"])
        run_cmd(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])
    else:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        run_cmd(["git", "clone", "--branch", branch, "--single-branch", authed, str(repo_dir)], safe=f"git clone --branch {branch} <token-authenticated-url> {repo_dir}")
        run_cmd(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])

    for path in [str(repo_dir), str(repo_dir / "src")]:
        if path not in sys.path:
            sys.path.insert(0, path)
    return repo_dir


def split_bootstrap_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project-repo-url", default=PROJECT_REPO_URL)
    parser.add_argument("--project-branch", default=PROJECT_BRANCH)
    parser.add_argument("--project-repo-dir", type=Path, default=PROJECT_REPO_DIR)
    parser.add_argument("--secret-name", default=SECRET_NAME)
    parser.add_argument("--skip-project-git", action="store_true")
    parser.add_argument("--drive-mount", type=Path, default=Path("/content/drive"))
    raw = strip_colab_kernel_args(list(sys.argv[1:] if argv is None else argv))
    args, rest = parser.parse_known_args(raw)
    return args, rest


def variant_slug(args: argparse.Namespace) -> str:
    slug = "dense" if args.attention == "dense" else f"{args.hops}hop"
    if args.global_vnode:
        slug += "_vnode"
    if args.rrwp_horizon >= 0:
        slug += f"_localrrwp_h{args.rrwp_horizon}"
    return slug


def default_drive_dir(args: argparse.Namespace) -> Path:
    return Path(f"/content/drive/MyDrive/grit_{TASKS[args.task]['label']}_{variant_slug(args)}")


def default_grit_repo_dir(args: argparse.Namespace) -> Path:
    return Path(f"/content/GRIT_{TASKS[args.task]['label']}_{variant_slug(args)}")


def default_name_tag(args: argparse.Namespace) -> str:
    parts = ["ColabDrive", variant_slug(args), "GRITwRRWP", TASKS[args.task]["label"], f"s{args.seed}"]
    return ".".join(parts)


def nested_get(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        current = current[key]
    return current


def nested_set(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = payload
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def semantically_equal(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - float(expected)) <= max(1e-12, abs(float(expected)) * 1e-8)
        except Exception:
            return False
    return False


def _replace_exact(path: Path, old: str, new: str, marker: str, label: str) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    if marker in text:
        log(f"[patch] {label}: already present")
        return False
    if old not in text:
        raise RuntimeError(f"Could not apply {label} to pinned GRIT source: {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    log(f"[patch] {label}: applied")
    return True


def _insert_after(path: Path, anchor: str, insertion: str, marker: str, label: str) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    if marker in text:
        log(f"[patch] {label}: already present")
        return False
    if anchor not in text:
        raise RuntimeError(f"Could not locate insertion point for {label}: {path}")
    path.write_text(text.replace(anchor, anchor + insertion, 1), encoding="utf-8")
    log(f"[patch] {label}: applied")
    return True


def apply_attention_rrwp_vnode_patch(repo_dir: Path) -> None:
    """Add exact k-hop support, local RRWP, and a communication-only VNode."""
    gt_config = repo_dir / "grit/config/gt_config.py"
    _replace_exact(
        gt_config,
        "    cfg.gt.attn.full_attn = True\n    cfg.gt.attn.norm_e = True\n",
        (
            "    cfg.gt.attn.full_attn = True\n"
            "    cfg.gt.attn.sparsity = \"full\"\n"
            "    cfg.gt.attn.hops = 1\n"
            "    cfg.gt.attn.global_vnode = False\n"
            "    cfg.gt.attn.norm_e = True\n"
        ),
        "cfg.gt.attn.global_vnode = False",
        "attention defaults",
    )

    posenc_config = repo_dir / "grit/config/posenc_config.py"
    _insert_after(
        posenc_config,
        "    cfg.posenc_RRWP.spd = False\n",
        "    cfg.posenc_RRWP.local_horizon = -1\n",
        "cfg.posenc_RRWP.local_horizon",
        "RRWP local-horizon default",
    )

    rrwp_transform = repo_dir / "grit/transform/rrwp.py"
    _replace_exact(
        rrwp_transform,
        (
            "                  add_identity=True,\n"
            "                  spd=False,\n"
            "                  **kwargs\n"
            "                  ):\n"
        ),
        (
            "                  add_identity=True,\n"
            "                  spd=False,\n"
            "                  local_horizon=-1,\n"
            "                  support_horizon=-1,\n"
            '                  support_index_attr="rrwp_attention_edge_index",\n'
            "                  **kwargs\n"
            "                  ):\n"
        ),
        'support_index_attr="rrwp_attention_edge_index"',
        "RRWP local/support arguments",
    )
    _insert_after(
        rrwp_transform,
        "    pe = torch.stack(pe_list, dim=-1) # n x n x k\n",
        (
            "\n"
            "    local_horizon = int(local_horizon)\n"
            "    support_horizon = int(support_horizon)\n"
            "    if support_horizon >= 1 and support_index_attr:\n"
            "        # Compute architectural support before optional PE truncation.\n"
            "        support_channels = support_horizon + 1 if add_identity else support_horizon\n"
            "        if support_channels > pe.size(-1):\n"
            "            raise ValueError(\n"
            "                f'support_horizon={support_horizon} requires {support_channels} channels; '\n"
            "                f'found {pe.size(-1)}.'\n"
            "            )\n"
            "        support = pe[..., :support_channels].abs().sum(dim=-1) > 0\n"
            "        support_row, support_col = support.nonzero(as_tuple=True)\n"
            "        data[support_index_attr] = torch.stack([support_col, support_row], dim=0)\n"
            "\n"
            "    if local_horizon >= 0:\n"
            "        # Channel 0 is I; channel k is P^k when add_identity=True.\n"
            "        keep_channels = local_horizon + 1 if add_identity else local_horizon\n"
            "        keep_channels = max(0, min(int(keep_channels), pe.size(-1)))\n"
            "        if keep_channels < pe.size(-1):\n"
            "            pe[..., keep_channels:] = 0\n"
        ),
        "data[support_index_attr] = torch.stack([support_col, support_row]",
        "frozen k-hop support and local RRWP",
    )

    posenc_stats = repo_dir / "grit/transform/posenc_stats.py"
    _insert_after(
        posenc_stats,
        "                            spd=param.spd, # by default False\n",
        (
            "                            local_horizon=param.get('local_horizon', -1),\n"
            "                            support_horizon=(cfg.gt.attn.hops if cfg.gt.attn.get('sparsity', 'full') == 'k_hop' else -1),\n"
        ),
        "support_horizon=(cfg.gt.attn.hops",
        "RRWP/attention horizon forwarding",
    )

    rrwp_encoder = repo_dir / "grit/encoder/rrwp_encoder.py"
    _replace_exact(
        rrwp_encoder,
        '                 mask_index_name="edge_index",\n                 ):\n',
        '                 mask_index_name="edge_index",\n                 max_hops=None,\n                 ):\n',
        "max_hops=None",
        "masked RRWP max-hops argument",
    )
    _replace_exact(
        rrwp_encoder,
        "        torch.nn.init.xavier_uniform_(self.fc.weight)\n        self.fill_value = 0.\n",
        (
            "        torch.nn.init.xavier_uniform_(self.fc.weight)\n"
            "        self.pad_to_full_graph = False\n"
            "        self.max_hops = None if max_hops is None else int(max_hops)\n"
            "        self.fill_value = 0.\n"
        ),
        "self.max_hops = None if max_hops is None else int(max_hops)",
        "masked RRWP k-hop state",
    )
    _replace_exact(
        rrwp_encoder,
        (
            "        rrwp_idx = batch.rrwp_index\n"
            "        rrwp_val = batch.rrwp_val\n"
            "        edge_index = batch.edge_index\n"
            "        edge_attr = batch.edge_attr\n"
            "        rrwp_val = self.fc(rrwp_val)\n"
            "        mask_index = batch.get(self.mask_index_name, None)\n"
            "        num_nodes = batch.num_nodes\n"
        ),
        (
            "        rrwp_idx = batch.rrwp_index\n"
            "        raw_rrwp_val = batch.rrwp_val\n"
            "        edge_index = batch.edge_index\n"
            "        edge_attr = batch.edge_attr\n"
            "        if self.max_hops is None or self.max_hops == 1:\n"
            "            mask_index = batch.get(self.mask_index_name, None)\n"
            "        else:\n"
            '            mask_index = batch.get("rrwp_attention_edge_index", None)\n'
            "            if mask_index is None:\n"
            "                # Compatibility fallback for pre-patch transformed data.\n"
            "                needed_channels = self.max_hops + 1\n"
            "                if needed_channels > raw_rrwp_val.size(1):\n"
            "                    raise ValueError(\n"
            "                        f'max_hops={self.max_hops} requires {needed_channels} channels; '\n"
            "                        f'found {raw_rrwp_val.size(1)}.'\n"
            "                    )\n"
            "                reachable = raw_rrwp_val[:, :needed_channels].abs().sum(dim=-1) > 0\n"
            "                mask_index = rrwp_idx[:, reachable]\n"
            "        rrwp_val = self.fc(raw_rrwp_val)\n"
            "        num_nodes = batch.num_nodes\n"
        ),
        'mask_index = batch.get("rrwp_attention_edge_index", None)',
        "exact <=k-hop attention support",
    )

    grit_model = repo_dir / "grit/network/grit_model.py"
    _replace_exact(
        grit_model,
        (
            "        if cfg.posenc_RRWP.enable:\n"
            '            self.rrwp_abs_encoder = register.node_encoder_dict["rrwp_linear"]\\\n'
            "                (cfg.posenc_RRWP.ksteps, cfg.gnn.dim_inner)\n"
            "            rel_pe_dim = cfg.posenc_RRWP.ksteps\n"
            '            self.rrwp_rel_encoder = register.edge_encoder_dict["rrwp_linear"] \\\n'
            "                (rel_pe_dim, cfg.gnn.dim_edge,\n"
            "                 pad_to_full_graph=cfg.gt.attn.full_attn,\n"
            "                 add_node_attr_as_self_loop=False,\n"
            "                 fill_value=0.\n"
            "                 )\n"
        ),
        (
            "        if cfg.posenc_RRWP.enable:\n"
            '            self.rrwp_abs_encoder = register.node_encoder_dict["rrwp_linear"]\\\n'
            "                (cfg.posenc_RRWP.ksteps, cfg.gnn.dim_inner)\n"
            "            rel_pe_dim = cfg.posenc_RRWP.ksteps\n"
            '            attn_sparsity = cfg.gt.attn.get("sparsity", "full")\n'
            '            if attn_sparsity == "full":\n'
            '                self.rrwp_rel_encoder = register.edge_encoder_dict["rrwp_linear"] \\\n'
            "                    (rel_pe_dim, cfg.gnn.dim_edge,\n"
            "                     pad_to_full_graph=True,\n"
            "                     add_node_attr_as_self_loop=False,\n"
            "                     fill_value=0.\n"
            "                     )\n"
            '            elif attn_sparsity == "k_hop":\n'
            '                self.rrwp_rel_encoder = register.edge_encoder_dict["masked_rrwp_linear"] \\\n'
            "                    (rel_pe_dim, cfg.gnn.dim_edge,\n"
            "                     add_node_attr_as_self_loop=False,\n"
            "                     fill_value=0.,\n"
            '                     mask_index_name="edge_index",\n'
            "                     max_hops=cfg.gt.attn.hops,\n"
            "                     )\n"
            "            else:\n"
            "                raise ValueError(f'Unsupported attention sparsity: {attn_sparsity!r}')\n"
        ),
        "max_hops=cfg.gt.attn.hops",
        "GRIT dense/k-hop RRWP switch",
    )

    _replace_exact(
        grit_model,
        "\n\n@register_network('GritTransformer')\nclass GritTransformer(torch.nn.Module):\n",
        (
            "\n\nclass GlobalVNode(torch.nn.Module):\n"
            '    """One learned graph token connected both ways to every real node."""\n'
            "\n"
            "    def __init__(self, dim):\n"
            "        super().__init__()\n"
            "        self.embedding = torch.nn.Parameter(torch.zeros(1, dim))\n"
            "\n"
            "    def forward(self, batch):\n"
            "        graph_id = batch.batch\n"
            "        num_real = batch.x.size(0)\n"
            "        num_graphs = int(graph_id.max().item()) + 1\n"
            "        device = batch.x.device\n"
            "        vnode_id = torch.arange(num_graphs, device=device) + num_real\n"
            "        real_id = torch.arange(num_real, device=device)\n"
            "        vnode_for_real = vnode_id[graph_id]\n"
            "        virtual_edges = torch.cat([\n"
            "            torch.stack([real_id, vnode_for_real]),\n"
            "            torch.stack([vnode_for_real, real_id]),\n"
            "            torch.stack([vnode_id, vnode_id]),\n"
            "        ], dim=1)\n"
            "        virtual_attr = batch.edge_attr.new_zeros(\n"
            "            virtual_edges.size(1), batch.edge_attr.size(1)\n"
            "        )\n"
            "        batch.x = torch.cat([\n"
            "            batch.x, self.embedding.to(dtype=batch.x.dtype).expand(num_graphs, -1)\n"
            "        ], dim=0)\n"
            "        batch.batch = torch.cat([\n"
            "            graph_id, torch.arange(num_graphs, device=device, dtype=graph_id.dtype)\n"
            "        ], dim=0)\n"
            "        batch.edge_index = torch.cat([batch.edge_index, virtual_edges], dim=1)\n"
            "        batch.edge_attr = torch.cat([batch.edge_attr, virtual_attr], dim=0)\n"
            "        batch.real_node_mask = torch.arange(num_real + num_graphs, device=device) < num_real\n"
            "        counts = torch.bincount(graph_id, minlength=num_graphs)\n"
            '        if batch.get("log_deg", None) is not None:\n'
            "            vnode_log_deg = torch.log(counts.to(batch.log_deg.dtype) + 1)\n"
            "            batch.log_deg = torch.cat([batch.log_deg.view(-1), vnode_log_deg])\n"
            '        if batch.get("deg", None) is not None:\n'
            "            batch.deg = torch.cat([batch.deg.view(-1), counts.to(batch.deg.dtype)])\n"
            "        return batch\n"
            "\n\n@register_network('GritTransformer')\n"
            "class GritTransformer(torch.nn.Module):\n"
        ),
        "class GlobalVNode(torch.nn.Module):",
        "global VNode module",
    )
    _replace_exact(
        grit_model,
        (
            "        if cfg.gnn.layers_pre_mp > 0:\n"
            "            self.pre_mp = GNNPreMP(\n"
            "                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)\n"
            "            dim_in = cfg.gnn.dim_inner\n"
            "\n"
            "        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \\\n"
        ),
        (
            "        if cfg.gnn.layers_pre_mp > 0:\n"
            "            self.pre_mp = GNNPreMP(\n"
            "                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)\n"
            "            dim_in = cfg.gnn.dim_inner\n"
            "\n"
            "        self.global_vnode = (\n"
            "            GlobalVNode(cfg.gnn.dim_inner)\n"
            '            if cfg.gt.attn.get("global_vnode", False) else None\n'
            "        )\n"
            "\n"
            "        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \\\n"
        ),
        "GlobalVNode(cfg.gnn.dim_inner)",
        "optional VNode construction",
    )
    _replace_exact(
        grit_model,
        (
            "        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)\n"
            "\n"
            "    def forward(self, batch):\n"
            "        for module in self.children():\n"
            "            batch = module(batch)\n"
            "\n"
            "        return batch\n"
        ),
        (
            "        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)\n"
            "\n"
            "    def forward(self, batch):\n"
            "        batch = self.encoder(batch)\n"
            '        if hasattr(self, "rrwp_abs_encoder"):\n'
            "            batch = self.rrwp_abs_encoder(batch)\n"
            "            batch = self.rrwp_rel_encoder(batch)\n"
            '        if hasattr(self, "pre_mp"):\n'
            "            batch = self.pre_mp(batch)\n"
            "        if self.global_vnode is not None:\n"
            "            batch = self.global_vnode(batch)\n"
            "        batch = self.layers(batch)\n"
            "        if self.global_vnode is not None:\n"
            "            # Communication-only: preserve official real-node pooling.\n"
            "            batch.x = batch.x[batch.real_node_mask]\n"
            "            batch.batch = batch.batch[batch.real_node_mask]\n"
            "        return self.post_mp(batch)\n"
        ),
        "Communication-only: preserve official real-node pooling",
        "VNode-aware forward/pooling",
    )


def apply_peptides_pandas_warning_patch(repo_dir: Path) -> None:
    """Avoid one pandas positional-index FutureWarning per Peptides graph."""
    for relative in (
        "grit/loader/dataset/peptides_functional.py",
        "grit/loader/dataset/peptides_structural.py",
    ):
        path = repo_dir / relative
        text = path.read_text(encoding="utf-8", errors="replace")
        updated = text.replace(
            "            smiles = smiles_list[i]\n",
            "            smiles = smiles_list.iloc[i]\n",
        )
        if relative.endswith("peptides_structural.py"):
            updated = updated.replace(
                "            data.y = torch.Tensor([y])\n",
                (
                    "            data.y = torch.from_numpy(\n"
                    "                y.to_numpy(dtype='float32')).view(1, -1)\n"
                ),
                1,
            )
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            log(f"[dataset-compat] Removed pandas warning hot loop: {path}")


def verify_attention_rrwp_vnode_patch(repo_dir: Path) -> None:
    required = {
        "grit/config/gt_config.py": ["cfg.gt.attn.hops = 1", "cfg.gt.attn.global_vnode = False"],
        "grit/config/posenc_config.py": ["cfg.posenc_RRWP.local_horizon = -1"],
        "grit/transform/rrwp.py": ["support_horizon = int(support_horizon)", "pe[..., keep_channels:] = 0"],
        "grit/transform/posenc_stats.py": ["support_horizon=(cfg.gt.attn.hops"],
        "grit/encoder/rrwp_encoder.py": ['batch.get("rrwp_attention_edge_index", None)'],
        "grit/network/grit_model.py": ["max_hops=cfg.gt.attn.hops", "class GlobalVNode(torch.nn.Module):", "Communication-only"],
    }
    errors: list[str] = []
    for relative, tokens in required.items():
        path = repo_dir / relative
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            compile(text, str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"{relative}: {exc}")
        missing = [token for token in tokens if token not in text]
        if missing:
            errors.append(f"{relative}: missing {missing}")
    if errors:
        raise RuntimeError("Incomplete Peptides k-hop/RRWP/VNode patch:\n  - " + "\n  - ".join(errors))
    log("[patch-check] exact k-hop, local RRWP, and global VNode support verified.")


def apply_recovery_checkpoint_patch(repo_dir: Path) -> None:
    """Add stable first/latest/best recovery checkpoints to official training."""
    custom_train = repo_dir / "grit/train/custom_train.py"
    _replace_exact(
        custom_train,
        "import logging\nimport time\n",
        "import logging\nimport os\nimport shutil\nimport time\n",
        "import os\nimport shutil\nimport time",
        "recovery-checkpoint imports",
    )
    recovery_block = (
        "\n"
        "            if os.environ.get('GRIT_FORCE_RECOVERY_CKPT', '0') == '1' and cfg.train.enable_ckpt:\n"
        "                try:\n"
        "                    recovery_period = int(os.environ.get('GRIT_RECOVERY_CKPT_PERIOD', '100'))\n"
        "                except ValueError:\n"
        "                    recovery_period = 100\n"
        "                recovery_period = max(0, recovery_period)\n"
        "                recovery_reason = None\n"
        "                if os.environ.get('GRIT_SAVE_FIRST_RECOVERY_CKPT', '1') == '1' and cur_epoch == start_epoch:\n"
        "                    recovery_reason = 'first_after_resume'\n"
        "                elif best_epoch == cur_epoch:\n"
        "                    recovery_reason = 'new_best'\n"
        "                elif recovery_period and cur_epoch > 0 and cur_epoch % recovery_period == 0:\n"
        "                    recovery_reason = f'period_{recovery_period}'\n"
        "                if recovery_reason is not None:\n"
        "                    save_ckpt(model, optimizer, scheduler, cur_epoch)\n"
        "                    recovery_path = get_ckpt_path(get_ckpt_epoch(cur_epoch))\n"
        "                    logging.info('Forced numbered recovery checkpoint saved (%s): %s', recovery_reason, recovery_path)\n"
        "                    recovery_dir = os.environ.get('GRIT_RECOVERY_CKPT_DIR', '')\n"
        "                    if recovery_dir:\n"
        "                        os.makedirs(recovery_dir, exist_ok=True)\n"
        "                        shutil.copy2(recovery_path, os.path.join(recovery_dir, 'latest.ckpt'))\n"
        "                        logging.info('Forced latest recovery checkpoint saved: %s', os.path.join(recovery_dir, 'latest.ckpt'))\n"
        "                        with open(os.path.join(recovery_dir, 'latest_epoch.txt'), 'w', encoding='utf-8') as f:\n"
        "                            f.write(f'{cur_epoch}\\n')\n"
        "                        if recovery_reason == 'new_best':\n"
        "                            shutil.copy2(recovery_path, os.path.join(recovery_dir, 'best.ckpt'))\n"
        "                            with open(os.path.join(recovery_dir, 'best_epoch.txt'), 'w', encoding='utf-8') as f:\n"
        "                                f.write(f'{cur_epoch}\\n')\n"
        "                        elif recovery_reason == 'first_after_resume':\n"
        "                            shutil.copy2(recovery_path, os.path.join(recovery_dir, 'first_after_resume.ckpt'))\n"
        "                        elif recovery_reason.startswith('period_'):\n"
        "                            shutil.copy2(recovery_path, os.path.join(recovery_dir, f'{recovery_reason}_epoch{cur_epoch}.ckpt'))\n"
        "                        logging.info('Stable recovery checkpoint copied (%s): %s', recovery_reason, recovery_dir)\n"
    )
    _insert_after(
        custom_train,
        (
            "            logging.info(\n"
            "                f\"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s \"\n"
            "                f\"(avg {np.mean(full_epoch_times):.1f}s) | \"\n"
            "                f\"Best so far: epoch {best_epoch}\\t\"\n"
            "                f\"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\\t\"\n"
            "                f\"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\\t\" \n"
            "                f\"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}\\n\"\n"
            "                f\"-----------------------------------------------------------\"\n"
            "            )\n"
        ),
        recovery_block,
        "GRIT_FORCE_RECOVERY_CKPT",
        "stable recovery checkpoints",
    )
    # Upgrade a checkout patched by the immediately preceding runner version,
    # whose recovery behavior was correct but lacked the legacy verifier phrase.
    _replace_exact(
        custom_train,
        "                        shutil.copy2(recovery_path, os.path.join(recovery_dir, 'latest.ckpt'))\n",
        (
            "                        shutil.copy2(recovery_path, os.path.join(recovery_dir, 'latest.ckpt'))\n"
            "                        logging.info('Forced latest recovery checkpoint saved: %s', os.path.join(recovery_dir, 'latest.ckpt'))\n"
        ),
        "Forced latest recovery checkpoint saved",
        "legacy recovery-verifier compatibility",
    )
    _replace_exact(
        custom_train,
        "                    recovery_path = get_ckpt_path(get_ckpt_epoch(cur_epoch))\n",
        (
            "                    recovery_path = get_ckpt_path(get_ckpt_epoch(cur_epoch))\n"
            "                    logging.info('Forced numbered recovery checkpoint saved (%s): %s', recovery_reason, recovery_path)\n"
        ),
        "Forced numbered recovery checkpoint saved",
        "current recovery-verifier compatibility",
    )


def expected_config_values(args: argparse.Namespace) -> dict[tuple[str, ...], Any]:
    info = TASKS[args.task]
    values: dict[tuple[str, ...], Any] = {
        ("metric_best",): info["metric_best"],
        ("dataset", "format"): "OGB",
        ("dataset", "name"): info["dataset_name"],
        ("dataset", "task"): "graph",
        ("dataset", "task_type"): info["task_type"],
        ("dataset", "transductive"): False,
        ("dataset", "node_encoder"): True,
        ("dataset", "node_encoder_name"): "Atom",
        ("dataset", "node_encoder_bn"): False,
        ("dataset", "edge_encoder"): True,
        ("dataset", "edge_encoder_name"): "Bond",
        ("dataset", "edge_encoder_bn"): False,
        ("posenc_RRWP", "enable"): True,
        ("posenc_RRWP", "ksteps"): info["rrwp_ksteps"],
        ("posenc_RRWP", "add_identity"): True,
        ("posenc_RRWP", "add_node_attr"): False,
        ("train", "mode"): "custom",
        ("train", "batch_size"): info["batch_size"],
        ("model", "type"): "GritTransformer",
        ("model", "loss_fun"): info["loss_fun"],
        ("model", "graph_pooling"): info["graph_pooling"],
        ("gt", "layer_type"): "GritTransformer",
        ("gt", "layers"): info["layers"],
        ("gt", "n_heads"): info["heads"],
        ("gt", "dim_hidden"): info["hidden"],
        ("gt", "dropout"): info["dropout"],
        ("gt", "attn_dropout"): info["attn_dropout"],
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
        ("gnn", "layers_post_mp"): info["layers_post_mp"],
        ("gnn", "dim_inner"): info["hidden"],
        ("gnn", "batchnorm"): True,
        ("gnn", "act"): "relu",
        ("gnn", "dropout"): 0.0,
        ("optim", "clip_grad_norm"): True,
        ("optim", "optimizer"): "adamW",
        ("optim", "weight_decay"): 0.0,
        ("optim", "base_lr"): info["base_lr"],
        ("optim", "max_epoch"): info["max_epoch"],
        ("optim", "scheduler"): "cosine_with_warmup",
        ("optim", "num_warmup_epochs"): info["warmup"],
        ("gt", "attn", "full_attn"): args.attention == "dense",
        ("gt", "attn", "sparsity"): "full" if args.attention == "dense" else "k_hop",
        ("gt", "attn", "hops"): args.hops,
        ("gt", "attn", "global_vnode"): args.global_vnode,
        ("posenc_RRWP", "local_horizon"): args.rrwp_horizon,
    }
    if info.get("metric_agg") is not None:
        values[("metric_agg",)] = info["metric_agg"]
    return values


def make_run_config(repo_dir: Path, args: argparse.Namespace) -> Path:
    import yaml

    info = TASKS[args.task]
    official = repo_dir / info["official_cfg"]
    generated = repo_dir / info["generated_cfg"]
    if not official.exists():
        raise FileNotFoundError(f"official Peptides config is missing: {official}")
    cfg = yaml.safe_load(official.read_text(encoding="utf-8"))
    nested_set(cfg, ("gt", "attn", "full_attn"), args.attention == "dense")
    nested_set(cfg, ("gt", "attn", "sparsity"), "full" if args.attention == "dense" else "k_hop")
    nested_set(cfg, ("gt", "attn", "hops"), args.hops)
    nested_set(cfg, ("gt", "attn", "global_vnode"), args.global_vnode)
    nested_set(cfg, ("posenc_RRWP", "local_horizon"), args.rrwp_horizon)
    nested_set(cfg, ("mlflow", "name"), f"{generated.stem}-{variant_slug(args)}")
    nested_set(cfg, ("wandb", "use"), False)
    if args.wandb_project:
        nested_set(cfg, ("wandb", "project"), args.wandb_project)
    header = (
        f"# Configurable attention/RRWP/VNode variant of {info['official_cfg']}.\n"
        "# Official task architecture and training settings are otherwise preserved.\n"
    )
    generated.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log(f"[config] wrote {args.task}/{variant_slug(args)} config: {generated}")
    return generated


def validate_config(repo_dir: Path, args: argparse.Namespace, *, allow_drift: bool) -> Path:
    import yaml

    info = TASKS[args.task]
    cfg_rel = info["generated_cfg"]
    cfg_path = repo_dir / cfg_rel
    if not cfg_path.exists():
        raise FileNotFoundError(f"GRIT config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for key_path, expected in expected_config_values(args).items():
        try:
            actual = nested_get(cfg, key_path)
        except Exception:
            errors.append(f"missing {'.'.join(key_path)}; expected {expected!r}")
            continue
        if not semantically_equal(actual, expected):
            errors.append(f"{'.'.join(key_path)}={actual!r}; expected {expected!r}")
    if errors:
        msg = f"{args.task}/{variant_slug(args)} config does not match expected setup:\n" + "\n".join(f"  - {e}" for e in errors)
        if not allow_drift:
            raise RuntimeError(msg + "\nPass --allow-upstream-config-drift to run anyway.")
        log("[config-warning] " + msg)

    info = TASKS[args.task]
    log(
        "[config] validated: "
        f"task={args.task} ({info['dataset_name']}), variant={variant_slug(args)}, RRWP-{info['rrwp_ksteps']}, "
        f"{info['layers']} layers, hidden={info['hidden']}, heads={info['heads']}, "
        f"batch={info['batch_size']}, max_epoch={info['max_epoch']}, "
        f"scheduler=cosine_with_warmup, warmup={info['warmup']}."
    )
    rrwp = "global" if args.rrwp_horizon < 0 else f"local<=P^{args.rrwp_horizon}"
    log(f"[control] attention={args.attention}, hops={args.hops}, RRWP={rrwp}, VNode={args.global_vnode}.")
    return cfg_path


def build_train_command(args: argparse.Namespace, cfg_path: Path) -> list[str]:
    results_dir = args.drive_dir / "results"
    dataset_dir = args.dataset_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = not args.checkpoint_every_epoch
    ckpt_clean = False if (args.keep_all_checkpoints or args.checkpoint_every_epoch or args.guaranteed_checkpoints) else True
    cmd = [
        sys.executable,
        "-u",
        "main.py",
        "--cfg",
        str(cfg_path),
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
        "posenc_RRWP.local_horizon",
        str(args.rrwp_horizon),
        "num_threads",
        str(args.num_threads),
    ]
    if args.accelerator:
        cmd.extend(["accelerator", args.accelerator])
    cmd.extend(args.cfg_overrides)
    return cmd


def checkpoint_inventory(drive_dir: Path, wrapper_log: Path, seed: int) -> Path:
    patterns = ("*.ckpt", "*.pt", "*.pth")
    result_root = drive_dir / "results"
    candidates = sorted(
        {path for pattern in patterns for path in result_root.rglob(pattern) if path.is_file()},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    sidecars: list[dict[str, Any]] = []
    for sidecar in result_root.rglob("best_epoch.txt") if result_root.exists() else []:
        try:
            sidecars.append({"path": str(sidecar), "best_epoch": sidecar.read_text(encoding="utf-8").strip()})
        except Exception:
            pass
    payload = {
        "seed": seed,
        "drive_dir": str(drive_dir),
        "wrapper_log": str(wrapper_log),
        "latest_checkpoint_by_mtime": str(candidates[0]) if candidates else None,
        "checkpoint_candidates": [
            {
                "path": str(path),
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
                "bytes": path.stat().st_size,
            }
            for path in candidates
        ],
        "best_epoch_sidecars": sidecars,
    }
    out = drive_dir / f"checkpoint_inventory_seed{seed}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (drive_dir / "latest_checkpoint_inventory.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    log(f"[checkpoint-inventory] latest={payload['latest_checkpoint_by_mtime']}")
    log(f"[checkpoint-inventory] wrote: {out}")
    return out


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train dense or exact k-hop GRIT with optional local RRWP/VNode on either Peptides task.",
        epilog=(
            "Examples:\n"
            "  --task func --attention khop --hops 1 --rrwp-horizon 1\n"
            "  --task struct --attention khop --hops 1 --global-vnode\n"
            "  --task func --attention khop --hops 2\n"
        ),
    )
    parser.add_argument("--task", choices=sorted(TASKS), default="func")
    parser.add_argument("--attention", choices=["dense", "khop"], default="dense")
    parser.add_argument("--hops", "-k", type=int, default=1, help="Attention radius; func supports 1..16 and struct 1..23.")
    parser.add_argument("--global-vnode", action="store_true", help="Add one learned 96D graph token to every attention layer; exclude it from pooling.")
    parser.add_argument(
        "--rrwp-horizon",
        type=int,
        default=-1,
        help="-1 keeps global RRWP; H>=0 keeps I and walk channels through P^H while retaining encoder width.",
    )
    parser.add_argument("--drive-dir", type=Path, default=None)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Shared task-specific dataset cache; created automatically when absent.",
    )
    parser.add_argument("--repo-dir", type=Path, default=None)
    parser.add_argument("--repo-url", default=OFFICIAL_GRIT_REPO)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--commit", default=OFFICIAL_GRIT_COMMIT)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--name-tag", default=None)
    parser.add_argument("--max-epoch", type=int, default=None, help="Default is the official config value for the selected task.")
    parser.add_argument("--ckpt-period", type=int, default=100)
    parser.add_argument("--recovery-ckpt-period", type=int, default=100)
    parser.add_argument("--rrwp-stream-chunk-size", type=int, default=32)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--pyg-version", default="2.2.0")
    parser.add_argument("--official-torch112", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-editable-install", action="store_true", help="Do not run pip install -e on the job-local GRIT checkout.")
    parser.add_argument("--force-fresh-repo", action="store_true")
    parser.add_argument("--allow-upstream-config-drift", action="store_true")
    parser.add_argument("--expected-params", type=int, default=None)
    parser.add_argument("--allow-param-count-drift", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--console-verbosity", choices=["compact", "standard", "full"], default="compact")
    parser.add_argument("--console-epoch-period", type=int, default=1)
    parser.add_argument("--accelerator", default="cuda:0")
    parser.add_argument("--auto-resume", action="store_true", default=True)
    parser.add_argument("--no-auto-resume", action="store_false", dest="auto_resume")
    parser.add_argument("--keep-all-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-every-epoch", action="store_true")
    parser.add_argument("--guaranteed-checkpoints", action="store_true", default=True)
    parser.add_argument("--no-guaranteed-checkpoints", action="store_false", dest="guaranteed_checkpoints")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("cfg_overrides", nargs=argparse.REMAINDER, help="Optional trailing GraphGym config overrides.")
    args = parser.parse_args(strip_colab_kernel_args(list(sys.argv[1:] if argv is None else argv)))
    if args.cfg_overrides and args.cfg_overrides[0] == "--":
        args.cfg_overrides = args.cfg_overrides[1:]
    max_hops = int(TASKS[args.task]["rrwp_ksteps"]) - 1
    if not 1 <= args.hops <= max_hops:
        parser.error(f"--hops must be between 1 and {max_hops} for Peptides-{args.task}")
    if not -1 <= args.rrwp_horizon <= max_hops:
        parser.error(f"--rrwp-horizon must be -1 or between 0 and {max_hops} for Peptides-{args.task}")
    if args.drive_dir is None:
        args.drive_dir = default_drive_dir(args)
    if args.dataset_dir is None:
        args.dataset_dir = Path(
            f"/content/drive/MyDrive/grit_peptides_{args.task}_shared_data"
        )
    if args.repo_dir is None:
        args.repo_dir = default_grit_repo_dir(args)
    if args.name_tag is None:
        args.name_tag = default_name_tag(args)
    if args.max_epoch is None:
        args.max_epoch = int(TASKS[args.task]["max_epoch"])
    return args


def main(argv: Sequence[str] | None = None) -> None:
    bootstrap, rest = split_bootstrap_args(argv)
    premount_drive(bootstrap.drive_mount)
    project_root = bootstrap_project_repo(
        bootstrap.project_repo_url,
        bootstrap.project_branch,
        bootstrap.project_repo_dir,
        bootstrap.secret_name,
        skip_git=bootstrap.skip_project_git,
    )
    log(f"[project] ready: {project_root}")

    args = parse_args(rest)

    try:
        from graph_specialisation_metrics.grit_patches import peptides as peptides_common
        from graph_specialisation_metrics.grit_patches import qm9_gap as base
    except ImportError:
        # Compatibility with the earlier Colab project branch layout.
        from experiments.peptides_struct.training import grit_peptides_struct_common as peptides_common
        from experiments.zinc.training import grit_zinc_core as base

    args.drive_dir.mkdir(parents=True, exist_ok=True)

    # The shim also handles dependency-version drift on older Python versions
    # (notably sklearn >=1.6 under Python 3.10 on CSD3).
    compat_shim_dir = base.write_py312_compat_shim(args.drive_dir)

    if not args.skip_install:
        base.install_dependencies(args)
        peptides_common.install_peptides_dependencies(base)
    else:
        log("[deps] skipping dependency installation (--skip-install).")

    commit = base.clone_or_update_repo(args.repo_dir, args.repo_url, args.branch, args.commit or None, args.force_fresh_repo)
    base.run_cmd(["git", "reset", "--hard", commit], cwd=args.repo_dir)

    peptides_common.apply_peptides_dataset_compat_patch(base, args.repo_dir)
    apply_peptides_pandas_warning_patch(args.repo_dir)
    peptides_common.apply_peptides_streaming_rrwp_patch(base, args.repo_dir)

    apply_attention_rrwp_vnode_patch(args.repo_dir)
    verify_attention_rrwp_vnode_patch(args.repo_dir)
    cfg_path = make_run_config(args.repo_dir, args)

    if args.guaranteed_checkpoints:
        apply_recovery_checkpoint_patch(args.repo_dir)
        base.verify_recovery_checkpoint_patch(args.repo_dir)

    if args.skip_editable_install:
        log("[deps] skipping editable GRIT install; main.py imports the job-local checkout.")
    else:
        base.install_grit_editable(args.repo_dir)
    cfg_path = validate_config(args.repo_dir, args, allow_drift=args.allow_upstream_config_drift)
    base.print_environment_summary(args.drive_dir, args.repo_dir, commit)
    log(f"[run] task={args.task} variant={variant_slug(args)} seed={args.seed} tag={args.name_tag}")
    log(f"[run] drive_dir={args.drive_dir}")
    log(f"[run] repo_dir={args.repo_dir}")

    cmd = build_train_command(args, cfg_path)
    if args.dry_run:
        log("[dry-run] training command:")
        log(" ".join(map(str, cmd)))
        return

    wrapper_log = args.drive_dir / "wrapper_logs" / f"grit_{TASKS[args.task]['label']}_{variant_slug(args)}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    train_env = base.env_with_py312_compat(compat_shim_dir)
    train_env["PYTHONWARNINGS"] = "ignore::FutureWarning"
    train_env["GRIT_PE_STREAM_CHUNK_SIZE"] = str(max(1, int(args.rrwp_stream_chunk_size)))
    if args.guaranteed_checkpoints:
        recovery_dir = (
            args.drive_dir
            / "results"
            / "_recovery_checkpoints"
            / f"{args.task}_{variant_slug(args)}_seed{args.seed}_{base.safe_path_fragment(args.name_tag)}"
        )
        recovery_dir.mkdir(parents=True, exist_ok=True)
        train_env["GRIT_FORCE_RECOVERY_CKPT"] = "1"
        train_env["GRIT_RECOVERY_CKPT_PERIOD"] = str(max(0, int(args.recovery_ckpt_period)))
        train_env["GRIT_RECOVERY_CKPT_DIR"] = str(recovery_dir)
        train_env["GRIT_SAVE_FIRST_RECOVERY_CKPT"] = "1"
        log("[checkpoint] recovery checkpoints enabled")
        log(f"[checkpoint] stable best path: {recovery_dir / 'best.ckpt'}")
        log(f"[checkpoint] first completed epoch after resume: {recovery_dir / 'first_after_resume.ckpt'}")
        log(f"[checkpoint] numbered snapshots every {args.recovery_ckpt_period} epochs")

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
    checkpoint_inventory(args.drive_dir, wrapper_log, args.seed)
    if rc != 0:
        raise SystemExit(rc)
    log("[done] training completed.")
    log(f"[done] full raw log: {wrapper_log}")
    log(f"[done] Drive results: {args.drive_dir / 'results'}")


if __name__ == "__main__":
    main()
