#!/usr/bin/env python3
"""Standalone Google Colab runner for the ZINC dissertation core procedure.

This runner is intentionally orchestration-only. It clones the current
methodology repo branch with the Colab secret ``dissertation_key``, discovers
the default Drive outputs from the dense GRIT and 1-hop GRIT ZINC Colab
training runners, prepares a ZINC config, and invokes the repo's
``graph_specialisation_metrics.main_procedure`` CLI for Steps 0-5.

The repo implementation runs the full GRIT/1-hop GRIT intervention procedure
when trained checkpoints are present. This file records any missing checkpoints,
dependency failures, or deliberately omitted validation references in Drive and,
by default, fails if the requested complete run cannot be completed. Pass
``--allow-incomplete`` only when you want a discovery/preflight artifact.
"""

from __future__ import annotations

import argparse
import ast
import csv
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


PUBLIC_REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
DEFAULT_BRANCH = "codex/cfim-grit-experiments"
DEFAULT_REPO_DIR = "/content/Graph-Specialisation-and-Metrics"
DEFAULT_SECRET_NAME = "dissertation_key"

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/graph_specialisation_metrics/zinc_main_procedure_colab"
DEFAULT_DENSE_DRIVE_DIR = "/content/drive/MyDrive/grit_zinc_official"
DEFAULT_ONEHOP_DRIVE_DIR = "/content/drive/MyDrive/grit_zinc_1hop"
DEFAULT_GIN_DRIVE_DIR = "/content/drive/MyDrive/gin_zinc_official"
DEFAULT_GIN_REPO_DIR = "/content/benchmarking-gnns_analysis"
DEFAULT_PYG_VERSION = "2.2.0"

OFFICIAL_GRIT_REPO = "https://github.com/LiamMa/GRIT.git"
OFFICIAL_GRIT_COMMIT = "6c988ea600a606fbb49a2246c64a2d37396b3ab5"
OFFICIAL_GIN_REPO = "https://github.com/graphdeeplearning/benchmarking-gnns.git"
OFFICIAL_GIN_COMMIT = "b6c407712fa576e9699555e1e035d1e327ccae6c"
EXPECTED_GRIT_PARAMS = 473_473
EXPECTED_OFFICIAL_GIN_PARAMS = 509_549
ONE_HOP_CFG_TEXT = """\
# Parameter-matched 1-hop sparse-control variant of the official ZINC GRIT RRWP config.
#
# This file intentionally keeps the official ZINC subset model, optimizer,
# training schedule, RRWP dimensions, and decoder settings. The only scientific
# intervention is `gt.attn.sparsity: one_hop`, which selects the masked RRWP edge
# encoder added by the local patch. That encoder keeps attention/message passing
# on original molecular bonds plus self-loops after RRWP encoding instead of the
# official complete-graph attention. The trainable parameter count remains
# 473,473, matching the official GRIT ZINC RRWP config.
out_dir: results
metric_best: mae
metric_agg: argmin
tensorboard_each_run: True
accelerator: "cuda:0"
mlflow:
  use: False
  project: Exp
  name: zinc-GRIT-RRWP-1hop
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
    sparsity: one_hop
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

MARKDOWN_REQUIRED_STEPS = {
    "0": "measurement_model_validation",
    "1": "performance_gap",
    "2": "usage_vs_causal_usage",
    "3": "distance_resolved_overfitting",
    "4": "mediator_patching",
    "5": "non_composable_gap_attribution",
}


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path | str | None = None,
    safe_display: str | None = None,
    check: bool = True,
    env: Mapping[str, str] | None = None,
    stream: bool = False,
) -> subprocess.CompletedProcess[str]:
    display = safe_display or " ".join(str(c) for c in cmd)
    print(f"[cmd] {display}", flush=True)
    if stream:
        merged_env = dict(env) if env is not None else None
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=merged_env,
            bufsize=1,
        )
        assert proc.stdout is not None
        lines: list[str] = []
        for line in proc.stdout:
            lines.append(line)
            print(line, end="", flush=True)
        returncode = proc.wait()
        stdout = "".join(lines)
        if check and returncode != 0:
            raise RuntimeError(f"command failed with exit code {returncode}: {display}")
        return subprocess.CompletedProcess([str(c) for c in cmd], returncode, stdout=stdout, stderr=None)
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=dict(env) if env is not None else None,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {display}")
    return proc


def pip_install(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_cmd([sys.executable, "-m", "pip", "install", *args], check=check)


def patch_dgl_graphbolt_import() -> bool:
    """Let DGL import on Colab Torch versions that lack a GraphBolt binary.

    The official Benchmarking-GNNs GIN path uses DGLGraph/message passing, not
    GraphBolt. Current Colab Torch wheels can be newer than DGL's packaged
    GraphBolt extension, causing ``import dgl`` to fail before the GIN code is
    reached. Stubbing GraphBolt is narrower than changing the DGL version and
    prevents partial failed imports from registering torchdata DataPipes twice.
    """

    import site
    import sysconfig

    roots: list[Path] = []
    for raw in [*site.getsitepackages(), sysconfig.get_paths().get("purelib", ""), sysconfig.get_paths().get("platlib", "")]:
        if raw:
            root = Path(str(raw))
            if root not in roots:
                roots.append(root)
    patched = False
    stub = '''"""Colab compatibility stub for optional DGL GraphBolt.

The dissertation GIN reference uses ordinary DGL graphs and message passing,
not GraphBolt. The pip DGL wheel can lack a GraphBolt extension for Colab's
newer torch build, so importing the real module fails before DGL itself is
usable. This stub is written by colab_zinc_main_procedure.py after installing
DGL.
"""

__all__ = []
'''
    for root in roots:
        init_py = root / "dgl" / "graphbolt" / "__init__.py"
        if not init_py.exists():
            continue
        text = init_py.read_text(encoding="utf-8")
        if "Colab compatibility stub for optional DGL GraphBolt" in text:
            patched = True
            continue
        backup = init_py.with_suffix(".py.original_graphbolt")
        if not backup.exists():
            backup.write_text(text, encoding="utf-8")
        init_py.write_text(stub, encoding="utf-8")
        print(f"[deps] stubbed optional DGL GraphBolt package: {init_py}", flush=True)
        patched = True
    if not patched:
        print("[deps-warning] DGL GraphBolt package was not found to patch.", flush=True)
    return patched


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return path


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    return path


def require_colab_token(secret_name: str) -> str:
    try:
        from google.colab import userdata
    except Exception as exc:  # pragma: no cover - outside Colab.
        raise RuntimeError("Run this file in Google Colab; it uses google.colab.userdata.") from exc
    token = userdata.get(secret_name)
    if not token:
        raise RuntimeError(f"Colab secret {secret_name!r} is missing or empty")
    secret = str(token).strip()
    prefix = "github_pat_" if secret.startswith("github_pat_") else secret[:4]
    print(f"[auth] secret {secret_name!r} loaded; length={len(secret)}, prefix={prefix!r}", flush=True)
    return secret


def mount_drive() -> None:
    try:
        from google.colab import drive
    except Exception as exc:  # pragma: no cover - outside Colab.
        raise RuntimeError("Google Drive mounting requires Google Colab") from exc
    drive.mount("/content/drive", force_remount=False)


def sanitize_repo_url(repo_url: str) -> str:
    repo_url = str(repo_url).strip()
    if repo_url.startswith("[") and "](" in repo_url and repo_url.endswith(")"):
        repo_url = repo_url.split("](", 1)[1][:-1]
    return repo_url


def repo_owner(public_url: str) -> str:
    public_url = sanitize_repo_url(public_url)
    if not public_url.startswith("https://github.com/"):
        raise ValueError("only https://github.com repo URLs are supported")
    return public_url.removeprefix("https://github.com/").split("/", 1)[0]


def auth_url_candidates(public_url: str, token: str, username: str | None = None) -> list[str]:
    public_url = sanitize_repo_url(public_url)
    user = quote(username or repo_owner(public_url), safe="")
    token_q = quote(token, safe="")
    suffix = public_url.removeprefix("https://github.com/")
    return [
        f"https://{user}:{token_q}@github.com/{suffix}",
        f"https://x-access-token:{token_q}@github.com/{suffix}",
    ]


def clone_or_update_repo(
    repo_url: str,
    branch: str,
    repo_dir: Path,
    token: str,
    github_username: str | None,
) -> None:
    repo_url = sanitize_repo_url(repo_url)
    candidates = auth_url_candidates(repo_url, token, github_username)
    authed = candidates[0]
    preflight: subprocess.CompletedProcess[str] | None = None
    for idx, candidate in enumerate(candidates, start=1):
        preflight = run_cmd(
            ["git", "ls-remote", "--heads", candidate, branch],
            safe_display=f"git ls-remote --heads <token-authenticated-url> {branch} [auth-form {idx}]",
            check=False,
        )
        if preflight.returncode == 0:
            authed = candidate
            break
    assert preflight is not None
    if preflight.returncode != 0:
        raise RuntimeError(
            "GitHub authentication failed. A fine-grained PAT must select this repo "
            "and grant Repository permissions: Contents=Read-only. If the branch is "
            "not pushed to GitHub, push it or pass --branch for a pushed branch."
        )
    if not preflight.stdout.strip():
        raise RuntimeError(f"GitHub authentication worked, but branch {branch!r} was not found")

    if (repo_dir / ".git").exists():
        run_cmd(
            ["git", "-C", str(repo_dir), "remote", "set-url", "origin", authed],
            safe_display=f"git -C {repo_dir} remote set-url origin <token-authenticated-url>",
        )
        run_cmd(["git", "-C", str(repo_dir), "fetch", "origin", branch])
        run_cmd(["git", "-C", str(repo_dir), "checkout", branch])
        run_cmd(["git", "-C", str(repo_dir), "pull", "--ff-only", "origin", branch])
        run_cmd(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])
        return

    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    run_cmd(
        ["git", "clone", "--branch", branch, "--single-branch", authed, str(repo_dir)],
        safe_display=f"git clone --branch {branch} <token-authenticated-url> {repo_dir}",
    )
    run_cmd(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])


def write_py312_compat_shim(base_dir: Path) -> Path:
    """Create a subprocess-local compatibility shim for old GRIT/PyG deps."""

    shim_dir = base_dir / "python312_compat"
    shim_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize = shim_dir / "sitecustomize.py"
    sitecustomize.write_text(
        r'''
import importlib.machinery
import os
import pkgutil
import site
import sys
import sysconfig

_preferred = []
for _p in list(site.getsitepackages()) + [
    sysconfig.get_paths().get("purelib", ""),
    sysconfig.get_paths().get("platlib", ""),
]:
    if _p and os.path.isdir(_p) and "/usr/local/" in _p and _p not in _preferred:
        _preferred.append(_p)
for _p in reversed(_preferred):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(1, _p)

if not hasattr(pkgutil, "ImpImporter"):
    class _CompatImpImporter:
        pass
    pkgutil.ImpImporter = _CompatImpImporter

if not hasattr(importlib.machinery.FileFinder, "find_module"):
    def _compat_find_module(self, fullname, path=None):
        spec = self.find_spec(fullname)
        return None if spec is None else spec.loader
    importlib.machinery.FileFinder.find_module = _compat_find_module

try:
    import torch
    _orig_torch_load = torch.load
    def _compat_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    if getattr(torch.load, "__name__", "") != "_compat_torch_load":
        torch.load = _compat_torch_load
except Exception:
    pass

_mod = sys.modules.get("pkg_resources")
_mod_file = getattr(_mod, "__file__", "") if _mod is not None else ""
if _mod_file.startswith("/usr/lib/python3/dist-packages/"):
    del sys.modules["pkg_resources"]

try:
    import inspect
    import numpy as _np
    import sklearn.metrics as _sk_metrics
    _mse_sig = inspect.signature(_sk_metrics.mean_squared_error)
    if "squared" not in _mse_sig.parameters:
        _orig_mean_squared_error = _sk_metrics.mean_squared_error
        def _compat_mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", squared=True):
            mse = _orig_mean_squared_error(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)
            return mse if squared else _np.sqrt(mse)
        _sk_metrics.mean_squared_error = _compat_mean_squared_error
except Exception:
    pass
'''.lstrip(),
        encoding="utf-8",
    )
    print(f"[compat] wrote Python compatibility shim: {sitecustomize}", flush=True)
    return shim_dir


def env_with_py312_compat(shim_dir: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if shim_dir is not None:
        old = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(shim_dir) + ((os.pathsep + old) if old else "")
    return env


def apply_colab_repo_hotfixes(repo_dir: Path) -> None:
    """Apply small runner-side fixes before installing the cloned repo.

    This keeps the notebook usable even when the selected GitHub branch lags
    behind the local Colab runner file.
    """
    adapters = repo_dir / "src" / "graph_specialisation_metrics" / "method_adapters.py"
    if not adapters.exists():
        print(f"[hotfix-warning] missing expected adapter source: {adapters}", flush=True)
        return
    text = adapters.read_text(encoding="utf-8")
    fixed = text.replace("torch.inference_mode()", "torch.no_grad()")
    if "import os\n" not in fixed:
        fixed = fixed.replace("import math\n", "import math\nimport os\n", 1)
    dgl_start = fixed.find("    @staticmethod\n    def _install_dgl_compat")
    dgl_end = fixed.find("\n    def _config_payload", dgl_start)
    if dgl_start != -1 and dgl_end != -1 and "DGL import failed for the official Benchmarking-GNNs GIN adapter" not in fixed[dgl_start:dgl_end]:
        new_dgl_compat = '''    @staticmethod
    def _install_dgl_compat() -> None:
        os.environ.setdefault("DGLBACKEND", "pytorch")
        try:
            import dgl  # noqa: F401
            import dgl.function as dgl_fn
        except Exception as exc:  # pragma: no cover - optional dependency.
            raise RuntimeError(
                "DGL import failed for the official Benchmarking-GNNs GIN adapter: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if not hasattr(dgl_fn, "copy_src") and hasattr(dgl_fn, "copy_u"):
            def copy_src(src: str | None = None, out: str | None = None, **kwargs: Any) -> Any:
                src = kwargs.get("src", src)
                out = kwargs.get("out", out)
                if src is None or out is None:
                    raise TypeError("copy_src requires src and out")
                return dgl_fn.copy_u(src, out)

            dgl_fn.copy_src = copy_src  # type: ignore[attr-defined]
        if not hasattr(dgl_fn, "copy_edge") and hasattr(dgl_fn, "copy_e"):
            def copy_edge(edge: str | None = None, out: str | None = None, **kwargs: Any) -> Any:
                edge = kwargs.get("edge", edge)
                out = kwargs.get("out", out)
                if edge is None or out is None:
                    raise TypeError("copy_edge requires edge and out")
                return dgl_fn.copy_e(edge, out)

            dgl_fn.copy_edge = copy_edge  # type: ignore[attr-defined]
        if not hasattr(dgl_fn, "copy_dst") and hasattr(dgl_fn, "copy_v"):
            def copy_dst(dst: str | None = None, out: str | None = None, **kwargs: Any) -> Any:
                dst = kwargs.get("dst", dst)
                out = kwargs.get("out", out)
                if dst is None or out is None:
                    raise TypeError("copy_dst requires dst and out")
                return dgl_fn.copy_v(dst, out)

            dgl_fn.copy_dst = copy_dst  # type: ignore[attr-defined]
'''
        fixed = fixed[:dgl_start] + new_dgl_compat + fixed[dgl_end:]
    if fixed != text:
        adapters.write_text(fixed, encoding="utf-8")
        print("[hotfix] updated method_adapters.py for Colab compatibility", flush=True)
    else:
        print("[hotfix] method_adapters.py already has required Colab compatibility", flush=True)

    intervention = repo_dir / "src" / "graph_specialisation_metrics" / "grit_intervention_procedure.py"
    if not intervention.exists():
        print(f"[hotfix-warning] missing expected intervention source: {intervention}", flush=True)
        return
    text = intervention.read_text(encoding="utf-8")
    fixed = text
    if "shortest_path_distance_matrix," not in fixed:
        fixed = fixed.replace(
            "    spearman_corr,\n",
            "    spearman_corr,\n    shortest_path_distance_matrix,\n",
        )
    old_distance_helper = '''def distance_matrix(graph: Any) -> torch.Tensor:
    if hasattr(graph, "distances") and isinstance(graph.distances, torch.Tensor):
        return graph.distances.detach().cpu().float()
    return all_pair_distances_or_compute(pyg_graph_view(graph)).cpu()
'''
    new_distance_helper = '''def distance_matrix(graph: Any) -> torch.Tensor:
    """Return molecular hop distance, not GRIT structural/RRWP fields."""
    if hasattr(graph, "edge_index") and isinstance(graph.edge_index, torch.Tensor):
        return shortest_path_distance_matrix(pyg_graph_view(graph)).detach().cpu().float()
    if hasattr(graph, "distances") and isinstance(graph.distances, torch.Tensor):
        return graph.distances.detach().cpu().float()
    return all_pair_distances_or_compute(pyg_graph_view(graph)).cpu()
'''
    fixed = fixed.replace(old_distance_helper, new_distance_helper)
    if "skip_failed_optional_adapters" not in fixed:
        fn_start = fixed.find("def instantiate_official_models(")
        fn_end = fixed.find("\ndef render_distance_profile(", fn_start)
        if fn_start != -1 and fn_end != -1:
            new_instantiate = '''def instantiate_official_models(config: Mapping[str, Any], discovery: Sequence[Mapping[str, Any]]) -> list[ModelRun]:
    out: list[ModelRun] = []
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(config.get("seeds", [0])[0])
    runtime_cfg = config.get("runtime", {}) if isinstance(config.get("runtime", {}), Mapping) else {}
    skip_failed_optional = bool(runtime_cfg.get("skip_failed_optional_adapters", True))
    for entry in discovery:
        if not entry.get("checkpoint_candidates") or not entry.get("config_candidates"):
            continue
        name = str(entry["model"])
        model_cfg = config["models"][name]
        adapter_kind = str(entry.get("adapter", model_cfg.get("adapter", ""))).strip().lower()
        model_device = str(model_cfg.get("device", device))
        role = str(model_cfg.get("role", ""))
        is_optional_reference = any(token in role.lower() for token in ("reference", "validation")) or name.lower() in {"gin", "gcn"}
        try:
            if adapter_kind == "official_grit":
                adapter = OfficialGRITAdapter(
                    repo_path=Path(str(model_cfg.get("repo_path", "external/GRIT"))),
                    config_path=Path(str(model_cfg.get("config_path") or entry["config_candidates"][0])),
                    checkpoint_path=Path(str(model_cfg.get("checkpoint_path") or entry["checkpoint_candidates"][0])),
                    variant=str(model_cfg.get("variant", name)),
                    official_commit=str(model_cfg.get("official_commit", "")) or None,
                    dataset_dir=Path(str(model_cfg["dataset_dir"])) if model_cfg.get("dataset_dir") else None,
                    device=model_device,
                    seed=seed,
                )
            elif adapter_kind in {"pyg_gin", "official_pyg_gin"}:
                adapter = OfficialPyGGINAdapter(
                    config_path=Path(str(model_cfg.get("config_path") or entry["config_candidates"][0])),
                    checkpoint_path=Path(str(model_cfg.get("checkpoint_path") or entry["checkpoint_candidates"][0])),
                    dataset_dir=Path(str(model_cfg["dataset_dir"])) if model_cfg.get("dataset_dir") else None,
                    device=model_device,
                    seed=seed,
                )
            elif adapter_kind in {"benchmarking_gnns_gin", "official_benchmarking_gnns_gin", "official_dgl_gin"}:
                adapter = OfficialBenchmarkingGNNsGINAdapter(
                    repo_path=Path(str(model_cfg.get("repo_path", "external/benchmarking-gnns"))),
                    config_path=Path(str(model_cfg.get("config_path") or entry["config_candidates"][0])),
                    checkpoint_path=Path(str(model_cfg.get("checkpoint_path") or entry["checkpoint_candidates"][0])),
                    dataset_dir=Path(str(model_cfg["dataset_dir"])) if model_cfg.get("dataset_dir") else None,
                    device=model_device,
                    seed=seed,
                    official_commit=str(model_cfg.get("official_commit", "")) or None,
                )
            else:
                continue
            if is_optional_reference and bool(model_cfg.get("validate_on_load", True)):
                _ = adapter.parameter_count()
                if adapter_kind in {"benchmarking_gnns_gin", "official_benchmarking_gnns_gin", "official_dgl_gin"}:
                    smoke_graphs = adapter.load_zinc_split("test", limit=1)
                    if smoke_graphs:
                        _ = adapter.forward(smoke_graphs[0])
        except Exception as exc:
            if is_optional_reference and skip_failed_optional:
                progress(f"skipping optional reference adapter {name}: {type(exc).__name__}: {exc}")
                continue
            raise
        out.append(ModelRun(name=name, adapter=adapter, role=role, variant=str(model_cfg.get("variant", ""))))
    return out
'''
            fixed = fixed[:fn_start] + new_instantiate + fixed[fn_end:]
    if fixed != text:
        intervention.write_text(fixed, encoding="utf-8")
        print("[hotfix] updated grit_intervention_procedure.py for molecular distances and optional references", flush=True)
    else:
        print("[hotfix] grit_intervention_procedure.py already has required Colab compatibility", flush=True)


def install_repo(repo_dir: Path, *, pyg_version: str) -> None:
    apply_colab_repo_hotfixes(repo_dir)
    os.environ.setdefault("DGLBACKEND", "pytorch")
    run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools<82", "wheel"])
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "networkx", "matplotlib", "numpy", "scipy", "pandas"])

    import importlib

    torch = importlib.import_module("torch")
    torch_version = str(torch.__version__).split("+")[0]
    cuda_version = getattr(torch.version, "cuda", None)
    cuda_tag = "cu" + cuda_version.replace(".", "") if cuda_version else "cpu"
    pyg_wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"
    print(f"[deps] Python: {sys.version.split()[0]} | torch: {torch.__version__} | CUDA: {cuda_version}", flush=True)
    print(f"[deps] PyG wheel index: {pyg_wheel_url}", flush=True)

    pip_install(["pyg-lib", "torch-scatter", "torch-sparse", "torch-cluster", "-f", pyg_wheel_url])
    spline_proc = pip_install(["torch-spline-conv", "-f", pyg_wheel_url], check=False)
    if spline_proc.returncode != 0:
        print("[deps-warning] torch-spline-conv wheel unavailable; continuing because GRIT ZINC RRWP does not require it.", flush=True)
    pip_install([f"torch-geometric=={pyg_version}"])
    pip_install(
        [
            "yacs==0.1.8",
            "pytorch-lightning==1.9.5",
            "torchmetrics==0.9.1",
            "opt_einsum>=3.3",
            "tensorboardX>=2.6,<2.7",
            "ogb==1.3.6",
            "wandb>=0.16,<0.18",
            "scikit-learn>=1.0",
        ]
    )
    dgl_proc = pip_install(["torchdata==0.8.0", "--no-deps"], check=False)
    if dgl_proc.returncode == 0:
        dgl_proc = pip_install(["dgl==2.1.0"], check=False)
    if dgl_proc.returncode == 0:
        patch_dgl_graphbolt_import()
    if dgl_proc.returncode == 0:
        dgl_verify = run_cmd(
            [
                sys.executable,
                "-c",
                "import torchdata.datapipes.iter; import dgl; import dgl.function; print('[deps] DGL import check passed')",
            ],
            check=False,
        )
    else:
        dgl_verify = dgl_proc
    if dgl_proc.returncode != 0 or dgl_verify.returncode != 0:
        print(
            "[deps-warning] DGL install/import failed; dense/1-hop GRIT can still run. "
            "The official Benchmarking-GNNs GIN reference will be attempted and skipped if unusable.",
            flush=True,
        )
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_dir), "--no-deps"])
    for path in [str(repo_dir), str(repo_dir / "src")]:
        if path not in sys.path:
            sys.path.insert(0, path)


def clone_official_grit(repo_dir: Path, *, force: bool = False) -> None:
    if force and repo_dir.exists():
        shutil.rmtree(repo_dir)
    if not (repo_dir / ".git").exists():
        run_cmd(["git", "clone", "--branch", "main", OFFICIAL_GRIT_REPO, str(repo_dir)])
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()
    if current != OFFICIAL_GRIT_COMMIT:
        run_cmd(["git", "fetch", "origin"], cwd=repo_dir)
        run_cmd(["git", "checkout", OFFICIAL_GRIT_COMMIT], cwd=repo_dir)
    print(f"[grit] official checkout {repo_dir} @ {OFFICIAL_GRIT_COMMIT}", flush=True)


def clone_official_gin(repo_dir: Path, *, force: bool = False) -> Path:
    if force and repo_dir.exists():
        shutil.rmtree(repo_dir)
    if not (repo_dir / ".git").exists():
        run_cmd(["git", "clone", OFFICIAL_GIN_REPO, str(repo_dir)])
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()
    if current != OFFICIAL_GIN_COMMIT:
        run_cmd(["git", "fetch", "origin"], cwd=repo_dir)
        run_cmd(["git", "checkout", OFFICIAL_GIN_COMMIT], cwd=repo_dir)
    print(f"[gin] official Benchmarking-GNNs checkout {repo_dir} @ {OFFICIAL_GIN_COMMIT}", flush=True)
    return repo_dir


def _read_text_preserve_newlines(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as f:
        return f.read()


def _write_text_preserve_newlines(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text)


def _native_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _replace_exact(path: Path, old: str, new: str, marker: str, label: str) -> bool:
    text = _read_text_preserve_newlines(path)
    if marker in text:
        print(f"[patch] {label}: already present", flush=True)
        return False
    newline = _native_newline(text)
    old_native = old.replace("\n", newline)
    new_native = new.replace("\n", newline)
    if old_native not in text:
        raise RuntimeError(
            f"Could not apply 1-hop patch segment `{label}` to {path}. "
            "The official GRIT source differs from the pinned layout."
        )
    _write_text_preserve_newlines(path, text.replace(old_native, new_native, 1))
    print(f"[patch] {label}: applied", flush=True)
    return True


def apply_embedded_parameter_matched_onehop_patch(repo_dir: Path, drive_root: Path) -> None:
    """Apply the parameter-matched 1-hop GRIT patch without repo helper imports."""
    print("\n[patch] Applying embedded parameter-matched 1-hop GRIT control patch.", flush=True)
    cfg_path = repo_dir / "configs" / "GRIT" / "zinc-GRIT-RRWP-1hop.yaml"
    current_cfg = _read_text_preserve_newlines(cfg_path) if cfg_path.exists() else ""
    if current_cfg != ONE_HOP_CFG_TEXT:
        _write_text_preserve_newlines(cfg_path, ONE_HOP_CFG_TEXT)
        print(f"[patch] wrote exact 1-hop ZINC config: {cfg_path}", flush=True)
    else:
        print(f"[patch] exact 1-hop ZINC config already present: {cfg_path}", flush=True)

    _replace_exact(
        repo_dir / "grit" / "config" / "gt_config.py",
        old=(
            '    cfg.gt.attn.full_attn = True\n'
            '    cfg.gt.attn.norm_e = True\n'
        ),
        new=(
            '    cfg.gt.attn.full_attn = True\n'
            '    cfg.gt.attn.sparsity = "full"\n'
            '    cfg.gt.attn.norm_e = True\n'
        ),
        marker='cfg.gt.attn.sparsity = "full"',
        label="default gt.attn.sparsity",
    )
    _replace_exact(
        repo_dir / "grit" / "encoder" / "rrwp_encoder.py",
        old=(
            '        torch.nn.init.xavier_uniform_(self.fc.weight)\n'
            '        self.fill_value = 0.\n'
        ),
        new=(
            '        torch.nn.init.xavier_uniform_(self.fc.weight)\n'
            '        self.pad_to_full_graph = False\n'
            '        self.fill_value = 0.\n'
        ),
        marker="self.pad_to_full_graph = False",
        label="masked RRWP encoder repr compatibility",
    )
    _replace_exact(
        repo_dir / "grit" / "encoder" / "rrwp_encoder.py",
        old=(
            '    def __repr__(self):\n'
            '        return f"{self.__class__.__name__}" \\\n'
            '               f"(pad_to_full_graph={self.pad_to_full_graph}," \\\n'
            '               f"fill_value={self.fill_value}," \\\n'
            '               f"{self.fc.__repr__()})"\n'
        ),
        new=(
            '    def __repr__(self):\n'
            '        return f"{self.__class__.__name__}" \\\n'
            '               f"(pad_to_full_graph={self.pad_to_full_graph}," \\\n'
            '               f"fill_value={self.fill_value}," \\\n'
            '               f"mask_index_name={self.mask_index_name}," \\\n'
            '               f"{self.fc.__repr__()})"\n'
        ),
        marker="mask_index_name={self.mask_index_name}",
        label="masked RRWP encoder repr mask label",
    )
    _replace_exact(
        repo_dir / "grit" / "network" / "grit_model.py",
        old=(
            '        if cfg.posenc_RRWP.enable:\n'
            '            self.rrwp_abs_encoder = register.node_encoder_dict["rrwp_linear"]\\\n'
            '                (cfg.posenc_RRWP.ksteps, cfg.gnn.dim_inner)\n'
            '            rel_pe_dim = cfg.posenc_RRWP.ksteps\n'
            '            self.rrwp_rel_encoder = register.edge_encoder_dict["rrwp_linear"] \\\n'
            '                (rel_pe_dim, cfg.gnn.dim_edge,\n'
            '                 pad_to_full_graph=cfg.gt.attn.full_attn,\n'
            '                 add_node_attr_as_self_loop=False,\n'
            '                 fill_value=0.\n'
            '                 )\n'
        ),
        new=(
            '        if cfg.posenc_RRWP.enable:\n'
            '            self.rrwp_abs_encoder = register.node_encoder_dict["rrwp_linear"]\\\n'
            '                (cfg.posenc_RRWP.ksteps, cfg.gnn.dim_inner)\n'
            '            rel_pe_dim = cfg.posenc_RRWP.ksteps\n'
            '            attn_sparsity = cfg.gt.attn.get("sparsity", "full")\n'
            '            if attn_sparsity == "full":\n'
            '                self.rrwp_rel_encoder = register.edge_encoder_dict["rrwp_linear"] \\\n'
            '                    (rel_pe_dim, cfg.gnn.dim_edge,\n'
            '                     pad_to_full_graph=cfg.gt.attn.full_attn,\n'
            '                     add_node_attr_as_self_loop=False,\n'
            '                     fill_value=0.\n'
            '                     )\n'
            '            elif attn_sparsity == "one_hop":\n'
            '                self.rrwp_rel_encoder = register.edge_encoder_dict["masked_rrwp_linear"] \\\n'
            '                    (rel_pe_dim, cfg.gnn.dim_edge,\n'
            '                     add_node_attr_as_self_loop=False,\n'
            '                     fill_value=0.,\n'
            '                     mask_index_name="edge_index",\n'
            '                     )\n'
            '            else:\n'
            '                raise ValueError(\n'
            '                    f"Unsupported cfg.gt.attn.sparsity={attn_sparsity!r}; "\n'
            '                    "expected \'full\' or \'one_hop\'."\n'
            '                )\n'
        ),
        marker='attn_sparsity = cfg.gt.attn.get("sparsity", "full")',
        label="GritTransformer RRWP sparsity switch",
    )
    write_json(
        drive_root / "patches" / "zinc_grit_rrwp_1hop_patch.json",
        {
            "official_repo": OFFICIAL_GRIT_REPO,
            "official_commit": OFFICIAL_GRIT_COMMIT,
            "one_hop_config": str(cfg_path),
            "parameter_count_guard": EXPECTED_GRIT_PARAMS,
            "scientific_change": "gt.attn.full_attn=False and gt.attn.sparsity=one_hop",
        },
    )


def prepare_grit_repos(repo_dir: Path, drive_root: Path, force: bool = False) -> tuple[Path, Path, Path, Path]:
    dense_repo = Path("/content/GRIT_dense_analysis")
    onehop_repo = Path("/content/GRIT_1hop_analysis")
    clone_official_grit(dense_repo, force=force)
    clone_official_grit(onehop_repo, force=force)

    apply_embedded_parameter_matched_onehop_patch(onehop_repo, drive_root)
    dense_cfg = dense_repo / "configs" / "GRIT" / "zinc-GRIT-RRWP.yaml"
    onehop_cfg = onehop_repo / "configs" / "GRIT" / "zinc-GRIT-RRWP-1hop.yaml"
    if not dense_cfg.exists():
        raise FileNotFoundError(f"missing dense GRIT config: {dense_cfg}")
    if not onehop_cfg.exists():
        raise FileNotFoundError(f"missing 1-hop GRIT config: {onehop_cfg}")
    return dense_repo, onehop_repo, dense_cfg, onehop_cfg


def checkpoint_candidates(root: Path) -> list[Path]:
    patterns = ["*.ckpt", "*checkpoint*.pt", "*checkpoint*.pth", "best*.pt", "best*.pth", "model*.pt"]
    out: list[Path] = []
    if root.exists():
        for pattern in patterns:
            out.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(set(out), key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)


def checkpoint_epoch(path: Path) -> int | None:
    text = str(path).lower()
    for pattern in (
        r"epoch[=_-]?(\d+)",
        r"ep[=_-]?(\d+)",
        r"ckpt[=_-]?(\d+)",
        r"/(\d+)\.ckpt$",
        r"/(\d+)\.(?:pt|pth)$",
    ):
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    try:
        return int(path.stem)
    except Exception:
        return None


def best_epoch_from_training_logs(source_drive_dir: Path, label: str) -> int | None:
    _, summary = parse_grit_training_logs(wrapper_logs(source_drive_dir), label)
    best_epoch = summary.get("best_epoch")
    return int(best_epoch) if best_epoch is not None else None


def checkpoint_from_audit(source_drive_dir: Path, *, allow_latest: bool = False) -> Path | None:
    for audit_path in [source_drive_dir / "latest_checkpoint_audit.json"]:
        if not audit_path.exists():
            continue
        try:
            with audit_path.open("r", encoding="utf-8") as f:
                audit = json.load(f)
        except Exception:
            continue
        for key in ("best_epoch_checkpoint_matches",):
            paths = audit.get(key) or []
            for raw_path in paths:
                path = Path(raw_path)
                if path.exists():
                    return path
        raw_latest = audit.get("latest_checkpoint_by_mtime") if allow_latest else None
        if raw_latest:
            latest = Path(raw_latest)
            if latest.exists():
                return latest
    return None


def best_named_checkpoint_candidates(candidates: Sequence[Path]) -> list[Path]:
    def score(path: Path) -> tuple[int, float, str]:
        lowered = str(path).lower()
        name = path.name.lower()
        if name in {"best.ckpt", "best.pt", "best.pth"}:
            priority = 0
        elif "best" in name:
            priority = 1
        elif "best" in lowered and "latest" not in name:
            priority = 2
        else:
            priority = 99
        return (priority, -path.stat().st_mtime, str(path))

    return [path for path in sorted(candidates, key=score) if score(path)[0] < 99]


def choose_checkpoint(results_root: Path, label: str) -> Path:
    candidates = checkpoint_candidates(results_root)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for {label} under {results_root}. "
            "The training run must have saved at least one best checkpoint first."
        )

    source_drive_dir = results_root.parent
    audited = checkpoint_from_audit(source_drive_dir, allow_latest=False)
    if audited is not None and audited in candidates:
        print(f"[checkpoint] {label}: {audited} (from latest_checkpoint_audit.json)", flush=True)
        return audited

    best_epoch = best_epoch_from_training_logs(source_drive_dir, label)
    if best_epoch is not None:
        matching = [path for path in candidates if checkpoint_epoch(path) == best_epoch]
        if matching:
            chosen = matching[0]
            print(f"[checkpoint] {label}: {chosen} (matched parsed best epoch {best_epoch})", flush=True)
            return chosen
        print(
            f"[checkpoint-warning] {label}: parsed best epoch {best_epoch}, "
            "but no checkpoint filename matched that epoch; falling back to newest checkpoint by mtime.",
            flush=True,
        )

    best_named = best_named_checkpoint_candidates(candidates)
    if best_named:
        chosen = best_named[0]
        print(f"[checkpoint] {label}: {chosen} (best-named checkpoint preferred over latest)", flush=True)
        return chosen

    audited_latest = checkpoint_from_audit(source_drive_dir, allow_latest=True)
    if audited_latest is not None and audited_latest in candidates:
        print(f"[checkpoint] {label}: {audited_latest} (latest checkpoint from audit fallback)", flush=True)
        return audited_latest

    chosen = candidates[0]
    print(f"[checkpoint] {label}: {chosen} (newest by mtime; epoch={checkpoint_epoch(chosen)})", flush=True)
    return chosen


def wrapper_logs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for base in [root / "wrapper_logs", root]:
        if base.exists():
            candidates.extend(path for path in base.rglob("*.log") if path.is_file())
            candidates.extend(path for path in base.rglob("*.out") if path.is_file())
    return sorted(set(candidates), key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)


def parse_grit_training_logs(log_paths: Sequence[Path], model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_epoch: dict[int, dict[str, Any]] = {}
    source_by_epoch: dict[int, str] = {}
    for path in log_paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for line in lines:
            stripped = line.strip()
            for split in ("train", "val", "test"):
                prefix = f"{split}: "
                if not stripped.startswith(prefix):
                    continue
                payload = stripped[len(prefix):]
                if not payload.startswith("{"):
                    continue
                try:
                    stats = ast.literal_eval(payload)
                    epoch = int(stats.get("epoch"))
                except Exception:
                    continue
                row = by_epoch.setdefault(epoch, {"model": model, "epoch": epoch})
                for key, value in stats.items():
                    if key == "epoch":
                        continue
                    out_key = f"{split}_{key}"
                    if out_key not in row:
                        row[out_key] = value
                source_by_epoch.setdefault(epoch, str(path))

    rows = []
    best_epoch: int | None = None
    best_val = float("inf")
    for epoch in sorted(by_epoch):
        row = by_epoch[epoch]
        row["source"] = source_by_epoch.get(epoch, "")
        rows.append(row)
        try:
            val = float(row.get("val_mae", row.get("val_loss", float("inf"))))
        except Exception:
            val = float("inf")
        if val < best_val:
            best_val = val
            best_epoch = epoch

    best_row = by_epoch.get(best_epoch, {}) if best_epoch is not None else {}
    summary = {
        "model": model,
        "num_rows": len(rows),
        "best_epoch": best_epoch,
        "best_val_mae": best_row.get("val_mae"),
        "best_val_loss": best_row.get("val_loss"),
        "best_test_mae": best_row.get("test_mae"),
        "best_test_loss": best_row.get("test_loss"),
        "log_files": [str(path) for path in log_paths[:20]],
    }
    return rows, summary


def prepare_model_artifact(
    *,
    model: str,
    source_drive_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    prepared_root: Path,
) -> dict[str, Any]:
    prepared_root.mkdir(parents=True, exist_ok=True)
    logs = wrapper_logs(source_drive_dir)
    rows, summary = parse_grit_training_logs(logs, model)
    metrics_csv = write_csv(prepared_root / "metrics" / "history_metrics.csv", rows)
    summary_json = write_json(prepared_root / "metrics" / "training_summary.json", summary)
    pointer = {
        "model": model,
        "source_drive_dir": str(source_drive_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "prepared_root": str(prepared_root),
        "metrics_csv": str(metrics_csv),
        "summary_json": str(summary_json),
        "logs_parsed": len(logs),
        "history_rows": len(rows),
    }
    write_json(prepared_root / "artifact_pointer.json", pointer)
    return pointer


def newest_existing(paths: Sequence[Path]) -> Path | None:
    existing = [path for path in paths if path.exists() and path.is_file()]
    return max(existing, key=lambda p: p.stat().st_mtime) if existing else None


def first_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.is_file():
            return path
    return None


def newest_files(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted((p for p in root.glob(pattern) if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)


def newest_rglob_files(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted((p for p in root.rglob(pattern) if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)


def discover_gin_artifact(gin_drive_dir: Path) -> dict[str, Path | None]:
    latest_outputs = gin_drive_dir / "latest_outputs"
    default_run_root = gin_drive_dir / "results" / "gin_zinc_500k_seed41"
    official_out_roots = [default_run_root / "official_out"]
    results_root = gin_drive_dir / "results"
    if results_root.exists():
        official_out_roots.extend(sorted(results_root.glob("*/official_out"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True))
    official_out_roots = list(dict.fromkeys(official_out_roots))
    checkpoint_candidates = [
        latest_outputs / "gin_zinc_seed41_best_checkpoint_pkl.pt",
        latest_outputs / "gin_zinc_seed41_best_checkpoint_pkl.pkl",
        *newest_files(latest_outputs, "*best*checkpoint*.pt"),
        *newest_files(latest_outputs, "*best*checkpoint*.pkl"),
        *newest_files(latest_outputs, "*best*state_dict*.pt"),
        *newest_files(latest_outputs, "*best*state_dict*.pkl"),
    ]
    for official_out in official_out_roots:
        checkpoint_candidates.extend(newest_rglob_files(official_out / "checkpoints", "best_val_mae_state_dict.pkl"))
    checkpoint_candidates.extend(
        [
            latest_outputs / "gin_zinc_seed41_latest_checkpoint_pkl.pt",
            latest_outputs / "gin_zinc_seed41_latest_checkpoint_pkl.pkl",
            *newest_files(latest_outputs, "*latest*checkpoint*.pt"),
            *newest_files(latest_outputs, "*latest*checkpoint*.pkl"),
            *newest_files(latest_outputs, "*latest*state_dict*.pt"),
            *newest_files(latest_outputs, "*latest*state_dict*.pkl"),
        ]
    )
    for official_out in official_out_roots:
        checkpoint_candidates.extend(newest_rglob_files(official_out / "checkpoints", "latest_state_dict.pkl"))
        checkpoint_candidates.extend(newest_rglob_files(official_out / "checkpoints", "*.pkl"))
    checkpoint = first_existing(
        checkpoint_candidates
    )
    config_candidates = [
        latest_outputs / "molecules_graph_regression_GIN_ZINC_500k.runtime.json",
        *newest_files(latest_outputs, "*.runtime.json"),
        *newest_files(latest_outputs, "*GIN*ZINC*.json"),
        *newest_files(gin_drive_dir, "**/*GIN*ZINC*.runtime.json"),
        default_run_root / "configs" / "molecules_graph_regression_GIN_ZINC_500k.runtime.json",
    ]
    for run_cfg_dir in sorted((gin_drive_dir / "results").glob("*/configs")) if (gin_drive_dir / "results").exists() else []:
        config_candidates.extend(newest_files(run_cfg_dir, "*.runtime.json"))
        config_candidates.extend(newest_files(run_cfg_dir, "*GIN*ZINC*.json"))
    config = newest_existing(
        config_candidates
    )
    history_candidates = [
        latest_outputs / "gin_zinc_seed41_history_csv.csv",
        *newest_files(latest_outputs, "*history*.csv"),
    ]
    best_json_candidates = [
        latest_outputs / "gin_zinc_seed41_best_json.json",
        *newest_files(latest_outputs, "*best*.json"),
    ]
    for official_out in official_out_roots:
        history_candidates.extend(newest_files(official_out / "results", "*_history.csv"))
        best_json_candidates.extend(newest_files(official_out / "results", "*_best.json"))
    history = newest_existing(
        history_candidates
    )
    best_json = newest_existing(
        best_json_candidates
    )
    raw_summary = newest_existing([latest_outputs / "training_summary.json", default_run_root / "training_summary.json"])
    return {
        "checkpoint": checkpoint,
        "config": config,
        "history": history,
        "best_json": best_json,
        "training_summary": raw_summary,
        "run_root": default_run_root if default_run_root.exists() else None,
        "official_out": next((root for root in official_out_roots if root.exists()), None),
        "searched_official_out_roots": [str(root) for root in official_out_roots],
        "checkpoint_candidates": [str(path) for path in checkpoint_candidates[:20]],
        "config_candidates": [str(path) for path in config_candidates[:20]],
    }


def prepare_gin_model_artifact(
    *,
    source_drive_dir: Path,
    repo_path: Path,
    artifact: Mapping[str, Path | None],
    prepared_root: Path,
) -> dict[str, Any] | None:
    checkpoint = artifact.get("checkpoint")
    config = artifact.get("config")
    if checkpoint is None or config is None:
        return None
    prepared_root.mkdir(parents=True, exist_ok=True)
    history_path = artifact.get("history")
    metrics_csv = None
    if history_path is not None and history_path.exists():
        metrics_csv = prepared_root / "metrics" / "history_metrics.csv"
        metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(history_path, metrics_csv)
    best_payload: dict[str, Any] = {}
    best_json = artifact.get("best_json")
    if best_json is not None and best_json.exists():
        try:
            best_payload = load_json(best_json)
        except Exception:
            best_payload = {}
    raw_summary_payload: dict[str, Any] = {}
    raw_summary = artifact.get("training_summary")
    if raw_summary is not None and raw_summary.exists():
        try:
            raw_summary_payload = load_json(raw_summary)
        except Exception:
            raw_summary_payload = {}
    summary = {
        "model": "gin",
        "adapter": "official_benchmarking_gnns_gin",
        "official_repo": OFFICIAL_GIN_REPO,
        "official_commit": OFFICIAL_GIN_COMMIT,
        "expected_parameter_count": EXPECTED_OFFICIAL_GIN_PARAMS,
        "observed_parameter_count": raw_summary_payload.get("observed_parameter_count"),
        "best_epoch": best_payload.get("best_epoch"),
        "best_val_mae": best_payload.get("best_val_mae"),
        "best_test_mae": best_payload.get("test_mae_at_best_val_epoch"),
        "best_train_mae": best_payload.get("train_mae_at_best_val_epoch"),
        "checkpoint_path": str(checkpoint),
        "config_path": str(config),
        "history_csv": str(history_path) if history_path is not None else None,
    }
    summary_json = write_json(prepared_root / "metrics" / "training_summary.json", summary)
    pointer = {
        "model": "gin",
        "adapter": "official_benchmarking_gnns_gin",
        "source_drive_dir": str(source_drive_dir),
        "repo_path": str(repo_path),
        "config_path": str(config),
        "checkpoint_path": str(checkpoint),
        "prepared_root": str(prepared_root),
        "metrics_csv": str(metrics_csv) if metrics_csv is not None else None,
        "summary_json": str(summary_json),
        "history_rows": "" if metrics_csv is None else "see_metrics_csv",
        "official_repo": OFFICIAL_GIN_REPO,
        "official_commit": OFFICIAL_GIN_COMMIT,
        "expected_parameter_count": EXPECTED_OFFICIAL_GIN_PARAMS,
    }
    write_json(prepared_root / "artifact_pointer.json", pointer)
    return pointer


def build_zinc_config(
    *,
    artifact_root: Path,
    dense_prepared: Path,
    onehop_prepared: Path,
    gin_prepared: Path | None,
    dense_dataset_dir: Path,
    onehop_dataset_dir: Path,
    gin_dataset_dir: Path | None,
    dense_repo: Path,
    onehop_repo: Path,
    gin_repo: Path | None,
    dense_cfg: Path,
    onehop_cfg: Path,
    gin_cfg: Path | None,
    dense_ckpt: Path,
    onehop_ckpt: Path,
    gin_ckpt: Path | None,
    single_seed: int,
    include_local_references: bool,
) -> dict[str, Any]:
    models: dict[str, Any] = {
        "dense_grit": {
            "adapter": "official_grit",
            "variant": "official",
            "role": "treatment",
            "official_repo": "https://github.com/LiamMa/GRIT",
            "repo_path": str(dense_repo),
            "artifact_root": str(dense_prepared),
            "dataset_dir": str(dense_dataset_dir),
            "config_path": str(dense_cfg),
            "checkpoint_path": str(dense_ckpt),
        },
        "grit_1hop": {
            "adapter": "official_grit",
            "variant": "1hop",
            "role": "parameter_matched_control",
            "official_repo": "https://github.com/LiamMa/GRIT",
            "repo_path": str(onehop_repo),
            "artifact_root": str(onehop_prepared),
            "dataset_dir": str(onehop_dataset_dir),
            "config_path": str(onehop_cfg),
            "checkpoint_path": str(onehop_ckpt),
        },
    }
    if gin_prepared is not None and gin_repo is not None and gin_cfg is not None and gin_ckpt is not None:
        models["gin"] = {
            "adapter": "official_benchmarking_gnns_gin",
            "role": "local_validation_reference",
            "official_repo": OFFICIAL_GIN_REPO,
            "official_commit": OFFICIAL_GIN_COMMIT,
            "device": "cpu",
            "repo_path": str(gin_repo),
            "artifact_root": str(gin_prepared),
            "dataset_dir": str(gin_dataset_dir or dense_dataset_dir),
            "config_path": str(gin_cfg),
            "checkpoint_path": str(gin_ckpt),
        }
    if include_local_references:
        if "gin" not in models:
            models["gin"] = {
                "adapter": "pyg_gin",
                "role": "local_validation_reference",
                "artifact_root": str(artifact_root / "missing_gin_reference"),
                "dataset_dir": str(dense_dataset_dir),
            }
        models["gcn"] = {
            "adapter": "pyg_gcn",
            "role": "local_validation_reference",
            "artifact_root": str(artifact_root / "missing_gcn_reference"),
        }

    step5_reference_models = ["dense_grit", "grit_1hop"]
    if "gin" in models:
        step5_reference_models.append("gin")
    if "gcn" in models:
        step5_reference_models.append("gcn")

    return {
        "artifact_root": str(artifact_root / "artifacts"),
        "dataset": {"name": "ZINC", "split": "official_subset", "task": "molecular_regression"},
        "seeds": [int(single_seed)],
        "primary_tau": 3,
        "far_thresholds": [2, 3, 4],
        "perturbation": {
            "carriage_primary": "integrated_gradients",
            "ig_baseline": "mean_node_embedding",
            "ig_steps": 32,
            "baseline_sample_graphs": 200,
            "swap_partners": 8,
            "batched_vjp": True,
            "swap_partner_policy": "different_type",
        },
        "models": models,
        "runtime": {"skip_failed_optional_adapters": True},
        "steps": {
            "0": {
                "name": "measurement_model_validation",
                "sample_graphs": 200,
                "ig_step_sweep": [16, 32, 64, 128, 256],
                "diagnostic_sample_graphs": 16,
                "baseline_sweep": ["mean_node_embedding", "zero_embedding"],
                "matched_target_ig_steps": 64,
                "matched_target_sources_per_graph": 4,
                "matched_target_partners_per_source": 2,
            },
            "1": {"name": "performance_gap", "reach_sweep": [1, 2, 3, 5, "dense"]},
            "2": {
                "name": "usage_vs_causal_usage",
                "sample_graphs": 200,
                "compare_attention_to_swaps": True,
                "run_layer_channel_split": True,
            },
            "3": {"name": "distance_resolved_overfitting", "sample_graphs": 200},
            "4": {
                "name": "mediator_patching",
                "sample_graphs": 100,
                "max_far_pairs_per_graph": 64,
                "depth_pairs_per_graph": 8,
                "all_distance": True,
                "min_distance": 2,
                "stratify_by_distance": True,
                "onset_fraction_threshold": 0.50,
                "min_effect_abs": 1.0e-6,
                "signal_gate": True,
                "signal_gate_quantile": 0.90,
                "clamp_mode": "detach",
                "run_analytic_patching_check": True,
                "run_clamp_negative_control": True,
                "composed_reference_max_direct_fraction": 0.20,
            },
            "5": {
                "name": "non_composable_gap_attribution",
                "sample_graphs": 200,
                "max_far_pairs_per_graph": "all",
                "interaction_pairs": 1000,
                "min_effect_abs": 1.0e-6,
                "signal_gate": True,
                "signal_gate_quantile": 0.90,
                "clamp_mode": "detach",
                "reference_models": step5_reference_models,
            },
        },
        "figures": {"dpi": 180},
        "colab_notes": {
            "requested_single_seed_only": True,
            "markdown_default_requires_three_seeds": True,
            "dense_default_drive_dir": DEFAULT_DENSE_DRIVE_DIR,
            "onehop_default_drive_dir": DEFAULT_ONEHOP_DRIVE_DIR,
            "gin_default_drive_dir": DEFAULT_GIN_DRIVE_DIR,
        },
    }


def write_yaml(path: Path, payload: Mapping[str, Any]) -> Path:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def run_main_procedure(
    repo_dir: Path,
    config_path: Path,
    *,
    steps: str,
    force: bool,
    analysis_preset: str,
    fast_dev_run: bool,
    dry_run: bool,
    env: Mapping[str, str] | None = None,
) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "graph_specialisation_metrics.main_procedure",
        "run",
        "--config",
        str(config_path),
        "--steps",
        str(steps),
        "--analysis-preset",
        str(analysis_preset),
    ]
    if force:
        cmd.append("--force")
    if fast_dev_run:
        cmd.append("--fast-dev-run")
    if dry_run:
        cmd.append("--dry-run")
    proc = run_cmd(cmd, cwd=repo_dir, check=True, env=env, stream=True)
    match = re.search(r"\[done\] main procedure artifacts: (.+)", proc.stdout)
    if not match:
        raise RuntimeError("main_procedure did not report an artifact root")
    return Path(match.group(1).strip())


def run_onehop_locality_preflight(
    repo_dir: Path,
    config_path: Path,
    drive_root: Path,
    *,
    sample_graphs: int,
    tolerance: float,
    env: Mapping[str, str] | None = None,
) -> Path:
    output = drive_root / "preflight" / "onehop_locality_certification.json"
    cmd = [
        sys.executable,
        "-m",
        "graph_specialisation_metrics.main_procedure",
        "verify-onehop",
        "--config",
        str(config_path),
        "--output",
        str(output),
        "--sample-graphs",
        str(int(sample_graphs)),
        "--tolerance",
        str(float(tolerance)),
    ]
    print("[preflight] certifying 1-hop GRIT direct attention/routing locality", flush=True)
    run_cmd(cmd, cwd=repo_dir, check=True, env=env, stream=True)
    print(f"[preflight] 1-hop locality certification written to {output}", flush=True)
    return output


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def assess_completion(artifact_root: Path) -> dict[str, Any]:
    status_path = artifact_root / "metrics" / "main_status.json"
    adapter_path = artifact_root / "metrics" / "adapter_checks.json"
    statuses = load_json(status_path) if status_path.exists() else {}
    adapter_checks = load_json(adapter_path) if adapter_path.exists() else {}
    incomplete = {
        step: status
        for step, status in statuses.items()
        if isinstance(status, Mapping)
        and str(status.get("status"))
        in {"requires_trained_model_intervention_hooks", "waiting_for_training_artifacts", "waiting_for_official_grit_artifacts", "failed"}
    }
    figures = sorted(str(path) for path in (artifact_root / "figures").glob("*") if path.is_file())
    return {
        "artifact_root": str(artifact_root),
        "status_by_step": statuses,
        "adapter_checks": adapter_checks,
        "incomplete_steps": incomplete,
        "figures": figures,
        "complete": not incomplete and all(str(i) in statuses for i in range(6)),
    }


def write_methodology_fidelity_audit(
    path: Path,
    *,
    config: Mapping[str, Any],
    completion: Mapping[str, Any],
    include_local_references: bool,
) -> None:
    notes = []
    if config.get("seeds") != [0]:
        notes.append("Single-seed run requested; markdown default requires three seeds for paper claims.")
    else:
        notes.append("Single-seed run requested by user; this deliberately deviates from the markdown's 3-seed final protocol.")
    model_names = set((config.get("models") or {}).keys())
    if not include_local_references:
        if "gin" in model_names:
            notes.append("Official Benchmarking-GNNs GIN ZINC reference included from default Drive artifacts; GCN remains omitted until trained/wired.")
        else:
            notes.append("GCN/GIN local validation references omitted because trained reference checkpoints were not available/requested.")
    if completion.get("incomplete_steps"):
        notes.append(
            "One or more required steps did not complete. Inspect completion.status_by_step for missing checkpoints, "
            "dependency errors, or failed intervention execution."
        )
    audit = {
        "markdown_steps_required": MARKDOWN_REQUIRED_STEPS,
        "config_steps": config.get("steps"),
        "config_primary_tau": config.get("primary_tau"),
        "config_far_thresholds": config.get("far_thresholds"),
        "config_perturbation": config.get("perturbation"),
        "models": config.get("models"),
        "completion": completion,
        "fidelity_notes": notes,
    }
    write_json(path, audit)


def copy_latest_outputs(drive_root: Path, artifact_root: Path) -> Path:
    latest = drive_root / "latest_outputs"
    if latest.exists():
        shutil.rmtree(latest)
    latest.mkdir(parents=True, exist_ok=True)
    for subdir in ["figures", "metrics"]:
        source = artifact_root / subdir
        if source.exists():
            shutil.copytree(source, latest / subdir)
    for name in ["manifest.json", "config.yaml"]:
        source = artifact_root / name
        if source.exists():
            shutil.copy2(source, latest / name)
    return latest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-url", default=PUBLIC_REPO_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--repo-dir", type=Path, default=Path(DEFAULT_REPO_DIR))
    parser.add_argument("--github-username", default=None)
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--drive-root", type=Path, default=Path(DEFAULT_DRIVE_ROOT))
    parser.add_argument("--dense-drive-dir", type=Path, default=Path(DEFAULT_DENSE_DRIVE_DIR))
    parser.add_argument("--onehop-drive-dir", type=Path, default=Path(DEFAULT_ONEHOP_DRIVE_DIR))
    parser.add_argument("--gin-drive-dir", type=Path, default=Path(DEFAULT_GIN_DRIVE_DIR))
    parser.add_argument("--gin-repo-dir", type=Path, default=Path(DEFAULT_GIN_REPO_DIR))
    parser.add_argument("--single-seed", type=int, default=0)
    parser.add_argument(
        "--prepared-id",
        default="zinc_single_seed_current",
        help="Stable prepared-artifact namespace. Keep fixed across Colab restarts to resume cached analysis.",
    )
    parser.add_argument(
        "--refresh-model-artifacts",
        action="store_true",
        help="Rediscover checkpoints and rewrite prepared model pointers. Omit this during resume so the config hash remains stable.",
    )
    parser.add_argument("--pyg-version", default=DEFAULT_PYG_VERSION)
    parser.add_argument("--steps", default="all", help="Main procedure steps to run, e.g. all or 0 or 0,2,4.")
    parser.add_argument(
        "--analysis-preset",
        choices=["full", "medium", "pilot", "smoke"],
        default="full",
        help="Bound expensive intervention counts. Use pilot/medium for analysis runs before full paper settings.",
    )
    parser.add_argument("--fast-dev-run", action="store_true", help="Use the main procedure fast-dev overrides for a quicker smoke run.")
    parser.add_argument("--force", action="store_true", help="Force rerun of main_procedure artifact generation instead of resuming completed steps.")
    parser.add_argument("--force-official-grit-reclone", action="store_true")
    parser.add_argument("--force-official-gin-reclone", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only run main_procedure discovery/status mode.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Do not raise if checkpoints/dependencies are missing or a preflight is requested.")
    parser.add_argument("--include-local-references", action="store_true", help="Include placeholder GCN/GIN entries required by the markdown gate.")
    parser.add_argument("--skip-gin-reference", action="store_true", help="Do not include the default official Benchmarking-GNNs GIN ZINC reference even if present on Drive.")
    parser.add_argument(
        "--allow-missing-gin-reference",
        action="store_true",
        default=True,
        help="Allow the run to continue if the default trained GIN reference cannot be discovered. Default: true.",
    )
    parser.add_argument(
        "--require-gin-reference",
        dest="allow_missing_gin_reference",
        action="store_false",
        help="Fail if the trained official Benchmarking-GNNs GIN reference is missing or cannot be loaded.",
    )
    parser.add_argument("--skip-onehop-locality-check", action="store_true", help="Skip the hard preflight that certifies the 1-hop control is local.")
    parser.add_argument("--onehop-locality-check-graphs", type=int, default=4, help="Number of ZINC test graphs used for the 1-hop locality preflight.")
    parser.add_argument("--onehop-locality-tolerance", type=float, default=1.0e-12, help="Allowed direct attention mass at molecular distance > 1.")
    parser.add_argument("--skip-git", action="store_true", help="Use an already-cloned repo-dir.")
    argv = list(sys.argv[1:] if argv is None else argv)
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "-f" and i + 1 < len(argv) and "kernel-" in argv[i + 1] and argv[i + 1].endswith(".json"):
            print(f"[args] ignoring notebook launcher arguments: {argv[i:i+2]}", flush=True)
            i += 2
            continue
        cleaned.append(argv[i])
        i += 1
    return parser.parse_args(cleaned)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    mount_drive()
    args.drive_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_git:
        token = require_colab_token(args.secret_name)
        clone_or_update_repo(args.repo_url, args.branch, args.repo_dir, token, args.github_username)
    else:
        print(f"[git] using existing repo: {args.repo_dir}", flush=True)
    compat_shim_dir = write_py312_compat_shim(args.drive_root)
    analysis_env = env_with_py312_compat(compat_shim_dir)
    install_repo(args.repo_dir, pyg_version=str(args.pyg_version))

    dense_repo, onehop_repo, dense_cfg, onehop_cfg = prepare_grit_repos(
        args.repo_dir,
        args.drive_root,
        force=bool(args.force_official_grit_reclone),
    )
    prepared = args.drive_root / "prepared_model_artifacts" / str(args.prepared_id)
    print(f"[prepared] using stable prepared artifact root: {prepared}", flush=True)
    dense_pointer_path = prepared / "dense_grit" / "artifact_pointer.json"
    onehop_pointer_path = prepared / "grit_1hop" / "artifact_pointer.json"
    gin_pointer_path = prepared / "gin" / "artifact_pointer.json"
    reuse_prepared = (
        not args.refresh_model_artifacts
        and dense_pointer_path.exists()
        and onehop_pointer_path.exists()
    )
    if reuse_prepared:
        dense_pointer = load_json(dense_pointer_path)
        onehop_pointer = load_json(onehop_pointer_path)
        dense_ckpt = Path(str(dense_pointer["checkpoint_path"]))
        onehop_ckpt = Path(str(onehop_pointer["checkpoint_path"]))
        if dense_ckpt.exists() and onehop_ckpt.exists():
            print(f"[prepared] reusing dense checkpoint: {dense_ckpt}", flush=True)
            print(f"[prepared] reusing 1-hop checkpoint: {onehop_ckpt}", flush=True)
        else:
            print("[prepared-warning] saved checkpoint pointer missing on Drive; refreshing model artifacts", flush=True)
            reuse_prepared = False
    if not reuse_prepared:
        dense_results = args.dense_drive_dir / "results"
        onehop_results = args.onehop_drive_dir / "results"
        dense_ckpt = choose_checkpoint(dense_results, "dense_grit")
        onehop_ckpt = choose_checkpoint(onehop_results, "grit_1hop")
        dense_pointer = prepare_model_artifact(
            model="dense_grit",
            source_drive_dir=args.dense_drive_dir,
            config_path=dense_cfg,
            checkpoint_path=dense_ckpt,
            prepared_root=prepared / "dense_grit",
        )
        onehop_pointer = prepare_model_artifact(
            model="grit_1hop",
            source_drive_dir=args.onehop_drive_dir,
            config_path=onehop_cfg,
            checkpoint_path=onehop_ckpt,
            prepared_root=prepared / "grit_1hop",
        )

    gin_pointer: dict[str, Any] | None = None
    gin_repo: Path | None = None
    gin_cfg: Path | None = None
    gin_ckpt: Path | None = None
    if not args.skip_gin_reference and not args.refresh_model_artifacts and gin_pointer_path.exists():
        candidate = load_json(gin_pointer_path)
        candidate_ckpt = Path(str(candidate.get("checkpoint_path", "")))
        candidate_cfg = Path(str(candidate.get("config_path", "")))
        candidate_repo = Path(str(candidate.get("repo_path", args.gin_repo_dir)))
        if candidate_ckpt.exists() and candidate_cfg.exists() and candidate_repo.exists():
            gin_pointer = candidate
            gin_ckpt = candidate_ckpt
            gin_cfg = candidate_cfg
            gin_repo = candidate_repo
            print(f"[prepared] reusing GIN checkpoint: {gin_ckpt}", flush=True)
        else:
            print("[prepared-warning] saved GIN pointer missing checkpoint/config/repo; rediscovering GIN artifacts", flush=True)
    if not args.skip_gin_reference and gin_pointer is None:
        gin_artifact = discover_gin_artifact(args.gin_drive_dir)
        if gin_artifact.get("checkpoint") is not None and gin_artifact.get("config") is not None:
            print(
                "[gin-discovery] "
                f"checkpoint={gin_artifact.get('checkpoint')} "
                f"config={gin_artifact.get('config')} "
                f"history={gin_artifact.get('history')} "
                f"best_json={gin_artifact.get('best_json')}",
                flush=True,
            )
            gin_repo = clone_official_gin(args.gin_repo_dir, force=bool(args.force_official_gin_reclone))
            gin_pointer = prepare_gin_model_artifact(
                source_drive_dir=args.gin_drive_dir,
                repo_path=gin_repo,
                artifact=gin_artifact,
                prepared_root=prepared / "gin",
            )
            if gin_pointer is not None:
                gin_ckpt = Path(str(gin_pointer["checkpoint_path"]))
                gin_cfg = Path(str(gin_pointer["config_path"]))
                print(f"[prepared] using official GIN checkpoint: {gin_ckpt}", flush=True)
        else:
            details = {
                "status": "missing_gin_reference",
                "gin_drive_dir": str(args.gin_drive_dir),
                "checkpoint": str(gin_artifact.get("checkpoint")) if gin_artifact.get("checkpoint") is not None else None,
                "config": str(gin_artifact.get("config")) if gin_artifact.get("config") is not None else None,
                "searched_official_out_roots": gin_artifact.get("searched_official_out_roots", []),
                "checkpoint_candidates": gin_artifact.get("checkpoint_candidates", []),
                "config_candidates": gin_artifact.get("config_candidates", []),
            }
            write_json(args.drive_root / "preflight" / "gin_reference_discovery_failed.json", details)
            message = f"No complete official GIN artifact found under {args.gin_drive_dir}."
            if not args.allow_missing_gin_reference and not args.include_local_references:
                raise RuntimeError(
                    message
                    + f" Wrote discovery details to {args.drive_root / 'preflight' / 'gin_reference_discovery_failed.json'}."
                    + " Pass --allow-missing-gin-reference to continue without GIN, or --skip-gin-reference to intentionally omit it."
                )
            print("[gin] optional reference unavailable; continuing with dense_grit and grit_1hop only.", flush=True)

    cfg = build_zinc_config(
        artifact_root=args.drive_root,
        dense_prepared=prepared / "dense_grit",
        onehop_prepared=prepared / "grit_1hop",
        gin_prepared=(prepared / "gin") if gin_pointer is not None else None,
        dense_dataset_dir=args.dense_drive_dir / "datasets",
        onehop_dataset_dir=args.onehop_drive_dir / "datasets",
        gin_dataset_dir=args.dense_drive_dir / "datasets",
        dense_repo=dense_repo,
        onehop_repo=onehop_repo,
        gin_repo=gin_repo,
        dense_cfg=dense_cfg,
        onehop_cfg=onehop_cfg,
        gin_cfg=gin_cfg,
        dense_ckpt=dense_ckpt,
        onehop_ckpt=onehop_ckpt,
        gin_ckpt=gin_ckpt,
        single_seed=args.single_seed,
        include_local_references=bool(args.include_local_references),
    )
    if not args.allow_missing_gin_reference:
        cfg.setdefault("runtime", {})["skip_failed_optional_adapters"] = False
    config_path = write_yaml(args.drive_root / "configs" / "zinc_main_procedure_colab.yaml", cfg)
    model_names = sorted((cfg.get("models") or {}).keys())
    print(f"[config] models included: {', '.join(model_names)}", flush=True)
    if "gin" not in model_names and not args.skip_gin_reference and not args.allow_missing_gin_reference:
        raise RuntimeError(
            "GIN was not included in the generated methodology config. "
            "This should only happen with --skip-gin-reference or default optional-GIN handling."
        )
    write_json(
        args.drive_root / "prepared_model_artifacts" / "latest_pointers.json",
        {
            "dense_grit": dense_pointer,
            "grit_1hop": onehop_pointer,
            "gin": gin_pointer,
            "config_path": str(config_path),
        },
    )

    print(f"[config] wrote {config_path}", flush=True)
    if not args.skip_onehop_locality_check:
        try:
            run_onehop_locality_preflight(
                args.repo_dir,
                config_path,
                args.drive_root,
                sample_graphs=int(args.onehop_locality_check_graphs),
                tolerance=float(args.onehop_locality_tolerance),
                env=analysis_env,
            )
        except Exception as exc:
            write_json(
                args.drive_root / "preflight" / "onehop_locality_certification_failed.json",
                {"status": "failed", "error": str(exc), "config_path": str(config_path)},
            )
            if not args.allow_incomplete:
                raise
            print(f"[preflight-warning] 1-hop locality certification failed but --allow-incomplete is set: {exc}", flush=True)
    artifact_root = run_main_procedure(
        args.repo_dir,
        config_path,
        steps=str(args.steps),
        force=bool(args.force),
        analysis_preset=str(args.analysis_preset),
        fast_dev_run=bool(args.fast_dev_run),
        dry_run=bool(args.dry_run),
        env=analysis_env,
    )
    completion = assess_completion(artifact_root)
    write_methodology_fidelity_audit(
        args.drive_root / "methodology_fidelity_audit.json",
        config=cfg,
        completion=completion,
        include_local_references=bool(args.include_local_references),
    )
    latest = copy_latest_outputs(args.drive_root, artifact_root)

    print("[completion]", json.dumps(completion, indent=2, sort_keys=True), flush=True)
    print(f"[done] artifacts: {artifact_root}", flush=True)
    print(f"[done] latest outputs: {latest}", flush=True)
    print(f"[done] fidelity audit: {args.drive_root / 'methodology_fidelity_audit.json'}", flush=True)

    if completion.get("incomplete_steps") and not args.allow_incomplete:
        incomplete_summary = {
            step: {
                "status": status.get("status"),
                "name": status.get("name"),
                "error": status.get("error"),
            }
            for step, status in completion.get("incomplete_steps", {}).items()
            if isinstance(status, Mapping)
        }
        raise SystemExit(
            "The current repo did not complete the full markdown procedure. "
            f"Incomplete steps: {json.dumps(incomplete_summary, sort_keys=True)}. "
            "Artifacts were written; inspect methodology_fidelity_audit.json for full details. "
            "Pass --allow-incomplete for discovery/preflight only."
        )


if __name__ == "__main__":
    main()
