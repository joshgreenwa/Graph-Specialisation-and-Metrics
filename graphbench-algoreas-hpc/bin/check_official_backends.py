#!/usr/bin/env python3
"""Preflight checks for official model backends.

This script is intentionally import-focused. It should be run on the HPC
environment before launching training arrays, after the official repos and
compiled PyG dependencies are installed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


EXPECTED_COMMITS = {
    "graphormer": "ac154fe4253d076a1c294f14be20dad0351cff3c",
    "graphgps": "28015707cbab7f8ad72bed0ee872d068ea59c94b",
    "grit": "6c988ea600a606fbb49a2246c64a2d37396b3ab5",
    "gnnplus": "0e02ad9acc2f1e54b5ad71c051bf5dfb1fcb4f28",
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


def default_project_root() -> Path:
    return Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()


def git_commit(path: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return out.strip()
    except Exception:
        return None


def check_import(module_name: str) -> CheckResult:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "no_version")
        return CheckResult(module_name, True, str(version))
    except Exception as exc:
        return CheckResult(module_name, False, f"{type(exc).__name__}: {exc}")


def check_file_import(label: str, path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult(label, False, f"missing: {path}")
    code = """
import importlib.util
import json
import sys
path = sys.argv[1]
name = sys.argv[2]
spec = importlib.util.spec_from_file_location(name, path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load spec for {path}")
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
print(json.dumps({"ok": True}))
"""
    try:
        out = subprocess.check_output(
            [sys.executable, "-c", code, str(path), f"_preflight_{label.replace('.', '_')}"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        detail = json.loads(out.strip().splitlines()[-1])
        return CheckResult(label, bool(detail["ok"]), str(path))
    except subprocess.CalledProcessError as exc:
        lines = (exc.output or "").strip().splitlines()
        return CheckResult(label, False, lines[-1] if lines else f"exit code {exc.returncode}")
    except Exception as exc:
        return CheckResult(label, False, f"{type(exc).__name__}: {exc}")


def add_path(path: Path) -> None:
    text = str(path.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


def check_repo(name: str, path: Path, expected_commit: str) -> list[CheckResult]:
    results: list[CheckResult] = []
    if not path.exists():
        return [CheckResult(f"{name}.path", False, f"missing: {path}")]
    results.append(CheckResult(f"{name}.path", True, str(path)))
    commit = git_commit(path)
    if commit is None:
        results.append(CheckResult(f"{name}.git", False, "not a readable git repository"))
    elif commit != expected_commit:
        results.append(CheckResult(f"{name}.commit", False, f"{commit} != expected {expected_commit}"))
    else:
        results.append(CheckResult(f"{name}.commit", True, commit))
    return results


def check_graphormer(path: Path) -> list[CheckResult]:
    results = check_repo("graphormer", path, EXPECTED_COMMITS["graphormer"])
    fairseq_dir = path / "fairseq"
    if (fairseq_dir / "fairseq").exists() or (fairseq_dir / "setup.py").exists():
        results.append(CheckResult("graphormer.fairseq_submodule", True, str(fairseq_dir)))
        add_path(fairseq_dir)
    else:
        results.append(
            CheckResult(
                "graphormer.fairseq_submodule",
                False,
                "missing or empty; clone Graphormer with --recurse-submodules or run git submodule update --init --recursive",
            )
        )
    add_path(path)
    for module_name in [
        "graphormer.modules.graphormer_layers",
        "graphormer.modules.graphormer_graph_encoder_layer",
        "graphormer.modules.graphormer_graph_encoder",
    ]:
        results.append(check_import(module_name))
    return results


def check_graphgps(path: Path) -> list[CheckResult]:
    results = check_repo("graphgps", path, EXPECTED_COMMITS["graphgps"])
    add_path(path)
    for module_name in [
        "graphgps.encoder.kernel_pos_encoder",
        "graphgps.layer.gps_layer",
        "graphgps.network.gps_model",
    ]:
        results.append(check_import(module_name))
    return results


def check_grit(path: Path) -> list[CheckResult]:
    results = check_repo("grit", path, EXPECTED_COMMITS["grit"])
    add_path(path)
    for module_name in [
        "grit.encoder.rrwp_encoder",
        "grit.layer.grit_layer",
    ]:
        results.append(check_import(module_name))
    return results


def check_gnnplus(path: Path) -> list[CheckResult]:
    results = check_repo("gnnplus", path, EXPECTED_COMMITS["gnnplus"])
    pkg = path / "GNNPlus" if (path / "GNNPlus").exists() else path
    file_checks = {
        "GNNPlus.encoder.kernel_pos_encoder": pkg / "encoder" / "kernel_pos_encoder.py",
        "GNNPlus.layer.gcn_conv_layer": pkg / "layer" / "gcn_conv_layer.py",
        "GNNPlus.layer.gine_conv_layer": pkg / "layer" / "gine_conv_layer.py",
        "GNNPlus.layer.gatedgcn_layer": pkg / "layer" / "gatedgcn_layer.py",
    }
    for label, file_path in file_checks.items():
        results.append(check_file_import(label, file_path))
    return results


def resolve_existing_path(path: Path, alternatives: list[Path]) -> Path:
    if path.exists():
        return path
    for alt in alternatives:
        if alt.exists():
            return alt
    return path


def parse_args() -> argparse.Namespace:
    root = default_project_root()
    external = root / "external"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="graphgps,static_grit,grit,gatedgcn_plus,gin_plus,gcn_plus")
    parser.add_argument("--graphormer-path", type=Path, default=Path(os.environ.get("GRAPHORMER_ROOT", external / "Graphormer")))
    parser.add_argument("--graphgps-path", type=Path, default=Path(os.environ.get("GRAPHGPS_ROOT", external / "GraphGPS")))
    parser.add_argument("--grit-path", type=Path, default=Path(os.environ.get("GRIT_ROOT", external / "GRIT")))
    parser.add_argument("--gnnplus-path", type=Path, default=Path(os.environ.get("GNNPLUS_ROOT", external / "GNNPlus")))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_models = {part.strip() for part in args.models.replace(";", ",").split(",") if part.strip()}
    needs_graphormer = "graphormer" in selected_models
    needs_graphgps = "graphgps" in selected_models
    needs_grit = bool({"grit", "static_grit"} & selected_models)
    needs_gnnplus = bool({"gcn_plus", "gin_plus", "gatedgcn_plus"} & selected_models)
    checks: list[CheckResult] = []
    checks.append(CheckResult("python", True, sys.version.replace("\n", " ")))
    base_modules = ["torch", "torch_geometric", "torch_scatter", "yacs", "wandb", "graphbench"]
    if needs_graphgps:
        base_modules.append("performer_pytorch")
    if needs_grit:
        base_modules.extend(["torch_sparse", "ogb", "opt_einsum"])
    for module_name in base_modules:
        checks.append(check_import(module_name))
    pyg_lib = check_import("pyg_lib")
    pyg_lib.required = False
    checks.append(pyg_lib)
    if needs_graphormer:
        checks.extend(check_graphormer(args.graphormer_path))
    if needs_graphgps:
        checks.extend(check_graphgps(args.graphgps_path))
    if needs_grit:
        checks.extend(check_grit(args.grit_path))
    if needs_gnnplus:
        gnnplus_path = resolve_existing_path(
            args.gnnplus_path,
            [default_project_root() / "external" / "tunedGNN-G"],
        )
        checks.extend(check_gnnplus(gnnplus_path))

    ok = True
    for result in checks:
        prefix = "OK" if result.ok else ("FAIL" if result.required else "WARN")
        print(f"[{prefix}] {result.name}: {result.detail}")
        ok = ok and (result.ok or not result.required)
    if not ok:
        print("\nOfficial backend preflight failed. Do not launch paper training arrays until all FAIL rows are resolved.")
        return 1
    print("\nOfficial backend preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
