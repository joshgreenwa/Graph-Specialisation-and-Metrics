# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Standalone Colab runner for the official GIN+ ZINC experiment.

This script does not reimplement or patch GIN+. It clones the official
LUOyk1999/GNNPlus repository, pins it to a verified commit, verifies the
official configs/gine/zinc.yaml byte-for-byte, and launches the repository's
own main.py once per seed.

The paper reports five independent runs. The official main.py starts at seed 0
and increments by one, so the paper-faithful default schedule used here is
0,1,2,3,4. The paper does not separately enumerate the numeric seed IDs; the
official README's ZINC example requests only two repeats. Pass --seeds 0,1 if
you specifically want the README example rather than the five-run paper
protocol.

Default Drive output root:

    /content/drive/MyDrive/ginplus_zinc_official

Each seed gets its own Drive-backed results directory, including official raw
logs, split statistics, subset_result.txt, and the final official checkpoint.
Completed seeds are skipped. Because the exact official config has
train.auto_resume=False, an interrupted seed is archived and restarted from
epoch 0 rather than changing the training trajectory with a non-official resume
override.

Paste this whole file into a Colab cell and run it. The executable block at the
bottom calls main([...]) with the paper-faithful defaults. You may also upload
or import it and call, for example:

    main([])
    main(["--skip-install"])
    main(["--seeds", "0,1"])
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Mapping, Sequence


OFFICIAL_REPO = "https://github.com/LUOyk1999/GNNPlus.git"
OFFICIAL_COMMIT = "0e02ad9acc2f1e54b5ad71c051bf5dfb1fcb4f28"
OFFICIAL_CONFIG_REL = Path("configs/gine/zinc.yaml")
OFFICIAL_CONFIG_SHA256 = (
    "894816439156cf82b18212f52aa5947f62d73a18e18af9ba750d160e648d9320"
)
EXPECTED_PARAMS = 477_241
PAPER_REFERENCE_MEAN = 0.065
PAPER_REFERENCE_SD = 0.004

ENVIRONMENT_SPEC = {
    "python": "3.10",
    "torch": "2.2.0+cu118",
    "torchvision": "0.17.0+cu118",
    "torchaudio": "2.2.0+cu118",
    "torch_geometric": "2.3.1",
    "scikit_learn": "1.4.0",
    "numpy": "1.26.4",
    "pyg_lib": "0.4.0+pt22cu118",
    "torch_scatter": "2.1.2+pt22cu118",
    "torch_sparse": "0.6.18+pt22cu118",
    "torch_cluster": "1.6.3+pt22cu118",
    "torch_spline_conv": "1.2.2+pt22cu118",
}

REQUIRED_CONFIG_FRAGMENTS = (
    "format: PyG-ZINC",
    "name: subset",
    "node_encoder_name: TypeDictNode+RWSE",
    "times_func: range(1,21)",
    "dim_pe: 28",
    "batch_size: 32",
    "layer_type: gine",
    "layers_mp: 12",
    "dim_inner: 80",
    "ffn: True",
    "residual: True",
    "base_lr: 0.001",
    "max_epoch: 2000",
    "num_warmup_epochs: 50",
    "weight_decay: 1e-5",
)


class CommandError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, flush=True)


def run_cmd(
    cmd: Sequence[object],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [str(item) for item in cmd]
    log("\n[cmd] " + " ".join(command))
    proc = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        check=False,
        text=True,
        capture_output=capture,
    )
    if proc.returncode != 0:
        details = ""
        if capture:
            details = f"\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        raise CommandError(
            f"Command failed with exit code {proc.returncode}: "
            f"{' '.join(command)}{details}"
        )
    return proc


def mount_drive(mount_point: Path) -> None:
    try:
        from google.colab import drive  # type: ignore

        log(f"[drive] Mounting Google Drive at {mount_point} ...")
        drive.mount(str(mount_point), force_remount=False)
    except ImportError:
        log("[drive] google.colab is unavailable; assuming Drive is already mounted.")
    if not mount_point.exists():
        raise RuntimeError(f"Drive mount point does not exist: {mount_point}")


def parse_seed_list(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Seeds must be unique")
    return seeds


def environment_key() -> str:
    return hashlib.sha256(
        json.dumps(ENVIRONMENT_SPEC, sort_keys=True).encode("utf-8")
    ).hexdigest()


def install_environment(venv_dir: Path, python: Path, skip_install: bool) -> None:
    marker = venv_dir / ".ginplus_environment.json"
    expected_key = environment_key()
    reusable = False
    if marker.exists() and python.exists():
        try:
            reusable = json.loads(marker.read_text())["key"] == expected_key
        except Exception:
            reusable = False

    if skip_install:
        if not reusable:
            raise RuntimeError(
                f"--skip-install was requested, but no matching environment exists at {venv_dir}"
            )
        log(f"[env] Reusing verified environment at {venv_dir}")
        return

    if reusable:
        log(f"[env] Reusing verified environment at {venv_dir}")
        return

    if shutil.which("uv") is None:
        run_cmd([sys.executable, "-m", "pip", "install", "-q", "uv"])
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv installation did not expose an executable")

    if venv_dir.exists():
        # Explicitly scoped to the disposable /content environment configured by this runner.
        shutil.rmtree(venv_dir)
    run_cmd([uv, "venv", "--python", "3.10", "--seed", venv_dir])
    pip = [python, "-m", "pip", "install", "--disable-pip-version-check"]

    run_cmd(
        pip
        + [
            "--index-url",
            "https://download.pytorch.org/whl/cu118",
            "--extra-index-url",
            "https://pypi.org/simple",
            "torch==2.2.0",
            "torchvision==0.17.0",
            "torchaudio==2.2.0",
        ]
    )
    run_cmd(
        pip
        + [
            "numpy==1.26.4",
            "scipy==1.12.0",
            "scikit-learn==1.4.0",
            "torch_geometric==2.3.1",
            "fsspec==2024.2.0",
            "rdkit==2023.9.5",
            "pytorch-lightning==2.2.5",
            "torchmetrics==1.3.2",
            "yacs==0.1.8",
            "networkx==3.2.1",
            "tensorboardX==2.6.2.2",
            "ogb==1.3.6",
            "wandb==0.16.3",
            "setuptools<81",
            "protobuf<5",
        ]
    )
    run_cmd(
        pip
        + [
            "--no-index",
            "--find-links",
            "https://data.pyg.org/whl/torch-2.2.0+cu118.html",
            "pyg_lib==0.4.0+pt22cu118",
            "torch_scatter==2.1.2+pt22cu118",
            "torch_sparse==0.6.18+pt22cu118",
            "torch_cluster==1.6.3+pt22cu118",
            "torch_spline_conv==1.2.2+pt22cu118",
        ]
    )
    marker.write_text(
        json.dumps({"key": expected_key, "spec": ENVIRONMENT_SPEC}, indent=2)
    )


def prepare_official_repo(repo_dir: Path) -> str:
    if not (repo_dir / ".git").exists():
        if repo_dir.exists():
            raise RuntimeError(
                f"{repo_dir} exists but is not a Git checkout; remove or rename it."
            )
        run_cmd(["git", "clone", "--filter=blob:none", OFFICIAL_REPO, repo_dir])
    else:
        origin = run_cmd(
            ["git", "-C", repo_dir, "remote", "get-url", "origin"], capture=True
        ).stdout.strip()
        normalized_origin = origin.rstrip("/").removesuffix(".git")
        normalized_expected = OFFICIAL_REPO.rstrip("/").removesuffix(".git")
        if normalized_origin != normalized_expected:
            raise RuntimeError(f"Unexpected repository origin: {origin}")
        dirty = run_cmd(
            [
                "git",
                "-C",
                repo_dir,
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            capture=True,
        ).stdout
        if dirty.strip():
            raise RuntimeError(
                "Official checkout has tracked modifications; refusing to overwrite them.\n"
                + dirty
            )
        run_cmd(["git", "-C", repo_dir, "fetch", "origin", OFFICIAL_COMMIT])

    run_cmd(["git", "-C", repo_dir, "checkout", "--detach", OFFICIAL_COMMIT])
    commit = run_cmd(
        ["git", "-C", repo_dir, "rev-parse", "HEAD"], capture=True
    ).stdout.strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"Commit mismatch: {commit} != {OFFICIAL_COMMIT}")
    log(f"[source] Official checkout pinned and clean: {commit}")
    return commit


def verify_config(repo_dir: Path) -> tuple[Path, bytes, str]:
    config_path = repo_dir / OFFICIAL_CONFIG_REL
    config_bytes = config_path.read_bytes()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    if config_sha != OFFICIAL_CONFIG_SHA256:
        raise RuntimeError(
            f"Official ZINC config digest changed: {config_sha} != "
            f"{OFFICIAL_CONFIG_SHA256}"
        )
    text = config_bytes.decode("utf-8")
    missing = [item for item in REQUIRED_CONFIG_FRAGMENTS if item not in text]
    if missing:
        raise RuntimeError(f"Official config semantic audit failed; missing: {missing}")
    log(f"[config] Verified {OFFICIAL_CONFIG_REL} (sha256={config_sha})")
    return config_path, config_bytes, text


def probe_runtime(python: Path) -> dict:
    probe = dedent(
        """
        import json, sys, torch, torch_geometric, sklearn, numpy
        info = {
            'python': sys.version.split()[0],
            'torch': torch.__version__,
            'torch_geometric': torch_geometric.__version__,
            'sklearn': sklearn.__version__,
            'numpy': numpy.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        print(json.dumps(info))
        """
    )
    result = run_cmd([python, "-c", probe], capture=True)
    info = json.loads(result.stdout.strip().splitlines()[-1])
    expected = {
        "python": "3.10",
        "torch": "2.2.0+cu118",
        "torch_geometric": "2.3.1",
        "sklearn": "1.4.0",
        "numpy": "1.26.4",
    }
    for key, expected_value in expected.items():
        actual = info[key]
        ok = (
            actual.startswith(expected_value + ".")
            if key == "python"
            else actual == expected_value
        )
        if not ok:
            raise RuntimeError(
                f"Runtime mismatch for {key}: {actual!r} != {expected_value!r}"
            )
    if not info["cuda_available"]:
        raise RuntimeError(
            "CUDA is unavailable. In Colab select Runtime > Change runtime type > GPU."
        )
    log("[runtime] " + json.dumps(info, sort_keys=True))
    return info


def write_provenance(
    *,
    provenance_dir: Path,
    python: Path,
    config_bytes: bytes,
    runtime_info: dict,
    seeds: Sequence[int],
    dataset_dir: Path,
    runs_dir: Path,
) -> Path:
    provenance_dir.mkdir(parents=True, exist_ok=True)
    freeze = run_cmd([python, "-m", "pip", "freeze"], capture=True).stdout
    (provenance_dir / "requirements_frozen.txt").write_text(freeze)
    (provenance_dir / "official_gine_zinc.yaml").write_bytes(config_bytes)
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper": "arXiv:2502.09263 / ICML 2025",
        "official_repo": OFFICIAL_REPO,
        "official_commit": OFFICIAL_COMMIT,
        "official_config": str(OFFICIAL_CONFIG_REL),
        "official_config_sha256": OFFICIAL_CONFIG_SHA256,
        "expected_parameters": EXPECTED_PARAMS,
        "seeds": list(seeds),
        "seed_basis": (
            "Paper reports 5 independent seeds; official main.py increments from "
            "base seed 0. Paper does not separately enumerate seed IDs; the README "
            "ZINC example requests only 2 repeats."
        ),
        "runtime": runtime_info,
        "dataset_cache": str(dataset_dir),
        "runs_root": str(runs_dir),
    }
    path = provenance_dir / "provenance.json"
    path.write_text(json.dumps(record, indent=2))
    log(f"[provenance] Saved {path}")
    return path


def directory_has_content(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def archive_attempt(seed_root: Path, results_dir: Path, raw_log: Path) -> Path:
    archive_parent = seed_root / "attempt_archives"
    archive_parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = archive_parent / timestamp
    suffix = 1
    while archive.exists():
        archive = archive_parent / f"{timestamp}_{suffix}"
        suffix += 1
    archive.mkdir()
    if results_dir.exists():
        shutil.move(str(results_dir), str(archive / "results"))
    if raw_log.exists():
        shutil.move(str(raw_log), str(archive / raw_log.name))
    completion = seed_root / "completed.json"
    if completion.exists():
        shutil.move(str(completion), str(archive / completion.name))
    log(f"[recovery] Archived prior seed output to {archive}")
    return archive


def stream_official_training(
    cmd: Sequence[object],
    *,
    cwd: Path,
    raw_log: Path,
    expected_params: int,
    console_epoch_period: int,
) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["WANDB_MODE"] = "disabled"  # The official config already has use=False.

    command = [str(item) for item in cmd]
    log("\n[official cmd] " + " ".join(command))
    log(f"[official cwd] {cwd}")
    log(f"[raw log] {raw_log}")
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    parameter_count: int | None = None
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    with raw_log.open("w", buffering=1) as log_file:
        for line in proc.stdout:
            log_file.write(line)
            stripped = line.rstrip()

            param_match = re.search(r"Num parameters:\s*([0-9,]+)", stripped)
            if param_match:
                parameter_count = int(param_match.group(1).replace(",", ""))
                log(stripped)
                if parameter_count != expected_params:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise RuntimeError(
                        f"Official model parameter mismatch: {parameter_count} != "
                        f"{expected_params}"
                    )
                continue

            epoch_match = re.search(r"> Epoch\s+(\d+):", stripped)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                if (
                    epoch < 3
                    or (epoch + 1) % console_epoch_period == 0
                    or epoch == 1999
                ):
                    log(stripped)
                continue

            important = (
                "[*] Run ID",
                "Start from epoch",
                "Checkpoint found",
                "Avg time per epoch",
                "Total train loop time",
                "Task done",
                "[*] All done",
                "Downloading",
                "Processing",
            )
            if any(token in stripped for token in important):
                log(stripped)

    return_code = proc.wait()
    if return_code != 0:
        raise CommandError(
            f"Official training failed with exit code {return_code}. Raw log: {raw_log}"
        )
    if parameter_count is None:
        raise RuntimeError(
            f"Training ended without the official parameter-count line. Raw log: {raw_log}"
        )
    return parameter_count


def read_json_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def parse_seed_result(results_dir: Path, seed: int) -> tuple[float, int | None]:
    val_records = read_json_lines(results_dir / "val" / "stats.json")
    test_records = read_json_lines(results_dir / "test" / "stats.json")
    if val_records and test_records:
        best_val = min(val_records, key=lambda record: float(record["mae"]))
        best_epoch = int(best_val["epoch"])
        test_by_epoch = {int(record["epoch"]): record for record in test_records}
        if best_epoch in test_by_epoch:
            return float(test_by_epoch[best_epoch]["mae"]), best_epoch

    result_file = results_dir / "subset_result.txt"
    if not result_file.exists():
        raise RuntimeError(f"Official result file is missing: {result_file}")
    pattern = re.compile(rf"\bseed_{seed}:.*?test_mae:\s*([0-9.eE+-]+)")
    matches = pattern.findall(result_file.read_text(errors="replace"))
    if not matches:
        raise RuntimeError(f"Could not parse seed {seed} test MAE from {result_file}")
    return float(matches[-1]), None


def run_seed(
    *,
    seed: int,
    python: Path,
    repo_dir: Path,
    config_path: Path,
    dataset_dir: Path,
    runs_dir: Path,
    console_epoch_period: int,
    force_rerun: bool,
) -> dict:
    seed_root = runs_dir / f"seed_{seed}"
    results_dir = seed_root / "results"
    raw_log = seed_root / f"official_stdout_seed{seed}.log"
    completion_path = seed_root / "completed.json"
    seed_root.mkdir(parents=True, exist_ok=True)

    if completion_path.exists() and not force_rerun:
        prior = json.loads(completion_path.read_text())
        retained_checkpoints = [
            Path(path) for path in prior.get("checkpoints", [])
        ]
        specification_matches = (
            prior.get("official_commit") == OFFICIAL_COMMIT
            and prior.get("config_sha256") == OFFICIAL_CONFIG_SHA256
            and prior.get("seed") == seed
            and prior.get("parameter_count") == EXPECTED_PARAMS
        )
        files_complete = (
            results_dir.exists()
            and retained_checkpoints
            and all(path.exists() for path in retained_checkpoints)
        )
        if specification_matches and files_complete:
            log(f"\n[skip] seed {seed} already completed: test MAE={prior['test_mae']}")
            return prior
        if not specification_matches:
            raise RuntimeError(
                f"Completion manifest for seed {seed} belongs to another specification."
            )
        log(
            f"[recovery] Seed {seed} has a matching completion manifest but missing "
            "result/checkpoint files; archiving and rerunning it."
        )

    if completion_path.exists() or directory_has_content(results_dir) or raw_log.exists():
        archive_attempt(seed_root, results_dir, raw_log)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Fast local cwd with only the official relative data/output paths routed to Drive.
    work_dir = Path(f"/content/ginplus_zinc_seed_{seed}")
    if work_dir.exists():
        # Explicitly scoped to this runner's seed-specific disposable /content directory.
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    os.symlink(dataset_dir, work_dir / "datasets", target_is_directory=True)
    os.symlink(results_dir, work_dir / "results", target_is_directory=True)

    command = [
        python,
        "-u",
        repo_dir / "main.py",
        "--cfg",
        config_path,
        "--repeat",
        "1",
        "seed",
        str(seed),
    ]
    started = datetime.now(timezone.utc)
    parameter_count = stream_official_training(
        command,
        cwd=work_dir,
        raw_log=raw_log,
        expected_params=EXPECTED_PARAMS,
        console_epoch_period=console_epoch_period,
    )
    finished = datetime.now(timezone.utc)

    test_mae, best_epoch = parse_seed_result(results_dir, seed)
    checkpoints = sorted(results_dir.glob("ckpt/*.ckpt"))
    if not checkpoints:
        raise RuntimeError(f"No official checkpoint retained for completed seed {seed}")

    record = {
        "seed": seed,
        "test_mae": test_mae,
        "best_validation_epoch": best_epoch,
        "parameter_count": parameter_count,
        "official_commit": OFFICIAL_COMMIT,
        "config_sha256": OFFICIAL_CONFIG_SHA256,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_hours": (finished - started).total_seconds() / 3600.0,
        "results_dir": str(results_dir),
        "raw_log": str(raw_log),
        "checkpoints": [str(path) for path in checkpoints],
        "command": [str(item) for item in command],
    }
    completion_path.write_text(json.dumps(record, indent=2))
    log(
        f"[complete] seed {seed}: test MAE={test_mae}; "
        f"best val epoch={best_epoch}; checkpoint={checkpoints[-1]}"
    )
    return record


def aggregate_results(records: Sequence[dict], drive_dir: Path) -> dict:
    maes = [float(record["test_mae"]) for record in records]
    mean_mae = statistics.fmean(maes)
    population_sd = math.sqrt(
        statistics.fmean([(value - mean_mae) ** 2 for value in maes])
    )
    aggregate = {
        "model": "official GIN+ (repository config name: gine)",
        "dataset": "ZINC subset",
        "metric": "test MAE at best validation-MAE epoch",
        "seeds": [record["seed"] for record in records],
        "test_mae_by_seed": {
            str(record["seed"]): record["test_mae"] for record in records
        },
        "best_validation_epoch_by_seed": {
            str(record["seed"]): record.get("best_validation_epoch")
            for record in records
        },
        "mean_test_mae": mean_mae,
        "population_sd_test_mae": population_sd,
        "paper_reference_mean": PAPER_REFERENCE_MEAN,
        "paper_reference_sd": PAPER_REFERENCE_SD,
        "official_commit": OFFICIAL_COMMIT,
        "config_sha256": OFFICIAL_CONFIG_SHA256,
        "parameter_count": EXPECTED_PARAMS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_json = drive_dir / "aggregate_summary.json"
    summary_json.write_text(json.dumps(aggregate, indent=2))

    summary_csv = drive_dir / "seed_results.csv"
    fields = [
        "seed",
        "test_mae",
        "best_validation_epoch",
        "elapsed_hours",
        "results_dir",
    ]
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fields})

    log("\nOfficial GIN+ / ZINC result")
    for record in records:
        log(
            f"  seed {record['seed']}: {record['test_mae']:.6f} "
            f"(best val epoch {record.get('best_validation_epoch')})"
        )
    log(f"  mean ± population s.d.: {mean_mae:.6f} ± {population_sd:.6f}")
    log(
        f"  paper reference:         {PAPER_REFERENCE_MEAN:.6f} ± "
        f"{PAPER_REFERENCE_SD:.6f}"
    )
    log(f"[summary] {summary_json}")
    log(f"[summary] {summary_csv}")
    return aggregate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the untouched official GIN+ implementation on ZINC in Colab, "
            "with per-seed Drive saves."
        )
    )
    parser.add_argument(
        "--drive-mount", type=Path, default=Path("/content/drive")
    )
    parser.add_argument(
        "--drive-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/ginplus_zinc_official"),
    )
    parser.add_argument(
        "--repo-dir", type=Path, default=Path("/content/GNNPlus_official")
    )
    parser.add_argument(
        "--venv-dir", type=Path, default=Path("/content/gnnplus_py310")
    )
    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=parse_seed_list("0,1,2,3,4"),
        help="Comma-separated seeds. Paper-faithful default: 0,1,2,3,4.",
    )
    parser.add_argument(
        "--console-epoch-period",
        type=int,
        default=10,
        help="Print every Nth official epoch summary; full logs always go to Drive.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Reuse an already verified /content environment.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Archive and rerun seeds even when their completion manifests are valid.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.console_epoch_period < 1:
        parser.error("--console-epoch-period must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    started = time.time()
    mount_drive(args.drive_mount)

    runs_dir = args.drive_dir / "runs"
    dataset_dir = args.drive_dir / "datasets"
    provenance_dir = args.drive_dir / "provenance"
    for path in (args.drive_dir, runs_dir, dataset_dir, provenance_dir):
        path.mkdir(parents=True, exist_ok=True)

    python = args.venv_dir / "bin/python"
    log(f"[spec] seeds={args.seeds}")
    log(f"[spec] Drive root={args.drive_dir}")
    log(f"[spec] dataset cache={dataset_dir}")
    install_environment(args.venv_dir, python, args.skip_install)
    prepare_official_repo(args.repo_dir)
    config_path, config_bytes, _ = verify_config(args.repo_dir)
    runtime_info = probe_runtime(python)
    write_provenance(
        provenance_dir=provenance_dir,
        python=python,
        config_bytes=config_bytes,
        runtime_info=runtime_info,
        seeds=args.seeds,
        dataset_dir=dataset_dir,
        runs_dir=runs_dir,
    )

    records = []
    for seed in args.seeds:
        records.append(
            run_seed(
                seed=seed,
                python=python,
                repo_dir=args.repo_dir,
                config_path=config_path,
                dataset_dir=dataset_dir,
                runs_dir=runs_dir,
                console_epoch_period=args.console_epoch_period,
                force_rerun=args.force_rerun,
            )
        )
    aggregate = aggregate_results(records, args.drive_dir)
    log(f"[done] Total wrapper time: {(time.time() - started) / 3600:.2f}h")
    log(f"[done] All outputs saved under: {args.drive_dir}")
    return aggregate


if __name__ == "__main__":
    # Default paste-and-run Colab command: paper-faithful seeds 0,1,2,3,4,
    # with all datasets, logs, checkpoints, provenance, and summaries on Drive.
    main(
        [
            # "--skip-install",  # Enable only when rerunning in the same Colab runtime.
            # "--force-rerun",   # Archives and reruns already completed seeds.
            "--drive-dir",
            "/content/drive/MyDrive/ginplus_zinc_official",
            "--seeds",
            "0,1,2,3,4",
            "--console-epoch-period",
            "10",
        ]
    )
