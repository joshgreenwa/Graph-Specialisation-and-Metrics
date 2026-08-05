#!/usr/bin/env python3
"""Manifest-driven one-GRIT-model-per-GPU launcher for Slurm/HPC.

This is orchestration only: task/model/training settings remain owned by the
validated ZINC, QM9-gap, and Peptides runners in this repository.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for _path in (REPO_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

OFFICIAL_GRIT_REPO = "https://github.com/LiamMa/GRIT.git"
OFFICIAL_GRIT_COMMIT = "6c988ea600a606fbb49a2246c64a2d37396b3ab5"
TASKS = ("zinc", "qm9_gap", "peptides_func", "peptides_struct")
TASK_HOP_LIMITS = {
    "zinc": 20,
    "qm9_gap": 20,
    "peptides_func": 16,
    "peptides_struct": 23,
}
READY_FILE = ".grit_base_dataset_ready.json"


@dataclass(frozen=True)
class Job:
    run_id: str
    task: str
    attention: str
    hops: int
    global_vnode: bool
    rrwp_horizon: int
    seed: int


def log(message: str) -> None:
    print(message, flush=True)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    try:
        return [int(item) for item in split_csv(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {value!r}") from exc


def parse_variant(value: str) -> tuple[str, int, bool, int]:
    match = re.fullmatch(
        r"(?P<base>dense|[1-9][0-9]*hop)"
        r"(?P<vnode>_vnode)?"
        r"(?:_localrrwp_h(?P<horizon>[0-9]+))?",
        value,
    )
    if not match:
        raise ValueError(
            f"invalid variant {value!r}; examples: dense, dense_vnode, 1hop, "
            "1hop_vnode, 2hop, 1hop_localrrwp_h1"
        )
    base = match.group("base")
    attention = "dense" if base == "dense" else "khop"
    hops = 1 if attention == "dense" else int(base.removesuffix("hop"))
    horizon = -1 if match.group("horizon") is None else int(match.group("horizon"))
    return attention, hops, bool(match.group("vnode")), horizon


def variant_slug(job: Job) -> str:
    slug = "dense" if job.attention == "dense" else f"{job.hops}hop"
    if job.global_vnode:
        slug += "_vnode"
    if job.rrwp_horizon >= 0:
        slug += f"_localrrwp_h{job.rrwp_horizon}"
    return slug


def validate_job(job: Job) -> None:
    if job.task not in TASKS:
        raise ValueError(f"unsupported task {job.task!r}; expected one of {TASKS}")
    if job.attention not in {"dense", "khop"}:
        raise ValueError(f"attention must be dense or khop, got {job.attention!r}")
    limit = TASK_HOP_LIMITS[job.task]
    if not 1 <= int(job.hops) <= limit:
        raise ValueError(f"{job.task} hops must be in 1..{limit}, got {job.hops}")
    if not -1 <= int(job.rrwp_horizon) <= limit:
        raise ValueError(f"{job.task} RRWP horizon must be -1 or 0..{limit}")
    if job.task == "zinc" and job.rrwp_horizon >= 0:
        raise ValueError("the general ZINC runner does not yet combine local RRWP with dense/k-hop/VNode")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job.run_id):
        raise ValueError(f"run_id contains unsafe path characters: {job.run_id!r}")


def jobs_from_grid(tasks: Iterable[str], variants: Iterable[str], seeds: Iterable[int]) -> list[Job]:
    jobs: list[Job] = []
    for task in tasks:
        for variant in variants:
            attention, hops, vnode, horizon = parse_variant(variant)
            for seed in seeds:
                provisional = Job(
                    run_id="pending",
                    task=task,
                    attention=attention,
                    hops=hops,
                    global_vnode=vnode,
                    rrwp_horizon=horizon,
                    seed=int(seed),
                )
                run_id = f"{task}.{variant_slug(provisional)}.s{seed}"
                job = Job(**{**asdict(provisional), "run_id": run_id})
                validate_job(job)
                jobs.append(job)
    return jobs


def write_manifest(path: Path, jobs: Sequence[Job]) -> None:
    if not jobs:
        raise ValueError("refusing to write an empty manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for job in jobs:
        validate_job(job)
        if job.run_id in seen:
            raise ValueError(f"duplicate run_id in manifest: {job.run_id}")
        seen.add(job.run_id)
    with path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(asdict(job), sort_keys=True) + "\n")


def read_manifest(path: Path) -> list[Job]:
    jobs: list[Job] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
            job = Job(**payload)
        except Exception as exc:
            raise ValueError(f"invalid manifest line {line_no} in {path}: {exc}") from exc
        validate_job(job)
        if job.run_id in seen:
            raise ValueError(f"duplicate run_id {job.run_id!r} at line {line_no}")
        seen.add(job.run_id)
        jobs.append(job)
    if not jobs:
        raise ValueError(f"manifest has no jobs: {path}")
    return jobs


def task_dataset_dir(dataset_root: Path, task: str) -> Path:
    return dataset_root / task


def dataset_ready_path(dataset_root: Path, task: str) -> Path:
    return task_dataset_dir(dataset_root, task) / READY_FILE


def require_dataset_ready(dataset_root: Path, task: str) -> None:
    marker = dataset_ready_path(dataset_root, task)
    if not marker.is_file():
        raise RuntimeError(
            f"base dataset for {task} is not staged ({marker} missing). "
            "Run the dataset staging Slurm job before launching the GPU array."
        )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def reserve_output(output_dir: Path, job: Job) -> None:
    spec_path = output_dir / "hpc_job.json"
    payload = asdict(job)
    if spec_path.exists():
        existing = json.loads(spec_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"output collision: {spec_path} belongs to a different job")
    else:
        atomic_write_json(spec_path, payload)


def record_slurm_attempt(output_dir: Path, job: Job, array_index: int) -> Path:
    """Persist the actual Slurm allocation used by this attempt."""
    job_id = os.environ.get("SLURM_JOB_ID", f"local-{os.getpid()}")
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", job_id)
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", str(array_index))
    safe_attempt = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{job_id}_{task_id}")
    path = output_dir / "hpc_attempts" / f"{safe_attempt}.json"
    atomic_write_json(
        path,
        {
            "run_id": job.run_id,
            "manifest_index": array_index,
            "slurm_job_id": job_id,
            "slurm_array_job_id": array_job_id,
            "slurm_array_task_id": task_id,
            "hostname": os.environ.get("SLURMD_NODENAME", os.uname().nodename),
        },
    )
    return path


def load_peptides_runner():
    path = REPO_ROOT / "experiments/peptides/training/GRIT_peptides_khop.py"
    spec = importlib.util.spec_from_file_location("grit_peptides_hpc_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Peptides runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runner_argv(
    job: Job,
    *,
    output_dir: Path,
    dataset_dir: Path,
    repo_dir: Path,
    grit_source: str,
    num_threads: int,
    accelerator: str,
    auto_resume: bool,
    recovery_ckpt_period: int = 10,
    wandb: bool = False,
    wandb_project: str | None = None,
) -> list[str]:
    common = [
        "--drive-mount", str(output_dir.parent),
        "--drive-dir", str(output_dir),
        "--dataset-dir", str(dataset_dir),
        "--repo-dir", str(repo_dir),
        "--repo-url", grit_source,
        "--seed", str(job.seed),
        "--name-tag", job.run_id,
        "--num-threads", str(num_threads),
        "--accelerator", accelerator,
        "--console-verbosity", "compact",
        "--recovery-ckpt-period", str(recovery_ckpt_period),
        "--skip-install",
        "--skip-editable-install",
    ]
    if not auto_resume:
        common.append("--no-auto-resume")
    if wandb:
        common.append("--wandb")
        if wandb_project:
            common.extend(["--wandb-project", wandb_project])
    if job.task == "zinc":
        return [
            *common,
            "--attention", job.attention,
            "--hops", str(job.hops),
            *(["--global-vnode"] if job.global_vnode else []),
        ]
    if job.task == "qm9_gap":
        return [
            *common,
            "--attention", job.attention,
            "--hops", str(job.hops),
            "--rrwp-horizon", str(job.rrwp_horizon),
            *(["--global-vnode"] if job.global_vnode else []),
        ]
    peptide_task = "func" if job.task == "peptides_func" else "struct"
    return [
        "--skip-project-git",
        *common,
        "--task", peptide_task,
        "--attention", job.attention,
        "--hops", str(job.hops),
        "--rrwp-horizon", str(job.rrwp_horizon),
        *(["--global-vnode"] if job.global_vnode else []),
    ]


def dispatch_job(job: Job, argv: Sequence[str]) -> None:
    log(f"[hpc] dispatch {job.run_id}: {' '.join(argv)}")
    if job.task == "zinc":
        from graph_specialisation_metrics.grit_patches import khop_zinc

        khop_zinc.main(argv)
    elif job.task == "qm9_gap":
        from graph_specialisation_metrics.grit_patches import qm9_gap

        qm9_gap.main(argv)
    else:
        load_peptides_runner().main(argv)


def clone_grit_for_staging(target: Path, source: str) -> Path:
    if target.exists():
        raise FileExistsError(f"refusing to replace existing staging checkout: {target}")
    subprocess.run(["git", "clone", "--quiet", source, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--quiet", OFFICIAL_GRIT_COMMIT], check=True)
    return target


def stage_one_dataset(task: str, dataset_root: Path, scratch_root: Path, grit_source: str) -> None:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    destination = task_dataset_dir(dataset_root, task)
    marker = dataset_ready_path(dataset_root, task)
    lock_dir = dataset_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{task}.lock").open("a+") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        if marker.is_file():
            log(f"[dataset] already staged: {task} -> {destination}")
            return
        destination.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        if task == "zinc":
            from torch_geometric.datasets import ZINC

            for split in ("train", "val", "test"):
                ZINC(root=str(destination), subset=True, split=split)
        elif task == "qm9_gap":
            from torch_geometric.datasets import QM9

            QM9(root=str(destination))
        else:
            from graph_specialisation_metrics.grit_patches import peptides as peptide_helpers

            runner = load_peptides_runner()
            scratch_root.mkdir(parents=True, exist_ok=True)
            staging_work = Path(
                tempfile.mkdtemp(prefix=f"grit_stage_{task}_", dir=scratch_root)
            )
            checkout = clone_grit_for_staging(staging_work / "GRIT", grit_source)
            base = SimpleNamespace(log=log)
            peptide_helpers.apply_peptides_dataset_compat_patch(base, checkout)
            runner.apply_peptides_pandas_warning_patch(checkout)
            filename = "peptides_functional.py" if task == "peptides_func" else "peptides_structural.py"
            class_name = "PeptidesFunctionalDataset" if task == "peptides_func" else "PeptidesStructuralDataset"
            module_path = checkout / "grit/loader/dataset" / filename
            spec = importlib.util.spec_from_file_location(f"stage_{task}", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            getattr(module, class_name)(root=str(destination))
        processed = [str(path.relative_to(destination)) for path in destination.rglob("*.pt")]
        if not processed:
            raise RuntimeError(f"dataset staging produced no processed .pt files under {destination}")
        atomic_write_json(
            marker,
            {"task": task, "processed_files": sorted(processed), "grit_commit": OFFICIAL_GRIT_COMMIT},
        )
        log(f"[dataset] staged {task}: {marker}")


def command_make_manifest(args: argparse.Namespace) -> None:
    tasks = split_csv(args.tasks)
    variants = split_csv(args.variants)
    seeds = parse_int_csv(args.seeds)
    jobs = jobs_from_grid(tasks, variants, seeds)
    write_manifest(args.output, jobs)
    log(f"[manifest] wrote {len(jobs)} jobs: {args.output}")
    log(f"[manifest] Slurm array range: 0-{len(jobs) - 1}")


def command_print_jobs(args: argparse.Namespace) -> None:
    jobs = read_manifest(args.manifest)
    for index, job in enumerate(jobs):
        print(f"{index:04d}  {job.run_id}")
    log(f"[manifest] {len(jobs)} jobs; array range 0-{len(jobs) - 1}")


def command_write_tracking(args: argparse.Namespace) -> None:
    jobs = read_manifest(args.manifest)
    parent_id = str(args.slurm_array_job_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "array_index",
                "array_job_id",
                "slurm_handle",
                "run_id",
                "task",
                "variant",
                "seed",
                "output_dir",
                "stdout_log",
                "stderr_log",
            ]
        )
        for index, job in enumerate(jobs):
            writer.writerow(
                [
                    index,
                    parent_id,
                    f"{parent_id}_{index}",
                    job.run_id,
                    job.task,
                    variant_slug(job),
                    job.seed,
                    args.output_root / job.run_id,
                    args.log_root / f"grit-molecules-{parent_id}-{index}.out",
                    args.log_root / f"grit-molecules-{parent_id}-{index}.err",
                ]
            )
    log(f"[tracking] wrote {len(jobs)} job mappings: {args.output}")


def command_status_tracking(args: argparse.Namespace) -> None:
    with args.tracking.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"tracking ledger has no jobs: {args.tracking}")
    parent_id = str(args.slurm_array_job_id or rows[0]["array_job_id"])
    states: dict[int, tuple[str, str, str, str]] = {}

    try:
        accounting = subprocess.run(
            [
                "sacct", "-X", "-n", "-P", "-j", parent_id,
                "--format=JobID,State,Elapsed,Timelimit,ExitCode",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        for line in accounting.stdout.splitlines():
            fields = line.strip().split("|")
            if len(fields) < 5:
                continue
            match = re.fullmatch(re.escape(parent_id) + r"_(\d+)", fields[0])
            if match:
                states[int(match.group(1))] = (fields[1], fields[2], fields[3], fields[4])
    except FileNotFoundError:
        pass

    try:
        queued = subprocess.run(
            [
                "squeue", "-h", "-r", "-j", parent_id,
                "-o", "%F|%K|%T|%M|%l|%R",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        for line in queued.stdout.splitlines():
            fields = line.strip().split("|", 5)
            if len(fields) == 6 and fields[0] == parent_id and fields[1].isdigit():
                states[int(fields[1])] = (fields[2], fields[3], fields[4], fields[5])
    except FileNotFoundError:
        pass

    display_rows = []
    for row in rows:
        index = int(row["array_index"])
        state, elapsed, limit, location = states.get(index, ("NOT_FOUND", "-", "-", "-"))
        if args.only_known and state == "NOT_FOUND":
            continue
        display_rows.append((index, f"{parent_id}_{index}", row["run_id"], state, elapsed, limit, location))

    print(f"Slurm array {parent_id} | mapping: {args.tracking}")
    print(f"{'IDX':>3}  {'SLURM HANDLE':<20}  {'RUN ID':<43}  {'STATE':<12}  {'ELAPSED':>10}  {'LIMIT':>10}  NODE/REASON")
    for index, handle, run_id, state, elapsed, limit, location in display_rows:
        print(
            f"{index:>3}  {handle:<20.20}  {run_id:<43.43}  "
            f"{state:<12.12}  {elapsed:>10.10}  {limit:>10.10}  {location}"
        )
    counts = Counter(row[3] for row in display_rows)
    summary = ", ".join(f"{state}={count}" for state, count in sorted(counts.items()))
    print(f"Summary: {summary or 'no matching rows'}")


def command_run(args: argparse.Namespace) -> None:
    if args.recovery_ckpt_period < 0:
        raise ValueError("--recovery-ckpt-period must be non-negative")
    jobs = read_manifest(args.manifest)
    if not 0 <= args.array_index < len(jobs):
        raise IndexError(f"array index {args.array_index} outside 0..{len(jobs) - 1}")
    job = jobs[args.array_index]
    dataset_dir = task_dataset_dir(args.dataset_root, job.task)
    if not args.allow_dataset_build:
        require_dataset_ready(args.dataset_root, job.task)
    output_dir = args.output_root / job.run_id
    reserve_output(output_dir, job)
    attempt_path = record_slurm_attempt(output_dir, job, args.array_index)
    job_token = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    array_token = os.environ.get("SLURM_ARRAY_TASK_ID", str(args.array_index))
    repo_dir = args.scratch_root / "repos" / f"{job.run_id}.{job_token}.{array_token}"
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    argv = runner_argv(
        job,
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        repo_dir=repo_dir,
        grit_source=args.grit_source,
        num_threads=args.num_threads,
        accelerator=args.accelerator,
        auto_resume=not args.no_auto_resume,
        recovery_ckpt_period=args.recovery_ckpt_period,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
    )
    log(f"[hpc] index={args.array_index} run_id={job.run_id}")
    log(f"[hpc] output={output_dir}")
    log(f"[hpc] dataset={dataset_dir}")
    log(f"[hpc] checkout={repo_dir}")
    log(f"[hpc] attempt={attempt_path}")
    if args.dry_run:
        log("[hpc] dry-run; model runner was not invoked")
        log("[hpc] runner argv: " + " ".join(argv))
        return
    dispatch_job(job, argv)


def command_stage(args: argparse.Namespace) -> None:
    tasks = split_csv(args.tasks)
    for task in tasks:
        stage_one_dataset(task, args.dataset_root, args.scratch_root, args.grit_source)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    make = subparsers.add_parser("make-manifest", help="Generate a JSONL task/architecture/seed grid.")
    make.add_argument("--tasks", default=",".join(TASKS))
    make.add_argument("--variants", default="dense,1hop,1hop_vnode,2hop,2hop_vnode")
    make.add_argument("--seeds", default="0")
    make.add_argument("--output", type=Path, required=True)
    make.set_defaults(func=command_make_manifest)

    show = subparsers.add_parser("print-jobs", help="Print manifest array indices.")
    show.add_argument("--manifest", type=Path, required=True)
    show.set_defaults(func=command_print_jobs)

    tracking = subparsers.add_parser("write-tracking", help="Write Slurm array-ID to run-ID mapping TSV.")
    tracking.add_argument("--manifest", type=Path, required=True)
    tracking.add_argument("--slurm-array-job-id", required=True)
    tracking.add_argument("--output-root", type=Path, required=True)
    tracking.add_argument("--log-root", type=Path, required=True)
    tracking.add_argument("--output", type=Path, required=True)
    tracking.set_defaults(func=command_write_tracking)

    status = subparsers.add_parser("status-tracking", help="Join a tracking ledger with squeue/sacct state.")
    status.add_argument("--tracking", type=Path, required=True)
    status.add_argument("--slurm-array-job-id", default=None)
    status.add_argument("--only-known", action="store_true")
    status.set_defaults(func=command_status_tracking)

    run = subparsers.add_parser("run", help="Run one manifest row (one GPU process).")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--array-index", type=int, required=True)
    run.add_argument("--dataset-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--scratch-root", type=Path, required=True)
    run.add_argument("--grit-source", default=OFFICIAL_GRIT_REPO)
    run.add_argument("--num-threads", type=int, default=8)
    run.add_argument("--accelerator", default="cuda:0")
    run.add_argument(
        "--recovery-ckpt-period",
        type=int,
        default=10,
        help="Write a persistent recovery snapshot every N epochs; 0 disables periodic snapshots.",
    )
    run.add_argument("--wandb", action="store_true")
    run.add_argument("--wandb-project", default=None)
    run.add_argument("--no-auto-resume", action="store_true")
    run.add_argument("--allow-dataset-build", action="store_true", help="Allow unsafe concurrent first-time dataset creation.")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=command_run)

    stage = subparsers.add_parser("stage-datasets", help="Serially download/process base datasets.")
    stage.add_argument("--tasks", default=",".join(TASKS))
    stage.add_argument("--dataset-root", type=Path, required=True)
    stage.add_argument("--scratch-root", type=Path, required=True)
    stage.add_argument("--grit-source", default=OFFICIAL_GRIT_REPO)
    stage.set_defaults(func=command_stage)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
